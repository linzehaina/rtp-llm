"""Cross-dimensional exact FP32 oracle and graph tests for fused prefill TopK.

M31_TOPK_TEST_EXTENSION can select a focused build of the identical production
CUDA translation unit when validating against an older immutable model wheel.
"""

import ast
import importlib
import importlib.util
import itertools
import os
import tempfile
import unittest
from pathlib import Path

import torch


def operator():
    extension = os.environ.get("M31_TOPK_TEST_EXTENSION")
    if extension:
        return importlib.import_module(extension).minimax_prefill_topk
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    return rtp_llm_ops.minimax_prefill_topk


def reference(scores, lengths, prefixes, k, init, local):
    positions = torch.tensor(
        [
            prefix + q
            for length, prefix in zip(lengths, prefixes)
            for q in range(length)
        ],
        device=scores.device,
        dtype=torch.int64,
    )
    end = positions // 128 + 1
    columns = torch.arange(scores.shape[2], device=scores.device)[None, :]
    live = columns < end[:, None]
    values = torch.where(scores.isnan(), -1e30, scores)
    values = torch.where((columns < init)[None, :, :] & live[None, :, :], 1e30, values)
    values = torch.where(
        (columns >= end[:, None] - local)[None, :, :] & live[None, :, :], 1e29, values
    )
    values = values.masked_fill(~live[None, :, :], float("-inf"))
    order = torch.argsort(values, dim=-1, descending=True, stable=True)
    out = torch.full(
        (*scores.shape[:2], k), -1, device=scores.device, dtype=torch.int32
    )
    take = min(k, scores.shape[2])
    out[:, :, :take] = order[:, :, :take].int()
    out.masked_fill_(
        torch.arange(k, device=scores.device)[None, None, :] >= end[None, :, None], -1
    )
    return out, positions, end


def legacy_kernels():
    # Load only the exact existing Triton bodies; the test need not import a
    # complete model runtime, fmha planning code, or its native dependencies.
    prefill = Path(__file__).resolve().parents[1] / "prefill"
    names = {
        "_compare_and_swap",
        "_bitonic_merge",
        "_topk_to_block_table_multirow_kernel",
        "_topk_to_block_table_tie_fallback_kernel",
    }
    pieces = ["import triton\nimport triton.language as tl\n"]
    for name in ("flash_with_topk_idx.py", "topk_bt_fused.py"):
        tree = ast.parse((prefill / name).read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                pieces.append(ast.unparse(node))
            elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name)
                and t.id in ("_HEUR_topk_to_block_table_kernel", "_topk_multirow_body")
                for t in node.targets
            ):
                pieces.append(ast.unparse(node))
    directory = tempfile.TemporaryDirectory(prefix="m31_legacy_topk_")
    path = Path(directory.name) / "kernels.py"
    path.write_text("\n\n".join(pieces))
    spec = importlib.util.spec_from_file_location("m31_legacy_topk_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return directory, module


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MinimaxPrefillTopkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.op = staticmethod(operator())  # A missing native op is a failure on GPU.
        cls.cases = 0
        torch.set_num_threads(1)
        torch.manual_seed(20261010)

    @staticmethod
    def triton_args(args):
        s, ti, bt, sl, cu, pre, qmax, k, init, local, emit, pages = args
        positional = [
            s,
            bt,
            sl,
            ti,
            1,
            128,
            cu,
            cu,
            pre,
            k,
            init,
            local,
            pages,
            *s.stride(),
            *bt.stride(),
            *ti.stride(),
        ]
        keywords = dict(
            NKV=s.shape[0],
            MASK_INIT=False,
            MASK_LOCAL=False,
            EMIT_TOPK_IDX=True,
            EMIT_BLOCK_TABLE=emit,
            BLOCK_Q=16,
            num_warps=4,
            num_stages=2,
        )
        grid = ((qmax + 15) // 16, pre.numel(), s.shape[0])
        return positional, keywords, grid

    @staticmethod
    def tie_pattern(args, pattern):
        s = args[0]
        k = args[7]
        s.copy_(torch.arange(s.shape[2], 0, -1, device=s.device, dtype=s.dtype))
        if pattern == "interior":
            s[:, :, 2] = s[:, :, 1]
        elif pattern == "cutoff":
            s[:, :, k - 1] = s[:, :, k - 2]
        elif pattern == "below":
            s[:, :, k + 1] = s[:, :, k]
        elif pattern == "all":
            s.fill_(1)

    def test_tie_flags_cover_selected_cutoff_and_below_cutoff_duplicates(self):
        for k, padded, emit, pattern in itertools.product(
            (16, 32),
            (False, True),
            (False, True),
            ("unique", "interior", "cutoff", "below", "all"),
        ):
            args, lengths, prefixes, bases = self.make_case(
                579, 4, k, [32], padded, emit
            )
            self.tie_pattern(args, pattern)
            flags = self.op(*args)
            self.assertEqual(flags.dtype, torch.int32)
            self.assertEqual(tuple(flags.shape), tuple(args[0].shape[:2]))
            self.assertTrue(
                torch.equal(
                    flags,
                    torch.full_like(flags, int(pattern not in ("unique", "below"))),
                )
            )
            self.verify(args, lengths, prefixes, bases)

    def test_legacy_tie_fallback_matches_original_ids_and_block_tables(self):
        directory, module = legacy_kernels()
        try:
            for k, padded, emit, pattern in itertools.product(
                (16, 32),
                (False, True),
                (False, True),
                ("unique", "interior", "cutoff", "all"),
            ):
                args, _, _, _ = self.make_case(579, 4, k, [32], padded, emit)
                args = list(args)
                args[3] = torch.empty(
                    args[0].shape[0] * args[0].shape[1],
                    device="cuda",
                    dtype=torch.int32,
                )
                self.tie_pattern(args, pattern)
                shadow = list(args)
                for i in (1, 2, 3):
                    shadow[i] = torch.empty_like(args[i])
                pos, kw, grid = self.triton_args(shadow)
                module._topk_to_block_table_multirow_kernel[grid](*pos, **kw)
                flags = self.op(*args)
                pos, kw, grid = self.triton_args(args)
                module._topk_to_block_table_tie_fallback_kernel[grid](
                    *pos, **kw, TIE_FLAGS=flags, TIE_STRIDE_H=flags.stride(0)
                )
                self.assertTrue(
                    torch.equal(args[1], shadow[1]), (k, padded, emit, pattern)
                )
                if emit:
                    self.assertTrue(torch.equal(args[2], shadow[2]))
                    self.assertTrue(torch.equal(args[3], shadow[3]))
                self.__class__.cases += 1
        finally:
            directory.cleanup()

    def test_legacy_tie_fallback_graph_replay_changes_tie_policy_on_gpu(self):
        directory, module = legacy_kernels()
        try:
            for k, emit in itertools.product((16, 32), (False, True)):
                args, _, _, _ = self.make_case(579, 4, k, [32], True, emit)
                args = list(args)
                args[3] = torch.empty(
                    args[0].shape[0] * args[0].shape[1],
                    device="cuda",
                    dtype=torch.int32,
                )
                pos, kw, grid = self.triton_args(args)

                def run():
                    flags = self.op(*args)
                    module._topk_to_block_table_tie_fallback_kernel[grid](
                        *pos, **kw, TIE_FLAGS=flags, TIE_STRIDE_H=flags.stride(0)
                    )

                self.tie_pattern(args, "unique")
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                for pattern in ("unique", "all", "cutoff", "interior"):
                    self.tie_pattern(args, pattern)
                    graph.replay()
                    shadow = list(args)
                    for i in (1, 2, 3):
                        shadow[i] = torch.empty_like(args[i])
                    spos, skw, sgrid = self.triton_args(shadow)
                    module._topk_to_block_table_multirow_kernel[sgrid](*spos, **skw)
                    self.assertTrue(torch.equal(args[1], shadow[1]), (k, emit, pattern))
                    if emit:
                        self.assertTrue(torch.equal(args[2], shadow[2]))
                        self.assertTrue(torch.equal(args[3], shadow[3]))
                    self.__class__.cases += 1
        finally:
            directory.cleanup()

    def test_legacy_fallback_marks_nonfinite_and_legacy_padding_score_hazards(self):
        directory, module = legacy_kernels()
        try:
            for k, kind in itertools.product(
                (16, 32), ("nan", "all_nan", "inf", "all_neginf", "tiny")
            ):
                args, _, _, _ = self.make_case(579, 4, k, [32], True, False, kind=kind)
                if kind == "tiny":
                    args[0].fill_(-1e31)
                flags = self.op(*args)
                if kind in ("all_nan", "inf", "all_neginf", "tiny"):
                    self.assertTrue(bool(flags.all()))
                shadow = list(args)
                shadow[1] = torch.empty_like(args[1])
                pos, kw, grid = self.triton_args(shadow)
                module._topk_to_block_table_multirow_kernel[grid](*pos, **kw)
                pos, kw, grid = self.triton_args(args)
                module._topk_to_block_table_tie_fallback_kernel[grid](
                    *pos, **kw, TIE_FLAGS=flags, TIE_STRIDE_H=flags.stride(0)
                )
                self.assertTrue(torch.equal(args[1], shadow[1]), (k, kind))
                self.__class__.cases += 1
        finally:
            directory.cleanup()

    def make_case(self, n, h, k, lengths, padded, emit, init=0, local=1, kind="normal"):
        lengths = [min(q, n * 128) for q in lengths]
        total = sum(lengths)
        prefixes = [max(0, n * 128 - q - (b * 19) % 128) for b, q in enumerate(lengths)]
        cu = torch.tensor(
            [0] + list(itertools.accumulate(lengths)), device="cuda", dtype=torch.int32
        )
        pre = torch.tensor(prefixes, device="cuda", dtype=torch.int32)
        score_base = torch.randn(
            h, total + (11 if padded else 0), n + (7 if padded else 0), device="cuda"
        )
        scores = score_base[:, :total, :n]
        if kind == "ties":
            scores.fill_(1)
        elif kind == "narrow":
            scores.mul_(1e-6).add_(1)
        elif kind == "negative":
            scores.abs_().neg_()
        elif kind == "nan":
            scores[:, :, ::7] = float("nan")
        elif kind == "all_nan":
            scores.fill_(float("nan"))
        elif kind == "inf":
            scores[:, :, ::7] = float("inf")
            scores[:, :, 1::7] = float("-inf")
        elif kind == "all_neginf":
            scores.fill_(float("-inf"))
        elif kind == "zeros":
            scores.zero_()
            scores[:, :, ::2] = -0.0
        out_base = torch.full(
            (h, total + (13 if padded else 0), k + (3 if padded else 0)),
            -77,
            device="cuda",
            dtype=torch.int32,
        )
        out = out_base[:, :total, :k]
        bt_base = torch.full(
            (total * h + 9, k + (5 if padded else 0)),
            -77,
            device="cuda",
            dtype=torch.int32,
        )
        bt = bt_base[: total * h, :k]
        sl_base = torch.full(
            (total * h * 2 + 9,), -77, device="cuda", dtype=torch.int32
        )
        sl = sl_base[: total * h * 2 : 2]
        args = (
            scores,
            out,
            bt,
            sl,
            cu,
            pre,
            max(lengths, default=0),
            k,
            init,
            local,
            emit,
            n + 17,
        )
        return args, lengths, prefixes, (score_base, out_base, bt_base, sl_base)

    def verify(self, args, lengths, prefixes, bases):
        scores, out, bt, sl, cu, pre, qmax, k, init, local, emit, npages = args
        expected, positions, end = reference(scores, lengths, prefixes, k, init, local)
        self.assertTrue(
            torch.equal(out, expected), f"indices differ: N={scores.shape[2]}, K={k}"
        )
        h, total = scores.shape[:2]
        # Padded retained capacity must never be written.
        self.assertTrue(bool((bases[1][:, total:, :] == -77).all()))
        self.assertTrue(bool((bases[1][:, :total, k:] == -77).all()))
        self.assertTrue(bool((bases[2][total * h :] == -77).all()))
        self.assertTrue(bool((bases[2][: total * h, k:] == -77).all()))
        self.assertTrue(bool((bases[3][1 : total * h * 2 : 2] == -77).all()))
        if emit and total:
            rows = out.permute(1, 0, 2).reshape(total * h, k).long()
            current = (positions // 128).repeat_interleave(h)[:, None]
            slots = torch.arange(k, device="cuda")[None, :].expand(total * h, -1)
            priorities = torch.where(
                rows < 0, k + 1, torch.where(rows == current, k, slots)
            )
            pages = rows.gather(1, priorities.argsort(dim=1, stable=True))
            pages += torch.arange(h, device="cuda").repeat(total)[:, None] * npages
            keep = end.clamp(max=k).repeat_interleave(h)
            valid = slots < keep[:, None]
            self.assertTrue(torch.equal(bt[valid], pages.int()[valid]))
            self.assertTrue(bool((bt[~valid] == -77).all()))
            has_local = (rows == current).any(dim=1)
            expected_sl = torch.where(
                has_local,
                (keep - 1) * 128 + (positions % 128 + 1).repeat_interleave(h),
                keep * 128,
            )
            self.assertTrue(torch.equal(sl, expected_sl.int()))
        elif not emit:
            self.assertTrue(bool((bt == -77).all()))
            self.assertTrue(bool((sl == -77).all()))
        self.__class__.cases += 1

    def test_shape_layout_and_emission_cross_product(self):
        ns = [
            1,
            15,
            16,
            17,
            31,
            32,
            33,
            127,
            128,
            129,
            255,
            256,
            257,
            320,
            383,
            384,
            385,
            448,
            511,
            512,
            513,
            578,
            579,
            580,
            639,
            640,
            641,
            767,
            768,
            769,
            1023,
            1024,
        ]
        for n, h, k, padded, emit in itertools.product(
            ns, [1, 2, 4, 8], [16, 32], [False, True], [False, True]
        ):
            with self.subTest(n=n, heads=h, k=k, padded=padded, emit=emit):
                args, lengths, prefixes, bases = self.make_case(
                    n, h, k, [3, 0, 5, 129], padded, emit
                )
                before = bases[0].clone()
                self.op(*args)
                self.verify(args, lengths, prefixes, bases)
                self.assertTrue(torch.equal(before, bases[0]))

    def test_special_values_and_initial_local_cross_product(self):
        for n, k, kind, init, local in itertools.product(
            [17, 579, 641],
            [16, 32],
            [
                "normal",
                "ties",
                "narrow",
                "negative",
                "nan",
                "all_nan",
                "inf",
                "all_neginf",
                "zeros",
            ],
            [0, 1, 4],
            [0, 1, 3],
        ):
            with self.subTest(n=n, k=k, kind=kind, init=init, local=local):
                args, lengths, prefixes, bases = self.make_case(
                    n, 4, k, [1, 0, 3, 7], True, True, init, local, kind
                )
                # Include short causal rows and a partial current block.
                prefixes[-1] = 0
                args[5][-1] = 0
                self.op(*args)
                self.verify(args, lengths, prefixes, bases)

    def test_graph_replay_reads_fresh_scores_prefix_and_segment_boundaries(self):
        for n, k, emit in itertools.product([256, 579, 1024], [16, 32], [False, True]):
            args, lengths, prefixes, bases = self.make_case(
                n, 4, k, [3, 0, 5, 7], True, emit
            )
            self.op(*args)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.op(*args)
            for revision in range(4):
                lengths = [1 + revision, 2, 6 - revision, 6]
                prefixes = [0, 128 * (n // 2) - revision, n * 128 - 24 - revision, 127]
                args[0].normal_()
                args[4].copy_(
                    torch.tensor(
                        [0] + list(itertools.accumulate(lengths)),
                        device="cuda",
                        dtype=torch.int32,
                    )
                )
                args[5].copy_(torch.tensor(prefixes, device="cuda", dtype=torch.int32))
                args[1].fill_(-77)
                args[2].fill_(-77)
                args[3].fill_(-77)
                graph.replay()
                self.verify(args, lengths, prefixes, bases)

    def test_subnormal_keys_with_cpu_ieee_oracle(self):
        # The Bazel CUDA toolchain enables FTZ. The TopK key transform must use
        # integer bits, so nonzero subnormals are not mistaken for signed zero.
        bits = torch.tensor(
            [-2147483645, -2147483646, -2147483647, 0, 1, 2, 3, -2147483648],
            dtype=torch.int32,
        )
        for n, k, emit in itertools.product([579, 1024], [16, 32], [False, True]):
            args, lengths, prefixes, bases = self.make_case(
                n, 4, k, [5], True, emit, local=0
            )
            row = bits.repeat((n + 7) // 8)[:n].view(torch.float32)
            args[0].copy_(row.to("cuda")[None, None, :])
            expected = row.argsort(descending=True, stable=True)[:k].int()
            self.op(*args)
            self.assertTrue(torch.equal(args[1].cpu(), expected.expand(4, 5, k)))
            self.verify(args, lengths, prefixes, bases)

    def test_empty_and_current_stream(self):
        for lengths in ([0, 0], [1], [3], [4], [5], [127, 129]):
            args, lens, prefixes, bases = self.make_case(
                579, 4, 16, lengths, True, True
            )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                args[0].fill_(2)
                self.op(*args)
            stream.synchronize()
            self.verify(args, lens, prefixes, bases)

    def test_invalid_host_arguments(self):
        args, _, _, _ = self.make_case(579, 4, 16, [3, 5], True, True)
        mutations = [
            (0, args[0].half()),
            (1, args[1].long()),
            (4, args[4].long()),
            (5, args[5].cpu()),
            (5, args[5][::2]),
            (6, -1),
            (7, 8),
            (8, -1),
            (9, -1),
            (11, 1),
            (11, 2**40),
            (2, args[2][:, ::2]),
        ]
        for index, value in mutations:
            bad = list(args)
            bad[index] = value
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                self.op(*bad)
        bad = list(args)
        bad[1] = args[0].view(torch.int32)[:, :, :16]
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            self.op(*bad)
        bad = list(args)
        bad[0] = args[0].expand(4, -1, -1)[:1].expand(4, -1, -1)
        with self.assertRaises(RuntimeError):
            self.op(*bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
