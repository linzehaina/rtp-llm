"""Independent CPU IEEE oracle, stream ordering and integer-boundary review tests.

Run with M31_TOPK_TEST_EXTENSION for a focused build of the production CUDA TU.
The >INT32 stride allocation is opt-in: M31_TOPK_LARGE_STRIDE=1 (~17 GiB).
"""

import itertools
import math
import os
import random
import struct
import unittest

import test_minimax_prefill_topk as topk_test
import torch


def fp32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def bits_tensor(bits):
    signed = [b if b < 0x80000000 else b - 0x100000000 for b in bits]
    return torch.tensor(signed, dtype=torch.int32).view(torch.float32)


def cpu_oracle(scores, lengths, prefixes, k, init, local):
    # Python float represents every FP32 value exactly. No GPU float arithmetic,
    # FP16 approximation, or reproduction of the CUDA integer key transform.
    rows = scores.cpu().tolist()
    positions = [p + q for length, p in zip(lengths, prefixes) for q in range(length)]
    result = []
    for head in rows:
        out = []
        for row, position in zip(head, positions):
            end = position // 128 + 1
            values = []
            for idx in range(end):
                value = fp32(0xF149F2CA) if math.isnan(row[idx]) else row[idx]
                if idx < init:
                    value = fp32(0x7149F2CA)  # 1e30f
                if idx >= end - local:
                    value = fp32(0x6FA18F08)  # 1e29f
                values.append((value, idx))
            order = sorted(values, key=lambda pair: (-pair[0], pair[1]))
            out.append([idx for _, idx in order[:k]] + [-1] * max(0, k - end))
        result.append(out)
    return torch.tensor(result, dtype=torch.int32), positions


def patterns(n):
    rng = random.Random(20261010)
    exponent_edges = [b for exp in range(1, 255) for b in ((exp << 23) - 1, exp << 23)]
    raw = {
        "adjacent_one": [0x3F800000 + i for i in range(n)],
        "adjacent_negative_one": [0xBF800000 + i for i in range(n)],
        "positive_subnormal": [i + 1 for i in range(n)],
        "negative_subnormal": [0x80000001 + i for i in range(n)],
        "minimum_normal_boundary": [0x00800000 - n // 2 + i for i in range(n)],
        "maximum_finite": [0x7F7FFFFF - i for i in range(n)],
        "negative_maximum_finite": [0xFF7FFFFF - i for i in range(n)],
        "half_overflow_boundary": [0x477FF000 - n // 2 + i for i in range(n)],
        "exponent_boundaries": exponent_edges,
        "signed_zeros": [0, 0x80000000],
        "nan_payloads": [0x7F800001, 0x7FC00000, 0x7FFFFFFF, 0xFF800001, 0xFFC00000],
        "infinities": [0x7F800000, 0xFF800000, 0x3F800000, 0xBF800000],
        "all_below_legacy_padding": [0xF149F2CB + i for i in range(n)],
        "random_raw_bits": [rng.getrandbits(32) for _ in range(n)],
    }
    return {
        name: (bits * ((n + len(bits) - 1) // len(bits)))[:n]
        for name, bits in raw.items()
    }


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PrefillTopkAdversarialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.op = staticmethod(topk_test.operator())
        cls.cases = 0
        cls.legacy_findings = []
        torch.set_num_threads(1)

    def make_case(self, n, h, k, lengths, padded, emit, init=0, local=0):
        return topk_test.MinimaxPrefillTopkTest.make_case(
            self, n, h, k, lengths, padded, emit, init, local
        )

    def check_math(self, args, lengths, prefixes, bases):
        expected, positions = cpu_oracle(
            args[0], lengths, prefixes, args[7], args[8], args[9]
        )
        self.assertTrue(
            torch.equal(args[1].cpu(), expected), "CPU IEEE TopK IDs differ"
        )
        h, total = args[0].shape[:2]
        k, emit, pages = args[7], args[10], args[11]
        self.assertTrue(bool((bases[1][:, total:, :] == -77).all()))
        self.assertTrue(bool((bases[1][:, :total, k:] == -77).all()))
        self.assertTrue(bool((bases[2][total * h :, :] == -77).all()))
        self.assertTrue(bool((bases[2][: total * h, k:] == -77).all()))
        self.assertTrue(bool((bases[3][1 : total * h * 2 : 2] == -77).all()))
        if emit:
            bt = args[2].cpu()
            sl = args[3].cpu()
            for q, position in enumerate(positions):
                for head in range(h):
                    ids = [i for i in expected[head, q].tolist() if i >= 0]
                    current = position // 128
                    selected = [i for i in ids if i != current] + (
                        [current] if current in ids else []
                    )
                    row = q * h + head
                    self.assertEqual(
                        bt[row, : len(ids)].tolist(),
                        [head * pages + i for i in selected],
                    )
                    self.assertTrue(bool((bt[row, len(ids) :] == -77).all()))
                    span = (
                        (len(ids) - 1) * 128 + position % 128 + 1
                        if current in ids
                        else len(ids) * 128
                    )
                    self.assertEqual(sl[row].item(), span)
        else:
            self.assertTrue(bool((args[2] == -77).all()))
            self.assertTrue(bool((args[3] == -77).all()))
        self.__class__.cases += 1

    def test_fp32_extremes_ulp_subnormal_nan_and_fp16_boundaries(self):
        for n, k, padded, emit in itertools.product(
            (579, 1024), (16, 32), (False, True), (False, True)
        ):
            for name, raw in patterns(n).items():
                with self.subTest(n=n, k=k, padded=padded, emit=emit, pattern=name):
                    args, lens, prefixes, bases = self.make_case(
                        n, 4, k, [3, 0, 5], padded, emit
                    )
                    for head in range(4):
                        args[0][head].copy_(bits_tensor(raw).roll(7 * head).to("cuda"))
                    before = args[0].view(torch.int32).clone()
                    self.op(*args)
                    self.check_math(args, lens, prefixes, bases)
                    self.assertTrue(torch.equal(args[0].view(torch.int32), before))

    def test_extreme_bias_counts_and_partial_causal_rows(self):
        for k, init, local in itertools.product(
            (16, 32), (0, 4, 2147483647), (0, 3, 2147483647)
        ):
            args, lens, prefixes, bases = self.make_case(
                579, 4, k, [1, 3, 5], True, True, init, local
            )
            prefixes[:] = [0, 127, 255]
            args[5].copy_(torch.tensor(prefixes, device="cuda", dtype=torch.int32))
            raw = patterns(579)["random_raw_bits"]
            args[0].copy_(bits_tensor(raw).to("cuda"))
            self.op(*args)
            self.check_math(args, lens, prefixes, bases)

    def test_full_legacy_compatibility_for_unique_numeric_extremes(self):
        directory, module = topk_test.legacy_kernels()
        try:
            for name, raw in patterns(579).items():
                args, lens, prefixes, bases = self.make_case(
                    579, 4, 16, [3, 0, 5], True, False
                )
                args[0].copy_(bits_tensor(raw).to("cuda"))
                expected, _ = cpu_oracle(args[0], lens, prefixes, 16, 0, 0)
                shadow = list(args)
                shadow[1] = torch.empty_like(args[1])
                pos, kw, grid = topk_test.MinimaxPrefillTopkTest.triton_args(shadow)
                module._topk_to_block_table_multirow_kernel[grid](*pos, **kw)
                flags = self.op(*args)
                native = args[1].cpu()
                pos, kw, grid = topk_test.MinimaxPrefillTopkTest.triton_args(args)
                module._topk_to_block_table_tie_fallback_kernel[grid](
                    *pos, **kw, TIE_FLAGS=flags, TIE_STRIDE_H=flags.stride(0)
                )
                full = args[1].cpu()
                legacy = shadow[1].cpu()
                finding = dict(
                    pattern=name,
                    flagged_rows=int(flags.sum().item()),
                    rows=flags.numel(),
                    native_math_equal=torch.equal(native, expected),
                    full_legacy_equal=torch.equal(full, legacy),
                    legacy_stable_oracle_equal=torch.equal(legacy, expected),
                    native_ids=native[0, 0].tolist(),
                    legacy_ids=legacy[0, 0].tolist(),
                )
                self.__class__.legacy_findings.append(finding)
                print("LEGACY_REVIEW", finding, flush=True)
                self.assertTrue(finding["native_math_equal"], name)
                self.assertTrue(finding["full_legacy_equal"], name)
                self.__class__.cases += 1
        finally:
            directory.cleanup()

    def test_three_streams_flag_lifetime_and_producer_consumer_order(self):
        directory, module = topk_test.legacy_kernels()
        try:
            streams = [torch.cuda.Stream() for _ in range(3)]
            tasks = []
            for stream, lengths in zip(
                streams, ([1, 0, 3, 4, 5], [15, 16, 17], [31, 32, 33])
            ):
                with torch.cuda.stream(stream):
                    args, _, _, _ = self.make_case(579, 4, 16, lengths, True, False)
                    pos, kw, grid = topk_test.MinimaxPrefillTopkTest.triton_args(args)
                    topk_test.MinimaxPrefillTopkTest.tie_pattern(args, "unique")
                    flags = self.op(*args)
                    module._topk_to_block_table_tie_fallback_kernel[grid](
                        *pos, **kw, TIE_FLAGS=flags, TIE_STRIDE_H=flags.stride(0)
                    )
                tasks.append((stream, args, pos, kw, grid))
            snapshots = []
            for revision in range(24):
                for stream, args, pos, kw, grid in tasks:
                    with torch.cuda.stream(stream):
                        pattern = ("unique", "all", "cutoff", "interior")[revision % 4]
                        topk_test.MinimaxPrefillTopkTest.tie_pattern(args, pattern)
                        shadow = list(args)
                        shadow[1] = torch.empty_like(args[1])
                        spos, skw, sgrid = topk_test.MinimaxPrefillTopkTest.triton_args(
                            shadow
                        )
                        module._topk_to_block_table_multirow_kernel[sgrid](*spos, **skw)
                        flags = self.op(*args)
                        module._topk_to_block_table_tie_fallback_kernel[grid](
                            *pos, **kw, TIE_FLAGS=flags, TIE_STRIDE_H=flags.stride(0)
                        )
                        del flags
                        # Try immediate same-stream allocator reuse after consumer enqueue.
                        torch.empty(
                            args[0].shape[:2], device="cuda", dtype=torch.int32
                        ).fill_(-12345)
                        snapshots.append((args[1].clone(), shadow[1]))
            for stream in streams:
                stream.synchronize()
            for actual, expected in snapshots:
                self.assertTrue(torch.equal(actual, expected))
                self.__class__.cases += 1
        finally:
            directory.cleanup()

    def test_int32_page_limit_and_unused_pages(self):
        for h in (1, 4, 8):
            args, lens, prefixes, bases = self.make_case(579, h, 16, [3], True, True)
            args = list(args)
            args[11] = 2147483647 // h
            self.op(*args)
            self.check_math(args, lens, prefixes, bases)
            args[11] += 1
            with self.assertRaisesRegex(RuntimeError, "page-ID overflow"):
                self.op(*args)
        args, lens, prefixes, bases = self.make_case(579, 4, 16, [3], True, False)
        args = list(args)
        args[11] = 1 << 40  # Unused when EmitBT=false; should have no effect.
        self.op(*args)
        self.check_math(args, lens, prefixes, bases)

    def test_graph_replay_updates_metadata_and_fallback_flags_together(self):
        directory, module = topk_test.legacy_kernels()
        try:
            for k, emit in itertools.product((16, 32), (False, True)):
                args, _, _, _ = self.make_case(579, 4, k, [3, 0, 5, 7], True, emit)
                # The reused legacy path requires contiguous sequence lengths.
                args = list(args)
                args[3] = torch.empty(4 * 15, device="cuda", dtype=torch.int32)
                pos, kw, grid = topk_test.MinimaxPrefillTopkTest.triton_args(args)

                def run():
                    flags = self.op(*args)
                    module._topk_to_block_table_tie_fallback_kernel[grid](
                        *pos, **kw, TIE_FLAGS=flags, TIE_STRIDE_H=flags.stride(0)
                    )

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                for revision in range(32):
                    lengths = ([3, 0, 5, 7], [5, 1, 3, 6])[revision % 2]
                    args[4].copy_(
                        torch.tensor(
                            [0] + list(itertools.accumulate(lengths)),
                            device="cuda",
                            dtype=torch.int32,
                        )
                    )
                    args[5].copy_(
                        torch.tensor(
                            [0, 127, 579 * 128 - 16, 255],
                            device="cuda",
                            dtype=torch.int32,
                        )
                    )
                    topk_test.MinimaxPrefillTopkTest.tie_pattern(
                        args, ("unique", "all", "cutoff", "interior")[revision % 4]
                    )
                    for index in (1, 2, 3):
                        args[index].fill_(-77)
                    graph.replay()
                    shadow = list(args)
                    for index in (1, 2, 3):
                        shadow[index] = torch.full_like(args[index], -77)
                    spos, skw, sgrid = topk_test.MinimaxPrefillTopkTest.triton_args(
                        shadow
                    )
                    module._topk_to_block_table_multirow_kernel[sgrid](*spos, **skw)
                    self.assertTrue(torch.equal(args[1], shadow[1]))
                    if emit:
                        self.assertTrue(torch.equal(args[2], shadow[2]))
                        self.assertTrue(torch.equal(args[3], shadow[3]))
                    self.__class__.cases += 1
        finally:
            directory.cleanup()

    def test_empty_huge_stride_validation_does_not_wrap(self):
        for tensor_index, last in ((0, 579), (1, 16)):
            args, _, _, _ = self.make_case(579, 0, 16, [2], False, False)
            args = list(args)
            args[tensor_index] = torch.empty_strided(
                (0, 2, last),
                (1 << 62, 1 << 62, 1),
                device="cuda",
                dtype=torch.float32 if tensor_index == 0 else torch.int32,
            )
            with self.assertRaisesRegex(RuntimeError, "nonoverlapping"):
                self.op(*args)
            self.__class__.cases += 1

    def test_scalar_narrowing_bounds_before_empty_return(self):
        args, _, _, _ = self.make_case(579, 0, 16, [], False, False)
        for index, limit in ((6, 2147483644), (8, 2147483647), (9, 2147483647)):
            good = list(args)
            good[index] = limit
            self.assertEqual(self.op(*good).numel(), 0)
            good[index] += 1
            with self.assertRaises(RuntimeError):
                self.op(*good)
            self.__class__.cases += 2
        for h, q in ((65536, 0), (0, 2147483648)):
            bad = list(args)
            bad[0] = torch.empty((h, q, 579), device="cuda")
            bad[1] = torch.empty((h, q, 16), device="cuda", dtype=torch.int32)
            with self.assertRaisesRegex(RuntimeError, "launch limits"):
                self.op(*bad)
            self.__class__.cases += 1
        bad = list(args)
        bad[4] = torch.zeros(65537, device="cuda", dtype=torch.int32)
        bad[5] = torch.zeros(65536, device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "launch limits"):
            self.op(*bad)
        self.__class__.cases += 1

    @unittest.skipUnless(
        os.environ.get("M31_TOPK_LARGE_STRIDE") == "1", "opt-in ~17 GiB allocation"
    )
    def test_real_pointer_offsets_above_int32(self):
        for axis in ("head", "query"):
            h, q = (2, 1) if axis == "head" else (1, 2)
            sp, tp = (1 << 31) + 4096, (1 << 31) + 128
            scores = torch.empty_strided(
                (h, q, 579),
                (sp if axis == "head" else q * sp, 579 if axis == "head" else sp, 1),
                device="cuda",
            )
            ti = torch.empty_strided(
                (h, q, 16),
                (tp if axis == "head" else q * tp, 16 if axis == "head" else tp, 1),
                device="cuda",
                dtype=torch.int32,
            )
            scores.copy_(torch.arange(579, device="cuda", dtype=torch.float32))
            scores[h - 1, q - 1].copy_(
                torch.arange(579, 0, -1, device="cuda", dtype=torch.float32)
            )
            ti.fill_(-77)
            cu = torch.tensor([0, q], device="cuda", dtype=torch.int32)
            pre = torch.tensor([579 * 128 - q], device="cuda", dtype=torch.int32)
            self.op(scores, ti, ti, ti, cu, pre, q, 16, 0, 0, False, 579)
            expected, _ = cpu_oracle(scores, [q], [579 * 128 - q], 16, 0, 0)
            self.assertTrue(torch.equal(ti.cpu(), expected))
            self.__class__.cases += 1
            del scores, ti, cu, pre
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main(verbosity=2)
