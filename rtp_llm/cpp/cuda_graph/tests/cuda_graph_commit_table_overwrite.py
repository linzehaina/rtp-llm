"""Actual CudaGraphRunner COMMIT preparation gate; no checkpoint/model weights.

This checks captured buffers and GPU route names, not only the isolated kernel.
Run after rebuilding libtest_cuda_graph_runner with its new default-false argument.
"""
import unittest

import torch

from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    get_typemeta,
)


_TABLES = ("kv_cache_kernel_block_id_device", "kv_cache_block_id_device")
_METADATA = (
    "cu_seqlens",
    "cu_kv_seqlens",
    "input_lengths",
    "prefix_lengths",
    "sequence_lengths_plus_1_d",
)
_OBSERVED = _TABLES + _METADATA + ("padding_offset",)


class _NoopAttention:
    def prepare_cuda_graph(self, _attention_inputs):
        return None


class _CommitTableEcho:
    """Retain real capture input views, and read tables/metadata in the graph."""

    def __init__(self):
        self.captured = {}

    def prepare_fmha_impl(self, _inputs, _capture):
        return _NoopAttention()

    def forward(self, inputs, _attention=None):
        attn = inputs.attention_inputs
        self.captured[attn.input_lengths.numel()] = {
            name: getattr(attn, name) for name in _OBSERVED
        }
        # GPU operations remain captured: replay consumes the current buffer bytes.
        checksum = sum(
            getattr(attn, name).sum(dtype=torch.int64)
            for name in _TABLES + _METADATA
        ).to(torch.float32)
        return PyModelOutputs(inputs.input_hiddens + checksum)


class TestCudaGraphCommitTableOverwrite(unittest.TestCase):
    batch = 32
    width = 5  # static verify4 + anchor; the proposal backbone remains gamma7.
    hidden_size = 8
    capture_cols = 708  # ceil(90112 / 128) + static verify4
    poison = -777

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip:
            raise unittest.SkipTest("CUDA table overwrite gate requires CUDA")

    def setUp(self):
        self.models = [_CommitTableEcho(), _CommitTableEcho()]
        self.runners = [CudaGraphRunner(), CudaGraphRunner()]
        for arm, (runner, model) in enumerate(zip(self.runners, self.models)):
            runner.init_decode(
                model,
                self.hidden_size,
                90112,
                128,
                128,
                [self.batch],
                "fp32",
                self.width,
                True,
                False,
                # Allow a padded batch to exercise the preparation fallback.
                False,
                commit_table_overwrite=bool(arm),
            )
        torch.cuda.synchronize()

    def tearDown(self):
        torch.cuda.synchronize()
        self.runners = None
        self.models = None

    def _inputs(self, arm, round_id, mode="dense", rows=None):
        rows = self.batch if rows is None else rows
        tokens = rows * self.width
        inputs = PyModelInputs()
        inputs.input_ids = torch.arange(tokens, device="cuda", dtype=torch.int32)
        inputs.input_hiddens = torch.arange(
            tokens * self.hidden_size, device="cuda", dtype=torch.float32
        ).view(tokens, self.hidden_size)
        attn = PyAttentionInputs()
        attn.input_lengths = torch.full(
            (rows,), self.width, device="cuda", dtype=torch.int32
        )
        attn.prefix_lengths = torch.arange(
            rows, device="cuda", dtype=torch.int32
        ) + 8 + round_id
        attn.sequence_lengths = torch.empty(0, device="cuda", dtype=torch.int32)
        attn.sequence_lengths_plus_1_d = attn.prefix_lengths + 1
        cu = torch.arange(rows + 1, device="cuda", dtype=torch.int32) * self.width
        attn.cu_seqlens = cu
        attn.cu_kv_seqlens = torch.cat(
            (torch.zeros(1, device="cuda", dtype=torch.int32),
             (attn.prefix_lengths + attn.input_lengths).cumsum(0, dtype=torch.int32))
        )
        attn.cu_seqlens_host = cu.cpu().pin_memory()
        # Context COMMIT leaves decode_cu_seqlens_d undefined, as production does.
        cols = self.capture_cols + 1 if mode == "capacity" else 626
        sources = []
        for table_id in range(2):
            if mode == "fill_only":
                src = torch.empty((0, 0), device="cuda", dtype=torch.int32)
            else:
                base = torch.arange(
                    rows * (cols + 3), device="cuda", dtype=torch.int32
                ).view(rows, cols + 3)
                src = base[:, :cols] + (round_id + 1) * 11 + table_id * 3
                # Restore a real row-strided view after the arithmetic operation.
                backing = torch.empty_like(base)
                backing[:, :cols].copy_(src)
                src = backing[:, :cols]
                src[0, 0] = -(round_id + 1)
            sources.append(src)
        if mode == "alias":
            sources[0] = self.models[arm].captured[self.batch][_TABLES[0]][:, :626]
        for name, src in zip(_TABLES, sources):
            setattr(attn, name, src)
        attn.kv_cache_kernel_block_id_host = sources[0].cpu().pin_memory()
        attn.kv_cache_block_id_host = sources[1].cpu().pin_memory()
        attn.padding_offset = torch.zeros(tokens, dtype=torch.int32)
        attn.is_prefill = True
        attn.is_target_verify = True
        attn.is_ragged_target_verify = False
        attn.total_tokens = tokens
        attn.context_total_kv_length = tokens
        attn.dtype = get_typemeta(torch.zeros(1, dtype=torch.float32))
        inputs.attention_inputs = attn
        return inputs

    def _expected(self, inputs, captured, mode):
        """CPU reference for the legacy fill -> metadata -> rectangular copy."""
        attn = inputs.attention_inputs
        rows = attn.input_lengths.numel()
        expected = {
            name: torch.full_like(tensor.cpu(), self.poison)
            for name, tensor in captured.items()
        }
        for table_id, name in enumerate(_TABLES):
            expected[name].zero_()
            src = getattr(attn, name).cpu()
            if mode == "alias" and table_id == 0:
                # Legacy fill zeros this source view before its strided copy.
                src = torch.zeros_like(src)
            nrows = min(src.shape[0], expected[name].shape[0])
            ncols = min(src.shape[1], expected[name].shape[1])
            expected[name][:nrows, :ncols] = src[:nrows, :ncols]
        for name in _METADATA:
            src = getattr(attn, name).cpu()
            expected[name][:src.numel()] = src
        if rows < self.batch:
            expected["input_lengths"][rows:] = 0
            expected["prefix_lengths"][rows:] = 0
            expected["sequence_lengths_plus_1_d"][rows:] = 0
            expected["cu_seqlens"][rows + 1:] = rows * self.width
            # Non-prefill runner does not fill the cu_kv tail: retain poison.
        # padding_offset is CPU capture storage and has no decode-mode update.
        return expected

    def _run_pair(self, round_id, mode="dense", rows=None):
        observed = []
        for arm, (runner, model) in enumerate(zip(self.runners, self.models)):
            captured = model.captured[self.batch]
            # Poison all captured bytes before every step, including padded tails.
            for tensor in captured.values():
                tensor.fill_(self.poison)
            inputs = self._inputs(arm, round_id, mode, rows)
            expected = self._expected(inputs, captured, mode)
            self.assertTrue(runner.canRun(inputs))
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                output = runner.forward(inputs)
                torch.cuda.synchronize()
            names = [event.name for event in prof.events()]
            routed = any("cudaGraphTableOverwriteKernel" in name for name in names)
            should_route = arm == 1 and mode in ("dense", "fill_only") and (
                rows is None or rows == self.batch
            )
            self.assertEqual(routed, should_route,
                             f"arm={arm}, mode={mode}, route kernel missing/unexpected")
            snapshot = {}
            for name, tensor in captured.items():
                value = tensor.cpu().clone()
                # Int32 byte equality, without probability tolerances.
                self.assertTrue(torch.equal(value.view(torch.uint8),
                                            expected[name].view(torch.uint8)), name)
                snapshot[name] = value
            checksum = sum(snapshot[name].sum(dtype=torch.int64)
                           for name in _TABLES + _METADATA).to(torch.float32)
            self.assertTrue(torch.equal(output.hidden_states.cpu(),
                                        inputs.input_hiddens.cpu() + checksum))
            observed.append(snapshot)
        for name in _OBSERVED:
            self.assertTrue(torch.equal(observed[0][name].view(torch.uint8),
                                        observed[1][name].view(torch.uint8)), name)

    def test_fresh_replay_whole_tables_and_five_metadata(self):
        for round_id in range(4):
            self._run_pair(round_id)

    def test_fill_only_and_fallback_then_fast_reentry(self):
        self._run_pair(0, "fill_only")
        self._run_pair(1, "alias")
        self._run_pair(2, "capacity")
        self._run_pair(3, rows=self.batch - 1)
        self._run_pair(4)


if __name__ == "__main__":
    unittest.main()
