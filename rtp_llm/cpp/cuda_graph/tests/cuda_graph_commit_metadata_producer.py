"""Native runner gate for the COMMIT metadata producer.

This exercises actual CudaGraphRunner preparation/replay. It does not instantiate
PyWrappedModel and cannot establish its buildPyAttentionInputs bypass or native
AsyncRunner lifecycle; those are separate delivery gates.
"""
import unittest

import torch

from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs, PyModelOutputs, get_typemeta

TABLES = ("kv_cache_kernel_block_id_device", "kv_cache_block_id_device")
GENERATED = ("cu_seqlens", "cu_kv_seqlens", "input_lengths", "prefix_lengths", "sequence_lengths_plus_1_d")
OBSERVED = TABLES + GENERATED + ("sequence_lengths", "decode_cu_seqlens_d", "padding_offset")


class _Attention:
    def __init__(self, owner):
        self.owner = owner

    def prepare_cuda_graph(self, inputs):
        self.owner.prepare_calls += 1
        if self.owner.raise_prepare:
            raise RuntimeError("injected COMMIT preparation failure")
        self.owner.last_scalars = (inputs.total_tokens, inputs.is_prefill, inputs.is_target_verify)


class _Echo:
    def __init__(self):
        self.captured = {}
        self.prepare_calls = 0
        self.raise_prepare = False
        self.last_scalars = None

    def prepare_fmha_impl(self, _inputs, _capture):
        return _Attention(self)

    def forward(self, inputs, _attention=None):
        attn = inputs.attention_inputs
        self.captured[attn.input_lengths.numel()] = {name: getattr(attn, name) for name in OBSERVED}
        # Every device metadata destination contributes to the graph result;
        # CPU padding is checked byte-for-byte outside the graph.
        checksum = sum(t.sum(dtype=torch.int64) for t in self.captured[attn.input_lengths.numel()].values() if isinstance(t, torch.Tensor) and t.is_cuda).to(torch.float32)
        return PyModelOutputs(inputs.input_hiddens + checksum)


class TestCommitMetadataProducer(unittest.TestCase):
    width = 5
    hidden = 8
    poison = -77

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip:
            raise unittest.SkipTest("native COMMIT producer requires CUDA")

    def setUp(self):
        self.owned = []

    def tearDown(self):
        torch.cuda.synchronize()
        self.owned.clear()

    def _runner(self, enabled=True, exact=True, target=True):
        model, runner = _Echo(), CudaGraphRunner()
        runner.init_decode(model, self.hidden, 1024, 128, 128, [2, 4], "fp32", self.width, target, False, exact,
                           commit_table_overwrite=True, commit_metadata_producer=enabled)
        self.owned.append((runner, model))
        torch.cuda.synchronize()
        return runner, model

    def _inputs(self, model, rows=4, step=0, mode="dense", compact=True):
        tokens = rows * self.width
        inputs = PyModelInputs()
        inputs.input_ids = torch.arange(tokens, device="cuda", dtype=torch.int32)
        inputs.input_hiddens = torch.arange(tokens * self.hidden, device="cuda", dtype=torch.float32).reshape(tokens, self.hidden)
        attn = PyAttentionInputs()
        attn.input_lengths = torch.full((rows,), self.width, dtype=torch.int32, device="cuda")
        attn.prefix_lengths = torch.arange(rows, dtype=torch.int32, device="cuda") + 8 + step * 3
        attn.sequence_lengths = torch.empty(0, dtype=torch.int32, device="cuda")
        attn.is_prefill, attn.is_target_verify, attn.is_ragged_target_verify = True, True, False
        attn.total_tokens = tokens
        attn.context_total_kv_length = tokens
        attn.dtype = get_typemeta(torch.zeros(1, dtype=torch.float32))
        key = rows if rows in model.captured else 4
        cols = model.captured[key][TABLES[0]].shape[1]
        source_cols = cols + 1 if mode == "capacity" else max(cols - 2, 1)
        for table_id, name in enumerate(TABLES):
            backing = torch.full((rows, source_cols + 3), 123456, dtype=torch.int32, device="cuda")
            source = backing[:, :source_cols]
            source.copy_(torch.arange(source.numel(), device="cuda", dtype=torch.int32).reshape(source.shape) + 11 * (step + 1) + table_id)
            if mode == "alias" and table_id == 0:
                source = model.captured[key][name][:rows, :source_cols]
            setattr(attn, name, source)
            host_name = "kv_cache_kernel_block_id_host" if table_id == 0 else "kv_cache_block_id_host"
            setattr(attn, host_name, source.cpu().pin_memory())
        if not compact:
            self._legacy_metadata(attn, rows)
        inputs.attention_inputs = attn
        return inputs

    def _legacy_metadata(self, attn, rows):
        attn.cu_seqlens = torch.arange(rows + 1, dtype=torch.int32, device="cuda") * self.width
        attn.cu_kv_seqlens = torch.cat((torch.zeros(1, dtype=torch.int32, device="cuda"), (attn.prefix_lengths + attn.input_lengths).cumsum(0, dtype=torch.int32)))
        attn.sequence_lengths_plus_1_d = attn.prefix_lengths + 1
        attn.cu_seqlens_host = attn.cu_seqlens.cpu().pin_memory()
        attn.padding_offset = torch.zeros(rows * self.width, dtype=torch.int32, device="cuda")

    def _snapshot(self, captured):
        return {name: tensor.cpu().clone() for name, tensor in captured.items()}

    def _expect(self, inputs, captured):
        attn = inputs.attention_inputs
        rows = attn.input_lengths.numel()
        result = self._snapshot(captured)
        for name in TABLES:
            result[name].zero_()
            src = getattr(attn, name).cpu()
            nr, nc = min(src.shape[0], result[name].shape[0]), min(src.shape[1], result[name].shape[1])
            result[name][:nr, :nc] = src[:nr, :nc]
        lens, prefixes = attn.input_lengths.cpu(), attn.prefix_lengths.cpu()
        values = {
            "input_lengths": lens, "prefix_lengths": prefixes,
            "sequence_lengths_plus_1_d": (attn.sequence_lengths_plus_1_d.cpu() if attn.sequence_lengths_plus_1_d is not None and attn.sequence_lengths_plus_1_d.numel() else prefixes + 1),
            "cu_seqlens": torch.cat((torch.zeros(1, dtype=torch.int32), lens.cumsum(0, dtype=torch.int32))),
            "cu_kv_seqlens": torch.cat((torch.zeros(1, dtype=torch.int32), (lens + prefixes).cumsum(0, dtype=torch.int32))),
        }
        for name, value in values.items():
            result[name][:value.numel()] = value
        if rows < result["input_lengths"].numel():
            for name in ("input_lengths", "prefix_lengths", "sequence_lengths_plus_1_d", "sequence_lengths"):
                result[name][rows:] = 0
            for name in ("cu_seqlens", "decode_cu_seqlens_d"):
                result[name][rows + 1:] = rows * self.width
        return result

    def _assert_snapshot(self, captured, expected):
        for name, value in self._snapshot(captured).items():
            self.assertTrue(torch.equal(value.view(torch.uint8), expected[name].view(torch.uint8)), name)

    def _run_fast(self, runner, model, rows=4, step=0):
        captured = model.captured[rows if rows in model.captured else 4]
        for tensor in captured.values():
            tensor.fill_(self.poison)
        inputs = self._inputs(model, rows, step)
        expected = self._expect(inputs, captured)
        outside = self._snapshot(model.captured[4]) if rows == 2 else None
        source_tables = {name: getattr(inputs.attention_inputs, name).clone() for name in TABLES}
        calls = model.prepare_calls
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
            self.assertTrue(runner.tryPrepareCommitMetadata(inputs))
            output = runner.forward(inputs).hidden_states.clone()
            torch.cuda.synchronize()
        self.assertGreater(model.prepare_calls, calls)
        names = [event.name for event in profile.events()]
        self.assertTrue(any("CommitMetadata" in name or "commitMetadata" in name for name in names), names)
        self._assert_snapshot(captured, expected)
        if outside is not None:
            for name, value in self._snapshot(model.captured[4]).items():
                start = captured[name].numel()
                self.assertTrue(torch.equal(value.reshape(-1)[start:], outside[name].reshape(-1)[start:]), name + " cross-bucket canary")
        for name, source in source_tables.items():
            self.assertTrue(torch.equal(getattr(inputs.attention_inputs, name), source), name + " source canary")
        checksum = sum(expected[name].sum(dtype=torch.int64) for name, tensor in captured.items() if tensor.is_cuda).to(torch.float32)
        self.assertTrue(torch.equal(output.cpu(), inputs.input_hiddens.cpu() + checksum))
        self.assertEqual(model.last_scalars, (rows * self.width, True, True))

    def test_four_fresh_rounds_and_cross_capture_keys(self):
        runner, model = self._runner()
        for step, rows in enumerate((4, 2, 4, 2)):
            self._run_fast(runner, model, rows, step)

    def test_explicit_sequence_plus_one_and_metadata_alias_rejection(self):
        runner, model = self._runner()
        inputs = self._inputs(model)
        inputs.attention_inputs.sequence_lengths_plus_1_d = inputs.attention_inputs.prefix_lengths + 37
        expected = self._expect(inputs, model.captured[4])
        self.assertTrue(runner.tryPrepareCommitMetadata(inputs))
        runner.forward(inputs)
        torch.cuda.synchronize()
        self._assert_snapshot(model.captured[4], expected)
        # Source aliases a producer-written destination: reject before writes.
        inputs = self._inputs(model)
        inputs.attention_inputs.prefix_lengths = model.captured[4]["prefix_lengths"]
        before = self._snapshot(model.captured[4])
        self.assertFalse(runner.tryPrepareCommitMetadata(inputs))
        self._assert_snapshot(model.captured[4], before)
        self._run_fast(runner, model, step=2)

    def test_fallback_then_fast_reentry(self):
        runner, model = self._runner()
        self._run_fast(runner, model)
        for mode in ("alias", "capacity"):
            inputs = self._inputs(model, mode=mode)
            before = self._snapshot(model.captured[4])
            self.assertFalse(runner.tryPrepareCommitMetadata(inputs), mode)
            self._assert_snapshot(model.captured[4], before)
            self._legacy_metadata(inputs.attention_inputs, 4)
            self.assertTrue(runner.canRun(inputs))
            expected = self._expect(inputs, model.captured[4])
            if mode == "alias":
                expected[TABLES[0]].zero_()
            runner.forward(inputs)
            torch.cuda.synchronize()
            self._assert_snapshot(model.captured[4], expected)
            self._run_fast(runner, model, step=3)

    def test_negative_capability_and_payload_geometry(self):
        runner, model = self._runner(enabled=False)
        inputs = self._inputs(model)
        before = self._snapshot(model.captured[4])
        self.assertFalse(runner.tryPrepareCommitMetadata(inputs))
        self._assert_snapshot(model.captured[4], before)
        self._legacy_metadata(inputs.attention_inputs, 4)
        self.assertTrue(runner.canRun(inputs))
        runner.forward(inputs)
        runner, model = self._runner(target=False)
        self.assertFalse(runner.tryPrepareCommitMetadata(self._inputs(model)))
        runner, model = self._runner()
        for change in ("hidden_width", "hidden_rows", "capture_key", "generic"):
            inputs = self._inputs(model, rows=3 if change == "capture_key" else 4)
            if change == "hidden_width":
                inputs.input_hiddens = inputs.input_hiddens[:, :-1].contiguous()
            elif change == "hidden_rows":
                inputs.input_hiddens = inputs.input_hiddens[:-1]
            elif change == "generic":
                inputs.attention_inputs.is_target_verify = False
            before = self._snapshot(model.captured[4])
            self.assertFalse(runner.tryPrepareCommitMetadata(inputs), change)
            self._assert_snapshot(model.captured[4], before)
        self._run_fast(runner, model, step=2)

    def test_padded_runner_tail_semantics(self):
        runner, model = self._runner(exact=False)
        for tensor in model.captured[4].values():
            tensor.fill_(self.poison)
        inputs = self._inputs(model, rows=3)
        self.assertTrue(runner.canRun(inputs))
        expected = self._expect(inputs, model.captured[4])
        self.assertTrue(runner.tryPrepareCommitMetadata(inputs))
        runner.forward(inputs)
        torch.cuda.synchronize()
        self._assert_snapshot(model.captured[4], expected)
        self._run_fast(runner, model)

    def test_side_stream_input_ready_and_delayed_consumer(self):
        # Real runner on a second CUDA stream; this is not native AsyncRunner.
        runner, model = self._runner()
        main, worker = torch.cuda.current_stream(), torch.cuda.Stream()
        inputs = self._inputs(model)
        expected = self._expect(inputs, model.captured[4])
        torch.cuda._sleep(2_000_000)
        inputs.attention_inputs.prefix_lengths.add_(9)
        expected["prefix_lengths"] += 9
        expected["sequence_lengths_plus_1_d"] += 9
        expected["cu_kv_seqlens"] += torch.arange(5, dtype=torch.int32) * 9
        ready, consumed = torch.cuda.Event(), torch.cuda.Event()
        ready.record(main)
        with torch.cuda.stream(worker):
            worker.wait_event(ready)
            self.assertTrue(runner.tryPrepareCommitMetadata(inputs))
            consumed.record(worker)
        main.wait_event(consumed)
        next_inputs = self._inputs(model, step=2)
        old_output = runner.forward(inputs).hidden_states
        # Consume the actual captured output view after a delay, before reuse.
        # Build next input host mirrors first so their .cpu() cannot hide the delay.
        torch.cuda._sleep(2_000_000)
        delayed = old_output.clone()
        ready.record(main)
        with torch.cuda.stream(worker):
            worker.wait_event(ready)
            self.assertTrue(runner.tryPrepareCommitMetadata(next_inputs))
            consumed.record(worker)
        main.wait_event(consumed)
        runner.forward(next_inputs)
        torch.cuda.synchronize()
        checksum = sum(expected[name].sum(dtype=torch.int64) for name, tensor in model.captured[4].items() if tensor.is_cuda).to(torch.float32)
        self.assertTrue(torch.equal(delayed.cpu(), inputs.input_hiddens.cpu() + checksum))

    def test_source_owner_release_with_delayed_producer_and_allocator_pressure(self):
        runner, model = self._runner()
        inputs = self._inputs(model)
        expected = self._expect(inputs, model.captured[4])
        cols = inputs.attention_inputs.kv_cache_block_id_device.shape[1]
        main, worker = torch.cuda.current_stream(), torch.cuda.Stream()
        ready, consumed = torch.cuda.Event(), torch.cuda.Event()
        ready.record(main)
        with torch.cuda.stream(worker):
            worker.wait_event(ready)
            # Long enough to release the main-stream owners while native raw
            # pointer reads are pending; no global synchronize hides that gap.
            torch.cuda._sleep(100_000_000)
            self.assertTrue(runner.tryPrepareCommitMetadata(inputs))
            consumed.record(worker)
        self.assertFalse(consumed.query(), "delayed producer completed before ownership test")
        del inputs
        # Inputs are no longer retained by this test. The runner stores weak
        # identities; allocator recordStream protects its pending device reads.
        pressure = []
        for _ in range(64):
            pressure.append(torch.full((4, cols + 3), -998, dtype=torch.int32, device="cuda"))
            pressure.append(torch.full((4,), -999, dtype=torch.int32, device="cuda"))
        main.wait_event(consumed)
        torch.cuda.synchronize()
        self._assert_snapshot(model.captured[4], expected)
        del pressure
        self._run_fast(runner, model, step=2)

    def test_inplace_metadata_and_semantic_source_edits(self):
        runner, model = self._runner()
        inputs = self._inputs(model)
        self.assertTrue(runner.tryPrepareCommitMetadata(inputs))
        inputs.attention_inputs.prefix_lengths.add_(11)
        expected = self._expect(inputs, model.captured[4])
        calls = model.prepare_calls
        self.assertTrue(runner.canRun(inputs))
        runner.forward(inputs)
        torch.cuda.synchronize()
        self.assertGreater(model.prepare_calls, calls)
        self._assert_snapshot(model.captured[4], expected)
        # A source classification change must not reuse a COMMIT prepared flag.
        inputs = self._inputs(model)
        self.assertTrue(runner.tryPrepareCommitMetadata(inputs))
        self._legacy_metadata(inputs.attention_inputs, 4)
        inputs.attention_inputs.is_prefill = False
        calls = model.prepare_calls
        self.assertTrue(runner.canRun(inputs))
        runner.forward(inputs)
        torch.cuda.synchronize()
        self.assertGreater(model.prepare_calls, calls)
        self._run_fast(runner, model, step=3)

    def test_stale_prepared_capture_key_rebuilds_metadata(self):
        runner, model = self._runner()
        self.assertTrue(runner.tryPrepareCommitMetadata(self._inputs(model, rows=4)))
        inputs = self._inputs(model, rows=2, step=3, compact=False)
        expected = self._expect(inputs, model.captured[2])
        calls = model.prepare_calls
        self.assertTrue(runner.canRun(inputs))
        runner.forward(inputs)
        torch.cuda.synchronize()
        self.assertGreater(model.prepare_calls, calls)
        self._assert_snapshot(model.captured[2], expected)

    def test_prepare_exception_then_retry(self):
        runner, model = self._runner()
        inputs = self._inputs(model)
        model.raise_prepare = True
        with self.assertRaisesRegex(RuntimeError, "injected COMMIT"):
            runner.tryPrepareCommitMetadata(inputs)
        model.raise_prepare = False
        legacy = self._inputs(model, rows=2, step=3, compact=False)
        expected = self._expect(legacy, model.captured[2])
        self.assertTrue(runner.canRun(legacy))
        runner.forward(legacy)
        torch.cuda.synchronize()
        self._assert_snapshot(model.captured[2], expected)
        self._run_fast(runner, model, step=2)


if __name__ == "__main__":
    unittest.main()
