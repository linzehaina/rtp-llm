"""CP1 draft commit slot mapping matches the original Torch contract."""

import types
import unittest

import torch
import triton

from rtp_llm.models_py.model_desc.minimax_m31_dspark import (
    MiniMaxM31DSparkModel,
    _map_commit_slots_cp1_kernel,
)
from rtp_llm.models_py.speculative.dspark_proposer_mixin import map_context_rows


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DSparkCommitSlotsTest(unittest.TestCase):
    def setUp(self):
        self.model = types.SimpleNamespace(cache_cp_size=1, page_size=128)

    def reference(self, requests, positions, table):
        valid = (requests >= 0) & (positions >= 0)
        safe_req = requests.clamp(min=0).to(torch.int64)
        safe_pos = positions.clamp(min=0).to(torch.int64)
        blocks = safe_pos // 128
        valid &= (safe_req < table.shape[0]) & (blocks < table.shape[1])
        if table.numel() == 0:
            return torch.full_like(safe_pos, -1), torch.zeros_like(valid)
        physical = table[
            safe_req.clamp(max=table.shape[0] - 1),
            blocks.clamp(max=table.shape[1] - 1),
        ]
        slots = physical.to(torch.int64) * 128 + safe_pos % 128
        valid &= physical >= 0
        return torch.where(valid, slots, -1), valid

    def check(self, requests, positions, table):
        expected = self.reference(requests, positions, table)
        actual = MiniMaxM31DSparkModel._commit_slots(
            self.model, requests, positions, table
        )
        for a, b in zip(expected, actual):
            self.assertTrue(torch.equal(a, b))

    def test_invalid_rows_strides_and_empty_table(self):
        torch.manual_seed(20261013)
        for n in (0, 1, 7, 224, 257):
            with self.subTest(n=n):
                pairs = torch.randint(-3, 9, (n, 2), device="cuda", dtype=torch.int32)
                requests, positions = pairs.unbind(1)
                if n:
                    positions[::3] = torch.randint(
                        0, 80000, (positions[::3].numel(),), device="cuda"
                    )
                base = torch.randint(-1, 700, (8, 18), device="cuda", dtype=torch.int32)
                table = base[:, ::2]
                self.check(requests, positions, table)
                self.check(requests, positions, table[:, :0])
                self.check(requests, positions, table[:0, :])
                self.check(
                    requests.to(torch.int64),
                    positions.to(torch.int64),
                    table.to(torch.int64),
                )

    def test_graph_replay_reads_fresh_rows_and_pages(self):
        pairs = torch.randint(0, 32, (224, 2), device="cuda", dtype=torch.int32)
        table = torch.randint(0, 20000, (32, 628), device="cuda", dtype=torch.int32)
        requests, positions = pairs.unbind(1)
        positions.mul_(128)
        self.check(requests, positions, table)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = MiniMaxM31DSparkModel._commit_slots(
                self.model, requests, positions, table
            )
        for seed in (41, 42, 43):
            torch.manual_seed(seed)
            pairs[:, 0].random_(-2, 35)
            pairs[:, 1].random_(-128, 81000)
            table.random_(-1, 20000)
            graph.replay()
            expected = self.reference(requests, positions, table)
            for a, b in zip(expected, actual):
                self.assertTrue(torch.equal(a, b))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DSparkCommitRowsTest(unittest.TestCase):
    def setUp(self):
        self.model = types.SimpleNamespace(cp_enabled=False)

    def fused(self, starts, lengths, ends, row_count):
        requests, positions, ctx = MiniMaxM31DSparkModel.map_commit_rows(
            self.model, starts, lengths, ends, row_count, None
        )
        self.assertTrue(all(a is b for a, b in zip(ctx, (starts, lengths, ends))))
        table = torch.zeros(
            (starts.numel(), 628), dtype=torch.int32, device=starts.device
        )
        slots = torch.empty(row_count, dtype=torch.int64, device=starts.device)
        valid = torch.empty(row_count, dtype=torch.bool, device=starts.device)
        _map_commit_slots_cp1_kernel[(triton.cdiv(row_count, 128),)](
            starts,
            lengths,
            ends,
            table,
            requests,
            positions,
            slots,
            valid,
            ROW_COUNT=row_count,
            BATCH=starts.numel(),
            PAGE=128,
            TABLE_ROWS=table.shape[0],
            TABLE_COLS=table.shape[1],
            SS=starts.stride(0),
            LS=lengths.stride(0),
            ES=ends.stride(0),
            TRS=table.stride(0),
            TCS=table.stride(1),
            SEARCH_STEPS=starts.numel().bit_length(),
            BLOCK=128,
            num_warps=4,
        )
        return requests, positions, slots, valid

    def expected(self, starts, lengths, ends, row_count):
        requests, positions = map_context_rows(starts, lengths, ends, row_count)
        valid = (
            (requests >= 0)
            & (requests < starts.numel())
            & (positions >= 0)
            & (positions // 128 < 628)
        )
        slots = torch.where(valid, positions % 128, -1)
        return requests, positions, slots, valid

    def check(self, starts, lengths, ends, row_count):
        expected = self.expected(starts, lengths, ends, row_count)
        actual = self.fused(starts, lengths, ends, row_count)
        for a, b in zip(expected, actual):
            self.assertTrue(torch.equal(a, b))

    def test_padding_zero_lengths_and_strides(self):
        for batch in (1, 2, 7, 32):
            lengths = torch.randint(0, 8, (batch,), device="cuda", dtype=torch.int64)
            lengths[0] = 0
            starts = lengths.cumsum(0) - lengths
            ends = (
                torch.randint(0, 1000, (batch,), device="cuda", dtype=torch.int64)
                + lengths
            )
            rows = int(lengths.sum().item())
            for row_count in sorted({1, max(1, rows), rows + 4, 224}):
                with self.subTest(batch=batch, row_count=row_count):
                    self.check(starts, lengths, ends, row_count)
                    self.check(
                        torch.stack((starts, starts), dim=1)[:, 0],
                        torch.stack((lengths, lengths), dim=1)[:, 0],
                        torch.stack((ends, ends), dim=1)[:, 0],
                        row_count,
                    )

    def test_graph_replay_reads_updated_metadata(self):
        lengths = torch.ones(32, dtype=torch.int64, device="cuda")
        starts = torch.empty_like(lengths)
        ends = torch.empty_like(lengths)
        graph = torch.cuda.CUDAGraph()
        starts.copy_(lengths.cumsum(0) - lengths)
        ends.copy_(lengths + 1000)
        self.check(starts, lengths, ends, 224)
        with torch.cuda.graph(graph):
            actual = self.fused(starts, lengths, ends, 224)
        for seed in (3, 7, 11):
            torch.manual_seed(seed)
            lengths.random_(0, 8)
            starts.copy_(lengths.cumsum(0) - lengths)
            ends.copy_(lengths + 1000 + seed)
            graph.replay()
            expected = self.expected(starts, lengths, ends, 224)
            for a, b in zip(expected, actual):
                self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
