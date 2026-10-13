"""Byte-exact comparison with the independent, original five-launch path."""

import unittest
from types import SimpleNamespace

import torch
import triton

from rtp_llm.models_py.modules.glm5_mega_moe.input_packer_triton import (
    fused_pack_mega_moe_inputs,
    stage_shared_expert_input_scales,
)
from rtp_llm.models_py.modules.glm5_mega_moe.mega_nvfp4_input_packer_triton import (
    fused_pack_mega_nvfp4_inputs,
)
from rtp_llm.models_py.modules.glm5_mega_moe.mega_nvfp4_se_input_packer_triton import (
    fused_pack_mega_nvfp4_se_inputs,
)

FIELDS = (
    "x",
    "x_sf",
    "x_gsf",
    "topk_idx",
    "topk_weights",
    "shared_l1_acts",
    "shared_l1_acts_sf",
)


def allocate_buffer(capacity, hidden=6144, topk=4, strided=False):
    # Identical to the wheel's max padding over all candidate block_m values.
    rows = max(
        capacity // bm * triton.cdiv(bm, 128) * 128
        for bm in (8, 16, 32, 64, 96, 128, 192)
    )

    def matrix(width, dtype):
        return torch.empty_strided(
            (capacity, width),
            (width + (11 if strided else 0), 1),
            dtype=dtype,
            device="cuda",
        )

    return SimpleNamespace(
        x=matrix(hidden // 2, torch.int8),
        x_sf=matrix(hidden // 64, torch.int32),
        x_gsf=torch.empty(capacity, dtype=torch.float32, device="cuda"),
        topk_idx=matrix(topk, torch.int64),
        topk_weights=matrix(topk, torch.float32),
        shared_l1_acts=matrix(hidden, torch.float8_e4m3fn),
        shared_l1_acts_sf=torch.empty_strided(
            (rows, hidden // 128),
            (2, rows * 2 + 7) if strided else (1, rows),
            dtype=torch.int32,
            device="cuda",
        ),
        scratch=torch.empty(
            (capacity, hidden // 128), dtype=torch.int32, device="cuda"
        ),
    )


def legacy_pack(x, weights, indices, buf, tokens, block_m):
    """The production sequence before fusion; no new-kernel helpers used."""
    fused_pack_mega_nvfp4_inputs(
        x,
        weights,
        indices,
        buf.x[:tokens],
        buf.x_sf[:tokens],
        buf.x_gsf[:tokens],
        buf.topk_idx[:tokens],
        buf.topk_weights[:tokens],
    )
    fused_pack_mega_moe_inputs(
        x,
        weights,
        indices,
        buf.shared_l1_acts[:tokens],
        buf.scratch[:tokens],
        buf.topk_idx[:tokens],
        buf.topk_weights[:tokens],
    )
    buf.shared_l1_acts_sf.zero_()
    stage_shared_expert_input_scales(
        buf.scratch[:tokens], buf.shared_l1_acts_sf, tokens, block_m
    )


def assert_buffers_equal(test, expected, actual, tokens, block_m):
    for name in FIELDS:
        a, b = getattr(expected, name), getattr(actual, name)
        if name == "shared_l1_acts_sf":
            rows = triton.cdiv(tokens, block_m) * triton.cdiv(block_m, 128) * 128
            a, b = a[:rows], b[:rows]
        # Compare FP8 bytes (including signed zeros) and FP32 bit patterns too.
        if a.dtype == torch.float8_e4m3fn:
            a, b = a.view(torch.uint8), b.view(torch.uint8)
        elif a.dtype == torch.float32:
            a, b = a.view(torch.int32), b.view(torch.int32)
        test.assertTrue(torch.equal(a, b), f"byte mismatch in {name}")


class NVFP4SEInputPackerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("NVFP4 SE pack numerical tests require a Blackwell GPU")
        torch.manual_seed(31013)

    def inputs(self, tokens, hidden=6144, topk=4, strided=False):
        x = torch.empty_strided(
            (tokens, hidden),
            (hidden + (13 if strided else 0), 1),
            dtype=torch.bfloat16,
            device="cuda",
        ).normal_()
        weights = torch.empty_strided(
            (tokens, topk), (topk + (9 if strided else 0), 1), device="cuda"
        ).uniform_()
        indices = torch.empty_strided(
            (tokens, topk),
            (topk + (9 if strided else 0), 1),
            dtype=torch.int64,
            device="cuda",
        ).random_(128)
        if tokens:
            x[0].zero_()
            x[0, 0] = -0.0
        if tokens > 1:
            x[1].mul_(1e-6)
        if tokens > 2:
            x[2].mul_(1e5)
        if tokens > 3:
            # Force row scale to depend on the final H block, not a K tile.
            x[3].fill_(1e-5)
            x[3, -1] = 16384
        return x, weights, indices

    def compare(self, tokens, capacity, block_m, hidden=6144, topk=4, strided=False):
        old, new = (allocate_buffer(capacity, hidden, topk, strided) for _ in range(2))
        for name in FIELDS:
            a, b = getattr(old, name), getattr(new, name)
            a.fill_(3)
            b.copy_(a)
        args = self.inputs(tokens, hidden, topk, strided)
        legacy_pack(*args, old, tokens, block_m)
        fused_pack_mega_nvfp4_se_inputs(*args, new, tokens, block_m)
        assert_buffers_equal(self, old, new, tokens, block_m)
        read_rows = triton.cdiv(tokens, block_m) * triton.cdiv(block_m, 128) * 128
        self.assertTrue(
            bool(torch.all(new.shared_l1_acts_sf[read_rows:] == 3)),
            "unread SF tail changed",
        )

    def test_rounding_boundaries_signed_zero_and_tiny_groups(self):
        tokens, hidden, bm = 160, 6144, 96
        old, new = (allocate_buffer(16512) for _ in range(2))
        for name in FIELDS:
            getattr(old, name).fill_(3)
            getattr(new, name).copy_(getattr(old, name))
        ladder = torch.tensor(
            [
                0.0,
                -0.0,
                0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5.0,
                6.0,
                -0.25,
                -0.75,
                -1.25,
                -1.75,
                -2.5,
                -5.0,
            ],
            dtype=torch.bfloat16,
            device="cuda",
        )
        x = ladder.repeat(tokens, hidden // 16)
        x[:, 128:256].mul_(1e-6)
        x[:, -1] = 2688  # exactly unit GSF; the earlier K16 groups have unit SF.
        w = torch.rand(tokens, 4, device="cuda")
        i = torch.randint(128, (tokens, 4), device="cuda")
        legacy_pack(x, w, i, old, tokens, bm)
        fused_pack_mega_nvfp4_se_inputs(x, w, i, new, tokens, bm)
        assert_buffers_equal(self, old, new, tokens, bm)
        self.assertTrue(bool(torch.all(new.x_gsf[:tokens] == 1)))

    def test_all_block_m_tails_and_strides(self):
        for bm in (8, 16, 32, 64, 96, 128, 192):
            for tokens in (0, 1, bm - 1, bm, bm + 1, 2 * bm + 1):
                for strided in (False, True):
                    with self.subTest(block_m=bm, tokens=tokens, strided=strided):
                        cap = triton.cdiv(max(tokens, 1), 384) * 384
                        self.compare(tokens, cap, bm, strided=strided)

    def test_actual_ep8_capacity_and_ep2_heuristic(self):
        import deep_gemm

        for ep, cap in ((8, 16512), (2, 8448)):
            for tokens in (32, 160, 257, 1023, 1024, 2049, 8192):
                bm = deep_gemm.get_block_m_for_mega_moe_nvfp4(ep, 128, cap, tokens, 4)
                with self.subTest(ep=ep, capacity=cap, tokens=tokens, block_m=bm):
                    self.compare(tokens, cap, bm)

    def test_other_aligned_hidden_and_topk(self):
        for hidden, topk in ((128, 1), (384, 7), (8192, 31)):
            self.compare(33, 384, 32, hidden, topk, strided=True)

    def test_reused_buffer_changed_rows_and_cuda_graph(self):
        old, new = (allocate_buffer(16512) for _ in range(2))
        for name in FIELDS:
            getattr(old, name).fill_(5)
            getattr(new, name).copy_(getattr(old, name))
        # Warm/capture per-shape graphs, then reuse the same output allocation.
        cases = []
        for tokens, bm in (
            (257, 192),
            (1, 8),
            (160, 96),
            (33, 32),
            (1023, 96),
            (1024, 192),
            (2049, 192),
            (0, 16),
        ):
            args = self.inputs(tokens)
            fused_pack_mega_nvfp4_se_inputs(*args, new, tokens, bm)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused_pack_mega_nvfp4_se_inputs(*args, new, tokens, bm)
            cases.append((args, tokens, bm, graph))
        for name in FIELDS:
            getattr(old, name).fill_(5)
            getattr(new, name).copy_(getattr(old, name))
        for repeat in range(3):
            for args, tokens, bm, graph in cases:
                x, w, i = args
                x.normal_() if repeat < 2 else x.zero_()
                w.uniform_()
                i.random_(128)
                legacy_pack(*args, old, tokens, bm)
                graph.replay()
                assert_buffers_equal(self, old, new, tokens, bm)


if __name__ == "__main__":
    unittest.main()
