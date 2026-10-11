"""Diagnostic smoke for the frozen Champion prefill's prior-GEMM NaN bug.

This intentionally expects the known defect. Once the kernel is fixed, change
the poisoned-run assertion to require finite output and keep the same fixture.
No model service or saved activation snapshot is needed.
"""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.champion_r8pkge import (
    make_chunk_runner,
)


class ChampionNonfiniteSmokeTest(unittest.TestCase):
    def test_prior_nan_gemm_triggers_nonfinite_attention(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")

        device = torch.device("cuda")
        torch.manual_seed(13)
        q = torch.randn(62, 64, 128, device=device, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        k = torch.randint(0, 128, (2, 4, 128, 64), device=device, dtype=torch.uint8)
        v = torch.randint(0, 128, (2, 4, 128, 64), device=device, dtype=torch.uint8)
        # E4M3 1.0, in the kernel's [pages * heads * tokens, groups] layout.
        ks = torch.full((1024, 8), 0x38, device=device, dtype=torch.uint8)
        vs = ks.clone()
        topk = torch.full((4, 62, 16), -1, device=device, dtype=torch.int32)
        topk[:, :, 0] = 0
        chunk = {
            "nb": 2,
            "cu_q": torch.tensor([0, 31, 62], device=device, dtype=torch.int32),
            "cu_k": torch.tensor([0, 62, 279], device=device, dtype=torch.int32),
            "seqused": torch.tensor([62, 217], device=device, dtype=torch.int32),
            "pt": torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0]], device=device, dtype=torch.int32),
            "max_q": 31,
        }
        runner = make_chunk_runner(q, k, v, ks, vs, 16, 128**-0.5, 128)
        self.assertIsNotNone(runner, "Champion rejected the smoke geometry")
        out = torch.empty(q.shape, device=device, dtype=torch.bfloat16)

        def nonfinite_count():
            out.zero_()
            runner(q, topk, chunk, out)
            torch.cuda.synchronize(device)
            return int((~torch.isfinite(out)).sum().item())

        self.assertEqual(nonfinite_count(), 0, "unpoisoned control is already invalid")
        self.assertTrue(torch.isfinite(q.float()).all().item())

        a = torch.full((2048, 2048), float("nan"), device=device, dtype=torch.bfloat16)
        b = torch.ones_like(a)
        gemm_out = a @ b
        torch.cuda.synchronize(device)
        self.assertNotEqual(gemm_out.data_ptr(), out.data_ptr())
        poisoned = nonfinite_count()
        print(f"Champion prior-NaN-GEMM smoke: {poisoned} nonfinite attention elements")
        self.assertGreater(poisoned, 0, "known Champion defect did not reproduce")
        self.assertTrue(torch.isfinite(runner.k8.float()).all().item())
        self.assertTrue(torch.isfinite(runner.v8.float()).all().item())

        finite_gemm_out = b @ b
        torch.cuda.synchronize(device)
        self.assertTrue(torch.isfinite(finite_gemm_out).all().item())
        self.assertEqual(nonfinite_count(), 0, "finite GEMM control did not recover")


if __name__ == "__main__":
    unittest.main()
