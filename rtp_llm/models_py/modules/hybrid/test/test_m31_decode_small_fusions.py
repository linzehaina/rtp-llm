"""Decode-shape numerical and replay checks for small fused kernels."""

import unittest

import torch

from rtp_llm.models_py.kernels.cuda.mxfp8_ops import pack_mxfp8_scale
from rtp_llm.models_py.modules.hybrid.minimax_m31_dspark_layer import (
    _add_and_mask_proposals,
)
from rtp_llm.models_py.triton_kernels.common.activation import (
    silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DecodeSmallFusionTest(unittest.TestCase):
    def test_grouped_swiglu_matches_row_group_reference_and_graph_replay(self):
        torch.manual_seed(20261012)
        for rows, width in ((160, 3072), (200, 3072), (224, 12288), (280, 12288)):
            with self.subTest(rows=rows, width=width):
                x = torch.randn(rows, 2 * width, dtype=torch.bfloat16, device="cuda")

                def reference():
                    q, scales = silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd(
                        x, 32, False, True, 1.702, 7.0, False
                    )
                    return q, pack_mxfp8_scale(scales, rows, width)

                def grouped():
                    return silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd(
                        x, 32, False, True, 1.702, 7.0, True
                    )

                grouped()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replayed = grouped()
                for small in (False, True):
                    if small:
                        x.normal_().mul_(1e-5)
                    expected = reference()
                    actual = grouped()
                    graph.replay()
                    for a, b in zip(expected, actual):
                        self.assertTrue(
                            torch.equal(
                                a.contiguous().view(torch.uint8),
                                b.contiguous().view(torch.uint8),
                            )
                        )
                    for a, b in zip(expected, replayed):
                        self.assertTrue(
                            torch.equal(
                                a.contiguous().view(torch.uint8),
                                b.contiguous().view(torch.uint8),
                            )
                        )

    def test_proposal_add_mask_matches_torch_for_variable_lengths(self):
        torch.manual_seed(20261013)
        for batch, width in ((32, 7), (40, 7), (32, 5), (40, 5)):
            with self.subTest(batch=batch, width=width):
                hidden = torch.randn(
                    batch * width, 6144, device="cuda", dtype=torch.bfloat16
                )
                mlp = torch.randn_like(hidden)
                lens = torch.randint(
                    0, width + 1, (batch,), device="cuda", dtype=torch.int32
                )

                def reference():
                    valid = (
                        torch.arange(width, device=hidden.device)[None, :]
                        < lens[:, None]
                    )
                    return torch.where(
                        valid[:, :, None], (hidden + mlp).view(batch, width, 6144), 0
                    )

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = _add_and_mask_proposals(hidden, mlp, lens, width)
                for _ in range(3):
                    lens.random_(width + 1)
                    hidden.normal_()
                    mlp.normal_()
                    graph.replay()
                    self.assertTrue(
                        torch.equal(
                            result.view(torch.uint8), reference().view(torch.uint8)
                        )
                    )


if __name__ == "__main__":
    unittest.main()
