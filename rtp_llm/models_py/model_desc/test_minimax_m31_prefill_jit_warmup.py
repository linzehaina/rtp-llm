"""Model-local structural JIT warmup dispatch tests."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc.minimax_m31 import MiniMaxM31Model
from rtp_llm.models_py.modules.hybrid.dense_mlp import (
    DenseMLP,
    _FusedFp8QuantParams,
)


class MiniMaxM31PrefillJitWarmupTest(unittest.TestCase):
    def test_matches_production_specializations_and_deduplicates_dense(self):
        weights = tuple(torch.zeros(128, dtype=torch.bfloat16) for _ in range(4))
        attn = SimpleNamespace(
            nvfp4_kv_cache=True,
            _flashinfer_nvfp4_prefill=True,
            _m31_raw_attention_norms=weights,
            cos_sin_cache=torch.ones((1, 64), dtype=torch.float32),
            head_num=64,
            kv_head_num=4,
            num_idx_heads=4,
            layernorm_eps=1e-6,
        )
        dense = object.__new__(DenseMLP)
        torch.nn.Module.__init__(dense)
        dense._fuse_silu_quant = True
        dense._down_proj_fp8_quant_params = _FusedFp8QuantParams(
            32, False, True, True
        )
        dense.swiglu_oai_params = (1.702, 7.0)
        dense.down_proj = torch.nn.Module()
        dense.down_proj.K = 3072
        dense.down_proj.weight = torch.nn.Parameter(torch.zeros(1))
        model = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=attn)], modules=lambda: [dense, dense]
        )

        with (
            patch(
                "rtp_llm.models_py.triton_kernels.minimax_m31_gemma_rope."
                "minimax_m31_gemma_norm_rope_"
            ) as gemma,
            patch(
                "rtp_llm.models_py.triton_kernels.common.activation."
                "silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd"
            ) as silu,
            patch("torch.cuda.synchronize") as synchronize,
        ):
            MiniMaxM31Model._prewarm_prefill_structural_kernels(model)

        self.assertEqual(gemma.call_count, 2)
        self.assertEqual(silu.call_count, 3)
        self.assertEqual(synchronize.call_count, 2)
        self.assertEqual([call.args[0].shape[0] for call in gemma.call_args_list], [32, 2048])
        self.assertEqual(
            [call.args[0].shape[0] for call in silu.call_args_list],
            [32, 64, 8192],
        )
        for call in gemma.call_args_list:
            qkv, index_q, index_k = call.args[:3]
            self.assertEqual(qkv.stride(0), 9856)
            self.assertEqual(index_q.stride(0), 9856)
            self.assertEqual(index_k.stride(0), 9856)
            self.assertEqual(call.kwargs["eps"], 1e-6)
            self.assertEqual(call.kwargs["num_q_heads"], 64)
            self.assertEqual(call.kwargs["num_kv_heads"], 4)
            self.assertEqual(call.kwargs["num_index_heads"], 4)
            self.assertEqual(call.kwargs["prefill_outputs"][2].shape[1], 1152)
        for call in silu.call_args_list:
            self.assertEqual(call.args[0].shape[1], 6144)
            self.assertEqual(call.kwargs["quant_group_size"], 32)
            self.assertTrue(call.kwargs["tma_packed_scales"])

    def test_ineligible_paths_do_not_warm(self):
        attn = SimpleNamespace(
            nvfp4_kv_cache=True,
            _flashinfer_nvfp4_prefill=False,
            _m31_raw_attention_norms=None,
            cos_sin_cache=None,
        )
        model = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=attn)], modules=lambda: []
        )
        with patch("torch.cuda.synchronize") as synchronize:
            MiniMaxM31Model._prewarm_prefill_structural_kernels(model)
        synchronize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
