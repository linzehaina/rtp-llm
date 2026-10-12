"""NVFP4 routed experts and an MXFP8 shared expert in one DeepGEMM kernel."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from .input_packer_triton import (
    fused_pack_mega_moe_inputs,
    stage_shared_expert_input_scales,
)
from .mega_buf import get_or_create_mega_output
from .mega_moe import GLM5MegaMoeCfg
from .mega_moe_nvfp4 import GLM5MegaMoENVFP4
from .quant_layouts import MXFP8_BLOCK, prepare_fp8_weight_scale_for_deepgemm

logger = logging.getLogger(__name__)


class GLM5MegaMoENVFP4SE(GLM5MegaMoENVFP4):
    """Run ``nvfp4_nvfp4_mega_moe`` with its fused MXFP8 shared branch."""

    def __init__(self, cfg: GLM5MegaMoeCfg):
        super().__init__(cfg)
        self._num_shared_experts = 0
        self._shared_l1_w: Optional[torch.Tensor] = None
        self._shared_l1_sf: Optional[torch.Tensor] = None
        self._shared_l2_w: Optional[torch.Tensor] = None
        self._shared_l2_sf: Optional[torch.Tensor] = None
        self._shared_input_sf: Optional[torch.Tensor] = None

    def _setup_buffer_and_warmup(self) -> None:
        # The wrapper installs routed weights first. Rendezvous only after the
        # shared weights are ready, with the final workspace layout on all ranks.
        if self._num_shared_experts == 0:
            return
        super()._setup_buffer_and_warmup()

    def _maybe_warmup_jit_once(self) -> None:
        self._shared_input_sf = get_or_create_mega_output(
            self._mega_buf.num_max_tokens_per_rank,
            self.cfg.dim // (4 * MXFP8_BLOCK),
            torch.int32,
            self._mega_l1_w.device,
        )
        super()._maybe_warmup_jit_once()

    def setup_shared_expert_from_fp8(
        self,
        w1_fp8: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_fp8: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> None:
        """Load full-rank MXFP8 weights; FFN w13 already has [gate | up] order."""
        import deep_gemm

        required = (
            "transform_shared_weights_for_mega_moe_nvfp4",
            "get_block_m_for_mega_moe_nvfp4",
        )
        missing = [name for name in required if not hasattr(deep_gemm, name)]
        if missing:
            raise RuntimeError(
                "mega_moe_nvfp4_se requires DeepGEMM fused shared-expert APIs; "
                f"missing {missing}"
            )
        cfg = self.cfg
        if cfg.n_activated_experts > 31:
            raise ValueError("mega_moe_nvfp4_se requires topk <= 31")
        if cfg.dim % 128 or cfg.moe_inter_dim % 128:
            raise ValueError("NVFP4 SE hidden and intermediate must align to 128")
        for name, weight, expected in (
            ("w1", w1_fp8, (2 * cfg.moe_inter_dim, cfg.dim)),
            ("w2", w2_fp8, (cfg.dim, cfg.moe_inter_dim)),
        ):
            if weight.dtype != torch.float8_e4m3fn or tuple(weight.shape) != expected:
                raise ValueError(
                    "mega_moe_nvfp4_se supports exactly one full MXFP8 shared "
                    f"expert; {name} must be e4m3 {expected}, got "
                    f"{weight.dtype} {tuple(weight.shape)}"
                )
        if self._mega_l1_w is None:
            raise RuntimeError("routed NVFP4 weights must be prepared first")
        device = self._mega_l1_w.device
        with torch.cuda.device(device):
            w1_fp8, w1_scale, w2_fp8, w2_scale = (
                t.to(device=device, non_blocking=True)
                for t in (w1_fp8, w1_scale, w2_fp8, w2_scale)
            )
            s1 = prepare_fp8_weight_scale_for_deepgemm(
                w1_scale, 2 * cfg.moe_inter_dim, cfg.dim, recipe=(1, MXFP8_BLOCK)
            )
            s2 = prepare_fp8_weight_scale_for_deepgemm(
                w2_scale, cfg.dim, cfg.moe_inter_dim, recipe=(1, MXFP8_BLOCK)
            )
            (self._shared_l1_w, self._shared_l1_sf), (
                self._shared_l2_w,
                self._shared_l2_sf,
            ) = deep_gemm.transform_shared_weights_for_mega_moe_nvfp4(
                (w1_fp8.contiguous(), s1), (w2_fp8.contiguous(), s2)
            )
        self._num_shared_experts = 1
        logger.info(
            "[MegaMoE NVFP4 SE] layer=%d fused shared=1 MXFP8 K32; "
            "operator=nvfp4_nvfp4_mega_moe package=%s extension=%s",
            cfg.layer_id,
            deep_gemm.__file__,
            deep_gemm._C.__file__,
        )
        self._setup_buffer_and_warmup()

    def _pack_inputs(self, x, weights, indices, tokens: int) -> None:
        import deep_gemm

        super()._pack_inputs(x, weights, indices, tokens)
        buf = self._mega_buf
        # Quantize the original BF16 input a second time, to MXFP8 K32. Routed
        # buf.x_sf contains UE4M3 K16 and cannot serve as the shared input SF.
        fused_pack_mega_moe_inputs(
            x,
            weights,
            indices,
            buf.shared_l1_acts[:tokens],
            self._shared_input_sf[:tokens],
            buf.topk_idx[:tokens],
            buf.topk_weights[:tokens],
        )
        block_m = deep_gemm.get_block_m_for_mega_moe_nvfp4(
            int(buf.group.size()),
            self.cfg.n_routed_experts,
            buf.num_max_tokens_per_rank,
            tokens,
            self.cfg.n_activated_experts,
        )
        # Clear scale padding only; the kernel owns its communication signals.
        buf.shared_l1_acts_sf.zero_()
        stage_shared_expert_input_scales(
            self._shared_input_sf[:tokens], buf.shared_l1_acts_sf, tokens, block_m
        )

    def _fused_shared_expert_kwargs(self, tokens: int) -> Dict[str, Any]:
        if self._num_shared_experts != 1 or any(
            t is None
            for t in (
                self._shared_l1_w,
                self._shared_l1_sf,
                self._shared_l2_w,
                self._shared_l2_sf,
            )
        ):
            raise RuntimeError("NVFP4 fused shared-expert weights are not initialized")
        return {
            "shared_l1_weights": (self._shared_l1_w, self._shared_l1_sf),
            "shared_l2_weights": (self._shared_l2_w, self._shared_l2_sf),
        }

    def clone_for_cuda_graph(self) -> "GLM5MegaMoENVFP4SE":
        clone = super().clone_for_cuda_graph()
        clone._num_shared_experts = self._num_shared_experts
        clone._shared_l1_w = self._shared_l1_w
        clone._shared_l1_sf = self._shared_l1_sf
        clone._shared_l2_w = self._shared_l2_w
        clone._shared_l2_sf = self._shared_l2_sf
        # Like the symmetric workspace, staging scratch is reused serially by
        # layers and graph clones. No new collective allocation during cloning.
        clone._shared_input_sf = self._shared_input_sf
        return clone
