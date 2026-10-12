"""FusedMoe-compatible NVFP4 MegaMoE wrapper with a fused MXFP8 shared expert."""

import torch

from rtp_llm.utils.model_weight import W

from .mega_moe_nvfp4_se import GLM5MegaMoENVFP4SE
from .mega_moe_nvfp4_wrapper import MegaMoeNvfp4Wrapper


class MegaMoeNvfp4SEWrapper(MegaMoeNvfp4Wrapper):
    """Retain NVFP4 prefill chunking/direct output and consume shared weights."""

    def _get_mega_moe_cls(self):
        return GLM5MegaMoENVFP4SE

    def __init__(
        self,
        config,
        parallelism_config,
        weights,
        moe_config=None,
        layer_idx: int = 0,
        max_generate_batch_size: int = 0,
    ):
        if parallelism_config.get_ffn_tp_size() != 1:
            raise ValueError(
                "mega_moe_nvfp4_se requires full shared weights on every EP "
                "rank (ffn_tp_size == 1)"
            )
        shared_keys = (W.ffn_w13, W.ffn_s13, W.ffn_w2, W.ffn_s2)
        missing = [key for key in shared_keys if weights.get(key) is None]
        if missing:
            raise ValueError(
                f"NVFP4 SE requires MXFP8 shared weights/scales: {missing}"
            )
        super().__init__(
            config,
            parallelism_config,
            weights,
            moe_config=moe_config,
            layer_idx=layer_idx,
            max_generate_batch_size=max_generate_batch_size,
        )
        w1, s1, w2, s2 = (weights.pop(key) for key in shared_keys)
        self.mega_moe.setup_shared_expert_from_fp8(
            w1_fp8=w1, w1_scale=s1, w2_fp8=w2, w2_scale=s2
        )
        del w1, s1, w2, s2
        torch.cuda.empty_cache()
