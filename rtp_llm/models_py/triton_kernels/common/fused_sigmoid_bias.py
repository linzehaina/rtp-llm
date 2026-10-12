"""FP32 sigmoid and correction bias with the same rounded intermediate."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _sigmoid_bias(X, BIAS, SCORES, CORRECTED,
                  SIZE: tl.constexpr, EXPERTS: tl.constexpr,
                  BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < SIZE
    x = tl.load(X + offset, valid, other=0)
    # Match the CUDA FP32 sigmoid's expf and rounded division. Fast exp2 or
    # approximate reciprocal can change a tied expert-selection boundary.
    score = tl.div_rn(1.0, 1.0 + libdevice.exp(-x))
    bias = tl.load(BIAS + offset % EXPERTS, valid, other=0)
    tl.store(SCORES + offset, score, valid)
    tl.store(CORRECTED + offset, score + bias, valid)


def fused_sigmoid_bias(scores: torch.Tensor, bias: torch.Tensor):
    """Return raw sigmoid scores and biased selection scores, both FP32."""
    if (
        scores.ndim != 2
        or bias.ndim != 1
        or scores.shape[1] != bias.numel()
        or scores.dtype != torch.float32
        or bias.dtype != torch.float32
        or not scores.is_cuda
        or scores.device != bias.device
        or not scores.is_contiguous()
        or not bias.is_contiguous()
    ):
        raise ValueError("sigmoid/bias requires contiguous CUDA FP32 rows and bias")
    raw = torch.empty_like(scores)
    corrected = torch.empty_like(scores)
    if scores.numel():
        _sigmoid_bias[(triton.cdiv(scores.numel(), 512),)](
            scores, bias, raw, corrected, scores.numel(), bias.numel(),
            BLOCK=512, num_warps=4)
    return raw, corrected
