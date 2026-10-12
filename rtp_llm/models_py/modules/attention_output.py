"""Internal attention-to-postnorm contract; default attention API stays Tensor."""
from typing import NamedTuple

import torch


class DeferredMaskedAttentionOutput(NamedTuple):
    # Invalid projected rows contribute positive zero BEFORE residual addition.
    # The residual and normalized output of those rows must not be zeroed.
    hidden_states: torch.Tensor
    valid_row_mask: torch.Tensor


def mask_attention_output(
    hidden_states: torch.Tensor,
    valid_row_mask: torch.Tensor,
    *,
    defer_mask: bool = False,
) -> torch.Tensor | DeferredMaskedAttentionOutput:
    """Defer only masks accepted by fused postnorm; preserve the legacy fallback.

    A single request expanded across verify rows can produce a legitimate
    stride-zero mask. torch.where supports it, while fused postnorm requires
    contiguous bool[T]. Keep that case on the original masking path.
    """
    if (
        defer_mask
        and hidden_states.dim() == 2
        and valid_row_mask.dtype == torch.bool
        and valid_row_mask.shape == (hidden_states.shape[0],)
        and valid_row_mask.device == hidden_states.device
        and valid_row_mask.is_contiguous()
    ):
        return DeferredMaskedAttentionOutput(hidden_states, valid_row_mask)
    return torch.where(
        valid_row_mask[:, None], hidden_states, torch.zeros_like(hidden_states)
    )
