"""Native warp TopK dispatch for the measured small-K M3 prefill geometry."""

import os
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _native_topk():
    # Older wheels can still execute the established Triton path.
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    return getattr(rtp_llm_ops, "minimax_prefill_topk", None)


def try_warp_prefill_topk(max_query_length, num_heads, num_blocks, args, kwargs):
    """Return GPU tie flags, or None for unsupported geometry; no synchronization."""
    # The caller reuses the historical bitonic body for flagged rows, preserving
    # its equal-score IDs. The measured M3 geometry uses this path by default.
    if os.environ.get("M3_PREFILL_TOPK_IMPL", "warp") != "warp":
        return None
    scores, bt, sl, topk_idx = args[:4]
    sample_interval, block_size, cu, cu_blocks, prefix = args[4:9]
    topk, init_blocks, local_blocks, num_pages = args[9:13]
    eligible = (
        torch.version.hip is None
        and scores.is_cuda
        and sample_interval == 1
        and block_size == 128
        and topk == 16
        and 512 <= num_blocks <= 1024
        and scores.shape[1] >= 1024
        and num_heads == 4
        and scores.shape[0] == num_heads == kwargs.get("NKV")
        and scores.shape[2] == num_blocks
        and cu.data_ptr() == cu_blocks.data_ptr()
        and not kwargs.get("MASK_INIT", False)
        and not kwargs.get("MASK_LOCAL", False)
        and kwargs.get("EMIT_TOPK_IDX", False)
        and not kwargs.get("EMIT_BLOCK_TABLE", False)
        and scores.dtype == torch.float32
        and scores.stride(2) == 1
        and topk_idx.dtype == torch.int32
        and topk_idx.stride(2) == 1
        # The reused legacy body evaluates output-index arithmetic in int32.
        and (num_heads - 1) * topk_idx.stride(0)
        + (scores.shape[1] - 1) * topk_idx.stride(1)
        + topk
        <= 2147483647
        and cu.dtype == torch.int32
        and prefix.dtype == torch.int32
        and cu.is_contiguous()
        and prefix.is_contiguous()
    )
    if not eligible:
        return None
    op = _native_topk()
    if op is None:
        return None
    return op(
        scores,
        topk_idx,
        bt,
        sl,
        cu,
        prefix,
        max_query_length,
        topk,
        init_blocks,
        local_blocks,
        kwargs.get("EMIT_BLOCK_TABLE", False),
        num_pages,
    )
