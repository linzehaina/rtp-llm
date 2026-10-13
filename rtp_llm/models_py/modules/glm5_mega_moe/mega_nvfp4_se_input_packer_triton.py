"""Single-launch NVFP4/MXFP8 input packing, including consumed shared SF padding.

DeepGEMM schedules ceil(T/BLOCK_M) shared blocks and loads exactly
SF_BLOCK_M=align(BLOCK_M,128) rows per block. The unused allocation tail is
not read. See scheduler/mega_moe_nvfp4.cuh:create_task and
sm100_nvfp4_nvfp4_mega_moe.cuh:run_a_load_block.

Each token/hidden-tile CTA reduces the entire hidden dimension before NVFP4 K16
quantization. Other CTAs clear only inactive swizzled SF rows: their writes
are disjoint from token CTAs, so no grid-wide synchronization is needed.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .mega_nvfp4_input_packer_triton import _cast_ue4m3_nearest

_QUANT_TILE_H = 1024
_NUM_WARPS = 4


@triton.jit(do_not_specialize=["tokens"])
def _pack_nvfp4_se_inputs_impl(
    x_ptr,
    weights_ptr,
    indices_ptr,
    fp4_ptr,
    sf_ptr,
    gsf_ptr,
    out_indices_ptr,
    out_weights_ptr,
    shared_ptr,
    shared_sf_ptr,
    tokens,
    layout: tl.constexpr,
    DECODE: tl.constexpr,
):
    H: tl.constexpr = layout[0]
    TOPK: tl.constexpr = layout[1]
    X_SM: tl.constexpr = layout[2]
    X_SK: tl.constexpr = layout[3]
    W_SM: tl.constexpr = layout[4]
    W_SK: tl.constexpr = layout[5]
    I_SM: tl.constexpr = layout[6]
    I_SK: tl.constexpr = layout[7]
    FP4_SM: tl.constexpr = layout[8]
    FP4_SK: tl.constexpr = layout[9]
    SF_SM: tl.constexpr = layout[10]
    SF_SK: tl.constexpr = layout[11]
    GSF_SM: tl.constexpr = layout[12]
    OI_SM: tl.constexpr = layout[13]
    OI_SK: tl.constexpr = layout[14]
    OW_SM: tl.constexpr = layout[15]
    OW_SK: tl.constexpr = layout[16]
    SH_SM: tl.constexpr = layout[17]
    SH_SK: tl.constexpr = layout[18]
    SH_SF_SM: tl.constexpr = layout[19]
    SH_SF_SK: tl.constexpr = layout[20]
    SF_READ_ROWS: tl.constexpr = layout[21]
    BLOCK_M: tl.constexpr = layout[22]
    ALIGNED_M: tl.constexpr = layout[23]
    BLOCK_H: tl.constexpr = layout[24]
    TILE_H: tl.constexpr = layout[25]
    NUM_TILES: tl.constexpr = layout[26]
    BLOCK_TOPK: tl.constexpr = layout[27]
    CLEAR_BLOCK: tl.constexpr = layout[28]
    pid = tl.program_id(0).to(tl.int64)
    if pid < tokens * NUM_TILES:
        row = pid // NUM_TILES
        tile = pid % NUM_TILES
        cols = tl.arange(0, BLOCK_H)
        values = tl.load(x_ptr + row * X_SM + cols * X_SK, cols < H, 0).to(tl.float32)
        # This full-H reduction must precede every K16 scale calculation.
        gsf = tl.maximum(tl.max(tl.abs(values), 0), 1.0e-30) / (6.0 * 448.0)
        if tile == 0:
            tl.store(gsf_ptr + row * GSF_SM, gsf)
        if DECODE:
            # Small-row hidden tiles trade redundant L2 reduction reads for
            # more independent CTAs and fewer quantization registers.
            cols = tile * TILE_H + tl.arange(0, TILE_H)
            values = tl.load(x_ptr + row * X_SM + cols * X_SK, cols < H, 0).to(
                tl.float32
            )
        # Prefill has one full-H CTA per token: reuse the original FP32 load.
        groups16 = tl.reshape(values, (TILE_H // 16, 16))
        amax16 = tl.max(tl.abs(groups16), 1)
        sf_code, sf_value = _cast_ue4m3_nearest(tl.div_rn(amax16 / 6.0, gsf))
        inverse = tl.div_rn(tl.div_rn(1.0, gsf), sf_value)
        scaled = tl.maximum(tl.minimum(groups16 * inverse[:, None], 6.0), -6.0)
        value0, value1 = tl.split(tl.reshape(scaled, (TILE_H // 2, 2)))
        # CUDA's native E2M1x2 conversion uses the same nearest-even ladder;
        # the first feature occupies the low nibble, including signed zero.
        packed_fp4 = tl.inline_asm_elementwise(
            "{ .reg .b8 packed; cvt.rn.satfinite.e2m1x2.f32 packed, $2, $1; "
            "mov.b32 $0, {packed, 0, 0, 0}; }",
            constraints="=r,f,f",
            args=[value0, value1],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        half_cols = tile * (TILE_H // 2) + tl.arange(0, TILE_H // 2)
        tl.store(
            fp4_ptr + row * FP4_SM + half_cols * FP4_SK,
            packed_fp4,
            half_cols < H // 2,
        )
        sf_bytes = tl.reshape(sf_code, (TILE_H // 64, 4))
        packed16 = tl.sum(sf_bytes << (tl.arange(0, 4)[None, :] * 8), 1)
        sf_cols = tile * (TILE_H // 64) + tl.arange(0, TILE_H // 64)
        tl.store(sf_ptr + row * SF_SM + sf_cols * SF_SK, packed16, sf_cols < H // 64)

        # Independent MXFP8 K32 scale, with UE8M0 ceil-to-power-of-two.
        groups32 = tl.reshape(values, (TILE_H // 32, 32))
        raw = tl.maximum(tl.max(tl.abs(groups32), 1), 1.0e-4) / 448.0
        bits = raw.to(tl.int32, bitcast=True)
        exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0)
        exp = tl.minimum(tl.maximum(exp, 1), 254)
        scale = (exp << 23).to(tl.float32, bitcast=True)
        q = tl.reshape(tl.clamp(groups32 / scale[:, None], -448.0, 448.0), (TILE_H,))
        tl.store(shared_ptr + row * SH_SM + cols * SH_SK, q.to(tl.float8e4nv), cols < H)
        exp4 = tl.reshape(exp, (TILE_H // 128, 4))
        packed32 = tl.sum(exp4 << (tl.arange(0, 4)[None, :] * 8), 1)
        within = row % BLOCK_M
        destination_row = (
            row // BLOCK_M * ALIGNED_M
            + within // 128 * 128
            + within % 32 * 4
            + within % 128 // 32
        )
        shared_cols = tile * (TILE_H // 128) + tl.arange(0, TILE_H // 128)
        tl.store(
            shared_sf_ptr + destination_row * SH_SF_SM + shared_cols * SH_SF_SK,
            packed32,
            shared_cols < H // 128,
        )
        if tile == 0:
            route = tl.arange(0, BLOCK_TOPK)
            w = tl.load(weights_ptr + row * W_SM + route * W_SK, route < TOPK, 0)
            idx = tl.load(indices_ptr + row * I_SM + route * I_SK, route < TOPK, 0)
            tl.store(out_weights_ptr + row * OW_SM + route * OW_SK, w, route < TOPK)
            tl.store(out_indices_ptr + row * OI_SM + route * OI_SK, idx, route < TOPK)
    else:
        # Iterate rows fastest, matching the real column-major SF allocation.
        offset = (pid - tokens * NUM_TILES) * CLEAR_BLOCK + tl.arange(0, CLEAR_BLOCK)
        clear_row = offset % SF_READ_ROWS
        col = offset // SF_READ_ROWS
        # Invert the UTCCP row permutation, including BLOCK_M=192.
        within_swizzle = clear_row % ALIGNED_M
        original_within = (
            within_swizzle // 128 * 128
            + within_swizzle % 4 * 32
            + within_swizzle % 128 // 4
        )
        original_token = clear_row // ALIGNED_M * BLOCK_M + original_within
        active = (original_within < BLOCK_M) & (original_token < tokens)
        tl.store(
            shared_sf_ptr + clear_row * SH_SF_SM + col * SH_SF_SK,
            0,
            (offset < SF_READ_ROWS * (H // 128)) & ~active,
        )


@triton.jit(do_not_specialize=["tokens"])
def _pack_nvfp4_se_decode_inputs_kernel(
    x_ptr,
    weights_ptr,
    indices_ptr,
    fp4_ptr,
    sf_ptr,
    gsf_ptr,
    out_indices_ptr,
    out_weights_ptr,
    shared_ptr,
    shared_sf_ptr,
    tokens,
    layout: tl.constexpr,
):
    _pack_nvfp4_se_inputs_impl(
        x_ptr,
        weights_ptr,
        indices_ptr,
        fp4_ptr,
        sf_ptr,
        gsf_ptr,
        out_indices_ptr,
        out_weights_ptr,
        shared_ptr,
        shared_sf_ptr,
        tokens,
        layout,
        True,
    )


@triton.jit(do_not_specialize=["tokens"])
def _pack_nvfp4_se_prefill_inputs_kernel(
    x_ptr,
    weights_ptr,
    indices_ptr,
    fp4_ptr,
    sf_ptr,
    gsf_ptr,
    out_indices_ptr,
    out_weights_ptr,
    shared_ptr,
    shared_sf_ptr,
    tokens,
    layout: tl.constexpr,
):
    _pack_nvfp4_se_inputs_impl(
        x_ptr,
        weights_ptr,
        indices_ptr,
        fp4_ptr,
        sf_ptr,
        gsf_ptr,
        out_indices_ptr,
        out_weights_ptr,
        shared_ptr,
        shared_sf_ptr,
        tokens,
        layout,
        False,
    )


def fused_pack_mega_nvfp4_se_inputs(
    x, weights, indices, buf, tokens: int, block_m: int
) -> None:
    """Write both expert input formats and zero the consumed shared SF padding.

    The buffer's communication workspace is not touched. All tensor strides are
    passed explicitly; this also supports diagnostic buffers with padded strides.
    """
    if x.dtype != torch.bfloat16 or not x.is_cuda or x.ndim != 2:
        raise ValueError("NVFP4 SE packing requires a CUDA BF16 [T,H] tensor")
    hidden = x.shape[1]
    if x.shape[0] != tokens or hidden % 128 or hidden <= 0:
        raise ValueError(
            "NVFP4 SE input rows must match tokens and H must align to 128"
        )
    if weights.dtype != torch.float32 or indices.dtype != torch.int64:
        raise ValueError("NVFP4 SE router weights/indices must be float32/int64")
    if weights.shape != indices.shape or weights.shape[0] != tokens:
        raise ValueError("NVFP4 SE router tensors must match [T,topk]")
    topk = weights.shape[1]
    if not 0 < topk <= 31 or block_m not in (8, 16, 32, 64, 96, 128, 192):
        raise ValueError("unsupported NVFP4 SE topk or DeepGEMM block_m")
    shared_sf = buf.shared_l1_acts_sf
    if (
        shared_sf.dtype != torch.int32
        or shared_sf.ndim != 2
        or shared_sf.shape[1] != hidden // 128
    ):
        raise ValueError("NVFP4 SE shared SF must be packed int32 [padded_rows,H/128]")
    aligned_m = triton.cdiv(block_m, 128) * 128
    if shared_sf.shape[0] < triton.cdiv(tokens, block_m) * aligned_m:
        raise ValueError("NVFP4 SE shared SF buffer is too small for swizzled input")
    tensors = [
        buf.x,
        buf.x_sf,
        buf.x_gsf,
        buf.topk_idx,
        buf.topk_weights,
        buf.shared_l1_acts,
        shared_sf,
    ]
    expected = [
        (torch.int8, hidden // 2),
        (torch.int32, hidden // 64),
        (torch.float32, None),
        (torch.int64, topk),
        (torch.float32, topk),
        (torch.float8_e4m3fn, hidden),
        (torch.int32, hidden // 128),
    ]
    for tensor, (dtype, width) in zip(tensors, expected):
        if (
            tensor.device != x.device
            or tensor.dtype != dtype
            or tensor.shape[0] < tokens
        ):
            raise ValueError("invalid NVFP4 SE output buffer device/dtype/capacity")
        if tensor.ndim != (1 if width is None else 2) or (
            width is not None and tensor.shape[1] != width
        ):
            raise ValueError("invalid NVFP4 SE output buffer shape")
    if weights.device != x.device or indices.device != x.device:
        raise ValueError("NVFP4 SE router tensors must be on the input device")
    clear_block = 1024
    read_rows = triton.cdiv(tokens, block_m) * aligned_m
    if tokens == 0:
        return
    tile_h = (
        triton.next_power_of_2(hidden)
        if tokens >= 1024
        else min(_QUANT_TILE_H, triton.next_power_of_2(hidden))
    )
    num_warps = 8 if tokens >= 1024 else _NUM_WARPS
    num_tiles = triton.cdiv(hidden, tile_h)
    grid = (tokens * num_tiles + triton.cdiv(read_rows * (hidden // 128), clear_block),)
    # Dispatch uses the actual tensor row count, never batch size or MTP step.
    kernel = (
        _pack_nvfp4_se_decode_inputs_kernel
        if tokens < 1024
        else _pack_nvfp4_se_prefill_inputs_kernel
    )
    layout = (
        hidden,
        topk,
        *x.stride(),
        *weights.stride(),
        *indices.stride(),
        *buf.x.stride(),
        *buf.x_sf.stride(),
        buf.x_gsf.stride(0),
        *buf.topk_idx.stride(),
        *buf.topk_weights.stride(),
        *buf.shared_l1_acts.stride(),
        *shared_sf.stride(),
        read_rows,
        block_m,
        aligned_m,
        triton.next_power_of_2(hidden),
        tile_h,
        num_tiles,
        triton.next_power_of_2(topk),
        clear_block,
    )
    kernel[grid](
        x,
        weights,
        indices,
        *tensors[:5],
        tensors[5],
        shared_sf,
        tokens,
        layout,
        num_warps=num_warps,
    )
