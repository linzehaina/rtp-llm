"""ctypes bridge for the our_score tcgen05 score kernels (M3.1 port of the
M3.0 b128/b128d campaign kernels).

Pattern: ``sparse_msa/prefill/champion_r8pkge.py`` -- content-verified CDLL,
field-order-exact ctypes Structure mirrors of the frozen C ABI
(``our_score_abi.h``, OUR_SCORE_ABI_VERSION 1), a keepalive ring holding
every Params-referenced tensor through the async launch, and the stream taken
from ``torch.cuda.current_stream(device).cuda_stream``.  No host-device syncs.

Entry points
------------
``launch_prefill(...)``        -- low-level prefill ABI launch over an
    already-staged page-major E4M3 scratch (used directly by the gate).
``run_prefill_score(ws, i, q8, page_offsets, score)`` -- production wrapper
    called from ``NativeIndexWorkspace.score_ourscore``; reads the workspace's
    staged scratch, safe page table and chunk metadata.
``our_decode_index_score(...)`` -- exact drop-in for
    ``decode.nvfp4_q8_index_score.q8kv4_index_score`` (same call signature);
    stages the referenced packed-NVFP4 index-K pages into a cached E4M3
    scratch (correctness-first v1: full-cache stage each call) and launches
    the decode ABI.

All index tensors must be torch.int32 (asserted; never int64).  The .so is
resolved from ``M3_OUR_SCORE_LIB`` or ``<this dir>/lib/libour_score.so``; a
missing/stale library raises RuntimeError with the build instructions -- the
gate degrades gracefully on that message.

Env gates live in the dispatch modules (topk_bt_fused.py /
q8kv4_decode.py), not here: default OFF keeps production bit-for-bit.
"""

import ctypes
import logging
import os
from collections import deque

import torch
import triton
import triton.language as tl

from ..decode.nvfp4_q8_index_score import _load_index_page_fp8

OUR_SCORE_ABI_VERSION = 1

_DEFAULT_LIB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "lib", "libour_score.so"
)

# ---------------------------------------------------------------- ctypes ABI


class OurPrefillParams(ctypes.Structure):
    """Exact field-order mirror of OurPrefillParams in our_score_abi.h."""

    _fields_ = [
        ("q8", ctypes.c_void_p),
        ("staged_k", ctypes.c_void_p),
        ("kv_indices", ctypes.c_void_p),
        ("cu_seqlens", ctypes.c_void_p),
        ("seq_lens", ctypes.c_void_p),
        ("prefix_lens", ctypes.c_void_p),
        ("page_offsets", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("kv_indices_len", ctypes.c_int64),
        ("rows", ctypes.c_int32),
        ("segs", ctypes.c_int32),
        ("blocks", ctypes.c_int32),
        ("phys_pages", ctypes.c_int32),
        ("out_stride_h", ctypes.c_int32),
        ("out_stride_r", ctypes.c_int32),
    ]


class OurDecodeParams(ctypes.Structure):
    """Exact field-order mirror of OurDecodeParams in our_score_abi.h."""

    _fields_ = [
        ("q8", ctypes.c_void_p),
        ("staged_k", ctypes.c_void_p),
        ("block_table", ctypes.c_void_p),
        ("seq_lens", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("batch", ctypes.c_int32),
        ("max_blocks", ctypes.c_int32),
        ("bt_stride", ctypes.c_int32),
        ("q8_stride_b", ctypes.c_int32),
        ("phys_pages", ctypes.c_int32),
        ("sm_scale", ctypes.c_float),
        ("init_blocks", ctypes.c_int32),
        ("local_blocks", ctypes.c_int32),
    ]


class OurFusedPrefillParams(ctypes.Structure):
    """Exact field-order mirror of OurFusedPrefillParams in our_score_abi.h.

    Added alongside the frozen v1 symbols (no version bump): the fused
    score+topk entry is gated on symbol AVAILABILITY, so an old .so paired
    with this bridge fails cleanly at :func:`fused_prefill_available`.
    """

    _fields_ = [
        ("q8", ctypes.c_void_p),
        ("staged_k", ctypes.c_void_p),
        ("kv_indices", ctypes.c_void_p),
        ("cu_seqlens", ctypes.c_void_p),
        ("seq_lens", ctypes.c_void_p),
        ("prefix_lens", ctypes.c_void_p),
        ("page_offsets", ctypes.c_void_p),
        ("topk_idx", ctypes.c_void_p),
        ("kv_indices_len", ctypes.c_int64),
        ("rows", ctypes.c_int32),
        ("segs", ctypes.c_int32),
        ("blocks", ctypes.c_int32),
        ("phys_pages", ctypes.c_int32),
        ("ti_stride_h", ctypes.c_int32),
        ("ti_stride_r", ctypes.c_int32),
        ("ti_stride_t", ctypes.c_int32),
        ("topk", ctypes.c_int32),
        ("init_blocks", ctypes.c_int32),
        ("local_blocks", ctypes.c_int32),
    ]


_LIB = None

# Params-referenced tensors must outlive the async launch; a bounded ring
# keeps the last launches alive (production reuses stable workspace
# addresses, so a shallow ring is ample -- same idea as the champion
# runner's single keepalive slot, generalized for stateless runners).
_KEEPALIVE = deque(maxlen=64)


def get_library():
    """CDLL of libour_score.so with the ABI version asserted on load."""
    global _LIB
    if _LIB is not None:
        return _LIB
    path = os.environ.get("M3_OUR_SCORE_LIB") or _DEFAULT_LIB
    if not os.path.exists(path):
        raise RuntimeError(
            f"libour_score.so not found at {path}. Build it first (Track A): "
            "cd rtp_llm/models_py/triton_kernels/sparse_msa/our_score && "
            "/usr/local/cuda-13.0/bin/nvcc -O3 -std=c++17 "
            "-gencode arch=compute_103a,code=sm_103a -I. -Icommon -Iprefill "
            "-Idecode -I/home/liukan.lk/m3pro-128blk-20261009/third_party/cutlass/include "
            "--expt-relaxed-constexpr --expt-extended-lambda -Xcompiler -fPIC -shared "
            "our_score_host.cu prefill/kernel_ourscore_prefill.cu "
            "decode/kernel_ourscore_decode.cu -o lib/libour_score.so "
            "-lcudart -lcuda -lm  (or run our_score/build.sh), or point "
            "M3_OUR_SCORE_LIB at an existing build."
        )
    lib = ctypes.CDLL(path)
    for symbol in (
        "our_score_abi_version",
        "our_prefill_launch",
        "our_decode_launch",
    ):
        if not hasattr(lib, symbol):
            raise RuntimeError(f"our_score binary {path} is missing {symbol}")
    lib.our_score_abi_version.argtypes = []
    lib.our_score_abi_version.restype = ctypes.c_int
    version = int(lib.our_score_abi_version())
    if version != OUR_SCORE_ABI_VERSION:
        raise RuntimeError(
            f"our_score ABI version {version} != bridge version "
            f"{OUR_SCORE_ABI_VERSION} ({path}); rebuild or update the bridge"
        )
    lib.our_prefill_launch.argtypes = [
        ctypes.POINTER(OurPrefillParams),
        ctypes.c_void_p,
    ]
    lib.our_prefill_launch.restype = ctypes.c_int
    lib.our_decode_launch.argtypes = [
        ctypes.POINTER(OurDecodeParams),
        ctypes.c_void_p,
    ]
    lib.our_decode_launch.restype = ctypes.c_int
    _LIB = lib
    logging.info("[M3 our_score] loaded %s (ABI %d)", path, version)
    return lib


def fused_prefill_available():
    """True when the loaded .so exports our_fused_prefill_launch.

    Raises whatever :func:`get_library` raises when the .so itself is
    missing/stale; the dispatch layer is expected to catch that and fall
    back to the two-step path.
    """
    lib = get_library()
    if not hasattr(lib, "our_fused_prefill_launch"):
        raise RuntimeError(
            f"our_score binary {os.environ.get('M3_OUR_SCORE_LIB') or _DEFAULT_LIB} "
            "predates the fused prefill entry (our_fused_prefill_launch); "
            "rebuild with our_score/build.sh"
        )
    if not hasattr(lib, "_fused_argtypes_bound"):
        lib.our_fused_prefill_launch.argtypes = [
            ctypes.POINTER(OurFusedPrefillParams),
            ctypes.c_void_p,
        ]
        lib.our_fused_prefill_launch.restype = ctypes.c_int
        lib._fused_argtypes_bound = True
    return True


def _launch(fn_name, struct_cls, values, tensors, device):
    """Field-count-guarded struct build + stream launch + keepalive."""
    if len(values) != len(struct_cls._fields_):
        raise TypeError(
            f"our_score {fn_name}: {len(values)} values vs "
            f"{len(struct_cls._fields_)} {struct_cls.__name__} fields"
        )
    lib = get_library()
    params = struct_cls(*values)
    stream = ctypes.c_void_p(
        torch.cuda.current_stream(device).cuda_stream
    )
    err = int(getattr(lib, fn_name)(ctypes.byref(params), stream))
    if err != 0:
        raise RuntimeError(f"our_score {fn_name} failed with error code {err}")
    _KEEPALIVE.append((tensors, params))
    return err


# ---------------------------------------------------------------- validation


def _check_i32(tensor, size, name):
    if (
        tensor is None
        or tensor.dtype != torch.int32
        or tensor.ndim != 1
        or tensor.numel() != size
        or tensor.stride(0) != 1
    ):
        raise ValueError(f"{name} must be contiguous int32 [{size}]")


def _same_device(reference, *tensors):
    for tensor in tensors:
        if tensor.device != reference.device:
            raise ValueError(
                f"our_score tensors must share one CUDA device "
                f"({reference.device} vs {tensor.device})"
            )
    if reference.device.type != "cuda":
        raise ValueError("our_score requires CUDA tensors")


def _check_no_alias(out, *inputs):
    out_storage = out.untyped_storage().data_ptr()
    out_end = out_storage + out.untyped_storage().nbytes()
    for tensor in inputs:
        if tensor is None or not isinstance(tensor, torch.Tensor):
            continue
        start = tensor.untyped_storage().data_ptr()
        end = start + tensor.untyped_storage().nbytes()
        if start < out_end and out_storage < end:
            raise ValueError(
                "our_score output must not alias any input tensor storage"
            )


# ---------------------------------------------------------------- prefill


def launch_prefill(
    q8,
    staged_k,
    kv_indices,
    cu_seqlens,
    seq_lens,
    prefix_lens,
    page_offsets,
    out,
    phys_pages=None,
):
    """Low-level prefill ABI launch (frozen semantics: see our_score_abi.h).

    ``out`` is pre-filled with -inf on-stream; the kernel writes only finite
    values for visible blocks.  Chunk-local rows; strided fp32 view with
    stride(2)==1.
    """
    if (
        q8.dtype != torch.float8_e4m3fn
        or q8.ndim != 3
        or tuple(q8.shape[1:]) != (4, 128)
        or not q8.is_contiguous()
    ):
        raise ValueError(
            "our_score prefill Q must be contiguous E4M3 [rows,4,128]"
        )
    if (
        staged_k.dtype != torch.float8_e4m3fn
        or staged_k.numel() % (128 * 128) != 0
        or not staged_k.is_contiguous()
    ):
        raise ValueError(
            "our_score staged K must be contiguous page-major E4M3 "
            "[(phys_pages+1)*128,128] (16 KiB per page)"
        )
    staged_pages = staged_k.numel() // (128 * 128) - 1
    if staged_pages < 1:
        raise ValueError("our_score staged K must hold at least one page")
    if phys_pages is None:
        phys_pages = staged_pages
    elif int(phys_pages) != staged_pages:
        raise ValueError(
            f"our_score phys_pages={phys_pages} disagrees with staged K "
            f"pages={staged_pages}"
        )
    if (
        kv_indices.dtype != torch.int32
        or kv_indices.ndim != 1
        or kv_indices.stride(0) != 1
    ):
        raise ValueError("our_score kv_indices must be contiguous int32")
    rows = int(q8.shape[0])
    segs = int(cu_seqlens.numel()) - 1
    _check_i32(cu_seqlens, segs + 1, "cu_seqlens")
    _check_i32(seq_lens, segs, "seq_lens")
    _check_i32(prefix_lens, segs, "prefix_lens")
    _check_i32(page_offsets, segs + 1, "page_offsets")
    if out.ndim != 3:
        raise ValueError("our_score prefill output must be 3-D float32")
    blocks = int(out.shape[2])
    if (
        out.dtype != torch.float32
        or out.ndim != 3
        or tuple(out.shape) != (4, rows, blocks)
        or out.stride(2) != 1
        or out.stride(1) < blocks
        or out.stride(0) < rows * out.stride(1)
    ):
        raise ValueError(
            "our_score prefill output must be nonoverlapping float32 "
            f"[4,{rows},{blocks}] with stride(2)==1"
        )
    _same_device(q8, staged_k, kv_indices, cu_seqlens, seq_lens, prefix_lens,
                 page_offsets, out)
    _check_no_alias(out, q8, staged_k, kv_indices, cu_seqlens, seq_lens,
                    prefix_lens, page_offsets)
    out.fill_(float("-inf"))
    values = [
        q8.data_ptr(),
        staged_k.data_ptr(),
        kv_indices.data_ptr(),
        cu_seqlens.data_ptr(),
        seq_lens.data_ptr(),
        prefix_lens.data_ptr(),
        page_offsets.data_ptr(),
        out.data_ptr(),
        int(kv_indices.numel()),
        rows,
        segs,
        blocks,
        int(phys_pages),
        int(out.stride(0)),
        int(out.stride(1)),
    ]
    _launch(
        "our_prefill_launch",
        OurPrefillParams,
        values,
        (q8, staged_k, kv_indices, cu_seqlens, seq_lens, prefix_lens,
         page_offsets, out),
        q8.device,
    )
    return out


def run_prefill_score(native_workspace, chunk_index, q8, cu_page_offsets,
                      score):
    """Production prefill entry: launch over a NativeIndexWorkspace chunk.

    ``native_workspace.stage``/``score_ourscore`` own the prologue (staged
    scratch, safe-table refresh, compact staging); this wrapper only mirrors
    the frozen ABI.  The page table passed to the kernel is the workspace's
    SAFE table (``safe_tables[chunk_index]``): invalid entries map to the
    sentinel page index == phys_pages (-> -inf per the ABI, never
    dereferenced), and in compact mode it carries the remapped dense page
    ids -- the raw ``chunk.kv_indices`` would misread under compact staging.
    """
    chunk = native_workspace.chunks[chunk_index]
    staged = native_workspace.staged
    safe = native_workspace.safe_tables[chunk_index]
    return launch_prefill(
        q8,
        staged,
        safe,
        chunk.cu_seqlens,
        chunk.seq_lens,
        chunk.prefix_lens,
        cu_page_offsets,
        score,
        phys_pages=native_workspace.stage_pages,
    )


# ---------------------------------------------------------------- fused prefill


def launch_fused_prefill(
    q8,
    staged_k,
    kv_indices,
    cu_seqlens,
    seq_lens,
    prefix_lens,
    page_offsets,
    topk_idx,
    blocks,
    topk,
    init_blocks,
    local_blocks,
    phys_pages=None,
):
    """Low-level FUSED prefill ABI launch (score + top-k in one kernel).

    ``topk_idx`` is pre-filled with -1 on-stream; the kernel defines every
    (h, r, slot) of the [4, rows, topk] strided view.  ``blocks`` is the
    chunk's block count (the width the two-step path's score tensor would
    have); it cannot be derived from ``topk_idx`` and must be passed
    explicitly.  Requires ``blocks <= 1024`` (single window) and
    ``topk == 16`` -- anything else raises, and the caller must fall back
    to the two-step path.
    """
    fused_prefill_available()
    if (
        q8.dtype != torch.float8_e4m3fn
        or q8.ndim != 3
        or tuple(q8.shape[1:]) != (4, 128)
        or not q8.is_contiguous()
    ):
        raise ValueError(
            "our_score prefill Q must be contiguous E4M3 [rows,4,128]"
        )
    if (
        staged_k.dtype != torch.float8_e4m3fn
        or staged_k.numel() % (128 * 128) != 0
        or not staged_k.is_contiguous()
    ):
        raise ValueError(
            "our_score staged K must be contiguous page-major E4M3 "
            "[(phys_pages+1)*128,128] (16 KiB per page)"
        )
    staged_pages = staged_k.numel() // (128 * 128) - 1
    if staged_pages < 1:
        raise ValueError("our_score staged K must hold at least one page")
    if phys_pages is None:
        phys_pages = staged_pages
    elif int(phys_pages) != staged_pages:
        raise ValueError(
            f"our_score phys_pages={phys_pages} disagrees with staged K "
            f"pages={staged_pages}"
        )
    if (
        kv_indices.dtype != torch.int32
        or kv_indices.ndim != 1
        or kv_indices.stride(0) != 1
    ):
        raise ValueError("our_score kv_indices must be contiguous int32")
    rows = int(q8.shape[0])
    segs = int(cu_seqlens.numel()) - 1
    _check_i32(cu_seqlens, segs + 1, "cu_seqlens")
    _check_i32(seq_lens, segs, "seq_lens")
    _check_i32(prefix_lens, segs, "prefix_lens")
    _check_i32(page_offsets, segs + 1, "page_offsets")
    blocks = int(blocks)
    if blocks < 1:
        raise ValueError("our_score fused prefill requires blocks >= 1")
    if blocks > 1024:
        raise ValueError(
            f"our_score fused prefill requires blocks <= 1024 (single "
            f"window); got {blocks}"
        )
    if int(topk) != 16:
        raise ValueError("our_score fused prefill requires topk == 16")
    if int(init_blocks) < 0 or int(local_blocks) < 0:
        raise ValueError("init_blocks/local_blocks must be non-negative")
    if topk_idx.ndim != 3:
        raise ValueError("our_score fused prefill output must be 3-D int32")
    if (
        topk_idx.dtype != torch.int32
        or tuple(topk_idx.shape) != (4, rows, 16)
        or topk_idx.stride(2) != 1
        or topk_idx.stride(1) < 16
        or topk_idx.stride(0) < rows * topk_idx.stride(1)
    ):
        raise ValueError(
            "our_score fused prefill output must be nonoverlapping int32 "
            f"[4,{rows},16] with stride(2)==1"
        )
    _same_device(q8, staged_k, kv_indices, cu_seqlens, seq_lens, prefix_lens,
                 page_offsets, topk_idx)
    _check_no_alias(topk_idx, q8, staged_k, kv_indices, cu_seqlens, seq_lens,
                    prefix_lens, page_offsets)
    topk_idx.fill_(-1)
    values = [
        q8.data_ptr(),
        staged_k.data_ptr(),
        kv_indices.data_ptr(),
        cu_seqlens.data_ptr(),
        seq_lens.data_ptr(),
        prefix_lens.data_ptr(),
        page_offsets.data_ptr(),
        topk_idx.data_ptr(),
        int(kv_indices.numel()),
        rows,
        segs,
        blocks,
        int(phys_pages),
        int(topk_idx.stride(0)),
        int(topk_idx.stride(1)),
        int(topk_idx.stride(2)),
        int(topk),
        int(init_blocks),
        int(local_blocks),
    ]
    _launch(
        "our_fused_prefill_launch",
        OurFusedPrefillParams,
        values,
        (q8, staged_k, kv_indices, cu_seqlens, seq_lens, prefix_lens,
         page_offsets, topk_idx),
        q8.device,
    )
    return topk_idx


def run_fused_prefill_topk(
    native_workspace, chunk_index, q8, cu_page_offsets, topk_idx, topk,
    init_blocks, local_blocks,
):
    """Production fused prefill entry over a NativeIndexWorkspace chunk.

    Same prologue contract as :func:`run_prefill_score` (the workspace's
    staged scratch + SAFE page table), but emits ``topk_idx`` directly
    (replacing the score step + the production topk kernel).  Requires the
    chunk's block count <= 1024 (single window) -- raises before launch so
    the dispatch layer can fall back to the two-step path.
    """
    chunk = native_workspace.chunks[chunk_index]
    staged = native_workspace.staged
    safe = native_workspace.safe_tables[chunk_index]
    rows = int(q8.shape[0])
    blocks = 0
    geometry = getattr(native_workspace, "_geometry", None)
    if geometry is not None and chunk_index < len(geometry):
        blocks = int(geometry[chunk_index][1])
    if blocks <= 0:
        blocks = (int(chunk.max_seqlen_k) + 127) // 128
    if blocks > 1024:
        raise ValueError(
            f"our_score fused prefill requires blocks <= 1024 (single "
            f"window); chunk has {blocks} -- fall back to the two-step path"
        )
    if rows != int(chunk.q_end - chunk.q_start):
        raise ValueError("fused prefill chunk geometry changed; rebuild workspace")
    return launch_fused_prefill(
        q8,
        staged,
        safe,
        chunk.cu_seqlens,
        chunk.seq_lens,
        chunk.prefix_lens,
        cu_page_offsets,
        topk_idx,
        blocks,
        topk,
        init_blocks,
        local_blocks,
        phys_pages=native_workspace.stage_pages,
    )


# ---------------------------------------------------------------- decode

# Cached E4M3 staging scratch [(pages+1),128,128] keyed by (device, pages):
# stable addresses matter for CUDA-graph capture/replay; the sentinel page at
# index `pages` is zeroed once at allocation and never rewritten.
_DECODE_STAGED_CACHE = {}


def _decode_staged(pages, device):
    key = (device.index, pages)
    scratch = _DECODE_STAGED_CACHE.get(key)
    if scratch is None:
        scratch = torch.empty(
            (pages + 1, 128, 128),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        scratch[pages].zero_()
        _DECODE_STAGED_CACHE[key] = scratch
    return scratch


@triton.jit(do_not_specialize=["NP"])
def _stage_our_decode_pages(K, S, O, NP, KS, SS, MMA: tl.constexpr):
    """Stage packed NVFP4 index-K pages to page-major E4M3 (16 KiB/page).

    Reuses the production dequant primitive (E2M1 x E4M3 group-16 scale,
    FP16 RN multiply, cvt.rn.satfinite.e4m3x2.f16x2, MMA-swizzle offsets
    when MMA) so decode scores match the Triton reader bit-domain.
    """
    page = tl.program_id(0).to(tl.int64)
    key = _load_index_page_fp8(K, S, page, page < NP, KS, SS, MMA)
    token = tl.arange(0, 128)
    dim = tl.arange(0, 128)
    tl.store(O + page * 16384 + token[:, None] * 128 + dim[None, :], key)


@torch.no_grad()
def our_decode_index_score(
    q,
    packed,
    scales,
    block_table,
    seq_lens,
    out,
    *,
    init_blocks,
    local_blocks,
    sm_scale,
    mma_scale_layout=False,
):
    """Exact drop-in for q8kv4_index_score (plain decode, query_width==1).

    Stages the packed NVFP4 index-K pages into the cached E4M3 scratch,
    pre-fills ``out`` with -inf on-stream, then launches the decode ABI
    (sm_scale applied after the per-block max; 1e30/1e29 sentinels and the
    [num_blocks, max_blocks) tail are written by the kernel every launch, so
    CUDA-graph replay cannot inherit stale scores).
    """
    if q.dtype != torch.float8_e4m3fn or tuple(q.shape[1:]) != (4, 128):
        raise ValueError("our_score decode Q must be E4M3 [batch,4,128]")
    if q.stride(2) != 1 or q.stride(1) != 128:
        raise ValueError("our_score decode Q rows must be contiguous [4,128]")
    if packed.dtype != torch.uint8 or scales.dtype not in (
        torch.uint8,
        torch.float8_e4m3fn,
    ):
        raise ValueError("packed index-K and E4M3 scales must be byte tensors")
    if packed.ndim != 2 or scales.ndim != 2:
        raise ValueError("packed index-K and scales must be [physical_page,bytes]")
    if packed.stride(1) != 1 or scales.stride(1) != 1:
        raise ValueError("packed index-K/scale byte rows must be contiguous")
    if packed.stride(0) < 128 * 64 or scales.stride(0) < 128 * 8:
        raise ValueError("index-K page strides are smaller than the Q8KV4 ABI")
    if packed.data_ptr() % 4 or packed.stride(0) % 4:
        raise ValueError("packed index-K page base/stride must be 4-byte aligned")
    if (
        block_table.dtype != torch.int32
        or block_table.ndim != 2
        or block_table.stride(1) != 1
    ):
        raise ValueError("block_table must be int32 [batch,max_blocks]")
    if seq_lens.dtype != torch.int32 or seq_lens.stride(0) != 1:
        raise ValueError("seq_lens must be contiguous int32 [batch]")
    batch = int(q.shape[0])
    if int(seq_lens.numel()) != batch:
        raise ValueError("seq_lens must be int32 [batch]")
    max_blocks = int(block_table.shape[1])
    expected = (4, batch, max_blocks)
    if (
        out.dtype != torch.float32
        or tuple(out.shape) != expected
        or not out.is_contiguous()
    ):
        raise ValueError(f"score output must be contiguous float32 {expected}")
    pages = int(packed.shape[0])
    _same_device(q, packed, scales, block_table, seq_lens, out)
    _check_no_alias(out, q, packed, scales, block_table, seq_lens)

    staged = _decode_staged(pages, q.device)
    _stage_our_decode_pages[(pages,)](
        packed,
        scales.view(torch.uint8),
        staged,
        pages,
        packed.stride(0),
        scales.stride(0),
        bool(mma_scale_layout),
        num_warps=4,
    )
    out.fill_(float("-inf"))
    values = [
        q.data_ptr(),
        staged.data_ptr(),
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        out.data_ptr(),
        batch,
        max_blocks,
        int(block_table.stride(0)),
        int(q.stride(0)),
        pages,
        ctypes.c_float(float(sm_scale)).value,
        int(init_blocks),
        int(local_blocks),
    ]
    _launch(
        "our_decode_launch",
        OurDecodeParams,
        values,
        (q, packed, scales, block_table, seq_lens, out, staged),
        q.device,
    )
    return out
