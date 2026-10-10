"""Frozen native hybrid decode attention for the Q8KV4 sparse decode chain.

One ctypes launch (``launch_native_hybrid_attention``) replaces the Triton
``_q8kv4_selected|four_page_partial`` + ``_q8kv4_combine`` attention stage of
``q8kv4_paged_sparse_decode``.  The binary is the 13-round decode campaign's
frozen champion P6-2-residual (``libdecode_hybrid_p6res.so``, see
``NATIVE_HYBRID_DECODE.md`` beside this file for provenance).

The champion was designed against this exact production cache layout and
consumes every input zero-copy:

* ``q8``            E4M3 ``[rows, 64, 128]``   -- the entry's workspace carrier
* K/V planes        uint8 ``[pages, 32768]``   -- ``NVFP4CacheLayout`` planar
  slices of ``packed_main``; scale planes uint8 ``[pages, 4096]`` in the MMA
  128x4 swizzle (``mma_scale_layout=True``), all with caller page strides
* ``topk_indices``  int32 ``[4, rows, 16]``    -- the entry's workspace buffer
* per-request page table (a stride-view of the per-row table; rows of one
  request share the table by the entry's contract)
* direct batch metadata: ``cu`` (query cumsum), ``cu_k`` (allocated-KV
  cumsum), ``used`` (per-request used KV).  ``lens`` stays NULL; per-row
  visibility is derived on device as ``used - (w - 1 - row)``.

The partial/page-LSE workspaces are byte-compatible with the production
``_Q8KV4DecodeWorkspace`` buffers (``[rows, 4, 16, 16, 128]`` BF16 and
``[rows, 4, 16, 16]`` FP32), so the entry passes its own cached tensors
through unchanged.  Output ``O`` is BF16 ``[rows, 64, 128]``; the fused
natural-log global LSE ``[rows, 64]`` is written as a free byproduct into a
shape-bucket-cached buffer (the production stage never materializes it).

Measured against the production Triton attention stage on L20D (19-cell
matrix, 2026-10-10): O bitwise-identical on every cell, attention-stage
kernel time 1.27x-3.39x faster (anchor dec-w6-bsz30-kv128000: 80.5us vs
262.8us).  CUDA-graph capture/replay was verified bitwise in the same
campaign; the launcher keeps graph safety by allocating only from the
shape-bucket caches below (stable addresses across replays).

Dispatch is reversible without code changes:

* ``MSA_USE_NATIVE_HYBRID_DECODE=0`` disables the route (default on).
* ``MSA_NATIVE_HYBRID_LIB=/path/to/lib.so`` loads an operator-supplied
  binary instead of the bundled artifact (sha not enforced for overrides).

The library resolves ``libcudart`` through the normal loader path and falls
back to the copy bundled with the installed torch on ``OSError``.  If the
binary cannot be loaded the route is skipped (once, with a warning) and the
caller falls back to the original attention stage.
"""

import ctypes
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch

from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
    NVFP4CacheLayout,
    NVFP4LogicalViews,
)

logger = logging.getLogger(__name__)

_LIB_FILENAME = "libdecode_hybrid_p6res.so"
_MAIN_ENTRY = "launch_native_hybrid_attention"
_BINARY_SHA256 = "e1d0b7efa3acea405cfd88764126bdfc652a25cd992e0fe22a4e3c634ea4a7b5"

# sm_103a build (L20D); the campaign never validated other devices.
_REQUIRED_CAPABILITY = (10, 3)
_MAX_QUERY_WIDTH = 8
# Static per-request allocated-KV upper bound for cu_k.  The kernels only use
# cu_k to clamp ``used`` (``min(used, allocated)``), so any bound >= the model
# context limit keeps production visibility semantics; a static tensor avoids
# a per-step cumsum launch.  Keep bsz * _ALLOC_KV_HINT inside int32.
_ALLOC_KV_HINT = 1 << 22
_MAX_REQUESTS = (1 << 31) // _ALLOC_KV_HINT - 1

# The kernels compute in the log2 domain: scores are pre-multiplied by
# 128^-0.5 * log2(e) in float32 (do NOT re-multiply).
ALPHA = ctypes.c_float(
    ctypes.c_float(128**-0.5).value * ctypes.c_float(math.log2(math.e)).value
).value


class _NativeHybridParams(ctypes.Structure):
    """ctypes mirror of the C ABI struct ``multi::Params`` (stage3/attention.cu)."""

    _fields_ = (
        [
            (name, ctypes.c_void_p)
            for name in (
                "q",
                "k",
                "v",
                "ks",
                "vs",
                "table",
                "topk",
                "lens",
                "valid",
                "partial",
                "out",
                "lse",
            )
        ]
        + [(name, ctypes.c_int) for name in ("rows", "pages", "max_pages")]
        + [
            (name, ctypes.c_int64)
            for name in (
                "kp",
                "vp",
                "ksp",
                "vsp",
                "table_row",
                "topk_head",
                "topk_row",
                "valid_stride",
            )
        ]
        + [
            ("alpha", ctypes.c_float),
            ("cu", ctypes.c_void_p),
            ("requests", ctypes.c_int),
            ("max_query_width", ctypes.c_int),
            ("finished", ctypes.c_void_p),
            ("tmem_trace", ctypes.c_void_p),
            ("global_lse", ctypes.c_void_p),
            ("cu_k", ctypes.c_void_p),
            ("used", ctypes.c_void_p),
        ]
    )


class _NativeHybridLibrary:
    """Content-SHA-verified ctypes handle of the frozen champion binary."""

    def __init__(self, path: Path, launch, binary_sha256: str):
        self.path = path
        self.launch = launch
        self.binary_sha256 = binary_sha256


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_LIBRARY: _NativeHybridLibrary | None = None
_LIBRARY_ERROR = ""


def _library_path() -> Path:
    override = os.environ.get("MSA_NATIVE_HYBRID_LIB")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / _LIB_FILENAME


def _open_with_cudart(path: Path):
    try:
        return ctypes.CDLL(str(path))
    except OSError:
        # libcudart may be missing from the loader path; preload the copy
        # bundled with the installed torch and retry once.
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        for candidate in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
            bundled = torch_lib / candidate
            if bundled.is_file():
                ctypes.CDLL(str(bundled), mode=ctypes.RTLD_GLOBAL)
                break
        return ctypes.CDLL(str(path))


def load_library() -> _NativeHybridLibrary | None:
    """Load (and memoize) the frozen champion binary; ``None`` on failure.

    The bundled artifact is hash-pinned: a corrupted or replaced file disables
    the route (with a warning) instead of serving unverified numerics.
    Explicit ``MSA_NATIVE_HYBRID_LIB`` overrides are operator-supplied and
    only logged.
    """
    global _LIBRARY, _LIBRARY_ERROR
    if _LIBRARY is not None or _LIBRARY_ERROR:
        return _LIBRARY
    path = _library_path()
    try:
        if not path.is_file():
            raise OSError(f"native hybrid decode binary not found: {path}")
        sha = _file_sha256(path)
        if (
            path == Path(__file__).resolve().parent / _LIB_FILENAME
            and sha != _BINARY_SHA256
        ):
            raise RuntimeError(
                f"native hybrid decode binary sha {sha} != pinned "
                f"{_BINARY_SHA256} ({path})"
            )
        library = _open_with_cudart(path)
        entry = getattr(library, _MAIN_ENTRY)
    except (OSError, AttributeError, RuntimeError) as error:
        _LIBRARY_ERROR = str(error)
        logger.warning("native hybrid decode disabled: %s", error)
        return None
    entry.argtypes = [
        ctypes.POINTER(_NativeHybridParams),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    entry.restype = ctypes.c_int
    _LIBRARY = _NativeHybridLibrary(path, entry, sha)
    if sha != _BINARY_SHA256:
        logger.info("native hybrid decode using override %s (sha256 %s)", path, sha)
    return _LIBRARY


def native_hybrid_decode_enabled() -> bool:
    """Per-call env read so the route can be toggled without re-import."""
    return os.environ.get("MSA_USE_NATIVE_HYBRID_DECODE", "1") == "1"


@dataclass
class _NativeHybridWorkspace:
    """Shape-bucket-cached metadata and output-side buffers.

    Geometry buffers (cu, cu_k) hold values that are constant per captured
    shape, so replays read the same device memory the capture bound.
    ``global_lse`` is the champion's fused natural-log LSE byproduct.
    """

    global_lse: torch.Tensor
    cu: torch.Tensor
    cu_k: torch.Tensor
    used: torch.Tensor | None

    _CACHE: ClassVar[dict[tuple, "_NativeHybridWorkspace"]] = {}

    @classmethod
    def acquire(
        cls, q8: torch.Tensor, requests: int, query_width: int
    ) -> "_NativeHybridWorkspace":
        rows = int(q8.shape[0])
        key = (str(q8.device), rows, requests, query_width)
        workspace = cls._CACHE.get(key)
        if workspace is None:
            device = q8.device
            cu = torch.arange(requests + 1, device=device, dtype=torch.int32)
            cu_k = torch.arange(requests + 1, device=device, dtype=torch.int32)
            workspace = cls(
                global_lse=torch.empty((rows, 64), dtype=torch.float32, device=device),
                cu=cu * query_width,
                cu_k=cu_k * _ALLOC_KV_HINT,
                used=(
                    torch.empty((requests,), dtype=torch.int32, device=device)
                    if query_width > 1
                    else None
                ),
            )
            cls._CACHE[key] = workspace
        return workspace


def _geometry_supported(
    q: torch.Tensor,
    layout: NVFP4CacheLayout,
    topk: int,
    query_width: int,
    rows: int,
    requests: int,
) -> bool:
    """Static geometry gate: the champion's validated M3.1 contract."""
    if not q.is_cuda or tuple(q.shape[1:]) != (64, 128):
        return False
    if layout.num_heads != 4 or layout.page_size != 128:
        return False
    if layout.head_dim != 128 or topk != 16:
        return False
    if rows <= 0 or rows % query_width or requests > _MAX_REQUESTS:
        return False
    try:
        capability = torch.cuda.get_device_capability(q.device)
    except Exception:  # pragma: no cover - defensive: malformed device
        return False
    return capability == _REQUIRED_CAPABILITY


def native_hybrid_decode_supported(
    q: torch.Tensor,
    layout: NVFP4CacheLayout,
    topk: int,
    mma_scale_layout: bool,
    query_width: int,
    query_cu_seqlens: torch.Tensor | None,
    block_table: torch.Tensor | None = None,
    output_format: str = "bf16",
) -> bool:
    """Static producer-side predicate: geometry, format and library only.

    No tensor readback, no device sync -- safe inside CUDA graph capture.
    Ragged (query_cu_seqlens) verification stays on the production path; the
    champion's ragged cu support was never campaign-validated.
    """
    if not native_hybrid_decode_enabled():
        return False
    if output_format != "bf16" or not mma_scale_layout:
        return False
    if query_cu_seqlens is not None:
        return False
    if not 1 <= query_width <= _MAX_QUERY_WIDTH:
        return False
    if block_table is not None and (
        block_table.dtype != torch.int32
        or block_table.ndim != 2
        or block_table.stride(1) != 1
    ):
        # Graph reserve tables are zero-copy strided column views; any row
        # stride is fine (the request view uses as_strided), only element
        # contiguity within a row is required.
        return False
    rows = int(q.shape[0])
    if not _geometry_supported(q, layout, topk, query_width, rows, rows // query_width):
        return False
    return load_library() is not None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def native_hybrid_decode_attention(
    q8: torch.Tensor,
    logical: NVFP4LogicalViews,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    layout: NVFP4CacheLayout,
    query_width: int,
    out: torch.Tensor,
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
) -> torch.Tensor:
    """Launch the frozen champion (K1 selected-page partial + K2 combine).

    All tensor arguments are the entry's own workspace/cache carriers
    (zero-copy).  Returns ``out`` (BF16 ``[rows, 64, 128]``); the fused
    natural-log global LSE lands in this route's cached buffer.
    """
    library = load_library()
    if library is None:
        raise RuntimeError(
            "native hybrid decode selected but the binary is unavailable"
        )
    rows = int(q8.shape[0])
    requests = rows // query_width
    max_pages = int(block_table.shape[1])
    _require(
        q8.dtype == torch.float8_e4m3fn
        and tuple(q8.shape) == (rows, 64, 128)
        and q8.is_contiguous(),
        f"q8 must be contiguous E4M3 [{rows},64,128]",
    )
    _require(
        topk_indices.dtype == torch.int32
        and tuple(topk_indices.shape) == (4, rows, 16),
        f"topk must be int32 [4,{rows},16], got "
        f"{topk_indices.dtype} {tuple(topk_indices.shape)}",
    )
    _require(
        block_table.dtype == torch.int32
        and block_table.ndim == 2
        and block_table.stride(1) == 1,
        "block_table must be int32 [rows,max_pages] with contiguous rows",
    )
    _require(
        seq_lens.dtype == torch.int32 and tuple(seq_lens.shape) == (rows,),
        f"seq_lens must be int32 [{rows}]",
    )
    _require(
        out.dtype == torch.bfloat16
        and tuple(out.shape) == (rows, 64, 128)
        and out.is_contiguous()
        and partial_out.is_contiguous()
        and partial_lse.is_contiguous(),
        "output buffers must be contiguous [rows,64,128] BF16 workspaces",
    )

    # Per-request page table: rows of one request share the table (entry
    # contract), so slice the first row of each request.  Graph reserve
    # tables are strided column views, hence as_strided (any row stride)
    # instead of a contiguous view.
    table_row_stride = block_table.stride(0)
    table = block_table.as_strided(
        (requests, max_pages),
        (query_width * table_row_stride, block_table.stride(1)),
    )

    workspace = _NativeHybridWorkspace.acquire(q8, requests, query_width)
    # Per-request used KV follows the production lens convention
    # lens[row] = used - (w - 1 - row); the last row of each request carries
    # the full used length.  Plain decode reuses seq_lens zero-copy.
    if query_width == 1 and seq_lens.is_contiguous():
        used = seq_lens
    else:
        lens_stride = seq_lens.stride(0)
        lens_2d = seq_lens.as_strided(
            (requests, query_width), (query_width * lens_stride, lens_stride)
        )
        workspace.used.copy_(lens_2d[:, query_width - 1])
        used = workspace.used

    k, v = logical.main_k_fp4, logical.main_v_fp4
    ks = logical.main_k_scale.view(torch.uint8)
    vs = logical.main_v_scale.view(torch.uint8)
    for name, plane in (("k", k), ("v", v), ("ks", ks), ("vs", vs)):
        _require(
            plane.stride(1) == 1
            and plane.data_ptr() % 4 == 0
            and plane.stride(0) % 4 == 0,
            f"{name} plane needs contiguous 4B-aligned rows",
        )

    params = _NativeHybridParams(
        q8.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        ks.data_ptr(),
        vs.data_ptr(),
        table.data_ptr(),
        topk_indices.data_ptr(),
        0,  # lens: NULL -- direct batch metadata ABI
        0,  # valid: NULL -- padded rows have seq_lens 0 (O stays 0)
        partial_out.data_ptr(),
        out.data_ptr(),
        partial_lse.data_ptr(),
        rows,
        int(layout.num_blocks),
        max_pages,
        k.stride(0),
        v.stride(0),
        ks.stride(0),
        vs.stride(0),
        table.stride(0),
        topk_indices.stride(0),
        topk_indices.stride(1),
        1,
        ALPHA,
        workspace.cu.data_ptr(),
        requests,
        query_width,
        0,
        0,  # finished, tmem_trace
        workspace.global_lse.data_ptr(),
        workspace.cu_k.data_ptr(),
        used.data_ptr(),
    )
    error = library.launch(
        ctypes.byref(params),
        ctypes.c_void_p(torch.cuda.current_stream(q8.device).cuda_stream),
        1,  # swizzled: MMA 128x4 scale layout (mma_scale_layout=True)
        0,  # mode 0: K1 + K2 in one call
    )
    if error:
        raise RuntimeError(f"native hybrid decode launch failed: CUDA error {error}")
    return out
