"""Champion R8-PKG-E prefill kernel bridge (MSAINTEGRATE108, Phase 3).

Env-gated integration of the frozen prefill champion
``frozen/glm/prefill-r8pkge-c58d6a73.so`` (R8-PKG-E, composite source
275bf397 + R4 TMA loader + PKG-S softmax diet + R6 late-den reorder + R8
chain trims) into the MiniMax-M3.1 native Q8KV4 prefill path.  The module is
imported lazily by ``topk_bt_fused._sparse_attn_chunked`` ONLY when
``MSA_INTEGRATION_CHAMPION=1``; the default production dispatch
(fmha_sm100 CSR build + ``sparse_atten_nvfp4_kv_func``) is untouched.

ABI: ctypes mirror of ``rtp_llm::Params`` in the champion kernel.cu
(12 pointers + 9 int64 + 4 int + alpha struct; identical field order to the
campaign binding harness/glm_kimi_bind.py).

INPUT CONTRACT (all zero-copy from the m3.1 native planes):
  q8              E4M3 [csz, 64, 128]          (scale1_query_fp8 output)
  k_pages         uint8 [pages, 4, 128, 64]    packed NVFP4 K plane
  v_pages         uint8 [pages, 4, 128, 64]    packed NVFP4 V plane
  k_scale/v_scale uint8 view of [pages, 4, 128, 8] E4M3 planes whose bytes are
                  already in the cuBLAS/cuDNN 128x4 MMA swizzle the champion
                  reads (so = page*4096 + head*1024 + mma_off(t, g))
  topk            int32 [4, csz, 16]           per-row selection, -1 padded
  page_table      int32 [nb, max_pages]        physical page ids per segment
  cu_q/cu_k/used  int32 chunk-local cu_seqlens (tokens) / causal KV lengths

SELECTION SEMANTICS (load-bearing, see the 13-round campaign contract): the
champion consumes ONE top-k list per 8-row query tile (it reads
``topk[head, tile_base_row, :]``), the production chunk-plan contract.  The
m3.1 topk lists are per-query-row, so this bridge reduces each tile to its
LAST row's list (widest causal window) before the launch -- the same
``_tile_share_q2k`` reduction the campaign harness applies
(harness/fixtures.py).  This runs INSIDE the champion call (its cost is part
of the variant).  On tile-shared selections the two contracts are identical
and the champion is bitwise-equal to production; on per-row selections this
is a documented reinterpretation (each 8-row tile attends the tile's shared
16 blocks with per-row causal masking, instead of each row's own 16).

Run with: /data0/liukan.lk/RTP-LLM/build_logs/msa-nvfp4-kda/venv/bin/python
"""

import ctypes
import logging
import os
from pathlib import Path

import torch

LOG2E = 1.4426950408889634

_LIB_DIR = Path(__file__).resolve().parent / 'champion_src' / 'lib'
_REPO_LIB = _LIB_DIR / 'prefill-r8pkge-c58d6a73.so'
_REPO_LIB_GLIBC232 = _LIB_DIR / 'prefill-r8pkge-c58d6a73-glibc232.so'
_CAMPAIGN_LIB = Path(
    '/data0/liukan.lk/RTP-LLM/build_logs/msa-integration-final-20261008/'
    'frozen/glm/prefill-r8pkge-c58d6a73.so')
_DEFAULT_SHA256 = (
    '654282ca25edce2c765d12941521ed9839d3dcf911cb98e666da1653130f23ca'
)

# Fixed champion geometry (asserted at bind time; anything else falls back).
PAGE, HEADS, QH, DIM, TOPK = 128, 4, 64, 128, 16


class Params(ctypes.Structure):
    """Exact ctypes mirror of rtp_llm::Params in the champion kernel.cu."""

    _fields_ = [(name, ctypes.c_void_p) for name in (
        'q', 'k', 'v', 'ks', 'vs', 'topk', 'table', 'cuq', 'cuk', 'used',
        'out', 'lse')] + [
        (name, ctypes.c_int64) for name in (
            'qn', 'qh', 'kp', 'vp', 'th', 'tq', 'tb', 'ksp', 'vsp')] + [
        (name, ctypes.c_int) for name in (
            'pages', 'max_pages', 'max_q', 'batches')] + [
        ('alpha', ctypes.c_float),
        ('debug', ctypes.c_void_p), ('k8', ctypes.c_void_p),
        ('v8', ctypes.c_void_p), ('rowlens', ctypes.c_void_p),
        ('partial', ctypes.c_void_p), ('partial_lse', ctypes.c_void_p),
        ('counts', ctypes.c_void_p)]


_LIB = None


def _file_sha256(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _preload_legacy_glibc_providers():
    """Best-effort RTLD_GLOBAL preload of the pre-2.34 split-out glibc libs.

    The champion binary was linked against glibc >= 2.34 (merged libc), so
    its DT_NEEDED lists only libc.so.6; on older stacks (the dev container
    runs glibc 2.32) the pthread/sem/dl/shm symbols live in libpthread /
    libdl / librt, which are then only present if the host process happens
    to load them. Preloading makes the dependency explicit. On glibc >= 2.34
    these are one-line compat stubs, so the preload is a no-op there.
    """
    for name in ('librt.so.1', 'libpthread.so.0', 'libdl.so.2', 'libutil.so.1'):
        try:
            ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


def _default_candidates():
    """Repo copy first (SHA-pinned), then its glibc-2.32 remap for hosts
    whose glibc cannot load the original, then the campaign freeze.  The
    original binary is linked against glibc >= 2.34; on older stacks CDLL
    raises and the remapped copy (identical GPU code, ELF version table
    remapped) loads instead."""
    yield _REPO_LIB, True
    yield _REPO_LIB_GLIBC232, False
    yield _CAMPAIGN_LIB, True


def get_library():
    """Content-SHA-verified CDLL of the frozen R8-PKG-E champion.

    The binary path is configurable via ``MSA_CHAMPION_LIB`` (preferred) or
    the campaign-era ``MSA_INTEGRATION_CHAMPION_LIB`` (both skip the SHA
    check, since the operator picked the file deliberately).  The default
    resolution tries the repo-committed champion binaries first and falls
    back to the campaign freeze; see ``champion_src/README.md``.
    """
    global _LIB
    if _LIB is not None:
        return _LIB
    override = (os.environ.get('MSA_CHAMPION_LIB')
                or os.environ.get('MSA_INTEGRATION_CHAMPION_LIB'))
    _preload_legacy_glibc_providers()
    if override:
        path = override
        sha = _file_sha256(path)
        lib = ctypes.CDLL(path)
    else:
        path = sha = lib = None
        for cand, check_sha in _default_candidates():
            if not cand.exists():
                continue
            cand_sha = _file_sha256(cand)
            if check_sha and cand_sha != _DEFAULT_SHA256:
                raise RuntimeError(
                    f'champion binary sha {cand_sha} != identity '
                    f'{_DEFAULT_SHA256} ({cand})')
            try:
                lib = ctypes.CDLL(str(cand))
            except OSError:
                continue  # glibc < 2.34: try the remapped copy next
            path, sha = str(cand), cand_sha
            break
        if lib is None:
            raise RuntimeError(
                'no loadable champion binary (expected champion_src/lib/'
                'prefill-r8pkge-c58d6a73.so); rebuild with champion_src/'
                'build.sh or set MSA_CHAMPION_LIB')
    if not hasattr(lib, 'msa_launch'):
        raise RuntimeError(f'champion binary {path} missing msa_launch')
    lib.msa_launch.argtypes = [ctypes.POINTER(Params), ctypes.c_void_p]
    lib.msa_launch.restype = ctypes.c_int
    _LIB = lib
    logging.info(
        '[MSA champion] loaded %s (sha256 %s..)', path, sha[:12])
    return lib


# Reusable E4M3 working caches (pages, 4, 128, 128), keyed by (pages, device).
# The champion chain decodes the selected packed-NVFP4 pages into these every
# call (mark + decode_working_cache); allocation is one-off per pool size.
_K8_CACHE = {}

# Reusable LSE buffers [csz, 64] FP32, keyed by (csz, device.index).
_LSE_CACHE = {}


def _lse_buffer(csz, device):
    key = (csz, device.index)
    if key not in _LSE_CACHE:
        _LSE_CACHE[key] = torch.empty(
            (csz, QH), dtype=torch.float32, device=device)
    return _LSE_CACHE[key]


def _working_caches(pages, device):
    key = (pages, device.index)
    if key not in _K8_CACHE:
        _K8_CACHE[key] = tuple(
            torch.empty((pages, HEADS, PAGE, DIM), dtype=torch.float8_e4m3fn,
                        device=device)
            for _ in range(2))
    return _K8_CACHE[key]


def _tile_last_row_index(cu_q, csz, device):
    """int64 [csz] gather index: every row adopts its 8-row tile's LAST row
    (production chunk-plan reduction, mirrors harness _tile_share_q2k).

    Fully vectorized on GPU (no host loop / no device sync): the python-loop
    form cost ~9 ms of host time per call at 48k rows, dwarfing the kernel.
    """
    nb = int(cu_q.numel()) - 1
    seg_lens = (cu_q[1:] - cu_q[:-1])
    rows = torch.arange(csz, device=device)
    seg = torch.repeat_interleave(torch.arange(nb, device=device), seg_lens)
    q0 = cu_q[seg]
    t = (rows - q0) // 8
    last = torch.minimum(q0 + t * 8 + 7, cu_q[seg + 1] - 1)
    return last.to(torch.int64)


class ChampionChunkRunner:
    """Per-forward champion bind over the m3.1 native planes.

    Built once per ``_sparse_attn_chunked`` call; ``__call__`` runs one chunk
    (tile reduction + msa_launch: mark + decode_working_cache + resident12).
    """

    def __init__(self, q, k_pages, v_pages, k_scale, v_scale, topk, sm_scale,
                 reduce_tiles=True):
        self.lib = get_library()
        self.device = q.device
        self.k_pages, self.v_pages = k_pages, v_pages
        self.k_scale, self.v_scale = k_scale, v_scale
        self.sm_scale = float(sm_scale)
        self.reduce_tiles = reduce_tiles
        self.pages = int(k_pages.shape[0])
        self.kp, self.vp = int(k_pages.stride(0)), int(v_pages.stride(0))
        self.scale_page_stride = int(k_scale.numel()) // self.pages
        self.k8, self.v8 = _working_caches(self.pages, self.device)
        self.alpha = self.sm_scale * LOG2E
        # counters for the integration log line
        self.chunks_run = 0

    def __call__(self, q8, topk_chunk, c, out):
        csz = int(q8.shape[0])
        nb = int(c["nb"])
        cu_q, cu_k, used, pt = c["cu_q"], c["cu_k"], c["seqused"], c["pt"]
        lse = _lse_buffer(csz, self.device)
        if self.reduce_tiles:
            # the tile map depends only on cu_q (chunk-stable): cache it in
            # the chunk dict so layers 2..N of a forward skip the rebuild
            src = c.get('_champ_tile_src')
            if src is None or int(src.numel()) != csz:
                src = _tile_last_row_index(cu_q, csz, self.device)
                c['_champ_tile_src'] = src
            topk_run = topk_chunk.index_select(1, src)
        else:
            topk_run = topk_chunk.contiguous()
        tensors = (q8, self.k_pages, self.v_pages, self.k_scale, self.v_scale,
                   topk_run, pt, cu_q, cu_k, used, out, lse)
        vals = [x.data_ptr() for x in tensors] + [
            q8.stride(0), q8.stride(1), self.kp, self.vp,
            topk_run.stride(0), topk_run.stride(1), pt.stride(0),
            self.scale_page_stride, self.scale_page_stride,
            self.pages, int(pt.shape[1]), int(c["max_q"]), nb,
            ctypes.c_float(self.alpha).value,
            0, self.k8.data_ptr(), self.v8.data_ptr(), 0, 0, 0, 0]
        if len(vals) != len(Params._fields_):
            raise TypeError(f'champion bind: {len(vals)} vals vs '
                            f'{len(Params._fields_)} Params fields')
        p = Params(*vals)
        stream = ctypes.c_void_p(
            torch.cuda.current_stream(self.device).cuda_stream)
        err = self.lib.msa_launch(ctypes.byref(p), stream)
        if err:
            raise RuntimeError(f'champion msa_launch failed: CUDA {err}')
        self.chunks_run += 1
        # keep Params-referenced buffers alive through the async launch
        self._keepalive = (tensors, topk_run, lse)
        return out


def make_chunk_runner(q, k_pages, v_pages, k_scale, v_scale, topk, sm_scale,
                      block_size_k):
    """Validate geometry and build a ChampionChunkRunner, or None to fall back.

    The champion's baked geometry: page 128, 4 KV heads, 128 head dim, topk 16,
    packed planes [pages, 4, 128, 64] uint8, 128x4-MMA scale planes.
    """
    if k_scale is None or v_scale is None:
        return None
    try:
        if (int(block_size_k) != PAGE or int(k_pages.shape[1]) != HEADS
                or int(k_pages.shape[2]) != PAGE
                or int(k_pages.shape[3]) != DIM // 2
                or tuple(v_pages.shape) != tuple(k_pages.shape)
                or k_pages.dtype != torch.uint8
                or int(topk) != TOPK
                or int(q.shape[1]) != QH or int(q.shape[2]) != DIM
                or int(k_scale.numel()) % int(k_pages.shape[0]) != 0):
            return None
        if not (k_pages.is_contiguous() and v_pages.is_contiguous()):
            return None
    except Exception:
        return None
    reduce_tiles = os.environ.get(
        'MSA_INTEGRATION_CHAMPION_TILE_REDUCE', '1') == '1'
    return ChampionChunkRunner(
        q, k_pages, v_pages, k_scale, v_scale, TOPK, sm_scale,
        reduce_tiles=reduce_tiles)
