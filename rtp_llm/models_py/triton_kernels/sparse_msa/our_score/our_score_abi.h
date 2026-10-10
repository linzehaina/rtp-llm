// our_score_abi.h -- FROZEN C ABI for the b128/b128d tcgen05 score-kernel port
// to MiniMax-M3.1 geometry (H=4 index heads, 128-token contiguous pages,
// head_dim=128, topk=16 blocks).
//
// This header is the parallel-work contract between the CUDA implementer
// (our_score_host.cu + prefill/kernel_ourscore_prefill.cu +
// decode/kernel_ourscore_decode.cu) and the Python ctypes bridge
// (our_score_bridge.py).  It was frozen BEFORE any implementation; neither
// side may change it without bumping OUR_SCORE_ABI_VERSION.
//
// Pattern: flat struct of device pointers + int scalars + one float,
// mirroring champion_r8pkge.py's ctypes mirror of rtp_llm::Params.
// Launch = (const Params*, cudaStream_t) -> int (0 = success, nonzero =
// error -> bridge raises RuntimeError).  All device pointers must live until
// stream completion (bridge keepalive); all index tensors are int32 (the
// bridge asserts dtype==torch.int32, never int64); no host-device syncs; all
// host-side ints come from Python chunk metadata.
//
// Row indices are CHUNK-LOCAL everywhere (r in [0,rows) -- no global q_start
// offset anywhere in the store).

#pragma once

#include <stdint.h>
#include <cuda_runtime.h>

#define OUR_SCORE_ABI_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

int our_score_abi_version(void); /* must return OUR_SCORE_ABI_VERSION; bridge asserts on load */

/* ---- PREFILL (replaces NativeIndexWorkspace.score's fmha+_copy call, same output contract) ---- */
typedef struct {
    /* inputs (device) */
    const void*       q8;            /* torch.float8_e4m3fn [rows,4,128], contiguous (byte stride row = 512) */
    const void*       staged_k;      /* torch.float8_e4m3fn page-major [(phys_pages+1)*128 rows,128 cols], row-major, 16KiB/page; page phys_pages = zero sentinel */
    const int32_t*    kv_indices;    /* flat per-segment physical page ids (chunk.ku_indices); entries may be -1 or >= phys_pages -> whole block -inf, NEVER dereferenced */
    const int32_t*    cu_seqlens;    /* [segs+1] chunk-local query-row offsets, cu_seqlens[0]==0, cu_seqlens[segs]==rows */
    const int32_t*    seq_lens;      /* [segs] segment KV lengths (tokens) */
    const int32_t*    prefix_lens;   /* [segs] causal prefix (production: segment prefix + segment-local row offset) */
    const int32_t*    page_offsets;  /* [segs+1] cumulative LOGICAL page counts; segment s owns kv_indices[page_offsets[s] .. page_offsets[s+1]) */
    /* output (device) */
    float*            out;           /* fp32 [4,rows,blocks] strided view; element index = h*out_stride_h + r*out_stride_r + b; out_stride_k==1 required */
    /* geometry scalars (host ints from Python chunk metadata) */
    int64_t           kv_indices_len;/* kv_indices.numel() for bounds checks */
    int32_t           rows;          /* chunk_q = q_end-q_start, in (0,16384] */
    int32_t           segs;          /* n_segments >= 1 */
    int32_t           blocks;        /* chunk_max_pages = ceil(chunk.max_seqlen_k/128) */
    int32_t           phys_pages;    /* valid physical pages in staged_k; sentinel page index == phys_pages */
    int32_t           out_stride_h;  /* output.stride(0) in elements, >= rows*out_stride_r */
    int32_t           out_stride_r;  /* output.stride(1) in elements, >= blocks */
} OurPrefillParams;
int our_prefill_launch(const OurPrefillParams* p, cudaStream_t stream);

/* ---- DECODE (drop-in for q8kv4_index_score, plain decode only: query_width==1, no query_cu_seqlens) ---- */
typedef struct {
    /* inputs (device) */
    const void*       q8;            /* torch.float8_e4m3fn [batch,4,128]; batch stride q8_stride_b BYTES; row (token) is contiguous [4,128] */
    const void*       staged_k;      /* same layout as prefill: page-major E4M3 [(phys_pages+1)*128,128], zero sentinel at page phys_pages (bridge-staged from packed NVFP4) */
    const int32_t*    block_table;   /* [batch, bt_stride] logical block -> physical page id; -1/invalid allowed -> -inf, never dereferenced */
    const int32_t*    seq_lens;      /* [batch] */
    /* output (device) */
    float*            out;           /* fp32 [4,batch,max_blocks] CONTIGUOUS: index = (h*batch + b)*max_blocks + blk */
    /* geometry scalars */
    int32_t           batch;
    int32_t           max_blocks;    /* logical block-table width (num columns of out) */
    int32_t           bt_stride;     /* block_table row stride in ELEMENTS (>= max_blocks) */
    int32_t           q8_stride_b;   /* q8 batch stride in BYTES (512 when contiguous) */
    int32_t           phys_pages;    /* valid pages in staged_k; sentinel == phys_pages */
    /* numerics (decode-only contract) */
    float             sm_scale;      /* plain fp32 multiplier applied AFTER the per-block max (production default 128**-0.5) */
    int32_t           init_blocks;   /* blk <  init_blocks (and blk < local_start) -> literal 1e30 */
    int32_t           local_blocks;  /* blk >= num_blocks - local_blocks -> literal 1e29 (applied AFTER init on overlap) */
} OurDecodeParams;
int our_decode_launch(const OurDecodeParams* p, cudaStream_t stream);

/* ---- FUSED PREFILL (score + top-k in one kernel; added WITHOUT touching the
   v1 symbols above -- the bridge gates on symbol availability, not the ABI
   version, so old .so <-> new bridge combinations fail cleanly) ---- */
typedef struct {
    /* inputs (device) -- identical contract to OurPrefillParams */
    const void*       q8;            /* torch.float8_e4m3fn [rows,4,128], contiguous (byte stride row = 512) */
    const void*       staged_k;      /* torch.float8_e4m3fn page-major [(phys_pages+1)*128 rows,128 cols], row-major, 16KiB/page; page phys_pages = zero sentinel */
    const int32_t*    kv_indices;    /* flat per-segment physical page ids (the workspace SAFE table in production); entries may be -1 or >= phys_pages -> block value -inf, NEVER dereferenced */
    const int32_t*    cu_seqlens;    /* [segs+1] chunk-local query-row offsets, cu_seqlens[0]==0, cu_seqlens[segs]==rows */
    const int32_t*    seq_lens;      /* [segs] segment KV lengths (tokens) */
    const int32_t*    prefix_lens;   /* [segs] causal prefix (production: segment prefix + segment-local row offset) */
    const int32_t*    page_offsets;  /* [segs+1] cumulative LOGICAL page counts; segment s owns kv_indices[page_offsets[s] .. page_offsets[s+1]) */
    /* output (device) */
    int32_t*          topk_idx;      /* int32 [4,rows,topk] strided view; element index = h*ti_stride_h + r*ti_stride_r + slot*ti_stride_t; ti_stride_t==1 required */
    /* geometry scalars (host ints from Python chunk metadata) */
    int64_t           kv_indices_len;/* kv_indices.numel() for bounds checks */
    int32_t           rows;          /* chunk_q = q_end-q_start, in (0,16384] */
    int32_t           segs;          /* n_segments >= 1 */
    int32_t           blocks;        /* chunk_max_pages = ceil(chunk.max_seqlen_k/128); MUST be <= 1024 (single window: grid z == 1) */
    int32_t           phys_pages;    /* valid physical pages in staged_k; sentinel page index == phys_pages */
    int32_t           ti_stride_h;   /* topk_idx.stride(0) in elements, >= rows*ti_stride_r */
    int32_t           ti_stride_r;   /* topk_idx.stride(1) in elements, >= topk */
    int32_t           ti_stride_t;   /* topk_idx.stride(2) in elements, MUST be 1 */
    int32_t           topk;          /* MUST be 16 (the register state is 16-slot) */
    int32_t           init_blocks;   /* blk < init_blocks -> forced value 1e30 (before local) */
    int32_t           local_blocks;  /* blk >= vbl-local_blocks -> forced value 1e29 (after init; overrides on overlap) */
} OurFusedPrefillParams;
int our_fused_prefill_launch(const OurFusedPrefillParams* p, cudaStream_t stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

/*
SEMANTICS (binding on both implementers):

PREFILL: for every (h in [0,4), r in [0,rows), b in [0,blocks)) the FULL
extent is defined after the call (bridge pre-fills out with -inf on-stream
via out.fill_(-inf) -- the pattern already used at
native_q8_index_score.py:502 -- and the kernel writes only finite values):
with segment s such that cu_seqlens[s] <= r < cu_seqlens[s+1],
q_pos = prefix_lens[s] + (r - cu_seqlens[s]), logical page index
lp = page_offsets[s] + b, phys = kv_indices[lp]:

  out[h,r,b] = RAW scale-one E4M3 dot max over visible tokens
               t in [b*128, b*128+128) with t <= q_pos AND t < seq_lens[s],
               of dot(q8[r,h,:], staged_k[phys, t, :]);

  -inf iff b*128 >= seq_lens[s], or b*128 > q_pos, or lp >= page_offsets[s+1],
       or phys invalid (<0 or >= phys_pages).

NO sm_scale, NO per-request q/k scales (xsc folded to literal 1.0f), no NaN
permitted.  Row indices are CHUNK-LOCAL (r in [0,rows) -- no global q_start
offset anywhere in the store).

FUSED PREFILL: emits topk_idx[h,r,slot] for EVERY (h, r, slot in [0,topk))
in one launch (no score tensor round trip).  Per row r (segment s, q_pos as
above), let vbl = min(ceil(min(q_pos+1, seq_lens[s]) / 128), blocks) -- the
kernel-2 `valid_blocks` = (prefix+row+128)//128 under the causal contract
q_pos < seq_lens (kernel-2 does not kv-clamp; production never violates the
contract).  Candidates are blocks b in [0, vbl) with value

  val(b) = RAW scale-one E4M3 dot max over visible tokens (same value the
           PREFILL entry would store at out[h,r,b]), then kernel-2's
           MASK_INIT=False / MASK_LOCAL=False overrides IN ITS ORDER:
           b < init_blocks -> 1e30; THEN b >= vbl - local_blocks -> 1e29
           (local overrides init on overlap).

Blocks whose val(b) is -inf (invalid page / never staged) are NOT
candidates: kernel-2's -1e30 0-padding entries outrank -inf scores, so both
paths emit -1 for slots past the candidate count.  topk_idx[h,r,slot] =
the slot-th best candidate id (value descending, ties -> smaller block id),
-1 for slot >= min(topk, vbl) or past the candidate count.  Candidate
values are held in f16 inside the fused state: selection SETS may differ
from an fp32 oracle when two candidates sit within one f16 ulp (near-tie);
the bridge/gate documents this tolerance.  SINGLE WINDOW only: blocks MUST
be <= 1024 (the launch grid is (ceil(rows/64), segs, 1); the finalize emits
directly because no page-window z-split can occur) -- larger block counts
return error 1 and the caller must fall back to the two-step path.

DECODE: out[h,b,blk] = sm_scale * max over t in [blk*128,
min((blk+1)*128, seq_lens[b])) of dot(q8[b,h,:],
staged_k[block_table[b,blk], t, :]) for valid blk < num_blocks(b) =
ceil(seq_lens[b]/128) and valid page; -inf for blk >= num_blocks, invalid
page, or invisible tokens (none in plain decode -- all t < seq_len visible);
overrides in this exact order: (1) if blk < init_blocks and blk <
local_start: 1e30; (2) then if blk >= num_blocks - local_blocks: 1e29 (local
overwrites init on overlap); out[(h*batch+b)*max_blocks + blk] for ALL blk in
[0,max_blocks) defined every call (tail [num_blocks,max_blocks) = -inf every
launch, CUDA-graph replay safe).  Sentinel literals are exact float
constants 1e30f/1e29f.  (Production-reader parity: the 1e30/1e29 sentinels
apply only to otherwise-valid blocks -- valid page id and blk < num_blocks;
an invalid page is -inf regardless of its init/local position, matching
nvfp4_q8_index_score.py's valid_page gating.)
*/
