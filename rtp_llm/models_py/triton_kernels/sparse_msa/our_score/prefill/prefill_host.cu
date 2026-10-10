// prefill_host.cu -- host launcher for our_prefill_kernel (the
// our_prefill_launch entry of the frozen our_score_abi.h C ABI).
//
// This file is #included by our_score_host.cu (the frozen build command
// compiles exactly three TUs; the prefill kernel itself lives in
// prefill/kernel_ourscore_prefill.cu and is launched cross-TU through
// common/our_kernels_decl.h).
//
// Port of b128_launch() from
//   /home/liukan.lk/m3pro-128blk-20261009/src/kernel_b128.cu
// with the M3.1 changes: int32 chunk metadata, grid (q_blocks, segs,
// page_chunks) with blockIdx.y = chunk-local segment (no y_order/LPT
// remap, no one-wave page-chunk heuristic -- admission bounds blocks <=
// 1024 so a single grid.z tile covers the whole block range; a defensive
// z-split is added for larger block counts), tensor-map cache keyed on
// (staged_k pointer, (phys_pages+1)*128 rows, device) re-encoded whenever
// the staged scratch moves, and full ABI argument validation.
//
// Error codes (nonzero -> the ctypes bridge raises RuntimeError):
//   0            success
//   1            invalid arguments (null pointer / out-of-range scalar)
//   2            cudaFuncSetAttribute (dynamic smem) failed
//   3            cuTensorMapEncodeTiled failed
//   4            cudaGetDevice failed
//   1000 + e     kernel launch failed with CUDA error e
//
// The caller (bridge) guarantees: all device pointers live until stream
// completion; the current CUDA device is the tensors' device; the launch
// is stream-ordered with the bridge's out.fill_(-inf) pre-fill; no
// host-device syncs happen here.

#include <mutex>

#include "our_kernels_decl.h"
#include "m3idx_types_m31.cuh"

namespace {

std::mutex g_prefill_mutex;

// cuTensorMapEncodeTiled cache: the descriptor must be re-encoded whenever
// the staged scratch base pointer, its row count, or the device changes
// (the scratch is reallocated per forward/epoch; a stale map silently reads
// the wrong pages).  Keyed exactly like b128host::g_map (ptr + rows), plus
// the device ordinal.
CUtensorMap g_map;
const void* g_key_ptr = nullptr;
int64_t g_key_rows = -1;
int g_key_dev = -1;

} // namespace

extern "C" int our_prefill_launch(const OurPrefillParams* p, cudaStream_t stream) {
    if (p == nullptr) return 1;
    if (p->q8 == nullptr || p->staged_k == nullptr || p->kv_indices == nullptr ||
        p->cu_seqlens == nullptr || p->seq_lens == nullptr ||
        p->prefix_lens == nullptr || p->page_offsets == nullptr || p->out == nullptr)
        return 1;
    if (p->rows <= 0 || p->rows > 16384) return 1;      // ABI: rows in (0,16384]
    if (p->segs < 1 || p->segs > 65535) return 1;       // grid.y limit
    if (p->blocks < 1) return 1;
    if (p->phys_pages < 1) return 1;                    // sentinel page == phys_pages
    if (p->kv_indices_len < 0) return 1;
    if (p->out_stride_r < p->blocks) return 1;          // ABI: stride(2)==1 implied,
    if ((int64_t)p->out_stride_h <                                    // stride(1) >= blocks,
        (int64_t)p->rows * (int64_t)p->out_stride_r) return 1;        // stride(0) >= rows*stride(1)

    const std::lock_guard<std::mutex> lock(g_prefill_mutex);

    if (!ourscore::prefill_prepare()) return 2;

    // ---- grid (q_blocks, segs, page_chunks): one CTA per (64-row tile,
    // segment, block-chunk).  rows covers every segment's q_len (each
    // segment's rows <= chunk rows); CTAs past a segment's tail exit early
    // in-kernel after reading cu_seqlens.  64 == prefill::kRowsPerCta
    // (2 M=128 tiles x 32 real rows x 4 heads).
    constexpr int kRowsPerCta = 64;
    const int q_blocks = (p->rows + kRowsPerCta - 1) / kRowsPerCta;
    // Admission bounds blocks <= 1024 (one grid.z tile).  Split defensively
    // if a future caller exceeds it so grid coverage never truncates.
    int page_chunks = 1;
    if (p->blocks > 1024)
        page_chunks = (p->blocks + 1023) / 1024;
    const int chunk_pages = (p->blocks + page_chunks - 1) / page_chunks;

    // ---- tensor map over staged_k as [(phys_pages+1)*128, 128] uint8
    // row-major (the sentinel page at index phys_pages is included in the
    // span but never dereferenced: invalid page ids never issue a box).
    // box (128 cols, 128 rows); SWIZZLE_128B matches the SW128 smem atoms;
    // single box == one 128-token page == one K stage == expect_tx 16384.
    const int64_t rows_total = ((int64_t)p->phys_pages + 1) * m3idx31::kPageSize;
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess) return 4;
    if (g_key_ptr != p->staged_k || g_key_rows != rows_total || g_key_dev != dev) {
        const cuuint64_t gdim[2] = {128, (cuuint64_t)rows_total};
        const cuuint64_t gstride[1] = {128};
        const cuuint32_t box[2] = {128, (cuuint32_t)128};
        const cuuint32_t estride[2] = {1, 1};
        CUresult r = cuTensorMapEncodeTiled(
            &g_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
            (void*)p->staged_k, gdim, gstride, box, estride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) return 3;
        g_key_ptr = p->staged_k;
        g_key_rows = rows_total;
        g_key_dev = dev;
    }

    const dim3 grid((unsigned)q_blocks, (unsigned)p->segs, (unsigned)page_chunks);
    ourscore::our_prefill_kernel<<<grid, 608, (size_t)ourscore::prefill_smem_bytes(), stream>>>(
        *p, g_map, chunk_pages, /*tma_mode=*/1);
    const cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : 1000 + (int)err;
}

// ---------------------------------------------------------------------------
// our_fused_prefill_launch -- fused score+topk entry (single-window only).
//
// Port of b128t_launch()'s kernel half (the M3.0 partials + merge-kernel
// half is DROPPED: the M3.1 grid z-splits only when blocks > 1024 and this
// entry requires blocks <= 1024, so one CTA window covers every row's whole
// block range and the finalize's direct topk_idx emit is final).  The
// tensor-map cache is shared with our_prefill_launch (same key: staged_k
// pointer, (phys_pages+1)*128 rows, device) under the same mutex.
//
// Error codes: same table as our_prefill_launch, plus: blocks > 1024 or
// topk != 16 -> 1 (caller must fall back to the two-step path).
// ---------------------------------------------------------------------------
extern "C" int our_fused_prefill_launch(const OurFusedPrefillParams* p,
                                        cudaStream_t stream) {
    if (p == nullptr) return 1;
    if (p->q8 == nullptr || p->staged_k == nullptr || p->kv_indices == nullptr ||
        p->cu_seqlens == nullptr || p->seq_lens == nullptr ||
        p->prefix_lens == nullptr || p->page_offsets == nullptr ||
        p->topk_idx == nullptr)
        return 1;
    if (p->rows <= 0 || p->rows > 16384) return 1;      // ABI: rows in (0,16384]
    if (p->segs < 1 || p->segs > 65535) return 1;       // grid.y limit
    if (p->blocks < 1) return 1;
    if (p->blocks > 1024) return 1;                     // single window: z == 1 only
    if (p->phys_pages < 1) return 1;                    // sentinel page == phys_pages
    if (p->kv_indices_len < 0) return 1;
    if (p->topk != 16) return 1;                        // register state is 16-slot
    if (p->init_blocks < 0 || p->local_blocks < 0) return 1;
    if (p->ti_stride_t != 1) return 1;                  // ABI: stride(2)==1
    if (p->ti_stride_r < p->topk) return 1;             // stride(1) >= topk,
    if ((int64_t)p->ti_stride_h <                                    // stride(0) >= rows*stride(1)
        (int64_t)p->rows * (int64_t)p->ti_stride_r) return 1;

    const std::lock_guard<std::mutex> lock(g_prefill_mutex);

    if (!ourscore::prefill_fused_prepare()) return 2;

    // ---- grid (q_blocks, segs, 1): one CTA per (64-row tile, segment);
    // z == 1 covers the whole block range (blocks <= 1024 admission).
    constexpr int kRowsPerCta = 64;
    const int q_blocks = (p->rows + kRowsPerCta - 1) / kRowsPerCta;
    const int chunk_pages = p->blocks;   // page_lo = 0, page_hi = min(blocks, tile_pages)

    // ---- tensor map over staged_k (shared cache with our_prefill_launch)
    const int64_t rows_total = ((int64_t)p->phys_pages + 1) * m3idx31::kPageSize;
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess) return 4;
    if (g_key_ptr != p->staged_k || g_key_rows != rows_total || g_key_dev != dev) {
        const cuuint64_t gdim[2] = {128, (cuuint64_t)rows_total};
        const cuuint64_t gstride[1] = {128};
        const cuuint32_t box[2] = {128, (cuuint32_t)128};
        const cuuint32_t estride[2] = {1, 1};
        CUresult r = cuTensorMapEncodeTiled(
            &g_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
            (void*)p->staged_k, gdim, gstride, box, estride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) return 3;
        g_key_ptr = p->staged_k;
        g_key_rows = rows_total;
        g_key_dev = dev;
    }

    const dim3 grid((unsigned)q_blocks, (unsigned)p->segs, 1u);
    ourscore::our_fused_prefill_kernel<<<grid, 608, (size_t)ourscore::prefill_fused_smem_bytes(), stream>>>(
        *p, g_map, chunk_pages, /*tma_mode=*/1);
    const cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : 1000 + (int)err;
}
