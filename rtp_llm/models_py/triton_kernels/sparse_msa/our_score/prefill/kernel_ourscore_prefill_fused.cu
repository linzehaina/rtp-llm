// kernel_ourscore_prefill_fused.cu -- M3.1 FUSED prefill score+topk kernel.
//
// Port of the M3.0 b128t fused-topk architecture
//   /home/liukan.lk/m3pro-128blk-20261009/src/kernel_b128t.cu
// (gate 4/4 PASS on the M3.0 testbed; see
//  /home/liukan.lk/m3pro-128blk-20261009/report/PREFILL_FUSION_RESULTS.md for
//  the architecture and the 8-bug ledger) onto the M3.1 carrier
// (kernel_ourscore_prefill.cu, GATE GREEN 16/16 bit-exact):
//
//   * The carrier's per-(row,head) block-score STORE is replaced by an
//     in-consume UNSORTED top-16 candidate state (8 half2 registers + f16
//     threshold; fire = hv > thr, insert = replace-the-min-slot via the
//     hmin2 tree -- O(1); the harmonic fire rate 24-40% makes a sorted
//     insert's serial shift unaffordable: that was the M3.0 +114% bug).
//   * Per-warp PRIVATE smem sets (16 warps x 32 lanes x 16 x (val,id)):
//     the carrier's stage-parity quarter pairing feeds each (row,head) from
//     two warps concurrently -- a shared set would race.
//   * KStages 8 -> 7 frees 16 KiB for the sets (32+32 = 64 KiB); total smem
//     ~208 KiB <= 227 KiB.  Lookahead 4 <= 6 keeps the loader pipeline.
//   * Finalize (M3.1 SIMPLIFICATION -- direct emit): selection-sort the
//     register set in registers (slot-carried ids, register bitmask; bool[]
//     arrays spill to local memory -- M3.0 bug 2), pair bar.sync (bar id
//     1..8, 64 threads), two-pointer merge of the parity pair, tie =
//     smaller block id, then write topk_idx DIRECTLY.  The M3.0 b128t
//     needed per-window GLOBAL partials + a merge kernel ONLY because its
//     grid z-splits KV across page-window CTAs (M3.0 bug 4); the M3.1
//     carrier z-splits only when blocks > 1024, and this entry REQUIRES
//     blocks <= 1024 (single window per row -- the E2E shape kv=80000 =
//     625 blocks -> z=1), so the finalize result is the FINAL selection.
//
// Selection semantics = the M3.1 prefill topk path it replaces (the
// `_topk_to_block_table_kernel` / `_topk_to_block_table_multirow_kernel`
// launched from flash_prefill_topk_to_block_tables_fp4 at its production
// parameters, MASK_INIT=False / MASK_LOCAL=False, EMIT_TOPK_IDX only):
//   - candidate domain: blocks b < vbl(r) where vbl = ceil(min(q_pos+1,
//     kv_len)/128) (kernel-2's valid_blocks = (prefix+row+128)//128 equals
//     this whenever q_pos < kv_len, the causal contract; also clamped to
//     `blocks`, the score-tensor width kernel-2 reads).
//   - value: the block score the carrier would store (RAW scale-one E4M3
//     dot max over visible tokens), with the forced overrides in kernel-2's
//     order: b < init_blocks -> 1e30 FIRST, then b >= vbl-local_blocks ->
//     1e29 (local OVERWRITES init on overlap, exactly like kernel-2's two
//     sequential tl.where's).
//   - -inf candidates (invalid page / no visible token / never staged) are
//     NOT inserted: kernel-2's -1e30 0-padding entries rank ABOVE -inf
//     candidates, so slots past the finite candidate count come out as -1
//     there, and skipping the insertion reproduces that exactly.
//   - nstore = min(topk, vbl): slots >= nstore (and slots past the
//     candidate count) are -1 padding.
//   - tie handling: kernel-2's bitonic network has an internal tie order;
//     the fused state stores values as f16 (near-ties may collapse or swap
//     order) and its merge breaks ties by smaller block id.  Selection SETS
//     are compared in the gate with an f16 near-tie tolerance.
//
// Everything else (loader warps, TMA single-box loader, mbarrier chain,
// MMA warp, TMEM double buffering, consume mapping, visibility rule) is
// byte-identical to the carrier kernel_ourscore_prefill.cu at KStages=7.

#include "our_common.cuh"
#include "our_kernels_decl.h"

#include <cuda_fp16.h>

namespace m3idx31 {
namespace prefill_fused {

using namespace cute;
using fp8e4m3 = ourscore::fp8e4m3;

constexpr int T = 128;                  // tokens per K stage == one page == one block
constexpr int KStages = 7;              // 8 -> 7: frees 16 KiB for the topk sets
constexpr int Lookahead = 4;            // K(t+L) issued at stage t (L-1 < KS)
constexpr uint32_t kTmemS0a = 0;        // tile0 S double buffer
constexpr uint32_t kTmemS0b = 128;
constexpr uint32_t kTmemS1a = 256;      // tile1 S double buffer
constexpr uint32_t kTmemS1b = 384;
constexpr uint32_t kTmemCols = 512;     // full TMEM (1 CTA/SM)
constexpr int kRowsPerCta = 64;         // 2 tiles x 32 real rows (x 4 heads)
constexpr int kTopk = 16;               // fused top-k slots (M3.1 contract)

// Fused top-k candidate sets: per consume warp, per lane, 16 (value,id) slots
// (32+32 = 64 KiB).  The carrier's stage-parity quarter pairing feeds each
// (row,head) from TWO warps (even/odd stages) concurrently -- each warp owns a
// PRIVATE set and the finalize merges the pair (shared sets would race).
struct Smem {
    alignas(1024) uint8_t q[2][128 * kIdxDim];    // 32 KB, SW128 K-major
    alignas(1024) uint8_t k[KStages][T * kIdxDim];// 112 KB, SW128 K-major
    uint64_t kfull[KStages];                      // loader -> MMA (cnt 2)
    uint64_t kempty[KStages];                     // MMA commit -> loader (cnt 1)
    cutlass::arch::ClusterTransactionBarrier tmafull[KStages];  // TMA expect_tx
    uint64_t sfull_t[2][2];                       // MMA commit -> tile's 4 warps (cnt 1)
    uint64_t sfree_t[2][2];                       // tile's warps -> MMA (cnt 4)
    int32_t qpos[kRowsPerCta];
    uint32_t tmem_base;
    alignas(128) float tkval[16][32][kTopk];
    alignas(128) int32_t tkid[16][32][kTopk];
};

} // namespace prefill_fused
} // namespace m3idx31

namespace ourscore {

// f16-domain min over the 8 half2 registers (M3.0 b128t's hmin16h2).
__device__ __forceinline__ __half fused_hmin16h2(const __half2 (&vh)[8]) {
    __half2 m01 = __hmin2(vh[0], vh[1]);
    __half2 m23 = __hmin2(vh[2], vh[3]);
    __half2 m45 = __hmin2(vh[4], vh[5]);
    __half2 m67 = __hmin2(vh[6], vh[7]);
    __half2 a = __hmin2(m01, m23);
    __half2 b = __hmin2(m45, m67);
    __half2 c = __hmin2(a, b);
    return __hmin(__low2half(c), __high2half(c));
}

__global__ void __launch_bounds__(608, 1) our_fused_prefill_kernel(
    const OurFusedPrefillParams p, const __grid_constant__ CUtensorMap tmap_kv,
    int chunk_pages, int tma_mode) {
    using namespace cute;
    using fp8e4m3 = ourscore::fp8e4m3;
    namespace pf = m3idx31::prefill_fused;
    using m3idx31::kIdxDim;
    using m3idx31::kIdxQHeads;
    using m3idx31::kPageSize;

    constexpr int T = pf::T;
    constexpr int KS = pf::KStages;
    constexpr int L = pf::Lookahead;

    extern __shared__ __align__(1024) uint8_t our_smem[];
    pf::Smem& sm = *reinterpret_cast<pf::Smem*>(our_smem);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;

    // grid (q_blocks, segs, 1): the fused entry requires blocks <= 1024, so
    // blockIdx.z == 0 and one CTA sees the WHOLE block range of its rows
    // (single window -- the direct emit in the finalize relies on this).
    const int s = (int)blockIdx.y;
    const int qb = (int)blockIdx.x;
    constexpr int kRealRows = pf::kRowsPerCta;   // 64

    const int seq_start = p.cu_seqlens[s];        // chunk-local row offset
    const int q_len = p.cu_seqlens[s + 1] - seq_start;
    if (qb * kRealRows >= q_len) return;
    const int kv_len = p.seq_lens[s];
    const int pages_b = (kv_len + kPageSize - 1) / kPageSize;
    const int pages_base = p.page_offsets[s];
    const int pages_end = p.page_offsets[s + 1];
    const int prefix = p.prefix_lens[s];

    const int row0 = qb * kRealRows;
    const int nrows = min(kRealRows, q_len - row0);

    // ---- prologue: loaders stage Q; tid 0 inits barriers ----
    if (warp < 2) {
        auto q_layout = [] {
            return tile_to_shape(UMMA::Layout_K_SW128_Atom<fp8e4m3>{},
                                 Shape<Int<128>, Int<kIdxDim>>{}, Step<_1, _2>{});
        }();
        const uint8_t* qsrc = reinterpret_cast<const uint8_t*>(p.q8)
                              + (size_t)(seq_start + row0) * (kIdxQHeads * kIdxDim);
        for (int i = tid; i < 2 * 128 * 8; i += 64) {   // 2 tiles x 128 rows x 8 chunks
            const int tile = i >> 10, r = (i >> 3) & 127, c = i & 7;
            const int real = tile * 32 + (r >> 2);      // CTA-local row 0-63
            const int src_row = (real < nrows) ? real : 0;
            const uint8_t* src = qsrc + (size_t)src_row * (kIdxQHeads * kIdxDim)
                                 + (size_t)(r & 3) * kIdxDim + (size_t)c * 16;
            uint32_t saddr = cute::cast_smem_ptr_to_uint(
                sm.q[tile] + q_layout(r, c * 16));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                         :: "r"(saddr), "l"(src));
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        // chunk-local q_pos: prefix + segment-local row offset (row0+tid).
        if (tid < kRealRows)
            sm.qpos[tid] = (tid < nrows) ? (prefix + row0 + tid) : -1;
    }
    if (tid == 0) {
        for (int i = 0; i < KS; ++i) {
            mbar_init(&sm.kfull[i], 2);
            mbar_init(&sm.kempty[i], 1);
            sm.tmafull[i].init(1);
        }
        for (int i = 0; i < 2; ++i) {
            mbar_init(&sm.sfull_t[0][i], 1);
            mbar_init(&sm.sfull_t[1][i], 1);
            mbar_init(&sm.sfree_t[0][i], 4);
            mbar_init(&sm.sfree_t[1][i], 4);
        }
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    __syncthreads();

    // page window (CTA-uniform; empty-window CTAs exit without leaking a
    // TMEM allocation).  z == 1 always here, so page_lo = 0.
    int p_last = -1;
    for (int r = 0; r < kRealRows; ++r) p_last = max(p_last, sm.qpos[r]);
    int tile_pages = min(pages_b, (p_last + kPageSize) / kPageSize);
    if (pages_end > pages_base)
        tile_pages = min(tile_pages, pages_end - pages_base);
    const int page_lo = (int)blockIdx.z * chunk_pages;
    const int page_hi = min(page_lo + chunk_pages, tile_pages);
    if (page_lo >= page_hi) return;

    if (warp == 18) {
        cute::TMEM::Allocator1Sm alloc;
        alloc.allocate(pf::kTmemCols, &sm.tmem_base);
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    __syncthreads();
    const uint32_t tmem = sm.tmem_base;

    const int nstages = page_hi - page_lo;   // one stage == one page == one block

    if (warp < 2) {
        // ================= loader warps (0-1, 64 threads) =================
        auto k_layout = [] {
            return tile_to_shape(UMMA::Layout_K_SW128_Atom<fp8e4m3>{},
                                 Shape<Int<T>, Int<kIdxDim>>{}, Step<_1, _2>{});
        }();
        auto page_id = [&](int pg0) -> int32_t {
            const int64_t lp = (int64_t)pages_base + pg0;
            if (lp < 0 || lp >= pages_end || lp >= p.kv_indices_len) return (int32_t)-1;
            return __ldg(p.kv_indices + lp);
        };
        auto issue_load_k = [&](int stg) {
            const int pg0 = page_lo + stg;
            const int32_t gd = page_id(pg0);
            const bool ok = (gd >= 0 && gd < p.phys_pages);
            if (tma_mode) {
                if (tid == 0) {
                    if (ok) {
                        sm.tmafull[stg % KS].arrive_and_expect_tx(
                            (uint32_t)(T * kIdxDim));
                        cute::SM90_TMA_LOAD_2D::copy(
                            &tmap_kv,
                            reinterpret_cast<uint64_t*>(&sm.tmafull[stg % KS]),
                            (uint64_t)cute::TMA::CacheHintSm100::EVICT_NORMAL,
                            sm.k[stg % KS], 0, (int32_t)((int64_t)gd * kPageSize));
                    } else {
                        sm.tmafull[stg % KS].arrive_and_expect_tx(0);
                    }
                }
                return;
            }
            // cp.async fallback: whole 128-token page from one physical page.
            if (ok) {
                const int c = tid & 7;                       // 8 x 16B chunks = 128B
                const int r0 = tid >> 3;                     // [0,8)
                #pragma unroll
                for (int kk = 0; kk < T / 8; ++kk) {
                    const int r = r0 + kk * 8;               // token row in stage
                    const uint8_t* src = reinterpret_cast<const uint8_t*>(p.staged_k)
                                         + ((size_t)gd * kPageSize + (size_t)r) * kIdxDim
                                         + (size_t)c * 16;
                    uint32_t saddr = cute::cast_smem_ptr_to_uint(
                        sm.k[stg % KS] + k_layout(r, c * 16));
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                                 :: "r"(saddr), "l"(src));
                }
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        };

        int issued = -1;
        for (int stg = 0; stg < nstages; ++stg) {
            const int want = min(stg + L, nstages) - 1;
            while (issued < want) {
                ++issued;
                if (issued >= KS)
                    mbar_wait(&sm.kempty[issued % KS],
                              (uint32_t)((issued / KS) - 1) & 1);
                issue_load_k(issued);
            }
            if (tma_mode) {
                if (stg == 0)
                    cpasync_wait_group(0);
                if (tid == 0)
                    sm.tmafull[stg % KS].wait((uint32_t)((stg / KS) & 1));
                __syncwarp();
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if ((tid & 31) == 0)
                    mbar_arrive(&sm.kfull[stg % KS]);
            } else {
                int allow = min(stg + L, nstages) - 1 - stg;
                if (allow > L - 1) allow = L - 1;
                if (allow < 0) allow = 0;
                cpasync_wait_group(allow);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if ((tid & 31) == 0)
                    mbar_arrive(&sm.kfull[stg % KS]);
            }
        }
    } else if (warp == 18) {
        // ================= MMA warp =================
        auto q_layout = [] {
            return tile_to_shape(UMMA::Layout_K_SW128_Atom<fp8e4m3>{},
                                 Shape<Int<128>, Int<kIdxDim>>{}, Step<_1, _2>{});
        }();
        auto k_layout = [] {
            return tile_to_shape(UMMA::Layout_K_SW128_Atom<fp8e4m3>{},
                                 Shape<Int<T>, Int<kIdxDim>>{}, Step<_1, _2>{});
        }();
        auto mma_s = make_tiled_mma(MMA_Atom<MMA_Traits<
            SM100_MMA_F8F6F4_SS, fp8e4m3, fp8e4m3, float,
            C<128>, C<T>,
            integral_constant<UMMA::Major, UMMA::Major::K>,
            integral_constant<UMMA::Major, UMMA::Major::K>,
            integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
            integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>>{});
        Tensor sQ0 = make_tensor(make_smem_ptr(reinterpret_cast<fp8e4m3*>(sm.q[0])), q_layout);
        Tensor sQ1 = make_tensor(make_smem_ptr(reinterpret_cast<fp8e4m3*>(sm.q[1])), q_layout);
        Tensor tAq0 = mma_s.get_slice(0).partition_fragment_A(sQ0);
        Tensor tAq1 = mma_s.get_slice(0).partition_fragment_A(sQ1);
        Tensor tS0 = partition_fragment_C(mma_s, make_shape(Int<128>{}, Int<T>{}));
        Tensor tS1 = tS0;

        for (int t = 0; t < nstages; ++t) {
            if (t >= 2) {
                mbar_wait(&sm.sfree_t[0][t & 1], (uint32_t)((t - 2) >> 1) & 1);
                mbar_wait(&sm.sfree_t[1][t & 1], (uint32_t)((t - 2) >> 1) & 1);
            }
            mbar_wait(&sm.kfull[t % KS], (uint32_t)(t / KS) & 1);
            Tensor sK = make_tensor(make_smem_ptr(
                reinterpret_cast<fp8e4m3*>(sm.k[t % KS])), k_layout);
            Tensor tBk = mma_s.get_slice(0).partition_fragment_B(sK);
            tS0.data() = make_tmem_ptr<float>(tmem + (t & 1 ? pf::kTmemS0b : pf::kTmemS0a));
            tS1.data() = make_tmem_ptr<float>(tmem + (t & 1 ? pf::kTmemS1b : pf::kTmemS1a));
            mma_s.accumulate_ = decltype(mma_s.accumulate_)::Zero;
            CUTE_UNROLL
            for (int uu = 0; uu < size<2>(tAq0); ++uu) {
                cute::gemm(mma_s, tAq0(_,_,uu), tBk(_,_,uu), tS0);
                mma_s.accumulate_ = decltype(mma_s.accumulate_)::One;
            }
            if ((tid & 31) == 0)
                mma_commit(&sm.sfull_t[0][t & 1]);   // tile0's group starts
            mma_s.accumulate_ = decltype(mma_s.accumulate_)::Zero;
            CUTE_UNROLL
            for (int uu = 0; uu < size<2>(tAq1); ++uu) {
                cute::gemm(mma_s, tAq1(_,_,uu), tBk(_,_,uu), tS1);
                mma_s.accumulate_ = decltype(mma_s.accumulate_)::One;
            }
            if ((tid & 31) == 0) {
                mma_commit(&sm.sfull_t[1][t & 1]);   // tile1's group starts
                mma_commit(&sm.kempty[t % KS]);
            }
        }
    } else {
        // ======== consume warps (2-17; 2-9 tile0, 10-17 tile1) ========
        // Carrier's stage-alternating quarter consume with the score store
        // REPLACED by the fused top-16 insertion, then the finalize merges
        // the parity pair and emits topk_idx directly (single window).
        const int cw = warp - 2;                    // consume warp index 0-15
        const int tile = cw >> 3;
        const int sparity = (cw >> 2) & 1;          // stage parity this warp owns
        const int lane = tid & 31;
        // TMEM quarter rule: the warp's collective ld accesses rows
        // 32*(GLOBAL warp & 3) -- with consume warps at global 2-17, groups
        // of 4 consecutive warps cover quarters {2,3,0,1}; mrow must follow
        // the hardware mapping, not the consume-warp index.
        const int mrow = 32 * (warp & 3) + lane;    // packed row in tile
        const int r_real = mrow >> 2;               // tile-local real row 0-31
        const int h = mrow & 3;                     // index head 0-3
        const int gr = tile * 32 + r_real;          // CTA-local row 0-63
        const int prow = sm.qpos[gr];               // causal bound of this row
        // warp-uniform visibility bound: the warp's 32 lanes cover real rows
        // 8*(warp&3) .. 8*(warp&3)+7 of this tile (4 heads each).
        int prow_hi_w = -1;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            prow_hi_w = max(prow_hi_w, sm.qpos[tile * 32 + 8 * (warp & 3) + j]);
        const uint32_t trow = (uint32_t)(mrow & ~31) << 16;   // TMEM quarter base
        const int nrows_tile = min(32, q_len - row0 - tile * 32);
        const bool tile_alive = nrows_tile > 0;
        const bool row_live = (r_real < nrows_tile) && (prow >= 0);

        // per-lane fused top-k state: UNSORTED 16-slot set in 8 half2
        // registers + f16 threshold (replace-the-min on fire).  Ids live in
        // the warp's private smem set and are touched once per fire.
        const int vbl = row_live
            ? min((min(prow + 1, kv_len) + 127) / 128, p.blocks)
            : 0;
        int32_t* const si = sm.tkid[cw][lane];
        __half2 vh[8];
        #pragma unroll
        for (int s = 0; s < 8; ++s) vh[s] = __half2half2(__float2half(-INFINITY));
        float thr = -INFINITY;
        int cnt = 0;

        for (int t = sparity; t < nstages; t += 2) {
            const int pg0 = page_lo + t;
            const int blk = pg0;                   // page index IS block index

            // (1) wait S(t) -- per-tile barrier: this stage's 4 owner warps
            //     wake as soon as their own UMMA commits.
            mbar_wait(&sm.sfull_t[tile][t & 1], (uint32_t)(t >> 1) & 1);
            asm volatile("tcgen05.fence::after_thread_sync;");

            // page validity for this block (CTA-uniform).
            const int64_t lp = (int64_t)pages_base + pg0;
            const bool lp_ok = (lp >= 0 && lp < pages_end && lp < p.kv_indices_len);
            const int32_t gd = lp_ok ? __ldg(p.kv_indices + lp) : (int32_t)-1;
            const bool page_ok = (lp_ok && gd >= 0 && gd < p.phys_pages);

            // (2) TMEM read, warp-local over the full block: half 0 first,
            //     reduce it, then half 1 into the same registers.
            const bool dead_block = pg0 * kPageSize > prow_hi_w;
            const bool dead_h1 = (pg0 * kPageSize + 64) > prow_hi_w;
            uint32_t rv[64];
            float m0 = -INFINITY, m1 = -INFINITY;
            bool need_ld1 = false;
            if (tile_alive && !dead_block && page_ok) {
                const uint32_t base = tmem
                    + (tile ? (t & 1 ? pf::kTmemS1b : pf::kTmemS1a)
                            : (t & 1 ? pf::kTmemS0b : pf::kTmemS0a))
                    + trow;
                tmem_ld64(base, *reinterpret_cast<uint32_t (*)[64]>(&rv[0]));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                __syncwarp();
                m0 = prefix_max64(&rv[0], vis_count(pg0 * 128, prow, kv_len));
                need_ld1 = !dead_h1;
                if (need_ld1) {
                    tmem_ld64(base + 64, *reinterpret_cast<uint32_t (*)[64]>(&rv[0]));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    __syncwarp();
                }
            }
            // (3) the quarter is fully read: release S(t) for UMMA(t+2).
            if (lane == 0)
                mbar_arrive(&sm.sfree_t[tile][t & 1]);
            if (need_ld1)
                m1 = prefix_max64(&rv[0], vis_count(pg0 * 128 + 64, prow, kv_len));

            // (4) FUSED INSERTION replaces the score store.  The value is
            //     what the carrier would store (RAW E4M3 dot max, xsc =
            //     1.0f); forced blocks take kernel-2's MASK_INIT=False /
            //     MASK_LOCAL=False constants, local AFTER init (overlap ->
            //     1e29, kernel-2's sequential tl.where order).  -inf
            //     candidates (invalid page) are NOT inserted: kernel-2's
            //     -1e30 0-padding outranks them, so both paths emit -1 for
            //     slots past the finite candidate count.
            if (row_live && blk < vbl) {
                float v = fmaxf(m0, m1);
                if (blk < p.init_blocks)
                    v = 1e30f;
                if (blk >= vbl - p.local_blocks)
                    v = 1e29f;
                if (v > -INFINITY) {
                    const __half hv = __float2half(v);
                    if (cnt < pf::kTopk) {
                        // fill phase: slots 0..15 in arrival order; once
                        // full, compute the f16-domain min (hmin2 tree).
                        if (cnt & 1)
                            vh[cnt >> 1] = __halves2half2(__low2half(vh[cnt >> 1]), hv);
                        else
                            vh[cnt >> 1] = __halves2half2(hv, __high2half(vh[cnt >> 1]));
                        si[cnt] = blk;
                        if (++cnt == pf::kTopk) thr = __half2float(fused_hmin16h2(vh));
                    } else if (__hgt(hv, __float2half(thr))) {
                        // fire: locate the min slot (value via hmin2 tree,
                        // then a slot match), replace it, recompute the min.
                        const __half mn = fused_hmin16h2(vh);
                        int m = 0;
                        #pragma unroll
                        for (int s = 0; s < 8; ++s) {
                            if (__low2half(vh[s]) == mn) { m = 2 * s; break; }
                            if (__high2half(vh[s]) == mn) { m = 2 * s + 1; break; }
                        }
                        if (m & 1)
                            vh[m >> 1] = __halves2half2(__low2half(vh[m >> 1]), hv);
                        else
                            vh[m >> 1] = __halves2half2(hv, __high2half(vh[m >> 1]));
                        si[m] = blk;
                        thr = __half2float(fused_hmin16h2(vh));
                    }
                }
            }
        }

        // (5) finalize: spill the register set to the private smem set
        //     (selection-sorted descending; rv's registers are dead
        //     post-loop), rendezvous with the parity partner, then the
        //     even-stage owner merges the two sorted 16-lists and emits
        //     topk_idx DIRECTLY (single window per row: z == 1).
        {
            float fv[pf::kTopk];
            #pragma unroll
            for (int s = 0; s < pf::kTopk; ++s)
                fv[s] = (s & 1) ? __half2float(__high2half(vh[s >> 1]))
                                : __half2float(__low2half(vh[s >> 1]));
            float* const svs = sm.tkval[cw][lane];
            int32_t* const ssi = sm.tkid[cw][lane];
            // snapshot the ids into registers FIRST: the sorted output
            // writes back into the same smem array the ids are read from
            // (in-place read-during-write would clobber unsorted slots --
            // M3.0 bug 3: duplicate/garbage ids).
            int32_t fid[pf::kTopk];
            #pragma unroll
            for (int s = 0; s < pf::kTopk; ++s) fid[s] = si[s];
            // register bitmask of taken slots (a bool[] array spills to
            // local memory -- M3.0 bug 2).  Unfilled slots hold value -inf
            // and are never selected (strictly-greater compare), so their
            // garbage ids cannot leak; the exhausted iterations emit the
            // (-inf, -1) pair.
            unsigned takenMask = 0;
            #pragma unroll
            for (int out = 0; out < pf::kTopk; ++out) {
                float best = -INFINITY; int bs = -1;
                #pragma unroll
                for (int s = 0; s < pf::kTopk; ++s) {
                    if (takenMask & (1u << s)) continue;
                    if (fv[s] > best) { best = fv[s]; bs = s; }
                }
                if (bs >= 0) {
                    takenMask |= 1u << bs;
                    svs[out] = best;
                    ssi[out] = fid[bs];
                } else {
                    svs[out] = -INFINITY;
                    ssi[out] = -1;
                }
            }
            // pair rendezvous: the quarter's two parity warps (64 threads)
            // sync on a named barrier so the partner's sorted set is
            // visible in smem before the merge reads it.
            const int pair_id = (cw >> 3) * 4 + (cw & 3);       // 0..7
            asm volatile("bar.sync %0, 64;" :: "r"(1 + pair_id));
            if (sparity == 0 && row_live) {
                const float* const pv = sm.tkval[cw ^ 4][lane];
                const int32_t* const pi = sm.tkid[cw ^ 4][lane];
                const int nstore = min(pf::kTopk, vbl);
                const int row_out = seq_start + row0 + gr;   // chunk-local row
                int32_t* const dst = p.topk_idx
                    + (size_t)h * (size_t)p.ti_stride_h
                    + (size_t)row_out * (size_t)p.ti_stride_r;
                int a = 0, b = 0;
                #pragma unroll
                for (int out = 0; out < pf::kTopk; ++out) {
                    bool take_a;
                    if (a >= pf::kTopk) take_a = false;
                    else if (b >= pf::kTopk) take_a = true;
                    else if (svs[a] > pv[b]) take_a = true;
                    else if (svs[a] < pv[b]) take_a = false;
                    else take_a = (ssi[a] <= pi[b]);   // tie: smaller block id
                    int32_t id;
                    if (take_a) {
                        id = ssi[a];
                        ++a;
                    } else {
                        id = pi[b];
                        ++b;
                    }
                    dst[(size_t)out * (size_t)p.ti_stride_t] =
                        (out < nstore && id >= 0) ? id : (int32_t)-1;
                }
            }
        }
    }

    __syncthreads();
    if (warp == 18) {
        cute::TMEM::Allocator1Sm alloc;
        alloc.free(tmem, pf::kTmemCols);
    }
}

// Host-side one-time setup, defined in the kernel's own TU so the
// cudaFuncSetAttribute symbol is the defining one.
bool prefill_fused_prepare() {
    static const bool ok = []() {
        return cudaFuncSetAttribute(
                   our_fused_prefill_kernel,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   prefill_fused_smem_bytes()) == cudaSuccess;
    }();
    return ok;
}

int prefill_fused_smem_bytes() {
    return (int)sizeof(m3idx31::prefill_fused::Smem);
}

} // namespace ourscore
