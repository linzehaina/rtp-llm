// kernel_ourscore_prefill.cu -- M3.1 prefill block-score kernel.
//
// Port of the M3.0 b128-SW128 champion
//   /home/liukan.lk/m3pro-128blk-20261009/src/kernel_b128.cu
// (warp-specialized tcgen05 F8F6F4 pipeline, SW128 TMA single-box loader,
// 608 threads / 19 warps, KStages=8, 1 CTA/SM, TMEM 512 cols) to the M3.1
// geometry frozen in our_score_abi.h:
//
//   * H=4 index heads (was 16).  Head packing into the M=128 tile:
//     packed row p = r_real*4 + h.  Per 32-lane TMEM quarter: 8 real rows
//     x 4 heads (was 2 x 16).  Consume mapping: mrow = 32*(warp&3)+lane,
//     r_real = mrow>>2, h = mrow&3, kRowsPerCta 16 -> 64 (2 tiles x 32
//     real rows), q_blocks = ceil(rows_c/64), qpos[64],
//     nrows_tile = min(32, ...).  Q staging: real = tile*32+(r>>2),
//     head = r&3, byte offset (r&3)*128.
//   * Pages: 128 contiguous tokens; one page == one scored block == one
//     T=128 K stage.  The b128 gp[8] 8-gather array collapses to ONE gather
//     gd = kv_indices[pages_base + pg0]; pg0 is the block index (blk = pg0).
//     chunk_pages needs no 8-alignment (stage boundaries are page
//     boundaries by construction).
//   * Per-block max: b128's per-16-token-page b128_half_max decomposition
//     is replaced by b128d's contiguous-prefix rule over the two 64-token
//     halves: n = clamp(min(prow+1, kv_len) - (pg0*128 + half*64), 0, 64)
//     (ourscore::vis_count / prefix_max64).
//   * Metadata is int32 (cu_seqlens/seq_lens/prefix_lens/page_offsets/
//     kv_indices); q_pos is derived in-kernel as prefix_lens[s] +
//     (row - cu_seqlens[s]); kv_len = seq_lens[s].
//   * Stores: out[h*out_stride_h + (seq_start+row0+gr)*out_stride_r + blk]
//     = fmaxf(m0, m1) with xsc = 1.0f literal (RAW E4M3 dots; the fmha
//     path uses sm_scale=q_scale=k_scale=1.0).  Rows are CHUNK-LOCAL.
//     The kernel writes ONLY finite visible scores; the bridge pre-fills
//     the whole out view with -inf on-stream, so every -inf case in the
//     ABI semantics is "skip the store".
//   * Invalid page ids (kv_indices entry <0 or >= phys_pages, or logical
//     page outside the segment's [page_offsets[s], page_offsets[s+1])
//     range / kv_indices_len): the whole block is left at -inf and the
//     page is NEVER dereferenced -- in TMA mode no box is issued at all
//     (an out-of-range coordinate would be an OOB TMA read); instead the
//     stage's transaction barrier is completed with expect_tx(0).
//   * TMA mode ALWAYS ON in production (host passes tma_mode=1): single
//     SM90_TMA_LOAD_2D box {128 cols, 128 rows} over staged_k viewed as
//     [(phys_pages+1)*128, 128] uint8, SWIZZLE_128B, L2_256B, expect_tx
//     16384, coordinate row gd*128.  The cp.async fallback is kept
//     (tma_mode=0) exactly like b128.
//
// Unchanged from b128: 608 threads / 19 warps (2 loader + 16 consume + 1
// MMA), KStages=8, Lookahead=4, smem ~161 KiB, 1 CTA/SM, TMEM 512 cols,
// SM100_MMA_F8F6F4_SS M=N=K=128, mbarrier chain (kfull/kempty/tmafull/
// sfull_t/sfree_t with identical counts and phase formulas), stage-
// alternating warp-local quarter consume.
//
// One deliberate deviation (safety fix, zero cost): in TMA mode the b128
// retire path never drained the Q cp.async group before the first kfull
// publish (empirically never lost -- the K(0) TMA DRAM round trip dwarfs
// the 32 KiB Q staging -- but not guaranteed).  This port retires it with
// cp.async.wait_group 0 at stage 0 (in TMA mode the Q group is the only
// cp.async group in flight).

#include "our_common.cuh"
#include "our_kernels_decl.h"

namespace m3idx31 {
namespace prefill {

using namespace cute;
using fp8e4m3 = ourscore::fp8e4m3;

constexpr int T = 128;                  // tokens per K stage == one page == one block
constexpr int KStages = 8;              // K smem buffers (1 CTA/SM, TMEM-forced)
constexpr int Lookahead = 4;            // K(t+L) issued at stage t
constexpr uint32_t kTmemS0a = 0;        // tile0 S double buffer
constexpr uint32_t kTmemS0b = 128;
constexpr uint32_t kTmemS1a = 256;      // tile1 S double buffer
constexpr uint32_t kTmemS1b = 384;
constexpr uint32_t kTmemCols = 512;     // full TMEM (1 CTA/SM)
constexpr int kRowsPerCta = 64;         // 2 tiles x 32 real rows (x 4 heads)

struct Smem {
    alignas(1024) uint8_t q[2][128 * kIdxDim];    // 32 KB, SW128 K-major
    alignas(1024) uint8_t k[KStages][T * kIdxDim];// 128 KB, SW128 K-major
    uint64_t kfull[KStages];                      // loader -> MMA (cnt 2)
    uint64_t kempty[KStages];                     // MMA commit -> loader (cnt 1)
    cutlass::arch::ClusterTransactionBarrier tmafull[KStages];  // TMA expect_tx
    uint64_t sfull_t[2][2];                       // MMA commit -> tile's 4 warps (cnt 1)
    uint64_t sfree_t[2][2];                       // tile's warps -> MMA (cnt 4)
    int32_t qpos[kRowsPerCta];
    uint32_t tmem_base;
};

} // namespace prefill
} // namespace m3idx31

namespace ourscore {

__global__ void __launch_bounds__(608, 1) our_prefill_kernel(
    const OurPrefillParams p, const __grid_constant__ CUtensorMap tmap_kv,
    int chunk_pages, int tma_mode) {
    using namespace cute;
    using fp8e4m3 = ourscore::fp8e4m3;
    namespace pf = m3idx31::prefill;
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

    // grid (q_blocks, segs, page_chunks): blockIdx.y IS the chunk-local
    // segment -- no y_order/LPT remap (M3.1 admission bounds the segment
    // count; scheduling is left to the default rasterizer).
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

    // page window (computed BEFORE any TMEM allocation: empty-window CTAs
    // exit here without leaking a TMEM allocation).  tile_pages bounds the
    // causally visible pages AND the segment's logical page range AND
    // kv_len pages (pages_b); kv_indices reads are additionally guarded by
    // lp_ok in loader and consume so no out-of-range entry is ever used.
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

    // RAW scale-one E4M3 dots (xsc folded to the literal 1.0f per ABI).
    // M3.0's q_scale/k_scale/sm_scale product is gone.
    const int nstages = page_hi - page_lo;   // one stage == one page == one block

    if (warp < 2) {
        // ================= loader warps (0-1, 64 threads) =================
        auto k_layout = [] {
            return tile_to_shape(UMMA::Layout_K_SW128_Atom<fp8e4m3>{},
                                 Shape<Int<T>, Int<kIdxDim>>{}, Step<_1, _2>{});
        }();
        // stage stg -> page pg0 = page_lo + stg (== block pg0 exactly: the
        // stage grid IS the page grid for 128-token pages).  M3.1: ONE
        // gather gd = kv_indices[pages_base + pg0]; validity-checked, never
        // dereferenced when invalid (no TMA box, no cp.async).
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
                        // box (128 cols, 128 rows) at global row gd*128
                        cute::SM90_TMA_LOAD_2D::copy(
                            &tmap_kv,
                            reinterpret_cast<uint64_t*>(&sm.tmafull[stg % KS]),
                            (uint64_t)cute::TMA::CacheHintSm100::EVICT_NORMAL,
                            sm.k[stg % KS], 0, (int32_t)((int64_t)gd * kPageSize));
                    } else {
                        // invalid page: complete the transaction barrier with
                        // zero expected bytes; no box is ever issued for an
                        // out-of-range coordinate (an OOB TMA read would
                        // fault / read the zero sentinel, both wrong here).
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
            // (1) keep the L-stage window full; gate each issue on the
            //     buffer's kempty (MMA commit of stage stg+L-1-KS, which is
            //     behind us since L-1 < KS)
            const int want = min(stg + L, nstages) - 1;
            while (issued < want) {
                ++issued;
                if (issued >= KS)
                    mbar_wait(&sm.kempty[issued % KS],
                              (uint32_t)((issued / KS) - 1) & 1);
                issue_load_k(issued);
            }
            // (2) retire K(stg) -- and Q, which rides the group order ahead
            //     of K(0) -- then publish kfull[stg].  TMA mode: the retire
            //     is the transaction barrier's phase flip (the copy engine's
            //     tx arrival completes it); cp.async mode: wait_group as
            //     before.  Safety fix vs b128: at stage 0 in TMA mode the Q
            //     cp.async group (the only one in flight) is drained so the
            //     first kfull publish guarantees Q has landed.
            if (tma_mode) {
                if (stg == 0)
                    cpasync_wait_group(0);
                // slot stg%KS completed once for this stage; phase = number of
                // prior completions of this slot mod 2 = (stg/KS)&1
                if (tid == 0)
                    sm.tmafull[stg % KS].wait((uint32_t)((stg / KS) & 1));
                __syncwarp();
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if ((tid & 31) == 0)          // converged: lane pred, no ELECT/VOTEU
                    mbar_arrive(&sm.kfull[stg % KS]);
            } else {
                int allow = min(stg + L, nstages) - 1 - stg;
                if (allow > L - 1) allow = L - 1;
                if (allow < 0) allow = 0;
                cpasync_wait_group(allow);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if ((tid & 31) == 0)          // converged: lane pred, no ELECT/VOTEU
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
            if ((tid & 31) == 0)              // converged: lane pred, no ELECT/VOTEU
                mma_commit(&sm.sfull_t[0][t & 1]);   // tile0's group starts
            mma_s.accumulate_ = decltype(mma_s.accumulate_)::Zero;
            CUTE_UNROLL
            for (int uu = 0; uu < size<2>(tAq1); ++uu) {
                cute::gemm(mma_s, tAq1(_,_,uu), tBk(_,_,uu), tS1);
                mma_s.accumulate_ = decltype(mma_s.accumulate_)::One;
            }
            if ((tid & 31) == 0) {            // converged: lane pred, no ELECT/VOTEU
                mma_commit(&sm.sfull_t[1][t & 1]);   // tile1's group starts
                // both UMMAs of stage t have read the K buffer: free it for
                // the loader's stage t+KS (async completion semantics -- a
                // plain mbar_arrive here would fire too early)
                mma_commit(&sm.kempty[t % KS]);
            }
        }
    } else {
        // ======== consume warps (2-17; 2-9 tile0, 10-17 tile1) ========
        // b128-2 stage-alternating warp-local quarter consume, re-mapped to
        // the H=4 packing: the two warps of a quarter pair {cw k, cw k+4}
        // (same TMEM quarter, same 32 packed rows = 8 real rows x 4 heads)
        // split the stage sequence by parity; each owner warp reads BOTH
        // 64-token halves of its stage's page (two x64 TMEM loads into the
        // same rv[64] registers) and produces the block max in registers.
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

        for (int t = sparity; t < nstages; t += 2) {
            const int pg0 = page_lo + t;
            const int blk = pg0;                   // page index IS block index

            // (1) wait S(t) -- per-tile barrier: this stage's 4 owner warps
            //     wake as soon as their own UMMA commits (the other parity's
            //     warps never wait on this slot)
            mbar_wait(&sm.sfull_t[tile][t & 1], (uint32_t)(t >> 1) & 1);
            asm volatile("tcgen05.fence::after_thread_sync;");

            // page validity for this block (CTA-uniform; invalid blocks are
            // left at the bridge's -inf pre-fill and their K is never read)
            const int64_t lp = (int64_t)pages_base + pg0;
            const bool lp_ok = (lp >= 0 && lp < pages_end && lp < p.kv_indices_len);
            const int32_t gd = lp_ok ? __ldg(p.kv_indices + lp) : (int32_t)-1;
            const bool page_ok = (lp_ok && gd >= 0 && gd < p.phys_pages);

            // (2) TMEM read, warp-local over the full block: half 0 first,
            //     reduce it, then half 1 into the same registers.  Warp-
            //     uniform dead-block skip drops both loads; a dead half 1
            //     (causal boundary) drops the second round trip; an invalid
            //     page drops both (uniform predicate -- the collective
            //     tcgen05.ld stays warp-converged).
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
            //     Dead/invalid blocks arrive immediately (their S is never
            //     read); live blocks arrive right after the second wait::ld,
            //     before half 1's reduce -- the MMA can start while it runs.
            if (lane == 0)
                mbar_arrive(&sm.sfree_t[tile][t & 1]);
            if (need_ld1)
                m1 = prefix_max64(&rv[0], vis_count(pg0 * 128 + 64, prow, kv_len));

            // (4) combine the halves and store the single 4B block score
            //     for this (row, head).  Visibility at block granularity:
            //     C128 = min(prow+1, kv_len) - blk*128 > 0 iff the block has
            //     at least one visible token; dead rows / blocks / invalid
            //     pages skip the store (out keeps the bridge's -inf).
            //     Rows are CHUNK-LOCAL: seq_start is cu_seqlens[s].
            if (r_real < nrows_tile && page_ok) {
                const int base128 = blk * 128;
                const int C128 = min(prow + 1, kv_len) - base128;
                if (C128 > 0) {
                    const int row_out = seq_start + row0 + gr;   // chunk-local
                    p.out[(size_t)h * (size_t)p.out_stride_h
                          + (size_t)row_out * (size_t)p.out_stride_r
                          + (size_t)blk] = fmaxf(m0, m1);        // xsc = 1.0f
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
bool prefill_prepare() {
    static const bool ok = []() {
        return cudaFuncSetAttribute(
                   our_prefill_kernel,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   prefill_smem_bytes()) == cudaSuccess;
    }();
    return ok;
}

int prefill_smem_bytes() {
    return (int)sizeof(m3idx31::prefill::Smem);
}

} // namespace ourscore
