// kernel_ourscore_decode.cu -- M3.1 decode block-score kernel.
//
// Port of the M3.0 b128d persistent stream-KV decode kernel
//   /home/liukan.lk/m3pro-128blk-20261009/src/kernel_b128d.cu
// to M3.1 geometry as a SIMPLER NON-PERSISTENT first cut (frozen design
// decision: correctness first; the persistent Unit/LPT/plan/fingerprint
// machinery is dropped, the validated b128d thread geometry and half-page
// stage pipeline are kept; a later persistent stream-KV variant can land
// behind the same ABI).
//
// M3.1 / M3.0 deltas:
//   * kRealRows 8 -> 32: thread = packed m-row, r_real = tid>>2, h = tid&3
//     (plain decode: ONE query token per request, so packed rows 0-3 are
//     the real (token 0 x 4 heads) rows; rows 4-127 are padding whose
//     stores are gated off).
//   * THE critical non-mechanical change (kPageSize 16 -> 128 makes
//     kPagesPerStage = T/kPageSize = 0 by integer division, which would
//     blow up nstages and mis-read the page table): T=64 stages are
//     re-expressed as HALF-PAGE stages -- stage s covers page
//     pg = page_lo + (s>>1) at token offset (s&1)*64; nstages = 2*n_pages;
//     the existing stage-PAIR m_blk accumulation now pairs exactly the two
//     halves of ONE 128-token page = one output block; blk = pg.
//   * Loader: ONE gather g = block_table[b*bt_stride + pg] (validity
//     checked: g < 0 or g >= phys_pages -> no load, block score -inf,
//     NEVER dereferenced); src = staged_k + (g*128 + (s&1)*64 + row)*128.
//   * Visibility: n = clamp(min(prow+1, seq_len) - (pg*128 + (s&1)*64),
//     0, 64) with decode q_pos = seq_len-1 (self-inclusive; plain decode
//     has no invisible tokens below seq_len).  seq_len is clamped to
//     max_blocks*128 exactly like the production Triton reader.
//   * Store: xsc = sm_scale only (plain fp32 multiplier AFTER the
//     per-block max -- never folded into the dot), plus the ABI override
//     order: (1) blk < init_blocks && blk < local_start -> literal 1e30f;
//     (2) then blk >= local_start (= max(0, num_blocks - local_blocks))
//     -> literal 1e29f (local overwrites init on overlap); -inf for
//     invalid pages and for the [num_blocks, max_blocks) tail -- plain
//     per-thread stores, no PAD_TILE clear kernel: the non-persistent
//     launch covers ALL (b, blk) pairs of [0, max_blocks) directly, so
//     the tail is rewritten to -inf on EVERY launch (CUDA-graph replay
//     safe).
//   * out index = (h*batch + b)*max_blocks + blk (contiguous [4,batch,
//     max_blocks], the ABI layout).
//
// Unchanged from b128d: 128 threads (4 CTA/SM, ~50 KiB smem), SW64 smem
// atoms, SM100_MMA_F8F6F4_SS M=128 x N=64 x K=128, 4 rotating K smem
// buffers with a 2-stage cp.async lookahead, single bar_s mbarrier fed by
// tcgen05.commit, TMEM 128 cols (S double buffer), the prologue/loop
// synchronization order (Q wait -> K(0) wait -> fence -> UMMA(0) ->
// K(1) issue; per-stage wait_group 1/0 + tcgen05 fences + __syncthreads
// before the next UMMA).

#include "our_common.cuh"
#include "our_kernels_decl.h"

namespace m3idx31 {
namespace decode {

using namespace cute;
using fp8e4m3 = ourscore::fp8e4m3;

constexpr int T = 64;                    // tokens per K stage (HALF of a 128-token page)
constexpr uint32_t kTmemS0 = 0;          // S double buffer (64 cols each)
constexpr uint32_t kTmemS1 = 64;
constexpr uint32_t kTmemCols = 128;
constexpr int kRealRows = 32;            // 32 real rows x 4 heads = 128 packed rows
static_assert(kRealRows * kIdxQHeads == 128,
              "the M=128 tile must be exactly packed by 32 real rows x 4 heads");
constexpr int kBlockChunk = 32;          // blocks (pages) per CTA along grid.z

struct Smem {
    alignas(1024) uint8_t q[128 * kIdxDim];        // 16 KB, SW64 K-major
    alignas(1024) uint8_t k[4][T * kIdxDim];       // 32 KB, SW64 K-major
    uint64_t bar_s;                                // UMMA completion (cnt 1)
    uint32_t tmem_base;
};
static_assert(sizeof(Smem) <= 56 * 1024,
              "4 x Smem must fit the SM smem budget at 4 CTAs/SM");

} // namespace decode
} // namespace m3idx31

namespace ourscore {

__global__ void __launch_bounds__(128, 4) our_decode_kernel(
    const OurDecodeParams p) {
    using namespace cute;
    using fp8e4m3 = ourscore::fp8e4m3;
    namespace dc = m3idx31::decode;
    using m3idx31::kIdxDim;
    using m3idx31::kIdxQHeads;
    using m3idx31::kPageSize;

    constexpr int T = dc::T;

    extern __shared__ __align__(1024) uint8_t our_smem[];
    dc::Smem& sm = *reinterpret_cast<dc::Smem*>(our_smem);
    const int tid = threadIdx.x;

    // grid (1, batch, block_chunks): plain decode only -- one query token
    // per request (grid.x is kept in the launch shape for symmetry; only
    // qb == 0 is meaningful).  The block window is computed BEFORE any smem
    // barrier / TMEM allocation so empty-window CTAs exit without leaking.
    if ((int)blockIdx.x != 0) return;
    const int b = (int)blockIdx.y;
    const int blk_lo = (int)blockIdx.z * dc::kBlockChunk;
    const int blk_hi = min(blk_lo + dc::kBlockChunk, p.max_blocks);
    if (blk_lo >= blk_hi) return;

    // ---- request geometry (device metadata; production-reader parity) ----
    int seq_len = __ldg(p.seq_lens + b);
    {
        const int cap = p.max_blocks * 128;   // block table only covers max_blocks
        if (seq_len > cap) seq_len = cap;
        if (seq_len < 0) seq_len = 0;
    }
    const int num_blocks = (seq_len + 127) / 128;
    const int local_start = max(0, num_blocks - p.local_blocks);
    const int nblocks_c = blk_hi - blk_lo;
    const int nstages = 2 * nblocks_c;       // half-page stages

    // ---- one-time setup ----
    if (tid == 0) mbar_init(&sm.bar_s, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    __syncthreads();
    if (tid < 32) {
        cute::TMEM::Allocator1Sm alloc;
        alloc.allocate(dc::kTmemCols, &sm.tmem_base);
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    __syncthreads();
    const uint32_t tmem = sm.tmem_base;

    auto q_layout = tile_to_shape(UMMA::Layout_K_SW64_Atom<fp8e4m3>{},
                                  Shape<Int<128>, Int<kIdxDim>>{}, Step<_1, _2>{});
    auto k_layout = tile_to_shape(UMMA::Layout_K_SW64_Atom<fp8e4m3>{},
                                  Shape<Int<T>, Int<kIdxDim>>{}, Step<_1, _2>{});

    auto mma_s = make_tiled_mma(MMA_Atom<MMA_Traits<
        SM100_MMA_F8F6F4_SS, fp8e4m3, fp8e4m3, float,
        C<128>, C<T>,
        integral_constant<UMMA::Major, UMMA::Major::K>,
        integral_constant<UMMA::Major, UMMA::Major::K>,
        integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
        integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>>{});
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<fp8e4m3*>(sm.q)), q_layout);
    Tensor tAq = mma_s.get_slice(0).partition_fragment_A(sQ);
    auto mk_Bk = [&](int kb) {
        Tensor sK = make_tensor(make_smem_ptr(
                reinterpret_cast<fp8e4m3*>(sm.k[kb])), k_layout);
        return mma_s.get_slice(0).partition_fragment_B(sK);
    };
    auto tBk0 = mk_Bk(0), tBk1 = mk_Bk(1), tBk2 = mk_Bk(2), tBk3 = mk_Bk(3);
    Tensor tS0 = partition_fragment_C(mma_s, make_shape(Int<128>{}, Int<T>{}));
    tS0.data() = make_tmem_ptr<float>(tmem + dc::kTmemS0);
    Tensor tS1 = tS0;
    tS1.data() = make_tmem_ptr<float>(tmem + dc::kTmemS1);

    auto issue_qk = [&](int kb, int sb) {
        auto& tBk = kb == 0 ? tBk0 : kb == 1 ? tBk1 : kb == 2 ? tBk2 : tBk3;
        auto& tS  = sb ? tS1 : tS0;
        mma_s.accumulate_ = decltype(mma_s.accumulate_)::Zero;
        CUTE_UNROLL
        for (int uu = 0; uu < size<2>(tAq); ++uu) {
            cute::gemm(mma_s, tAq(_,_,uu), tBk(_,_,uu), tS);
            mma_s.accumulate_ = decltype(mma_s.accumulate_)::One;
        }
        if (cute::elect_one_sync())
            mma_commit(&sm.bar_s);
    };

    // ---- stage Q: the request's single token row [4,128] (512 B) into
    // packed rows 0-3; packed rows 4-127 duplicate row 0 (their scores are
    // computed but never stored -- the row_active gate below). ----
    {
        const uint8_t* qsrc = reinterpret_cast<const uint8_t*>(p.q8)
                              + (size_t)b * (size_t)p.q8_stride_b;
        for (int i = tid; i < 128 * 8; i += 128) {
            const int r = i >> 3, c = i & 7;
            const uint8_t* src = qsrc            // real row is always 0 (plain decode)
                                 + (size_t)(r & 3) * kIdxDim + (size_t)c * 16;
            uint32_t saddr = cute::cast_smem_ptr_to_uint(
                sm.q + q_layout(r, c * 16));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                         :: "r"(saddr), "l"(src));
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }

    // ---- K loader: stage stg covers page blk_lo + (stg>>1) at token
    // offset (stg&1)*64.  ONE gather per page from block_table; invalid
    // page ids skip the loads entirely (empty cp.async group) -- the K
    // smem keeps stale bytes and the consume masks them out with -inf. ----
    auto issue_load_k = [&](int stg) {
        const int pg = blk_lo + (stg >> 1);
        const int half = stg & 1;
        const int32_t g = __ldg(p.block_table + (size_t)b * (size_t)p.bt_stride + pg);
        const bool ok = (g >= 0 && g < p.phys_pages);
        if (ok) {
            const uint8_t* base = reinterpret_cast<const uint8_t*>(p.staged_k)
                                  + ((size_t)g * kPageSize + (size_t)half * 64) * kIdxDim;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {        // 64 rows x 8 chunks / 128 threads
                const int idx = tid + i * 128;
                const int r = idx >> 3;           // token row 0..63 within the half
                const int c = idx & 7;
                const uint8_t* src = base + (size_t)r * kIdxDim + (size_t)c * 16;
                uint32_t saddr = cute::cast_smem_ptr_to_uint(
                    sm.k[stg % 4] + k_layout(r, c * 16));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                             :: "r"(saddr), "l"(src));
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };

    // ---- prologue (b128d order: Q wait -> K(0) issue+wait -> fence ->
    // UMMA(0) -> K(1) issue) ----
    {
        asm volatile("cp.async.wait_group 0;" ::: "memory");   // Q landed
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        issue_load_k(0);
        asm volatile("cp.async.wait_group 0;" ::: "memory");   // K(0) landed
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        if (tid < 32) issue_qk(0, 0);
        if (nstages > 1)
            issue_load_k(1);
    }

    // ---- consume mapping: thread = packed m-row ----
    const int mrow = tid;
    const int r_real = mrow >> 2;
    const int h = mrow & 3;
    const bool row_active = (r_real == 0);          // plain decode: token 0 only
    const int prow = row_active ? (seq_len - 1) : -1;   // decode q_pos = seq_len-1

    // block accumulator: the T=64 stage PAIR (even+odd) is one 128-token
    // page = one output block; m_blk lives in registers across the pair.
    float m_blk = -INFINITY;

    for (int t = 0; t < nstages; ++t) {
        if (t + 2 < nstages)
            issue_load_k(t + 2);

        // wait QK(t): the (t+1)-th bar_s arrival completes phase t
        mbar_wait(&sm.bar_s, (unsigned)t & 1);
        asm volatile("tcgen05.fence::after_thread_sync;");

        uint32_t rv[64];
        {
            const uint32_t base = tmem + (t & 1 ? dc::kTmemS1 : dc::kTmemS0)
                                + ((uint32_t)(mrow & ~31) << 16);
            tmem_ld32(base + 0,  *reinterpret_cast<uint32_t (*)[32]>(&rv[0]));
            tmem_ld32(base + 32, *reinterpret_cast<uint32_t (*)[32]>(&rv[32]));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
        }

        // stage = half page: visible tokens are a contiguous prefix
        // (visibility is monotone in the token index): n = clamp(
        // min(prow+1, seq_len) - (pg*128 + half*64), 0, 64).  An invalid
        // page contributes nothing (m stays -INFINITY, K never read).
        const int pg = blk_lo + (t >> 1);
        const int half = t & 1;
        const int32_t g = __ldg(p.block_table + (size_t)b * (size_t)p.bt_stride + pg);
        const bool page_ok = (g >= 0 && g < p.phys_pages);
        const int n = vis_count(pg * 128 + half * 64, prow, seq_len);
        float m = -INFINITY;
        if (page_ok)
            m = prefix_max64(&rv[0], n);
        m_blk = (t & 1) ? fmaxf(m_blk, m) : m;

        if ((t & 1) || t == nstages - 1) {
            const int blk = pg;                    // == blk_lo + (t >> 1)
            if (row_active) {
                float val;
                if (!page_ok || blk >= num_blocks) {
                    // invalid page (regardless of init/local position) or
                    // tail block: -inf, rewritten on EVERY launch so a
                    // shorter request can never inherit a previous
                    // replay's tail scores.
                    val = -INFINITY;
                } else {
                    // sm_scale is a plain fp32 multiplier AFTER the
                    // per-block max (never log2e-scaled, never folded into
                    // the dot); overrides in the exact ABI order.
                    val = p.sm_scale * m_blk;
                    if (blk < p.init_blocks && blk < local_start)
                        val = 1e30f;
                    if (blk >= local_start)
                        val = 1e29f;
                }
                p.out[((size_t)h * (size_t)p.batch + (size_t)b) * (size_t)p.max_blocks
                      + (size_t)blk] = val;
            }
        }

        if (t + 1 < nstages) {
            if (t + 2 < nstages)
                asm volatile("cp.async.wait_group 1;" ::: "memory");
            else
                asm volatile("cp.async.wait_group 0;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
            __syncthreads();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (tid < 32) issue_qk((t + 1) % 4, (t + 1) & 1);
        }
    }

    __syncthreads();
    if (tid < 32) {
        cute::TMEM::Allocator1Sm alloc;
        alloc.free(tmem, dc::kTmemCols);
    }
}

// Host-side one-time setup, defined in the kernel's own TU.
bool decode_prepare() {
    static const bool ok = []() {
        return cudaFuncSetAttribute(
                   our_decode_kernel,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   decode_smem_bytes()) == cudaSuccess;
    }();
    return ok;
}

int decode_smem_bytes() {
    return (int)sizeof(m3idx31::decode::Smem);
}

} // namespace ourscore

// ---------------- host launch (the our_decode_launch ABI entry) ----------------
//
// Error codes (nonzero -> the ctypes bridge raises RuntimeError):
//   0 success; 1 invalid arguments; 2 cudaFuncSetAttribute failed;
//   1000 + e kernel launch failed with CUDA error e.
//
// The caller (bridge) guarantees: the E4M3 staging scratch is cached by
// (device, pages) so its address is stable across calls (CUDA-graph
// capture/replay safety); all device pointers live until stream
// completion; the current device is the tensors' device; no host-device
// syncs happen here.
extern "C" int our_decode_launch(const OurDecodeParams* p, cudaStream_t stream) {
    if (p == nullptr) return 1;
    if (p->q8 == nullptr || p->staged_k == nullptr || p->block_table == nullptr ||
        p->seq_lens == nullptr || p->out == nullptr)
        return 1;
    if (p->batch < 1 || p->batch > 65535) return 1;     // grid.y limit
    if (p->max_blocks < 1) return 1;
    if (p->bt_stride < p->max_blocks) return 1;         // ABI: elements
    if (p->q8_stride_b < 4 * 128 || (p->q8_stride_b & 15) != 0)
        return 1;   // >= one [4,128] row, 16B-aligned for cp.async 16B chunks
    if (p->phys_pages < 1) return 1;                    // sentinel page == phys_pages
    if (p->init_blocks < 0 || p->local_blocks < 0) return 1;

    if (!ourscore::decode_prepare()) return 2;

    // grid (q_tiles=1, batch, block_chunks): every (b, blk) pair of
    // [0, batch) x [0, max_blocks) is covered -- the tail [num_blocks,
    // max_blocks) is rewritten to -inf by the same launch.
    constexpr int kBlockChunk = m3idx31::decode::kBlockChunk;
    const int block_chunks = (p->max_blocks + kBlockChunk - 1) / kBlockChunk;
    const dim3 grid(1, (unsigned)p->batch, (unsigned)block_chunks);
    ourscore::our_decode_kernel<<<grid, 128, (size_t)ourscore::decode_smem_bytes(), stream>>>(*p);
    const cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : 1000 + (int)err;
}
