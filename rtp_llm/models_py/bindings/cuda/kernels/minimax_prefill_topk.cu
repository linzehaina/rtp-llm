#include "rtp_llm/models_py/bindings/cuda/kernels/minimax_prefill_topk.h"

#ifndef USE_ROCM
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <climits>
#endif

namespace torch_ext {
#ifndef USE_ROCM
namespace {

constexpr int      kBlockSize  = 128;
constexpr int      kRowsPerCTA = 4;
constexpr unsigned kFullWarp   = 0xffffffffu;

__device__ uint32_t score_key(float value) {
    // Canonicalize signed zero: ties follow FP32 numeric comparison, not sign bits.
    uint32_t bits = __float_as_uint(value);
    // Bazel enables FTZ; compare bits so nonzero subnormals remain distinct.
    if ((bits & 0x7fffffffu) == 0)
        bits = 0;
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

template<int Capacity, int K, bool EmitBT>
__global__ void minimax_prefill_warp_topk_kernel(const float* __restrict__ scores,
                                                 int32_t* __restrict__ topk_idx,
                                                 int32_t* __restrict__ block_tables,
                                                 int32_t* __restrict__ seq_lens,
                                                 int32_t* __restrict__ tie_flags,
                                                 const int32_t* __restrict__ cu,
                                                 const int32_t* __restrict__ prefix,
                                                 int64_t total_q,
                                                 int64_t score_h,
                                                 int64_t score_q,
                                                 int64_t out_h,
                                                 int64_t out_q,
                                                 int64_t bt_row,
                                                 int64_t seq_stride,
                                                 int     n,
                                                 int     heads,
                                                 int     qmax,
                                                 int     init,
                                                 int     local,
                                                 int     num_pages) {
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int batch = blockIdx.y;
    const int head  = blockIdx.z;
    const int q     = blockIdx.x * kRowsPerCTA + warp;
    const int lo    = cu[batch];
    const int hi    = cu[batch + 1];
    CUDA_KERNEL_ASSERT(lo >= 0 && hi >= lo && hi <= total_q);
    CUDA_KERNEL_ASSERT(batch != 0 || lo == 0);
    CUDA_KERNEL_ASSERT(batch != gridDim.y - 1 || hi == total_q);
    CUDA_KERNEL_ASSERT(hi - lo <= qmax && prefix[batch] >= 0);
    // Uniform within a warp: every live warp participates with all 32 lanes.
    if (q >= hi - lo)
        return;
    const int64_t position = static_cast<int64_t>(prefix[batch]) + q;
    const int     end      = (position + kBlockSize) / kBlockSize;
    CUDA_KERNEL_ASSERT(end <= n);
    const int    keep          = min(K, end);
    const int    current_block = position / kBlockSize;
    const float* row           = scores + static_cast<int64_t>(head) * score_h + static_cast<int64_t>(lo + q) * score_q;

    // One coalesced pass. For N=579 each lane holds 20 keys, with implicit indices.
    uint32_t values[Capacity / 32];
    bool     local_special = false;
#pragma unroll
    for (int j = 0; j < Capacity / 32; ++j) {
        const int idx = j * 32 + lane;
        uint32_t  key = 0;
        if (idx < n && idx < end) {
            float value = row[idx];
            if (isnan(value))
                value = -1e30f;
            if (idx < init)
                value = 1e30f;
            if (idx >= end - local)
                value = 1e29f;
            // The legacy floating-point sorting network can propagate infinities
            // through masked arithmetic, even if that candidate is not selected.
            local_special |= (__float_as_uint(value) & 0x7fffffffu) >= 0x7f800000u;
            key = score_key(value);
        }
        values[j] = key;
    }

    int      nonlocal_count  = 0;
    bool     has_current     = false;
    uint32_t previous_winner = 0;
    bool     has_tie         = __any_sync(kFullWarp, local_special);
#pragma unroll 1
    for (int t = 0; t < K; ++t) {
        uint32_t best     = 0;
        int      best_idx = INT_MAX;
#pragma unroll
        for (int j = 0; j < Capacity / 32; ++j) {
            if (values[j] > best) {
                best     = values[j];
                best_idx = j * 32 + lane;
            }
        }
#if __CUDA_ARCH__ >= 800
        const uint32_t winner = __reduce_max_sync(kFullWarp, best);
        const int      idx    = __reduce_min_sync(kFullWarp, best == winner ? best_idx : INT_MAX);
#else
        uint32_t winner = best;
        for (int offset = 16; offset; offset /= 2)
            winner = max(winner, __shfl_xor_sync(kFullWarp, winner, offset));
        int idx = best == winner ? best_idx : INT_MAX;
        for (int offset = 16; offset; offset /= 2)
            idx = min(idx, __shfl_xor_sync(kFullWarp, idx, offset));
#endif
        if (t < keep) {
            has_tie |= t > 0 && winner == previous_winner;
            previous_winner = winner;
        }
        if (lane == 0) {
            topk_idx[static_cast<int64_t>(head) * out_h + static_cast<int64_t>(lo + q) * out_q + t] =
                t < keep ? idx : -1;
            if (EmitBT && t < keep) {
                const bool is_current = idx == current_block;
                const int  pos        = is_current ? keep - 1 : nonlocal_count++;
                has_current |= is_current;
                block_tables[(static_cast<int64_t>(lo + q) * heads + head) * bt_row + pos] = head * num_pages + idx;
            }
        }
#pragma unroll
        for (int j = 0; j < Capacity / 32; ++j)
            if (j * 32 + lane == idx)
                values[j] = 0;
    }
    // Adjacent selected ties and an unselected tie at the cutoff are exactly
    // the rows whose IDs can differ from the historical bitonic network.
    // Legacy masked candidates use -1e30 rather than a key outside the domain.
    // Re-run legacy sorting if those candidates can affect the selected cutoff.
    bool cutoff_tie = previous_winner <= score_key(-1e30f);
    if (!has_tie) {
#pragma unroll
        for (int j = 0; j < Capacity / 32; ++j)
            cutoff_tie |= values[j] == previous_winner;
        has_tie = __any_sync(kFullWarp, cutoff_tie);
    }
    if (lane == 0)
        tie_flags[static_cast<int64_t>(head) * total_q + lo + q] = has_tie;
    if (EmitBT && lane == 0) {
        seq_lens[(static_cast<int64_t>(lo + q) * heads + head) * seq_stride] =
            has_current ? (keep - 1) * kBlockSize + position % kBlockSize + 1 : keep * kBlockSize;
    }
}

template<int Capacity, int K>
void launch(const torch::Tensor& scores,
            torch::Tensor&       out,
            torch::Tensor&       bt,
            torch::Tensor&       sl,
            const torch::Tensor& cu,
            const torch::Tensor& prefix,
            torch::Tensor&       tie_flags,
            int                  qmax,
            int                  init,
            int                  local,
            bool                 emit_bt,
            int                  num_pages) {
    const dim3 grid((qmax + kRowsPerCTA - 1) / kRowsPerCTA, cu.numel() - 1, scores.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream(scores.get_device());
#define M3_LAUNCH(BT)                                                                                                  \
    minimax_prefill_warp_topk_kernel<Capacity, K, BT>                                                                  \
        <<<grid, 128, 0, stream>>>(scores.data_ptr<float>(),                                                           \
                                   out.data_ptr<int32_t>(),                                                            \
                                   emit_bt ? bt.data_ptr<int32_t>() : nullptr,                                         \
                                   emit_bt ? sl.data_ptr<int32_t>() : nullptr,                                         \
                                   tie_flags.data_ptr<int32_t>(),                                                      \
                                   cu.data_ptr<int32_t>(),                                                             \
                                   prefix.data_ptr<int32_t>(),                                                         \
                                   scores.size(1),                                                                     \
                                   scores.stride(0),                                                                   \
                                   scores.stride(1),                                                                   \
                                   out.stride(0),                                                                      \
                                   out.stride(1),                                                                      \
                                   emit_bt ? bt.stride(0) : 0,                                                         \
                                   emit_bt ? sl.stride(0) : 0,                                                         \
                                   scores.size(2),                                                                     \
                                   scores.size(0),                                                                     \
                                   qmax,                                                                               \
                                   init,                                                                               \
                                   local,                                                                              \
                                   num_pages)
    if (emit_bt) {
        M3_LAUNCH(true);
    } else {
        M3_LAUNCH(false);
    }
#undef M3_LAUNCH
}

template<int K>
void dispatch(const torch::Tensor& s,
              torch::Tensor&       o,
              torch::Tensor&       bt,
              torch::Tensor&       sl,
              const torch::Tensor& cu,
              const torch::Tensor& pre,
              torch::Tensor&       tie_flags,
              int                  qmax,
              int                  init,
              int                  local,
              bool                 emit,
              int                  np) {
#define M3_DISPATCH(C) launch<C, K>(s, o, bt, sl, cu, pre, tie_flags, qmax, init, local, emit, np)
    if (s.size(2) <= 32) {
        M3_DISPATCH(32);
    } else if (s.size(2) <= 128) {
        M3_DISPATCH(128);
    } else if (s.size(2) <= 256) {
        M3_DISPATCH(256);
    } else if (s.size(2) <= 384) {
        M3_DISPATCH(384);
    } else if (s.size(2) <= 512) {
        M3_DISPATCH(512);
    } else if (s.size(2) <= 640) {
        M3_DISPATCH(640);
    } else if (s.size(2) <= 768) {
        M3_DISPATCH(768);
    } else {
        M3_DISPATCH(1024);
    }
#undef M3_DISPATCH
}

void check_tensor(const torch::Tensor& value, const torch::Tensor& scores, at::ScalarType dtype, const char* name) {
    TORCH_CHECK(value.is_cuda() && value.device() == scores.device(), name, " must be on scores' CUDA device");
    TORCH_CHECK(value.scalar_type() == dtype, name, " has an invalid dtype");
}

void check_rows(const torch::Tensor& value, const char* name) {
    TORCH_CHECK(value.stride(2) == 1
                    && value.stride(1) >= value.size(2)
                    // Empty tensors can carry arbitrarily large strides without
                    // allocating storage. Division avoids overflowing Q * stride.
                    && value.stride(0) / value.stride(1) >= value.size(1),
                name,
                " requires nonoverlapping head/query rows and a contiguous last dimension");
}

void check_disjoint(const torch::Tensor& a, const torch::Tensor& b) {
    if (a.numel() == 0 || b.numel() == 0)
        return;
    auto extent = [](const torch::Tensor& t) {
        int64_t last = 0;
        for (int d = 0; d < t.dim(); ++d)
            last += (t.size(d) - 1) * t.stride(d);
        return static_cast<uintptr_t>((last + 1) * t.element_size());
    };
    const auto begin_a = reinterpret_cast<uintptr_t>(a.data_ptr());
    const auto begin_b = reinterpret_cast<uintptr_t>(b.data_ptr());
    TORCH_CHECK(begin_a + extent(a) <= begin_b || begin_b + extent(b) <= begin_a,
                "TopK outputs must not overlap inputs or other outputs");
}

}  // namespace
#endif

torch::Tensor minimax_prefill_topk(const torch::Tensor& scores,
                                   torch::Tensor&       topk_idx,
                                   torch::Tensor&       block_tables,
                                   torch::Tensor&       output_seq_lens,
                                   const torch::Tensor& cu_seqlens,
                                   const torch::Tensor& prefix_lens,
                                   int64_t              max_query_length,
                                   int64_t              topk,
                                   int64_t              init_blocks,
                                   int64_t              local_blocks,
                                   bool                 emit_block_table,
                                   int64_t              num_pages) {
#ifndef USE_ROCM
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == at::kFloat && scores.dim() == 3,
                "scores must be a 3D CUDA float32 tensor");
    const c10::cuda::CUDAGuard guard(scores.device());
    check_tensor(topk_idx, scores, at::kInt, "topk_idx");
    check_tensor(cu_seqlens, scores, at::kInt, "cu_seqlens");
    check_tensor(prefix_lens, scores, at::kInt, "prefix_lens");
    TORCH_CHECK(topk == 16 || topk == 32, "topk must be 16 or 32");
    TORCH_CHECK(scores.size(2) > 0 && scores.size(2) <= 1024, "block count must be in [1, 1024]");
    TORCH_CHECK(topk_idx.dim() == 3 && topk_idx.size(0) == scores.size(0) && topk_idx.size(1) == scores.size(1)
                    && topk_idx.size(2) == topk,
                "topk_idx shape must be [heads, queries, topk]");
    check_rows(scores, "scores");
    check_rows(topk_idx, "topk_idx");
    TORCH_CHECK(cu_seqlens.dim() == 1 && prefix_lens.dim() == 1 && cu_seqlens.is_contiguous()
                    && prefix_lens.is_contiguous() && cu_seqlens.numel() == prefix_lens.numel() + 1,
                "cu_seqlens/prefix_lens must be contiguous 1D segment metadata");
    TORCH_CHECK(max_query_length >= 0 && max_query_length <= INT_MAX - 3 && scores.size(1) <= INT_MAX
                    && scores.size(0) <= 65535 && prefix_lens.numel() <= 65535,
                "query/head/segment extent exceeds launch limits");
    TORCH_CHECK(init_blocks >= 0 && init_blocks <= INT_MAX && local_blocks >= 0 && local_blocks <= INT_MAX,
                "init/local block counts must be nonnegative int32");
    check_disjoint(topk_idx, scores);
    check_disjoint(topk_idx, cu_seqlens);
    check_disjoint(topk_idx, prefix_lens);
    if (emit_block_table) {
        check_tensor(block_tables, scores, at::kInt, "block_tables");
        check_tensor(output_seq_lens, scores, at::kInt, "output_seq_lens");
        const int64_t rows = scores.size(0) * scores.size(1);
        TORCH_CHECK(block_tables.dim() == 2 && block_tables.size(0) == rows && block_tables.size(1) == topk
                        && block_tables.stride(1) == 1 && block_tables.stride(0) >= topk,
                    "invalid block_tables layout");
        TORCH_CHECK(output_seq_lens.dim() == 1 && output_seq_lens.numel() == rows && output_seq_lens.stride(0) > 0,
                    "invalid output_seq_lens layout");
        TORCH_CHECK(num_pages >= scores.size(2) && num_pages <= INT_MAX
                        && (scores.size(0) == 0 || num_pages <= INT_MAX / scores.size(0)),
                    "num_pages must cover blocks without int32 page-ID overflow");
        for (const auto& output : {block_tables, output_seq_lens}) {
            check_disjoint(output, scores);
            check_disjoint(output, cu_seqlens);
            check_disjoint(output, prefix_lens);
            check_disjoint(output, topk_idx);
        }
        check_disjoint(block_tables, output_seq_lens);
    }
    auto tie_flags = torch::empty({scores.size(0), scores.size(1)}, scores.options().dtype(at::kInt));
    if (scores.size(0) == 0 || scores.size(1) == 0)
        return tie_flags;
    TORCH_CHECK(max_query_length > 0 && prefix_lens.numel() > 0, "nonempty scores require segments");
    // Only narrow a page count after its range was checked. The index-only
    // specializations never use it, so do not truncate an unused int64 argument.
    const int checked_num_pages = emit_block_table ? static_cast<int>(num_pages) : 0;
    if (topk == 16)
        dispatch<16>(scores,
                     topk_idx,
                     block_tables,
                     output_seq_lens,
                     cu_seqlens,
                     prefix_lens,
                     tie_flags,
                     max_query_length,
                     init_blocks,
                     local_blocks,
                     emit_block_table,
                     checked_num_pages);
    else
        dispatch<32>(scores,
                     topk_idx,
                     block_tables,
                     output_seq_lens,
                     cu_seqlens,
                     prefix_lens,
                     tie_flags,
                     max_query_length,
                     init_blocks,
                     local_blocks,
                     emit_block_table,
                     checked_num_pages);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return tie_flags;
#else
    TORCH_CHECK(false, "minimax_prefill_topk is not supported on ROCm");
#endif
}

}  // namespace torch_ext
