#pragma once

#include <array>
#include <initializer_list>
#include <limits>
#include <torch/torch.h>
#include "rtp_llm/models_py/bindings/cuda/kernels/cuda_graph_prepare.h"

namespace rtp_llm {
namespace detail {

struct GraphTableTensorPair {
    const torch::Tensor* src = nullptr;
    const torch::Tensor* dst = nullptr;
};

struct GraphTableAddressRange {
    uintptr_t begin = 0;
    uintptr_t end = 0;
};

inline bool graphTableRangesOverlap(GraphTableAddressRange a, GraphTableAddressRange b) {
    return a.begin < b.end && b.begin < a.end;
}

// This is a host-only plan: no device reads, allocation, stream query, or sync.
// Failure leaves the output empty and the caller executes the complete legacy
// fill/copy sequence. In particular, partial metadata fills are never dropped.
inline bool makeGraphTableOverwritePlan(
    const std::array<GraphTableTensorPair, kMaxCudaGraphPrepareFillRegions>& tables,
    int count,
    const CudaGraphPrepareFillParams& fills,
    std::initializer_list<const torch::Tensor*> metadata,
    CudaGraphTableOverwriteParams& output) {
    output.count = 0;
    if (count <= 0 || count > kMaxCudaGraphPrepareFillRegions || fills.region_count != count) {
        return false;
    }
    CudaGraphTableOverwriteParams candidate;
    std::array<GraphTableAddressRange, kMaxCudaGraphPrepareFillRegions> reads{}, writes{};
    std::array<bool, kMaxCudaGraphPrepareFillRegions> matched{};
    int device = -1;
    for (int i = 0; i < count; ++i) {
        if (!tables[i].src || !tables[i].dst) {
            return false;
        }
        const auto& src = *tables[i].src;
        const auto& dst = *tables[i].dst;
        if (!dst.defined() || !dst.is_cuda() || dst.scalar_type() != torch::kInt32 || dst.dim() != 2
            || !dst.is_contiguous() || dst.numel() <= 0) {
            return false;
        }
        if (device < 0) {
            device = dst.get_device();
        }
        if (dst.get_device() != device) {
            return false;
        }
        auto& task = candidate.tasks[i];
        task.dst = dst.data_ptr<int32_t>();
        task.dst_rows = dst.size(0);
        task.dst_cols = dst.size(1);
        const auto begin = reinterpret_cast<uintptr_t>(task.dst);
        if (dst.nbytes() > std::numeric_limits<uintptr_t>::max() - begin) {
            return false;
        }
        writes[i] = {begin, begin + dst.nbytes()};
        // Undefined/empty sources have exactly the legacy fill-only effect.
        if (src.defined() && src.numel() > 0) {
            if (!src.is_cuda() || src.get_device() != device || src.scalar_type() != torch::kInt32
                || src.dim() != 2 || src.stride(1) != 1 || src.stride(0) < src.size(1)
                || src.size(0) > task.dst_rows || src.size(1) > task.dst_cols) {
                return false;
            }
            task.src = src.data_ptr<int32_t>();
            task.src_rows = src.size(0);
            task.src_cols = src.size(1);
            task.src_stride = src.stride(0);
            const auto src_begin = reinterpret_cast<uintptr_t>(task.src);
            // Tensor views have already passed Torch storage bounds. Use the
            // bounding span, including row gaps, conservatively for aliases.
            const auto elements = uint64_t(task.src_rows - 1) * uint64_t(task.src_stride)
                                  + uint64_t(task.src_cols);
            if (elements > (std::numeric_limits<uintptr_t>::max() - src_begin) / sizeof(int32_t)) {
                return false;
            }
            reads[i] = {src_begin, src_begin + elements * sizeof(int32_t)};
        }
        bool found = false;
        for (int f = 0; f < fills.region_count; ++f) {
            const auto& fill = fills.regions[f];
            if (!matched[f] && fill.ptr == task.dst && fill.count == dst.numel() && fill.value == 0) {
                matched[f] = true;
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < count; ++j) {
            if ((i != j && graphTableRangesOverlap(writes[i], writes[j]))
                || graphTableRangesOverlap(writes[i], reads[j])) {
                return false;
            }
        }
        for (const auto* tensor : metadata) {
            if (!tensor || !tensor->defined() || !tensor->is_cuda() || tensor->numel() <= 0
                || tensor->get_device() != device) {
                continue;
            }
            // Protect entire metadata storage, including non-contiguous views.
            const auto begin = reinterpret_cast<uintptr_t>(tensor->storage().data_ptr().get());
            const auto bytes = tensor->storage().nbytes();
            if (bytes > std::numeric_limits<uintptr_t>::max() - begin) {
                return false;
            }
            const GraphTableAddressRange range{begin, begin + bytes};
            if (graphTableRangesOverlap(writes[i], range) || graphTableRangesOverlap(reads[i], range)) {
                return false;
            }
        }
    }
    candidate.count = count;
    output = candidate;
    return true;
}

}  // namespace detail
}  // namespace rtp_llm
