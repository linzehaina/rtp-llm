#include "rtp_llm/models_py/bindings/cuda/kernels/commit_metadata_producer.h"

#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <limits>

namespace rtp_llm {
namespace {

constexpr int kTableBlocks = 8;
constexpr int kThreads = 256;

__global__ void commitMetadataProducerKernel(CudaGraphCommitMetadataParams params) {
    const int table_blocks = kTableBlocks * params.tables.count;
    if (blockIdx.x < table_blocks) {
        const auto task = params.tables.tasks[blockIdx.x / kTableBlocks];
        const int stripe = blockIdx.x % kTableBlocks;
        const int64_t size = task.dst_rows * task.dst_cols;
        for (int64_t i = int64_t(stripe) * blockDim.x + threadIdx.x;
             i < size; i += int64_t(kTableBlocks) * blockDim.x) {
            const int64_t row = i / task.dst_cols;
            const int64_t col = i % task.dst_cols;
            task.dst[i] = task.src && row < task.src_rows && col < task.src_cols ?
                              task.src[row * task.src_stride + col] : 0;
        }
        return;
    }

    // Every cumulative row reads the current source independently. No metadata
    // CTA reads a table or vector written by another CTA in this launch.
    for (int64_t row = threadIdx.x; row <= params.batch_capacity; row += blockDim.x) {
        if (row < params.batch_capacity) {
            const bool live = row < params.batch_size;
            params.output_input_lengths[row] = live ? params.input_lengths[row] : 0;
            params.output_prefix_lengths[row] = live ? params.prefix_lengths[row] : 0;
            params.output_sequence_lengths_plus_1[row] =
                !live ? 0 : params.sequence_lengths_plus_1 ? params.sequence_lengths_plus_1[row] :
                            static_cast<int32_t>(static_cast<uint32_t>(params.prefix_lengths[row]) + 1U);
            if (!live && params.output_sequence_lengths) {
                params.output_sequence_lengths[row] = 0;
            }
        }
        if (row <= params.batch_size) {
            uint32_t tokens = 0;
            uint32_t kv = 0;
            for (int prev = 0; prev < row; ++prev) {
                tokens += static_cast<uint32_t>(params.input_lengths[prev]);
                kv += static_cast<uint32_t>(params.input_lengths[prev])
                      + static_cast<uint32_t>(params.prefix_lengths[prev]);
            }
            params.output_cu_seqlens[row] = static_cast<int32_t>(tokens);
            params.output_cu_kv_seqlens[row] = static_cast<int32_t>(kv);
        } else {
            params.output_cu_seqlens[row] = params.last_valid;
            // cu_kv tail and the live decode-cu values are deliberately untouched.
            if (params.output_decode_cu_seqlens) {
                params.output_decode_cu_seqlens[row] = params.last_valid;
            }
        }
    }
}

} // namespace

void invokeCudaGraphCommitMetadataProducer(CudaGraphCommitMetadataParams params, cudaStream_t stream) {
    TORCH_CHECK(params.tables.count >= 0 && params.tables.count <= kMaxCudaGraphPrepareFillRegions,
                "invalid commit metadata table count");
    TORCH_CHECK(params.batch_size >= 0 && params.batch_size <= params.batch_capacity,
                "invalid commit metadata batch size/capacity");
    TORCH_CHECK(params.batch_capacity >= 0, "negative commit metadata capacity");
    TORCH_CHECK(params.output_cu_seqlens && params.output_cu_kv_seqlens,
                "null commit metadata cumulative output");
    TORCH_CHECK(params.batch_capacity == 0 ||
                    (params.output_input_lengths && params.output_prefix_lengths
                     && params.output_sequence_lengths_plus_1),
                "null commit metadata row output");
    TORCH_CHECK(params.batch_size == 0 || (params.input_lengths && params.prefix_lengths),
                "null commit metadata live source");
    for (int i = 0; i < params.tables.count; ++i) {
        const auto& task = params.tables.tasks[i];
        TORCH_CHECK(task.dst_rows >= 0 && task.dst_cols >= 0,
                    "negative commit table destination dimensions");
        TORCH_CHECK(task.dst_cols == 0 ||
                        task.dst_rows <= std::numeric_limits<int64_t>::max() / task.dst_cols,
                    "commit table destination size overflow");
        TORCH_CHECK(task.dst_rows == 0 || task.dst_cols == 0 || task.dst,
                    "null commit table destination");
        if (task.src) {
            TORCH_CHECK(task.src_rows >= 0 && task.src_cols >= 0
                            && task.src_rows <= task.dst_rows && task.src_cols <= task.dst_cols
                            && task.src_stride >= task.src_cols,
                        "invalid commit table live rectangle/stride");
            TORCH_CHECK(task.src_rows <= 1 || task.src_stride == 0 ||
                            task.src_rows - 1 <=
                                (std::numeric_limits<int64_t>::max() - task.src_cols) / task.src_stride,
                        "commit table source offset overflow");
        }
    }
    commitMetadataProducerKernel<<<kTableBlocks * params.tables.count + 1, kThreads, 0, stream>>>(params);
}

} // namespace rtp_llm
