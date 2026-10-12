#include "rtp_llm/models_py/bindings/cuda/kernels/compact_spec_vocab_mask.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <limits>
#include <algorithm>

namespace rtp_llm {
namespace {

// Contiguous FP32 channels: one aligned vector and one allow nibble per
// thread. All-allow vectors are untouched; all-denied vectors need no load.
__global__ void applyCompactSpecVocabMaskFloat4Kernel(float* logits,
                                                     const int32_t* bitmask,
                                                     int64_t vocab_size,
                                                     int64_t words,
                                                     int64_t row_stride,
                                                     float masked_value) {
    const int64_t row = blockIdx.y;
    const int64_t token = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (token >= vocab_size) {
        return;
    }
    const uint32_t word = static_cast<uint32_t>(bitmask[row * words + token / 32]);
    const uint32_t allow = (word >> (token % 32)) & 0xf;
    if (allow == 0xf) {
        return;
    }
    auto* vector = reinterpret_cast<float4*>(logits + row * row_stride + token);
    float4 value;
    if (allow == 0) {
        value = make_float4(masked_value, masked_value, masked_value, masked_value);
    } else {
        value = *vector;
        if (!(allow & 1)) value.x = masked_value;
        if (!(allow & 2)) value.y = masked_value;
        if (!(allow & 4)) value.z = masked_value;
        if (!(allow & 8)) value.w = masked_value;
    }
    *vector = value;
}

template<typename scalar_t>
__global__ void applyCompactSpecVocabMaskKernel(scalar_t* logits,
                                               const int32_t* bitmask,
                                               int64_t vocab_size,
                                               int64_t words,
                                               int64_t row_stride,
                                               int64_t token_stride,
                                               int64_t tiles_per_row,
                                               int64_t total_tiles,
                                               scalar_t masked_value) {
    for (int64_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        const int64_t row = tile / tiles_per_row;
        // Four coalesced stripes per CTA reduce scheduling overhead without
        // changing the scalar store semantics or imposing an alignment rule.
        const int64_t token_base = (tile % tiles_per_row) * (blockDim.x * 4) + threadIdx.x;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int64_t token = token_base + item * blockDim.x;
            if (token < vocab_size) {
                const uint32_t word = static_cast<uint32_t>(bitmask[row * words + token / 32]);
                if ((word & (uint32_t{1} << (token % 32))) == 0) {
                    logits[row * row_stride + token * token_stride] = masked_value;
                }
            }
        }
    }
}

template<typename token_t>
__global__ void buildStaticThinkSpecMaskKernel(const token_t* draft, const int32_t* boundaries,
                                               int32_t* mask, int32_t* caps, int64_t p, int64_t words) {
    const int64_t b = blockIdx.x;
    __shared__ int limit;
    const int32_t active = boundaries[b * 3];
    const int32_t begin = boundaries[b * 3 + 1], end = boundaries[b * 3 + 2];
    if (threadIdx.x == 0) {
        limit = static_cast<int>(p);
        if (active) {
            for (int i = 0; i < p; ++i) {
                // Match the legacy torch int64->int32 draft conversion.
                const int32_t t = static_cast<int32_t>(draft[b * p + i]);
                if (t < 0 || static_cast<int64_t>(t) >= words * 32
                    || t == begin || t == end) { limit = i; break; }
            }
        }
        caps[b] = limit;
    }
    __syncthreads();
    for (int64_t i = threadIdx.x; i < (p + 1) * words; i += blockDim.x) {
        const int64_t row = i / words, word = i % words;
        uint32_t bits = 0xffffffffu;
        if (active && row <= limit) {
            if (begin >= 0 && begin / 32 == word) bits &= ~(uint32_t{1} << (begin % 32));
            if (end >= 0 && end / 32 == word) bits &= ~(uint32_t{1} << (end % 32));
        }
        mask[b * (p + 1) * words + i] = static_cast<int32_t>(bits);
    }
}

}  // namespace

void buildStaticThinkSpecMask(const torch::Tensor& draft, const torch::Tensor& boundaries,
                             const torch::Tensor& bitmask, const torch::Tensor& cap) {
    TORCH_CHECK(draft.is_cuda() && boundaries.is_cuda() && bitmask.is_cuda() && cap.is_cuda(),
                "static Think mask requires CUDA tensors");
    TORCH_CHECK(draft.device() == boundaries.device() && draft.device() == bitmask.device()
                && draft.device() == cap.device(), "static Think mask device mismatch");
    TORCH_CHECK(draft.dim() == 2 && draft.is_contiguous()
                && (draft.scalar_type() == torch::kInt32 || draft.scalar_type() == torch::kInt64),
                "static Think mask requires contiguous int32/int64 draft");
    const int64_t b = draft.size(0), p = draft.size(1);
    TORCH_CHECK(b > 0 && b <= 2147483647 && p > 0 && p < 2147483647,
                "static Think mask invalid batch/budget");
    TORCH_CHECK(boundaries.dim() == 2 && boundaries.size(0) == b && boundaries.size(1) == 3 && boundaries.is_contiguous()
                && boundaries.scalar_type() == torch::kInt32, "static Think mask invalid descriptors");
    TORCH_CHECK(bitmask.dim() == 2 && bitmask.size(0) == b * (p + 1) && bitmask.size(1) > 0
                && bitmask.is_contiguous() && bitmask.scalar_type() == torch::kInt32,
                "static Think mask invalid packed output");
    TORCH_CHECK(cap.dim() == 1 && cap.size(0) == b && cap.is_contiguous()
                && cap.scalar_type() == torch::kInt32, "static Think mask invalid cap output");
    const c10::cuda::CUDAGuard guard(draft.device());
    if (draft.scalar_type() == torch::kInt32)
        buildStaticThinkSpecMaskKernel<<<static_cast<unsigned>(b), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            draft.data_ptr<int32_t>(), boundaries.data_ptr<int32_t>(), bitmask.data_ptr<int32_t>(),
            cap.data_ptr<int32_t>(), p, bitmask.size(1));
    else
        buildStaticThinkSpecMaskKernel<<<static_cast<unsigned>(b), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            draft.data_ptr<int64_t>(), boundaries.data_ptr<int32_t>(), bitmask.data_ptr<int32_t>(),
            cap.data_ptr<int32_t>(), p, bitmask.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void applyCompactSpecVocabMask(const torch::Tensor& logits, const torch::Tensor& bitmask) {
    TORCH_CHECK(logits.is_cuda() && bitmask.is_cuda(), "compact spec mask requires CUDA tensors");
    TORCH_CHECK(logits.device() == bitmask.device(), "compact spec mask device mismatch");
    TORCH_CHECK(logits.dim() == 2 && bitmask.dim() == 2, "compact spec mask requires matrices");
    TORCH_CHECK(logits.is_floating_point(), "compact spec mask requires floating logits");
    TORCH_CHECK(bitmask.scalar_type() == torch::kInt32 && bitmask.is_contiguous(),
                "compact spec mask requires contiguous int32 bitmask");
    const int64_t rows = logits.size(0);
    const int64_t vocab = logits.size(1);
    const int64_t words = (vocab + 31) / 32;
    TORCH_CHECK(bitmask.size(0) == rows && bitmask.size(1) == words, "compact spec mask shape mismatch");
    if (rows > 0 && vocab > 0) {
        TORCH_CHECK(logits.stride(0) > 0 && logits.stride(1) > 0
                        && (logits.stride(0) >= vocab * logits.stride(1)
                            || logits.stride(1) >= rows * logits.stride(0)),
                    "compact spec mask requires nonoverlapping logits rows");
    }
    const c10::cuda::CUDAGuard guard(logits.device());
    constexpr int threads = 256;
    constexpr int tokens_per_tile = threads * 4;
    const int64_t tiles_per_row = (vocab + tokens_per_tile - 1) / tokens_per_tile;
    const int64_t total_tiles = rows * tiles_per_row;
    const int blocks = static_cast<int>(std::min<int64_t>(total_tiles, 65535));
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                  logits.scalar_type(), "applyCompactSpecVocabMask", [&] {
        // Match masked_fill_'s checked scalar conversion. In particular, the
        // legacy -FLT_MAX sentinel overflows FP16 and must still be rejected.
        const scalar_t masked_value = at::Scalar(-std::numeric_limits<float>::max()).to<scalar_t>();
        if (rows == 0 || vocab == 0) {
            return;
        }
        if (logits.scalar_type() == torch::kFloat32 && vocab % 4 == 0
            && logits.stride(1) == 1 && logits.stride(0) % 4 == 0
            && reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0
            && rows <= 65535 && tiles_per_row <= 65535) {
            applyCompactSpecVocabMaskFloat4Kernel<<<dim3(static_cast<unsigned>(tiles_per_row),
                                                        static_cast<unsigned>(rows)),
                                                    threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                logits.data_ptr<float>(), bitmask.data_ptr<int32_t>(), vocab, words,
                logits.stride(0), static_cast<float>(masked_value));
            return;
        }
        applyCompactSpecVocabMaskKernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            logits.data_ptr<scalar_t>(), bitmask.data_ptr<int32_t>(), vocab, words,
            logits.stride(0), logits.stride(1), tiles_per_row, total_tiles, masked_value);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace rtp_llm
