// Test-only: include the actual production TU, so the probe calls its exact helper.
#include "rtp_llm/models_py/bindings/cuda/kernels/minimax_prefill_topk.cu"
namespace torch_ext {
__global__ void key_probe_kernel(const float* in, int64_t* out, int64_t length) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < length; i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        out[i] = score_key(in[i]);
}
torch::Tensor probe_score_keys(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kFloat && input.is_contiguous());
    const c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty(input.sizes(), input.options().dtype(at::kLong));
    if (input.numel()) {
        key_probe_kernel<<<std::min<int64_t>((input.numel() + 255) / 256, 4096), 256, 0,
                            at::cuda::getCurrentCUDAStream(input.get_device())>>>(
            input.data_ptr<float>(), output.data_ptr<int64_t>(), input.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}
}
