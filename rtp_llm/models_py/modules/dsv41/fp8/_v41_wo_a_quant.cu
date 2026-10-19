#include "_v41_wo_a_quant.cuh"

#define sm100_fp8_fp4_gemm_1d1d_impl v41_wo_a_group32_epilogue
#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>
#undef sm100_fp8_fp4_gemm_1d1d_impl

using namespace deep_gemm;

// Keep the vendor mainloop and storage definition; only specialize fixed
// schedules for the O-A geometry (8 groups, N=1024, K=4096).
template<unsigned M, unsigned N, unsigned Stages, bool Swap>
static void instantiate_v41_wo_a() {
    auto kernel = reinterpret_cast<void*>(&v41_wo_a_group32_epilogue<
        cute::UMMA::Major::K, cute::UMMA::Major::K,
        32, 32, 32, 0, 1024, 4096, M, N, 128, 8,
        128, 128, 128, Stages, 2, 128, 128, 2, Swap,
        V41_NUM_SMS, Swap, true, GemmType::Batched, false,
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::float_e4m3_t,
        epilogue::operators::V41CastToFP8>);
}

static void instantiate_schedules() {
    instantiate_v41_wo_a<32, 128, 11, true>();
    instantiate_v41_wo_a<128, 256, 5, false>();
    instantiate_v41_wo_a<192, 128, 7, true>();
}

template<unsigned M, unsigned N, unsigned Stages, bool Swap>
using Storage = layout::SM100FP8FP4GemmSharedStorage<
    Stages, 2, 2, Swap ? M / 2 : M, Swap ? N : N / 2,
    128, Swap ? 16 : M, 128, (M + 127) / 128 * 128,
    (N + 127) / 128 * 128, 1,
    cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::float_e4m3_t>;

extern "C" __device__ __constant__ unsigned v41_wo_a_metadata[6] = {
    sizeof(Storage<32, 128, 11, true>),
    sizeof(Storage<128, 256, 5, false>),
    sizeof(Storage<192, 128, 7, true>),
    V41_NUM_SMS, sizeof(EpilogueOperatorArgs), offsetof(EpilogueOperatorArgs, alpha)};
