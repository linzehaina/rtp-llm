#pragma once

#include <deep_gemm/epilogue/operators.cuh>

namespace deep_gemm::epilogue::operators {

// Quantize FP32 accumulators directly; no intermediate BF16 rounding.
struct V41CastToFP8: CastToFP8WithClamp {
    using CastToFP8WithClamp::CastToFP8WithClamp;
    __device__ void init(const uint32_t&, const uint32_t&) {}
    __device__ void finalize() const {}

    __device__ static float mul_ftz(float a, float b) {
        float out;
        asm("mul.ftz.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
        return out;
    }

    __device__ unsigned scale_exp(float amax) const {
        amax = fmaxf(amax, 0x1p-126f);
        const float raw = args.alpha != 0.0f ? fmaxf(amax * (1.0f / 448.0f), 1.0e-10f) :
                                             mul_ftz(amax, 1.0f / 448.0f);
        const unsigned bits = __float_as_uint(raw);
        return ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0);
    }

    template<bool Swap, uint32_t Count>
    __device__ void prepare_fp8(const uint32_t (&values)[Count],
                               float2 (&fp32)[Count / 2], uint2& sf,
                               const uint32_t&, const uint32_t&) {
        float ax = 0.0f, ay = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < Count / 2; ++i) {
            fp32[i] = make_float2(__uint_as_float(values[i * 2]), __uint_as_float(values[i * 2 + 1]));
            ax = fmaxf(ax, fabsf(fp32[i].x));
            ay = fmaxf(ay, fabsf(fp32[i].y));
        }
        if constexpr (Swap) {
            ax = math::warp_reduce_max<4, true>(ax);
            ay = math::warp_reduce_max<4, true>(ay);
        } else {
            ax = ay = fmaxf(ax, ay);
        }
        sf.x = scale_exp(ax);
        sf.y = Swap ? scale_exp(ay) : sf.x;
    }

    template<uint32_t Count>
    __device__ void cast_fp8(const float2 (&fp32)[Count],
                            uint32_t (&packed)[Count / 2], const uint2& sf) const {
        const float sx = math::get_ue8m0_sf_inv<float>(sf.x);
        const float sy = math::get_ue8m0_sf_inv<float>(sf.y);
        #pragma unroll
        for (uint32_t i = 0; i < Count / 2; ++i) {
            packed[i] = __nv_fp8x4_e4m3(make_float4(
                mul_ftz(fp32[i * 2].x, sx), mul_ftz(fp32[i * 2].y, sy),
                mul_ftz(fp32[i * 2 + 1].x, sx), mul_ftz(fp32[i * 2 + 1].y, sy))).__x;
        }
    }
};

}  // namespace deep_gemm::epilogue::operators
