// our_common.cuh -- device-side helpers shared by the prefill and decode
// score kernels.  All inline device functions are lifted verbatim from the
// read-only M3.0 champion sources
//   /home/liukan.lk/m3pro-128blk-20261009/src/kernel_b128.cu   (b128_ld64,
//     b128_mbar_init/arrive/wait, b128_mma_commit, b128_wait_group)
//   /home/liukan.lk/m3pro-128blk-20261009/src/kernel_b128d.cu  (b128d_ld32)
// plus the M3.1 contiguous-prefix visibility helpers that replace b128's
// per-16-token-page b128_half_max decomposition (b128d's prefix-max rule
// widened to 64-token halves of a 128-token page).
#pragma once

#include <cstdint>
#include <cuda_fp8.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cuda.h>

#include "m3idx_types_m31.cuh"

namespace ourscore {

using fp8e4m3 = cutlass::float_e4m3_t;

// ---------------- mbarrier / cp.async / TMEM primitives (b128, verbatim) ----

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    uint32_t a = cute::cast_smem_ptr_to_uint(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(count));
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    uint32_t a = cute::cast_smem_ptr_to_uint(bar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a));
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
    uint32_t a = cute::cast_smem_ptr_to_uint(bar);
    asm volatile(
        "{\n\t.reg .pred P;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONE;\n\t"
        "bra LAB_WAIT;\n\t"
        "DONE:\n\t}"
        :: "r"(a), "r"(parity));
}

__device__ __forceinline__ void mma_commit(uint64_t* bar) {
    uint32_t a = cute::cast_smem_ptr_to_uint(bar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void cpasync_wait_group(int n) {
    switch (n) {
    case 0: asm volatile("cp.async.wait_group 0;" ::: "memory"); break;
    case 1: asm volatile("cp.async.wait_group 1;" ::: "memory"); break;
    case 2: asm volatile("cp.async.wait_group 2;" ::: "memory"); break;
    default: asm volatile("cp.async.wait_group 3;" ::: "memory"); break;
    }
}

// tcgen05.ld of one full 64-token half (b128's x64 form).
__device__ __forceinline__ void tmem_ld64(uint32_t addr, uint32_t (&r)[64]) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];"
        : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
          "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
          "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
          "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31]),
          "=r"(r[32]),"=r"(r[33]),"=r"(r[34]),"=r"(r[35]),"=r"(r[36]),"=r"(r[37]),"=r"(r[38]),"=r"(r[39]),
          "=r"(r[40]),"=r"(r[41]),"=r"(r[42]),"=r"(r[43]),"=r"(r[44]),"=r"(r[45]),"=r"(r[46]),"=r"(r[47]),
          "=r"(r[48]),"=r"(r[49]),"=r"(r[50]),"=r"(r[51]),"=r"(r[52]),"=r"(r[53]),"=r"(r[54]),"=r"(r[55]),
          "=r"(r[56]),"=r"(r[57]),"=r"(r[58]),"=r"(r[59]),"=r"(r[60]),"=r"(r[61]),"=r"(r[62]),"=r"(r[63])
        : "r"(addr));
}

// tcgen05.ld of one 32-column chunk (b128d's x32 form; decode reads a full
// 64-token half with two of these).
__device__ __forceinline__ void tmem_ld32(uint32_t addr, uint32_t (&r)[32]) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
          "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
          "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
          "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
        : "r"(addr));
}

// ---------------- M3.1 visibility helpers ----------------
//
// M3.1 pages are 128 contiguous tokens and visibility is monotone in the
// token index, so a 64-token half's visible tokens are a contiguous PREFIX
// of rv[0..64):  n = clamp(min(prow+1, kv_len) - base, 0, 64).  This is
// b128d's prefix rule; it replaces b128's per-16-token-page
// b128_half_max / b128_max16 decomposition.

// Visible-token count of the 64-token half starting at token offset `base`
// for a query row with causal bound `prow` (q_pos, inclusive) and segment
// KV length `kv_len`.
__device__ __forceinline__ int vis_count(int base, int prow, int kv_len) {
    int n = min(prow + 1, kv_len) - base;
    return max(0, min(n, 64));
}

// Max over the first n tokens of a 64-token half held in rv[0..64) TMEM
// registers; n <= 0 (fully invisible half) returns -INFINITY.
__device__ __forceinline__ float prefix_max64(const uint32_t* rv, int n) {
    float m = -INFINITY;
    #pragma unroll
    for (int uu = 0; uu < 64; ++uu)
        if (uu < n)
            m = fmaxf(m, reinterpret_cast<const float&>(rv[uu]));
    return m;
}

} // namespace ourscore
