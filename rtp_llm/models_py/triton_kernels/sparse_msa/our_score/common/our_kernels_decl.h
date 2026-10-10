// our_kernels_decl.h -- cross-TU declarations for the our_score kernels.
//
// Build layout (frozen build command compiles exactly three TUs):
//   TU1 our_score_host.cu        -> includes prefill/prefill_host.cu; exports
//                                    our_score_abi_version + our_prefill_launch
//   TU2 prefill/kernel_ourscore_prefill.cu -> defines our_prefill_kernel
//                                    + ourscore::prefill_prepare()
//   TU3 decode/kernel_ourscore_decode.cu   -> defines our_decode_kernel,
//                                    ourscore::decode_prepare() and exports
//                                    our_decode_launch
//
// Kernel launches from TU1 reference the TU2 kernel through this
// declaration (host-side stub symbols are resolved when the .so links);
// cudaFuncSetAttribute runs inside the DEFINING TU via the *_prepare()
// functions so the attribute is always set on the defining symbol.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "our_score_abi.h"

namespace ourscore {

// ---- prefill (prefill/kernel_ourscore_prefill.cu) ----
// chunk_pages: pages (=blocks) covered per grid.z tile; tma_mode: 1 = TMA
// single-box loader (production path), 0 = cp.async fallback.
__global__ void our_prefill_kernel(const OurPrefillParams params,
                                    const __grid_constant__ CUtensorMap tmap_kv,
                                    int chunk_pages, int tma_mode);

// Host-side one-time setup (cudaFuncSetAttribute for ~161 KiB dynamic smem).
// Returns false on CUDA error.  Safe to call concurrently (C++ magic static).
bool prefill_prepare();

// sizeof of the prefill kernel's dynamic smem block (the Smem layout is
// private to the kernel TU; host launchers need only the byte size).
int prefill_smem_bytes();

// ---- decode (decode/kernel_ourscore_decode.cu) ----
__global__ void our_decode_kernel(const OurDecodeParams params);

// Host-side one-time setup (dynamic smem attribute).  Returns false on error.
bool decode_prepare();

// sizeof of the decode kernel's dynamic smem block.
int decode_smem_bytes();

// ---- fused prefill (prefill/kernel_ourscore_prefill_fused.cu) ----
// Same loader/MMA/consume structure as our_prefill_kernel at KStages=7 with
// the score store replaced by the fused top-16 state; emits topk_idx
// directly (requires blocks <= 1024, i.e. grid z == 1).
__global__ void our_fused_prefill_kernel(const OurFusedPrefillParams params,
                                          const __grid_constant__ CUtensorMap tmap_kv,
                                          int chunk_pages, int tma_mode);

// Host-side one-time setup (dynamic smem attribute for ~208 KiB).
// Returns false on CUDA error.  Safe to call concurrently (C++ magic static).
bool prefill_fused_prepare();

// sizeof of the fused prefill kernel's dynamic smem block.
int prefill_fused_smem_bytes();

} // namespace ourscore
