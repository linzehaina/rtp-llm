// our_score_host.cu -- top-level translation unit of libour_score.so.
//
// The frozen build command compiles exactly three TUs:
//   this file (which #includes prefill/prefill_host.cu, exporting
//   our_prefill_launch and our_score_abi_version),
//   prefill/kernel_ourscore_prefill.cu (the prefill kernel),
//   decode/kernel_ourscore_decode.cu (the decode kernel + its launcher,
//   exporting our_decode_launch).
//
// Exactly three extern "C" symbols are exported, matching our_score_abi.h:
//   our_score_abi_version, our_prefill_launch, our_decode_launch.

#include "our_score_abi.h"

#include "prefill/prefill_host.cu"

extern "C" int our_score_abi_version(void) {
    return OUR_SCORE_ABI_VERSION;
}
