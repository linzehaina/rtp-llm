#!/usr/bin/env bash
# build.sh -- rebuild the frozen R8-PKG-E prefill champion .so (sm_103a).
#
# Exact reproduction of the MSAINTEGRATE108 campaign build
# (cpp_backend.py: Backend(decoded=True, resident12=True)):
#   nvcc -std=c++17 -O3 -gencode=arch=compute_103a,code=sm_103a
#        --fmad=false --expt-relaxed-constexpr -shared -Xcompiler=-fPIC
#        -Xptxas=-v -lineinfo
#        -DMSA_DECODED=1 -DMSA_RESIDENT12=1 (all other MSA_* = 0)
#        -I<cutlass> kernel.cu -o lib/prefill-r8pkge.so
#
# CUTLASS: the campaign used the fmha-wheel CUTLASS 4.x headers
# (msa-nvfp4-kda/dependencies/fmha-wheel/fmha_sm100/cutlass/include);
# the m3pro checkout (CUTLASS 4.4.2) is an acceptable substitute —
# both provide the tcgen05/TMEM CuTe atoms this kernel uses.
#
# The committed lib/prefill-r8pkge-c58d6a73.so is the SHA-pinned
# campaign binary (654282ca...f23ca); a rebuild is functionally
# identical but not guaranteed bit-identical.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

NVCC="${NVCC:-/usr/local/cuda-13.0/bin/nvcc}"

CAMPAIGN_CUTLASS="/data0/liukan.lk/RTP-LLM/build_logs/msa-nvfp4-kda/dependencies/fmha-wheel/fmha_sm100/cutlass/include"
M3PRO_CUTLASS="/home/liukan.lk/m3pro-128blk-20261009/third_party/cutlass/include"

CUTLASS_INC="${CUTLASS_INC:-}"
if [ -z "$CUTLASS_INC" ]; then
    if [ -d "$CAMPAIGN_CUTLASS/cutlass" ]; then
        CUTLASS_INC="$CAMPAIGN_CUTLASS"
    else
        CUTLASS_INC="$M3PRO_CUTLASS"
    fi
fi
echo "CUTLASS include: $CUTLASS_INC"

mkdir -p lib
OUT="${1:-lib/prefill-r8pkge-rebuilt.so}"
LOG="$OUT.build.log"

"$NVCC" -std=c++17 -O3 \
    -gencode=arch=compute_103a,code=sm_103a \
    --fmad=false --expt-relaxed-constexpr \
    -shared -Xcompiler=-fPIC \
    -Xptxas=-v -lineinfo \
    -DMSA_DUAL=0 -DMSA_DECODED=1 -DMSA_LATE_MERGE=0 \
    -DMSA_COMPACT=0 -DMSA_DUAL_QUERY=0 -DMSA_SPLIT_MERGE=0 \
    -DMSA_RESIDENT=0 -DMSA_RESIDENT2=0 -DMSA_RESIDENT3=0 -DMSA_RESIDENT4=0 \
    -DMSA_RESIDENT5=0 -DMSA_RESIDENT6=0 -DMSA_RESIDENT7=0 -DMSA_RESIDENT8=0 \
    -DMSA_RESIDENT9=0 -DMSA_RESIDENT11=0 -DMSA_RESIDENT12=1 -DMSA_RESIDENT13=0 \
    -DMSA_RESIDENT14=0 -DMSA_Q_TMEM=0 \
    -I. -I"$CUTLASS_INC" \
    kernel.cu \
    -o "$OUT" -lcudart -lcuda -lm 2>&1 | tee "$LOG"

echo "--- exported msa_ symbols ---"
nm -D "$OUT" | grep ' msa_' | sort

echo "--- sha256 ---"
sha256sum "$OUT"
