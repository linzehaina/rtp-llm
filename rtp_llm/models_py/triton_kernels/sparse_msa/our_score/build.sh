#!/usr/bin/env bash
# build.sh -- build libour_score.so for the M3.1 our_score kernels (sm_103a).
#
# Mirrors the m3pro Makefile NVFLAGS (CUDA 13.0, -O3 -std=c++17,
# compute_103a/sm_103a, --expt-relaxed-constexpr --expt-extended-lambda)
# plus -shared/-fPIC.  CUTLASS headers: the m31 repo's 3rdparty/cutlass is
# checked FIRST but contains only BUILD stubs (no include/cutlass --
# verified on disk), so the effective include is the read-only m3pro
# checkout (CUTLASS 4.4.2, exactly what b128/b128d were built against);
# including headers via -I is read-only use.
#
# Output: our_score/lib/libour_score.so (+ build log with -Xptxas -v
# register/spill counts; sha256 printed).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

NVCC="${NVCC:-/usr/local/cuda-13.0/bin/nvcc}"
ARCH="${ARCH:-compute_103a}"
CODE="${CODE:-sm_103a}"

M31_CUTLASS="/home/liukan.lk/msa-m31-work/rtp-llm-m31/3rdparty/cutlass/include"
M3PRO_CUTLASS="/home/liukan.lk/m3pro-128blk-20261009/third_party/cutlass/include"

CUTLASS_INC=""
if [ -d "$M31_CUTLASS/cutlass" ]; then
    CUTLASS_INC="$M31_CUTLASS"
else
    # verified fallback: m31 3rdparty/cutlass is BUILD-stubs only
    CUTLASS_INC="$M3PRO_CUTLASS"
fi
echo "CUTLASS include: $CUTLASS_INC"

mkdir -p lib
LOG="lib/build.log"

"$NVCC" -O3 -std=c++17 \
    -gencode arch=$ARCH,code=$CODE \
    -I. -Icommon -Iprefill -Idecode -I"$CUTLASS_INC" \
    --expt-relaxed-constexpr --expt-extended-lambda \
    -Xcompiler -fPIC -shared \
    -Xptxas -v \
    our_score_host.cu prefill/kernel_ourscore_prefill.cu prefill/kernel_ourscore_prefill_fused.cu decode/kernel_ourscore_decode.cu \
    -o lib/libour_score.so \
    -lcudart -lcuda -lm 2>&1 | tee "$LOG"

echo "--- exported our_* symbols ---"
nm -D lib/libour_score.so | grep ' our_' | sort

echo "--- sha256 ---"
sha256sum lib/libour_score.so
