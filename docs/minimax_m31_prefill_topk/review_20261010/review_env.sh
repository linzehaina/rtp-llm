#!/bin/bash
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-13.2
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++ CUDAHOSTCXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export PATH=/opt/rh/gcc-toolset-12/root/usr/bin:/usr/local/cuda-13.2/bin:/opt/conda310/bin:/usr/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=10.3 CUDA_VISIBLE_DEVICES=${M31_GPU_IDS:-0} MAX_JOBS=4
export PYTHONPATH=/data4/aozhengkai.azk/m31_warp_20261010/review_20261010:${PYTHONPATH:-}
export TMPDIR=/data4/aozhengkai.azk/m31_warp_20261010/review_20261010/cache/tmp TORCH_EXTENSIONS_DIR=/data4/aozhengkai.azk/m31_warp_20261010/review_20261010/cache/torch_extensions DG_JIT_CACHE_DIR=/data4/aozhengkai.azk/m31_warp_20261010/review_20261010/cache/deep_gemm XDG_CACHE_HOME=/data4/aozhengkai.azk/m31_warp_20261010/review_20261010/cache/xdg CUDA_CACHE_PATH=/data4/aozhengkai.azk/m31_warp_20261010/review_20261010/cache/cuda TRITON_CACHE_DIR=/data4/aozhengkai.azk/m31_warp_20261010/review_20261010/cache/triton
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$DG_JIT_CACHE_DIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR"
exec "$@"
