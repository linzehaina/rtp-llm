# M3.1 FlashInfer Q8KV4 / TopK: CUDA 13.4 build and JIT audit

Status: 2026-10-11 candidate `feat/m31_topk_q8_model_1010` at `5168295546` plus
uncommitted Q8KV4 integration and JIT fixes. This is an opt-in experiment, not a
production-default recommendation. The original PD4+4 service was restored and
smoke-tested after the experiments.

Current source uses one Prefill switch: `M3_NVFP4_PREFILL_BACKEND=fmha`
(default), `flashinfer_q16` (FI BF16 compute), or `flashinfer_q8` (RTP Q8KV4
JIT). The old `M3_NVFP4_PREFILL_COMPUTE` setting is rejected for NVFP4; old
environment names below document historical measurements, not launch commands
for the current source. Q16 never loads the RTP Q8 module, and Q8 does not
warm up or execute the FI BF16 MSA kernel unless the opt-in shadow diagnostic
is enabled. The RTP Q8 `.cu` is submitted with this repo; the FI source repo
has no Q8 patch, while the final image still needs a pinned, compatible FI
wheel and CUDA 13.4 toolchain.

## Current verdict and evidence boundary

| Gate | Current evidence | Decision boundary |
|---|---|---|
| FI/CUDA 13.4 build | Private FI source wheel `0.6.12+250aae19`, private nvcc/headers/runtime 13.4.92, `sm_103a` Q8 module and full-model cold/hot smoke passed. Exact-service-environment prewarm left FI BF16 MSA, Q8 MSA, and RoPE Ninja logs unchanged through later model runs. | Rebuild and repeat on the final image and every deployed architecture; the private candidate is not a production AOT/cache wheel. |
| Q8KV4 compute | BF16 Q is cast on load; FP8 QK/FP4 K and BF16 PV have independent synthetic oracles and same-input FI BF16 shadow data. Observed real active-page K scales did not enter the FP8 saturation regime on tested BS20/60 and LongBench160. | Extreme synthetic K/V scales still fail strict oracle checks, and natural model attention differs from BF16. Keep BF16 the default and Q8 opt-in. |
| Native Prefill TopK | Model dispatch was observed at BS20/60; six-batch matched bitonic/warp arms saved 2.20%/1.74% full Prefill latency. A real-score shadow matched 800/800 ordered index tensors. | Derived long-context 40-item quality repeats were noisy; no broad quality non-inferiority or production throughput claim. |
| Dataset precision and MAL | Final-byte Q8 arm at concurrency 32: GSM8K1314 math-equivalent 1,230/1,314, legacy string score 1,116/1,314, MAL 4.16158; MATH500 383/500, MAL 3.85076. Both completed with zero request errors. An earlier same-manifest BF16 GSM run scored 1,227 math-equivalent; an earlier paired MATH run scored BF16 387 versus Q8 375. | Separate processes and stochastic decoding prevent a precision-equivalence or speedup claim; never mix GSM scorers. MATH changed IDs and layer/logit causality remain open. |
| JIT and stability | Final-byte full MATH500 and GSM8K1314 completed. The MATH window added four finite TopK/RMSNorm structural cubins; the following GSM window added none. Tiled SwiGLU stayed at 36 cubins. Exact-final-source FI prewarm was then verified by a second fresh process, Ninja dry runs and unchanged three-module hashes through a real PD4+4 smoke. Original PD4+4 was restored after each campaign. | A changed Q8 `.cu` had forced FI recompilation on an earlier first smoke despite a previously warm workspace. Exact-source prewarm closes that FI path for tested `sm_103a`, but Triton cold-request JIT and final-image/other-architecture signoff remain open. |

The sections below retain exact commands, workspaces, manifest hashes, raw
measurements, negative gates, and source/installed boundaries. A validation
result can be complete even when it is negative; it is not permission to
promote Q8KV4 or TopK as a production default.

## What needs CUDA 13.4

`msa_prefill_q8kv4.cu` uses PTX
`mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4` for the QK K-stage and rejects
`nvcc` below 13.4. BF16-Q/packed-KV4 FlashInfer MSA does not require that
instruction. The model flag `M3_NVFP4_PREFILL_BACKEND=flashinfer_q8` selects
Q8KV4; `flashinfer_q16` selects FI BF16, while `fmha` remains the default.
The new Q8 path keeps BF16 Q at the adapter boundary and converts it while
loading Q; it does not allocate a global Q8 buffer. PV remains BF16 MMA.

The tested container has `/usr/local/cuda-13.2`, Torch `2.11.0+cu130`, and
driver `580.105.08`. In the model A/B, `FLASHINFER_NVCC` selected an external
13.4.92 compiler and `FLASHINFER_EXTRA_CUDAFLAGS` added 13.4 headers, while
`CUDA_HOME=/usr/local/cuda-13.2`. Both FI RoPE and Q8 MSA compiled and ran,
but their shared objects linked container CUDA 13.2 `libcudart.so.13`.
Separately, an isolated **CPU-only compile/link** using 13.4 headers/runtime
also passed for Q8 MSA. A tiny GPU operator comparison of that exact `.so`
passed mixed scales and reproduced the extreme-scale failure described below;
it is not a full-model or long-running 13.4 runtime qualification.

## Reproducible private toolchain

Use an isolated target directory in the build/runtime image. Do not replace
`/usr/local/cuda` or the resident PD service while qualifying the candidate.
The package versions below match the tested binaries. A CPU-only build with
private 13.4 `CUDA_HOME` is proven. A later isolated FI wheel and GPU test
also validated both BF16 and Q8 MSA JIT modules against the same private 13.4
headers/runtime; it does not qualify a complete production image.

```bash
TOOLS=/data0/ruixuan.zrx/minimax-m31-dev/1010_topk_q8_model_candidate/cuda134
python3 -m pip install --target "$TOOLS" --no-deps \
  nvidia-cuda-nvcc==13.4.92 \
  nvidia-cuda-runtime==13.4.92 \
  nvidia-cuda-cccl==13.3.4.3.1
CUDA134="$TOOLS/nvidia/cu13"
"$CUDA134/bin/nvcc" --version
rg 'CUDA_VERSION|CUDART_VERSION' "$CUDA134/include/cuda.h" \
  "$CUDA134/include/cuda_runtime_api.h"
# pip runtime wheels expose the versioned soname, but FI's linker uses -lcudart.
ln -sfn lib "$CUDA134/lib64"
ln -sfn libcudart.so.13 "$CUDA134/lib/libcudart.so"
mkdir -p "$CUDA134/lib/stubs"
ln -sfn /usr/local/cuda/lib64/stubs/libcuda.so \
  "$CUDA134/lib/stubs/libcuda.so"
# If nvcc and runtime came from separate packages/locations, include nvcc's
# matching CRT headers under CUDA_HOME as well. Missing include/crt makes the
# generated Ninja depfile permanently dirty and recompiles on every process.
test -f "$CUDA134/include/crt/host_config.h"
```

In an isolated test process, set the following **before importing FlashInfer**.
`FLASHINFER_WORKSPACE_BASE` must be writable and distinct for each tested
image/source/toolchain combination; FlashInfer's normal workspace name uses
its version and CUDA architecture, not the compiler/header version. A full
13.4 link points `CUDA_HOME` and `LD_LIBRARY_PATH` at the same private
toolchain. The private `libcuda.so` stub is link-only; the loaded
`libcuda.so.1` must still resolve to the host driver.

```bash
export CUDA_HOME="$CUDA134"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CPATH="$CUDA134/include${CPATH:+:$CPATH}"
export FLASHINFER_NVCC="$CUDA134/bin/nvcc"
export FLASHINFER_EXTRA_CUDAFLAGS="-I$CUDA134/include"
export LD_LIBRARY_PATH="$CUDA134/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_WORKSPACE_BASE=/data0/ruixuan.zrx/minimax-m31-dev/fi-q8-cold-134
export M3_NVFP4_PREFILL_BACKEND=flashinfer_q8
export M3_PREFILL_TOPK_IMPL=warp
unset FLASHINFER_DISABLE_JIT
```

`CPATH` is required for the **full model**, not just FI: DeepGEMM follows
`CUDA_HOME` during MXFP8 weight-scale packing and invokes the pip-installed
13.4 `nvcc` without an explicit include path. In the tested pip layout,
omitting `CPATH` failed model startup with `cuda_runtime.h: No such file or
directory`; a direct `nvcc -E` repro failed without it and passed with it.

Build/install RTP from the **candidate worktree** with its existing Bazel
workflow and the same CUDA toolchain selection; verify the resulting installed
`rtp_llm` path, native library hashes, and that
`sparse_msa/prefill/msa_prefill_q8kv4.cu` is included. The source declares
the CUDA file in `rtp_llm/models_py/triton_kernels/BUILD`. The current candidate
wheel at `.../1010_topk_q8_model_candidate/install` was built earlier; it is
not evidence of a complete 13.4 Bazel rebuild.

After a **cold** isolated launch, inspect the actual generated recipe and
library dependencies, not only `nvcc --version`:

```bash
find "$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer" -path \
  '*/cached_ops/rtp_m31_prefill_q8kv4_sm103a/build.ninja' -print
rg '^(nvcc|cuda_home|ldflags) =|cuda_cflags =|cuda134' \
  "$FLASHINFER_WORKSPACE_BASE"/.cache/flashinfer/*/103a/cached_ops/rtp_m31_prefill_q8kv4_sm103a/build.ninja
ldd "$FLASHINFER_WORKSPACE_BASE"/.cache/flashinfer/*/103a/cached_ops/rtp_m31_prefill_q8kv4_sm103a/*.so
```

Require `nvcc 13.4`, headers reporting `13040`, a real `sm_103a` cubin, and
the intended runtime library path. Build the `sm_100a` variant separately if
that hardware is a deployment target. Do not infer device compatibility from
the host-reported `nvidia-smi` CUDA label.
The pip-installed private nvcc package used here does not include
`cuobjdump`; use the target image's `/usr/local/cuda/bin/cuobjdump --list-elf`
on the generated `.so` (or install matching CUDA binary utilities) to inspect
the cubin architecture. The current private Q8 module reports
`rtp_m31_prefill_q8kv4_sm103a.1.sm_103a.cubin`.

The isolated generated FI/Ninja recipe was replayed from an empty directory
with the private `CUDA_HOME`, private 13.4 `libcudart`, and the container's
driver stub. Both NVCC compilation and linking passed. `cuobjdump --list-elf`
showed `rtp_m31_prefill_q8kv4_sm103a.1.sm_103a.cubin`; `ldd` with the private
library path resolved 13.4 `libcudart.so.13`; immediate Ninja rerun printed
`no work to do`. Artifact:
`.../1010_topk_q8_model_candidate/jit_full134_native_layout/rtp_m31_prefill_q8kv4_sm103a/`
with `.so` SHA256 `a5bb5185d95e8ba71f4df751e90244b13ef970dbf7b174df207b0c021ae7df98`.
`tvm_ffi.load_module` also loaded it. The recipe's first attempt exposed
the pip-layout missing `lib64/libcudart.so` linker name; the symlinks above
are therefore required when using these pip packages as `CUDA_HOME`.

The isolated GPU probe used the exact artifact above with BF16 Q, 16 NVFP4
pages and 8 query rows against CuTe Q8KV4. Mixed scales passed all
`65,536/65,536` values at `atol=rtol=0.02` (max absolute error `0.01953125`);
extreme scales failed `55,974/65,536` (max absolute error `2256`). Its process
mapped **two** CUDA runtime libraries: the private 13.4.92
`libcudart.so.13` (`cudaRuntimeGetVersion=13040`) and Torch's bundled 13.0
library (`13000`). This is a mixed-runtime integration that needs target-image
qualification, not proof that the whole environment can be upgraded safely.

### Build a matching FlashInfer Python wheel

The measured FI Python package is `0.6.12+ee2b8834`. A newer local FI checkout
is at `250aae19`; it contains subsequent JIT-cache CI fixes, but **was not**
the package used for the A/B numbers here. To qualify an FI upgrade, choose a
single reviewed FI commit, build its Python wheel in an isolated checkout, and
pair its JIT-cache wheel with that same source/public version. For example:

```bash
FI_REPO=/data0/ruixuan.zrx/minimax-m31-dev/1009_flashinfer_dependency_audit
FI_SHA=250aae19fe282d2bcf971dc0f9523062addee949
FI_SRC=/data0/ruixuan.zrx/minimax-m31-dev/fi-134-source-candidate
git -C "$FI_REPO" worktree add --detach "$FI_SRC" "$FI_SHA"
cd "$FI_SRC"
git submodule update --init 3rdparty/cccl 3rdparty/cutlass 3rdparty/spdlog
export CUDA_HOME="$CUDA134"
export FLASHINFER_NVCC="$CUDA134/bin/nvcc"
export FLASHINFER_CUDA_ARCH_LIST='10.0a 10.3a'
export FLASHINFER_LOCAL_VERSION="$(git rev-parse --short=8 HEAD)"
python3 -m pip install 'setuptools>=77,<82' packaging wheel build \
  'apache-tvm-ffi>=0.1.6,!=0.1.8,!=0.1.8.post0,<0.2'
python3 -m build --wheel --no-isolation --outdir "$FI_SRC/dist"
python3 -m pip install --no-deps --target "$FI_SRC/install" \
  "$FI_SRC"/dist/flashinfer_python-*.whl
```

The recipe has now been exercised in an isolated worktree at
`.../minimax-m31-dev/fi_134_source_candidate_1010`, using Python 3.10,
setuptools 81.0.0 and FI source SHA `250aae19fe282d2bcf971dc0f9523062addee949`.
The resulting **Python/source wheel** is
`.../1010_topk_q8_model_candidate/fi134_wheel_dist/flashinfer_python-0.6.12+250aae19-py3-none-any.whl`
(SHA256 `b343932f07a91a32a0b6c97cb5fdde38c8f5f2c3a220dfda99c81de770c75d49`,
about 14 MiB). It contains `msa_ops/_nvfp4_prefill_sm100.py`,
`trace/templates/activation.py`, and `data/csrc/msa_prefill_nvfp4_specialized.cu`.
Private `pip install --no-deps --target .../fi134_wheel_install` and cold import
reported version `0.6.12+250aae19` and the expected 4-KV-head page layout.
The wheel is **not** a precompiled 13.4 cubin/JIT-cache wheel and does not by
itself constitute a complete 13.4 FI release.

The host's existing nvcc 13.4.92 and the task-private 13.4 runtime/headers
were combined only inside that test directory. The first GPU JIT attempt
exposed a packaging trap: the private runtime headers lacked `include/crt`,
which nvcc took from its own installation. The generated Ninja depfile named
`$CUDA_HOME/include/crt/*.h`, so every new process rebuilt the Q8 module
despite a valid `.so`. After linking the **same-version nvcc CRT headers** into
that location, `ninja -d explain -n` printed `no work to do` for both FI BF16
MSA and RTP Q8 MSA. The repository GPU contract test
`FlashInferNVFP4PrefillTest.test_rtp_writer_scales_against_fp32_reference`
passed both before and after, then took 2.054 s warm versus about 17 s when
the missing CRT forced JIT rebuild; the two `.so` mtimes remained unchanged.
Both generated recipes selected the private 13.4 nvcc/headers and `sm_103a`,
and both `.so` files resolved the private 13.4 `libcudart.so.13` under the
test library path. These are isolated operator gates, not a full-model or
long-running mixed-runtime signoff.

A later single-operator rerun showed why the toolchain exports above must be
set for **every** process: with the private FI wheel but no `CUDA_HOME` or
`FLASHINFER_NVCC`, FI rewrote the warm Q8 Ninja recipe to use container
`/usr/local/cuda/bin/nvcc` (13.2) and failed at the kernel's explicit 13.4
guard before launching the GPU kernel. Repeating with the complete private
13.4 environment compiled and ran successfully; `ninja -n -d explain` then
again reported `no work to do`. A warm cache does not protect against a
different compiler selection on a later process launch.

A fresh-workspace, two-process GPU check used the private FI Python wheel,
installed candidate RTP package, and complete 13.4 exports above, with
`FLASHINFER_WORKSPACE_BASE=.../1010_topk_q8_model_candidate/fi134_cold_replay_1010`.
It invoked the source test file directly (the installed RTP wheel omits tests):

```bash
python /data0/ruixuan.zrx/rtp_llm_m31_topk_q8_worktree/rtp_llm/models_py/triton_kernels/sparse_msa/test/test_flashinfer_nvfp4_prefill.py \
  FlashInferNVFP4PrefillTest.test_rtp_writer_scales_against_fp32_reference
```

Both processes passed. The first took 18.836 s of **whole-test wall time** and
created BF16 MSA and Q8KV4 `sm_103a` `.so` files; the second took 2.168 s.
Their mtimes remained `22:23:14.495` and `22:23:24.187`, respectively, after
the second process. Both generated recipes selected the private 13.4 nvcc and
headers, and both `ninja -n -d explain` checks said `no work to do`.
With the private library path, `ldd` resolved Q8's `libcudart.so.13` to the
private 13.4 runtime and `libcuda.so.1` to the host driver. This tests JIT
reuse on one Blackwell GPU in an isolated container, not TopK bucket coverage,
full-model startup, or the final production image.

Execute the build in a disposable container with pinned Torch/CuTe/TVM-FFI
dependencies from the target image; inspect the resulting wheel's METADATA,
packaged source, SHA256 and cold import.
Do not use this repository's `build_wheel.sh` on a shared checkout: it deletes
build/`.so` files and uninstalls FI packages. The split cache CI currently
defaults to a `cu132` build; changing the compiler alone does not generate a
13.4 cache wheel. The custom RTP Q8 JIT spec is not part of FI's standard
952-module inventory, so either add/package that spec explicitly and validate
it in the target image, or ship the complete private 13.4 toolchain and permit
runtime JIT. Do not set `FLASHINFER_DISABLE_VERSION_CHECK=1` to hide a wheel
pairing error.

### Full-model private FI 13.4 cold/hot smoke

An isolated candidate PD4+4 was launched with the private FI Python wheel
`0.6.12+250aae19`, private CUDA 13.4 `CUDA_HOME`, `nvcc`, headers, runtime,
and the `CPATH` export above. The first startup **without** `CPATH` failed
before any request: all Prefill ranks reached DeepGEMM MXFP8 scale packing,
whose `sm_103f` JIT compile reported `cuda_runtime.h` missing. This is a real
upgrade integration failure, not a Q8 MSA runtime result. With `CPATH`, the
Prefill service became healthy and the frozen first four MATH500 IDs completed
4/4, zero request errors and valid per-DP native MAL (`3.14` aggregate). A
second fresh P4/D4 process reused exactly the first Prefill FI JIT workspace
and again completed 4/4; all four response texts matched the first process
and the earlier installed-FI Q8 diagnostic run.

The first process's request window was 67.312 s, with individual request
latencies 51.0-67.3 s; the second process's window was 5.807 s, with
individual latencies 3.63-5.80 s. This is a cold-versus-warm **whole-request**
observation, not a steady kernel-speed comparison. The cold cache created
`rtp_m31_prefill_q8kv4_sm103a.so` and `rope.so`; the Q8 `.so` mtime remained
unchanged after the second process and `ninja -d explain -n` reported `no work
to do`. Its generated Ninja recipe selected the private 13.4 `nvcc` and
headers. With the private runtime path, `ldd` resolved Q8's `libcudart.so.13`
to 13.4 and `libcuda.so.1` to the host driver. The live Prefill rank mapped
both private 13.4 `libcudart.so.13` and Torch's packaged CUDA runtime: mixed
runtime compatibility beyond this smoke remains unqualified. The original
resident PD4+4 was restored afterward and passed five idle endpoints plus
four P-to-D smoke RPCs.

The host-shared DeepGEMM `.dj/cache` showed no newly modified `kernel.cubin`
in the hot process window (26 existing `.committed` markers refreshed),
consistent with reuse but not an isolated-image cache proof. The private
launcher now resolves `--prefill-jit-workspace` to an absolute path so a
relative argument cannot silently create per-attempt caches; the recorded
cold/hot test already used the same explicit absolute path.

Artifacts: `.../dataset_quality/q8kv4_fi134_model_smoke_1011/` (expected
negative startup gate), `.../q8kv4_fi134_model_cpath_1011/` (cold success),
`.../q8kv4_fi134_model_hot_1011/` (same-cache process success), and
`.../1010_topk_q8_model_candidate/restore_health_and_smoke.json`. This closes
the narrow full-model `sm_103a` cold/hot smoke gap, **not** final-image AOT,
long-running stability, all TopK/Cake buckets, or a full 13.4 toolchain
promotion.

### Prewarm before serving traffic

With the exact RTP/FI installations and CUDA environment above, run this as
a **blocking readiness step** on one GPU of each deployment architecture,
before exposing Prefill to traffic. The service must use the same writable
`FLASHINFER_WORKSPACE_BASE`, compiler paths, and package sources. A cache
compiled for `sm_103a` does not qualify `sm_100a`.

```bash
RTP_INSTALL=/data0/ruixuan.zrx/minimax-m31-dev/1010_topk_q8_model_candidate/install
FI_INSTALL=/data0/ruixuan.zrx/minimax-m31-dev/1010_topk_q8_model_candidate/fi134_wheel_install
export PYTHONPATH="$RTP_INSTALL:$FI_INSTALL${PYTHONPATH:+:$PYTHONPATH}"
export FLASHINFER_WORKSPACE_BASE=/path/to/shared-writable-fi-cache
# Inherit the service's exact CUDA_HOME, CC, CXX, CPATH, FLASHINFER_NVCC,
# FLASHINFER_EXTRA_CUDAFLAGS, and LD_LIBRARY_PATH here.
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
from flashinfer.rope import get_rope_module
from flashinfer.msa_ops import msa_prefill_nvfp4_specialized_warmup
from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.flashinfer_nvfp4_prefill import (
    _q8kv4_module, _q8kv4_target,
)

msa_prefill_nvfp4_specialized_warmup(torch.device("cuda:0"))
_q8kv4_module(_q8kv4_target(torch.device("cuda:0")))
get_rope_module()
PY
```

On this host, an empty isolated `sm_103a` workspace took 7.852 s for BF16
MSA, 7.286 s for Q8 MSA, and 23.664 s for RoPE. Independent second
processes with identical environment took 0.492 s, 0.009 s, and 0.006 s
respectively; all three `.so` files remained in the same cache and their
Ninja dry runs reported `no work to do`. Artifact:
`.../1010_topk_q8_model_candidate/fi134_single_gpu_prewarm_1011/`.
Fail readiness if any module build/load fails, the expected `.so` is
absent, or a same-environment second process rebuilds it. Do not set
`FLASHINFER_DISABLE_JIT=1` as a shortcut: the custom RTP Q8 JitSpec still
requires JIT to be permitted. This prewarms only FI BF16 MSA, Q8 MSA, and
RoPE; it does not cover Triton TopK fallback, Cake/DSpark shape buckets, or
DeepGEMM's startup-time kernels. The final image must repeat this gate and
then verify a real post-readiness request has no compile event.

The stronger **model-environment** gate is mandatory: use the exact Prefill
`launch.json` compiler variables and package paths for the blocking prewarm,
record the three module directories' `.ninja_log` hashes and `.so` mtimes,
start a fresh model process, issue one representative post-readiness request,
then check that neither hashes nor mtimes changed. Also run
`ninja -d explain -n` in each module directory; all must say `no work to do`.
A standalone second-process cache hit is insufficient. FI's
`flashinfer/jit/env.py` names the workspace by FI version and CUDA arch,
not compiler or flags; `jit/cpp_ext.py` regenerates Ninja commands from
`CUDA_HOME`, `CXX`, `FLASHINFER_NVCC`, and extra flags. The workspace can
therefore look warm while a service's different build recipe forces a
recompile. Record the build environment alongside the cache artifact.

## Current A/B evidence

The two arms used the same candidate wheel, warp TopK setting, normal rejection
sampling, original D4 package/config, and one warmup plus three formal
fixed-shape Prefill rounds. Times below are per-request Prefill latency from
the frozen metrics JSON; they are not a TopK-only effect.

| CP4 shape | BF16 MSA | Q8KV4 MSA | Q8 change | Success |
|---|---:|---:|---:|---:|
| BS20, 74,011 input, 66,560 reuse | 1401.259 ms | 1386.017 ms | -1.09% | 60/60 each |
| BS60, 80,000 input, 75,776 reuse | 2527.288 ms | 2449.438 ms | -3.08% | 180/180 each |

Artifacts: `.../1010_topk_q8_model_candidate/bs20_{bf16,q8kv4}_cuda134/metrics.json`
and corresponding `bs60_*` files. This is one matched campaign per shape,
not repeat-run variance or a production throughput result. The metrics pool
60/180 per-request observations from only **three** BS20/BS60 batches per arm;
batchmates are correlated and cannot be treated as 60/180 independent trials.
Reported per-request prefill variance is 251.83/2334.59 ms² for BS20
BF16/Q8 and 1130.93/500.28 ms² for BS60 BF16/Q8. The BS20 Q8 tail is wider
than BF16, so its observed 15.24 ms mean saving especially needs another
matched batch-level campaign before a stable-gain claim.

A second fixed-shape campaign used the same installed candidate and settings,
one warmup plus **six** formal rounds per arm. Each number is the mean
per-request Prefill latency; the batch, not the request, is the independent
measurement unit. Run order was BS20 BF16/Q8 followed by BS60 Q8/BF16.

| CP4 shape | BF16 MSA | Q8KV4 MSA | Q8 change | Success per arm |
|---|---:|---:|---:|---:|
| BS20, 74,011 input, 66,560 reuse | 1400.227 ms | 1355.185 ms | -3.22% | 120/120 |
| BS60, 80,000 input, 75,776 reuse | 2508.276 ms | 2443.232 ms | -2.59% | 360/360 |

Artifacts: `.../1010_topk_q8_model_candidate/bs{20,60}_{bf16,q8kv4}_repeat6_1010/metrics.json`.
This repeat supports a modest fixed-shape Prefill gain with warp TopK held
constant; it does not isolate TopK, measure production throughput, qualify a
full CUDA 13.4 runtime upgrade, or establish quality non-inferiority. After
the campaign, the original PD4+4 container was restarted; five endpoints
reported alive/idle, and four P-to-D lane requests completed eight tokens
each before returning to idle. See `.../1010_topk_q8_model_candidate/restore_health_and_smoke.json`.

Same 160 LongBench IDs, complete PD4+4 runs: BF16 score `66.4320`, Q8 score
`67.5942`, 160/160 success in each arm. Native Decode MAL was `3.2004`
(`1581/494`) versus `3.3287` (`1661/499`). Twenty-five IDs changed score:
13 favor Q8 and 12 favor BF16; do not call the one-run `+1.16` a stable quality
improvement. Four Q8 and five BF16 samples reached the length cap in the
respective untruncated-score accounting. The dataset's longest prompt is about
22k tokens, so it does not exercise the 512-block warp TopK dispatch threshold.
Evidence: `.../dataset_quality/{bf16_retry4,q8kv4_retry3}/longbench160/`
`scores.summary.json` and `integrity.json`.

The frozen manifest SHA, 160-ID set, greedy sampling, 512-token cap,
four-DP/two-workers-per-DP lane contract and disabled reuse cache match.
Paired per-ID scores favor Q8 on 13 IDs, BF16 on 12, and tie on 135. This is
one run per arm, not a statistical quality superiority claim. The paired
mean difference is `+1.162` score points; a 10,000-resample item bootstrap
(fixed seed, resampling the 160 paired score differences) gives a 95% interval
of approximately `[-2.39, +4.86]` points. This interval describes variation
over sampled **items**, not repeated-run stochasticity or the exact score of
this frozen 160-item set. It includes a small regression, so the one run does
not prove quality non-inferiority. Native MAL's `1581/494` and `1661/499`
are separately valid run-window counts, but four DP ranks and one run do not
support a stable MAL-gain claim. The request
timings **do not form a fair performance A/B**: BF16's first eight requests
(two per DP) had first-token times about 55-56 s and total times 55-57 s,
whereas Q8 had no request above 2.01 s. BF16's first request window began
near 18:25:22; its FI BF16 MSA `.so` was created at 18:25:34 and FI RoPE
`.so` at 18:26:18. The Q8 arm's corresponding `.so` files had already been
created at 18:18:29 and 18:19:07 before its request window near 18:20:17.
This is direct first-request cold-JIT contamination in the BF16 arm; do not
attribute the LongBench latency difference to Q8 compute. The scored outputs
and isolated native MAL counters remain valid observations under their own
request contract, but that first MAL difference alone had no repeated-run
stability bound.

A second **same-ID PD4+4 quality/MAL pair** ran in reverse arm order, Q8KV4
then BF16. Each arm used a fresh isolated P4/D4 process and JIT cache, ran
20 warmup LongBench requests before the 160 formal IDs, and used the same
manifest/source SHA, greedy sampling, 512-token cap, normal rejection, four
DP ranks at two workers per rank, and disabled Prefill reuse. All four formal
campaigns have the same SHA256 of `jq -c '.selected_ids' contract.json`
(`04f03539b1191b33ecdea4f9343e0b8e8e8713e6f9451b8d0036fa30b45e833e`); each
completed 160/160 requests with strict per-PID native MAL validation.
Comparing the actual new launch records, Prefill environment keys differ
only in `M3_NVFP4_PREFILL_COMPUTE` and the arm-private
`FLASHINFER_WORKSPACE_BASE`; Decode differs only in its private JIT path.

| LongBench160 run | BF16 raw score | Q8KV4 raw score | Q8 minus BF16 | BF16 native MAL | Q8KV4 native MAL |
|---|---:|---:|---:|---:|---:|
| First pair | 66.4320 | 67.5942 | +1.1621 | 1581/494 = 3.2004 | 1661/499 = 3.3287 |
| Warmed repeat | 66.3581 | 66.3955 | +0.0374 | 1736/549 = 3.1621 | 1767/527 = 3.3529 |

Repeat artifacts:
`.../dataset_quality/{bf16,q8kv4}_repeat2_1010/longbench160/`
`{contract,integrity,scores.summary,longbench_scores}.json`. Both arms had
160/160 success, valid native metric windows across all four Decode ranks,
and 4/5 Q8/BF16 length-capped samples, respectively. In the warmed repeat,
paired per-ID scores favor Q8 on 15, BF16 on 14, and tie on 131; 121/160
response hashes match across arms. A 10,000-resample **item** bootstrap of
that repeat's 160 paired score differences gives a 95% interval of roughly
`[-3.54, +3.63]` points. It is not a repeated-run confidence interval.
Within-arm response hashes matched across runs on 151/160 BF16 and 154/160
Q8 IDs; two Q8 TREC scores flipped from 1 to 0, contributing most of Q8's
`-1.199`-point run-to-run change. BF16 moved `-0.074` points. Thus both
paired MAL observations favor Q8, but neither MAL nor LongBench score has a
credible stability/non-inferiority bound from only two process-level runs.
The second pair was warmed before formal requests, but its request latency
is still not a matched Prefill performance benchmark.

The upstream TopK patch is already in candidate source and installed wheel:
`warp_topk.py` calls native `rtp_llm_ops.minimax_prefill_topk` for Hkv4,
TopK16, page128, 512-1024 KV blocks, and at least 1024 score rows, with
legacy bitonic handling ties. A CPU import from the installed wheel confirmed
the native binding exists. Both A/B launch records set
`M3_PREFILL_TOPK_IMPL=warp`. The earlier dedicated TopK campaign measured
three-chunk operator time `1193.510 -> 689.796 us`, pure Prefill model-forward
`1634.49 -> 1609.97 ms` (1.50%), exact ordered IDs on 35,788,800 score rows,
and 140/140 paired first tokens. Those measurements used a different frozen
model campaign and must not be added to the Q8 A/B savings. The candidate's
BS20/74k and BS60/80k rank-0 model dispatches are confirmed below; the other
ranks are not yet profiled.

A new installed-wheel GPU dispatch probe used H4/Q1024/N579/K16, matching
the supported production geometry. The full `_launch_topk_to_block_table`
wrapper called the native op once and produced 4,096/4,096 ordered-ID rows
matching the no-tie reference. With a cutoff tie, the native op again fired
once and marked all 4,096 rows for the Triton fallback; the **full wrapper**
then matched the historical bitonic output on all 4,096 rows. A naive
mathematical reference differed on 1,536 rows because its tie ID choice is
not the legacy bitonic contract; this is not a candidate regression. The
candidate source and installed package hashes match for both `warp_topk.py`
and `topk_bt_fused.py`. This establishes package/dispatch functionality, not
model-run branch-hit attribution or an additional E2E speedup measurement by
itself.

A subsequent **model-run profile** closes the branch-hit gap for one target
shape: `.../1010_topk_q8_model_candidate/bs20_q8kv4_profile_topk_1010/`
`timelines/prefill_wr0_1.json` has rank 0/world size 4 provenance and 3,728
GPU kernel events. It contains 180 launches of native
`minimax_prefill_warp_topk_kernel<640,16,false>` and 180 launches of
`_topk_to_block_table_tie_fallback_kernel`, along with 60 sparse-prefill
kernel launches. The native TopK launch count proves actual model dispatch at
BS20/74,011 with Q8KV4; the fallback count shows the tie-preserving path was
also invoked. This one-rank profiler run is **not** a matched performance arm,
does not prove the same count on every rank/shape, and must not be compared
numerically with non-profiled model latency. The original PD4+4 service was
restored again afterward; four lane RPCs completed and all five endpoints
returned alive/idle.

The corresponding BS60/80,000 Q8KV4 profile at
`.../1010_topk_q8_model_candidate/bs60_q8kv4_profile_topk_1010/`
`timelines/prefill_wr0_1.json` contains 4,874 GPU kernel events with rank
0/world size 4 provenance: **240 native warp TopK**, 240 tie-fallback, and
60 sparse-prefill launches. Its separate non-profiled measurement round had
60/60 successful requests with exact 80,000 input and 75,776 reuse; the
profiled latency is not a comparable A/B datapoint. Across this fresh model
process, the existing BF16-MSA, Q8KV4-MSA and RoPE FI `.so` cache entries
kept exactly the same file list, sizes and modification times. This rules out
recompilation of those three warmed modules for this shape, not cold-start
JIT or new buckets. The original PD4+4 service was restored after this run;
all four lane RPCs completed eight tokens and all five endpoints returned
alive/idle.

### TopK-only fixed-shape model A/B

The private `run_prefill_fixed.py` runner now accepts `--topk warp|bitonic`
(default `warp`). Its only model-path switch is
`M3_PREFILL_TOPK_IMPL`: `bitonic` makes `try_warp_prefill_topk` return `None`
and executes the established Triton bitonic path. Q8KV4 attention, candidate
RTP install, checkpoint, CP4 topology, sampling, reuse, and six formal batches
per arm were held fixed. Run order was BS20 bitonic/warp, then BS60
warp/bitonic. Each arm also had one excluded warmup batch.

| CP4 input/reuse | Bitonic TopK Prefill | Warp TopK Prefill | Warp saving | Formal success |
|---|---:|---:|---:|---:|
| BS20, 74,011 / 66,560 | 1,384.070 ms | 1,353.596 ms | 30.474 ms (2.20%) | 120/120 each |
| BS60, 80,000 / 75,776 | 2,491.683 ms | 2,448.346 ms | 43.337 ms (1.74%) | 360/360 each |

Artifacts: `.../1010_topk_q8_model_candidate/bs{20,60}_q8kv4`
`{_topk_bitonic,}_topkab_1010_r1/{metrics.json,responses.jsonl}`.
The sorted input-SHA256 multiset matched exactly between TopK arms for each
shape (BS20 digest `848bcf92...9c3f0`; BS60 `18cb6101...5d1ed`). The
reported latencies are full Prefill request means, not TopK kernel timings;
requests within each batch are correlated, so 120/360 are not independent
replicates. This single order-reversed six-batch campaign shows a modest
fixed-shape model benefit, not a broad throughput or statistical gain bound.
The branch switch is established by the recorded `docker exec` CLI arguments,
the container's initial environment (no `M31_DIRECT_PREFILL_READY` override),
and the runner's re-exec assignment; this campaign did not capture per-arm
TopK kernel counts. The harness's `test_info.json` has stale generic fields
(`num_measures=1`, reuse rate 0) and is not used for the A/B contract.
After review, the runner gained a pre-launch effective-value check and writes
`effective_config.json` for subsequent campaigns; those additions are not
retroactive evidence for these four runs.

These fixed-perf requests produce short sampled outputs: same-input text
matched on 65/147 BS20 and 209/427 BS60 responses across bitonic/warp,
versus 72/147 and 212/427 when comparing this warp arm with an earlier warp
arm under the same input hashes. Thus text equality here cannot establish
TopK precision or a quality delta. The separate native TopK tie-preserving
4,096-row exact check above remains the operator-level contract.

### Full GSM8K1314, 32 concurrent requests

The original GSM8K test file has 1,319 rows; this campaign uses IDs 0..1313
to match the historical full-1314 convention. The private
`prepare_gsm8k1314_manifest.py` builds and shuffles a frozen manifest (SHA256
`29296ca57d4afe241aeabe5e0393b0fe5f839e2a1b97977936289b1b892f9d01`).
Its first 200 IDs were checked exactly against the prior formal GSM8K200
prompts, answers, and 512-token caps. All four runs used the same 1,314 IDs
and order, 32 clients (eight per Decode DP), greedy requests, normal rejection,
disabled cache reuse, the same installed source and D4 package, and separate
private JIT caches. Each run completed a separate 32-request warmup before the
formal window. The Prefill launch environments matched after removing only
`M3_NVFP4_PREFILL_COMPUTE` and the JIT workspace path; Decode environments
matched after removing only the JIT workspace path.

| Formal arm, legacy exact-string scorer | Correct / scored | Accuracy | Native MAL | Completion | Peak active clients / DP |
|---|---:|---:|---:|---:|---|
| BF16 1 | 1,142 / 1,314 | 86.9102% | 143,443 / 34,621 = 4.1432 | 1,314 / 1,314 | 8, 8, 8, 8 |
| Q8KV4 1 | 1,116 / 1,314 | 84.9315% | 147,128 / 35,592 = 4.1337 | 1,314 / 1,314 | 8, 8, 8, 8 |
| Q8KV4 2 | 1,120 / 1,314 | 85.2359% | 146,856 / 35,552 = 4.1307 | 1,314 / 1,314 | 8, 8, 8, 8 |
| BF16 2 | 1,138 / 1,314 | 86.6058% | 144,468 / 34,856 = 4.1447 | 1,314 / 1,314 | 8, 8, 8, 8 |

The second pair reversed service order (Q8 then BF16), with the same 32-item
warmup and frozen manifest. All four strict native MAL windows were valid,
without request or scorer errors; the launch environments match after removing
only the compute-mode flag and private JIT workspace. **The accuracy column
above is not valid numeric-answer accuracy.** The inherited GSM8K scorer
compares extracted strings literally and recognizes only a single-brace-level
`\boxed{...}`. It misclassifies `826.00` against gold `826`, and parses
`\boxed{6{,}000 \text{ gallons}}` as `000` instead of `6000`. Thus the
apparent Q8 deficits of 26 and 18 are dominated by scorer-contract artifacts
and cannot be interpreted as a confirmed model-quality regression.

A sidecar rescored all four **unchanged saved responses** with the existing
MATH500 `math_verify.parse+verify` vendor. Each row retains both decisions and
the parsed prediction; raw original scores remain untouched. Synthetic
`826.00`, nested `6{,}000`, and truly wrong `113` cases passed 3/3. All 1,314
IDs and response/manifest hashes were checked; duplicate rows are rejected.
`finish_reason=length` is counted wrong in the full-set score, even if a
truncated reasoning trace contains the gold number (one, zero, three, and one
such cases across the four arms). Flip-up audits found 67-78 pure
numeric-format equivalents and 23-36 nested-thousands answers per arm; the
remaining completed flip-ups were checked as boxed time/unit answers. The
flip-downs were inspected separately. The untruncated comparison also reports
its eligible denominator explicitly.

| Formal arm | Completed math-equivalent / 1,314 | Accuracy, length wrong | Untruncated math-equivalent / eligible |
|---|---:|---:|---:|
| BF16 1 | 1,235 | 93.99% | 1,235 / 1,307 |
| Q8KV4 1 | 1,227 | 93.38% | 1,227 / 1,302 |
| Q8KV4 2 | 1,233 | 93.84% | 1,233 / 1,297 |
| BF16 2 | 1,231 | 93.68% | 1,231 / 1,303 |

On **common non-truncated IDs** within each pair, pair 1 was BF16 1,230 vs
Q8 1,226 of 1,300, whereas pair 2 was BF16 1,224 vs Q8 1,231 of 1,293.
The direction reverses. On the full set with length counted wrong, Q8 is
eight answers lower in pair 1 and two higher in pair 2.
This removes the claimed repeatable GSM8K loss under a numeric answer contract,
but two process-level pairs still do not prove a quality non-inferiority bound.
Keep Q8 opt-in and BF16 default while the independent extreme-scale oracle and
broader dataset gates remain open.

The first pair had only 69/1,314 byte-identical full response texts, so the
generation trajectory is sensitive even with greedy request settings. Short
GSM8K prompts do not exercise the long-context warp TopK dispatch; this is a
Q8-versus-BF16 model-quality check, not a TopK quality check. Native MAL was
4.1432/4.1447 for BF16 and 4.1337/4.1307 for Q8, a small, stable direction
but not a release-level gain claim. Formal request windows were 51.5/40.3 s
for BF16 and 45.0/44.7 s for Q8: the throughput ordering reversed on repeat,
so this campaign does not prove a steady throughput gain.
Each Q8 arm used a fresh FI cache. Its Q8 MSA module `.ninja_log` has a header
plus one object and one shared-library build record (three lines total);
the `.so` timestamps were 79 and 94 seconds before the corresponding formal
request windows. This excludes a repeated rebuild of that FI module during
these formal windows, but says nothing conclusive about every Triton bucket or
the final production image.
Evidence: `.../dataset_quality/{bf16_gsm8k1314_bf16_1011,`
`q8kv4_gsm8k1314_q8kv4_1011}/gsm8k1314/`
and `.../dataset_quality/{q8kv4_gsm8k1314_q8_repeat2_1011,`
`bf16_gsm8k1314_bf_repeat2_1011}/gsm8k1314/`, each with
`{contract,integrity,scores.summary,scores}.json*`, `responses.jsonl`,
`gsm8k_numeric_scores.jsonl`, `gsm8k_numeric_scores.summary.json`, and a
sibling `gsm8k_warm32/` directory. After the repeat, the original PD4+4 was
restored and verified with five idle endpoints and four P-to-D lane RPCs;
see `.../1010_topk_q8_model_candidate/restore_health_and_smoke.json`.

### GSM8K1314 high-concurrency repeat (2026-10-11)

A fresh BF16-then-Q8KV4 PD4+4 pair used the same 1,314-ID manifest SHA256
`29296ca57d4afe241aeabe5e0393b0fe5f839e2a1b97977936289b1b892f9d01`,
greedy sampling, 512-token cap, no cache reuse, 32 separate warmup requests,
and eight clients per Decode DP (32 total) for the formal window. Both formal
runs completed 1,314/1,314 requests with zero request errors and strictly valid
native MAL on all four Decode ranks. The historical numeric-string GSM8K scorer
is not the quality result; unchanged responses were rescored with the same
`math_verify.parse+verify` vendor as MATH500, counting length truncation wrong.

| Arm | Math-equivalent / 1,314 | Request window | Native MAL | Length stops |
|---|---:|---:|---:|---:|
| BF16 | 1,227 (93.38%) | 39.064 s | 4.15969 | 11 |
| Q8KV4 | 1,228 (93.46%) | 37.126 s | 4.15575 | 14 |

The one-answer Q8 advantage comprises 31 BF16-only and 32 Q8-only correct
IDs; the 1.94 s shorter request window is also a single-pair observation.
Neither proves quality or throughput non-inferiority. These short
prompts do not exercise the native long-context TopK dispatch. A preceding
BF16 trial with 16 clients per DP (64 total) **failed**: 629/1,314 request
errors, including HTTP 500 and connection resets. Errors were concentrated on
DP1 (315/329) and DP3 (314/328), while DP0/DP2 had none; its legacy score and
20.245 s request window are invalid as quality/performance evidence. The
Prefill launch has `concurrency_limit=60`, but the rank asymmetry is not
explained by that limit alone; the available logs do not establish a unique
root cause. Do not claim 64-concurrency stability or attribute this BF16-arm
failure to Q8KV4. The original resident PD4+4 was restored afterward; the
five idle endpoints and four P-to-D lane smoke RPCs passed.

Artifacts: `.../dataset_quality/bf16_gsm8k64_bf_1011/gsm8k1314_retry32/`,
`.../dataset_quality/q8kv4_gsm8k32_q8_1011/gsm8k1314/`, the rejected 64-client
trial in `.../dataset_quality/bf16_gsm8k64_bf_1011/gsm8k1314/`, and
`.../1010_topk_q8_model_candidate/restore_health_and_smoke.json`.

### LongBench160 repeat: TopK dispatch was not exercised

Two fresh isolated P4/D4 services used the same candidate RTP install and
fixed D4 package. Only Prefill `M3_PREFILL_TOPK_IMPL` changed (`bitonic` then
`warp`); the per-arm `prefill/launch.json` records the requested value. **This
was not an executed TopK A/B:** the installed `try_warp_prefill_topk` requires
`512 <= num_blocks <= 1024` with 128-token blocks (at least 65,536 tokens),
whereas the longest formal prompt was 22,006 tokens. Both arms therefore took
the legacy TopK path. Kernel traces were not captured for these runs. After
removing the per-arm `FLASHINFER_WORKSPACE_BASE` path, the D4 environments
matched exactly; the P4 environments matched after also removing the TopK
flag. Both arms used the identical LongBench160 manifest SHA256
`7265b462...e1a8fd`, 20 warmup requests, then 160 formal IDs with greedy
request sampling, normal rejection, a 512-token cap, no Prefill cache reuse,
and two clients per Decode DP. The selected IDs, scorer source hashes, and
normalized request arguments matched. Each arm completed 160/160 formal
requests, with no errors and a strictly valid native MAL window on all four
Decode ranks.

| Requested Prefill TopK | LongBench raw score | Untruncated score | Native MAL | Accepted / streams | Completion |
|---|---:|---:|---:|---:|---:|
| Bitonic | 66.9817 | 68.2151 | 3.3139 | 1,763 / 532 | 160/160 |
| Warp | 66.3517 | 67.5690 | 3.3411 | 1,724 / 516 | 160/160 |

The second run minus the first is -0.6299 raw-score points and +0.0272 MAL
(+0.82%); these differences are **run-to-run variation on the same TopK
path**, not evidence of a warp TopK precision effect.
Text SHA256 matched on 157/160 same-ID outputs; per-ID LongBench score matched
on 158/160. The score difference comes from one TREC item (1 to 0) and one
Qasper item (0.10989 to 0.10204); a third changed output kept the same score.
An earlier warp-requested repeat matched text on only 155/160 IDs, consistent
with this variability. A TopK quality gate still needs matched requests above
the dispatch threshold plus per-arm native-kernel invocation evidence; this
dataset run establishes neither a TopK precision effect nor a statistical
quality non-inferiority bound. The printed generic `by_dataset.scored=0` is not the
LongBench quality score; use the separate `scores.summary.json` and
`longbench_scores.json` artifacts.

Artifacts: `.../1010_topk_q8_model_candidate/dataset_quality/`
`q8kv4_topk_quality_{bitonic,warp}_1010/{prefill,decode}/launch.json`,
`longbench160/{contract,integrity,scores.summary,longbench_scores}.json`,
and `longbench160/responses.jsonl`.
Both candidate P4/D4 services exited 0. The original PD4+4 service was then
restored after preserving its old owner/log files under
`*.pre_topk_quality_1010.*`; five workers were alive/idle and four real P-to-D
lane RPCs each generated eight tokens. The restoration artifact is
`.../1010_topk_q8_model_candidate/restore_health_and_smoke.json`.

Source-identity caveat: each P4 `launch.json` prepended the candidate `install`
to `PYTHONPATH` and its `rtp_llm/libs` to `LD_LIBRARY_PATH`; P4 console paths
also show Python imports from that install. At the short-input campaign,
installed `msa_attention.py`, `flashinfer_nvfp4_prefill.py`, and `warp_topk.py`
hash-matched the worktree (`321b15c3...`, `d17bac66...`, `6472920f...`).
For the later long-context campaign, `warp_topk.py` in both the worktree and
private install was `7c71a7e2...` after adding opt-in dispatch tracing.
However,
`longbench160/launch_record.json` inherited some old `frozen_python_sha256`,
`candidate_delta`, and native-library hash fields from the source launch
template. Those fields are not runtime-identity evidence for this campaign;
no contemporaneous module-path/native-library map was archived. Final-image
validation must record loaded paths and hashes from live rank processes.

### Long-context TopK path and quality stress A/B

The official 160-item LongBench subset cannot exercise native TopK. An
untruncated local `gov_report` example reached about 50k tokenizer tokens,
still below the 65,536-token dispatch threshold. To make a matched path test
without relabeling it as an official benchmark, the private
`.../1010_topk_q8_model_candidate/prepare_topk_long_manifest.py` builds a
**derived stress set**: 40 evenly spaced scored LongBench rows, each preceded
by the same unrelated article text, with its original task, answer, and ID
preserved. The corrected derived manifest is
`.../1010_topk_q8_model_candidate/topk_longbench_stress40_v2_1010.jsonl`,
SHA256 `589787f986975007bab535682cdf34ba5df66d1411a65598a552b5d1e3e4912f`.
Its `.meta.json` sidecar records the source manifest, distractor, and four
tokenizer-file hashes. The original length bucket is preserved separately as
`topk_stress_original_bucket`; the derived bucket is `64-80k`. The first
generated `topk_longbench_stress40_1010.jsonl` is retained as a historical
pre-review artifact and must not be used for the A/B.
The exact tokenizer preflight gives 71,939-71,940 tokens per prompt; all 40
IDs are unique, all turns equal their prompts, all answers are present, and
the existing evaluator's `select_items(..., {"longbench": 40})` accepts them.
These are **not** official LongBench scores. Each served request reported
72,103-72,104 `usage.prompt_tokens`, above the 65,536-token gate; Prefill
`M3_PREFILL_TOPK_TRACE=1` logged `num_blocks=564` on all four ranks. Its
bitonic-arm logs reported `path=disabled`, while warp-arm logs reported
`path=native`, with each rank reaching at least count 3,000. This proves the
native operator entrypoint was actually called, not merely requested by an
environment flag; it is not a full per-request CUDA kernel trace.

The isolated P4/D4 runs held Q8KV4, the D4 package, manifest SHA, sampling
(`temperature=0`, `top_k=1`, `top_p=1`), normal rejection, no Prefill cache
reuse, a 512-token cap, and two clients per Decode DP fixed. Normalized P4
launch records matched after removing only the TopK flag and per-arm JIT
workspace; D4 records matched after removing the JIT workspace. All 40
same-ID serialized request messages matched. Each service first ran the same
four excluded smoke requests, then 40 formal requests. The four-item smoke
completions were 4/4, but their native MAL windows were **invalid** because
rank 0 received no request. Only the 40-item formal windows below are used.

| Arm (run order) | Derived raw score | Untruncated score | Native MAL | Accepted / streams | Stop / length | Formal |
|---|---:|---:|---:|---:|---:|---:|
| Bitonic 1 | 65.1153 | 67.6350 | 3.2211 | 612 / 190 | 38 / 2 | 40/40 |
| Warp 1 | 61.2797 | 64.7593 | 3.2769 | 639 / 195 | 37 / 3 | 40/40 |
| Bitonic 2 | 58.9266 | 62.2553 | 3.1791 | 639 / 201 | 37 / 3 | 40/40 |
| Warp 2 | 63.6382 | 67.1331 | 3.0896 | 621 / 201 | 37 / 3 | 40/40 |

All four formal runs had no request errors, all four Decode ranks active,
`integrity.complete=true`, `native_mal_strict_valid=true`, and candidate
Decode/Prefill exit code 0. The first cross-path comparison was -3.8356
raw-score points and +0.0559 MAL (warp minus bitonic), but the second was
+4.7116 points and -0.0896 MAL. Bitonic repeated against itself differed by
6.1888 raw-score points and matched text on only 26/40 IDs; warp against
itself matched 27/40. Cross-path text equality ranged from 24/40 to 27/40.
This 40-item, heavily padded stress set is useful for dispatch and completion
but too variable for a quality non-inferiority or regression claim. The
generic `by_dataset.scored=0` output is the MT-Bench path; the LongBench
scorer's actual results are in `scores.summary.json` and
`longbench_scores.json`. No simultaneous GPU peak-memory or clean
first-request/JIT latency comparison was captured in these quality runs.

An additional opt-in real-score diagnostic compared native warp TopK indices
against the historical multirow bitonic kernel on the same Prefill inputs.
With `M3_PREFILL_TOPK_SHADOW_LIMIT=200`, all four Prefill ranks reached 200
checks, for 800 exact index-tensor comparisons and zero mismatches. The four
long-context requests completed without request errors; their native MAL
window is deliberately **invalid** because Decode rank 0 received no request.
The diagnostic synchronizes and doubles TopK computation, so its latency is
not a performance result. This is sampled operator equivalence, not a model
quality non-inferiority result. Evidence: `.../dataset_quality/`
`q8kv4_topk_shadow_warp_stderr_1011/prefill/console.log` and
`longbench_stress4/{responses.jsonl,integrity.json}`.

Artifacts: `.../1010_topk_q8_model_candidate/dataset_quality/`
`q8kv4_topk_long_{bitonic,warp}_rdma_1010/` and
`q8kv4_topk_long_{bitonic,warp}_repeat_1011/`, each with
`longbench_stress40/{contract,integrity,scores.summary,longbench_scores}.json`,
`longbench_stress40/responses.jsonl`, `prefill/console.log`, and
`{prefill,decode}/launch.json`. An earlier startup attempt in
`m31_topk_q8_candidate_1010` exited before requests because that container
had no `/dev/infiniband`; it is preserved as
`q8kv4_topk_long_bitonic_1010/` and is **not** operator instability.
The successful arms used `m31_topk_q8_quality_rdma_memlock_1010`, whose RDMA
device is mounted. The original PD4+4 container was restored after each
campaign; final `restore_health_and_smoke.json` records five alive/idle
workers and four real P-to-D lane RPCs, eight output tokens each.

Each arm had its own FlashInfer JIT workspace. The final modification times
of its Prefill Q8 MSA and RoPE `.so` files preceded that arm's 40-item formal
request window, so those two cold builds occurred during the excluded smoke
stage rather than the scored window. This does not establish coverage of
other JIT modules or final-image cold-start latency.

Next quality gate: a larger natural long-context corpus with repeated,
order-balanced runs and actual path instrumentation, plus an operator-level
snapshot of real IndexScore/TopK tensors to isolate selection equality.

## JIT and online-regression audit

| Path | Current evidence | Online risk / remaining gate |
|---|---|---|
| FI RoPE | Both dataset arms have cold `rope.so`; Ninja took about 25-30 s per isolated cache. BF16's first eight LongBench requests waited about 55-56 s for their first token while MSA/RoPE JIT completed. | Missing AOT or unwritable cache is a demonstrated first-request latency regression, not just a theoretical startup risk. Warm before traffic or ship matching AOT; wrong 13.4 compiler/13.2 header combination previously failed. |
| FI BF16 NVFP4 MSA | BF16 arm built `msa_prefill_nvfp4_specialized_sm103a.so` in about 10 s. | BF16 path is JIT too; verify AOT coverage and cold start independently of Q8. |
| FI Q8KV4 MSA | Q8 arm built `rtp_m31_prefill_q8kv4_sm103a.so` in about 8.5 s. | Custom module is absent from the current FI AOT cache. `FLASHINFER_DISABLE_JIT` blocks its `JitSpec.build`, even when a normal JIT `.so` is present; package an AOT module or leave JIT enabled with the complete toolchain. |
| Native Prefill TopK | Installed-wheel H4/Q1024/N579/K16 probe hit native once; cutoff ties marked 4,096 rows and matched legacy bitonic exactly. Its Triton fallback and legacy cubins kept identical timestamps across a second process. Rank-0 model profiles recorded BS20 180 and BS60 240 native launches, each with the same number of tie-fallback launches. | Native part does not require FI JIT; fallback does. Repeat on the final target image and other ranks/buckets; the isolated cache probe and profiler runs are not cold first-request latency evidence. |
| CP writer, Gemma norm/RoPE, MoE MXFP8 scale packing | Commit `e1c866a811` removes live-row/capacity specialization. On this candidate and an isolated Triton cache, 3 dynamic-geometry GPU tests plus the norm/RoPE hash-reuse GPU test passed with changing counts. | Repeat on the final image and check compile counts across warm/cold model requests; small operator tests are not a release gate. |
| Cake decode / DSpark SWA | Triton kernels remain in these paths. In an isolated cache, Cake BF16-Q alias GPU test passed (7.8 s) and DSpark SWA-FA4 tail/replay GPU test passed (17.8 s). A fresh-cache Cake metadata/selected-scales/Graph replay test passed twice in separate processes: three cubins were created once, with unchanged timestamps on rerun. | Same-bucket repeated JIT was not observed, but Cake `ROWS/COLS` and similar constexpr geometry can create new cubins for un-warmed Graph buckets. Current LongBench exercises original D4, not a fresh candidate Decode package; these small tests do not measure first-request latency across every bucket. CuTe printed deprecation warnings, not a compile failure. |

An additional fresh-cache audit on the current installed candidate exercised
`TestNVFP4PrefixRestore.test_fi_page_pitch_scale_permutation_and_graph_replay`,
`NativeIndexRuntimeShapeTest.test_shape_changes_preserve_packed_reader_and_score_layout`,
and `CPWriterGPURegressionTest.test_production_wrapper_small_exact_and_changed_graph`.
All three passed on the first and second **separate Python processes**.
The first pass created five Triton cubins for prefix restore/IndexScore and
14 CP writer cubins across the test's distinct structural configurations,
including the exact BS20/74,011 CP writer shape. The second pass created
none and left all 19 cubin timestamps unchanged. This rules out repeated
compilation for those exercised configurations in the private cache, not
unbounded specialization for every possible live shape. The writer test also
emitted a PyTorch `CUDA Graph is empty` warning; its value/Graph assertions
passed, but this warning prevents treating the test alone as full production
Graph-coverage evidence.

FI JIT source here uses `FLASHINFER_NVCC` for the compiler but `CUDA_HOME` for
headers/link libraries. Its cache namespace is FI version plus architecture;
Ninja records commands/dependencies, but a deployment that ships only a stale
`.so`, omits its source/toolchain, disables JIT, or changes library paths is
not covered by the current warm-cache experiments. Existing CI's split FI
JIT-cache plan has local completeness checks for its standard inventory, but
the custom RTP Q8 module is a separate spec and is not implicitly in that
inventory. Perform an image-level cold start with an empty writable cache,
then a warm restart with no new Ninja actions, under both BF16 and Q8 flags.
Record compile wall time, `build.ninja`, `.ninja_log`, `ldd`, rank health, and
first-request latency separately from steady-state timing.

The writable-cache requirement is observable before any kernel launch: an
isolated import without `FLASHINFER_WORKSPACE_BASE` failed with
`PermissionError` opening the shared `~/.cache/flashinfer/.../flashinfer_jit.log`.
With a task-private writable workspace, both CPU Q8/BF16 dispatch tests passed
(`python -m unittest ...FlashInferQ8KV4DispatchTest`, 2/2). A separate
`FLASHINFER_DISABLE_JIT=1` probe found no AOT Q8 module and raised
`MissingJITCacheError` before compilation. These are cold-import/JIT policy
checks, not GPU kernel correctness tests. Each service UID therefore needs a
writable private cache or a complete validated AOT package at startup.

The candidate's historical dynamic-JIT regression tests also ran in the
separate `jit_triton_dynamic` cache. `test_m31_dynamic_kernel_geometry` passed
3/3 (runtime-unspecialized bounds/stride, CP writer row+capacity changes,
MXFP8 scale row+stride changes); `ContiguousNormRopeTest.test_live_rows_reuse_binary_and_mask_tail`
passed 1/1. Both hash-reuse cases compared compiled binary hashes across
different live sizes. This narrows the known repeated-specialization risk for
those kernels, but does not quantify FI cold-start latency or cover all
Decode Graph buckets.

Additional historical Decode operator probes with the same private Triton
cache passed: `DSparkFA4Test.test_original_table_tail_and_replay` (1/1,
17.765 s, including compilation and multiple Graph replays) and
`CakeMetadataGpuTest.test_masked_direct_bf16_query_alias` (1/1, 7.824 s).
The times are whole unittest wall times, **not** steady kernel latency.
The DSpark test emitted CuTe API deprecation warnings but no error.

## Blocking correctness gap and promotion gates

The Q8 K-stage multiplies E2M1 by E4M3 into E4M3 with saturation. Valid cache
blocks can have `6 * block_scale > 448`; the exact full-13.4-link artifact
still failed the extreme-scale comparison above. Thus replacing the mixed
13.2/13.4 link with 13.4 runtime does not fix the numerical gap. The
LongBench score does not prove the high-scale regime is safe. Keep Q8 opt-in
and BF16 default.

An isolated 8-query/16-page cross-scale probe (seed 0, BF16 Q, full-13.4
linked artifact, 65,536 output elements per row) localizes the gap:

| K E4M3 scale | V E4M3 scale | CuTe FP8-Q failed at 0.02+2% | CuTe BF16-Q failed at 0.02+2% |
| --- | --- | ---: | ---: |
| mixed | mixed | 0 | 0 |
| extreme | mixed | 9,199 | 59,562 |
| mixed | extreme | 65,120 | 7,466 |
| extreme | extreme | 55,974 | 61,004 |

Here `mixed` uses scale codes `28/37/3f/45`, and `extreme` uses
`70/78/7c/7e` (hex). CuTe sets **both** K and V compute dtype from Q dtype:
FP8-Q uses FP8 K and FP8 V, whereas BF16-Q uses BF16 K and BF16 V.
The experimental RTP kernel uses FP8 K and BF16 V. Therefore the dominant
V-extreme difference from CuTe FP8-Q is a real compute-dtype/saturation
semantic difference, not proof that RTP's V-scale address calculation is
wrong. Comparing V-extreme against CuTe BF16-Q shrinks max absolute difference
from 55 to 0.5, though 7,466 near-zero/small outputs still exceed the strict
0.02+2% threshold. Conversely, CuTe BF16-Q is not a matching QK reference
when K scales are extreme. The K-only mismatch against CuTe FP8-Q remains
9,199 failures (max absolute 0.625) and needs separate QK/softmax diagnosis.
CuTe FP8-Q also stores softmax probabilities in FP8 for its FP8 PV MMA,
whereas RTP's PV probabilities are BF16; the full algorithms cannot be expected
to be bit-identical merely because their QK K stage is FP8.

The V-only probe with **Q exactly zero** removes all QK/logit effects. It
still fails 65,120/65,536 against CuTe FP8-Q (max absolute 47.75), but only
2,144/65,536 against CuTe BF16-Q (max absolute 0.5). This directly supports
the V compute-dtype explanation. With K-only extreme and V mixed, shrinking
Q amplitude from 0.03 to 0.003 to 0.0003 reduces CuTe FP8-Q failures from
9,199 to 40 to zero, respectively; high-range QK/softmax rounding is a
plausible contributor, but not yet a localized root cause.

An independent zero-Q oracle unpacks the 4-bit V codes, decodes E2M1 values,
applies the V E4M3 block scales in BF16, and averages the causal prefix of
the 16 selected pages. Its result is **bit-identical to RTP** for qlen/seed
`8/0`, `4/1`, and `16/2` (65,536 + 32,768 + 131,072 output elements). For
`8/0`, CuTe FP8-Q differs from this BF16-value oracle on 65,088 elements;
CuTe BF16-Q differs on 2,160 at the 0.02+2% threshold (max absolute 0.5).
This validates the RTP V scale swizzle and uniform-weight BF16-V route for
these cases; it does **not** validate nonzero-Q softmax or broad model quality.

For nonzero Q, a second independent oracle uses the rounded FP8 Q, decodes
K from E2M1/E4M3 with FP8 E4M3 saturation, decodes V to BF16, then computes
FP32 causal softmax and PV for one query head (1,024 output values per case).
At `atol=0.02, rtol=0.02` against this **Q8K/BF16V** semantic reference:

| K/V scales, seed | RTP failures | CuTe FP8-Q failures | RTP max absolute error |
| --- | ---: | ---: | ---: |
| mixed/mixed, 0 | 0 | 0 | 0.001953 |
| extreme/mixed, 0 | 0 | 152 | 0.125 |
| extreme/mixed, 7 | 0 | 156 | 0.125 |
| mixed/extreme, 0 | 21 | 1,017 | 0.25 |
| mixed/extreme, 3 (13.4 rerun) | 21 | 1,017 | 0.25 |
| mixed/extreme, 7 | 29 | 1,021 | 0.25 |
| extreme/extreme, 0 | 29 | 937 | 16 |

The FP32-softmax oracle rounds only the final output to BF16; an additional
BF16-probability oracle gives nearly the same pattern. These synthetic checks
show the custom path is generally closer to its intended mixed-precision
semantics than CuTe's FP8-PV path, but the remaining failures at extreme V
scales are real under the chosen strict tolerance. They do not establish
real-cache scale prevalence, checkpoint quality, or online MAL stability.

A separate private operator prototype tried to downscale every E4M3 K scale
by 8 (exact exponent change for normal scales) and multiply QK logits back
by 8. It did **not** solve the oracle gap: mixed scales still passed all
65,536 values, but extreme scales failed 63,179/65,536 (max absolute error
3,136), worse than the original. Lowering Q amplitude also did not remove
the discrepancy, so attributing it solely to large-logit softmax was premature.
The prototype lives only in
`.../1010_topk_q8_model_candidate/msa_prefill_q8kv4_scale8_experiment.cu`
and was not wired into the candidate model. Next gates are an independent
FP32/BF16 dequant reference, broader K/V scale coverage, and a separately
verified QK fix or shape/scale dispatch; no high-range fallback is yet proven.

A test-only diagnostic sampled **active physical working pages from the actual
model Prefill path** after the FI page table was built. For each segment, it
kept columns whose page start was below `seqused_k`, deduplicated the physical
page IDs, and scanned both E4M3 scale planes. The first positive E4M3 code
for `block_scale > 448/6` is `0x6A` (80.0); the preceding `0x69` is 72.0.
The diagnostic ran one warmup plus one measured batch per shape, on all four
CP ranks, with a temporary instrumentation hook in the private installed
adapter. The hook was removed afterward and the installed adapter SHA256 again
matched the worktree (`d17bac66...e29cb`). Its host synchronizations make
these runs **invalid as performance A/Bs**.

| Real-model synthetic input | Operator records | Max active pages in one call | Max K scale code/value | Max V scale code/value | K codes >= `0x6A` |
|---|---:|---:|---:|---:|---:|
| BS20, 74,011 input, 66,560 reuse | 960 (240/rank) | 11,580 = 20 x 579 | `0x41` / 2.25 | `0x5A` / 20.0 | 0 |
| BS60, 80,000 input, 75,776 reuse | 960 (240/rank) | 37,500 = 60 x 625 | `0x41` / 2.25 | `0x5A` / 20.0 | 0 |

All 20/20 and 60/60 measured requests succeeded. Artifacts:
`.../1010_topk_q8_model_candidate/bs{20,60}_q8kv4_scale_diag_1010/`
`scale_diag/scales.pid*.jsonl` and `metrics.json`. Thus the FP8 K saturation
condition did not arise on these two synthetic fixed-shape inputs, including
calls covering the complete logical history. This is not a distribution
estimate for natural LongBench prompts or online traffic, and it does not
explain the separate strict-oracle residual at extreme V scales.

The same temporary diagnostic also ran on the natural LongBench160 prompt
set: 20 warmup requests plus all 160 formal IDs, with 160/160 formal
completion and a valid native MAL measurement window. Four Prefill rank files
contain 3,240 operator-call records each (12,960 total). Across active
physical pages selected by `page_table` and `seqused_k`, `q_rows` ranged from
828 to 22,654 and active pages from 17 to 711 per call. The maximum K scale
code was `0x41` (2.25), maximum V scale code `0x59` (18.0), and **zero K or V
scale entries were `>= 0x6A`**. Artifacts:
`.../1010_topk_q8_model_candidate/dataset_quality/q8kv4_natural_scale_diag_1010/`
`scale_diag/scales.pid*.jsonl` and `longbench160/integrity.json`.

The hook was removed after the run; the private installed adapter again
matches the candidate source SHA256 (`d17bac66...e29cb`). Instrumentation
synchronizes the host, so this run's latency, MAL, and score are **not** a
matched performance or quality A/B. The observed absence of high K scales
on every active page is stronger than a selected-token-only check for these
prompts, but does not bound unseen online inputs. The synthetic extreme-V
strict-oracle failure remains; do not infer production precision from this
range sample alone.

## Review and release boundary

The final scoped dirty candidate received one fresh **read-only independent
RTP-LLM code review** after the two matched Prefill campaigns and two
same-ID LongBench/MAL pairs. No new independently confirmed code blocker was
found. The reviewer confirmed opt-in routing/default BF16 behavior, explicit
unsupported-architecture failure, source-to-installed hashes, the raw
fixed-shape metrics, rank-0 native TopK trace counts, and LongBench scores.
The reviewer did not run GPU tests or independently recompute the synthetic
extreme-scale oracle. A separate installed-package CPU dispatch check ran
`FlashInferQ8KV4DispatchTest` (2/2 passed); `py_compile` of the changed
Python files and `git diff --check` also passed. The warmed Q8 FI module's
generated Ninja recipe currently reports `no work to do`.

The candidate remains **opt-in experimental, not production-ready**. Natural
LongBench160 and the two fixed-shape model inputs showed no high K scales, but
broader online scale coverage and the high-scale behavior remain open. Before
production: resolve or prove a valid range gate for `6 * K_block_scale > 448`,
rerun extreme/mixed independent oracles and sanitizer, profile native TopK on
remaining ranks/buckets, and qualify cold/warm starts plus changing shapes on
the final image with the matching 13.4 toolchain/AOT and no unplanned JIT.
The two quality/MAL pairs and six-round fixed-shape Prefill repeats are
evidence, not a statistical quality non-inferiority or broad stability bound;
expand those gates for release. Re-review any subsequent code delta before
claiming completion. Preserve the resident PD4+4 service during isolated
qualification.

## Full dataset accuracy, 2026-10-11

The frozen PD4+4 BF16 and Q8KV4 candidate arms used the same manifest SHA256
`7265b4621b30a03734223003938bed01399b40667ccc7b2044d15841bde1a8fd`,
greedy sampling, disabled Prefill reuse, normal rejection, and four Decode DP
ranks. MATH500/HumanEval ran at 8 client workers per DP (32 concurrent),
512-token cap; AIME24 ran at 2 workers per DP, 16,384-token cap. All 664/664
MATH/HumanEval and 30/30 AIME requests succeeded in each arm. A separate
per-response audit checked unique IDs, HTTP 200, no request error, score counts,
and the PID-bound native MAL report. Candidate-generated HumanEval failures
are wrong answers, not transport failures: BF16 had 2 SyntaxErrors; Q8 had a
SyntaxError, a NameError in generated code, and a sandbox-rejected `__getitem__`
attribute. The inherited wrapper's `integrity.complete=false` reflects those
score errors, while the independent `strict_native_report.valid=true` and
`open3_audit.json` pass for both arms.

| Dataset and scoring contract | BF16 | Q8KV4 | Same-ID comparison |
|---|---:|---:|---|
| GSM8K1314, math-equivalent numeric answer, `length` wrong; first full pair | 1235/1314 | 1227/1314 | Common nontruncated: 1230 vs 1226 / 1300 |
| GSM8K1314, same contract; reverse-order repeat | 1231/1314 | 1233/1314 | Common nontruncated: 1224 vs 1231 / 1293 |
| MATH500, inherited `math_verify.parse+verify` (may score a `length` response) | 389/500 (77.8%) | 371/500 (74.2%) | Both right 345; BF-only 44, Q8-only 26, both wrong 85 |
| HumanEval164, AST-guarded execution | 84/164 (51.22%) | 83/164 (50.61%) | Both right 57; BF-only 27, Q8-only 26, both wrong 54 |
| AIME24 30, math-equivalent answer and `length` wrong | 24/30 | 25/30 | Both right 23; BF-only 1, Q8-only 2, both wrong 4 |

For MATH500, BF16/Q8 had 63/61 `length` responses; on the 410 items where
both stopped normally, BF-only/Q8-only was 26/14. The overall 18-item Q8 loss
is a **precision warning**, not a passing non-inferiority result. The paired
70-discordance exact two-sided sign test is `p=0.0414` without multiple-test or
run-to-run correction; it does not identify which operator caused the difference.
Repeat the pair with reversed service order and localize changed logits/TopK/
attention on discrepant IDs before enabling Q8 by default. HumanEval's one-item
net difference and AIME's one-item net difference are not stable superiority
claims. AIME BF16/Q8 had 5/4 `length` responses; the inherited numeric-string
scorer reported 27/30 and 26/30 but overcounted respectively 3 and 1 truncated
responses. The math-equivalent, length-wrong sidecar is the reported contract.

For the mixed MATH/HumanEval windows, BF16 native MAL was
`143022/36403 = 3.92885`, Q8 was `144149/36541 = 3.94486`. For AIME24 it was
`200818/67701 = 2.96625` versus `179272/62013 = 2.89088`. These are
per-window observations, not a matched throughput or stable MAL claim. The
prior warmed LongBench160 pair remains BF16 `66.3581` versus Q8 `66.3955`
raw score with 160/160 successful requests; the first pair was `66.4320`
versus `67.5942`. The 160-item subset did not exercise the new TopK dispatch.
This campaign has no MT-Bench judge score; do not label it an accuracy pass.

Artifacts: `.../1010_topk_q8_model_candidate/dataset_quality/`
`{bf16_open3_bf16_1011,q8kv4_open3_q8kv4_1011}/`
`{math_human664,aime30}/`, including `responses.jsonl`, `scores.jsonl`,
`strict_native_report.json`, `open3_audit.json`, and
`aime24_math_scores.summary.json`. After candidate shutdown, the original
`m31_decode_next_restored_pd4p4_framework_overlap_1010` service was restarted;
`restore_health_and_smoke.json` records five idle/alive ports and successful
8-token P-to-D smoke requests on all four Decode lanes. The candidate is still
opt-in experimental; the MATH500 loss and earlier extreme-scale FP32-oracle
residual remain release blockers.

### Reverse-order MATH500 repeat and cold JIT

A second same-ID 500-item MATH500 pair ran in reverse service order, Q8KV4
then BF16, with fresh P4/D4 processes and isolated FI JIT workspaces. The
manifest SHA256 remained `7265b462...da8fd`; both `selected_ids` arrays hash
to `890b56cdb435faf1e079a9c9590b239dd44322ae8ce8fc88684f028c407c7902`.
The Prefill launch environments differ only in
`M3_NVFP4_PREFILL_COMPUTE` and the arm-private `FLASHINFER_WORKSPACE_BASE`.
Both arms completed 500/500 requests with no request errors, all four DP ranks
reached eight active client workers, and their PID-bound native MAL reports
were valid.

| MATH500 pair | BF16 | Q8KV4 | Same-ID BF-only / Q8-only | Both normal-stop BF-only / Q8-only |
|---|---:|---:|---:|---:|
| First pair, BF16 then Q8 | 389/500 | 371/500 | 44 / 26 | 26 / 14 (410 common stops) |
| Reverse pair, Q8 then BF16 | 388/500 | 384/500 | 37 / 33 | 20 / 15 (407 common stops) |

BF16 score moved by one item; Q8 moved by 13. Within-arm exact response-text
matches across repeats were 235/500 BF16 and 253/500 Q8; cross-arm matches
were only 16/500 and 18/500. Across both repeats, 23 IDs were correct both
times under BF16 and wrong both times under Q8, versus 12 IDs in the opposite
direction. Of those persistent cases, 14/8 respectively had normal-stop
outputs in both runs; the rest included `length` outputs. The two pairs both
favor BF16, but the gap is not stable in magnitude. These are repeated fixed
IDs, not independent 1,000-item samples. The lower Q8 score remains a
precision warning, while a one-pair `p=0.0414` is not a robust release gate.
The native Prefill TopK optimization is ineligible on this MATH500 set:
prompts ranged from 187 to 925 tokens, far below its 512-KV-block minimum.
The Q8-vs-BF16 compute backend is selected without a length cutoff in the
FlashInfer adapter, so the current differential concerns Q8 MSA prefill and
run-to-run generation effects, not the new native TopK dispatch. Per-layer
logit/operator-input localization is still required to establish causality.

The reverse pair intentionally used empty FI workspaces and no excluded warmup.
In the Q8 arm the formal window opened at 02:32:53; its Q8 MSA `.so` appeared
at 02:33:01 (Ninja 7.4 s) and FI RoPE `.so` at 02:33:27 (Ninja 24.0 s).
First-token times for the slowest first requests were about 44.9 s. The BF16
arm similarly built its FI MSA `.so` in 7.6 s and RoPE `.so` in 23.8 s, with
about 44.7 s first-token time. These cold-start observations are not steady
kernel latency and do not prove a Q8-specific JIT regression. They do prove
that a final image without matching AOT or prewarm would expose multi-second
first-request stalls in **both** arms; the custom Q8 module is not in the
existing FI AOT cache.

Repeat artifacts:
`.../1010_topk_q8_model_candidate/dataset_quality/`
`{q8kv4_math_rev_q8_1011,bf16_math_rev_bf_1011}/math500/`, with contracts,
responses, scores, integrity, native reports, and per-arm JIT `.ninja_log`.
After this repeat the original PD4+4 container was restored again; the latest
`restore_health_and_smoke.json` records five idle/alive endpoints and four
successful P-to-D requests. No release/rollout decision changes: Q8 stays
opt-in, and MATH500 needs backend-specific precision localization plus a
longer quality stability gate.

### Same-input model Q and attention shadow diagnostic

An opt-in `M3_FI_Q8_SHADOW_DIR` hook in the RTP FlashInfer adapter ran the
BF16 MSA kernel after the production Q8KV4 kernel on the **same** BF16 Q,
packed FP4 K/V, page table, TopK selection, and scales. It returned the
original Q8 output unchanged. The frozen MATH500 manifest's first four IDs
completed 4/4 with no request errors. They formed two Prefill calls: one
231-token request (`Q rows=58` per CP rank), then a batch of three requests
(`Q rows=204` per CP rank). There were 4 CP ranks x 60 layers x 2 calls =
480 JSON records; the earlier one-call probe's 240 records were **one
request across four CP ranks**, not four distinct inputs. The two diagnostic
runs produced identical full response text for all four IDs, and their native
MAL counters matched exactly. Diagnostic latency is excluded from any
performance comparison.

Across these 480 layer/rank/call records, actual BF16 Q had absolute maximum
5.5-11.875 (median 8.5625) and RMS 0.711-1.364 (median 1.215); no Q value
exceeded the E4M3 saturation threshold 448. A PyTorch BF16-Q to E4M3 and
back cast proxy had relative RMS 2.620-2.724% (median 2.662%); it has not
been proven bitwise identical to the CUDA kernel's conversion instruction.
Q8KV4 versus BF16 MSA output
relative RMS was 0.587-5.272%, median 3.126%, P90 4.071%. The first call
and three-request batch respectively had median output relative RMS 3.070%
and 3.155%. These numbers establish a real same-input operator difference
at natural MATH scales, without FP8 saturation. They do **not** isolate the
Q cast from the FP4-K compute path, measure layer-hidden/logit divergence,
or explain which MATH500 answers changed. Next precision work should replay
persistent BF-only IDs and compare Q-only/K-path controls plus the first
divergent model logits.

Artifacts: `.../dataset_quality/q8kv4_math_q8shadow4_1011/{math4,shadow}/`
and the first-call-only cross-check at
`.../dataset_quality/q8kv4_math_q8shadow_1011/{math4,shadow}/`.
The diagnostic is disabled unless its environment variable is set; the
candidate launch record records the opt-in. Diagnostic file I/O failures are
logged and do not replace the Q8 output; CUDA shadow-compute failures are not
suppressed. Its focused CPU mock suite passed 3/3, including an injected
write failure, and the original resident PD4+4 was again restored and checked via
five idle endpoints and four P-to-D smoke RPCs.

### Observed-scale semantic oracle and JIT environment

The earlier `mixed` scale probe topped out below the cache values sampled on
natural LongBench inputs. The private single-operator fixture now includes
`observed_k = {0x28,0x37,0x3f,0x41}` and
`observed_v = {0x28,0x45,0x51,0x59}`. These sets include the observed maximum
codes, but deliberately assign them randomly to many blocks; they are a
synthetic stress distribution, **not** the measured cache histogram. The
fixture uses BF16 Q fused to E4M3 in the operator, 16 selected NVFP4 pages,
and an independent FP32-softmax/Q8K-BF16V oracle for one query head. It
compares BF16 outputs with `atol=0.02, rtol=0.02`.

| Q length | Input Q amplitude | Seeds | Oracle elements | Q8 values over tolerance | Max absolute difference |
|---:|---:|---|---:|---:|---:|
| 8 | 0.03 | 0,1,3,7 | 4,096 | 0 | 0.015625 |
| 8 | 0.3 | 0,3,7 | 3,072 | 0 | 0.0625 |
| 64,128 | 0.3 | 0 each | 24,576 | 0 | 0.125 |
| 8 | 1.0 | 0,3,7 | 3,072 | 5 (1,4,0) | 0.5 |

All 12 outputs were finite. At the observed scale-code range, the moderate-Q
cases pass this semantic tolerance even at larger query lengths; the
high-amplitude Q cases do not pass a strict zero-failure gate. The actual
model Q distribution was not captured, so this cannot explain or waive the
MATH500 score difference. CuTe FP8-Q uses FP8 PV rather than this kernel's
BF16 PV and remains an intentionally different numerical reference. The JSON
results also include a BF16-probability oracle; its high-Q failures are
`0,5,4`, versus the table's FP32-probability `1,4,0`. Do not mix those
probability-rounding contracts. Results are in
`.../1010_topk_q8_model_candidate/observed_scale_oracle_1011/`;
the initial 12 JSON files were produced by `compare_q8_cute_candidate.py`
(then SHA256
`128438416b1cd12e244ec4f9dfdb7c0a835abc324b04076694dcf77182f3c47a`).
The installed and source `msa_prefill_q8kv4.cu` hashes matched at
`f332d3fca9ed7b683796dc5f32f9369768ec86b9d1d87414bdb629d01f40f867`.

One probe initially **did** rebuild an already warm Q8 `.so` in a new process:
its ad hoc Docker environment defaulted to `CUDA_HOME=/usr/local/cuda` and
`CXX=c++`, while the model launcher used `CUDA_HOME=/usr/local/cuda-13.2` and
`CXX=/usr/bin/g++`. The generated Ninja command hash changed, adding one
object/link pair to `.ninja_log`; this was an environment mismatch, not
shape-based repeated JIT. On an untouched model cache, `ninja -d explain -n`
reported `no work to do`. A fresh standalone probe with the exact model
`CUDA_HOME`, `CC=/usr/bin/gcc`, `CXX=/usr/bin/g++`, `FLASHINFER_NVCC`, and
`FLASHINFER_EXTRA_CUDAFLAGS` left both Q8 `.so` and `.ninja_log` mtimes
unchanged (`1791655911`) and again passed the seed-0 oracle. The final image
must freeze these compiler paths/flags and AOT or prewarm policy together;
matching only the Python wheel version is insufficient.

### Identical-input FI BF16 versus Q8KV4

To separate operator semantics from CuTe's different FP8-PV path, the private
fixture now calls **both** FI BF16 MSA and the RTP FI-algorithm Q8KV4 module
on identical BF16 Q, packed FP4 K/V, E4M3 scales, TopK rows and page table.
The Q8 module casts BF16 Q to E4M3 on load. The same independent reference
decodes K to BF16 for a BF16-QK/FP32-softmax/BF16-V oracle, in addition to the
previous Q8K/BF16V oracle. In the initial all-16-pages-selected, KV-head-0
fixture, both FI paths had zero values
beyond `0.02+2%` against their **own** oracle at Q amplitude 0.03 and 0.3,
including 64- and 128-query cases. At Q amplitude 1.0, each path had one
strict-tolerance failure for seed 0. This checks the common page/scale layout
only for that head and all-page selection; it does not make their outputs
interchangeable.

| Q length | Q amplitude | Seeds | Direct Q8 versus FI BF16 relative RMS | Values over `0.02+2%` (all heads) |
|---:|---:|---|---:|---:|
| 8 | 0.03 | 0,3 | 0.59–0.62% | 0/65,536 each |
| 8 | 0.3 | 0,3,7 | 7.50–8.15% | 42,714–43,350 / 65,536 |
| 64,128 | 0.3 | 0 each | 7.63%, 7.78% | 347,428/524,288; 695,367/1,048,576 |
| 8 | 1.0 | 0,3,7 | 10.81–11.16% | 52,936–53,514 / 65,536 |

In the single-head FP32 semantic decomposition at Q amplitude 0.3, converting
only Q to E4M3 produced 3.46–5.12% relative RMS versus BF16; converting only
K to E4M3 produced 6.25–8.44%; converting both was non-additive. At 64/128
queries (seed 0), Q-only was 3.77/3.90% and K-only was 6.11/6.16%. This
suggests K-stage FP8 rounding is a major contributor **for the synthetic
observed-code distribution**; the real cache scale histogram and actual model
Q amplitudes were not captured. The model MATH500 gap still needs same-input
per-layer activation/logit tracing before causality or a quality fix can be
claimed. Data: `.../1010_topk_q8_model_candidate/`
`observed_scale_fi_ab_{1011,oracle_1011,decomp_1011}/*.json`; the BF16 and Q8 FI JIT modules
both reported `ninja: no work to do` after repeated exact-environment runs.

A follow-up fixture made the selection genuinely sparse: 32 physical pages,
16 distinct TopK pages per query, with **different page sets for each of four
KV heads**. Its oracle masks unselected and right-causal token positions,
then independently checks the first query head mapped to each KV head. On
observed K/V scale-code sets at Q amplitude 0.3, all five cases passed both
semantic oracles at `0.02+2%`:

| Query length, seed | Checked values per path (4 KV heads) | Q8 failures | BF16 failures | Direct Q8/BF16 relative RMS (all 64 query heads) |
|---|---:|---:|---:|---:|
| 8; 0,3,7 | 4,096 each | 0 | 0 | 7.47–7.48% |
| 64; 0 | 32,768 | 0 | 0 | 7.68% |
| 128; 0 | 65,536 | 0 | 0 | 7.69% |

At Q amplitude 1.0 and query length 8 (seed 0), the sparse case had 14 Q8
and 9 BF16 strict-tolerance failures out of 4,096 checked values, so neither
path earns a broad high-amplitude zero-failure claim. The 32-page cases
validate the tested sparse page/head/scale addressing much more directly than
the initial all-page probe; they still use synthetic random FP4 values,
synthetic scale-code frequencies, one head per KV group, and no actual model
Q tensors. Artifacts:
`.../1010_topk_q8_model_candidate/observed_scale_sparse_1011/`.
The final sparse fixture SHA256 is
`88d3c64df4bda3c3e6da4e54ff45273daf56c5cb0074066f6385e065244b6fae`;
earlier full-page all-head comparison JSON came from intermediate fixture
revisions and should be interpreted with their saved arguments/results rather
than assumed to have the final head-specific TopK construction.
As a V-addressing control, the 32-page sparse case with Q exactly zero made
the BF16 and Q8 FI outputs identical; Q8 also matched the independent
selected-page causal V mean on all 65,536 output values at `0.02+2%`.
The 16-page all-selected regression case continued to pass the four-KV-head
oracle after the TopK/head variation was added.

### Full GSM8K1314 under private FI/CUDA 13.4 (2026-10-11)

The same frozen 1,314-ID manifest (`gsm8k_full1314_1011.jsonl`, SHA256
`29296ca57d4afe241aeabe5e0393b0fe5f839e2a1b97977936289b1b892f9d01`)
was run on separate BF16 and Q8KV4 PD4+4 processes. Only Prefill used the
private FI `0.6.12+250aae19` wheel and CUDA 13.4 compiler/runtime; Decode
retained the candidate image. Each arm had a 32-request warmup, then a fresh
full run with cache reuse disabled, greedy sampling, 512 max output tokens,
and 8 client workers per DP (32 total). The native integrity and MAL checks
passed in both formal runs. The legacy numeric scorer is unsuitable for
units/thousands/decimal formatting; quality below uses the saved
`math_verify.parse+verify` rescore on exactly the same responses.

| Arm | Completed/errors | `math_verify` correct | Native MAL | Request window | Completion tokens | Finish: stop/length |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 1314/1314, 0 | 1236 (94.06%) | 4.12062 | 180.721 s | 142,941 | 1304/10 |
| Q8KV4 | 1314/1314, 0 | 1225 (93.23%) | 4.12063 | 45.508 s | 147,417 | 1293/21 |

Both runs reached `[8,8,8,8]` peak per-DP client concurrency. Same-ID
`math_verify` outcomes: 1,200 both correct, 36 BF16-only, 25 Q8-only, 53
both wrong. Thus the paired net gap is 11 items, not 11 one-way failures.
Against the most recent old-FI full run at 32 concurrency, Q8 moved from
1228 to 1225 (15 old-only, 12 new-only); BF16 moved from 1227 to 1236
(10 old-only, 19 new-only). Earlier old-FI paired rounds also varied. These
results show complete execution and no large GSM8K collapse; they do **not**
prove identical answers, isolate the cause of the 11-item BF16/Q8 gap, or
resolve the separate MATH500 precision gap. Q8 generated 4,476 more tokens
and had 11 more length finishes. The 45.5/180.7-second windows are
full-model measurements of these particular arm/order/path runs, not a
single-operator speedup or a controlled FI-version speedup; further matched
repeats and phase-level profiling are needed before assigning the difference.

Artifacts: `.../1010_topk_q8_model_candidate/dataset_quality/`
`{bf16_gsm1314_fi134_bf_1011,q8kv4_gsm1314_fi134_q8_1011}/gsm8k1314/`
(`integrity.json`, `responses.jsonl`, `gsm8k_numeric_scores.jsonl`,
`gsm8k_numeric_scores.summary.json`). The single-GPU prewarm **did** compile
all three modules, and a second standalone process reused them. Nevertheless,
the real BF16 model startup recompiled FI BF16 MSA and RoPE, and the Q8
model startup recompiled RTP Q8 MSA in that same workspace. Each module's
`.ninja_log` has a second object/link set with different command hashes;
the final `.so` timestamps were approximately 04:38:37, 04:39:04, and
04:46:51 respectively. The RTP Q8 source mtime did not change. The original
prewarm command environment was not recorded, so the exact changed flag
cannot be proven; a compiler or environment mismatch is the leading
explanation. As a cross-check, the surviving independent FI 13.4 workspace
`fi134_gpu_test_workspace` and the model-generated recipe have identical
BF16/Q8 MSA Ninja files after normalizing workspace paths **except** that
the model uses `cxx=/usr/bin/g++` instead of `c++` and adds
`-ccbin /usr/bin/gcc`. Either command difference is sufficient to invalidate
Ninja's object hash. The overwritten first recipe in the shared prewarm
workspace cannot be recovered from this comparison. Post-run Ninja dry runs
for all three modules now say `no work to do`; that is a **post-rebuild**
state, not proof of a clean initial startup. This is a concrete case where
successful standalone prewarm did not prevent model startup recompilation,
including older FI MSA/RoPE modules. A release image still needs
exact-module AOT/prewarm with the **identical** service compiler environment,
fresh-process cold/warm checks, and a no-build post-readiness request. After
the campaign, the private service exited
cleanly; the original `m31_decode_next_restored_pd4p4_framework_overlap_1010`
was restarted. Its five ports (23100, 24100, 24109, 24118, 24127) were
alive and idle, and four real 8-token P-to-D smoke requests succeeded;
`restore_health_and_smoke.json` records final `resident: true`.

### Exact-service-environment JIT replay (2026-10-11)

To test the suspected compiler-command mismatch without changing the model,
the eight JIT-relevant environment values (`PYTHONPATH`, `LD_LIBRARY_PATH`,
`CUDA_HOME`, `CC`, `CXX`, `CPATH`, `FLASHINFER_NVCC`, and
`FLASHINFER_EXTRA_CUDAFLAGS`) were taken from the saved Q8 Prefill
`launch.json`. A host-side structured JSON read supplied these values via
`docker exec -e`; the target container has no `jq`. A first attempt using
container-side `jq` failed before module compilation because it produced an
empty `PYTHONPATH`, so its workspace was not reused. The successful run used
the fresh `fi134_service_env_prewarm_v2_1011` workspace, CUDA device 0, and
the same installed RTP/FI package paths as the model.

| Module | Fresh prewarm | Second fresh process | `.ninja_log` and `.so` after second process |
|---|---:|---:|---|
| FI BF16 MSA | 7.982 s | 0.363 s | Unchanged; `ninja -d explain -n`: no work |
| RTP Q8 MSA | 7.391 s | 0.006 s | Unchanged; no work |
| FI RoPE | 23.722 s | 0.008 s | Unchanged; no work |

The new Q8 PD4+4 model process used that **same** workspace. Its Prefill
`launch.json` matched all eight recorded environment values. Immediately
after model startup, all three `.ninja_log` SHA256 values and `.so` mtimes
still matched the prewarm snapshot. A real 32-request GSM8K warmup at 8
clients per DP then completed 32/32 with zero request errors, native MAL
4.24933, and strict native validation. After the request, the three cache
hashes/mtimes remained unchanged and all Ninja dry runs still said `no work
to do`. For the Q8 model, this directly exercises RTP Q8 MSA and RoPE;
BF16 MSA's unchanged cache is an availability check, **not** a BF16-model
request-path check. This proves no FI-module recompilation for the tested
`sm_103a` Q8 startup/request shape with the exact service environment. It
does not qualify the final deployment image, other CUDA architectures,
DeepGEMM/Triton/Cake buckets, or AOT packaging.

Artifacts: `.../1010_topk_q8_model_candidate/fi134_service_env_prewarm_v2_1011/`
and `dataset_quality/q8kv4_jit_exactenv_q8_1011/gsm8k_warm32/`. After the
test, the candidate ranks exited. Restarting the preserved original PD4+4
first hit `launch_roles.py`'s exclusive-create guards: old `process.json`
contained dead PIDs, then old `console.log` blocked `open('x')`. Both sets
were archived and renamed (not deleted) before restart. The final
`restore_health_and_smoke.json` confirms five alive/idle ports and four
successful real 8-token P-to-D requests, with `resident: true`. A production
restart procedure must handle those stale owner/log files only after
verifying the old PIDs are dead; `docker start` alone is not readiness.

### Full MATH500 pair under private FI/CUDA 13.4 (2026-10-11)

The frozen MATH500 manifest above still hashes to
`7265b4621b30a03734223003938bed01399b40667ccc7b2044d15841bde1a8fd`.
Two fresh PD4+4 processes ran Q8KV4 then BF16, using the same private FI
13.4 wheel, the exact-service-environment prewarmed workspace, greedy
sampling, no Prefill reuse, 512-token cap, and 8 clients per DP. Their
Prefill `launch.json` environments are identical after removing only
`M3_NVFP4_PREFILL_COMPUTE`. Each formal run reached `[8,8,8,8]` peak
per-DP concurrency; all 500 IDs completed with zero request errors and
strictly valid PID-bound native MAL.

| FI 13.4 arm | `math_verify` correct | BF16-only / Q8-only on same IDs | Normal stop / length | Native MAL | Request window |
|---|---:|---:|---:|---:|---:|
| BF16 | 389/500 (77.8%) | 38 / 34 | 439 / 61 | 3.85803 | 41.917 s |
| Q8KV4 | 385/500 (77.0%) | 38 / 34 | 439 / 61 | 3.85259 | 46.412 s |

There were 351 both-correct and 77 both-wrong IDs. On the 416 IDs where
both arms stopped normally, BF16-only/Q8-only was 25/18. Against the most
recent old-FI pair, BF16 moved 388 to 389 (12 old-only, 13 new-only) and
Q8 moved 384 to 385 (16 old-only, 17 new-only). Across the first old-FI
pair, the reverse old-FI pair, and this FI 13.4 pair, 18 IDs were BF16-only
in **all three**, while 11 were Q8-only in all three. The new net four-item
gap is within observed process-level variation and does **not** prove
non-inferiority or that CUDA 13.4 fixed Q8 precision. The persistent
one-sided IDs and the normal-stop gap remain a precision warning requiring
same-input per-layer/logit localization. These short MATH prompts do not
exercise the native long-context TopK dispatch.

The Q8 arm emitted 111,053 completion tokens versus 110,673 for BF16.
The 46.412/41.917-second request windows are not a controlled kernel-speed
comparison; this order-only pair is quality/stability evidence. The three
FI module `.ninja_log` hashes and `.so` mtimes in
`fi134_service_env_prewarm_v2_1011` remained unchanged after **both** model
arms, extending the exact-environment no-recompile check to BF16 MSA as
well as Q8 MSA/RoPE for these shapes.

Artifacts: `.../1010_topk_q8_model_candidate/dataset_quality/`
`{q8kv4_math500_fi134_q8_1011,bf16_math500_fi134_bf_1011}/math500/`
(`contract.json`, `responses.jsonl`, `scores.jsonl`, `integrity.json`).
Afterward the original PD4+4 service was restored again; final
`restore_health_and_smoke.json` reports five alive/idle ports and four
successful real P-to-D 8-token requests, with `resident: true`.

### Triton request-shape JIT audit and fixes (2026-10-11)

The FI module no-rebuild gate did not cover RTP-owned Triton kernels. In the
original FI 13.4 MATH500 Q8 formal window (46.412 s), the shared Triton
cache gained 59 `_prepare_topk` and 38 `_clear_unwritten_page_tail` cubins;
the BF16 window gained 52 and 7 respectively. Timestamp/name attribution
alone is not process-exclusive, but the FI adapter had `ROWS`/`COLS` and
`TABLE_STRIDE`/`SEGMENTS_PER_REQUEST` as `tl.constexpr`. Those values change
with each batch. The candidate now treats them as non-specialized runtime
scalars; page-byte layout remains compile-time fixed. A focused GPU test
checked three distinct query/page/segment geometries and tail-byte outputs
with **one cubin per kernel** in an isolated cache. The complete FI adapter
test module passed 6/6. CUDA Graph median hot latencies for `_prepare_topk`
were 1.416 to 2.028 us at 256 rows/8 columns and 1.515 to 2.426 us at
1024 rows/579 columns (old to new); tail clear was 1.667 to 1.705 us.
Thus the no-JIT change has a small per-launch cost rather than a free kernel
speedup.

The first real-model candidate with only that adapter change used an
isolated Prefill `TRITON_CACHE_DIR`. After a 32-request warmup and full
MATH500 run, both target kernels remained at **one cubin each**; 500/500
requests succeeded, no errors, `math_verify` 377/500, native MAL 3.85212.
Its formal request window was 36.717 s for 110,914 completion tokens. The
same cache exposed a second, independent JIT problem: BF16 Prefill router
cubins grew from 4 after warmup to 81 after formal MATH500. Its `ROWS` was
also `tl.constexpr`, used only for tail-row masking. The candidate now makes
that scalar non-specialized while keeping the two weight strides constant.
Focused BF16 router tests passed changing row counts, both weight layouts,
Graph replay, and the int32/one-million-row boundary; the isolated test
cache held **two cubins** total, one per weight layout. CUDA Graph medians
(old to new) were 23.930 to 22.324 us at 128 rows, 24.351 to 22.760 us
at 1024 rows, and 65.531 to 69.963 us at 8192 rows.

With both fixes mechanically synchronized into the private candidate install,
a second isolated-cache Q8 PD4+4 arm ran the same 32 warmup and full 500
MATH requests. All 500 completed with zero errors; `math_verify` was
382/500 and native MAL 3.86372. The router and both FI adapter kernels were
**one cubin each before and after the formal run**. The formal request
window was 26.729 s for 111,643 completion tokens, 9.988 s shorter than
the adapter-only arm despite 729 more completion tokens. This single-order
comparison strongly supports a request-time JIT cost but is not a repeated,
order-balanced kernel-only or end-to-end gain bound; score differences are
within the previously observed stochastic MATH variation.

The fresh cache also created 36 `_silu_and_mul_mxfp8_quant_tiled_kernel`
cubins during the 32-request warmup, with no additional copies during the
subsequent formal 500 requests. That is a remaining **cold first-request**
JIT risk, separate from the fixed formal-window churn. The private launcher
now accepts `--prefill-triton-cache` and records the resolved path in
Prefill `launch.json`, so future counts can be attributed to one candidate
instead of the shared user cache. Artifacts:
`.../1010_topk_q8_model_candidate/dataset_quality/`
`{q8kv4_jit_dynamic_q8_1011,q8kv4_jit_router_q8_1011}/`
(`triton_prefill/`, `math_warm32/`, `math500/`). The original PD4+4 was
restored after each arm; the final smoke artifact again reports five
alive/idle ports and four successful real P-to-D requests.

### Full GSM8K1314 after the Triton JIT fixes (2026-10-11)

The private FI/CUDA 13.4 Q8KV4 candidate with both dynamic-geometry fixes ran
the frozen 1,314-ID GSM8K manifest (SHA256
`29296ca57d4afe241aeabe5e0393b0fe5f839e2a1b97977936289b1b892f9d01`).
The 32-request warmup completed without errors. The formal run used greedy
sampling, a 512-token cap, disabled cache reuse, and eight clients per Decode
DP (32 total). It completed **1,314/1,314 with zero request errors**, reaching
`[8,8,8,8]` peak per-DP client concurrency. Its request window was **37.332 s**
(about 35.20 requests/s), native MAL was **4.14548** with strict per-rank
metric/PID validation, and finish reasons were 1,298 stop and 16 length.
The inherited string scorer marked 1,113 correct; the established
`math_verify.parse+verify` rescore marked **1,224/1,314 (93.15%)** correct,
counting length truncation wrong. This is one Q8 run, not a paired BF16
noninferiority result; the earlier FI13.4 BF16 full run scored 1,236/1,314
under the same rescorer, but was a separate run before these JIT fixes.

In the isolated Prefill Triton cache, `_prepare_topk`,
`_clear_unwritten_page_tail`, and `_prefill_bf16_router` each produced exactly
**one cubin** across the warmup and full formal run. Artifacts:
`.../1010_topk_q8_model_candidate/dataset_quality/`
`q8kv4_gsm1314_jitfinal_q8_run2_1011/{gsm8k_warm32,gsm8k1314,triton_prefill}/`.
The first launch attempt on the host failed before service startup because
that host Python lacks `requests`; the successful launch ran in the candidate
container. After the run, the original resident container was restored. Its
start script requires absent `process.json`/`console.log`, so the stale files
from its prior stopped instance were archived unchanged before restart.
`restore_health_and_smoke.json` confirms five alive/idle ports and four
successful real P-to-D eight-token requests.

### GSM8K formal-window JIT audit beyond the fixed kernels

The same isolated cache shows that the 32-request warmup did **not** cover
every RTP Triton specialization. From the formal request start at 06:06:47
through completion, 13 additional cubins appeared: seven
`_scatter_wire_multirow`, three `_silu_and_mul_mxfp8_quant_tiled_kernel`,
and one each of `_pack_wire`, `_topk_to_block_table_kernel`, and
`_silu_and_mul_post_quant_dense_packed_kernel`. Their mtimes span 06:06:48
to 06:07:08, within the 37.332 s request window. The three FI module
`.ninja_log` mtimes remained at 05:00, so this is RTP Triton activity, not
new FI MSA/RoPE compilation. Cubin timestamps identify compilation, but
do not measure per-request stall or prove that all 13 were on the critical
path.

The tiled SwiGLU kernel has a 12-tile-pair by three-warp (36-config)
`triton.autotune` sweep keyed by hidden width. Its warmup generated 36
cubins; the three late variants have a `T` divisible-by-16 compiler hint
absent from the warmup variants. The wire scatter kernel passes `ROWS`,
`SOURCE_ROWS`, `WORK_PAGES`, and `PERSIST_PAGES` as ordinary integer
arguments; the seven late TTIR signatures vary their divisible-by-16 hints.
The block-table TopK late variant changes the score-head-stride alignment
hint. These observations point to a finite alignment/shape-specialization
tail rather than evidence of unbounded per-request JIT, but a production
image still needs a warmup corpus covering the live alignment classes or
measured runtime-scalar changes. Do not call the 32-request warmup a complete
no-JIT gate for all older operators. The three adapter/router kernels fixed
above remained one cubin each throughout the same window.

### CP wire alignment-specialization fix (2026-10-11)

The seven late `_scatter_wire_multirow` cubins above differed in the
divisible-by-16 compiler hints for `ROWS`, `SOURCE_ROWS`, `WORK_PAGES`, and
`PERSIST_PAGES`. Those four values are runtime bounds/address inputs; the
layout strides, `FI_WORKING_LAYOUT`, and tile width remain structural
compile-time parameters. The candidate now marks the four integers
`do_not_specialize`; `_pack_wire` similarly leaves only `ROWS` dynamic.
The private installed file was synchronized byte-for-byte with the source
(SHA256 `6b9f58b3062c4a1c5364d9a03855d86cf0a9eb24c10db3ddbea1cb52d29b5a08`),
with its former version archived as `nvfp4_cp_wire.pre_dynamic_scalars_1011.py`.

A new GPU test compared scatter output byte-for-byte with an independent
wire-to-page reference across four row/source/page geometries, actual
cross-page slots, and both MMA and FI working-scale layouts. It also checked
pack prefix equality across two row counts and compared packed bytes with
the independent CPU `reference_quantize` oracle. The initial test passed
2/2 on two processes; the strengthened oracle/cross-page version passed
2/2 on another process. The isolated cache had one pack cubin and one scatter cubin
per working layout, with unchanged mtimes on the second run. The existing
CPU wire contract suites passed 36/36. Old installed code versus new source
CUDA Graph hot medians (us) were:

| Rows | Pack old/new | Scatter old/new, FI layout |
|---:|---:|---:|
| 256 | 2.236 / 2.239 | 1.850 / 1.879 |
| 4,096 | 20.002 / 19.942 | 6.071 / 6.032 |

The model gate used the same private FI 13.4 Q8KV4 wheel, TopK route and
exact-service-environment FI workspace, changing only this CP wire module.
The first warmup attempt was rejected by the initial health gate because
Prefill `/worker_status` timed out; it sent no scored requests. After
Prefill became idle, the 32-request retry passed with zero errors. Its
isolated Triton cache held one cubin each for pack, scatter, adapter
prepare/tail, and the BF16 router. A subsequent full MATH500 run completed
**500/500, zero request errors**, reached `[8,8,8,8]` per-DP client
concurrency, scored **378/500**, and passed strict native MAL at **3.85929**.
The formal request window was **26.578 s** for **111,153 completion tokens**;
all five named cubin counts remained one before and after formal traffic.
The prior router-fix arm took 26.729 s for 111,643 tokens. These two
single-order runs show no obvious hot-path regression, not a matched
end-to-end speedup or a quality-equivalence proof. Other older kernels can
still add finite alignment variants, and final-image cold/warm coverage
remains open.

Artifacts: `.../1010_topk_q8_model_candidate/triton_cp_wire_dynamic_{test,bench}_1011/`
and `dataset_quality/q8kv4_cpwire_dynamic_q8_1011/`. After this arm, the
original resident PD4+4 again passed five idle/alive endpoints and four
real eight-token P-to-D smoke requests; see the updated
`restore_health_and_smoke.json`.

### Working-tail scratch-stride JIT fix (2026-10-11)

After the CP wire fix, the next isolated MATH500 formal window still created
four RTP Triton cubins: Gemma norm/RoPE one, dense SwiGLU post-quant one,
and `_clear_packed_working_tail_scales_kernel` two. The tail-scale kernel's
warmup versions had `SCRATCH_SEQ_LEN=256,512`; the formal-window versions
had `768,1024`, as seen in their TTIR page-address multipliers. Source used
`SCRATCH_SEQ_LEN: tl.constexpr` even though this request-dependent,
page-aligned value only calculates `(batch * SCRATCH_SEQ_LEN + length)/128`.
That is a concrete new-bucket JIT risk, distinct from FI Ninja recompilation.

The candidate now marks `SCRATCH_SEQ_LEN` non-specialized and passes it as a
runtime scalar while preserving compile-time page strides, scale layout,
head/group counts, and tile width. Installed candidate and source hashes
match (SHA256 `4892cde5765360efef50f9f51a749dc7246f27c1a16c9394fd7e00c0b39a444d`);
the former installed file was archived as
`nvfp4_kv_cache.pre_dynamic_tail_stride_1011.py`. A GPU test checked exact
K/V/index scale bytes for scratch strides 256/512/768/1024 under both MMA
and FI working layouts. The existing ragged-tail/capacity and CUDA Graph
replay test also passed. Both tests passed again in a separate process;
the isolated cache held one cubin per layout with unchanged mtimes. Old/new
CUDA Graph hot timings (us) were B20/stride256 1.162/1.342,
B60/stride1024 1.464/1.453, and B20/stride16384 1.335/1.332. The small
stride has a 0.18 us single-run cost; these isolated numbers are not an
end-to-end performance bound.

With this file synchronized into the same private FI13.4 Q8KV4 candidate,
32 MATH warmup requests passed with zero errors. Its isolated Prefill
cache had one tail-scale cubin before formal traffic. The full MATH500
window completed **500/500 with zero request errors**, scored **375/500**,
reached `[8,8,8,8]` per-DP client concurrency, and passed strict native
MAL at **3.86662**. It took **26.550 s** for **111,477 completion tokens**;
tail-scale, wire pack/scatter, FI adapter prepare/tail, and BF16 router
cubin counts all stayed one. The immediately prior CP-wire-only window
was 26.578 s for 111,153 tokens: consistent with no obvious model hot-path
penalty, but not an order-balanced gain or precision-equivalence claim.
This formal window still created three other RTP Triton cubins
(`_gemma_norm_rope`, `_silu_and_mul_post_quant_dense_packed_kernel`,
`_pack_nvfp4_inputs_vector_kernel`), so an all-operator no-JIT gate is
not yet met. The exact-environment FI RoPE/BF16 MSA/Q8 MSA `.ninja_log`
mtimes remained unchanged since 05:00.

Artifacts: `.../1010_topk_q8_model_candidate/`
`triton_tail_stride_dynamic_{test,bench}_1011/` and
`dataset_quality/q8kv4_tail_stride_dynamic_q8_1011/`. The original
resident PD4+4 was restored again; the latest
`restore_health_and_smoke.json` records five alive/idle ports and four
successful real P-to-D eight-token requests.

The three remaining late cubins in this run are finite structural variants,
not request-length scalar specialization. `_gemma_norm_rope` already leaves
`ROWS` non-specialized; its wrapper selects `GROUP=1` for small/contiguous
outputs and `GROUP=8` for large in-place Prefill (`rows>=32768`, or
`rows>=2048` without contiguous outputs). The warm and late TTIR differ in
that row tile. Dense SwiGLU post-quant selects `GROUPS_PER_CTA=4` only for
the measured direct-TMA BF16 case with `64<=T<=128`, hidden width
3072/12288, alpha 1.702, and clamp 7; otherwise it uses 1. The two TTIR
shapes are respectively one group and four groups. The MoE vector packer
chooses `BLOCK_M=8/16` for enumerated small token buckets and 64 for
`tokens>=4096`; the warm cubin in this cache has 64 rows per tile and the
late cubin has 16. These branches are deliberate performance policies and
were left unchanged. They do mean a 32-request warmup is not a complete
first-request-JIT gate: a final-image warmup must hit the applicable branches
or ship matching precompiled binaries, then demonstrate no new cubins during
the measured live workload. No FI `.ninja_log` changed in this model run.

The MoE packer gap has a source-level cause and a formal-window timestamp.
`GLM5MegaMoE._maybe_warmup_jit_once()` calls
`generate_mega_moe_jit_token_counts()`, which selects one token count per
DeepGEMM `(block_m, store_block_m, experts_per_wave)` signature. The NVFP4
input packer independently selects vector `BLOCK_M=8` for tokens 20/25,
`BLOCK_M=16` for 40/60/80/100, and `BLOCK_M=64` for tokens >=4096.
The warmup generator does not include these packer tile signatures in its
coverage key, so DeepGEMM bucket coverage does not guarantee packer coverage.
In the Q8 MATH500 cache, the `BLOCK_M=64` packer cubin appeared at
06:46:10 CST, whereas `BLOCK_M=16` appeared at 06:48:32 CST. The formal
request window started at 06:48:21.831 CST, proving a packer compile during
measured traffic. This is a finite missing-warmup branch, not the unbounded
request-scalar specialization fixed above. No isolated compile stall was
attributed to one request; final-image qualification must cover the 8/16/64
packer branches and both Gemma/SwiGLU structural branches before accepting a
zero-postwarm-JIT claim. Do not infer that the entire 26.550 s MATH window
was slowed by this one compile.

An uncommitted follow-up in `mega_moe_nvfp4.py` now runs a packer-only
initialization warmup after the existing DeepGEMM warmup. It selects the
smallest reachable token counts for the default fused packer's vector
`BLOCK_M=8/16` policies (20 and 40 for the production 6144-hidden/topk4
shape), skips nonmatching shapes and explicit tile overrides, and reuses the
existing MegaMoE buffer. It does not add a DeepGEMM forward for these two
counts or alter the hot-path dispatcher. The existing max-token representative
continues to cover `BLOCK_M=64` when the configured capacity reaches 4096.
The packer-only step runs per NVFP4 instance rather than inheriting the
base class's one-time DeepGEMM key; that key omits packer mode and override,
so a torch/override-to-fused configuration switch would otherwise skip it.

Isolated validation in the private candidate container: CPU dispatcher tests
8/8, full NVFP4 packer GPU tests 6/6, and `py_compile` of all four scoped
Python files passed. A fresh one-GPU Triton cache at
`.../1010_topk_q8_model_candidate/packer_jit_warmup_v2_1011/` contained exactly
two vector packer cubins immediately after the model warmup method; their TTIR
row ranges were 8 and 16. Subsequent direct packer calls with 20/40/60/100
tokens left the cubin count at two. The base DeepGEMM warmup was mocked in
this isolated cache check to prove the *additional* packer work independently;
the real class method still calls it first. This fix has **not** been synced
into the installed service or tested in a full cold PD4+4 startup, so the
final-image no-postwarm-JIT and startup-cost gates remain open.
An earlier fresh-cache packer-only timing probe (before the configuration-
switch review fix, with identical two-kernel work) took 1.139 s cold and
0.546 ms hot in one process; these are initialization costs, not request
latencies or a full-model startup measurement.
The four-file implementation received a fresh read-only RTP review and delta
re-review with no remaining actionable findings. Its configuration-switch
regression also passed when the outer test process set
`GLM5_MEGA_MOE_JIT_WARMUP=0` and
`GLM5_MEGA_MOE_NVFP4_PACK_BLOCK_M=4`; the test fixes its enabled-state
environment explicitly. The review did not run GPU or inspect a production
image, so it is not an E2E signoff.

The other two late structural branches were independently exercised in a
fresh single-GPU Triton cache (`.../1010_topk_q8_model_candidate/`
`remaining_structural_jit_v2_1011/`) without changing runtime code. The
MiniMax Prefill producer form of Gemma norm/RoPE used 64 Q heads, four KV
heads, four index heads and `[T,9856]` BF16 projection storage. Warmup at
T=32 and T=2048 produced its `GROUP=1/8` cubins (four/one warp); subsequent
T=33,2051,4096 calls left the count at **2**. Dense SwiGLU post-quant used
BF16 `[T,6144]`, MXFP8 group32, packed TMA scales, alpha 1.702 and clamp
7.0. Warmup at T=32 and T=64 produced `GROUPS_PER_CTA=1/4`; T=48,96,128
left its cubin count at **2**. The output scale is packed `[T,24]` int32,
not uncompressed `[T,96]`; the first fixture failed only its incorrect
shape assertion and was rerun from a new cache with that assertion corrected.

A separate fresh-cache timing probe (`remaining_structural_jit_timing_1011`)
measured cold/hot callable durations including compilation, Python,
allocations and synchronization: Gemma GROUP1 888.773/0.371 ms, Gemma GROUP8
1059.162/0.345 ms, SwiGLU GROUP1 763.597/0.166 ms, and SwiGLU GROUP4
110.668/0.145 ms. These isolate a plausible first-request stall if a branch
is not prewarmed; the hot figures are **not** kernel-only latency. No startup
prewarm or AOT package was changed for these two kernels, so final-image
coverage and actual model TTFT impact remain unverified.

### Final-candidate full GSM8K1314 and BF16 comparison (2026-10-11)

After the CP-wire and working-tail JIT fixes, a fresh Q8/BF16 comparison used
the same private FI13.4 package and installed model code, frozen 1,314-ID
GSM8K manifest (`29296ca57d4afe241aeabe5e0393b0fe5f839e2a1b97977936289b1b892f9d01`),
greedy sampling, max 512 output tokens, no cache reuse, and 32 concurrent
clients (eight per Decode DP). Each arm had 32 successful warmup requests.
The Prefill launch records had identical commands and source hashes; their
environment maps differed only in the isolated `FLASHINFER_WORKSPACE_BASE`.
The quality arm selected Q8KV4 versus BF16. Strict native MAL passed in every
formal window.

| Formal window | Complete / errors | Quality score | MAL | Request window | Completion tokens |
|---|---:|---:|---:|---:|---:|
| Q8 GSM8K1314 | 1,314 / 0 | 1,242 / 1,314 (94.52%) | 4.16104 | 36.028 s | 146,666 |
| BF16 GSM8K1314 | 1,314 / 0 | 1,224 / 1,314 (93.15%) | 4.16909 | 35.141 s | 143,292 |
| Q8 MATH500 | 500 / 0 | 375 / 500 (75.0%) | 3.86662 | 26.550 s | 111,477 |
| BF16 MATH500 | 500 / 0 | 387 / 500 (77.4%) | 3.86270 | 27.068 s | 110,346 |

GSM uses the same `math_verify.parse+verify` numeric equivalence as MATH500;
truncated answers count wrong. The older string-only GSM scorer reported
1,130/1,314 Q8 and 1,127/1,314 BF16, so its numbers must not be mixed with
the math-equivalent scores above. Same-ID GSM outcomes were 1,200 both
correct, 42 Q8-only, 24 BF16-only, and 48 neither; only 62 complete response
texts matched. Q8 led by 18 items in this ordered pair, but a previous
FI13.4 full GSM pair went the other way (BF16 1,236; Q8 1,225). This is
sampling variation, not evidence that Q8 improves accuracy.

The MATH500 arms used the same code snapshot but were run in separate
processes: Q8 immediately before this GSM pair and BF16 after the BF16 GSM
run. Same-ID outcomes were 348 both correct, 27 Q8-only, 39 BF16-only, and
86 neither; only 17 full response texts matched. Q8 trails BF16 by 12 items
in this comparison. Among the 407 IDs where **both** responses ended with
`stop`, Q8 scored 357 and BF16 365 (21 BF16-only, 13 Q8-only); among 37
both-`length` IDs, the scores were 3 and 7; among 56 IDs with different
finish reasons, both arms scored 15. Thus the net Q8 deficit is not solely
a truncation artifact, and the both-`stop` BF16-only IDs are the highest-value
inputs for a first-divergent-layer/logit trace. The formal durations above
are **not** a kernel-speed A/B: output-token counts, process order, and cache
state differed. Dedicated fixed-shape operator/model performance evidence is
reported earlier.
Example both-`stop` BF16-only IDs in the frozen MATH manifest are
`test/algebra/1184.json`, `test/algebra/2626.json`, and
`test/algebra/509.json`; repeat them before treating any one as a persistent
precision regression.

### Complete high-concurrency GSM8K repeat and tiled-SwiGLU JIT check

The `q8kv4_tiledprewarm_q8_1011` PD4+4 candidate used the same frozen
GSM8K1314 manifest, greedy sampling, 512-token limit, disabled cache reuse,
and eight workers per Decode DP (32 peak active clients). The formal run
`dataset_quality/q8kv4_tiledprewarm_q8_1011/gsm8k1314_w32/` completed
1,314/1,314 calls with zero request errors in a 35.426 s request window;
all four lanes reached eight active clients. It generated 1302 `stop` and 12
`length` responses. Native MAL was 4.160018 and passed the strict per-rank
report/bucket check. The legacy string scorer gave 1,121/1,314 (85.31%).
The saved responses were rescored with the same `math_verify.parse+verify`
numeric-equivalence script used above: 1,235/1,314 (93.99%), with truncated
answers counted wrong. Both score artifacts and the manifest hash are in the
run directory. Against the earlier BF16 full run on this *same manifest and
generation contract*, the numeric result was 1,227/1,314; same-ID outcomes
were 1200 both correct, 35 Q8-only, 27 BF16-only, and 52 neither. The legacy
BF16 score was 1,132/1,314, so its opposite ordering illustrates why the
scorer must be named. Different service processes and output-token totals
still prevent a causal precision or throughput conclusion from this pair.

This run followed a MATH500 formal window on the same candidate service.
The isolated Prefill Triton cache had no `.cubin` mtime inside the GSM formal
request window, but this is a **warm-cache** observation: it does not prove
that all first-request JIT is eliminated. The installed model package in this
run included startup prewarm for the Gemma, small SwiGLU and tiled MXFP8
SwiGLU structural shapes. Its tiled prewarm generated 36 autotune candidates
at T=8192 before business requests; the first MATH500 formal window still
generated one tiled cubin for an unaligned runtime T plus four other
structural cubins. The source-only follow-up changed the tiled kernel to
`@triton.jit(do_not_specialize=["T"])`. On GPU0 with a fresh independent
Triton cache, T=8192 generated 36 cubins; a subsequent T=1537 kept the count
at 36, and the existing Torch oracle for T=1537/H=3072 passed. This exact
source patch was **not** overlaid into the installed package for the GSM run.

After the campaign, the candidate launcher exited and ports
23100/23101/24100/24109/24118/24127 were free. The original
`m31_decode_next_restored_pd4p4_framework_overlap_1010` container was
restarted. Five worker-status endpoints reported alive/idle; four real
64-input/8-output P-to-D requests passed with `pd_sep=true`, and the endpoints
remained idle afterward. The restore evidence is
`1010_topk_q8_model_candidate/restore_health_and_smoke.json`.

Remaining gate: install the `T` non-specialization patch in an isolated
candidate image, test cold first-request and repeated full-model windows,
and account for the two TopK block-table and two fused-RMSNorm structural
cubins seen in the prior MATH formal window. Review the final scoped code
patch before release; this isolated test and the warm-cache GSM run are not a
production-image JIT signoff.

The current TTIR signatures narrow the four remaining cubins. The two
`_fused_add_rmsnorm_fp8_quant_*singlepass_kernel` families each have an
`int8` scale-pointer version (packed/TMA scale bytes) and an `fp32` version
(row-major MXFP8 scales). These are distinct ABI/output layouts, not merely
request-length specialization. The `_topk_to_block_table_kernel` cache has
three signatures across smoke and first MATH formal: one has aligned
`stride_s_h` and no runtime `num_pages`/`stride_s_n`; two include those
runtime scalars and differ in the compiler's divisibility hint for
`stride_s_h`. The score tensor layout varies between branches. A targeted
prewarm must cover the actual model layouts, then confirm no new signature
under fresh-cache model traffic; forcing all branches into one generic
kernel without a matched hot-latency check would be unjustified.

### Installed `T` non-specialization full-model check

The source `activation.py` (SHA256
`8307428bef668ea5d7a97550b349dd001d5a2c113399cecc2faacb6fc264f1a3`)
was mechanically overlaid into the private candidate's installed package,
after backing up its previous file under `prewarm_overlay_backup_1011/`.
The new arm `dataset_quality/q8kv4_tilednospec_q8_1011/` used a fresh
Prefill Triton cache and the exact prewarmed FI13.4 workspace from the prior
campaign. Before business requests, the model startup generated 36 tiled
SwiGLU autotune cubins, two Gemma, two small SwiGLU and the expected MoE
packer variants. Five service endpoints were alive and idle. Four real P-to-D
smoke requests completed; the first cold request took about 7.53 s and the
next three about 0.60-0.62 s. That first-request difference is not attributed
to one kernel without a timeline.

| Formal window, same PD4+4 process | Calls / request errors | MATH500 | Strict MAL | Request window | New Prefill Triton cubins |
|---|---:|---:|---:|---:|---|
| First | 500 / 0 | 383/500 | 3.857915 | 28.534 s | two TopK block-table and two fused-RMSNorm layout/scale variants |
| Repeat | 500 / 0 | 378/500 | 3.856563 | 25.412 s | none |

Both reached `[8,8,8,8]` peak active clients and had complete strict native
per-rank reports. Tiled SwiGLU remained at **36 cubins** through the first
formal window, eliminating the previously observed unaligned-T late cubin
in this workload. No `.cubin` mtime fell inside the repeat window. FI RoPE,
BF16 MSA and Q8 MSA `.ninja_log` mtimes remained at the earlier prewarm,
before either formal run. The 3.122 s first/repeat request-window gap is not
a JIT latency estimate: output texts, finish reasons and token totals differ.
The first window's four structural compiles and the cold-smoke stall remain
release risks. Neither arm is a final production image.

The cold-smoke cache timeline supplies stronger evidence of that risk:
between 08:34:06 and 08:34:13 local time, the first real P-to-D request
created about 15 Prefill Triton cubins, including score, TopK, CP wire,
working-tail, RMSNorm, router and page-commit kernels. The first request's
7.53 s cost and those cubins overlap in time; this establishes cold-cache
JIT exposure, not a per-kernel causal latency attribution. In contrast,
startup had already completed the 36 tiled SwiGLU autotune cubins before
business traffic. Any release image without a verified Triton cache package
or representative startup prewarm may move substantial compilation back onto
its first real request, even though steady-state repeat traffic was JIT-free.

A fresh independent read-only RTP review covered the incremental JIT patch
in `minimax_m31.py`, its new startup test, the MoE packer/warmup and tests,
`activation.py`, CP wire/working-tail and related tests. It found no confirmed
code blocker in that scope. The reviewer did not run GPU tests or review the
remaining TopK/RMSNorm structural kernels as part of this incremental pass,
and did not sign off a final deployment image or output equivalence. Thus
the reviewed JIT subset is stable as an experiment, not production-complete.

`--warm_up 1` is not a drop-in mitigation for this service. The tested
`minimax_m31_vl` launch explicitly used `--warm_up 0`, and each Prefill rank
logged `NormalEngine.cc:136 skip warm up`. The generic C++ warmup guard also
requires `!model_config_.mm_model_config.is_multimodal`, so simply toggling
the flag for an M3.1-VL image would still need source/config verification and
would not automatically exercise the actual paged sparse Prefill/PD path.
The release preflight should therefore keep a new instance **out of external
traffic** while running real representative P-to-D requests (short first
request plus long/prefix-reuse and the TopK/RMSNorm layout branches), then
check that a repeated same-shape window creates no new Triton cubins and no
FI Ninja work. Only after health, real PD output, precision/MAL and cache
checks pass should it enter service. This is a proposed deployment gate, not
a preflight facility already integrated into the service.

### Final scoped review delta and validation boundary

A complementary fresh read-only review of the current Q8KV4/TopK integration
confirmed two small issues outside the prior JIT-only review. Opt-in Q8 and
TopK shadow diagnostics synchronized GPU tensors and would conflict with a
future CUDA Graph capture; the current Prefill service has Graph disabled,
so this was not an observed production failure. The Q8 CUDA kernel used
`(kv_len + 127) >> 7` before clamping to the page-table width; an inconsistent
near-`INT_MAX` length could overflow that signed addition, although the
configured model limit is 1M tokens. The fixes skip shadow comparison during
capture with a one-time warning, without consuming its diagnostic limit, and
compute ceil-pages by quotient plus nonzero remainder before clamping.

Focused CPU tests passed: Q8 shadow/dispatch 4/4, TopK shadow 4/4, and
TopK dispatch 7/7. An arithmetic check covered lengths 0, 1, 127, 128,
129, 1,048,576, 2,147,483,520, 2,147,483,521 and `INT_MAX`; all matched
the mathematical ceil-divide and capped at the page-table capacity. The
modified `.cu` was compiled from the current worktree path by the private
CUDA 13.4.92 FI JIT in a fresh workspace and the RTP-writer-scale FP32
operator oracle passed on GPU0 (1/1). The original resident PD4+4 stayed
alive/idle throughout. A fresh delta-only read-only re-review found no
confirmed blocker; neither reviewer independently reran GPU tests.

The three changed runtime files now byte-match the private candidate install,
with the prior installed versions saved under `review_delta_backup_1011/`.
The earlier full GSM/MATH model campaigns predate this final delta; the
following fresh candidate arm validates the final installed bytes. Real
capture-mode and `INT_MAX` GPU input tests, extreme-scale Q8 precision,
first-request JIT mitigation, and final-image qualification remain open.

### Final-byte full-model replay and stale FI-cache recompilation

`dataset_quality/q8kv4_reviewdelta_q8_1011/` ran the private installed
candidate after the three reviewed runtime files were synchronized from the
worktree. It used the same frozen MATH500 and GSM8K1314 manifests, greedy
sampling, disabled cache reuse, 512-token cap, normal rejection, PD4+4,
and eight client workers per Decode DP (32 total). The Prefill Triton cache
was empty at launch; the FI13.4 workspace was the same previously prewarmed
workspace. Five endpoints reached alive/idle. Four real P-to-D smoke calls
passed, with `pd_sep=true` and eight output tokens each.

| Final-byte formal window | Complete / errors | Score | Strict native MAL | Request window | Peak clients per DP |
|---|---:|---:|---:|---:|---|
| MATH500 | 500 / 0 | 383/500 | 3.850760 | 28.523 s | `[8,8,8,8]` |
| GSM8K1314 | 1314 / 0 | 1230/1314 (93.61%, math-equivalent; `length` wrong) | 4.161585 | 34.479 s | `[8,8,8,8]` |

The inherited GSM string scorer gave 1116/1314 (84.93%); it must not be
compared with the math-equivalent score. The same-manifest older BF16 numeric
run gave 1227/1314, but different processes and changing outputs prevent an
accuracy-superiority claim. The final-byte MATH/GSM windows had respectively
434/66 and 1302/12 `stop`/`length` finish reasons. MATH's first formal
window generated two TopK block-table and two fused-RMSNorm structural
cubins; the subsequent GSM formal window generated no Prefill Triton cubin.

The first smoke request took 18.615 s, then the remaining three took about
0.59-0.62 s. The Q8 FI module's `.ninja_log` in the *reused* FI workspace
changed from its 05:00 prewarm timestamp to 09:01 during that first request,
while BF16 MSA and RoPE logs stayed unchanged. The changed Q8 `.cu` source
invalidated the old generated recipe and forced a new FI JIT compile. This is
a concrete upgrade hazard: an existing FI cache directory is not enough when
source/toolchain changes. The 18.615 s whole-request cost includes other
first-request work, so it is not an isolated Q8 compilation duration.
Production preflight must use the exact final source/toolchain to build the
FI cache **before** external traffic and verify `ninja -n -d explain` reports
no work for every relevant module. The final image has not passed this gate.

### Reproducible exact-source FI prewarm after a Q8 source change

The stale-cache failure above was followed by a positive control using the
same final installed bytes and launch environment, but a **fresh** workspace
`1010_topk_q8_model_candidate/fi134_finalsource_prewarm_1011`. Run the
following from the host after choosing the final candidate's Prefill
`launch.json`. The script reads structured JSON; it does not assume that
`jq` exists inside the container. Keep the workspace out of public traffic
until both this step and the real-request check pass.

```bash
python3 - <<'PY'
import json
import subprocess
from pathlib import Path

root = Path('/data0/ruixuan.zrx/minimax-m31-dev/1010_topk_q8_model_candidate')
launch = root / 'dataset_quality/q8kv4_reviewdelta_q8_1011/prefill/launch.json'
workspace = root / 'fi134_finalsource_prewarm_1011'
keys = ('PYTHONPATH', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'CC', 'CXX',
        'CPATH', 'FLASHINFER_NVCC', 'FLASHINFER_EXTRA_CUDAFLAGS')
record = json.loads(launch.read_text())['environment']
env = {key: record[key] for key in keys}
env.update(FLASHINFER_WORKSPACE_BASE=str(workspace), CUDA_VISIBLE_DEVICES='0')
code = '''
import time, torch, flashinfer, flashinfer.rope as rope
from flashinfer.msa_ops import msa_prefill_nvfp4_specialized_warmup
from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.flashinfer_nvfp4_prefill import _q8kv4_module, _q8kv4_target
device = torch.device('cuda:0')
print('FlashInfer', flashinfer.__version__, flashinfer.__file__)
tasks = (
    ('bf16_msa', lambda: msa_prefill_nvfp4_specialized_warmup(device)),
    ('q8_msa', lambda: _q8kv4_module(_q8kv4_target(device))),
    ('rope', rope.get_rope_module),
)
for name, build in tasks:
    start = time.perf_counter()
    build()
    print(name, time.perf_counter() - start, flush=True)
'''
command = ['docker', 'exec']
for key, value in env.items():
    command += ['-e', f'{key}={value}']
command += ['m31_topk_q8_quality_rdma_memlock_1010', 'python3', '-c', code]
subprocess.run(command, check=True)  # Repeat in a second fresh process.
PY
```

On this final source, the first process took 8.076/7.392/24.083 s for FI
BF16 MSA/RTP Q8 MSA/RoPE. The second fresh process took
0.472/0.005/0.008 s. All three module directories under
`$workspace/.cache/flashinfer/0.6.12+250aae19/103a/cached_ops/` reported
`ninja -n -d explain: no work to do`; Q8's generated recipe selected the
private CUDA 13.4.92 nvcc and the final installed `msa_prefill_q8kv4.cu`.
Before model startup, the three `.ninja_log` SHA256 hashes were respectively
`5c9ceb0a...e1fca8f`, `343cd923...435db56`, and
`318713ce...e69631` (BF16/Q8/RoPE). A new PD4+4 process using **that same
workspace** left all three hashes unchanged after startup and after four
real 64-input/8-output P-to-D requests. Its first request took 6.634 s;
subsequent requests took about 0.61 s. The previous stale-source-cache arm
took 18.615 s on its first request and rewrote only the Q8 `.ninja_log`.
These whole-request times are not a controlled FI compile-time A/B; the hash
and Ninja evidence is the decisive cache result. Triton still compiled other
first-request kernels in the fresh service, so this procedure closes only
the tested FI module-rebuild path, not all cold-request JIT.

Artifacts: `fi134_finalsource_prewarm_1011/` and
`dataset_quality/q8kv4_finalprewarm_q8_1011/candidate_pd_smoke.json`.
After the test the candidate exited; the original PD4+4 was restored and
passed five idle endpoints plus four real P-to-D requests, recorded in the
latest `restore_health_and_smoke.json`.

Afterward the candidate launcher exited, all six service ports were free,
and the original PD4+4 was restarted. Five worker-status endpoints were
alive/idle both before and after four successful real P-to-D requests. The
latest proof is `1010_topk_q8_model_candidate/restore_health_and_smoke.json`.

After this experiment the candidate launcher exited, all six ports were free,
and the original PD4+4 container was restarted. Five worker-status endpoints
were alive/idle before and after four successful 64-input/8-output real
P-to-D requests with `pd_sep=true`. The latest restore artifact is
`1010_topk_q8_model_candidate/restore_health_and_smoke.json`.

The Q8 GSM and BF16 GSM isolated Triton caches each retained one cubin for
the six repaired kernels (`_clear_packed_working_tail_scales_kernel`,
`_clear_unwritten_page_tail`, `_pack_wire`, `_prefill_bf16_router`,
`_prepare_topk`, `_scatter_wire_multirow`). Both GSM arms reached peak client
concurrency `[8,8,8,8]`; Q8 stop/length was 1300/14 and BF16 1302/12.
The three structural late-cubin variants above still need final-image
prewarm/AOT coverage. No precision promotion or production-default switch
follows from this test.

Raw Q8 GSM: `.../1010_topk_q8_model_candidate/dataset_quality/`
`q8kv4_finaljit_gsm_q8_1011/gsm8k1314/`.
Raw BF16 GSM and MATH: `.../dataset_quality/`
`bf16_finaljit_gsm_bf_1011/{gsm8k1314,math500}/`.
Raw Q8 MATH: `.../dataset_quality/`
`q8kv4_tail_stride_dynamic_q8_1011/math500/`.
Each directory contains response-level JSONL, scores and strict integrity
records. After the pair, the original PD4+4 resident was restored; the
restore smoke record confirms five idle/alive ports and four real P-to-D
eight-token requests.
