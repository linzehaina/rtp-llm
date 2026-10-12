# MiniMax-M3.1 Decode 空泡、逐算子分析与后续优化清单

日期：2026-10-10。用户要求先查看最新 timeline，再决定优化方向。本轮仅补充源码/trace 分析及独立 Graph launch 用例，不启动模型 E2E，不修改生产优化代码。

## 证据与配置

- 最新 timeline：`/data0/ruixuan.zrx/profiling_results/m31_decode_dp8_bs32_80000_dspark_static4_sequence_staging_20261010`，八个 `bs32_seq80000_decode_wr*_1.json`。
- DP8/TP1/EP8，BS32/card，物理 history 80000，static verify4，gamma7；这是 decode perftest/强制接受诊断，不是自然拒绝采样的质量或生产 MAL 结果。
- frozen Python：`/data0/ruixuan.zrx/minimax-m31-dev/1010_decode_next/gpu_phase_integration/framework_overlap/runtime_sequence_length_candidate/rtp_llm`。主 worktree 的额外 WIP 不等于 trace 中已执行。
- rank0 中间轮 target 的全部 1908 个 kernel 的名称、grid、block 与 `framework_overlap/current_target_operator_ledger/nodes.json` 逐项一致。完整逐算子清单沿用该 ledger，下面补充本次指定的算子及空泡。
- 正式 first-pair candidate 单卡 TPS 5330.42，相对 baseline 5490.66 为 -2.92%；ABBA 被用户暂停，不能声称本次 staging 已有 E2E 收益。此前另一轮 framework ABBA +4.92% 不能归因于本轮新改动。

## Gate dtype：默认代码与这份 trace 必须区分

当前 `models/minimax_m3.py::_router_dtype()` 默认返回 FP32，`M3_ROUTER_BF16=1` 时取消该 FP32 强制转换；M3.1 weight 定义复用这个函数。`GenericMoe._compute_router_logits()` 在 gate weight 为 FP32 时先将 activation 转 FP32。batch-invariant M3.1 router 也检查 FP32 weight。

但本次 trace 对应 `framework_overlap/sequence_length_model_abba01/candidate_a/plan.json` 显式设置 **`M3_ROUTER_BF16=1`**，且匹配节点存在 BF16 split-K epilogue 和 logits BF16→FP32 copy。此前关于 BF16 gate 的说明只适用于这份冻结 trace，不能当作当前默认 FP32 gate 的计算说明。

默认 FP32 gate：BF16 activation → FP32 activation → FP32 gate GEMM → FP32 logits；后面的 `router_logits.float()` 对已经 FP32 的 tensor 无需再启动转换 kernel。因此不能把 trace 的 57 个 logits cast 节点算作 FP32 gate 路径也能消除的节点。后续用 FP32 gate 验证路由融合，保留 checkpoint weight 和计算精度，不以降低 router 精度换性能。

## 1. Target CUDA Graph launch 前空泡

对 CPU API 与 GPU kernel 按 correlation 对齐，rank0 四轮如下，单位 µs：

|轮次|API duration|API begin 到首个 GPU kernel|API end 到首个 GPU kernel|
|---|---:|---:|---:|
|1|3224.386|3222.84|-1.55|
|2|2194.99|2198.62|3.63|
|3|2062.69|2066.15|3.46|
|4|2043.12|2046.73|3.60|

其他 rank 也在 API 返回附近开始，部分在 API 返回前约 0.4–1.4ms 已开始。rank0 第二轮最近的前序 DtoD 在 API begin 前约 1227µs 就结束。因此主要延迟位于 API 调用内部，不能解释为提交完成后 GPU 等待 1–2ms，也不能归因于已完成的 prepareInputData memcpy。

`cpp/cuda_graph/cuda_graph_runner.cc::replayGraph` 关闭诊断时直接 `graph.replay()`，该位置没有新增框架 CPU synchronize。追踪优先级是 replay/driver/profiling 内部成本。

已有 native journal：`framework_overlap/candidate_diag_retry01/target_launch_observer_comparison.json`。正式 profiler-off replay 中位 17.7585µs、thread CPU 17.3925µs；profile active 中位 2097.062µs、thread CPU 2085.341µs。主要是 CPU 工作，不是同线程被挂起的毫秒等待。但此记录测的是带诊断 bookkeeping 的 replayGraph 窗口，而不是裸 API；当次 GPU marker 21/24 缺失，不能用于证明完整 GPU 空泡已解释。

历史 EAGLE1 1893-node target 在 profile 中也有 2.0–2.36ms launch。历史分析没有完成该空泡的性能修复验证；FA4 构造重构/内存优化不能充当 launch 问题已解决的证据。

## 2. Proposal CUDA Graph launch 前空泡

rank0 四轮 API duration 为 342.38、190.76、176.72、181.29µs；API end 到首个 GPU kernel 分别为 5.15、5.82、5.59、5.01µs。proposal 127 nodes，target 1908 nodes。前两处共用一次低开销 profiler-off 验证，而不是分别重启两轮模型。

## 独立 Graph launch 用例

源码、原始 samples、correlation、完整 trace：`/data0/ruixuan.zrx/minimax-m31-dev/1010_cglaunch_singleop`。`probe.cu` 和 `run_probe.py`。

- 构造 1、127、1908 个串行 kernel 节点，每次每节点向同一计数器 +1，检查测量区间计数增量严格为 nodes×launches。
- 原生 CUDA：C++ steady_clock 仅包住 `cudaGraphLaunch`。PyTorch：计时完整 `CUDAGraph.replay()`，包含 Python/C++ wrapper，不能称为裸 API。
- 每个 graph 先首次执行，再预热 10 次。同步在计时区间外，使每次调用面对空 stream，避免积压提交被误判为 API 本身的成本。
- OFF 和 PROFILE 为不同进程；OFF 每项 200 次，PROFILE 每项 12 次。PROFILE 只在预热后启用 CPU+CUDA/record_shapes，并按 correlation 要求每次 launch 有正确数量的 GPU 节点。
- 使用空闲 GPU UUID `GPU-6ad27e1f-f4b0-d701-c8db-517dc7356374`，不停止/重启 resident 服务。Torch 2.11.0+cu130，SM103；native nvcc 13.2/static cudart，Torch runtime 13.0，实际 runtime/driver 信息在结果 JSON，不能假定两套 runtime 完全一致。
- 最终数值见 `off_results.json` 和 `profile_results.json`，本文后附汇总。首次 launch 是单个观察，不能当稳态分布。GPU first-start latency 仅在 profile 中可关联，OFF 本轮只验证 CPU 提交耗时。

用例可验证逐节点 profiling 是否足以产生类似毫秒 API 时长，但 synthetic 单 stream 图不是模型图；不包含真实 memcpy/event/MegaMoE/DP 依赖，不能取代模型 E2E，也不能把 profile 差值直接从模型延迟里扣掉。

官方资料：

- [NVIDIA CUDA 12.6 Graph 实测](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)：重复纯 kernel 串行图约 2.5µs+1ns/node；首次上传和非 kernel/多 root 图有不同成本。其硬件数据不是本机性能阈值。
- [NVIDIA profiling 建议](https://docs.nvidia.com/dl-cuda-graph/latest/troubleshooting/performance-issues.html)：node 粒度有显著 observer overhead，graph 粒度较低。

## 3. Metrics 后 schedule 的真实暴露成本

按该卡所有 GPU kernel 区间做 union，以下不是简单将整个 CPU scope 当空泡：

|rank/选取轮次|schedule µs|GPU kernel 覆盖 µs|未覆盖 µs|
|---|---:|---:|---:|
|rank0 最慢|392.82|206.83|185.99|
|rank2 最慢|329.32|4.26|325.06|
|rank7 最慢|301.32|119.61|181.70|

rank0 后三轮 schedule 77.96、65.92、101.04µs，大部分被 GPU 工作覆盖。因此约 300µs 暴露确实存在，但不是每轮 schedule 固定造成 300µs。

代码链：`MtpExecutor::collectDecodeMetrics` → 同 step consume metrics → executor 返回 → `NormalEngine` 下一轮 `schedule` → `BatchDecodeScheduler` 逐 stream 状态机 → `StreamCacheResource::incrKVBlock` → `KVCacheManager::malloc`。

- metrics event 是 accept_len 汇总及 8B D2H 的侧流就绪事件，不等于全设备 synchronize；同 step 消费仍约束主 CPU 线程进度。
- 每个上述慢轮都有 32 次 incrKVBlock 和嵌套的 32 次 KVCacheManager::malloc。rank2 incrKVBlock 总时长约 210.85µs，占 schedule 329.32µs 的主要部分；嵌套 malloc 时长不能再相加。
- malloc 是 cache key 更新、资源检查、allocator 入口，不意味着 32 次 CUDA allocation。尚余约 118µs 没有细分，不臆测全是 allocator/锁。
- `reserve_step=21` 是 DSpARK 3×gamma7 的资源预留，不是 21 次计算；static verify4 不授权将 reserve 降到4。

## 4. 用户指定小算子的业务逻辑与优化

紧邻 post-attention dual-output RMSNorm 的 `gpu_kernel_impl_nocast…where` 与 O projection 后的 where 是同一处，不能重复计成本。通用 `nocast` 名字本身不提供业务语义，要看完整 functor；该 graph 还有 router_bias、hidden_capture 等 nocast 节点。

|操作|计算/正确性要求|优化方向|
|---|---|---|
|O 后 BF16 FillFunctor|zeros_like，供 where 的零分支|无需单独生成整块零张量|
|O 后 where、dual norm 前 nocast|无效 speculative/padding 行 output=0 后进入残差/norm|将谓词融入 add+RMSNorm；无效行保持原 residual，不能清 residual|
|_fused_query_cast|Cake main Q 仍 BF16，仅 index Q BF16→scale1 FP8 供 IndexScore|融入 norm/RoPE index-Q producer；保留先 BF16 round 再 FP8 cast|
|_prepare_metadata|clamp causal length、校验 logical/physical page、无效行处理、16 selected pages 排序、int32 table/length|与选举融合；进一步改 producer/consumer ABI 才能省其他 adapter 工作|
|long FillFunctor|每调用清 seen(page,head) 选举字段，防旧 winner 污染|generation-tag 候选，需严格验证回绕/复用/Graph replay；不能直接删|
|_selected_valid_prefix|atomic_max 选共享 physical page 的最长可见 prefix 和唯一 owner|与 metadata 生成融合；保留共享页/不同长度语义|
|_refresh_selected_scales|唯一 owner 复制当前 layer 的 selected K/V scales，转换 Cake 布局/V swizzle，无效尾部 scale 清零|量级较大，评估减少 adapter 转换或 consumer 原生读 RTP 布局；不直接删|

scale refresh 不搬完整 KV、不重新量化 KV；每层历史 scale 不同，不能跨层复用上层内容。选举到 refresh 需要全 grid 完成的依赖，直接合成普通多 CTA kernel 没有合法全局 barrier。

rank0 中间轮 GPU kernel sum（profile 时间，不能相加当 E2E 收益）：

|操作|nodes|sum µs|
|---|---:|---:|
|query cast|60|87.52|
|metadata|60|161.25|
|seen zero|60|72.64|
|prefix/owner election|60|99.74|
|scale refresh|60|618.08|
|output fill + where|120|339.52|

### Gate 到 TopK

这份 BF16 trace 顺序：gate GEMM → split-K reduce → logits float → sigmoid → correction bias add → group top2 sum → group/expert select、权重归一化/scaling。

- split-K reduce 合并 GEMM partial，是当前 GEMM 算法必要节点；后续可独立比较非 split-K GEMM，不能直接删除 reduce。
- sigmoid 是 expert score，不是 softmax。
- correction bias 只影响 selection 排序；最终 weights 使用未加 bias 的 sigmoid scores。
- group score 是每组 top2 score 之和，然后选 group、选 expert，按配置 renormalize 和 routed scaling。
- trace 中 logits float/sigmoid/bias/group2 合计 497.32µs/228 nodes。BF16 trace 若四步变一步可少 171 nodes；**当前默认 FP32 gate 没有 logits cast，若 sigmoid/bias/group2 三步变一步，则少114 nodes**。都是候选节点数，不是已完成融合或性能收益。
- shared+routed FFN 输出 add 保留 BF16 round，再在下一层加 residual。进一步融合不能把两次 BF16 rounding 偷换成一次 FP32 sum。hidden_capture 是供 draft 的 hidden+residual 写入，不是可直接删除的重复 residual。

## 待执行清单：不将候选误写成已完成

|优先级|事项|当前状态|验证与接受条件|
|---|---|---|---|
|P0|确认 graph profiling 对 launch 的放大|独立单算子已采集，结果见附件|裸 API/PyTorch 两路径、计数正确、trace 节点与 correlation 完整|
|P0|原模型 profiler-off 的真正 GPU 启动空泡|未执行|冻结版本和 workload，低开销 native clocks/GPU 边界，区分前序工作结束、API begin/end、首节点；不复用覆盖失败的 GPU ring|
|P1|KV 资源管理无变化快速路径|未执行|逐请求需求不变时减少重复工作；跨 block/滑窗回收/结束/错误/资源不足仍走完整路径，验证 slot/KV reuse 和 drain|
|P1|metrics 与下一轮计算解耦|仅设计候选|带 step 身份的 pending 快照、event ready 消费、最后一轮 flush；保持指标窗口/统计正确性，不直接删 sync|
|P1|output mask 融合进 post norm|已有独立算子 WIP，不在此 trace|padded/无效行/动态 mask/Graph 输入变化、残差 BF16 等价；收益成立后再模型验证|
|P1|FP32 gate 的 sigmoid/bias/group2 融合|未执行|默认 FP32 路径，用真实形状比较 expert IDs、ties、weights；不能沿用 BF16 cast 节点预算|
|P1|index-Q cast 融入 norm/RoPE|未完成模型验证|BF16→FP8 舍入、strides/tails/canary/Graph 和 IndexScore/TopK 一致|
|P2|metadata + owner election 融合|未执行|共享 physical page、不同 causal length、invalid page、empty rows、复用和 Graph 动态输入|
|P2|scale layout adapter 降成本|未执行|保持 RTP KV ABI、PD transfer、dirty refresh、页复用、invalid-tail 防 NaN；先单算子与内存预算|
|P2|seen generation tag / shared add / router GEMM selector|未执行|分别独立验证代次回绕、BF16 rounding、GEMM 精度与实际总成本，不合并归因|

顺序沿用 prefill 优化文档：从真实 producer/consumer 消掉中间内存访问和 launch，先测正确性及链路收益；模型 profiler-off 性能与真实 GPU 暴露成本分开；候选收益成立/bugfix正确后才一次独立 completion review。用户 review 后才继续未授权的模型优化/E2E。

生产优化代码、本轮模型 E2E 和原模型 profiler-off GPU 启动延迟仍未验证。独立 Graph 用例的毫秒 profile 开销不能换算为生产 TPS 收益。

## Graph launch 单算子最终结果

最终源码完成后的两个进程均正常退出；每项 OFF 200 次、PROFILE 12 次，全部计数和 profile correlation/node 数断言通过。以下为中位数，单位 µs。

|backend|kernel nodes|OFF CPU|PROFILE CPU|PROFILE API返回→首kernel|
|---|---:|---:|---:|---:|
|native|1|1.538|5.209|3.423|
|torch|1|2.865|7.560|3.058|
|native|127|2.157|210.332|7.453|
|torch|127|3.546|211.029|7.368|
|native|1908|3.442|3113.726|5.941|
|torch|1908|4.777|1913.220|6.117|

原生 1908-node 的 OFF 首次 launch 为 1014.845µs（单次观察）；预热后中位 3.442µs。首次上传和重复 replay 必须分开。

结果支持：profiling 本身足以在该主机制造 target/proposal 所见量级的 API 内部耗时。并不证明原模型关闭 profiling 后没有其他空泡，更不代表获得毫秒级生产收益。native CUDA runtime=13020，driver API version=13000，Torch CUDA=13.0；native static cudart 与 Torch runtime 分开，不能把两种 backend 的绝对差值解释为 Python overhead 的精确值。

## 2026-10-11 最新 Decode 优化的组合检查

本轮只读源码并做临时文件的三方文本合并检查；未合并工作树、未启动 GPU/E2E。远端 ls-remote 已确认：brucelee `feat/minimax_m3.1` 和 `feat/m31-decode-champion-kernel` 均为 `a93a77d6477ea475402f37b75c40547c6fa41ed1`；internal `feat/minimax_m3.1` 为 `80bde5c11129a94a9055167ec2a60c396cb9f033`，额外为 prefill Champion 短请求 smoke。

比较基线为我们 HEAD `208f1d2f3f` 加当前 uncommitted WIP；新分支从同一 HEAD 演进。Decode 主改动为 `21194fcab8` native attention、`bc12d07d06` strided Graph reserve table、`23475e535d` rows16–31 dispatch threshold。其他大量新增文件属于 prefill TopK/our_score/Champion/FI 后端及 JIT 工作，不应当作同一个 decode 优化全量引入。

### 可以组合的部分

- 新分支没有修改 C++ scheduler/AsyncRunner/MtpExecutor/Graph runner/compact vocab mask，也没有修改 generic_moe 或 router weight dtype。我们的固定 DP phase、异步 input/输出 bookkeeping、compact mask、previous sequence lengths 和 metrics 调度候选与新 attention 没有直接代码重叠。保留各自独立的正确性/收益归因。
- native 消费 RTP 原有 packed K/V 和 MMA scales，并接受 page stride、strided request table。因此我们的 fused decode KV addressing/writer 与之 ABI 兼容，仍须验证 writer→native consumer、tail、dirty refresh 和页复用。
- O projection 之后的 DeferredMaskedAttentionOutput→post RMSNorm 是 backend 下游契约，可保留。native 不传显式 valid mask，只依赖 seq_lens/used，且 uniform width 从最后一行推导 used；不能据此删除 O 后的业务 mask。必须验证 padded request、零长度和 per-row visibility。
- query cast 融入 Gemma norm/RoPE 的方向仍成立。native 需要 main Q 和 index Q 都为 FP8；Cake 只需要 index Q 的 FP8。现有 `minimax_m31_gemma_rope_` 已有 `query_fp8_outputs` carrier，接入时应按 backend 选择输出，并保留 BF16 round 后再 FP8 cast。
- MoE FP32 router 前处理融合与新 attention 独立；pack tile 优化和 JIT warmup 可结合。新分支的 kernel M 使用 do_not_specialize，warmup 为每个 tile 选一个代表形状，不能把 warmup 等同于所有动态/Graph 行数验证。

### 三个不能直接叠加的点

1. **MXFP8 输出与 native 路由互斥。**我们的 `_use_q8kv4_mxfp8_output` 扩展到 T160；BS32 static4 的 T160 将选 `_output_format=mxfp8`。native supported 明确拒绝非 BF16 输出，所以文本合并后仍不会执行 native。ragged query_cu_seqlens 也明确被拒绝。开 `MSA_USE_NATIVE_HYBRID_DECODE=1` 不能越过这些 gate。
   - 第一阶段可让 native 写 BF16 shape-cached output，再接现有 `_quantize_bf16_attention_mxfp8`，返回 Q8KV4Mxfp8DecodeResult 并沿用 O projection 的 packed scales；先保留已验证舍入边界，测整链收益。
   - 第二阶段才考虑将 packed MXFP8 输出融入 native combine。当前交付是 hash-pinned binary，修改其 GPU 数学/输出 ABI 需要准确可重建的 decode 源码及重新 gate，不能只改 ctypes 返回类型。
2. **BF16 main-Q 与 FP8 main-Q 的计算路径不同。**新选择 native 后 use_cake=False，`fused_query_cast(...cast_main_query=True)` 将 main Q 转 FP8。其 19-cell O bitwise 文档对照的是原 FP8-Q Triton reader，不证明与我们 BF16-Q Cake 路径 bitwise 等价，也不证明模型质量保持。需要针对真实 Q/scales、自然拒绝采样/MAL 和同 ID 质量重新验证；不能直接套用其相对 Triton 的 1.27–3.39x 当作相对 Cake 的收益。
3. **MoE packer 有文本冲突。**我们在 inline tile dict 增加 T160→M16；新代码将 dict 提到 `_SMALL_VECTOR_TILES` 并用于 warmup。应保留新 constant/warmup 结构，把 `160:16` 加入 constant，保留我们的 T160 correctness/Graph tests。覆盖任意一边都会丢另一边能力。

临时三方检查中，q8kv4_decode.py、msa_attention.py、nvfp4_kv_cache.py、minimax_m31.py、CUDA kernels BUILD 和 packer tests 可自动文本合并；packer implementation 是上述冲突。自动文本合并通过不等于运行语义组合通过。

### Cake adapter 优化要按 backend 收缩

native 直接读取 RTP scales，不走 Cake `_prepare_metadata / seen.zero_ / _selected_valid_prefix / _refresh_selected_scales`。这些候选对 native 路径不再有意义；若 native 的 MXFP8 接口和精度门禁通过，则避免在 native 路径重复插入 Cake adapter。保留 Cake fallback 所需实现及测试。一个 ctypes 调用包含 K1 partial 和 K2 combine，不能写成一个 GPU kernel 或据此把 graph 节点数视为1。

### Prefill Champion 的已知失败不能误算到 Decode

`80bde5c111` smoke 和 `/data0/ruixuan.zrx/minimax-m31-dev/1010_topk_q8_model_candidate/champion_quality_e2e_1011/CHAMPION_NAN_DIAGNOSIS_1011.md` 记录的是 **prefill R8-PKG-E Champion**：full quality 592/1024 request errors、短20请求4次HTTP500，prior NaN GEMM 可触发非有限输出，精确可重建源码仍缺失。这不是 decode P6-2-residual binary 同样失败的证据，但阻止将整套 prefill Champion 一并作为已通过路径引入。

### 后续顺序

1. 选择性接入 decode 三个提交；保留我们的 scheduler/mask/writer/norm/packing WIP，解决 T160 constant 冲突；本轮未执行这一步。
2. 用固定相同 Q/KV/TopK 分别比较当前 Cake、原 Triton、新 native；拆清 main-Q dtype 变化和 kernel 实现变化。
3. 为 native 接上 packed MXFP8 输出，先 wrapper+现有 quant，再考虑 K2 融合；测 attention→quant→O 整链，不相加孤立 kernel 数字。
4. 验证 uniform/ragged fallback、invalid/padded rows、Graph exact/padded、strided reserve table、页复用和 stream ownership；default FP32 gate 保持不变。
5. 相应门禁通过后，按用户授权再做 profiler-off model/E2E、正常拒绝采样与质量；完成稳定候选的独立 review 后才提交。原模型 CPU schedule/KV 无变化快速路径仍需独立推进，native attention 不会自动消掉它。

## 2026-10-11 Decode Champion 精度补测：禁止接入

上节条件接入计划被本轮精度结果阻断。原始 `a93a77d647` launcher、原始 P6-res binary（SHA256 `c1506bb46a56f4eb343aa6aead09f4dc72d79674ea74193558de6e14004b8954`）对固定 T160/width5 FP8-Q 输入重放600次（200 eager +400 Graph）：native有13次错误（7 eager、6 Graph），最大绝对误差67.375；相同输入的 retained Triton 参考600次均与保存期望完全一致。短60次原始 launcher 重放曾全过，说明短测不足。扩展矩阵也出现 max35.5625 和6.75 的有限输出错误。不能将此问题直接等同 prefill Champion 的 NaN 问题；具体内核状态/同步根因尚未定位。

本轮临时 native 接入已经撤回，保留此前 WIP，未启用 prefill、未更改 resident 服务。MXFP8 wrapper 微测虽有179.216→49.600us的算子时间差，但错误候选不能用于模型，更不是E2E收益。后续优先用固定失败输入和可重建源码定位 K1/K2 状态、同步和 workspace 生命周期；修复精度后才恢复 packed-output/mask 接入及模型质量/MAL门禁。CPU schedule 和 profiler-off 空泡仍独立跟踪。

完整配置、证据边界、失败输入及复现命令见 [Decode precision report](/data0/ruixuan.zrx/minimax-m31-dev/1011_decode_champion_precision/DECODE_PRECISION_REPORT.md)，核心结果为同目录 `replay_original_600_results.json`，最小脚本 `replay_saved.py`。

## 2026-10-11 不启用 Champion 的继续优化

按用户要求保留原 attention 路径，选择性合入新代码的 MoE packer 小 tile 预热及共享 selector，并保留 T160→M16。另为 hybrid KV allocator 增加 FULL/SWA 无扩容路径：省去完整 block 表回滚快照，仍执行 tail 校验与 SWA skipped 回收；LINEAR 回填、跨页扩容、分配失败继续原路径。无数学/量化/缓存 ABI 改动。

CPU selector8、GPU数值/Graph6、HybridType15、CP shard6、HybridPool45测试通过。32独立请求/80K/page128/reserve21 的 allocator callable 微测9.99182→4.33976us，减少5.65206us（56.6%）；不含 engine.schedule、cache-key hash 和GPU工作，不能解释整个300us schedule空泡。预热收益也不能写作稳态TPS收益。六文件独立只读review无确认 blocker，完整wheel构建通过且不含Champion文件。

报告/源码SHA/日志/复现脚本见 [No-Champion optimization report](/data0/ruixuan.zrx/minimax-m31-dev/1011_decode_no_champion/OPTIMIZATION_REPORT.md)。后续优先做 profiler-off GPU边界/剩余schedule暴露成本，再推进FP32 gate sigmoid/bias/group2融合及Cake adapter优化。旧postnorm冻结源在MtpExecutor存在漂移，不能复用旧数字作为本轮单变量E2E基线。本轮未启动模型E2E，无新timeline/TPS结论，resident服务未变动。


## 2026-10-12 final fresh-wheel validation and unresolved variance

Latest final repaired candidate full CPU+CUDA timeline: `/data0/ruixuan.zrx/profiling_results/m31_decode_dp8_bs32_80000_dspark_static4_no_champion_final_20261012`. Eight rank exports each4BS32steps/12Graph launches/8588kernels. These are post-formal profiled samples, not profiler-off throughput evidence. Actual final compute native SHA50e7613d5221faf051ebaf90fb8da0b88bc001dfb9511df7c5e300390c4cfa6e; Champion OFF, FP32 gate.

Final seven profiler-off DP8 BS32/card fake80K static4/gamma7 rounds:1792/1792 successful, TPOT7.371477ms, derived/card4341.0567TPS /4.3410567TPMS (260463.40TPM). Compared with retained clean baseline ABBA pool4143.6219TPS, +4.764789%. Earlier full ABBA candidate pool5428.8407TPS/+31.0168% is historical and **not reproduced by final fresh-wheel confirmation**. Final seven round means5.688245..10.798983ms, no outlier removal; not a new final ABBA. Full client-window/card baseline620.0521→final608.2272TPS (−1.9071%), so no clientcompletion gain claim. Stable performance-gain acceptance remains unpassed; investigate matched final-candidate/baseline variance before declaring performance complete.

Target profiledGraph API median2123.968us, start→GPU2065.400us, return→GPU2.721us; proposal231.582/237.252/5.529us. Some targetGPU begins beforeAPIreturn. Schedule per-rank medians66.64–91.57us, maxima435.55us. This does not establish equivalent profiler-off bubbles; prior controlled launch probes show profiling amplification. Graph queue delay behind runningGPU must not be counted as idle. Preserve all-recorded-stream GPU unions and separate inclusiveCPU/overlappingeventwaits.

Real PD4+4 same-ID660 clean/final candidate, same actual-loaded originalP native: zeroerrors, Math381/500(76.2%) andall500outputsidentical; LongBench157/160identical,3changedtext withno per-ID scoredrop. Raw66.87212695→66.88693159;4truncatedeach, untruncated68.05715927→68.07234352. Strict4DP nativeMAL3.606535864→3.604785241(−0.0485403%); normalrejection/noreuse. Stochastic16 each and separateDP8fake normalrejection256 pass; not proof of DP8 all-real-input quality or stochastic bitwiseequivalence.

Full accumulated runtime/framework/kernel review plus final deltas found and resolved B1 verify stride0-mask compatibility: legalexpanded Bool[T] views now use legacywhere fallback; onlycorrect-shape contiguous same-device Bool[T] defers mask. No normABI relaxation, no invalid-row NaN propagation beforeresidual. ActualPD B1..16 Graphcapture and6maskCPU/GPU/Graph tests pass; rebuiltnative230/1skip andactualGraph3pass. Controller exactownedID cleanup andstop-RPC restorationwindow fixes have7+2fakeDocker regressions. Remaining untested model-boundaries include actualcancellation/timeoutstress, beam/multi-return, HIP anddedicatedCPnogrowthintegration.

Original PD4+4 container restored with originalconfiguration:5health endpointsidle, actualnativeHTTP andall8loadednativeSHAmatch. Campaign/arm/cleanup/restoreexit0. No commits/push in this round. Documentation update occurs after immutable validation; production source/native unchanged.

Full evidence, necessity/compatibility table and gates: `/data0/ruixuan.zrx/minimax-m31-dev/1011_decode_final_validation/VALIDATION_REPORT.md`; final receipts under`v2_b1_mask_fallback/retry_correct_import/`.

Backlog order: (1) explain final roundvariance and freshmatched baselineconfirmation, tracingCPU readiness/cohort andactualGPUidle independently; (2) FP32gate/group/topk fusion with precisiongate; (3) querycast/preparemetadata/selectedvalidprefix/refreshedscales adapterfusion withragged/static/cache/stream contracts; (4) actualcancellation/beam/CP boundary E2E. KeepChampionOFF andlifetime/epoch/Graph/allocatorfallbackguards. Profilekernel sum is not a promised additive E2E saving.


## 2026-10-12 full multi-dataset accuracy and production cleanup audit (ongoing)

Artifact root: `/data0/ruixuan.zrx/minimax-m31-dev/1012_decode_complete_quality_variance`. Each arm completes all 5681 frozen IDs: GSM8K1319, Math500500, HumanEval164, AIME24 30, LongBench-E3668 covering all13 official English E tasks. Same checkpoint/P runtime, PD CP4+DP4/EP4, DecodeFP32 router, staticverify4/proposal7, normal rejection, ChampionOFF. Decode candidate compute SHA256 `50e7613d5221faf051ebaf90fb8da0b88bc001dfb9511df7c5e300390c4cfa6e`.

| Dataset | Clean baseline | Optimized candidate | Boundary |
|---|---:|---:|---|
| Math500 | 421/500 (84.2%) | 421/500 (84.2%) | no-thinking8192; both3length-ended |
| GSM8K | 1143/1319 (86.6566%) | 1143/1319 (86.6566%) | no-thinking8192; both2length-ended |
| HumanEval | 142/164 (86.5854%) | 142/164 (86.5854%) | actual official-test execution in owned isolated Docker;22failedtests,0guardrejects |
| AIME24 | 26/30 (86.6667%) | 26/30 (86.6667%) | thinking32768/output32768; baseline4 and candidate3length-ended; raw score includes correct answers at cutoff |
| LongBench-E | 55.4712117402 | 55.4394854554 | official task metrics/task macro; both710length-ended retained |

Both full arms have zero request errors, valid strict native accepted/proposed counters and all4D rank evidence. Native MAL3.0638055644→3.0652013522. Math/GSM/HumanEval full responses identical;8AIME texts differ with no answer-correctness changes. LongBench283 texts differ:120 score increases,114 decreases,49 equal scores. Taskmacro delta−0.0317262848 points is a measured small difference, not proof of statistical equivalence or a confirmed optimization regression. Same-config repeated sentinel controls remain pending.

Confirmed old measurement problems:
- Previous Math512 budget yielded381/500. All40 newlycorrect IDs in the full8192 run had hit the old512 cap;0 newlywrong. The larger mixed suite changes batching too, so this is per-ID cutoff evidence rather than an isolated budget-only A/B.
- Old HumanEval extraction used the last fenced block, often examples/doctests, losing real entry functions. A legacy guard also rejected valid dict `__getitem__` and constant `__import__('re')`. Corrected extraction preserves generated implementations and official tests, retains denied unsafe dunder/import controls, and records identical scorer hashes across arms. Obsolete88/164 score is archived; corrected142/164 recovers54 and loses0.
- Expanded-budget cutoff subsets are supplementary and still running for candidate. Baseline Math3 capped cases all eventually stop correctly, but2GSM and2AIME remain wrong at larger caps. Own-truncated-ID subsets differ between arms and cannot be treated as matched full higher-budget accuracy.

Cleanup audit: current tracked diff71files includes14testfiles (+2208 lines),46productionfiles (+2036),4BUILDfiles and7doc relocations (−2025 lines with destination untracked). Untracked additions are inventoried separately. A prepared, unapplied removal patch retires25files/1029lines of private CPU/GPU journals, captured phase markers, node counting, cross-framework export interfaces and observer-only profiler settings. Preserve supported profiler/NVTX, safety assertions, relevant tests/docs and independent CP/prefill WIP. Branch audit covers GitHub208f1d2f, other open source5dcf6da5, internal80bde5c1; full patches are in `production_cleanup_audit`.

Acceptance is not yet complete: cutoff supplements, LongBench repetition control, actual probe-data variance analysis, cleanup implementation/build/compatibility gates, final independent review and clean collector-off stable performance remain. Previous31% claim remains unreproduced; no new final gain is claimed from diagnostic arms.

### 2026-10-12 fixed-contract precision follow-up and mid-iteration stall evidence

The expanded-output cutoff follow-ups completed without transport/native-accounting errors. They are supplements, not replacements for the full 8192-token ordinary benchmark. Baseline Math500's three capped rows all answered correctly at 32768; candidate answered two correctly, while `test/intermediate_algebra/2046.json` reached 32768 without an answer (baseline stopped correctly at 29746). These original follow-ups used different DP routes/cohorts. Preserve this failure and do not declare full precision equivalence from the identical primary 421/500 aggregate.

The exact installed Prefill request resolver and checkpoint tokenizer were probed on CPU. The request resolves to DISABLED, kwargs include `thinking_mode=disabled` and `enable_thinking=False`, rendered prefix ends in `</mm:think>`, and rendered input length is 200, matching both actual requests. Checkpoint template SHA256 is `01a13fa194313b925b4f7b69b7aa315aa13ee68a47d4ddb4a8e73e2cf8768ff8`. MiniMaxM3ReasoningDetector independently forces reasoning parsing until a closing marker; `reasoning_content` alone does not prove incorrect request mode. The actual request is `/v1/chat/completions` through the Python Prefill frontend; missing thinking fields in the separate C++ API are not established as this campaign's cause.

Prepared controls at `/data0/ruixuan.zrx/minimax-m31-dev/1012_decode_longbench_repeat_control`: 64 LongBench-E diagnostic sentinel IDs, two repeats per arm with identical order/routes/one worker per DP; plus four explicitly labelled same-question Math2046 replicas, one on each DP, two repeats per arm at 32768; then matched target-only controls to separate target arithmetic from speculative accept/commit state. These are diagnostic subsets, not full benchmark scores. They must run only after the existing DP8 controls and exact resident PD restoration complete successfully.

Profiler-off diagnostic candidate DP8 completed eight formal rounds with 256/256 successful requests each. TPOT means span 5.476--8.266 ms. All 24 GPU phase exports passed their validity/capture-generation/ordinal checks. A concrete mid-iteration event occurred in round 3, target replay index 27: DP0 submitted at round+12624.407 ms, other ranks at +11923.432--11923.854 ms. DP0's graph took 23.389 ms while other graphs took about 724 ms; the CPU replay calls took roughly 32--42 microseconds. DP0's previous accept-length metrics wait ended at +11922.498 ms, leaving about 701 ms before its next proposal. This localizes one propagation chain to a host gap before proposal submission, with collective waiting inside the other ranks' target graphs. It is not initial request readiness, and does not identify the specific stalled host function yet. Additional mid-iteration submission gaps occurred on DP4 and DP1, so rank0-only attribution is unsupported. Raw same-device GPU deltas and common CPU CLOCK_MONOTONIC_RAW timestamps are separate clocks; no absolute cross-device GPU clock alignment is assumed.

Evidence: `1012_decode_complete_quality_variance/candidate_diag/{journal_analysis.json,round_gpu_phase_summary.json,target_verify_slow_by_round.json,mid_iteration_host_stalls.json}`. Concurrent rank wait sums are not critical-path latency. Initial replay exclusion here is only for mechanism classification; every formal round and every request remains in the performance accounting. These candidate timings include private endpoint markers and cannot be presented as final clean-runtime performance gains. Client process-count and NVML toggles do not independently explain the observed variation. Normal backend startup already calls `gc.collect(); gc.freeze()` (confirmed in all eight rank logs), so missing startup GC freeze is not an established cause.

Cleanup is prepared but not yet applied while precision controls still depend on the frozen source/runtime: current probes remove approximately 1029 lines across 25 files; latest open-source/internal snapshots remove approximately 346 lines across 10 files. Keep required event ordering, exception cleanup, KV leases, compact-mask fallback, seed-offset semantics and ordinary profiler/NVTX. Set experimental native Hybrid/Champion decode default OFF rather than accepting its isolated bitwise claim after the prior-GEMM nonfinite failure. Restore internal perf target's validated FP32 decode router and non-Champion Prefill defaults. Remove internal CI temporary public-key printing narrowly. Tests enforcing production dispatch and Triton device_assert contracts remain.

Outstanding: exact host-function attribution and collector-off reproduction of the stalls; matched Math2046 and LongBench repeat controls; source cleanup/build and relevant compatibility gates; one final fresh independent read-only review; clean matched repeatable performance and new timeline; original PD4+4 HTTP/native/idle restoration. No final optimization or precision signoff is claimed by these interim artifacts.

### 2026-10-12 identical-run controls and explicit remaining roots

Normal speculative controls completed: baseline64 selected LongBench sentinels change42texts/29scores across two identical-contract repeats; candidate changes43texts/32scores. These deliberately selected IDs are diagnostics, not a random cohort or full-score uncertainty estimate. Fixed Math2046 (`test/intermediate_algebra/2046.json`) four-DP replicas yield baseline1/4 then3/4 correct, candidate1/4 then2/4 at32768. Both versions can change correctness on the same fixed DP. This establishes baseline variability, not candidate equivalence. Receipts: `precision_repeat_comparison.json` plus separate baseline/candidate campaign states.

Dedicated non-speculative engines use SP_TYPE=none in both P/D; the earlier attempt to request target-only on a speculative engine was rejected by the harness before requests. Retain that negative gate rather than bypassing it. New target-only baseline repeats still change40/64texts and31scores. Global-serial target-only baseline repeats still change19/24texts and15scores. Neither speculative disablement nor one global request lane alone resolves the common instability. Serial control changes both P and D concurrency; it cannot assign a root to P batching. Candidate target-only and long Math controls remain active; no final precision signoff.

The common P runtime uses BF16 router and previously enabled CP packed-KV overlap/suffix wire; D uses FP32 router. A serial P-only first-token/logit probe is prepared with exact rendered inputs200 and16863, no D KV transfer or speculative verify. CPU preflight confirms thinking DISABLED and exact recorded input lengths. Run only after current exclusive quality traffic drains; subsequent logits/KV/target-step evidence must determine the first divergent state. The presence of reasoning_content does not show an ignored thinking-mode request.

Profiler-off diagnostic TPOT variance corresponds closely to measured steady-iteration host stalls. Round3 versus7 TPOT difference2.779770ms corresponds to700.975ms/255=2.748922ms; round4 versus8 difference2.038386ms corresponds to533.866ms/255=2.093593ms; round6 versus10 difference2.059143ms corresponds to530.571ms/255=2.080672ms. These are same-engine/cell observations, not final gains or subtractable production savings. The CPU blocking function is still unresolved. Initial rank readiness is not used to explain these mid-iteration stalls.

Each D worker currently has roughly223–225threads and all-core affinity; cgroup quota is unlimited and throttling counters zero. No production Torch thread limit is configured. CPU-team contention is a hypothesis, not established causality. An external99Hz user-space perf/DWARF probe with CLOCK_MONOTONIC_RAW and immutable candidate50 compares inherited teams against OMP/MKL/OpenBLAS/NumExpr=1, constant8client processes, no NVML polling. CPU profiler-clock preflight passes with `--clockid monotonic_raw`; `mono_raw` was rejected before service mutation. Diagnostic timings are excluded from final collector-off performance.

Prepared cleanup now removes29current-patch files/net1222lines and26week-audit files/net886lines, including temporary benchmark bodies in regression tests and vendor timestamp/printf/tensor-dump hooks. It adds meaningful CP FULL/SWA no-growth compatibility tests and metadata-only native-hybrid dispatch tests, retaining mathematical/cache/ownership guards and ABI slots. These are unapplied plans; compile/runtime gates and fresh independent final review remain required. Latest remotely verified refs: GitHub208f1d2f3f, open5dcf6da59d, internal388fc8d421 (the new internal commit points its submodule at open5dcf). Original dirty work and historical binaries are preserved.

### 2026-10-12 P-only first-logit localization and live backend correction

Dedicated target-only controls have completed both arms, including serial controls and Math repeats, with positive gates exit0 and exact original PD4+4 restoration. Consolidated counts: `1012_decode_complete_quality_variance/precision_repeat_comparison.json`. Target-only baseline changes40/64texts, candidate44/64 across identical-contract repeats. Common variability remains a failure to localize, not precision equivalence.

After restoration, serial original-P-only first-logit requests with no D KV transfer or speculative target verify completed12/12. Same exact inputs200/16863/21000 tokens cycled four times, no prefix reuse, topk1/temperature0, thinkingDISABLED. Input200 is bitwise identical across four repetitions. Input16863 logits differ by as much as8.98496 from the first repetition and argmax switches2230→18845. Input21000 logits differ by as much as9.56688 and argmax switches13394→2765. Returned token agrees with logits argmax in every request. No NaN/+Inf was observed. Evidence: `1012_decode_complete_quality_variance/prefill_first_logits_control/{records.json,complete.json,*.logits}`. These large long-input changes precede Decode and cannot be explained merely as decode sampling, KV transfer or last-bit target arithmetic.

IMPORTANT backend correction: installed msa_attention.py has FMHA as its environment fallback, but actual restored Prefill parent PID810673 and original launch record explicitly set `M3_NVFP4_PREFILL_BACKEND=flashinfer`. The worker's process-title rewrite leaves `/proc/<worker>/environ` unsuitable for inferring missing flags; inspect the launch parent and role record. Original D uses FMHA for its separate path. The P package has no Champion bridge. The initial inference from the module default was wrong and corrected before constructing live controls.

Frozen counterfactual campaign: `1012_prefill_logits_root_control`. Same original P/D runtime and weights, three independent P controls: suffix-wire off only; overlap off only (also disables suffix-wire applicability by source contract); FMHA backend only. Same twelve exact serial P-only requests per arm, all five HTTP endpoints idle before/after. Runtime/native/Python hashes and actual launch-parent flags are required. Dedicated owned containers are stopped, then original PD4+4 is restored with four native routed requests and eight native worker hashes. CPU probe preflight passes with all exact lengths/modes. Controls are diagnostic and do not replace full dataset quality or clean performance gates. No production cleanup has been applied yet.

First counterfactual result (twelve successful requests): suffix-wire OFF still gives maximum repeated first-logit differences7.224424 (16863) and9.240940 (21000), with LCC first IDs2765/95760/2765/424. Short200 remains bitwise stable. This rules out disabling suffix wire alone as a sufficient remedy; it does not prove the wire is numerically equivalent. Overlap/backend controls remain pending.

The first control launch failed before any probe request because the validation helper computed a mapped-library relative path against an unresolved symlinked Decode install root. The owned container stopped and original PD4+4 HTTP/native/hash restoration completed successfully. Preserve `1012_prefill_logits_root_control/failed_control_restore_receipt.json` and original failure logs. Corrected helper uses resolved roots; a CPU preflight verifies all eight original workers' resolved paths/hashes. Fresh output/container names under `retry_resolved_provider` preserve the failed attempt, and the new controller records exceptions immediately and distinguishes control failure from successful service restoration. Do not merge the failed attempt into precision data.

### Confirmed long-Prefill reproducibility root and causal repair (2026-10-12)

FMHA-only control completes12/12; all logits for200/16863/21000 inputs are bitwise identical across four serial repeats. Disabling all overlap is insufficient (max delta9.338907/7.774391). Actual FlashInfer source uses a128-slot union hash. Above128blocks (16384tokens), linear-probing atomicCAS decides hash placement and therefore accumulation order; the installed module explicitly limits bitwise reproducibility to16384tokens. Both unstable prompts exceed that boundary.

Controlled actual P rank0 attention input: Q[4216,64,128], raw packed pool[132,73728]. Five replay calls use identical unchanged Q/K/V/scales/TopK/metadata hashes; four outputs differ by1–5BF16elements, max0.00390625. A separate baseline replay20times differs19times. Source pointer/provenance: `1012_flashinfer_prefill_singleop_capture/capture/operator_capture/{inputs.pt,receipt.json}` and `1012_flashinfer_union_determinism/operator_baseline/receipt.json`.

Narrow fix in isolated FlashInfer source250aae19: above128blocks only, sort compacted full entries by block code, retaining all query-membership bits; <=128 preserves historical collision-free order. Use warp shuffles and three cross-warp shared exchanges, no global scratch. Preserve softmax range-replay, empty-union/TMEM/numerical/layout guards. A versioned deterministic_v1 JIT URI prevents loading old AOT/JIT kernels; provenance marks the two formerly upstream-identical sources as modified and retains both old/new hashes. CPU package tests8/8pass.

Fixed actual operator repeats20/20bitwise stable; empty,128block full,129block full/partial collision gates pass. Sixteen sampled real rows versus independent FP32 oracle: relativeRMSE0.000780183, cosine0.999999642, maxBF16delta0.0078125. Isolated event mediansbaseline0.369120ms/fixed0.375088ms (~+1.62%); timings are one controlled operator comparison, not production/model performance proof. Fresh TP1/2/4, ragged, high-id/full-union, range-replay and Graph regressions are prepared and not yet run.

Whole Prefill causal control retains original FlashInfer/overlap/wire/BF16gate/native/weights and replaces only that FI fix, with no private capture hooks. All12requests pass:200 tokens→2421,16863→18845,21000→2765, all repeated logits bitwise identical (maxdelta0). This confirms the repair removes the observed long-Prefill first-logit instability. It does not sign off full multi-dataset generation accuracy, the separate steady Decode host stalls or the cleaned production candidate. Dedicated control and restore receipts: `1012_flashinfer_union_determinism/prefill_fixed_control/`. Broader validation/cleanup/final fresh review remains required.

### 2026-10-12 additional validation and production cleanup

FlashInfer deterministic-union GPU regressions passed7/7 (gpu_determinism_tests_private_pytest.log): Hkv1/2/4,128/129/132/257 blocks, ragged lengths/partial tails, range replay, full128-member union/high65535 block code, empty union/TMEM reuse and20 CUDA Graph replays. Whole-P first logits were bitwise stable for all12 requests; full-generation quality on the final cleaned runtime remains pending.

CPU-stack diagnostic first attempt failed before measurement. Native logs/stack_trace.log attributes SIGABRT to GraphReplayDiagnostics construction; configured replay_diags directory did not exist. No usable performance/CPU-stack result was generated. Exact original PD4+4 restoration passed. Retry1012_decode_host_stall_control_retry precreates and writability-tests both arm directories. Runtime/native remain frozen50.

Scoped cleanup is now applied:29 current-worktree files remove1222 net lines of private journals/GPU markers/export APIs/embedded exploratory benchmarks;26 isolated latest-week files remove886 net lines of shadow comparisons/dispatch traces/tensor dumps/vendor timing probes/CI public-key debug output. Native hybrid decode is explicit opt-in, default OFF. Original files are archived in production_cleanup_audit/retired_probe_source and retired_week_diagnostics. Production lifetime/events/device/fallback safeguards and correctness tests are retained. Fresh build, complete model validation, stable collector-off performance and final independent review remain required.

### 2026-10-12 清理后候选与验证状态

新的生产 wheel 已从实际 Bazel worktree 编译，SHA256 `792a8d10d695295653c314fbc976c67a4aba5fc235c6d11241787cd4ce7824c2`，安装在 `minimax-m31-dev/1012_decode_production_cleanup_validation/native_install`。原生 payload 不含退休的 CPU journal/GPU marker/export 诊断接口。一般/同步原生229项、异步37项和 Graph15项均通过；同步与异步 MTP test 的不同环境契约已在 BUILD 分开，不在同进程切换缓存开关。部分用例跨配置重复，不能加和宣称全部独立用例。完整模型精度仍待全量5681条/分支结果，路径 `minimax-m31-dev/1012_decode_clean_production_validation`；两个分支共同使用确定性 FlashInfer P 修复，D对比 clean208 与清理后新构建，Champion关闭。

时间戳 CPU 采集复现 metrics wait 结束到下一次 proposal 提交928.966ms，而 proposal API24.304µs。对应进程在该区间无用户态 CPU 样本；此采集排除内核，不能据此认定具体阻塞调用或 CPU copy 根因。采集器生命周期 gate 失败，保留失败及数据；perftest本身完成，精确原 PD4+4/native身份/HTTP/idle恢复通过；bounded_threads 分支未运行。因此未改生产线程设置，也未将本轮计为稳定性能结果。详细证据边界见 `1012_decode_host_stall_timestamped_control/default_threads/cpu_fixed/evidence_boundary.json`。

无采集器的清理后性能计划在 `1012_decode_clean_stable_performance`：B-C-C-B，两次冷启动/分支，每次2warm+12formal，客户端8×32，DP8/BS32/card/fake80K/static4/gamma7/FP32 gate；所有正式轮保留，另采8rank Torch timeline，计时与trace分开。目前仅准备/import preflight通过，未执行或作收益结论。

#### 最终候选更新：legacy draft producer 顺序

新增可控用例证明，旧 fallback 在 producer event 前对非连续 GPU draft 做 reshape 会复制旧数据；跨 GPU draft 同样不能只在 runner 的 stream 上 wait。两个用例修复前失败，修复后通过。现在同卡先 wait 再 reshape/cast/D2H；跨卡明确等待源 event 完成并保留旧 runner-stream H2D staging 完成屏障，普通 static Think 快路径不变。该修复只作兼容性/正确性结论。另修正 AsyncRunner CUDA 用例自身遗漏的输入初始化跨 stream 屏障；生产 AsyncRunner 未因此修改。

重新构建的最终 wheel SHA `430092fbfd169de5e14dae3d50022dd3bfd9335262529a451dc3003aac6dc4aa`，Transformer SHA `4e73aadcdba6d448e65903835d74b2fc9304ab553383a077c57c1ae8fd2242e2`。一般/同步231、异步37、Graph15项通过；不同环境重复用例不加和成独立数量。两个精度分支中 baseline native208保持不变，candidate 尚未启动时已换成上述新 wheel，因此后续 candidate 全量结果将验证该版本。完整5681/分支对照仍进行中，性能/最终独立review尚未完成。

#### 2026-10-12 全量精度、清理审查与正式性能进展（覆盖前述待验证状态）

最终4e原生候选已完成 matched5681/分支：Math500421/500、GSM8K1143/1319、正式HumanEval142/164，生成输出均与208基线逐条一致；LongBench-E13任务宏平均55.421246→55.439471；原始AIME数字提取26→27/30；native MAL3.054361→3.044463。两组请求错误均为0。原始长度截断分母保留，普通数据集扩预算补测B7/C8完成，GSM122与AIME-I-12仍在32768/65536处截断。每组真实4D取消/服务deadline8204及后续成功请求、原始服务恢复和原生身份检查通过。证据在1012_decode_clean_quality_final。

AIME两组原始30条均没有最终assistant正文，数字评分来自reasoning fallback，不能宣称完成答案准确率通过。另复现了共用旧M3解析器忽略force_reasoning=False、将disabled-thinking正文放进reasoning字段的问题；隔离latest_open_source修复显式模式，同时保留未指定M3默认语义，37 CPU真实tokenizer测试、144其他detector兼容对照通过。该Python修复尚未部署到冻结4e候选，不影响上述生成数值对照。短thinking边界协议对照严格排在正式性能/恢复之后，不能将generic与Mm marker不匹配的源码观察直接写成AIME根因。

清理最终scope：当前W29files净减1222行；最近一周latest35files增87/删1006，internal2files增7/删20。去除临时CPU journal/GPU markers、影子比较、TopK trace、tensor dump、vendor timestamp/printf/probe exports及CI debug打印。标准profiling、安全检查和默认OFF实验算子保留；历史SHA固定二进制未采用，清理源代码sm103a compile-only通过。不能宣称全部实验功能删除。框架tracked68files中新增测试2078行、生产1803行，另有docs搬迁；规模分类不替代语义审查。

一次新鲜独立只读审查已完成，未发现可确认的新增P0/P1/P2正确性阻断项；确认mask整任务fallback、event-before-read、foreign-device fence、Graph generation/device/alias/legacy回退、批量owner/cohort/epoch、pending/KV异常cleanup、postnorm正零与TP/stride-zero回退、FI union membership及barrier和parser默认兼容。完整结果在production_cleanup_audit/independent_clean_decode_review.json；正式稳定性能与AIME回答完成仍是验收缺口，审查不替代这些实测。

无profiler/NVML/外部CPU collector的B-C-C-B正式对照正在运行，每次冷启动2warm+12formal，保留每轮。已完成：B0 TPOTmedian7.066190ms/TPS4528.68，C0median6.045857ms/TPS5292.91，C1median5.7324ms/TPS5582.46。C两组TPOT CV约13%，B0约20%；候选慢轮仍存在，暂不称稳定性能或空泡已解决。最后B1、独立48-step八rank timeline及最终PD4+4恢复待完成。最终数据入口1012_decode_clean_stable_performance/stable_performance_summary.json；TPS按32并发/native TPOT换算，TPMS=TPS/1000，与client completion吞吐分开。

四组正式数据现已全部完成，覆盖前述B1待测状态：B/C各24轮，median TPS/card4333.313937→5418.698559，+25.04745%；TPMS4.333314→5.418699；C nativeTPOTmedian5.905477ms/p958.133657ms，TPSCV15.40084%→11.33063%。C TPS p05/p953934.514675/5784.124210，B2878.954528/4768.436857。全部48轮及四组独立normal-rejection256请求功能gate均通过。独立reviewer从raw accounting复算所有轮次、decode_time/255、32000/TPOT、分位数/CV和median收益，完全一致。该收益是forced-accept/fake-history派生native吞吐，不是normal-rejection吞吐或client完成吞吐。最新48-step八rank observer和最终服务恢复仍进行中。

## 2026-10-12 corrected M3 wire contract and final timeline update

The 12-request, four-lane control on original restored PD established two separate frontend/config defects: disabled thinking was interpreted as reasoning despite explicit force_reasoning=False, and nonstream HTTP called stateless parse_non_stream on each native output packet. A captured stream already contained token200060 (`</mm:think>`) yet final answer was still classified as reasoning. M3 now preserves incremental close-state for both HTTP modes. Native GenerateOutput.output_ids are incremental packet tokens, not cumulative. New finish flushing handles textless EOS and partial ordinary marker prefixes in both detector buffers; pending suffix follows any buffered normal text. This model-local change and M3 detector changes leave shared renderer/other detectors unchanged. Negative regressions are retained; final40 CPU tests, zero skips, pass. Fresh readonly reviewer confirmed earlier P2 fixed and no new confirmed blockers. These tests do not establish full dataset quality.

A new full fresh baseline/candidate campaign is running at `/data0/ruixuan.zrx/minimax-m31-dev/1012_decode_corrected_reasoning_full_quality`: Math500500, GSM8K1319, HumanEval164, AIME2430, LongBench-E3668; same manifest/order, native rejection, static4/FP32, Champion off. Common P frontend overlay contains only the two fixed parser files and preserves P native libraries and deterministic FlashInfer. AIME uses native begin_think_token_ids=[200059], end_think_token_ids=[200060] matching M3, output/thinking32768; ordinary output8192 and disabled thinking; official LongBench budgets. Both arms rerun all5681 instead of reusing old outputs. Short protocol gate requires final content on every D lane before full traffic. Primary capped and expanded-cutoff results remain separate. Original PD4+4 is restored and functionally/hash verified by controller finally.

Latest standalone observer timeline: `/data0/ruixuan.zrx/minimax-m31-dev/1012_decode_clean_stable_performance/candidate_timeline/results/timelines`, eight rank files, each48real BS32decode scopes/101252GPU kernels/144Graph launches. CUPTI correlation proves target graph API median ~1.81–2.98ms by rank, but first GPU after API return is only~2.45–3.88us; proposal API~155–244us and postreturn~5–6us. These are observer-visible CPU API wall times, not profiler-off API timings; queued postdraft launch latency is not GPU idle.

Rank1 next-proposal index33 has102613.674us union-of-all-GPU-activity idle and103152.801us metrics-end to next proposal-start. Concrete CPU chain: NormalModelInputGatherer.publish_core_tensors_to_cuda → aten::to/_to_copy → aten::empty_strided → cudaEventQuery (102462.022us). Tensor shape[1,32,627], int32, stride[20064,627,1], identifies fresh per-step KV block-table H2D allocation. This locates one long bubble in CachingAllocator event-query/allocation; driver contention, OS descheduling and profiler callback cost still need discrimination. It does not prove every profiler-off slow round has this cause. Next bounded hypothesis: persistent KV-table storage with safe consumer/Graph-generation leases and copy events; no unsafe single-buffer reuse or global synchronization. Not implemented or claimed as a gain.

Formal B-C-C-B12rounds/arm (48total), observer-off DP8 BS32/card physical fakeKV80000 static4 forced acceptance, native normal rejection gates separate: median cardTPS4333.313937→5418.698559 (+25.04745%); candidate TPMS5.418699, TPM325121.9135; TPOTmedian5.905477ms,p958.133657. Candidate TPS p053934.514675/p955784.124210,CV11.33063%; baselineCV15.40084%. All slow rounds retained. This is repeatable central tendency with residual variance, not fully stable production normal-rejection throughput. Eight-rank latest timeline and exact original PD restore completed successfully.

### Online gate found and reproduced wire double parsing

The first corrected frontend short protocol gate correctly blocked full dataset traffic: disabled thinking produced final content, enabled M3 boundary remained reasoning-only. Follow-up eight-request capture on all four D lanes (`1012_decode_corrected_reasoning_full_quality_retry/thinking_protocol_root_cause_evidence.json`) proves raw tokens `[2421,200060,92,128640,123,52,125,200020]` decode to `We</mm:think>\boxed{4}[e~[` while HTTP content=None/reasoning=`We\boxed{4}`. Native begin/end IDs are exactly200059/200060; the closing marker was actually emitted. All eight requests remained short/normal rejection and did not masquerade as full accuracy.

Root cause is a second parser above the model detector: normal-only fallback and final flush return plain strings, and `CustomChatRenderer._split_reasoning_text_and_content` reparses those using its separate still-enabled `ThinkStatus`. M3 now bypasses that second classification for already-classified string content; existing DeltaMessage tool/reasoning fields still delegate normally. Tests now cover the real wire split and final flush, not just `_update_single_status`:32negative subcases before fix;40CPU tests/zero skips pass afterward. Fresh reviewer found no new content blocker. Existing generic nonstream reasoning usage counting may overwrite on content-only deltas; this separate accounting limitation is not claimed fixed and is not native MAL. Full corrected dataset campaign must use this final fix. First failed campaign preserved, exact original PD4+4 functional/hash restore verified afterward; retry diagnostic controller also restores original PD.

### Final wire integration gate passed; fresh full quality active

Final corrected frontend at `/data0/ruixuan.zrx/minimax-m31-dev/1012_decode_corrected_wire_full_quality` passed all8 four-lane enabled/disabled short requests. The raw native token IDs match the previous failing frontend on every request: enabled reason is now `We`, final content is `\boxed{4}`; disabled content is the boxed answer with no reasoning. `matched_wire_parser_evidence.json` proves this integration fix changes field classification rather than native numerical generation on these controls. Full5681 baseline and candidate are both fresh, same manifest/order; no previous output reuse. Baseline full generation is active, candidate follows. This gate is not a replacement for full precision. Exact original PD4+4 restore with role verifier launch inputs is controller-owned in finally.

### Allocator follow-up boundary (verified Torch distribution2.11.0)

The running campaign image reports Torch2.11.0 by package metadata, checked without GPU calls. Upstream [v2.11.0 CUDACachingAllocator](https://github.com/pytorch/pytorch/blob/v2.11.0/c10/cuda/CUDACachingAllocator.cpp#L1311-L1336) processes pending cross-stream allocation events at the start of ordinary malloc, before choosing an allocation-size pool; [process_events](https://github.com/pytorch/pytorch/blob/v2.11.0/c10/cuda/CUDACachingAllocator.cpp#L3494-L3532) queries outstanding events and moves on when a stream's event is not ready. Therefore a stall observed during KV-table allocation need not belong to that table's lifetime: it can be triggered by prior allocations. Merely pooling one table may move the query to the next prefix/core/temporary allocation. This upstream mechanism plus trace is an inference, not a C++ stack proof of the packaged binary or proof of all profiler-off stalls.

A bounded next diagnosis should compare identical operator allocation/reuse with profiler off/on and record event-query API wall time separately from thread CPU time/descheduling, then identify the outstanding cross-stream consumer and pool. Keep it outside formal measurement, preserve stream-lifetime tracking, and do not introduce global synchronization, disable event recycling, or change the allocator globally based only on one trace. The frozen candidate and full-quality run are unchanged.

### LongBench historical score audit (2026-10-12)

The historical final-wire report covered 160 samples across eight tasks; its official legacy score was 66.237738. The corrected-wire full baseline covers 3668 samples across thirteen tasks and has task-macro score 56.058134. These totals are not directly comparable. The eight historical tasks, evaluated over their full current subsets, average 67.030043. Five newly included tasks have scores: gov_report 33.566056, multi_news 24.016027, samsum 39.656644, lcc 56.95 and repobench-p 38.326667; including them lowers the thirteen-task macro.

All historical 160 source IDs occur in the new baseline. Their prompts, answers, class lists and token budgets are identical. Rescoring both historical and current baseline responses with the exact historical scorer functions yields 66.416310 -> 66.870355 (+0.454045 points). The historical report labeled official used text-F1 on two TREC classification rows and therefore reported 66.237738; that label was inaccurate. This establishes no aggregate drop on that matched subset, but does not establish absence of individual regressions: hotpotqa drops 4.104747 points on its twenty matched samples, while qasper rises 5.827373. The native/runtime/frontend differ between dates, so historical matching cannot isolate the new decode optimization. Candidate full quality is still running; final acceptance requires the fresh same-contract baseline/candidate comparison. Evidence: /data0/ruixuan.zrx/minimax-m31-dev/1012_decode_corrected_wire_full_quality/longbench_historical_score_audit.json.

Historical HotpotQA delta detail: one matched sample c32e0b8f765bc19e2031e307cc3124da79557c834fa6f2c1 changed from `No.` to `No. Gregory La Cava was a film director, while Mike Judge was an actor, animator, filmmaker, and musician.` against answer `no`, reducing F1 1 -> 0.117647. This is an output verbosity/instruction-following difference, not a flipped yes/no answer; it remains a real benchmark loss and is not waived.

### Profiler-off formal-round arrival audit (2026-10-12)

All 48 formal B-C-C-B rounds have exactly 256 matched client records. Candidate fastest and slowest rounds have TPOT 5.499392 and 8.240604 ms, respectively; client request-start spreads are 72.738331 and 65.457445 ms. The corresponding baseline extremes are 6.655914 and 11.828722 ms, with start spreads 63.803461 and 67.384078 ms. Pearson correlations between start spread and TPOT over the 24 rounds per variant are -0.124613 baseline and -0.223579 candidate. These observations do not support a simple larger client-start-spread explanation for slower rounds. Client start is not server arrival, so this is not proof of driver or allocator causality. Median within-round per-request TPOT CV is 0.141766% baseline / 0.053492% candidate, although candidate_1 round4 reaches9.297525%; retain this exception rather than assuming identical cohorts in every round. Evidence: /data0/ruixuan.zrx/minimax-m31-dev/1012_decode_clean_stable_performance/formal_round_arrival_spread_audit.json.

### Remaining target-verify small kernels: explicit unfinished work (2026-10-12)

Rank0 CUPTI correlations cover48 actual target graphs, with1788 kernels per graph. Each graph still contains60 `_fused_query_cast` (median1.344us),60 `_prepare_metadata` (2.688us),60 `_selected_valid_prefix` (1.600us),60 `_refresh_selected_scales` (10.241us). The combined FillFunctor group is62/graph:61 long fills (60 Cake seen clears plus one other initialization) and one BF16 fill at initial residual creation. Do not call all62 O-projection zero fills. On the sampled sparse-layer chain O-projection GEMM is immediately followed by fused postnorm: its former zero-fill/where pair has been eliminated. Router still has separate cast, partials/reduce, sigmoid, FP32 bias add and selection kernels.

The fused-query-cast name denotes one standalone conversion launch; it does not prove integration into norm/RoPE. Current Cake reads BF16 main Q and only needs the index Q conversion. Decode callers do not pass the existing `query_fp8_outputs` producer interface; target verify already uses contiguous BF16 outputs, and the current producer validation explicitly rejects combining contiguous and FP8 outputs. Proper integration therefore needs a reviewed model-local producer/consumer contract (including independent storage and BF16-then-E4M3 rounding), not just deletion of the conversion call. Reusing only index-Q FP8 for Cake avoids unnecessary main-Q FP8 writes; fallback readers must retain their own required main-Q conversion. Verify bitwise outputs, all padding/ragged/stride shapes and Graph replay, and measure producer-plus-consumer before adoption.

Current measured median per-card throughput improvement does not establish elimination of GPU bubbles or these small kernels. Preserve the unresolved100ms allocation/event-query observer stall and Graph API observer boundaries. GPU-focused backlog: (1) integrate query rounding producer output, (2) reduce Cake metadata/seen/prefix/scales adapter work while preserving shared-page ownership and partial-prefix scale semantics, (3) remove router BF16->FP32 materialization by FP32 conversion on load and assess sigmoid/bias/selection epilogues with exact FP32/tie gates, (4) isolate profiler-off allocator/host scheduling stalls. Latest source integration and its new timeline are not yet validated. Evidence: /data0/ruixuan.zrx/minimax-m31-dev/1012_decode_clean_stable_performance/remaining_small_kernels_target_verify.json.
