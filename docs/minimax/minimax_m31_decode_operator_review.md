# M3.1 Decode 第一批优化：单算子 review

本次按用户要求止于实现、单算子验证和契约回归。没有运行模型 perftest、PDFusion/PD E2E，没有部署、替换 resident wheel、重启服务或提交分支。基线为 `399db00d611eddf52263a5ca3e2c09fae6ef01de`，改动位于 `rtp_llm_m31_dspark_worktree`。工作树已有 Prefill WIP 保留；本批不把那些修改算作 Decode 收益。

原始逐算子与空泡分析入口见 [设计文档](minimax_m31_decode_optimization.md)。本批先消除 CPU 展开、逐请求 Tensor 对象和明确的冗余计算；毫秒级 cudaGraphLaunch 空白尚需 review 后以 matched timeline 定位，不能宣称空泡问题已经解决。

## 实现与边界

| 项目 | 实现 | 保持的语义 / fallback |
|---|---|---|
| P0-A compact spec mask | CPU processor/merge结果以 int32 bitset直接H2D；verify consumer直接mask logits；160×200064从32,010,240B降至4,001,280B | bit1允许；旧dense bool1禁止。普通phase不消费spec mask；HIP保留dense。旧FP32哨兵是 `-FLT_MAX`，允许位置含NaN payload按位保留 |
| P0-B batch device owner | stream保存owner+row；稳定cohort直接消费batch seq/tokens/length/anchor/end，避免64个准备narrow及发布热路径逐字段view、重复cat | 检查ID、epoch、行顺序、CUDA/device/contiguous/schema；重排、取消、宽度变化回退。每轮独立owner，不用两slot循环覆盖；lazy getters保留旧Tensor接口 |
| P1-C static prefix LM head（撤回路由） | 候选BS32 M224→M128通过按位门禁，但实际cuBLAS微测变慢，生产继续全gamma LM head | 保留sampler显式head_steps候选接口及精确/RNG测试；尚未启用，避免以FLOPs下降替代性能证据 |
| P1-B force override mask | 一次CPU bool staging + 一次H2D替代每request GPU scalar fill | 仍运行正常rejection与failure逻辑。这是perftest辅助路径，不代表真实服务收益 |
| P2 norm/RoPE producer | target小T contiguous producer上限128→160，直接生成4个consumer布局 | main Q保持BF16；M1、Prefill路径保留；无额外降精度 |
| P2 attention输出量化 | packed quant上限100→160，一次生成FP8值及UE8M0 consumer布局 | 仍先生成BF16 attention output；没有融合Cake attention epilogue，保持BF16舍入 |

batch owner主线程只发布描述与ownership。worker持有owner，ready-event wait位于异常清理作用域；protected/unprotected失败均释放pending claim。成功请求由specUpdate更新legacy SP mirrors，失败请求由worker从该轮captured owner发布保留anchor/propose字段，避免失败请求留下undefined字段。旧SP读取不被删除以掩盖问题。

## 单算子正确性

- Compact mask：FP32/BF16/FP16旧转换行为、尾部、全禁用、允许NaN payload、连续/转置/strided布局、float4对齐与大row fallback；eager consumer和representation错误拒绝。FP16旧 `-FLT_MAX` 转换会溢出拒绝，保持该行为。
- Runner：processor语义、applied IDs、merge/spec cap、batch grow/shrink；compact scratch按flattened prefix reshape，避免P/W缩小时错stride。
- LM head：实际 `cublas_gemm_bf16_bf16_fp32`，V200064/K6144、B1/3/32、gamma7/prefix4；比较FP32输出按位相等。更换M可能改变cuBLAS算法，此精确门禁只覆盖当前GPU/library及这些shape，不能当所有环境证明。
- Sampler：B1/3/32、prefix1/4/7、mapped/unmapped、3轮；tokens/probabilities/后续sample和随机draw/RNG state逐项相等。包含force、非force、混合失败与非法head契约。非法契约测试局部关闭core-dump以检查异常，退出时恢复全局配置，生产断言未改。
- Owner/dispatch：旧getter按需view与storage存活、stable cohort失效、GPU真实长度与host allocation上界区分、混合失败/缺失状态/历史宽度、异常清理。它们是native/mock契约测试，不是完整模型E2E。
- Norm/RoPE：T129/143/159/160及原小shape，consumer布局、dtype、值与in-place V保持，CUDA Graph多轮mutation按位比较。
- Packed quant：T1/3/100/101/127/128/129/143/159/160；零值/极小/放大输入及padding，与FlashInfer FP8 bytes和独立scale packing逐字节比较；Graph多轮比较。非Cake partial combine仍按BF16中间结果验证。

## 性能口径

Python T160 microbenchmark使用50次CUDA活动采样；表中为每replay的kernel duration之和，另存5×200次replay event median。GPU0测试前近空闲，GPU4–7有既有任务；这只是单GPU合成输入结果。

| T160算子链 | 原kernel数 | 新kernel数 | 原kernel时间 µs | 新kernel时间 µs |
|---|---:|---:|---:|---:|
| norm/RoPE + 4个布局copy | 5 | 1 | 22.726 | 10.072 |
| BF16输出→MXFP8 + pack | 2 | 1 | 4.341 | 2.005 |

CPU mask微测为预分配buffer上的raw unpack/copy，不含processor、merge或allocation：dense unpack 32,036.3µs，compact copy 306.626µs。不能将差值直接作为每轮收益；原worker工作有与target重叠的部分。

GPU event单算子测量（100次mask、20次LM head；含提交间隙，不能冒充kernel duration）：

| 算子 / 传输 | 原 µs | 候选 µs | 决策 |
|---|---:|---:|---|
| dense / compact mask应用 | 45.925 | 32.065 | 保留对齐FP32 float4路径，其他布局generic fallback |
| dense / compact mask H2D | 580.155 | 76.426 | 保留，传输字节8倍减少 |
| cuBLAS LM head M224 / M128 | 429.002 | 459.571 | 未证明收益，撤回生产缩行路由 |

LM head测试的单次顺序微测不能证明稳定回归大小，但已足够否定“计算量更少一定更快”的假设。后续若重新提出，需要排查cuBLAS算法选择并做交错重复/纯kernel采样。保留真实GEMM精确性、sampler后续RNG契约测试。没有因该候选通过正确性就默认上线。

## Review 后才做的工作

1. 同源码/依赖/配置先做无profiler、CUDA activity-only、完整profile三个控制，定位cudaGraphLaunch毫秒开销是生产还是观测开销。
2. exact DP8 BS32/80K/static4 matched timeline，统计全stream GPU union、target-end→mask-ready暴露尾部、完整轮间gap、各rank首计算入场和首MoE peer wait；正式无profilerTPS/TPM单列。不能仅测dispatch返回耗时。
3. 非force真实接受、动态verify、普通decode/PDFusion，检查跨轮取消、EOS、换序、page边界和正常reject回滚。长请求与resident/peak内存、Graph pool/KV容量另验。
4. 通用sampler config预打包、commit metadata直接producer、greedy省softmax、MoE及attention epilogue等未在本批实施。动态cohort目前安全fallback，尚未实现GPU row-map重排加速。

证据目录：`/data0/ruixuan.zrx/minimax-m31-dev/1009_decode_operator_optimization`。保留早期失败日志与修复过程，最终结论以验收摘要指定的最终build/test文件为准。


## 最终验收摘要（2026-10-09）

六个native目标在撤回LM head缩行后重新构建成功：`build_route_withdrawal.log`。共76项native回归通过：sampler21、runner4、batch38、owner1、executor7、compact5。撤回后的batch38/owner1/executor7已再次运行通过，其余算子实现未变。2项disabled GPU性能case单独运行通过，CPU mask微测另跑；不混入正确性计数。Python norm1及quant2测试通过，包含上述多shape/多轮Graph字节门禁。

- 最终状态：[review_gate_results.json](/data0/ruixuan.zrx/minimax-m31-dev/1009_decode_operator_optimization/review_gate_results.json)。
- 纯Decode改动：[decode_only.patch](/data0/ruixuan.zrx/minimax-m31-dev/1009_decode_operator_optimization/decode_only.patch)。已排除原有及并行更新的Prefill修改，msa_attention仅含两处Decode阈值hunk；patch可应用于冻结的HEAD+preexisting.patch基线。
- [source_binary_manifest.json](/data0/ruixuan.zrx/minimax-m31-dev/1009_decode_operator_optimization/source_binary_manifest.json)记录源文件与六个测试binary SHA256。共享文件整体hash包含既有WIP，不表示Prefill内容属于本批。
- kernel活动原始数据：`python_single_op_perf.json`及四个JSON trace；mask性能：`compact_perf_float4.log`、`compact_cpu_perf.log`；LM候选：`lm_head_perf.log`。

独立源码复审发现并修复临时Tensor悬空引用、ready wait异常清理范围、失败请求legacy SP mirror缺失。最终复审未发现新的确定阻塞项。实际CUDA wait故障尚未注入，E2E/动态请求生命周期与服务内存验收仍未做。
