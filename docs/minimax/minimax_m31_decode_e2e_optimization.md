# MiniMax-M3.1 Decode E2E 空泡定位与首批优化验证

## 验证对象和边界

2026-10-10，工作树 `rtp_llm_m31_dspark_worktree`。此前逐算子逻辑、是否需要优化及方案见 [逐算子设计](minimax_m31_decode_optimization.md)，实现与单算子门禁见 [operator review](minimax_m31_decode_operator_review.md)。本文件记录用户 review 后授权的 E2E 验证，不把单算子门禁替代模型验证。

冻结 workload 为 TP1 / EP8 / DP8，每 DP BS32、全局256请求，80,000 fake physical history，输出256，DSpark gamma7，static verify4，target KV4/indexer K4、draft BF16 KV，MegaMoE NVFP4、CUDA Graph、异步调度启用。固定阶段序列没有 `engine.normal.cpu_phase_agreement`。这是 Decode perftest，跳过真实 Prefill，并启用 force accept；不作为模型精度或生产正常拒绝采样吞吐结论。

首轮运行保持历史 capture buckets `1,2,4,8,16,20,24,32`。后续同 wheel profiler 诊断双方均仅 capture32，便于缩短冷启动，只有相同 BS32 graph 实际运行；该设置变化必须在性能比较中保留，不能合并为首轮精确契约。

原始测试目录 `/data0/ruixuan.zrx/minimax-m31-dev/1010_decode_e2e`。历史 baseline 来源 `1009_restore_scheduler_dp8_bs32_80k/status.json` 及用户给定旧 timeline。首轮 native wheel来源 reviewed Decode C++；运行时 Python 为历史冻结包加三处 reviewed Decode threshold 修改，隔离并行 Prefill 修改。`initial_source_identity.json`、`runtime_identity.json` 记录实际 SHA256，不以当前 HEAD 代替加载包。诊断版本另有 `runtime_diagnostics_identity.json`。

## 首轮无 profiler 正式结果

3轮共768/768请求成功。平均 Decode 每输出 token 时间6.096669 ms，按 `BS32 / seconds_per_token` 计算，单卡 TPS 5248.768、TPM314926.061。历史同契约值7.101002 ms、TPS4506.406，因此本次测得延迟下降14.14%、吞吐增加16.47%。这是历史同契约与当前运行的比较，没有新跑原代码交错 A/B，不能夸大重复稳定性。

Timeline（8 rank，每 rank16完整轮）：
`/data0/ruixuan.zrx/profiling_results/m31_decode_dp8_bs32_80000_dspark_static4_compact_batch_20261010`

正式性能结果与 profiler 采集分别运行。下表是 FULL profiler trace 的稳态120 rank-step观测，不能替代上面的无 profiler 性能。

| 项目 | 旧 | 优化后 | 实际证据 / 业务含义 |
|---|---:|---:|---|
| dispatch narrow / cat 次数 | 256 / 3 | 0 / 0 | 批次owner快路径真正执行，消除逐请求视图与重复拼接 |
| dispatch CPU ms | 1.118769 | 0.171320 | 保留三个 success mask where，发布owner及row |
| round-state准备cat次数 | 2 | 0 | anchor/committed-end直接取批次Tensor；3个cached input narrow仍在 |
| gather输入 CPU ms | 0.227601 | 0.237695 | 此scope原本就没有narrow/cat，不能声称gather也因此加速 |
| CPU mask准备 ms | 29.121723 | 1.743621 | 包含processor/merge/alloc/packing，不能全归因一个loop |
| mask H2D字节 | 32,010,240 | 4,001,280 | CPU bitset直接复制；GPU按bit消费，不展开bool矩阵 |
| mask H2D GPU ms | 0.590959 | 0.122845 | 所有128轮均为compact上传，未走dense fallback |
| target尾部mask上传前暴露GPU空白 ms | 1.190484 | 0.015493 | 全stream activity union，不是worker总CPU时间 |
| 完整轮间GPU空白 ms | 0.753199 | 0.151000 | 上轮GPU最后活动结束→下一轮gather首copy |
| CPU process窗口内GPU空白 ms | 6.304058 | 2.776856 | 包括所有stream，busy-wait算活动 |

规范比较与逐scope算子计数见 `trace_compare/initial_implementation/{comparison.md,initial_implementation_report.md,phase_operator_counts.csv,annotated_gaps.csv}`。首轮target图节点2268→1968，减少300；它包含T160 norm/RoPE consumer copy与输出quant变化，因此 graph API下降不能单独归因mask或owner。

## 剩余空泡的具体调用链

首轮剩余>=20us大空泡主要关联下一项 `cudaGraphLaunch`，平均1.240407 ms/rank-step。target inputcopy→firstnode真GPU空白平均0.965114 ms，其中0.962048 ms发生在graph API尚未返回时，API返回→firstnode仅0.003054 ms。源码没有逐node发射loop，只执行一次 `graph.replay()`。该形态需要对照 profiler 开销，不能归因DP负载或第一次MoE。

另有一次明确异常：rank1 step7 `CudaGraphRunner::forward` 的 `forward_event_.record()` 主机耗时32.1547 ms，阻挡forward返回后的 `PyWrappedModel` hidden clone提交。target graph GPU最后活动在process+30.829738 ms结束，下一clone copy直到+37.248699 ms，形成6.418961 ms真GPU空白。后一个copy API本身17.666us，主要问题是提交晚。该event保护graph buffer后续使用，不能凭一例profiler outlier直接删除。

## 同 wheel graph replay 诊断

新增opt-in mmap记录graph.replay前后 wall与thread CPU，运行热路径没有写日志、文件IO或device synchronize。env未设置时不创建诊断，不改变正常replay路径。节点数在capture时获取。mmap ABI/容量/发布前缀与进程SIGTERM后的读取通过CPU门禁，CUDA13诊断wheel构建成功。

诊断FULL同一进程8rank、target BS32/1968nodes：

| 条件 | 样本数 | replay wall median / p95 µs | thread CPU median µs |
|---|---:|---:|---:|
| pristine OFF（首次profile前至少1秒） | 2576 | 18.5085 / 34.486 | 17.6365 |
| FULL profiler已开启 | 128 | 2084.487 / 3617.323 | 2072.324 |

因此**毫秒级graph replay CPU开销由开启profiler触发**：同一图、同一进程的CPU实耗增长，绝不能称为生产固有DP协商、MoE peer等待或纯线程抢占。计时只包graph.replay，不涵盖eventRecord，故不能凭这一表解释32ms eventRecord异常。未profile标志但位于profile启停后的样本单独列为transition/tail，避免污染pristine OFF统计。

首轮CPU优化解决真实的mask展开/传输和批次元数据开销；完整profiler会另行引入可见GPU空白。优化和采集手段必须分别处理。CUDA-only对照、正常拒绝采样、服务恢复结果将在完成后追加。

## 同 wheel CUDA-only及正常拒绝采样最终结果

CUDA-only BS32/80K重跑退出0。3轮正式768/768成功，逐响应 `input_len=80000` / `output_len=256`，平均6.362983 ms/output token（单卡TPS5029.088，TPM301745.269）。此诊断设置只capture32，另有opt-in计时，须与首轮原capture配置的5248.768 TPS分别列出。FULL诊断组为6.843688 ms、TPS4675.841；三组都在正式测量后才启用profiler，运行差异/噪声仍在，不能把一次16.47%解释为稳定的普遍收益。

| 同wheel target1968节点 | pristine OFF wall median / thread CPU µs | profile wall median / thread CPU µs |
|---|---:|---:|
| FULL CPU+CUDA / shapes on | 18.5085 / 17.6365 | 2084.487 / 2072.324 |
| CUDA-only / shapes off | 18.299 / 17.441 | 2140.7815 / 2127.031 |

CUDA-only依然存在毫秒提交空泡，不能用“关CPU trace”宣称解决。CUDA-only无CPU scope，离线以127/1968/88节点完整signature和proposal→target→commit顺序识别轮次：8rank各16轮开始、15轮完整，末commit只有37–61个节点（严格subset验证），排除该截断末轮；FULL对照也限制15轮，并各丢第一轮，比较112稳态rank-round。API median FULL2.057576 ms / CUDA-only2.089572 ms；最大CUDA-only真GPU idle3.045016 ms中3.043330 ms与graph API内部重合。5035次eventRecord最大147.647us，无>=1ms；第二FULL 5060次最大207.186us，无>=1ms，原32ms异常两次均未复现，但内部根因仍未定论。

CUDA-only rank1 ordinal1有1197.980ms长MegaMoE kernel，按active保留原统计，不把它计为空白，也不自动称DP负载根因；报告另给共同剔除该ordinal的敏感性结果。完整结果：`trace_compare/diagnostic_cuda_only_retry/comparison_report.md`，输入snapshot保留未被后续移动覆盖的8trace。

正常拒绝采样 (`force_sp_accept=false`) 使用同一engine，256/256成功，全部input80000/output256。原始门禁 `candidate_cuda_only_retry/normal_rejection_gate.json`；这是fake history、top_k1/static4的功能与跨轮状态检查，位于profile之后，不作为正常reject生产吞吐、精度或native MAL。单独统计正式性能时只取profile之前的3轮。

补充脚本最初误把BS1 grid warmup当作BS32 normal gate并断言256，首次CUDA-only运行在8个小批次正常成功后退出；它不计入正式数据。失败记录保留在 `candidate_cuda_only/failure.json`，修复只对 `batch_size == dp_size*32` 执行该门禁，重跑目录独立，未覆盖失败。

## profiler结束后的残留与采集方案

实测同1968节点加法图：pristine OFF5.1955us，FULL1992.2115us，CUDA-only1856.49us。另一独立进程，FULL结束后的replay仍1148.402us；设置 `TEARDOWN_CUPTI=1` 后结束时恢复5.881us。这与模型mmap的profile-transition/tail约1.2ms相符：**仅看 `profilerEnabled=false` 不能证明CUPTI开销已消失**。

[Kineto源码](https://github.com/pytorch/kineto/blob/main/libkineto/src/CuptiActivityApi.cpp)的teardown路径是opt-in异步finalize；[PyTorch官方问题记录](https://github.com/pytorch/pytorch/issues/144455)说明默认不关闭CUPTI及CUDA Graph reinit兼容性历史。上面是本机CUDA13单次micro门禁，不代表多轮profile/reinit、RTP多线程engine与所有Graph资源已兼容，因此不擅自开启生产默认teardown。

可执行的测量约束：正式所有形状先测，profile放最后；正式测量和profiling使用独立engine进程；对常驻生产服务采集后检查OFF graph耗时，不用profile之后的请求证明无采集吞吐。需要每算子duration时保留节点采集，并明确该trace的空白受采集扰动；需要阶段间真实空泡时评估graph粒度采集，分开两种证据。

Nsight graph粒度单算子尝试超过210秒未产出benchmark结果，已停止owned进程并保留partial `.nsys-rep` / SQLite。报告只有3条completed CUDA API及Graph Creation，无replay/kernel activity，并警告未收齐事件；不能解释后续CPU忙内部原因，也不能将该方案列为实测通过。详见 `graph_trace_control/nsys_graph_failed.json`。

采样HBM最大值226385–227051 MiB/卡（约221.08–221.73GiB），是离散nvidia-smi采样最大值，不是精确allocator峰值；物理KV仍21859 blocks，2,797,952 token slots，足够本用例32×80000。没有借用更短sequence、更小batch或降低reserve完成验证；resident/Graph pool精确单变量内存比较仍未完成。

## PD4+4 服务恢复

按用户要求，全部owned DP8进程及单算子任务退出后，启动容器 `m31_decode_e2e_restored_pd4p4_1010`（原冻结镜像、P/D安装包和数学配置），仅调整本次日志/artifact目录及可写JIT目录。P GPU0–3、TP4/EP4/CP4；D GPU4–7、TP1/EP4/DP4；RDMA开启。使用原冻结服务运行包，不代表已部署本批新Decode wheel。

Prefill23100及Decode24100/24109/24118/24127均alive=true、running_query_len=waiting_query_len=0且无running_task_info。四个D分别完成真实token-ID PD请求：64输入、8输出、pd_sep=true；逐增量tensor验证无缺失/重复、词表范围和最终finish，最终全部空闲。服务留运行。验收为 `restored_pd4p4/restore_health_and_smoke.json`，实际8rank worker映射包及SHA256为 `restored_pd4p4/live_runtime_identity.json`。未测试长上下文/压力/新优化版本PD精度，不能借恢复smoke宣称这些完成。
