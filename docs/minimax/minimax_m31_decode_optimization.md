# MiniMax-M3.1 Decode 优化、精度与性能结论

合并原逐算子设计、首批审查/E2E、后续算子、空泡分析和Radix死锁报告。过时实施步骤已清理，仅保留方案、结论及证据入口。以下为截至2026-10-12的文档记录，本次整理未重新运行模型；不同日期/运行包的结果不能合为同一验收。

## 实施方案与取舍

| 方案 | 状态与结论 |
| --- | --- |
| 紧凑spec vocab mask | CPU bitset直接H2D，GPU按bit消费；T160/V200064由32010240→4001280字节。保持FP32旧−FLT_MAX哨兵、允许位置NaN payload、其他phase及HIP回退 |
| batch设备状态owner | 消除逐请求Tensor view/重复cat；按ID/cohort/epoch/行顺序校验，保留owner、失败状态发布、pending claim清理与旧getter兼容 |
| T160 norm/RoPE与输出量化 | producer直接生成consumer布局，BF16舍入保持；单算子22.726→10.072μs、4.341→2.005μs，不可相加当模型收益 |
| LM head静态prefix缩行 | M224→M128精度门禁通过但cuBLAS429.002→459.571μs，撤回生产路由，保留全gamma计算 |
| seed/table/完整COMMIT producer | seed上传合并、table完整覆盖、元数据融合；保留Graph代次/别名/容量/stream/readiness检查及完整legacy回退。单算子有效，四组模型未接受完整producer E2E收益 |
| sampler配置fresh slab | 正确性通过，CPU构造/copy约1.985→7.466μs，退役，恢复原字段分配 |
| Champion/Hybrid低精度候选 | 此前GEMM非有限/精度失败，默认关闭；不据孤立按位结果接入 |
| norm/postnorm与静态采样整理 | 保留正零、stride0/TP回退、mask整任务fallback、批量owner/epoch及event-before-read；最终以新wheel模型门禁为准 |
| 私有诊断清理 | 当前29文件净减1222行；周审计35文件增87/删1006、内部2文件增7/删20。去私有日志/marker/影子对照/导出/CI debug，保留标准profiler/NVTX和安全测试 |

模型与缓存/DSpark契约见[设计](minimax_m31_nvfp4_native_kernel_design.md)，配置含义见[环境审计](minimax_m31_env_switch_audit.md)。

## 历史性能阶段：不能累加或替代最终结果

共同诊断主场景为DP8、每DP BS32、80000物理fake history、输出256、gamma7/static verify4、FP32或明确记录的router配置。fake history与强制接受测量不能当真实普通拒绝采样质量/生产吞吐。

| 阶段 | 结果 | 边界 |
| --- | --- | --- |
| 首批compact mask/owner E2E | 768/768，6.096669ms/token、5248.768TPS/card；对历史基线4506.406为+16.47% | 没有新交错基线，不能证明普遍稳定收益；正式与profile分开 |
| seed/table | 768/768正式、256/256普通fake-history功能，5.770009ms/token | 比此前5.623579顺序控制慢2.60%，不接受E2E收益 |
| 完整COMMIT producer四组 | 5120正式+1024普通功能全通过；OFF6.046268568、ON6.157925579ms/token | 单算子10.379→6.226μs不能替代模型结论；未接受E2E收益 |
| 早期完整ABBA候选 | 5428.8407TPS、相对旧池+31.0168% | 最终fresh wheel未复现，历史结论撤回为最终性能依据 |
| 最终fresh wheel七轮 | 1792/1792、TPOT7.371477ms、4341.0567TPS；对旧基线池+4.764789% | 5.688245..10.798983ms波动，全client吞吐620.0521→608.2272TPS；非新最终ABBA |
| 清理后B-C-C-B正式 | B/C各24轮，TPS/card中位4333.313937→5418.698559（+25.04745%） | 观察器/NVML/外部CPU collector关闭；fakeKV/强制接受，四组普通拒绝采样256请求仅独立功能门禁 |

最后一组：候选TPMS5.418699、TPM325121.9135，TPOT中位5.905477ms/p958.133657ms；TPS p053934.514675/p955784.124210，CV11.33063%，基线CV15.40084%。48正式轮及慢轮全部保留，原始计数由独立审查复算。证明中心趋势可重复，不能称波动消除或稳定生产普通拒绝吞吐，更不能与client completion混用。证据 `/data0/ruixuan.zrx/minimax-m31-dev/1012_decode_clean_stable_performance/stable_performance_summary.json`。

Radix修复运行时最终性能/最新timeline仍在 `1012_decode_radix_final_performance` 采集，前述较早二进制吞吐不能自动继承为该修复版本验收。

## GPU 空泡与观察器边界

同进程target Graph1968节点：pristine OFF replay wall/threadCPU中位18.5085/17.6365μs，FULL2084.487/2072.324μs；CUDA-only也约2140.7815/2127.031μs。独立1908节点控制同样显示观察器CPU成本。profile停止后可残留约1.2ms开销，`profilerEnabled=false` 不证明CUPTI卸载；teardown仅孤立试验通过，不能直接设生产默认。

正式性能在profile前或独立engine测，profile置最后；按全记录stream kernel/memcpy/memset并集定义idle，wait kernel计活跃，排队等待GPU不算idle。inclusive CPU、event wait和跨rank耗时不可相加。rank局部ordinal不等于全局逻辑轮，首profiler初始化返回差曾达1.811983s，并被MoE peer等待吸收。

已定位两个不同层次的停顿：

- OFF诊断轮中一rank在proposal提交前约701ms host空隙，其他rank target Graph约724ms等待；不属于初始请求到达。CPU阻塞函数仍未明确，CPU线程组竞争仅假设；不能以缺启动GC freeze解释，日志已有 `gc.collect(); gc.freeze()`。
- 最新八rank observer各48真实BS32 scope/101252kernel/144Graph提交。rank1 proposal索引33有102613.674μs全stream idle，CPU链 `publish_core_tensors_to_cuda → aten::to/_to_copy → aten::empty_strided → cudaEventQuery`，Query102462.022μs，shape[1,32,627] Int32对应新KV表H2D分配。一例定位allocator查询，不能证明所有OFF慢轮同因。

Torch2.11.0上游malloc先处理此前跨stream未完成事件，再选size pool。因此KV表分配处Query未必由该表自身引起，单表池化可能把Query移至下一allocation；这是机制加trace推断，非安装二进制完整C++栈。保留recordStream/lifetime，不能凭长尾删正确性wait、禁event回收或加全局同步。

48正式轮client起点跨度与TPOT相关基线−0.124613/候选−0.223579，不支持简单“客户端起点越散越慢”，但client起点不是服务端到达，不能据此确定driver/allocator因果。证据 `formal_round_arrival_spread_audit.json`。

## 精度测量纠正与最终交付

早期Math预算512得381/500，新8192的40新增正确ID均触及旧上限，零新增错误；不能继续沿用短预算代表完整精度。HumanEval旧last-code-fence提取丢入口实现，旧88/164过时；修正入口感知提取并执行官方测试后142/164，保留安全拒绝及两组相同scorer hash。评测分数必须注明生成预算、提取和官方指标。

### 长 Prefill 不确定性：已定位并修复，独立于 Decode

同输入串行P-only、无D传输/推测verify：短200token稳定，16863/21000首logit最大变化8.98496/9.56688；实际P使用FlashInfer，不是模块默认FMHA。只关wire或overlap不足，FMHA控制12/12按位稳定。

FI使用128槽union hash，>128block/16384token时atomicCAS线性探测位置改变累加顺序。真实不变Q/K/V/scale/TopK输入20replay有19次变化。修复仅>128block对完整entry按block code排序，保留membership全部位、softmax/TMEM/布局保护及≤128历史顺序，用新JIT URI避免旧缓存。

修复单算子20/20、全P12/12首logit稳定；GPU回归7/7含Hkv1/2/4、128/129/132/257block、变长/高ID/Graph/空union。孤立event0.369120→0.375088ms（+1.62%），不能外推模型收益或全生成精度。证据 `1012_flashinfer_union_determinism`。

### M3 前端协议：字段分类修复

共同旧前端忽略显式force_reasoning=False；非stream逐packet无状态解析丢关闭状态；已分类字符串又被CustomChatRenderer独立ThinkStatus二次分类。修复M3显式模式、增量close-state及结束flush，绕过已分类字符串二次解析，保留DeltaMessage工具/推理字段和其他detector兼容。

四D路八enabled/disabled短请求通过，native token与失败前相同；enabled reasoning=`We`、最终content=`\boxed{4}`，disabled仅最终答案。40CPU/零skip及独立review通过，不能替代全质量。通用nonstream usage计数覆盖问题未修复，且非native MAL。证据 `1012_decode_corrected_wire_full_quality/matched_wire_parser_evidence.json`。

### Radix TopK 重归一化死锁与续跑范围

32client PD4+4在5409成功后停滞。CUDA13.2 cuda-gdb定位 `RadixTopKRenormProbKernel_MultiCTA<1024,4,false,float,int>`，60block/1024线程、剩余非leader block23；CPU metrics/transfer event等待是GPU未完成后果，不是已证实其他服务干扰或GraphLaunch瓶颈。

leader通过最终到达屏障后清零counter，延迟CTA可能未观察目标便只见0、永久等待。仅移除该kernel终末histogram/counter重置；调用方每次新分配清零1MiB私有state，pivot/同分/归一化不变。后续复用须在全部前CTA完成后初始化并保持stream顺序。

测试专用延迟复现旧超时exit124，修复完成且3010560概率精确匹配；V200064、B3/15/80、TopK1/8/不限、各12轮独立threshold参考通过。生产无延迟hook；`native_radix_fix_install` Python字节相同，六native仅librtp_compute_ops.so改变。证据 `1012_decode_submission_candidate/radix_renorm_barrier_probe`、`radix_fix_runtime_identity.json`。

完整5681生成完成后，用户将主要LongBench交付收缩为历史八任务160样本，每任务20。**最终候选由5409旧二进制成功响应加272修复后新响应组成，没有新生成基线。** 27传输timeout按状态归档排除，完成错误答案全部计入。不能称修复二进制新生成5681条；native MAL仅新272窗口。

| 主要数据集 | 最终候选 | 结论边界 |
| --- | ---: | --- |
| Math500 | 418/500 | 与基线差异供审阅，不声称零回退 |
| GSM8K | 1137/1319 | 同上 |
| HumanEval | 143/164 | 实际官方测试 |
| AIME24 | 完成最终答案26/30 | 区分旧仅reasoning fallback分数 |
| LongBench历史160 | 66.7683125；基线66.8703545 | 保存响应按同官方指标重评 |
| native MAL | 2.80470635 | 仅272请求修复窗口 |

主要评分/native MAL/sampled16/生命周期门禁通过执行与完整性检查；一次新最终只读review无已确认可执行生产阻塞，但分差不等于精度等价。证据 `1012_decode_latest_candidate_full_quality/{dataset_score_handoff.json,quality_radix_state.json,quality_candidate/resume_radix_checkpoint}`。

历史LongBench160旧标“官方”66.237738使用两TREC text-F1，标签不准确；以同历史scorer重评66.416310→新基线66.870355。13任务3668完整基线56.058134不能与八任务160直比。匹配hotpotqa仍下降4.104747、qasper升5.827373，汇总上升不能抹除逐项损失。审计 `1012_decode_corrected_wire_full_quality/longbench_historical_score_audit.json`。

## 未完成优化与上线边界

- 修复二进制最终匹配性能可重复性及最新timeline；sample/propose CPU空泡和OFF host/allocator根因。
- target每图仍有60次query cast、metadata、valid-prefix、scale refresh；FillFunctor62次不是全O-projection清零，原O-projection zero-fill/where已融合去除。
- Cake只需index-Q FP8；producer接入须独立storage及BF16后E4M3舍入、变长/stride/padding/Graph证明，不可只删cast或多写主Q FP8。
- Cake共享页/seen/prefix/scale适配器，router加载时FP32转换与sigmoid/bias/selection融合，须精确FP32/同分及producer+consumer实测。
- 其余未测beam/多返回、所需CP/HIP/长时间压力及发布CI须各自门禁。历史PD恢复为各活动当时HTTP/native/hash/空闲结果，本次文档整理不核验或改动常驻服务。

证据主目录：`1010_decode_e2e`、`1010_decode_next`、`1011_decode_final_validation`、`1012_decode_complete_quality_variance`、`1012_decode_clean_quality_final`、`1012_decode_clean_stable_performance`，均位于 `/data0/ruixuan.zrx/minimax-m31-dev`。详细过程已从文档清理；原报告历史由Git保留，原始实验材料仍在这些目录。
