# M3.1 Decode：逐算子分析后的优化设计与验收

## 冻结范围与结论边界

基线 timeline：`/data0/ruixuan.zrx/profiling_results/m31_decode_dp8_bs32_80000_dspark_static4_old_scheduler_20261009`。DP8、每DP BS32、80K、static verify4，恢复固定Prefill/Decode阶段顺序。实际没有Prefill；fake history KV、force accept5；fake_balance_expert=0。异步stream开启不等于CPU逻辑能全部被GPU隐藏。

基线源码HEAD 399db00d611eddf52263a5ca3e2c09fae6ef01de；Python算子解释使用冻结BS20 isolated_install，BS32复用该wheel。详细台账是rank0第6个采样轮（step_index=5）全部2658事件，跨rank空泡统计覆盖8rank×16轮，稳态去首轮。详细算子文档和HTML保留源码推断与直接trace证据的区别。以下保留原设计，第一批实现与单算子验证见 [review材料](minimax_m31_decode_operator_review.md)。没有改动运行服务，E2E等待用户review。

主要真空泡在CPU尚未提交下一个任务及graph launch尚未完成处。Target图内部没有>=20us空泡。首MoE中的peer等待吸收入场差异，是上游不齐的结果。算子融合主要减少有效任务、内存流量与graph节点数，不能拿其耗时直接当空泡收益。

## 借鉴Prefill优化的具体原则

参考 `docs/minimax_m31_prefill_attention_optimization.md`：让producer直接生成consumer布局；消除不必要物化与重复metadata；有生命周期证明才能复用；保持BF16舍入和量化语义；用全stream GPU union分析critical path。Decode不直接搬用Prefill CP allgather方案。这里的长CPU dense mask及per-request Tensor view是优先对象；不额外扩大80K历史的working scratch，不用更低精度KV换容量。

## P0-A：紧凑spec vocab mask，消除CPU展开尾部

现状：worker拿到512B候选后执行processor/merge，并把160×200064的bitset展开为32,010,240B bool，再上传GPU。D2H sync返回到上传提交平均29.122ms，包含processor、merge、分配与展开，不能全部归因展开loop。大部分已与target重叠，但rank0该轮仍暴露5.423ms真GPU空白；上传约0.592ms。

修改边界：CUDA 的 SpecLogitsVerifyRunner保留CPU processor输出的紧凑mask；HIP 尚无 packed consumer，维持原 CPU bool 展开及 dense 上传协议。CUDA compact mask 大小为 160×ceil(200064/32)×4=4,001,280B；LogitsProcessorStates的MTP_VERIFY消费端增加明确格式字段和GPU bit-test masked-fill路径。优先让logits消费kernel读取bitset直接写旧 `BaseLogitsProcessor::neg_inf` 哨兵（FP32 为 `-FLT_MAX`，并非负无穷），避免先生成32MB bool再读一次。若先做GPU展开作为过渡，要单独标记其32MB写入成本，不能称融合完成。不要改变其他phase现有mask表示。

CPU处理器、mask合并、applied_processors、spec_cap、失败传播维持现有语义。ready_event保护H2D完成，consumed_event保护buffer复用，CPU owner保活到异步copy完成。buffer pool按真正未完成消费者决定容量，不能每轮覆盖同一块buffer。针对vocab尾部、全禁用、单token放行、混合processor及取消做逐元素等价。先在worker增加独立计时，分开processor/merge/alloc/展开/H2D，避免将未被消除的processor尾部误计收益。

验收：dense mask与compact mask作用后的logits逐元素相同（含旧 mask 哨兵、允许位置的 NaN/Inf，以及旧 dtype 转换的溢出拒绝行为）；draft/target接受结果及失败状态相同。相同timeline中32MB H2D变为4MB，CPU展开loop消失，target-end→mask-ready暴露尾部缩短。未profile正式perftest必须另测，不能从29ms推出每轮29ms收益。

## P0-B：batch设备状态，移除dispatch/gather逐stream Tensor view

现状：MtpExecutor.cc旧state准备64次narrow累计257us，GPU空白433us；最后GPU任务到下一轮首copy之间191次narrow累计480us，真空白1.149ms。narrow没有GPU kernel，但CPU构建Tensor对象会迟交任务。各rank已有GPU任务覆盖程度不同，造成下一轮入场不齐。metrics本轮28us不是主因。

修改边界：定义batch device-state owner，持有tokens、accept lengths、committed ends、anchor、hidden/KV引用；stream保存owner+row+epoch而不是每字段立即narrow。稳定cohort直接使用batched tensor；新请求、重排、结束和取消通过GPU row-map gather/scatter更新。buildDSparkRoundState不再逐stream拼anchors/committed ends。真正需要独立Tensor的RPC/外部接口延迟物化view，不在launch前热路径物化。迁移保留旧路径fallback，但不能让fallback每步隐式触发。

正确性契约：普通reject回滚、失败请求where保留旧状态、实际GPU committed length、request顺序和epoch一致；不能把host预分配上界作为真实提交长度。worker和下一graph还消费旧buffer时禁止复用。graph输出会被replay覆盖，需要owner生命周期和completion event证明，不能只凭“两轮轮换”假设无并发。保留固定阶段顺序和PDFusion兼容性，不恢复CPU phase agreement。

验收：稳定BS32热路径的64/191次narrow消失或移到不阻塞GPU的消费端，两个cat不再重新拼完整旧state；取消、混合成功失败、partial batch、换序、cohort变更、普通非force接受均逐轮比较token、accept_len、KV committed end。测完整GPU末端→下一gather首copy，而非dispatch CPU返回→下一process。

## P0-C：定位graph launch毫秒开销，不能盲改MoE

113/120稳态target提交有>=20us空白，条件均值1.959ms、中位2.209ms；首kernel一般紧随cudaGraphLaunch返回约5us。已定位runtime提交边界，未证明内部是driver、CUPTI、线程抢占哪一项。

先做同一wheel同一参数：无profiler正式测试、只采CUDA activity、完整CPU+CUDA profile三组；固定warmup、轮数、sampling和force配置。记录cudaGraphLaunch主机耗时、GPU首node、CPU调度/线程状态、CUPTI活动处理；只增加有界局部观测，不加barrier。若只full-profile有毫秒开销，修正观测方式，不宣称生产优化。若无profiler仍存在，再围绕graph节点数量和提交路径做单变量实验。

后续producer融合减少节点是独立候选，必须报告launch空白是否实际随节点减少。graph_.replay()外不能用额外全局sync“修复”。也不优先加入新的CPU线程去争抢同一GIL/driver锁。

## P1-A：提前准备Sampler与commit metadata

rank0 target sampling输入前空白473us，reject后commit metadata前265us。将稳定temperature/topk/seed/config以及host小数组预打包到固定pinned staging，在target运行期间准备；一次合并上传替代零散分配/toCUDA。动态request配置用config epoch失效，seed及RNG offset仍按request row归属，禁止跨请求复用随机状态。

commit的cu_seq、cu_kv_seq、padding_offset由一个GPU producer直接写graph-owned布局，移除重复fill/contiguous copy；不可改为host猜测GPU接受长度。KV block table仅在allocation/cohort epoch未变时复用，不是shape相同就复用；80K跨page边界必须更新。proposal/target/commit metadata buffer生命周期分别受consumer event保护。

验收：小H2D、copy/fill数量与提交延迟下降；page边界、长度0/1、最大长度、partial batch、重排对齐一致。不同graph/replay不会消费前轮metadata。

## P1-B：force mask批量生成与greedy采样冗余

perftest force override目前32次GPU标量fill，合并成一个批量mask producer或一次小H2D，再做batched override。仍对失败请求保留accept1/失败fallback；不能提前跳过正常验证路径得到虚假benchmark加速。单独报告benchmark辅助代码节省与真实服务节省。

greedy候选：processor完成后，top_k=1分支直接argmax logits并生成后续需要的一热target distribution，研究省掉softmax和renorm。普通rejection仍需要target distribution，不能只产token。仅当temperature、非有限值、tie规则、all_probs、cum_log_probs和RNG契约都证明等价才启用；其他情况沿用原路径。此项须有单算子精度及性能证据再合入。

## P1-C：static4只减少proposal LMhead的无用行

proposal gamma7 backbone输出224行；static4实际只采128行，但当前LMhead仍处理224行。保持7行query block及其attention关系，仅改变LMhead gather索引为每请求被采样的前4行，并显式定义head_steps=4契约。SpeculativeSampler现有[B*gamma,V]检查和view也要同步改为显式head_steps，禁止修改backbone width来冒充等价。

留存4步Markov顺序依赖；未采3步的RNG reservation保持现有逻辑。验证相同backbone前缀hidden/logits、采样token/概率、后续target RNG位置及置信度输入。该优化是删掉LMhead冗余行，不能声称整套5层proposal减少3/7。

## P2：图内算子优化

详见逐角色矩阵及全部事件决策。优先研究Gemma norm/RoPE直接产consumer布局（T160不命中现有<=128 contiguous分支），避免Q/K/indexQ/indexK独立copy；保持BF16主Q，不能因indexQ是FP8就将主Q改FP8。量化scale直接产UE8M0消费布局，减少transpose/pack。允许拓展attention输出量化的小T路由前必须microbench T160；现有<=100路径仍先生成BF16 attention output，不能宣称已融合attention epilogue。真正epilogue量化是另一项更复杂实现。

shared expert与routed pack可研究独立工作重叠，但共享SM可能拖慢persistent MegaMoE，不能仅凭“两条stream”判定收益。TopK到MegaMoE该轮48.155us中约46us为shared expert、激活量化、GSF及pack，末pack到Mega仅0.364us；不是48us空泡。不优先改已98.57us量级的非首层MegaMoE或给首MoE加barrier。

保持量化RNE、UE8M0、E4M3/E2M1、global scale、BF16舍入、tail mask和CUDA Graph稳定地址。每项单独对照，不能把小算子duration sum视作可得端到端收益。

## 验收顺序与未完成项

1. 冻结基线wheel/source/config及正式BS32与BS20；性能正式结果与profile另列。
2. 每项改动先有源码链、tensor/生命周期契约，再单算子正确性+性能；一次只改变一项。
3. exact DP8 BS32/80K/static4 matched trace：全stream空泡、节点数、mask tail、graph launch、完整轮间gap、各rank入场skew；同时无profiler报告单卡TPS/TPM。
4. 非force真实接受、动态verify、普通decode/PDFusion：token、概率、拒绝回滚、EOS/取消、并发cohort变化、worker异常。不能以force接受case证明真实speculative正确性。
5. PD4+4真实模型质量和长请求稳定性，resident/peak memory、CUDA Graph pool、KV容量；确认新scratch没有侵蚀80K服务容量。

第一批已实现 compact mask、稳定cohort批量设备状态、force mask批量上传及T160两项producer优化，验证范围见review材料。static前缀LM head候选按位正确，但实际cuBLAS微测未显示收益，本批撤回生产路由。P0-C graph launch定位、P1-A通用sampler/commit metadata改造、greedy省softmax等仍待后续阶段。异步调度已开启，但CPU dense-mask和逐stream对象工作仍会在GPU任务耗尽后暴露；优化目标是及时提供下一段合法设备工作，而不是再加同步屏障。单算子结果不能证明完整轮间空泡已消除。

## 完整算子清单入口

- [交互逐事件清单](/data0/ruixuan.zrx/minimax-m31-dev/1009_old_scheduler_decode_analysis/operator_explorer.html)：2658事件逐条逻辑、shape、耗时、源码、优化与验证。
- [完整Markdown](/data0/ruixuan.zrx/minimax-m31-dev/1009_old_scheduler_decode_analysis/operator_complete_report.md)：全部算子与方案。
- [Target 59角色优化矩阵](/data0/ruixuan.zrx/minimax-m31-dev/1009_old_scheduler_decode_analysis/target_operator_optimization_plan.md)。
- [其余390事件优化方案](/data0/ruixuan.zrx/minimax-m31-dev/1009_old_scheduler_decode_analysis/non_target_operator_optimization_plan.md)。
- [空泡逐段证据](/data0/ruixuan.zrx/minimax-m31-dev/1009_old_scheduler_decode_analysis/gpu_idle_report.md)。
