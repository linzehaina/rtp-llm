# MiniMax-M3.1 Decode 第二批算子与同步门禁

最新空泡、gate dtype 区分、Graph launch 独立用例和待优化清单见 [2026-10-10 Decode 空泡分析](minimax_m31_decode_bubble_analysis_20261010.md)。候选清单不代表已完成模型收益验证。

2026-10-10。完整逐算子计算逻辑和优先级沿用 [Decode设计](minimax_m31_decode_optimization.md)，第一批/E2E证据见 [首批E2E](minimax_m31_decode_e2e_optimization.md)。本批根目录 `/data0/ruixuan.zrx/minimax-m31-dev/1010_decode_next`，仍处于验证中，不能当作全部优化完成。

## KV 地址查找与持久缓存写入

原路径每层先对 token-row `seq_lens` 计算 `pos=lens-1`，`column=max(pos,0)//128`，从对应物理页表读取 page ID；仅正 page ID 合法，物理 slot 为 `page*128+pos%128`，非法行给 -1。下一 kernel 每个 token/plane 一个 CTA，写 K、V、共享 indexer-K 的 NVFP4 bytes 及 E4M3 group16 scales；MMA scales 保持 128x4 swizzle。60层重复一次临时 slot tensor及launch。

新 Decode-only writer 在 CTA 内读取当前长度和页表，按原 int64 算术计算 slot，然后执行相同量化/store。无跨轮地址缓存，无新 workspace，无数据精度变化。限定 int32 metadata、d128/page128及有界 pool；int64/generic仍 mapper+旧writer。MSA 保留 static paged gate，逻辑页和物理页相同才选择新入口；不同页仍使用旧 `self.page_size`。V保留 fused projection 的 strided view，不增加 contiguous copy。

6项 CPU/GPU tests及5项冻结MSA route tests通过，包含独立缓存字节oracle、完整未写区/canary、无效page/pool/极值长度、Graph输入变化与fallback。T160/H4/D128/page128，80K位置、626列table及strided V的无profiler Graph event整链中位数 **3.585600→2.426027us**。这是单算子门禁，不能乘60后宣称整轮一定节省对应时间。

## Routed MoE NVFP4 打包

保留逐行 GSF=max(abs(BF16 hidden))/(6*448)及1e-30下限。group16量化 scale 使用原 E4M3 ties/subnormal/clamp；值编码E2M1并打包，同步写router IDs和weights。原T160走 scalar M4；只将 hidden6144/topk4/T160 加入已测 vector M16 selector，显式BLOCK_M环境覆盖仍优先。

24 cases覆盖T159/160/161、contiguous/strided输出、random/zero/signedzero/subnormal；五输出与旧default和DeepGEMM oracle逐字节一致，canary及Graph动态输入通过。接入后原5项GPU测试通过。T160含GSF的整链中位数 **13.856750→4.556250us**，不含GSF的pack-only **12.226000→2.907250us**。没有改变MegaMoE通信、expert分配、FC1/FC2或precision。

## Sampler 候选不接入

直接省softmax或onehot会改变概率舍入及ties。隔离max-pivot top-k1候选虽然T160/V200064测得201.416636→139.015837us，但uniform T32/V200065严格字节门禁失败，baseline本身也不稳定；不放宽容差，也不以actualshape通过覆盖失败。该候选未接入native库。保持dense RNG reserve、renorm sum/PTX reciprocal及正常拒绝采样契约。

## Metrics 同步定位

三组FULL八卡timeline均是 dispatch→consume→同step report。最新control稳态120steps的metrics EventSync平均23.463126ms，期间GPU活动22.723823ms，空闲0.739303ms；其中0.587297ms位于target图内，0.152006ms位于图后。8B metrics D2H完成后到sync返回之间GPU空白仅4.207506us。120/120返回时commit Graph仍未结束，平均还剩350.527us。不能把23ms全算空泡，不能把重叠的main/worker wait相加。保留same-step最后一轮指标和统计窗口。

## 独立 EventRecord/Graph 计时

新native计时仅环境启用时生效，保存wall/thread-CPU到预分配mmap，无热路径文件写入、日志或同步；保护性event保留。匹配control的target OFF2576次Graph launch中位18.1435us，FULL128次中位2101.57us。target OFF2568次EventRecord中位1.343us/最大15.283us，FULL128次中位4.097us/最大292.706us。未复现历史32ms异常，不能声称已证明其driver内部原因。

## E2E 状态

冻结TP1/EP8/DP8、BS32/global256、80K fake history、输出256、gamma7/static4、KV4/indexerK4+draftBF16、async/drop-broad-sync开启、capture32、reserve45000MiB/KV131072MiB。三组使用完全相同的六个 native 库。control exit0，3轮768/768，6.1080906862745ms/token，单卡TPS5238.952996/TPM314337.179753。首次 candidate 也完成768/768，但为9.034411371527765ms/token，性能门禁失败。其 KV 公共文件复制了工作区中无关 Prefill 修改，属于运行时隔离错误；保留原始运行时和失败结果，不能将这些修改直接认定为回退原因。clean_candidate 从 control 公共文件出发，仅新增两个 Decode 定义，原有56个顶层 AST 节点全部相同；MSA 接线和打包策略保持最小差异，重新通过6项 writer 和5项路由测试。干净组首次启动在rank2 Graph clone时CUDA OOM（本进程216.40GiB、free76.44MiB），尚无正式请求；外部verify_copy_gpu2的运行时间覆盖异常窗口，未同时采样其内存，故仅记录并发干扰嫌疑。该容器已自行退出。缓存/预留容量不变，在clean_candidate_retry重新启动并完成。正式3轮768/768，分别5.986686596、5.931383655、6.762478876ms/token，平均6.226849709ms/token，折算单卡TPS5139.035226、TPM308342.113548；对照6.108090686，不能宣布E2E提升。正常rejection功能检查256/256在首次profiler之前完成，仍非真实模型精度/MAL。八份FULL timeline归档到 `/data0/ruixuan.zrx/profiling_results/m31_decode_dp8_bs32_80000_dspark_static4_next_ops_clean_20261010`。

首次 candidate 的 FULL 稳态图节点1968→1908；mapper+writer 合计0.264101→0.170671ms，input pack合计0.741742→0.180268ms。所有真实 kernel/memcpy/memset 的时间并集之外，target空闲均值0.663368→0.6422ms。该 capture 的均值被15个异常长的首个MegaMoE kernel 主导，最长8.278秒；这些属于GPU活动，不归为空泡。去掉双方相同 ordinal 集合的敏感性分析显示普通 step 更短，但不能用这一筛选抹去异常或解释正式吞吐回退。详见 `1010_decode_next/regression_gpu_audit/report.md`。

本批单算子门禁及新的metrics/EventRecord证据不代表正常真实模型精度、dynamic verify、普通PDFusion/PD吞吐或所有86类候选均已验证。PD4+4须在本轮GPU工作后恢复并检查真实请求链路。

## 干净组剩余空泡定位

干净FULL target graph节点1908，120个稳态ranksteps的target空闲平均0.640778ms，对照0.663368ms。target跨度中位25.108712ms，对照25.856115ms；保留全部异常后的平均34.451854ms，对照26.805294ms，不以中位数替代异常证据。pack family平均1.358669→0.796388ms；mapper+writer0.264101→fused0.169869ms。

两个rank1实际GPU空闲异常定位到分配/提交前的cudaEventQuery：ordinal2 main gather查询61.969515ms，对应62.079803ms空闲；ordinal7 worker查询123.121061ms（无CPU scope，无法区分host/device分配器）、main等待specworker125.519ms，target结束后连续GPU空闲101.453447ms。ordinal7 main另有CUDA目的empty_strided分配中的5.945983ms查询/6.060430ms空闲。这里明确区分了CPU allocator/API阻塞、真实all-stream空闲、以及其他rank首个MegaMoE的GPU活动；driver内部锁/CUPTI因果仍未证明，不能直接移除生命周期事件。无profiler阶段Graph提交中位18.8595us，forwardEventRecord1.38us/最大16.924us，历史32ms记录异常未复现。

## PD4+4 恢复验证

冻结原P/D包恢复，容器 `m31_decode_next_restored_pd4p4_retry_1010`，P0–3/D4–7。首次恢复因host共享shm旧root锁名与private容器PID冲突，改host PID namespace避开，未删除旧锁。5个HTTP端点alive/idle，4个D lane真实native PD请求各64输入/8输出、pd_sep=true且完成，8rank库路径/SHA核对通过。证据 `1010_decode_next/restored_pd4p4_retry/restore_health_and_smoke.json` 与 `live_runtime_identity.json`。这是原服务链路恢复，未将第二批候选发布到PD，也不等于模型精度或PD性能签收。


### CPU staging / 主线程 deferred core 下一轮（2026-10-10）

SpecLogitsVerifyRunner CPU pinned staging 的 9 项 native 校验通过；单算子相同 draft D2H/既有同步/CPU复制/H2D 整链 200.8095→200.3 μs，不能扩成显著 E2E 加速结论。主线程仅在真实、非空、纯 Decode、有效且顺序一致的 DSpARK batch owner 下，保留三组 fresh pageable CPU core 数据供校验和 owner-loss fallback，延后其无用初次 CUDA publication。DSpARK proposal/static/adaptive verify 始终重建 CUDA core，所以该路径不依赖全局 RTP_LLM_DEVICE_INPUT。历史三轮全部 8 rank 的该开关均 unset→0；下一轮保持 0，不启用其他旧 MTP 行为。

修订版 native：完整 42/42，unset/0/1 各 4/4 定向检查；普通 gather 回归 6/6。新增 MROPE factor3 与已分片 CP2 page metadata 的精确对照，仅证明 gather 保留 metadata，并不证明真实 TP/CP collective。测试首版漏声明两组 cache_specs，原 legacy gather 先失败；已补齐真实 spec，并保留 default_v2 失败记录，修正结果位于 main_input_gate_default_v3。

隔离运行包 runtime_cpu_staging 的 905 个 Python 文件与 runtime_clean 完全一致；六个 native 库中仅 libth_transformer.so 更新。首次 exact DP8/BS32/80K/static4 CPU arm 在正式请求前失败，原因是测试准备漏建 Graph replay 诊断目录；零正式响应，不能用于性能结论。恢复脚本同时暴露宿主 /usr/bin/python3 为 3.6 的解释器错误，重试 controller 改用自身 3.12 interpreter，实际模型 runtime 仍为 conda310。完整日志保留在 cpu_staging_candidate，当前相同候选/容量/workload 的 retry 已补目录并验证 exact runtime UID 可写。

CPU staging/deferred-only retry 完成，正常 rejection 功能检查 256/256、正式 OFF 768/768；但三轮 10.718361/8.720703/13.159835 ms/token，平均 10.866300，单卡 TPS2944.884692，性能失败。FULL 证明旧的三组 main core pinned/H2D 在稳态 120 steps 消失，但长查询转移到其他 pinned 分配：rank0 ordinal3 的 KV table [1,32,626] 查询 60.512414 ms，对应连续 GPU 空闲 60.568 ms；rank0 ordinal9 的 sampler [160] 查询 121.975011 ms，暴露 99.763 ms 空闲；rank1 ordinal7 同类查询 84.601760 ms，暴露 61.717 ms 空闲。不能以减少分配数宣称消除根因。失败 timeline 独立保留在 m31_decode_dp8_bs32_80000_dspark_static4_cpu_staging_regression_20261010。


### Pinned allocator 后台轮询：单变量 E2E（2026-10-10）

安装 Torch 2.11.0+cu130 的 CachingHostAllocator 源头：默认 eager pinned allocate 会在请求线程执行 process_events/query_event，检查释放 buffer 的异步拷贝是否完成，再决定是否可复用；这不是业务 phase 协商，也不是需要删除的 metrics synchronize。allocate 内处理已释放 buffer 的多个事件会让新的一次 pinned empty 背上历史事件轮询耗时。现有生命周期事件必须保留。开启支持的 pinned_use_background_threads 后，非 capture 默认池的事件处理改在后台线程进行。driver/CUPTI 为何令某次 query 达到 60–122 ms 仍未证明；此次验证的是把这项轮询移出请求关键路径。

先用独立 pinned H2D/Graph 测试验证 buffer bytes 与 event 生命周期，foreground main query45次，background main0/后台226次；该小测试未复现毫秒长尾，所以不能单独证明真实空泡解决。随后 exact DP8/BS32/80K/static4 用完全相同 runtime_cpu_staging、容量与 workload，仅将配置改为：

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,pinned_use_background_threads:True
```

正式 OFF 三轮 5.496053202/5.424919179/5.949764246 ms/token，768/768 成功，均值 **5.623578876 ms/token，单卡 TPS5690.326518 / TPM341419.591059**。正常 rejection 功能 gate 256/256 在 profiler 之前完成。相对同运行库的 foreground 10.866300 ms/token，延迟减少48.25%；相对当日早先 clean6.226850，吞吐提高10.73%，相对 control6.108091提高8.62%。历史 control/clean 非交错 AA，不能把差值全部分摊给每项 C++/kernel 优化。

FULL 稳态 ordinal1–15 共120 ranksteps：整轮 all-stream idle 均值 **5.442715→2.877179 ms**、最大 **104.516548→5.398949 ms**，中位3.183130→3.155088 ms；target 图内 idle 均值0.641574→0.638055 ms，变化很小。target span 均值42.669646→24.656324 ms、中位25.090041→24.644962 ms。首 MoE 等待所在的 GPU 活动算入 active，不改称空泡；MegaMoE 合计均值24.533401→6.525296 ms，说明 peer 晚提交引出的长活动等待一并缩短。逐线程 API 审计见 `1010_decode_next/cpu_background_idle_audit`：128步骤全部main Query3444→411，最大121975.011→10.952 us；剩余关联 empty_strided/resize/empty，没有 pinned=True CPU allocate scope。新出现20263次仅轮询线程Query，该线程与spec H2D线程不同，没有被记录的memcpy提交；线程名称/动态调用栈未捕获，所以后台host allocator归属基于配置、源码与线程行为联合归因。最长后台Query49.256238 ms期间，62次main pinned allocate继续，最大4.224 us；GPU idle总6.034874 ms、最大连续1.794861 ms，主线程跨完整step持续推进。全128步骤最大单个连续GPU空闲99.762934→3.577169 ms，区别于120稳态步骤的每step空闲时间总和。长尾未转移到同级GraphLaunch/EventRecord/H2D/HostAlloc；driver内部长Query原因仍未证明。

原始 graph 提交 OFF median19.470→19.3055 us，FULL3189.237→3205.122 us；EventRecord OFF1.340→1.320 us，FULL4.082→4.210 us。残余 FULL 约3 ms提交不由此设置消除，不能冒充生产 OFF 同等空泡。八卡 timeline：`/data0/ruixuan.zrx/profiling_results/m31_decode_dp8_bs32_80000_dspark_static4_pinned_background_20261010`，manifest 包含 SHA、冻结运行库和请求指标契约。

本轮结束后恢复容器 m31_decode_next_restored_pd4p4_background_1010，P0–3/D4–7 使用原冻结包；五 HTTP alive/idle、四路 native PD64输入/8输出、八 rank 库路径/SHA 全部通过。该恢复不是将优化配置发布到 PD。合成 fake-history/force-accept 性能通过不等于真实模型质量、动态请求生命周期、optimized PD 性能与峰值内存签收。


### Sampler seed/offset：动态数组合并（2026-10-10）

原 helper 对每个 dense row 按原 generator mapping 调用 get_seed_and_offset(32)，分别填两块 pinned Int64[R]、两次 H2D。新 helper 仅改为 fresh pinned Int64[2,R]、相同循环与 cast、一次 H2D，返回两个 contiguous row views，共有同一 fresh CUDA storage。保留 dense reservation 后才 index_select live rows，默认/undefined/重复请求 generator、Markov/skipped/rejection RNG 顺序未动。高位 seed 仍通过 Int64 bits ABI，未改变消费者概率或采样数学；host owner 仍由 TensorHolder 和 pinned allocator copy-event 保护，不增加同步或复用池。

新增3项native门禁全部通过（R0/1/3/32/160/224、多mapping、高seed、合法offset>2^32、最终state/next draw，真实dense/compact top-k/top-p/joint旧新消费者，双非默认stream24轮null/释放holder+allocator压力）。既有CUDA Compact sampler10/10、SpeculativeSampler21/21通过，共34个实际执行的正确性/回归测试，不把XML里的disabled tests计为执行。首次R0测试误要求空tensor is_pinned=true，实际旧新空tensor都没有host allocation；修正对照断言并保留 attempt_v1。回归runner首次漏runfiles环境，fixture启动即abort；修正后通过，失败记录单列 regression_attempt_v1。

独立review无剩余源码阻断。T160/V200064/k4，500个交错pair、同pair相同RNG state、实际 top_k_sampling_from_probs 消费者、无CUPTI：CPU构造+API提交 raw median98.864→91.7605us；CUDA event构造+copy+top-k链 median959.696→951.088us。原C++日志打印上中位数，与CSV conventional median略有差别，以 seed_offset_gate/summary.json 和 benchmark_samples.csv 为准。事件包含CPU提交间隙，并非纯kernel时间；不含softmax/renorm或外围compact处理，不等于完整Sampler/Decode收益。该改动已在工作区接入，尚未更新模型wheel/E2E/PD；服务仍是原冻结包。


### Commit table 完整覆盖：隔离 native 门禁（2026-10-10）

最新 FULL 的 commit 表仍是先清零、再按byte作二维 strided copy。候选每个destination int32只写一次：落在当前source矩形内取src[row*stride+col]，其余填0；保持signed page bit模式，不缓存页表、不改变source ready或任何KV/row/math。

隔离extension直接编译冻结的原native cuda_graph_prepare.cu 与 fuse_copy_kernel.cu作baseline：fill+连续metadata copy+strided table copy；candidate复用原连续copy，加一个全表覆盖kernel。12组case包括0/小/奇数列/625–627/live窄于capture/不同rowstride/多table、毒化目标、完整bytes/零padding/canary；双臂Graph相同地址4轮fresh值、跨streamproducer ready→consumer wait、alias/type/inner-stride/capacity拒绝前无写，以及计时后完整oracle全部通过。

初版用了711列和两组metadata，结果独立保存，不作为实际形状最终门禁。COMMIT static4 的sp_steps取4，capture width=ceil(90112/128)+4=708；cu/cuKV各33、input/prefix/sequence_lengths_plus_1各32，共5组648B，decode_cu在context branch无source故skip。最终无CUPTI的15组交错paired Graph测试、每graph32重复×100replay、两表src[32,626] contiguous/capture[32,708] allocator-aligned，全子链median **14.344950→5.718950us**。builder、tableH2D和token/features payload保留且不含在此计时中，不能称整个prepare或E2E收益。

一次matched脚本替换错误导致NameError，发生在native table case之前，保留matched_attempt_v1；修正后最终pass。源码与证据位于 table_overwrite_gate；尚未将候选接到真实CudaGraphRunner，必须继续实施COMMIT capability、table-only填充覆盖、source/destination/metadata别名检查及完整legacy fallback，并复用原replay/worker/sourceready生命周期，再做native/e2e验证。


### Table overwrite 已接入生产 COMMIT 源码并通过实际 runner 门禁（2026-10-10）

MiniMaxM31DSparkModel 的 CP1/cacheCP1 暴露显式 capability，PyWrappedModel 仅在 COMMIT role 读取，GraphParams 默认关闭；普通 target/propose/其他模型沿用原路。每轮 rebuild host-only descriptor，source/destination 是既有 Tensor，不新增 GPU buffer、不缓存 table values/shape、不改 ready/consumed event。

资格 helper 要求全部 fill 都是匹配的完整 table zero-fill；2D Int32 同设备、连续 destination、合法 source row stride/live rectangle、容量匹配；检查所有 table destination/read span 与彼此及连续 metadata source/destination storage 的别名。任何条件不满足执行完整旧 fill+copy。成功时保留原 5 metadata D2D，把旧 fill+strided byte-copy 换成完整 Int32 overwrite；padding 全写0，undefined/empty source 保留 fill-only 语义。真实上游 [group,B,C] 经 PyWrappedModel 的 group 索引为 [B,C]，不是从 profiler allocation shape 推测输入 ndim。

新增 production native **4/4**：实际新旧 CUDA kernel + independent CPU bytes + guard、Graph 四轮 fresh values、fill-only/非tablefill/容量/dtype/别名拒绝、跨stream source-ready/multi-table。actual CudaGraphRunner Python **2/2**：两 table 626→708、完整5 metadata/CPU padding逐byte对照；profiler kernel names 明确断言 optin dense/fill-only 命中 overwrite，legacy/alias/capacity/padded B31 均不命中，并验证 fallback→fast 重入。最小 mock forward 在捕获图中实际读取 table/metadata，返回 checksum 校验，不是只验证 descriptor。测试工厂新增的 capability 参数默认 false，保留既有测试调用。

证据目录 `1010_decode_next/table_overwrite_gate`：production_native.xml/log、actual_runner_v1.log、production_summary.json/source SHA、production_review.md、build_production_v4.log。实际 runner 采用 inline prepare/replay；尚不证明 async worker/cancellation teardown 或真实 KV writer/模型质量。先前 matched isolated device chain **14.344950→5.718950 us** 仍只用于该子链，不包含本次 host eligibility 扫描，也不能直接当成 E2E 收益。

### Sampler config fresh slab：正确性通过，CPU 性能失败，已撤下（2026-10-10）

实作 fresh pinned Byte[29R] + 七32bit字段/一个bool的独立 typed views，CPU fill/normalization/H2D 条件未改。6项新增门禁通过：R0/odd/highR布局+独立memcpy byte oracle、实际 stream score/beam/多return fill、fresh config与owner、字段写隔离、非默认stream排入10ms callback后异步field/nested-offset H2D+source release+64轮CPU allocation pressure、实际 execSampleGreedy token/prob bits/RNG/final normalized fields 新旧对照。既有 normal processor6/6、MTP processor42/42，共54项实际执行测试；XML disabled test未计入。

独立 C++ paired CPU bench 固定 background allocator，两个shape各9轮×1000pairs/phase，所有IO在计时后。构造+析构：R160旧 **1.921us**、slab **7.363us**；加相同CPU memcpy：旧 **1.985us**、slab **7.466us**，p95 **2.148→7.677us**。R32 加copy **1.938→7.354us**，9轮一致。phase仅8个config字段 allocation/views/optional memcpy/destroy，不含H2D、sampling、完整gatherer/模型。减少allocator调用却增加narrow/dtype-view元数据开销，性能为负。R160 payload4640B不等于allocator省内存：power2 buckets旧7424B、新8192B，且共同storage会耦合字段保留时间。

因此生产 Gatherer **逐byte恢复本候选前快照**；候选源码、测试、BUILD stanza 与全部 native XML/raw JSONL 保存在 `1010_decode_next/sampler_config_slab_candidate/retired_source` 与 `decision.json`，不进入新候选包。继续保留原8字段分配，由已验证 background host allocator 移出主线程 Query 长尾。不因allocation计数更少而启用更慢的路径。

新的 seed+table candidate wheel已构建并冻结于 `runtime_seed_table`：905个Python文件仅增加COMMIT capability那一个文件，全部6个native库来自同一个新wheel，其中librtp_compute_ops.so/libth_transformer.so变更；来源见runtime_seed_table_identity.json。8卡模型E2E及真实quality/dynamic/optimized PD/memory仍待验证。当前原冻结PD4+4未停止，五HTTP alive/running0/waiting0。新增源码门禁与原PD恢复健康均不能代替完整模型签收。

## 2026-10-10 seed/table DP8 E2E 门禁

三轮 OFF 全部768/768成功，普通rejection功能256/256成功；均值5.770009ms/token，单卡TPS5545.92/TPM332755.13。前一轮相同pinned-background控制5.623579ms/token/TPS5690.33，非交错三轮候选延迟+2.60%、TPS-2.54%，范围重叠，本轮不接受E2E提速结论。fakehistory/forceaccept指标不能证明真实模型质量或纯model_forward性能。

8rank FULL 各16local step：128/128 COMMIT命中新overwrite，旧fill/strided0；builder/metacopy各128且proposal/target无误启；128/128seed H2D 2×1280B→1×2560B，bytes不变。COMMIT准备kernel-sum steady median33.8735→17.12us。GPU全stream并集求idle：steady每step总idle mean2.877179→3.004683ms，target内idle0.638055→0.641047ms，全部128step连续idle最大3.577169→3.543167ms，残余空泡未改善。请求main EventQuery≥1ms0，max8.977us；background仍有64.780msQuery，不宣称driver长尾消失。

FULL中秒级targetspan是MegaMoE活跃kernel，不能算GPUidle。rank0/1/4与其余rank的profile起点差1.25–1.38s，local ordinal不能对齐globalstep；未直接记录profiler.start/stop/export调用链，边界因果未证实。targetGraphLaunch FULL median≈3.203ms而OFF18.582us，不能把FULL几毫秒launch当生产收益。

完整结果：`/data0/ruixuan.zrx/minimax-m31-dev/1010_decode_next/seed_table_e2e_report.md`。公开8timeline：`/data0/ruixuan.zrx/profiling_results/m31_decode_dp8_bs32_80000_dspark_static4_seed_table_pinned_background_20261010`。真实模型精度/native接受率、动态生命周期、完整metadata producer、优化后PD4+4吞吐/peak memory仍待验证。

## 2026-10-10 CUDA-only同包复测

同source/runtime/workload，仅FULL/on→CUDA/off。formal768/768、normal fake-history functional256/256，native response6.127287934ms/token，单卡TPS5222.54/TPM313352.34，不接受E2E提速。8rank识别128targetgraph×1908kernels+57Mega，120完整GPUcycles，cpu_op/InputDims0但CUPTI/runtimecorrelation保留。相同cycle定义FULL→CUDA-only idle median3.320205→1.587121ms，mean3.184537→1.966406ms，最大连续gap3.543167→2.216863ms，>3ms gap7→0；candidate代码未变，此下降只证明FULL采集放大空泡，不作业务优化收益。

GraphLaunch OFFmedian18.712us、CUDA-only2000.4025us、此前FULL3203.4505us；CUDA-only仍不能当OFFGPUidle。最大CUDA-onlygap的2216.409/2216.863us与targetGraphLaunch重叠。OFF4038调用仍有一次1.186ms/threadCPU22.035us，OS/driver因果未证实。最长664ms GPUspan为MegaMoE活跃wait，不能算idle或归因DP业务负载。

完整报告`/data0/ruixuan.zrx/minimax-m31-dev/1010_decode_next/cuda_only_e2e_report.md`。原冻结PD4+4恢复并验证5HTTP/4真实PD/8rankSHA，常驻m31_decode_next_restored_pd4p4_cuda_only_1010。Full COMMIT producer及受限stream-eventwait仍待native生命周期与E2E门禁，不以设计替代实施。


## 2026-10-10 Full COMMIT metadata producer: focused gates

Static verify4 COMMIT now attempts the compact path before allocating cu/cuKV/padding or launching implicit prefix+1. Exact graph key and real token/hidden geometry are checked, then a host-only all-or-nothing alias plan validates full storages. One launch produces live metadata and fully overwrites block tables. Legacy decode-style cuKV tails and live sequence/decode-cu bytes remain unchanged; captured CPU padding remains unchanged. Non-COMMIT, CP/cacheCP>1, ragged, eager/uncaptured geometry, position/MM/extra derived-source inputs use complete legacy preparation.

Prepared state records capture generation, batch/tokens and weak TensorImpl identities with Torch mutation versions. Wrapper and native runner both reject stale preparation. Explicit KV refresh invalidates it. Raw CPU/CUDA edits must obey the existing prepare-to-forward immutability/invalidation contract. The runner records allocator streams for raw-pointer metadata/table reads; existing input-ready/forward-ready events, CPU forward-event wait, TensorHolder, result clones and publication wait are retained. Large target features are not retained merely for the identity check.

Stable CUDA13 production/models/factory build passed. Native correctness 5/5, table regression 4/4, actual native CudaGraphRunner 10/10, legacy runner regression 2/2 passed. The 10 runner tests also passed with the frozen new wheel libraries. They cover bytes/canaries, changing values/buckets, padded tails, aliases/capacity, complete fallback, in-place updates, delayed side-stream consumers, source-owner release under allocator pressure, and preparation failure recovery. They do not instantiate PyWrappedModel or native AsyncRunner; actual model routing/math/quality remain an E2E gate.

Paired 9x40 microbenchmark, B=C32/T160, one strided 626-to-708 table: old builder+optional prefix add+five copies+table overwrite versus full producer. Device graph-chain median was 10.379 to 6.226 us (implicit plus1), and 10.233 to 6.225 us (explicit). Native host submission was 17.166 to 2.690 us and 7.776 to 2.627 us. Required table H2D, temporary allocation, wrapper preflight, source ownership and model work are excluded. POD-only plan timing is not the cost of runner host validation. No service speedup is inferred.

Artifacts: `/data0/ruixuan.zrx/minimax-m31-dev/1010_decode_next/full_commit_producer_gate/`; frozen runtime `runtime_full_commit` changes only libth_transformer.so and the explicit MiniMax-M3.1 model capability compared with `runtime_seed_table`. A DP8 BS32/80K/static4 CUDA-only E2E campaign is running under `full_commit_candidate`; results and original-PD restoration are still pending.

## 2026-10-10 Full COMMIT E2E outcome

DP8 BS32/80K/static4 formal responses passed 768/768, normal-rejection fake-history functional responses passed 256/256. All eight ranks mapped the frozen candidate library. Producer routing passed 128/128 sampled steps: builders 48 to 32 per rank, fused copies three to two per cycle, legacy table overwrite 16 to zero per rank. Native response timing rounds were 7.697438, 7.246741 and 5.425546 ms/token, mean 6.789908; derived single-card TPS 4712.88 and TPM 282772.59. This is full decode-phase response accounting excluding first token, not pure model_forward time or a real-history quality result. Sequential earlier control 6.127288 ms/token is not a paired causal A/B. No E2E speedup is accepted.

Matched CUDA-only trace definition retains 120 complete cycles and all outliers. GPU all-stream idle median increased 1587.121 to 1814.003 us; mean 1966.406 to 2276.498 us. Rank0 cycle8 contains a 7502.340 us H2D-to-arange gap. A cudaEventQuery on the graph-launch-owner thread overlaps 7472.902 us, 99.61%; no target GraphLaunch overlaps that gap. This establishes the API interval covering the gap, not its allocator/driver/OS cause. Main Query counts decrease 714 to 531 across eight ranks; existing recordStream protection cannot be removed based on this observation. Query caller stacks and thread CPU time are missing. Full report: `1010_decode_next/full_commit_cycle_audit/report.md` and `runtime_api_report.md`. Next causal gate is matched OFF instrumentation and producer on/off control, retaining source lifetimes.

Original frozen PD4+4 was restored in `m31_decode_next_restored_pd4p4_full_commit_1010`: five HTTP endpoints healthy and idle, four real native PD routes passed, eight mapped frozen rank identities verified. Restoration evidence is `1010_decode_next/restored_pd4p4_full_commit/restore_health_and_smoke.json` and `live_runtime_identity.json`. Candidate optimization is not installed into this resident original service.

## 2026-10-10 Query caller diagnosis and same-native control

Artifact-only LD_PRELOAD diagnosis now records the wall/threadCPU/callsite of each delegated actual cudaEventQuery, with durable mmap prefixes and independent >=100us long-call records. CPU mocks cover exact-one-call semantics, errno, concurrency, fork, capacity, _exit/SIGKILL and late RTLD_LOCAL resolution. Real Torch CUDA Event pending/ready gate passed: two actual Query records, pending600 and ready0, with no extra Query from diagnosis. Estimated mock overhead is0.369us per Query; prefaulted host memory is about40.5MiB/process. This is diagnosis, not a production optimization. Long-ring/caller/thread saturation must be inspected.

Same native library and original safeguards are retained in four frozen model controls: producer OFF/ON/ON/OFF, five formal rounds per arm followed by normal-rejection functional gate and CUDA-only route verification. OFF differs by only one model-owned Python capability assignment. Parent and workers use identical diagnosis in all arms. Native-clock request windows and worker executable mappings support stage/caller attribution. Fresh campaign is `1010_decode_next/full_commit_abba_v2`; the first campaign exited before model startup because the parent had not imported Torch before resolving cudart, and its finally block restored and verified original PD4+4. V2 adds four actual-runner import/resolve preflights before stopping services. Its results remain pending.

A separate raw GraphLaunch observer test with1908 repeated-add nodes has twelve fresh-process controls. Median of three round medians is OFF5.181us, CUDA profiling3085.366us, CUDA with external correlation disabled1955.036us, FULL1963.750us; corresponding threadCPU indicates substantial observer CPU work. Isolated graph results do not quantify production idle or prove model speedup. Mixed-process exploration shows residual profiling overhead and is retained but rejected as clean OFF evidence. See `1010_decode_next/graph_profile_control_gate/README.md`.

## 2026-10-10 OFF1 Query caller and trace coverage correction

Same-native producer OFF1 passed1280/1280 formal plus256/256 normal rejection requests and8rank CUDA-only traces. Native response decode mean6.227295594ms/token is diagnostic-instrumented, and does not establish producer benefit. Pending ON1/ON2/OFF2 are in `full_commit_abba_v3`; original PD4+4 was restored between OFF1 and ON1, so ordering is explicitly interrupted.

Mapped Torch ELF caller verification distinguishes device allocator, pinned-host allocator, NCCL WorkNCCL and FlightRecorder. Main launch-owner long records have zero drops: formal7calls; max146.588521ms wall,0.331398ms threadCPU, caller DeviceCachingAllocator::process_events. Formal global509.406574ms query belongs to background pinned-host allocator. NCCL-only long ring overflow prevents complete global per-phase statistics; it does not invalidate main long coverage.

Both main long Queries in outer response09 occur AFTER the recorded16step CUPTI trace (rank0 about1.188653s,rank1 about0.582504s after trace end). They cannot explain gaps inside this timeline. Client profile request window and active CUPTI trace window must not be conflated. Formal main stalls lack simultaneous GPU coverage. Low threadCPU does not distinguish driver blocking from OS descheduling, and caller identity is not proof of a particular allocation or recordStream source.

OFF1 recorded all-stream activity-union audit:120complete cycles; cycle median29.6214ms; idle median1.699712ms, mean2.079637ms; largest individual gap2.010386ms from DtoD to target embedding. These are process-local, profiler-observed kernel/memcpy/memset gaps; wait kernels count active and other-process GPU activity is unobserved. Main long Query causality is not inferred. Sources: `full_commit_abba_v2/audit/off_1_report.md`, `off_1_cycles/summary.json`; precise Query-to-trace coverage report is in the same audit directory.

### ON1: first-profile start skew and observer evidence

ON1 completed1280formal/256normal requests and8traces, five round means5.429082169,5.462289461,6.288642233,5.569785309,5.961885983ms/token; mean5.742337031. Comparison remains provisional pending ON2/OFF2. Complete main long coverage shows2formaldevice-allocatorQueries(max36.145362ms) versus OFF1 seven(max146.588521ms). ON1 response09 long63.471467ms is579.130490ms AFTER active trace; no GPU idle inference.

All120ON1 cycles retained, including1.416928s cycle with1.4132s active union. Median cycle29.509ms and idle1.7895ms versusOFF29.621ms/1.6997ms; largest singlegap2.091ms. The large cycle is not1.4s idle. Common native target indices615..629 alignGPUtargettails2–5us, whereas index615 CPUsubmit startspread1.386940s. Initial trace ordinal differsbyone onrank1/2, so ordinal-only pairing is wrong. Local runnerindex is a correlation counter rather than globally certified logicalroundID.

Engine logs show8rank synchronous TorchProfiler::start entryspread8.139ms; prepareProfiler+enableProfiler durations4.461–6.272s and returns spread1.811983s. First native launch startspread1.811971s closely matches. Thus profiler initialization blocks each mainengine thread and its variable return precedes initial graphsubmit skew; firstMoEabsorbs earlier arrival. Source/log audit provides direct observer evidence, but does not identify the internal CUPTI function or quantify formal profiler-OFF submission gaps. TP-only enable_all_rank withTP1 does not broadcast DP; prearm configuration still completed11s beforestart and is not the immediate1.8s delay.

ON/OFF target native nonprofileGraphLaunch medians18.282/18.273us; CUDA-onlyprofile2062/2105us, predominantly threadCPU. Top20gap botharms sits inside these APIcalls. Artifactreports in `full_commit_abba_v3/audit/`. No unprofiled whole-device idle claim or accepted producer speedup.

### Four-arm completion and restore

All5120formal+1024normalrequests passed,32timelines archived at `/data0/ruixuan.zrx/profiling_results/m31_decode_dp8_bs32_80000_dspark_static4_commit_producer_controls_20261010`. OFF group6.046268568ms/token versusON6.157925579ms/token; noacceptedE2Ebenefit. Mainformal longQuerycounts7/2/6/2 withcompletecoverage. ON2alone34–36ms EventSync-covered idle tailsrequire exactcaller/phase and profiler-OFF GPUevidence; neitherproducer causality norremovingcorrectnesswaits accepted.

OriginalfrozenPD4+4restored,actualrequest+8workermappedlibrarySHAverified;5idleendpoints recheckedafterstandaloneGPUringgate. Native ring9×400pairedgatepassedincludingordinal/overflow/lane/key andunits;2markerGPUbatchdelta2.072640us, modelintegrationpending. Safeexportplan follows NormalEngine.stop loopjoin+existingdeviceSync, notperstepwaits.
