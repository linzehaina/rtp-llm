# MiniMax-M3.1 环境开关审计

2026-10-04 在既有 DSpark 工作树中，基于 `69be6060368b246939b67396872d36bfb6ff18bb` 审计。未提交的性能候选另行标明。开关是否有效取决于该次源码，不能套用其他工作树的历史配置。

## 范围与计数

检查 M3.1 提交 `5cff7481dc`、`f1701e2688`、`e2bc3169c7`、`eff9080c40`、`36c5883242`、`fbc5ca1c33`、`69be606036`，以及共享内存缓存提交 `f901e0ea71` 和 GCC13 提交 `af4403eb17`。对照各引入提交的父版本，统计生产源码新增/删除的名称。测试、kernel constexpr 名称及既有共享启动配置不计为新功能开关。

M3.1/DSpark/MegaMoE 中仍有效的新开关共七个，加上三个共享内存缓存线程控制共十个；这不是部署 PD 所需环境变量的总数。早期临时数学路径和 MoE mock 选择器已删除。

## 仍有效的新功能开关

| 开关 | 引入提交 | 含义/默认值 | 部署建议 |
| --- | --- | --- | --- |
| `NVFP4_KV_CACHE` | e2bc3169c7 | 默认 0。启用主 KV 和 idxK 打包存储，E2M1 数值、每 16 元素一个 E4M3 scale，以及原生 Q8K4 读写 | M3.1 两个 PD 角色均设 1；与目标 FP8/INT8 KV 选择器互斥。保留为实际缓存格式选择 |
| `RTP_LLM_DSPARK_CUDA_GRAPH` | 5cff7481dc | 默认关闭，仅字面值 1 启用独立精确形状 DSpark propose/commit 图 | Decode 设 1，同时配置目标图启用/捕获桶；Prefill 保持 eager，可设 0。此开关本身不启用目标图 |
| `GLM5_MEGA_MOE_NVFP4_INPUT_PACKER` | f1701e2688 | 默认 `fused`；`auto` 当前同样选择融合实现；`torch` 选择参考打包 | 生产可省略；显式 torch 用于正确性/回退。后续可弃用冗余 auto 别名。选择的是激活/路由打包，而非 MegaMoE GEMM 或权重格式 |
| `GLM5_MEGA_MOE_NVFP4_PACK_BLOCK_M` | f1701e2688 | 可覆盖为 1/2/4/8/16。自动策略在 tokens≥1024，或 H6144/topk4 且行数 80/96/112/128 时选 16，否则选 4 | 省略以保留按形状调优；勿对所有小 Decode 形状强制 16。覆盖用于已测调优/诊断 |
| `SP_DSPARK_VERIFY_TOKENS` | fbc5ca1c33 | 默认 0 表示 gamma。含义随模式变化：完整静态验证；自适应初始平均额外行预算；无模式且旧 bool 为 false 时的固定前缀预算 | 保留；它不是草稿块大小，也不是接受率保证 |
| `SP_DSPARK_VERIFY_MODE` | 69be606036 | `static`、`adaptive`，或默认空值以沿用旧 bool/预算语义 | 新部署显式设置模式；替代旧 bool 的模式选择，不能替代 token 预算 |
| `SP_DSPARK_ADAPTIVE_VERIFY` | 69be606036 | 默认 false；兼容旧配置，用置信度排序压缩验证 | 新部署省略并使用 VERIFY_MODE；暂保留旧配置/pickle 兼容。static 与 true 同设会拒绝 |

### 验证模式的准确语义

发布的 gamma7 权重保持 `GEN_NUM_PER_CIRCLE=7`；草稿骨干始终生成七行，与目标验证预算无关。

- `SP_DSPARK_VERIFY_MODE=static`：关闭置信度规划，验证全部 gamma 候选及 anchor；VERIFY_TOKENS 只能为 0 或 7。0 表示使用 gamma。
- `SP_DSPARK_VERIFY_MODE=adaptive`：置信度规划器压缩真实目标行。VERIFY_TOKENS 控制初始总额外行预算 `min(B * budget, B * gamma)`，并非各请求固定长度或上限。每请求仍有 anchor，候选数可不同。
- 模式为空且旧 bool 为 false：沿用固定前缀行为，如 VERIFY_TOKENS=4 表示各请求验证前四个候选及 anchor；不能与显式 static 全量验证混淆。

因此 adaptive、budget=4、B=16 提供 64 个额外行预算及 16 个 anchor，各请求长度由置信度决定。adaptive 预算 0 使用完整 gamma 预算，不能用于减少目标计算。

新自适应部署片段须一致应用于两个 PD 角色：

```text
GEN_NUM_PER_CIRCLE=7
SP_DSPARK_VERIFY_MODE=adaptive
SP_DSPARK_VERIFY_TOKENS=4
```

预算 4 是当时已测配置，不是通用生产最优值；须结合普通拒绝采样延迟、MAL 和 batch 大小调优。强制接受是独立诊断。

## 新增共享内存缓存控制

| 开关 | 默认值/优先级 | 建议 |
| --- | --- | --- |
| `RTP_LLM_MEMORY_CACHE_H2D_WAIT_DONE_THREADS` | 16；覆盖公共控制 | 控制等待 H2D 完成的 host 线程，不控制 GPU copy stream 或 PD RDMA 带宽 |
| `RTP_LLM_MEMORY_CACHE_D2H_WAIT_DONE_THREADS` | 24；覆盖公共控制 | 独立 D2H 完成等待池；增加线程可能仅增加 CPU 消耗 |
| `RTP_LLM_MEMORY_CACHE_WAIT_DONE_THREADS` | 方向值缺失/为空时的公共回退 | 新配置避免同时设置公共和方向开关；保留公共值用于兼容 |

三者均由 f901e0ea71 引入；接受 1..64 线程，非法值回退到各方向默认值。仅启用 Memory Cache 时相关，不影响纯 device-cache-hit GPU 计算。既有共享开关 `RTP_LLM_MEMORY_CACHE_WAIT_DONE_QUEUE_SIZE` 默认 1000，不是新 M3.1 功能开关。

## 已删除或无效配置

以下名称在审计工作树中没有生产读取方，应从部署环境列表移除：

- `M31_DSPARK_CANDIDATE_MATH`：69be606036 删除。发布 DSpark 使用检查点验证的 Gemma norm、非因果 query block 和检查点滑动窗口。
- `M3_M31_MOCK_NVFP4_MOE`：f1701e2688 删除，真实 NVFP4 MegaMoE 替代 mock。
- `M3_IDX_PAGED`：e2bc3169c7 删除；分页 indexer cache 不再是可选平面缓存模式。
- `M3_LOAD_MSA_INDEX`、`M3_MSA_RAW_IDX_MXFP8`：36c5883242 删除，稀疏层/权重处理不再需要人工开关。
- `M3_MSA_FUSED_KV_GATHER`、`M3_MSA_KV_GATHER_BLOCK_T`、`M3_MSA_IDX_GATHER_BLOCK_T`、`M3_DISABLE_TRTLLM_GEN`：36c5883242 随旧平面 scratch 路由删除。
- 69be606036 删除的详细诊断选择器：`RTP_LLM_ASYNC_DEBUG`、`RTP_LLM_COMPARE_SP_PREFILL`、`RTP_LLM_DEBUG_MTP_ACCEPT`、`RTP_LLM_DEBUG_MTP_DECODE_DATA`、`RTP_LLM_DEBUG_MTP_PREFILL_DATA`、`RTP_LLM_DEBUG_TARGET_VERIFY_INPUT`、`RTP_LLM_PD_DEBUG`。

`M3_MSA_CP_COMPACT_PREFILL` 仍用于旧 M3，但启用 NVFP4 时明确排除。M3.1 配置应省略；不能据此删除旧 M3 代码。

## 仍相关的既有配置

- `SP_TYPE=dspark`、`SP_MODEL_TYPE=minimax_m31_dspark`、`SP_CHECKPOINT_PATH`、`SP_ACT_TYPE=BF16`、`SP_FP8_KV_CACHE=0` 属于既有推测解码配置。当前 ModelFactory 仍要求 SP_CHECKPOINT_PATH；从目标根目录/嵌套草稿检查点加载不等于可省略该字段。目标 embedding/LM head 共享，无需为草稿独立计算/复制权重。
- `MOE_STRATEGY=mega_moe_nvfp4`、VL 请求的模型类型 `minimax_m31_vl`、fastsafetensors 和 `FORCE_CPU_LOAD_WEIGHTS=0` 仍是本部署要求。
- `PREFILL_CP_SIZE=4`、`PREFILL_CP_KV_CACHE_SHARDED=1`、各角色 CP rotate 模式、Decode DP 拓扑和 Graph 捕获桶仍重要，属于共享配置。
- `M3_MSA_INDEX_SCORE_CHUNK_ROWS`、`M3_SPARSE_ATTN_CHUNK_ENABLE/SIZE`、`DSV4_MOE_CHUNK_PREFILL/TOKENS` 分别控制工作区/分块。attention partial 工作区、index scores 和 EP MoE collective 调度几何不同，不能合成一个数值。
- `M3_ROUTER_BF16`：0 选择 FP32 router，1 选择 BF16 router。2026-10-06 已测候选按已批准 BF16 策略在两个 PD 角色均设 1。FP32 Prefill 持有展开 buffer，BF16 保留普通 linear 路径。对照不得暗改精度或旧 M3 语义。
- `M3_SPARSE_ATTN_FP8_PARTIAL` 仍是旧/全局方案选择器。已测 BF16-partial 契约保持未设/0。BF16 partial O 不等于旧 BF16 KV 工作页，也不代表 Q8K4 attention 回退。
- `SP_DETERMINISTIC_DRAFT_EXACT_MATCH=0` 属于普通拒绝采样配置；不能开启 exact-match/force-accept 诊断后报告生产随机精度或 MAL。

## 建议清理顺序

1. 删除无效环境名并修正文档，无需改变运行时语义。
2. 新配置仅用 VERIFY_MODE 选择模式，VERIFY_TOKENS 保留为独立预算。后续弃用兼容 bool 时须配套迁移和序列化配置测试。
3. 无 profiling 证据时省略 packer 模式/tile 覆盖及公共内存线程覆盖；有 H2D/D2H 吞吐需求时保留已测方向值。
4. 保留 router 精度选择器并明确记录实测值。当时候选使用 BF16；删除选择器或强制 FP32 不在该次发布范围。
5. 当时未提交的 norm/RoPE 分组未新增环境变量。TopK32 平面 combine 能力从依赖 callable 协商；旧依赖沿用原生未展平路径。交付该优化须同时更新已验证依赖和 RTP 源码。

本文描述审计时源码语义，不验证其他已安装 wheel、线上环境、新验证预算或新增依赖包。
