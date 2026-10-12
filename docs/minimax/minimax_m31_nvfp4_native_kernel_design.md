# MiniMax-M3.1 模型、DSpark 与 NVFP4 缓存设计

合并原 DSpark、Q8KV4 一期和原生 NVFP4 设计。本文保留有效契约，过时实现仅记录方案与结论。配置开关统一见[环境审计](minimax_m31_env_switch_audit.md)，模型验证见[Prefill](minimax_m31_prefill_attention_optimization.md)与[Decode](minimax_m31_decode_optimization.md)。以下为截至2026-10-12的文档证据，不代表本次重新验证当前运行时。

## 存储、工作表示与计算

| 对象 | 契约 |
| --- | --- |
| 目标主 K/V、indexer K 持久缓存 | E2M1 打包值，每16个元素一个E4M3 scale；不得扩张为全历史BF16工作页 |
| 主 attention Q | 当前FI/Cake路径使用BF16；不能从Q8KV4名称推断所有主Q/MMA均为FP8 |
| 选中 KV tile | 按实际后端在片上解包/反量化，FI路径以BF16参与MMA/softmax |
| indexer | Q转换、K读取和score后端须按实际版本确认；不能把历史单算子实现当当前加载后端 |
| DSpark 草稿 | 真实preview2权重，独立BF16 KV，共享目标embedding/LM head，五层滑动窗口稠密FFN Transformer |
| PD | 只传持久字节，value/scale/index附属区完整到达后才发布；工作scratch不作为前缀缓存传输 |

`NVFP4_KV_CACHE` 与目标FP8/INT8 KV互斥，关闭时旧M3路径保持兼容。原生路由不满足时应明确失败，不能静默恢复全历史BF16物化。历史文档的固定backend、旧MTP草稿配置已被后续实际FI/DSpark运行记录覆盖；实际后端须查启动父进程、源码和加载包身份。

## 数值与布局

每组16个BF16值，FP32计算 `amax`，`scale_fp32=clamp(amax/6,1/512,448)`，存储转换后的E4M3 scale。编码使用实际存储scale：在原数值域比较 `abs(x)` 与 `threshold*stored_scale`，避免先除法把精确中点推过阈值。反量化为 `decode_e2m1(code)*scale_e4m3`，无tensor级外部scale。

| 从上到下判定的归一化幅值 | E2M1幅值编码 | 解码值 |
| --- | ---: | ---: |
| >5.0 | 7 | 6.0 |
| ≥3.5 | 6 | 4.0 |
| >2.5 | 5 | 3.0 |
| ≥1.75 | 4 | 2.0 |
| >1.25 | 3 | 1.5 |
| ≥0.75 | 2 | 1.0 |
| >0.25 | 1 | 0.5 |
| 其他 | 0 | 0.0 |

幅值0时符号位归零；偶数维存低半字节、随后奇数维存高半字节。scale转换前钳位，量化使用舍入后scale，CPU/Triton/CUDA/QAT须一致。

H为本地KV head数，P为页大小，D为主head维度，I为indexer维度：

| 区域 | 每block字节数 |
| --- | --- |
| 主打包K/V | `2*H*P*D/2` |
| 主K/V scale | `2*H*P*D/16` |
| 打包indexer-K | `P*I/2` |
| indexer scale | `P*I/16` |

`kv_cache_base` 保存主打包K/V，`kv_scale_base` 保存主scale及indexer数值/scale。合计每标量9/16字节，为BF16的28.125%。维度须为16倍数，C++缓存布局是尺寸/偏移唯一依据，Python仅校验并建view，不能从dtype推附属偏移或伪装成FP8缓存。

## 写入、所有权与生命周期

普通目标attention及DSpark commit/query写入共用codec。无效slot `-1` 不改缓存；拒绝后复用slot须同时覆盖数值和scale。CP在writer前完成本地物理slot映射；前缀恢复对原始value/side使用相同收集计划及逻辑恢复索引，不能对打包字节套token-major BF16 gather。

workspace按实际布局、桶、forward/layer epoch管理。CUDA Graph地址稳定，实时metadata原地刷新，补齐尾部在消费者可见前清理。跨stream保留producer-ready、consumer-complete及owner/lease，桶扩容不得破坏已捕获指针。短fake stream不足TopK16时输出有效ID及尾部−1，不扩大fake cache或新增BF16分支。

PD须传逻辑 `prefill_cp_size`，不能从attention TP推导；CP2/TP1时推错会发0、破坏page-RR归属。不透明整block传输须保留完整 `kv`/`kv_scale` 字节数，缺scale硬失败。`worker_status=finished` 仅说明RPC阶段结束，HTTP/native输出须独立核验。任意attention-TP重分片不能靠混合附属区的一个bool表达，须显式支持与门禁。

## DSpark 模型与验证模式

发布检查点使用weight-plus-one Gemma RMS norm、投影后hidden norm、非因果query block及检查点滑动窗口；目标特征层为3/17/31/45/59。旧 `M31_DSPARK_CANDIDATE_MATH` 已删除。集成要求CUDA，不能宣称HIP严格随机拒绝采样已验证。

```text
MODEL_TYPE=minimax_m31
LOAD_METHOD=fastsafetensors
FORCE_CPU_LOAD_WEIGHTS=0
MOE_STRATEGY=mega_moe_nvfp4
NVFP4_KV_CACHE=1
FP8_KV_CACHE=0
SP_TYPE=dspark
SP_MODEL_TYPE=minimax_m31_dspark
SP_CHECKPOINT_PATH=<preview2-checkpoint>/dspark
SP_ACT_TYPE=BF16
SP_FP8_KV_CACHE=0
GEN_NUM_PER_CIRCLE=7
SP_DSPARK_VERIFY_MODE=adaptive
SP_DSPARK_VERIFY_TOKENS=4
SEQ_SIZE_PER_BLOCK=128
KERNEL_SEQ_SIZE_PER_BLOCK=128
```

两个PD角色模式/预算一致；完整语义由环境审计维护。草稿始终生成七行，adaptive预算不是每请求固定长度。P使用eager CP4与分片KV；目标Graph和草稿 `RTP_LLM_DSPARK_CUDA_GRAPH` 分别启用，prompt初始化保持eager。

DSpark对采样前驱施加Markov bias，保留真实proposal分布供拒绝采样。历史验证只用已提交历史及当前位置之前proposal，拒绝KV复用时覆盖。终止发布防止异步继续追加。强制接受仅诊断，精度/MAL须普通拒绝采样。

## 历史方案与结论

| 阶段 | 方案 | 结论与证据边界 |
| --- | --- | --- |
| Q8KV4一期，2026-09-22 | FP4持久缓存，经有界请求表转BF16工作页后复用旧attention | 已退役，不作为服务回退。PD2+2十贪心请求成功，热中位242.7ms；小BF16参考185–191ms，转换成本明显，不能推断当前原生FP4性能 |
| 原生Decode早期 | 融合writer、原生index score/TopK/metadata、CuTe读打包KV | 主/indexer不物化历史BF16。Q4/K4因TopK精度下降拒绝；按实际后端分别验证。覆盖长度1至1M、变长/非法ID/Graph及>2^31字节偏移 |
| 2026-09-24 PD4+4 | 原生KV4，Graph关闭 | 连续5/5算术请求输出84；该日Graph开启在初始化SIGABRT，不能以关闭结果替代。独立target-only DP8 Graph128/128不证明同日PD/DSpark Graph |
| 2026-09-28真实DSpark PD4+4 | P CP4/EP4、D DP4/EP4、目标/草稿Graph、预算5、普通拒绝 | 冒烟8/8、历史处理12/12；GSM原始432/500、独立数值审计465/500；LongBench68.19/160，保留三截断；MAL3.794/2.997。为旧快照，不能验证不同版本/模式 |

发布须分别证明源码/安装Python/native/checkpoint身份、实际每DP流量、缓存字节/scale、Graph/生命周期、完整质量和性能。TCP证据不能替代RDMA，单算子精度或历史冒烟不能替代新版本生产签收。
