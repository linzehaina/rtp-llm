# MiniMax M3 / M3.1 文档

文档统一使用中文，按主题合并。过时方案仅保留方法、结论和证据路径，不维护历史实施流水账。源码标识、配置、命令及原始数据保留原样。

| 入口 | 内容 |
| --- | --- |
| [模型、DSpark与NVFP4缓存设计](minimax_m31_nvfp4_native_kernel_design.md) | 存储/工作表示/计算、数值布局、PD与Graph生命周期、DSpark配置；旧BF16工作页方案只留历史结论 |
| [环境开关审计](minimax_m31_env_switch_audit.md) | 验证模式和预算、有效/已删除开关、部署建议；按审计日期理解 |
| [Prefill优化与验证](minimax_m31_prefill_attention_optimization.md) | CP4/FI/R16、Warp TopK、性能和数值审查、生产边界 |
| [Decode优化、精度与性能](minimax_m31_decode_optimization.md) | 首批及后续方案取舍、观察器/空泡、最终性能、全质量与Radix修复续跑范围 |
| [TopK简短复现入口](minimax_m31_prefill_topk/reproduction.md) | 原始附件、测试入口和环境限制 |

本目录共六篇Markdown（含此索引），不再保留重复专题入口。TopK附件已从 `docs/minimax_m31_prefill_topk` 归入本目录；原JSON/CSV/图片及复现代码保留，历史审查说明并入Prefill文档，旧检查日志已清理。

结论来自截至2026-10-12的既有报告，整理未重跑模型或核验当前服务。重点边界：R16实验默认关闭且未证明新通信重叠；BF16历史工作页已退役；fakeKV/强制接受吞吐不能当生产普通拒绝吞吐；Radix后最终候选混合5409旧响应及272修复响应，不能写为修复二进制全新5681条。未完成项统一列在Prefill/Decode文末。
