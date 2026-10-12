# Prefill TopK 复现与证据入口

方案、形状范围、性能口径和数值/同步限制统一见[Prefill优化](../minimax_m31_prefill_attention_optimization.md)。不再重复历史执行步骤。

本目录保留JSON/CSV/图表和 `review_20261010` 验证附件。focused源码/测试入口为 `setup.py`、`check_keys.py`、`run_review.py`、`run_original.py`、`negative_metadata.py`、`run_sanitizers.sh`。需要仓库对应生产源码置于其要求的 `source/rtp_llm/...` 结构；运行前按实际环境调整 `review_env.sh` 的host/container/data路径。缺native export、跳过或mock不能当GPU验证。

常规生产测试为 `test_minimax_prefill_topk.py`、`test_prefill_topk_adversarial.py`；`M31_TOPK_LARGE_STRIDE=1` 启用真实大stride用例。新环境须核验CUDA Runtime架构、构建产物和GPU可用性，再按已授权验证范围运行。

历史算子环境为azk114、Runtime CC10.3/148SM、CUDA13.2/GCC12/PyTorch2.11.0+cu130、sm103；审查在azk112物理GPU1。benchmark固定随机种子、5次预热、30次计时，缓存清理在计时外，性能包含native主kernel及兼容回退。

完整运行包为交付 `outputs/kernel_review_20261010`，模型/分析材料在原交付包的 `model_reproduction/`、`analysis/`。历史路径 `/data4/aozhengkai.azk/m31_warp_20261010`、端口28300/28301只描述原环境。仓库附件不包含权重、冻结native library和全部模型依赖；focused测试不替代完整Bazel/wheel/CI。此次整理未运行GPU或服务。
