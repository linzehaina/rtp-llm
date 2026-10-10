# M3.1 prefill TopK 最终兼容版复现

包内含生产 CUDA 源、thin pybind、原 bitonic body、GPU 条件回退、1670 个正向交叉用例、107 点 benchmark 和 NCU 入口。benchmark 的 warp 直接运行主 kernel + 兼容回退，不受生产 shape guard 限制；生产默认仅选择 H4/K16/N512..1024/Q≥1024/indices-only，其余保留原实现。

本次 azk114，CUDA Runtime CC10.3、148 SM，CUDA13.2/GCC12/PyTorch2.11.0+cu130，使用 sm_103。先核验 CUDA Runtime 架构；每次 GPU 工作或变体启动前，对所需物理 GPU 每秒采样，覆盖完整 5 秒且全部低于 1024 MiB，保存记录。A/B 同卡串行。在对应 GPU 容器中进入此目录，按实际设备设置 CUDA_VISIBLE_DEVICES。

```bash
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export TORCH_CUDA_ARCH_LIST=10.3
export MAX_JOBS=4
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python setup_kernel.py build_ext --inplace

export M31_TOPK_TEST_EXTENSION=m31_prefill_topk_test
python source/rtp_llm/models_py/triton_kernels/sparse_msa/test/test_minimax_prefill_topk.py -v
CUDA_VISIBLE_DEVICES= python source/rtp_llm/models_py/triton_kernels/sparse_msa/test/test_prefill_warp_dispatch.py -v
# 每条 GPU 命令启动前分别重新完成 GPU 空闲检查。
python production_benchmark.py --variant baseline --suite compat_final --iters 30
python production_benchmark.py --variant warp --suite compat_final --iters 30
```

results/compat_final_{baseline,warp}/results.json 包含 107 点；每点 .json.gz 包含 GPU trace。固定随机种子、5 次预热、30 次计时，每次计时前的 256 MiB 清缓存位于计时区间外。kernel_sum_us 为算子各 kernel 耗时中位数之和；event_us_median 也包含 GPU 执行间隔。先校验分数、范围、唯一性和 padding；精确历史 ID 另由交叉测试和真实模型 audit 验证。

```bash
ncu --profile-from-start off --set full --replay-mode kernel --launch-count 1 --force-overwrite \
  --kernel-name 'regex:.*topk_to_block_table_multirow_kernel.*' \
  --export ncu_baseline python ncu_case.py --variant baseline
# 下一条命令前重新完成 GPU 空闲检查；新版采集两个 kernel。
ncu --profile-from-start off --set full --replay-mode kernel --launch-count 2 --force-overwrite \
  --kernel-name 'regex:.*(minimax_prefill_warp_topk_kernel|topk_to_block_table_tie_fallback_kernel).*' \
  --export ncu_warp python ncu_case.py --variant warp

compute-sanitizer --tool memcheck --error-exitcode 86 python \
  source/rtp_llm/models_py/triton_kernels/sparse_msa/test/test_minimax_prefill_topk.py \
  MinimaxPrefillTopkTest.test_empty_and_current_stream \
  MinimaxPrefillTopkTest.test_graph_replay_reads_fresh_scores_prefix_and_segment_boundaries \
  MinimaxPrefillTopkTest.test_subnormal_keys_with_cpu_ieee_oracle \
  MinimaxPrefillTopkTest.test_legacy_tie_fallback_graph_replay_changes_tie_policy_on_gpu \
  MinimaxPrefillTopkTest.test_legacy_tie_fallback_matches_original_ids_and_block_tables \
  MinimaxPrefillTopkTest.test_tie_flags_cover_selected_cutoff_and_below_cutoff_duplicates \
  MinimaxPrefillTopkTest.test_legacy_fallback_marks_nonfinite_and_legacy_padding_score_hazards -v
```

synccheck 使用相同方法，去掉 subnormal 方法；日志中 memcheck/synccheck 分别覆盖 160/152 个正向子用例，0 errors。NCU 需 profiling 权限，本次 NCU2026.1.1。代码采用 -ftz=true 匹配 Bazel，subnormal 零判断使用位模式。

model_reproduction/ 保存启动、客户端、original-body patch、audit、清理脚本和白名单模型配置。路径保留实际 /data4/aozhengkai.azk/m31_warp_20261010、端口 28300/28301；跨目录复现须统一调整路径、容器名及配置。包不含权重、原冻结 native library 或模型依赖，实际目录保留了这些资产。搬运原运行包须 rsync -aH 保留 native library 硬链接。analysis/compat_model_sequence.py 保存真实串行 A/B 的完整命令和 gate 顺序，可作为重跑入口参考。

冻结包仅增加 native export（每个 spawn worker 由 sitecustomize 注入）和 TopK import/分派/条件回退。原 bitonic body AST 未改；baseline 设置 M3_PREFILL_TOPK_IMPL=bitonic，warp 设置 warp。主统计为 baseline_compat/formal_steady 与 warp_compat/formal_steady，phase_base800；冷一组、稳定六组，每组实际一次 BS20/74011输入/66560复用/1输出。每实现四 rank timeline、两个 forward；独立 audit 为 warp_compat_audit，不计时。140/140 首 token 一致，35,788,800 行 ordered IDs 一致。

真实服务和测试容器已关闭，资产/缓存/日志保留。完整 wheel/全仓 CI 未验证：Bazel rules_python fetch 在容器 SSH host-key 校验处阻塞；focused build 和实际模型替换验证不替代完整构建。
