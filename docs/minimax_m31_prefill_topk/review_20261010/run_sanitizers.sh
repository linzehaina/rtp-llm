#!/bin/bash
set -euo pipefail
review=/data4/aozhengkai.azk/m31_warp_20261010/review_20261010
root=/data4/aozhengkai.azk/m31_warp_20261010
for mode in memcheck synccheck racecheck initcheck; do
 python "$root/gpu_gate.py" "$review/gate_${mode}.json" 1
 docker exec -u "$(id -u):$(id -g)" -e M31_REVIEW_RESULT="${mode}.json" azk_m31_review_20261010 bash review_env.sh compute-sanitizer --tool "$mode" --error-exitcode 99 python run_review.py > "$review/${mode}.log" 2>&1
 tail -n 8 "$review/${mode}.log"
done
for name in negative_prefix max_int_prefix bad_cu_start bad_cu_end descending_cu qmax_too_small; do
 python "$root/gpu_gate.py" "$review/gate_negative_${name}.json" 1
 docker exec -u "$(id -u):$(id -g)" azk_m31_review_20261010 bash review_env.sh python negative_metadata.py "$name" > "$review/negative_${name}.log" 2>&1
 tail -n 1 "$review/negative_${name}.log"
done
