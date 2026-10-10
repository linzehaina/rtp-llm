# Native hybrid decode attention (P6-2-residual champion + dispatch threshold fix)

`libdecode_hybrid_p6res.so` is the decode champion of the 13-round
Q8KV4 decode optimization campaign (2026-10-09/10) plus the P8-SMALLB
dispatch threshold fix (2026-10-11). It replaces the Triton
`_q8kv4_selected|four_page_partial` + `_q8kv4_combine` attention stage of
`q8kv4_paged_sparse_decode` with one ctypes launch
(`launch_native_hybrid_attention`, swizzled=1, mode=0: K1 selected-page
partial + K2 combine in a single call).

## Provenance

- Binary sha256:
  `c1506bb46a56f4eb343aa6aead09f4dc72d79674ea74193558de6e14004b8954`
  (previous champion build `e1d0b7ef...`, frozen
  `codex/decode-hybrid-p6res-e1d0b7ef.so` = `P6-2-residual/it1.so`)
- Campaign source tree: `experiments/decode_opt/P6-2-residual/src/`
  (P6-2 residual-decode cut on the P9-A QMUL4 line; full identity in the
  campaign's `frozen/IDENTITY.json`, variant `codex_decode_hybrid_p6res`)
  with the one-line dispatch change of
  `experiments/decode_opt/P9-SHIP16/src/stage3/attention.cu`:
  `launch_native_hybrid_attention` routes `args->rows >= 16` to the native
  tcgen05 two-page K1 (was `>= 32`). Device kernels are byte-identical to
  the champion (all 19 fatbin functions have identical SASS opcode
  sequences; the change is host-side dispatch only — the two paths were
  both already bitwise-gated, rows 16-31 just take the faster one).
- Built with the frozen champion recipe (nvcc 13.4.92, sm_103a, ring
  geometry flags; toolchain control rebuild of the unpatched tree
  reproduced the champion's SASS exactly; byte-identity of the control is
  impossible because nvcc bakes a PID-derived temp name into the fatbin),
  links only `libcudart.so.13` + standard C/C++ runtime.
- Launcher: `native_hybrid_decode.py` (same directory). The bundled binary
  is hash-pinned at load; a mismatch disables the route with a warning.

## Dispatch threshold change (P8-SMALLB evidence, 2026-10-10)

The champion routed rows 16-31 to the warp-register K1; the P8-SMALLB
probe (9 cells x 7 launch-path variants, cold-L2 CUDA-graph p50, same
process) showed the native tcgen05 K1 is faster across that whole window
while remaining O-bitwise vs production on every cell:

| cell (rows) | champion dispatch | native K1 forced | delta |
|---|---|---|---|
| dec-w6-bsz4-kv128000 (24) | 32.10 us | 22.03 us | **-31.4%** |
| dec-w1-bsz16-kv81920 (16) | 24.96 us | 20.75 us | **-16.9%** |
| dec-w4-bsz4-kv65536 (16) | 24.54 us | 20.46 us | **-16.6%** |
| dec-w6-bsz30-kv128000 (180, anchor) | 80.72 us | 80.70 us | parity |

Same-session paired spot-check of the shipped binary vs the frozen
champion twin (10 cold AB/BA CUDA-graph pairs, seed 20261007, distinct
idle GPUs): dec-w6-bsz4 22.42 vs 28.53 us (**-21.4%**), dec-w1-bsz16
19.87 vs 24.35 us (**-18.4%**), anchor dec-w6-bsz30 79.65 vs 79.87 us
(**-0.3%**, parity).

## Re-gate (9 P8 cells x 2 seeds, fresh process, 2026-10-11)

Full campaign gate on the shipped binary (bitwise O + LSE vs the
production chain, NaN-aware stats, rep-to-rep determinism x2, CUDA-graph
capture + 10 replays):

| cell | rows | O bitwise vs prod | O max_abs | mismatch rows | NaN rows | det x2 | graph |
|---|---|---|---|---|---|---|---|
| dec-w6-bsz30-kv128000 (anchor) | 180 | yes | 0.0 | 0 | 0 | yes | 10/10 |
| dec-w6-bsz4-kv128000 | 24 | yes | 0.0 | 0 | 0 | yes | 10/10 |
| dec-w1-bsz16-kv81920 | 16 | yes | 0.0 | 0 | 0 | yes | 10/10 |
| dec-w4-bsz4-kv65536 | 16 | yes | 0.0 | 0 | 0 | yes | 10/10 |
| dec-w1-bsz4-kv65536 | 4 | yes | 0.0 | 0 | 0 | yes | 10/10 |
| dec-w4-bsz1-kv81920 | 4 | yes | 0.0 | 0 | 0 | yes | 10/10 |
| dec-w1-bsz1-kv128000 | 1 | yes | 0.0 | 0 | 0 | yes | 10/10 |
| dec-w6-bsz1-kv128000 | 6 | yes | 0.0 | 0 | 0 | yes | 10/10 |
| dec-w1-bsz8-kv81920 | 8 | yes | 0.0 | 0 | 0 | yes | 10/10 |

18/18 runs PASS (both seeds per cell). LSE finite_max_abs 9.5e-7 <= 5e-2
(the pre-existing champion fused-LSE delta, identical to every prior
champion gate run); O is bitwise-identical to production on all cells —
expected, since the threshold only changes WHICH existing path runs.

## Validation standing (19-cell decode matrix, L20D, 2026-10-10)

- Output `O` bitwise-identical to the production Triton attention stage on
  every cell (0 mismatch rows, 0 one-sided NaN); fused natural-log global
  LSE within 1 float ulp of the harness reduce.
- Attention-stage kernel time 1.27x-3.39x faster than the Triton reader
  (anchor dec-w6-bsz30-kv128000: 80.5us vs 262.8us); full decode chain
  (cast+index+topk shared) 1.11x-2.07x, geomean 1.545x. With the rows 16-31
  re-route, the small-batch cells improve a further 17-21%.
- Rep-to-rep bitwise deterministic; CUDA-graph capture/replay bitwise-equal
  to eager; no process-static device state.

See the integration report under
`build_logs/msa-integration-final-20261008/integration_m31/decode_bench/`
(campaign workspace) for the full protocol and verdict tables.

## Dispatch contract

Selected only when: SM 10.3 device, `mma_scale_layout=True`, BF16 output
format, uniform `query_width` 1..8 (no ragged `query_cu_seqlens`), 64 Q
heads / 4 KV heads / head_dim 128 / page 128 / topk 16. Consumes the entry's
own shape-cached workspace carriers (`q8`, `topk_i32`, `output`,
`partial_output`, `partial_lse`) and the cache planes zero-copy; per-request
page table via a strided view of the per-row table (rows of one request
share the table by the entry contract); `cu`/`cu_k` are shape-bucket-cached
constant device tensors, `used` is the live per-request used-KV derived from
`seq_lens` (zero-copy at width 1). `lens` stays NULL: per-row visibility is
derived on device as `used - (w - 1 - row)`.

## Toggles

- `MSA_USE_NATIVE_HYBRID_DECODE=0` — disable; restores the original
  Triton partial+combine / Cake dispatch (read per call, no re-import).
- `MSA_NATIVE_HYBRID_LIB=/path/lib.so` — load an operator-supplied binary
  (sha logged, not enforced) instead of the bundled artifact.

Ragged target verification, MXFP8 output, non-BF16-ready geometry and other
devices keep the original attention stage.
