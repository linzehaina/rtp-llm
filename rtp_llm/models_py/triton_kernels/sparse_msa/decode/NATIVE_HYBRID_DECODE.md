# Native hybrid decode attention (frozen champion P6-2-residual)

`libdecode_hybrid_p6res.so` is the frozen decode champion of the 13-round
Q8KV4 decode optimization campaign (2026-10-09/10). It replaces the Triton
`_q8kv4_selected|four_page_partial` + `_q8kv4_combine` attention stage of
`q8kv4_paged_sparse_decode` with one ctypes launch
(`launch_native_hybrid_attention`, swizzled=1, mode=0: K1 selected-page
partial + K2 combine in a single call).

## Provenance

- Binary: `decode-hybrid-p6res-e1d0b7ef.so`
- SHA256: `e1d0b7efa3acea405cfd88764126bdfc652a25cd992e0fe22a4e3c634ea4a7b5`
- Campaign source tree: `experiments/decode_opt/P6-2-residual/src/`
  (P6-2 residual-decode cut on the P9-A QMUL4 line; full identity in the
  campaign's `frozen/IDENTITY.json`, variant `codex_decode_hybrid_p6res`)
- Built with CUDA 13.0.48, `-gencode=arch=compute_103a,code=sm_103a`;
  links only `libcudart.so.13` + standard C/C++ runtime
- Launcher: `native_hybrid_decode.py` (same directory). The bundled binary
  is hash-pinned at load; a mismatch disables the route with a warning.

## Validation standing (19-cell decode matrix, L20D, 2026-10-10)

- Output `O` bitwise-identical to the production Triton attention stage on
  every cell (0 mismatch rows, 0 one-sided NaN); fused natural-log global
  LSE within 1 float ulp of the harness reduce.
- Attention-stage kernel time 1.27x-3.39x faster than the Triton reader
  (anchor dec-w6-bsz30-kv128000: 80.5us vs 262.8us); full decode chain
  (cast+index+topk shared) 1.11x-2.07x, geomean 1.545x.
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
