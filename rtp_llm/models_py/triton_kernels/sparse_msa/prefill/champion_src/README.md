# champion_src — frozen R8-PKG-E prefill attention champion

Source and binaries for the prefill sparse-attention champion integrated by
`../champion_r8pkge.py` (env gate `MSA_INTEGRATION_CHAMPION=1`).

## Contents

| File | Role |
|------|------|
| `kernel.cu` | Frozen composite champion source (MSAINTEGRATE108 campaign, source sha256 `44e87f0a...`) |
| `resident12.cuh` | resident12 grouped-attention include (mark → working-cache → grouped resident12 chain, PDL) |
| `build.sh` | Reproduction build (Backend(decoded=True, resident12=True)) |
| `lib/prefill-r8pkge-c58d6a73.so` | The SHA-pinned campaign binary (sha256 `654282ca...f23ca` — verified on load) |
| `lib/prefill-r8pkge-c58d6a73-glibc232.so` | Same GPU code, ELF version table remapped for glibc-2.32 hosts (dev container) |

## Provenance

- Campaign: MSAINTEGRATE108 (2026-10-08), 275/275 prefill points fp64-equivalent,
  2.03x geo at model-forward level vs the production sparse chain.
- Frozen source: `RTP-LLM/build_logs/msa-integration-final-20261008/frozen/glm/`
  (kernel.cu + resident12.cuh, verbatim copies).
- Build identity: `Backend(decoded=True, resident12=True)` from the campaign's
  `cpp_backend.py` — nvcc CUDA 13.0, sm_103a, `--fmad=false`, CUTLASS 4.x headers.

## Rebuild

```bash
./build.sh                # -> lib/prefill-r8pkge-rebuilt.so
./build.sh my_output.so   # custom output name
```

The committed binaries are the tested artifacts; a rebuild is functionally
identical but not guaranteed bit-identical.

## Correctness on this branch

`test_champion_prefill_gate` (same package, `test/`): champion vs independent
fp32 reference rel_l2 = 2.6% (E4M3 operand noise level, matching the campaign's
fp64 oracle to 2.58%), champion vs production route rel_l2 = 2.6e-05.
