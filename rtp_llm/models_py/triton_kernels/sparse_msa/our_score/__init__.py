"""our_score: env-gated tcgen05 score kernels ported from the M3.0 b128/b128d
campaign to M3.1 geometry (H=4 index heads, 128-token contiguous pages).

Off by default; the production dispatch is untouched unless
``M3_OUR_SCORE=1`` is set (lazy per-call read, like
``MSA_INTEGRATION_CHAMPION``).  Contents:

* ``our_score_abi.h``        -- frozen C ABI contract (shared with the .so).
* ``our_score_bridge.py``    -- ctypes bridge: prefill launch over the staged
  E4M3 scratch + decode launch with bridge-side NVFP4->E4M3 staging.
* ``lib/libour_score.so``    -- built by ``build.sh`` (Track A) from
  ``our_score_host.cu`` + ``prefill/`` + ``decode/`` kernel sources.
"""
