"""Single-launch Triton top-16 for the MiniMax-M3.1 decode chain (reg16w1).

Decode-campaign adopted pair component (r1 H5, revived by the r2 H5-stack
policy rule): replaces the production native ``rtp_llm_ops.minimax_decode_topk``
(4-round smem radix select, 9.02us mean on the 15-condition matrix) with one
Triton kernel over a (row, head) grid, ``num_warps=1`` (the gated ``reg16w1``
config):

  * REGISTER-RESIDENT selection: the whole score row (max_blocks <= 1024 ->
    one BLOCK=1024 tile) is loaded once, then 16 rounds of a fused
    max+argmax reduction (``tl.max(..., return_indices=True,
    return_indices_tie_break_left=True)``) pick the winners.  The fused
    reduction's tie rule (value1 == value2 and index1 < index2) is EXACTLY
    the value-desc / id-asc tie contract, verified in the container's triton
    3.6.0 ``standard.py`` (``_argmax_combine``).
  * masked tail: positions >= ceil(seq_len/block_size) load as -inf (the
    production workspace tail is stale garbage; the native op masks by
    seq_lens the same way).
  * the 16 winning ids are emitted in canonical ID-ASCENDING slot order
    (``tl.sort`` ascending) -- deterministic slot order, which canonicalizes
    the downstream native-champion attention reduction order (the root-caused
    source of the production 1-bf16-ulp output wobble).

Gates (r1/r2, 15/15 conditions): topk SET-equal vs the native op on the same
score tensor (direct + dispatch paths), static contract (no duplicate ids,
ids in range, id-ascending slots), attention output within 1 bf16 ulp, whole
chain (topk tensor + attention output) bitwise across reruns.

Measured (L20D, kernel-time): 7.93us mean vs 9.02us native = 1.13us saving
(0.88x); stacked with the N1 score kernel the PAIR saves 2.58us mean
(76.00 vs 78.58us pair kernel-time, > the pre-registered 2us bar).

DEPLOYMENT CAVEAT (eager host launch): the Python-launched Triton topk adds
~+23.5us to the eager (non-graph) full chain vs the native op -- host launch
serialization, kernel time unaffected.  Production CUDA-graph deployments do
not pay it; eager-Python-dispatch deployments must capture the topk call in
a graph or port the launcher to C++ (see pair_adoption/README.md).
"""

import torch
import triton
import triton.language as tl

NEG_INF = tl.constexpr(float("-inf"))


@triton.jit
def _msa_topk_reg16_kernel(
    s_ptr,            # score: [heads, rows, max_blocks] fp32
    out_ptr,          # topk idx: [heads, rows, TOPK] int32
    seq_lens_ptr,     # [rows] int32
    block_size,       # 128
    stride_s_h,
    stride_s_b,
    stride_s_k,
    stride_o_h,
    stride_o_b,
    stride_o_t,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)          # row
    pid_h = tl.program_id(1)          # idx head
    seq_len = tl.load(seq_lens_ptr + pid_b)
    nb = (seq_len + block_size - 1) // block_size

    off = tl.arange(0, BLOCK)
    s = tl.load(
        s_ptr + pid_h * stride_s_h + pid_b * stride_s_b + off * stride_s_k,
        mask=off < nb,
        other=NEG_INF,
    )

    # 16 rounds of fused max+argmax.  tie_break_left (value1 == value2 and
    # index1 < index2) implements value-desc / id-asc exactly.
    ids = tl.zeros((TOPK,), dtype=tl.int32)
    for t in tl.static_range(TOPK):
        v, j = tl.max(
            s, axis=0, return_indices=True,
            return_indices_tie_break_left=True,
        )
        j = j.to(tl.int32)
        valid = v != NEG_INF            # nb < TOPK defensive (never here)
        # Sort invalid slots after real IDs; -1 would sort first and cause
        # the valid-prefix mask below to discard blocks on short sequences.
        j = tl.where(valid, j, 2147483647)
        ids = tl.where(tl.arange(0, TOPK) == t, j, ids)
        s = tl.where(off == j, NEG_INF, s)

    # canonical id-ascending slot order (deterministic downstream consume)
    ids = tl.sort(ids, descending=False)
    mslot = tl.arange(0, TOPK) < tl.minimum(TOPK, nb)
    ids = tl.where(mslot & (ids != 2147483647), ids, -1)

    o_ptrs = (out_ptr + pid_h * stride_o_h + pid_b * stride_o_b
              + tl.arange(0, TOPK) * stride_o_t)
    tl.store(o_ptrs, ids)


class TritonReg16Topk:
    """Drop-in ``minimax_decode_topk`` built on _msa_topk_reg16_kernel.

    Mirrors the native op's calling convention exactly:
        fn(score, seq_lens, out, block_size, topk) -> writes out
    score [heads, rows, max_blocks] fp32, seq_lens [rows] int32,
    out [heads, rows, topk] int32 (workspace.topk_i32, graph-stable buffer).
    """

    def __init__(self, num_warps=1, num_stages=1, name=None):
        self.num_warps = int(num_warps)
        self.num_stages = int(num_stages)
        self.name = name or f"reg16w{num_warps}"
        self.kernel_holder = {}

    @torch.no_grad()
    def __call__(self, score, seq_lens, out, block_size, topk):
        heads, rows, max_blocks = (int(x) for x in score.shape)
        topk = int(topk)
        block_size = int(block_size)
        if score.dtype != torch.float32:
            raise ValueError(f"score must be fp32, got {score.dtype}")
        if not seq_lens.is_contiguous():
            seq_lens = seq_lens.contiguous()
        block = max(triton.next_power_of_2(max_blocks), 16)
        if block > 4096:
            raise ValueError(f"max_blocks {max_blocks} too large for reg16")
        grid = (rows, heads)
        self.kernel_holder["kernel"] = _msa_topk_reg16_kernel[grid](
            score,
            out,
            seq_lens,
            block_size,
            score.stride(0),
            score.stride(1),
            score.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            TOPK=topk,
            BLOCK=block,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )
        return out

    def kernel_meta(self):
        try:
            k = self.kernel_holder.get("kernel")
            if k is None:
                return {}
            return {
                "n_regs": int(k.n_regs),
                "n_spills": int(k.n_spills),
                "shared_bytes": int(k.metadata.shared),
            }
        except Exception:  # noqa: BLE001
            return {}


# the gated reg16w1 config (num_warps=1): 79 regs, 0 spills, 0 smem
triton_reg16_topk = TritonReg16Topk(num_warps=1)
