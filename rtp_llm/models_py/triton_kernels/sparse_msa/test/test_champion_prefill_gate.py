# Copyright 2025. All rights reserved.
"""Gate: frozen R8-PKG-E champion prefill route vs production on THIS stack.

MSAINTEGRATE108 Phase-3 integration gate for the m31-e2e branch.  The frozen
champion binary (frozen/glm/prefill-r8pkge-c58d6a73.so) was validated on the
campaign stack; this gate re-proves, on the CURRENT branch dependency stack,
that

  1. ``make_chunk_runner`` accepts the branch's native packed-NVFP4 planes
     (geometry check passes -> the champion route really dispatches, no
     silent production fallback),
  2. the champion produces finite, numerically valid output against an
     independent FP32 reference built from an exact nibble/MMA-swizzle decode
     of the same packed planes, and
  3. the champion output stays in the same numeric class as the branch's
     production fmha ``sparse_atten_nvfp4_kv_func`` route (both are quantized
     Q8KV4 kernels; they are not expected to be bitwise-equal).

Selection semantics: the champion consumes ONE top-k list per 8-row query
tile (the production chunk-plan contract).  The fixture builds TILE-SHARED
selections (every row of a tile carries the tile's last row's list), so the
champion's tile reduction is the identity and both routes compute exactly
the reference's per-row selection.  Prefixes are multiples of 8 so no tile
straddles a 128-token block boundary; consecutive rows in a tile differ by
at most one causal block, so every row keeps >= 15 valid selected blocks.

Geometry matches the MiniMax-M3.1 E2E shape class the champion targets:
page 128, 4 KV heads, 64 Q heads, dim 128, topk 16, two ragged segments
(one odd-KV tail page).
"""

import json
import os
import unittest
from unittest.mock import patch

import torch


def _ensure_cutlass_dsl_on_path() -> None:
    """Make the cutlass DSL importable (same injection as
    models_py/modules/factory/linear/factory.py). Under bazel runfiles the
    ``nvidia_cutlass_dsl.pth`` site hook does not run, so ``cutlass`` (and
    hence ``fmha_sm100``) is unimportable."""
    import sys

    try:
        import nvidia_cutlass_dsl
    except Exception:
        return
    pkg_dir = nvidia_cutlass_dsl.__path__[0]
    python_packages_dir = os.path.join(pkg_dir, "python_packages")
    if os.path.isdir(python_packages_dir) and python_packages_dir not in sys.path:
        sys.path.insert(0, python_packages_dir)


_ensure_cutlass_dsl_on_path()

from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (  # noqa: E402
    quantize_main_index_rows_to_planes,
)
from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import (  # noqa: E402
    topk_bt_fused as op,
)

SM_SCALE = 128**-0.5
TOPK = 16
PAGE = 128
HEADS = 4
Q_HEADS = 64
DIM = 128
GROUP = 16  # Q heads per KV head

# (q_len, kv_len): prefix = kv - q is a multiple of 8 for both segments.
SEGMENTS = ((1536, 8192), (517, 8197))
# Selection-independent bisect fixture: <=16 blocks per segment so every
# causally-valid block is always inside the top-16 (the selection plays no
# role; any residual mismatch is decode/masking/GQA, not topk plumbing).
SEGMENTS_FULL = ((256, 2048), (136, 2048))


def _reference_planes(dev, segments=SEGMENTS):
    """Build packed NVFP4 K/V planes with the production quantizer."""
    torch.manual_seed(20261011)
    kv_lens = [kv for (_, kv) in segments]
    tokens = sum(kv_lens)
    pages = sum((kv + PAGE - 1) // PAGE for kv in kv_lens)
    permutation = torch.randperm(pages, device=dev).to(torch.int32)
    k = torch.randn(tokens, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 0.5
    v = torch.randn(tokens, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 0.5
    index_k = torch.randn(tokens, 1, DIM, device=dev, dtype=torch.bfloat16) * 0.5
    # token -> physical slot through the per-segment logical page map
    slots = []
    page_cursor = 0
    for (_, kv_len) in segments:
        n_blocks = (kv_len + PAGE - 1) // PAGE
        pos = torch.arange(kv_len, device=dev)
        blocks = permutation[page_cursor : page_cursor + n_blocks].long()
        slots.append(blocks[pos // PAGE] * PAGE + pos % PAGE)
        page_cursor += n_blocks
    slots = torch.cat(slots).long()
    kp = torch.zeros(pages, HEADS, PAGE, DIM // 2, device=dev, dtype=torch.uint8)
    vp = torch.zeros_like(kp)
    ks = torch.zeros(pages, HEADS, PAGE, 8, device=dev, dtype=torch.float8_e4m3fn)
    vs = torch.zeros_like(ks)
    ip = torch.zeros(pages, 1, PAGE, DIM // 2, device=dev, dtype=torch.uint8)
    isc = torch.zeros(pages, 1, PAGE, 8, device=dev, dtype=torch.float8_e4m3fn)
    quantize_main_index_rows_to_planes(k, v, index_k, slots, kp, ks, vp, vs, ip, isc)
    return kp, vp, ks, vs, permutation


def _nibbles(plane):
    """Packed-NVFP4 nibble decode -> fp32 values [pages, heads, 128, 128]."""
    dev = plane.device
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)
    codes = torch.empty(
        plane.shape[0], HEADS, PAGE, DIM, device=dev, dtype=torch.uint8)
    codes[..., ::2], codes[..., 1::2] = plane & 15, plane >> 4
    value = lut[(codes & 7).long()]
    return torch.where(codes & 8 != 0, -value, value)


def _decode_physical(kp, vp, ks, vs, v_scale_off=None):
    """Independent nibble + MMA-swizzle decode -> PHYSICAL fp32 K/V
    [pages, 4, 128, 128] (post-E4M3 recast, kernel operand semantics).

    ``v_scale_off`` optionally overrides the V-scale byte-offset matrix
    ([128 tokens, 8 groups] -> offset in one head's 1024-byte scale tile);
    the default is the 128x4 MMA swizzle both planes are written with.
    """
    pages = int(kp.shape[0])
    dev = kp.device
    token = torch.arange(PAGE, device=dev)[:, None]
    group = torch.arange(8, device=dev)[None, :]
    mma = group // 4 * 512 + token % 32 * 16 + token // 32 * 4 + group % 4

    def scale_of(plane, off):
        flat = plane.float().reshape(pages, HEADS, PAGE * 8)
        return flat[..., off]  # [pages, heads, 128 tokens, 8 groups]

    def recast(x):
        # The kernels consume the decoded NVFP4 values as E4M3 MMA operands:
        # saturate the dequantized product to the E4M3 range and round through
        # the format (campaign reference_fp64 parity).
        return x.clamp(-448, 448).to(torch.float8_e4m3fn).float()

    off_v = mma if v_scale_off is None else v_scale_off
    k_dec = recast(_nibbles(kp) * scale_of(ks, mma).repeat_interleave(16, -1))
    v_dec = recast(_nibbles(vp) * scale_of(vs, off_v).repeat_interleave(16, -1))
    return k_dec, v_dec


def _decode_logical_kv(kp, vp, ks, vs, permutation):
    """Independent nibble + MMA-swizzle decode -> logical fp32 K/V [tok,4,128].

    The physical decode is [page, head, token, dim]; the logical view must
    move token ahead of head BEFORE flattening (page, token) into rows -- a
    bare reshape scrambles the head/token axes (root cause of the original
    gate failure: kernels and campaign fp64 reference agreed to 2.6e-2 while
    this oracle diverged at 1.2 rel_l2).
    """
    k_dec, v_dec = _decode_physical(kp, vp, ks, vs)
    # permutation[b] is the physical page holding logical block b.
    phys = permutation.long()
    k_log = k_dec[phys].permute(0, 2, 1, 3).reshape(-1, HEADS, DIM)
    v_log = v_dec[phys].permute(0, 2, 1, 3).reshape(-1, HEADS, DIM)
    return k_log, v_log


def _tile_shared_topk(q8, k_log, segments=SEGMENTS):
    """Block scores from the decoded K; per-row top-16, then tile-share."""
    dev = q8.device
    lists = []
    row_base, kv_base = 0, 0
    for (q_len, kv_len) in segments:
        n_blocks = (kv_len + PAGE - 1) // PAGE
        prefix = kv_len - q_len
        pos = torch.arange(q_len, device=dev) + prefix
        valid_blocks = (pos[:, None] + 1 + PAGE - 1) // PAGE  # [rows, 1]
        # block score = max causally-valid q.k over the group's Q heads and
        # the block's tokens -> [rows, kv heads, blocks]
        score = torch.empty(q_len, HEADS, n_blocks, device=dev, dtype=torch.float32)
        for h in range(HEADS):
            qk = torch.einsum(
                "rhd,kd->rhk",
                q8[row_base : row_base + q_len, h * GROUP : (h + 1) * GROUP].float(),
                k_log[kv_base : kv_base + kv_len, h, :],
            )
            for b in range(n_blocks):
                lo, hi = b * PAGE, min((b + 1) * PAGE, kv_len)
                tok = torch.arange(lo, hi, device=dev)
                causal = (tok[None, None, :] <= pos[:, None, None]).float()
                score[:, h, b] = (qk[:, :, lo:hi] * causal).amax(dim=(1, 2))
        blocks = torch.arange(n_blocks, device=dev)
        score = torch.where(
            blocks[None, None, :] < valid_blocks[:, :, None], score, float("-inf")
        )
        # per-row top-16 (largest first), then tile-share to the tile's LAST
        # row's list (the champion chunk-plan contract)
        top = score.topk(TOPK, dim=-1).indices  # [rows, heads, 16]
        row_last = torch.tensor(
            [min((r // 8) * 8 + 7, q_len - 1) for r in range(q_len)], device=dev
        )
        shared = top[row_last]  # [rows, heads, 16]
        n_valid = (blocks[None, None, :] < valid_blocks[row_last][:, :, None]).sum(-1)  # [rows, 1]
        slots = torch.arange(TOPK, device=dev)[None, None, :]
        shared = torch.where(
            slots < n_valid[..., None], shared, torch.full_like(shared, -1)
        )
        lists.append(shared.permute(1, 0, 2).contiguous().int())  # [heads, rows, 16]
        row_base += q_len
        kv_base += kv_len
    return torch.cat(lists, dim=1)


def _fp32_reference(q8, k_log, v_log, topk_idx, segments=SEGMENTS):
    """Independent per-row selected-block causal attention (fp32)."""
    dev = q8.device
    out = torch.empty(q8.shape[0], Q_HEADS, DIM, device=dev, dtype=torch.float32)
    row_base, kv_base = 0, 0
    for (q_len, kv_len) in segments:
        prefix = kv_len - q_len
        pos = torch.arange(q_len, device=dev) + prefix
        n_blocks = (kv_len + PAGE - 1) // PAGE
        sel = topk_idx[:, row_base : row_base + q_len, :].permute(1, 0, 2)  # [r,h,16]
        acc = torch.zeros(q_len, HEADS, n_blocks, device=dev, dtype=torch.int32)
        acc.scatter_add_(2, sel.long().clamp(min=0), (sel >= 0).to(torch.int32))
        block_ok = acc > 0
        tok = torch.arange(kv_len, device=dev)
        tok_ok = block_ok[:, :, tok // PAGE] & (tok[None, None, :] <= pos[:, None, None])
        for h in range(HEADS):
            q_group = q8[row_base : row_base + q_len, h * GROUP : (h + 1) * GROUP].float()
            scores = torch.einsum(
                "rhd,kd->rhk", q_group, k_log[kv_base : kv_base + kv_len, h, :]
            ) * SM_SCALE
            scores = scores.masked_fill(~tok_ok[:, h, None, :], float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[row_base : row_base + q_len, h * GROUP : (h + 1) * GROUP] = (
                torch.einsum(
                    "rhk,kd->rhd", probs, v_log[kv_base : kv_base + kv_len, h, :]
                )
            )
        row_base += q_len
        kv_base += kv_len
    return out


class ChampionPrefillGateTest(unittest.TestCase):
    report = {}

    def _run_case(self, segments, label, champion_lib_required=True):
        dev = torch.device("cuda")
        kp, vp, ks, vs, permutation = _reference_planes(dev, segments)
        kv_lens = [kv for (_, kv) in segments]
        rows = sum(q for (q, _) in segments)
        q = torch.randn(rows, Q_HEADS, DIM, device=dev, dtype=torch.bfloat16) * 0.5
        q8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        op.scale1_query_fp8(q, q8)
        k_log, v_log = _decode_logical_kv(kp, vp, ks, vs, permutation)
        topk_idx = _tile_shared_topk(q8, k_log, segments)
        reference = _fp32_reference(q8, k_log, v_log, topk_idx, segments)
        self.assertTrue(torch.isfinite(reference).all().item(), "reference has NaN")

        cu_q = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor([q for (q, _) in segments]), 0)),
            device=dev,
            dtype=torch.int32,
        )
        seq_lens = torch.tensor(kv_lens, device=dev, dtype=torch.int32)
        prefix_lens = torch.tensor(
            [kv - q for (q, kv) in segments], device=dev, dtype=torch.int32
        )
        k_scale = ks.view(torch.uint8).view(-1, 8)
        v_scale = vs.view(torch.uint8).view(-1, 8)

        def run(champion):
            plan = op.build_sparse_attn_plan(
                cu_q, seq_lens, prefix_lens, Q_HEADS, HEADS, PAGE, TOPK
            )
            env = {"MSA_INTEGRATION_CHAMPION": "1"} if champion else {}
            # The champion bridge resolves the repo-committed binaries by
            # default (original, then the glibc-2.32 remapped copy); the
            # legacy build_logs remap is only used when it still exists.
            if "MSA_CHAMPION_LIB" not in os.environ:
                legacy = (
                    "/data0/liukan.lk/m31-e2e/build_logs/champion_prefill_run/"
                    "prefill-r8pkge-c58d6a73-glibc232.so"
                )
                if os.path.exists(legacy):
                    env["MSA_CHAMPION_LIB"] = legacy
            with patch.dict(os.environ, env):
                return op.sparse_prefill_from_topk_fp4(
                    q, kp, vp, k_scale, v_scale, topk_idx, permutation, plan,
                    TOPK, PAGE, SM_SCALE,
                )

        import rtp_llm.models_py.triton_kernels.sparse_msa.prefill.champion_r8pkge as champ

        made = []
        original = champ.make_chunk_runner

        def spy(*args, **kwargs):
            runner = original(*args, **kwargs)
            made.append(runner)
            return runner

        champ.make_chunk_runner = spy
        try:
            out_prod = run(champion=False)
            out_champ = run(champion=True)
        finally:
            champ.make_chunk_runner = original

        def delta(a, b):
            a, b = a.float(), b.float()
            return {
                "max_abs": (a - b).abs().max().item(),
                "relative_l2": ((a - b).norm() / b.norm()).item(),
            }

        if champion_lib_required:
            self.assertEqual(len(made), 1, "champion runner must be built once")
            self.assertIsNotNone(made[0], "champion geometry rejected branch planes")
        self.assertTrue(torch.isfinite(out_champ).all().item(), "champion output NaN/Inf")
        self.assertTrue(torch.isfinite(out_prod).all().item(), "production output NaN/Inf")
        # per-segment diagnostics localize any oracle divergence
        seg_rels = []
        lo = 0
        for (q_len, _) in segments:
            seg_rels.append(delta(out_prod[lo : lo + q_len], reference[lo : lo + q_len])["relative_l2"])
            lo += q_len
        head_rels = [
            delta(out_prod[:, h * GROUP : (h + 1) * GROUP],
                  reference[:, h * GROUP : (h + 1) * GROUP])["relative_l2"]
            for h in range(HEADS)
        ]
        result = dict(
            champion_vs_reference=delta(out_champ, reference),
            production_vs_reference=delta(out_prod, reference),
            champion_vs_production=delta(out_champ, out_prod),
            per_segment_rel_l2=seg_rels,
            per_head_rel_l2=head_rels,
            rows=rows,
            pages=int(kp.shape[0]),
        )
        print(f"[gate:{label}] {result}")
        return result

    def test_champion_route_matches_reference_and_production(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        # Selection-independent bisect fixture first: every causally-valid
        # block is inside the top-16, so a mismatch here isolates
        # decode/masking/GQA errors from topk plumbing.
        full = self._run_case(SEGMENTS_FULL, "full-selection")
        self.assertLess(full["champion_vs_reference"]["relative_l2"], 0.05, full)
        self.assertLess(full["champion_vs_production"]["relative_l2"], 0.05, full)
        # Main E2E-shaped fixture (topk actually selects 16 of 53-65 blocks).
        main = self._run_case(SEGMENTS, "e2e-shape")
        self.report.update(
            full_selection=full,
            e2e_shape=main,
            topk=TOPK,
        )
        self.assertLess(main["champion_vs_reference"]["relative_l2"], 0.05, main)
        self.assertLess(main["production_vs_reference"]["relative_l2"], 0.05, main)
        self.assertLess(main["champion_vs_production"]["relative_l2"], 0.05, main)


if __name__ == "__main__":
    result = unittest.main(exit=False).result
    if torch.cuda.is_available():
        print(json.dumps(ChampionPrefillGateTest.report, indent=2))
    raise SystemExit(not result.wasSuccessful())
