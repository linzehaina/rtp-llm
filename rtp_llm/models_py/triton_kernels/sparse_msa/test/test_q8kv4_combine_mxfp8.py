"""Match fused output bytes/scales to BF16 combine + production MXFP8 quant."""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.sparse_msa.decode.nvfp4_q8_attention import (
    LOG2E_F32,
    _q8kv4_combine,
)
from rtp_llm.models_py.triton_kernels.sparse_msa.decode.nvfp4_q8_combine_mxfp8 import (
    _combine_mxfp8,
    _quantize_bf16_attention_mxfp8,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CombineMxfp8Test(unittest.TestCase):
    def test_quantizer_masks_raw_invalid_rows_and_graph_replay(self):
        for rows in (1, 5, 100, 160):
            for dtype in (torch.int32, torch.int64):
                for mask_kind in ("none", "strided", "broadcast"):
                    x = torch.randn(rows, 64, 128, device="cuda", dtype=torch.bfloat16)
                    lens = torch.full((rows * 2,), 100, device="cuda", dtype=dtype)[::2]
                    if mask_kind == "none":
                        mask = None
                    elif mask_kind == "strided":
                        mask = torch.ones(rows * 2, device="cuda", dtype=torch.bool)[::2]
                    else:
                        mask = torch.ones(1, device="cuda", dtype=torch.bool).expand(rows)
                    aligned = (rows + 3) // 4 * 4
                    actual = torch.empty_like(x, dtype=torch.float8_e4m3fn)
                    expected = torch.empty_like(actual)
                    scales = torch.empty_strided((rows, 64), (1, aligned), device="cuda", dtype=torch.int32)
                    expected_scales = torch.empty_strided((rows, 64), (1, aligned), device="cuda", dtype=torch.int32)

                    def launch():
                        _quantize_bf16_attention_mxfp8[(rows, 4)](
                            x, actual.view(torch.uint8), scales, ALIGNED_M=aligned,
                            LENS=lens, MASK=mask, LS=lens.stride(0),
                            MS=mask.stride(0) if mask is not None else 0,
                            HAS_LENS=True, HAS_MASK=mask is not None, num_warps=4)

                    launch()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        launch()
                    for live in (rows, 0, rows // 2, rows):
                        x.normal_()
                        lens.copy_(torch.where(torch.arange(rows, device="cuda") < live, 100, -1))
                        if mask is not None:
                            mask.fill_(live > 0)
                        active = lens > 0
                        if mask is not None:
                            active = active & mask
                        x[~active] = float("nan")
                        clean = torch.where(active[:, None, None], x, 0)
                        _quantize_bf16_attention_mxfp8[(rows, 4)](
                            clean, expected.view(torch.uint8), expected_scales,
                            ALIGNED_M=expected_scales.stride(1), num_warps=4)
                        graph.replay()
                        self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))
                        self.assertTrue(torch.equal(scales, expected_scales))

    def test_quantization_and_graph_replay(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("MXFP8 fused consumer is gated to SM103")
        for rows in (1, 3, 5, 20, 40, 100, 101, 129, 160):
            with self.subTest(rows=rows):
                self.check_rows(rows)

    def test_bf16_attention_quant_bytes_and_graph_t160(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("MXFP8 consumer is gated to SM103")
        import flashinfer

        for rows in (1, 3, 100, 101, 127, 128, 129, 143, 159, 160):
            with self.subTest(rows=rows):
                x = torch.randn(rows, 64, 128, device="cuda", dtype=torch.bfloat16)
                fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
                aligned = (rows + 3) // 4 * 4
                scales = torch.empty_strided((rows, 64), (1, aligned), device="cuda", dtype=torch.int32)

                def launch():
                    _quantize_bf16_attention_mxfp8[(rows, 4)](
                        x, fp8.view(torch.uint8), scales, ALIGNED_M=aligned, num_warps=4
                    )

                def check():
                    expected, sf = flashinfer.mxfp8_quantize(
                        x.reshape(rows, -1), is_sf_swizzled_layout=False,
                        alignment=32, backend="cute-dsl"
                    )
                    shifts = torch.arange(4, device="cuda", dtype=torch.int64) * 8
                    packed = (sf.reshape(rows, 64, 4).to(torch.int64) << shifts).sum(-1).to(torch.int32)
                    self.assertTrue(torch.equal(fp8.view(torch.uint8).reshape(rows, -1), expected.view(torch.uint8)))
                    self.assertTrue(torch.equal(scales, packed))

                launch()
                check()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch()
                pointers = (fp8.data_ptr(), scales.data_ptr())
                for amplitude in (0., 1e-8, 1e-4, 64.):
                    x.normal_().mul_(amplitude)
                    # Inactive bucket rows must remain zero after quantization.
                    x[rows // 2:].zero_()
                    graph.replay()
                    check()
                    self.assertEqual(pointers, (fp8.data_ptr(), scales.data_ptr()))
                del graph

    def check_rows(self, rows):
        torch.manual_seed(1847 + rows)
        partial = torch.randn(rows, 4, 16, 16, 128, device="cuda", dtype=torch.bfloat16)
        lse = torch.randn(rows, 4, 16, 16, device="cuda")
        counts = torch.randint(0, 17, (rows, 4), device="cuda", dtype=torch.int32)
        valid = torch.ones(rows, device="cuda", dtype=torch.bool)
        bf16 = torch.empty(rows, 64, 128, device="cuda", dtype=torch.bfloat16)
        fp8 = torch.empty_like(bf16, dtype=torch.float8_e4m3fn)
        aligned_m = (rows + 3) // 4 * 4
        scales = torch.empty_strided(
            (rows, 64), (1, aligned_m), device="cuda", dtype=torch.int32
        )
        mask_args = dict(
            valid_token_mask=valid,
            HAS_VALID_TOKEN_MASK=True,
            valid_token_mask_stride=valid.stride(0),
        )

        def launch():
            _combine_mxfp8[(rows, 4)](
                partial,
                lse,
                counts,
                fp8.view(torch.uint8),
                scales,
                ALIGNED_M=aligned_m,
                LOG2E=LOG2E_F32,
                num_warps=4,
                **mask_args
            )

        def check():
            _q8kv4_combine[(rows, 4)](
                partial,
                lse,
                counts,
                bf16,
                bf16.stride(0),
                bf16.stride(1),
                LOG2E=LOG2E_F32,
                num_warps=4,
                **mask_args
            )
            import flashinfer

            expected, scale_u8 = flashinfer.mxfp8_quantize(
                bf16.reshape(rows, -1),
                is_sf_swizzled_layout=False,
                alignment=32,
                backend="cute-dsl",
            )
            # Independent byte packing reference for the int32 TMA ABI;
            # avoid coupling the oracle to the production Triton packer.
            scale_groups = scale_u8.reshape(rows, 64, 4).to(torch.int64)
            shifts = torch.arange(4, device="cuda", dtype=torch.int64) * 8
            expected_scales = (scale_groups << shifts).sum(-1).to(torch.int32)
            self.assertTrue(
                torch.equal(
                    fp8.view(torch.uint8).reshape(rows, -1), expected.view(torch.uint8)
                )
            )
            self.assertTrue(torch.equal(scales, expected_scales))

        launch()
        check()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                launch()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        pointers = (fp8.data_ptr(), scales.data_ptr())
        for live, amplitude in (
            (0, 1.0),
            (rows // 2, 1e-4),
            (rows, 64.0),
            (max(0, rows - 1), 1.0),
        ):
            valid.copy_(torch.arange(rows, device="cuda") < live)
            partial.normal_().mul_(amplitude)
            partial[~valid] = float("nan")
            counts.random_(0, 17)
            counts[0, 0] = 0
            lse.normal_().mul_(64)
            graph.replay()
            check()
            self.assertEqual(pointers, (fp8.data_ptr(), scales.data_ptr()))


if __name__ == "__main__":
    unittest.main()
