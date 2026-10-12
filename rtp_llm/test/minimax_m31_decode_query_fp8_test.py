"""Decode index-Q producer preserves the existing BF16-to-E4M3 boundary."""
import unittest

import torch

from rtp_llm.models_py.triton_kernels.minimax_m31_gemma_rope import (
    minimax_m31_gemma_norm_rope_,
)
from rtp_llm.models_py.triton_kernels.sparse_msa.decode.nvfp4_q8_query_cast import (
    fused_query_cast,
)
from rtp_llm.models_py.triton_kernels.sparse_msa.decode.q8kv4_decode import (
    decode_index_query_fp8_buffer,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DecodeIndexQueryFp8Test(unittest.TestCase):
    def fixture(self, rows):
        torch.manual_seed(20261012)
        source = torch.randn(rows, 9856, device="cuda", dtype=torch.bfloat16)
        weights = tuple(torch.randn(128, device="cuda", dtype=torch.bfloat16) for _ in range(4))
        positions = torch.arange(rows, device="cuda", dtype=torch.int32) % 1024
        cache = torch.randn(1024, 64, device="cuda", dtype=torch.float32)
        return source, weights, positions, cache

    def call(self, storage, weights, positions, cache, outputs=None, index_fp8=None):
        minimax_m31_gemma_norm_rope_(
            storage[:, :9216], storage[:, 9216:9728], storage[:, 9728:],
            weights, positions, cache,
            num_q_heads=64, num_kv_heads=4, num_index_heads=4,
            contiguous_outputs=outputs, index_query_fp8_output=index_fp8,
        )

    def check(self, rows, contiguous, graph=False):
        source, weights, positions, cache = self.fixture(rows)
        old, new = source.clone(), source.clone()
        if contiguous:
            outputs = [tuple(torch.empty((rows, heads, 128), dtype=torch.bfloat16, device="cuda")
                             for heads in (64, 4, 4, 1)) for _ in range(2)]
        else:
            outputs = [None, None]
        expected = torch.empty((rows, 4, 128), dtype=torch.float8_e4m3fn, device="cuda")
        actual = torch.empty_like(expected)
        unused_main = torch.empty((rows, 64, 128), dtype=torch.float8_e4m3fn, device="cuda")

        def reference():
            self.call(old, weights, positions, cache, outputs[0])
            if contiguous:
                q, iq = outputs[0][0], outputs[0][2]
            else:
                q = old[:, :8192].reshape(rows, 64, 128).contiguous()
                iq = old[:, 9216:9728].reshape(rows, 4, 128).contiguous()
            fused_query_cast(q, iq, unused_main, expected, cast_main_query=False)

        def fused():
            self.call(new, weights, positions, cache, outputs[1], actual)

        reference()
        fused()
        self.assertTrue(torch.equal(old.view(torch.int16), new.view(torch.int16)))
        self.assertTrue(torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)))
        if contiguous:
            for a, b in zip(*outputs):
                self.assertTrue(torch.equal(a.view(torch.int16), b.view(torch.int16)))
        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graphs = []
            for function in (reference, fused):
                capture = torch.cuda.CUDAGraph()
                with torch.cuda.graph(capture, stream=stream):
                    function()
                graphs.append(capture)
            torch.cuda.current_stream().wait_stream(stream)
            for shift in (1, 17, 0):
                positions.add_(shift).remainder_(1024)
                old.copy_(source)
                new.copy_(source)
                for capture in graphs:
                    capture.replay()
                self.assertTrue(torch.equal(old.view(torch.int16), new.view(torch.int16)))
                self.assertTrue(torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)))

    def test_index_only_and_contiguous_outputs_match_legacy_rounding(self):
        for rows in (1, 5, 20, 25, 40, 100, 160, 256):
            for contiguous in (False, True):
                with self.subTest(rows=rows, contiguous=contiguous):
                    self.check(rows, contiguous)

    def test_graph_replay_preserves_changed_positions(self):
        self.check(160, True, graph=True)
        self.check(256, True, graph=True)

    def test_index_output_cannot_alias_contiguous_output(self):
        source, weights, positions, cache = self.fixture(20)
        outputs = tuple(torch.empty((20, heads, 128), dtype=torch.bfloat16, device="cuda")
                        for heads in (64, 4, 4, 1))
        alias = outputs[2].view(torch.float8_e4m3fn).reshape(-1)[:20 * 4 * 128].reshape(20, 4, 128)
        with self.assertRaisesRegex(ValueError, "independent storage"):
            self.call(source, weights, positions, cache, outputs, alias)

    def test_empty_rows_and_invalid_output_dtype(self):
        source, weights, positions, cache = self.fixture(0)
        output = torch.empty((0, 4, 128), device="cuda", dtype=torch.float8_e4m3fn)
        self.call(source, weights, positions, cache, index_fp8=output)
        source, weights, positions, cache = self.fixture(20)
        output = torch.empty((20, 4, 128), device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "contiguous E4M3"):
            self.call(source, weights, positions, cache, index_fp8=output)

    def test_cached_outputs_are_owned_by_the_current_stream(self):
        query = torch.empty((40, 512), dtype=torch.bfloat16, device="cuda")
        first = decode_index_query_fp8_buffer(query)
        self.assertIs(first, decode_index_query_fp8_buffer(query))
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            second = decode_index_query_fp8_buffer(query)
            self.assertIs(second, decode_index_query_fp8_buffer(query))
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        for invalid in (
            torch.empty((5, 512), dtype=torch.bfloat16, device="cuda"),
            torch.empty((257, 512), dtype=torch.bfloat16, device="cuda"),
            torch.empty((40, 256), dtype=torch.bfloat16, device="cuda"),
            torch.empty((40, 512), dtype=torch.float32, device="cuda"),
            torch.empty((), dtype=torch.bfloat16, device="cuda"),
            torch.empty((40, 512), dtype=torch.bfloat16),
        ):
            self.assertIsNone(decode_index_query_fp8_buffer(invalid))


if __name__ == "__main__":
    unittest.main()
