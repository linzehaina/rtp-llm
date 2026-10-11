"""Last-page initialization is bounded and preserves all valid MMA scale bytes."""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
    clear_packed_working_tail_scales,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class WorkingTailScalesTest(unittest.TestCase):
    def test_changing_scratch_stride_and_working_layout(self):
        heads, groups = 4, 8
        for fi_layout in (False, True):
            for stride in (256, 512, 768, 1024):
                with self.subTest(fi_layout=fi_layout, stride=stride):
                    lengths = (137, stride - 1)
                    pages = stride // 64 + 2
                    main = torch.full(
                        (2, pages, heads * 128 * groups), 127,
                        device="cuda", dtype=torch.uint8,
                    )
                    index = torch.full(
                        (pages, 128 * groups), 127,
                        device="cuda", dtype=torch.uint8,
                    )
                    expected_main, expected_index = main.cpu().clone(), index.cpu().clone()
                    for batch, length in enumerate(lengths):
                        page = (batch * stride + length) // 128
                        for row in range(length % 128, 128):
                            for group in range(groups):
                                mma = (
                                    group // 4 * 512 + row % 32 * 16
                                    + row // 32 * 4 + group % 4
                                )
                                k_offset = row * groups + group if fi_layout else mma
                                v_offset = (
                                    ((row // 4) * 4 + group // 2) * 8
                                    + (group % 2) * 4 + row % 4
                                ) if fi_layout else mma
                                for head in range(heads):
                                    expected_main[0, page, head * 1024 + k_offset] = 0
                                    expected_main[1, page, head * 1024 + v_offset] = 0
                                expected_index[page, mma] = 0
                    lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
                    clear_packed_working_tail_scales(
                        main[0].view(torch.float8_e4m3fn),
                        main[1].view(torch.float8_e4m3fn),
                        index.view(torch.float8_e4m3fn),
                        lens, stride, heads, 128, 128,
                        fi_working_layout=fi_layout,
                    )
                    torch.testing.assert_close(main.cpu(), expected_main, rtol=0, atol=0)
                    torch.testing.assert_close(index.cpu(), expected_index, rtol=0, atol=0)

    def test_ragged_tail_only_with_capacity_larger_than_live_pages(self):
        lengths = [0, 128, 137, 255, 256]
        stride, pages, heads, groups = 384, 18, 4, 8
        # Larger-capacity planes sliced like the shared pool; stride(0) is the
        # page stride, not a presumed tightly packed two-plane allocation.
        main = torch.full(
            (2, pages, heads * 128 * groups), 127, device="cuda", dtype=torch.uint8
        )
        index = torch.full((pages, 128 * groups), 127, device="cuda", dtype=torch.uint8)
        expected_main, expected_index = main.cpu().clone(), index.cpu().clone()
        for b, length in enumerate(lengths):
            if length % 128:
                page = (b * stride + length) // 128
                for row in range(length % 128, 128):
                    for group in range(groups):
                        offset = (
                            group // 4 * 512 + row % 32 * 16 + row // 32 * 4 + group % 4
                        )
                        for head in range(heads):
                            expected_main[:, page, head * 128 * groups + offset] = 0
                        expected_index[page, offset] = 0
        lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)

        def run():
            clear_packed_working_tail_scales(
                main[0, :15].view(torch.float8_e4m3fn),
                main[1, :15].view(torch.float8_e4m3fn),
                index[:15].view(torch.float8_e4m3fn),
                lens,
                stride,
                heads,
                128,
                128,
            )

        run()
        torch.testing.assert_close(main.cpu(), expected_main, rtol=0, atol=0)
        torch.testing.assert_close(index.cpu(), expected_index, rtol=0, atol=0)
        # CUDA Graph changes actual tail boundaries in place, including a
        # page-aligned request (no tail) after a previous partial-page replay.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        main.fill_(127)
        index.fill_(127)
        lens.fill_(128)
        graph.replay()
        self.assertTrue((main == 127).all().item())
        self.assertTrue((index == 127).all().item())
        lens.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        graph.replay()
        torch.testing.assert_close(main.cpu(), expected_main, rtol=0, atol=0)
        torch.testing.assert_close(index.cpu(), expected_index, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
