import unittest

import torch

from rtp_llm.models_py.triton_kernels.sparse_msa.decode.minimax_decode_topk_reg16 import (
    triton_reg16_topk,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DecodeTopkReg16Test(unittest.TestCase):
    @staticmethod
    def oracle(scores, lengths):
        result = []
        for head in scores.cpu().tolist():
            rows = []
            for values, length in zip(head, lengths.cpu().tolist()):
                blocks = (length + 127) // 128
                live = [i for i in range(blocks) if values[i] != float("-inf")]
                winners = sorted(live, key=lambda i: (-values[i], i))[:16]
                rows.append(sorted(winners) + [-1] * (16 - len(winners)))
            result.append(rows)
        return result

    def check(self, scores, lengths):
        output = torch.full(
            (*scores.shape[:2], 16), -99, dtype=torch.int32, device="cuda"
        )
        triton_reg16_topk(scores, lengths, output, 128, 16)
        self.assertEqual(output.cpu().tolist(), self.oracle(scores, lengths))

    def test_short_sequences_preserve_every_valid_block(self):
        lengths = torch.arange(18, device="cuda", dtype=torch.int32) * 128
        scores = torch.zeros((4, 18, 32), device="cuda", dtype=torch.float32)
        self.check(scores, lengths)

    def test_ragged_lengths_ties_and_stale_padding(self):
        for width in (31, 128, 627, 1024, 4096):
            with self.subTest(width=width):
                lengths = torch.tensor(
                    [0, 1, 127, 128, 129, (width - 1) * 128, width * 128],
                    device="cuda",
                    dtype=torch.int32,
                )
                scores = (torch.arange(width, device="cuda") % 7).float()
                scores = scores.expand(4, lengths.numel(), width).clone()
                for row, length in enumerate(lengths.cpu().tolist()):
                    scores[:, row, (length + 127) // 128 :] = 1.0e30
                self.check(scores, lengths)

    def test_strided_inputs(self):
        storage = torch.arange(4 * 3 * 128, device="cuda").float()
        scores = storage.reshape(4, 3, 128)[:, :, ::2]
        lengths = torch.tensor(
            [1, -99, 129, -99, 64 * 128, -99],
            device="cuda",
            dtype=torch.int32,
        )[::2]
        self.check(scores, lengths)

    def test_graph_replay_with_changing_live_lengths(self):
        scores = torch.zeros((4, 3, 32), device="cuda", dtype=torch.float32)
        lengths = torch.tensor([128, 15 * 128, 32 * 128], device="cuda", dtype=torch.int32)
        output = torch.empty((4, 3, 16), device="cuda", dtype=torch.int32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            triton_reg16_topk(scores, lengths, output, 128, 16)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            triton_reg16_topk(scores, lengths, output, 128, 16)
        torch.cuda.current_stream().wait_stream(stream)
        for live in ([0, 128, 17 * 128], [15 * 128, 16 * 128, 32 * 128]):
            lengths.copy_(torch.tensor(live, device="cuda", dtype=torch.int32))
            graph.replay()
            self.assertEqual(output.cpu().tolist(), self.oracle(scores, lengths))


if __name__ == "__main__":
    unittest.main()
