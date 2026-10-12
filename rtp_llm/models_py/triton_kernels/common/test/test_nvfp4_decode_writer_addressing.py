"""Independent byte oracle and pending GPU gates for the Decode-only candidate.

CPU helpers intentionally do not use the production encoder or torch FP8 cast.
GPU tests compare whole guarded backing allocations, including unwritten bytes.
"""
import math
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.triton_kernels.common import nvfp4_kv_cache as nv


def slot_oracle(length, table_row, page_size, num_blocks):
    pos = int(length) - 1
    if pos < 0 or pos // page_size >= len(table_row):
        return -1
    page = int(table_row[pos // page_size])
    slot = page * page_size + pos % page_size
    return slot if page > 0 and 0 <= slot < num_blocks * page_size else -1


def e4m3_value(code):
    exponent, mantissa = code >> 3, code & 7
    return mantissa / 512 if exponent == 0 else (1 + mantissa / 8) * 2 ** (exponent - 7)


def group_byte_oracle(values):
    if len(values) != 16 or not all(math.isfinite(x) for x in values):
        raise ValueError("oracle accepts sixteen finite values")
    raw = min(448, max(1 / 512, max(abs(x) for x in values) / 6))
    # Independent E4M3 RNE: enumerate its 127 positive finite representations.
    scale_code = min(range(127), key=lambda c: (abs(e4m3_value(c) - raw), c & 1))
    scale = e4m3_value(scale_code)
    codes = []
    for value in values:
        magnitude = abs(value)
        if magnitude > 5 * scale:
            code = 7
        elif magnitude >= 3.5 * scale:
            code = 6
        elif magnitude > 2.5 * scale:
            code = 5
        elif magnitude >= 1.75 * scale:
            code = 4
        elif magnitude > 1.25 * scale:
            code = 3
        elif magnitude >= .75 * scale:
            code = 2
        elif magnitude > .25 * scale:
            code = 1
        else:
            code = 0
        codes.append(code | (8 if value < 0 and code else 0))
    return [codes[i] | (codes[i + 1] << 4) for i in range(0, 16, 2)], scale_code


def cache_byte_oracle(k, v, index_k, lengths, tables, blocks, heads, page_size,
                      base_stride, side_stride, guard, mma):
    base = bytearray([165]) * (blocks * base_stride)
    side = bytearray([165]) * (blocks * side_stride)
    packed_plane = heads * page_size * 64
    scale_plane = heads * page_size * 8
    index_begin = 2 * scale_plane
    for row, length in enumerate(lengths):
        slot = slot_oracle(length, tables[row], page_size, blocks)
        if slot < 0:
            continue
        block, offset = divmod(slot, page_size)
        for plane, payload in enumerate((k[row], v[row], [index_k[row]])):
            for head, values in enumerate(payload):
                for group in range(8):
                    packed, scale = group_byte_oracle(values[group * 16:(group + 1) * 16])
                    if mma:
                        scale_at = (group // 4) * 512 + (offset % 32) * 16 + (offset // 32) * 4 + group % 4
                    else:
                        scale_at = offset * 8 + group
                    if plane < 2:
                        at = block * base_stride + guard + plane * packed_plane + (head * page_size + offset) * 64 + group * 8
                        scale_at += block * side_stride + guard + plane * scale_plane + head * page_size * 8
                        base[at:at + 8] = bytes(packed)
                    else:
                        at = block * side_stride + guard + index_begin + offset * 64 + group * 8
                        side[at:at + 8] = bytes(packed)
                        scale_at += block * side_stride + guard + index_begin + page_size * 64
                    side[scale_at] = scale
    return base, side


class TestDecodeWriterCPUOracle(unittest.TestCase):
    def test_slot_page_and_pool_bounds(self):
        self.assertEqual(slot_oracle(128, [1, 2], 128, 4), 255)
        self.assertEqual(slot_oracle(129, [1, 2], 128, 4), 256)
        for length, table in ((0, [1]), (-2147483648, [1]), (129, [1]), (1, [0]), (1, [-1]), (1, [4]), (2147483647, [1]), (1, [])):
            self.assertEqual(slot_oracle(length, table, 128, 4), -1)
        self.assertEqual(slot_oracle(1, [2147483647], 128, 4), -1)

    def test_e2m1_ties_and_signed_zero(self):
        values = [0, .25, .75, 1.25, 1.75, 2.5, 3.5, 5, 6, -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5]
        packed, scale = group_byte_oracle(values)
        self.assertEqual(scale, 56)  # E4M3(1.0)
        codes = [0, 0, 2, 2, 4, 4, 6, 6, 7, 0, 10, 10, 12, 12, 14, 14]
        self.assertEqual(packed, [codes[i] | (codes[i + 1] << 4) for i in range(0, 16, 2)])
        self.assertEqual(group_byte_oracle([0] * 16), ([0] * 8, 1))

    def test_e4m3_midpoint_tie(self):
        # amax/6 = 1.0625 ties E4M3 1.0(code56) / 1.125(code57).
        self.assertEqual(group_byte_oracle([6.375] + [0] * 15)[1], 56)
        self.assertEqual(group_byte_oracle([1e9] + [0] * 15)[1], 126)


    def test_guarded_oracle_byte_extent_and_unwritten_tail(self):
        k = [[[0] * 128 for _ in range(4)]]
        guard, base_stride, side_stride = 17, 65536 + 66, 17408 + 66
        for mma in (False, True):
            base, side = cache_byte_oracle(k, k, [[0] * 128], [128], [[1]], 3, 4, 128, base_stride, side_stride, guard, mma)
            self.assertEqual(sum(x != 165 for x in base), 512)
            self.assertEqual(sum(x != 165 for x in side), 136)
            for data, stride in ((base, base_stride), (side, side_stride)):
                self.assertEqual(data[:stride], bytearray([165]) * stride)
                self.assertEqual(data[2 * stride:], bytearray([165]) * stride)
                self.assertEqual(data[stride:stride + guard], bytearray([165]) * guard)
                self.assertEqual(data[2 * stride - guard:2 * stride], bytearray([165]) * guard)
            untouched = cache_byte_oracle(k, k, [[0] * 128], [128], [[0]], 3, 4, 128, base_stride, side_stride, guard, mma)
            self.assertTrue(all(x == 165 for data in untouched for x in data))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA gate pending")
class TestDecodeWriterAddressingCUDA(unittest.TestCase):
    def make_case(self, rows, *, empty_table=False):
        heads, blocks, guard = 4, rows + 3, 17
        carrier = torch.randn(rows, 1232, device="cuda", dtype=torch.bfloat16)
        k = carrier[:, :512].reshape(rows, heads, 128)
        v = carrier[:, 512:1024].reshape(rows, heads, 128)
        idx = carrier[:, 1024:1152]
        lengths = torch.zeros(rows * 2, device="cuda", dtype=torch.int32)[::2]
        table = torch.ones(rows * 2, 10, device="cuda", dtype=torch.int32)[::2, ::2]
        if empty_table:
            table = table[:, :0]
        else:
            lengths.copy_(torch.tensor([1, 128, 129, 640, 641, 0, -1, -2147483648, 2147483647] * ((rows + 8) // 9), dtype=torch.int32, device="cuda")[:rows])
            table.copy_((torch.arange(rows, device="cuda", dtype=torch.int32) + 1)[:, None].expand_as(table))
            table[::7] = 0
            table[1::11] = -1
            table[2::13] = blocks
            table[3::17] = 2147483647
        base_width = 2 * heads * 128 * 64 + 32
        side_width = 2 * heads * 128 * 8 + 128 * (64 + 8) + 32
        pairs = []
        for _ in range(2):
            base = torch.full((blocks, base_width + 2 * guard), 165, device="cuda", dtype=torch.uint8)
            side = torch.full((blocks, side_width + 2 * guard), 165, device="cuda", dtype=torch.uint8)
            layout = nv.cache_layout(base[:, guard:-guard], side[:, guard:-guard], heads, 128, 128)
            pairs.append((base, side, layout))
        return k, v, idx, lengths, table, pairs, guard

    def check_bytes(self, case, mma):
        k, v, idx, lengths, table, pairs, guard = case
        expected = cache_byte_oracle(k.cpu().float().tolist(), v.cpu().float().tolist(), idx.cpu().float().tolist(), lengths.cpu().tolist(), table.cpu().tolist(), pairs[0][2].num_blocks, 4, 128, pairs[0][0].stride(0), pairs[0][1].stride(0), guard, mma)
        for base, side, _ in pairs:
            for actual, oracle in zip((base, side), expected):
                self.assertEqual(bytes(actual.cpu().flatten().tolist()), bytes(oracle))

    def test_all_bytes_tail_empty_and_strided_inputs(self):
        torch.manual_seed(101001)
        for rows in (0, 1, 33, 160, 161):
            for empty in (False, True):
                for mma in (False, True):
                    with self.subTest(rows=rows, empty=empty, mma=mma):
                        case = self.make_case(rows, empty_table=empty)
                        k, v, idx, lengths, table, pairs, _ = case
                        snapshots = [x.clone() for x in (k, v, idx, lengths, table)]
                        nv.quantize_decode_main_index_rows(k, v, idx, lengths, table, pairs[0][2], mma_scale_layout=mma)
                        slots = nv.build_decode_physical_slots(lengths, table)
                        nv.quantize_main_index_rows(k, v, idx, slots, pairs[1][2], mma_scale_layout=mma)
                        torch.cuda.synchronize()
                        self.check_bytes(case, mma)
                        for actual, snapshot in zip((k, v, idx, lengths, table), snapshots):
                            self.assertTrue(torch.equal(actual, snapshot))

    def test_graph_fresh_lengths_table_and_unwritten_canaries(self):
        case = self.make_case(160)
        k, v, idx, lengths, table, pairs, _ = case
        graphs = []
        for fused, (_, _, layout) in zip((True, False), pairs):
            def write():
                if fused:
                    nv.quantize_decode_main_index_rows(k, v, idx, lengths, table, layout, mma_scale_layout=True)
                else:
                    nv.quantize_main_index_rows(k, v, idx, nv.build_decode_physical_slots(lengths, table), layout, mma_scale_layout=True)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    write()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                write()
            graphs.append(graph)
        for state in ("page_boundary", "new_tables_same_storage", "all_invalid"):
            for base, side, _ in pairs:
                base.fill_(165)
                side.fill_(165)
            if state == "all_invalid":
                lengths.zero_()
                table.fill_(1)
            else:
                lengths.fill_(128 if state == "page_boundary" else 129)
                table.copy_((torch.arange(160, device="cuda", dtype=torch.int32) + (1 if state == "page_boundary" else 2))[:, None].expand_as(table))
                table[::9] = -1
                k.add_(.125)
                v.sub_(.125)
                idx.mul_(2)
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            self.check_bytes(case, True)

    def test_int64_and_generic_keep_original_fallback(self):
        case = self.make_case(33)
        k, v, idx, lengths, table, pairs, _ = case
        # Values above int32 remain int64 when the original mapper is invoked.
        wide = table.long()
        wide[::2] = 2**31 + 7
        with mock.patch.object(nv, "build_decode_physical_slots", wraps=nv.build_decode_physical_slots) as mapper, mock.patch.object(nv, "_quantize_decode_main_index_rows_d128_kernel") as fused:
            nv.quantize_decode_main_index_rows(k, v, idx, lengths.long(), wide, pairs[0][2], mma_scale_layout=True)
            mapper.assert_called_once()
            self.assertIs(mapper.call_args.args[1], wide)
            fused.__getitem__.assert_not_called()
        # Non-d128 index geometry is supported by the existing generic writer.
        with mock.patch.object(nv, "build_decode_physical_slots", wraps=nv.build_decode_physical_slots) as mapper, mock.patch.object(nv, "_quantize_decode_main_index_rows_d128_kernel") as fused:
            nv.quantize_decode_main_index_rows(k, v, idx[:, :64], lengths, table, pairs[0][2], mma_scale_layout=True)
            mapper.assert_called_once()
            fused.__getitem__.assert_not_called()


if __name__ == "__main__":
    unittest.main()
