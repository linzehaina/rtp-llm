import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.nvfp4_cp_wire import (
    pack_cp_nvfp4_wire,
    scatter_cp_nvfp4_wire,
)
from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import reference_quantize


_PLANE_SIZES = (32768, 4096, 32768, 4096, 8192, 1024)


def _planes(pages):
    result = []
    for i, width in enumerate(_PLANE_SIZES):
        dtype = torch.uint8 if i % 2 == 0 else torch.float8_e4m3fn
        plane = torch.empty((pages, width), dtype=dtype, device="cuda")
        plane.view(torch.uint8).fill_(0xA5)
        result.append(plane)
    return tuple(result)


def _mma_scale_offset(row, group):
    return (group // 4) * 512 + (row % 32) * 16 + (row // 32) * 4 + group % 4


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class NVFP4CPWireGPUTest(unittest.TestCase):
    def test_pack_rows_reuse_and_contents(self):
        packed = torch.arange(16 * 1152, device="cuda", dtype=torch.float32)
        packed = ((packed % 137) - 68).reshape(16, 1152).to(torch.bfloat16) / 32
        short = torch.empty((7, 648), device="cuda", dtype=torch.uint8)
        full = torch.empty((16, 648), device="cuda", dtype=torch.uint8)
        pack_cp_nvfp4_wire(packed[:7], short)
        pack_cp_nvfp4_wire(packed, full)
        torch.testing.assert_close(short, full[:7], rtol=0, atol=0)
        codes, scales = reference_quantize(packed[:7].cpu().reshape(7, 9, 8, 16))
        expected = torch.empty((7, 9, 72), dtype=torch.uint8)
        expected[:, :, :64] = codes.reshape(7, 9, 64)
        expected[:, :, 64:] = scales.view(torch.uint8)
        torch.testing.assert_close(short.cpu(), expected.reshape(7, 648), rtol=0, atol=0)

    def test_scatter_changing_geometry_and_fi_layout(self):
        for fi_layout in (False, True):
            for rows, source_rows, work_pages, persist_pages in (
                (7, 17, 1, 1),
                (16, 32, 16, 16),
                (17, 33, 2, 1),
                (31, 48, 1, 16),
            ):
                with self.subTest(fi_layout=fi_layout, rows=rows):
                    source = [(r * 3 + 1) % source_rows for r in range(rows)]
                    work_slots = [(work_pages - 1) * 128 + r * 3 for r in range(rows)]
                    persist_slots = [(persist_pages - 1) * 128 + r * 2 + 1 for r in range(rows)]
                    wire_cpu = torch.arange(source_rows * 648, dtype=torch.int32)
                    wire_cpu = ((wire_cpu * 29 + 7) % 256).to(torch.uint8).reshape(source_rows, 648)
                    wire = wire_cpu.to("cuda")
                    maps = tuple(
                        torch.tensor(values, dtype=torch.int64, device="cuda")
                        for values in (source, work_slots, persist_slots)
                    )
                    working = _planes(work_pages)
                    persistent = _planes(persist_pages)
                    scatter_cp_nvfp4_wire(
                        wire, *maps, working, persistent,
                        fi_working_layout=fi_layout, rows_per_cta=16,
                    )

                    expected = [
                        [bytearray([0xA5]) * (pages * width) for width in _PLANE_SIZES]
                        for pages in (work_pages, persist_pages)
                    ]
                    for row, src in enumerate(source):
                        record = bytes(wire_cpu[src].tolist())
                        for plane in range(9):
                            codes = record[plane * 72:plane * 72 + 64]
                            scales = record[plane * 72 + 64:plane * 72 + 72]
                            for target, slot in enumerate((work_slots[row], persist_slots[row])):
                                block, off = divmod(slot, 128)
                                head = plane if plane < 4 else plane - 4
                                if plane == 8:
                                    code_plane, scale_plane = 4, 5
                                    code_offset = block * 8192 + off * 64
                                else:
                                    code_plane, scale_plane = (0, 1) if plane < 4 else (2, 3)
                                    code_offset = block * 32768 + (head * 128 + off) * 64
                                expected[target][code_plane][code_offset:code_offset + 64] = codes
                                for group, value in enumerate(scales):
                                    if plane == 8:
                                        scale_offset = block * 1024 + _mma_scale_offset(off, group)
                                    elif target == 0 and fi_layout and plane < 4:
                                        scale_offset = block * 4096 + head * 1024 + off * 8 + group
                                    elif target == 0 and fi_layout:
                                        v_offset = ((off // 4) * 4 + group // 2) * 8 + (group % 2) * 4 + off % 4
                                        scale_offset = block * 4096 + head * 1024 + v_offset
                                    else:
                                        scale_offset = block * 4096 + head * 1024 + _mma_scale_offset(off, group)
                                    expected[target][scale_plane][scale_offset] = value

                    for target, planes in enumerate((working, persistent)):
                        for i, plane in enumerate(planes):
                            self.assertEqual(
                                plane.view(torch.uint8).cpu().contiguous().numpy().tobytes(),
                                bytes(expected[target][i]),
                                (fi_layout, rows, target, i),
                            )


if __name__ == "__main__":
    unittest.main()
