"""Metadata-only dispatch regressions; no GPU or native extension required."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch


def load_dispatch():
    path = Path(__file__).resolve().parents[1] / "prefill" / "warp_topk.py"
    spec = importlib.util.spec_from_file_location("warp_dispatch_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TensorMetadata:
    def __init__(self, shape, dtype, pointer=1):
        self.shape, self.dtype, self.pointer = shape, dtype, pointer
        self.is_cuda = True

    def stride(self, dim):
        return 1 if dim == len(self.shape) - 1 else self.shape[-1]

    def data_ptr(self):
        return self.pointer

    def is_contiguous(self):
        return True


class PrefillWarpDispatchTest(unittest.TestCase):
    def setUp(self):
        self.module = load_dispatch()
        s = TensorMetadata((4, 16384, 579), torch.float32)
        ti = TensorMetadata((4, 16384, 16), torch.int32, 2)
        cu = TensorMetadata((19,), torch.int32, 3)
        prefix = TensorMetadata((18,), torch.int32, 4)
        self.args = [s, object(), object(), ti, 1, 128, cu, cu, prefix, 16, 0, 1, 579]
        self.kwargs = dict(
            NKV=4,
            MASK_INIT=False,
            MASK_LOCAL=False,
            EMIT_TOPK_IDX=True,
            EMIT_BLOCK_TABLE=False,
        )

    def test_default_returns_gpu_flags_for_compatible_fallback(self):
        flags = object()
        op = Mock(return_value=flags)
        with patch.object(self.module, "_native_topk", return_value=op), patch.dict(
            "os.environ", {}, clear=True
        ):
            self.assertIs(
                self.module.try_warp_prefill_topk(932, 4, 579, self.args, self.kwargs),
                flags,
            )
        op.assert_called_once()

    def test_scope_boundaries_and_legacy_output_index_overflow(self):
        with patch.dict("os.environ", {"M3_PREFILL_TOPK_IMPL": "warp"}), patch.object(
            self.module, "_native_topk", return_value=Mock()
        ) as load:
            for blocks in (512, 1024):
                args = list(self.args)
                args[0] = TensorMetadata((4, 1024, blocks), torch.float32)
                self.assertIsNotNone(
                    self.module.try_warp_prefill_topk(
                        1024, 4, blocks, args, self.kwargs
                    )
                )
            load.reset_mock()
            args = list(self.args)
            args[3] = TensorMetadata((4, 16384, 16), torch.int32)
            args[3].stride = lambda dim: (2147483647, 16, 1)[dim]
            self.assertIsNone(
                self.module.try_warp_prefill_topk(932, 4, 579, args, self.kwargs)
            )
            load.assert_not_called()

    def test_native_receives_original_views_and_emission_parameters(self):
        op = Mock()
        with patch.object(self.module, "_native_topk", return_value=op), patch.dict(
            "os.environ", {"M3_PREFILL_TOPK_IMPL": "warp"}
        ):
            self.assertTrue(
                self.module.try_warp_prefill_topk(932, 4, 579, self.args, self.kwargs)
            )
        op.assert_called_once_with(
            self.args[0],
            self.args[3],
            self.args[1],
            self.args[2],
            self.args[6],
            self.args[8],
            932,
            16,
            0,
            1,
            False,
            579,
        )

    def test_unmeasured_geometry_and_mask_contracts_keep_bitonic(self):
        scenarios = [
            (4, 2),
            (5, 64),
            (7, TensorMetadata((19,), torch.int32, 99)),
            (9, 8),
            (9, 32),
            (0, TensorMetadata((4, 16384, 579), torch.float16)),
            (0, TensorMetadata((4, 255, 579), torch.float32)),
            (0, TensorMetadata((4, 1023, 579), torch.float32)),
            (3, TensorMetadata((4, 16384, 16), torch.int64)),
            (6, TensorMetadata((19,), torch.int64)),
        ]
        with patch.dict("os.environ", {"M3_PREFILL_TOPK_IMPL": "warp"}), patch.object(
            self.module, "_native_topk"
        ) as load:
            for idx, value in scenarios:
                args = list(self.args)
                args[idx] = value
                with self.subTest(idx=idx):
                    self.assertFalse(
                        self.module.try_warp_prefill_topk(
                            932, 4, 579, args, self.kwargs
                        )
                    )
            for blocks in (32, 128, 255, 256, 384, 448, 511, 1025, 2048, 8192):
                self.assertFalse(
                    self.module.try_warp_prefill_topk(
                        932, 4, blocks, self.args, self.kwargs
                    )
                )
            for heads in (1, 2, 8):
                args = list(self.args)
                args[0] = TensorMetadata((heads, 16384, 579), torch.float32)
                kwargs = dict(self.kwargs, NKV=heads)
                self.assertFalse(
                    self.module.try_warp_prefill_topk(932, heads, 579, args, kwargs)
                )
            for key, value in (
                ("NKV", 2),
                ("MASK_INIT", True),
                ("MASK_LOCAL", True),
                ("EMIT_TOPK_IDX", False),
                ("EMIT_BLOCK_TABLE", True),
            ):
                kwargs = dict(self.kwargs)
                kwargs[key] = value
                self.assertFalse(
                    self.module.try_warp_prefill_topk(932, 4, 579, self.args, kwargs)
                )
            load.assert_not_called()

    def test_explicit_rollback_and_older_wheel(self):
        with patch.dict(
            "os.environ", {"M3_PREFILL_TOPK_IMPL": "bitonic"}
        ), patch.object(self.module, "_native_topk") as load:
            self.assertFalse(
                self.module.try_warp_prefill_topk(932, 4, 579, self.args, self.kwargs)
            )
            load.assert_not_called()
        with patch.dict("os.environ", {"M3_PREFILL_TOPK_IMPL": "warp"}), patch.object(
            self.module, "_native_topk", return_value=None
        ):
            self.assertFalse(
                self.module.try_warp_prefill_topk(932, 4, 579, self.args, self.kwargs)
            )

    def test_native_errors_propagate(self):
        with patch.dict("os.environ", {"M3_PREFILL_TOPK_IMPL": "warp"}), patch.object(
            self.module,
            "_native_topk",
            return_value=Mock(side_effect=RuntimeError("invalid input")),
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid input"):
                self.module.try_warp_prefill_topk(932, 4, 579, self.args, self.kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
