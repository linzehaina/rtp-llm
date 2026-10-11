"""CPU-only checks for the opt-in native-versus-legacy TopK diagnostic."""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import topk_bt_fused


class FakeKernel:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls += 1
            args[3].copy_(self.value)

        return launch


class PrefillTopkShadowTest(unittest.TestCase):
    def setUp(self):
        base = torch.arange(32, dtype=torch.int32).reshape(2, 4, 4)
        self.output = base[:, :3, :]
        self.args = (None, None, None, self.output)
        topk_bt_fused._topk_shadow_checked = 0
        topk_bt_fused._topk_shadow_mismatches = 0

    def test_disabled_does_not_launch_or_change_output(self):
        kernel = FakeKernel(self.output)
        with patch.object(topk_bt_fused, "_TOPK_SHADOW_LIMIT", 0), patch.object(
            topk_bt_fused, "_topk_to_block_table_multirow_kernel", kernel
        ):
            topk_bt_fused._shadow_check_topk(3, 1, 2, self.args, {})
        self.assertEqual(kernel.calls, 0)
        self.assertEqual(topk_bt_fused._topk_shadow_checked, 0)

    def test_capture_skips_shadow_without_consuming_limit(self):
        kernel = FakeKernel(self.output)
        trace = io.StringIO()
        args = (None, None, None, SimpleNamespace(is_cuda=True))
        with patch.object(topk_bt_fused, "_TOPK_SHADOW_LIMIT", 1), patch.object(
            topk_bt_fused, "_topk_shadow_capture_warned", False
        ), patch.object(
            topk_bt_fused, "_topk_to_block_table_multirow_kernel", kernel
        ), patch.object(
            topk_bt_fused.torch.cuda, "is_current_stream_capturing", return_value=True
        ), patch.object(topk_bt_fused.sys, "stderr", trace):
            topk_bt_fused._shadow_check_topk(3, 1, 2, args, {})
            topk_bt_fused._shadow_check_topk(3, 1, 2, args, {})
        self.assertEqual(kernel.calls, 0)
        self.assertEqual(topk_bt_fused._topk_shadow_checked, 0)
        self.assertEqual(trace.getvalue().count("skipped during CUDA Graph capture"), 1)

    def test_equal_and_mismatch_are_counted_without_changing_output(self):
        original = self.output.clone()
        kernel = FakeKernel(original)
        trace = io.StringIO()
        with patch.object(topk_bt_fused, "_TOPK_SHADOW_LIMIT", 2), patch.object(
            topk_bt_fused, "_topk_to_block_table_multirow_kernel", kernel
        ), patch.object(topk_bt_fused.sys, "stderr", trace):
            topk_bt_fused._shadow_check_topk(3, 1, 2, self.args, {})
            kernel.value = torch.full_like(original, -1)
            topk_bt_fused._shadow_check_topk(3, 1, 2, self.args, {})
            topk_bt_fused._shadow_check_topk(3, 1, 2, self.args, {})
        self.assertEqual(kernel.calls, 2)
        self.assertEqual(topk_bt_fused._topk_shadow_checked, 2)
        self.assertEqual(topk_bt_fused._topk_shadow_mismatches, 1)
        self.assertTrue(torch.equal(self.output, original))
        self.assertIn("checked=1 mismatches=0 equal=True", trace.getvalue())
        self.assertIn("checked=2 mismatches=1 equal=False", trace.getvalue())

    def test_limit_logs_final_success(self):
        kernel = FakeKernel(self.output.clone())
        trace = io.StringIO()
        with patch.object(topk_bt_fused, "_TOPK_SHADOW_LIMIT", 2), patch.object(
            topk_bt_fused, "_topk_to_block_table_multirow_kernel", kernel
        ), patch.object(topk_bt_fused.sys, "stderr", trace):
            topk_bt_fused._shadow_check_topk(3, 1, 2, self.args, {})
            topk_bt_fused._shadow_check_topk(3, 1, 2, self.args, {})
        self.assertIn("checked=2 mismatches=0 equal=True", trace.getvalue())


if __name__ == "__main__":
    unittest.main()
