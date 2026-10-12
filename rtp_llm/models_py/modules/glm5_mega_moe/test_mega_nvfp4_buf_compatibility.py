"""Default routed-only calls remain compatible with older DeepGEMM wheels."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location(
    "mega_nvfp4_buf_under_test", Path(__file__).with_name("mega_nvfp4_buf.py")
)
buf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(buf)


class BufferCompatibilityTest(unittest.TestCase):
    def setUp(self):
        buf._MEGA_NVFP4_BUF_CACHE.clear()
        self.group = SimpleNamespace(size=lambda: 8)

    def test_default_works_with_legacy_signature_and_reuses_buffer(self):
        calls = []
        result = SimpleNamespace()
        def legacy(group, num_experts, num_max_tokens_per_rank, num_topk,
                   hidden, intermediate_hidden, activation="swiglu"):
            calls.append(group)
            return result
        def size(group_size, experts, tokens, topk, hidden, intermediate,
                 activation):
            return (1234, None)
        api = SimpleNamespace(get_symm_buffer_for_mega_moe_nvfp4=legacy,
                              _C=SimpleNamespace(get_symm_buffer_size_for_mega_moe_nvfp4=size))
        with patch.dict("sys.modules", {"deep_gemm": api}):
            self.assertEqual(buf.estimate_mega_moe_nvfp4_symm_buffer_bytes(
                8, 128, 256, 8, 4096, 1024), 1234)
            self.assertIs(buf.get_or_create_mega_buf_nvfp4(
                self.group, 128, 256, 8, 4096, 1024), result)
            self.assertIs(buf.get_or_create_mega_buf_nvfp4(
                self.group, 128, 256, 8, 4096, 1024), result)
        self.assertEqual(len(calls), 1)

    def test_shared_experts_are_forwarded_and_have_a_separate_cache_key(self):
        create = Mock(side_effect=[SimpleNamespace(), SimpleNamespace()])
        size = Mock(return_value=(5678, None))
        api = SimpleNamespace(get_symm_buffer_for_mega_moe_nvfp4=create,
                              _C=SimpleNamespace(get_symm_buffer_size_for_mega_moe_nvfp4=size))
        with patch.dict("sys.modules", {"deep_gemm": api}):
            plain = buf.get_or_create_mega_buf_nvfp4(self.group, 128, 256, 8, 4096, 1024)
            shared = buf.get_or_create_mega_buf_nvfp4(
                self.group, 128, 256, 8, 4096, 1024, num_shared_experts=1)
        self.assertIsNot(plain, shared)
        self.assertNotIn("num_shared_experts", create.call_args_list[0].kwargs)
        self.assertEqual(create.call_args_list[1].kwargs["num_shared_experts"], 1)
        self.assertEqual(size.call_args_list[1].args[-1], 1)

    def test_unsupported_shared_experts_do_not_fall_back_to_plain_buffer(self):
        def legacy(group, num_experts, num_max_tokens_per_rank, num_topk,
                   hidden, intermediate_hidden, activation="swiglu"):
            return SimpleNamespace()
        api = SimpleNamespace(get_symm_buffer_for_mega_moe_nvfp4=legacy,
                              _C=SimpleNamespace(get_symm_buffer_size_for_mega_moe_nvfp4=Mock(side_effect=TypeError)))
        with patch.dict("sys.modules", {"deep_gemm": api}):
            with self.assertRaises(TypeError):
                buf.get_or_create_mega_buf_nvfp4(
                    self.group, 128, 256, 8, 4096, 1024, num_shared_experts=1)
        self.assertFalse(buf._MEGA_NVFP4_BUF_CACHE)


if __name__ == "__main__":
    unittest.main()
