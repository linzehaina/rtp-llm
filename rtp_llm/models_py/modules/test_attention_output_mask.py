"""Attention/postnorm ABI regressions, including real B1 verify-mask metadata."""
import ast
import importlib.util
from pathlib import Path
import unittest

import torch

_SPEC = importlib.util.spec_from_file_location(
    'attention_output_under_test', Path(__file__).with_name('attention_output.py'))
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
DeferredMaskedAttentionOutput = _MODULE.DeferredMaskedAttentionOutput
mask_attention_output = _MODULE.mask_attention_output


def metadata_builder():
    # Run the production metadata builder without importing the GPU model stack.
    source = Path(__file__).parent / 'hybrid/msa_attention.py'
    tree = ast.parse(source.read_text())
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef)
                    and x.name == '_build_target_verify_token_metadata')
    env = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(source), 'exec'), env)
    return env[function.name]


class AttentionOutputMaskTest(unittest.TestCase):
    def test_single_request_verify_metadata_uses_legacy_stride_zero_fallback(self):
        build = metadata_builder()
        for width in (2, 4, 5, 8):
            for valid in (False, True):
                with self.subTest(width=width, valid=valid):
                    _, _, mask = build(torch.tensor([123]), torch.tensor([width if valid else 0]),
                                       width, torch.device('cpu'))
                    self.assertEqual(mask.stride(), (0,))
                    output = torch.randn(width, 128, dtype=torch.bfloat16)
                    actual = mask_attention_output(output, mask, defer_mask=True)
                    self.assertIsInstance(actual, torch.Tensor)
                    self.assertTrue(torch.equal(actual, torch.where(mask[:, None], output, torch.zeros_like(output))))

    def test_single_token_and_multi_request_metadata_keep_supported_deferred_path(self):
        build = metadata_builder()
        for batch, width in ((1, 1), (2, 5), (32, 5)):
            with self.subTest(batch=batch, width=width):
                _, _, mask = build(torch.arange(batch), torch.full((batch,), width),
                                   batch * width, torch.device('cpu'))
                output = torch.randn(batch * width, 128, dtype=torch.bfloat16)
                actual = mask_attention_output(output, mask, defer_mask=True)
                self.assertIsInstance(actual, DeferredMaskedAttentionOutput)
                self.assertIs(actual.hidden_states, output)
                self.assertIs(actual.valid_row_mask, mask)

    def test_noncontiguous_mask_retains_legacy_masking(self):
        mask = torch.tensor([True, False, False, False, True, False])[::2]
        self.assertFalse(mask.is_contiguous())
        output = torch.randn(3, 128)
        actual = mask_attention_output(output, mask, defer_mask=True)
        self.assertIsInstance(actual, torch.Tensor)
        self.assertTrue(torch.equal(actual, torch.where(mask[:, None], output, torch.zeros_like(output))))

    def test_invalid_nan_rows_become_positive_zero_before_residual(self):
        output = torch.full((5, 128), float('nan'), dtype=torch.bfloat16)
        mask = torch.tensor([False]).expand(5)
        actual = mask_attention_output(output, mask, defer_mask=True)
        self.assertFalse(torch.isnan(actual).any())
        self.assertFalse(torch.signbit(actual).any())
        residual = torch.randn_like(output)
        self.assertTrue(torch.equal(actual + residual, residual))

    def test_explicit_non_deferred_path_preserves_tensor_api(self):
        output = torch.randn(4, 128)
        mask = torch.tensor([True, False, True, False])
        actual = mask_attention_output(output, mask, defer_mask=False)
        self.assertIsInstance(actual, torch.Tensor)
        self.assertTrue(torch.equal(actual, torch.where(mask[:, None], output, torch.zeros_like(output))))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for actual norm/Graph regression')
    def test_broadcast_mask_eager_and_graph_replay_read_fresh_scalar(self):
        from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import fused_add_rmsnorm_fp8_quant_with_bf16_output
        device = torch.device('cuda')
        output = torch.randn(5, 6144, device=device, dtype=torch.bfloat16)
        output[1] = float('nan')
        initial = torch.randn_like(output)
        residual = initial.clone()
        scalar = torch.tensor([False], device=device)
        mask = scalar.expand(5)
        weight = torch.ones(6144, device=device, dtype=torch.bfloat16)
        def run():
            residual.copy_(initial)
            value = mask_attention_output(output, mask, defer_mask=True)
            self.assertIsInstance(value, torch.Tensor)
            return fused_add_rmsnorm_fp8_quant_with_bf16_output(value, residual, weight,
                        group_size=32, round_to_pow2=True)
        run()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(3): run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream): captured = run()
        for flag in (False, True, False):
            # Make valid rows finite; preserve an invalid NaN case for the false mask.
            output.fill_(1.25 if flag else float('nan'))
            scalar.fill_(flag)
            reference_residual = initial.clone()
            reference = fused_add_rmsnorm_fp8_quant_with_bf16_output(
                torch.where(mask[:, None], output, torch.zeros_like(output)),
                reference_residual, weight, group_size=32, round_to_pow2=True)
            graph.replay();torch.cuda.synchronize()
            self.assertTrue(torch.equal(residual, reference_residual))
            for actual, expected in zip(captured, reference):
                self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))

if __name__ == '__main__': unittest.main(verbosity=2)
