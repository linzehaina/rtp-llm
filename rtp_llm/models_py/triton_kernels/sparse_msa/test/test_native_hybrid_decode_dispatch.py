"""CPU dispatch gates for the experimental native attention backend."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.models_py.triton_kernels.sparse_msa.decode import native_hybrid_decode as backend


class NativeHybridDecodeDispatchTest(unittest.TestCase):
    def supported(self, q=None, **overrides):
        args = dict(q=q if q is not None else SimpleNamespace(shape=(4, 64, 128)),
                    layout=object(), topk=16, mma_scale_layout=True,
                    query_width=4, query_cu_seqlens=None)
        args.update(overrides)
        return backend.native_hybrid_decode_supported(**args)

    def test_default_and_explicit_disable_never_inspect_or_load(self):
        for setting in (None, "0", "false", ""):
            with self.subTest(setting=setting), patch.dict(os.environ, {}, clear=True), \
                    patch.object(backend, "_geometry_supported") as geometry, \
                    patch.object(backend, "load_library") as load:
                if setting is not None:
                    os.environ["MSA_USE_NATIVE_HYBRID_DECODE"] = setting
                self.assertFalse(self.supported(q=object()))
                geometry.assert_not_called()
                load.assert_not_called()

    def test_explicit_opt_in_preserves_supported_dispatch(self):
        with patch.dict(os.environ, {"MSA_USE_NATIVE_HYBRID_DECODE": "1"}), \
                patch.object(backend, "_geometry_supported", return_value=True) as geometry, \
                patch.object(backend, "load_library", return_value=object()) as load:
            self.assertTrue(self.supported())
            geometry.assert_called_once()
            load.assert_called_once_with()

    def test_opt_in_unsupported_metadata_keeps_production_fallback(self):
        cases = [dict(query_cu_seqlens=object()), dict(query_width=0),
                 dict(query_width=9), dict(output_format="mxfp8"),
                 dict(mma_scale_layout=False)]
        with patch.dict(os.environ, {"MSA_USE_NATIVE_HYBRID_DECODE": "1"}), \
                patch.object(backend, "_geometry_supported") as geometry, \
                patch.object(backend, "load_library") as load:
            for case in cases:
                with self.subTest(case=case):
                    self.assertFalse(self.supported(q=object(), **case))
            geometry.assert_not_called()
            load.assert_not_called()

    def test_unsupported_geometry_does_not_load_native_library(self):
        with patch.dict(os.environ, {"MSA_USE_NATIVE_HYBRID_DECODE": "1"}), \
                patch.object(backend, "_geometry_supported", return_value=False), \
                patch.object(backend, "load_library") as load:
            self.assertFalse(self.supported())
            load.assert_not_called()

    def test_missing_library_falls_back_after_valid_metadata(self):
        with patch.dict(os.environ, {"MSA_USE_NATIVE_HYBRID_DECODE": "1"}), \
                patch.object(backend, "_geometry_supported", return_value=True), \
                patch.object(backend, "load_library", return_value=None) as load:
            self.assertFalse(self.supported())
            load.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
