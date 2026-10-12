import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.models.downstream_modules.decision_sources import (
    detect_kev_source,
    detect_native_decision_source,
    resolve_jev_labels,
    resolve_kev_markers,
)


class _FakeTokenizer:
    """Minimal tokenizer: A..Z single-token; kev markers are single tokens."""

    def __init__(self):
        self._ids = {c: 1000 + i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
        self._ids.update(
            {
                "<state>": 2001,
                "<question>": 2002,
                "<opt>": 2003,
                "</opt>": 2004,
                "<decide>": 2005,
            }
        )

    def __call__(self, text, add_special_tokens=False):
        ids = []
        rest = text
        while rest:
            for token, tid in self._ids.items():
                if len(token) > 1 and rest.startswith(token):
                    ids.append(tid)
                    rest = rest[len(token):]
                    break
            else:
                ch = rest[0]
                ids.append(self._ids.get(ch, 7))
                rest = rest[1:]
        return SimpleNamespace(input_ids=ids)


def _write_jev_release(root: Path, wide_codes: int = 26):
    adapter = root / "adapter_vllm"
    adapter.mkdir()
    verbalizers = ["false", "true", "0", "1", "2", "3", "4", "5"] + list("ABCDEFGHIJKLMNOP")
    verbalizer_ids = [10, 11, 12, 13, 14, 15, 16, 17] + [
        1000 + i for i in range(16)
    ]
    bias = [0.01 * i for i in range(24)]
    (adapter / "decision_head.json").write_text(
        json.dumps(
            {
                "bias": bias,
                "verbalizer_ids": verbalizer_ids,
                "slots": {
                    "num_slots": 24,
                    "ranges": {
                        "noul": [0, 2],
                        "score": [2, 8],
                        "choice": [8, 24],
                    },
                    "verbalizers": verbalizers,
                    "template_version": "bare-v1",
                },
            }
        )
    )
    (root / "calibration.json").write_text(
        json.dumps({"per_kind": {"noul": 1.5, "choice": 2.0, "score": 1.1}})
    )
    return verbalizers, verbalizer_ids, bias


class JevNativeSourceTest(unittest.TestCase):
    def test_detect_and_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            verbalizers, verbalizer_ids, bias = _write_jev_release(root)

            settings = detect_native_decision_source(str(root), None)
            self.assertIsNotNone(settings)
            self.assertEqual(settings["head_type"], "linear")
            self.assertEqual(settings["prompt_format"], "bare")
            self.assertEqual(settings["head_weight"], "lm_head.weight")
            self.assertEqual(settings["slot_ranges"]["noul"], [0, 2])
            self.assertEqual(settings["temperatures"], {"noul": 1.5, "choice": 2.0, "score": 1.1})

            resolved = resolve_jev_labels(settings, _FakeTokenizer())
            # 24 trained slots + 26 wide single-letter codes
            self.assertEqual(resolved["num_labels"], 24 + 26)
            self.assertEqual(
                resolved["slot_ranges"]["choice"], [8, 24 + 26]
            )
            self.assertEqual(
                resolved["label_token_ids"][:24], verbalizer_ids
            )
            # wide codes follow the tokenizer's own ids
            self.assertEqual(
                resolved["label_token_ids"][24:], [1000 + i for i in range(26)]
            )
            self.assertEqual(resolved["head_bias"][:24], bias)
            self.assertEqual(resolved["head_bias"][24:], [0.0] * 26)
            # the detector's state is not mutated by resolve
            self.assertEqual(settings["slot_ranges"]["choice"], [8, None])

    def test_no_adapter_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(detect_native_decision_source(tmp, None))

    def test_wrong_temperature_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_jev_release(root)
            (root / "calibration.json").write_text(
                json.dumps({"per_kind": {"noul": -3.0, "choice": "bad"}})
            )
            settings = detect_native_decision_source(str(root), None)
            self.assertEqual(settings.get("temperatures") or {}, {})


class KevNativeSourceTest(unittest.TestCase):
    def _write_kev_adapter(self, root: Path):
        state = {
            "q.weight": torch.randn(32, 64),
            "q.bias": torch.randn(32),
            "k.weight": torch.randn(32, 64),
            "k.bias": torch.randn(32),
        }
        torch.save({"head": state, "base": "Qwen/Qwen2.5-0.5B", "lora": 16}, root / "head.pt")
        (root / "eval.json").write_text(
            json.dumps({"temperature_scaling": {"T": 2.5}})
        )
        return state

    def test_detect_and_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self._write_kev_adapter(root)

            settings = detect_kev_source(str(root))
            self.assertIsNotNone(settings)
            self.assertEqual(settings["head_type"], "pointer")
            self.assertEqual(settings["prompt_format"], "kev_branch")
            self.assertEqual(settings["proj_dim"], 32)
            self.assertAlmostEqual(settings["scale"], 1.0 / math.sqrt(32))
            self.assertEqual(settings["temperature"], 2.5)
            self.assertTrue(
                torch.equal(settings["head_tensors"]["q.weight"], state["q.weight"])
            )

            resolved = resolve_kev_markers(settings, _FakeTokenizer())
            # markers resolve through the tokenizer; <decide> is not a single
            # token in the fake, so expect the failure path
            self.assertIn("marker_token_ids", resolved)
            for name in ("state", "question", "option", "option_end", "decide"):
                self.assertIn(name, resolved["marker_token_ids"])

    def test_decision_source_routing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            base.mkdir()
            adapter = Path(tmp) / "adapter"
            adapter.mkdir()
            self._write_kev_adapter(adapter)
            # base has no native layout; the adapter is reachable only via
            # the explicit decision_source
            self.assertIsNone(detect_native_decision_source(str(base), None))
            settings = detect_native_decision_source(str(base), str(adapter))
            self.assertIsNotNone(settings)
            self.assertEqual(settings["head_type"], "pointer")


if __name__ == "__main__":
    unittest.main()
