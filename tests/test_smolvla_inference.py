"""Exercise SmolVLA's actual queue methods without loading model weights.

Only the inference methods are extracted; model prediction is replaced with
known tensors so batching, queue reuse and recorded-action isolation are tested.
"""
from __future__ import annotations

import ast
from collections import deque
from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from common.constants import ACTION
from common.policies.utils import populate_queues


class SmolVLAInferenceTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "common/policies/smolvla/modeling_smolvla.py"
        tree = ast.parse(path.read_text())
        policy = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SmolVLAPolicy")
        methods = [n for n in policy.body if isinstance(n, ast.FunctionDef)
                   and n.name in {"select_action", "predict_action_chunk"}]
        env = {"torch": torch, "Tensor": torch.Tensor, "ACTION": ACTION,
               "populate_queues": populate_queues}
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                  *methods], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
        self.select_action = env["select_action"]
        self.predict_action_chunk = env["predict_action_chunk"]
        self.chunks = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
        self.policy = SimpleNamespace(
            eval=mock.Mock(), _prepare_batch=lambda batch: dict(batch),
            _queues={ACTION: deque(maxlen=2)}, config=SimpleNamespace(n_action_steps=2),
            _get_action_chunk=mock.Mock(return_value=self.chunks),
        )

    def test_select_reuses_chunk_and_refills_in_batched_action_order(self):
        # Two-dimensional actions also ensure inference has no XYZ plotting assumption.
        out = io.StringIO()
        with redirect_stdout(out):
            first = self.select_action(self.policy, {"observation.state": torch.zeros(2, 2)})
            second = self.select_action(self.policy, {})
            third = self.select_action(self.policy, {})
        torch.testing.assert_close(first, self.chunks[:, 0])
        torch.testing.assert_close(second, self.chunks[:, 1])
        torch.testing.assert_close(third, self.chunks[:, 0])
        self.assertEqual(self.policy._get_action_chunk.call_count, 2)
        self.assertEqual(out.getvalue(), "")

    def test_select_ignores_recorded_actions_in_offline_batch(self):
        result = self.select_action(self.policy, {ACTION: torch.full((2, 2), -99.0)})
        torch.testing.assert_close(result, self.chunks[:, 0])
        self.policy._get_action_chunk.assert_called_once()

    def test_predict_chunk_preserves_pending_actions_and_passes_noise(self):
        pending = torch.full((2, 2), -1.0)
        self.policy._queues[ACTION].append(pending)
        noise = torch.zeros_like(self.chunks)
        result = self.predict_action_chunk(self.policy, {ACTION: torch.full((2, 2), -99.0)}, noise)
        torch.testing.assert_close(result, self.chunks)
        self.assertEqual(len(self.policy._queues[ACTION]), 1)
        torch.testing.assert_close(self.policy._queues[ACTION][0], pending)
        self.assertIs(self.policy._get_action_chunk.call_args.args[1], noise)


if __name__ == "__main__":
    unittest.main()
