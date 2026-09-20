from __future__ import annotations

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ModuleNotFoundError:

    def register_cpu_ci(**_kwargs):
        return None


register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = ROOT / "python/sglang/srt/retire/worker_authority.py"
MODEL_RUNNER_PATH = ROOT / "python/sglang/srt/model_executor/model_runner.py"
QWEN2_PATH = ROOT / "python/sglang/srt/models/qwen2.py"
SCHEDULER_PATH = ROOT / "python/sglang/srt/managers/scheduler.py"

SPEC = importlib.util.spec_from_file_location(
    "sglang_retire_worker_authority", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
WORKER_AUTHORITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WORKER_AUTHORITY
SPEC.loader.exec_module(WORKER_AUTHORITY)


class _FakeState:
    def __init__(self, *, requires_device_predicates=False):
        self.bound = None
        self.finished = False
        self.step_requires_device_predicates = requires_device_predicates
        self.step_authorities = ()
        self.waited = None

    def bind_step_authority(
        self,
        request_ids,
        request_tags,
        scheduled_token_count=0,
    ):
        self.bound = (request_ids, request_tags, scheduled_token_count)
        self.step_authorities = tuple(sorted(set(request_tags.values())))

    def wait_for_versions(self, authorities):
        self.waited = authorities

    def finish_step(self):
        self.finished = True


def _fake_retire_modules(state):
    package = types.ModuleType("retire_serving")
    package.__path__ = []
    mailbox = types.ModuleType("retire_serving.epoch_mailbox")
    mailbox.get_worker_epoch_state = lambda: state
    return {
        "retire_serving": package,
        "retire_serving.epoch_mailbox": mailbox,
    }


class TestRetireWorkerAuthority(unittest.TestCase):
    def test_bind_carries_exact_request_coordinates_and_token_count(self):
        state = _FakeState()
        batch = SimpleNamespace(
            rids=["a", "b"],
            retire_worker_authorities=[(2, 7), (3, 9)],
            global_num_token_non_padded_cpu=8192,
            input_ids=None,
        )
        with patch.dict(sys.modules, _fake_retire_modules(state)):
            observed = WORKER_AUTHORITY.bind_forward_authority(batch)
            WORKER_AUTHORITY.finish_forward_authority(observed)

        self.assertIs(observed, state)
        self.assertEqual(
            state.bound,
            (["a", "b"], {"a": (2, 7), "b": (3, 9)}, 8192),
        )
        self.assertTrue(state.finished)

    def test_guarded_bind_establishes_device_mirror_before_forward(self):
        state = _FakeState(requires_device_predicates=True)
        batch = SimpleNamespace(
            rids=["a", "b"],
            retire_worker_authorities=[(2, 7), (3, 9)],
            global_num_token_non_padded_cpu=8192,
            input_ids=None,
        )
        with patch.dict(sys.modules, _fake_retire_modules(state)):
            WORKER_AUTHORITY.bind_forward_authority(batch)

        self.assertEqual(state.waited, [(2, 7), (3, 9)])

    def test_bind_rejects_authority_without_initialized_relay(self):
        batch = SimpleNamespace(
            rids=["a"],
            retire_worker_authorities=[(2, 7)],
            global_num_token_non_padded_cpu=1,
            input_ids=None,
        )
        with (
            patch.dict(sys.modules, _fake_retire_modules(None)),
            self.assertRaisesRegex(RuntimeError, "without an initialized"),
        ):
            WORKER_AUTHORITY.bind_forward_authority(batch)

    def test_model_runner_binds_before_forward_and_disarms_in_finally(self):
        tree = ast.parse(MODEL_RUNNER_PATH.read_text(encoding="utf-8"))
        model_runner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
        )
        forward = next(
            node
            for node in model_runner.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        calls = [
            node.func.id
            for node in ast.walk(forward)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        self.assertIn("bind_forward_authority", calls)
        self.assertIn("finish_forward_authority", calls)
        self.assertTrue(any(isinstance(node, ast.Try) for node in ast.walk(forward)))

        raw = next(
            node
            for node in model_runner.body
            if isinstance(node, ast.FunctionDef) and node.name == "_forward_raw"
        )
        graph_guards = [
            node
            for node in ast.walk(raw)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "forward_requires_layer_safe_points"
        ]
        self.assertEqual(len(graph_guards), 3)

        split_guard = next(
            node
            for node in ast.walk(raw)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "RuntimeError"
            and "split prefill" in ast.unparse(node.exc)
        )
        self.assertIsNotNone(split_guard)

    def test_qwen_layer_loop_has_bounded_safe_point_before_layer_launch(self):
        tree = ast.parse(QWEN2_PATH.read_text(encoding="utf-8"))
        model = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Qwen2Model"
        )
        forward = next(
            node
            for node in model.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        loop = next(node for node in ast.walk(forward) if isinstance(node, ast.For))
        break_nodes = [node for node in ast.walk(loop) if isinstance(node, ast.Break)]
        modulo_nodes = [node for node in ast.walk(loop) if isinstance(node, ast.Mod)]
        plan_calls = [
            node
            for node in ast.walk(forward)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "layer_safe_point_plan"
        ]
        self.assertEqual(len(plan_calls), 1)
        self.assertEqual(len(break_nodes), 1)
        self.assertGreaterEqual(len(modulo_nodes), 1)

    def test_scheduler_attests_mailbox_before_host_advance_and_writer_drain(self):
        tree = ast.parse(SCHEDULER_PATH.read_text(encoding="utf-8"))
        scheduler = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
        )
        advance = next(
            node
            for node in scheduler.body
            if isinstance(node, ast.FunctionDef) and node.name == "retire_advance"
        )
        calls = [node for node in ast.walk(advance) if isinstance(node, ast.Call)]
        mailbox_line = min(
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Name) and node.func.id == "host_check_version"
        )
        advance_line = min(
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Attribute) and node.func.attr == "advance"
        )
        synchronize_line = min(
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Attribute) and node.func.attr == "synchronize"
        )
        self.assertLess(mailbox_line, advance_line)
        self.assertLess(advance_line, synchronize_line)


if __name__ == "__main__":
    unittest.main()
