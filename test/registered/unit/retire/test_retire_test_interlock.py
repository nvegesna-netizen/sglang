from __future__ import annotations

import asyncio
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.srt.retire.authority import RetireAuthorityTag
    from sglang.srt.retire.test_interlock import (
        RetireTestInterlock,
        RetireTestInterlockError,
    )
except ModuleNotFoundError:
    # Keep this dependency-free test runnable before the SGLang image exists.
    def register_cpu_ci(**_kwargs):
        return None

    for module_name in tuple(sys.modules):
        if module_name == "sglang" or module_name.startswith("sglang."):
            sys.modules.pop(module_name, None)
    package_paths = {
        "sglang": Path(__file__).resolve().parents[4] / "python/sglang",
        "sglang.srt": Path(__file__).resolve().parents[4] / "python/sglang/srt",
        "sglang.srt.retire": (
            Path(__file__).resolve().parents[4] / "python/sglang/srt/retire"
        ),
    }
    for module_name, package_path in package_paths.items():
        package = types.ModuleType(module_name)
        package.__path__ = [str(package_path)]
        sys.modules[module_name] = package

    authority_path = package_paths["sglang.srt.retire"] / "authority.py"
    authority_spec = importlib.util.spec_from_file_location(
        "sglang.srt.retire.authority", authority_path
    )
    assert authority_spec is not None and authority_spec.loader is not None
    authority_module = importlib.util.module_from_spec(authority_spec)
    sys.modules[authority_spec.name] = authority_module
    authority_spec.loader.exec_module(authority_module)

    interlock_path = package_paths["sglang.srt.retire"] / "test_interlock.py"
    interlock_spec = importlib.util.spec_from_file_location(
        "sglang.srt.retire.test_interlock", interlock_path
    )
    assert interlock_spec is not None and interlock_spec.loader is not None
    interlock_module = importlib.util.module_from_spec(interlock_spec)
    sys.modules[interlock_spec.name] = interlock_module
    interlock_spec.loader.exec_module(interlock_module)

    RetireAuthorityTag = authority_module.RetireAuthorityTag
    RetireTestInterlock = interlock_module.RetireTestInterlock
    RetireTestInterlockError = interlock_module.RetireTestInterlockError


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def authority() -> RetireAuthorityTag:
    return RetireAuthorityTag(
        tenant_id="tenant",
        scope_id="scope",
        epoch=0,
        generation=1,
    )


class TestRetireTestInterlock(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def write_arm(
        self,
        *,
        component: str = "tokenizer",
        point: str = "tokenizer_before_client_publication",
        phase: str = "any",
        occurrence: int = 1,
    ) -> None:
        payload = {
            "artifact": "retire_test_interlock_arm",
            "schema_version": 1,
            "nonce": "nonce-1",
            "component": component,
            "point": point,
            "request_id": "request-1",
            "authority": authority().to_dict(),
            "phase": phase,
            "occurrence": occurrence,
            "timeout_s": 1,
        }
        (self.root / "arm.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def environment(self) -> dict[str, str]:
        return {
            "SGLANG_RETIRE_TEST_FAULTS": "1",
            "SGLANG_RETIRE_TEST_INTERLOCK_DIR": str(self.root),
        }

    async def wait_for_path(self, path: Path) -> None:
        async def wait() -> None:
            while not path.is_file():
                await asyncio.sleep(0)

        await asyncio.wait_for(wait(), timeout=1)

    async def test_pause_emits_reached_and_completed_receipts(self):
        self.write_arm()
        with patch.dict(os.environ, self.environment(), clear=False):
            interlock = RetireTestInterlock.from_environment("tokenizer")
            assert interlock is not None
            task = asyncio.create_task(
                interlock.pause_async(
                    point="tokenizer_before_client_publication",
                    request_id="request-1",
                    authority=authority(),
                )
            )
            await self.wait_for_path(self.root / "reached.json")
            reached = json.loads(
                (self.root / "reached.json").read_text(encoding="utf-8")
            )
            self.assertTrue(reached["injection_applied"])
            self.assertEqual(reached["authority"], authority().to_dict())

            release = {
                "artifact": "retire_test_interlock_release",
                "schema_version": 1,
                "nonce": "nonce-1",
                "component": "tokenizer",
                "point": "tokenizer_before_client_publication",
                "request_id": "request-1",
                "advance_receipt_sha256": "a" * 64,
                "injection_applied": True,
            }
            (self.root / "release.json").write_text(
                json.dumps(release, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.assertTrue(await task)

        completed = json.loads(
            (self.root / "completed.json").read_text(encoding="utf-8")
        )
        self.assertEqual(completed["advance_receipt_sha256"], "a" * 64)
        self.assertTrue(completed["injection_applied"])

    async def test_nonmatching_arm_does_not_emit_a_marker(self):
        self.write_arm(point="tokenizer_after_bind_before_scheduler_admission")
        with patch.dict(os.environ, self.environment(), clear=False):
            interlock = RetireTestInterlock.from_environment("tokenizer")
            assert interlock is not None
            paused = await interlock.pause_async(
                point="tokenizer_before_client_publication",
                request_id="request-1",
                authority=authority(),
            )
        self.assertFalse(paused)
        self.assertFalse((self.root / "reached.json").exists())

    def test_occurrence_selects_the_second_matching_boundary(self):
        self.write_arm(
            component="scheduler",
            point="scheduler_before_result_commit",
            occurrence=2,
        )
        with patch.dict(os.environ, self.environment(), clear=False):
            interlock = RetireTestInterlock.from_environment("scheduler")
            assert interlock is not None
            self.assertFalse(
                interlock.reach(
                    point="scheduler_before_result_commit",
                    request_id="request-1",
                    authority=authority(),
                )
            )
            self.assertFalse((self.root / "reached.json").exists())
            self.assertTrue(
                interlock.reach(
                    point="scheduler_before_result_commit",
                    request_id="request-1",
                    authority=authority(),
                )
            )
            reached = json.loads(
                (self.root / "reached.json").read_text(encoding="utf-8")
            )
            self.assertEqual(reached["occurrence"], 2)

    def test_result_phase_selects_decode_without_counting_prefill(self):
        self.write_arm(
            component="scheduler",
            point="scheduler_before_result_commit",
            phase="decode",
        )
        with patch.dict(os.environ, self.environment(), clear=False):
            interlock = RetireTestInterlock.from_environment("scheduler")
            assert interlock is not None
            self.assertFalse(
                interlock.reach(
                    point="scheduler_before_result_commit",
                    request_id="request-1",
                    authority=authority(),
                    phase="prefill",
                )
            )
            self.assertFalse((self.root / "reached.json").exists())
            self.assertTrue(
                interlock.reach(
                    point="scheduler_before_result_commit",
                    request_id="request-1",
                    authority=authority(),
                    phase="decode",
                )
            )
            reached = json.loads(
                (self.root / "reached.json").read_text(encoding="utf-8")
            )
            self.assertEqual(reached["phase"], "decode")

    def test_synchronous_reach_and_poll_are_nonblocking_and_idempotent(self):
        self.write_arm(component="scheduler", point="scheduler_before_result_commit")
        with patch.dict(os.environ, self.environment(), clear=False):
            interlock = RetireTestInterlock.from_environment("scheduler")
            assert interlock is not None
            reached = interlock.reach(
                point="scheduler_before_result_commit",
                request_id="request-1",
                authority=authority(),
            )
            self.assertTrue(reached)
            self.assertTrue((self.root / "reached.json").is_file())
            self.assertFalse(interlock.poll_release())

            # A cooperative caller may encounter the same deferred item more than
            # once. Reaching it again must not overwrite its immutable receipt.
            reached_bytes = (self.root / "reached.json").read_bytes()
            self.assertTrue(
                interlock.reach(
                    point="scheduler_before_result_commit",
                    request_id="request-1",
                    authority=authority(),
                )
            )
            self.assertEqual((self.root / "reached.json").read_bytes(), reached_bytes)

            release = {
                "artifact": "retire_test_interlock_release",
                "schema_version": 1,
                "nonce": "nonce-1",
                "component": "scheduler",
                "point": "scheduler_before_result_commit",
                "request_id": "request-1",
                "advance_receipt_sha256": "b" * 64,
                "injection_applied": True,
            }
            (self.root / "release.json").write_text(
                json.dumps(release, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.assertFalse(
                interlock.poll_release_after_authority_advance(
                    authority_is_current=True
                )
            )
            self.assertTrue(
                interlock.poll_release_after_authority_advance(
                    authority_is_current=False
                )
            )
            self.assertTrue(interlock.poll_release())

        completed = json.loads(
            (self.root / "completed.json").read_text(encoding="utf-8")
        )
        self.assertEqual(completed["advance_receipt_sha256"], "b" * 64)
        self.assertEqual(completed["authority"], authority().to_dict())

    def test_interlock_directory_requires_explicit_fault_gate(self):
        self.write_arm()
        environment = {"SGLANG_RETIRE_TEST_INTERLOCK_DIR": str(self.root)}
        with (
            patch.dict(os.environ, environment, clear=True),
            self.assertRaisesRegex(
                RetireTestInterlockError, "SGLANG_RETIRE_TEST_FAULTS=1"
            ),
        ):
            RetireTestInterlock.from_environment("tokenizer")

    def test_outcome_binds_completed_receipt_and_requires_stale_authority(self):
        self.write_arm(component="scheduler", point="scheduler_before_result_commit")
        with patch.dict(os.environ, self.environment(), clear=False):
            interlock = RetireTestInterlock.from_environment("scheduler")
            assert interlock is not None
            self.assertTrue(
                interlock.reach(
                    point="scheduler_before_result_commit",
                    request_id="request-1",
                    authority=authority(),
                )
            )
            release = {
                "artifact": "retire_test_interlock_release",
                "schema_version": 1,
                "nonce": "nonce-1",
                "component": "scheduler",
                "point": "scheduler_before_result_commit",
                "request_id": "request-1",
                "advance_receipt_sha256": "c" * 64,
                "injection_applied": True,
            }
            (self.root / "release.json").write_text(
                json.dumps(release, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.assertTrue(interlock.poll_release())
            with self.assertRaisesRegex(
                RetireTestInterlockError, "while authority is current"
            ):
                interlock.record_outcome(
                    "stale_result_reclaimed_without_token_or_cache_commit",
                    authority_is_current=True,
                )
            interlock.record_outcome(
                "stale_result_reclaimed_without_token_or_cache_commit",
                authority_is_current=False,
            )

        completed_sha256 = hashlib.sha256(
            (self.root / "completed.json").read_bytes()
        ).hexdigest()
        outcome = json.loads((self.root / "outcome.json").read_text(encoding="utf-8"))
        self.assertEqual(outcome["completed_sha256"], completed_sha256)
        self.assertEqual(
            outcome["outcome"],
            "stale_result_reclaimed_without_token_or_cache_commit",
        )
        self.assertFalse(outcome["local_authority_current"])
        self.assertTrue(outcome["injection_applied"])

    async def test_release_requires_exact_identity_and_advance_hash(self):
        self.write_arm()
        with patch.dict(os.environ, self.environment(), clear=False):
            interlock = RetireTestInterlock.from_environment("tokenizer")
            assert interlock is not None
            task = asyncio.create_task(
                interlock.pause_async(
                    point="tokenizer_before_client_publication",
                    request_id="request-1",
                    authority=authority(),
                )
            )
            await self.wait_for_path(self.root / "reached.json")
            release = {
                "artifact": "retire_test_interlock_release",
                "schema_version": 1,
                "nonce": "nonce-1",
                "component": "tokenizer",
                "point": "tokenizer_before_client_publication",
                "request_id": "request-1",
                "advance_receipt_sha256": "not-a-hash",
                "injection_applied": True,
            }
            (self.root / "release.json").write_text(
                json.dumps(release, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RetireTestInterlockError, "lowercase SHA-256"):
                await task

    def test_tokenizer_interlocks_are_at_the_named_pipeline_boundaries(self):
        tokenizer_path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/managers/tokenizer_manager.py"
        )
        tree = ast.parse(tokenizer_path.read_text(encoding="utf-8"))
        generate = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "generate_request"
        )
        stream = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_stream_one_response"
        )

        def call_lines(node, method):
            return [
                item.lineno
                for item in ast.walk(node)
                if isinstance(item, ast.Call)
                and isinstance(item.func, ast.Attribute)
                and item.func.attr == method
            ]

        def constant_line(node, value):
            return next(
                item.lineno
                for item in ast.walk(node)
                if isinstance(item, ast.Constant) and item.value == value
            )

        init_line = call_lines(generate, "_init_req_state")[0]
        bind_pause_line = constant_line(
            generate, "tokenizer_after_bind_before_scheduler_admission"
        )
        scheduler_send_line = call_lines(generate, "_send_one_request")[0]
        self.assertLess(init_line, bind_pause_line)
        self.assertLess(bind_pause_line, scheduler_send_line)

        publication_pause_line = constant_line(
            stream, "tokenizer_before_client_publication"
        )
        publication_gate_line = call_lines(
            stream, "_retire_require_client_publication"
        )[0]
        output_yield_lines = [
            item.lineno
            for item in ast.walk(stream)
            if isinstance(item, ast.Yield)
            and isinstance(item.value, ast.Name)
            and item.value.id in {"out", "abort_out"}
        ]
        self.assertLess(publication_pause_line, publication_gate_line)
        self.assertTrue(output_yield_lines)
        self.assertLess(publication_gate_line, min(output_yield_lines))

    def test_scheduler_result_interlock_preserves_authority_ingestion_order(self):
        scheduler_path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/managers/scheduler.py"
        )
        tree = ast.parse(scheduler_path.read_text(encoding="utf-8"))
        event_loop = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "event_loop_normal"
        )

        def call_lines(method):
            return [
                item.lineno
                for item in ast.walk(event_loop)
                if isinstance(item, ast.Call)
                and isinstance(item.func, ast.Attribute)
                and item.func.attr == method
            ]

        ingest_line = call_lines("ingest_requests")[0]
        admission_poll_line = call_lines("_retire_poll_deferred_admission")[0]
        poll_line = call_lines("_retire_poll_deferred_result")[0]
        output_poll_line = call_lines("retire_poll_deferred_output")[0]
        run_line = call_lines("run_batch")[0]
        defer_line = call_lines("_retire_defer_before_result_commit")[0]
        result_line = call_lines("process_batch_result")[0]
        self.assertLess(ingest_line, admission_poll_line)
        self.assertLess(admission_poll_line, poll_line)
        self.assertLess(poll_line, output_poll_line)
        self.assertLess(run_line, defer_line)
        self.assertLess(defer_line, result_line)

        result_poll = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_retire_poll_deferred_result"
        )
        result_process_line = next(
            item.lineno
            for item in ast.walk(result_poll)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "process_batch_result"
        )
        result_outcome_line = next(
            item.lineno
            for item in ast.walk(result_poll)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "record_outcome"
        )
        self.assertLess(result_process_line, result_outcome_line)

        result_defer = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_retire_defer_before_result_commit"
        )
        result_defer_calls = {
            item.func.attr
            for item in ast.walk(result_defer)
            if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
        }
        result_defer_constants = {
            item.value
            for item in ast.walk(result_defer)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        self.assertIn("is_extend", result_defer_calls)
        self.assertIn("is_decode", result_defer_calls)
        self.assertIn("prefill", result_defer_constants)
        self.assertIn("decode", result_defer_constants)

        init_method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "init_retire_test_interlock"
        )
        init_constants = {
            item.value
            for item in ast.walk(init_method)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        self.assertIn("scheduler_before_result_commit", init_constants)
        self.assertIn("scheduler_after_admission_before_model_launch", init_constants)
        self.assertTrue(
            any("overlap scheduling disabled" in value for value in init_constants)
        )
        self.assertTrue(
            any("single scheduler rank" in value for value in init_constants)
        )

    def test_detokenizer_interlock_polls_without_blocking_authority_input(self):
        detokenizer_path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/managers/detokenizer_manager.py"
        )
        tree = ast.parse(detokenizer_path.read_text(encoding="utf-8"))
        event_loop = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "event_loop"
        )

        def call_lines(method):
            return [
                item.lineno
                for item in ast.walk(event_loop)
                if isinstance(item, ast.Call)
                and isinstance(item.func, ast.Attribute)
                and item.func.attr == method
            ]

        poll_deferred_line = call_lines("_retire_poll_deferred_output")[0]
        socket_poll_line = call_lines("poll")[0]
        socket_recv_line = next(
            item.lineno
            for item in ast.walk(event_loop)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id == "sock_recv"
        )
        defer_line = call_lines("_retire_defer_after_validation")[0]
        dispatch_line = call_lines("_request_dispatcher")[0]
        self.assertLess(poll_deferred_line, socket_poll_line)
        self.assertLess(socket_poll_line, socket_recv_line)
        self.assertLess(socket_recv_line, defer_line)
        self.assertLess(defer_line, dispatch_line)

        defer_method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_retire_defer_after_validation"
        )
        validation_line = next(
            item.lineno
            for item in ast.walk(defer_method)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "_retire_validate_batch_token_id_output"
        )
        reach_line = next(
            item.lineno
            for item in ast.walk(defer_method)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "reach"
        )
        self.assertLess(validation_line, reach_line)
        point_constants = {
            item.value
            for item in ast.walk(defer_method)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        self.assertIn(
            "detokenizer_after_validation_before_publication", point_constants
        )

        poll_method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_retire_poll_deferred_output"
        )
        poll_calls = {
            item.func.attr
            for item in ast.walk(poll_method)
            if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
        }
        self.assertIn("poll_release_after_authority_advance", poll_calls)
        self.assertIn("is_current", poll_calls)
        self.assertIn("record_outcome", poll_calls)

    def test_scheduler_admission_interlock_follows_waiting_queue_append(self):
        scheduler_path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/managers/scheduler.py"
        )
        tree = ast.parse(scheduler_path.read_text(encoding="utf-8"))
        add_request = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_add_request_to_queue"
        )
        append_line = next(
            item.lineno
            for item in ast.walk(add_request)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "append"
            and isinstance(item.func.value, ast.Attribute)
            and item.func.value.attr == "waiting_queue"
        )
        interlock_line = next(
            item.lineno
            for item in ast.walk(add_request)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "_retire_defer_after_admission"
        )
        self.assertLess(append_line, interlock_line)

        poll_admission = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_retire_poll_deferred_admission"
        )
        poll_calls = {
            item.func.attr
            for item in ast.walk(poll_admission)
            if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
        }
        self.assertIn("poll_release_after_authority_advance", poll_calls)
        self.assertIn("record_outcome", poll_calls)

    def test_scheduler_output_interlock_precedes_payload_send(self):
        output_path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/managers/scheduler_components/output_streamer.py"
        )
        tree = ast.parse(output_path.read_text(encoding="utf-8"))
        stream = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_stream_output_generation"
        )

        def call_line(method):
            return next(
                item.lineno
                for item in ast.walk(stream)
                if isinstance(item, ast.Call)
                and isinstance(item.func, ast.Attribute)
                and item.func.attr == method
            )

        payload_line = call_line("to_payload")
        defer_line = call_line("_retire_defer_before_output")
        send_line = call_line("_send_generation_payload")
        self.assertLess(payload_line, defer_line)
        self.assertLess(defer_line, send_line)

        poll_method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "retire_poll_deferred_output"
        )
        constants = {
            item.value
            for item in ast.walk(poll_method)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        calls = {
            item.func.attr
            for item in ast.walk(poll_method)
            if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
        }
        self.assertIn("poll_release_after_authority_advance", calls)
        self.assertIn("_send_generation_payload", calls)
        self.assertIn("record_outcome", calls)
        self.assertIn("type", constants)
        self.assertIn("abort", constants)

    def test_tokenizer_outcomes_precede_admission_and_client_yield(self):
        tokenizer_path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/managers/tokenizer_manager.py"
        )
        tree = ast.parse(tokenizer_path.read_text(encoding="utf-8"))
        generate = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "generate_request"
        )
        stream = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_stream_one_response"
        )

        admission_outcome = next(
            item.lineno
            for item in ast.walk(generate)
            if isinstance(item, ast.Constant)
            and item.value == "stale_request_rejected_before_scheduler_admission"
        )
        scheduler_send = next(
            item.lineno
            for item in ast.walk(generate)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "_send_one_request"
        )
        publication_outcome = next(
            item.lineno
            for item in ast.walk(stream)
            if isinstance(item, ast.Constant)
            and item.value == "stale_frontend_output_rejected_before_client_yield"
        )
        output_yields = [
            item.lineno
            for item in ast.walk(stream)
            if isinstance(item, ast.Yield)
            and isinstance(item.value, ast.Name)
            and item.value.id in {"out", "abort_out"}
        ]
        self.assertLess(admission_outcome, scheduler_send)
        self.assertTrue(output_yields)
        self.assertLess(publication_outcome, min(output_yields))


if __name__ == "__main__":
    unittest.main()
