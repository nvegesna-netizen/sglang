from __future__ import annotations

import asyncio
import ast
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
    ) -> None:
        payload = {
            "artifact": "retire_test_interlock_arm",
            "schema_version": 1,
            "nonce": "nonce-1",
            "component": component,
            "point": point,
            "request_id": "request-1",
            "authority": authority().to_dict(),
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


if __name__ == "__main__":
    unittest.main()
