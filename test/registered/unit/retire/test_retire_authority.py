from __future__ import annotations

import ast
import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ModuleNotFoundError:
    # Keep this dependency-free test runnable before the SGLang image exists.
    def register_cpu_ci(**_kwargs):
        return None


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


MODULE_PATH = (
    Path(__file__).resolve().parents[4] / "python/sglang/srt/retire/authority.py"
)
TOKENIZER_MANAGER_PATH = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/managers/tokenizer_manager.py"
)
SPEC = importlib.util.spec_from_file_location("sglang_retire_authority", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AUTHORITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUTHORITY
SPEC.loader.exec_module(AUTHORITY)

RetireAuthorityError = AUTHORITY.RetireAuthorityError
RetireAuthorityTable = AUTHORITY.RetireAuthorityTable
RetireAuthorityTag = AUTHORITY.RetireAuthorityTag


def tag(epoch: int, generation: int, *, tenant: str = "tenant"):
    return RetireAuthorityTag(
        tenant_id=tenant,
        scope_id="scope",
        epoch=epoch,
        generation=generation,
    )


class TestRetireAuthority(unittest.TestCase):
    def test_parse_requires_exact_valid_fields(self):
        value = tag(0, 0).to_dict()
        self.assertEqual(RetireAuthorityTag.from_value(value), tag(0, 0))
        with self.assertRaises(RetireAuthorityError):
            RetireAuthorityTag.from_value({**value, "extra": 1})
        with self.assertRaises(RetireAuthorityError):
            RetireAuthorityTag.from_value({**value, "generation": -1})
        with self.assertRaises(RetireAuthorityError):
            RetireAuthorityTag.from_value({**value, "epoch": True})

    def test_parse_carries_atomic_worker_mailbox_authority(self):
        value = {
            **tag(0, 0).to_dict(),
            "mailbox_slot": 7,
            "mailbox_version": 11,
        }
        parsed = RetireAuthorityTag.from_value(value)
        self.assertEqual(parsed.worker_authority, (7, 11))
        self.assertEqual(parsed.to_dict(), value)
        for missing in ("mailbox_slot", "mailbox_version"):
            malformed = dict(value)
            malformed.pop(missing)
            with self.subTest(missing=missing), self.assertRaises(RetireAuthorityError):
                RetireAuthorityTag.from_value(malformed)

    def test_bind_initializes_epoch_zero_and_is_idempotent(self):
        table = RetireAuthorityTable()
        table.bind("request", tag(0, 0))
        table.bind("request", tag(0, 0))
        self.assertTrue(table.is_current(tag(0, 0)))
        self.assertEqual(table.snapshot()["requests"]["request"], tag(0, 0).to_dict())

    def test_worker_mailbox_revokes_before_host_table_advance(self):
        worker_tag = RetireAuthorityTag(
            tenant_id="tenant",
            scope_id="scope",
            epoch=0,
            generation=1,
            mailbox_slot=3,
            mailbox_version=5,
        )
        table = RetireAuthorityTable()
        table.bind("request", worker_tag)

        package = types.ModuleType("retire_serving")
        package.__path__ = []
        mailbox = types.ModuleType("retire_serving.epoch_mailbox")
        observed = {"admission_safe": True}
        mailbox.host_check_version = lambda *_args: dict(observed)
        modules = {
            "retire_serving": package,
            "retire_serving.epoch_mailbox": mailbox,
        }
        with patch.dict(sys.modules, modules):
            self.assertTrue(table.is_current(worker_tag))
            observed["admission_safe"] = False
            self.assertFalse(table.is_current(worker_tag))

    def test_bind_rejects_future_epoch_without_advance(self):
        table = RetireAuthorityTable()
        with self.assertRaisesRegex(RetireAuthorityError, "only epoch zero"):
            table.bind("request", tag(1, 2))

    def test_advance_revokes_old_and_admits_exact_successor(self):
        table = RetireAuthorityTable()
        table.bind("old-b", tag(0, 1))
        table.bind("old-a", tag(0, 1))

        result = table.advance(
            tenant_id="tenant",
            scope_id="scope",
            retired_epoch=0,
            new_epoch=1,
            generation=2,
        )

        self.assertEqual(result.retired_request_ids, ("old-a", "old-b"))
        self.assertFalse(table.is_current(tag(0, 1)))
        table.bind("successor", tag(1, 2))
        self.assertTrue(table.is_current(tag(1, 2)))

    def test_advance_before_old_arrival_closes_queued_race(self):
        table = RetireAuthorityTable()
        table.advance(
            tenant_id="tenant",
            scope_id="scope",
            retired_epoch=0,
            new_epoch=1,
            generation=2,
        )
        with self.assertRaisesRegex(RetireAuthorityError, "not current"):
            table.bind("late-old", tag(0, 1))
        table.bind("successor", tag(1, 2))

    def test_advance_rejects_skips_replays_and_generation_reuse(self):
        table = RetireAuthorityTable()
        table.bind("old", tag(0, 1))
        for values in (
            dict(retired_epoch=0, new_epoch=2, generation=2),
            dict(retired_epoch=1, new_epoch=2, generation=2),
            dict(retired_epoch=0, new_epoch=1, generation=1),
        ):
            with self.subTest(values=values), self.assertRaises(RetireAuthorityError):
                table.advance(tenant_id="tenant", scope_id="scope", **values)

    def test_tenants_with_same_scope_are_independent(self):
        table = RetireAuthorityTable()
        table.bind("a", tag(0, 1, tenant="a"))
        table.bind("b", tag(0, 1, tenant="b"))
        table.advance(
            tenant_id="a",
            scope_id="scope",
            retired_epoch=0,
            new_epoch=1,
            generation=2,
        )
        self.assertFalse(table.is_current(tag(0, 1, tenant="a")))
        self.assertTrue(table.is_current(tag(0, 1, tenant="b")))

    def test_concurrent_old_bind_cannot_commit_after_advance(self):
        table = RetireAuthorityTable()
        table.bind("initial", tag(0, 1))
        start = threading.Barrier(3)
        outcomes = []

        def bind_old():
            start.wait()
            try:
                table.bind("racing-old", tag(0, 1))
                outcomes.append("bound")
            except RetireAuthorityError:
                outcomes.append("rejected")

        def advance():
            start.wait()
            table.advance(
                tenant_id="tenant",
                scope_id="scope",
                retired_epoch=0,
                new_epoch=1,
                generation=2,
            )
            outcomes.append("advanced")

        threads = [threading.Thread(target=bind_old), threading.Thread(target=advance)]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join()

        self.assertIn("advanced", outcomes)
        self.assertFalse(table.is_current(tag(0, 1)))
        with self.assertRaises(RetireAuthorityError):
            table.require_current(tag(0, 1), "pre-launch")

    def test_publication_rechecks_frame_admitted_before_advance(self):
        table = RetireAuthorityTable()
        old = tag(0, 1)
        table.bind("old", old)

        # This is the earlier component's output-admission check.
        table.require_publication(
            old,
            terminal_abort=False,
            has_payload=True,
            phase="tokenizer enqueue",
        )
        table.advance(
            tenant_id="tenant",
            scope_id="scope",
            retired_epoch=0,
            new_epoch=1,
            generation=2,
        )

        with self.assertRaisesRegex(
            RetireAuthorityError, "tokenizer-manager client publication"
        ):
            table.require_publication(
                old,
                terminal_abort=False,
                has_payload=True,
                phase="tokenizer-manager client publication",
            )

    def test_retired_publication_allows_only_empty_terminal_abort(self):
        table = RetireAuthorityTable()
        old = tag(0, 1)
        table.bind("old", old)
        table.advance(
            tenant_id="tenant",
            scope_id="scope",
            retired_epoch=0,
            new_epoch=1,
            generation=2,
        )

        table.require_publication(
            old,
            terminal_abort=True,
            has_payload=False,
            phase="tokenizer-manager client publication",
        )
        for terminal_abort, has_payload in ((False, False), (True, True)):
            with (
                self.subTest(terminal_abort=terminal_abort, has_payload=has_payload),
                self.assertRaises(RetireAuthorityError),
            ):
                table.require_publication(
                    old,
                    terminal_abort=terminal_abort,
                    has_payload=has_payload,
                    phase="tokenizer-manager client publication",
                )

    def test_tokenizer_stream_rechecks_authority_before_every_client_yield(self):
        tree = ast.parse(TOKENIZER_MANAGER_PATH.read_text(encoding="utf-8"))
        stream = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_stream_one_response"
        )
        gate_lines = [
            node.lineno
            for node in ast.walk(stream)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_retire_require_client_publication"
        ]
        output_yield_lines = [
            node.lineno
            for node in ast.walk(stream)
            if isinstance(node, ast.Yield)
            and isinstance(node.value, ast.Name)
            and node.value.id in {"out", "abort_out"}
        ]
        self.assertEqual(len(gate_lines), 1)
        self.assertTrue(output_yield_lines)
        self.assertLess(gate_lines[0], min(output_yield_lines))


if __name__ == "__main__":
    unittest.main()
