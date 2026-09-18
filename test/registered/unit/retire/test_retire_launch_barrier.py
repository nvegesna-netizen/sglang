from __future__ import annotations

import __future__
import ast
from collections import deque
from http import HTTPStatus
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import MagicMock

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ModuleNotFoundError:
    # Keep this source-level test runnable before the SGLang image exists.
    def register_cpu_ci(**_kwargs):
        return None


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


ROOT = Path(__file__).resolve().parents[4]
SCHEDULER_PATH = ROOT / "python/sglang/srt/managers/scheduler.py"
PROCESSOR_PATH = (
    ROOT / "python/sglang/srt/managers/scheduler_components/batch_result_processor.py"
)
AUTHORITY_PATH = ROOT / "python/sglang/srt/retire/authority.py"


def _load_function(path: Path, name: str, namespace: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(
        compile(
            module,
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[name]


AUTHORITY_SPEC = importlib.util.spec_from_file_location(
    "sglang_retire_launch_barrier_authority", AUTHORITY_PATH
)
assert AUTHORITY_SPEC is not None and AUTHORITY_SPEC.loader is not None
AUTHORITY = importlib.util.module_from_spec(AUTHORITY_SPEC)
sys.modules[AUTHORITY_SPEC.name] = AUTHORITY
AUTHORITY_SPEC.loader.exec_module(AUTHORITY)

RetireAuthorityError = AUTHORITY.RetireAuthorityError
RetireAuthorityTable = AUTHORITY.RetireAuthorityTable
RetireAuthorityTag = AUTHORITY.RetireAuthorityTag


class FINISH_ABORT:
    def __init__(self, message: str, status_code=None):
        self.message = message
        self.status_code = status_code


class BaseSpecWorker:
    pass


class _FakeReq:
    def __init__(self, rid: str, authority: RetireAuthorityTag):
        self.rid = rid
        self.retire_authority = authority
        self.to_finish = None
        self.finished_reason = None
        self.return_logprob = False

    def finished(self) -> bool:
        return self.finished_reason is not None


class _SchedulerHarness:
    _retire_resolve_launch_barrier = _load_function(
        SCHEDULER_PATH,
        "_retire_resolve_launch_barrier",
        {
            "FINISH_ABORT": FINISH_ABORT,
            "HTTPStatus": HTTPStatus,
            "RetireAuthorityError": RetireAuthorityError,
            "logger": logging.getLogger(__name__),
        },
    )

    def collect_inflight_reqs(self):
        batches = (self.running_batch, self.last_batch)
        return {req for batch in batches if batch is not None for req in batch.reqs}


class _ProcessorHarness:
    abort_before_model_launch = _load_function(
        PROCESSOR_PATH,
        "abort_before_model_launch",
        {"BaseSpecWorker": BaseSpecWorker, "FINISH_ABORT": FINISH_ABORT},
    )


def _tag(epoch: int, generation: int) -> RetireAuthorityTag:
    return RetireAuthorityTag(
        tenant_id="tenant",
        scope_id="scope",
        epoch=epoch,
        generation=generation,
    )


def _authority_with_old_and_current() -> RetireAuthorityTable:
    authority = RetireAuthorityTable()
    authority.bind("stale", _tag(0, 0))
    authority.advance(
        tenant_id="tenant",
        scope_id="scope",
        retired_epoch=0,
        new_epoch=1,
        generation=1,
    )
    authority.bind("current", _tag(1, 1))
    return authority


def _scheduler(*, overlap: bool, stale: _FakeReq, current: _FakeReq):
    scheduler = _SchedulerHarness()
    scheduler._retire_launch_barrier_pending = True
    scheduler.retire_authority = _authority_with_old_and_current()
    scheduler.enable_overlap = overlap
    scheduler.running_batch = SimpleNamespace(reqs=[stale, current])
    scheduler.last_batch = None
    scheduler.result_queue = deque()
    scheduler._pending_chunked_abort_req = None
    scheduler.batch_result_processor = MagicMock()
    scheduler._release_aborted_request = MagicMock()
    scheduler.beam_coordinator = MagicMock()
    scheduler.process_batch_result = MagicMock()
    return scheduler


class TestRetireLaunchBarrier(unittest.TestCase):
    def test_overlap_commits_queued_mixed_result_before_next_launch(self):
        stale = _FakeReq("stale", _tag(0, 0))
        current = _FakeReq("current", _tag(1, 1))
        stale.to_finish = FINISH_ABORT("RETIRE superseded")
        scheduler = _scheduler(overlap=True, stale=stale, current=current)
        last_batch = SimpleNamespace(reqs=[stale, current])
        queued_batch = SimpleNamespace(reqs=[stale, current])
        queued_result = object()
        scheduler.last_batch = last_batch
        scheduler.result_queue.append((queued_batch, queued_result))

        def commit_result(batch, result):
            self.assertIs(batch, queued_batch)
            self.assertIs(result, queued_result)
            stale.finished_reason = stale.to_finish
            stale.to_finish = None

        scheduler.process_batch_result.side_effect = commit_result

        drained = scheduler._retire_resolve_launch_barrier()

        self.assertTrue(drained)
        scheduler.process_batch_result.assert_called_once_with(
            queued_batch, queued_result
        )
        scheduler.batch_result_processor.abort_before_model_launch.assert_not_called()
        self.assertTrue(stale.finished())
        self.assertFalse(current.finished())
        self.assertIs(scheduler.last_batch, last_batch)
        self.assertEqual(len(scheduler.result_queue), 0)
        self.assertFalse(scheduler._retire_launch_barrier_pending)

    def test_nonoverlap_reclaims_only_stale_request_exactly_once(self):
        stale = _FakeReq("stale", _tag(0, 0))
        current = _FakeReq("current", _tag(1, 1))
        reason = FINISH_ABORT("RETIRE superseded")
        stale.to_finish = reason
        scheduler = _scheduler(overlap=False, stale=stale, current=current)
        scheduler.last_batch = SimpleNamespace(reqs=[stale])

        def finish(req, observed_reason):
            self.assertIs(req, stale)
            self.assertIs(observed_reason, reason)
            req.finished_reason = observed_reason
            req.to_finish = None

        scheduler.batch_result_processor.abort_before_model_launch.side_effect = finish

        drained = scheduler._retire_resolve_launch_barrier()

        self.assertFalse(drained)
        scheduler.batch_result_processor.abort_before_model_launch.assert_called_once_with(
            stale, reason
        )
        scheduler._release_aborted_request.assert_called_once_with("stale")
        scheduler.beam_coordinator.retire_group.assert_called_once_with(stale)
        self.assertTrue(stale.finished())
        self.assertFalse(current.finished())
        self.assertFalse(scheduler._retire_launch_barrier_pending)

    def test_pending_chunked_abort_remains_owned_by_chunk_cleanup(self):
        stale = _FakeReq("stale", _tag(0, 0))
        current = _FakeReq("current", _tag(1, 1))
        stale.to_finish = FINISH_ABORT("RETIRE superseded")
        scheduler = _scheduler(overlap=False, stale=stale, current=current)
        scheduler._pending_chunked_abort_req = stale

        scheduler._retire_resolve_launch_barrier()

        scheduler.batch_result_processor.abort_before_model_launch.assert_not_called()
        self.assertFalse(stale.finished())

    def test_overlap_state_mismatch_fails_closed(self):
        stale = _FakeReq("stale", _tag(0, 0))
        current = _FakeReq("current", _tag(1, 1))
        scheduler = _scheduler(overlap=True, stale=stale, current=current)
        scheduler.result_queue.append((object(), object()))

        with self.assertRaisesRegex(
            RetireAuthorityError, "overlap result without last_batch"
        ):
            scheduler._retire_resolve_launch_barrier()
        self.assertTrue(scheduler._retire_launch_barrier_pending)

    def test_overlap_event_loop_skips_both_old_result_pop_sites_after_drain(self):
        source_text = SCHEDULER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source_text)
        event_loop = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "event_loop_overlap"
        )
        source = ast.get_source_segment(source_text, event_loop)
        assert source is not None
        self.assertIn(
            "retire_overlap_result_drained = self._retire_resolve_launch_barrier()",
            source,
        )
        self.assertEqual(source.count("not retire_overlap_result_drained"), 2)

    def test_model_launch_guard_remains_independent_of_barrier(self):
        source_text = SCHEDULER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source_text)
        run_batch = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "run_batch"
        )
        calls = [
            node
            for node in ast.walk(run_batch)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        require_current = next(
            node for node in calls if node.func.attr == "require_current"
        )
        validate_resume = next(
            node for node in calls if node.func.attr == "_retire_validate_resume_launch"
        )
        self.assertLess(require_current.lineno, validate_resume.lineno)

    def test_result_commit_preserves_specific_supersede_reason(self):
        source = PROCESSOR_PATH.read_text(encoding="utf-8")
        self.assertEqual(
            source.count("if retire_revoked and req.to_finish is None:"), 2
        )


class TestAbortBeforeModelLaunch(unittest.TestCase):
    def test_cleanup_and_terminal_output_are_ordered_once(self):
        processor = _ProcessorHarness()
        events: list[str] = []
        processor.token_to_kv_pool_allocator = MagicMock()
        processor.token_to_kv_pool_allocator.free_group_begin.side_effect = lambda: (
            events.append("begin")
        )
        processor.token_to_kv_pool_allocator.free_group_end.side_effect = lambda: (
            events.append("end")
        )
        processor.output_streamer = MagicMock()
        processor.output_streamer.stream_output.side_effect = lambda *_args: (
            events.append("stream")
        )
        processor._handle_sampling_mask_abort = MagicMock(
            side_effect=lambda _req: events.append("cleanup")
        )
        processor.draft_worker = object()
        req = _FakeReq("stale", _tag(0, 0))
        reason = FINISH_ABORT("RETIRE superseded")
        req.to_finish = reason

        processor.abort_before_model_launch(req, reason)

        self.assertEqual(events, ["begin", "cleanup", "end", "stream"])
        processor._handle_sampling_mask_abort.assert_called_once_with(req)
        processor.output_streamer.stream_output.assert_called_once_with([req], False)
        self.assertIs(req.finished_reason, reason)
        self.assertIsNone(req.to_finish)

    def test_speculative_worker_is_notified_without_natural_stop(self):
        processor = _ProcessorHarness()
        processor.token_to_kv_pool_allocator = MagicMock()
        processor.output_streamer = MagicMock()
        processor._handle_sampling_mask_abort = MagicMock()
        processor.draft_worker = BaseSpecWorker()
        processor.draft_worker.note_request_finished = MagicMock()
        req = _FakeReq("stale", _tag(0, 0))
        reason = FINISH_ABORT("RETIRE superseded")

        processor.abort_before_model_launch(req, reason)

        processor.draft_worker.note_request_finished.assert_called_once_with(
            rid="stale", natural_stop=False
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
