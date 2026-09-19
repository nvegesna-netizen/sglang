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


class RetireKVInheritanceError(RuntimeError):
    pass


def prepare_abort(req, message, status_code=None):
    req.finished_reason = FINISH_ABORT(message, status_code)


class _PrefixIndices(list):
    def tolist(self):
        return list(self)


class _FakeReq:
    def __init__(self, rid: str, authority: RetireAuthorityTag):
        self.rid = rid
        self.retire_authority = authority
        self.to_finish = None
        self.finished_reason = None
        self.return_logprob = False
        self.cache_salt = "shared-salt"
        self.prefix_indices = _PrefixIndices([30, 31, 32, 33])
        self.full_untruncated_fill_ids = [10, 11, 12, 13, 99]
        self.last_node = object()
        self.lock_receipt = object()
        self.time_stats = SimpleNamespace(trace_ctx=SimpleNamespace(abort=MagicMock()))

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
    _retire_filter_invalid_resume_requests = _load_function(
        SCHEDULER_PATH,
        "_retire_filter_invalid_resume_requests",
        {
            "FINISH_ABORT": FINISH_ABORT,
            "HTTPStatus": HTTPStatus,
            "RetireAuthorityError": RetireAuthorityError,
            "RetireKVInheritanceError": RetireKVInheritanceError,
            "logger": logging.getLogger(__name__),
            "prepare_abort": prepare_abort,
        },
    )
    retire_prepare_resume = _load_function(
        SCHEDULER_PATH,
        "retire_prepare_resume",
        {},
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


class _LaunchRegistry:
    def __init__(self, errors=None):
        self.errors = errors or {}
        self.calls = []

    def validate_launch(self, **kwargs):
        self.calls.append(kwargs)
        error = self.errors.get(kwargs["successor_request_id"])
        if error is not None:
            raise error
        return object(), object()


def _launch_scheduler(reqs, *, errors=None, remote_result=None):
    scheduler = _SchedulerHarness()
    scheduler.retire_authority = _authority_with_old_and_current()
    scheduler.retire_kv_inheritance = _LaunchRegistry(errors)
    scheduler._retire_prepared_successors = {req.rid for req in reqs}
    scheduler.tree_cache = MagicMock()
    scheduler._release_aborted_request = MagicMock()
    scheduler.beam_coordinator = MagicMock()
    scheduler.output_streamer = MagicMock()
    scheduler.world_group = MagicMock()

    def gather(local):
        if remote_result is None:
            return [local]
        return [local, remote_result(local)]

    scheduler.world_group.all_gather_object.side_effect = gather
    return scheduler


class TestRetireResumeAdmission(unittest.TestCase):
    def test_successful_prepare_marks_successor_for_collective_admission(self):
        scheduler = _SchedulerHarness()
        scheduler._retire_prepared_successors = set()
        reservation = SimpleNamespace(
            slot_digest="slots",
            source_slot_digest="source-slots",
            source_token_digest="source-tokens",
        )
        scheduler.retire_kv_inheritance = MagicMock()
        scheduler.retire_kv_inheritance.reserve.return_value = reservation
        scheduler.world_group = MagicMock()
        scheduler.world_group.all_gather_object.side_effect = lambda local: [local]
        scheduler.ps = SimpleNamespace(pp_rank=0, tp_rank=0)

        receipts = scheduler.retire_prepare_resume(
            tenant_id="tenant",
            scope_id="scope",
            retired_epoch=0,
            source_request_id="source",
            successor_request_id="successor",
            new_epoch=1,
            generation=1,
            reuse_tokens=4,
            block_size=4,
            cache_salt="salt",
            successor_token_ids=[1, 2, 3, 4],
            nonce="nonce",
        )

        self.assertTrue(receipts[0]["prepared"])
        self.assertEqual(scheduler._retire_prepared_successors, {"successor"})

    def test_failed_prepare_does_not_mark_successor(self):
        scheduler = _SchedulerHarness()
        scheduler._retire_prepared_successors = set()
        scheduler.retire_kv_inheritance = MagicMock()
        scheduler.retire_kv_inheritance.reserve.side_effect = RetireKVInheritanceError(
            "reserve failed"
        )
        scheduler.world_group = MagicMock()
        scheduler.world_group.all_gather_object.side_effect = lambda local: [local]
        scheduler.ps = SimpleNamespace(pp_rank=0, tp_rank=0)

        receipts = scheduler.retire_prepare_resume(
            tenant_id="tenant",
            scope_id="scope",
            retired_epoch=0,
            source_request_id="source",
            successor_request_id="successor",
            new_epoch=1,
            generation=1,
            reuse_tokens=4,
            block_size=4,
            cache_salt="salt",
            successor_token_ids=[1, 2, 3, 4],
            nonce="nonce",
        )

        self.assertFalse(receipts[0]["prepared"])
        self.assertEqual(scheduler._retire_prepared_successors, set())
        scheduler.retire_kv_inheritance.cancel_reservation.assert_called_once_with(
            "successor"
        )

    def test_invalid_successor_is_rejected_without_removing_valid_peer(self):
        invalid = _FakeReq("invalid", _tag(1, 1))
        valid = _FakeReq("valid", _tag(1, 1))
        scheduler = _launch_scheduler(
            [invalid, valid],
            errors={
                "invalid": RetireKVInheritanceError(
                    "successor physical slots differ from reservation"
                )
            },
        )
        invalid_last_node = invalid.last_node

        accepted = scheduler._retire_filter_invalid_resume_requests([invalid, valid])

        self.assertEqual(accepted, [valid])
        self.assertTrue(invalid.finished())
        self.assertFalse(valid.finished())
        scheduler.tree_cache.dec_lock_ref.assert_called_once_with(
            invalid_last_node, invalid.lock_receipt
        )
        self.assertIsNone(invalid.last_node)
        scheduler._release_aborted_request.assert_called_once_with("invalid")
        scheduler.beam_coordinator.retire_group.assert_called_once_with(invalid)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [invalid], False
        )
        invalid.time_stats.trace_ctx.abort.assert_called_once()

    def test_all_invalid_successors_leave_no_launchable_request(self):
        invalid = _FakeReq("invalid", _tag(1, 1))
        scheduler = _launch_scheduler(
            [invalid],
            errors={"invalid": RetireKVInheritanceError("injected mismatch")},
        )

        accepted = scheduler._retire_filter_invalid_resume_requests([invalid])

        self.assertEqual(accepted, [])
        scheduler.output_streamer.stream_output.assert_called_once()

    def test_remote_rank_failure_rejects_successor_on_local_rank(self):
        req = _FakeReq("successor", _tag(1, 1))

        def remote(local):
            copied = {key: dict(value) for key, value in local.items()}
            copied["successor"]["error"] = (
                "RetireKVInheritanceError: remote physical-slot mismatch"
            )
            return copied

        scheduler = _launch_scheduler([req], remote_result=remote)

        accepted = scheduler._retire_filter_invalid_resume_requests([req])

        self.assertEqual(accepted, [])
        self.assertIn("rank 1", req.finished_reason.message)

    def test_unprepared_cold_successor_bypasses_inheritance_collective(self):
        req = _FakeReq("cold", _tag(1, 1))
        scheduler = _launch_scheduler([req])
        scheduler._retire_prepared_successors.clear()
        scheduler.retire_kv_inheritance.validate_launch = MagicMock(return_value=None)

        accepted = scheduler._retire_filter_invalid_resume_requests([req])

        self.assertEqual(accepted, [req])
        scheduler.retire_kv_inheritance.validate_launch.assert_not_called()
        scheduler.world_group.all_gather_object.assert_not_called()
        scheduler.output_streamer.stream_output.assert_not_called()

    def test_prepared_reservation_disagreement_rejects_successor(self):
        req = _FakeReq("successor", _tag(1, 1))

        def remote(local):
            copied = {key: dict(value) for key, value in local.items()}
            copied["successor"]["prepared"] = False
            return copied

        scheduler = _launch_scheduler([req], remote_result=remote)

        accepted = scheduler._retire_filter_invalid_resume_requests([req])

        self.assertEqual(accepted, [])
        self.assertIn("disagreed", req.finished_reason.message)

    def test_participant_request_set_disagreement_rejects_successor(self):
        req = _FakeReq("successor", _tag(1, 1))

        def remote(_local):
            return {}

        scheduler = _launch_scheduler([req], remote_result=remote)

        accepted = scheduler._retire_filter_invalid_resume_requests([req])

        self.assertEqual(accepted, [])
        self.assertIn("validation set disagreed", req.finished_reason.message)

    def test_prefill_rejection_precedes_batch_materialization(self):
        source = SCHEDULER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        prefill = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_get_new_batch_prefill_raw"
        )
        segment = ast.get_source_segment(source, prefill)
        assert segment is not None
        self.assertLess(
            segment.index("_retire_filter_invalid_resume_requests"),
            segment.index("ScheduleBatch.init_new"),
        )
        self.assertLess(
            segment.index("_retire_filter_invalid_resume_requests"),
            segment.index("new_batch.prepare_for_extend()"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
