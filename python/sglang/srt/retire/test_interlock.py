"""Receipt-producing interlocks for deterministic RETIRE race tests.

The interlock is inert unless both ``SGLANG_RETIRE_TEST_FAULTS=1`` and an
absolute ``SGLANG_RETIRE_TEST_INTERLOCK_DIR`` are present. A test controller
places one immutable ``arm.json`` in that directory before engine launch,
waits for ``reached.json``, advances authority, and atomically publishes
``release.json``. The runtime never uses elapsed time as evidence that the
requested race boundary was reached.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

from sglang.srt.retire.authority import RetireAuthorityTag


class RetireTestInterlockError(RuntimeError):
    """Raised when a test interlock is malformed, unsafe, or times out."""


_ARM_KEYS = {
    "artifact",
    "schema_version",
    "nonce",
    "component",
    "point",
    "request_id",
    "authority",
    "phase",
    "occurrence",
    "timeout_s",
}
_RELEASE_KEYS = {
    "artifact",
    "schema_version",
    "nonce",
    "component",
    "point",
    "request_id",
    "advance_receipt_sha256",
    "injection_applied",
}
_OUTCOMES = {
    "tokenizer_after_bind_before_scheduler_admission": (
        "stale_request_rejected_before_scheduler_admission"
    ),
    "scheduler_after_admission_before_model_launch": (
        "stale_admission_dropped_before_model_launch"
    ),
    "scheduler_before_result_commit": (
        "stale_result_reclaimed_without_token_or_cache_commit"
    ),
    "scheduler_before_output": "stale_scheduler_output_dropped_before_transport",
    "detokenizer_after_validation_before_publication": (
        "stale_detokenizer_output_dropped_before_tokenizer"
    ),
    "tokenizer_before_client_publication": (
        "stale_frontend_output_rejected_before_client_yield"
    ),
}


def _read_json_object(
    path: Path, *, expected_keys: set[str]
) -> tuple[dict[str, Any], str]:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("interlock input is not a regular file")
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RetireTestInterlockError(
            f"cannot read test interlock {path.name}"
        ) from error
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise RetireTestInterlockError(
            f"test interlock {path.name} has unexpected fields"
        )
    return payload, hashlib.sha256(raw).hexdigest()


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RetireTestInterlockError(f"{field} must be a non-empty string")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise RetireTestInterlockError(f"refusing to overwrite {path.name}")
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


@dataclass(frozen=True, slots=True)
class RetireTestArm:
    nonce: str
    component: str
    point: str
    request_id: str
    authority: RetireAuthorityTag
    phase: str
    occurrence: int
    timeout_s: float

    @classmethod
    def from_path(cls, path: Path) -> RetireTestArm:
        payload, _ = _read_json_object(path, expected_keys=_ARM_KEYS)
        if payload["artifact"] != "retire_test_interlock_arm":
            raise RetireTestInterlockError("unexpected test interlock arm artifact")
        if payload["schema_version"] != 1:
            raise RetireTestInterlockError("unsupported test interlock arm schema")
        try:
            authority = RetireAuthorityTag.from_value(payload["authority"])
        except Exception as error:
            raise RetireTestInterlockError(
                "invalid test interlock authority"
            ) from error
        if authority is None:
            raise RetireTestInterlockError("test interlock authority is missing")
        phase = _nonempty_string(payload["phase"], "phase")
        if phase not in {"any", "prefill", "decode"}:
            raise RetireTestInterlockError("unsupported test interlock phase")
        if payload["point"] != "scheduler_before_result_commit" and phase != "any":
            raise RetireTestInterlockError(
                "only the scheduler result interlock accepts a selected phase"
            )
        occurrence = payload["occurrence"]
        if (
            not isinstance(occurrence, int)
            or isinstance(occurrence, bool)
            or occurrence <= 0
        ):
            raise RetireTestInterlockError("occurrence must be a positive integer")
        timeout_s = payload["timeout_s"]
        if (
            not isinstance(timeout_s, (int, float))
            or isinstance(timeout_s, bool)
            or timeout_s <= 0
        ):
            raise RetireTestInterlockError("timeout_s must be positive")
        return cls(
            nonce=_nonempty_string(payload["nonce"], "nonce"),
            component=_nonempty_string(payload["component"], "component"),
            point=_nonempty_string(payload["point"], "point"),
            request_id=_nonempty_string(payload["request_id"], "request_id"),
            authority=authority,
            phase=phase,
            occurrence=occurrence,
            timeout_s=float(timeout_s),
        )


class RetireTestInterlock:
    """One-shot test interlock loaded from an immutable per-engine directory."""

    def __init__(self, root: Path, component: str, arm: RetireTestArm) -> None:
        self.root = root
        self.component = component
        self.arm = arm
        self._reached_authority: RetireAuthorityTag | None = None
        self._deadline: float | None = None
        self._completed = False
        self._match_count = 0

    @classmethod
    def from_environment(cls, component: str) -> RetireTestInterlock | None:
        root_raw = os.environ.get("SGLANG_RETIRE_TEST_INTERLOCK_DIR")
        if not root_raw:
            return None
        if os.environ.get("SGLANG_RETIRE_TEST_FAULTS") != "1":
            raise RetireTestInterlockError(
                "RETIRE test interlock requires SGLANG_RETIRE_TEST_FAULTS=1"
            )
        root = Path(root_raw)
        if (
            not root.is_absolute()
            or root.is_symlink()
            or not root.is_dir()
            or root.resolve() != root
        ):
            raise RetireTestInterlockError(
                "SGLANG_RETIRE_TEST_INTERLOCK_DIR must be an existing absolute directory"
            )
        arm = RetireTestArm.from_path(root / "arm.json")
        if arm.component != component:
            return None
        return cls(root=root, component=component, arm=arm)

    def matches(
        self,
        *,
        point: str,
        request_id: str,
        authority: RetireAuthorityTag | None,
        phase: str = "any",
    ) -> bool:
        return (
            not self._completed
            and authority is not None
            and point == self.arm.point
            and request_id == self.arm.request_id
            and authority == self.arm.authority
            and (self.arm.phase == "any" or phase == self.arm.phase)
        )

    def reach(
        self,
        *,
        point: str,
        request_id: str,
        authority: RetireAuthorityTag | None,
        phase: str = "any",
    ) -> bool:
        if not self.matches(
            point=point,
            request_id=request_id,
            authority=authority,
            phase=phase,
        ):
            return False
        assert authority is not None
        if self._reached_authority is not None:
            if self._reached_authority != authority:
                raise RetireTestInterlockError(
                    "test interlock reached with conflicting authority"
                )
            return True
        self._match_count += 1
        if self._match_count != self.arm.occurrence:
            return False
        reached = {
            "artifact": "retire_test_interlock_reached",
            "schema_version": 1,
            "nonce": self.arm.nonce,
            "component": self.component,
            "point": point,
            "request_id": request_id,
            "authority": authority.to_dict(),
            "phase": phase,
            "occurrence": self.arm.occurrence,
            "process_id": os.getpid(),
            "reached_monotonic_ns": time.monotonic_ns(),
            "injection_applied": True,
        }
        _atomic_json(self.root / "reached.json", reached)
        self._reached_authority = authority
        self._deadline = time.monotonic() + self.arm.timeout_s
        return True

    def poll_release(self) -> bool:
        """Complete a reached interlock without blocking its caller.

        Synchronous runtime loops call this method while cooperatively deferring the
        armed item. It returns ``False`` until the controller publishes a release,
        returns ``True`` after completion, and fails closed after the arm timeout.
        """
        if self._completed:
            return True
        if self._reached_authority is None or self._deadline is None:
            return False
        release_path = self.root / "release.json"
        if not release_path.is_file():
            if time.monotonic() >= self._deadline:
                raise RetireTestInterlockError(
                    f"timed out waiting to release {self.component}:{self.arm.point}"
                )
            return False
        release, release_sha256 = _read_json_object(
            release_path, expected_keys=_RELEASE_KEYS
        )
        expected = {
            "artifact": "retire_test_interlock_release",
            "schema_version": 1,
            "nonce": self.arm.nonce,
            "component": self.component,
            "point": self.arm.point,
            "request_id": self.arm.request_id,
            "injection_applied": True,
        }
        for field, value in expected.items():
            if release[field] != value:
                raise RetireTestInterlockError(
                    f"test interlock release {field} mismatch"
                )
        receipt_sha = release["advance_receipt_sha256"]
        if (
            not isinstance(receipt_sha, str)
            or len(receipt_sha) != 64
            or any(character not in "0123456789abcdef" for character in receipt_sha)
        ):
            raise RetireTestInterlockError(
                "advance_receipt_sha256 must be a lowercase SHA-256"
            )
        completed = {
            "artifact": "retire_test_interlock_completed",
            "schema_version": 1,
            "nonce": self.arm.nonce,
            "component": self.component,
            "point": self.arm.point,
            "request_id": self.arm.request_id,
            "authority": self._reached_authority.to_dict(),
            "advance_receipt_sha256": receipt_sha,
            "release_sha256": release_sha256,
            "completed_monotonic_ns": time.monotonic_ns(),
            "injection_applied": True,
        }
        _atomic_json(self.root / "completed.json", completed)
        self._completed = True
        return True

    def require_within_timeout(self, phase: str) -> None:
        if self._deadline is None:
            raise RetireTestInterlockError(
                "test interlock timeout checked before the arm was reached"
            )
        if time.monotonic() >= self._deadline:
            raise RetireTestInterlockError(
                f"timed out waiting for {self.component}:{phase}"
            )

    def record_outcome(self, outcome: str, *, authority_is_current: bool) -> None:
        """Bind a stage-specific stale action to the completed release chain."""

        if not self._completed or self._reached_authority is None:
            raise RetireTestInterlockError(
                "cannot record a test interlock outcome before completion"
            )
        expected = _OUTCOMES.get(self.arm.point)
        if outcome != expected:
            raise RetireTestInterlockError(
                f"unexpected outcome for {self.component}:{self.arm.point}"
            )
        if authority_is_current:
            raise RetireTestInterlockError(
                "cannot record stale test outcome while authority is current"
            )
        completed_raw = (self.root / "completed.json").read_bytes()
        completed_sha256 = hashlib.sha256(completed_raw).hexdigest()
        _atomic_json(
            self.root / "outcome.json",
            {
                "artifact": "retire_test_interlock_outcome",
                "schema_version": 1,
                "nonce": self.arm.nonce,
                "component": self.component,
                "point": self.arm.point,
                "request_id": self.arm.request_id,
                "authority": self._reached_authority.to_dict(),
                "outcome": outcome,
                "local_authority_current": False,
                "completed_sha256": completed_sha256,
                "outcome_monotonic_ns": time.monotonic_ns(),
                "injection_applied": True,
            },
        )

    def poll_release_after_authority_advance(
        self, *, authority_is_current: bool
    ) -> bool:
        """Wait cooperatively for both controller release and local advance."""
        if not self.poll_release():
            return False
        if authority_is_current:
            self.require_within_timeout("local authority advance")
            return False
        return True

    async def pause_async(
        self,
        *,
        point: str,
        request_id: str,
        authority: RetireAuthorityTag | None,
        phase: str = "any",
    ) -> bool:
        if not self.reach(
            point=point,
            request_id=request_id,
            authority=authority,
            phase=phase,
        ):
            return False
        while not self.poll_release():
            await asyncio.sleep(0.001)
        return True
