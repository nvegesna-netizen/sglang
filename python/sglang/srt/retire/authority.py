"""Fail-closed host authority state for versioned SGLang requests.

This module deliberately has no accelerator or SGLang imports.  The tokenizer,
scheduler, detokenizer, and tests can use the same validation and ordering rules
without initializing a serving runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Mapping


class RetireAuthorityError(RuntimeError):
    """Raised when a tagged operation violates authority ordering."""


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RetireAuthorityError(f"{field} must be a non-empty string")
    return value


def _nonnegative_integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RetireAuthorityError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class RetireAuthorityTag:
    tenant_id: str
    scope_id: str
    epoch: int
    generation: int

    def __post_init__(self) -> None:
        _nonempty_string(self.tenant_id, "tenant_id")
        _nonempty_string(self.scope_id, "scope_id")
        _nonnegative_integer(self.epoch, "epoch")
        _nonnegative_integer(self.generation, "generation")

    @property
    def scope_key(self) -> tuple[str, str]:
        return self.tenant_id, self.scope_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "scope_id": self.scope_id,
            "epoch": self.epoch,
            "generation": self.generation,
        }

    @classmethod
    def from_value(cls, value: Any) -> RetireAuthorityTag | None:
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise RetireAuthorityError("retire authority must be an object")
        expected = {"tenant_id", "scope_id", "epoch", "generation"}
        if set(value) != expected:
            raise RetireAuthorityError(
                "retire authority fields must be exactly " + ", ".join(sorted(expected))
            )
        return cls(
            tenant_id=_nonempty_string(value["tenant_id"], "tenant_id"),
            scope_id=_nonempty_string(value["scope_id"], "scope_id"),
            epoch=_nonnegative_integer(value["epoch"], "epoch"),
            generation=_nonnegative_integer(value["generation"], "generation"),
        )


@dataclass(frozen=True, slots=True)
class RetireAdvanceResult:
    tenant_id: str
    scope_id: str
    retired_epoch: int
    new_epoch: int
    generation: int
    retired_request_ids: tuple[str, ...]


class RetireAuthorityTable:
    """Linearizable host table for scope authority and bound requests.

    An unobserved scope accepts only epoch zero at request bind.  An advance may
    establish an unobserved scope at its new epoch; this closes the case where a
    control message reaches a scheduler before an older queued request does.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._current: dict[tuple[str, str], RetireAuthorityTag] = {}
        self._requests: dict[str, RetireAuthorityTag] = {}

    def bind(self, request_id: str, tag: RetireAuthorityTag) -> None:
        request_id = _nonempty_string(request_id, "request_id")
        with self._lock:
            previous = self._requests.get(request_id)
            if previous is not None:
                if previous != tag:
                    raise RetireAuthorityError(
                        f"request {request_id!r} was rebound with different authority"
                    )
                return

            current = self._current.get(tag.scope_key)
            if current is None:
                if tag.epoch != 0:
                    raise RetireAuthorityError(
                        "an uninitialized scope accepts only epoch zero"
                    )
                self._current[tag.scope_key] = tag
                current = tag
            if current.epoch != tag.epoch or current.generation != tag.generation:
                raise RetireAuthorityError(
                    "request authority is not current: "
                    f"request=({tag.epoch},{tag.generation}) "
                    f"current=({current.epoch},{current.generation})"
                )
            self._requests[request_id] = tag

    def advance(
        self,
        *,
        tenant_id: str,
        scope_id: str,
        retired_epoch: int,
        new_epoch: int,
        generation: int,
    ) -> RetireAdvanceResult:
        tenant_id = _nonempty_string(tenant_id, "tenant_id")
        scope_id = _nonempty_string(scope_id, "scope_id")
        retired_epoch = _nonnegative_integer(retired_epoch, "retired_epoch")
        new_epoch = _nonnegative_integer(new_epoch, "new_epoch")
        generation = _nonnegative_integer(generation, "generation")
        if new_epoch != retired_epoch + 1:
            raise RetireAuthorityError("new_epoch must equal retired_epoch + 1")

        key = tenant_id, scope_id
        with self._lock:
            current = self._current.get(key)
            if current is not None:
                if current.epoch != retired_epoch:
                    raise RetireAuthorityError(
                        f"advance expected epoch {retired_epoch}, current is {current.epoch}"
                    )
                if generation <= current.generation:
                    raise RetireAuthorityError(
                        "advance generation must be greater than the current generation"
                    )
            next_tag = RetireAuthorityTag(
                tenant_id=tenant_id,
                scope_id=scope_id,
                epoch=new_epoch,
                generation=generation,
            )
            self._current[key] = next_tag
            retired = tuple(
                sorted(
                    request_id
                    for request_id, request_tag in self._requests.items()
                    if request_tag.scope_key == key
                    and request_tag.epoch == retired_epoch
                )
            )
            return RetireAdvanceResult(
                tenant_id=tenant_id,
                scope_id=scope_id,
                retired_epoch=retired_epoch,
                new_epoch=new_epoch,
                generation=generation,
                retired_request_ids=retired,
            )

    def is_current(self, tag: RetireAuthorityTag | None) -> bool:
        if tag is None:
            return True
        with self._lock:
            current = self._current.get(tag.scope_key)
            return current is not None and (
                current.epoch == tag.epoch
                and current.generation == tag.generation
            )

    def require_current(self, tag: RetireAuthorityTag | None, phase: str) -> None:
        if not self.is_current(tag):
            assert tag is not None
            raise RetireAuthorityError(
                f"stale RETIRE authority at {phase}: "
                f"tenant={tag.tenant_id!r} scope={tag.scope_id!r} "
                f"epoch={tag.epoch} generation={tag.generation}"
            )

    def current(self, tenant_id: str, scope_id: str) -> RetireAuthorityTag | None:
        with self._lock:
            return self._current.get((tenant_id, scope_id))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            scopes = [
                tag.to_dict()
                for _, tag in sorted(self._current.items(), key=lambda item: item[0])
            ]
            requests = {
                request_id: tag.to_dict()
                for request_id, tag in sorted(self._requests.items())
            }
        return {"scopes": scopes, "requests": requests}
