"""Host-side certificates for RETIRE physical KV-prefix inheritance.

This module has no torch or scheduler imports.  It records only immutable
evidence; the scheduler remains responsible for acquiring and releasing the
corresponding RadixCache node lock.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Iterable


class RetireKVInheritanceError(RuntimeError):
    """Raised when physical KV inheritance cannot be certified."""


def _require_nonempty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise RetireKVInheritanceError(f"{field} must be a non-empty string")


def _require_nonnegative(value: int, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RetireKVInheritanceError(f"{field} must be a non-negative integer")


def sequence_digest(values: Iterable[int]) -> str:
    """Return an unambiguous SHA-256 digest for a non-negative integer list."""

    materialized = tuple(values)
    digest = hashlib.sha256()
    digest.update(struct.pack(">Q", len(materialized)))
    for value in materialized:
        _require_nonnegative(value, "sequence value")
        digest.update(struct.pack(">Q", value))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RetirePinnedPrefix:
    tenant_id: str
    scope_id: str
    retired_epoch: int
    generation: int
    source_request_id: str
    cache_salt: str
    block_size: int
    token_ids: tuple[int, ...]
    slot_ids: tuple[int, ...]
    writer_drained_sequence: int

    def __post_init__(self) -> None:
        for value, field in (
            (self.tenant_id, "tenant_id"),
            (self.scope_id, "scope_id"),
            (self.source_request_id, "source_request_id"),
            (self.cache_salt, "cache_salt"),
        ):
            _require_nonempty(value, field)
        for value, field in (
            (self.retired_epoch, "retired_epoch"),
            (self.generation, "generation"),
            (self.writer_drained_sequence, "writer_drained_sequence"),
        ):
            _require_nonnegative(value, field)
        if self.block_size <= 0:
            raise RetireKVInheritanceError("block_size must be positive")
        if len(self.token_ids) != len(self.slot_ids):
            raise RetireKVInheritanceError("token and physical-slot lengths differ")
        if len(self.token_ids) % self.block_size:
            raise RetireKVInheritanceError("pinned prefix is not block aligned")
        sequence_digest(self.token_ids)
        sequence_digest(self.slot_ids)

    @property
    def key(self) -> tuple[str, str, int, str]:
        return (
            self.tenant_id,
            self.scope_id,
            self.retired_epoch,
            self.source_request_id,
        )

    @property
    def pinned_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def token_digest(self) -> str:
        return sequence_digest(self.token_ids)

    @property
    def slot_digest(self) -> str:
        return sequence_digest(self.slot_ids)


@dataclass(frozen=True, slots=True)
class RetireResumeReservation:
    successor_request_id: str
    source_key: tuple[str, str, int, str]
    new_epoch: int
    generation: int
    reuse_tokens: int
    cache_salt: str
    token_digest: str
    slot_ids: tuple[int, ...]
    source_token_digest: str
    source_slot_digest: str

    @property
    def slot_digest(self) -> str:
        return sequence_digest(self.slot_ids)


class RetireKVInheritanceRegistry:
    """Scheduler-thread registry for pinned prefixes and pending handoffs."""

    def __init__(self) -> None:
        self._snapshots: dict[tuple[str, str, int, str], RetirePinnedPrefix] = {}
        self._reservations: dict[str, RetireResumeReservation] = {}
        self._reserved_sources: dict[tuple[str, str, int, str], str] = {}

    def install(self, snapshot: RetirePinnedPrefix) -> None:
        if snapshot.key in self._snapshots:
            raise RetireKVInheritanceError("duplicate pinned-prefix snapshot")
        self._snapshots[snapshot.key] = snapshot

    def probe(
        self,
        *,
        tenant_id: str,
        scope_id: str,
        retired_epoch: int,
        source_request_id: str,
        requested_tokens: int,
        block_size: int,
    ) -> tuple[RetirePinnedPrefix, int] | None:
        _require_nonnegative(requested_tokens, "requested_tokens")
        if block_size <= 0 or requested_tokens % block_size:
            raise RetireKVInheritanceError(
                "requested reuse must align to a positive block size"
            )
        key = tenant_id, scope_id, retired_epoch, source_request_id
        snapshot = self._snapshots.get(key)
        if snapshot is None or snapshot.block_size != block_size:
            return None
        available = min(snapshot.pinned_tokens, requested_tokens)
        available -= available % block_size
        return snapshot, available

    def reserve(
        self,
        *,
        tenant_id: str,
        scope_id: str,
        retired_epoch: int,
        source_request_id: str,
        successor_request_id: str,
        new_epoch: int,
        generation: int,
        reuse_tokens: int,
        block_size: int,
        cache_salt: str,
        successor_token_ids: tuple[int, ...],
    ) -> RetireResumeReservation:
        _require_nonempty(successor_request_id, "successor_request_id")
        if successor_request_id in self._reservations:
            raise RetireKVInheritanceError("duplicate successor reservation")
        if new_epoch != retired_epoch + 1:
            raise RetireKVInheritanceError("successor epoch is not consecutive")
        result = self.probe(
            tenant_id=tenant_id,
            scope_id=scope_id,
            retired_epoch=retired_epoch,
            source_request_id=source_request_id,
            requested_tokens=reuse_tokens,
            block_size=block_size,
        )
        if result is None:
            raise RetireKVInheritanceError("source prefix is not pinned")
        snapshot, available = result
        if reuse_tokens <= 0 or available != reuse_tokens:
            raise RetireKVInheritanceError("requested prefix is not fully available")
        if generation <= snapshot.generation:
            raise RetireKVInheritanceError("successor generation did not advance")
        if cache_salt != snapshot.cache_salt:
            raise RetireKVInheritanceError("successor cache salt differs from source")
        if (
            tuple(successor_token_ids[:reuse_tokens])
            != snapshot.token_ids[:reuse_tokens]
        ):
            raise RetireKVInheritanceError("successor token prefix differs from source")
        if snapshot.key in self._reserved_sources:
            raise RetireKVInheritanceError("source prefix already has a successor")
        reservation = RetireResumeReservation(
            successor_request_id=successor_request_id,
            source_key=snapshot.key,
            new_epoch=new_epoch,
            generation=generation,
            reuse_tokens=reuse_tokens,
            cache_salt=cache_salt,
            token_digest=sequence_digest(snapshot.token_ids[:reuse_tokens]),
            slot_ids=snapshot.slot_ids[:reuse_tokens],
            source_token_digest=snapshot.token_digest,
            source_slot_digest=snapshot.slot_digest,
        )
        self._reservations[successor_request_id] = reservation
        self._reserved_sources[snapshot.key] = successor_request_id
        return reservation

    def verify_launch(
        self,
        *,
        successor_request_id: str,
        tenant_id: str,
        scope_id: str,
        epoch: int,
        generation: int,
        cache_salt: str,
        token_ids: tuple[int, ...],
        slot_ids: tuple[int, ...],
    ) -> RetirePinnedPrefix | None:
        reservation = self._reservations.get(successor_request_id)
        if reservation is None:
            return None
        expected_scope = reservation.source_key[:2]
        if (tenant_id, scope_id) != expected_scope:
            raise RetireKVInheritanceError("successor scope differs from reservation")
        if (epoch, generation) != (
            reservation.new_epoch,
            reservation.generation,
        ):
            raise RetireKVInheritanceError(
                "successor authority differs from reservation"
            )
        if cache_salt != reservation.cache_salt:
            raise RetireKVInheritanceError(
                "successor cache salt differs from reservation"
            )
        if (
            sequence_digest(token_ids[: reservation.reuse_tokens])
            != reservation.token_digest
        ):
            raise RetireKVInheritanceError(
                "successor token digest differs from reservation"
            )
        if tuple(slot_ids[: reservation.reuse_tokens]) != reservation.slot_ids:
            raise RetireKVInheritanceError(
                "successor physical slots differ from reservation"
            )

        snapshot = self._snapshots.pop(reservation.source_key)
        del self._reserved_sources[reservation.source_key]
        del self._reservations[successor_request_id]
        return snapshot

    def cancel_reservation(
        self, successor_request_id: str
    ) -> RetirePinnedPrefix | None:
        reservation = self._reservations.pop(successor_request_id, None)
        if reservation is None:
            return None
        self._reserved_sources.pop(reservation.source_key, None)
        return self._snapshots.get(reservation.source_key)

    def pop_unreserved_scope(
        self, tenant_id: str, scope_id: str, retired_epoch: int
    ) -> tuple[RetirePinnedPrefix, ...]:
        keys = [
            key
            for key in self._snapshots
            if key[:3] == (tenant_id, scope_id, retired_epoch)
            and key not in self._reserved_sources
        ]
        return tuple(self._snapshots.pop(key) for key in keys)

    def snapshot(self) -> dict[str, int]:
        return {
            "pinned_prefixes": len(self._snapshots),
            "pending_reservations": len(self._reservations),
        }

    def scope_state(
        self, tenant_id: str, scope_id: str, retired_epoch: int
    ) -> dict[str, int]:
        prefix = tenant_id, scope_id, retired_epoch
        keys = {key for key in self._snapshots if key[:3] == prefix}
        return {
            "pinned_prefixes": len(keys),
            "pending_reservations": sum(key in self._reserved_sources for key in keys),
        }
