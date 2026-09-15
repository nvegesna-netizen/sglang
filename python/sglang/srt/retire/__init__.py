"""Versioned execution-authority primitives for RETIRE integration."""

from sglang.srt.retire.authority import (
    RetireAdvanceResult,
    RetireAuthorityError,
    RetireAuthorityTable,
    RetireAuthorityTag,
)
from sglang.srt.retire.kv_inheritance import (
    RetireKVInheritanceError,
    RetireKVInheritanceRegistry,
    RetirePinnedPrefix,
    RetireResumeReservation,
    sequence_digest,
)

__all__ = [
    "RetireAdvanceResult",
    "RetireAuthorityError",
    "RetireAuthorityTable",
    "RetireAuthorityTag",
    "RetireKVInheritanceError",
    "RetireKVInheritanceRegistry",
    "RetirePinnedPrefix",
    "RetireResumeReservation",
    "sequence_digest",
]
