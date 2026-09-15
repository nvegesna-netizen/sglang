from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ModuleNotFoundError:
    # Keep this dependency-free test runnable before the SGLang image exists.
    def register_cpu_ci(**_kwargs):
        return None


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


MODULE_PATH = (
    Path(__file__).resolve().parents[4] / "python/sglang/srt/retire/kv_inheritance.py"
)
SPEC = importlib.util.spec_from_file_location(
    "sglang_retire_kv_inheritance", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
KV = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = KV
SPEC.loader.exec_module(KV)

RetireKVInheritanceError = KV.RetireKVInheritanceError
RetireKVInheritanceRegistry = KV.RetireKVInheritanceRegistry
RetirePinnedPrefix = KV.RetirePinnedPrefix
sequence_digest = KV.sequence_digest
standard_cache_is_certifiable = KV.standard_cache_is_certifiable


def prefix(*, request_id: str = "old", generation: int = 2):
    return RetirePinnedPrefix(
        tenant_id="tenant",
        scope_id="scope",
        retired_epoch=0,
        generation=generation,
        source_request_id=request_id,
        cache_salt="shared-salt",
        block_size=4,
        token_ids=(10, 11, 12, 13, 14, 15, 16, 17),
        slot_ids=(30, 31, 32, 33, 40, 41, 42, 43),
        writer_drained_sequence=9,
    )


class TestRetireKVInheritanceRegistry(unittest.TestCase):
    def test_only_standard_single_group_cache_is_certifiable(self):
        standard = {
            "hybrid_swa": False,
            "hybrid_ssm": False,
            "speculative": False,
            "diffusion": False,
            "disaggregated": False,
            "hierarchical_cache": False,
            "rust_frontend": False,
            "kv_pool_type": "PagedTokenToKVPoolAllocator",
            "tree_cache_type": "RadixCache",
        }
        self.assertTrue(standard_cache_is_certifiable(**standard))
        mutations = {
            "hybrid_swa": True,
            "hybrid_ssm": True,
            "speculative": True,
            "diffusion": True,
            "disaggregated": True,
            "hierarchical_cache": True,
            "rust_frontend": True,
            "kv_pool_type": "TokenToKVPoolAllocator",
            "tree_cache_type": "ChunkCache",
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                self.assertFalse(
                    standard_cache_is_certifiable(**{**standard, field: value})
                )

    def test_sequence_digest_is_length_delimited_and_rejects_negative_values(self):
        self.assertNotEqual(sequence_digest((1, 23)), sequence_digest((12, 3)))
        with self.assertRaises(RetireKVInheritanceError):
            sequence_digest((1, -1))

    def test_snapshot_requires_equal_block_aligned_token_and_slot_rows(self):
        with self.assertRaisesRegex(RetireKVInheritanceError, "lengths differ"):
            RetirePinnedPrefix(
                tenant_id="tenant",
                scope_id="scope",
                retired_epoch=0,
                generation=0,
                source_request_id="old",
                cache_salt="salt",
                block_size=4,
                token_ids=(1, 2, 3, 4),
                slot_ids=(1, 2, 3),
                writer_drained_sequence=0,
            )
        with self.assertRaisesRegex(RetireKVInheritanceError, "block aligned"):
            RetirePinnedPrefix(
                tenant_id="tenant",
                scope_id="scope",
                retired_epoch=0,
                generation=0,
                source_request_id="old",
                cache_salt="salt",
                block_size=4,
                token_ids=(1, 2, 3),
                slot_ids=(4, 5, 6),
                writer_drained_sequence=0,
            )

    def test_probe_is_tenant_scoped_block_aligned_and_bounded(self):
        registry = RetireKVInheritanceRegistry()
        snapshot = prefix()
        registry.install(snapshot)
        self.assertEqual(
            registry.probe(
                tenant_id="tenant",
                scope_id="scope",
                retired_epoch=0,
                source_request_id="old",
                requested_tokens=4,
                block_size=4,
            ),
            (snapshot, 4),
        )
        self.assertIsNone(
            registry.probe(
                tenant_id="other",
                scope_id="scope",
                retired_epoch=0,
                source_request_id="old",
                requested_tokens=4,
                block_size=4,
            )
        )
        with self.assertRaisesRegex(RetireKVInheritanceError, "align"):
            registry.probe(
                tenant_id="tenant",
                scope_id="scope",
                retired_epoch=0,
                source_request_id="old",
                requested_tokens=3,
                block_size=4,
            )

    def test_reservation_requires_same_tokens_salt_and_advanced_generation(self):
        registry = RetireKVInheritanceRegistry()
        registry.install(prefix())
        base = dict(
            tenant_id="tenant",
            scope_id="scope",
            retired_epoch=0,
            source_request_id="old",
            successor_request_id="new",
            new_epoch=1,
            generation=3,
            reuse_tokens=4,
            block_size=4,
            cache_salt="shared-salt",
            successor_token_ids=(10, 11, 12, 13, 99),
        )
        reservation = registry.reserve(**base)
        self.assertEqual(reservation.slot_ids, (30, 31, 32, 33))

        for change, message in (
            ({"successor_request_id": "bad-salt", "cache_salt": "other"}, "salt"),
            (
                {
                    "successor_request_id": "bad-token",
                    "successor_token_ids": (10, 11, 12, 99),
                },
                "token prefix",
            ),
            ({"successor_request_id": "bad-generation", "generation": 2}, "generation"),
        ):
            second = RetireKVInheritanceRegistry()
            second.install(prefix())
            with self.assertRaisesRegex(RetireKVInheritanceError, message):
                second.reserve(**{**base, **change})

    def test_launch_consumes_pin_only_for_exact_physical_mapping(self):
        registry = RetireKVInheritanceRegistry()
        snapshot = prefix()
        registry.install(snapshot)
        registry.reserve(
            tenant_id="tenant",
            scope_id="scope",
            retired_epoch=0,
            source_request_id="old",
            successor_request_id="new",
            new_epoch=1,
            generation=3,
            reuse_tokens=4,
            block_size=4,
            cache_salt="shared-salt",
            successor_token_ids=(10, 11, 12, 13, 99),
        )
        with self.assertRaisesRegex(RetireKVInheritanceError, "physical slots"):
            registry.verify_launch(
                successor_request_id="new",
                tenant_id="tenant",
                scope_id="scope",
                epoch=1,
                generation=3,
                cache_salt="shared-salt",
                token_ids=(10, 11, 12, 13, 99),
                slot_ids=(30, 31, 32, 99),
            )
        self.assertEqual(
            registry.snapshot(),
            {"pinned_prefixes": 1, "pending_reservations": 1},
        )
        released = registry.verify_launch(
            successor_request_id="new",
            tenant_id="tenant",
            scope_id="scope",
            epoch=1,
            generation=3,
            cache_salt="shared-salt",
            token_ids=(10, 11, 12, 13, 99),
            slot_ids=(30, 31, 32, 33),
        )
        assert released is not None
        released_snapshot, released_reservation = released
        self.assertEqual(released_snapshot, snapshot)
        self.assertEqual(released_reservation.successor_request_id, "new")
        self.assertEqual(
            registry.snapshot(),
            {"pinned_prefixes": 0, "pending_reservations": 0},
        )

    def test_scope_reclaim_keeps_reserved_source_until_cancelled(self):
        registry = RetireKVInheritanceRegistry()
        snapshot = prefix()
        registry.install(snapshot)
        registry.reserve(
            tenant_id="tenant",
            scope_id="scope",
            retired_epoch=0,
            source_request_id="old",
            successor_request_id="new",
            new_epoch=1,
            generation=3,
            reuse_tokens=4,
            block_size=4,
            cache_salt="shared-salt",
            successor_token_ids=snapshot.token_ids,
        )
        self.assertEqual(registry.pop_unreserved_scope("tenant", "scope", 0), ())
        self.assertEqual(registry.cancel_reservation("new"), snapshot)
        self.assertEqual(
            registry.pop_unreserved_scope("tenant", "scope", 0), (snapshot,)
        )

    def test_cancel_rejects_a_different_source_scope_without_mutation(self):
        registry = RetireKVInheritanceRegistry()
        snapshot = prefix()
        registry.install(snapshot)
        registry.reserve(
            tenant_id="tenant",
            scope_id="scope",
            retired_epoch=0,
            source_request_id="old",
            successor_request_id="new",
            new_epoch=1,
            generation=3,
            reuse_tokens=4,
            block_size=4,
            cache_salt="shared-salt",
            successor_token_ids=snapshot.token_ids,
        )

        with self.assertRaisesRegex(RetireKVInheritanceError, "different source scope"):
            registry.cancel_reservation(
                "new", expected_source_scope=("other", "scope", 0)
            )
        self.assertEqual(
            registry.snapshot(),
            {"pinned_prefixes": 1, "pending_reservations": 1},
        )

    def test_fault_corruption_changes_only_reserved_expected_slot(self):
        registry = RetireKVInheritanceRegistry()
        snapshot = prefix()
        registry.install(snapshot)
        reservation = registry.reserve(
            tenant_id="tenant",
            scope_id="scope",
            retired_epoch=0,
            source_request_id="old",
            successor_request_id="new",
            new_epoch=1,
            generation=3,
            reuse_tokens=4,
            block_size=4,
            cache_salt="shared-salt",
            successor_token_ids=snapshot.token_ids,
        )

        before, after = registry.test_corrupt_reservation_slot("new", slot_offset=2)
        self.assertEqual(before, reservation)
        self.assertEqual(after.slot_ids, (30, 31, 33, 33))
        self.assertNotEqual(after.slot_digest, before.slot_digest)
        self.assertEqual(
            registry.snapshot(), {"pinned_prefixes": 1, "pending_reservations": 1}
        )
        with self.assertRaisesRegex(RetireKVInheritanceError, "physical slots"):
            registry.verify_launch(
                successor_request_id="new",
                tenant_id="tenant",
                scope_id="scope",
                epoch=1,
                generation=3,
                cache_salt="shared-salt",
                token_ids=snapshot.token_ids,
                slot_ids=snapshot.slot_ids[:4],
            )


if __name__ == "__main__":
    unittest.main()
