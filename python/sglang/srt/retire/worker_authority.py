"""Worker-side RETIRE authority binding for SGLang model forwards.

The frontend publishes scope versions through RETIRE's mmap mailbox before it
enters SGLang's synchronous collective RPC.  Each scheduler/model process maps
that file independently.  Long, fully tagged forwards may then stop submitting
layers at a bounded model safe point; short steps retain SGLang's ordinary
model-step boundary.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def initialize_worker_authority(device: Any) -> Any | None:
    """Initialize the relay only for an explicitly configured campaign."""

    if not os.environ.get("RETIRE_EPOCH_MAILBOX_PATH"):
        return None
    if not str(device).startswith("cuda"):
        raise RuntimeError("SGLang RETIRE worker authority requires a CUDA device")

    from retire_serving.epoch_mailbox import start_worker_epoch_relay

    return start_worker_epoch_relay(device)


def bind_forward_authority(forward_batch: ForwardBatch) -> Any | None:
    """Bind this exact batch before graph selection and model submission."""

    from retire_serving.epoch_mailbox import get_worker_epoch_state

    rids = list(forward_batch.rids or ())
    authorities = list(forward_batch.retire_worker_authorities or ())
    if authorities and len(authorities) != len(rids):
        raise RuntimeError(
            "RETIRE worker authority cardinality differs from request IDs"
        )

    state = get_worker_epoch_state()
    has_authority = any(authority is not None for authority in authorities)
    if state is None:
        if has_authority:
            raise RuntimeError(
                "RETIRE worker authority arrived without an initialized mailbox relay"
            )
        return None

    request_tags = {
        rid: authority
        for rid, authority in zip(rids, authorities, strict=True)
        if authority is not None
    }
    scheduled_tokens = forward_batch.global_num_token_non_padded_cpu
    if scheduled_tokens is None:
        scheduled_tokens = (
            int(forward_batch.input_ids.numel())
            if forward_batch.input_ids is not None
            else 0
        )
    state.bind_step_authority(
        rids,
        request_tags,
        scheduled_token_count=int(scheduled_tokens),
    )
    if state.step_requires_device_predicates:
        state.wait_for_versions(list(state.step_authorities))
    return state


def forward_requires_layer_safe_points() -> bool:
    """Return whether the currently bound step must stay out of CUDA graphs."""

    from retire_serving.epoch_mailbox import get_worker_epoch_state

    state = get_worker_epoch_state()
    return bool(state is not None and state.step_requires_device_predicates)


def finish_forward_authority(state: Any | None) -> None:
    if state is not None:
        state.finish_step()


def layer_safe_point_plan():
    """Build a TP-consistent bounded layer plan for the current SGLang step."""

    if not os.environ.get("RETIRE_EPOCH_MAILBOX_PATH"):
        return None

    from retire_serving.epoch_mailbox import worker_layer_safe_point_plan
    from sglang.srt.runtime_context import get_parallel

    return worker_layer_safe_point_plan(lambda: get_parallel().tp_group)


def worker_authority_audit() -> dict[str, Any]:
    """Return local mechanism counters for a scheduler-rank receipt."""

    from retire_serving.epoch_mailbox import (
        get_worker_epoch_state,
        layer_safe_point_horizon,
        layer_safe_point_min_tokens,
        worker_epoch_audit,
    )

    configured = bool(os.environ.get("RETIRE_EPOCH_MAILBOX_PATH"))
    state = get_worker_epoch_state()
    return {
        "backend": "sglang",
        "mailbox_configured": configured,
        "worker_relay_initialized": state is not None,
        "layer_horizon": layer_safe_point_horizon(),
        "layer_safe_point_min_tokens": layer_safe_point_min_tokens(),
        **worker_epoch_audit(),
    }
