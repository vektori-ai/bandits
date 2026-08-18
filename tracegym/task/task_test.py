"""Tests for pre-state reconstruction, filler generation and task mining."""

from __future__ import annotations

import pytest

from tracegym.contracts import EntityRows, TaskCase
from tracegym.task._testdata import (
    RETAIL_SCHEMA,
    WRITE_TOOLS,
    corpus,
    ep_cross_entity,
    ep_double_refund,
    ep_refund_ok,
    ep_status_update,
)
from tracegym.task.filler import FillerError, fill_task, generate_filler
from tracegym.task.mine import mine_task, mine_tasks
from tracegym.task.prestate import reconstruct_final_state, reconstruct_pre_state


def _rows(pre_state: tuple[EntityRows, ...], entity: str) -> list[dict]:
    for er in pre_state:
        if er.entity == entity:
            return list(er.rows)
    return []


# --------------------------------------------------------------------------
# THE test: post-state must not leak backward
# --------------------------------------------------------------------------


def test_post_write_read_does_not_leak_into_pre_state():
    """ep-refund-ok reads order 7741 twice: 'delivered' at step 2 (pre-write) and
    'refunded' at step 5 (post-write). The pre-state must be the step-2 value.

    If this test ever fails, every rollout for this task starts in the solved
    state and the reward signal is silently dead."""
    pre = reconstruct_pre_state(ep_refund_ok(), RETAIL_SCHEMA)
    orders = _rows(pre.to_entity_rows(), "orders")
    assert len(orders) == 1
    assert orders[0]["order_id"] == 7741
    assert orders[0]["status"] == "delivered"
    assert orders[0]["status"] != "refunded"
    # and the rejection is auditable, not silent
    assert any(b["step"] == 5 and b["entity"] == "orders" for b in pre.blocked)
    assert pre.first_write_step == 4


def test_write_response_row_is_not_pre_state_evidence():
    """The refund_order response itself carries status='refunded'. It is a write,
    so it never contributes rows to the pre-state."""
    pre = reconstruct_pre_state(ep_refund_ok(), RETAIL_SCHEMA)
    assert all(r["status"] == "delivered" for r in _rows(pre.to_entity_rows(), "orders"))


def test_explicit_write_tool_override_matches_schema_derived_set():
    a = reconstruct_pre_state(ep_refund_ok(), RETAIL_SCHEMA)
    b = reconstruct_pre_state(ep_refund_ok(), RETAIL_SCHEMA, write_tools=WRITE_TOOLS)
    assert a.to_entity_rows() == b.to_entity_rows()


# --------------------------------------------------------------------------
# per-entity, per-row tracking
# --------------------------------------------------------------------------


def test_read_of_other_entity_after_write_is_still_pre_state():
    """A write to orders says nothing about products, so a later get_product is
    still evidence of the starting world."""
    pre = reconstruct_pre_state(ep_cross_entity(), RETAIL_SCHEMA)
    products = _rows(pre.to_entity_rows(), "products")
    assert [p["sku"] for p in products] == ["SKU-BLU-1"]


def test_read_of_other_row_of_written_entity_after_write_is_still_pre_state():
    """Per-row, not per-entity: refunding 7750 does not invalidate a later read
    of order 7742."""
    pre = reconstruct_pre_state(ep_cross_entity(), RETAIL_SCHEMA)
    ids = {r["order_id"]: r for r in _rows(pre.to_entity_rows(), "orders")}
    assert set(ids) == {7750, 7742}
    assert ids[7750]["status"] == "delivered"  # pre-write value
    assert ids[7742]["status"] == "in_transit"
    assert pre.blocked == []


def test_unidentifiable_write_target_dirties_the_whole_entity():
    """When a write names no row, we cannot tell which rows it touched, so all
    later reads of that entity are refused."""
    trace = ep_cross_entity()
    stripped = trace.model_copy(
        update={
            "invocations": tuple(
                inv.model_copy(update={"arguments": {}, "response": {"ok": True}})
                if inv.tool == "refund_order"
                else inv
                for inv in trace.invocations
            )
        }
    )
    pre = reconstruct_pre_state(stripped, RETAIL_SCHEMA)
    ids = {r["order_id"] for r in _rows(pre.to_entity_rows(), "orders")}
    assert ids == {7750}  # 7742 was read after the un-attributable write
    assert any("whole entity" in n for n in pre.notes)
    # products, a different entity, is untouched by the blanket dirty flag
    assert _rows(pre.to_entity_rows(), "products")


def test_error_response_is_not_pre_state_evidence():
    """{'error': 'not_found', 'order_id': 9999} describes an ABSENT row. Seeding
    it would invert the task."""
    pre = reconstruct_pre_state(ep_double_refund(), RETAIL_SCHEMA)
    assert _rows(pre.to_entity_rows(), "orders") == []


def test_first_observation_wins_and_only_fills_gaps():
    trace = ep_status_update()
    partial = trace.invocations[0].model_copy(
        update={"step": 0, "response": {"order_id": 7742, "status": "in_transit"}}
    )
    later = trace.invocations[0].model_copy(update={"step": 1, "call_id": "c1"})
    t2 = trace.model_copy(update={"invocations": (partial, later)})
    pre = reconstruct_pre_state(t2, RETAIL_SCHEMA)
    row = _rows(pre.to_entity_rows(), "orders")[0]
    assert row["status"] == "in_transit"
    assert row["sku"] == "SKU-YEL-7"  # gap filled by the later pre-write read


# --------------------------------------------------------------------------
# solvability
# --------------------------------------------------------------------------


def test_unsolvable_task_is_flagged():
    """ep-double-refund writes before it reads, so no pre-state row backs the
    ids in its instruction. Unflagged, every rollout fails identically and the
    pass@k gate reads it as 'beyond the model's ability'."""
    task, reason, pre = mine_task(ep_double_refund(), RETAIL_SCHEMA)
    assert reason is None and task is not None and pre is not None
    warnings = task.provenance["solvability_warnings"]
    assert warnings
    assert any("7741" in w for w in warnings)
    assert any("88" in w for w in warnings)


def test_solvable_task_has_no_warning():
    task, _, _ = mine_task(ep_refund_ok(), RETAIL_SCHEMA)
    assert task is not None
    assert task.provenance["solvability_warnings"] == ()


# --------------------------------------------------------------------------
# mining
# --------------------------------------------------------------------------


def test_mine_tasks_over_the_corpus():
    result = mine_tasks(corpus(), RETAIL_SCHEMA)
    ids = {t.task_id for t in result.tasks}
    assert ids == {
        "task-ep-refund-ok",
        "task-ep-cross-entity",
        "task-ep-double-refund",
        "task-ep-status-update",
    }
    reasons = {s["trace_id"]: s["reason"] for s in result.skipped}
    assert "no invocations" in reasons["ep-chatter"]
    assert "no user message" in reasons["ep-headless"]
    assert [t.task_id for t in result.warned] == ["task-ep-double-refund"]


def test_task_carries_tools_outcome_and_provenance():
    task, _, _ = mine_task(ep_refund_ok(), RETAIL_SCHEMA)
    assert task is not None
    assert task.outcome is True
    assert task.tools == (
        "get_customer",
        "search_orders",
        "get_order",
        "get_store_policy",
        "refund_order",
        "send_email",
    )
    assert task.provenance["source_trace_id"] == "ep-refund-ok"
    assert task.provenance["invocation_count"] == 7


# --------------------------------------------------------------------------
# filler
# --------------------------------------------------------------------------


def _observed_strings() -> set[str]:
    out: set[str] = set()
    for e in RETAIL_SCHEMA.entities:
        for f in e.fields:
            out |= {str(v) for v in f.sample_values}
    return out


def test_filler_never_emits_an_observed_real_value():
    real = _observed_strings()
    for entity in RETAIL_SCHEMA.entities:
        rows = generate_filler(entity, count=6, seed=7)
        for row in rows:
            for name, value in row.items():
                if isinstance(value, bool):
                    continue  # booleans have two values; shape and value coincide
                assert str(value) not in real, f"{entity.name}.{name} leaked {value!r}"


def test_filler_is_deterministic_across_runs():
    e = RETAIL_SCHEMA.entity("orders")
    assert e is not None
    assert generate_filler(e, count=5, seed=42) == generate_filler(e, count=5, seed=42)
    assert generate_filler(e, count=5, seed=42) != generate_filler(e, count=5, seed=43)


def test_filler_copies_shapes():
    e = RETAIL_SCHEMA.entity("orders")
    assert e is not None
    for row in generate_filler(e, count=5, seed=3):
        assert isinstance(row["order_id"], int)
        assert len(str(row["order_id"])) == 4
        assert row["placed_at"][4] == "-" and len(row["placed_at"]) == 10
        assert row["sku"].startswith(("A", "B", "C")) or row["sku"][3] == "-"
        assert row["sku"].count("-") == 2


def test_filler_primary_keys_are_unique_and_avoid_task_rows():
    task, _, _ = mine_task(ep_refund_ok(), RETAIL_SCHEMA)
    assert task is not None
    filled = fill_task(task, RETAIL_SCHEMA, seed=11, count_per_entity=5)
    orders = _rows(filled, "orders")
    assert orders[0]["order_id"] == 7741  # the real row is first and unmodified
    assert orders[0]["status"] == "delivered"
    assert len({r["order_id"] for r in orders}) == len(orders)


def test_filler_skips_static_snapshot_entities():
    e = RETAIL_SCHEMA.entity("store_policy")
    assert e is not None
    assert generate_filler(e, count=5, seed=1) == ()


def test_preserve_values_is_opt_in_for_enums():
    e = RETAIL_SCHEMA.entity("orders")
    assert e is not None
    rows = generate_filler(e, count=5, seed=5, preserve_values={"status"})
    assert all(r["status"] in {"delivered", "refunded", "in_transit"} for r in rows)


def test_filler_refuses_when_the_value_space_is_exhausted():
    e = RETAIL_SCHEMA.entity("products")
    assert e is not None
    tiny = e.model_copy(
        update={
            "fields": (
                e.fields[0].model_copy(update={"json_types": ("boolean",)}),
            ),
            "primary_key": "sku",
        }
    )
    with pytest.raises(FillerError):
        generate_filler(tiny, count=5, seed=1)


# --------------------------------------------------------------------------
# final-state reconstruction (consumed by verifier synthesis)
# --------------------------------------------------------------------------


def test_final_state_is_the_post_write_view():
    final = reconstruct_final_state(ep_refund_ok(), RETAIL_SCHEMA)
    row = next(iter(final["orders"].values()))
    assert row["status"] == "refunded"
    # fields the entity does not declare never enter the state
    assert "refund_amount_cents" not in row


def test_mine_task_returns_a_taskcase_contract():
    task, _, _ = mine_task(ep_refund_ok(), RETAIL_SCHEMA)
    assert isinstance(task, TaskCase)
