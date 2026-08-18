"""Tests for verifier synthesis, grading and anti-cheat.

The load-bearing pair is `test_verifier_passes_on_a_correct_rollout` and
`test_verifier_fails_when_the_store_was_written_directly`: same verifier, same
final state, different means. If both do not hold, the environment rewards
cheating.
"""

from __future__ import annotations

import pytest

from tracegym.contracts import (
    AssertionKind,
    Effect,
    EnvManifest,
    ToolClass,
    Verifier,
)
from tracegym.task._testdata import (
    RETAIL_SCHEMA,
    customer,
    ep_double_refund,
    ep_refund_ok,
    make_trace,
    order,
    product,
)
from tracegym.task.mine import mine_task
from tracegym.verify.anticheat import RolloutAction, RolloutRecord, check_rollout, enforce
from tracegym.verify.run import UnreviewedVerifierError, evaluate
from tracegym.verify.synthesize import UnlabeledTraceError, synthesize_verifier

ADA = customer(88, "Ada Lovelace", "ada@example.com", "gold")
ORDER_PRE = order(7741, 88, "SKU-RED-9", "delivered", 4200, "2026-07-02")
ORDER_POST = order(7741, 88, "SKU-RED-9", "refunded", 4200, "2026-07-02")
RED = product("SKU-RED-9", "Red Widget", 4200, True)

TOOL_CLASSES = {
    "get_customer": ToolClass.READ,
    "search_orders": ToolClass.READ,
    "get_order": ToolClass.READ,
    "get_product": ToolClass.READ,
    "get_store_policy": ToolClass.READ,
    "refund_order": ToolClass.WRITE,
    "update_order_status": ToolClass.WRITE,
    "send_email": ToolClass.EXTERNAL,
}
MANIFEST = EnvManifest(
    env_id="env-1",
    task_id="task-ep-refund-ok",
    schema_digest="deadbeef",
    tool_classes=TOOL_CLASSES,
    static_entities=("store_policy",),
)


def refund_verifier(reviewed: bool = True) -> Verifier:
    trace = ep_refund_ok()
    task, _, _ = mine_task(trace, RETAIL_SCHEMA)
    assert task is not None
    v = synthesize_verifier(task, trace, RETAIL_SCHEMA)
    return v.model_copy(update={"reviewed_by": "a-human"}) if reviewed else v


def final_state(order_row=None, customers=None, products=None):
    return {
        "customers": customers if customers is not None else [ADA],
        "orders": [order_row if order_row is not None else ORDER_POST],
        "products": products if products is not None else [RED],
    }


HONEST_ACTIONS = (
    RolloutAction(tool="get_customer", arguments={"customer_id": 88}, step=0),
    RolloutAction(tool="get_order", arguments={"order_id": 7741}, step=1),
    RolloutAction(tool="refund_order", arguments={"order_id": 7741, "amount_cents": 4200}, step=2),
    RolloutAction(tool="send_email", arguments={"to_customer_id": 88, "subject": "Your refund",
                                                "body": "done"}, step=3),
)


def record(actions=HONEST_ACTIONS, **kw) -> RolloutRecord:
    base = dict(
        task_id="task-ep-refund-ok",
        actions=actions,
        manifest=MANIFEST,
        pre_state={"customers": [ADA], "orders": [ORDER_PRE], "products": [RED]},
        final_state=final_state(),
        primary_keys={"customers": "customer_id", "orders": "order_id", "products": "sku"},
        resource_reads=(),
        network_calls=(),
    )
    base.update(kw)
    return RolloutRecord(**base)


# --------------------------------------------------------------------------
# synthesis
# --------------------------------------------------------------------------


def test_synthesized_assertions_match_the_plan_example():
    v = refund_verifier()
    kinds = {(a.kind, a.entity, a.field, a.tool) for a in v.assertions}
    assert (AssertionKind.STATE_EQUALS, "orders", "status", None) in kinds
    assert (AssertionKind.STATE_UNCHANGED, "customers", None, None) in kinds
    assert (AssertionKind.EFFECT_COUNT, None, None, "send_email") in kinds
    equals = next(a for a in v.assertions if a.kind is AssertionKind.STATE_EQUALS)
    assert equals.expected == "refunded"
    assert equals.row_key == {"order_id": 7741}
    assert all(a.description for a in v.assertions)


def test_synthesis_refuses_a_negative_outcome():
    trace = ep_double_refund()
    task, _, _ = mine_task(trace, RETAIL_SCHEMA)
    assert task is not None
    with pytest.raises(UnlabeledTraceError, match="outcome=False"):
        synthesize_verifier(task, trace, RETAIL_SCHEMA)


def test_synthesis_refuses_an_unlabeled_outcome():
    trace = ep_refund_ok().model_copy(update={"outcome": None})
    task, _, _ = mine_task(trace, RETAIL_SCHEMA)
    assert task is not None
    with pytest.raises(UnlabeledTraceError):
        synthesize_verifier(task, trace, RETAIL_SCHEMA)


def test_verifier_starts_unreviewed_and_refuses_to_grade():
    v = refund_verifier(reviewed=False)
    assert v.reviewed_by is None
    with pytest.raises(UnreviewedVerifierError):
        evaluate(v, final_state(), [Effect(tool="send_email")])
    # dry runs must say so out loud
    assert evaluate(v, final_state(), [Effect(tool="send_email")], allow_unreviewed=True).passed


def test_static_snapshot_entity_gets_no_assertions():
    v = refund_verifier()
    assert all(a.entity != "store_policy" for a in v.assertions)


# --------------------------------------------------------------------------
# grading
# --------------------------------------------------------------------------


def test_verifier_passes_on_a_correct_rollout():
    v = refund_verifier()
    r = evaluate(v, final_state(), [Effect(tool="send_email", step=3)])
    assert r.passed
    assert r.reward == 1.0


def test_verifier_tolerates_filler_rows():
    """The materialized env has synthetic filler the trace never saw. Extra rows
    must not trip STATE_UNCHANGED, or every real rollout fails."""
    v = refund_verifier()
    state = final_state(
        customers=[ADA, customer(41, "Zzz", "z@example.invalid", "bronze")],
        products=[RED, product("QQQ-ZZZ-1", "thing", 10, False)],
    )
    assert evaluate(v, state, [Effect(tool="send_email")]).passed


def test_verifier_fails_when_the_target_field_is_wrong():
    v = refund_verifier()
    r = evaluate(v, final_state(order_row=ORDER_PRE), [Effect(tool="send_email")])
    assert not r.passed and r.reward == 0.0
    bad = [x for x in r.results if not x.passed]
    assert [x.assertion.kind for x in bad] == [AssertionKind.STATE_EQUALS]
    assert bad[0].actual == "delivered"


def test_state_unchanged_catches_collateral_damage():
    """Target field right, unrelated entity damaged. Without STATE_UNCHANGED this
    rollout would score 1.0 and the policy would learn that wrecking the
    customers table is free."""
    v = refund_verifier()
    damaged = dict(ADA, tier="platinum")
    r = evaluate(v, final_state(customers=[damaged]), [Effect(tool="send_email")])
    assert not r.passed
    failed = [x for x in r.results if not x.passed]
    assert [x.assertion.entity for x in failed] == ["customers"]
    assert failed[0].assertion.kind is AssertionKind.STATE_UNCHANGED

    # deleting the row entirely is caught by the same assertion
    assert not evaluate(v, final_state(customers=[]), [Effect(tool="send_email")]).passed


def test_state_unchanged_catches_damage_within_the_changed_row():
    v = refund_verifier()
    mangled = dict(ORDER_POST, total_cents=0)
    r = evaluate(v, final_state(order_row=mangled), [Effect(tool="send_email")])
    assert not r.passed
    failed = [x for x in r.results if not x.passed]
    assert failed[0].assertion.kind is AssertionKind.STATE_UNCHANGED
    assert failed[0].actual == {"total_cents": 0}


def test_effect_count_catches_zero_and_two_emails():
    v = refund_verifier()
    none = evaluate(v, final_state(), [])
    assert not none.passed
    zero = next(x for x in none.results if x.assertion.kind is AssertionKind.EFFECT_COUNT)
    assert zero.actual == 0

    two = evaluate(v, final_state(), [Effect(tool="send_email", step=3),
                                      Effect(tool="send_email", step=4)])
    assert not two.passed
    dup = next(x for x in two.results if x.assertion.kind is AssertionKind.EFFECT_COUNT)
    assert dup.actual == 2


def test_partial_credit_and_all_or_nothing_differ():
    v = refund_verifier()
    # status refunded (right) but no email sent (wrong): partially correct
    strict = evaluate(v, final_state(), [])
    loose = evaluate(v, final_state(), [], mode="partial")
    assert strict.reward == 0.0
    assert 0.0 < loose.reward < 1.0
    assert loose.reward == pytest.approx(3 / 4)
    assert strict.passed is loose.passed is False


def test_evaluate_rejects_an_empty_verifier():
    empty = Verifier(verifier_id="v", task_id="t", assertions=(), reviewed_by="h")
    with pytest.raises(ValueError, match="no assertions"):
        evaluate(empty, {})
    with pytest.raises(ValueError, match="unknown reward mode"):
        evaluate(refund_verifier(), final_state(), [], mode="vibes")  # type: ignore[arg-type]


def test_row_exists_and_row_absent_round_trip():
    created = make_trace(
        "ep-create",
        "Open order 7799 for customer 88 and then cancel order 7741.",
        [
            ("get_order", {"order_id": 7741}, ORDER_PRE),
            ("update_order_status", {"order_id": 7799, "status": "delivered"},
             {"order_id": 7799, "status": "delivered"}),
            ("update_order_status", {"order_id": 7741, "status": "cancelled"},
             {"order_id": 7741, "status": "cancelled"}),
            ("get_order", {"order_id": 7741}, {"error": "not_found", "order_id": 7741},
             "error", "not_found"),
        ],
        "Done.",
        True,
    )
    task, _, _ = mine_task(created, RETAIL_SCHEMA)
    assert task is not None
    v = synthesize_verifier(task, created, RETAIL_SCHEMA).model_copy(
        update={"reviewed_by": "a-human"}
    )
    kinds = {a.kind for a in v.assertions}
    assert AssertionKind.ROW_EXISTS in kinds
    assert AssertionKind.ROW_ABSENT in kinds

    good = {"orders": [{"order_id": 7799, "status": "delivered"}], "customers": [], "products": []}
    assert evaluate(v, good).passed
    stale = {"orders": [{"order_id": 7799, "status": "delivered"}, ORDER_PRE],
             "customers": [], "products": []}
    assert not evaluate(v, stale).passed  # 7741 must be gone
    missing = {"orders": [], "customers": [], "products": []}
    assert not evaluate(v, missing).passed  # 7799 must exist


# --------------------------------------------------------------------------
# anti-cheat
# --------------------------------------------------------------------------


def test_verifier_fails_when_the_store_was_written_directly():
    """The definition-of-done test. Identical final state, identical effects -
    but no refund_order call, so the store was mutated behind the tools."""
    v = refund_verifier()
    result = evaluate(v, final_state(), [Effect(tool="send_email")])
    assert result.passed  # state-wise indistinguishable from the honest rollout

    cheating = record(actions=(
        RolloutAction(tool="get_order", arguments={"order_id": 7741}, step=0),
        RolloutAction(tool="send_email", arguments={"to_customer_id": 88}, step=1),
    ))
    report = check_rollout(cheating, result)
    assert not report.clean
    guards = {f.guard for f in report.failures}
    assert "state_changed_without_write_call" in guards
    assert report.failures[0].evidence["changes"][0]["fields"] == ["status"]

    final = enforce(result, report)
    assert final.passed is False and final.reward == 0.0


def test_honest_rollout_is_clean():
    v = refund_verifier()
    result = evaluate(v, final_state(), [Effect(tool="send_email")])
    report = check_rollout(record(), result)
    assert report.clean, report.to_json()
    assert enforce(result, report).reward == 1.0


def test_write_call_naming_a_different_row_does_not_launder_the_change():
    v = refund_verifier()
    result = evaluate(v, final_state(), [Effect(tool="send_email")])
    laundering = record(actions=(
        RolloutAction(tool="refund_order", arguments={"order_id": 9999, "amount_cents": 1}, step=0),
        RolloutAction(tool="send_email", arguments={"to_customer_id": 88}, step=1),
    ))
    report = check_rollout(laundering, result)
    assert "unattributed_row_change" in {f.guard for f in report.failures}


def test_direct_store_writes_reported_by_the_runtime_are_fatal():
    report = check_rollout(record(direct_store_writes=({"sql": "UPDATE orders SET status=..."},)))
    assert "direct_store_write" in {f.guard for f in report.failures}


def test_reading_the_verifier_or_the_ledger_fails():
    for resource in ("/env/verifiers/ver-task-ep-refund-ok.json",
                     "/env/effect_ledger.jsonl",
                     "expected_state.json",
                     "tracegym/verify/run.py"):
        report = check_rollout(record(resource_reads=(resource,)))
        assert "forbidden_read" in {f.guard for f in report.failures}, resource

    sneaky = record(actions=(
        RolloutAction(tool="get_product", arguments={"sku": "../../answer_key"}, step=0),
    ))
    assert "forbidden_read" in {f.guard for f in check_rollout(sneaky).failures}


def test_assertions_passing_with_zero_tool_calls_is_flagged():
    v = refund_verifier()
    # a leaked pre-state: the env already starts refunded and an email is pre-logged
    result = evaluate(v, final_state(), [Effect(tool="send_email")])
    assert result.passed
    idle = record(actions=(), pre_state=record().final_state)
    report = check_rollout(idle, result)
    assert "assertions_passed_with_zero_tool_calls" in {f.guard for f in report.failures}
    assert enforce(result, report).reward == 0.0


def test_network_reporting_is_never_silently_assumed():
    unknown = check_rollout(record(network_calls=None))
    assert "network_guard_unverified" in {f.guard for f in unknown.findings}
    assert unknown.clean  # a warning, not a failure
    called = check_rollout(record(network_calls=("https://example.com",)))
    assert "network_access" in {f.guard for f in called.failures}


def test_report_serializes_for_a_human():
    report = check_rollout(record(direct_store_writes=({"sql": "DELETE FROM customers"},)))
    blob = report.to_json()
    assert blob["clean"] is False
    assert blob["findings"][0]["guard"] == "direct_store_write"
    assert "mutations" in blob["findings"][0]["evidence"]
