from __future__ import annotations

import pytest
from pydantic import ValidationError

from bandits.verify.models import (
    CheckOperator,
    CheckSpec,
    VerifierDraft,
    VerifierMode,
    VerifierSpec,
    VerifierStatus,
)
from bandits.verify.review import (
    PromotionBlocker,
    assess_promotion,
    load_reviewed_verifier,
    review_verifier,
    save_reviewed_verifier,
)
from bandits.verify.validate import Agreement, GameabilityResult, Validation


def _inputs():
    spec = VerifierSpec(
        verifier_id="v1",
        family_id="f1",
        task_set_id="ts1",
        mode=VerifierMode.REPLAY,
        status=VerifierStatus.EXECUTABLE,
        inputs=("terminal_evidence:command_exit_code",),
        checks=(
            CheckSpec(
                check_id="c1",
                claim="command_exit_code",
                operator=CheckOperator.EXIT_CODE_ZERO,
                expected=0,
                supporting_evidence_ids=("e1",),
                description="exit zero",
            ),
        ),
        unknown_when=("missing exit",),
        blind_spots=("wrong command",),
        gaming_hypotheses=("run true",),
    )
    draft = VerifierDraft(task_set_id="ts1", analysis_id="a1", family_id="f1", verifiers=(spec,))
    validation = Validation(
        source_draft_id="d1",
        family_id="f1",
        label_set_id="l1",
        agreements=(
            Agreement(
                verifier_id="v1",
                split="held_out",
                labeled=1,
                agreed=1,
                disagreed=0,
                unscored=0,
                agreement=1,
            ),
        ),
        labels_used=2,
        success_labels=1,
        failure_labels=1,
    )
    return draft, validation


def test_review_requires_matching_validation_and_explicit_acceptance() -> None:
    draft, validation = _inputs()
    with pytest.raises(ValueError, match="must not be empty"):
        review_verifier(draft, "d1", validation, "val1", "v1", " ")
    with pytest.raises(ValueError, match="another verifier draft"):
        review_verifier(draft, "wrong", validation, "val1", "v1", "owner-1")

    unmeasured = validation.replace(agreements=())
    with pytest.raises(ValueError, match="no validation measurements"):
        review_verifier(draft, "d1", unmeasured, "val1", "v1", "owner-1")


def test_review_round_trips_as_immutable_derived_artifact(tmp_path) -> None:
    from bandits.store import DerivedStore

    draft, validation = _inputs()
    reviewed = review_verifier(draft, "d1", validation, "val1", "v1", "owner-1")
    store = DerivedStore(tmp_path / ".bandits")

    envelope = save_reviewed_verifier(reviewed, store)

    assert load_reviewed_verifier(envelope.artifact_id, store) == reviewed
    assert reviewed.spec.status is VerifierStatus.REVIEWED


def _validation(**overrides) -> Validation:
    _, validation = _inputs()
    return validation.replace(**overrides)


def test_a_measurement_that_measured_nothing_blocks_promotion() -> None:
    """An Agreement existing is not the same as that Agreement saying anything."""
    empty = Agreement(verifier_id="v1", split="fit", labeled=0, agreed=0, disagreed=0, unscored=0)

    blockers = assess_promotion(_validation(agreements=(empty,)), "v1")

    assert {item.code for item in blockers} == {"no_held_out_measurement"}


def test_all_held_out_runs_unscorable_blocks_promotion() -> None:
    blind = Agreement(
        verifier_id="v1", split="held_out", labeled=3, agreed=0, disagreed=0, unscored=3
    )

    blockers = assess_promotion(_validation(agreements=(blind,)), "v1")

    assert {item.code for item in blockers} == {"no_scorable_held_out"}


def test_labels_of_one_verdict_block_promotion() -> None:
    blockers = assess_promotion(
        _validation(labels_used=4, success_labels=4, failure_labels=0), "v1"
    )

    assert {item.code for item in blockers} == {"single_verdict_labels"}


def test_a_working_attack_blocks_promotion() -> None:
    gamed = GameabilityResult(
        verifier_id="v1",
        hypothesis="Write the status field directly.",
        constructed={"status": "refunded"},
        passed=True,
        forged_facts=1,
    )

    blockers = assess_promotion(_validation(gameability=(gamed,)), "v1")

    assert {item.code for item in blockers} == {"gameable"}


def test_sufficient_evidence_promotes_cleanly() -> None:
    draft, validation = _inputs()

    reviewed = review_verifier(draft, "d1", validation, "val-1", "v1", "ticket-7")

    assert reviewed.spec.status is VerifierStatus.REVIEWED
    assert not reviewed.accepted_risks
    assert reviewed.success_threshold == validation.success_threshold


def test_review_refuses_insufficient_evidence_by_default() -> None:
    draft, _ = _inputs()
    weak = _validation(success_labels=1, failure_labels=0)

    with pytest.raises(ValueError, match="single_verdict_labels"):
        review_verifier(draft, "d1", weak, "val-1", "v1", "ticket-7")


def test_an_override_is_a_different_artifact_not_the_same_one() -> None:
    """An owner going against the evidence must not look like one who did not."""
    draft, _ = _inputs()
    weak = _validation(success_labels=1, failure_labels=0)

    reviewed = review_verifier(draft, "d1", weak, "val-1", "v1", "ticket-7", accept_risks=True)

    assert reviewed.spec.status is VerifierStatus.RISK_ACCEPTED
    assert [item.code for item in reviewed.accepted_risks] == ["single_verdict_labels"]


def test_a_risk_accepted_spec_cannot_pose_as_an_ordinary_review() -> None:
    draft, validation = _inputs()
    reviewed = review_verifier(draft, "d1", validation, "val-1", "v1", "ticket-7")

    with pytest.raises(ValidationError, match="cannot carry accepted risks"):
        reviewed.replace(accepted_risks=(PromotionBlocker(code="gameable", detail="x"),))
