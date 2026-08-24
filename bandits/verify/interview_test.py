from __future__ import annotations

import pytest

from bandits.store import DerivedStore
from bandits.verify import (
    CheckOperator,
    CheckSpec,
    VerifierDraft,
    VerifierMode,
    VerifierSpec,
    VerifierStatus,
    answer_question,
    load_interview,
    next_question,
    save_interview,
    start_interview,
)


def _draft() -> VerifierDraft:
    spec = VerifierSpec(
        verifier_id="verifier-one",
        family_id="family-one",
        task_set_id="taskset-one",
        mode=VerifierMode.REPLAY,
        inputs=("terminal_evidence:final_state_field:status",),
        checks=(
            CheckSpec(
                check_id="check-one",
                claim="final_state_field:status",
                operator=CheckOperator.EQUALS,
                expected="done",
                supporting_evidence_ids=("ev-one",),
                description="Status is done.",
            ),
        ),
        unknown_when=("status missing",),
        blind_spots=("status alone may be insufficient",),
        gaming_hypotheses=("set status directly",),
    )
    return VerifierDraft(
        task_set_id="taskset-one",
        analysis_id="analysis-one",
        family_id="family-one",
        verifiers=(spec,),
    )


def test_interview_is_bounded_and_asks_one_question_at_a_time() -> None:
    interview = start_interview(_draft(), "draft-one")

    assert len(interview.questions) == 3
    assert next_question(interview).field == "expected"
    interview = answer_question(interview, '"completed"')
    assert next_question(interview).field == "blind_spots"
    interview = answer_question(interview, "amount may be wrong")
    assert next_question(interview).field == "gaming_hypotheses"
    interview = answer_question(interview, "write status without refunding")

    assert interview.complete
    assert next_question(interview) is None
    assert interview.draft.verifiers[0].checks[0].expected == "completed"


def test_revision_changes_only_the_answer_target() -> None:
    original = _draft()
    interview = answer_question(start_interview(original, "draft-one"), '"completed"')
    before = original.verifiers[0]
    after = interview.draft.verifiers[0]

    assert after.checks[0].expected == "completed"
    assert after.blind_spots == before.blind_spots
    assert after.gaming_hypotheses == before.gaming_hypotheses
    assert after.status is VerifierStatus.SUGGESTED


def test_completed_interview_refuses_extra_answers() -> None:
    interview = start_interview(_draft(), "draft-one")
    for _ in range(3):
        interview = answer_question(interview, "")

    with pytest.raises(ValueError, match="already complete"):
        answer_question(interview, "extra")


def test_interview_round_trips_with_draft_as_parent(tmp_path) -> None:
    interview = start_interview(_draft(), "draft-one")
    for _ in range(3):
        interview = answer_question(interview, "")
    store = DerivedStore(tmp_path / ".bandits")

    envelope = save_interview(interview, store)

    assert envelope.parent_artifact_id == "draft-one"
    assert load_interview(envelope.artifact_id, store) == interview
