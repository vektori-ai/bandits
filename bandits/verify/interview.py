"""Bounded, one-question-at-a-time review of deterministic verifier drafts."""

from __future__ import annotations

import hashlib
import json

from bandits.store import DerivedEnvelope, DerivedStore
from bandits.verify.models import (
    CheckOperator,
    InterviewAnswer,
    InterviewQuestion,
    VerifierDraft,
    VerifierInterview,
)

_HAS_EXPECTED_VALUE = frozenset({CheckOperator.EQUALS, CheckOperator.EXACT_OUTPUT})


def _questions(draft: VerifierDraft) -> tuple[InterviewQuestion, ...]:
    questions: list[InterviewQuestion] = []
    for spec in draft.verifiers:
        questions.extend(
            InterviewQuestion(
                question_id=f"q-{spec.verifier_id}-{check.check_id}-expected",
                verifier_id=spec.verifier_id,
                check_id=check.check_id,
                field="expected",
                current_value=check.expected,
                prompt=(
                    f"For {check.claim}, what exact value represents success? "
                    f"Press Enter to keep {check.expected!r}."
                ),
            )
            # Only checks that compare against a value have one to confirm. An
            # invariant compares two observed fields and a no-error check
            # compares nothing; asking their expected value is a question with
            # no answer, and it teaches the owner the interview is boilerplate.
            for check in spec.checks
            if check.operator in _HAS_EXPECTED_VALUE
        )
        questions.extend(
            (
                InterviewQuestion(
                    question_id=f"q-{spec.verifier_id}-blind-spot",
                    verifier_id=spec.verifier_id,
                    field="blind_spots",
                    prompt="Name one additional blind spot, or press Enter if none.",
                ),
                InterviewQuestion(
                    question_id=f"q-{spec.verifier_id}-gaming",
                    verifier_id=spec.verifier_id,
                    field="gaming_hypotheses",
                    prompt="Name one additional way this check could be gamed, or press Enter if none.",
                ),
            )
        )
    return tuple(questions)


def start_interview(draft: VerifierDraft, source_draft_id: str) -> VerifierInterview:
    questions = _questions(draft)
    return VerifierInterview(
        source_draft_id=source_draft_id,
        draft=draft,
        questions=questions,
        complete=not questions,
    )


def next_question(interview: VerifierInterview) -> InterviewQuestion | None:
    if interview.complete:
        return None
    return interview.questions[interview.next_question_index]


def _parse_expected(value: str, current: object) -> object:
    if not value.strip():
        return current
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def answer_question(interview: VerifierInterview, value: str) -> VerifierInterview:
    question = next_question(interview)
    if question is None:
        raise ValueError("interview is already complete")

    revised = []
    for spec in interview.draft.verifiers:
        if spec.verifier_id != question.verifier_id:
            revised.append(spec)
            continue
        if question.field == "expected":
            # Every check is carried through; only the one the question names is
            # revised. Rebuilding from a single check would drop the others.
            revised.append(
                spec.replace(
                    checks=tuple(
                        check.replace(expected=_parse_expected(value, check.expected))
                        if check.check_id == question.check_id
                        else check
                        for check in spec.checks
                    )
                )
            )
        else:
            additions = (value.strip(),) if value.strip() else ()
            revised.append(
                spec.replace(**{question.field: getattr(spec, question.field) + additions})
            )

    index = interview.next_question_index + 1
    return interview.replace(
        draft=interview.draft.replace(verifiers=tuple(revised)),
        answers=interview.answers
        + (InterviewAnswer(question_id=question.question_id, value=value),),
        next_question_index=index,
        complete=index == len(interview.questions),
    )


def compute_interview_id(interview: VerifierInterview) -> str:
    digest = hashlib.sha256(interview.model_dump_json().encode()).hexdigest()
    return f"verifier-interview-{digest[:16]}"


def save_interview(interview: VerifierInterview, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_interview_id(interview),
        kind="verifier_interview",
        parent_artifact_id=interview.source_draft_id,
        payload=interview.model_dump_json().encode(),
        summary={"answers": len(interview.answers), "complete": int(interview.complete)},
    )


def load_interview(interview_id: str, store: DerivedStore) -> VerifierInterview:
    return VerifierInterview.model_validate_json(store.read_payload(interview_id))
