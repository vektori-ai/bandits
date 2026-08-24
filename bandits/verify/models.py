"""Verifier contracts: specs first, executable code later."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import Field, model_validator

from bandits.traces import Contract


class VerifierMode(str, Enum):
    REPLAY = "replay"
    """Scores only evidence captured in a historical trace."""

    LIVE_QUERY = "live_query"
    """Queries an external source of truth and may eventually gate RL."""


class VerifierStatus(str, Enum):
    SUGGESTED = "suggested"
    EXECUTABLE = "executable"
    CALIBRATED = "calibrated"
    REVIEWED = "reviewed"
    REJECTED = "rejected"


class CheckOperator(str, Enum):
    EQUALS = "equals"
    EXIT_CODE_ZERO = "exit_code_zero"
    EXACT_OUTPUT = "exact_output"


class CheckSpec(Contract):
    check_id: str
    claim: str
    operator: CheckOperator
    expected: Any = None
    weight: float = Field(default=1.0, gt=0)
    supporting_evidence_ids: tuple[str, ...]
    description: str


class VerifierSpec(Contract):
    schema_version: int = 1
    verifier_id: str
    family_id: str
    task_set_id: str
    mode: VerifierMode
    status: VerifierStatus = VerifierStatus.SUGGESTED
    inputs: tuple[str, ...]
    checks: tuple[CheckSpec, ...]
    unknown_when: tuple[str, ...]
    blind_spots: tuple[str, ...]
    gaming_hypotheses: tuple[str, ...]
    provenance: Literal["rule", "model", "human"] = "rule"
    validation_artifact_id: str | None = None
    human_acceptance_id: str | None = None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> VerifierSpec:
        if not self.checks:
            raise ValueError("a verifier must contain at least one check")
        if self.mode is VerifierMode.REPLAY and any(i.startswith("live:") for i in self.inputs):
            raise ValueError("a replay verifier cannot declare a live input")
        if self.status in {VerifierStatus.CALIBRATED, VerifierStatus.REVIEWED}:
            if not self.validation_artifact_id:
                raise ValueError(f"{self.status.value} requires a validation artifact")
        if self.status is VerifierStatus.REVIEWED and not self.human_acceptance_id:
            raise ValueError("reviewed requires explicit human acceptance")
        return self


class SubScore(Contract):
    check_id: str
    score: float | None
    details: dict[str, Any] = {}
    evidence_ids: tuple[str, ...] = ()


class Result(Contract):
    verifier_id: str
    score: float | None
    subscores: tuple[SubScore, ...]
    details: dict[str, Any] = {}

    @model_validator(mode="after")
    def unknown_is_consistent(self) -> Result:
        known = [part.score for part in self.subscores if part.score is not None]
        if self.score is None and known:
            raise ValueError("an unknown aggregate cannot contain known subscores")
        if self.score is not None and not known:
            raise ValueError("a known aggregate requires at least one known subscore")
        return self


class VerifierDraft(Contract):
    schema_version: int = 1
    task_set_id: str
    analysis_id: str
    family_id: str
    verifiers: tuple[VerifierSpec, ...]
    unresolved: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_references(self) -> VerifierDraft:
        ids = [spec.verifier_id for spec in self.verifiers]
        if len(ids) != len(set(ids)):
            raise ValueError("verifier draft contains duplicate verifier ids")
        for spec in self.verifiers:
            if spec.family_id != self.family_id or spec.task_set_id != self.task_set_id:
                raise ValueError("verifier draft contains a spec for another family or task set")
        return self


class InterviewQuestion(Contract):
    question_id: str
    verifier_id: str
    field: Literal["expected", "blind_spots", "gaming_hypotheses"]
    prompt: str
    current_value: Any = None


class InterviewAnswer(Contract):
    question_id: str
    value: str


class VerifierInterview(Contract):
    schema_version: int = 1
    source_draft_id: str
    draft: VerifierDraft
    questions: tuple[InterviewQuestion, ...]
    answers: tuple[InterviewAnswer, ...] = ()
    next_question_index: int = 0
    complete: bool = False

    @model_validator(mode="after")
    def validate_progress(self) -> VerifierInterview:
        if self.next_question_index != len(self.answers):
            raise ValueError("interview progress must match its recorded answers")
        if self.next_question_index > len(self.questions):
            raise ValueError("interview progressed past its final question")
        if self.complete != (self.next_question_index == len(self.questions)):
            raise ValueError("interview completion flag disagrees with its progress")
        expected_ids = [q.question_id for q in self.questions[: self.next_question_index]]
        if [a.question_id for a in self.answers] != expected_ids:
            raise ValueError("answers must correspond to questions in order")
        return self
