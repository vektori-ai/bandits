"""Verifier contracts: specs first, executable code later."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import Field, model_validator

from bandits.analyze.models import EvidenceKind, kind_rank
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

    RISK_ACCEPTED = "risk_accepted"
    """An owner promoted this over evidence that said not to.

    Separate from ``reviewed`` because the two are not the same artifact. A
    verifier accepted despite a working attack, or with nothing held out to
    measure it on, must not be indistinguishable downstream from one that
    cleared every check — the override is exactly what a later reader needs to
    see, and a shared status would hide it.
    """

    REJECTED = "rejected"


_ACCEPTED_STATUSES = frozenset({VerifierStatus.REVIEWED, VerifierStatus.RISK_ACCEPTED})
"""Statuses that only a named human decision can reach."""

_MEASURED_STATUSES = frozenset({VerifierStatus.CALIBRATED}) | _ACCEPTED_STATUSES
"""Statuses that require a validation artifact behind them."""


class CheckOperator(str, Enum):
    EQUALS = "equals"
    """A named state field or recorded score equals a fixed value."""

    EXIT_CODE_ZERO = "exit_code_zero"
    EXACT_OUTPUT = "exact_output"

    STATE_INVARIANT = "state_invariant"
    """A terminal field agrees with the initial state it is derived from.

    The check a single field cannot express. A fixed value says a step
    happened; a value that still agrees with what it came from says the step
    was *correct* — that the amount returned matched the amount taken, that the
    row count after a migration matched the row count before it.
    """

    NO_SPAN_ERROR = "no_span_error"
    """Nothing in the episode reported an error."""

    RUBRIC_AT_LEAST = "rubric_at_least"
    """A model judge scored the run at or above a threshold.

    The only operator that can reach a family recording no structured state.
    Its evidence is a judgement, not an observation, and ranks accordingly.
    """


class CheckSpec(Contract):
    check_id: str
    claim: str
    operator: CheckOperator
    expected: Any = None
    weight: float = Field(default=1.0, gt=0)
    supporting_evidence_ids: tuple[str, ...]
    description: str

    evidence_kind: EvidenceKind = EvidenceKind.OBSERVED_TRACE
    """The trust class of the evidence this check reads.

    Recorded on the check so a verifier's standing can be judged from the spec
    alone, without re-deriving it from whichever traces happened to support it.
    """


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

    @property
    def rests_only_on_self_report(self) -> bool:
        """True when nothing but the agent's own claim supports this verifier."""
        return all(check.evidence_kind is EvidenceKind.AGENT_SELF_REPORT for check in self.checks)

    @property
    def weakest_evidence_kind(self) -> EvidenceKind:
        return min((check.evidence_kind for check in self.checks), key=kind_rank)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> VerifierSpec:
        if not self.checks:
            raise ValueError("a verifier must contain at least one check")
        if self.rests_only_on_self_report and self.status in _MEASURED_STATUSES:
            # The agent asserting it finished is never sufficient on its own. A
            # check reading only that may be drafted and run, so its disagreement
            # with better evidence stays visible, but it can never be promoted.
            raise ValueError(
                f"{self.status.value} requires evidence stronger than agent self-report"
            )
        if self.mode is VerifierMode.REPLAY and any(i.startswith("live:") for i in self.inputs):
            raise ValueError("a replay verifier cannot declare a live input")
        if self.status in _MEASURED_STATUSES and not self.validation_artifact_id:
            raise ValueError(f"{self.status.value} requires a validation artifact")
        if self.status in _ACCEPTED_STATUSES and not self.human_acceptance_id:
            raise ValueError(f"{self.status.value} requires explicit human acceptance")
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
        """Fail closed on the aggregate, but never at the cost of the breakdown.

        An unknown aggregate may — and should — carry known subscores. Which
        checks passed while another was unscorable is the whole record of how a
        verifier behaved, and it is what reward-hacking analysis reads.
        """
        if not self.subscores:
            raise ValueError("a result must contain at least one subscore")
        known = [part.score for part in self.subscores if part.score is not None]
        if self.score is not None and len(known) != len(self.subscores):
            raise ValueError("a known aggregate cannot contain an unknown subscore")
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

    check_id: str | None = None
    """Which check an ``expected`` answer revises. Required for that field.

    Without it a revision could only ever address one check, and answering a
    question would silently discard the rest of a composite verifier.
    """

    @model_validator(mode="after")
    def check_is_targeted(self) -> InterviewQuestion:
        if self.field == "expected" and not self.check_id:
            raise ValueError("an expected-value question must name the check it revises")
        return self


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
