"""Human verdicts about what actually happened in a run.

A label is the only thing in this system that can settle a disagreement between
verifiers, and the only evidence a verifier can be calibrated against. It lives
outside ``verify`` on purpose: whether a run succeeded is a fact about the run,
not about whichever check happened to ask.

Labels are expensive, so they are spent where they buy the most — the runs a
family's verifiers disagree about. One label there resolves an ambiguity that
every verifier in the family shares.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import Enum

from pydantic import model_validator

from bandits.analyze.models import Evidence, EvidenceKind, Visibility
from bandits.store import DerivedEnvelope, DerivedStore
from bandits.traces import Contract


class Verdict(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"

    UNCLEAR = "unclear"
    """A human looked and could not tell.

    Kept rather than discarded: a run experts cannot adjudicate is a fact about
    the task, and silently dropping it would inflate every agreement rate that
    follows.
    """


class Label(Contract):
    label_id: str
    trace_id: str
    family_id: str
    verdict: Verdict
    labeler: str
    rationale: str = ""
    created_at: str
    prompted_by: str | None = None
    """The run whose disagreement surfaced this trace, when one did."""

    def as_evidence(self) -> Evidence:
        """A label is evidence like any other, and ranks accordingly."""
        return Evidence(
            evidence_id=f"ev-{self.label_id}",
            claim="human_label",
            value={"verdict": self.verdict.value, "rationale": self.rationale},
            visibility=Visibility.POST_HOC,
            provenance="human",
            strength="strong" if self.verdict is not Verdict.UNCLEAR else "weak",
            kind=EvidenceKind.HUMAN_LABEL,
            trace_id=self.trace_id,
        )


class LabelSet(Contract):
    schema_version: int = 1
    task_set_id: str
    family_id: str
    labels: tuple[Label, ...]

    @model_validator(mode="after")
    def validate_labels(self) -> LabelSet:
        trace_ids = [label.trace_id for label in self.labels]
        if len(trace_ids) != len(set(trace_ids)):
            raise ValueError("a label set holds at most one verdict per trace")
        for label in self.labels:
            if label.family_id != self.family_id:
                raise ValueError(f"label {label.label_id} belongs to another family")
        return self

    def verdicts(self) -> dict[str, Verdict]:
        return {label.trace_id: label.verdict for label in self.labels}

    def adjudicated(self) -> dict[str, Verdict]:
        """Only the verdicts that settle something; unclear ones cannot."""
        return {
            trace_id: verdict
            for trace_id, verdict in self.verdicts().items()
            if verdict is not Verdict.UNCLEAR
        }


def make_label(
    *,
    trace_id: str,
    family_id: str,
    verdict: Verdict,
    labeler: str,
    rationale: str = "",
    prompted_by: str | None = None,
) -> Label:
    digest = hashlib.sha256(f"{family_id}\0{trace_id}\0{labeler}".encode()).hexdigest()
    return Label(
        label_id=f"label-{digest[:16]}",
        trace_id=trace_id,
        family_id=family_id,
        verdict=verdict,
        labeler=labeler,
        rationale=rationale,
        created_at=datetime.now(UTC).isoformat(),
        prompted_by=prompted_by,
    )


def compute_label_set_id(label_set: LabelSet) -> str:
    digest = hashlib.sha256(label_set.model_dump_json().encode()).hexdigest()
    return f"labels-{digest[:16]}"


def save_label_set(label_set: LabelSet, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_label_set_id(label_set),
        kind="label_set",
        parent_artifact_id=label_set.task_set_id,
        payload=label_set.model_dump_json().encode(),
        summary={
            "labels": len(label_set.labels),
            "adjudicated": len(label_set.adjudicated()),
        },
    )


def load_label_set(label_set_id: str, store: DerivedStore) -> LabelSet:
    return LabelSet.model_validate_json(store.read_payload(label_set_id))
