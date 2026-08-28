"""Run drafted verifiers over the traces they were drafted from.

A drafted check is a hypothesis. Until it has been run, an owner asked to review
it is reviewing prose. This module produces the evidence that makes the review
concrete: what each verifier said about each historical run, and — more usefully
— the runs where two verifiers for the same family disagree.

Disagreement is where human labeling pays for itself. A trace all k verifiers
agree on teaches nothing; a trace they split on is exactly one ambiguity, and
resolving it moves every verifier at once.
"""

from __future__ import annotations

import hashlib

from bandits.analyze.models import CorpusAnalysis, TaskSet
from bandits.store import DerivedEnvelope, DerivedStore
from bandits.traces import Contract
from bandits.verify.execute import execute_verifier
from bandits.verify.models import Result, VerifierDraft


class TraceOutcome(Contract):
    """What one verifier said about one historical run."""

    trace_id: str
    verifier_id: str
    result: Result


class Disagreement(Contract):
    """One run the family's verifiers do not agree about."""

    trace_id: str
    scores: dict[str, float | None]
    """Aggregate score per verifier id. None is unknown, never zero."""

    kind: str
    """``split`` when they disagree on a score, ``coverage`` when some cannot score it."""


class DraftRun(Contract):
    schema_version: int = 1
    source_draft_id: str
    family_id: str
    outcomes: tuple[TraceOutcome, ...]
    disagreements: tuple[Disagreement, ...] = ()
    unscorable_trace_ids: tuple[str, ...] = ()
    """Runs no verifier could score at all. Reported, never counted as failures."""

    def scores_for(self, trace_id: str) -> dict[str, float | None]:
        return {o.verifier_id: o.result.score for o in self.outcomes if o.trace_id == trace_id}


def run_draft(draft: VerifierDraft, analysis: CorpusAnalysis, task_set: TaskSet) -> DraftRun:
    """Score every fit trace in the family with every verifier in the draft."""
    family = task_set.family_by_id().get(draft.family_id)
    if family is None:
        raise ValueError(f"unknown family id: {draft.family_id!r}")

    # Fit only. Scoring the held-out side here would spend the split before
    # calibration ever gets to use it.
    trace_ids = family.fit_trace_ids or family.trace_ids
    by_trace: dict[str, list] = {}
    for item in analysis.evidence:
        by_trace.setdefault(item.trace_id, []).append(item)

    outcomes: list[TraceOutcome] = []
    disagreements: list[Disagreement] = []
    unscorable: list[str] = []

    for trace_id in trace_ids:
        evidence = tuple(by_trace.get(trace_id, ()))
        scores: dict[str, float | None] = {}
        for spec in draft.verifiers:
            result = execute_verifier(spec, evidence)
            outcomes.append(
                TraceOutcome(trace_id=trace_id, verifier_id=spec.verifier_id, result=result)
            )
            scores[spec.verifier_id] = result.score

        known = {score for score in scores.values() if score is not None}
        if not known:
            unscorable.append(trace_id)
        elif len(known) > 1:
            disagreements.append(Disagreement(trace_id=trace_id, scores=scores, kind="split"))
        elif any(score is None for score in scores.values()):
            # Not a contradiction, but still worth a label: one check sees the
            # run and another is blind to it, and only a human can say which
            # blindness matters.
            disagreements.append(Disagreement(trace_id=trace_id, scores=scores, kind="coverage"))

    return DraftRun(
        source_draft_id=compute_run_source_id(draft),
        family_id=draft.family_id,
        outcomes=tuple(outcomes),
        disagreements=tuple(disagreements),
        unscorable_trace_ids=tuple(unscorable),
    )


def compute_run_source_id(draft: VerifierDraft) -> str:
    digest = hashlib.sha256(draft.model_dump_json().encode()).hexdigest()
    return f"verifier-draft-{digest[:16]}"


def compute_run_id(run: DraftRun) -> str:
    digest = hashlib.sha256(run.model_dump_json().encode()).hexdigest()
    return f"verifier-run-{digest[:16]}"


def save_draft_run(run: DraftRun, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_run_id(run),
        kind="verifier_run",
        parent_artifact_id=run.source_draft_id,
        payload=run.model_dump_json().encode(),
        summary={
            "outcomes": len(run.outcomes),
            "disagreements": len(run.disagreements),
            "unscorable": len(run.unscorable_trace_ids),
        },
    )


def load_draft_run(run_id: str, store: DerivedStore) -> DraftRun:
    return DraftRun.model_validate_json(store.read_payload(run_id))
