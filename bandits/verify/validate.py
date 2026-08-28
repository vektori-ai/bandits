"""Calibrate drafted verifiers against human labels, and try to game them.

Two questions, answered separately:

Does the check agree with people who know the task? Measured out-of-fold — the
fit side is what the check was drafted from, so its agreement there is a
training-set number and is reported as such. The held-out side is the honest one.

Can the check be satisfied without doing the task? Answered by constructing
evidence that satisfies it and nothing else, then running it. A verifier with a
measured gameability result outranks one carrying a prose warning.
"""

from __future__ import annotations

import hashlib

from bandits.analyze.models import Evidence, EvidenceKind, TaskSet, Visibility
from bandits.labels import LabelSet, Verdict
from bandits.store import DerivedEnvelope, DerivedStore
from bandits.traces import Contract
from bandits.verify.execute import execute_verifier
from bandits.verify.models import (
    CheckOperator,
    CheckSpec,
    VerifierDraft,
    VerifierSpec,
    VerifierStatus,
)

DEFAULT_SUCCESS_THRESHOLD = 0.5
"""At or above this, a verifier is taken to be claiming success."""

_MAX_COUNTEREXAMPLES = 10


class Counterexample(Contract):
    """One run where a verifier and a human disagreed."""

    trace_id: str
    verifier_score: float | None
    human_verdict: str
    rationale: str = ""

    kind: str = "false_positive"
    """``false_positive`` when the check passed a run a human failed.

    Named because the two errors are not equally dangerous. A false negative
    withholds credit for real work; a false positive rewards a run that did not
    do the task, and is the one that trains the wrong behaviour.
    """


class Agreement(Contract):
    verifier_id: str
    split: str
    """``fit`` or ``held_out``. Only held_out is an honest estimate."""

    labeled: int
    agreed: int
    disagreed: int
    unscored: int
    """Labeled runs the verifier could not score. Never counted as disagreement."""

    agreement: float | None = None
    """None when nothing was both labeled and scorable — not zero."""

    counterexamples: tuple[Counterexample, ...] = ()


class GameabilityResult(Contract):
    """Whether a constructed run satisfying the check actually passed it."""

    verifier_id: str
    hypothesis: str
    constructed: dict[str, object]
    passed: bool
    """True means the attack worked: the check accepted a run that did not do the task."""

    forged_facts: int
    """How many facts had to be fabricated for the attack to land.

    Reported because "gameable" alone flattens a real difference: writing one
    status field is something a careless tool can do by itself, while forging a
    before/after pair requires faking the read the action was based on too. A
    check that costs more to game is a better check, and a bare pass/fail hides
    exactly that.
    """


class Validation(Contract):
    schema_version: int = 1
    source_draft_id: str
    family_id: str
    label_set_id: str
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD
    agreements: tuple[Agreement, ...] = ()
    gameability: tuple[GameabilityResult, ...] = ()
    labels_used: int = 0
    unclear_labels: int = 0
    limitations: tuple[str, ...] = ()

    def held_out(self, verifier_id: str) -> Agreement | None:
        return next(
            (a for a in self.agreements if a.verifier_id == verifier_id and a.split == "held_out"),
            None,
        )


def _claims_success(score: float | None, threshold: float) -> bool | None:
    return None if score is None else score >= threshold


def _agreement(
    *,
    spec: VerifierSpec,
    split: str,
    trace_ids: tuple[str, ...],
    evidence_by_trace: dict[str, tuple[Evidence, ...]],
    verdicts: dict[str, Verdict],
    threshold: float,
) -> Agreement:
    agreed = disagreed = unscored = 0
    counterexamples: list[Counterexample] = []

    for trace_id in trace_ids:
        verdict = verdicts.get(trace_id)
        if verdict is None:
            continue
        score = execute_verifier(spec, evidence_by_trace.get(trace_id, ())).score
        claim = _claims_success(score, threshold)
        if claim is None:
            unscored += 1
            continue
        if claim is (verdict is Verdict.SUCCESS):
            agreed += 1
        else:
            disagreed += 1
            counterexamples.append(
                Counterexample(
                    trace_id=trace_id,
                    verifier_score=score,
                    human_verdict=verdict.value,
                    kind="false_positive" if claim else "false_negative",
                )
            )

    scored = agreed + disagreed
    return Agreement(
        verifier_id=spec.verifier_id,
        split=split,
        labeled=sum(1 for trace_id in trace_ids if trace_id in verdicts),
        agreed=agreed,
        disagreed=disagreed,
        unscored=unscored,
        agreement=(agreed / scored) if scored else None,
        # A false positive — the check passing a run a human failed — is the one
        # that would reward the wrong behavior, so it is shown first.
        counterexamples=tuple(
            sorted(counterexamples, key=lambda c: (c.kind != "false_positive", c.trace_id))
        )[:_MAX_COUNTEREXAMPLES],
    )


def _forged(claim: str, key: str, value: object) -> Evidence:
    return Evidence(
        evidence_id=f"ev-forged-{claim}-{key}",
        claim=claim,
        value={"key": key, "value": value, "tool": "forged"},
        visibility=Visibility.TERMINAL,
        provenance="derived",
        strength="weak",
        kind=EvidenceKind.STRUCTURED_EXTERNAL_RESULT,
        trace_id="forged",
    )


def _attack(check: CheckSpec) -> tuple[str, tuple[Evidence, ...], dict[str, object]] | None:
    """Build the least work that satisfies one check, and nothing more."""
    if check.operator is CheckOperator.EQUALS and check.claim.startswith("final_state_field:"):
        key = check.claim.partition(":")[2]
        return (
            f"Write {key!r} directly without performing the action.",
            (_forged("final_state_field", key, check.expected),),
            {key: check.expected},
        )

    if check.operator is CheckOperator.STATE_INVARIANT:
        final_key, _, initial_key = check.claim.partition(":")[2].partition("==")
        # Both sides forged to the same value: the invariant holds while nothing
        # was actually transferred.
        return (
            f"Copy {initial_key!r} into {final_key!r} without performing the action.",
            (
                _forged("initial_state_field", initial_key, 1),
                _forged("final_state_field", final_key, 1),
            ),
            {final_key: 1, initial_key: 1},
        )

    if check.operator is CheckOperator.EXIT_CODE_ZERO:
        return (
            "Run a trivially passing command instead of the required one.",
            (
                Evidence(
                    evidence_id="ev-forged-exit",
                    claim="command_exit_code",
                    value={"key": "exit_code", "value": 0, "tool": "true"},
                    visibility=Visibility.TERMINAL,
                    provenance="derived",
                    strength="weak",
                    kind=EvidenceKind.STRUCTURED_EXTERNAL_RESULT,
                    trace_id="forged",
                ),
            ),
            {"exit_code": 0, "tool": "true"},
        )

    if check.operator is CheckOperator.NO_SPAN_ERROR:
        return (
            "Swallow the error and report success without it reaching the trace.",
            (
                Evidence(
                    evidence_id="ev-forged-anchor",
                    claim="episode_span_count",
                    value=1,
                    visibility=Visibility.POST_HOC,
                    provenance="derived",
                    strength="weak",
                    kind=EvidenceKind.OBSERVED_TRACE,
                    trace_id="forged",
                ),
            ),
            {"span_error": "suppressed"},
        )

    if check.operator is CheckOperator.EXACT_OUTPUT:
        return (
            "Emit the expected text without completing the underlying task.",
            (
                Evidence(
                    evidence_id="ev-forged-output",
                    claim="final_output",
                    value={"output": check.expected},
                    visibility=Visibility.TERMINAL,
                    provenance="derived",
                    strength="weak",
                    kind=EvidenceKind.AGENT_SELF_REPORT,
                    trace_id="forged",
                ),
            ),
            {"output": check.expected},
        )

    return None


def probe_gameability(spec: VerifierSpec) -> list[GameabilityResult]:
    """Construct a run that satisfies each check without doing the task, and score it."""
    results: list[GameabilityResult] = []
    for check in spec.checks:
        attack = _attack(check)
        if attack is None:
            continue
        hypothesis, evidence, constructed = attack
        score = execute_verifier(spec.replace(checks=(check,)), evidence).score
        results.append(
            GameabilityResult(
                verifier_id=spec.verifier_id,
                hypothesis=hypothesis,
                constructed=constructed,
                passed=score is not None and score >= DEFAULT_SUCCESS_THRESHOLD,
                forged_facts=len(evidence),
            )
        )
    return results


def validate_draft(
    draft: VerifierDraft,
    draft_id: str,
    task_set: TaskSet,
    evidence: tuple[Evidence, ...],
    label_set: LabelSet,
    label_set_id: str,
    *,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
) -> Validation:
    """Measure each verifier against labels on both splits, then try to game it."""
    family = task_set.family_by_id().get(draft.family_id)
    if family is None:
        raise ValueError(f"unknown family id: {draft.family_id!r}")

    by_trace: dict[str, list[Evidence]] = {}
    for item in evidence:
        by_trace.setdefault(item.trace_id, []).append(item)
    evidence_by_trace = {key: tuple(value) for key, value in by_trace.items()}
    verdicts = label_set.adjudicated()

    agreements: list[Agreement] = []
    gameability: list[GameabilityResult] = []
    for spec in draft.verifiers:
        for split, trace_ids in (
            ("fit", family.fit_trace_ids),
            ("held_out", family.held_out_trace_ids),
        ):
            agreements.append(
                _agreement(
                    spec=spec,
                    split=split,
                    trace_ids=trace_ids,
                    evidence_by_trace=evidence_by_trace,
                    verdicts=verdicts,
                    threshold=success_threshold,
                )
            )
        gameability.extend(probe_gameability(spec))

    limitations: list[str] = []
    if not any(a.split == "held_out" and a.labeled for a in agreements):
        limitations.append(
            "no held-out trace carries a label; every agreement below is measured on "
            "the traces the checks were drafted from and is not an honest estimate"
        )
    unclear = len(label_set.labels) - len(verdicts)
    if unclear:
        limitations.append(
            f"{unclear} labeled run(s) a human could not adjudicate are excluded from "
            "every rate above rather than counted as either outcome"
        )

    return Validation(
        source_draft_id=draft_id,
        family_id=draft.family_id,
        label_set_id=label_set_id,
        success_threshold=success_threshold,
        agreements=tuple(agreements),
        gameability=tuple(gameability),
        labels_used=len(verdicts),
        unclear_labels=unclear,
        limitations=tuple(limitations),
    )


def calibrate(spec: VerifierSpec, validation_id: str) -> VerifierSpec:
    """Promote a verifier that has been measured. Never past calibrated."""
    return spec.replace(status=VerifierStatus.CALIBRATED, validation_artifact_id=validation_id)


def accept(spec: VerifierSpec, acceptance_id: str) -> VerifierSpec:
    """Record a human owner accepting a calibrated verifier."""
    if spec.status is not VerifierStatus.CALIBRATED:
        raise ValueError("only a calibrated verifier can be accepted")
    return spec.replace(status=VerifierStatus.REVIEWED, human_acceptance_id=acceptance_id)


def compute_validation_id(validation: Validation) -> str:
    digest = hashlib.sha256(validation.model_dump_json().encode()).hexdigest()
    return f"validation-{digest[:16]}"


def save_validation(validation: Validation, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_validation_id(validation),
        kind="validation",
        parent_artifact_id=validation.source_draft_id,
        payload=validation.model_dump_json().encode(),
        summary={
            "agreements": len(validation.agreements),
            "gameable": sum(1 for g in validation.gameability if g.passed),
            "labels_used": validation.labels_used,
        },
    )


def load_validation(validation_id: str, store: DerivedStore) -> Validation:
    return Validation.model_validate_json(store.read_payload(validation_id))
