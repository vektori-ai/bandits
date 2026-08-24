"""Propose small deterministic replay verifiers from observed terminal evidence."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from bandits.analyze.models import CorpusAnalysis, Evidence, TaskFamily, TaskSet, Visibility
from bandits.store import DerivedEnvelope, DerivedStore
from bandits.verify.models import (
    CheckOperator,
    CheckSpec,
    VerifierDraft,
    VerifierMode,
    VerifierSpec,
)


def _stable_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _spec_id(family_id: str, claim: str, expected: Any) -> str:
    payload = f"{family_id}\0{claim}\0{_stable_value(expected)}"
    return f"verifier-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _eligible(family: TaskFamily, analysis: CorpusAnalysis) -> list[Evidence]:
    trace_ids = set(family.fit_trace_ids or family.trace_ids)
    outcome_ids = {
        evidence_id
        for task in analysis.tasks
        if task.trace_id in trace_ids
        for evidence_id in task.outcome_evidence_ids
    }
    return [
        item
        for item in analysis.evidence
        if item.evidence_id in outcome_ids and item.visibility is Visibility.TERMINAL
    ]


def _proposal(
    *,
    family: TaskFamily,
    task_set_id: str,
    claim: str,
    operator: CheckOperator,
    expected: Any,
    evidence: list[Evidence],
    description: str,
    blind_spots: tuple[str, ...],
    gaming: tuple[str, ...],
) -> VerifierSpec:
    verifier_id = _spec_id(family.family_id, claim, expected)
    return VerifierSpec(
        verifier_id=verifier_id,
        family_id=family.family_id,
        task_set_id=task_set_id,
        mode=VerifierMode.REPLAY,
        inputs=(f"terminal_evidence:{claim}",),
        checks=(
            CheckSpec(
                check_id=f"check-{verifier_id.removeprefix('verifier-')}",
                claim=claim,
                operator=operator,
                expected=expected,
                supporting_evidence_ids=tuple(sorted(e.evidence_id for e in evidence)),
                description=description,
            ),
        ),
        unknown_when=(f"terminal evidence for {claim!r} is absent",),
        blind_spots=blind_spots,
        gaming_hypotheses=gaming,
    )


def draft_verifiers(
    task_set: TaskSet,
    task_set_id: str,
    analysis: CorpusAnalysis,
    family_id: str,
    *,
    limit: int = 3,
) -> VerifierDraft:
    """Draft up to ``limit`` independent checks, without claiming they establish success."""
    family = task_set.family_by_id().get(family_id)
    if family is None:
        raise ValueError(f"unknown family id: {family_id!r}")
    if task_set.corpus_id != analysis.corpus_id:
        raise ValueError("task set and analysis refer to different corpora")
    if limit < 1:
        raise ValueError("limit must be at least 1")

    evidence = _eligible(family, analysis)
    proposals: list[VerifierSpec] = []

    exits = [item for item in evidence if item.claim == "command_exit_code"]
    if exits:
        proposals.append(
            _proposal(
                family=family,
                task_set_id=task_set_id,
                claim="command_exit_code",
                operator=CheckOperator.EXIT_CODE_ZERO,
                expected=0,
                evidence=exits,
                description="Require the recorded terminal command to exit successfully.",
                blind_spots=("A zero exit code does not prove the command tested the intended behavior.",),
                gaming=("Run a harmless command that exits zero instead of the required check.",),
            )
        )

    states = [item for item in evidence if item.claim == "final_state_field"]
    grouped: dict[tuple[str, str], list[Evidence]] = {}
    for item in states:
        key = str(item.value.get("key"))
        value = item.value.get("value")
        grouped.setdefault((key, _stable_value(value)), []).append(item)
    for (key, _), support in sorted(
        grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])
    ):
        expected = support[0].value.get("value")
        proposals.append(
            _proposal(
                family=family,
                task_set_id=task_set_id,
                claim=f"final_state_field:{key}",
                operator=CheckOperator.EQUALS,
                expected=expected,
                evidence=support,
                description=f"Require terminal field {key!r} to equal the observed value {expected!r}.",
                blind_spots=("The field may be necessary but not sufficient for task success.",),
                gaming=(f"Set {key!r} without completing the other intended state changes.",),
            )
        )

    unresolved: list[str] = []
    if not proposals:
        unresolved.append(
            "no deterministic terminal exit code or structured final-state field was recorded"
        )
    elif len(proposals) == 1:
        unresolved.append("only one independent deterministic verifier pattern was supported")

    return VerifierDraft(
        task_set_id=task_set_id,
        analysis_id=task_set.analysis_id,
        family_id=family_id,
        verifiers=tuple(proposals[:limit]),
        unresolved=tuple(unresolved),
    )


def compute_verifier_draft_id(draft: VerifierDraft) -> str:
    digest = hashlib.sha256(draft.model_dump_json().encode()).hexdigest()
    return f"verifier-draft-{digest[:16]}"


def save_verifier_draft(draft: VerifierDraft, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_verifier_draft_id(draft),
        kind="verifier_draft",
        parent_artifact_id=draft.task_set_id,
        payload=draft.model_dump_json().encode(),
        summary={"verifiers": len(draft.verifiers), "unresolved": len(draft.unresolved)},
    )


def load_verifier_draft(draft_id: str, store: DerivedStore) -> VerifierDraft:
    return VerifierDraft.model_validate_json(store.read_payload(draft_id))
