"""Execute deterministic replay verifier specs against recorded evidence."""

from __future__ import annotations

from bandits.analyze.models import Evidence
from bandits.verify.models import CheckOperator, CheckSpec, Result, SubScore, VerifierSpec


def _matches(check: CheckSpec, evidence: Evidence) -> bool:
    if check.operator is CheckOperator.EXIT_CODE_ZERO:
        return evidence.claim == "command_exit_code" and evidence.value.get("value") == 0
    if check.operator is CheckOperator.EXACT_OUTPUT:
        return evidence.claim == "final_output" and evidence.value.get("output") == check.expected
    if check.operator is CheckOperator.EQUALS and check.claim.startswith("final_state_field:"):
        key = check.claim.partition(":")[2]
        return (
            evidence.claim == "final_state_field"
            and evidence.value.get("key") == key
            and evidence.value.get("value") == check.expected
        )
    return False


def _relevant(check: CheckSpec, evidence: Evidence) -> bool:
    if check.operator is CheckOperator.EXIT_CODE_ZERO:
        return evidence.claim == "command_exit_code"
    if check.operator is CheckOperator.EXACT_OUTPUT:
        return evidence.claim == "final_output"
    if check.operator is CheckOperator.EQUALS and check.claim.startswith("final_state_field:"):
        key = check.claim.partition(":")[2]
        return evidence.claim == "final_state_field" and evidence.value.get("key") == key
    return False


def execute_verifier(spec: VerifierSpec, evidence: tuple[Evidence, ...]) -> Result:
    """Score recorded evidence; absent required evidence produces unknown, never failure."""
    subscores: list[SubScore] = []
    for check in spec.checks:
        relevant = tuple(item for item in evidence if _relevant(check, item))
        if not relevant:
            subscores.append(
                SubScore(
                    check_id=check.check_id,
                    score=None,
                    details={"reason": "required terminal evidence is absent"},
                )
            )
            continue
        passed = any(_matches(check, item) for item in relevant)
        subscores.append(
            SubScore(
                check_id=check.check_id,
                score=1.0 if passed else 0.0,
                evidence_ids=tuple(item.evidence_id for item in relevant),
                details={"matched": passed},
            )
        )

    if any(part.score is None for part in subscores):
        # A composite verifier fails closed when any required check is unavailable.
        return Result(
            verifier_id=spec.verifier_id,
            score=None,
            subscores=tuple(
                part.model_copy(update={"score": None}) if part.score is not None else part
                for part in subscores
            ),
            details={"reason": "one or more required checks are unknown"},
        )
    total_weight = sum(check.weight for check in spec.checks)
    score = (
        sum(part.score * check.weight for part, check in zip(subscores, spec.checks, strict=True))
        / total_weight
        if total_weight
        else None
    )
    return Result(verifier_id=spec.verifier_id, score=score, subscores=tuple(subscores))
