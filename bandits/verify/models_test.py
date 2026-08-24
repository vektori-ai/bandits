from __future__ import annotations

import pytest
from pydantic import ValidationError

from bandits.verify.models import (
    CheckOperator,
    CheckSpec,
    Result,
    SubScore,
    VerifierMode,
    VerifierSpec,
    VerifierStatus,
)


def _spec(**updates) -> VerifierSpec:
    values = {
        "verifier_id": "verifier-one",
        "family_id": "family-one",
        "task_set_id": "taskset-one",
        "mode": VerifierMode.REPLAY,
        "inputs": ("terminal_evidence:command_exit_code",),
        "checks": (
            CheckSpec(
                check_id="check-one",
                claim="command_exit_code",
                operator=CheckOperator.EXIT_CODE_ZERO,
                expected=0,
                supporting_evidence_ids=("ev-one",),
                description="Exit zero.",
            ),
        ),
        "unknown_when": ("exit code absent",),
        "blind_spots": ("wrong command",),
        "gaming_hypotheses": ("run true",),
    }
    values.update(updates)
    return VerifierSpec(**values)


def test_reviewed_requires_validation_and_explicit_human_acceptance() -> None:
    with pytest.raises(ValidationError, match="validation artifact"):
        _spec(status=VerifierStatus.REVIEWED)
    with pytest.raises(ValidationError, match="human acceptance"):
        _spec(status=VerifierStatus.REVIEWED, validation_artifact_id="validation-one")

    reviewed = _spec(
        status=VerifierStatus.REVIEWED,
        validation_artifact_id="validation-one",
        human_acceptance_id="acceptance-one",
    )
    assert reviewed.status is VerifierStatus.REVIEWED


def test_unknown_result_is_none_not_zero() -> None:
    result = Result(
        verifier_id="verifier-one",
        score=None,
        subscores=(SubScore(check_id="check-one", score=None),),
    )
    assert result.score is None


def test_unknown_aggregate_cannot_hide_a_known_subscore() -> None:
    with pytest.raises(ValidationError, match="unknown aggregate"):
        Result(
            verifier_id="verifier-one",
            score=None,
            subscores=(SubScore(check_id="check-one", score=0.0),),
        )


def test_replay_verifier_cannot_claim_a_live_input() -> None:
    with pytest.raises(ValidationError, match="replay verifier"):
        _spec(inputs=("live:orders_database",))
