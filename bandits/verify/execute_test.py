from __future__ import annotations

from bandits.analyze.models import Evidence, EvidenceKind, Visibility
from bandits.verify import (
    CheckOperator,
    CheckSpec,
    VerifierMode,
    VerifierSpec,
    execute_verifier,
)


def _evidence(claim: str, value: object) -> Evidence:
    return Evidence(
        evidence_id=f"ev-{claim}",
        claim=claim,
        value=value,
        visibility=Visibility.TERMINAL,
        provenance="observed",
        strength="strong",
        kind=EvidenceKind.STRUCTURED_EXTERNAL_RESULT,
        trace_id="trace-one",
    )


def _spec(operator: CheckOperator, claim: str, expected: object) -> VerifierSpec:
    return VerifierSpec(
        verifier_id="verifier-one",
        family_id="family-one",
        task_set_id="taskset-one",
        mode=VerifierMode.REPLAY,
        status="executable",
        inputs=(f"terminal_evidence:{claim}",),
        checks=(
            CheckSpec(
                check_id="check-one",
                claim=claim,
                operator=operator,
                expected=expected,
                supporting_evidence_ids=("ev-support",),
                description="A deterministic check.",
            ),
        ),
        unknown_when=("evidence absent",),
        blind_spots=("may be insufficient",),
        gaming_hypotheses=("fake the checked value",),
    )


def test_exit_code_check_executes() -> None:
    spec = _spec(CheckOperator.EXIT_CODE_ZERO, "command_exit_code", 0)

    passed = execute_verifier(
        spec, (_evidence("command_exit_code", {"key": "exit_code", "value": 0}),)
    )
    failed = execute_verifier(
        spec, (_evidence("command_exit_code", {"key": "exit_code", "value": 1}),)
    )

    assert passed.score == 1.0
    assert failed.score == 0.0


def test_structured_field_check_executes() -> None:
    spec = _spec(CheckOperator.EQUALS, "final_state_field:status", "refunded")
    evidence = _evidence(
        "final_state_field", {"key": "status", "value": "refunded", "tool": "refund"}
    )

    assert execute_verifier(spec, (evidence,)).score == 1.0


def test_missing_required_evidence_is_unknown() -> None:
    spec = _spec(CheckOperator.EQUALS, "final_state_field:status", "refunded")

    result = execute_verifier(spec, ())

    assert result.score is None
    assert result.subscores[0].score is None


def test_exact_output_check_executes_without_semantic_guessing() -> None:
    spec = _spec(CheckOperator.EXACT_OUTPUT, "final_output", "OK")

    assert execute_verifier(spec, (_evidence("final_output", {"output": "OK"}),)).score == 1.0
    assert execute_verifier(spec, (_evidence("final_output", {"output": "Okay"}),)).score == 0.0
