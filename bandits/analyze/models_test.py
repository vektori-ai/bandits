from __future__ import annotations

import pytest

from bandits.analyze.models import (
    Evidence,
    LeakageError,
    Visibility,
    build_task_candidate,
    evidence_id,
)


def _evidence(visibility: Visibility, claim: str = "instruction") -> Evidence:
    return Evidence(
        evidence_id=evidence_id(trace_id="trace-1", claim=claim),
        claim=claim,
        value="Refund order 7741",
        visibility=visibility,
        provenance="observed",
        strength="strong",
        trace_id="trace-1",
    )


def test_prompt_built_from_at_start_evidence_is_accepted() -> None:
    task = build_task_candidate(
        task_id="task-trace-1",
        trace_id="trace-1",
        instruction="Refund order 7741",
        prompt_evidence=(_evidence(Visibility.AT_START),),
        trajectory_span_ids=("span-1",),
        terminal_span_ids=("span-2",),
        outcome_evidence=(),
    )
    assert task.prompt_evidence_ids == ("ev-trace-1-trace-instruction",)


@pytest.mark.parametrize(
    "visibility", [Visibility.DURING, Visibility.TERMINAL, Visibility.POST_HOC]
)
def test_prompt_built_from_later_evidence_raises(visibility: Visibility) -> None:
    with pytest.raises(LeakageError, match="not knowable at t=0"):
        build_task_candidate(
            task_id="task-trace-1",
            trace_id="trace-1",
            instruction="Refund order 7741",
            prompt_evidence=(_evidence(visibility, claim="final_state_field"),),
            trajectory_span_ids=(),
            terminal_span_ids=(),
            outcome_evidence=(),
        )


def test_evidence_ids_are_stable_for_the_same_fact() -> None:
    assert evidence_id(trace_id="trace-1", claim="command_exit_code", span_id="span-2") == (
        "ev-trace-1-span-2-command-exit-code"
    )


def test_strength_ranks_order_conflicting_evidence() -> None:
    strong = _evidence(Visibility.TERMINAL).model_copy(update={"strength": "strong"})
    weak = _evidence(Visibility.TERMINAL).model_copy(update={"strength": "weak"})
    assert strong.strength_rank > weak.strength_rank
