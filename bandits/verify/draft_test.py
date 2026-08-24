from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from bandits.analyze import analyze_corpus, compute_analysis_id, mine_task_set
from bandits.ingest import load_corpus
from bandits.store import DerivedStore
from bandits.traces import Span, SpanKind, Trace, TraceCorpus
from bandits.verify import (
    VerifierMode,
    VerifierStatus,
    draft_verifiers,
    load_verifier_draft,
    save_verifier_draft,
)

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _task_set(path: Path):
    analysis = analyze_corpus(load_corpus(path, "otlp"))
    analysis_id = compute_analysis_id(analysis)
    return analysis, mine_task_set(analysis, analysis_id, budget=10)


def test_coding_family_proposes_exit_code_check() -> None:
    analysis, task_set = _task_set(FIXTURES / "traces.coding.otlp.jsonl")
    family = next(f for f in task_set.families if "parser" in f.descriptor)

    draft = draft_verifiers(task_set, "taskset-test", analysis, family.family_id)

    spec = next(v for v in draft.verifiers if v.checks[0].claim == "command_exit_code")
    assert spec.mode is VerifierMode.REPLAY
    assert spec.status is VerifierStatus.EXECUTABLE
    assert spec.checks[0].expected == 0
    assert spec.blind_spots
    assert spec.gaming_hypotheses


def test_support_family_proposes_structured_state_check() -> None:
    analysis, task_set = _task_set(FIXTURES / "traces.support.otlp.jsonl")
    family = next(f for f in task_set.families if "refund" in f.descriptor)

    draft = draft_verifiers(task_set, "taskset-test", analysis, family.family_id)

    checks = [v.checks[0] for v in draft.verifiers]
    assert any(check.claim == "final_state_field:status" for check in checks)
    assert all(check.supporting_evidence_ids for check in checks)


def test_no_deterministic_signal_is_explicitly_unresolved() -> None:
    analysis, task_set = _task_set(FIXTURES / "traces.support.otlp.jsonl")
    family = next(f for f in task_set.families if "help" in f.descriptor)

    draft = draft_verifiers(task_set, "taskset-test", analysis, family.family_id)

    assert draft.verifiers == ()
    assert any("no deterministic" in reason for reason in draft.unresolved)


def test_draft_round_trips_with_task_set_as_parent(tmp_path) -> None:
    analysis, task_set = _task_set(FIXTURES / "traces.support.otlp.jsonl")
    family = next(f for f in task_set.families if "refund" in f.descriptor)
    draft = draft_verifiers(task_set, "taskset-test", analysis, family.family_id)
    store = DerivedStore(tmp_path / ".bandits")

    envelope = save_verifier_draft(draft, store)

    assert envelope.parent_artifact_id == "taskset-test"
    assert envelope.kind == "verifier_draft"
    assert load_verifier_draft(envelope.artifact_id, store) == draft


def test_exact_output_is_proposed_only_when_the_instruction_requires_exact_text() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    corpus = TraceCorpus(
        source="test",
        traces=(
            Trace(
                trace_id="exact-one",
                source="test",
                source_digest="a" * 64,
                task="Return exactly OK",
                spans=(
                    Span(
                        span_id="reply",
                        kind=SpanKind.MODEL,
                        name="assistant",
                        started_at=now,
                        ended_at=now,
                        output="OK",
                    ),
                ),
            ),
        ),
    )
    analysis = analyze_corpus(corpus)
    task_set = mine_task_set(analysis, compute_analysis_id(analysis), budget=1)

    draft = draft_verifiers(
        task_set, "taskset-test", analysis, task_set.families[0].family_id
    )

    assert draft.verifiers[0].checks[0].operator.value == "exact_output"
    assert "agent output" in draft.verifiers[0].blind_spots[0]
