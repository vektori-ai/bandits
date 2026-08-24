from __future__ import annotations

from pathlib import Path

from bandits.ingest.otlp import load_otlp
from bandits.traces import SpanKind, SpanStatus

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "traces.otlp.jsonl"


def test_groups_spans_into_traces() -> None:
    corpus = load_otlp(FIXTURE)

    assert corpus.source == "otlp"
    assert {t.trace_id for t in corpus.traces} == {"trace-1", "trace-2"}

    trace_1 = next(t for t in corpus.traces if t.trace_id == "trace-1")
    assert trace_1.task == "Refund order 7741"
    assert [s.span_id for s in trace_1.spans] == ["span-1", "span-2"]
    assert trace_1.spans[1].parent_span_id == "span-1"
    assert trace_1.spans[1].kind is SpanKind.TOOL
    assert trace_1.spans[1].arguments == {"order_id": "7741"}
    assert trace_1.spans[1].output == {"status": "refunded"}


def test_error_status_is_captured() -> None:
    corpus = load_otlp(FIXTURE)
    trace_2 = next(t for t in corpus.traces if t.trace_id == "trace-2")
    tool_span = next(s for s in trace_2.spans if s.kind is SpanKind.TOOL)
    assert tool_span.status is SpanStatus.ERROR


def test_malformed_line_becomes_an_issue_not_a_failure() -> None:
    corpus = load_otlp(FIXTURE)
    assert len(corpus.issues) == 1
    assert corpus.issues[0].kind == "malformed_json"
    # the well-formed record on the line after it still loads
    assert any(t.trace_id == "trace-2" and len(t.spans) == 2 for t in corpus.traces)


def test_all_traces_share_the_file_digest() -> None:
    corpus = load_otlp(FIXTURE)
    digests = {t.source_digest for t in corpus.traces}
    assert len(digests) == 1
