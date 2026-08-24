from __future__ import annotations

from pathlib import Path

from bandits.ingest.chat_json import load_chat_json
from bandits.traces import SpanKind

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "traces.chat.jsonl"


def test_pairs_tool_call_with_tool_result() -> None:
    corpus = load_chat_json(FIXTURE)

    assert corpus.source == "chat-json"
    assert len(corpus.traces) == 1
    trace = corpus.traces[0]
    assert trace.task == "Refund order 7741"

    tool_span = next(s for s in trace.spans if s.kind is SpanKind.TOOL)
    model_span = next(s for s in trace.spans if s.kind is SpanKind.MODEL and s.arguments)
    assert tool_span.parent_span_id == model_span.span_id
    assert tool_span.arguments == {}
    assert tool_span.output == {"status": "refunded"}
    assert model_span.arguments == {"order_id": "7741"}


def test_spans_are_tagged_synthetic_time() -> None:
    corpus = load_chat_json(FIXTURE)
    assert all(s.attributes.get("synthetic_time") is True for s in corpus.traces[0].spans)


def test_no_user_message_becomes_an_issue(tmp_path) -> None:
    path = tmp_path / "no_user.json"
    path.write_text('[{"role": "assistant", "content": "hi"}]')

    corpus = load_chat_json(path)

    assert corpus.traces == ()
    assert corpus.issues[0].kind == "no_user_message"


def test_prose_tool_result_is_kept_as_text() -> None:
    """Only JSON-shaped results are parsed; prose is genuinely prose."""
    corpus = load_chat_json(FIXTURE)
    trace = corpus.traces[0]

    final_reply = trace.spans[-1]
    assert final_reply.output == "Order 7741 has been refunded."


def test_bare_message_array_declares_no_lineage() -> None:
    corpus = load_chat_json(FIXTURE)
    assert corpus.traces[0].lineage_id is None


def test_wrapped_conversation_carries_its_session_id(tmp_path) -> None:
    source = tmp_path / "chat.json"
    source.write_text(
        '[{"session_id": "sess-9", "messages": ['
        '{"role": "user", "content": "Refund order 7741"},'
        '{"role": "assistant", "content": "Done."}]}]'
    )

    corpus = load_chat_json(source)

    assert corpus.traces[0].lineage_id == "sess-9"
