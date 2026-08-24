"""Chat JSON adapter: the shape almost any agent framework can dump with zero
observability tooling — a plain list of ``{role, content, tool_calls}`` messages.

An assistant message with ``tool_calls`` becomes one MODEL span per requested
call; the following ``tool``-role message with a matching ``tool_call_id``
becomes that call's child TOOL span. Chat JSON carries no real timestamps, so
spans get sequential synthetic ones — ordinal order only, never treated as
measured latency, and tagged ``attributes={"synthetic_time": True}`` so nothing
downstream mistakes assigned order for real timing.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from bandits.traces import Span, SpanKind, SpanStatus, Trace, TraceCorpus, TraceIssue

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _conversations(payload: object) -> list[list[dict]]:
    """A file is either one conversation (a bare message array) or a list of them."""
    if (
        isinstance(payload, list)
        and payload
        and all(isinstance(item, dict) and "role" in item for item in payload)
    ):
        return [payload]
    if isinstance(payload, list):
        return [c.get("messages", []) for c in payload if isinstance(c, dict)]
    if isinstance(payload, dict):
        return [payload.get("messages", [])]
    return []


def _tool_calls(message: dict) -> list[dict]:
    raw = message.get("tool_calls")
    return [c for c in raw if isinstance(c, dict)] if isinstance(raw, list) else []


def _convert_conversation(
    messages: list[dict],
    *,
    trace_id: str,
    source_digest: str,
    location: str,
    issues: list[TraceIssue],
) -> Trace | None:
    task = next(
        (
            m.get("content")
            for m in messages
            if m.get("role") in ("user", "human") and m.get("content")
        ),
        None,
    )
    if task is None:
        issues.append(
            TraceIssue(
                kind="no_user_message", detail="conversation has no user message", location=location
            )
        )
        return None

    spans: list[Span] = []
    pending_calls: dict[
        str, str
    ] = {}  # tool_call_id -> span_id of the MODEL span that requested it
    ordinal = 0

    for index, message in enumerate(messages):
        role = message.get("role")
        timestamp = _EPOCH + timedelta(seconds=ordinal)

        if role == "assistant":
            calls = _tool_calls(message)
            if calls:
                for call in calls:
                    function = (
                        call.get("function") if isinstance(call.get("function"), dict) else call
                    )
                    name = function.get("name")
                    call_id = call.get("id") or call.get("tool_call_id")
                    if not isinstance(name, str) or not name:
                        issues.append(
                            TraceIssue(
                                kind="tool_call_missing_name",
                                detail=f"message {index} has a tool call with no readable name",
                                location=location,
                            )
                        )
                        continue
                    span_id = f"{trace_id}:span-{ordinal}"
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {"raw": arguments}
                    spans.append(
                        Span(
                            span_id=span_id,
                            parent_span_id=None,
                            kind=SpanKind.MODEL,
                            name=name,
                            started_at=timestamp,
                            ended_at=timestamp,
                            arguments=arguments if isinstance(arguments, dict) else {},
                            attributes={"synthetic_time": True},
                        )
                    )
                    if isinstance(call_id, str) and call_id:
                        pending_calls[call_id] = span_id
                    ordinal += 1
            else:
                content = message.get("content")
                span_id = f"{trace_id}:span-{ordinal}"
                spans.append(
                    Span(
                        span_id=span_id,
                        parent_span_id=None,
                        kind=SpanKind.MODEL,
                        name="assistant",
                        started_at=timestamp,
                        ended_at=timestamp,
                        output=content,
                        attributes={"synthetic_time": True},
                    )
                )
                ordinal += 1

        elif role == "tool":
            call_id = message.get("tool_call_id")
            parent_span_id = pending_calls.get(call_id) if isinstance(call_id, str) else None
            name = message.get("name") or "tool"
            span_id = f"{trace_id}:span-{ordinal}"
            spans.append(
                Span(
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    kind=SpanKind.TOOL,
                    name=name,
                    started_at=timestamp,
                    ended_at=timestamp,
                    output=message.get("content"),
                    status=SpanStatus.OK,
                    attributes={"synthetic_time": True},
                )
            )
            if parent_span_id is None:
                issues.append(
                    TraceIssue(
                        kind="unpaired_tool_result",
                        detail=f"message {index}: no matching assistant tool_call for tool_call_id {call_id!r}",
                        location=location,
                    )
                )
            ordinal += 1

    if not spans:
        issues.append(
            TraceIssue(
                kind="empty_conversation",
                detail="conversation has no assistant or tool message",
                location=location,
            )
        )
        return None

    return Trace(
        trace_id=trace_id,
        source="chat-json",
        source_digest=source_digest,
        task=task,
        spans=tuple(spans),
    )


def load_chat_json(path: Path) -> TraceCorpus:
    """Read one chat-JSON export (one conversation or an array of them) into a
    :class:`~bandits.traces.TraceCorpus`."""
    raw = path.read_bytes()
    source_digest = hashlib.sha256(raw).hexdigest()

    issues: list[TraceIssue] = []
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        issues.append(TraceIssue(kind="malformed_json", detail=str(exc), location=str(path)))
        return TraceCorpus(source="chat-json", traces=(), issues=tuple(issues))

    conversations = _conversations(payload)
    traces: list[Trace] = []
    for index, messages in enumerate(conversations):
        location = f"{path}[{index}]"
        trace = _convert_conversation(
            messages,
            trace_id=f"{path.stem}-{index}",
            source_digest=source_digest,
            location=location,
            issues=issues,
        )
        if trace is not None:
            traces.append(trace)

    return TraceCorpus(source="chat-json", traces=tuple(traces), issues=tuple(issues))
