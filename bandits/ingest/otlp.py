"""OTLP adapter.

The native format: a trace is already a flat stream of spans, each declaring its
own trace id, span id, optional parent span id, and start/end time. Nothing here
reconstructs anything the export didn't already say directly.

Expected record shape, one JSON object per line::

    {
      "trace_id": "trace-1",
      "span_id": "span-1",
      "parent_span_id": null,
      "name": "gpt-5",
      "start_time": "2026-01-01T00:00:00Z",
      "end_time": "2026-01-01T00:00:01Z",
      "attributes": {
        "gen_ai.operation.name": "chat",
        "task": "Refund order 7741"
      }
    }

``attributes["gen_ai.operation.name"]`` decides the span kind: ``"chat"`` is a
MODEL span, ``"execute_tool"`` is a TOOL span. Anything else is a
:class:`~bandits.traces.TraceIssue`, not a guess.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from bandits.traces import Span, SpanKind, SpanStatus, Trace, TraceCorpus, TraceIssue

_OPERATION_TO_KIND = {
    "chat": SpanKind.MODEL,
    "execute_tool": SpanKind.TOOL,
}


class OtlpFormatError(ValueError):
    """The file itself is not readable as OTLP JSONL. Raised, not swallowed."""


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_record(record: dict, *, location: str) -> tuple[Span | None, str | None, str | None]:
    """Returns ``(span, trace_id, task)`` for a valid record, or ``(None, None, None)``
    plus an appended issue when the record can't become a span."""
    trace_id = record.get("trace_id")
    span_id = record.get("span_id")
    name = record.get("name")
    attributes = record.get("attributes") or {}
    operation = attributes.get("gen_ai.operation.name")
    kind = _OPERATION_TO_KIND.get(operation)
    started_at = _parse_timestamp(record.get("start_time"))
    ended_at = _parse_timestamp(record.get("end_time"))

    missing = [
        field
        for field, value in (
            ("trace_id", trace_id),
            ("span_id", span_id),
            ("name", name),
        )
        if not isinstance(value, str) or not value
    ]
    if missing:
        raise _RecordError(f"missing or invalid field(s): {', '.join(missing)}", location)
    if kind is None:
        raise _RecordError(
            f"unrecognized gen_ai.operation.name {operation!r}; expected 'chat' or 'execute_tool'",
            location,
        )
    if started_at is None or ended_at is None:
        raise _RecordError("start_time/end_time must be ISO-8601 timestamps", location)

    if kind is SpanKind.TOOL:
        arguments = attributes.get("gen_ai.tool.call.arguments") or {}
        output = attributes.get("gen_ai.tool.call.result")
    else:
        arguments = attributes.get("gen_ai.request.arguments") or {}
        output = attributes.get("gen_ai.completion")

    status = SpanStatus.ERROR if attributes.get("status") == "error" else SpanStatus.OK
    parent_span_id = record.get("parent_span_id") or None
    task = attributes.get("task") if parent_span_id is None else None

    span = Span(
        span_id=span_id,
        parent_span_id=parent_span_id,
        kind=kind,
        name=name,
        started_at=started_at,
        ended_at=ended_at,
        status=status,
        arguments=arguments if isinstance(arguments, dict) else {},
        output=output,
        attributes=attributes,
    )
    return span, trace_id, task


class _RecordError(Exception):
    def __init__(self, detail: str, location: str) -> None:
        super().__init__(detail)
        self.detail = detail
        self.location = location


def load_otlp(path: Path) -> TraceCorpus:
    """Read one OTLP JSONL export into a :class:`TraceCorpus`."""
    raw = path.read_bytes()
    source_digest = hashlib.sha256(raw).hexdigest()

    spans_by_trace: dict[str, list[Span]] = {}
    task_by_trace: dict[str, str] = {}
    issues: list[TraceIssue] = []

    for index, line in enumerate(raw.split(b"\n")):
        if not line.strip():
            continue
        location = f"{path}:{index + 1}"
        try:
            record = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            issues.append(TraceIssue(kind="malformed_json", detail=str(exc), location=location))
            continue
        if not isinstance(record, dict):
            issues.append(
                TraceIssue(
                    kind="malformed_record",
                    detail=f"expected a JSON object, got {type(record).__name__}",
                    location=location,
                )
            )
            continue
        try:
            span, trace_id, task = _parse_record(record, location=location)
        except _RecordError as exc:
            issues.append(
                TraceIssue(kind="malformed_span", detail=exc.detail, location=exc.location)
            )
            continue
        spans_by_trace.setdefault(trace_id, []).append(span)
        if task is not None:
            task_by_trace[trace_id] = task

    traces = tuple(
        Trace(
            trace_id=trace_id,
            source="otlp",
            source_digest=source_digest,
            task=task_by_trace.get(trace_id),
            spans=tuple(sorted(spans, key=lambda s: (s.started_at, s.span_id))),
        )
        for trace_id, spans in sorted(spans_by_trace.items())
    )
    return TraceCorpus(source="otlp", traces=traces, issues=tuple(issues))
