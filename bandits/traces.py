"""Canonical trace model.

A trace is an ordered set of spans. Every ingest adapter's only job is to produce
that list correctly for its source format; nothing downstream ever looks at a raw
export again once it has been turned into a ``TraceCorpus``.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    """Base for every model in this module: immutable, no silent extra fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class SpanKind(str, Enum):
    MODEL = "model"
    """One call to a language model."""

    TOOL = "tool"
    """One tool call."""


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


class Span(Contract):
    """One node in a trace: either a model call or a tool call."""

    span_id: str
    parent_span_id: str | None = None
    """None marks the root span of the trace."""

    kind: SpanKind
    name: str
    """Tool name for a TOOL span; model name for a MODEL span."""

    started_at: datetime
    ended_at: datetime
    status: SpanStatus = SpanStatus.OK
    arguments: dict[str, Any] = Field(default_factory=dict)
    """Tool call arguments, or the prompt/input for a model call."""

    output: Any = None
    """Tool response, or the completion text for a model call."""

    attributes: dict[str, Any] = Field(default_factory=dict)
    """Anything else the source declared that doesn't have a dedicated field."""


class TraceIssue(Contract):
    """One source record that could not be normalized. Never silently dropped."""

    kind: str
    detail: str
    location: str | None = None


class Trace(Contract):
    """One normalized agent episode."""

    trace_id: str
    source: str
    """Declared adapter name that produced this trace, e.g. 'otlp'."""

    source_digest: str
    """sha256 hex of the exact source bytes this trace came from."""

    task: str | None = None
    """The user-facing instruction, when the source declares one."""

    spans: tuple[Span, ...]


class TraceCorpus(Contract):
    """The normalized result of one ingest run."""

    source: str
    traces: tuple[Trace, ...]
    issues: tuple[TraceIssue, ...] = ()
