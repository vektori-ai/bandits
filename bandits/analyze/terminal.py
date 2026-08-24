"""Shared terminal-span semantics for task and outcome extraction."""

from __future__ import annotations

from bandits.traces import Span, SpanKind


def terminal_spans(spans: tuple[Span, ...]) -> tuple[Span, ...]:
    """Return the final span and a directly preceding tool result, when present."""
    if not spans:
        return ()
    last = spans[-1]
    if last.kind is SpanKind.MODEL and len(spans) >= 2 and spans[-2].kind is SpanKind.TOOL:
        return (spans[-2], last)
    return (last,)
