"""Read outcome evidence off a trace, without deciding the outcome.

Every detector here answers "what did the source actually record?", never "did
this run succeed?". A tool whose result contains the word ``refunded`` is
evidence about a field, not a verdict; collapsing the two is the mistake that
makes a corpus look better than it was.
"""

from __future__ import annotations

from typing import Any

from bandits.analyze.models import Evidence, Visibility, evidence_id
from bandits.traces import Span, SpanKind, SpanStatus, Trace

_EXIT_CODE_KEYS = ("exit_code", "exitcode", "returncode", "return_code")
"""Keys that carry a process result across the exporters seen so far."""

_SCORE_KEYS = ("score", "rating", "feedback", "user_feedback", "evaluation")
"""Keys under which a source may have recorded its own judgement of the run."""

_STATE_KEYS = ("status", "state")


def _evidence(
    span: Span | None,
    *,
    trace_id: str,
    claim: str,
    value: Any,
    visibility: Visibility,
    strength: str,
) -> Evidence:
    span_id = span.span_id if span is not None else None
    return Evidence(
        evidence_id=evidence_id(trace_id=trace_id, claim=claim, span_id=span_id),
        claim=claim,
        value=value,
        visibility=visibility,
        provenance="observed",
        strength=strength,  # type: ignore[arg-type]
        trace_id=trace_id,
        span_id=span_id,
    )


def _first_key(payload: Any, keys: tuple[str, ...]) -> tuple[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        if key in payload:
            return key, payload[key]
    return None


def extract_outcome_evidence(trace: Trace) -> tuple[Evidence, ...]:
    """Collect every recorded signal about how this episode went."""
    found: dict[str, Evidence] = {}
    last_span_id = trace.spans[-1].span_id if trace.spans else None

    def record(evidence: Evidence) -> None:
        found.setdefault(evidence.evidence_id, evidence)

    for span in trace.spans:
        is_last = span.span_id == last_span_id
        visibility = Visibility.TERMINAL if is_last else Visibility.DURING

        if span.status is SpanStatus.ERROR:
            record(
                _evidence(
                    span,
                    trace_id=trace.trace_id,
                    claim="span_error",
                    value={"name": span.name, "kind": span.kind.value},
                    visibility=visibility,
                    strength="strong",
                )
            )

        exit_code = _first_key(span.output, _EXIT_CODE_KEYS) or _first_key(
            span.attributes, _EXIT_CODE_KEYS
        )
        if exit_code is not None:
            record(
                _evidence(
                    span,
                    trace_id=trace.trace_id,
                    claim="command_exit_code",
                    value={"key": exit_code[0], "value": exit_code[1], "tool": span.name},
                    visibility=visibility,
                    strength="strong",
                )
            )

        recorded_score = _first_key(span.attributes, _SCORE_KEYS)
        if recorded_score is not None:
            # Moderate, not strong: the source recorded a judgement but not what
            # produced it, so its trustworthiness is unknown until someone says.
            record(
                _evidence(
                    span,
                    trace_id=trace.trace_id,
                    claim="recorded_score",
                    value={"key": recorded_score[0], "value": recorded_score[1]},
                    visibility=Visibility.POST_HOC,
                    strength="moderate",
                )
            )

        if span.kind is SpanKind.TOOL and is_last:
            state = _first_key(span.output, _STATE_KEYS)
            if state is not None:
                record(
                    _evidence(
                        span,
                        trace_id=trace.trace_id,
                        claim="final_state_field",
                        value={"key": state[0], "value": state[1], "tool": span.name},
                        visibility=Visibility.TERMINAL,
                        strength="moderate",
                    )
                )

        if span.kind is SpanKind.TOOL and span.output is None and span.status is SpanStatus.OK:
            # An absent result is absent. It is never an empty successful result.
            record(
                _evidence(
                    span,
                    trace_id=trace.trace_id,
                    claim="missing_tool_result",
                    value={"tool": span.name},
                    visibility=visibility,
                    strength="strong",
                )
            )

    return tuple(found.values())
