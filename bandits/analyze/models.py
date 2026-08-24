"""Derived analysis contracts.

These live beside the corpus, never on it. The same trace may be reassessed by a
newer analysis policy or corrected by a reviewer; that must produce a new
artifact rather than rewrite the evidence it was derived from.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from bandits.traces import Contract


class Visibility(str, Enum):
    """When a fact becomes knowable, relative to one episode."""

    AT_START = "at_start"
    """Available to a live request at t=0: the instruction and permitted context."""

    DURING = "during"
    """Observed only as the episode runs."""

    TERMINAL = "terminal"
    """The final output or final observed state."""

    POST_HOC = "post_hoc"
    """Knowable only afterwards: outcome labels, totals, downstream corrections."""


_STRENGTH_ORDER: dict[str, int] = {"strong": 3, "moderate": 2, "weak": 1}


class Evidence(Contract):
    """One fact read off a trace, carrying where it came from and when it was knowable."""

    evidence_id: str
    claim: str
    """Short stable name for what is being asserted, e.g. 'command_exit_code'."""

    value: Any
    visibility: Visibility
    provenance: Literal["observed", "derived", "model", "human"]
    """'observed' means read directly off a span; nothing here infers."""

    strength: Literal["strong", "moderate", "weak"]
    trace_id: str
    span_id: str | None = None

    @property
    def strength_rank(self) -> int:
        """Higher wins when two pieces of evidence disagree."""
        return _STRENGTH_ORDER[self.strength]


class LeakageError(ValueError):
    """A task prompt was built from a fact that is not knowable at t=0.

    Raised rather than silently dropped: a task set that leaks its own answer
    into the prompt scores suspiciously well and is very hard to notice later.
    """


class TaskCandidate(Contract):
    """One episode, reshaped into something a new attempt could be asked to do."""

    task_id: str
    trace_id: str
    instruction: str | None = None
    """None when the source never declared one; see ``limitations``."""

    prompt_evidence_ids: tuple[str, ...] = ()
    """Facts the prompt may be built from. Enforced ``at_start`` — see :func:`build_task_candidate`."""

    trajectory_span_ids: tuple[str, ...] = ()
    terminal_span_ids: tuple[str, ...] = ()
    outcome_evidence_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    """What this extraction could not recover. Never filled in with a guess."""


def build_task_candidate(
    *,
    task_id: str,
    trace_id: str,
    instruction: str | None,
    prompt_evidence: tuple[Evidence, ...],
    trajectory_span_ids: tuple[str, ...],
    terminal_span_ids: tuple[str, ...],
    outcome_evidence: tuple[Evidence, ...],
    limitations: tuple[str, ...] = (),
) -> TaskCandidate:
    """Assemble a task candidate, refusing any prompt fact that is not ``at_start``."""
    leaked = [e for e in prompt_evidence if e.visibility is not Visibility.AT_START]
    if leaked:
        detail = ", ".join(f"{e.evidence_id} ({e.visibility.value})" for e in leaked)
        raise LeakageError(f"prompt evidence for {task_id} is not knowable at t=0: {detail}")

    return TaskCandidate(
        task_id=task_id,
        trace_id=trace_id,
        instruction=instruction,
        prompt_evidence_ids=tuple(e.evidence_id for e in prompt_evidence),
        trajectory_span_ids=trajectory_span_ids,
        terminal_span_ids=terminal_span_ids,
        outcome_evidence_ids=tuple(e.evidence_id for e in outcome_evidence),
        limitations=limitations,
    )


class CorpusAnalysis(Contract):
    """The deterministic read of one corpus: what was asked, and what is known about how it went."""

    schema_version: int = 1
    corpus_id: str
    """Parent artifact. Every derived artifact records the one it came from."""

    source: str
    tasks: tuple[TaskCandidate, ...]
    evidence: tuple[Evidence, ...]
    limitations: tuple[str, ...] = ()

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {e.evidence_id: e for e in self.evidence}


_SLUG_UNSAFE = re.compile(r"[^a-z0-9]+")


def evidence_id(*, trace_id: str, claim: str, span_id: str | None = None) -> str:
    """Deterministic id, so re-analyzing the same corpus lands on the same ids."""
    parts = [trace_id, span_id or "trace", claim]
    slug = _SLUG_UNSAFE.sub("-", "-".join(parts).lower()).strip("-")
    return f"ev-{slug}"
