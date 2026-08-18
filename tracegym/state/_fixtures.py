"""Test helpers: build contract objects without depending on stages 1 and 2.

Stage 3 must be developable and testable before ingest (stage 1) and surface
(stage 2) exist, so this module converts
``tests/fixtures/make_corpus.py:episodes()`` straight into
:class:`~tracegym.contracts.Trace` objects, and builds a
:class:`~tracegym.contracts.ToolSurface` from
``tests/fixtures/expected.json:expected_tool_classes``.

This is *not* an ingest implementation - it is a fixture loader for this
module's own tests, and nothing in :mod:`tracegym.state` imports it at runtime.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from tracegym.contracts import (
    CallStatus,
    InvocationPoint,
    Message,
    ToolClass,
    ToolProfile,
    ToolSurface,
    Trace,
    TraceCorpus,
)

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _load_make_corpus() -> Any:
    path = FIXTURE_DIR / "make_corpus.py"
    spec = importlib.util.spec_from_file_location("tracegym_fixture_make_corpus", path)
    if spec is None or spec.loader is None:  # pragma: no cover - fixture must exist
        raise RuntimeError(f"cannot load fixture generator at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected() -> dict[str, Any]:
    """The golden ``expected.json``, the ground truth this stage reproduces."""
    return json.loads((FIXTURE_DIR / "expected.json").read_text())


def episodes() -> list[tuple]:
    return list(_load_make_corpus().episodes())


def trace_from_episode(episode: tuple) -> Trace:
    eid, instruction, calls, answer, outcome = episode
    digest = hashlib.sha256(json.dumps(episode, default=str).encode()).hexdigest()
    invocations = tuple(
        InvocationPoint(
            call_id=f"call_{eid}_{i}",
            trace_id=eid,
            step=i,
            tool=tool,
            arguments=args,
            response=resp,
            status=CallStatus.ERROR if status == "error" else CallStatus.OK,
            error_kind=err,
        )
        for i, (tool, args, resp, status, err) in enumerate(calls)
    )
    return Trace(
        trace_id=eid,
        source="fixture",
        source_digest=digest,
        messages=(
            Message(role="user", content=instruction),
            Message(role="assistant", content=answer),
        ),
        invocations=invocations,
        outcome=outcome,
    )


def golden_corpus() -> TraceCorpus:
    """The full golden corpus as contract objects."""
    return TraceCorpus(
        source="fixture",
        traces=tuple(trace_from_episode(e) for e in episodes()),
    )


def golden_surface() -> ToolSurface:
    """A ToolSurface carrying exactly the classes ``expected.json`` declares."""
    classes = expected()["expected_tool_classes"]
    return ToolSurface(
        tools=tuple(
            ToolProfile(
                name=name,
                tool_class=ToolClass(value),
                class_confidence=1.0,
                class_evidence=("fixture: expected.json",),
            )
            for name, value in sorted(classes.items())
        )
    )


def corpus_from_calls(
    calls_by_trace: dict[str, list[tuple[str, dict, Any]]],
) -> TraceCorpus:
    """Build a tiny corpus from ``{trace_id: [(tool, arguments, response), ...]}``."""
    traces = []
    for tid, calls in calls_by_trace.items():
        traces.append(
            Trace(
                trace_id=tid,
                source="synthetic",
                source_digest="0" * 64,
                invocations=tuple(
                    InvocationPoint(
                        call_id=f"{tid}-{i}",
                        trace_id=tid,
                        step=i,
                        tool=tool,
                        arguments=args,
                        response=resp,
                    )
                    for i, (tool, args, resp) in enumerate(calls)
                ),
            )
        )
    return TraceCorpus(source="synthetic", traces=tuple(traces))


__all__ = [
    "FIXTURE_DIR",
    "corpus_from_calls",
    "episodes",
    "expected",
    "golden_corpus",
    "golden_surface",
    "trace_from_episode",
]
