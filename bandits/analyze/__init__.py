"""Deterministic analysis of a trace corpus: task candidates and outcome evidence.

Nothing here is inferred by a model. Every value is read directly off spans that
are already in the corpus, and anything the source did not record is named as a
limitation rather than filled in. Model-assisted analysis is a later, explicitly
labelled addition on top of this layer, never a replacement for it.
"""

from __future__ import annotations

from bandits.analyze.analysis import (
    analyze_corpus,
    compute_analysis_id,
    load_analysis,
    save_analysis,
)
from bandits.analyze.models import (
    CorpusAnalysis,
    Evidence,
    LeakageError,
    TaskCandidate,
    Visibility,
    build_task_candidate,
)
from bandits.analyze.outcomes import extract_outcome_evidence
from bandits.analyze.tasks import extract_task

__all__ = [
    "CorpusAnalysis",
    "Evidence",
    "LeakageError",
    "TaskCandidate",
    "Visibility",
    "analyze_corpus",
    "build_task_candidate",
    "compute_analysis_id",
    "extract_outcome_evidence",
    "extract_task",
    "load_analysis",
    "save_analysis",
]
