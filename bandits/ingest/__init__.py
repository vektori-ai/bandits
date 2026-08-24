"""Ingest adapters: turn a raw trace export into a :class:`~bandits.traces.TraceCorpus`.

Format is always declared by the caller, never guessed from file content — a
mislabeled export would otherwise parse into a corpus with the wrong spans in it
and look like a valid, if odd, trace rather than an error.
"""

from __future__ import annotations

from pathlib import Path

from bandits.ingest.chat_json import load_chat_json
from bandits.ingest.otlp import load_otlp
from bandits.redact import DEFAULT_RULESET, RedactionRuleset
from bandits.traces import TraceCorpus

CANONICAL_SOURCES: tuple[str, ...] = ("otlp", "chat-json")

_LOADERS = {
    "otlp": load_otlp,
    "chat-json": load_chat_json,
}


class UnknownSourceError(ValueError):
    """Raised for a source name that is not in :data:`CANONICAL_SOURCES`."""


def load_corpus(
    path: str | Path, source: str, ruleset: RedactionRuleset = DEFAULT_RULESET
) -> TraceCorpus:
    """Read a raw export into a :class:`TraceCorpus` using the declared adapter."""
    loader = _LOADERS.get(source)
    if loader is None:
        raise UnknownSourceError(
            f"unknown source {source!r}; declare one of {list(CANONICAL_SOURCES)}"
        )
    return loader(Path(path), ruleset)


__all__ = [
    "CANONICAL_SOURCES",
    "UnknownSourceError",
    "load_corpus",
]
