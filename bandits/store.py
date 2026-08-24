"""Local content-addressed artifact store.

Every ingested corpus is written once under an id derived from its own content
and never mutated afterward. Ingesting the same file twice lands on the same id
and is a no-op; two different corpora landing on the same id — unreachable in
practice, since ids are content hashes, but not trusted blindly — raises instead
of silently overwriting.
"""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from bandits.traces import TraceCorpus


class Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ArtifactEnvelope(Contract):
    schema_version: int = 1
    artifact_id: str
    created_at: str
    source_path: str
    source: str
    trace_count: int
    span_count: int
    issue_count: int


class ArtifactConflict(ValueError):
    """An existing artifact at this id has different content than what was just written."""


def compute_artifact_id(corpus: TraceCorpus) -> str:
    digest = hashlib.sha256(corpus.model_dump_json().encode("utf-8")).hexdigest()
    return f"corpus-{digest[:16]}"


def _atomic_write(path: Path, data: bytes) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(data)
    os.replace(tmp_path, path)


class ArtifactStore:
    def __init__(self, project_dir: Path | str = Path(".bandits")) -> None:
        self._project_dir = Path(project_dir)
        self._artifacts_dir = self._project_dir / "artifacts"

    def _dir(self, artifact_id: str) -> Path:
        return self._artifacts_dir / artifact_id

    def write(self, corpus: TraceCorpus, *, source_path: str) -> ArtifactEnvelope:
        artifact_id = compute_artifact_id(corpus)
        artifact_dir = self._dir(artifact_id)
        corpus_bytes = corpus.model_dump_json().encode("utf-8")

        if artifact_dir.exists():
            existing_bytes = (artifact_dir / "corpus.json").read_bytes()
            if existing_bytes != corpus_bytes:
                raise ArtifactConflict(
                    f"artifact {artifact_id} already exists with different content"
                )
            return self.read_envelope(artifact_id)

        artifact_dir.mkdir(parents=True)
        envelope = ArtifactEnvelope(
            artifact_id=artifact_id,
            created_at=datetime.now(UTC).isoformat(),
            source_path=source_path,
            source=corpus.source,
            trace_count=len(corpus.traces),
            span_count=sum(len(t.spans) for t in corpus.traces),
            issue_count=len(corpus.issues),
        )
        _atomic_write(artifact_dir / "corpus.json", corpus_bytes)
        _atomic_write(artifact_dir / "envelope.json", envelope.model_dump_json().encode("utf-8"))
        return envelope

    def read(self, artifact_id: str) -> TraceCorpus:
        return TraceCorpus.model_validate_json(
            (self._dir(artifact_id) / "corpus.json").read_bytes()
        )

    def read_envelope(self, artifact_id: str) -> ArtifactEnvelope:
        return ArtifactEnvelope.model_validate_json(
            (self._dir(artifact_id) / "envelope.json").read_bytes()
        )

    def list(self) -> list[ArtifactEnvelope]:
        if not self._artifacts_dir.exists():
            return []
        envelopes = [
            self.read_envelope(entry.name)
            for entry in self._artifacts_dir.iterdir()
            if entry.is_dir()
        ]
        return sorted(envelopes, key=lambda e: e.created_at, reverse=True)


class DerivedEnvelope(Contract):
    """Header for an artifact derived from another one.

    Kept deliberately ignorant of what it wraps: the store persists opaque JSON so
    that adding an analysis, verifier, or export type never reaches back into it.
    """

    schema_version: int = 1
    artifact_id: str
    kind: str
    """What produced this, e.g. 'analysis'."""

    parent_artifact_id: str
    """The artifact this was derived from. Never empty — lineage is the point."""

    created_at: str
    summary: dict[str, int] = {}


class DerivedStore:
    """Derived artifacts, stored beside corpora rather than among them.

    A separate directory so that :meth:`ArtifactStore.list` keeps returning
    corpora only, and so an analysis can never be mistaken for source evidence.
    """

    def __init__(self, project_dir: Path | str = Path(".bandits")) -> None:
        self._derived_dir = Path(project_dir) / "derived"

    def _dir(self, artifact_id: str) -> Path:
        return self._derived_dir / artifact_id

    def write(
        self,
        artifact_id: str,
        *,
        kind: str,
        parent_artifact_id: str,
        payload: bytes,
        summary: dict[str, int] | None = None,
    ) -> DerivedEnvelope:
        artifact_dir = self._dir(artifact_id)
        if artifact_dir.exists():
            existing = (artifact_dir / "payload.json").read_bytes()
            if existing != payload:
                raise ArtifactConflict(
                    f"derived artifact {artifact_id} already exists with different content"
                )
            return self.read_envelope(artifact_id)

        artifact_dir.mkdir(parents=True)
        envelope = DerivedEnvelope(
            artifact_id=artifact_id,
            kind=kind,
            parent_artifact_id=parent_artifact_id,
            created_at=datetime.now(UTC).isoformat(),
            summary=summary or {},
        )
        _atomic_write(artifact_dir / "payload.json", payload)
        _atomic_write(artifact_dir / "envelope.json", envelope.model_dump_json().encode("utf-8"))
        return envelope

    def read_payload(self, artifact_id: str) -> bytes:
        return (self._dir(artifact_id) / "payload.json").read_bytes()

    def read_envelope(self, artifact_id: str) -> DerivedEnvelope:
        return DerivedEnvelope.model_validate_json(
            (self._dir(artifact_id) / "envelope.json").read_bytes()
        )

    def list(self, *, kind: str | None = None) -> list[DerivedEnvelope]:
        if not self._derived_dir.exists():
            return []
        envelopes = [
            self.read_envelope(entry.name)
            for entry in self._derived_dir.iterdir()
            if entry.is_dir()
        ]
        if kind is not None:
            envelopes = [e for e in envelopes if e.kind == kind]
        return sorted(envelopes, key=lambda e: e.created_at, reverse=True)
