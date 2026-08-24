"""Command-line interface: ingest a trace export, then inspect what's stored."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from bandits.analyze import analyze_corpus, save_analysis
from bandits.ingest import CANONICAL_SOURCES, UnknownSourceError, load_corpus
from bandits.store import ArtifactStore, DerivedStore

app = typer.Typer(add_completion=False)
console = Console()

_MAX_INLINE_ISSUES = 3
_DEFAULT_PROJECT = Path(".")


@app.command()
def ingest(
    path: Path,
    source: str = typer.Option(..., "--source", help=f"One of: {', '.join(CANONICAL_SOURCES)}"),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Load a trace export into the local artifact store."""
    try:
        corpus = load_corpus(path, source)
    except UnknownSourceError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    store = ArtifactStore(project / ".bandits")
    envelope = store.write(corpus, source_path=str(path))

    console.print(f"artifact_id: {envelope.artifact_id}")
    console.print(f"source:      {envelope.source}")
    console.print(f"traces:      {envelope.trace_count}")
    console.print(f"spans:       {envelope.span_count}")
    console.print(f"issues:      {envelope.issue_count}")
    for issue in corpus.issues[:_MAX_INLINE_ISSUES]:
        location = f" at {issue.location}" if issue.location else ""
        console.print(f"  - {issue.kind}{location}: {issue.detail}")
    remaining = envelope.issue_count - _MAX_INLINE_ISSUES
    if remaining > 0:
        console.print(f"  (+{remaining} more — see `bandits show {envelope.artifact_id} --issues`)")


@app.command(name="list")
def list_artifacts(project: Path = typer.Option(_DEFAULT_PROJECT, "--project")) -> None:
    """List every artifact in the local store."""
    store = ArtifactStore(project / ".bandits")
    table = Table("artifact_id", "source", "traces", "spans", "issues", "created_at")
    for envelope in store.list():
        table.add_row(
            envelope.artifact_id[:19],
            envelope.source,
            str(envelope.trace_count),
            str(envelope.span_count),
            str(envelope.issue_count),
            envelope.created_at,
        )
    console.print(table)


@app.command()
def show(
    artifact_id: str,
    trace: str = typer.Option(None, "--trace"),
    issues: bool = typer.Option(False, "--issues"),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Inspect one stored artifact."""
    store = ArtifactStore(project / ".bandits")
    corpus = store.read(artifact_id)

    if issues:
        table = Table("kind", "location", "detail")
        for issue in corpus.issues:
            table.add_row(issue.kind, issue.location or "", issue.detail)
        console.print(table)
        return

    if trace is not None:
        traced = next((t for t in corpus.traces if t.trace_id == trace), None)
        if traced is None:
            console.print(f"[red]error:[/red] no trace {trace!r} in {artifact_id}")
            raise typer.Exit(code=1)
        table = Table("span_id", "parent_span_id", "kind", "name", "status", "output")
        for span in traced.spans:
            output = str(span.output)
            table.add_row(
                span.span_id,
                span.parent_span_id or "",
                span.kind.value,
                span.name,
                span.status.value,
                output[:80],
            )
        console.print(table)
        return

    table = Table("trace_id", "task", "spans")
    for traced in corpus.traces:
        task = (traced.task or "")[:60]
        table.add_row(traced.trace_id, task, str(len(traced.spans)))
    console.print(table)


@app.command()
def analyze(
    artifact_id: str,
    tasks: bool = typer.Option(False, "--tasks", help="List every extracted task candidate."),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Extract task candidates and outcome evidence from a stored corpus."""
    store = ArtifactStore(project / ".bandits")
    try:
        corpus = store.read(artifact_id)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no artifact {artifact_id!r}")
        raise typer.Exit(code=1) from exc

    analysis = analyze_corpus(corpus)
    envelope = save_analysis(analysis, DerivedStore(project / ".bandits"))

    console.print(f"analysis_id: {envelope.artifact_id}")
    console.print(f"corpus:      {analysis.corpus_id}")
    console.print(f"tasks:       {len(analysis.tasks)}")
    console.print(f"evidence:    {len(analysis.evidence)}")

    if tasks:
        table = Table("task_id", "instruction", "outcome evidence", "limitations")
        for task in analysis.tasks:
            table.add_row(
                task.task_id,
                (task.instruction or "[dim]none declared[/dim]")[:40],
                str(len(task.outcome_evidence_ids)),
                str(len(task.limitations)),
            )
        console.print(table)

    # Printed last and never suppressed: what could not be read matters as much
    # as what could, and burying it under a summary is how a corpus gets trusted
    # further than its evidence supports.
    for limitation in analysis.limitations:
        console.print(f"[yellow]limitation:[/yellow] {limitation}")


if __name__ == "__main__":
    app()
