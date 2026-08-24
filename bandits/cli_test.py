from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from bandits.cli import app
from bandits.ingest.otlp import load_otlp
from bandits.store import compute_artifact_id

runner = CliRunner()
FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "traces.otlp.jsonl"


def test_ingest_prints_artifact_summary(tmp_path) -> None:
    result = runner.invoke(
        app, ["ingest", str(FIXTURE), "--source", "otlp", "--project", str(tmp_path)]
    )

    assert result.exit_code == 0
    assert "artifact_id: corpus-" in result.stdout
    assert "traces:      2" in result.stdout


def test_ingest_unknown_source_exits_nonzero(tmp_path) -> None:
    result = runner.invoke(
        app, ["ingest", str(FIXTURE), "--source", "nope", "--project", str(tmp_path)]
    )
    assert result.exit_code == 1


def test_list_shows_ingested_artifact(tmp_path) -> None:
    runner.invoke(app, ["ingest", str(FIXTURE), "--source", "otlp", "--project", str(tmp_path)])

    result = runner.invoke(app, ["list", "--project", str(tmp_path)])

    assert result.exit_code == 0
    assert "otlp" in result.stdout


def test_show_lists_traces_then_one_traces_spans(tmp_path) -> None:
    runner.invoke(app, ["ingest", str(FIXTURE), "--source", "otlp", "--project", str(tmp_path)])
    artifact_id = compute_artifact_id(load_otlp(FIXTURE))

    overview = runner.invoke(app, ["show", artifact_id, "--project", str(tmp_path)])
    assert overview.exit_code == 0
    assert "trace-1" in overview.stdout
    assert "trace-2" in overview.stdout

    detail = runner.invoke(
        app, ["show", artifact_id, "--trace", "trace-1", "--project", str(tmp_path)]
    )
    assert detail.exit_code == 0
    assert "lookup_order" in detail.stdout


CODING_FIXTURE = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "traces.coding.otlp.jsonl"
)


def test_analyze_reports_tasks_and_never_hides_limitations(tmp_path) -> None:
    runner.invoke(app, ["ingest", str(FIXTURE), "--source", "otlp", "--project", str(tmp_path)])
    artifact_id = compute_artifact_id(load_otlp(FIXTURE))

    result = runner.invoke(app, ["analyze", artifact_id, "--tasks", "--project", str(tmp_path)])

    assert result.exit_code == 0
    assert "analysis_id: analysis-" in result.stdout
    assert "task-trace-1" in result.stdout
    assert "limitation:" in result.stdout


def test_analyze_reads_a_coding_corpus_too(tmp_path) -> None:
    runner.invoke(
        app, ["ingest", str(CODING_FIXTURE), "--source", "otlp", "--project", str(tmp_path)]
    )
    artifact_id = compute_artifact_id(load_otlp(CODING_FIXTURE))

    result = runner.invoke(app, ["analyze", artifact_id, "--tasks", "--project", str(tmp_path)])

    assert result.exit_code == 0
    assert "task-code-1" in result.stdout


def test_analyze_unknown_artifact_exits_nonzero(tmp_path) -> None:
    result = runner.invoke(app, ["analyze", "corpus-nope", "--project", str(tmp_path)])
    assert result.exit_code == 1
