from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from bandits.analyze import load_task_set
from bandits.cli import app
from bandits.ingest.otlp import load_otlp
from bandits.store import DerivedStore, compute_artifact_id
from bandits.verify import load_interview, load_verifier_draft

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


_BLANK_ANSWERS = "\n" * 60
"""Enough blank answers to walk any draft's interview to completion."""

SUPPORT_FIXTURE = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "traces.support.otlp.jsonl"
)


def _mined(tmp_path: Path) -> str:
    """Ingest, analyze and mine the support fixture; return the task set id."""
    runner.invoke(
        app, ["ingest", str(SUPPORT_FIXTURE), "--source", "otlp", "--project", str(tmp_path)]
    )
    corpus_id = compute_artifact_id(load_otlp(SUPPORT_FIXTURE))
    analyzed = runner.invoke(app, ["analyze", corpus_id, "--project", str(tmp_path)])
    analysis_id = analyzed.stdout.split()[1]
    mined = runner.invoke(app, ["mine", analysis_id, "--budget", "10", "--project", str(tmp_path)])
    assert mined.exit_code == 0, mined.stdout
    return mined.stdout.split()[1]


def test_mine_reports_coverage_and_unfilled_slots(tmp_path) -> None:
    runner.invoke(
        app, ["ingest", str(SUPPORT_FIXTURE), "--source", "otlp", "--project", str(tmp_path)]
    )
    corpus_id = compute_artifact_id(load_otlp(SUPPORT_FIXTURE))
    analysis_id = runner.invoke(
        app, ["analyze", corpus_id, "--project", str(tmp_path)]
    ).stdout.split()[1]

    result = runner.invoke(app, ["mine", analysis_id, "--budget", "10", "--project", str(tmp_path)])

    assert result.exit_code == 0
    assert "taskset_id:  taskset-" in result.stdout
    assert "coverage:" in result.stdout
    assert "missing slot" in result.stdout


def test_families_shows_one_family_in_full(tmp_path) -> None:
    task_set_id = _mined(tmp_path)
    store = DerivedStore(tmp_path / ".bandits")
    family = load_task_set(task_set_id, store).families[0]

    result = runner.invoke(
        # The table truncates ids for width, so the id comes from the artifact.
        app,
        ["families", task_set_id, "--family", family.family_id, "--project", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert family.descriptor in result.stdout
    assert family.medoid_trace_id in result.stdout
    assert "held_out" in result.stdout


def test_families_rejects_an_unknown_family(tmp_path) -> None:
    task_set_id = _mined(tmp_path)

    result = runner.invoke(
        app, ["families", task_set_id, "--family", "family-nope", "--project", str(tmp_path)]
    )

    assert result.exit_code == 1


def test_merge_families_writes_a_new_task_set(tmp_path) -> None:
    task_set_id = _mined(tmp_path)
    store = DerivedStore(tmp_path / ".bandits")
    families = load_task_set(task_set_id, store).families

    result = runner.invoke(
        app,
        # fmt: off
        [
            "merge-families",
            task_set_id,
            families[0].family_id,
            families[1].family_id,
            "--project",
            str(tmp_path),
        ],
        # fmt: on
    )

    assert result.exit_code == 0
    merged_id = result.stdout.split()[1]
    assert merged_id != task_set_id
    assert load_task_set(task_set_id, store).families == families, "the original is untouched"
    assert len(load_task_set(merged_id, store).families) == len(families) - 1


def test_mine_unknown_analysis_exits_nonzero(tmp_path) -> None:
    result = runner.invoke(app, ["mine", "analysis-nope", "--project", str(tmp_path)])
    assert result.exit_code == 1


def test_draft_verifier_writes_suggested_replay_specs(tmp_path) -> None:
    task_set_id = _mined(tmp_path)
    store = DerivedStore(tmp_path / ".bandits")
    family = next(
        item for item in load_task_set(task_set_id, store).families if "refund" in item.descriptor
    )

    result = runner.invoke(
        app,
        [
            "draft-verifier",
            task_set_id,
            "--family",
            family.family_id,
            "--project",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    draft_id = result.stdout.split()[1]
    draft = load_verifier_draft(draft_id, store)
    assert draft.verifiers
    assert all(spec.status.value == "executable" for spec in draft.verifiers)
    assert all(spec.mode.value == "replay" for spec in draft.verifiers)


def test_draft_verifier_rejects_unknown_family(tmp_path) -> None:
    task_set_id = _mined(tmp_path)

    result = runner.invoke(
        app,
        [
            "draft-verifier",
            task_set_id,
            "--family",
            "family-nope",
            "--project",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1


def test_interview_verifier_completes_a_bounded_review(tmp_path) -> None:
    task_set_id = _mined(tmp_path)
    store = DerivedStore(tmp_path / ".bandits")
    family = next(
        item for item in load_task_set(task_set_id, store).families if "refund" in item.descriptor
    )
    drafted = runner.invoke(
        app,
        [
            "draft-verifier",
            task_set_id,
            "--family",
            family.family_id,
            "--project",
            str(tmp_path),
        ],
    )
    draft_id = drafted.stdout.split()[1]

    result = runner.invoke(
        app,
        ["interview-verifier", draft_id, "--project", str(tmp_path)],
        input=_BLANK_ANSWERS,
    )

    assert result.exit_code == 0, result.stdout
    assert "status:       complete" in result.stdout
    interview_id = next(
        line.split(maxsplit=1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("interview_id:")
    )
    interview = load_interview(interview_id, store)
    assert interview.complete
    assert len(interview.answers) == len(interview.questions)
    # Blind spots and gaming are asked of every check; an expected value only of
    # the checks that compare against one.
    assert all(
        question.check_id is not None
        for question in interview.questions
        if question.field == "expected"
    )
    assert interview.draft.verifiers[0].status.value == "executable"


def test_draft_verifier_can_run_the_interview_inline(tmp_path) -> None:
    task_set_id = _mined(tmp_path)
    store = DerivedStore(tmp_path / ".bandits")
    family = next(
        item for item in load_task_set(task_set_id, store).families if "refund" in item.descriptor
    )

    result = runner.invoke(
        app,
        [
            "draft-verifier",
            task_set_id,
            "--family",
            family.family_id,
            "--interview",
            "--project",
            str(tmp_path),
        ],
        input=_BLANK_ANSWERS,
    )

    assert result.exit_code == 0, result.stdout
    assert "verifier_draft_id:" in result.stdout
    assert "interview_id:" in result.stdout
    assert "status:       complete" in result.stdout


def _drafted_family(tmp_path, descriptor_word: str) -> tuple[str, str]:
    """Ingest, analyze, mine and draft; return (draft_id, family_id)."""
    task_set_id = _mined(tmp_path)
    store = DerivedStore(tmp_path / ".bandits")
    family = next(
        f for f in load_task_set(task_set_id, store).families if descriptor_word in f.descriptor
    )
    drafted = runner.invoke(
        app,
        # fmt: off
        [
            "draft-verifier",
            task_set_id,
            "--family",
            family.family_id,
            "--project",
            str(tmp_path),
        ],
        # fmt: on
    )
    assert drafted.exit_code == 0, drafted.stdout
    return drafted.stdout.split()[1], family.family_id


def test_label_then_validate_separates_the_right_check(tmp_path) -> None:
    """addr-1 changed the address; addr-2 and addr-3 only looked the order up."""
    draft_id, _ = _drafted_family(tmp_path, "address")

    labelled = runner.invoke(
        app,
        ["label", draft_id, "--labeler", "owner", "--project", str(tmp_path)],
        input="s\nchanged it\nf\nonly looked\nf\nonly looked\n",
    )
    assert labelled.exit_code == 0, labelled.stdout
    label_set_id = labelled.stdout.split("label_set_id:")[1].split()[0]

    result = runner.invoke(
        app,
        # fmt: off
        [
            "validate-verifier",
            draft_id,
            "--labels",
            label_set_id,
            "--project",
            str(tmp_path),
        ],
        # fmt: on
    )

    assert result.exit_code == 0, result.stdout
    assert "validation_id: validation-" in result.stdout
    # One hypothesis is right and one is wrong; the run where the wrong one
    # would have rewarded doing nothing is named.
    assert "100%" in result.stdout and "0%" in result.stdout
    assert "false_positive" in result.stdout
    assert "gamed" in result.stdout


def test_labeling_can_be_quit_early(tmp_path) -> None:
    draft_id, _ = _drafted_family(tmp_path, "address")

    result = runner.invoke(
        app,
        ["label", draft_id, "--labeler", "owner", "--project", str(tmp_path)],
        input="s\nfine\nq\n",
    )

    assert result.exit_code == 0
    assert "labels:       1" in result.stdout


def test_validate_verifier_needs_an_existing_label_set(tmp_path) -> None:
    draft_id, _ = _drafted_family(tmp_path, "address")

    result = runner.invoke(
        app,
        # fmt: off
        [
            "validate-verifier",
            draft_id,
            "--labels",
            "labels-nope",
            "--project",
            str(tmp_path),
        ],
        # fmt: on
    )

    assert result.exit_code == 1
