"""The stage runner — PLAN.md Step 1.

Every guard here is one the monolithic runner does not have: it cannot skip a phase that is already
current, cannot refuse one whose inputs are absent, and cannot report what a re-run invalidated.
The adapters themselves are stubbed; what is under test is the orchestration around them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dr2_podcast import stage as stage_mod
from dr2_podcast.artifacts import ArtifactError
from dr2_podcast.manifest import Manifest
from dr2_podcast.schemas import SchemaValidationError
from dr2_podcast.stage import (
    StageError,
    load_run_config,
    main,
    run_stage,
    write_run_config,
)


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    for sub in ("research", "scripts", "audio", "meta"):
        (tmp_path / sub).mkdir()
    write_run_config(tmp_path, topic="ビタミンDと骨折", language="ja", target_length_minutes=25)
    return tmp_path


@pytest.fixture(autouse=True)
def _clean_adapters():
    """Adapters are module-global; no test may leak one into another."""
    original = dict(stage_mod.ADAPTERS)
    yield
    stage_mod.ADAPTERS.clear()
    stage_mod.ADAPTERS.update(original)


def _stub(name: str, writes: dict[str, str]) -> list[str]:
    """Register an adapter that writes fixed contents, and record when it ran."""
    calls: list[str] = []

    def _adapter(run_dir: Path, run_config: dict[str, Any]) -> None:
        calls.append(run_config["topic"])
        for artifact, text in writes.items():
            path = run_dir / artifact
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    stage_mod.ADAPTERS[name] = _adapter
    return calls


FRAMING_OUTPUTS = {
    "research/research_framing.md": "# framing\n",
    "research/domain_classification.json": '{"domain": "clinical"}',
}


# --------------------------------------------------------------------------- #
# run_config: the run's parameters as an artifact
# --------------------------------------------------------------------------- #
def test_run_config_round_trips(run_dir: Path) -> None:
    config = load_run_config(run_dir)
    assert config["topic"] == "ビタミンDと骨折"
    assert config["language"] == "ja"
    assert config["target_length_minutes"] == 25


def test_a_missing_run_config_stops_the_stage_with_advice(tmp_path: Path) -> None:
    (tmp_path / "meta").mkdir()
    _stub("framing", FRAMING_OUTPUTS)
    with pytest.raises(StageError, match="--topic"):
        run_stage(tmp_path, "framing")


def test_an_invalid_run_config_is_refused_rather_than_written(run_dir: Path) -> None:
    with pytest.raises(SchemaValidationError):
        write_run_config(run_dir, topic="", language="ja", target_length_minutes=25)
    assert load_run_config(run_dir)["topic"] == "ビタミンDと骨折", "the good version survives"


def test_a_corrupt_run_config_raises(run_dir: Path) -> None:
    (run_dir / "meta/run_config.json").write_text("{ not json")
    _stub("framing", FRAMING_OUTPUTS)
    with pytest.raises(ArtifactError):
        run_stage(run_dir, "framing")


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
def test_a_stage_that_is_not_separable_yet_says_so(run_dir: Path) -> None:
    """The six phase-1 sub-stages are declared but blocked on Step 10; the refusal names it."""
    with pytest.raises(StageError, match="not separable yet"):
        run_stage(run_dir, "keywords")


def test_a_stage_with_no_adapter_says_what_is_missing(run_dir: Path) -> None:
    with pytest.raises(StageError, match="no adapter yet"):
        run_stage(run_dir, "blueprint")


def test_an_unknown_stage_raises(run_dir: Path) -> None:
    with pytest.raises(KeyError):
        run_stage(run_dir, "nonesuch")


def test_a_stage_whose_inputs_are_absent_refuses_and_names_the_producer(run_dir: Path) -> None:
    _stub("research", {a: "x" for a in ("research/affirmative_case.md",)})
    with pytest.raises(StageError, match=r"missing input.*run stage 'framing'"):
        run_stage(run_dir, "research")


# --------------------------------------------------------------------------- #
# Running, skipping, forcing
# --------------------------------------------------------------------------- #
def test_a_stage_runs_records_and_reports(run_dir: Path) -> None:
    calls = _stub("framing", FRAMING_OUTPUTS)
    assert "complete" in run_stage(run_dir, "framing")
    assert calls == ["ビタミンDと骨折"], "the adapter received the run config from disk"

    manifest = Manifest.load(run_dir)
    assert manifest.status("framing") == "complete"
    recorded = {ref["artifact"] for ref in manifest.record_for("framing")["outputs"]}
    assert recorded == set(FRAMING_OUTPUTS)


def test_a_current_stage_is_skipped_not_rerun(run_dir: Path) -> None:
    """Re-running a current stage would stale everything downstream of it for no reason."""
    calls = _stub("framing", FRAMING_OUTPUTS)
    run_stage(run_dir, "framing")
    assert "skipped" in run_stage(run_dir, "framing")
    assert len(calls) == 1


def test_force_reruns_a_current_stage(run_dir: Path) -> None:
    calls = _stub("framing", FRAMING_OUTPUTS)
    run_stage(run_dir, "framing")
    run_stage(run_dir, "framing", force=True)
    assert len(calls) == 2


def test_rerunning_a_stage_reports_what_it_made_stale(run_dir: Path) -> None:
    """PLAN.md Step 1's exit criterion: `stage keywords` alone re-runs against an existing run dir —
    and the runner has to say what that invalidated rather than leaving it to be discovered."""
    _stub("framing", FRAMING_OUTPUTS)
    run_stage(run_dir, "framing")
    _stub("research", {a: f"contents of {a}" for a in stage_mod.get_stage("research").produces})
    run_stage(run_dir, "research")

    _stub("framing", {**FRAMING_OUTPUTS, "research/research_framing.md": "# a different framing\n"})
    outcome = run_stage(run_dir, "framing", force=True)
    assert "now stale: research" in outcome
    assert Manifest.load(run_dir).status("research") == "stale"


def test_a_failing_adapter_records_the_failure_and_reraises(run_dir: Path) -> None:
    def _explode(run_dir: Path, run_config: dict[str, Any]) -> None:
        raise RuntimeError("vLLM unreachable")

    stage_mod.ADAPTERS["framing"] = _explode
    with pytest.raises(RuntimeError, match="vLLM unreachable"):
        run_stage(run_dir, "framing")

    manifest = Manifest.load(run_dir)
    assert manifest.status("framing") == "failed"
    assert "vLLM unreachable" in manifest.record_for("framing")["stale_reason"]
    assert manifest.record_for("framing")["attempts"][-1]["outcome"] == "failed"


def test_a_stage_that_does_not_write_what_it_promised_fails_closed(run_dir: Path) -> None:
    _stub("framing", {"research/research_framing.md": "# only one of two outputs\n"})
    with pytest.raises(ArtifactError, match="declared it produces"):
        run_stage(run_dir, "framing")


def test_leftover_candidates_are_cleared_before_a_stage_runs(run_dir: Path) -> None:
    stray = run_dir / "research" / "research_framing.md.candidate"
    stray.write_text("half a file from a killed run")
    _stub("framing", FRAMING_OUTPUTS)
    assert "cleared 1 stale candidate" in run_stage(run_dir, "framing")
    assert not stray.exists()


# --------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------- #
def test_cli_runs_a_stage_and_exits_zero(run_dir: Path, capsys: pytest.CaptureFixture) -> None:
    _stub("framing", FRAMING_OUTPUTS)
    assert main(["framing", "--run", str(run_dir)]) == 0
    assert "complete" in capsys.readouterr().out


def test_cli_creates_the_run_config_from_topic(tmp_path: Path) -> None:
    (tmp_path / "meta").mkdir()
    _stub("framing", FRAMING_OUTPUTS)
    assert main(["framing", "--run", str(tmp_path), "--topic", "睡眠と記憶", "--language", "ja"]) == 0
    assert load_run_config(tmp_path)["topic"] == "睡眠と記憶"


def test_cli_reports_a_refusal_on_stderr_and_exits_one(run_dir: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["keywords", "--run", str(run_dir)]) == 1
    assert "not separable yet" in capsys.readouterr().err


def test_cli_rejects_a_missing_run_directory(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["framing", "--run", str(tmp_path / "nope")]) == 2
    assert "not a directory" in capsys.readouterr().err


def test_cli_status_lists_every_available_stage(run_dir: Path, capsys: pytest.CaptureFixture) -> None:
    _stub("framing", FRAMING_OUTPUTS)
    run_stage(run_dir, "framing")
    assert main(["framing", "--run", str(run_dir), "--status"]) == 0
    out = capsys.readouterr().out
    assert "framing" in out and "complete" in out
    assert "blueprint" in out and "pending" in out
