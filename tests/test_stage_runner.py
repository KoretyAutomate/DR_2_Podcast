"""The stage runner: refusals, guards, running, skipping, forcing — PLAN.md Step 1.

Every guard here is one the monolithic runner does not have.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dr2_podcast import stage as stage_mod
from dr2_podcast.artifacts import ArtifactError
from dr2_podcast.manifest import Manifest
from dr2_podcast.schemas import SchemaValidationError
from dr2_podcast.stage import StageError, load_run_config, run_stage, write_run_config

from tests._stage_fixtures import FRAMING_OUTPUTS, _clean_adapters, _stub, run_dir

__all__ = ["FRAMING_OUTPUTS", "_clean_adapters", "_stub", "run_dir"]


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
    """Every available stage HAS an adapter now, so this removes one to check the refusal survives —
    it is the message a future stage will meet before its adapter is written."""
    stage_mod.ADAPTERS.pop("blueprint", None)
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


# prepush codex 2026-08-12 [P1]: currency did not include the run config, so rewriting --topic on
# an existing run left every stage "current" and the runner skipped them — leaving artifacts about
# the old topic beside a config file describing the new one.
def test_changing_the_topic_makes_completed_stages_not_current(run_dir: Path) -> None:
    calls = _stub("framing", FRAMING_OUTPUTS)
    run_stage(run_dir, "framing")
    assert len(calls) == 1

    write_run_config(run_dir, topic="まったく別の話題", language="ja", target_length_minutes=25)
    assert "complete" in run_stage(run_dir, "framing")
    assert len(calls) == 2, "a stage completed for a different topic is not current for this one"


def test_rewriting_the_run_config_unchanged_does_not_invalidate(run_dir: Path) -> None:
    """created_at moves on every rewrite; only the semantic fields are part of identity."""
    calls = _stub("framing", FRAMING_OUTPUTS)
    run_stage(run_dir, "framing")
    write_run_config(run_dir, topic="ビタミンDと骨折", language="ja", target_length_minutes=25)
    assert "skipped" in run_stage(run_dir, "framing")
    assert len(calls) == 1


# prepush codex 2026-08-12 [P1]: existence is not currency. After a config change every upstream
# record stops being current without any file disappearing, so a downstream stage would consume
# artifacts built under the old configuration and record itself complete under the new one.
def test_a_stage_refuses_to_consume_outputs_of_a_stage_that_is_not_current(run_dir: Path) -> None:
    _stub("framing", FRAMING_OUTPUTS)
    run_stage(run_dir, "framing")
    research_calls = _stub("research", {a: f"contents of {a}" for a in stage_mod.get_stage("research").produces})

    write_run_config(run_dir, topic="別の話題", language="ja", target_length_minutes=25)
    with pytest.raises(StageError, match="are not current"):
        run_stage(run_dir, "research")
    assert research_calls == []


# prepush codex 2026-08-12: an optional input that is absent is not read, so demanding its producer
# be current made an English episode unable to run `blueprint` at all — `translate` produces the
# translated SOT that no English run has, and it does not even have an adapter.
def test_an_absent_optional_input_does_not_demand_its_producer(run_dir: Path) -> None:
    _stub("framing", FRAMING_OUTPUTS)
    run_stage(run_dir, "framing")
    _stub("research", {a: f"contents of {a}" for a in stage_mod.get_stage("research").produces})
    run_stage(run_dir, "research")
    _stub("url_validation", {a: "{}" for a in stage_mod.get_stage("url_validation").produces})
    run_stage(run_dir, "url_validation")

    calls = _stub("blueprint", {a: f"blueprint {a}" for a in stage_mod.get_stage("blueprint").produces})
    run_stage(run_dir, "blueprint")
    assert calls, "no translated SOT on disk, so translate is not a producer of anything read here"


def test_a_present_optional_input_does_demand_its_producer(run_dir: Path) -> None:
    _stub("framing", FRAMING_OUTPUTS)
    run_stage(run_dir, "framing")
    _stub("research", {a: f"contents of {a}" for a in stage_mod.get_stage("research").produces})
    run_stage(run_dir, "research")
    _stub("url_validation", {a: "{}" for a in stage_mod.get_stage("url_validation").produces})
    run_stage(run_dir, "url_validation")
    (run_dir / "research/source_of_truth_ja.md").write_text("translated, by nobody the manifest knows")

    _stub("blueprint", {a: f"blueprint {a}" for a in stage_mod.get_stage("blueprint").produces})
    with pytest.raises(StageError, match="translate"):
        run_stage(run_dir, "blueprint")


def test_force_consumes_the_artifacts_as_they_stand(run_dir: Path) -> None:
    """The escape hatch is explicit and named, not a silent default."""
    _stub("framing", FRAMING_OUTPUTS)
    run_stage(run_dir, "framing")
    research_calls = _stub("research", {a: f"contents of {a}" for a in stage_mod.get_stage("research").produces})
    write_run_config(run_dir, topic="別の話題", language="ja", target_length_minutes=25)
    run_stage(run_dir, "research", force=True)
    assert len(research_calls) == 1


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


# prepush codex 2026-08-12 [P2]: output hashing used to happen outside the failure handler, so an
# adapter that returned normally without writing what it declared left "running" on disk with no
# failed attempt — a stage reported as live after the process had exited.
def test_a_broken_output_contract_is_persisted_as_a_failure_not_left_running(run_dir: Path) -> None:
    _stub("framing", {"research/research_framing.md": "# only one of two outputs\n"})
    with pytest.raises(ArtifactError):
        run_stage(run_dir, "framing")

    persisted = Manifest.load(run_dir)
    assert persisted.status("framing") == "failed"
    assert "declared it produces" in persisted.record_for("framing")["stale_reason"]
    # prepush codex 2026-08-12: one execution must not leave both a complete and a failed attempt.
    outcomes = [a["outcome"] for a in persisted.record_for("framing")["attempts"]]
    assert outcomes == ["failed"], outcomes


def test_leftover_candidates_are_cleared_before_a_stage_runs(run_dir: Path) -> None:
    stray = run_dir / "research" / "research_framing.md.candidate"
    stray.write_text("half a file from a killed run")
    _stub("framing", FRAMING_OUTPUTS)
    assert "cleared 1 stale candidate" in run_stage(run_dir, "framing")
    assert not stray.exists()
