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
    assert persisted.record_for("framing")["attempts"][-1]["outcome"] == "failed"


def test_leftover_candidates_are_cleared_before_a_stage_runs(run_dir: Path) -> None:
    stray = run_dir / "research" / "research_framing.md.candidate"
    stray.write_text("half a file from a killed run")
    _stub("framing", FRAMING_OUTPUTS)
    assert "cleared 1 stale candidate" in run_stage(run_dir, "framing")
    assert not stray.exists()


# --------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------- #
# prepush codex 2026-08-12 [P1]: `sot` and `url_validation` are independent branches, so two
# stages against one run is a real shape. Both would load the manifest, both would save a private
# copy, and the later save would erase the other's status — and they share manifest.json.candidate.
def test_a_second_stage_refuses_while_another_holds_the_run(run_dir: Path) -> None:
    _stub("framing", FRAMING_OUTPUTS)
    with stage_mod.run_lock(run_dir), pytest.raises(StageError, match="another stage is already running"):
        run_stage(run_dir, "framing")


def test_the_lock_is_released_afterwards(run_dir: Path) -> None:
    _stub("framing", FRAMING_OUTPUTS)
    run_stage(run_dir, "framing")
    with stage_mod.run_lock(run_dir):
        pass  # acquiring again must not raise


def test_the_lock_is_released_even_when_a_stage_fails(run_dir: Path) -> None:
    def _explode(run_dir: Path, run_config: dict[str, Any]) -> None:
        raise RuntimeError("boom")

    stage_mod.ADAPTERS["framing"] = _explode
    with pytest.raises(RuntimeError):
        run_stage(run_dir, "framing")
    with stage_mod.run_lock(run_dir):
        pass


# prepush codex 2026-08-12 [P2]: SchemaValidationError was not in the CLI's handled set, so an
# invalid --topic produced a traceback instead of the intended ERROR line and exit code.
def test_cli_reports_a_schema_violation_instead_of_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    (tmp_path / "meta").mkdir()
    _stub("framing", FRAMING_OUTPUTS)
    assert main(["framing", "--run", str(tmp_path), "--topic", "", "--language", "ja"]) == 1
    assert "ERROR" in capsys.readouterr().err


# prepush codex 2026-08-12 [P1]: the config write happened before the lock was acquired, so an
# invocation could rewrite the topic of a run that was already executing — the running stage would
# carry on with the old parameters while the directory described the new ones.
def test_the_run_config_is_not_rewritten_while_another_stage_holds_the_run(run_dir: Path) -> None:
    _stub("framing", FRAMING_OUTPUTS)
    before = (run_dir / "meta/run_config.json").read_text()
    with stage_mod.run_lock(run_dir):
        assert main(["framing", "--run", str(run_dir), "--topic", "乗っ取られた話題"]) == 1
    assert (run_dir / "meta/run_config.json").read_text() == before


# prepush codex 2026-08-12 [P2]: `--topic ""` read as "option omitted" under a truthiness check, so
# the command silently ran against the previous topic instead of rejecting the request.
def test_an_empty_topic_is_rejected_rather_than_ignored(run_dir: Path, capsys: pytest.CaptureFixture) -> None:
    calls = _stub("framing", FRAMING_OUTPUTS)
    assert main(["framing", "--run", str(run_dir), "--topic", ""]) == 1
    assert "ERROR" in capsys.readouterr().err
    assert calls == [], "the stage must not run against a topic nobody asked for"
    assert load_run_config(run_dir)["topic"] == "ビタミンDと骨折"


# prepush codex 2026-08-12: the parser defaults were copied into the new config unconditionally, so
# changing the topic of an English 60-minute run silently made it a Japanese 25-minute one — and
# those fields are part of stage identity, so it invalidated every completed stage on the way past.
def test_changing_only_the_topic_keeps_the_other_settings(tmp_path: Path) -> None:
    (tmp_path / "meta").mkdir()
    write_run_config(tmp_path, topic="original", language="en", target_length_minutes=60)
    _stub("framing", FRAMING_OUTPUTS)
    assert main(["framing", "--run", str(tmp_path), "--topic", "a new topic"]) == 0

    config = load_run_config(tmp_path)
    assert config == {**config, "topic": "a new topic", "language": "en", "target_length_minutes": 60}


def test_a_first_run_config_still_gets_the_defaults(tmp_path: Path) -> None:
    (tmp_path / "meta").mkdir()
    _stub("framing", FRAMING_OUTPUTS)
    assert main(["framing", "--run", str(tmp_path), "--topic", "brand new"]) == 0
    config = load_run_config(tmp_path)
    assert config["language"] == "ja"
    assert config["target_length_minutes"] == 25


def test_an_explicit_option_still_overrides(tmp_path: Path) -> None:
    (tmp_path / "meta").mkdir()
    write_run_config(tmp_path, topic="original", language="en", target_length_minutes=60)
    _stub("framing", FRAMING_OUTPUTS)
    assert main(["framing", "--run", str(tmp_path), "--topic", "t", "--language", "ja"]) == 0
    assert load_run_config(tmp_path)["language"] == "ja"


# prepush codex 2026-08-12, round 2 on the same fix — three edge cases it introduced.
def test_language_alone_updates_an_existing_run(tmp_path: Path) -> None:
    (tmp_path / "meta").mkdir()
    write_run_config(tmp_path, topic="original", language="en", target_length_minutes=60)
    _stub("framing", FRAMING_OUTPUTS)
    assert main(["framing", "--run", str(tmp_path), "--language", "ja"]) == 0
    config = load_run_config(tmp_path)
    assert config["language"] == "ja"
    assert config["topic"] == "original", "the topic it did not mention is preserved"
    assert config["target_length_minutes"] == 60


def test_a_zero_target_length_is_rejected_not_defaulted(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    (tmp_path / "meta").mkdir()
    write_run_config(tmp_path, topic="original", language="en", target_length_minutes=60)
    _stub("framing", FRAMING_OUTPUTS)
    assert main(["framing", "--run", str(tmp_path), "--target-length", "0"]) == 1
    assert "ERROR" in capsys.readouterr().err
    assert load_run_config(tmp_path)["target_length_minutes"] == 60


def test_settings_without_a_topic_on_a_fresh_run_says_so(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    (tmp_path / "meta").mkdir()
    _stub("framing", FRAMING_OUTPUTS)
    assert main(["framing", "--run", str(tmp_path), "--language", "ja"]) == 1
    assert "Pass --topic" in capsys.readouterr().err


def test_a_corrupt_run_config_during_a_topic_update_is_an_error_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/run_config.json").write_text("{ not json")
    _stub("framing", FRAMING_OUTPUTS)
    assert main(["framing", "--run", str(tmp_path), "--topic", "t"]) == 1
    assert "ERROR" in capsys.readouterr().err


# prepush codex 2026-08-12: a free-string language let `--language fr` overwrite the run config and
# only then fail the stage with a KeyError from SUPPORTED_LANGUAGES.
def test_an_unsupported_language_is_rejected_before_it_is_committed(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    (tmp_path / "meta").mkdir()
    write_run_config(tmp_path, topic="original", language="en", target_length_minutes=60)
    _stub("framing", FRAMING_OUTPUTS)
    assert main(["framing", "--run", str(tmp_path), "--language", "fr"]) == 1
    assert "ERROR" in capsys.readouterr().err
    assert load_run_config(tmp_path)["language"] == "en", "the good config survives"


def test_the_language_enum_matches_the_supported_languages() -> None:
    """Two lists that must agree; this is what keeps them agreeing."""
    from dr2_podcast.pipeline import SUPPORTED_LANGUAGES
    from dr2_podcast.schemas import load_schema

    enum = load_schema("run_config")["properties"]["language"]["enum"]
    assert sorted(enum) == sorted(SUPPORTED_LANGUAGES)


# prepush codex 2026-08-12: the config was written before the manifest was loaded, so a corrupt
# manifest left the run described by parameters its artifacts were not generated from.
def test_a_corrupt_manifest_does_not_get_the_run_config_changed_underneath_it(run_dir: Path) -> None:
    _stub("framing", FRAMING_OUTPUTS)
    run_stage(run_dir, "framing")
    (run_dir / "meta/manifest.json").write_text("{ not json")
    before = load_run_config(run_dir)

    assert main(["framing", "--run", str(run_dir), "--topic", "something else"]) == 1
    assert load_run_config(run_dir) == before, "the run's source of truth was not touched"


# prepush codex 2026-08-12: the boundary caught only validation errors, so a backend that was down
# or any CrewAI failure produced a traceback instead of the intended ERROR line and exit code.
def test_cli_reports_an_adapter_failure_as_an_error(run_dir: Path, capsys: pytest.CaptureFixture) -> None:
    def _explode(run_dir: Path, run_config: dict[str, Any]) -> None:
        raise RuntimeError("no response from the LLM backend")

    stage_mod.ADAPTERS["framing"] = _explode
    assert main(["framing", "--run", str(run_dir)]) == 1
    assert "no response from the LLM backend" in capsys.readouterr().err
    assert Manifest.load(run_dir).status("framing") == "failed", "recorded before it was reported"


def test_a_deliberate_stop_is_not_swallowed(run_dir: Path) -> None:
    """KeyboardInterrupt derives from BaseException and must still reach the caller."""

    def _interrupt(run_dir: Path, run_config: dict[str, Any]) -> None:
        raise KeyboardInterrupt

    stage_mod.ADAPTERS["framing"] = _interrupt
    with pytest.raises(KeyboardInterrupt):
        main(["framing", "--run", str(run_dir)])


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


# prepush codex 2026-08-12 [P2]: --status returned outside the exception handler, so a corrupt
# manifest or run config produced a traceback instead of the documented ERROR line and exit code.
@pytest.mark.parametrize("artifact", ["meta/manifest.json", "meta/run_config.json"])
def test_cli_status_reports_a_corrupt_artifact_as_an_error(
    run_dir: Path, capsys: pytest.CaptureFixture, artifact: str
) -> None:
    _stub("framing", FRAMING_OUTPUTS)
    run_stage(run_dir, "framing")
    (run_dir / artifact).write_text("{ not json")
    assert main(["framing", "--run", str(run_dir), "--status"]) == 1
    assert "ERROR" in capsys.readouterr().err


# prepush codex 2026-08-12 [P1]: the CLI advertised every separable stage as though it could run,
# while ADAPTERS is empty in production — so every one of them refused. The two facts are kept
# apart (unavailable = the pipeline cannot separate it; not runnable = no adapter written yet) and
# the CLI now reports which is which instead of promising something it cannot do.
def test_nothing_is_advertised_as_runnable_without_an_adapter() -> None:
    stage_mod.ADAPTERS.clear()
    assert stage_mod.runnable_stage_names() == ()
    assert "Runnable now: NONE" in stage_mod.build_parser().format_help()


def test_a_registered_adapter_is_advertised_as_runnable() -> None:
    stage_mod.ADAPTERS.clear()
    _stub("framing", FRAMING_OUTPUTS)
    assert stage_mod.runnable_stage_names() == ("framing",)
    assert "Runnable now: framing" in stage_mod.build_parser().format_help()


def test_status_marks_stages_that_have_no_adapter(run_dir: Path, capsys: pytest.CaptureFixture) -> None:
    stage_mod.ADAPTERS.clear()
    assert main(["framing", "--run", str(run_dir), "--status"]) == 0
    out = capsys.readouterr().out
    assert "[no adapter]" in out
    assert "No stage adapter is registered" in out


def test_cli_status_lists_every_available_stage(run_dir: Path, capsys: pytest.CaptureFixture) -> None:
    _stub("framing", FRAMING_OUTPUTS)
    run_stage(run_dir, "framing")
    assert main(["framing", "--run", str(run_dir), "--status"]) == 0
    out = capsys.readouterr().out
    assert "framing" in out and "complete" in out
    assert "blueprint" in out and "pending" in out
