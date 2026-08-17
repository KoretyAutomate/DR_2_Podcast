"""Stage adapters, part three: the derived blueprint shape, promotion, and the publish sheet.

Split from test_adapters_scripts.py to stay under the repo's file-size ceiling; see test_adapters.py
for what a mutation matrix over adapters is testing.

What these pin is the part of a stage that is not the stage's own work: a run whose promotion fails
partway must leave the run directory exactly as it found it, and an artifact that names a path must
name one that still exists after promotion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dr2_podcast import adapters
from dr2_podcast.stage import write_run_config
from tests.test_adapters_scripts import RUN_CONFIG, _render_into_staging


@pytest.fixture(autouse=True)
def _no_backend_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let these tests depend on whether vLLM happens to be up. See test_adapters_scripts."""
    monkeypatch.setattr("dr2_podcast.pipeline.get_final_model_string", lambda: "test-model")


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    for sub in ("research", "scripts", "audio", "meta"):
        (tmp_path / sub).mkdir()
    write_run_config(tmp_path, topic="ビタミンDと骨折", language="ja", target_length_minutes=25)
    return tmp_path


# --------------------------------------------------------------------------- #
# The derived blueprint shape — PLAN.md Step 2
# --------------------------------------------------------------------------- #
def _pack_with(steps_absent=()) -> dict:
    def entry(n):
        absent = n in steps_absent
        return {
            "step": n, "question_ja": f"質問{n}",
            "answer": {"unavailable": "no frozen prior"} if absent else {"count": 1},
            "sot_sections": [] if absent else ["4.1"], "locators": [],
            "verdict_contribution": "neutral",
            "sufficiency": "absent" if absent else "complete",
        }

    steps = {str(n): entry(n) for n in (1, 2, 3, 4, 5, 6, 8, 9, 10)}
    steps["10"]["answer"] = {"confidence_ja": "高い", "grade_level": "moderate"}
    return {"schema_version": 1, "sot_domain": "clinical", "steps": steps}


def test_the_blueprint_stage_derives_the_episode_shape(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json

    from dr2_podcast.adapters.script_stages import _write_blueprint_scaffold

    (run_dir / "research/source_of_truth.md").write_text("# Source of Truth\n\nBody.\n")
    (run_dir / "research/step_pack.json").write_text(_json.dumps(_pack_with()))

    assert _write_blueprint_scaffold(run_dir) is True
    scaffold = _json.loads((run_dir / "research/blueprint.json").read_text())
    assert scaffold["authored"] is False, "the shape is derived; the words are Claude's"
    assert scaffold["opening"]["hedge_level"] == "高い"
    assert [s["step"] for s in scaffold["steps"]] == [1, 2, 3, 4, 5, 6, 8, 9, 10]


def test_no_step_pack_means_no_derived_shape(run_dir: Path) -> None:
    from dr2_podcast.adapters.script_stages import _write_blueprint_scaffold

    (run_dir / "research/source_of_truth.md").write_text("# Source of Truth\n")
    assert _write_blueprint_scaffold(run_dir) is False
    assert not (run_dir / "research/blueprint.json").exists()


def test_a_pack_missing_a_mandatory_step_declines_rather_than_inventing_one(
    run_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Today's real state: nothing writes the frozen prior, so steps 1 and 9 are absent. A scaffold
    whose opening stated a prior nobody set would be worse than no scaffold."""
    import json as _json
    import logging

    from dr2_podcast.adapters.script_stages import _write_blueprint_scaffold

    (run_dir / "research/source_of_truth.md").write_text("# Source of Truth\n")
    (run_dir / "research/step_pack.json").write_text(_json.dumps(_pack_with(steps_absent=(1, 9))))

    with caplog.at_level(logging.WARNING):
        assert _write_blueprint_scaffold(run_dir) is False
    assert not (run_dir / "research/blueprint.json").exists()
    assert any("事前確率" in record.message for record in caplog.records), "and it says which step"


# prepush codex 2026-08-13: promote() replaced targets one at a time, so an interruption partway
# through left a NEW script.txt beside an OLD wav — a mixed set, both files looking current, which
# is the exact state staging exists to prevent.
def test_a_promotion_that_fails_partway_puts_everything_back(run_dir: Path, monkeypatch) -> None:
    import os as _os

    from dr2_podcast.adapters._common import promote, staging_dir

    (run_dir / "scripts/script.txt").write_text("the previously accepted plain text")
    (run_dir / "audio/audio.wav").write_bytes(b"the previously accepted audio")

    real_replace = _os.replace
    calls = {"n": 0}

    def _dies_on_the_second_promotion(src, dst):
        # Every replace of a STAGED file is a promotion; the rollback set-asides are not counted.
        if str(src).startswith(str(run_dir / "meta")) and not str(src).endswith(".promote_rollback"):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("the disk went away mid-promotion")
        return real_replace(src, dst)

    with pytest.raises(OSError, match="mid-promotion"), staging_dir(run_dir) as staging:
        (staging / "scripts").mkdir(parents=True, exist_ok=True)
        (staging / "audio").mkdir(parents=True, exist_ok=True)
        (staging / "scripts/script.txt").write_text("the new plain text")
        (staging / "audio/audio.wav").write_bytes(b"the new audio")
        monkeypatch.setattr(_os, "replace", _dies_on_the_second_promotion)
        promote(staging, run_dir)

    monkeypatch.undo()
    assert (run_dir / "scripts/script.txt").read_text() == "the previously accepted plain text"
    assert (run_dir / "audio/audio.wav").read_bytes() == b"the previously accepted audio"
    assert not list(run_dir.rglob("*.promote_rollback")), "and it leaves no sidecars behind"


def test_a_complete_promotion_leaves_no_sidecars(run_dir: Path) -> None:
    from dr2_podcast.adapters._common import promote, staging_dir

    (run_dir / "scripts/script.txt").write_text("old")
    with staging_dir(run_dir) as staging:
        (staging / "scripts").mkdir(parents=True, exist_ok=True)
        (staging / "scripts/script.txt").write_text("new")
        assert promote(staging, run_dir) == ["scripts/script.txt"]

    assert (run_dir / "scripts/script.txt").read_text() == "new"
    assert not list(run_dir.rglob("*.promote_rollback"))


def test_a_first_promotion_that_fails_leaves_no_new_files_behind(run_dir: Path, monkeypatch) -> None:
    """Restoring the REPLACED files is not enough when the targets are new: an interrupted first
    render would otherwise leave a new audio.wav with no script.txt — a partial set either way."""
    import os as _os

    from dr2_podcast.adapters._common import promote, staging_dir

    real_replace = _os.replace
    calls = {"n": 0}

    def _dies_on_the_second_promotion(src, dst):
        if str(src).startswith(str(run_dir / "meta")):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("the disk went away mid-promotion")
        return real_replace(src, dst)

    with pytest.raises(OSError, match="mid-promotion"), staging_dir(run_dir) as staging:
        (staging / "audio").mkdir(parents=True, exist_ok=True)
        (staging / "audio/audio.wav").write_bytes(b"new audio")
        (staging / "audio/audio_mixed.wav").write_bytes(b"new mix")
        monkeypatch.setattr(_os, "replace", _dies_on_the_second_promotion)
        promote(staging, run_dir)

    monkeypatch.undo()
    assert not (run_dir / "audio/audio.wav").exists(), "a half-promoted render leaves nothing behind"
    assert not (run_dir / "audio/audio_mixed.wav").exists()


# prepush codex 2026-08-14, and it only appeared once both change-sets were committed together: the
# publish sheet records an ABSOLUTE audio path, and the staged audio adapter renders into
# meta/.stage_staging — so a sheet written during the render pointed into a directory that promote()
# was about to delete. A sheet whose one job is to name the file to upload, naming nothing.
def test_the_publish_sheet_is_written_after_promotion_not_during_the_render(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Path] = {}

    def _record(output_dir):
        seen["dir"] = Path(output_dir)
        return Path(output_dir) / "meta/publish_sheet.md"

    (run_dir / "scripts/script_final.md").write_text("Host 1: hello\n")
    monkeypatch.setattr("dr2_podcast.tools.publish_sheet.write_publish_sheet", _record)
    monkeypatch.setattr(
        "dr2_podcast.pipeline._run_audio_pipeline",
        _render_into_staging("audio.wav", "script.txt"),
    )
    adapters.audio(run_dir, RUN_CONFIG)

    assert seen["dir"] == run_dir, "the sheet must name the run, not the scratch tree"
    assert ".stage_staging" not in str(seen["dir"])


def test_the_monolithic_path_still_writes_its_own_sheet() -> None:
    """The suppression is scoped to staging, not to the feature."""
    from dr2_podcast.pipeline import _is_staging_dir

    assert _is_staging_dir(Path("/runs/ep1/meta/.stage_staging")) is True
    assert _is_staging_dir(Path("/runs/ep1")) is False
