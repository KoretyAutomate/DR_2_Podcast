"""Stage adapters, part two: translate, url_validation, draft, polish, audio.

Split from test_adapters.py to stay under the repo's file-size ceiling; see that file for what a
mutation matrix over adapters is testing.

Originally:

An adapter's job is to reconstruct, from the run directory alone, the state the monolithic runner
built in memory. What is tested here is that reconstruction and the fail-closed behaviour; the LLM
calls themselves are stubbed, because a test that needs vLLM up is a test that does not run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dr2_podcast import adapters
from dr2_podcast.artifacts import ArtifactError
from dr2_podcast.stage import write_run_config
from dr2_podcast.stages import ADAPTERS


@pytest.fixture(autouse=True)
def _no_backend_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let these tests depend on whether vLLM happens to be up.

    initialise_run_globals probes the backend before building the LLM handles. Left real, this file
    passes or fails according to what is running on the machine — which is how it passed in
    isolation and failed in the suite.
    """
    monkeypatch.setattr("dr2_podcast.pipeline.get_final_model_string", lambda: "test-model")


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    for sub in ("research", "scripts", "audio", "meta"):
        (tmp_path / sub).mkdir()
    write_run_config(tmp_path, topic="ビタミンDと骨折", language="ja", target_length_minutes=25)
    return tmp_path


RUN_CONFIG = {"topic": "ビタミンDと骨折", "language": "ja", "target_length_minutes": 25}


# --------------------------------------------------------------------------- #
# translate
# --------------------------------------------------------------------------- #
def test_translate_writes_the_translated_source_of_truth(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (run_dir / "research/source_of_truth.md").write_text("# Source of Truth\n\nBody.\n")
    monkeypatch.setattr(
        "dr2_podcast.pipeline._translate_sot_pipelined",
        lambda text, language, config: "# 真実の源\n\n本文。\n",
    )
    adapters.translate(run_dir, RUN_CONFIG)
    assert (run_dir / "research/source_of_truth_ja.md").read_text().startswith("# 真実の源")


def test_translate_does_nothing_for_an_english_episode(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The output is optional for exactly this reason."""

    def _never(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("an English episode has nothing to translate")

    monkeypatch.setattr("dr2_podcast.pipeline._translate_sot_pipelined", _never)
    adapters.translate(run_dir, {**RUN_CONFIG, "language": "en"})
    assert not list((run_dir / "research").glob("source_of_truth_*.md"))


def test_translate_fails_closed_on_an_empty_translation(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The phase returns None and carries on, building the episode from the wrong language."""
    (run_dir / "research/source_of_truth.md").write_text("# Source of Truth\n\nBody.\n")
    monkeypatch.setattr("dr2_podcast.pipeline._translate_sot_pipelined", lambda text, lang, cfg: "")
    with pytest.raises(ArtifactError, match="produced nothing"):
        adapters.translate(run_dir, RUN_CONFIG)
    assert not (run_dir / "research/source_of_truth_ja.md").exists()


# --------------------------------------------------------------------------- #
# url_validation
# --------------------------------------------------------------------------- #
def test_url_validation_reads_its_input_from_disk(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (run_dir / "research/research_sources.json").write_text(
        json.dumps(
            {
                "affirmative": [{"url": "https://example.org/a", "title": "A"}],
                "falsification": [{"url": "https://example.org/b"}],
            }
        )
    )
    checked: dict[str, Any] = {}

    def _fake_validate(urls: list[str], max_workers: int = 15) -> dict[str, str]:
        checked["urls"] = urls
        return dict.fromkeys(urls, "Valid")

    monkeypatch.setattr("dr2_podcast.tools.link_validator.validate_multiple_urls_parallel", _fake_validate)
    adapters.url_validation(run_dir, RUN_CONFIG)

    assert checked["urls"] == ["https://example.org/a", "https://example.org/b"]
    results = json.loads((run_dir / "research/url_validation_results.json").read_text())
    assert results["https://example.org/a"] == "Valid"


# prepush codex 2026-08-12: the phase removes broken URLs from the library; dropping that would
# leave staged runs citing sources the pipeline has already determined are unusable.
def test_url_validation_filters_the_broken_sources(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (run_dir / "research/research_sources.json").write_text(
        json.dumps(
            {
                "affirmative": [
                    {"url": "https://ok.example/a", "title": "good"},
                    {"url": "https://dead.example/b", "title": "broken"},
                ],
                "falsification": [{"url": "https://err.example/c"}],
            }
        )
    )
    monkeypatch.setattr(
        "dr2_podcast.tools.link_validator.validate_multiple_urls_parallel",
        lambda urls, max_workers=15: {
            "https://ok.example/a": "Valid (200)",
            "https://dead.example/b": "Broken (404)",
            "https://err.example/c": "ERROR: timeout",
        },
    )
    adapters.url_validation(run_dir, RUN_CONFIG)

    filtered = json.loads((run_dir / "research/research_sources_validated.json").read_text())
    assert [e["url"] for e in filtered["affirmative"]] == ["https://ok.example/a"]
    assert filtered["falsification"] == []

    untouched = json.loads((run_dir / "research/research_sources.json").read_text())
    assert len(untouched["affirmative"]) == 2, "the producer's own artifact is not edited"


# prepush codex 2026-08-12: LinkValidatorTool._run returns "✗ ERROR: …" with a leading marker, so
# the phase's startswith("ERROR") test misses it and an unusable citation proceeds downstream.
@pytest.mark.parametrize(
    "status",
    ["✗ ERROR: connection reset", "ERROR: timeout", "✗ Broken Link (Status: 404 Not Found)", "✗ Invalid URL: loop"],
)
def test_every_rejected_status_shape_is_filtered(status: str) -> None:
    sources = {"affirmative": [{"url": "https://bad.example/x"}, {"url": "https://good.example/y"}]}
    results = {"https://bad.example/x": status, "https://good.example/y": "✓ Valid (200)"}
    filtered = adapters._without_broken(sources, results)
    assert [e["url"] for e in filtered["affirmative"]] == ["https://good.example/y"], status


# prepush codex 2026-08-12: the filtered artifact was written and then read by nobody — the tools
# still opened research_sources.json, so rejected URLs reached the blueprint anyway.
def test_the_agents_read_the_validated_library_when_it_exists(run_dir: Path) -> None:
    from dr2_podcast.pipeline import research_sources_file

    (run_dir / "research/research_sources.json").write_text("{}")
    assert research_sources_file(run_dir).name == "research_sources.json"

    (run_dir / "research/research_sources_validated.json").write_text("{}")
    assert research_sources_file(run_dir).name == "research_sources_validated.json"


# prepush codex 2026-08-13: the legacy runner regenerates research_sources.json in place and knows
# nothing about the validated copy, so a directory that had once been driven by stages would keep
# serving sources from the previous research result.
def test_a_regenerated_library_wins_over_a_stale_validated_copy(run_dir: Path) -> None:
    import os

    from dr2_podcast.pipeline import research_sources_file

    validated = run_dir / "research/research_sources_validated.json"
    raw = run_dir / "research/research_sources.json"
    validated.write_text("{}")
    raw.write_text("{}")
    os.utime(validated, (1_000_000, 1_000_000))
    os.utime(raw, (2_000_000, 2_000_000))
    assert research_sources_file(run_dir).name == "research_sources.json"

    os.utime(validated, (3_000_000, 3_000_000))
    assert research_sources_file(run_dir).name == "research_sources_validated.json"


def test_validation_gates_the_blueprint() -> None:
    """Required, not optional: an optional gate lets the blueprint run before validation ever did,
    research_sources_file() falls back to the raw library, and rejected URLs reach the episode."""
    from dr2_podcast.stages import direct_producers, get_stage

    assert "research/research_sources_validated.json" in get_stage("blueprint").consumes
    assert "url_validation" in direct_producers("blueprint")


def test_url_validation_fails_closed_on_a_missing_sources_file(run_dir: Path) -> None:
    with pytest.raises(ArtifactError, match="cannot read"):
        adapters.url_validation(run_dir, RUN_CONFIG)


def test_urls_are_found_at_any_nesting_depth() -> None:
    """The sources document's shape has changed before; a shape-specific reader would miss URLs."""
    found = adapters._iter_urls(
        {"a": [{"url": "u1"}], "b": {"c": {"d": [{"url": "u2"}]}}, "url": "u3", "n": None}
    )
    assert sorted(found) == ["u1", "u2", "u3"]


# --------------------------------------------------------------------------- #
# draft and polish
# --------------------------------------------------------------------------- #
INVENTORY = {"Act 1": [{"question": "What is the claim?", "answer": "That vitamin D helps."}]}


def _script_inputs(run_dir: Path) -> None:
    (run_dir / "research/source_of_truth.md").write_text("# Source of Truth\n\nBody.\n")
    (run_dir / "meta/blueprint_inventory.json").write_text(json.dumps(INVENTORY))


def test_draft_builds_from_the_inventory_and_the_sot(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _script_inputs(run_dir)
    seen: dict[str, Any] = {}

    def _fake_sectional(inventory: dict, ctx: Any, *, _call_smart_model: Any) -> tuple[str, int]:
        seen.update(inventory=inventory, ctx=ctx)
        return "Host 1: hello\nHost 2: hi\n", 4200

    monkeypatch.setattr("dr2_podcast.pipeline._run_sectional_draft", _fake_sectional)
    adapters.draft(run_dir, RUN_CONFIG)

    assert seen["inventory"] == INVENTORY
    assert seen["ctx"].sot_content.startswith("# Source of Truth")
    assert seen["ctx"].session_roles, "the roles framing chose reach the draft"
    assert seen["ctx"].target_length_int > 0, "not the sentinel 0"
    assert (run_dir / "scripts/script_draft.md").read_text().startswith("Host 1:")


def test_draft_fails_closed_on_an_empty_script(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _script_inputs(run_dir)
    monkeypatch.setattr("dr2_podcast.pipeline._run_sectional_draft", lambda *a, **k: ("   ", 0))
    with pytest.raises(ArtifactError, match="produced no script"):
        adapters.draft(run_dir, RUN_CONFIG)
    assert not (run_dir / "scripts/script_draft.md").exists()


def test_draft_fails_closed_without_the_inventory(run_dir: Path) -> None:
    (run_dir / "research/source_of_truth.md").write_text("# Source of Truth\n")
    with pytest.raises(ArtifactError, match="cannot read"):
        adapters.draft(run_dir, RUN_CONFIG)


def test_polish_recomputes_the_draft_count_rather_than_reading_a_stored_one(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A number derivable from the artifact it describes is a second source of truth waiting to
    disagree with it — so it is recomputed with the same function phase 5 used."""
    _script_inputs(run_dir)
    draft_text = "Host 1: " + "word " * 300 + "\n"
    (run_dir / "scripts/script_draft.md").write_text(draft_text)
    seen: dict[str, Any] = {}

    def _fake_polish(text: str, count: int, inventory: dict, ctx: Any, refs: Any, max_attempts: int = 3) -> tuple:
        seen.update(text=text, count=count, inventory=inventory, refs=refs)
        return "Host 1: polished\n", None

    monkeypatch.setattr("dr2_podcast.pipeline._run_polish_loop", _fake_polish)
    adapters.polish(run_dir, RUN_CONFIG)

    from dr2_podcast import pipeline

    expected = pipeline._count_words(draft_text, pipeline.language_config)
    assert seen["count"] == expected
    assert (run_dir / "scripts/script_polished.md").read_text() == "Host 1: polished\n"


def test_polish_primes_the_task_the_loop_reads_the_draft_from(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _script_inputs(run_dir)
    (run_dir / "scripts/script_draft.md").write_text("Host 1: the draft\n")
    seen: dict[str, Any] = {}

    def _fake_polish(text: str, count: int, inventory: dict, ctx: Any, refs: Any, max_attempts: int = 3) -> tuple:
        seen["script_task_output"] = refs.script_task.output.raw
        seen["base_desc"] = refs.polish_base_desc
        return "polished", None

    monkeypatch.setattr("dr2_podcast.pipeline._run_polish_loop", _fake_polish)
    adapters.polish(run_dir, RUN_CONFIG)
    assert seen["script_task_output"] == "Host 1: the draft\n"
    assert seen["base_desc"], "the polish task's own description is the base in a fresh process"


# prepush codex 2026-08-13: a fresh process rebuilds the translation task EMPTY, so a Japanese
# episode would be polished against no translated evidence at all — and CrewAI context resolution
# can fail on an output-less task. The persisted translation is that output.
def test_polish_primes_the_translation_task_for_a_japanese_run(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _script_inputs(run_dir)
    (run_dir / "scripts/script_draft.md").write_text("Host 1: the draft\n")
    (run_dir / "research/source_of_truth_ja.md").write_text("# 翻訳された真実の源\n")
    seen: dict[str, Any] = {}

    def _fake_polish(text: str, count: int, inventory: dict, ctx: Any, refs: Any, max_attempts: int = 3) -> tuple:
        task = refs.translation_task
        seen["translated"] = task.output.raw if getattr(task, "output", None) else None
        return "polished", None

    monkeypatch.setattr("dr2_podcast.pipeline._run_polish_loop", _fake_polish)
    adapters.polish(run_dir, RUN_CONFIG)
    # A MARKER, not the text: the full SOT in a context task overflows the window and sends CrewAI
    # into an infinite summariser loop — pipeline.py:2419 records 36 cycles and 9.6 hours wasted.
    assert "Translation complete" in seen["translated"]
    assert "source_of_truth_ja.md" in seen["translated"]
    assert "翻訳された真実の源" not in seen["translated"]
    assert len(seen["translated"]) < 400


def test_polish_declares_the_translated_sot_it_polishes_against() -> None:
    """prepush codex 2026-08-13: it was loaded and used but not declared, so regenerating the
    translation left the polish current despite being based on obsolete evidence."""
    from dr2_podcast.stages import direct_producers, get_stage

    assert "research/source_of_truth_{language}.md" in get_stage("polish").optional_consumes
    assert "translate" in direct_producers("polish")


def test_an_english_run_leaves_the_translation_task_alone(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _script_inputs(run_dir)
    (run_dir / "scripts/script_draft.md").write_text("Host 1: the draft\n")
    seen: dict[str, Any] = {}

    def _fake_polish(text: str, count: int, inventory: dict, ctx: Any, refs: Any, max_attempts: int = 3) -> tuple:
        seen["output"] = getattr(refs.translation_task, "output", None)
        return "polished", None

    monkeypatch.setattr("dr2_podcast.pipeline._run_polish_loop", _fake_polish)
    adapters.polish(run_dir, {**RUN_CONFIG, "language": "en"})
    assert seen["output"] is None


def test_polish_fails_closed_on_an_empty_result(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _script_inputs(run_dir)
    (run_dir / "scripts/script_draft.md").write_text("Host 1: the draft\n")
    monkeypatch.setattr("dr2_podcast.pipeline._run_polish_loop", lambda *a, **k: ("  ", None))
    with pytest.raises(ArtifactError, match="produced no script"):
        adapters.polish(run_dir, RUN_CONFIG)
    assert not (run_dir / "scripts/script_polished.md").exists()


def test_both_script_stages_build_the_same_context(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One helper, not two copies: the two drifting apart is how a draft and its polish end up
    written to different targets."""
    _script_inputs(run_dir)
    (run_dir / "scripts/script_draft.md").write_text("Host 1: the draft\n")
    contexts: list[Any] = []

    def _record_draft(inventory: dict, ctx: Any, **kwargs: Any) -> tuple[str, int]:
        contexts.append(ctx)
        return "Host 1: x\n", 10

    def _record_polish(t: str, c: int, i: dict, ctx: Any, refs: Any, max_attempts: int = 3) -> tuple[str, None]:
        contexts.append(ctx)
        return "polished", None

    monkeypatch.setattr("dr2_podcast.pipeline._run_sectional_draft", _record_draft)
    monkeypatch.setattr("dr2_podcast.pipeline._run_polish_loop", _record_polish)
    adapters.draft(run_dir, RUN_CONFIG)
    adapters.polish(run_dir, RUN_CONFIG)

    first, second = contexts
    for field in ("target_length_int", "target_instruction", "session_roles", "sot_content"):
        assert getattr(first, field) == getattr(second, field), field


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #
def _render_into_staging(*files: str, duration: float = 24.5) -> Any:
    """A stand-in for _run_audio_pipeline that resolves its paths the way the real one does.

    It calls pipeline.output_path rather than joining strings, because that resolution is where the
    bug was: output_path falls back to a FLAT path when the subdirectory is missing, so a stand-in
    that writes to explicit subpaths would pass while the real renderer put everything in the wrong
    place.
    """

    def _fake_render(script_text: str, output_dir: Path, language_config: dict) -> tuple[Any, float]:
        from dr2_podcast.pipeline import output_path

        written = None
        for name in files:
            path = output_path(Path(output_dir), name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"RIFF" if name.endswith(".wav") else b"Host 1: hello")
            written = written or path
        return written, duration

    return _fake_render


def test_audio_renders_from_the_script_on_disk(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (run_dir / "scripts/script_final.md").write_text("Host 1: hello\nHost 2: hi\n")
    seen: dict[str, Any] = {}
    render = _render_into_staging("audio.wav", "audio_mixed.wav", "script.txt")

    def _spy(script_text: str, output_dir: Path, language_config: dict) -> tuple[Any, float]:
        seen.update(script=script_text, output_dir=output_dir, language_config=language_config)
        return render(script_text, output_dir, language_config)

    monkeypatch.setattr("dr2_podcast.pipeline._run_audio_pipeline", _spy)
    adapters.audio(run_dir, RUN_CONFIG)

    assert seen["script"].startswith("Host 1:")
    assert seen["output_dir"] != run_dir, "rendered into staging, not over the live artifacts"
    assert seen["language_config"]["speech_rate"]
    assert (run_dir / "audio/audio.wav").read_bytes() == b"RIFF", "promoted into place"
    assert (run_dir / "scripts/script.txt").exists()
    assert not (run_dir / "meta/.audio_staging").exists(), "staging cleaned up"


# prepush codex 2026-08-13: _run_audio_pipeline writes straight to the final paths, so an
# interrupted render destroyed the previous good audio or left a truncated WAV that looks finished
# — and a WAV's truncation is not visible until someone listens to it.
def test_a_failed_render_leaves_the_previous_audio_intact(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (run_dir / "scripts/script_final.md").write_text("Host 1: hello\n")
    good = run_dir / "audio/audio.wav"
    good.write_bytes(b"THE PREVIOUS GOOD RENDER")

    def _dies_midway(script_text: str, output_dir: Path, language_config: dict) -> tuple[Any, float]:
        from dr2_podcast.pipeline import output_path

        half = output_path(Path(output_dir), "audio.wav")
        half.parent.mkdir(parents=True, exist_ok=True)
        half.write_bytes(b"trunc")
        raise RuntimeError("TTS engine died")

    monkeypatch.setattr("dr2_podcast.pipeline._run_audio_pipeline", _dies_midway)
    with pytest.raises(RuntimeError, match="TTS engine died"):
        adapters.audio(run_dir, RUN_CONFIG)

    assert good.read_bytes() == b"THE PREVIOUS GOOD RENDER"
    assert not (run_dir / "meta/.audio_staging").exists()


def test_a_zero_duration_render_is_not_promoted(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (run_dir / "scripts/script_final.md").write_text("Host 1: hello\n")
    good = run_dir / "audio/audio.wav"
    good.write_bytes(b"THE PREVIOUS GOOD RENDER")
    monkeypatch.setattr(
        "dr2_podcast.pipeline._run_audio_pipeline", _render_into_staging("audio.wav", duration=0)
    )
    with pytest.raises(ArtifactError, match="no duration"):
        adapters.audio(run_dir, RUN_CONFIG)
    assert good.read_bytes() == b"THE PREVIOUS GOOD RENDER"


def test_audio_does_not_need_the_llm_backend(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rendering needs the TTS engines and the language config, not the Crews — building them would
    make audio unrenderable whenever vLLM happens to be down."""

    def _explode() -> str:
        raise AssertionError("the audio stage must not touch the LLM backend")

    monkeypatch.setattr("dr2_podcast.pipeline.get_final_model_string", _explode)
    (run_dir / "scripts/script_final.md").write_text("Host 1: hello\n")
    monkeypatch.setattr("dr2_podcast.pipeline._run_audio_pipeline", _render_into_staging("audio.wav"))
    adapters.audio(run_dir, RUN_CONFIG)


def test_audio_fails_closed_when_nothing_was_rendered(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The phase logs a warning and returns, so a run reaches its terminal state with no audio."""
    (run_dir / "scripts/script_final.md").write_text("Host 1: hello\n")
    monkeypatch.setattr("dr2_podcast.pipeline._run_audio_pipeline", lambda *a: (None, None))
    with pytest.raises(ArtifactError, match="produced no file"):
        adapters.audio(run_dir, RUN_CONFIG)


def test_audio_fails_closed_without_a_final_script(run_dir: Path) -> None:
    with pytest.raises(ArtifactError, match="cannot read"):
        adapters.audio(run_dir, RUN_CONFIG)


# --------------------------------------------------------------------------- #
# sot — deliberately NOT adapted; see the note in dr2_podcast/adapters.py
# --------------------------------------------------------------------------- #
def test_sot_has_no_adapter_and_the_reason_is_recorded() -> None:
    """Writing it proved its input artifact cannot exist in the assumed form: _serialize_dataclass
    repr-stringifies the report objects, so `audit` round-trips as the literal text
    "namespace(report='…')" and no rehydration can recover the structure the builder needs."""
    assert "sot" not in ADAPTERS
    source = Path(adapters.__file__).read_text()
    assert "repr-stringifies" in source, "the reason has to travel with the code"


def test_the_serialiser_really_does_destroy_the_report_structure() -> None:
    """The claim above, pinned. If this ever starts passing structure through, the sot adapter
    becomes writable and this test is the signal."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from gen_sot_golden import _pipeline_data, _reports

    from dr2_podcast.pipeline import _serialize_dataclass

    serialised = _serialize_dataclass(_reports(_pipeline_data()))
    assert isinstance(serialised["audit"], str), "a dict here would mean the structure survived"
    assert serialised["audit"].startswith("namespace(")
