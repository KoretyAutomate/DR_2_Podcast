"""Stage identity: what a fingerprint hashes, and what it deliberately does not.

Split out of test_manifest.py to stay under the repo's file-size ceiling. Every entry here answers
one question: would this change what the stage produces? If yes it is hashed; if no it is excluded,
with the reason stated. Identity is scoped per stage, so a setting one stage reads must not restale
another's forty-minute work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dr2_podcast.manifest import Manifest, config_fingerprint
from dr2_podcast.stages import (
    get_stage,
)

CONFIG = {"SMART_MODEL": "test-model", "LLM_BASE_URL": "http://localhost:8000/v1"}


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    for sub in ("research", "scripts", "audio", "meta"):
        (tmp_path / sub).mkdir()
    return tmp_path


def _write(run_dir: Path, artifact: str, text: str) -> None:
    path = run_dir / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _complete_framing(manifest: Manifest, run_dir: Path, framing: str = "framing v1") -> None:
    _write(run_dir, "research/research_framing.md", framing)
    _write(run_dir, "research/domain_classification.json", '{"domain": "clinical"}')
    _write(run_dir, "meta/session_roles.json", '{"hosts_setting": "", "roles": {"presenter": "Host 1"}}')
    manifest.start("framing", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.complete("framing")


def _complete_research(manifest: Manifest, run_dir: Path) -> None:
    for artifact in get_stage("research").produces:
        _write(run_dir, artifact, f"contents of {artifact}")
    manifest.start("research", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.complete("research")



def test_config_fingerprint_is_stable_and_sensitive() -> None:
    assert config_fingerprint(CONFIG) == config_fingerprint(dict(reversed(list(CONFIG.items()))))
    assert config_fingerprint(CONFIG) != config_fingerprint({**CONFIG, "LLM_BASE_URL": "http://elsewhere/v1"})


def test_config_fingerprint_reads_the_real_config_without_arguments() -> None:
    assert len(config_fingerprint()) == 64


# prepush codex 2026-08-12: an ALLOWLIST of output-affecting settings was wrong within a day — it
# named four and missed TTS_SPEED_SCALE, TTS_RANDOM_VOICE, TTS_INTONATION_OVERRIDES and the rest,
# so an .env change producing a different waveform left the audio stage "current". And it named
# LLM_BASE_URL, the ENV VAR, which the config module exposes as SMART_BASE_URL — so getattr hashed
# None forever and the endpoint invalidated nothing. The set is derived now, not maintained.
@pytest.mark.parametrize(
    "name",
    [
        "SMART_MODEL",
        "SMART_BASE_URL",
        "TTS_ENGINE_JA",
        "TTS_ENGINE_EN",
        "TTS_API_URL",
        "TTS_RANDOM_VOICE",
        "TTS_SPEED_SCALE",
        "TTS_SPEED_OVERRIDES",
        "TTS_INTONATION_SCALE",
        "TTS_INTONATION_OVERRIDES",
        "TTS_HOST1_ID",
        "TTS_HOST2_ID",
        "SCREENING_TOP_N",
        "TIER_CASCADE_THRESHOLD",
        "MIN_TIER3_STUDIES",
    ],
)
def test_every_output_affecting_setting_is_part_of_identity(name: str) -> None:
    from dr2_podcast.manifest import config_identity_values

    assert name in config_identity_values(), f"{name} can change output but does not invalidate a stage"


def test_changing_any_identity_setting_changes_the_fingerprint() -> None:
    from dr2_podcast.manifest import config_identity_values

    base = config_identity_values()
    baseline = config_fingerprint(base)
    for name, value in base.items():
        altered = "sentinel" if not isinstance(value, bool) else not value
        assert config_fingerprint({**base, name: altered}) != baseline, name


def test_the_excluded_settings_each_have_a_stated_reason() -> None:
    """A denylist is only safe while every entry is justified where it is written."""
    import dr2_podcast.manifest as manifest_module
    from dr2_podcast.manifest import CONFIG_IDENTITY_EXCLUDE

    lines = Path(manifest_module.__file__).read_text().splitlines()
    for name in CONFIG_IDENTITY_EXCLUDE:
        entry = next(i for i, line in enumerate(lines) if line.strip().startswith(f'"{name}"'))
        assert lines[entry - 1].strip().startswith("#"), f"{name} is excluded with no comment saying why"


def test_the_output_root_is_not_part_of_identity() -> None:
    """Where runs are written is not what they contain; hashing it would stale every stage on a
    machine with a different output root."""
    from dr2_podcast.manifest import config_identity_values

    assert "OUTPUT_DIR_OVERRIDE" not in config_identity_values()


def test_dict_ordering_cannot_move_the_fingerprint() -> None:
    overrides = {1138003200: 1.0, 1937616896: 1.2}
    reversed_overrides = dict(reversed(list(overrides.items())))
    assert config_fingerprint({"TTS_SPEED_OVERRIDES": overrides}) == config_fingerprint(
        {"TTS_SPEED_OVERRIDES": reversed_overrides}
    )


# prepush codex 2026-08-12: limiting this to the initialiser missed PODCAST_HOSTS in assign_roles
# and TTS_GLOSSARY_ENABLED in the audio engine — both change what a run produces while completed
# stages compared as current. The scan now covers the whole package, and anything new has to be
# classified as content-affecting or explicitly excluded before this passes.
def test_every_environment_read_in_the_package_is_classified() -> None:
    import re

    from dr2_podcast.manifest import CONTENT_ENV_KEYS, ENV_IDENTITY_EXCLUDE

    # [A-Z][A-Z0-9_]* — the first version was [A-Z_]+, which silently skipped every variable with a
    # DIGIT in it: TTS_HOST1_ID, TTS_HOST2_ID, S2_API_KEY. The guard passed by not looking, which is
    # the failure mode it exists to prevent.
    name = r"[A-Z][A-Z0-9_]*"
    pattern = re.compile(
        rf"os\.(?:getenv|environ\.get)\(\s*[\"']({name})[\"']|os\.environ\[\s*[\"']({name})[\"']"
    )
    package = Path(__file__).resolve().parent.parent / "dr2_podcast"
    found: set[str] = set()
    for source in package.rglob("*.py"):
        for a, b in pattern.findall(source.read_text(encoding="utf-8")):
            found.add(a or b)
    assert found, "the scan matched nothing — it has stopped tracking how the package reads env"

    classified = set(CONTENT_ENV_KEYS) | ENV_IDENTITY_EXCLUDE
    unclassified = sorted(found - classified)
    assert not unclassified, (
        f"{unclassified} are read from the environment but are neither part of stage identity nor "
        f"excluded with a reason. Add each to CONTENT_ENV_KEYS or ENV_IDENTITY_EXCLUDE."
    )


@pytest.mark.parametrize("name", ["PODCAST_HOSTS", "TTS_GLOSSARY_ENABLED", "PODCAST_CHANNEL_INTRO"])
def test_the_settings_that_were_missed_are_now_in_the_fingerprint(name: str) -> None:
    from dr2_podcast.manifest import config_identity_values

    assert f"env:{name}" in config_identity_values()


def test_a_changed_channel_brief_invalidates_a_stage() -> None:
    from dr2_podcast.manifest import config_identity_values

    base = config_identity_values()
    altered = {**base, "env:PODCAST_CHANNEL_INTRO": "a completely different show"}
    assert config_fingerprint(base) != config_fingerprint(altered)


# prepush codex 2026-08-13: a global fingerprint coupled unrelated stages — changing
# TTS_SPEED_SCALE made framing and research non-current, and since producers must be current before
# a stage runs, an audio-only tweak forced the whole ~28-minute research chain to re-run first.
def test_an_audio_setting_does_not_invalidate_the_research_chain() -> None:
    from dr2_podcast.manifest import config_identity_values

    base = config_identity_values()
    changed = {**base, "TTS_SPEED_SCALE": 9.9}
    for upstream in ("framing", "research", "blueprint", "draft"):
        assert config_fingerprint(base, stage=upstream) == config_fingerprint(changed, stage=upstream), upstream
    assert config_fingerprint(base, stage="audio") != config_fingerprint(changed, stage="audio")


def test_a_model_change_still_invalidates_every_llm_stage() -> None:
    from dr2_podcast.manifest import config_identity_values

    base = config_identity_values()
    changed = {**base, "SMART_MODEL": "some-other-model"}
    for stage in ("framing", "research", "translate", "blueprint", "draft", "polish", "audit"):
        assert config_fingerprint(base, stage=stage) != config_fingerprint(changed, stage=stage), stage


def test_an_unmapped_stage_keeps_the_whole_configuration() -> None:
    """The safe default: a new stage over-invalidates rather than quietly ignoring a setting nobody
    remembered to classify."""
    from dr2_podcast.manifest import config_identity_values

    base = config_identity_values()
    assert config_fingerprint(base, stage="a_stage_nobody_mapped") == config_fingerprint(base)


def test_every_mapped_stage_is_a_real_stage() -> None:
    from dr2_podcast.manifest import STAGE_CONFIG_GROUPS
    from dr2_podcast.stages import STAGE_NAMES

    assert set(STAGE_CONFIG_GROUPS) <= set(STAGE_NAMES)


def test_every_available_stage_has_a_scoped_identity() -> None:
    """An unmapped stage still works, but silently taking the whole configuration is a decision
    somebody should have made on purpose."""
    from dr2_podcast.manifest import STAGE_CONFIG_GROUPS
    from dr2_podcast.stages import AVAILABLE_STAGE_NAMES

    assert set(AVAILABLE_STAGE_NAMES) <= set(STAGE_CONFIG_GROUPS)


# prepush codex 2026-08-13: configuration is not the only thing that changes what a stage produces.
# The TTS glossary is applied inside clean_script_for_tts, so editing it changes the rendered speech
# while the script stays byte-identical — PLAN.md Step 12 makes the same point.
def test_the_tts_glossary_is_part_of_the_audio_stage_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    import dr2_podcast.manifest as manifest_module

    base = manifest_module.config_identity_values()
    before = config_fingerprint(base, stage="audio")
    monkeypatch.setattr(
        manifest_module, "_data_asset_values", lambda stage: {"data:glossary": "a different glossary"}
    )
    assert config_fingerprint(base, stage="audio") != before


def test_a_data_asset_belongs_only_to_the_stage_that_reads_it() -> None:
    from dr2_podcast.manifest import _data_asset_values

    assert _data_asset_values("audio"), "audio depends on the glossary"
    assert _data_asset_values("research") == {}, "research does not read it, so it must not stale on it"


def test_a_missing_data_asset_is_recorded_rather_than_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent is a state, not a non-answer: a deleted glossary changes the render too."""
    import dr2_podcast.manifest as manifest_module

    monkeypatch.setattr(manifest_module, "STAGE_DATA_ASSETS", {"audio": ("dr2_podcast/data/nonesuch.json",)})
    assert manifest_module._data_asset_values("audio") == {"data:dr2_podcast/data/nonesuch.json": None}


def test_the_run_config_is_part_of_identity() -> None:
    run_config = {"topic": "A", "language": "ja", "target_length_minutes": 25}
    assert config_fingerprint(CONFIG, run_config=run_config) != config_fingerprint(CONFIG)
    assert config_fingerprint(CONFIG, run_config=run_config) != config_fingerprint(
        CONFIG, run_config={**run_config, "topic": "B"}
    )


def test_run_config_bookkeeping_fields_are_not_part_of_identity() -> None:
    """Rewriting the file with the same parameters must not invalidate completed work."""
    run_config = {"topic": "A", "language": "ja", "target_length_minutes": 25}
    assert config_fingerprint(CONFIG, run_config=run_config) == config_fingerprint(
        CONFIG, run_config={**run_config, "created_at": "2026-08-12T00:00:00+09:00", "notes": "hi"}
    )


# prepush codex 2026-08-13: PODCAST_LENGTH picks a mode from a table for the monolithic runner, but
# a staged run passes target_length_minutes into initialise_run_globals and that argument overrides
# the lookup. Hashing the env var into staged identity made framing — and the whole research and
# script chain behind it — non-current whenever an unrelated shell setting moved.
def test_a_staged_fingerprint_ignores_the_env_var_its_run_config_supersedes() -> None:
    values = {"env:PODCAST_LENGTH": "long", "env:MODEL_NAME": "m"}
    run_config = {"topic": "t", "language": "en", "target_length_minutes": 25}
    before = config_fingerprint(values, run_config, "framing")
    after = config_fingerprint({**values, "env:PODCAST_LENGTH": "short"}, run_config, "framing")
    assert before == after, "the staged path never reads it"


def test_the_effective_length_still_moves_a_staged_fingerprint() -> None:
    """The control: what the staged run actually uses must still invalidate."""
    values = {"env:PODCAST_LENGTH": "long", "env:MODEL_NAME": "m"}
    base = {"topic": "t", "language": "en", "target_length_minutes": 25}
    assert config_fingerprint(values, base, "framing") != config_fingerprint(
        values, {**base, "target_length_minutes": 12}, "framing"
    )


def test_the_legacy_fingerprint_still_hashes_it() -> None:
    """Without a run config there is nothing to supersede it, and it really does steer the run."""
    values = {"env:PODCAST_LENGTH": "long", "env:MODEL_NAME": "m"}
    assert config_fingerprint(values, None, "framing") != config_fingerprint(
        {**values, "env:PODCAST_LENGTH": "short"}, None, "framing"
    )


# prepush codex 2026-08-13 [P1]: identity hashed configuration and run parameters but not the code.
# Deploy a change to an adapter and every existing run still reported the stage current, so the
# runner skipped it and the run kept artifacts the current implementation would not produce.
def test_a_changed_implementation_makes_a_stage_not_current(tmp_path) -> None:
    from dr2_podcast import manifest as manifest_mod

    values = {"env:MODEL_NAME": "m"}
    before = config_fingerprint(values, None, "audio")

    # A stage whose implementation file cannot be read has a DIFFERENT fingerprint, which is the
    # same property as "its bytes changed" — that is what makes this checkable without editing the
    # repo's own source mid-test.
    original = manifest_mod.STAGE_IMPLEMENTATION["audio"]
    manifest_mod.STAGE_IMPLEMENTATION["audio"] = original + ("dr2_podcast/not_a_real_module.py",)
    manifest_mod.implementation_closure.cache_clear()
    try:
        assert config_fingerprint(values, None, "audio") != before
    finally:
        manifest_mod.STAGE_IMPLEMENTATION["audio"] = original
        manifest_mod.implementation_closure.cache_clear()


def test_a_stages_identity_moves_only_for_its_own_code(tmp_path) -> None:
    """Per stage, not one build identifier: a typo fix in the audio engine must not restale the
    forty-minute research stage."""
    from dr2_podcast import manifest as manifest_mod

    values = {"env:MODEL_NAME": "m"}
    research_before = config_fingerprint(values, None, "research")

    original = manifest_mod.STAGE_IMPLEMENTATION["audio"]
    manifest_mod.STAGE_IMPLEMENTATION["audio"] = original + ("dr2_podcast/not_a_real_module.py",)
    manifest_mod.implementation_closure.cache_clear()
    try:
        assert config_fingerprint(values, None, "research") == research_before
    finally:
        manifest_mod.STAGE_IMPLEMENTATION["audio"] = original
        manifest_mod.implementation_closure.cache_clear()


def test_every_runnable_stage_declares_the_code_it_runs() -> None:
    """A stage missing from the map is one whose implementation nobody hashes — it would keep
    skipping across a deploy, silently, which is the defect this table exists for."""
    from dr2_podcast.manifest import STAGE_IMPLEMENTATION
    from dr2_podcast.stages import STAGES

    runnable = {stage.name for stage in STAGES if stage.available}
    assert runnable <= set(STAGE_IMPLEMENTATION), sorted(runnable - set(STAGE_IMPLEMENTATION))


def test_every_declared_implementation_file_exists() -> None:
    """A path that has been renamed hashes as None forever, which reads as 'unchanged'."""
    from pathlib import Path as _Path

    from dr2_podcast.manifest import STAGE_IMPLEMENTATION

    root = _Path(__file__).resolve().parent.parent
    for stage, relatives in STAGE_IMPLEMENTATION.items():
        for relative in relatives:
            assert (root / relative).exists(), f"{stage} names {relative}, which is not there"


# prepush codex 2026-08-13 [P2], twice. The hand-written table missed pipeline.py, then
# pipeline_flow.py, then prompt_strings.py — a list nobody can verify by reading it is not a
# guarantee, so what is hashed is now the IMPORT CLOSURE of the roots.
def test_every_stage_hashes_the_module_its_phase_functions_live_in() -> None:
    from dr2_podcast.manifest import implementation_closure

    for stage in ("framing", "research", "blueprint", "draft", "polish", "audit", "audio"):
        files = implementation_closure(stage)
        assert "dr2_podcast/pipeline.py" in files, stage
        assert "dr2_podcast/adapters/_common.py" in files, stage


def test_the_closure_reaches_what_a_curated_list_kept_missing() -> None:
    from dr2_podcast.manifest import implementation_closure

    # research calls _save_research_reports / _read_candidate_counts from pipeline_flow, imported
    # inside the adapter's function body — which is why the walk reads every Import node, not just
    # the ones at module level.
    assert "dr2_podcast/pipeline_flow.py" in implementation_closure("research")
    # the prompts that decide what blueprint, draft and polish produce
    for stage in ("blueprint", "draft", "polish"):
        assert "dr2_podcast/prompt_strings.py" in implementation_closure(stage), stage


def test_the_closures_are_near_identical_and_that_is_the_honest_outcome() -> None:
    """Measured, not assumed. pipeline.py is in every stage's roots and imports nearly the whole
    package, so per-stage roots do not buy per-stage precision: a code change anywhere restales
    every stage. Over-invalidation is the direction this module chooses — a re-run costs time,
    while under-invalidation ships artifacts built by code that no longer exists.

    If the phase functions ever move out of pipeline.py, this test is what will notice."""
    from dr2_podcast.manifest import implementation_closure

    research = set(implementation_closure("research"))
    audio = set(implementation_closure("audio"))
    assert "dr2_podcast/audio/engine.py" in research, "reachable through pipeline.py, today"
    # A RATIO, not a count: adding a research-only module widens the difference, which is the
    # healthy direction and must not fail this test. What is being asserted is that the shared core
    # still dominates — i.e. that per-stage roots are not yet buying per-stage precision. When that
    # stops being true, this fails and the comment above needs rewriting rather than the number.
    shared, differing = research & audio, research ^ audio
    assert len(shared) > 3 * len(differing), (
        f"{len(shared)} shared vs {len(differing)} differing — the closures have started to separate"
    )


def test_every_stage_hashes_the_adapter_module_that_registers_it() -> None:
    """The narrower half of the same property: a stage whose own adapter file is unhashed keeps
    skipping across a change to the code that runs it."""
    import inspect

    from dr2_podcast.manifest import STAGE_IMPLEMENTATION
    from dr2_podcast.stage import load_adapters
    from dr2_podcast.stages import ADAPTERS

    load_adapters()
    for stage, files in STAGE_IMPLEMENTATION.items():
        adapter = ADAPTERS.get(stage)
        if adapter is None:
            continue
        source = inspect.getsourcefile(adapter)
        assert source is not None
        module = Path(source).name
        assert any(f.endswith(module) for f in files), f"{stage} does not hash {module}"
