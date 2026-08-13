"""The manifest, the stage graph, and atomic artifact I/O — PLAN.md Step 8 + Step 1.

The property under test throughout is the one the plan's first draft asserted without designing:
that re-running one stage cannot leave a downstream artifact falsely current.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dr2_podcast.artifacts import (
    ArtifactError,
)
from dr2_podcast.manifest import Manifest, config_fingerprint, manifest_errors
from dr2_podcast.schemas import SchemaValidationError, load_example
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
    _write(run_dir, "meta/session_roles.json", '{"presenter": "Host 1"}')
    manifest.start("framing", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.complete("framing")


def _complete_research(manifest: Manifest, run_dir: Path) -> None:
    for artifact in get_stage("research").produces:
        _write(run_dir, artifact, f"contents of {artifact}")
    manifest.start("research", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.complete("research")


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #
def test_the_canonical_manifest_example_validates() -> None:
    assert manifest_errors(load_example("manifest")) == []


def test_a_fresh_run_starts_with_an_empty_manifest(run_dir: Path) -> None:
    manifest = Manifest.load(run_dir)
    assert manifest.status("framing") == "pending"
    assert not manifest.is_current("framing")


def test_a_completed_stage_is_current_and_round_trips(run_dir: Path) -> None:
    manifest = Manifest.load(run_dir)
    _complete_framing(manifest, run_dir)
    manifest.save()

    reloaded = Manifest.load(run_dir)
    assert reloaded.status("framing") == "complete"
    assert reloaded.is_current("framing", config_sha256=config_fingerprint(CONFIG))
    assert reloaded.document == manifest.document


def test_a_changed_config_makes_a_stage_not_current(run_dir: Path) -> None:
    """`.env` or model changes count as input changes — the same inputs under a different model
    are not the same run."""
    manifest = Manifest.load(run_dir)
    _complete_framing(manifest, run_dir)
    other = config_fingerprint({**CONFIG, "SMART_MODEL": "some-other-model"})
    assert not manifest.is_current("framing", config_sha256=other)
    assert manifest.is_current("framing", config_sha256=config_fingerprint(CONFIG))


def test_a_touched_output_makes_a_stage_not_current(run_dir: Path) -> None:
    manifest = Manifest.load(run_dir)
    _complete_framing(manifest, run_dir)
    _write(run_dir, "research/research_framing.md", "edited by hand")
    assert not manifest.is_current("framing")
    assert any("changed since this stage ran" in reason for reason in manifest.drift("framing"))


def test_a_deleted_artifact_counts_as_drift_not_as_unchanged(run_dir: Path) -> None:
    manifest = Manifest.load(run_dir)
    _complete_framing(manifest, run_dir)
    (run_dir / "research/domain_classification.json").unlink()
    assert any("missing" in reason for reason in manifest.drift("framing"))


# --------------------------------------------------------------------------- #
# The property the whole step exists for
# --------------------------------------------------------------------------- #
def test_rerunning_an_upstream_stage_marks_downstream_stale(run_dir: Path) -> None:
    """PLAN.md Step 8's exit criterion: re-run an upstream stage on a completed run dir and
    confirm every downstream stage is marked stale rather than reused."""
    manifest = Manifest.load(run_dir)
    _complete_framing(manifest, run_dir)
    _complete_research(manifest, run_dir)
    assert manifest.status("research") == "complete"

    # framing runs again and produces something different
    _complete_framing(manifest, run_dir, framing="framing v2 — different question")

    assert manifest.status("research") == "stale"
    assert "research/research_framing.md changed" in manifest.record_for("research")["stale_reason"]
    assert not manifest.is_current("research")


def test_an_unchanged_rerun_does_not_stale_downstream(run_dir: Path) -> None:
    """Staleness follows the hash, not the fact of a re-run — re-deriving the same bytes is a no-op."""
    manifest = Manifest.load(run_dir)
    _complete_framing(manifest, run_dir)
    _complete_research(manifest, run_dir)
    _complete_framing(manifest, run_dir)  # identical contents
    assert manifest.status("research") == "complete"


def test_staleness_reaches_a_stage_whose_own_inputs_have_not_moved_yet(run_dir: Path) -> None:
    """The case a purely hash-based rule misses. When framing changes, `research` is stale but has
    not re-run, so `sot`'s recorded inputs still hash the same. `sot` is nonetheless not current:
    it is consistent with artifacts that are known to be out of date, and `research` is about to
    re-run and change them."""
    manifest = Manifest.load(run_dir)
    _complete_framing(manifest, run_dir)
    _complete_research(manifest, run_dir)
    _write(run_dir, "research/source_of_truth.md", "sot v1")
    manifest.start("sot", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.complete("sot")
    assert manifest.status("sot") == "complete"

    _complete_framing(manifest, run_dir, framing="framing v2")
    assert manifest.status("research") == "stale"
    assert manifest.status("sot") == "stale"
    assert "research is not current" in manifest.record_for("sot")["stale_reason"]
    assert manifest.status("blueprint") == "pending", "never-run stages stay pending, not stale"


# prepush codex 2026-08-12 [P1]: the failure path marked only the failing stage, so a descendant
# whose own inputs happened not to move stayed falsely current behind a stage known to be broken.
def test_a_failed_rerun_invalidates_everything_behind_it(run_dir: Path) -> None:
    manifest = Manifest.load(run_dir)
    _complete_framing(manifest, run_dir)
    _complete_research(manifest, run_dir)
    _write(run_dir, "research/source_of_truth.md", "sot v1")
    _write(run_dir, "research/research_sources_validated.json", "{}")
    manifest.start("sot", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.complete("sot")
    for artifact in get_stage("blueprint").produces:
        _write(run_dir, artifact, f"blueprint v1: {artifact}")
    manifest.start("blueprint", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.complete("blueprint")
    assert manifest.status("blueprint") == "complete"

    # research is re-run, rewrites one output, and then fails
    _write(run_dir, "research/grade_synthesis.md", "half-rewritten")
    manifest.start("research", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.fail("research", "vLLM died mid-synthesis")
    manifest.invalidate_downstream("research")

    assert manifest.status("sot") == "stale"
    assert manifest.status("blueprint") == "stale", "a descendant cannot stay current behind a failure"
    assert "is not current" in manifest.record_for("blueprint")["stale_reason"]


def test_a_stage_that_did_not_write_what_it_promised_fails_closed(run_dir: Path) -> None:
    manifest = Manifest.load(run_dir)
    _write(run_dir, "research/research_framing.md", "only half the outputs")
    manifest.start("framing", model="test-model", config_sha256=config_fingerprint(CONFIG))
    with pytest.raises(ArtifactError, match="declared it produces"):
        manifest.complete("framing")


def test_an_unresolved_placeholder_is_never_hashed_as_a_literal_path() -> None:
    """Without substitutions the pattern is skipped, not written into the manifest verbatim."""
    from dr2_podcast.stages import resolve

    assert resolve(("research/source_of_truth_{language}.md",)) == ()
    assert resolve(("research/source_of_truth_{language}.md",), {"language": "ja"}) == (
        "research/source_of_truth_ja.md",
    )


# prepush codex 2026-08-12: blueprint reads grade_synthesis.md and the translated SOT, but declared
# neither — so regenerating either left an existing blueprint "current" and skipped, even though
# re-running it would have produced different output.
def test_an_optional_input_is_hashed_when_present(run_dir: Path) -> None:
    manifest = Manifest.load(run_dir)
    _complete_framing(manifest, run_dir)
    _complete_research(manifest, run_dir)
    _write(run_dir, "research/source_of_truth.md", "sot")
    _write(run_dir, "research/research_sources_validated.json", "{}")
    _write(run_dir, "research/source_of_truth_ja.md", "translated v1")
    for artifact in get_stage("blueprint").produces:
        _write(run_dir, artifact, f"blueprint: {artifact}")
    manifest.start("blueprint", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.complete("blueprint", {"language": "ja"})

    recorded = {ref["artifact"] for ref in manifest.record_for("blueprint")["inputs"]}
    assert "research/source_of_truth_ja.md" in recorded
    assert "research/grade_synthesis.md" in recorded

    _write(run_dir, "research/source_of_truth_ja.md", "translated v2 — different wording")
    assert not manifest.is_current("blueprint")


def test_an_absent_optional_input_is_not_a_failure(run_dir: Path) -> None:
    """An English episode has no translated SOT."""
    manifest = Manifest.load(run_dir)
    _complete_framing(manifest, run_dir)
    _complete_research(manifest, run_dir)
    _write(run_dir, "research/source_of_truth.md", "sot")
    _write(run_dir, "research/research_sources_validated.json", "{}")
    for artifact in get_stage("blueprint").produces:
        _write(run_dir, artifact, f"blueprint: {artifact}")
    manifest.start("blueprint", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.complete("blueprint")
    assert manifest.is_current("blueprint")


def test_optional_outputs_may_be_absent(run_dir: Path) -> None:
    """A translated SOT only exists for a non-English episode; its absence is not a failure."""
    manifest = Manifest.load(run_dir)
    _complete_framing(manifest, run_dir)
    _complete_research(manifest, run_dir)
    _write(run_dir, "research/source_of_truth.md", "sot")
    manifest.start("translate", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.complete("translate")
    assert manifest.status("translate") == "complete"
    assert manifest.record_for("translate")["outputs"] == []


# --------------------------------------------------------------------------- #
# Attempts: transport retries are not revision rounds
# --------------------------------------------------------------------------- #
def test_transport_retries_are_counted_separately(run_dir: Path) -> None:
    """Conflating a retry with a revision round silently shortens a loop bounded at three."""
    manifest = Manifest.load(run_dir)
    manifest.start("framing", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.record_attempt("framing", "transport", "empty response")
    manifest.record_attempt("framing", "transport", "malformed JSON")
    manifest.record_attempt("framing", "complete")
    assert manifest.transport_retries("framing") == 2
    assert [a["number"] for a in manifest.record_for("framing")["attempts"]] == [1, 2, 3]


def test_a_failed_stage_records_why(run_dir: Path) -> None:
    manifest = Manifest.load(run_dir)
    manifest.start("framing", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.fail("framing", "vLLM unreachable")
    manifest.save()
    assert Manifest.load(run_dir).record_for("framing")["stale_reason"] == "vLLM unreachable"


# --------------------------------------------------------------------------- #
# Modes and corruption
# --------------------------------------------------------------------------- #
def test_legacy_and_staged_manifests_are_separate_files(run_dir: Path) -> None:
    """Sequencing item 1: the two modes must not collide on filenames while both are live."""
    assert Manifest.path_for(run_dir, "staged") != Manifest.path_for(run_dir, "legacy")
    Manifest.load(run_dir, "staged").save()
    Manifest.load(run_dir, "legacy").save()
    assert (run_dir / "meta/manifest.json").exists()
    assert (run_dir / "meta/manifest_legacy.json").exists()


def test_opening_a_manifest_in_the_wrong_mode_raises(run_dir: Path) -> None:
    Manifest.load(run_dir, "legacy").save()
    (run_dir / "meta/manifest.json").write_text((run_dir / "meta/manifest_legacy.json").read_text())
    with pytest.raises(ArtifactError, match="opened as"):
        Manifest.load(run_dir, "staged")


def test_a_corrupt_manifest_raises_rather_than_resetting(run_dir: Path) -> None:
    """Resetting on a read error turns "I cannot tell what ran" into "nothing ran" — which is the
    pipeline.py:502 behaviour this replaces."""
    path = Manifest.path_for(run_dir, "staged")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not json")
    with pytest.raises(ArtifactError):
        Manifest.load(run_dir)


def test_a_manifest_that_violates_its_schema_raises_on_load(run_dir: Path) -> None:
    path = Manifest.path_for(run_dir, "staged")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "mode": "staged", "stages": {"framing": {}}}))
    with pytest.raises(SchemaValidationError):
        Manifest.load(run_dir)


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
