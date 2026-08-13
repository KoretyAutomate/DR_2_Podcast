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
    clear_candidates,
    read_json_strict,
    read_text_strict,
    sha256_file,
    write_atomic,
    write_json_atomic,
)
from dr2_podcast.manifest import Manifest, config_fingerprint, manifest_errors
from dr2_podcast.schemas import SchemaValidationError, load_example
from dr2_podcast.stages import (
    AVAILABLE_STAGE_NAMES,
    STAGE_NAMES,
    STAGES,
    direct_consumers,
    downstream_of,
    get_stage,
    producer_of,
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
    manifest.start("framing", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.complete("framing")


def _complete_research(manifest: Manifest, run_dir: Path) -> None:
    for artifact in get_stage("research").produces:
        _write(run_dir, artifact, f"contents of {artifact}")
    manifest.start("research", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.complete("research")


# --------------------------------------------------------------------------- #
# The stage graph
# --------------------------------------------------------------------------- #
def test_the_plan_s_fourteen_stage_names_are_all_declared() -> None:
    """Declared-but-unavailable, not omitted — the target shape stays visible."""
    for name in ("framing", "keywords", "search", "screen", "extract", "synthesize", "grade",
                 "sot", "translate", "blueprint", "draft", "polish", "audit", "audio"):
        assert name in STAGE_NAMES


def test_the_six_phase_one_substages_are_declared_unavailable_with_a_reason() -> None:
    """They cannot be separated before Step 10 splits _run_research_track; saying so beats pretending."""
    for name in ("keywords", "search", "screen", "extract", "synthesize", "grade"):
        stage = get_stage(name)
        assert not stage.available
        assert "Step 10" in stage.unavailable_reason
    assert get_stage("research").available, "the transitional composite has to be usable meanwhile"


def test_every_available_stage_declares_at_least_one_output() -> None:
    for name in AVAILABLE_STAGE_NAMES:
        stage = get_stage(name)
        assert stage.produces or stage.optional_outputs, name


def test_every_consumed_artifact_has_a_declared_producer() -> None:
    """An input nothing writes is a graph that cannot be resolved from disk."""
    for stage in STAGES:
        for artifact in stage.consumes:
            assert producer_of(artifact) is not None, f"{stage.name} consumes unproduced {artifact}"


def test_no_two_stages_claim_the_same_output() -> None:
    seen: dict[str, str] = {}
    for stage in STAGES:
        for artifact in stage.produces + stage.optional_outputs:
            assert artifact not in seen, f"{artifact} claimed by {seen.get(artifact)} and {stage.name}"
            seen[artifact] = stage.name


def test_downstream_is_transitive_and_ordered() -> None:
    assert direct_consumers("framing") == ("research",)
    chain = downstream_of("framing")
    assert {"research", "sot", "blueprint", "draft", "polish", "audit", "audio"} <= set(chain)
    assert chain.index("draft") < chain.index("polish") < chain.index("audit")


def test_the_graph_is_acyclic() -> None:
    for name in STAGE_NAMES:
        assert name not in downstream_of(name), f"{name} is downstream of itself"


def test_an_unavailable_stage_must_say_why() -> None:
    from dr2_podcast.stages import Stage

    with pytest.raises(ValueError, match="says nothing about why"):
        Stage("x", (), (), "python", available=False)


def test_unknown_stage_names_list_the_alternatives() -> None:
    with pytest.raises(KeyError, match="known:"):
        get_stage("nonesuch")


# --------------------------------------------------------------------------- #
# Atomic writes
# --------------------------------------------------------------------------- #
def test_write_atomic_returns_the_hash_it_wrote(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    digest = write_atomic(path, "hello")
    assert digest == sha256_file(path)
    assert path.read_text() == "hello"


def test_an_empty_write_is_refused(tmp_path: Path) -> None:
    """An empty artifact overwriting a good one is the silent degradation this prevents."""
    path = tmp_path / "a.md"
    write_atomic(path, "the good version")
    with pytest.raises(ArtifactError, match="empty"):
        write_atomic(path, "   \n")
    assert path.read_text() == "the good version"


def test_a_failed_validation_leaves_the_previous_version_intact(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    write_atomic(path, '{"good": true}')

    def _reject(payload: bytes) -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        write_atomic(path, '{"bad": true}', validate=_reject)
    assert path.read_text() == '{"good": true}'
    assert not list(tmp_path.glob("*.candidate")), "the candidate must be cleaned up"


def test_a_schema_violating_json_never_becomes_the_artifact(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    with pytest.raises(SchemaValidationError):
        write_json_atomic(path, {"schema_version": 1, "mode": "nonsense", "stages": {}}, schema="manifest")
    assert not path.exists()


def test_json_writes_refuse_nan(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_json_atomic(tmp_path / "x.json", {"x": float("nan")})


def test_read_json_strict_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "gone.json"
    with pytest.raises(ArtifactError, match="cannot read"):
        read_json_strict(missing)
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(ArtifactError, match="not valid JSON"):
        read_json_strict(broken)
    nan = tmp_path / "nan.json"
    nan.write_text('{"x": NaN}')
    with pytest.raises(ArtifactError, match="not valid JSON"):
        read_json_strict(nan)


def test_read_text_strict_refuses_an_empty_artifact(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("  \n ")
    with pytest.raises(ArtifactError, match="empty"):
        read_text_strict(path)


def test_leftover_candidates_are_removed_not_recovered(run_dir: Path) -> None:
    """A candidate means a run died between writing and renaming. It is never a valid artifact."""
    stray = run_dir / "research" / "sot.md.candidate"
    stray.write_text("half a file")
    assert clear_candidates(run_dir) == [stray]
    assert not stray.exists()


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
    assert "research is stale" in manifest.record_for("sot")["stale_reason"]
    assert manifest.status("blueprint") == "pending", "never-run stages stay pending, not stale"


def test_a_stage_that_did_not_write_what_it_promised_fails_closed(run_dir: Path) -> None:
    manifest = Manifest.load(run_dir)
    _write(run_dir, "research/research_framing.md", "only half the outputs")
    manifest.start("framing", model="test-model", config_sha256=config_fingerprint(CONFIG))
    with pytest.raises(ArtifactError, match="declared it produces"):
        manifest.complete("framing")


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


# prepush codex 2026-08-12 [P2]: the identity list named LLM_BASE_URL, which is the ENV VAR — the
# config module exposes it as SMART_BASE_URL (config.py:10). getattr therefore hashed None forever,
# so changing the endpoint invalidated nothing, contradicting the module's own stated contract.
def test_every_identity_key_exists_on_config() -> None:
    from dr2_podcast import config
    from dr2_podcast.manifest import CONFIG_IDENTITY_KEYS

    missing = [key for key in CONFIG_IDENTITY_KEYS if not hasattr(config, key)]
    assert not missing, f"identity keys absent from dr2_podcast.config, so they hash None: {missing}"


def test_the_endpoint_is_part_of_identity() -> None:
    from dr2_podcast import config
    from dr2_podcast.manifest import CONFIG_IDENTITY_KEYS

    base = {key: getattr(config, key, None) for key in CONFIG_IDENTITY_KEYS}
    assert config_fingerprint(base) != config_fingerprint({**base, "SMART_BASE_URL": "http://elsewhere/v1"})


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
