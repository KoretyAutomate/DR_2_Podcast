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
    _write(run_dir, "meta/session_roles.json", '{"hosts_setting": "", "roles": {"presenter": "Host 1"}}')
    manifest.start("framing", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.complete("framing")


def _complete_framing_prior(manifest: Manifest, run_dir: Path) -> None:
    """plan_search consumes the prior, so the prior has to exist and have a producer of record."""
    for artifact in get_stage("framing_prior").produces:
        _write(run_dir, artifact, f"contents of {artifact}")
    manifest.start("framing_prior", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.complete("framing_prior")


def _complete_plan_search(manifest: Manifest, run_dir: Path) -> None:
    """Step 10 put a stage between framing and the search, so research has a producer to be current
    against — without this, research is stale for a reason that has nothing to do with the test."""
    for artifact in get_stage("plan_search").produces:
        _write(run_dir, artifact, f"contents of {artifact}")
    manifest.start("plan_search", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.complete("plan_search")


def _complete_research(manifest: Manifest, run_dir: Path) -> None:
    # Its inputs as well as its outputs: since Step 10 the stage consumes the two strategy files and
    # the approval, and Manifest.complete() hashes what it read.
    for artifact in get_stage("research").consumes:
        if not (run_dir / artifact).exists():
            _write(run_dir, artifact, f"contents of {artifact}")
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
    _complete_framing_prior(manifest, run_dir)
    _complete_plan_search(manifest, run_dir)
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
    _complete_framing_prior(manifest, run_dir)
    _complete_plan_search(manifest, run_dir)
    _complete_research(manifest, run_dir)
    _complete_framing(manifest, run_dir)  # identical contents
    assert manifest.status("research") == "complete"


def test_staleness_reaches_a_stage_whose_own_inputs_have_not_moved_yet(run_dir: Path) -> None:
    """The case a purely hash-based rule misses. When framing changes, `research` is stale but has
    not re-run, so `blueprint`'s recorded inputs still hash the same. `blueprint` is nonetheless not
    current: it is consistent with artifacts that are known to be out of date, and `research` is
    about to re-run and change them."""
    manifest = Manifest.load(run_dir)
    _complete_framing(manifest, run_dir)
    _complete_framing_prior(manifest, run_dir)
    _complete_plan_search(manifest, run_dir)
    _complete_research(manifest, run_dir)
    _write(run_dir, "research/research_sources_validated.json", "{}")
    for artifact in get_stage("blueprint").produces:
        _write(run_dir, artifact, f"blueprint v1: {artifact}")
    manifest.start("blueprint", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.complete("blueprint")
    assert manifest.status("blueprint") == "complete"

    _complete_framing(manifest, run_dir, framing="framing v2")
    assert manifest.status("research") == "stale"
    assert manifest.status("blueprint") == "stale"
    assert "research is not current" in manifest.record_for("blueprint")["stale_reason"]
    assert manifest.status("draft") == "pending", "never-run stages stay pending, not stale"


# prepush codex 2026-08-12 [P1]: the failure path marked only the failing stage, so a descendant
# whose own inputs happened not to move stayed falsely current behind a stage known to be broken.
def test_a_failed_rerun_invalidates_everything_behind_it(run_dir: Path) -> None:
    manifest = Manifest.load(run_dir)
    _complete_framing(manifest, run_dir)
    _complete_framing_prior(manifest, run_dir)
    _complete_plan_search(manifest, run_dir)
    _complete_research(manifest, run_dir)
    _write(run_dir, "research/research_sources_validated.json", "{}")
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
    _complete_framing_prior(manifest, run_dir)
    _complete_plan_search(manifest, run_dir)
    _complete_research(manifest, run_dir)
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


# prepush codex 2026-08-13: invalidation checked every producer the GRAPH allows, so an English
# blueprint — which records no translated SOT and never read one — went stale the next time any
# unrelated stage completed, purely because `translate` sits pending forever.
def test_a_producer_of_an_input_the_stage_never_read_does_not_stale_it(run_dir: Path) -> None:
    manifest = Manifest.load(run_dir)
    _complete_framing(manifest, run_dir)
    _complete_framing_prior(manifest, run_dir)
    _complete_plan_search(manifest, run_dir)
    _complete_research(manifest, run_dir)
    _write(run_dir, "research/research_sources_validated.json", "{}")
    for artifact in get_stage("blueprint").produces:
        _write(run_dir, artifact, f"blueprint: {artifact}")
    manifest.start("blueprint", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.complete("blueprint", {"language": "en"})
    assert manifest.status("blueprint") == "complete"

    recorded = {ref["artifact"] for ref in manifest.record_for("blueprint")["inputs"]}
    assert not any("source_of_truth_" in a for a in recorded), "an English run reads no translation"

    # url_validation completes with byte-identical output; translate is pending and always will be.
    for artifact in get_stage("url_validation").produces:
        _write(run_dir, artifact, "{}")
    manifest.start("url_validation", model="test-model", config_sha256=config_fingerprint(CONFIG))
    manifest.complete("url_validation")

    assert manifest.status("blueprint") == "complete", "a stage cannot stale on a producer it never read"


def test_an_absent_optional_input_is_not_a_failure(run_dir: Path) -> None:
    """An English episode has no translated SOT."""
    manifest = Manifest.load(run_dir)
    _complete_framing(manifest, run_dir)
    _complete_framing_prior(manifest, run_dir)
    _complete_plan_search(manifest, run_dir)
    _complete_research(manifest, run_dir)
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
    _complete_framing_prior(manifest, run_dir)
    _complete_plan_search(manifest, run_dir)
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
