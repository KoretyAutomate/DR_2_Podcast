"""Stage adapters — PLAN.md Step 1's second half.

An adapter's job is to reconstruct, from the run directory alone, the state the monolithic runner
built in memory. What is tested here is that reconstruction and the fail-closed behaviour; the LLM
calls themselves are stubbed, because a test that needs vLLM up is a test that does not run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import logging

import pytest

from dr2_podcast import adapters
from dr2_podcast.adapters import _common, research_stages
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
# Registration
# --------------------------------------------------------------------------- #
def test_the_adapters_register_themselves_against_declared_stages() -> None:
    assert {
        "framing",
        "research",
        "url_validation",
        "translate",
        "blueprint",
        "draft",
        "polish",
        "audit",
        "audio",
    } <= set(ADAPTERS)


def test_registering_an_unknown_stage_is_refused() -> None:
    """A registry keyed by free strings would let a typo silently register nothing runnable."""
    from dr2_podcast.stages import register

    with pytest.raises(KeyError, match="unknown stage"):

        @register("nonesuch")
        def _adapter(run_dir: Path, run_config: dict[str, Any]) -> None:
            pass


# --------------------------------------------------------------------------- #
# The state reconstruction that makes a staged run possible at all
# --------------------------------------------------------------------------- #
def test_prepare_run_rebuilds_the_state_the_crews_read(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The globals the Crew builders read are set from the run config, in a process that never saw
    the monolithic runner's argv."""
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    pipeline = _common._prepare_run(run_dir, RUN_CONFIG)

    assert pipeline.output_dir == run_dir
    assert pipeline.topic_name == "ビタミンDと骨折"
    assert pipeline.language == "ja"
    assert pipeline.framing_task is not None, "the crew objects were constructed"
    assert pipeline.framing_agent is not None


def test_the_target_length_comes_from_the_run_config_not_the_environment(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PODCAST_LENGTH drives the monolithic runner; a staged run's minutes are part of its identity,
    so they have to be what actually applies."""
    monkeypatch.setenv("PODCAST_LENGTH", "short")
    pipeline = _common._prepare_run(run_dir, {**RUN_CONFIG, "target_length_minutes": 40})
    expected = 40 * pipeline.language_config["speech_rate"]
    assert pipeline.target_length_int == expected


# prepush codex 2026-08-12: _target_min is read directly by _create_agents_and_tasks and three more
# CrewBuildConfig sites. Left at its sentinel 0 by the extracted initialiser, every staged draft and
# polish prompt would have asked for a 0-minute episode while target_script carried the right count.
def test_the_target_minutes_global_is_set_not_left_at_its_sentinel(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _common._prepare_run(run_dir, {**RUN_CONFIG, "target_length_minutes": 33})
    assert pipeline._target_min == 33
    assert pipeline.target_length_int == 33 * pipeline.language_config["speech_rate"]


# prepush codex 2026-08-12: assign_roles() is random under the default PODCAST_HOSTS=random, and
# every stage is a fresh process — so framing, blueprint and the script phases would each build
# prompts with DIFFERENT host roles, with no manifest identity change to show for it.
def test_the_host_roles_are_assigned_once_and_then_reused(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODCAST_HOSTS", "random")
    first = _common._session_roles(run_dir)
    assert (run_dir / "meta/session_roles.json").exists()

    seen: list[int] = []

    def _reassign() -> dict[str, Any]:
        seen.append(1)
        return {"changed": True}

    monkeypatch.setattr("dr2_podcast.pipeline.assign_roles", _reassign)
    for _ in range(5):
        assert _common._session_roles(run_dir) == first
    assert seen == [], "a second process must read the roles, never reassign them"


# prepush codex 2026-08-13: a forced framing rerun after a transient failure reassigned the hosts
# under PODCAST_HOSTS=random, silently swapping presenter and questioner and invalidating every
# downstream script. Reassignment follows the SETTING changing, not the rerun happening.
def test_a_forced_rerun_keeps_the_roles_when_the_setting_is_unchanged(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PODCAST_HOSTS", "random")
    first = _common._session_roles(run_dir, reassign=True)
    for _ in range(5):
        assert _common._session_roles(run_dir, reassign=True) == first


def test_changing_the_hosts_setting_does_reassign(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODCAST_HOSTS", "random")
    _common._session_roles(run_dir, reassign=True)

    monkeypatch.setenv("PODCAST_HOSTS", "host1_leads")
    replacement = {"presenter": {"label": "Host 1"}, "questioner": {"label": "Host 2"}}
    monkeypatch.setattr("dr2_podcast.pipeline.assign_roles", lambda: replacement)
    assert _common._session_roles(run_dir, reassign=True) == replacement


# prepush codex 2026-08-13: a changed PODCAST_HOSTS makes framing stale, but rerunning it read the
# old assignment straight back while the manifest recorded the stage as current under the new one.
def test_framing_reassigns_the_roles_when_the_setting_changed(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dr2_podcast.pipeline import assign_roles

    monkeypatch.setenv("PODCAST_HOSTS", "host1_leads")

    old = assign_roles()
    new = {role: {**spec, "personality": "a different persona"} for role, spec in old.items()}
    # The stored shape carries the SETTING the roles were chosen under, which is what makes
    # "the configuration changed" answerable when the assignment itself is random.
    (run_dir / "meta/session_roles.json").write_text(
        json.dumps({"hosts_setting": "a previous setting", "roles": old})
    )
    monkeypatch.setattr("dr2_podcast.pipeline.assign_roles", lambda: new)
    _stub_framing(monkeypatch, "# framing")
    adapters.framing(run_dir, RUN_CONFIG)
    assert json.loads((run_dir / "meta/session_roles.json").read_text())["roles"] == new


def test_a_stage_that_only_reads_the_roles_leaves_them_alone(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dr2_podcast.pipeline import assign_roles

    chosen = assign_roles()
    other = {role: {**spec, "personality": "reassigned"} for role, spec in chosen.items()}
    (run_dir / "meta/session_roles.json").write_text(
        json.dumps({"hosts_setting": "", "roles": chosen}, ensure_ascii=False)
    )
    monkeypatch.setattr("dr2_podcast.pipeline.assign_roles", lambda: other)
    _common._prepare_run(run_dir, RUN_CONFIG)
    assert json.loads((run_dir / "meta/session_roles.json").read_text())["roles"] == chosen


def test_prepare_run_uses_the_persisted_roles(run_dir: Path) -> None:
    pipeline = _common._prepare_run(run_dir, RUN_CONFIG)
    stored = json.loads((run_dir / "meta/session_roles.json").read_text())
    assert stored["roles"] == pipeline.SESSION_ROLES


def test_initialise_run_globals_is_the_one_owner_of_that_state() -> None:
    """It was extracted from pipeline.py's __main__ so both runners share it. If an adapter
    reimplemented it, the two would drift and produce different episodes from the same inputs."""
    import inspect

    from dr2_podcast import pipeline

    assert "initialise_run_globals" in inspect.getsource(_common._prepare_run)
    assert callable(pipeline.initialise_run_globals)


# prepush codex 2026-08-12: a run directory given as a relative path outside cwd (`--run
# ../episode`) is itself a traversal, which CrewAI rejects exactly as it rejects the relpath form.
@pytest.mark.parametrize("shape", ["absolute", "relative-outside"])
def test_task_output_paths_never_contain_a_traversal(tmp_path: Path, shape: str) -> None:
    import os

    from dr2_podcast.pipeline_crew import _task_output_file

    target = tmp_path / "research" / "research_framing.md"
    given = target if shape == "absolute" else Path(os.path.relpath(target))
    assert ".." not in _task_output_file(given)


# --------------------------------------------------------------------------- #
# framing
# --------------------------------------------------------------------------- #
class _FakeOutput:
    def __init__(self, raw: str) -> None:
        self.raw = raw


class _FakeClassification:
    def __init__(self) -> None:
        from dr2_podcast.research.domain_classifier import ResearchDomain

        self.domain = ResearchDomain.CLINICAL
        self.confidence = 0.92
        self.reasoning = "vitamin D and fracture is a clinical question"
        self.suggested_framework = "PICO"
        self.primary_databases = ["PubMed", "Cochrane"]


def _stub_framing(monkeypatch: pytest.MonkeyPatch, produced: str) -> dict[str, Any]:
    """Stub the classifier and the Crew, recording what the crew was handed."""
    seen: dict[str, Any] = {}
    # The adapter resolves this in ITS module namespace, so the patch has to land there — patching
    # dr2_podcast.adapters would leave the real classifier running against a live backend.
    monkeypatch.setattr(research_stages, "_classify_domain", lambda topic: _FakeClassification())

    class _FakeCrew:
        def __init__(self, agents: list, tasks: list, **kwargs: Any) -> None:
            seen["task"] = tasks[0]
            seen["agents"] = agents

        def kickoff(self) -> None:
            seen["task"].output = _FakeOutput(produced)

    monkeypatch.setattr("crewai.Crew", _FakeCrew)
    return seen


def test_framing_writes_both_artifacts(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _stub_framing(monkeypatch, "# Research Framing\n\n## 1. Core Research Questions\n…")
    adapters.framing(run_dir, RUN_CONFIG)

    assert (run_dir / "research/research_framing.md").read_text().startswith("# Research Framing")
    classification = json.loads((run_dir / "research/domain_classification.json").read_text())
    assert classification["domain"] == "clinical"
    assert classification["databases"] == ["PubMed", "Cochrane"]
    assert seen["task"] is not None


def test_framing_appends_the_domain_directive(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _stub_framing(monkeypatch, "# framing")
    adapters.framing(run_dir, RUN_CONFIG)
    assert "DOMAIN CONTEXT" in seen["task"].description
    assert "PICO" in seen["task"].description


def test_framing_does_not_let_crewai_write_the_artifact(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The task carries an output_file, which would write the artifact unatomically mid-run."""
    seen = _stub_framing(monkeypatch, "# framing")
    adapters.framing(run_dir, RUN_CONFIG)
    assert seen["task"].output_file is None


def test_framing_fails_closed_on_an_empty_crew_output(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """phase_0_framing logs 'continuing' and returns '', sending the run into search with no
    framework. A stage that produced nothing has failed."""
    _stub_framing(monkeypatch, "   \n  ")
    with pytest.raises(ArtifactError, match="returned nothing"):
        adapters.framing(run_dir, RUN_CONFIG)
    assert not (run_dir / "research/research_framing.md").exists()


def test_a_social_science_topic_gets_the_peco_directive(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dr2_podcast.research.domain_classifier import ResearchDomain

    classification = _FakeClassification()
    classification.domain = ResearchDomain.SOCIAL_SCIENCE
    note = _common._domain_note(classification)
    assert "PECO" in note
    assert "Do NOT use clinical terminology" in note


# --------------------------------------------------------------------------- #
# research
# --------------------------------------------------------------------------- #
def _research_inputs(run_dir: Path) -> None:
    (run_dir / "research/research_framing.md").write_text("# Research Framework\n\nQuestions.\n")
    (run_dir / "research/domain_classification.json").write_text('{"domain": "clinical"}')


def _stub_research(
    monkeypatch: pytest.MonkeyPatch,
    *,
    aff: int = 50,  # comfortably above EVIDENCE_LIMITED_THRESHOLD
    neg: int = 12,
    sot: str = "# Source of Truth\n\n## Abstract\n…",
    reports: Any = None,
) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    async def _fake_deep_research(*, topic: str, config: Any, framing_context: str, output_dir: str) -> Any:
        seen.update(topic=topic, framing=framing_context, domain=config.domain, output_dir=output_dir)
        return reports if reports is not None else {"audit": object()}

    monkeypatch.setattr("dr2_podcast.research.clinical.run_deep_research", _fake_deep_research)
    monkeypatch.setattr("dr2_podcast.pipeline_flow._read_candidate_counts", lambda d, log: (aff, neg))
    monkeypatch.setattr("dr2_podcast.pipeline_flow._save_research_reports", lambda r, d, log: None)
    monkeypatch.setattr("dr2_podcast.pipeline_flow._save_sources_json", lambda r, d, log: None)

    def _fake_sot(*, topic: str, reports: Any, domain: str) -> str:
        seen["sot_domain"] = domain
        return sot

    monkeypatch.setattr("dr2_podcast.pipeline.build_imrad_sot", _fake_sot)
    return seen


def test_research_runs_and_writes_the_source_of_truth(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The SOT is produced HERE because phase 1 produces it here — on the live reports dict that
    cannot cross a process boundary."""
    _research_inputs(run_dir)
    seen = _stub_research(monkeypatch)
    research_stages.research(run_dir, RUN_CONFIG)

    assert seen["topic"] == "ビタミンDと骨折"
    assert seen["framing"].startswith("# Research Framework")
    assert seen["domain"] == "clinical"
    assert seen["sot_domain"] == "clinical"
    assert (run_dir / "research/source_of_truth.md").read_text().startswith("# Source of Truth")


def test_research_takes_the_domain_from_the_classification_artifact(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _research_inputs(run_dir)
    (run_dir / "research/domain_classification.json").write_text('{"domain": "social_science"}')
    seen = _stub_research(monkeypatch)
    research_stages.research(run_dir, RUN_CONFIG)
    assert seen["domain"] == "social_science"


def test_an_unrecognised_domain_falls_back_to_clinical(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The phase does the same; an unknown label must not reach the research config."""
    _research_inputs(run_dir)
    (run_dir / "research/domain_classification.json").write_text('{"domain": "astrology"}')
    seen = _stub_research(monkeypatch)
    research_stages.research(run_dir, RUN_CONFIG)
    assert seen["domain"] == "clinical"


def test_no_affirmative_candidates_is_a_terminal_verdict(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """InsufficientEvidenceError propagates unchanged — it is a real finding about the topic, with a
    report written for the human who has to rephrase it."""
    from dr2_podcast.pipeline import InsufficientEvidenceError

    _research_inputs(run_dir)
    _stub_research(monkeypatch, aff=0, neg=7)
    written: dict[str, Any] = {}
    monkeypatch.setattr(
        "dr2_podcast.pipeline._write_insufficient_evidence_report",
        lambda topic, a, n, d: written.update(topic=topic, aff=a, neg=n),
    )
    with pytest.raises(InsufficientEvidenceError, match="0 candidates"):
        research_stages.research(run_dir, RUN_CONFIG)
    assert written["neg"] == 7
    assert not (run_dir / "research/source_of_truth.md").exists()


def test_limited_evidence_is_declared_at_the_top_of_the_document(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A thin evidence base is stated where anyone reading the SOT meets it first, not buried."""
    _research_inputs(run_dir)
    _stub_research(monkeypatch, aff=2)
    research_stages.research(run_dir, RUN_CONFIG)
    sot = (run_dir / "research/source_of_truth.md").read_text()
    assert sot.startswith("## Evidence Quality Notice")
    assert sot.index("# Source of Truth") > 0


def test_a_healthy_evidence_base_gets_no_notice(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _research_inputs(run_dir)
    _stub_research(monkeypatch, aff=50)
    research_stages.research(run_dir, RUN_CONFIG)
    assert (run_dir / "research/source_of_truth.md").read_text().startswith("# Source of Truth")


# prepush codex 2026-08-13 read this as a P1 — the adapter passes the RUN ROOT to run_deep_research
# while _read_candidate_counts appears to look under research/, which would report zero candidates
# after a successful search and raise InsufficientEvidenceError on every staged run. It is a false
# positive: both sides apply the same "use research/ when it exists" rule, the producer inline
# (clinical.py:3810) and the reader through pipeline.output_path. That agreement is load-bearing and
# was nowhere pinned, so it is pinned here — if either side stops applying the rule, this fails.
def test_the_screening_files_are_written_where_the_candidate_count_looks_for_them(run_dir: Path) -> None:
    import json as _json

    from dr2_podcast import pipeline as _pipeline
    from dr2_podcast.pipeline_flow import _read_candidate_counts

    # Exactly what run_deep_research does with the output_dir the adapter hands it.
    out = Path(str(run_dir))
    research_dir = out / "research"
    written = (research_dir if research_dir.is_dir() else out) / "screening_results_aff.json"
    written.write_text(_json.dumps({"total_candidates": 7}))

    assert Path(_pipeline.output_path(run_dir, "screening_results_aff.json")) == written
    assert _read_candidate_counts(run_dir, logging.getLogger(__name__))[0] == 7


def test_research_fails_closed_on_an_empty_source_of_truth(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The phase catches everything and logs 'continuing without deep research', so a run whose
    research never happened goes on to write an episode from nothing."""
    _research_inputs(run_dir)
    _stub_research(monkeypatch, sot="   ")
    with pytest.raises(ArtifactError, match="nothing to write an episode from"):
        research_stages.research(run_dir, RUN_CONFIG)
    assert not (run_dir / "research/source_of_truth.md").exists()


def test_research_lets_a_pipeline_failure_out(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _research_inputs(run_dir)

    async def _explode(**kwargs: Any) -> Any:
        raise RuntimeError("PubMed unreachable")

    monkeypatch.setattr("dr2_podcast.research.clinical.run_deep_research", _explode)
    with pytest.raises(RuntimeError, match="PubMed unreachable"):
        research_stages.research(run_dir, RUN_CONFIG)


# prepush codex 2026-08-13: run_deep_research writes incrementally and _save_research_reports skips
# a report it does not have, so a rerun producing fewer artifacts left the previous run's files in
# place — and Manifest.complete() saw every declared path and recorded a MIXED set as one run.
def test_a_rerun_that_leaves_a_previous_artifact_behind_is_refused(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dr2_podcast.stages import get_stage

    _research_inputs(run_dir)
    stale = run_dir / "research/grade_synthesis.md"
    stale.write_text("# GRADE from a previous, different run\n")

    # Everything except the stale file gets rewritten by this run.
    def _write_most(reports: Any, directory: Path, log: Any) -> None:
        for artifact in get_stage("research").produces:
            if artifact in ("research/grade_synthesis.md", "research/source_of_truth.md"):
                continue
            path = directory / artifact
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("written by this run")

    _stub_research(monkeypatch)
    monkeypatch.setattr("dr2_podcast.pipeline_flow._save_research_reports", _write_most)

    with pytest.raises(ArtifactError, match="previous execution"):
        research_stages.research(run_dir, RUN_CONFIG)
    assert stale.read_text().startswith("# GRADE from a previous")


def test_a_rerun_that_rewrites_everything_is_accepted(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dr2_podcast.stages import get_stage

    _research_inputs(run_dir)
    for artifact in get_stage("research").produces:
        path = run_dir / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("from a previous run")

    def _write_all(reports: Any, directory: Path, log: Any) -> None:
        for artifact in get_stage("research").produces:
            if artifact == "research/source_of_truth.md":
                continue
            (directory / artifact).write_text("written by this run")

    _stub_research(monkeypatch)
    monkeypatch.setattr("dr2_podcast.pipeline_flow._save_research_reports", _write_all)
    research_stages.research(run_dir, RUN_CONFIG)
    assert (run_dir / "research/grade_synthesis.md").read_text() == "written by this run"


def test_research_fails_closed_without_a_framing_document(run_dir: Path) -> None:
    (run_dir / "research/domain_classification.json").write_text('{"domain": "clinical"}')
    with pytest.raises(ArtifactError, match="cannot read"):
        research_stages.research(run_dir, RUN_CONFIG)


# --------------------------------------------------------------------------- #
# blueprint
# --------------------------------------------------------------------------- #
BLUEPRINT_TEXT = """# Episode Blueprint

## 5. Discussion Points
### Act 1
- Q: What is the claim?
  A: That vitamin D prevents fractures.
"""


def _stub_blueprint(monkeypatch: pytest.MonkeyPatch, produced: str) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def _fake_kickoff(factory, task, translation_task, language, sot, budget):
        seen["sot"] = sot
        seen["task"] = task
        task.output = _FakeOutput(produced)

    monkeypatch.setattr("dr2_podcast.pipeline_crew._crew_kickoff_guarded", _fake_kickoff)
    monkeypatch.setattr("dr2_podcast.pipeline.summarize_report", lambda text, role, topic: f"summary of {role}")
    return seen


def _blueprint_inputs(run_dir: Path) -> None:
    (run_dir / "research/source_of_truth.md").write_text("# Source of Truth\n\nBody.\n")
    (run_dir / "research/domain_classification.json").write_text('{"domain": "clinical"}')
    (run_dir / "research/research_sources_validated.json").write_text("{}")


def test_blueprint_writes_the_document_and_the_inventory(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The inventory is the process boundary: phases 5 and 6 take it as an argument, so a staged
    run needs it on disk."""
    _blueprint_inputs(run_dir)
    _stub_blueprint(monkeypatch, BLUEPRINT_TEXT)
    adapters.blueprint(run_dir, RUN_CONFIG)

    assert (run_dir / "research/EPISODE_BLUEPRINT.md").read_text().startswith("# Episode Blueprint")
    inventory = json.loads((run_dir / "meta/blueprint_inventory.json").read_text())
    assert inventory, "section 5 parsed into something downstream can use"


def test_blueprint_strips_think_blocks(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _blueprint_inputs(run_dir)
    _stub_blueprint(monkeypatch, "<think>internal reasoning</think>\n" + BLUEPRINT_TEXT)
    adapters.blueprint(run_dir, RUN_CONFIG)
    assert "internal reasoning" not in (run_dir / "research/EPISODE_BLUEPRINT.md").read_text()


def test_blueprint_passes_the_translated_sot_when_there_is_one(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _blueprint_inputs(run_dir)
    (run_dir / "research/source_of_truth_ja.md").write_text("# 真実の源\n\n本文。\n")
    seen = _stub_blueprint(monkeypatch, BLUEPRINT_TEXT)
    adapters.blueprint(run_dir, RUN_CONFIG)
    assert seen["sot"].translated_sot_file is not None
    assert seen["sot"].translated_sot_summary


def test_blueprint_tolerates_no_translation(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An English episode has no translated SOT; that is not a failure."""
    _blueprint_inputs(run_dir)
    seen = _stub_blueprint(monkeypatch, BLUEPRINT_TEXT)
    adapters.blueprint(run_dir, RUN_CONFIG)
    assert seen["sot"].translated_sot_file is None
    assert seen["sot"].translated_sot_summary == ""


def test_blueprint_fails_closed_on_an_empty_crew_output(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _blueprint_inputs(run_dir)
    _stub_blueprint(monkeypatch, "<think>only reasoning</think>")
    with pytest.raises(ArtifactError, match="returned nothing"):
        adapters.blueprint(run_dir, RUN_CONFIG)
    assert not (run_dir / "research/EPISODE_BLUEPRINT.md").exists()


def test_blueprint_fails_closed_without_a_source_of_truth(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (run_dir / "research/domain_classification.json").write_text('{"domain": "clinical"}')
    _stub_blueprint(monkeypatch, BLUEPRINT_TEXT)
    with pytest.raises(ArtifactError, match="cannot read"):
        adapters.blueprint(run_dir, RUN_CONFIG)
