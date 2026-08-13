"""Stage adapters — PLAN.md Step 1's second half.

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
# Registration
# --------------------------------------------------------------------------- #
def test_the_adapters_register_themselves_against_declared_stages() -> None:
    assert {"framing", "url_validation", "blueprint", "translate", "audio", "draft", "polish"} <= set(
        ADAPTERS
    )


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
    pipeline = adapters._prepare_run(run_dir, RUN_CONFIG)

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
    pipeline = adapters._prepare_run(run_dir, {**RUN_CONFIG, "target_length_minutes": 40})
    expected = 40 * pipeline.language_config["speech_rate"]
    assert pipeline.target_length_int == expected


# prepush codex 2026-08-12: _target_min is read directly by _create_agents_and_tasks and three more
# CrewBuildConfig sites. Left at its sentinel 0 by the extracted initialiser, every staged draft and
# polish prompt would have asked for a 0-minute episode while target_script carried the right count.
def test_the_target_minutes_global_is_set_not_left_at_its_sentinel(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = adapters._prepare_run(run_dir, {**RUN_CONFIG, "target_length_minutes": 33})
    assert pipeline._target_min == 33
    assert pipeline.target_length_int == 33 * pipeline.language_config["speech_rate"]


# prepush codex 2026-08-12: assign_roles() is random under the default PODCAST_HOSTS=random, and
# every stage is a fresh process — so framing, blueprint and the script phases would each build
# prompts with DIFFERENT host roles, with no manifest identity change to show for it.
def test_the_host_roles_are_assigned_once_and_then_reused(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODCAST_HOSTS", "random")
    first = adapters._session_roles(run_dir)
    assert (run_dir / "meta/session_roles.json").exists()

    seen: list[int] = []

    def _reassign() -> dict[str, Any]:
        seen.append(1)
        return {"changed": True}

    monkeypatch.setattr("dr2_podcast.pipeline.assign_roles", _reassign)
    for _ in range(5):
        assert adapters._session_roles(run_dir) == first
    assert seen == [], "a second process must read the roles, never reassign them"


# prepush codex 2026-08-13: a changed PODCAST_HOSTS makes framing stale, but rerunning it read the
# old assignment straight back while the manifest recorded the stage as current under the new one.
def test_framing_reassigns_the_roles_when_it_reruns(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dr2_podcast.pipeline import assign_roles

    old = assign_roles()
    new = {role: {**spec, "personality": "a different persona"} for role, spec in old.items()}
    (run_dir / "meta/session_roles.json").write_text(json.dumps(old))
    monkeypatch.setattr("dr2_podcast.pipeline.assign_roles", lambda: new)
    _stub_framing(monkeypatch, "# framing")
    adapters.framing(run_dir, RUN_CONFIG)
    assert json.loads((run_dir / "meta/session_roles.json").read_text()) == new


def test_a_stage_that_only_reads_the_roles_leaves_them_alone(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dr2_podcast.pipeline import assign_roles

    chosen = assign_roles()
    other = {role: {**spec, "personality": "reassigned"} for role, spec in chosen.items()}
    (run_dir / "meta/session_roles.json").write_text(json.dumps(chosen, ensure_ascii=False))
    monkeypatch.setattr("dr2_podcast.pipeline.assign_roles", lambda: other)
    adapters._prepare_run(run_dir, RUN_CONFIG)
    assert json.loads((run_dir / "meta/session_roles.json").read_text()) == chosen


def test_prepare_run_uses_the_persisted_roles(run_dir: Path) -> None:
    pipeline = adapters._prepare_run(run_dir, RUN_CONFIG)
    assert json.loads((run_dir / "meta/session_roles.json").read_text()) == pipeline.SESSION_ROLES


def test_initialise_run_globals_is_the_one_owner_of_that_state() -> None:
    """It was extracted from pipeline.py's __main__ so both runners share it. If an adapter
    reimplemented it, the two would drift and produce different episodes from the same inputs."""
    import inspect

    from dr2_podcast import pipeline

    assert "initialise_run_globals" in inspect.getsource(adapters._prepare_run)
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
    monkeypatch.setattr(adapters, "_classify_domain", lambda topic: _FakeClassification())

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
    note = adapters._domain_note(classification)
    assert "PECO" in note
    assert "Do NOT use clinical terminology" in note


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
