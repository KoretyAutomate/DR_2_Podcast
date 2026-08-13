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
    assert {"framing", "url_validation", "blueprint"} <= set(ADAPTERS)


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


def test_blueprint_declares_the_validated_library_as_an_input() -> None:
    """Declared, so producing it makes an existing blueprint stale rather than silently ignored."""
    from dr2_podcast.stages import direct_producers, get_stage

    assert "research/research_sources_validated.json" in get_stage("blueprint").optional_consumes
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
