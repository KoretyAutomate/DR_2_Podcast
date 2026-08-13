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
    assert {"framing", "sot", "url_validation"} <= set(ADAPTERS)


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


def test_initialise_run_globals_is_the_one_owner_of_that_state() -> None:
    """It was extracted from pipeline.py's __main__ so both runners share it. If an adapter
    reimplemented it, the two would drift and produce different episodes from the same inputs."""
    import inspect

    from dr2_podcast import pipeline

    assert "initialise_run_globals" in inspect.getsource(adapters._prepare_run)
    assert callable(pipeline.initialise_run_globals)


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
# sot
# --------------------------------------------------------------------------- #
def test_sot_does_not_need_the_llm_backend(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """prepush codex 2026-08-12: sot is declared engine="python" but went through _prepare_run,
    which probes the backend ten times before failing. A deterministic stage that cannot run while
    vLLM is down is not deterministic in any useful sense."""

    def _explode() -> str:
        raise AssertionError("the SOT stage must not touch the LLM backend")

    monkeypatch.setattr("dr2_podcast.pipeline.get_final_model_string", _explode)
    (run_dir / "research/domain_classification.json").write_text('{"domain": "clinical"}')
    (run_dir / "meta/deep_reports.json").write_text('{"affirmative": {"report": "x"}}')
    monkeypatch.setattr("dr2_podcast.pipeline_sot.build_imrad_sot", lambda **kwargs: "# SOT\n")
    adapters.sot(run_dir, RUN_CONFIG)
    assert (run_dir / "research/source_of_truth.md").exists()


def test_sot_names_the_artifact_the_monolithic_flow_never_had_to_persist(run_dir: Path) -> None:
    """build_imrad_sot consumes a live dict in the flow; across a process boundary it has to be a
    file, and saying which one beats failing on a KeyError deep inside the builder."""
    (run_dir / "research/domain_classification.json").write_text('{"domain": "clinical"}')
    with pytest.raises(ArtifactError, match="deep_reports.json"):
        adapters.sot(run_dir, RUN_CONFIG)


def test_sot_builds_the_document_from_disk(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (run_dir / "research/domain_classification.json").write_text('{"domain": "clinical"}')
    (run_dir / "meta/deep_reports.json").write_text('{"affirmative": {"report": "x"}}')
    seen: dict[str, Any] = {}

    def _fake_build(**kwargs: Any) -> str:
        seen.update(kwargs)
        return "# Source of Truth\n\n## Abstract\n…"

    monkeypatch.setattr("dr2_podcast.pipeline_sot.build_imrad_sot", _fake_build)
    adapters.sot(run_dir, RUN_CONFIG)

    assert seen["topic"] == "ビタミンDと骨折"
    assert seen["domain"] == "clinical"
    assert (run_dir / "research/source_of_truth.md").read_text().startswith("# Source of Truth")


def test_sot_fails_closed_on_an_empty_document(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (run_dir / "research/domain_classification.json").write_text('{"domain": "clinical"}')
    (run_dir / "meta/deep_reports.json").write_text("{}")
    monkeypatch.setattr("dr2_podcast.pipeline_sot.build_imrad_sot", lambda **kwargs: "   ")
    with pytest.raises(ArtifactError, match="empty source of truth"):
        adapters.sot(run_dir, RUN_CONFIG)
