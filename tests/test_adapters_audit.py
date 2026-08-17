"""The audit stage: the accuracy audit, its deterministic gates, the correction pass, finalize.

Split from test_adapters.py to stay under the repo's file-size ceiling; see that file for what a
mutation matrix over adapters is testing.

Originally:

An adapter's job is to reconstruct, from the run directory alone, the state the monolithic runner
built in memory. What is tested here is that reconstruction and the fail-closed behaviour; the LLM
calls themselves are stubbed, because a test that needs vLLM up is a test that does not run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dr2_podcast import adapters
from dr2_podcast.artifacts import ArtifactError
from dr2_podcast.artifacts import write_atomic as _real_write_atomic
from dr2_podcast.stage import write_run_config


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
# audit
# --------------------------------------------------------------------------- #
class _FakeOutput:
    """What CrewAI leaves on a task once it has run."""

    def __init__(self, raw: str) -> None:
        self.raw = raw


def _audit_inputs(run_dir: Path) -> None:
    (run_dir / "scripts/script_polished.md").write_text("Host 1: the polished script\n")
    (run_dir / "research/source_of_truth.md").write_text("# Source of Truth\n\nBody.\n")


# prepush codex 2026-08-13 named `blueprint` for this; it is `audit`. The library tools belong to
# auditor_agent (pipeline_crew.py:328) and producer_agent carries none at all (:356), so the
# blueprint never opens the library — but the auditor does, through research_sources_file(), which
# consults all three files to decide which library to serve.
def test_the_stage_that_reads_the_library_declares_all_three_files() -> None:
    from dr2_podcast.stages import get_stage

    consumed = set(get_stage("audit").consumes)
    assert {
        "research/research_sources.json",
        "research/research_sources_validated.json",
        "research/research_sources_validated.sha256",
    } <= consumed


def test_the_blueprint_declares_the_validated_library_as_a_gate_only() -> None:
    """producer_agent has no tools, so this is an ordering gate rather than a read: an episode must
    not be designed before its citations were checked at all."""
    from dr2_podcast.stages import get_stage

    assert "research/research_sources_validated.json" in get_stage("blueprint").consumes
    assert "research/research_sources.json" not in get_stage("blueprint").consumes


def _stub_audit(
    monkeypatch: pytest.MonkeyPatch,
    *,
    verdict: str = "PASS — no drift found",
    citation_issues: list | None = None,
    grade_issues: list | None = None,
    corrected: str | None = "Host 1: the corrected script\n",
) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    class _FakeCrew:
        def __init__(self, agents: list, tasks: list, **kwargs: Any) -> None:
            seen["task"] = tasks[0]

        def kickoff(self) -> None:
            seen["task"].output = _FakeOutput(verdict)

    monkeypatch.setattr("crewai.Crew", _FakeCrew)
    # Keyed on the text: the polished script has the issues, the corrected one does not. A stub
    # that returned the same issues for every input would make every correction fail the re-check.
    monkeypatch.setattr(
        "dr2_podcast.pipeline_flow._deterministic_gate_issues",
        lambda polished, sot, log: ([], []) if polished.startswith("Host 1: the corrected")
        else (citation_issues or [], grade_issues or []),
    )

    def _correct(**kwargs: Any) -> str | None:
        seen["correction_ran"] = True
        return corrected

    monkeypatch.setattr("dr2_podcast.pipeline_flow._run_inline_correction", _correct)
    monkeypatch.setattr("dr2_podcast.pipeline_flow._write_accuracy_corrections_md", lambda *a: None)

    def _finalize(polished, task, language, config, output_dir, corrected_text=None):
        seen["finalized_from"] = corrected_text or polished
        seen["finalized_in"] = output_dir
        (Path(output_dir) / "scripts/script_final.md").write_text(seen["finalized_from"])
        return seen["finalized_from"]

    monkeypatch.setattr("dr2_podcast.pipeline._finalize_script", _finalize)
    return seen


def test_audit_writes_the_report_and_finalises_the_script(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _audit_inputs(run_dir)
    seen = _stub_audit(monkeypatch)
    adapters.audit(run_dir, RUN_CONFIG)

    assert (run_dir / "research/accuracy_audit.md").read_text() == "PASS — no drift found"
    assert (run_dir / "scripts/script_final.md").exists()
    assert seen.get("correction_ran") is None, "a clean audit does not trigger the correction pass"
    assert seen["finalized_from"].startswith("Host 1: the polished script")


def test_audit_puts_the_polished_script_in_the_task_context(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _audit_inputs(run_dir)
    seen = _stub_audit(monkeypatch)
    adapters.audit(run_dir, RUN_CONFIG)
    context = seen["task"].context
    assert context, "the auditor reads the script through the task context"
    assert context[0].output.raw.startswith("Host 1: the polished script")


def test_a_deterministic_gate_alone_triggers_the_correction(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The citation and GRADE gates are independent triggers — the LLM verdict need not say FAIL."""
    _audit_inputs(run_dir)
    seen = _stub_audit(monkeypatch, verdict="PASS", citation_issues=["fabricated PMID 99999999"])
    adapters.audit(run_dir, RUN_CONFIG)
    assert seen["correction_ran"] is True
    assert seen["finalized_from"].startswith("Host 1: the corrected script")


def test_an_unrepairable_script_is_not_rendered(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The flow finalises the UNCORRECTED script and proceeds to audio. A script this pipeline's own
    gate rejected and could not repair needs a human, not a render."""
    _audit_inputs(run_dir)
    _stub_audit(monkeypatch, citation_issues=["fabricated PMID"], corrected=None)
    with pytest.raises(ArtifactError, match="needs a human"):
        adapters.audit(run_dir, RUN_CONFIG)
    assert not (run_dir / "scripts/script_final.md").exists()


# prepush codex 2026-08-13: a corrections report from an EARLIER audit survived a later run whose
# gate did not fire, and the manifest recorded it as this execution's output — a file describing a
# different script and a different verdict.
def test_a_stale_corrections_report_does_not_survive_a_clean_audit(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _audit_inputs(run_dir)
    stale = run_dir / "research/ACCURACY_CORRECTIONS.md"
    stale.write_text("# corrections from a previous audit of a different script\n")
    _stub_audit(monkeypatch)

    adapters.audit(run_dir, RUN_CONFIG)
    assert not stale.exists()


# prepush codex 2026-08-13. Finalisation was handed a staging directory so its bare open() could
# not truncate the previous script — but it reads research/source_of_truth*.md and
# research/grade_synthesis.md from that same directory to run validate_grade_consistency, and a
# staging tree has neither. The check found no inputs and skipped itself on every staged run, in
# silence. The write is atomic at its source now, so finalisation sees the real run directory.
def test_finalisation_can_reach_the_research_inputs_its_grade_check_needs(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _audit_inputs(run_dir)
    (run_dir / "research/grade_synthesis.md").write_text("### Overall Certainty\nLOW\n")
    seen = _stub_audit(monkeypatch)
    adapters.audit(run_dir, RUN_CONFIG)

    finalize_dir = Path(seen["finalized_in"])
    assert (finalize_dir / "research/grade_synthesis.md").exists(), (
        "finalisation cannot run validate_grade_consistency from a directory without the research inputs"
    )
    assert (finalize_dir / "research/source_of_truth.md").exists()


def test_the_final_script_is_written_atomically(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real _finalize_script, not a stub: a failure partway through must leave the previously
    accepted script intact rather than a truncated one that looks finished."""
    from dr2_podcast import pipeline as real_pipeline

    previous = run_dir / "scripts/script_final.md"
    previous.write_text("Host 1: the previously accepted final script\n")

    def _dies_on_validate(payload: bytes) -> None:
        raise RuntimeError("finalisation died after the candidate was written")

    monkeypatch.setattr(real_pipeline, "_add_reaction_guidance", lambda text, cfg: text)
    monkeypatch.setattr(
        "dr2_podcast.artifacts.write_atomic",
        lambda path, data, **kw: _real_write_atomic(path, data, validate=_dies_on_validate),
    )
    with pytest.raises(RuntimeError, match="finalisation died"):
        real_pipeline._finalize_script(
            "Host 1: a new script\n", None, "en", {}, run_dir, corrected_text=None
        )
    assert previous.read_text().startswith("Host 1: the previously accepted")
    assert not list((run_dir / "scripts").glob("*.candidate")), "the candidate must not be left behind"


def test_a_leftover_final_script_is_not_recorded_as_this_runs_output(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staging used to prove authorship by construction. Without it, the mtime snapshot must."""
    _audit_inputs(run_dir)
    (run_dir / "scripts/script_final.md").write_text("Host 1: a previous execution's final script\n")
    _stub_audit(monkeypatch)

    def _writes_nothing(polished, task, language, config, output_dir, corrected_text=None):
        return polished

    monkeypatch.setattr("dr2_podcast.pipeline._finalize_script", _writes_nothing)
    with pytest.raises(ArtifactError, match="previous execution"):
        adapters.audit(run_dir, RUN_CONFIG)


def test_audit_fails_closed_on_an_empty_verdict(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _audit_inputs(run_dir)
    _stub_audit(monkeypatch, verdict="   ")
    with pytest.raises(ArtifactError, match="unaudited script"):
        adapters.audit(run_dir, RUN_CONFIG)
    assert not (run_dir / "research/accuracy_audit.md").exists()


def test_audit_fails_closed_without_a_polished_script(run_dir: Path) -> None:
    (run_dir / "research/source_of_truth.md").write_text("# Source of Truth\n")
    with pytest.raises(ArtifactError, match="cannot read"):
        adapters.audit(run_dir, RUN_CONFIG)


# prepush codex 2026-08-13: the deterministic gates ran against the POLISHED script only, so a
# correction that introduced a new unsupported citation or GRADE contradiction was finalised and
# rendered having never passed anything. PLAN.md Step 5 requires every correction to re-enter them.
def test_a_correction_that_still_fails_the_gates_is_refused(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _audit_inputs(run_dir)
    seen = _stub_audit(monkeypatch, citation_issues=["fabricated PMID 99999999"])

    def _still_broken(polished, sot, log):
        # The polished script has one issue; the corrected one still does.
        return (["fabricated PMID 99999999"], [])

    monkeypatch.setattr("dr2_podcast.pipeline_flow._deterministic_gate_issues", _still_broken)
    with pytest.raises(ArtifactError, match="still fails the deterministic gates"):
        adapters.audit(run_dir, RUN_CONFIG)
    assert seen["correction_ran"] is True, "it ran, and then its output was checked"


def test_a_correction_that_fixes_the_problem_is_accepted(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control: re-checking must not reject a correction that worked."""
    _audit_inputs(run_dir)
    seen = _stub_audit(monkeypatch, citation_issues=["fabricated PMID 99999999"])

    def _fixed_by_the_correction(polished, sot, log):
        return ([], []) if polished.startswith("Host 1: the corrected") else (["fabricated PMID"], [])

    monkeypatch.setattr("dr2_podcast.pipeline_flow._deterministic_gate_issues", _fixed_by_the_correction)
    adapters.audit(run_dir, RUN_CONFIG)
    assert seen["finalized_from"].startswith("Host 1: the corrected")
