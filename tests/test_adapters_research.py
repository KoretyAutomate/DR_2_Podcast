"""The research stage: phase 1, the source of truth it builds, and its GRADE record.

Split out of test_adapters.py to stay under the repo's file-size ceiling. An adapter's job is to
reconstruct, from the run directory alone, the state the monolithic runner built in memory; what is
tested here is that reconstruction and the fail-closed behaviour, with the LLM calls stubbed —
a test that needs vLLM up is a test that does not run.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import logging

import pytest

from dr2_podcast.adapters import research_stages
from dr2_podcast.artifacts import ArtifactError
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
# research
# --------------------------------------------------------------------------- #
def _research_inputs(run_dir: Path, domain: str = "clinical") -> None:
    from tests._stage_fixtures import plan_and_approve

    (run_dir / "research/research_framing.md").write_text("# Research Framework\n\nQuestions.\n")
    (run_dir / "research/domain_classification.json").write_text(f'{{"domain": "{domain}"}}')
    # Step 10: the stage will not search without approved strategies, so a fixture that writes only
    # a framing no longer describes a runnable run.
    plan_and_approve(run_dir)


#: A validated structured GRADE record, as step 7 now produces. Modifier-free on purpose: the
#: adapter cares that there IS an assessment, and grade_errors is exercised where it is built.
GRADE_RECORD = {"schema_version": 1, "level": "moderate", "downgrades": [], "upgrades": []}


def _stub_research(
    monkeypatch: pytest.MonkeyPatch,
    *,
    aff: int = 50,  # comfortably above EVIDENCE_LIMITED_THRESHOLD
    neg: int = 12,
    sot: str = "# Source of Truth\n\n## Abstract\n…",
    reports: Any = None,
) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    async def _fake_deep_research(
        *, topic: str, config: Any, framing_context: str, output_dir: str, plans: Any = None
    ) -> Any:
        seen.update(topic=topic, framing=framing_context, domain=config.domain, output_dir=output_dir,
                    plans=plans)
        if reports is not None:
            return reports
        return {"audit": object(), "pipeline_data": {"grade_record": GRADE_RECORD}}

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


# prepush codex 2026-08-13 [P1]: _grade_record raises when it cannot ground the assessment, but the
# synthesis CALL has a degraded mode — a timeout returns "GRADE synthesis failed. Raw inputs below."
# and never reaches the record pass. The stage completed anyway, because the artifact is optional in
# the graph (social science has no GRADE modifiers) and absence cannot carry a clinical requirement.
def test_a_clinical_run_without_a_grade_record_is_a_failed_stage(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _research_inputs(run_dir)
    _stub_research(monkeypatch, reports={"audit": object(), "pipeline_data": {"grade_record": None}})
    with pytest.raises(ArtifactError, match="no structured GRADE record"):
        research_stages.research(run_dir, RUN_CONFIG)


def test_a_clinical_run_writes_its_grade_record_beside_the_prose(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json as _json

    _research_inputs(run_dir)
    _stub_research(monkeypatch)
    research_stages.research(run_dir, RUN_CONFIG)

    written = _json.loads((run_dir / "research/grade_synthesis.json").read_text())
    assert written["level"] == "moderate"


def test_a_social_science_run_needs_no_record(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """It has an evidence-quality ladder rather than GRADE's modifier arithmetic, so requiring the
    artifact would fail every social-science run on a file that correctly does not exist."""
    _research_inputs(run_dir, domain="social_science")
    _stub_research(monkeypatch, reports={"audit": object(), "pipeline_data": {"grade_record": None}})
    research_stages.research(run_dir, RUN_CONFIG)
    assert not (run_dir / "research/grade_synthesis.json").exists()


# PLAN.md Step 9b: the pack is a projection of the SOT the stage just wrote, so it is generated
# here — where both the SOT and the pipeline_data that produced it are in hand.
def test_the_research_stage_writes_the_step_pack(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json

    from dr2_podcast.research.clinical import Finding, FundingBlock
    from dr2_podcast.schemas import validate_step_pack

    sot = (
        "# Source of Truth\n\n## 1. Abstract\na\n\n## 2. Methods\nm\n\n"
        "### 3.3 Clinical Impact\ni\n\n### 4.1 Study Characteristics\nc\n\n## 5. Discussion\nd\n"
    )
    extraction = SimpleNamespace(
        pmid="1", title="t", study_design="parallel RCT", author_group="Tanaka H; Osaka",
        trial_registration="NCT01", risk_of_bias="low",
        funding=FundingBlock(funding_raw="NIA", funding_category="government",
                             funding_disclosure="disclosed", funding_source_type="api_metadata"),
        findings=[Finding(population="p", intervention="i", comparator="c", endpoint="hip fracture",
                          direction="decrease", finding_key="k" * 40)],
    )
    _research_inputs(run_dir)
    _stub_research(
        monkeypatch,
        sot=sot,
        reports={
            "audit": object(),
            "pipeline_data": {
                "grade_record": GRADE_RECORD,
                "aff_extractions": [extraction],
                "fal_extractions": [],
                "metrics": {"aff_wide_net_total": 30, "aff_screened_in": 5},
                "impacts": [],
            },
        },
    )
    research_stages.research(run_dir, RUN_CONFIG)

    pack = _json.loads((run_dir / "research/step_pack.json").read_text())
    validate_step_pack(pack, {"research/source_of_truth.md": sot})
    assert pack["steps"]["2"]["answer"]["records_identified"] == 30
    assert pack["steps"]["3"]["answer"]["rct"] == 1


def test_a_run_with_no_extractions_writes_no_pack(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """There is nothing to project. An empty pack would be a document asserting that nine questions
    were asked of no evidence."""
    _research_inputs(run_dir)
    _stub_research(
        monkeypatch,
        reports={"audit": object(), "pipeline_data": {"grade_record": GRADE_RECORD,
                                                      "aff_extractions": [], "fal_extractions": []}},
    )
    research_stages.research(run_dir, RUN_CONFIG)
    assert not (run_dir / "research/step_pack.json").exists()


def test_a_stale_pack_does_not_survive_a_run_that_produces_none(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = run_dir / "research/step_pack.json"
    stale.write_text('{"schema_version": 1, "sot_domain": "clinical", "steps": {}}')
    _research_inputs(run_dir)
    _stub_research(
        monkeypatch,
        reports={"audit": object(), "pipeline_data": {"grade_record": GRADE_RECORD,
                                                      "aff_extractions": [], "fal_extractions": []}},
    )
    research_stages.research(run_dir, RUN_CONFIG)
    assert not stale.exists(), "it describes a different run's evidence"


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
# Step 10 — the boundary between planning and searching
# --------------------------------------------------------------------------- #
# PLAN.md's exit criterion is specific: assert the CALL COUNT, not just the verdict. A gate that
# refuses after the search has already run has cost the four minutes and the PubMed rate limit it
# exists to protect.
def _searching_is_forbidden(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    async def _must_not_search(**kwargs):
        calls.append(kwargs.get("topic", "?"))
        raise AssertionError("a search ran on an unapproved strategy")

    monkeypatch.setattr("dr2_podcast.research.clinical.run_deep_research", _must_not_search)
    return calls


def test_an_unapproved_strategy_issues_no_search_at_all(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _research_inputs(run_dir)
    (run_dir / "meta/strategy_approval.json").unlink()
    calls = _searching_is_forbidden(monkeypatch)

    with pytest.raises(ArtifactError, match="not approved"):
        research_stages.research(run_dir, RUN_CONFIG)
    assert calls == [], "the refusal has to come before the search, not after it"


@pytest.mark.parametrize(
    "artifact",
    ["research/search_strategy_aff.json", "research/research_framing.md"],
)
def test_an_artifact_edited_after_approval_issues_no_search(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, artifact: str
) -> None:
    """The framing case is the one a strategy-only hash would have let through: the strategies are
    approved BY COMPARISON with the framing, so a changed framing invalidates the comparison."""
    _research_inputs(run_dir)
    (run_dir / artifact).write_text("something the approver never read\n")
    calls = _searching_is_forbidden(monkeypatch)

    with pytest.raises(ArtifactError, match="not approved"):
        research_stages.research(run_dir, RUN_CONFIG)
    assert calls == []


def test_the_search_runs_against_the_strategies_that_were_approved(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rebuilt from the FILES, never from anything held in memory: the approval is a statement about
    those bytes, so a search against anything else is covered by no approval at all."""
    _research_inputs(run_dir)
    seen = _stub_research(monkeypatch)
    research_stages.research(run_dir, RUN_CONFIG)

    plans = seen["plans"]
    assert plans["aff_plan"].role == "affirmative"
    assert plans["fal_plan"].role == "adversarial"
    assert plans["aff_plan"].tier1.intervention == ["vitamin D"]


def test_plan_search_stops_before_the_search(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stage exists so a strategy is on disk with nothing searched. Before the split there was
    no such point in the pipeline at all."""
    import dataclasses

    from dr2_podcast.research.clinical import TieredSearchPlan, TierKeywords

    (run_dir / "research/research_framing.md").write_text("# Research Framework\n\nQuestions.\n")
    (run_dir / "research/domain_classification.json").write_text('{"domain": "clinical"}')
    calls = _searching_is_forbidden(monkeypatch)

    def _tier(word):
        return TierKeywords(intervention=[word], outcome=["fracture"], population=["adults"], rationale="r")

    async def _plan(*, topic, config, framing_context, log):
        return {
            "decomposition": {"canonical_terms": ["vitamin D"]},
            "aff_plan": TieredSearchPlan(pico={"population": "adults"}, tier1=_tier("vitamin D"),
                                         tier2=_tier("cholecalciferol"), tier3=_tier("secosteroid"),
                                         role="affirmative"),
            "fal_plan": TieredSearchPlan(pico={"population": "adults"}, tier1=_tier("vitamin D harm"),
                                         tier2=_tier("hypercalcaemia"), tier3=_tier("secosteroid"),
                                         role="adversarial"),
        }

    monkeypatch.setattr("dr2_podcast.research.clinical.plan_search", _plan)
    research_stages.plan_search(run_dir, RUN_CONFIG)

    assert calls == []
    assert not (run_dir / "meta/strategy_approval.json").exists(), "approving is not this stage's job"
    import json as _json

    aff = _json.loads((run_dir / "research/search_strategy_aff.json").read_text())
    assert aff["role"] == "affirmative"
    assert aff["tier1"]["intervention"] == ["vitamin D"]
    assert dataclasses.is_dataclass(TieredSearchPlan)
