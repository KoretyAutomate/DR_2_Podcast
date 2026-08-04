"""Characterization tests for Orchestrator.run — the 7-step pipeline.

run() is 156 statements, most of it two near-identical inline track closures
that differ only in which researcher they use and how their log lines are
labelled. Written before extracting that duplication.

Every researcher, fetcher and synthesis call is stubbed, so what is pinned is
the orchestration: both tracks running, the per-track metrics, retracted-paper
filtering, the domain switch for Step 6, and the totals handed to Step 7.
"""

import asyncio
from types import SimpleNamespace

import pytest

from dr2_podcast.research import clinical as cl
from dr2_podcast.research.clinical import Orchestrator, TieredSearchPlan, TierKeywords, WideNetRecord


def _plan(role):
    kw = TierKeywords(intervention=["i"], outcome=["o"], population=[], rationale="r")
    return TieredSearchPlan(
        pico={"population": "p", "intervention": "i", "comparison": "c", "outcome": "o"},
        tier1=kw,
        tier2=kw,
        tier3=kw,
        role=role,
    )


def _record(i, prefix, retracted=False):
    meta = cl.PaperMetadata(is_retracted=retracted) if retracted else None
    return WideNetRecord(
        pmid=f"{prefix}{i}",
        doi=f"10.1/{prefix}{i}",
        title=f"{prefix} study {i}",
        abstract="abs",
        study_type="RCT",
        sample_size=None,
        primary_objective=None,
        year=2024,
        journal="J",
        authors="A",
        url=f"https://ex.org/{prefix}{i}",
        source_db="pubmed",
        research_tier=1,
        paper_metadata=meta,
    )


def _extraction(i, prefix):
    return cl.DeepExtraction(pmid=f"{prefix}{i}", doi=f"10.1/{prefix}{i}", title=f"{prefix} study {i}", url="")


class FakeResearcher:
    """Stands in for lead_researcher / counter_researcher."""

    def __init__(self, prefix, n_records=4, n_top=2, retracted_idx=None):
        self.prefix = prefix
        self.n_records = n_records
        self.n_top = n_top
        self.retracted_idx = set(retracted_idx or ())
        self.calls = []

    async def _decompose_topic(self, topic, framing_context):
        self.calls.append("decompose")
        return {"canonical_terms": ["term"], "related_concepts": ["concept"]}

    async def _formulate_tiered_strategy(self, topic, role, framing_context, decomposition, log=None):
        self.calls.append(("strategy", role))
        return _plan(role)

    async def _tiered_search(self, plan, log):
        self.calls.append("search")
        return [_record(i, self.prefix) for i in range(self.n_records)], 2

    async def _screen_and_prioritize(self, records, plan, topic="", log=None):
        self.calls.append("screen")
        return [_record(i, self.prefix, retracted=(i in self.retracted_idx)) for i in range(self.n_top)]

    async def _deep_extract_batch(self, fulltexts, records, pico, log, output_dir=None):
        self.calls.append("extract")
        return [_extraction(i, self.prefix) for i in range(len(records))]

    async def _build_case(self, topic, plan, extractions, role, log):
        """_build_case returns the case TEXT, not a report object."""
        self.calls.append(("case", role))
        return f"{self.prefix} case"


@pytest.fixture
def orch(monkeypatch):
    o = Orchestrator.__new__(Orchestrator)
    o.domain = "clinical"
    o.fast_model_available = False

    async def _aclose():
        return None

    o.openalex = SimpleNamespace(close=_aclose)
    o.eric = SimpleNamespace(close=_aclose)
    o._metadata_cache = SimpleNamespace(close=lambda: None)
    o.lead_researcher = FakeResearcher("aff")
    o.counter_researcher = FakeResearcher("fal")

    class FakeFetcher:
        async def fetch_all(self, records):
            # one failure per track, so the ok/err split is observable
            return [SimpleNamespace(error=None) for _ in records[:-1]] + [
                SimpleNamespace(error="timeout") for _ in records[-1:]
            ]

    o.fulltext_fetcher = FakeFetcher()

    async def no_enrich(records, log):
        return records

    o._enrich_with_metadata = no_enrich

    captured = {}

    async def fake_grade(*args, **kwargs):
        captured["grade_args"] = args
        captured["grade_kwargs"] = kwargs
        return "GRADE TEXT"

    o._grade_synthesis = fake_grade
    o._save_artifacts = lambda *a, **k: captured.setdefault("saved", (a, k))
    o._extractions_to_sources = lambda extractions, role: [
        SimpleNamespace(role=role, url="", summary="s", error=None, title="t", query="", goal="", metadata=None)
        for _ in extractions
    ]
    # Step 6 math is exercised by its own tests; keep it inert elsewhere.
    from dr2_podcast.research import clinical_math

    monkeypatch.setattr(clinical_math, "batch_calculate", lambda ex: [])
    monkeypatch.setattr(clinical_math, "format_math_report", lambda imps: "MATH")

    o.captured = captured
    return o


def _run(o, **kw):
    return asyncio.run(o.run("Topic", **kw))


class TestOrchestratorRun:
    def test_both_tracks_run_every_step(self, orch):
        _run(orch)
        for researcher in (orch.lead_researcher, orch.counter_researcher):
            names = [c if isinstance(c, str) else c[0] for c in researcher.calls]
            assert "search" in names
            assert "screen" in names
            assert "extract" in names
            assert "case" in names

    def test_the_two_tracks_use_their_own_roles(self, orch):
        _run(orch)
        assert ("strategy", "affirmative") in orch.lead_researcher.calls
        assert ("case", "affirmative") in orch.lead_researcher.calls
        assert ("strategy", "adversarial") in orch.counter_researcher.calls
        assert ("case", "falsification") in orch.counter_researcher.calls

    def test_totals_handed_to_grade_are_the_sum_of_both_tracks(self, orch):
        _run(orch)
        # positional: topic, aff_track, fal_track, math_report, search_date, log
        _, aff, fal, _, _, _ = orch.captured["grade_args"]
        assert aff.plan.role == "affirmative", "the tracks must not be passed in the wrong order"
        assert fal.plan.role == "adversarial"
        assert aff.wide_net_total + fal.wide_net_total == 8, "4 wide-net records per track"
        assert aff.screened_in + fal.screened_in == 4, "2 screened per track"
        assert aff.fulltext_ok + fal.fulltext_ok == 2, "one full-text failure per track"
        assert aff.fulltext_err + fal.fulltext_err == 2

    def test_retracted_papers_are_filtered_before_extraction(self, orch):
        orch.lead_researcher = FakeResearcher("aff", retracted_idx={0})
        result = _run(orch)
        pd = result["pipeline_data"]
        assert len(pd["aff_extractions"]) == 1, "the retracted paper must not reach extraction"
        assert len(pd["fal_extractions"]) == 2, "the other track is unaffected"
        # screened_in is counted BEFORE the retraction filter, so it is unchanged
        _, aff, fal, _, _, _ = orch.captured["grade_args"]
        assert aff.screened_in + fal.screened_in == 4

    def test_clinical_domain_uses_arr_nnt_math(self, orch, monkeypatch):
        from dr2_podcast.research import clinical_math

        called = {}

        def spy(ex):
            called["clinical"] = ex
            return []

        monkeypatch.setattr(clinical_math, "batch_calculate", spy)
        monkeypatch.setattr(clinical_math, "format_math_report", lambda imps: "ARR REPORT")
        _run(orch)
        assert "clinical" in called
        assert orch.captured["grade_args"][3] == "ARR REPORT"

    def test_social_domain_uses_effect_size_math(self, orch, monkeypatch):
        orch.domain = "social_science"
        import dr2_podcast.research.effect_size_math as esm

        monkeypatch.setattr(esm, "batch_calculate", lambda ex: [])
        monkeypatch.setattr(esm, "format_effect_size_report", lambda imps: "EFFECT REPORT")
        _run(orch)
        assert orch.captured["grade_args"][3] == "EFFECT REPORT"

    def test_artifacts_are_saved_only_when_an_output_dir_is_given(self, orch, tmp_path):
        _run(orch)
        assert "saved" not in orch.captured
        _run(orch, output_dir=str(tmp_path))
        assert "saved" in orch.captured
        args, _kwargs = orch.captured["saved"]
        _out, aff_track, fal_track, _math = args
        assert aff_track.plan.role == "affirmative", "affirmative track must be the first argument"
        assert fal_track.plan.role == "adversarial"

    def test_result_carries_pipeline_data_for_both_tracks(self, orch):
        result = _run(orch)
        pd = result["pipeline_data"]
        assert pd["aff_strategy"].role == "affirmative"
        assert pd["fal_strategy"].role == "adversarial"
        assert len(pd["aff_extractions"]) == 2
        assert len(pd["fal_extractions"]) == 2
        assert pd["metrics"]["aff_wide_net_total"] == 4
        assert pd["metrics"]["fal_wide_net_total"] == 4

    def test_reports_are_returned_for_lead_counter_and_audit(self, orch):
        result = _run(orch)
        assert result["lead"].report == "aff case"
        assert result["counter"].report == "fal case"
        assert result["audit"].report == "GRADE TEXT"

    def test_decomposition_runs_once_before_the_tracks(self, orch):
        _run(orch)
        assert orch.lead_researcher.calls[0] == "decompose"
        assert "decompose" not in orch.counter_researcher.calls
