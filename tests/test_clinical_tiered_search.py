"""Characterization tests for the Step 2 tiered cascade in clinical.py.

_tiered_search (complexity 34, 101 statements) and _tiered_search_social
(complexity 30, 101 statements) are near-duplicates: both run a three-tier
cascade, deduplicate as they go, supplement with Google Scholar, and stop at
TIER_CASCADE_THRESHOLD. Written before extracting that shared machinery.

Every network client is stubbed. The agent is built with __new__ and attribute
injection, matching tests/test_t3_3_retry_logic.py.
"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from dr2_podcast.config import TIER_CASCADE_THRESHOLD
from dr2_podcast.research import clinical as cl
from dr2_podcast.research.clinical import ResearchAgent, TieredSearchPlan, TierKeywords


def _tier(terms, outcome=("effect",), population=()):
    return TierKeywords(intervention=list(terms), outcome=list(outcome), population=list(population), rationale="r")


def _plan(t1=("aspirin",), t2=("acetylsalicylic",), t3=("salicylate",)):
    return TieredSearchPlan(
        pico={"population": "adults", "intervention": "aspirin", "comparison": "placebo", "outcome": "MI"},
        tier1=_tier(t1),
        tier2=_tier(t2),
        tier3=_tier(t3),
        role="affirmative",
    )


def _articles(n, prefix="p", start=0):
    return [
        {
            "pmid": f"{prefix}{i}",
            "doi": f"10.1/{prefix}{i}",
            "title": f"Study {prefix}{i}",
            "abstract": "abs",
            "study_type": "RCT",
            "year": 2020,
            "journal": "J",
            "authors": "A",
            "url": f"https://pubmed.example/{prefix}{i}",
        }
        for i in range(start, start + n)
    ]


@pytest.fixture
def agent(monkeypatch):
    a = ResearchAgent.__new__(ResearchAgent)
    a._domain = "clinical"
    a.fast_worker = None
    a.search = SimpleNamespace(pubmed=SimpleNamespace())
    a._openalex = SimpleNamespace()
    a._eric = SimpleNamespace()

    # No Scholar unless a test opts in.
    _disable_scholar(monkeypatch)
    return a


def _disable_scholar(monkeypatch, results=None, connected=False):
    class FakeSearxng:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def validate_connection(self):
            return connected

        async def search(self, query, engines=None, num_results=0):
            return results or []

    monkeypatch.setattr(cl, "SearxngClient", FakeSearxng)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# _tiered_search — clinical (PubMed)
# ---------------------------------------------------------------------------


class TestTieredSearchClinical:
    def test_tier_one_alone_satisfies_the_threshold_and_stops_the_cascade(self, agent):
        queries = []

        async def search_extended(query, max_results=200):
            queries.append(query)
            return _articles(TIER_CASCADE_THRESHOLD)

        agent.search.pubmed.search_extended = search_extended
        records, highest = _run(agent._tiered_search(_plan(), log=lambda *a: None))

        assert len(queries) == 1, "cascade must stop once the threshold is reached"
        assert highest == 1
        assert len(records) == TIER_CASCADE_THRESHOLD
        assert all(r.research_tier == 1 for r in records)
        assert all(r.source_db == "pubmed" for r in records)

    def test_cascade_continues_through_all_three_tiers_when_under_threshold(self, agent):
        queries = []

        async def search_extended(query, max_results=200):
            queries.append(query)
            return _articles(2, prefix=f"t{len(queries)}_")

        agent.search.pubmed.search_extended = search_extended
        records, highest = _run(agent._tiered_search(_plan(), log=lambda *a: None))

        assert len(queries) == 3
        assert highest == 3
        assert len(records) == 6
        assert {r.research_tier for r in records} == {1, 2, 3}

    def test_records_are_deduplicated_by_pmid_across_tiers(self, agent):
        async def search_extended(query, max_results=200):
            return _articles(3)  # identical every tier

        agent.search.pubmed.search_extended = search_extended
        records, _ = _run(agent._tiered_search(_plan(), log=lambda *a: None))
        assert len(records) == 3
        assert [r.pmid for r in records] == ["p0", "p1", "p2"]

    def test_dedupe_by_pmid_alone_when_urls_differ(self, agent):
        """Isolates the PMID check — the URL check cannot mask it here."""
        calls = {"n": 0}

        async def search_extended(query, max_results=200):
            calls["n"] += 1
            arts = _articles(2)
            for a in arts:  # same pmids, different URLs each tier
                a["url"] = f"https://mirror{calls['n']}.example/{a['pmid']}"
            return arts

        agent.search.pubmed.search_extended = search_extended
        records, _ = _run(agent._tiered_search(_plan(), log=lambda *a: None))
        assert [r.pmid for r in records] == ["p0", "p1"]

    def test_dedupe_by_url_alone_when_pmids_are_absent(self, agent):
        """Isolates the URL check — records with no PMID still deduplicate."""

        async def search_extended(query, max_results=200):
            arts = _articles(2)
            for a in arts:
                a["pmid"] = ""
            return arts

        agent.search.pubmed.search_extended = search_extended
        records, _ = _run(agent._tiered_search(_plan(), log=lambda *a: None))
        assert len(records) == 2

    def test_a_tier_without_intervention_keywords_is_skipped(self, agent):
        queries = []

        async def search_extended(query, max_results=200):
            queries.append(query)
            return _articles(1, prefix=f"q{len(queries)}_")

        agent.search.pubmed.search_extended = search_extended
        plan = _plan()
        plan.tier2 = _tier(())  # no intervention terms
        records, highest = _run(agent._tiered_search(plan, log=lambda *a: None))

        assert len(queries) == 2, "tier 2 contributes no query"
        assert {r.research_tier for r in records} == {1, 3}

    def test_a_failing_tier_does_not_abort_the_cascade(self, agent):
        calls = {"n": 0}

        async def search_extended(query, max_results=200):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("pubmed down")
            return _articles(2, prefix=f"ok{calls['n']}_")

        agent.search.pubmed.search_extended = search_extended
        records, highest = _run(agent._tiered_search(_plan(), log=lambda *a: None))
        assert calls["n"] == 3
        assert len(records) == 4
        assert highest == 3

    def test_zero_results_trigger_the_intervention_only_fallback(self, agent):
        queries = []

        async def search_extended(query, max_results=200):
            queries.append(query)
            # first three (the cascade) return nothing; the fallback then succeeds
            return [] if len(queries) <= 3 else _articles(2, prefix=f"fb{len(queries)}_")

        agent.search.pubmed.search_extended = search_extended
        records, _ = _run(agent._tiered_search(_plan(), log=lambda *a: None))

        assert len(queries) > 3, "fallback must run a second round of queries"
        assert records, "fallback records must reach the pool"
        # the fallback query drops the outcome clause
        assert "effect" not in queries[3]
        assert "aspirin" in queries[3]

    def test_fallback_does_not_run_when_the_cascade_found_anything(self, agent):
        queries = []

        async def search_extended(query, max_results=200):
            queries.append(query)
            return _articles(1, prefix=f"x{len(queries)}_")

        agent.search.pubmed.search_extended = search_extended
        _run(agent._tiered_search(_plan(), log=lambda *a: None))
        assert len(queries) == 3

    def test_scholar_supplements_the_pool_with_tier_one_terms(self, agent, monkeypatch):
        _disable_scholar(
            monkeypatch,
            connected=True,
            results=[
                {"url": "https://scholar.example/a", "title": "Scholar A", "content": "snippet"},
                {"url": "https://scholar.example/b", "title": "Scholar B", "content": "snippet"},
            ],
        )

        async def search_extended(query, max_results=200):
            return _articles(1)

        agent.search.pubmed.search_extended = search_extended
        records, _ = _run(agent._tiered_search(_plan(), log=lambda *a: None))

        scholar = [r for r in records if r.source_db == "scholar"]
        assert len(scholar) == 2
        assert all(r.research_tier == 1 for r in scholar)
        assert all(r.pmid is None for r in scholar)

    def test_scholar_results_are_deduplicated_against_pubmed_urls(self, agent, monkeypatch):
        _disable_scholar(
            monkeypatch,
            connected=True,
            results=[{"url": "https://pubmed.example/p0", "title": "dupe", "content": "s"}],
        )

        async def search_extended(query, max_results=200):
            return _articles(1)

        agent.search.pubmed.search_extended = search_extended
        records, _ = _run(agent._tiered_search(_plan(), log=lambda *a: None))
        assert [r.source_db for r in records] == ["pubmed"]

    def test_junk_scholar_urls_are_dropped(self, agent, monkeypatch):
        _disable_scholar(
            monkeypatch,
            connected=True,
            results=[
                {"url": "https://www.dictionary.com/browse/aspirin", "title": "junk", "content": "s"},
                {"url": "https://scholar.example/good", "title": "good", "content": "s"},
            ],
        )

        async def search_extended(query, max_results=200):
            return _articles(1)

        agent.search.pubmed.search_extended = search_extended
        records, _ = _run(agent._tiered_search(_plan(), log=lambda *a: None))
        scholar = [r for r in records if r.source_db == "scholar"]
        assert [r.title for r in scholar] == ["good"]

    def test_pool_is_capped_at_500_records(self, agent):
        async def search_extended(query, max_results=200):
            return _articles(600)

        agent.search.pubmed.search_extended = search_extended
        records, _ = _run(agent._tiered_search(_plan(), log=lambda *a: None))
        assert len(records) == 500

    def test_fast_model_typing_updates_untyped_records(self, agent):
        async def search_extended(query, max_results=200):
            arts = _articles(2)
            for a in arts:
                a["study_type"] = "other"
            return arts

        agent.search.pubmed.search_extended = search_extended
        agent.fast_worker = SimpleNamespace()

        async def fake_screen(records):
            return [{"study_type": "RCT", "sample_size": "n=100", "primary_objective": "obj"} for _ in records]

        agent._fast_screen_abstracts = fake_screen
        records, _ = _run(agent._tiered_search(_plan(), log=lambda *a: None))
        assert all(r.study_type == "RCT" for r in records)
        assert all(r.sample_size == "n=100" for r in records)

    def test_social_domain_delegates_to_the_social_variant(self, agent):
        agent._domain = "social_science"
        called = {}

        async def fake_social(plan, log):
            called["yes"] = True
            return ([], 0)

        agent._tiered_search_social = fake_social
        _run(agent._tiered_search(_plan(), log=lambda *a: None))
        assert called == {"yes": True}


# ---------------------------------------------------------------------------
# _tiered_search_social — OpenAlex + ERIC
# ---------------------------------------------------------------------------


def _oa(n, start=0):
    return [
        {
            "title": f"OA {i}",
            "doi": f"10.3/oa{i}",
            "abstract_text": "abs",
            "publication_year": 2021,
            "url": f"https://openalex.example/{i}",
        }
        for i in range(start, start + n)
    ]


def _eric(n, start=0):
    return [
        {
            "title": f"ERIC {i}",
            "description": "abs",
            "publication_year": 2022,
            "url": f"https://eric.example/{i}",
        }
        for i in range(start, start + n)
    ]


class TestTieredSearchSocial:
    @pytest.fixture
    def social_agent(self, agent):
        agent._domain = "social_science"

        async def oa_search(query, per_page=50):
            return _oa(2)

        async def eric_search(query, max_results=30):
            return _eric(2)

        agent._openalex.search_works = oa_search
        agent._eric.search = eric_search
        return agent

    def test_both_sources_contribute_records(self, social_agent):
        records, highest = _run(social_agent._tiered_search_social(_plan(), log=lambda *a: None))
        sources = {r.source_db for r in records}
        assert "openalex" in sources
        assert "eric" in sources
        assert highest >= 1

    def test_records_are_deduplicated_by_title_across_tiers(self, social_agent):
        records, _ = _run(social_agent._tiered_search_social(_plan(), log=lambda *a: None))
        titles = [r.title for r in records]
        assert len(titles) == len(set(titles))

    def test_dedupe_by_title_alone_when_dois_differ(self, social_agent):
        """Isolates the title check.

        OpenAlex records derive their URL from the DOI, so varying the DOI is
        what makes the URL check unable to mask the title check.
        """
        calls = {"n": 0}

        async def oa_search(query, per_page=50):
            calls["n"] += 1
            rows = _oa(2)
            for r in rows:  # same titles, different DOIs (hence URLs) each tier
                r["doi"] = f"10.9/mirror{calls['n']}-{r['title']}"
            return rows

        async def no_eric(query, max_results=30):
            return []

        social_agent._openalex.search_works = oa_search
        social_agent._eric.search = no_eric
        records, _ = _run(social_agent._tiered_search_social(_plan(), log=lambda *a: None))
        assert [r.title for r in records] == ["OA 0", "OA 1"]

    def test_a_tier_with_no_terms_is_skipped(self, social_agent):
        seen = []

        async def oa_search(query, per_page=50):
            seen.append(query)
            return _oa(1)

        social_agent._openalex.search_works = oa_search
        plan = _plan()
        plan.tier2 = TierKeywords(intervention=[], outcome=[], population=[], rationale="r")
        _run(social_agent._tiered_search_social(plan, log=lambda *a: None))
        assert len(seen) == 2

    def test_a_failing_source_does_not_abort_the_other(self, social_agent):
        async def oa_boom(query, per_page=50):
            raise RuntimeError("openalex down")

        social_agent._openalex.search_works = oa_boom
        records, _ = _run(social_agent._tiered_search_social(_plan(), log=lambda *a: None))
        assert records, "ERIC results must still land in the pool"
        assert all(r.source_db == "eric" for r in records)
