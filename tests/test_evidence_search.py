"""Tests for the evidence-search allowlist (PLAN.md Step 1).

Everything here is offline. The one thing this tool must never do is let an
unvouched domain into a bundle, and that property cannot be demonstrated by
running a live search — a search that happens to return only good domains
proves nothing about the filter.

Three contracts:
  1. Subdomain matching is anchored on a dot, so a lookalike host cannot borrow
     an allowlisted domain's name.
  2. `_why` documents a tier; it is not a domain that vouches for anything.
  3. Rejected domains are never fetched. `deep_dive()` scrapes everything it
     finds, which is why this module does not use it — and why the test asserts
     on *what was requested*, not merely on what came back.
"""

from __future__ import annotations

import json

import pytest

from dr2_podcast.tools import evidence_search as ev

ALLOWLIST = {
    "_doc": "fixture",
    "primary_literature": {
        "_why": "Tier 1",
        "pubmed.ncbi.nlm.nih.gov": "PubMed records",
        "ncbi.nlm.nih.gov": "E-utilities/PMC",
    },
    "guidelines": {"_why": "Tier 2", "who.int": "WHO"},
}


@pytest.fixture
def allowlist_file(tmp_path):
    path = tmp_path / "evidence_domains.json"
    path.write_text(json.dumps(ALLOWLIST), encoding="utf-8")
    return path


# ------------------------------------------------------------------ matching


def test_exact_and_subdomain_hosts_are_allowed(allowlist_file):
    allow = ev.load_allowlist(path=allowlist_file)
    assert allow.match("https://pubmed.ncbi.nlm.nih.gov/12345/") == (
        "pubmed.ncbi.nlm.nih.gov", "primary_literature",
    )
    # www. and other subdomains of an allowlisted domain
    assert allow.match("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1/")[1] == "primary_literature"
    assert allow.match("https://who.int/news-room/fact-sheets/x")[0] == "who.int"


@pytest.mark.parametrize(
    "url",
    [
        "https://evilncbi.nlm.nih.gov/paper",                 # glued onto the leftmost label
        "https://pubmed.ncbi.nlm.nih.gov.attacker.test/x",    # allowlisted name as a prefix
        "https://who.int.evil.test/fact-sheet",
        "https://notwho.int/fact-sheet",
        "https://healthblog.example.com/what-a-p-value-means",
    ],
)
def test_lookalike_hosts_are_rejected(allowlist_file, url):
    """The dot in `.{domain}` is the whole defence. Without it every one of
    these reads as an authority, and the allowlist becomes decorative."""
    assert ev.load_allowlist(path=allowlist_file).match(url) is None


def test_host_matching_ignores_case_port_and_trailing_dot(allowlist_file):
    allow = ev.load_allowlist(path=allowlist_file)
    for url in ("https://WHO.int/x", "https://who.int.:443/x", "https://who.int:8443/x"):
        assert allow.match(url) is not None, url


def test_why_is_documentation_not_a_domain(allowlist_file):
    allow = ev.load_allowlist(path=allowlist_file)
    assert "_why" not in allow.tier_of
    assert allow.why["primary_literature"] == "Tier 1"


# --------------------------------------------------------------------- tiers


def test_tier_selection_narrows_the_allowlist(allowlist_file):
    """Step 6's guideline reviewer needs guidelines only — a PubMed hit is not
    an answer to "what do current clinical guidelines say"."""
    allow = ev.load_allowlist(["guidelines"], path=allowlist_file)
    assert allow.tiers == ["guidelines"]
    assert allow.match("https://who.int/x") is not None
    assert allow.match("https://pubmed.ncbi.nlm.nih.gov/1/") is None


def test_unknown_tier_is_an_error_not_an_empty_filter(allowlist_file):
    with pytest.raises(ev.EvidenceSearchError, match="unknown tier"):
        ev.load_allowlist(["guidlines"], path=allowlist_file)  # typo


def test_a_tier_with_no_domains_refuses_rather_than_allowing_everything(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"guidelines": {"_why": "nothing yet"}}), encoding="utf-8")
    with pytest.raises(ev.EvidenceSearchError, match="refusing to search unfiltered"):
        ev.load_allowlist(path=path)


def test_missing_allowlist_is_an_error(tmp_path):
    with pytest.raises(ev.EvidenceSearchError, match="allowlist missing"):
        ev.load_allowlist(path=tmp_path / "nope.json")


def test_the_shipped_allowlist_loads_and_covers_the_four_planned_tiers():
    """Guards the real data file: PLAN.md Step 1 names these four tiers, and
    Step 6 selects `guidelines` by name."""
    allow = ev.load_allowlist()
    assert allow.tiers == ["guidelines", "japanese_authority", "methodology_statistics", "primary_literature"]
    assert allow.match("https://pubmed.ncbi.nlm.nih.gov/1/")[1] == "primary_literature"
    assert allow.match("https://training.cochrane.org/handbook")[1] == "methodology_statistics"
    assert allow.match("https://www.mhlw.go.jp/x")[1] == "japanese_authority"


# ------------------------------------------------------------------- bundle


class _Hit:
    def __init__(self, url, title="t", snippet="s", engine="e"):
        self.url, self.title, self.snippet, self.engine = url, title, snippet, engine


class _Page:
    def __init__(self, url):
        self.url, self.title, self.content, self.word_count, self.error = url, "T", "body " * 50, 50, None


class _FakeAgent:
    """Stands in for search_agent.py, recording every URL actually fetched."""

    def __init__(self, hits):
        self.hits, self.fetched = hits, []
        agent = self

        class SearxngClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def search(self, query, num_results=5, language="en", engines=None):
                return [_Hit(u) for u in agent.hits]

        class DeepResearch:
            def __init__(self, client):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def fetch_page_content(self, url):
                agent.fetched.append(url)
                return _Page(url)

        self.SearxngClient, self.DeepResearch = SearxngClient, DeepResearch


@pytest.fixture
def fake_agent(monkeypatch, allowlist_file):
    def _install(hits):
        agent = _FakeAgent(hits)
        monkeypatch.setattr(ev, "_load_search_agent", lambda: agent)
        monkeypatch.setattr(ev, "DATA_FILE", allowlist_file)
        return agent

    return _install


def test_rejected_domains_are_never_fetched(fake_agent):
    """The reason this module calls search() + fetch_page_content() instead of
    deep_dive(): deep_dive scrapes every result, so the banned page would be
    downloaded and only then discarded."""
    agent = fake_agent([
        "https://pubmed.ncbi.nlm.nih.gov/111/",
        "https://healthblog.example.com/post",
        "https://who.int/fact-sheet",
    ])
    ev.search_evidence("statins reduce LDL", opts=ev.SearchOptions(scrape=5))

    assert agent.fetched == ["https://pubmed.ncbi.nlm.nih.gov/111/", "https://who.int/fact-sheet"]
    assert "healthblog.example.com" not in " ".join(agent.fetched)


def test_bundle_has_the_shape_plan_specifies(fake_agent):
    fake_agent(["https://pubmed.ncbi.nlm.nih.gov/111/", "https://blog.example.com/a"])
    bundle = ev.search_evidence("a claim", queries=["q1", "q2"], opts=ev.SearchOptions(scrape=1))

    assert set(bundle) >= {"claim", "queries", "allowed_hits", "rejected_domains", "scraped"}
    assert bundle["claim"] == "a claim"
    assert bundle["queries"] == ["q1", "q2"]
    assert [h["url"] for h in bundle["allowed_hits"]] == ["https://pubmed.ncbi.nlm.nih.gov/111/"]
    assert bundle["allowed_hits"][0]["tier"] == "primary_literature"
    # counted, not listed: one blog seen twice is a different problem from two blogs
    assert bundle["rejected_domains"] == {"blog.example.com": 2}
    assert len(bundle["scraped"]) == 1


def test_the_same_url_from_two_queries_appears_once(fake_agent):
    fake_agent(["https://pubmed.ncbi.nlm.nih.gov/111/"])
    bundle = ev.search_evidence("c", queries=["q1", "q2"], opts=ev.SearchOptions(scrape=0))
    assert len(bundle["allowed_hits"]) == 1


def test_scraped_content_is_capped_for_the_refuter_prompt(fake_agent):
    """Bundles are pasted into a prompt whose budget is shared with the script
    excerpt being judged (PLAN.md: Codex cannot fetch anything itself)."""
    fake_agent(["https://pubmed.ncbi.nlm.nih.gov/111/"])
    bundle = ev.search_evidence("c", opts=ev.SearchOptions(scrape=1, max_chars=20))
    page = bundle["scraped"][0]
    assert len(page["content"]) == 20
    assert page["truncated"] is True


def test_no_allowlisted_hit_is_reported_not_crashed(fake_agent):
    agent = fake_agent(["https://blog.example.com/a"])
    bundle = ev.search_evidence("an unsupported claim", opts=ev.SearchOptions(scrape=3))
    assert bundle["allowed_hits"] == []
    assert agent.fetched == []


def test_cli_exit_status_distinguishes_no_evidence_from_failure(fake_agent, capsys):
    fake_agent(["https://blog.example.com/a"])
    assert ev.main(["a claim", "--scrape", "0"]) == 1  # searched fine, found nothing vouched
    assert ev.main(["a claim", "--tier", "nosuchtier"]) == 2  # could not search at all


def test_a_broad_tier_does_not_relabel_a_narrower_tiers_domain(tmp_path):
    """Measured 2026-08-16: `--tier guidelines` returned three PubMed papers
    labelled `guidelines`, because ncbi.nlm.nih.gov sits under the guideline
    publisher nih.gov. Asking "what do current guidelines say" and being handed
    research papers is the failure Step 6 exists to avoid.

    Classification uses the whole allowlist and resolves to the most specific
    domain; only admission is narrowed by tier.
    """
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps({
        "primary_literature": {"_why": "T1", "ncbi.nlm.nih.gov": "PubMed/PMC"},
        "guidelines": {"_why": "T2", "nih.gov": "NIH guidance", "who.int": "WHO"},
    }), encoding="utf-8")

    pubmed = "https://www.ncbi.nlm.nih.gov/pubmed/41815847"

    every = ev.load_allowlist(path=path)
    assert every.match(pubmed) == ("ncbi.nlm.nih.gov", "primary_literature")

    guidelines_only = ev.load_allowlist(["guidelines"], path=path)
    assert guidelines_only.classify(pubmed)[1] == "primary_literature"  # still knows what it is
    assert guidelines_only.match(pubmed) is None                        # and will not admit it
    assert guidelines_only.match("https://www.nih.gov/news/x")[1] == "guidelines"
    assert guidelines_only.match("https://who.int/publications/i/item/9789241549028/") is not None
