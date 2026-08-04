"""Characterization tests for metadata_clients.enrich_papers_metadata.

Written before splitting that 89-statement, complexity-29 function into one
enricher per source. All three API clients are stubbed, so these test the
aggregation itself: which source wins, how sources are recorded, that one
failing API does not take the others down, and how the derived fields merge.

Existing coverage was two cases (empty list, no clients).
"""

import asyncio

import pytest

from dr2_podcast.research.metadata_clients import enrich_papers_metadata


class FakeOpenAlex:
    def __init__(self, by_doi=None, by_pmid=None, raise_batch=False, raise_pmid=False):
        self.by_doi = by_doi or {}
        self.by_pmid = by_pmid or {}
        self.raise_batch = raise_batch
        self.raise_pmid = raise_pmid
        self.batch_calls = []

    async def batch_get_works(self, dois):
        self.batch_calls.append(list(dois))
        if self.raise_batch:
            raise RuntimeError("openalex down")
        return [dict(v, doi=k) for k, v in self.by_doi.items()]

    async def get_work_by_pmid(self, pmid):
        if self.raise_pmid:
            raise RuntimeError("openalex pmid down")
        return self.by_pmid.get(pmid)


class FakeS2:
    def __init__(self, results=None, raise_batch=False):
        self.results = results or []
        self.raise_batch = raise_batch
        self.ids_seen = []

    async def batch_get_papers(self, ids):
        self.ids_seen = list(ids)
        if self.raise_batch:
            raise RuntimeError("s2 down")
        return self.results


class FakeCrossref:
    def __init__(self, by_doi=None, raise_batch=False):
        self.by_doi = by_doi or {}
        self.raise_batch = raise_batch

    async def batch_get_works(self, dois):
        if self.raise_batch:
            raise RuntimeError("crossref down")
        return [dict(v, doi=k) for k, v in self.by_doi.items()]


def _enrich(papers, **clients):
    return asyncio.run(enrich_papers_metadata(papers, **clients))


class TestOpenAlex:
    def test_fields_and_source_recorded(self):
        oa = FakeOpenAlex(
            by_doi={
                "10.1/a": {
                    "openalex_id": "W1",
                    "cited_by_count": 42,
                    "fwci": 2.5,
                    "is_retracted": True,
                    "funding": ["NIH"],
                    "concepts": ["cardiology"],
                    "abstract_text": "abstract",
                }
            }
        )
        [ep] = _enrich([{"doi": "10.1/a"}], openalex_client=oa)
        assert ep.openalex_id == "W1"
        assert ep.cited_by_count == 42
        assert ep.fwci == 2.5
        assert ep.openalex_is_retracted is True
        assert ep.openalex_funding == ["NIH"]
        assert ep.abstract_text == "abstract"
        assert ep.enrichment_sources == ["openalex"]

    def test_pmid_fallback_when_no_doi(self):
        oa = FakeOpenAlex(by_pmid={"999": {"openalex_id": "W9", "cited_by_count": 7}})
        [ep] = _enrich([{"pmid": "999"}], openalex_client=oa)
        assert ep.openalex_id == "W9"
        assert ep.cited_by_count == 7
        assert ep.enrichment_sources == ["openalex"]

    def test_pmid_fallback_is_skipped_when_the_doi_batch_already_matched(self):
        oa = FakeOpenAlex(
            by_doi={"10.1/a": {"openalex_id": "W1"}},
            by_pmid={"999": {"openalex_id": "SHOULD-NOT-BE-USED"}},
        )
        [ep] = _enrich([{"doi": "10.1/a", "pmid": "999"}], openalex_client=oa)
        assert ep.openalex_id == "W1"

    def test_batch_failure_is_non_fatal(self):
        oa = FakeOpenAlex(raise_batch=True)
        [ep] = _enrich([{"doi": "10.1/a"}], openalex_client=oa)
        assert ep.enrichment_sources == []
        assert ep.openalex_id == ""

    def test_pmid_fallback_failure_is_non_fatal(self):
        oa = FakeOpenAlex(raise_pmid=True)
        [ep] = _enrich([{"pmid": "999"}], openalex_client=oa)
        assert ep.enrichment_sources == []

    def test_no_dois_means_no_batch_call(self):
        oa = FakeOpenAlex()
        _enrich([{"pmid": "1"}], openalex_client=oa)
        assert oa.batch_calls == []


class TestSemanticScholar:
    def test_prefers_doi_prefixed_id_and_falls_back_to_pmid(self):
        s2 = FakeS2()
        _enrich([{"doi": "10.1/a", "pmid": "1"}, {"pmid": "2"}], s2_client=s2)
        assert s2.ids_seen == ["DOI:10.1/a", "PMID:2"]

    def test_fields_and_source_recorded(self):
        s2 = FakeS2(
            results=[
                {
                    "doi": "10.1/a",
                    "s2_id": "S1",
                    "citation_count": 10,
                    "influential_citation_count": 3,
                    "fields_of_study": ["Medicine"],
                    "tldr": "short",
                }
            ]
        )
        [ep] = _enrich([{"doi": "10.1/a"}], s2_client=s2)
        assert ep.s2_id == "S1"
        assert ep.s2_citation_count == 10
        assert ep.influential_citation_count == 3
        assert ep.fields_of_study == ["Medicine"]
        assert ep.tldr == "short"
        assert ep.enrichment_sources == ["semantic_scholar"]

    def test_doi_match_wins_over_pmid_match(self):
        """A paper carrying both identifiers must resolve by DOI, not PMID."""
        s2 = FakeS2(
            results=[
                {"doi": "10.1/a", "s2_id": "BY-DOI"},
                {"pmid": "999", "s2_id": "BY-PMID"},
            ]
        )
        [ep] = _enrich([{"doi": "10.1/a", "pmid": "999"}], s2_client=s2)
        assert ep.s2_id == "BY-DOI"

    def test_batch_failure_is_non_fatal(self):
        [ep] = _enrich([{"doi": "10.1/a"}], s2_client=FakeS2(raise_batch=True))
        assert ep.enrichment_sources == []


class TestCrossref:
    def test_fields_and_source_recorded(self):
        cr = FakeCrossref(
            by_doi={
                "10.1/a": {
                    "is_referenced_by_count": 5,
                    "funder": ["Wellcome"],
                    "is_retracted": True,
                    "is_corrected": True,
                    "clinical_trial_numbers": ["NCT1"],
                }
            }
        )
        [ep] = _enrich([{"doi": "10.1/a"}], crossref_client=cr)
        assert ep.crossref_citation_count == 5
        assert ep.crossref_funders == ["Wellcome"]
        assert ep.crossref_is_retracted is True
        assert ep.clinical_trial_numbers == ["NCT1"]
        assert ep.enrichment_sources == ["crossref"]

    def test_batch_failure_is_non_fatal(self):
        [ep] = _enrich([{"doi": "10.1/a"}], crossref_client=FakeCrossref(raise_batch=True))
        assert ep.enrichment_sources == []


class TestDerivedFields:
    def test_retraction_is_true_if_any_source_says_so(self):
        oa = FakeOpenAlex(by_doi={"10.1/a": {"is_retracted": True}})
        cr = FakeCrossref(by_doi={"10.1/a": {"is_retracted": False}})
        [ep] = _enrich([{"doi": "10.1/a"}], openalex_client=oa, crossref_client=cr)
        assert ep.is_retracted is True

    def test_not_retracted_when_neither_source_says_so(self):
        oa = FakeOpenAlex(by_doi={"10.1/a": {"is_retracted": False}})
        cr = FakeCrossref(by_doi={"10.1/a": {"is_retracted": False}})
        [ep] = _enrich([{"doi": "10.1/a"}], openalex_client=oa, crossref_client=cr)
        assert ep.is_retracted is False

    def test_funding_is_merged_and_deduplicated_preserving_order(self):
        oa = FakeOpenAlex(by_doi={"10.1/a": {"funding": ["NIH", "Wellcome"]}})
        cr = FakeCrossref(by_doi={"10.1/a": {"funder": ["Wellcome", "ERC"]}})
        [ep] = _enrich([{"doi": "10.1/a"}], openalex_client=oa, crossref_client=cr)
        assert ep.all_funding_sources == ["NIH", "Wellcome", "ERC"]

    def test_best_citation_count_is_the_maximum_available(self):
        oa = FakeOpenAlex(by_doi={"10.1/a": {"cited_by_count": 10}})
        s2 = FakeS2(results=[{"doi": "10.1/a", "citation_count": 25}])
        cr = FakeCrossref(by_doi={"10.1/a": {"is_referenced_by_count": 18}})
        [ep] = _enrich([{"doi": "10.1/a"}], openalex_client=oa, s2_client=s2, crossref_client=cr)
        assert ep.best_citation_count == 25

    def test_best_citation_count_is_none_when_no_source_has_one(self):
        [ep] = _enrich([{"doi": "10.1/a"}])
        assert ep.best_citation_count is None

    def test_all_three_sources_are_recorded_together(self):
        oa = FakeOpenAlex(by_doi={"10.1/a": {"openalex_id": "W1"}})
        s2 = FakeS2(results=[{"doi": "10.1/a", "s2_id": "S1"}])
        cr = FakeCrossref(by_doi={"10.1/a": {"is_referenced_by_count": 1}})
        [ep] = _enrich([{"doi": "10.1/a"}], openalex_client=oa, s2_client=s2, crossref_client=cr)
        assert ep.enrichment_sources == ["openalex", "semantic_scholar", "crossref"]


class TestOrderAndIdentity:
    def test_output_order_matches_input_order(self):
        papers = [{"doi": f"10.1/{i}"} for i in range(5)]
        result = _enrich(papers)
        assert [ep.doi for ep in result] == [p["doi"] for p in papers]

    def test_papers_without_doi_or_pmid_still_produce_an_entry(self):
        [ep] = _enrich([{}])
        assert ep.doi == ""
        assert ep.pmid == ""
        assert ep.enrichment_sources == []

    def test_empty_input(self):
        assert _enrich([]) == []
