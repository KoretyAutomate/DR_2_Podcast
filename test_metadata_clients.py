"""
Unit tests for metadata_clients.py — offline tests using sample JSON fixtures.
"""

import asyncio
import os
import sqlite3
import tempfile
import time

import pytest

from metadata_clients import (
    RateLimiter,
    MetadataCache,
    OpenAlexClient,
    SemanticScholarClient,
    CrossrefClient,
    ERICClient,
    EnrichedPaper,
    enrich_papers_metadata,
)


# ──────────────────────────────────────────────────────────────
# RateLimiter Tests
# ──────────────────────────────────────────────────────────────

class TestRateLimiter:

    def test_basic_rate_limiting(self):
        """Two requests at 5 req/s should take at least 0.2s total."""
        limiter = RateLimiter(rate=5.0)

        async def timed_requests():
            start = asyncio.get_event_loop().time()
            async with limiter:
                pass
            async with limiter:
                pass
            elapsed = asyncio.get_event_loop().time() - start
            return elapsed

        elapsed = asyncio.run(timed_requests())
        assert elapsed >= 0.15, f"Expected >= 0.15s, got {elapsed:.3f}s"

    def test_high_rate_no_delay(self):
        """At 1000 req/s, two requests should be near-instant."""
        limiter = RateLimiter(rate=1000.0)

        async def fast_requests():
            start = asyncio.get_event_loop().time()
            async with limiter:
                pass
            async with limiter:
                pass
            return asyncio.get_event_loop().time() - start

        elapsed = asyncio.run(fast_requests())
        assert elapsed < 0.05, f"Expected < 0.05s, got {elapsed:.3f}s"


# ──────────────────────────────────────────────────────────────
# MetadataCache Tests
# ──────────────────────────────────────────────────────────────

class TestMetadataCache:

    def test_put_and_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test_cache.db")
            cache = MetadataCache(db_path=db_path, ttl_days=30)
            try:
                cache.put("openalex", "doi:10.1234/test", {"cited_by_count": 42})
                result = cache.get("openalex", "doi:10.1234/test")
                assert result is not None
                assert result["cited_by_count"] == 42
            finally:
                cache.close()

    def test_get_missing_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test_cache.db")
            cache = MetadataCache(db_path=db_path)
            try:
                result = cache.get("openalex", "doi:nonexistent")
                assert result is None
            finally:
                cache.close()

    def test_ttl_expiration(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test_cache.db")
            cache = MetadataCache(db_path=db_path, ttl_days=0)  # 0 days = immediate expiry
            try:
                # Manually insert with old timestamp
                cache.conn.execute(
                    "INSERT INTO metadata_cache (api_name, identifier, data, fetched_at) "
                    "VALUES (?, ?, ?, ?)",
                    ("test_api", "test_id", '{"val": 1}', time.time() - 100),
                )
                cache.conn.commit()
                result = cache.get("test_api", "test_id")
                assert result is None, "Expired entry should not be returned"
            finally:
                cache.close()

    def test_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test_cache.db")
            cache = MetadataCache(db_path=db_path)
            try:
                cache.put("api", "key1", {"v": 1})
                cache.put("api", "key1", {"v": 2})
                result = cache.get("api", "key1")
                assert result["v"] == 2
            finally:
                cache.close()

    def test_different_api_same_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test_cache.db")
            cache = MetadataCache(db_path=db_path)
            try:
                cache.put("api_a", "key1", {"source": "a"})
                cache.put("api_b", "key1", {"source": "b"})
                assert cache.get("api_a", "key1")["source"] == "a"
                assert cache.get("api_b", "key1")["source"] == "b"
            finally:
                cache.close()


# ──────────────────────────────────────────────────────────────
# OpenAlex Client Tests
# ──────────────────────────────────────────────────────────────

class TestOpenAlexClient:

    def test_reconstruct_abstract(self):
        """Test inverted index → text reconstruction."""
        inverted = {
            "This": [0],
            "is": [1, 5],
            "a": [2],
            "test.": [3],
            "It": [4],
            "working.": [6],
        }
        text = OpenAlexClient._reconstruct_abstract(inverted)
        assert text == "This is a test. It is working."

    def test_reconstruct_abstract_empty(self):
        assert OpenAlexClient._reconstruct_abstract(None) == ""
        assert OpenAlexClient._reconstruct_abstract({}) == ""

    def test_reconstruct_abstract_single_word(self):
        assert OpenAlexClient._reconstruct_abstract({"Hello": [0]}) == "Hello"

    def test_normalize_work(self):
        """Test normalization of a sample OpenAlex work record."""
        client = OpenAlexClient()
        raw = {
            "id": "https://openalex.org/W12345",
            "doi": "https://doi.org/10.1038/s41586-020-2649-2",
            "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/32845998"},
            "cited_by_count": 1500,
            "fwci": 12.5,
            "is_retracted": False,
            "type": "article",
            "grants": [{"funder_display_name": "NIH"}],
            "concepts": [
                {"display_name": "Machine Learning", "score": 0.8},
                {"display_name": "Noise", "score": 0.1},  # below threshold
            ],
            "open_access": {"is_oa": True, "oa_status": "gold"},
            "abstract_inverted_index": {"Sample": [0], "abstract.": [1]},
            "referenced_works": ["https://openalex.org/W999"],
            "authorships": [],
        }
        result = client._normalize_work(raw)
        assert result["doi"] == "10.1038/s41586-020-2649-2"
        assert result["pmid"] == "32845998"
        assert result["cited_by_count"] == 1500
        assert result["fwci"] == 12.5
        assert result["is_retracted"] is False
        assert result["funding"] == ["NIH"]
        assert result["concepts"] == ["Machine Learning"]
        assert result["open_access"]["is_oa"] is True
        assert result["abstract_text"] == "Sample abstract."

    def test_normalize_work_missing_fields(self):
        """Test normalization with minimal data."""
        client = OpenAlexClient()
        raw = {"id": "W1"}
        result = client._normalize_work(raw)
        assert result["openalex_id"] == "W1"
        assert result["doi"] == ""
        assert result["pmid"] == ""
        assert result["cited_by_count"] == 0
        assert result["fwci"] is None
        assert result["is_retracted"] is False
        assert result["funding"] == []
        assert result["concepts"] == []


# ──────────────────────────────────────────────────────────────
# Semantic Scholar Client Tests
# ──────────────────────────────────────────────────────────────

class TestSemanticScholarClient:

    def test_normalize_paper(self):
        client = SemanticScholarClient()
        raw = {
            "paperId": "abc123",
            "title": "Test Paper",
            "citationCount": 250,
            "influentialCitationCount": 15,
            "fieldsOfStudy": ["Medicine", "Computer Science"],
            "tldr": {"text": "This paper is about testing."},
            "isOpenAccess": True,
            "externalIds": {"DOI": "10.1234/test", "PubMed": "12345678"},
        }
        result = client._normalize_paper(raw)
        assert result["s2_id"] == "abc123"
        assert result["citation_count"] == 250
        assert result["influential_citation_count"] == 15
        assert result["fields_of_study"] == ["Medicine", "Computer Science"]
        assert result["tldr"] == "This paper is about testing."
        assert result["doi"] == "10.1234/test"
        assert result["pmid"] == "12345678"

    def test_normalize_paper_missing_fields(self):
        client = SemanticScholarClient()
        raw = {"paperId": "xyz"}
        result = client._normalize_paper(raw)
        assert result["s2_id"] == "xyz"
        assert result["citation_count"] == 0
        assert result["influential_citation_count"] == 0
        assert result["tldr"] == ""
        assert result["doi"] == ""

    def test_normalize_paper_tldr_none(self):
        client = SemanticScholarClient()
        raw = {"paperId": "x", "tldr": None}
        result = client._normalize_paper(raw)
        assert result["tldr"] == ""


# ──────────────────────────────────────────────────────────────
# Crossref Client Tests
# ──────────────────────────────────────────────────────────────

class TestCrossrefClient:

    def test_check_retraction_true(self):
        """Retracted paper detected via update-to field."""
        raw = {"update-to": [{"type": "retraction", "DOI": "10.1234/retracted"}]}
        assert CrossrefClient._check_retraction(raw) is True

    def test_check_retraction_false(self):
        raw = {"update-to": []}
        assert CrossrefClient._check_retraction(raw) is False

    def test_check_retraction_via_relation(self):
        raw = {"update-to": [], "relation": {"is-retracted-by": [{"id": "10.xxx"}]}}
        assert CrossrefClient._check_retraction(raw) is True

    def test_check_retraction_no_field(self):
        raw = {}
        assert CrossrefClient._check_retraction(raw) is False

    def test_check_correction(self):
        raw = {"update-to": [{"type": "correction"}]}
        assert CrossrefClient._check_correction(raw) is True

    def test_check_correction_erratum(self):
        raw = {"update-to": [{"type": "erratum"}]}
        assert CrossrefClient._check_correction(raw) is True

    def test_check_correction_false(self):
        raw = {"update-to": [{"type": "retraction"}]}
        assert CrossrefClient._check_correction(raw) is False

    def test_normalize_work(self):
        client = CrossrefClient()
        raw = {
            "DOI": "10.1234/test",
            "is-referenced-by-count": 100,
            "references-count": 30,
            "funder": [{"name": "NIH"}, {"name": "NSF"}],
            "clinical-trial-number": [
                {"clinical-trial-number": "NCT001234"}
            ],
            "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}],
            "update-to": [],
            "relation": {},
        }
        result = client._normalize_work(raw)
        assert result["doi"] == "10.1234/test"
        assert result["is_referenced_by_count"] == 100
        assert result["funder"] == ["NIH", "NSF"]
        assert result["clinical_trial_numbers"] == ["NCT001234"]
        assert result["is_retracted"] is False
        assert result["is_corrected"] is False

    def test_normalize_work_retracted(self):
        client = CrossrefClient()
        raw = {
            "DOI": "10.1234/retracted",
            "is-referenced-by-count": 5,
            "update-to": [{"type": "retraction"}],
        }
        result = client._normalize_work(raw)
        assert result["is_retracted"] is True


# ──────────────────────────────────────────────────────────────
# ERIC Client Tests
# ──────────────────────────────────────────────────────────────

class TestERICClient:

    def test_normalize_result(self):
        client = ERICClient()
        raw = {
            "id": "EJ1234567",
            "title": "Reading Intervention Study",
            "author": ["Smith, John", "Doe, Jane"],
            "description": "A study on reading.",
            "subject": ["Reading", "Elementary Education"],
            "educationLevel": ["Grade 3", "Grade 4"],
            "peerreviewed": "T",
            "publicationType": ["Journal Article"],
            "url": "https://eric.ed.gov/?id=EJ1234567",
            "publicationDateYear": "2022",
            "source": "Journal of Education",
        }
        result = client._normalize_result(raw)
        assert result["eric_id"] == "EJ1234567"
        assert result["peer_reviewed"] is True
        assert result["year"] == 2022
        assert "Reading" in result["subject"]

    def test_normalize_result_not_peer_reviewed(self):
        client = ERICClient()
        raw = {"id": "ED123", "peerreviewed": "F"}
        result = client._normalize_result(raw)
        assert result["peer_reviewed"] is False

    def test_extract_year_missing(self):
        assert ERICClient._extract_year({}) is None
        assert ERICClient._extract_year({"publicationDateYear": "abc"}) is None


# ──────────────────────────────────────────────────────────────
# EnrichedPaper Tests
# ──────────────────────────────────────────────────────────────

class TestEnrichedPaper:

    def test_to_dict_minimal(self):
        ep = EnrichedPaper(doi="10.1234/test")
        d = ep.to_dict()
        assert d["doi"] == "10.1234/test"
        assert d["is_retracted"] is False
        assert d["is_corrected"] is False

    def test_to_dict_full(self):
        ep = EnrichedPaper(
            doi="10.1234/test",
            pmid="12345",
            cited_by_count=100,
            fwci=3.5,
            is_retracted=True,
            all_funding_sources=["NIH"],
            enrichment_sources=["openalex", "crossref"],
        )
        d = ep.to_dict()
        assert d["cited_by_count"] == 100
        assert d["fwci"] == 3.5
        assert d["is_retracted"] is True
        assert d["all_funding_sources"] == ["NIH"]


# ──────────────────────────────────────────────────────────────
# Aggregator Tests (offline — no API calls)
# ──────────────────────────────────────────────────────────────

class TestAggregator:

    def test_enrich_empty_list(self):
        result = asyncio.run(enrich_papers_metadata([]))
        assert result == []

    def test_enrich_no_clients(self):
        papers = [{"doi": "10.1234/test", "pmid": "12345"}]
        result = asyncio.run(enrich_papers_metadata(papers))
        assert len(result) == 1
        assert result[0].doi == "10.1234/test"
        assert result[0].pmid == "12345"
        assert result[0].enrichment_sources == []
        assert result[0].is_retracted is False
