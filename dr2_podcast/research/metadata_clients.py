"""
Async API clients for academic metadata enrichment.

Free APIs:
- OpenAlex (https://openalex.org) — citation counts, FWCI, funding, retraction status
- Semantic Scholar (https://api.semanticscholar.org) — influential citations, TLDR, fields of study
- Crossref (https://api.crossref.org) — funder registry, retraction/correction status, clinical trial numbers
- ERIC (https://api.ies.ed.gov/eric/) — education research database (social science pipeline)

All clients share:
- RateLimiter: async token-bucket rate limiter
- MetadataCache: SQLite-backed cache with 30-day TTL

Zero cost — all APIs are free tier.
"""

import asyncio
import atexit
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Shared Infrastructure
# ──────────────────────────────────────────────────────────────

class RateLimiter:
    """Async token-bucket rate limiter.

    Usage:
        limiter = RateLimiter(rate=10)  # 10 req/sec
        async with limiter:
            await do_request()
    """

    def __init__(self, rate: float = 10.0):
        self.rate = rate
        self.min_interval = 1.0 / rate
        self._last_request = 0.0
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_request
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last_request = asyncio.get_event_loop().time()
        return self

    async def __aexit__(self, *exc):
        pass


class MetadataCache:
    """SQLite-backed metadata cache with configurable TTL.

    Follows PageCache pattern from clinical_research.py.
    Default TTL: 30 days.
    DB path: ~/.cache/dr2podcast/metadata_cache.db
    """

    def __init__(self, db_path: str = None, ttl_days: int = 30):
        if db_path is None:
            db_path = os.path.expanduser("~/.cache/dr2podcast/metadata_cache.db")
        self.db_path = db_path
        self.ttl_seconds = ttl_days * 86400
        self._closed = False

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata_cache "
            "(api_name TEXT, identifier TEXT, data TEXT, fetched_at REAL, "
            "PRIMARY KEY (api_name, identifier))"
        )
        # Cleanup expired entries
        cutoff = time.time() - self.ttl_seconds
        deleted = self.conn.execute(
            "DELETE FROM metadata_cache WHERE fetched_at < ?", (cutoff,)
        ).rowcount
        self.conn.commit()
        if deleted:
            logger.info(f"MetadataCache: cleaned {deleted} expired entries")

        # Ensure connection is closed on interpreter shutdown
        atexit.register(self.close)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def get(self, api_name: str, identifier: str) -> Optional[dict]:
        cutoff = time.time() - self.ttl_seconds
        row = self.conn.execute(
            "SELECT data FROM metadata_cache "
            "WHERE api_name = ? AND identifier = ? AND fetched_at > ?",
            (api_name, identifier, cutoff),
        ).fetchone()
        if row:
            try:
                return json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    def put(self, api_name: str, identifier: str, data: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata_cache "
            "(api_name, identifier, data, fetched_at) VALUES (?, ?, ?, ?)",
            (api_name, identifier, json.dumps(data), time.time()),
        )
        self.conn.commit()

    def close(self):
        if not self._closed:
            self._closed = True
            self.conn.close()


# ──────────────────────────────────────────────────────────────
# OpenAlex Client
# ──────────────────────────────────────────────────────────────

class OpenAlexClient:
    """Client for the OpenAlex API (https://api.openalex.org).

    Auth: free, polite pool via OPENALEX_EMAIL env var.
    Rate: 10 req/s (polite pool).
    """

    BASE_URL = "https://api.openalex.org"

    def __init__(self, email: str = None, cache: MetadataCache = None):
        self.email = email or os.getenv("OPENALEX_EMAIL", "")
        self.cache = cache
        self.limiter = RateLimiter(rate=10.0)
        self._http = httpx.AsyncClient(timeout=30)

    async def close(self):
        """Close the underlying HTTP client."""
        await self._http.aclose()

    def _headers(self) -> dict:
        h = {"Accept": "application/json"}
        if self.email:
            h["User-Agent"] = f"DR2Podcast/1.0 (mailto:{self.email})"
        return h

    def _params(self) -> dict:
        p = {}
        if self.email:
            p["mailto"] = self.email
        return p

    @staticmethod
    def _reconstruct_abstract(inverted_index: Optional[dict]) -> str:
        """Reconstruct abstract text from OpenAlex inverted index format."""
        if not inverted_index:
            return ""
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in word_positions)

    def _normalize_work(self, raw: dict) -> dict:
        """Normalize an OpenAlex work record to a standard dict."""
        oa = raw.get("open_access", {})
        funding_list = []
        for f in raw.get("grants", []):
            funder_name = f.get("funder_display_name") or f.get("funder", "")
            if funder_name:
                funding_list.append(funder_name)
        if not funding_list:
            for a in raw.get("authorships", []):
                for inst in a.get("institutions", []):
                    if inst.get("type") == "funder":
                        funding_list.append(inst.get("display_name", ""))

        doi_raw = raw.get("doi", "") or ""
        doi = doi_raw.replace("https://doi.org/", "")

        ids = raw.get("ids", {})
        pmid_raw = ids.get("pmid", "") or ""
        pmid = pmid_raw.replace("https://pubmed.ncbi.nlm.nih.gov/", "").strip("/")

        concepts = [
            c.get("display_name", "")
            for c in raw.get("concepts", [])
            if c.get("score", 0) > 0.3
        ]

        return {
            "openalex_id": raw.get("id", ""),
            "doi": doi,
            "pmid": pmid,
            "cited_by_count": raw.get("cited_by_count", 0),
            "fwci": raw.get("fwci"),
            "is_retracted": raw.get("is_retracted", False),
            "funding": funding_list,
            "concepts": concepts,
            "referenced_works": raw.get("referenced_works", []),
            "type": raw.get("type", ""),
            "open_access": {
                "is_oa": oa.get("is_oa", False),
                "oa_status": oa.get("oa_status", ""),
            },
            "abstract_text": self._reconstruct_abstract(
                raw.get("abstract_inverted_index")
            ),
        }

    async def get_work_by_doi(self, doi: str) -> Optional[dict]:
        """Look up a single work by DOI."""
        if self.cache:
            cached = self.cache.get("openalex", f"doi:{doi}")
            if cached:
                return cached
        try:
            async with self.limiter:
                url = f"{self.BASE_URL}/works/https://doi.org/{doi}"
                resp = await self._http.get(url, headers=self._headers(), params=self._params())
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                result = self._normalize_work(resp.json())
                if self.cache:
                    self.cache.put("openalex", f"doi:{doi}", result)
                return result
        except Exception as e:
            logger.warning(f"OpenAlex DOI lookup failed for {doi}: {e}")
            return None

    async def get_work_by_pmid(self, pmid: str) -> Optional[dict]:
        """Look up a single work by PMID."""
        if self.cache:
            cached = self.cache.get("openalex", f"pmid:{pmid}")
            if cached:
                return cached
        try:
            async with self.limiter:
                url = f"{self.BASE_URL}/works/pmid:{pmid}"
                resp = await self._http.get(url, headers=self._headers(), params=self._params())
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                result = self._normalize_work(resp.json())
                if self.cache:
                    self.cache.put("openalex", f"pmid:{pmid}", result)
                return result
        except Exception as e:
            logger.warning(f"OpenAlex PMID lookup failed for {pmid}: {e}")
            return None

    async def batch_get_works(self, identifiers: List[str]) -> List[dict]:
        """Batch lookup up to 50 works via pipe-separated filter.

        identifiers: list of DOIs (e.g. ["10.1234/abc", "10.5678/def"])
        """
        if not identifiers:
            return []
        results = []
        # Process in batches of 50 (OpenAlex limit)
        for i in range(0, len(identifiers), 50):
            batch = identifiers[i:i + 50]
            filter_str = "|".join(f"https://doi.org/{d}" for d in batch)
            try:
                async with self.limiter:
                    params = {**self._params(), "filter": f"doi:{filter_str}", "per_page": 50}
                    resp = await self._http.get(
                        f"{self.BASE_URL}/works",
                        headers=self._headers(),
                        params=params,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for work in data.get("results", []):
                        results.append(self._normalize_work(work))
            except Exception as e:
                logger.warning(f"OpenAlex batch lookup failed: {e}")
        return results

    async def search_works(
        self, query: str, filters: Optional[dict] = None,
        per_page: int = 25
    ) -> List[dict]:
        """Full-text search with optional filters.

        filters: dict of OpenAlex filter keys, e.g.
            {"from_publication_date": "2020-01-01", "type": "article"}
        """
        try:
            params = {**self._params(), "search": query, "per_page": per_page}
            if filters:
                filter_parts = []
                for k, v in filters.items():
                    filter_parts.append(f"{k}:{v}")
                params["filter"] = ",".join(filter_parts)

            async with self.limiter:
                resp = await self._http.get(
                    f"{self.BASE_URL}/works",
                    headers=self._headers(),
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()
                return [self._normalize_work(w) for w in data.get("results", [])]
        except Exception as e:
            logger.warning(f"OpenAlex search failed: {e}")
            return []


# ──────────────────────────────────────────────────────────────
# Semantic Scholar Client
# ──────────────────────────────────────────────────────────────

class SemanticScholarClient:
    """Client for the Semantic Scholar Graph API.

    Auth: free API key via S2_API_KEY env var (higher rate limits).
    Rate: 1 req/s with key, 100 req/5min without.
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    DEFAULT_FIELDS = "paperId,title,citationCount,influentialCitationCount,fieldsOfStudy,tldr,isOpenAccess,externalIds"

    def __init__(self, api_key: str = None, cache: MetadataCache = None):
        self.api_key = api_key or os.getenv("S2_API_KEY", "")
        self.cache = cache
        rate = 1.0 if self.api_key else 0.3  # conservative without key
        self.limiter = RateLimiter(rate=rate)
        self._http = httpx.AsyncClient(timeout=30)

    async def close(self):
        """Close the underlying HTTP client."""
        await self._http.aclose()

    def _headers(self) -> dict:
        h = {"Accept": "application/json"}
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    def _normalize_paper(self, raw: dict) -> dict:
        tldr = raw.get("tldr")
        tldr_text = tldr.get("text", "") if isinstance(tldr, dict) else ""
        ext_ids = raw.get("externalIds", {}) or {}
        return {
            "s2_id": raw.get("paperId", ""),
            "title": raw.get("title", ""),
            "citation_count": raw.get("citationCount", 0),
            "influential_citation_count": raw.get("influentialCitationCount", 0),
            "fields_of_study": raw.get("fieldsOfStudy") or [],
            "tldr": tldr_text,
            "is_open_access": raw.get("isOpenAccess", False),
            "doi": ext_ids.get("DOI", ""),
            "pmid": str(ext_ids.get("PubMed", "")),
        }

    async def get_paper(self, paper_id: str, fields: str = None) -> Optional[dict]:
        """Look up a single paper. Accepts DOI:xxx, PMID:xxx, or S2 paper ID."""
        cache_key = f"s2:{paper_id}"
        if self.cache:
            cached = self.cache.get("semantic_scholar", cache_key)
            if cached:
                return cached
        try:
            async with self.limiter:
                params = {"fields": fields or self.DEFAULT_FIELDS}
                resp = await self._http.get(
                    f"{self.BASE_URL}/paper/{paper_id}",
                    headers=self._headers(),
                    params=params,
                )
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                result = self._normalize_paper(resp.json())
                if self.cache:
                    self.cache.put("semantic_scholar", cache_key, result)
                return result
        except Exception as e:
            logger.warning(f"Semantic Scholar lookup failed for {paper_id}: {e}")
            return None

    async def batch_get_papers(
        self, paper_ids: List[str], fields: str = None
    ) -> List[dict]:
        """Batch lookup up to 500 papers via POST."""
        if not paper_ids:
            return []
        results = []
        batch_fields = fields or self.DEFAULT_FIELDS
        # Process in batches of 500
        for i in range(0, len(paper_ids), 500):
            batch = paper_ids[i:i + 500]
            try:
                async with self.limiter:
                    resp = await self._http.post(
                        f"{self.BASE_URL}/paper/batch",
                        headers=self._headers(),
                        params={"fields": batch_fields},
                        json={"ids": batch},
                    )
                    resp.raise_for_status()
                    for paper in resp.json():
                        if paper:
                            results.append(self._normalize_paper(paper))
            except Exception as e:
                logger.warning(f"Semantic Scholar batch lookup failed: {e}")
        return results

    async def search_papers(
        self, query: str, fields_of_study: Optional[List[str]] = None,
        year: Optional[str] = None, limit: int = 25
    ) -> List[dict]:
        """Relevance search with optional field/year filters."""
        try:
            params = {
                "query": query,
                "limit": min(limit, 100),
                "fields": self.DEFAULT_FIELDS,
            }
            if fields_of_study:
                params["fieldsOfStudy"] = ",".join(fields_of_study)
            if year:
                params["year"] = year

            async with self.limiter:
                resp = await self._http.get(
                    f"{self.BASE_URL}/paper/search",
                    headers=self._headers(),
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()
                return [self._normalize_paper(p) for p in data.get("data", [])]
        except Exception as e:
            logger.warning(f"Semantic Scholar search failed: {e}")
            return []


# ──────────────────────────────────────────────────────────────
# Crossref Client
# ──────────────────────────────────────────────────────────────

class CrossrefClient:
    """Client for the Crossref API (https://api.crossref.org).

    Auth: none (polite pool via CROSSREF_MAILTO env var in User-Agent).
    Rate: 10 req/s polite pool.
    """

    BASE_URL = "https://api.crossref.org"

    def __init__(self, mailto: str = None, cache: MetadataCache = None):
        self.mailto = mailto or os.getenv("CROSSREF_MAILTO", "")
        self.cache = cache
        self.limiter = RateLimiter(rate=10.0)
        self._http = httpx.AsyncClient(timeout=15)

    async def close(self):
        """Close the underlying HTTP client."""
        await self._http.aclose()

    def _headers(self) -> dict:
        h = {"Accept": "application/json"}
        if self.mailto:
            h["User-Agent"] = f"DR2Podcast/1.0 (mailto:{self.mailto})"
        return h

    @staticmethod
    def _check_retraction(raw: dict) -> bool:
        """Check if a work has been retracted via the updated-by field."""
        for update in raw.get("update-to", []):
            if update.get("type", "").lower() == "retraction":
                return True
        # Also check relation field
        for rel in raw.get("relation", {}).get("is-retracted-by", []):
            return True
        return False

    @staticmethod
    def _check_correction(raw: dict) -> bool:
        """Check if a work has been corrected."""
        for update in raw.get("update-to", []):
            utype = update.get("type", "").lower()
            if utype in ("correction", "erratum"):
                return True
        return False

    def _normalize_work(self, raw: dict) -> dict:
        funders = []
        for f in raw.get("funder", []):
            name = f.get("name", "")
            if name:
                funders.append(name)

        clinical_trials = []
        for ct in raw.get("clinical-trial-number", []):
            num = ct.get("clinical-trial-number", "")
            if num:
                clinical_trials.append(num)

        licenses = []
        for lic in raw.get("license", []):
            url = lic.get("URL", "")
            if url:
                licenses.append(url)

        return {
            "doi": raw.get("DOI", ""),
            "is_referenced_by_count": raw.get("is-referenced-by-count", 0),
            "funder": funders,
            "reference_count": raw.get("references-count", 0),
            "is_retracted": self._check_retraction(raw),
            "is_corrected": self._check_correction(raw),
            "clinical_trial_numbers": clinical_trials,
            "license": licenses,
        }

    async def get_work(self, doi: str) -> Optional[dict]:
        """Look up a single work by DOI."""
        if self.cache:
            cached = self.cache.get("crossref", f"doi:{doi}")
            if cached:
                return cached
        try:
            async with self.limiter:
                resp = await self._http.get(
                    f"{self.BASE_URL}/works/{doi}",
                    headers=self._headers(),
                )
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                msg = resp.json().get("message", {})
                result = self._normalize_work(msg)
                if self.cache:
                    self.cache.put("crossref", f"doi:{doi}", result)
                return result
        except Exception as e:
            logger.warning(f"Crossref lookup failed for {doi}: {e}")
            return None

    async def batch_get_works(self, dois: List[str]) -> List[dict]:
        """Sequential lookup with rate limiting (Crossref has no batch endpoint)."""
        results = []
        for doi in dois:
            result = await self.get_work(doi)
            if result:
                results.append(result)
        return results


# ──────────────────────────────────────────────────────────────
# ERIC Client
# ──────────────────────────────────────────────────────────────

class ERICClient:
    """Client for the ERIC API (https://api.ies.ed.gov/eric/).

    Auth: none.
    Rate: 5 req/s (conservative).
    Solr query syntax for education research.
    """

    BASE_URL = "https://api.ies.ed.gov/eric/"

    def __init__(self, cache: MetadataCache = None):
        self.cache = cache
        self.limiter = RateLimiter(rate=5.0)
        self._http = httpx.AsyncClient(timeout=20)

    async def close(self):
        """Close the underlying HTTP client."""
        await self._http.aclose()

    def _normalize_result(self, raw: dict) -> dict:
        return {
            "eric_id": raw.get("id", ""),
            "title": raw.get("title", ""),
            "author": raw.get("author", []),
            "description": raw.get("description", ""),
            "subject": raw.get("subject", []),
            "education_level": raw.get("educationLevel", []),
            "peer_reviewed": raw.get("peerreviewed", "T") == "T",
            "publication_type": raw.get("publicationType", []),
            "url": raw.get("url", ""),
            "year": self._extract_year(raw),
            "source": raw.get("source", ""),
        }

    @staticmethod
    def _extract_year(raw: dict) -> Optional[int]:
        date_str = raw.get("publicationDateYear")
        if date_str:
            try:
                return int(date_str)
            except (ValueError, TypeError):
                pass
        return None

    async def search(
        self, query: str, max_results: int = 50,
        filters: Optional[dict] = None
    ) -> List[dict]:
        """Search ERIC using Solr query syntax.

        filters: optional dict, e.g.
            {"educationlevel": "Early Childhood Education", "peerreviewed": "T"}
        """
        search_parts = [query]
        if filters:
            for k, v in filters.items():
                search_parts.append(f'{k}:"{v}"')
        search_str = " AND ".join(search_parts)

        try:
            async with self.limiter:
                params = {
                    "search": search_str,
                    "rows": min(max_results, 200),
                    "format": "json",
                }
                resp = await self._http.get(self.BASE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
                docs = data.get("response", {}).get("docs", [])
                return [self._normalize_result(d) for d in docs]
        except Exception as e:
            logger.warning(f"ERIC search failed: {e}")
            return []


# ──────────────────────────────────────────────────────────────
# Aggregator
# ──────────────────────────────────────────────────────────────

@dataclass
class EnrichedPaper:
    """Merged metadata from all API sources for a single paper."""
    doi: str = ""
    pmid: str = ""
    # OpenAlex
    openalex_id: str = ""
    cited_by_count: Optional[int] = None
    fwci: Optional[float] = None
    openalex_is_retracted: Optional[bool] = None
    openalex_funding: List[str] = field(default_factory=list)
    openalex_concepts: List[str] = field(default_factory=list)
    abstract_text: str = ""
    # Semantic Scholar
    s2_id: str = ""
    s2_citation_count: Optional[int] = None
    influential_citation_count: Optional[int] = None
    fields_of_study: List[str] = field(default_factory=list)
    tldr: str = ""
    # Crossref
    crossref_citation_count: Optional[int] = None
    crossref_funders: List[str] = field(default_factory=list)
    crossref_is_retracted: Optional[bool] = None
    crossref_is_corrected: Optional[bool] = None
    clinical_trial_numbers: List[str] = field(default_factory=list)
    # Derived
    is_retracted: bool = False
    is_corrected: bool = False
    all_funding_sources: List[str] = field(default_factory=list)
    best_citation_count: Optional[int] = None
    enrichment_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if v is not None and v != "" and v != [] and v is not False:
                d[k] = v
        # Always include retraction/correction booleans
        d["is_retracted"] = self.is_retracted
        d["is_corrected"] = self.is_corrected
        return d


async def enrich_papers_metadata(
    papers: List[dict],
    openalex_client: Optional[OpenAlexClient] = None,
    s2_client: Optional[SemanticScholarClient] = None,
    crossref_client: Optional[CrossrefClient] = None,
) -> List[EnrichedPaper]:
    """Batch-enrich papers from all three metadata APIs.

    papers: list of dicts with at least 'doi' and/or 'pmid' keys.
    Returns list of EnrichedPaper with merged data from all sources.
    """
    enriched = []

    # Index papers by doi and pmid for matching
    paper_by_doi = {}
    paper_by_pmid = {}
    for p in papers:
        doi = p.get("doi", "")
        pmid = p.get("pmid", "")
        ep = EnrichedPaper(doi=doi, pmid=pmid)
        if doi:
            paper_by_doi[doi] = ep
        if pmid:
            paper_by_pmid[pmid] = ep
        enriched.append(ep)

    # --- OpenAlex batch ---
    if openalex_client:
        dois = [p.doi for p in enriched if p.doi]
        if dois:
            try:
                oa_results = await openalex_client.batch_get_works(dois)
                oa_by_doi = {r["doi"]: r for r in oa_results if r.get("doi")}
                for ep in enriched:
                    oa = oa_by_doi.get(ep.doi)
                    if oa:
                        ep.openalex_id = oa.get("openalex_id", "")
                        ep.cited_by_count = oa.get("cited_by_count")
                        ep.fwci = oa.get("fwci")
                        ep.openalex_is_retracted = oa.get("is_retracted", False)
                        ep.openalex_funding = oa.get("funding", [])
                        ep.openalex_concepts = oa.get("concepts", [])
                        ep.abstract_text = oa.get("abstract_text", "")
                        ep.enrichment_sources.append("openalex")
            except Exception as e:
                logger.warning(f"OpenAlex batch enrichment failed: {e}")

        # Fall back to PMID lookup for papers without DOI
        for ep in enriched:
            if ep.pmid and not ep.openalex_id:
                try:
                    oa = await openalex_client.get_work_by_pmid(ep.pmid)
                    if oa:
                        ep.openalex_id = oa.get("openalex_id", "")
                        ep.cited_by_count = oa.get("cited_by_count")
                        ep.fwci = oa.get("fwci")
                        ep.openalex_is_retracted = oa.get("is_retracted", False)
                        ep.openalex_funding = oa.get("funding", [])
                        ep.openalex_concepts = oa.get("concepts", [])
                        ep.abstract_text = oa.get("abstract_text", "")
                        ep.enrichment_sources.append("openalex")
                except Exception as e:
                    logger.warning(f"OpenAlex PMID fallback failed for {ep.pmid}: {e}")

    # --- Semantic Scholar batch ---
    if s2_client:
        s2_ids = []
        for ep in enriched:
            if ep.doi:
                s2_ids.append(f"DOI:{ep.doi}")
            elif ep.pmid:
                s2_ids.append(f"PMID:{ep.pmid}")
        if s2_ids:
            try:
                s2_results = await s2_client.batch_get_papers(s2_ids)
                s2_by_doi = {r["doi"]: r for r in s2_results if r.get("doi")}
                s2_by_pmid = {r["pmid"]: r for r in s2_results if r.get("pmid")}
                for ep in enriched:
                    s2 = s2_by_doi.get(ep.doi) or s2_by_pmid.get(ep.pmid)
                    if s2:
                        ep.s2_id = s2.get("s2_id", "")
                        ep.s2_citation_count = s2.get("citation_count")
                        ep.influential_citation_count = s2.get("influential_citation_count")
                        ep.fields_of_study = s2.get("fields_of_study", [])
                        ep.tldr = s2.get("tldr", "")
                        ep.enrichment_sources.append("semantic_scholar")
            except Exception as e:
                logger.warning(f"Semantic Scholar batch enrichment failed: {e}")

    # --- Crossref ---
    if crossref_client:
        dois = [p.doi for p in enriched if p.doi]
        if dois:
            try:
                cr_results = await crossref_client.batch_get_works(dois)
                cr_by_doi = {r["doi"]: r for r in cr_results if r.get("doi")}
                for ep in enriched:
                    cr = cr_by_doi.get(ep.doi)
                    if cr:
                        ep.crossref_citation_count = cr.get("is_referenced_by_count")
                        ep.crossref_funders = cr.get("funder", [])
                        ep.crossref_is_retracted = cr.get("is_retracted", False)
                        ep.crossref_is_corrected = cr.get("is_corrected", False)
                        ep.clinical_trial_numbers = cr.get("clinical_trial_numbers", [])
                        ep.enrichment_sources.append("crossref")
            except Exception as e:
                logger.warning(f"Crossref batch enrichment failed: {e}")

    # --- Merge derived fields ---
    for ep in enriched:
        # Retraction: true if ANY source says retracted
        ep.is_retracted = bool(ep.openalex_is_retracted or ep.crossref_is_retracted)
        ep.is_corrected = bool(ep.crossref_is_corrected)

        # Merge funding from all sources
        seen = set()
        for src in ep.openalex_funding + ep.crossref_funders:
            if src and src not in seen:
                ep.all_funding_sources.append(src)
                seen.add(src)

        # Best citation count (prefer OpenAlex FWCI-aware count)
        counts = [c for c in [ep.cited_by_count, ep.s2_citation_count, ep.crossref_citation_count] if c is not None]
        ep.best_citation_count = max(counts) if counts else None

    return enriched
