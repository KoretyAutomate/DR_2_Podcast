"""
Deep Research Agent - Evidence-Based Clinical Research Pipeline

Optimized for Nvidia DGX Spark (128GB Unified Memory):
- SMART MODEL (Qwen2.5-32B-Instruct-AWQ) on port 8000: Reasoning, planning, evaluation
- FAST MODEL (Phi-4 Mini via Ollama) on port 11434: Parallel content summarization

Architecture (7-Step Clinical Pipeline — parallel a/b tracks):
  Steps 1a–5a (Affirmative) run in parallel with Steps 1b–5b (Falsification):
    Step 1: PICO/MeSH/Boolean search strategy (Smart Model)
    Step 2: Wide net — up to 500 results (PubMed + Scholar + Fast Model screening)
    Step 3: Screen → top 20 (Smart Model)
    Step 4: Deep extraction — full text retrieval + clinical variable extraction (Fast Model)
    Step 5: Case synthesis (Smart Model) — affirmative (5a) or falsification (5b)
  Step 6: Deterministic math — ARR/NNT (Python, no LLM)
  Step 7: GRADE synthesis (Smart Model)

Author: DR_2_Podcast Team
"""

import asyncio
import json
import logging
import os
import re
import sqlite3
import time
import datetime
from dotenv import load_dotenv

load_dotenv()
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import xml.etree.ElementTree as ET

import httpx
from bs4 import BeautifulSoup
from openai import AsyncOpenAI

from search_service import SearxngClient, SearchResult

logger = logging.getLogger(__name__)

# --- Configuration ---

SMART_MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-32B-Instruct-AWQ")
SMART_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")

FAST_MODEL = os.getenv("FAST_MODEL_NAME", "llama3.2:1b")
FAST_BASE_URL = os.getenv("FAST_LLM_BASE_URL", "http://localhost:11434/v1")

MAX_INPUT_TOKENS = 32000
MAX_CONCURRENT_SUMMARIES = 10
MAX_RESEARCH_ITERATIONS = 3
SCRAPING_TIMEOUT = 20.0
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
TIER_CASCADE_THRESHOLD = 50   # Tier 3 ultrawide fires if T1+T2 candidates < this
MIN_TIER3_STUDIES = 3          # Minimum Tier 3 (compound-class) studies if available
MAX_TIER3_RATIO = 0.5          # Tier 3 never exceeds 50% of max_select unless Tier 1+2 insufficient

JUNK_DOMAINS = {
    "dictionary.com", "merriam-webster.com", "thefreedictionary.com",
    "cambridge.org", "wiktionary.org", "vocabulary.com",
    "thesaurus.com", "urbandictionary.com",
    "facebook.com", "fb.com", "twitter.com", "instagram.com", "tiktok.com",
    "pinterest.com", "reddit.com", "youtube.com", "support.google.com",
    "lkong.com", "rctslabs.com",
    "starbucks.com", "amazon.com", "walmart.com",
    "dailythemedcrosswordanswers.com", "crosswordanswers.com",
}

def is_junk_url(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    return any(junk in domain for junk in JUNK_DOMAINS)


# --- URL Cache ---

CACHE_TTL_DAYS = 7


class PageCache:
    """SQLite-backed URL cache to avoid re-scraping across pipeline runs."""

    def __init__(self, db_path: str = None, ttl_days: int = CACHE_TTL_DAYS):
        if db_path is None:
            db_path = os.path.expanduser("~/.cache/dr2podcast/url_cache.db")
        self.db_path = db_path
        self.ttl_seconds = ttl_days * 86400

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS page_cache "
            "(url TEXT PRIMARY KEY, title TEXT, content TEXT, word_count INTEGER, fetched_at REAL)"
        )
        # Cleanup expired entries
        cutoff = time.time() - self.ttl_seconds
        deleted = self.conn.execute("DELETE FROM page_cache WHERE fetched_at < ?", (cutoff,)).rowcount
        self.conn.commit()
        if deleted:
            logger.info(f"PageCache: cleaned {deleted} expired entries")

    def get(self, url: str):
        """Return a FetchedPage if cached and not expired, else None."""
        cutoff = time.time() - self.ttl_seconds
        row = self.conn.execute(
            "SELECT url, title, content, word_count FROM page_cache WHERE url = ? AND fetched_at > ?",
            (url, cutoff)
        ).fetchone()
        if row:
            # Import here to avoid circular reference at class definition time
            return FetchedPage(url=row[0], title=row[1], content=row[2], word_count=row[3])
        return None

    def put(self, page) -> None:
        """Store a successfully fetched page in cache."""
        if page.error or not page.content:
            return
        self.conn.execute(
            "INSERT OR REPLACE INTO page_cache (url, title, content, word_count, fetched_at) VALUES (?, ?, ?, ?, ?)",
            (page.url, page.title, page.content, page.word_count, time.time())
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


# --- Data Models ---

@dataclass
class ResearchQuery:
    query: str
    goal: str

@dataclass
class FetchedPage:
    url: str
    title: str
    content: str
    word_count: int
    error: Optional[str] = None

@dataclass
class StudyMetadata:
    """Structured metadata extracted from a scientific source."""
    study_type: Optional[str] = None       # RCT, meta-analysis, cohort, observational, etc.
    sample_size: Optional[str] = None      # "n=1234" or None
    key_result: Optional[str] = None       # Main quantitative finding
    publication_year: Optional[int] = None
    journal_name: Optional[str] = None
    authors: Optional[str] = None          # "First Author et al."
    effect_size: Optional[str] = None      # "HR 0.82", "OR 1.5", "d=0.3"
    limitations: Optional[str] = None      # Author-stated limitations
    demographics: Optional[str] = None     # "age 25-45, 60% female, healthy adults"
    funding_source: Optional[str] = None   # "Industry-funded", "NIH grant", "Independent", etc.

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "StudyMetadata":
        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})

@dataclass
class SummarizedSource:
    url: str
    title: str
    summary: str
    query: str
    goal: str
    error: Optional[str] = None
    metadata: Optional[StudyMetadata] = None

@dataclass
class SearchMetrics:
    """PRISMA-style search flow metrics for auto-generated methodology sections."""
    search_date: str                    # ISO date
    databases_searched: List[str]       # ["PubMed", "Google Scholar", "Google", "Bing", "Brave"]
    total_identified: int               # raw results before dedup
    total_after_dedup: int              # after dedup
    total_fetched: int                  # pages fetched
    total_fetch_errors: int             # fetch failures
    total_with_content: int             # pages with extractable content
    total_summarized: int               # successfully summarized
    academic_sources: int               # pubmed + scholar count
    general_web_sources: int            # general web count
    tier1_sufficient_count: int = 0     # queries where Tier 1 was sufficient
    tier3_expanded_count: int = 0       # queries that needed Tier 3
    wide_net_total: int = 0             # total records from wide net search (Step 2)
    screened_in: int = 0                # records selected after screening (Step 3)
    fulltext_retrieved: int = 0         # full-text articles successfully retrieved (Step 4)
    fulltext_errors: int = 0            # full-text retrieval failures

@dataclass
class ResearchReport:
    topic: str
    role: str
    sources: List[SummarizedSource]
    report: str
    iterations_used: int
    total_urls_fetched: int
    total_summaries: int
    total_errors: int
    duration_seconds: float
    search_metrics: Optional[SearchMetrics] = None


# --- New Pipeline Data Models ---

@dataclass
class SearchStrategy:
    """PICO framework + MeSH terms + Boolean search strings from Step 1."""
    pico: Dict[str, str]                    # P, I, C, O
    mesh_terms: Dict[str, List[str]]        # population/intervention/outcome MeSH
    search_strings: Dict[str, str]          # pubmed_primary, pubmed_broad, cochrane, scholar
    role: str                               # "affirmative" or "adversarial"

@dataclass
class TierKeywords:
    """Plain keyword lists for one search tier — NO Boolean/MeSH syntax."""
    intervention: List[str]   # exact terms for the intervention at this tier
    outcome: List[str]        # outcome terms at this tier
    population: List[str]     # population terms
    rationale: str            # scientist's justification for this tier's scope

@dataclass
class TieredSearchPlan:
    """Three-tier keyword plan produced by the scientist and approved by the Auditor."""
    pico: Dict[str, str]       # P, I, C, O — used downstream in _build_case, screening
    tier1: TierKeywords        # Exact folk/named terms → "Established evidence"
    tier2: TierKeywords        # Canonical scientific synonyms, same substance → "Supporting evidence"
    tier3: TierKeywords        # Active compound class / mechanism → "Speculative extrapolation"
    role: str                  # "affirmative" | "adversarial"
    auditor_approved: bool = False
    auditor_notes: str = ""
    revision_count: int = 0

@dataclass
class WideNetRecord:
    """Lightweight screening record — no full text, just title + abstract metadata."""
    pmid: Optional[str]
    doi: Optional[str]
    title: str
    abstract: str
    study_type: str
    sample_size: Optional[str]
    primary_objective: Optional[str]
    year: Optional[int]
    journal: Optional[str]
    authors: Optional[str]
    url: str
    source_db: str                          # "pubmed", "cochrane_central", "scholar"
    research_tier: Optional[int] = None    # 1=exact folk  2=scientific synonyms  3=compound class
    relevance_score: Optional[float] = None

@dataclass
class DeepExtraction:
    """Clinical variable extraction from full-text articles (Step 4)."""
    pmid: Optional[str]
    doi: Optional[str]
    title: str
    url: str
    attrition_pct: Optional[str] = None
    effect_size: Optional[str] = None
    demographics: Optional[str] = None
    follow_up_period: Optional[str] = None
    funding_source: Optional[str] = None
    conflicts_of_interest: Optional[str] = None
    biological_mechanism: Optional[str] = None
    control_event_rate: Optional[float] = None      # CER — needed for Step 7
    experimental_event_rate: Optional[float] = None  # EER — needed for Step 7
    primary_outcome: Optional[str] = None
    secondary_outcomes: Optional[List[str]] = None
    blinding: Optional[str] = None
    randomization_method: Optional[str] = None
    intention_to_treat: Optional[bool] = None
    sample_size_total: Optional[int] = None
    sample_size_intervention: Optional[int] = None
    sample_size_control: Optional[int] = None
    study_design: Optional[str] = None
    risk_of_bias: Optional[str] = None
    raw_facts: str = ""

    def to_dict(self) -> dict:
        d = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None and v != "" and v != []:
                d[f.name] = v
        return d


# --- Worker Services (IO + Fast Model) ---

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MIN_ACADEMIC_RESULTS = 5  # Sufficiency threshold for Tier 1


class PubMedClient:
    """Search PubMed via NCBI E-utilities (free, no API key needed for <3 req/sec).

    Enhanced for clinical pipeline: supports Boolean/MeSH queries, retmax up to 500,
    and extracts PublicationType, DOI, MeSH headings, and structured abstracts.
    """

    def __init__(self):
        self.api_key = os.getenv("PUBMED_API_KEY")

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """Legacy search — returns simple dicts for backward compatibility."""
        records = await self.search_extended(query, max_results=max_results)
        return [
            {"url": r["url"], "title": r.get("title", ""), "snippet": r.get("abstract", "")[:500]}
            for r in records
        ]

    async def search_extended(self, query: str, max_results: int = 500,
                               sort: str = "relevance") -> List[Dict[str, Any]]:
        """Enhanced search returning rich article records with metadata.

        Returns list of dicts with: pmid, doi, title, abstract (full), study_type,
        publication_types, mesh_headings, journal, authors, year, url, abstract_sections.
        """
        results = []
        try:
            async with httpx.AsyncClient(timeout=30) as http:
                # Step 1: esearch to get PMIDs
                params = {
                    "db": "pubmed",
                    "term": query,
                    "retmax": min(max_results, 500),
                    "retmode": "json",
                    "sort": sort,
                }
                if self.api_key:
                    params["api_key"] = self.api_key

                resp = await http.get(f"{PUBMED_BASE_URL}/esearch.fcgi", params=params)
                resp.raise_for_status()
                search_result = resp.json().get("esearchresult", {})
                id_list = search_result.get("idlist", [])
                if not id_list:
                    return []

                logger.info(f"PubMed esearch returned {len(id_list)} IDs for query: {query[:80]}")

                # Step 2: efetch in batches of 200 (NCBI recommended max)
                for batch_start in range(0, len(id_list), 200):
                    batch_ids = id_list[batch_start:batch_start + 200]
                    fetch_params = {
                        "db": "pubmed",
                        "id": ",".join(batch_ids),
                        "retmode": "xml"
                    }
                    if self.api_key:
                        fetch_params["api_key"] = self.api_key

                    resp = await http.get(f"{PUBMED_BASE_URL}/efetch.fcgi", params=fetch_params)
                    resp.raise_for_status()

                    results.extend(self._parse_articles_xml(resp.text))

                    # Rate limiting: 0.4s delay between batches (3 req/sec without API key)
                    if not self.api_key and batch_start + 200 < len(id_list):
                        await asyncio.sleep(0.4)

        except Exception as e:
            logger.warning(f"PubMed search failed: {e}")
        return results

    def _parse_articles_xml(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse PubMed efetch XML into rich article records."""
        results = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.warning(f"PubMed XML parse error: {e}")
            return []

        for article in root.findall(".//PubmedArticle"):
            try:
                record = self._parse_single_article(article)
                if record:
                    results.append(record)
            except Exception as e:
                logger.debug(f"Failed to parse article: {e}")
        return results

    def _parse_single_article(self, article) -> Optional[Dict[str, Any]]:
        """Parse a single PubmedArticle XML element."""
        pmid_el = article.find(".//PMID")
        if pmid_el is None:
            return None
        pmid = pmid_el.text

        # Title
        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()) if title_el is not None else ""

        # Abstract — handle structured abstracts (multiple AbstractText elements with labels)
        abstract_parts = []
        abstract_sections = {}
        for abs_el in article.findall(".//AbstractText"):
            label = abs_el.get("Label", "")
            text = "".join(abs_el.itertext()).strip()
            if label:
                abstract_sections[label] = text
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # DOI
        doi = None
        for id_el in article.findall(".//ArticleId"):
            if id_el.get("IdType") == "doi":
                doi = id_el.text
                break
        # Also check ELocationID
        if not doi:
            for eloc in article.findall(".//ELocationID"):
                if eloc.get("EIdType") == "doi":
                    doi = eloc.text
                    break

        # Publication types
        pub_types = []
        for pt in article.findall(".//PublicationType"):
            if pt.text:
                pub_types.append(pt.text)

        # Derive study_type from PublicationType (no LLM needed)
        study_type = self._classify_study_type(pub_types)

        # MeSH headings
        mesh_headings = []
        for mh in article.findall(".//MeshHeading/DescriptorName"):
            if mh.text:
                mesh_headings.append(mh.text)

        # Journal
        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else None

        # Year
        year = None
        year_el = article.find(".//PubDate/Year")
        if year_el is not None and year_el.text:
            try:
                year = int(year_el.text)
            except ValueError:
                pass
        if not year:
            medline_year = article.find(".//MedlineDate")
            if medline_year is not None and medline_year.text:
                import re as _re
                m = _re.search(r"(\d{4})", medline_year.text)
                if m:
                    year = int(m.group(1))

        # Authors
        author_list = article.findall(".//Author")
        authors = None
        if author_list:
            first = author_list[0]
            last_name = first.findtext("LastName", "")
            if last_name:
                authors = f"{last_name} et al." if len(author_list) > 1 else last_name

        return {
            "pmid": pmid,
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "study_type": study_type,
            "publication_types": pub_types,
            "mesh_headings": mesh_headings,
            "journal": journal,
            "authors": authors,
            "year": year,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "abstract_sections": abstract_sections,
        }

    @staticmethod
    def _classify_study_type(pub_types: List[str]) -> str:
        """Classify study type from PubMed PublicationType elements (no LLM)."""
        pt_lower = [p.lower() for p in pub_types]
        if any("meta-analysis" in p for p in pt_lower):
            return "meta-analysis"
        if any("systematic review" in p for p in pt_lower):
            return "systematic-review"
        if any("randomized controlled trial" in p for p in pt_lower):
            return "RCT"
        if any("clinical trial" in p for p in pt_lower):
            return "clinical-trial"
        if any("observational study" in p for p in pt_lower):
            return "observational"
        if any("cohort" in p for p in pt_lower):
            return "cohort"
        if any("case report" in p for p in pt_lower):
            return "case-report"
        if any("review" in p for p in pt_lower):
            return "review"
        if any("guideline" in p or "practice guideline" in p for p in pt_lower):
            return "guideline"
        if any("retracted publication" in p for p in pt_lower):
            return "retracted"
        return "other"


def _dedup_and_filter(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Deduplicate results by URL and filter junk domains."""
    seen = set()
    unique = []
    for r in results:
        url = r["url"]
        if url not in seen and not is_junk_url(url):
            seen.add(url)
            unique.append(r)
    return unique


class SearchService:
    """Tiered search: Academic sources first (PubMed + Google Scholar), then general web."""

    def __init__(self, brave_api_key: str = ""):
        self.brave_api_key = brave_api_key
        self.pubmed = PubMedClient()
        # Tier tracking counters for SearchMetrics
        self.tier1_sufficient = 0
        self.tier3_expanded = 0
        self.academic_count = 0
        self.general_count = 0
        self.total_identified_raw = 0

    async def _extract_searxng_results(self, raw: list) -> List[Dict[str, str]]:
        """Extract url/title/snippet from SearXNG raw results."""
        results = []
        for r in raw:
            url = r.get("url", "") if isinstance(r, dict) else getattr(r, "url", "")
            title = r.get("title", "") if isinstance(r, dict) else getattr(r, "title", "")
            snippet = r.get("content", "") if isinstance(r, dict) else getattr(r, "snippet", "")
            if url:
                results.append({"url": url, "title": title, "snippet": snippet})
        return results

    async def search(self, query: str, max_results: int = 10, min_academic: int = MIN_ACADEMIC_RESULTS) -> List[Dict[str, str]]:
        academic_results = []

        # Tier 1a: PubMed
        pubmed_results = await self.pubmed.search(query, max_results=max_results)
        academic_results.extend(pubmed_results)

        # Tier 1b: Google Scholar via SearXNG
        try:
            async with SearxngClient() as client:
                if await client.validate_connection():
                    raw = await client.search(query, engines=['google scholar'], num_results=max_results)
                    academic_results.extend(await self._extract_searxng_results(raw))
        except Exception as e:
            logger.warning(f"Google Scholar search failed: {e}")

        self.total_identified_raw += len(academic_results)
        academic_results = _dedup_and_filter(academic_results)

        # Tier 2: Sufficiency check
        if len(academic_results) >= min_academic:
            logger.info(f"[Tier 1: Academic] {len(academic_results)} results — sufficient, skipping general web")
            self.tier1_sufficient += 1
            self.academic_count += len(academic_results[:max_results])
            return academic_results[:max_results]

        logger.info(f"[Tier 3: General web] expanding search — only {len(academic_results)} academic results")
        self.tier3_expanded += 1

        # Tier 3: General web (existing behavior)
        general_results = []
        try:
            async with SearxngClient() as client:
                if await client.validate_connection():
                    raw = await client.search(query, engines=['google', 'bing', 'brave'], num_results=max_results)
                    general_results.extend(await self._extract_searxng_results(raw))
        except Exception as e:
            logger.warning(f"SearXNG general search failed: {e}")

        if self.brave_api_key and len(general_results) < max_results:
            try:
                headers = {"X-Subscription-Token": self.brave_api_key, "Accept": "application/json"}
                async with httpx.AsyncClient(timeout=15) as http:
                    resp = await http.get(
                        "https://api.search.brave.com/res/v1/web/search",
                        params={"q": query, "count": min(max_results, 20)},
                        headers=headers
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        for r in data.get("web", {}).get("results", []):
                            general_results.append({
                                "url": r.get("url", ""),
                                "title": r.get("title", ""),
                                "snippet": r.get("description", "")
                            })
            except Exception as e:
                logger.warning(f"BraveSearch failed: {e}")

        # Merge: academic first (prioritized), then general
        self.total_identified_raw += len(general_results)
        all_results = academic_results + general_results
        deduped = _dedup_and_filter(all_results)[:max_results]
        self.academic_count += len(academic_results)
        self.general_count += len(deduped) - min(len(academic_results), len(deduped))
        return deduped


class ContentFetcher:
    """Async parallel content fetcher."""

    def __init__(self, max_concurrent: int = 10, cache: PageCache = None):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.cache = cache

    async def fetch_page(self, url: str) -> FetchedPage:
        async with self.semaphore:
            # Check cache first
            if self.cache:
                cached = self.cache.get(url)
                if cached is not None:
                    logger.debug(f"Cache hit: {url}")
                    return cached
            try:
                headers = {"User-Agent": USER_AGENT}
                async with httpx.AsyncClient(
                    timeout=SCRAPING_TIMEOUT, follow_redirects=True, headers=headers
                ) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.text, "lxml")
                    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
                        tag.decompose()
                    content_el = (
                        soup.find("main") or soup.find("article") or
                        soup.find("div", class_=re.compile(r"content|main-content|post-content|article")) or
                        soup.find("body")
                    )
                    text = content_el.get_text(separator=" ", strip=True) if content_el else ""
                    max_chars = MAX_INPUT_TOKENS * 4
                    if len(text) > max_chars:
                        text = text[:max_chars] + "..."
                    title = soup.title.string.strip() if soup.title and soup.title.string else ""
                    page = FetchedPage(url=url, title=title, content=text, word_count=len(text.split()))
                    if self.cache:
                        self.cache.put(page)
                    return page
            except httpx.HTTPStatusError as e:
                return FetchedPage(url=url, title="", content="", word_count=0, error=f"HTTP {e.response.status_code}")
            except Exception as e:
                return FetchedPage(url=url, title="", content="", word_count=0, error=str(e)[:200])

    async def fetch_all(self, urls: List[str]) -> List[FetchedPage]:
        return await asyncio.gather(*[self.fetch_page(url) for url in urls])


class FastWorker:
    """Uses fast model for parallel content summarization."""

    def __init__(self, client: AsyncOpenAI, model: str = FAST_MODEL):
        self.client = client
        self.model = model
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_SUMMARIES)

    def _parse_metadata_from_response(self, raw_text: str) -> Tuple[str, Optional[StudyMetadata]]:
        """Parse FACTS and METADATA sections from fast model response.

        Returns (facts_text, metadata_or_none). On parse failure, returns
        original text with None metadata (graceful fallback).
        """
        # Split on METADATA: marker
        marker = "METADATA:"
        marker_idx = raw_text.find(marker)
        if marker_idx == -1:
            return raw_text.strip(), None

        facts_text = raw_text[:marker_idx].strip()
        # Remove "FACTS:" prefix if present
        if facts_text.upper().startswith("FACTS:"):
            facts_text = facts_text[6:].strip()

        json_part = raw_text[marker_idx + len(marker):].strip()

        # Extract JSON using brace-depth tracking
        brace_start = json_part.find("{")
        if brace_start == -1:
            return facts_text, None

        depth = 0
        brace_end = -1
        for i in range(brace_start, len(json_part)):
            if json_part[i] == "{":
                depth += 1
            elif json_part[i] == "}":
                depth -= 1
                if depth == 0:
                    brace_end = i
                    break

        if brace_end == -1:
            return facts_text, None

        json_str = json_part[brace_start:brace_end + 1]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.debug(f"Failed to parse metadata JSON: {json_str[:200]}")
            return facts_text, None

        # Convert "null" strings and null values to None
        cleaned = {}
        for k, v in data.items():
            if v is None or v == "null" or v == "":
                continue
            cleaned[k] = v

        if not cleaned:
            return facts_text, None

        try:
            metadata = StudyMetadata.from_dict(cleaned)
            return facts_text, metadata
        except Exception:
            return facts_text, None

    async def summarize(self, page: FetchedPage, goal: str, query: str) -> SummarizedSource:
        if page.error or not page.content.strip():
            return SummarizedSource(
                url=page.url, title=page.title, summary="",
                query=query, goal=goal, error=page.error or "Empty content"
            )
        content = page.content[:MAX_INPUT_TOKENS * 4]
        system_prompt = (
            f"You are a precise scientific data extractor. Extract facts relevant to: '{goal}'.\n\n"
            f"OUTPUT FORMAT (follow exactly):\n"
            f"FACTS:\n"
            f"- [fact 1]\n"
            f"- [fact 2]\n"
            f"...\n\n"
            f"METADATA:\n"
            f'{{"study_type":"RCT|meta-analysis|cohort|observational|animal_model|review|mechanism|guideline|general",'
            f'"sample_size":"n=X or null",'
            f'"key_result":"main quantitative finding or null",'
            f'"publication_year":YYYY or null,'
            f'"journal_name":"journal name or null",'
            f'"authors":"First Author et al. or null",'
            f'"effect_size":"HR/OR/d value or null",'
            f'"limitations":"key limitation or null",'
            f'"demographics":"age range, sex ratio, population description or null",'
            f'"funding_source":"Industry/Government/Independent/Unknown or null"}}\n\n'
            f"Rules:\n"
            f"- Be extremely concise in facts\n"
            f"- Use null (not quotes) for unknown metadata fields\n"
            f"- If no relevant information: output 'NO RELEVANT DATA' with no metadata"
        )
        async with self.semaphore:
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Source URL: {page.url}\n\nContent:\n{content}"}
                    ],
                    max_tokens=1536, temperature=0.1, timeout=180
                )
                raw_text = resp.choices[0].message.content.strip()
                facts_text, metadata = self._parse_metadata_from_response(raw_text)
                return SummarizedSource(
                    url=page.url, title=page.title, summary=facts_text,
                    query=query, goal=goal, metadata=metadata
                )
            except Exception as e:
                logger.warning(f"Fast model failed for {page.url}: {str(e)[:100]}")
                return SummarizedSource(url=page.url, title=page.title, summary="", query=query, goal=goal, error=str(e)[:200])

    async def summarize_batch(self, pages: List[FetchedPage], goal: str, query: str) -> List[SummarizedSource]:
        return await asyncio.gather(*[self.summarize(page, goal, query) for page in pages])


# --- Smart Model: The Researcher Agent ---

class ResearchAgent:
    """
    A smart-model-driven researcher that iteratively delegates to workers.

    The agent:
    1. Plans what to search (based on its role and what's missing)
    2. Delegates search + summarization to SearchService + FastWorker
    3. Evaluates gathered evidence
    4. Identifies gaps and generates new queries
    5. Repeats until satisfied or max iterations reached
    6. Writes a final report
    """

    def __init__(
        self,
        smart_client: AsyncOpenAI,
        fast_worker: Optional[FastWorker],
        search_service: SearchService,
        fetcher: ContentFetcher,
        smart_model: str = SMART_MODEL,
        results_per_query: int = 5,
        max_iterations: int = MAX_RESEARCH_ITERATIONS
    ):
        self.smart_client = smart_client
        self.smart_model = smart_model
        self.fast_worker = fast_worker
        self.search = search_service
        self.fetcher = fetcher
        self.results_per_query = results_per_query
        self.max_iterations = max_iterations

    async def _call_smart(self, system: str, user: str, max_tokens: int = 2048, temperature: float = 0.3) -> str:
        """Call the smart model."""
        resp = await self.smart_client.chat.completions.create(
            model=self.smart_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens, temperature=temperature, timeout=300
        )
        return resp.choices[0].message.content.strip()

    def _parse_json_queries(self, raw: str) -> List[ResearchQuery]:
        """Parse JSON query list from smart model output."""
        if "```" in raw:
            match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
            if match:
                raw = match.group(1).strip()
        try:
            plans = json.loads(raw)
            return [ResearchQuery(query=p["query"], goal=p["goal"]) for p in plans]
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning(f"Failed to parse queries JSON, raw: {raw[:300]}")
            return []

    def _format_evidence_so_far(self, summaries: List[SummarizedSource]) -> str:
        """Format collected evidence for the smart model to review."""
        good = [s for s in summaries if s.summary and s.summary != "NO RELEVANT DATA" and not s.error]
        if not good:
            return "No evidence collected yet."
        blocks = []
        for s in good:
            blocks.append(f"- [{s.title or 'Untitled'}]({s.url}): {s.summary[:300]}")
        return "\n".join(blocks)

    async def _search_and_summarize(
        self, queries: List[ResearchQuery], seen_urls: set, log
    ) -> Tuple[List[SummarizedSource], int, int]:
        """Execute search + fetch + summarize for a batch of queries."""
        all_summaries = []
        total_fetched = 0
        total_errors = 0

        for rq in queries:
            # Search
            results = await self.search.search(rq.query, max_results=self.results_per_query)
            urls = [r["url"] for r in results if r["url"] not in seen_urls]
            for u in urls:
                seen_urls.add(u)

            if not urls:
                log(f"      [{rq.goal[:40]}] No new URLs")
                continue

            log(f"      [{rq.goal[:40]}] {len(urls)} URLs → fetching...")

            # Fetch
            pages = await self.fetcher.fetch_all(urls)
            good_pages = [p for p in pages if not p.error and p.content.strip()]
            total_fetched += len(pages)
            total_errors += sum(1 for p in pages if p.error)

            if not good_pages:
                log(f"      [{rq.goal[:40]}] No pages fetched")
                continue

            log(f"      [{rq.goal[:40]}] {len(good_pages)}/{len(pages)} fetched → summarizing...")
            
            # Log sources for UI visualization
            for p in good_pages:
                log(f"[SOURCE] {p.url}")

            # Summarize with fast model
            if self.fast_worker:
                batch = await self.fast_worker.summarize_batch(good_pages, rq.goal, rq.query)
            else:
                # Fallback to smart model
                batch = []
                for p in good_pages:
                    content = p.content[:MAX_INPUT_TOKENS * 4]
                    try:
                        summary = await self._call_smart(
                            f"Extract facts relevant to: '{rq.goal}'. Bulleted list only. Be concise.",
                            f"Source: {p.url}\n\n{content}",
                            max_tokens=1024, temperature=0.1
                        )
                        batch.append(SummarizedSource(url=p.url, title=p.title, summary=summary, query=rq.query, goal=rq.goal))
                    except Exception as e:
                        batch.append(SummarizedSource(url=p.url, title=p.title, summary="", query=rq.query, goal=rq.goal, error=str(e)[:200]))

            good = sum(1 for s in batch if s.summary and not s.error)
            log(f"      [{rq.goal[:40]}] {good}/{len(good_pages)} summarized")
            all_summaries.extend(batch)

        return all_summaries, total_fetched, total_errors

    async def research(self, topic: str, role: str, role_instructions: str, log=print) -> ResearchReport:
        """
        Run iterative research as the given role.

        Args:
            topic: Research topic
            role: Role name (e.g. "Lead Researcher", "Counter Researcher")
            role_instructions: Specific instructions for this role
            log: Logging callback
        """
        start_time = time.time()
        all_summaries: List[SummarizedSource] = []
        seen_urls: set = set()
        total_fetched = 0
        total_errors = 0

        log(f"\n  {'─'*60}")
        log(f"  {role.upper()}: Starting iterative research")
        log(f"  Topic: {topic}")
        log(f"  Max iterations: {self.max_iterations}")
        log(f"  {'─'*60}")

        for iteration in range(self.max_iterations):
            log(f"\n  ── Iteration {iteration + 1}/{self.max_iterations} ──")

            # Step 1: Smart model plans what to search
            if iteration == 0:
                plan_prompt = (
                    f"You are a {role}. {role_instructions}\n\n"
                    f"Topic: {topic}\n\n"
                    f"Generate 5-7 specific search queries to begin your research.\n"
                    f"Return ONLY a JSON array: [{{\"query\": \"...\", \"goal\": \"...\"}}]"
                )
            else:
                evidence_summary = self._format_evidence_so_far(all_summaries)
                plan_prompt = (
                    f"You are a {role}. {role_instructions}\n\n"
                    f"Topic: {topic}\n\n"
                    f"Evidence gathered so far ({len([s for s in all_summaries if s.summary and not s.error])} sources):\n"
                    f"{evidence_summary}\n\n"
                    f"Based on what you have, identify 3-5 specific GAPS in your evidence.\n"
                    f"Generate NEW targeted search queries to fill those gaps.\n"
                    f"If evidence is sufficient, return an empty array: []\n\n"
                    f"Return ONLY a JSON array: [{{\"query\": \"...\", \"goal\": \"...\"}}]"
                )

            log(f"    Planning: asking smart model for queries...")
            raw_plan = await self._call_smart(
                "You are a research planning expert. Return ONLY valid JSON arrays.",
                plan_prompt, max_tokens=2048, temperature=0.3
            )
            queries = self._parse_json_queries(raw_plan)

            if not queries:
                log(f"    Smart model returned no new queries — evidence deemed sufficient")
                break

            log(f"    Plan: {len(queries)} queries")
            for i, q in enumerate(queries, 1):
                log(f"      {i}. [{q.goal[:50]}] {q.query}")

            # Step 2: Delegate search + summarization to workers
            log(f"    Delegating to search + summarize workers...")
            batch_summaries, batch_fetched, batch_errors = await self._search_and_summarize(
                queries, seen_urls, log
            )
            all_summaries.extend(batch_summaries)
            total_fetched += batch_fetched
            total_errors += batch_errors

            good_count = len([s for s in all_summaries if s.summary and not s.error])
            log(f"    Iteration {iteration + 1} complete: {good_count} total good sources")

        # Step 3: Smart model writes final report
        good_summaries = [s for s in all_summaries if s.summary and s.summary != "NO RELEVANT DATA" and not s.error]
        log(f"\n  Writing final report from {len(good_summaries)} sources...")

        evidence_blocks = []
        for s in good_summaries:
            meta_line = ""
            if s.metadata:
                m = s.metadata
                parts = []
                if m.study_type:
                    parts.append(f"Type: {m.study_type}")
                if m.sample_size:
                    parts.append(f"N: {m.sample_size}")
                if m.journal_name:
                    parts.append(f"Journal: {m.journal_name}")
                if m.publication_year:
                    parts.append(f"Year: {m.publication_year}")
                if m.effect_size:
                    parts.append(f"Effect: {m.effect_size}")
                if m.authors:
                    parts.append(f"Authors: {m.authors}")
                if m.demographics:
                    parts.append(f"Pop: {m.demographics}")
                if m.funding_source:
                    parts.append(f"Funding: {m.funding_source}")
                if parts:
                    meta_line = f"**Study Metadata:** {' | '.join(parts)}\n"
            evidence_blocks.append(
                f"### Source: {s.title or s.url}\n"
                f"**URL:** {s.url}\n"
                f"**Research Goal:** {s.goal}\n"
                f"{meta_line}"
                f"**Extracted Facts:**\n{s.summary}\n"
            )
        aggregated = "\n---\n".join(evidence_blocks) if evidence_blocks else "No evidence gathered."
        if len(aggregated) > 80000:
            aggregated = aggregated[:80000] + "\n\n[...truncated...]"

        report_system = (
            f"You are a {role}. {role_instructions}\n\n"
            f"Write a comprehensive research report based ONLY on the evidence provided.\n"
            f"Each source includes structured metadata (study type, sample size, effect size, journal, funding).\n"
            f"Weight evidence accordingly: meta-analyses > RCTs > cohort > observational > reviews > opinion.\n\n"
            f"Structure:\n"
            f"1. Abstract (3-4 sentence summary of the overall findings)\n"
            f"2. Key Findings (grouped by evidence tier):\n"
            f"   - Meta-Analyses and Systematic Reviews\n"
            f"   - Randomized Controlled Trials\n"
            f"   - Cohort and Observational Studies\n"
            f"   - Reviews and Expert Opinion\n"
            f"3. Evidence Table (clean markdown table):\n"
            f"   | Author (Year) | Study Type | N | Key Finding | Effect Size | Funding | Journal |\n"
            f"   | --- | --- | --- | --- | --- | --- | --- |\n"
            f"   [Fill from source metadata — one row per source]\n"
            f"4. Limitations\n"
            f"5. References (standardized format: 'Author et al. (Year). Title. Journal. URL')\n\n"
            f"Citation rules:\n"
            f"- In body text, cite as 'Author et al. (Year)' when metadata is available\n"
            f"- Fall back to (URL) only when no author/year metadata exists\n"
            f"- Report specific sample sizes and effect sizes when available"
        )

        try:
            report_text = await self._call_smart(
                report_system,
                f"Topic: {topic}\nSources: {len(good_summaries)}\n\nEVIDENCE:\n\n{aggregated}",
                max_tokens=8000, temperature=0.2
            )
        except Exception as e:
            logger.error(f"Report synthesis failed: {e}")
            report_text = f"# {role} Report: {topic}\n\n*Synthesis failed ({e}). Raw evidence below.*\n\n{aggregated}"

        duration = time.time() - start_time
        log(f"  {role} complete: {len(good_summaries)} sources, {duration:.0f}s")

        # Build search metrics from SearchService counters
        svc = self.search
        metrics = SearchMetrics(
            search_date=datetime.date.today().isoformat(),
            databases_searched=["PubMed", "Google Scholar", "Google", "Bing", "Brave"],
            total_identified=svc.total_identified_raw,
            total_after_dedup=len(seen_urls),
            total_fetched=total_fetched,
            total_fetch_errors=total_errors,
            total_with_content=total_fetched - total_errors,
            total_summarized=len(good_summaries),
            academic_sources=svc.academic_count,
            general_web_sources=svc.general_count,
            tier1_sufficient_count=svc.tier1_sufficient,
            tier3_expanded_count=svc.tier3_expanded,
        )

        return ResearchReport(
            topic=topic,
            role=role,
            sources=all_summaries,
            report=report_text,
            iterations_used=min(iteration + 1, self.max_iterations),
            total_urls_fetched=total_fetched,
            total_summaries=len(good_summaries),
            total_errors=total_errors,
            duration_seconds=duration,
            search_metrics=metrics,
        )

    # --- New Clinical Pipeline Methods (Steps 1-6) ---

    def _parse_json_response(self, raw: str) -> Any:
        """Parse JSON from smart model output, handling code blocks."""
        if "```" in raw:
            match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
            if match:
                raw = match.group(1).strip()
        return json.loads(raw)

    async def _decompose_topic(self, topic: str, framing_context: str = "") -> dict:
        """Pre-PICO: fast model extracts canonical scientific terms from folk-language topic."""
        system = (
            "You are a biomedical terminology specialist. Given a research topic (possibly in "
            "colloquial form), extract canonical scientific terms used in PubMed/MeSH searches.\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "canonical_terms": ["term1", ...],\n'
            '  "related_concepts": ["concept1", ...],\n'
            '  "population_terms": ["term1", ...]\n'
            "}\n"
            "canonical_terms: 4-8 scientific synonyms for the intervention/exposure.\n"
            "related_concepts: 3-5 related research areas that may have evidence.\n"
            "population_terms: 2-4 population descriptors.\n"
            "Use only real MeSH-compatible scientific terminology."
        )
        context = f"\n\nRESEARCH FRAMING CONTEXT:\n{framing_context[:2000]}" if framing_context else ""
        user = f"Research topic: {topic}{context}"
        try:
            # Use fast model if available, else fall back to smart
            if self.fast_worker:
                resp = await self.fast_worker.client.chat.completions.create(
                    model=self.fast_worker.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    max_tokens=512, temperature=0.2, timeout=60
                )
                raw = resp.choices[0].message.content.strip()
            else:
                raw = await self._call_smart(system, user, max_tokens=512, temperature=0.2)
            data = self._parse_json_response(raw)
            return {
                "canonical_terms": data.get("canonical_terms", []),
                "related_concepts": data.get("related_concepts", []),
                "population_terms": data.get("population_terms", []),
            }
        except Exception as e:
            logger.warning(f"Topic decomposition failed: {e}")
            return {"canonical_terms": [], "related_concepts": [], "population_terms": []}

    # ------------------------------------------------------------------ #
    #  STEP 1 — Tiered keyword generation + Auditor gate                  #
    # ------------------------------------------------------------------ #

    async def _generate_tiered_keywords(
        self, topic: str, role: str, framing_context: str,
        decomposition: Optional[Dict], auditor_feedback: str = "",
        log=print
    ) -> TieredSearchPlan:
        """Generate three-tier keyword plan as plain lists — NO Boolean/MeSH syntax."""
        log(f"    [Step 1] Generating tiered keywords ({role})...")

        adversarial_note = ""
        if role == "adversarial":
            adversarial_note = (
                "\n\nADVERSARIAL ROLE: Each tier must surface CONTRADICTING or NULL evidence.\n"
                "Target: null results, adverse effects, dose-response harms, tolerance, "
                "withdrawal, conflicting findings, methodological concerns.\n"
                "Same tier-boundary rules apply: Tier 1 = exact substance + exact harm outcome, "
                "Tier 2 = same substance, broader harm synonyms, Tier 3 = compound class + mechanism harms."
            )

        framing_note = f"\n\nRESEARCH FRAMING:\n{framing_context[:3000]}" if framing_context else ""

        decomp_note = ""
        if decomposition and any(decomposition.values()):
            canonical = ", ".join(decomposition.get("canonical_terms", []))
            related = ", ".join(decomposition.get("related_concepts", []))
            decomp_note = (
                f"\n\nCANONICAL SCIENTIFIC TERMS: {canonical}"
                f"\nRELATED CONCEPTS: {related}"
            )

        revision_note = ""
        if auditor_feedback:
            revision_note = (
                f"\n\n⚠ AUDITOR REJECTED YOUR PREVIOUS KEYWORDS with this feedback:\n"
                f"{auditor_feedback}\n"
                "You MUST revise your keywords to address this feedback."
            )

        system = (
            "You are a systematic review scientist generating search keyword tiers.\n\n"
            "TASK: Produce three keyword tiers for a cascading PubMed search.\n"
            "Each tier is a set of PLAIN KEYWORD LISTS — no Boolean operators, no MeSH notation, "
            "no brackets, no field tags. Just simple English phrases.\n\n"
            "TIER DEFINITIONS:\n\n"
            "TIER 1 — 'Established evidence' (strictest):\n"
            "  Intervention: exact folk/common names for *this specific substance* only.\n"
            "  Example for coffee: [\"coffee\", \"coffee drinking\", \"coffee consumption\"]\n"
            "  Do NOT include caffeine — caffeine also comes from tea, energy drinks, etc.\n"
            "  Outcome: direct primary outcome labels as they appear in clinical trial titles.\n"
            "  Example: [\"work productivity\", \"job performance\", \"occupational performance\"]\n"
            "  Population: specific population relevant to the research question.\n"
            "  Example: [\"working adults\", \"employees\"]\n\n"
            "TIER 2 — 'Supporting evidence' (broadened scope, same substance):\n"
            "  Intervention: <<INHERITED FROM TIER 1 — do not generate, will be copied automatically>>\n"
            "  Outcome: SUPERSET of Tier 1 outcomes. Include ALL Tier 1 outcome terms PLUS broader\n"
            "    proxy outcomes and related clinical endpoints. Must be strictly broader, never narrower.\n"
            "  Example: [\"work productivity\", \"job performance\", \"cognitive performance\", "
            "\"alertness\", \"executive function\", \"mental performance\"]\n"
            "  Population: BROADER than Tier 1. Widen to more general populations.\n"
            "  Example: [\"adults\", \"healthy adults\"]\n\n"
            "TIER 3 — 'Speculative extrapolation' (compound class):\n"
            "  Intervention: active compound class / mechanism (source ambiguity accepted).\n"
            "  Example: [\"caffeine\", \"methylxanthine\", \"adenosine antagonist\", \"caffeinated beverage\"]\n"
            "  These results require inference (e.g., caffeine from any source → coffee effect) and "
            "will be flagged as speculative in the output.\n"
            "  Outcome: <<INHERITED FROM TIER 2 — do not generate, will be copied automatically>>\n"
            "  Population: <<INHERITED FROM TIER 2 — do not generate, will be copied automatically>>\n\n"
            "RULES:\n"
            "- Keyword lists must contain plain phrases only — no AND, OR, NOT, [MeSH], [tiab], etc.\n"
            "- Each term should be 1-4 words max.\n"
            "- Tier 1 intervention must NOT contain compound-class terms.\n"
            "- Tier 2 outcome MUST include ALL Tier 1 outcome terms plus additional broader terms.\n"
            "- Also produce a PICO summary.\n\n"
            "Return ONLY valid JSON (note: Tier 2 has no intervention, Tier 3 has no outcome/population):\n"
            "{\n"
            '  "pico": {"population": "...", "intervention": "...", "comparison": "...", "outcome": "..."},\n'
            '  "tier1": {\n'
            '    "intervention": ["term1", "term2"],\n'
            '    "outcome": ["term1", "term2"],\n'
            '    "population": ["term1", "term2"],\n'
            '    "rationale": "Why these exact terms belong at Tier 1"\n'
            '  },\n'
            '  "tier2": {"outcome": ["all tier1 outcomes + broader terms"], "population": ["broader"], "rationale": "..."},\n'
            '  "tier3": {"intervention": ["compound class terms"], "rationale": "..."}\n'
            "}"
            f"{adversarial_note}{framing_note}{decomp_note}{revision_note}"
        )

        user = f"Research topic: {topic}"

        raw = await self._call_smart(system, user, max_tokens=2048, temperature=0.2)
        try:
            data = self._parse_json_response(raw)

            def parse_tier(d: dict) -> TierKeywords:
                return TierKeywords(
                    intervention=d.get("intervention", []),
                    outcome=d.get("outcome", []),
                    population=d.get("population", []),
                    rationale=d.get("rationale", ""),
                )

            plan = TieredSearchPlan(
                pico=data.get("pico", {}),
                tier1=parse_tier(data.get("tier1", {})),
                tier2=parse_tier(data.get("tier2", {})),
                tier3=parse_tier(data.get("tier3", {})),
                role=role,
            )

            # --- Deterministic tier inheritance ---
            # Tier 2 inherits intervention from Tier 1
            plan.tier2.intervention = list(plan.tier1.intervention)
            # Tier 3 inherits outcome and population from Tier 2
            plan.tier3.outcome = list(plan.tier2.outcome)
            plan.tier3.population = list(plan.tier2.population)

            log(f"    [Step 1] Tier 1 intervention: {plan.tier1.intervention[:3]}")
            log(f"    [Step 1] Tier 2 intervention (inherited T1): {plan.tier2.intervention[:3]}")
            log(f"    [Step 1] Tier 3 intervention: {plan.tier3.intervention[:3]}")
            log(f"    [Step 1] Tier 2 outcome (superset of T1): {plan.tier2.outcome[:5]}")
            log(f"    [Step 1] Tier 3 outcome (inherited T2): {plan.tier3.outcome[:5]}")
            return plan
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Tier keyword generation parse failed: {e} — using fallback")
            fallback_terms = [w for w in topic.split() if len(w) > 3][:4]
            return TieredSearchPlan(
                pico={"population": "general", "intervention": topic,
                      "comparison": "control", "outcome": "primary outcome"},
                tier1=TierKeywords(intervention=fallback_terms, outcome=[], population=[], rationale="fallback"),
                tier2=TierKeywords(intervention=fallback_terms, outcome=[], population=[], rationale="fallback"),
                tier3=TierKeywords(intervention=fallback_terms, outcome=[], population=[], rationale="fallback"),
                role=role,
            )

    async def _audit_tier_plan(
        self, plan: TieredSearchPlan, topic: str, log=print
    ) -> tuple:
        """Auditor reviews all three tiers in one call. Returns (approved: bool, notes: str)."""
        log(f"    [Auditor] Reviewing tier keyword plan...")

        system = (
            "You are The Auditor — a systematic review methodologist.\n\n"
            "Review the three-tier keyword plan below for this research topic.\n"
            "NOTE: Tier 2 intervention is inherited from Tier 1, and Tier 3 outcome/population "
            "are inherited from Tier 2 — these are enforced by code. Focus your review on "
            "the LLM-generated fields.\n\n"
            "For each tier, check ALL of the following:\n\n"
            "1. INTERVENTION ANCHOR — Tier 1 intervention must NOT include compound-class terms.\n"
            "   Tier 2 intervention is inherited from Tier 1 (enforced by code, no need to check).\n"
            "   Tier 3 intervention must be compound class / mechanism, one step removed.\n\n"
            "2. OUTCOME BROADENING — Tier 2 outcome terms MUST be a strict superset of Tier 1.\n"
            "   They must include ALL Tier 1 outcome terms plus additional broader proxies.\n"
            "   Tier 2 must NEVER be narrower than Tier 1.\n"
            "   Tier 3 outcome is inherited from Tier 2 (enforced by code, no need to check).\n\n"
            "3. POPULATION BROADENING — Tier 2 population must be BROADER than Tier 1.\n"
            "   If Tier 1 has 'working adults', Tier 2 should have 'adults' or 'healthy adults'.\n"
            "   Tier 3 population is inherited from Tier 2 (enforced by code, no need to check).\n\n"
            "4. NO BOOLEAN SYNTAX — keyword lists must contain plain phrases only "
            "(no AND, OR, NOT, [MeSH], [tiab], parentheses, or other operators).\n\n"
            "5. COVERAGE — Tier 1 needs >=2 intervention + >=2 outcome terms.\n"
            "   Tier 2 needs >=2 outcome terms (more than Tier 1) + >=2 population terms.\n"
            "   Tier 3 needs >=2 intervention terms.\n\n"
            "Return ONLY valid JSON:\n"
            '{"approved": true/false, "tier1_ok": true/false, "tier2_ok": true/false, '
            '"tier3_ok": true/false, "notes": "Specific actionable feedback — name which tier '
            'failed and exactly what to change. Empty string if approved."}'
        )

        user = (
            f"Research topic: {topic}\n\n"
            f"PICO: {json.dumps(plan.pico)}\n\n"
            f"Tier 1 (Established evidence):\n"
            f"  Intervention: {plan.tier1.intervention}\n"
            f"  Outcome: {plan.tier1.outcome}\n"
            f"  Population: {plan.tier1.population}\n"
            f"  Rationale: {plan.tier1.rationale}\n\n"
            f"Tier 2 (Supporting evidence — intervention inherited from Tier 1):\n"
            f"  Intervention: {plan.tier2.intervention}  [INHERITED from Tier 1]\n"
            f"  Outcome: {plan.tier2.outcome}\n"
            f"  Population: {plan.tier2.population}\n"
            f"  Rationale: {plan.tier2.rationale}\n\n"
            f"Tier 3 (Speculative extrapolation — outcome/population inherited from Tier 2):\n"
            f"  Intervention: {plan.tier3.intervention}\n"
            f"  Outcome: {plan.tier3.outcome}  [INHERITED from Tier 2]\n"
            f"  Population: {plan.tier3.population}  [INHERITED from Tier 2]\n"
            f"  Rationale: {plan.tier3.rationale}"
        )

        raw = await self._call_smart(system, user, max_tokens=1024, temperature=0.1)
        try:
            data = self._parse_json_response(raw)
            approved = bool(data.get("approved", False))
            notes = data.get("notes", "")
            if approved:
                log(f"    [Auditor] APPROVED")
            else:
                log(f"    [Auditor] REJECTED — {notes[:200]}")
            return approved, notes
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Audit response parse failed: {e} — auto-approving to unblock pipeline")
            return True, "Auto-approved (parse error)"

    async def _formulate_tiered_strategy(
        self, topic: str, role: str, framing_context: str,
        decomposition: Optional[Dict], log=print
    ) -> TieredSearchPlan:
        """Step 1: Scientist generates tier keywords, Auditor reviews; loop until approved."""
        MAX_REVISIONS = 2
        feedback = ""

        for attempt in range(MAX_REVISIONS + 1):
            log(f"\n    [Step 1] Generating tier keywords (attempt {attempt + 1}/{MAX_REVISIONS + 1})...")
            plan = await self._generate_tiered_keywords(
                topic, role, framing_context, decomposition,
                auditor_feedback=feedback, log=log
            )
            approved, feedback = await self._audit_tier_plan(plan, topic, log=log)
            plan.revision_count = attempt

            if approved:
                plan.auditor_approved = True
                plan.auditor_notes = feedback
                return plan

        # Max revisions exhausted — warn and proceed
        logger.warning(
            f"Tier plan not approved after {MAX_REVISIONS} revisions "
            f"({role}) — proceeding with last draft. Notes: {feedback[:200]}"
        )
        log(f"    [Auditor] WARNING: proceeding with unapproved plan after {MAX_REVISIONS} revisions")
        plan.auditor_approved = False
        plan.auditor_notes = f"Not approved after {MAX_REVISIONS} revisions: {feedback}"
        return plan

    def _build_tier_query(self, tier: TierKeywords, extra_filters: str = "") -> str:
        """Deterministic PubMed Boolean builder — no LLM. AND between groups, OR within groups."""
        def group(terms: List[str]) -> str:
            quoted = [f'"{t}"[Title/Abstract]' for t in terms if t.strip()]
            return "(" + " OR ".join(quoted) + ")" if quoted else ""

        parts = [g for g in [
            group(tier.intervention),
            group(tier.outcome),
            group(tier.population),
        ] if g]

        query = " AND ".join(parts)
        if extra_filters and query:
            query += f" AND {extra_filters}"
        return query

    # ------------------------------------------------------------------ #
    #  STEP 2 — Tiered cascade search                                      #
    # ------------------------------------------------------------------ #

    async def _tiered_search(
        self, plan: TieredSearchPlan, log=print
    ) -> tuple:
        """Step 2: Run tiered PubMed cascade. Stop when pool >= TIER_THRESHOLD.

        Returns:
            (List[WideNetRecord], int) — records and highest tier reached.
        """
        log(f"    [Step 2] Tiered cascade search ({plan.role})...")
        all_records: List[WideNetRecord] = []
        seen_pmids: set = set()
        seen_urls: set = set()
        highest_tier = 0

        pubmed = self.search.pubmed

        tier_configs = [
            (1, plan.tier1, "Humans[MeSH] AND English[la]"),
            (2, plan.tier2, "Humans[MeSH]"),
            (3, plan.tier3, ""),
        ]

        for tier_num, tier_kw, filters in tier_configs:
            if not tier_kw.intervention:
                log(f"    [Tier {tier_num}] No intervention keywords — skipping")
                continue

            query = self._build_tier_query(tier_kw, filters)
            if not query:
                log(f"    [Tier {tier_num}] Empty query — skipping")
                continue

            log(f"    [Tier {tier_num}] Query: {query[:140]}...")
            try:
                articles = await pubmed.search_extended(query, max_results=200)
                added = 0
                for art in articles:
                    pmid = art.get("pmid", "")
                    url = art.get("url", "")
                    if pmid and pmid in seen_pmids:
                        continue
                    if url and url in seen_urls:
                        continue
                    if pmid:
                        seen_pmids.add(pmid)
                    if url:
                        seen_urls.add(url)
                    all_records.append(WideNetRecord(
                        pmid=pmid,
                        doi=art.get("doi"),
                        title=art.get("title", ""),
                        abstract=art.get("abstract", ""),
                        study_type=art.get("study_type", "other"),
                        sample_size=None,
                        primary_objective=None,
                        year=art.get("year"),
                        journal=art.get("journal"),
                        authors=art.get("authors"),
                        url=url,
                        source_db="pubmed",
                        research_tier=tier_num,
                    ))
                    added += 1
                log(f"    [Tier {tier_num}] +{added} new records (pool: {len(all_records)})")
            except Exception as e:
                logger.warning(f"Tier {tier_num} PubMed search failed: {e}")

            highest_tier = tier_num

            if len(all_records) >= TIER_CASCADE_THRESHOLD:
                log(f"    [Tier {tier_num}] Threshold ({TIER_CASCADE_THRESHOLD}) reached — stopping cascade")
                break

        # Scholar search — Tier 1 plain-text keywords always
        scholar_query = " ".join(plan.tier1.intervention + plan.tier1.outcome)
        if scholar_query.strip():
            try:
                async with SearxngClient() as client:
                    if await client.validate_connection():
                        raw = await client.search(scholar_query, engines=['google scholar'], num_results=100)
                        scholar_added = 0
                        for r in raw:
                            url = r.get("url", "") if isinstance(r, dict) else getattr(r, "url", "")
                            if not url or url in seen_urls or is_junk_url(url):
                                continue
                            seen_urls.add(url)
                            title = r.get("title", "") if isinstance(r, dict) else getattr(r, "title", "")
                            snippet = r.get("content", "") if isinstance(r, dict) else getattr(r, "snippet", "")
                            all_records.append(WideNetRecord(
                                pmid=None, doi=None,
                                title=title, abstract=snippet,
                                study_type="other",
                                sample_size=None, primary_objective=None,
                                year=None, journal=None, authors=None,
                                url=url, source_db="scholar",
                                research_tier=1,
                            ))
                            scholar_added += 1
                        log(f"    [Scholar] +{scholar_added} records (Tier 1 keywords)")
            except Exception as e:
                logger.warning(f"Google Scholar search failed: {e}")

        log(f"    [Step 2] Total pool: {len(all_records)} records (highest tier: {highest_tier})")

        # Fast-model screening for study_type / sample_size on "other" records
        needs_screening = [r for r in all_records if r.study_type == "other" and r.abstract]
        if needs_screening and self.fast_worker:
            log(f"    [Step 2] Fast-model typing {len(needs_screening)} abstracts...")
            screened = await self._fast_screen_abstracts(needs_screening)
            screening_map = {id(r): s for r, s in zip(needs_screening, screened)}
            for r in all_records:
                if id(r) in screening_map:
                    s = screening_map[id(r)]
                    if s.get("study_type"):
                        r.study_type = s["study_type"]
                    if s.get("sample_size"):
                        r.sample_size = s["sample_size"]
                    if s.get("primary_objective"):
                        r.primary_objective = s["primary_objective"]

        return all_records[:500], highest_tier


    async def _fast_screen_abstracts(self, records: List[WideNetRecord]) -> List[Dict]:
        """Use fast model to extract study_type, sample_size, primary_objective from abstracts."""
        semaphore = asyncio.Semaphore(10)

        async def screen_one(record: WideNetRecord) -> Dict:
            async with semaphore:
                try:
                    resp = await self.fast_worker.client.chat.completions.create(
                        model=self.fast_worker.model,
                        messages=[
                            {"role": "system", "content": (
                                "Extract from this abstract:\n"
                                '- study_type: RCT | meta-analysis | systematic-review | cohort | '
                                'case-control | cross-sectional | case-report | in-vitro | animal-model | '
                                'review | guideline | other\n'
                                '- sample_size: "n=X" or null\n'
                                '- primary_objective: one sentence or null\n'
                                'Return JSON only: {"study_type":"...","sample_size":"...","primary_objective":"..."}'
                            )},
                            {"role": "user", "content": f"Title: {record.title}\n\nAbstract: {record.abstract[:2000]}"}
                        ],
                        max_tokens=256, temperature=0.1, timeout=60
                    )
                    raw = resp.choices[0].message.content.strip()
                    # Parse JSON from response
                    if "```" in raw:
                        match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
                        if match:
                            raw = match.group(1).strip()
                    return json.loads(raw)
                except Exception:
                    return {}

        results = await asyncio.gather(*[screen_one(r) for r in records])
        return list(results)

    async def _screen_and_prioritize(
        self, records: List[WideNetRecord], strategy: TieredSearchPlan,
        max_select: int = 20, topic: str = "", log=print
    ) -> List[WideNetRecord]:
        """Step 3: Smart model screens wide net records → top 20 with tier-aware priority."""
        if not records:
            log(f"    [Step 3] No records to screen")
            return []

        pico_str = json.dumps(strategy.pico)

        # Group records by tier
        tier_groups: Dict[int, List[WideNetRecord]] = {1: [], 2: [], 3: []}
        for r in records:
            tier = r.research_tier if r.research_tier in (1, 2, 3) else 3
            tier_groups[tier].append(r)

        log(f"    [Step 3] Pool by tier: T1={len(tier_groups[1])}, T2={len(tier_groups[2])}, T3={len(tier_groups[3])}")

        # Screen each tier independently with tier-appropriate intervention
        screened: Dict[int, List[WideNetRecord]] = {}
        for tier_num in [1, 2, 3]:
            tier_records = tier_groups[tier_num]
            if not tier_records:
                screened[tier_num] = []
                continue
            # Use tier-appropriate intervention for relevance gate
            if tier_num <= 2:
                tier_intervention = ", ".join(strategy.tier1.intervention)
            else:
                tier_intervention = ", ".join(strategy.tier3.intervention)
            screened[tier_num] = await self._screen_chunk(
                tier_records, 0, pico_str, max_select, topic, log,
                intervention_override=tier_intervention,
            )
            log(f"    [Step 3] Tier {tier_num}: {len(screened[tier_num])} passed screening")

        # Priority fill: Tier 1 → Tier 2 → Tier 3 (with cap)
        t3_available = len(screened[3])
        min_t3 = min(MIN_TIER3_STUDIES, t3_available)
        tier3_cap = int(max_select * MAX_TIER3_RATIO)
        tier12_budget = max_select - min_t3

        selected: List[WideNetRecord] = list(screened[1][:tier12_budget])
        remaining12 = tier12_budget - len(selected)
        if remaining12 > 0:
            selected.extend(screened[2][:remaining12])

        remaining = max_select - len(selected)
        if len(selected) >= tier12_budget:
            tier3_slots = min_t3
        elif len(selected) + tier3_cap >= max_select:
            tier3_slots = min(remaining, tier3_cap)
        else:
            tier3_slots = remaining

        selected.extend(screened[3][:tier3_slots])

        log(f"    [Step 3] Final selection: T1={sum(1 for s in selected if s.research_tier==1)}, "
            f"T2={sum(1 for s in selected if s.research_tier==2)}, "
            f"T3={sum(1 for s in selected if s.research_tier==3)}, total={len(selected)}")

        return selected

    async def _screen_chunk(
        self, records: List[WideNetRecord], offset: int,
        pico_str: str, max_select: int, topic: str, log,
        intervention_override: str = ""
    ) -> List[WideNetRecord]:
        """Screen a chunk of records with the smart model."""
        compact = []
        for i, r in enumerate(records):
            compact.append({
                "idx": offset + i,
                "title": r.title[:150],
                "type": r.study_type,
                "n": r.sample_size,
                "year": r.year,
                "journal": r.journal,
                "abstract": r.abstract[:300],
            })

        # Extract intervention text for relevance gate
        if intervention_override:
            intervention_text = intervention_override
        else:
            try:
                pico_data = json.loads(pico_str)
                intervention_text = pico_data.get("intervention", "the PICO intervention")
            except (json.JSONDecodeError, AttributeError):
                intervention_text = "the PICO intervention"

        topic_line = f"RESEARCH TOPIC: {topic}\n" if topic else ""

        system = (
            "You are a systematic review screener performing title/abstract screening.\n\n"
            f"{topic_line}"
            f"PICO: {pico_str}\n\n"
            "SCREENING IS A TWO-STAGE PROCESS. You MUST apply both stages in order.\n\n"
            "═══ STAGE 1: RELEVANCE GATE (mandatory, apply first) ═══\n"
            f"Does the study directly investigate {intervention_text}?\n"
            "A study MUST explicitly examine, measure, or review the PICO intervention to pass.\n"
            "Studies about DIFFERENT interventions (e.g., exercise, other drugs, supplements, "
            "devices, or procedures unrelated to the PICO intervention) MUST be EXCLUDED — "
            "regardless of how methodologically rigorous they are.\n"
            "If a study does not pass the relevance gate, do NOT select it.\n\n"
            "═══ STAGE 2: RIGOR RANKING (among relevant studies only) ═══\n"
            "From the studies that PASSED Stage 1, apply these criteria:\n"
            "INCLUSION CRITERIA:\n"
            "- Human clinical studies (RCTs, meta-analyses, systematic reviews, large cohort studies)\n"
            "- Sample size >= 30 participants (prefer >= 100)\n"
            "- Published in peer-reviewed journals\n\n"
            "EXCLUSION CRITERIA:\n"
            "- Animal models / in vitro studies\n"
            "- Case reports (n < 5)\n"
            "- Conference abstracts without full data\n"
            "- Retracted publications\n"
            "- Duplicate reports of the same study\n\n"
            f"From the relevant studies, select the TOP {max_select} most rigorous.\n"
            "Rank by: meta-analyses first, then RCTs (by sample size), then large cohort studies.\n\n"
            "Return ONLY a JSON array of selected indices:\n"
            '[{"index": 0, "reason": "Meta-analysis of 45 RCTs, n=12,000, directly studies [intervention]"}, ...]'
        )

        user = json.dumps(compact, ensure_ascii=False)

        raw = await self._call_smart(system, user, max_tokens=2048, temperature=0.1)
        try:
            selections = self._parse_json_response(raw)
            selected_indices = set()
            for s in selections[:max_select]:
                idx = s.get("index", s.get("idx", -1))
                local_idx = idx - offset
                if 0 <= local_idx < len(records):
                    selected_indices.add(local_idx)

            result = [records[i] for i in sorted(selected_indices)]
            log(f"    [Step 3] Selected {len(result)} from {len(records)} records")
            return result
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Screening parse failed: {e}, returning top records by study type")
            # Fallback: prioritize by study type
            priority = {"meta-analysis": 0, "systematic-review": 1, "RCT": 2,
                        "clinical-trial": 3, "cohort": 4, "observational": 5}
            sorted_records = sorted(records, key=lambda r: priority.get(r.study_type, 99))
            return sorted_records[:max_select]

    async def _deep_extract_batch(
        self, articles, records: List[WideNetRecord], pico: Dict[str, str], log=print
    ) -> List[DeepExtraction]:
        """Step 4: Extract clinical variables from full-text articles using fast model."""
        log(f"    [Step 4] Deep extraction from {len(articles)} articles...")
        semaphore = asyncio.Semaphore(10)

        async def extract_one(article, record: WideNetRecord) -> DeepExtraction:
            text = getattr(article, 'full_text', '') or record.abstract
            if not text.strip():
                return DeepExtraction(
                    pmid=record.pmid, doi=record.doi, title=record.title,
                    url=record.url, raw_facts="No content available"
                )

            async with semaphore:
                try:
                    content = text[:MAX_INPUT_TOKENS * 4]
                    system_prompt = (
                        "You are a clinical data extraction specialist. Read this study and extract "
                        "ALL of the following variables. Use null for any field not found.\n\n"
                        "Return ONLY valid JSON:\n"
                        "{\n"
                        '  "attrition_pct": "exact dropout percentage or null",\n'
                        '  "effect_size": "primary effect with CI (e.g. HR 0.76, 95% CI 0.65-0.89) or null",\n'
                        '  "demographics": "age range, sex ratio, population or null",\n'
                        '  "follow_up_period": "duration (e.g. 5.2 years median) or null",\n'
                        '  "funding_source": "exact funding source or null",\n'
                        '  "conflicts_of_interest": "declared COI or None declared or null",\n'
                        '  "biological_mechanism": "mechanism/pathway or null",\n'
                        '  "control_event_rate": 0.15,\n'
                        '  "experimental_event_rate": 0.10,\n'
                        '  "primary_outcome": "exact primary endpoint or null",\n'
                        '  "secondary_outcomes": ["endpoint1", "endpoint2"],\n'
                        '  "blinding": "double-blind | single-blind | open-label | null",\n'
                        '  "randomization_method": "method or null",\n'
                        '  "intention_to_treat": true,\n'
                        '  "sample_size_total": 1000,\n'
                        '  "sample_size_intervention": 500,\n'
                        '  "sample_size_control": 500,\n'
                        '  "study_design": "parallel RCT | crossover RCT | meta-analysis | cohort | etc.",\n'
                        '  "risk_of_bias": "low | some concerns | high | unclear",\n'
                        '  "raw_facts": "3-5 key findings as bullet points"\n'
                        "}"
                    )

                    if self.fast_worker:
                        resp = await self.fast_worker.client.chat.completions.create(
                            model=self.fast_worker.model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": f"Title: {record.title}\n\nContent:\n{content}"}
                            ],
                            max_tokens=1536, temperature=0.1, timeout=180
                        )
                    else:
                        resp = await self.smart_client.chat.completions.create(
                            model=self.smart_model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": f"Title: {record.title}\n\nContent:\n{content}"}
                            ],
                            max_tokens=1536, temperature=0.1, timeout=300
                        )

                    raw = resp.choices[0].message.content.strip()
                    data = self._parse_json_response(raw)

                    # Safely parse numeric fields
                    def safe_float(v):
                        if v is None or v == "null":
                            return None
                        try:
                            return float(v)
                        except (ValueError, TypeError):
                            return None

                    def safe_int(v):
                        if v is None or v == "null":
                            return None
                        try:
                            return int(v)
                        except (ValueError, TypeError):
                            return None

                    def safe_bool(v):
                        if v is None or v == "null":
                            return None
                        if isinstance(v, bool):
                            return v
                        return None

                    def safe_str(v):
                        if v is None or v == "null" or v == "":
                            return None
                        return str(v)

                    def safe_list(v):
                        if isinstance(v, list):
                            return v
                        return None

                    return DeepExtraction(
                        pmid=record.pmid,
                        doi=record.doi,
                        title=record.title,
                        url=record.url,
                        attrition_pct=safe_str(data.get("attrition_pct")),
                        effect_size=safe_str(data.get("effect_size")),
                        demographics=safe_str(data.get("demographics")),
                        follow_up_period=safe_str(data.get("follow_up_period")),
                        funding_source=safe_str(data.get("funding_source")),
                        conflicts_of_interest=safe_str(data.get("conflicts_of_interest")),
                        biological_mechanism=safe_str(data.get("biological_mechanism")),
                        control_event_rate=safe_float(data.get("control_event_rate")),
                        experimental_event_rate=safe_float(data.get("experimental_event_rate")),
                        primary_outcome=safe_str(data.get("primary_outcome")),
                        secondary_outcomes=safe_list(data.get("secondary_outcomes")),
                        blinding=safe_str(data.get("blinding")),
                        randomization_method=safe_str(data.get("randomization_method")),
                        intention_to_treat=safe_bool(data.get("intention_to_treat")),
                        sample_size_total=safe_int(data.get("sample_size_total")),
                        sample_size_intervention=safe_int(data.get("sample_size_intervention")),
                        sample_size_control=safe_int(data.get("sample_size_control")),
                        study_design=safe_str(data.get("study_design")),
                        risk_of_bias=safe_str(data.get("risk_of_bias")),
                        raw_facts=safe_str(data.get("raw_facts")) or "",
                    )
                except Exception as e:
                    logger.warning(f"Deep extraction failed for {record.title[:50]}: {e}")
                    return DeepExtraction(
                        pmid=record.pmid, doi=record.doi, title=record.title,
                        url=record.url, raw_facts=f"Extraction failed: {str(e)[:100]}"
                    )

        # Log sources for UI visualization
        for record in records:
            log(f"[SOURCE] {record.url}")

        results = await asyncio.gather(*[
            extract_one(art, rec) for art, rec in zip(articles, records)
        ])
        good = sum(1 for r in results if r.raw_facts and "failed" not in r.raw_facts.lower())
        log(f"    [Step 4] Extracted data from {good}/{len(results)} articles")
        return list(results)

    async def _build_case(
        self, topic: str, strategy: TieredSearchPlan,
        extractions: List[DeepExtraction], case_type: str, log=print
    ) -> str:
        """Step 5/6: Smart model builds affirmative or falsification case from extraction data."""
        log(f"    [Step {'5' if case_type == 'affirmative' else '6'}] Building {case_type} case...")

        pico_str = json.dumps(strategy.pico)

        # Build extraction data for prompt
        extraction_blocks = []
        for i, ex in enumerate(extractions, 1):
            block = f"Study {i}: {ex.title}\n"
            if ex.study_design:
                block += f"  Design: {ex.study_design}\n"
            if ex.sample_size_total:
                block += f"  N: {ex.sample_size_total}\n"
            if ex.effect_size:
                block += f"  Effect: {ex.effect_size}\n"
            if ex.control_event_rate is not None:
                block += f"  CER: {ex.control_event_rate}\n"
            if ex.experimental_event_rate is not None:
                block += f"  EER: {ex.experimental_event_rate}\n"
            if ex.demographics:
                block += f"  Demographics: {ex.demographics}\n"
            if ex.follow_up_period:
                block += f"  Follow-up: {ex.follow_up_period}\n"
            if ex.blinding:
                block += f"  Blinding: {ex.blinding}\n"
            if ex.risk_of_bias:
                block += f"  Risk of bias: {ex.risk_of_bias}\n"
            if ex.funding_source:
                block += f"  Funding: {ex.funding_source}\n"
            if ex.raw_facts:
                block += f"  Key findings: {ex.raw_facts}\n"
            extraction_blocks.append(block)

        extractions_text = "\n".join(extraction_blocks)
        if len(extractions_text) > 60000:
            extractions_text = extractions_text[:60000] + "\n[...truncated...]"

        if case_type == "affirmative":
            system = (
                "You are a Lead Researcher writing the AFFIRMATIVE case for the following hypothesis.\n\n"
                f"PICO: {pico_str}\n\n"
                f"You have deeply extracted data from {len(extractions)} clinical studies. "
                "Analyze this evidence and write a comprehensive argument FOR the hypothesis.\n\n"
                "Structure:\n"
                "1. Clinical Significance: How large are the observed effects? Clinically meaningful?\n"
                "2. Biological Plausibility: What mechanisms support efficacy?\n"
                "3. Consistency: Do multiple independent studies converge?\n"
                "4. Dose-Response: Evidence of a dose-response relationship?\n"
                "5. Strength of Evidence: Rate as STRONG / MODERATE / WEAK / INSUFFICIENT\n"
                "6. Evidence Table:\n"
                "   | Study | Design | N | Effect Size | CER | EER | Follow-up | Bias Risk |\n"
                "7. Key Supporting Citations (Author et al. (Year) format)\n\n"
                "Be precise. Cite specific numbers. Do not speculate beyond the data."
            )
        else:
            system = (
                "You are an Adversarial Researcher writing the FALSIFICATION case against the following hypothesis.\n\n"
                f"PICO: {pico_str}\n\n"
                "Your mandate: Find every reason this intervention may NOT work, may cause harm, or may be overstated.\n\n"
                "Structure:\n"
                "1. Adverse Effects: What harms have been documented?\n"
                "2. Null Results: Which studies found NO significant effect?\n"
                "3. Methodological Concerns: Poor blinding, high attrition, small samples, short follow-up\n"
                "4. Funding Bias: Industry-funded studies vs. independent results\n"
                "5. Publication Bias: Evidence of selective reporting or p-hacking\n"
                "6. Biological Implausibility: Any mechanistic concerns?\n"
                "7. Evidence Table (same format as affirmative)\n"
                "8. Strength of Counter-Evidence: STRONG / MODERATE / WEAK / INSUFFICIENT"
            )

        try:
            report = await self._call_smart(
                system,
                f"Topic: {topic}\n\nEXTRACTED STUDY DATA:\n\n{extractions_text}",
                max_tokens=6000, temperature=0.2
            )
            has_synthetic, flagged = _detect_synthetic_citations(report, extractions)
            if has_synthetic:
                warning = (
                    "\n\n---\n"
                    "⚠ **SYNTHETIC CITATION WARNING** — the following references were cited but "
                    f"not found in the {len(extractions)} retrieved studies. "
                    "They may be hallucinated:\n"
                    + "\n".join(f"- {r}" for r in flagged)
                    + "\n---\n"
                )
                report = warning + report
                if not extractions:
                    logger.critical(f"SYNTHETIC CITATIONS DETECTED (0 studies input): {flagged}")
            return report
        except Exception as e:
            logger.error(f"Build case ({case_type}) failed: {e}")
            return f"# {case_type.title()} Case: {topic}\n\n*Case synthesis failed ({e}).*\n\n{extractions_text}"


_CITATION_RE = re.compile(r'\b([A-Z][a-z]+(?:\s+et\s+al\.)?)\s+\((\d{4})\)')


def _detect_synthetic_citations(
    report: str, extractions: list
) -> tuple:
    """Cross-check cited references against retrieved study corpus.

    Returns:
        (has_synthetic: bool, flagged: List[str])
    """
    found = _CITATION_RE.findall(report)
    if not found:
        return False, []
    if not extractions:
        # No studies were input → any citation is hallucinated
        return True, [f"{a} ({y})" for a, y in found]
    # Build corpus of known author/title text from extractions
    known = " ".join(
        f"{ex.title or ''} {ex.raw_facts or ''}" for ex in extractions
    ).lower()
    flagged = [
        f"{a} ({y})" for a, y in found
        if a.replace("et al.", "").strip().lower() not in known
    ]
    return bool(flagged), flagged


# --- Orchestrator: Full Pipeline ---

class Orchestrator:
    """
    Runs the full DR_2_Podcast evidence-based clinical research pipeline.

    7-Step Pipeline (affirmative + falsification tracks run in parallel):
    Steps 1a–5a: Affirmative track (PICO → wide net → screen → extract → case)
    Steps 1b–5b: Falsification track (adversarial PICO → same pipeline → case)
    Step 6: Deterministic math (ARR/NNT from Python, no LLM)
    Step 7: GRADE synthesis (Smart Model)
    """

    def __init__(
        self,
        smart_base_url: str = SMART_BASE_URL,
        fast_base_url: str = FAST_BASE_URL,
        smart_model: str = SMART_MODEL,
        fast_model: str = FAST_MODEL,
        brave_api_key: str = "",
        results_per_query: int = 5,
        max_iterations: int = MAX_RESEARCH_ITERATIONS,
        fast_model_available: bool = True
    ):
        self.smart_client = AsyncOpenAI(base_url=smart_base_url, api_key="NA")
        self.fast_client = AsyncOpenAI(base_url=fast_base_url, api_key="NA") if fast_model_available else None
        self.smart_model = smart_model

        fast_worker = FastWorker(self.fast_client, fast_model) if fast_model_available else None
        search_svc = SearchService(brave_api_key)
        self._page_cache = PageCache()
        fetcher = ContentFetcher(max_concurrent=15, cache=self._page_cache)

        self.lead_researcher = ResearchAgent(
            self.smart_client, fast_worker, search_svc, fetcher,
            smart_model, results_per_query, max_iterations
        )
        self.counter_researcher = ResearchAgent(
            self.smart_client, fast_worker, search_svc, fetcher,
            smart_model, results_per_query, max_iterations
        )
        self.fast_model_available = fast_model_available

        # Full-text fetcher for Step 4
        from fulltext_fetcher import FullTextFetcher
        self.fulltext_fetcher = FullTextFetcher(max_concurrent=5, cache=self._page_cache)

    async def run(self, topic: str, framing_context: str = "", progress_callback=None,
                  output_dir: str = None) -> Dict[str, ResearchReport]:
        """Run the full 7-step clinical research pipeline.

        Args:
            topic: Research topic
            framing_context: Optional research framing document to guide searches
            progress_callback: Optional callback for progress messages
            output_dir: Optional directory to save intermediate artifacts

        Returns:
            Dict[str, ResearchReport] with keys: "lead", "counter", "audit"
        """
        import clinical_math

        start_time = time.time()

        def log(msg: str):
            logger.info(msg)
            print(msg)
            if progress_callback:
                progress_callback(msg)

        mode = "DUAL-MODEL" if self.fast_model_available else "SINGLE-MODEL"
        log(f"\n{'='*70}")
        log(f"DEEP RESEARCH AGENT - Evidence-Based Clinical Pipeline ({mode})")
        log(f"{'='*70}")
        log(f"Topic: {topic}")
        if framing_context:
            log(f"Research framing provided: {len(framing_context)} chars")
        log(f"{'='*70}")

        # Counters for metrics
        aff_wide_net_total = 0
        aff_screened_in = 0
        aff_fulltext_ok = 0
        aff_fulltext_err = 0
        fal_wide_net_total = 0
        fal_screened_in = 0
        fal_fulltext_ok = 0
        fal_fulltext_err = 0

        # --- Phase 0: Concept Decomposition (C2) ---
        log(f"\n{'='*70}")
        log(f"PHASE 0: CONCEPT DECOMPOSITION")
        log(f"{'='*70}")
        decomposition = await self.lead_researcher._decompose_topic(topic, framing_context)
        if decomposition.get("canonical_terms"):
            log(f"  Canonical terms: {', '.join(decomposition['canonical_terms'])}")
        if decomposition.get("related_concepts"):
            log(f"  Related concepts: {', '.join(decomposition['related_concepts'])}")

        # --- Affirmative Track (Steps 1-5) ---
        async def affirmative_track():
            nonlocal aff_wide_net_total, aff_screened_in, aff_fulltext_ok, aff_fulltext_err

            log(f"\n{'='*70}")
            log(f"STEP 1a: TIERED KEYWORD GENERATION + AUDITOR GATE (Affirmative)")
            log(f"{'='*70}")
            plan = await self.lead_researcher._formulate_tiered_strategy(
                topic, "affirmative", framing_context, decomposition, log=log
            )

            log(f"\n{'='*70}")
            log(f"STEP 2a: TIERED CASCADE SEARCH (Affirmative)")
            log(f"{'='*70}")
            records, aff_highest_tier = await self.lead_researcher._tiered_search(plan, log)
            aff_wide_net_total = len(records)

            log(f"\n{'='*70}")
            log(f"STEP 3a: SCREENING ({len(records)} → top 20) (Affirmative)")
            log(f"{'='*70}")
            top_records = await self.lead_researcher._screen_and_prioritize(
                records, plan, topic=topic, log=log
            )
            aff_screened_in = len(top_records)

            log(f"\n{'='*70}")
            log(f"STEP 4a: DEEP EXTRACTION ({len(top_records)} articles) (Affirmative)")
            log(f"{'='*70}")
            fulltexts = await self.fulltext_fetcher.fetch_all(top_records)
            aff_fulltext_ok = sum(1 for ft in fulltexts if not ft.error)
            aff_fulltext_err = sum(1 for ft in fulltexts if ft.error)
            log(f"    Full-text retrieved: {aff_fulltext_ok}/{len(fulltexts)}")

            extractions = await self.lead_researcher._deep_extract_batch(
                fulltexts, top_records, plan.pico, log
            )

            log(f"\n{'='*70}")
            log(f"STEP 5a: AFFIRMATIVE CASE")
            log(f"{'='*70}")
            case_report = await self.lead_researcher._build_case(
                topic, plan, extractions, "affirmative", log
            )

            return plan, records, top_records, extractions, case_report, aff_highest_tier

        # --- Falsification Track (Steps 1'-4', 6) ---
        async def falsification_track():
            nonlocal fal_wide_net_total, fal_screened_in, fal_fulltext_ok, fal_fulltext_err

            log(f"\n{'='*70}")
            log(f"STEP 1b: TIERED KEYWORD GENERATION + AUDITOR GATE (Falsification)")
            log(f"{'='*70}")
            plan = await self.counter_researcher._formulate_tiered_strategy(
                topic, "adversarial", framing_context, decomposition, log=log
            )

            log(f"\n{'='*70}")
            log(f"STEP 2b: TIERED CASCADE SEARCH (Falsification)")
            log(f"{'='*70}")
            records, fal_highest_tier = await self.counter_researcher._tiered_search(plan, log)
            fal_wide_net_total = len(records)

            log(f"\n{'='*70}")
            log(f"STEP 3b: SCREENING ({len(records)} → top 20) (Falsification)")
            log(f"{'='*70}")
            top_records = await self.counter_researcher._screen_and_prioritize(
                records, plan, topic=topic, log=log
            )
            fal_screened_in = len(top_records)

            log(f"\n{'='*70}")
            log(f"STEP 4b: DEEP EXTRACTION ({len(top_records)} articles) (Falsification)")
            log(f"{'='*70}")
            fulltexts = await self.fulltext_fetcher.fetch_all(top_records)
            fal_fulltext_ok = sum(1 for ft in fulltexts if not ft.error)
            fal_fulltext_err = sum(1 for ft in fulltexts if ft.error)
            log(f"    Full-text retrieved: {fal_fulltext_ok}/{len(fulltexts)}")

            extractions = await self.counter_researcher._deep_extract_batch(
                fulltexts, top_records, plan.pico, log
            )

            log(f"\n{'='*70}")
            log(f"STEP 5b: FALSIFICATION CASE")
            log(f"{'='*70}")
            case_report = await self.counter_researcher._build_case(
                topic, plan, extractions, "falsification", log
            )

            return plan, records, top_records, extractions, case_report, fal_highest_tier

        # --- Run both tracks in parallel ---
        log(f"\n{'='*70}")
        log(f"RUNNING AFFIRMATIVE & FALSIFICATION TRACKS IN PARALLEL")
        log(f"{'='*70}")

        (aff_strategy, aff_records, aff_top, aff_extractions, aff_case, aff_highest_tier), \
        (fal_strategy, fal_records, fal_top, fal_extractions, fal_case, fal_highest_tier) = await asyncio.gather(
            affirmative_track(), falsification_track()
        )

        # --- Step 7: Deterministic Math ---
        log(f"\n{'='*70}")
        log(f"STEP 6: DETERMINISTIC MATH (ARR/NNT)")
        log(f"{'='*70}")
        all_extractions = aff_extractions + fal_extractions
        impacts = clinical_math.batch_calculate(all_extractions)
        math_report = clinical_math.format_math_report(impacts)
        log(f"    Calculated clinical impact for {len(impacts)} studies with CER+EER data")
        if impacts:
            for imp in impacts:
                log(f"      {imp.study_id}: NNT={imp.nnt:.1f} ({imp.direction})")

        # --- Step 8: GRADE Synthesis ---
        log(f"\n{'='*70}")
        log(f"STEP 7: GRADE SYNTHESIS")
        log(f"{'='*70}")

        search_date = datetime.date.today().isoformat()
        total_wide = aff_wide_net_total + fal_wide_net_total
        total_screened = aff_screened_in + fal_screened_in
        total_ft_ok = aff_fulltext_ok + fal_fulltext_ok
        total_ft_err = aff_fulltext_err + fal_fulltext_err

        audit_text = await self._grade_synthesis(
            topic, aff_case, fal_case, math_report,
            aff_strategy, fal_strategy,
            total_wide, total_screened, total_ft_ok, total_ft_err,
            search_date, log
        )

        # --- Save intermediate artifacts ---
        if output_dir:
            self._save_artifacts(
                output_dir, aff_strategy, fal_strategy,
                aff_records, fal_records, aff_top, fal_top,
                math_report,
                aff_highest_tier=aff_highest_tier,
                fal_highest_tier=fal_highest_tier,
            )

        # --- Build backward-compatible return ---
        # Convert extractions to SummarizedSource for compatibility
        aff_sources = self._extractions_to_sources(aff_extractions, "affirmative")
        fal_sources = self._extractions_to_sources(fal_extractions, "falsification")

        combined_metrics = SearchMetrics(
            search_date=search_date,
            databases_searched=["PubMed", "Google Scholar"],
            total_identified=total_wide,
            total_after_dedup=total_wide,  # dedup happens inside PubMedClient
            total_fetched=total_ft_ok + total_ft_err,
            total_fetch_errors=total_ft_err,
            total_with_content=total_ft_ok,
            total_summarized=len(aff_extractions) + len(fal_extractions),
            academic_sources=total_wide,
            general_web_sources=0,
            wide_net_total=total_wide,
            screened_in=total_screened,
            fulltext_retrieved=total_ft_ok,
            fulltext_errors=total_ft_err,
        )

        lead_duration = time.time() - start_time
        lead_report = ResearchReport(
            topic=topic, role="Lead Researcher",
            sources=aff_sources,
            report=aff_case,
            iterations_used=0,
            total_urls_fetched=aff_fulltext_ok + aff_fulltext_err,
            total_summaries=len(aff_extractions),
            total_errors=aff_fulltext_err,
            duration_seconds=lead_duration,
            search_metrics=SearchMetrics(
                search_date=search_date,
                databases_searched=["PubMed", "Google Scholar"],
                total_identified=aff_wide_net_total,
                total_after_dedup=aff_wide_net_total,
                total_fetched=aff_fulltext_ok + aff_fulltext_err,
                total_fetch_errors=aff_fulltext_err,
                total_with_content=aff_fulltext_ok,
                total_summarized=len(aff_extractions),
                academic_sources=aff_wide_net_total,
                general_web_sources=0,
                wide_net_total=aff_wide_net_total,
                screened_in=aff_screened_in,
                fulltext_retrieved=aff_fulltext_ok,
                fulltext_errors=aff_fulltext_err,
            ),
        )

        counter_report = ResearchReport(
            topic=topic, role="Counter Researcher",
            sources=fal_sources,
            report=fal_case,
            iterations_used=0,
            total_urls_fetched=fal_fulltext_ok + fal_fulltext_err,
            total_summaries=len(fal_extractions),
            total_errors=fal_fulltext_err,
            duration_seconds=lead_duration,
            search_metrics=SearchMetrics(
                search_date=search_date,
                databases_searched=["PubMed", "Google Scholar"],
                total_identified=fal_wide_net_total,
                total_after_dedup=fal_wide_net_total,
                total_fetched=fal_fulltext_ok + fal_fulltext_err,
                total_fetch_errors=fal_fulltext_err,
                total_with_content=fal_fulltext_ok,
                total_summarized=len(fal_extractions),
                academic_sources=fal_wide_net_total,
                general_web_sources=0,
                wide_net_total=fal_wide_net_total,
                screened_in=fal_screened_in,
                fulltext_retrieved=fal_fulltext_ok,
                fulltext_errors=fal_fulltext_err,
            ),
        )

        audit_report = ResearchReport(
            topic=topic, role="Auditor",
            sources=aff_sources + fal_sources,
            report=audit_text,
            iterations_used=0,
            total_urls_fetched=(aff_fulltext_ok + aff_fulltext_err + fal_fulltext_ok + fal_fulltext_err),
            total_summaries=len(aff_extractions) + len(fal_extractions),
            total_errors=aff_fulltext_err + fal_fulltext_err,
            duration_seconds=time.time() - start_time,
            search_metrics=combined_metrics,
        )

        total_time = time.time() - start_time
        log(f"\n{'='*70}")
        log(f"ALL RESEARCH COMPLETE in {total_time:.0f}s")
        log(f"  Affirmative: {len(aff_extractions)} studies from {aff_wide_net_total} candidates")
        log(f"  Falsification: {len(fal_extractions)} studies from {fal_wide_net_total} candidates")
        log(f"  Clinical math: {len(impacts)} studies with NNT data")
        log(f"  Total articles analyzed: {len(all_extractions)}")
        log(f"{'='*70}\n")

        return {
            "lead": lead_report,
            "counter": counter_report,
            "audit": audit_report,
            # Raw pipeline data for IMRaD SOT assembly (additive — backward-compatible)
            "pipeline_data": {
                "aff_strategy": aff_strategy,
                "fal_strategy": fal_strategy,
                "aff_extractions": aff_extractions,
                "fal_extractions": fal_extractions,
                "aff_top": aff_top,
                "fal_top": fal_top,
                "math_report": math_report,
                "impacts": impacts,
                "framing_context": framing_context,
                "search_date": search_date,
                "aff_highest_tier": aff_highest_tier,
                "fal_highest_tier": fal_highest_tier,
                "metrics": {
                    "aff_wide_net_total": aff_wide_net_total,
                    "aff_screened_in": aff_screened_in,
                    "aff_fulltext_ok": aff_fulltext_ok,
                    "aff_fulltext_err": aff_fulltext_err,
                    "fal_wide_net_total": fal_wide_net_total,
                    "fal_screened_in": fal_screened_in,
                    "fal_fulltext_ok": fal_fulltext_ok,
                    "fal_fulltext_err": fal_fulltext_err,
                },
            },
        }

    async def _grade_synthesis(
        self, topic: str, aff_case: str, fal_case: str, math_report: str,
        aff_strategy: TieredSearchPlan, fal_strategy: TieredSearchPlan,
        total_wide: int, total_screened: int, total_ft_ok: int, total_ft_err: int,
        search_date: str, log=print
    ) -> str:
        """Step 8: GRADE framework synthesis by the Auditor."""
        log(f"    [Step 7] GRADE synthesis...")

        pico_str = json.dumps(aff_strategy.pico)

        audit_system = (
            "You are The Auditor — an independent scientific arbiter.\n\n"
            "You have received:\n"
            "1. The AFFIRMATIVE CASE (arguing FOR the intervention)\n"
            "2. The FALSIFICATION CASE (arguing AGAINST the intervention)\n"
            "3. DETERMINISTIC MATH (Python-calculated ARR, RRR, NNT — these numbers are EXACT, not LLM-generated)\n\n"
            f"PICO Framework: {pico_str}\n\n"
            "Your task: Issue a GRADE-framework synthesis.\n\n"
            "Structure:\n"
            "1. Executive Summary (3-4 sentences)\n"
            "2. Evidence Profile\n"
            "   - Study designs: [list study types included]\n"
            "   - Total participants across key studies: N = X\n"
            "   - Risk of bias assessment: [summary]\n"
            "   - Consistency: [do studies agree?]\n"
            "   - Directness: [do studies directly measure the outcome of interest?]\n"
            "   - Precision: [are confidence intervals narrow?]\n"
            "   - Publication bias: [any evidence of selective reporting?]\n\n"
            "3. GRADE Assessment\n"
            "   Start at HIGH for RCTs, LOW for observational. Then apply modifiers:\n"
            "   DOWNGRADE for: Risk of bias, Inconsistency, Indirectness, Imprecision, Publication bias\n"
            "   UPGRADE for: Large effect, Dose-response, Plausible confounders would reduce effect\n"
            "   FINAL GRADE: HIGH | MODERATE | LOW | VERY LOW\n\n"
            "4. Clinical Impact (from deterministic math)\n"
            "   - Include the NNT table directly (do NOT recalculate — use the exact numbers provided)\n"
            "   - Interpret the NNT in clinical context\n\n"
            "5. Balanced Verdict\n"
            "   - What does the weight of evidence actually say?\n"
            "   - What are the key caveats?\n"
            "   - What would change the conclusion?\n\n"
            "6. Recommendations for Further Research\n\n"
            "7. PRISMA Flow Diagram (text-based)\n"
            f"   Records identified: {total_wide}\n"
            f"   Screened (top studies): {total_screened}\n"
            f"   Full-text retrieved: {total_ft_ok}\n"
            f"   Full-text errors: {total_ft_err}\n"
            f"   Included in synthesis: {total_screened}\n\n"
            "8. Consolidated Evidence Table\n"
            "   | Study | Design | N | Effect | CER | EER | ARR | NNT | Bias Risk | GRADE Impact |\n\n"
            "9. Full Reference List\n\n"
            "CRITICAL RULES:\n"
            "- NEVER recalculate ARR or NNT — use the Python-provided numbers exactly\n"
            "- Be heavily caveated — acknowledge uncertainty\n"
            "- Flag any potential conflicts of interest\n"
            "- Distinguish between statistical significance and clinical significance\n"
            "- Note that absence of evidence is not evidence of absence"
        )

        combined_input = (
            f"TOPIC: {topic}\n\n"
            f"=== SEARCH METHODOLOGY ===\n"
            f"Search date: {search_date}\n"
            f"Databases: PubMed (MeSH Boolean), Google Scholar\n"
            f"Records identified: {total_wide}\n"
            f"Screened to top studies: {total_screened}\n"
            f"Full-text retrieved: {total_ft_ok} (errors: {total_ft_err})\n\n"
            f"=== AFFIRMATIVE CASE ===\n{aff_case}\n\n"
            f"=== FALSIFICATION CASE ===\n{fal_case}\n\n"
            f"=== DETERMINISTIC MATH (Python-calculated, NOT LLM) ===\n{math_report}\n"
        )

        if len(combined_input) > 80000:
            combined_input = combined_input[:80000] + "\n\n[...truncated...]"

        try:
            resp = await self.smart_client.chat.completions.create(
                model=self.smart_model,
                messages=[
                    {"role": "system", "content": audit_system},
                    {"role": "user", "content": combined_input}
                ],
                max_tokens=8000, temperature=0.2, timeout=300
            )
            audit_text = resp.choices[0].message.content.strip()
            log(f"    [Step 7] GRADE synthesis complete ({len(audit_text)} chars)")
            return audit_text
        except Exception as e:
            logger.error(f"GRADE synthesis failed: {e}")
            return (
                f"# GRADE Synthesis: {topic}\n\n"
                f"*GRADE synthesis failed ({e}). Raw inputs below.*\n\n"
                f"{combined_input}"
            )

    @staticmethod
    def _extractions_to_sources(extractions: List[DeepExtraction], role: str) -> List[SummarizedSource]:
        """Convert DeepExtraction list to SummarizedSource for backward compatibility."""
        sources = []
        for ex in extractions:
            metadata = StudyMetadata(
                study_type=ex.study_design,
                sample_size=str(ex.sample_size_total) if ex.sample_size_total else None,
                key_result=ex.effect_size,
                journal_name=None,
                authors=None,
                effect_size=ex.effect_size,
                limitations=ex.attrition_pct,
                demographics=ex.demographics,
                funding_source=ex.funding_source,
            )
            sources.append(SummarizedSource(
                url=ex.url,
                title=ex.title,
                summary=ex.raw_facts,
                query=role,
                goal=role,
                metadata=metadata,
            ))
        return sources

    @staticmethod
    def _save_artifacts(
        output_dir: str,
        aff_strategy: TieredSearchPlan, fal_strategy: TieredSearchPlan,
        aff_records: List[WideNetRecord], fal_records: List[WideNetRecord],
        aff_top: List[WideNetRecord], fal_top: List[WideNetRecord],
        math_report: str,
        aff_highest_tier: int = 1,
        fal_highest_tier: int = 1,
    ):
        """Save intermediate pipeline artifacts to output directory."""
        import dataclasses
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Strategy files — TieredSearchPlan serialized via dataclasses.asdict
        with open(out / "search_strategy_aff.json", 'w') as f:
            json.dump(dataclasses.asdict(aff_strategy), f, indent=2)
        with open(out / "search_strategy_neg.json", 'w') as f:
            json.dump(dataclasses.asdict(fal_strategy), f, indent=2)

        # Screening decisions (one file per track) — full candidate list for debugging
        def _record_to_dict(r, selected: bool) -> dict:
            return {
                "selected": selected,
                "pmid": r.pmid,
                "doi": r.doi,
                "title": r.title,
                "study_type": r.study_type,
                "sample_size": r.sample_size,
                "year": r.year,
                "journal": r.journal,
                "authors": r.authors,
                "source_db": r.source_db,
                "research_tier": r.research_tier,
                "url": r.url,
                "abstract_snippet": (r.abstract or "")[:300],
            }

        tier_labels = {1: "established", 2: "supporting", 3: "speculative"}

        def _screening_payload(records, top, highest_tier):
            selected_set = {id(r) for r in top}
            by_source: dict = {}
            for r in records:
                by_source[r.source_db] = by_source.get(r.source_db, 0) + 1
            return {
                # Top-level summary (kept for backward compat with pipeline.py gate check)
                "total_candidates": len(records),
                "selected_count": len(top),
                "highest_tier_reached": highest_tier,
                "tier_label": tier_labels.get(highest_tier, "unknown"),
                "by_source_db": by_source,
                # Full record lists for debugging
                "selected_records": [_record_to_dict(r, True) for r in top],
                "all_candidates": [
                    _record_to_dict(r, id(r) in selected_set) for r in records
                ],
            }

        with open(out / "screening_results_aff.json", 'w') as f:
            json.dump(_screening_payload(aff_records, aff_top, aff_highest_tier),
                      f, indent=2, ensure_ascii=False)
        with open(out / "screening_results_neg.json", 'w') as f:
            json.dump(_screening_payload(fal_records, fal_top, fal_highest_tier),
                      f, indent=2, ensure_ascii=False)

        # Math report
        with open(out / "clinical_math.md", 'w') as f:
            f.write(math_report)


# --- Convenience functions ---

async def run_deep_research(
    topic: str,
    brave_api_key: str = "",
    results_per_query: int = 8,
    max_iterations: int = MAX_RESEARCH_ITERATIONS,
    fast_model_available: bool = True,
    framing_context: str = "",
    output_dir: str = None
) -> Dict[str, ResearchReport]:
    orchestrator = Orchestrator(
        brave_api_key=brave_api_key,
        results_per_query=results_per_query,
        max_iterations=max_iterations,
        fast_model_available=fast_model_available
    )
    return await orchestrator.run(topic, framing_context=framing_context, output_dir=output_dir)


async def main():
    """Test the evidence-based clinical research pipeline."""
    import os
    topic = "does coffee intake improve cognitive performance and productivity?"
    brave_key = os.getenv("BRAVE_API_KEY", "")

    # Check fast model
    fast_available = True
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{FAST_BASE_URL}/models")
            fast_available = resp.status_code == 200
    except Exception:
        fast_available = False

    if not fast_available:
        print("NOTE: Fast model not available. Using smart-only mode.")

    from pathlib import Path
    output_dir = Path("research_outputs/test_deep_agent")
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = await run_deep_research(
        topic=topic,
        brave_api_key=brave_key,
        results_per_query=5,
        fast_model_available=fast_available,
        output_dir=str(output_dir)
    )

    # Save reports
    report_filenames = {"lead": "affirmative_case.md", "counter": "falsification_case.md", "audit": "grade_synthesis.md"}
    for role, report in reports.items():
        filename = output_dir / report_filenames.get(role, f"{role}.md")
        with open(filename, "w") as f:
            f.write(report.report)
        print(f"Saved {role} report: {filename} ({len(report.report)} chars)")

    print(f"\nTotal sources: {reports['audit'].total_summaries}")
    print(f"Total time: {reports['audit'].duration_seconds:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
