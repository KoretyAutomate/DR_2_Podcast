"""
Deep Research Agent - Evidence-Based Clinical Research Pipeline

Optimized for Nvidia DGX Spark (128GB Unified Memory):
- SMART MODEL (configured via MODEL_NAME env var) on port 8000: every LLM call in this module.

Single-model since 2026-08-10. The second endpoint (FAST MODEL, qwen3.5:9b via Ollama on
port 11434) was removed: measured on this GB10 box it decoded at 21 tok/s against the Smart
model's 27 tok/s, because Ollama falls back to CPU whenever vLLM holds the GPU. It was the
slower of the two, and its absence had two silent-degradation paths (abstract typing skipped
outright; SOT condensation truncating to 6000 chars).

Architecture (7-Step Clinical Pipeline — parallel a/b tracks):
  Pre-step: Concept Decomposition — canonical scientific terms from a folk topic
  Steps 1a–5a (Affirmative) run in parallel with Steps 1b–5b (Falsification):
    Step 1: Tiered keyword generation (Scientist) → Auditor gate → loop until approved
    Step 2: Cascading PubMed search — Tier1 → if pool<50 add Tier2 → if still<50 add Tier3 + Scholar
    Step 3: Tier-aware screening → top 20 (priority fill T1→T2→T3)
    Step 4: Deep extraction — full text retrieval + clinical variable extraction
    Step 5: Case synthesis — affirmative (5a) or falsification (5b)
  Step 6: Deterministic math — ARR/NNT (Python, no LLM)
  Step 7: GRADE synthesis (Smart Model)

Author: DR_2_Podcast Team
"""

import asyncio
import atexit
import contextlib
import json
import logging
import os
import re
import sqlite3
import time
import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dr2_podcast.pipeline_types import DeepResearchResult

from dr2_podcast.pipeline_types import StudyMetadata, SummarizedSource, SearchMetrics, ResearchReport
from dr2_podcast.utils import (
    gated_create,
    strip_think_blocks,
    is_safe_url,
    safe_bool,
    safe_float,
    safe_int,
    safe_str,
    async_call_smart,
    SmartCallOptions,
    safe_message_text,
    QWEN3_NO_THINK_EXTRA_BODY,
)
from dr2_podcast.config import (
    SMART_MODEL,
    SMART_BASE_URL,
    SCRAPING_TIMEOUT,
    USER_AGENT,
    TIER_CASCADE_THRESHOLD,
    MIN_TIER3_STUDIES,
    MAX_TIER3_RATIO,
)
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import defusedxml.ElementTree as ET

import httpx
from bs4 import BeautifulSoup
import openai
from openai import AsyncOpenAI

from dr2_podcast.research.search_service import SearxngClient

logger = logging.getLogger(__name__)

# --- Configuration (local constants — shared ones imported from config.py) ---

MAX_INPUT_TOKENS = 32000
# Safe char budget for Smart Model content (32K-token context).
# Reserve ~3,200 tokens for system prompt + completion → ~29K tokens available.
# Qwen3 tokenizer is ~1.5 chars/token on medical/CJK text → ~43K chars headroom;
# kept at 29K for safety margin under worst-case all-CJK input.
_SMART_CONTENT_CHARS = 29_000
MAX_RESEARCH_ITERATIONS = 3

JUNK_DOMAINS = {
    "dictionary.com",
    "merriam-webster.com",
    "thefreedictionary.com",
    "cambridge.org",
    "wiktionary.org",
    "vocabulary.com",
    "thesaurus.com",
    "urbandictionary.com",
    "facebook.com",
    "fb.com",
    "twitter.com",
    "instagram.com",
    "tiktok.com",
    "pinterest.com",
    "reddit.com",
    "youtube.com",
    "support.google.com",
    "lkong.com",
    "rctslabs.com",
    "starbucks.com",
    "amazon.com",
    "walmart.com",
    "dailythemedcrosswordanswers.com",
    "crosswordanswers.com",
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
        self._closed = False

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

        # Ensure connection is closed on interpreter shutdown
        atexit.register(self.close)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def get(self, url: str):
        """Return a FetchedPage if cached and not expired, else None."""
        cutoff = time.time() - self.ttl_seconds
        row = self.conn.execute(
            "SELECT url, title, content, word_count FROM page_cache WHERE url = ? AND fetched_at > ?", (url, cutoff)
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
            (page.url, page.title, page.content, page.word_count, time.time()),
        )
        self.conn.commit()

    def close(self):
        if not self._closed:
            self._closed = True
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
    error: str | None = None


# --- New Pipeline Data Models ---


@dataclass
class TierKeywords:
    """Plain keyword lists for one search tier — NO Boolean/MeSH syntax."""

    intervention: list[str]  # exact terms for the intervention at this tier
    outcome: list[str]  # outcome terms at this tier
    population: list[str]  # population terms
    rationale: str  # scientist's justification for this tier's scope


@dataclass
class TieredSearchPlan:
    """Three-tier keyword plan produced by the scientist and approved by the Auditor."""

    pico: dict[str, str]  # P, I, C, O — used downstream in _build_case, screening
    tier1: TierKeywords  # Exact folk/named terms → "Established evidence"
    tier2: TierKeywords  # Canonical scientific synonyms, same substance → "Supporting evidence"
    tier3: TierKeywords  # Active compound class / mechanism → "Speculative extrapolation"
    role: str  # "affirmative" | "adversarial"
    auditor_approved: bool = False
    auditor_notes: str = ""
    revision_count: int = 0


@dataclass
class WideNetRecord:
    """Lightweight screening record — no full text, just title + abstract metadata."""

    pmid: str | None
    doi: str | None
    title: str
    abstract: str
    study_type: str
    sample_size: str | None
    primary_objective: str | None
    year: int | None
    journal: str | None
    authors: str | None
    url: str
    source_db: str  # "pubmed", "cochrane_central", "scholar"
    research_tier: int | None = None  # 1=exact folk  2=scientific synonyms  3=compound class
    relevance_score: float | None = None
    paper_metadata: Optional["PaperMetadata"] = None


def _findings_block(ex: "DeepExtraction") -> str:
    """Every finding a paper reported, one line each, for the case-synthesis prompt."""
    lines = []
    for f in ex.findings or []:
        parts = [f.endpoint or "unnamed endpoint"]
        if f.timepoint:
            parts.append(f"@ {f.timepoint}")
        if f.direction:
            parts.append(f.direction)
        if f.value is not None:
            parts.append(f"{f.value}{f.unit or ''}")
        if f.ci_low is not None and f.ci_high is not None:
            parts.append(f"95% CI {f.ci_low} to {f.ci_high}")
        if f.p_value is not None:
            parts.append(f"p={f.p_value}")
        if f.control_event_rate is not None and f.experimental_event_rate is not None:
            parts.append(f"CER {f.control_event_rate} / EER {f.experimental_event_rate}")
        if f.is_primary:
            parts.append("[primary]")
        lines.append(f"  Finding: {' | '.join(parts)}\n")
    if lines:
        return "".join(lines)
    # No findings and no rates is a legacy record; saying nothing is better than implying a null
    # result the paper never reported.
    if ex.control_event_rate is not None and ex.experimental_event_rate is not None:
        return f"  CER: {ex.control_event_rate}\n  EER: {ex.experimental_event_rate}\n"
    return ""


def _funding_line(ex: "DeepExtraction") -> str:
    """Funding as the block states it, with the provenance the reader needs to weigh it."""
    funding = ex.funding
    if funding is None or funding.funding_disclosure == "unknown":
        return f"  Funding: {ex.funding_source}\n" if ex.funding_source else ""
    if funding.funding_disclosure == "undisclosed":
        return "  Funding: the paper does not state its funding (not the same as unknown)\n"
    verified = "quoted from the paper" if funding.funding_source_type == "extracted_text" else "API metadata, unverified"
    return f"  Funding: {funding.funding_raw} [{funding.funding_category}; {verified}]\n"


def locate_span(text: str, span: str) -> tuple[int, str] | None:
    """Where ``span`` occurs in ``text``, as ``(offset, literal_text)``, or None if it does not.

    The model supplies the QUOTED SPAN; Python finds the offset. Asking a model for a character
    offset would be asking it to count, and a number it invented would satisfy the locator contract
    while pointing nowhere — the contract's whole value is that a Python check can refute it. A span
    that cannot be found is a fabricated quote, and the finding carrying it is dropped.

    Whitespace is normalised for the SEARCH, because extracted full text re-wraps lines and a model
    quoting it will not reproduce the wrap. But the second element is the LITERAL substring of
    ``text`` that matched, not the model's version of it, and that is what a caller must store:
    ``verify_locator_span`` asserts ``text[offset:offset + len(span)] == span``, so storing the
    model's spaces against an offset into line-wrapped source builds a locator that cannot verify.
    """
    if not span or not text:
        return None
    direct = text.find(span)
    if direct >= 0:
        return direct, span
    collapsed = " ".join(span.split())
    if not collapsed:
        return None
    pattern = re.compile(r"\s+".join(re.escape(word) for word in collapsed.split()))
    match = pattern.search(text)
    if not match:
        return None
    return match.start(), match.group(0)


def _primary_flag(raw_value: Any, endpoint: str) -> bool:
    """Whether the model actually said this finding is the primary one."""
    stated = safe_bool(raw_value)
    if stated is None and raw_value is not None:
        logger.warning(
            "is_primary for %r was %r, not a JSON boolean — treating it as not primary",
            endpoint[:60],
            raw_value,
        )
    return stated is True


@dataclass
class FundingBlock:
    """Paper-level funding, as five fields rather than one free-text line.

    Funding has two provenances and only one can satisfy the locator contract: the extractor falls
    back to ``paper_metadata.funding_sources`` from API metadata, which exists nowhere in the paper
    text. ``undisclosed`` (the paper is silent) is NOT ``unknown`` (extraction failed) — Ep09's
    thesis makes that distinction the finding, so they are counted separately.
    """

    funding_raw: str | None = None
    funding_category: str = "unknown"
    funding_disclosure: str = "unknown"
    funding_source_type: str = "none"
    funding_locator: dict | None = None

    def to_dict(self) -> dict:
        return {
            "funding_raw": self.funding_raw,
            "funding_category": self.funding_category,
            "funding_disclosure": self.funding_disclosure,
            "funding_source_type": self.funding_source_type,
            "funding_locator": self.funding_locator,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "FundingBlock":
        return cls(**{k: v for k, v in (d or {}).items() if k in {f.name for f in fields(cls)}})


@dataclass
class Finding:
    """One result: a (population, intervention, comparator, endpoint, timepoint) tuple.

    A PAPER IS NOT A FINDING. The extraction prompt already asks for a primary outcome, a
    secondary_outcomes list and 3-5 key findings, so one paper routinely reports benefit on one
    endpoint and a null result on another. Hanging a singular direction off the paper forces an
    arbitrary pick and silently discards the rest, which breaks replication grouping.
    """

    population: str
    intervention: str
    comparator: str
    endpoint: str
    timepoint: str | None = None
    direction: str = "null_result"
    value: float | None = None
    unit: str | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    is_primary: bool = False
    control_event_rate: float | None = None
    experimental_event_rate: float | None = None
    outcome_is_adverse: bool | None = None
    finding_key: str = ""
    locators: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "Finding":
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in fields(cls)}})


@dataclass
class DeepExtraction:
    """Clinical variable extraction from full-text articles (Step 4)."""

    pmid: str | None
    doi: str | None
    title: str
    url: str
    attrition_pct: str | None = None
    effect_size: str | None = None
    demographics: str | None = None
    follow_up_period: str | None = None
    funding_source: str | None = None
    conflicts_of_interest: str | None = None
    biological_mechanism: str | None = None
    control_event_rate: float | None = None  # CER — needed for Step 7
    experimental_event_rate: float | None = None  # EER — needed for Step 7
    outcome_is_adverse: bool | None = None  # True = event is bad (default assumption)
    primary_outcome: str | None = None
    secondary_outcomes: list[str] | None = None
    blinding: str | None = None
    randomization_method: str | None = None
    intention_to_treat: bool | None = None
    sample_size_total: int | None = None
    sample_size_intervention: int | None = None
    sample_size_control: int | None = None
    study_design: str | None = None
    risk_of_bias: str | None = None
    research_tier: int | None = None  # 1=folk 2=synonym 3=compound
    raw_facts: str = ""
    paper_metadata: Optional["PaperMetadata"] = None
    # --- Step 9a ---------------------------------------------------------
    # findings[] is the real result set; the paper-level effect fields above are DERIVED from the
    # primary finding so every existing consumer keeps working until they are migrated.
    findings: list = field(default_factory=list)
    funding: Optional["FundingBlock"] = None
    trial_registration: str | None = None  # NCT/UMIN — a trial has exactly one, so paper-level
    author_group: str | None = None  # normalised, for counting DISTINCT groups per finding_key

    def to_dict(self) -> dict:
        """Every field, INCLUDING the null ones.

        It used to drop anything None or empty. Absent cannot distinguish "we looked and the paper
        does not say" from "this producer version does not set the field", and the extraction
        contract in dr2_podcast/schemas requires the key to be present and explicitly null. Nested
        records serialise through their own to_dict rather than being handed out as objects.
        """
        d = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "paper_metadata":
                d[f.name] = v.to_dict() if v is not None else None
            elif f.name == "funding":
                d[f.name] = v.to_dict() if v is not None else FundingBlock().to_dict()
            elif f.name == "findings":
                d[f.name] = [finding.to_dict() for finding in (v or [])]
            else:
                d[f.name] = v
        return d


@dataclass
class PaperMetadata:
    """External API metadata for a paper (OpenAlex, Semantic Scholar, Crossref).

    All fields optional — pipeline degrades gracefully if APIs are unreachable.
    """

    citation_count: int | None = None
    influential_citation_count: int | None = None
    fwci: float | None = None
    funding_sources: list[str] | None = None
    is_retracted: bool | None = None
    is_corrected: bool | None = None
    has_clinical_trial_number: bool | None = None
    clinical_trial_numbers: list[str] | None = None
    enrichment_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None and v != "" and v != []:
                d[f.name] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PaperMetadata":
        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


# --- Worker Services (IO + LLM) ---

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MIN_ACADEMIC_RESULTS = 5  # Sufficiency threshold for Tier 1


class PubMedClient:
    """Search PubMed via NCBI E-utilities (free, no API key needed for <3 req/sec).

    Enhanced for clinical pipeline: supports Boolean/MeSH queries, retmax up to 500,
    and extracts PublicationType, DOI, MeSH headings, and structured abstracts.
    """

    def __init__(self):
        self.api_key = os.getenv("PUBMED_API_KEY")

    async def search(self, query: str, max_results: int = 10) -> list[dict[str, str]]:
        """Legacy search — returns simple dicts for backward compatibility."""
        records = await self.search_extended(query, max_results=max_results)
        return [{"url": r["url"], "title": r.get("title", ""), "snippet": r.get("abstract", "")[:500]} for r in records]

    async def search_extended(
        self, query: str, max_results: int = 500, sort: str = "relevance"
    ) -> list[dict[str, Any]]:
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
                    batch_ids = id_list[batch_start : batch_start + 200]
                    fetch_params = {"db": "pubmed", "id": ",".join(batch_ids), "retmode": "xml"}
                    if self.api_key:
                        fetch_params["api_key"] = self.api_key

                    resp = await http.get(f"{PUBMED_BASE_URL}/efetch.fcgi", params=fetch_params)
                    resp.raise_for_status()

                    results.extend(self._parse_articles_xml(resp.text))

                    # Rate limiting: 0.4s delay between batches (3 req/sec without API key)
                    if not self.api_key and batch_start + 200 < len(id_list):
                        await asyncio.sleep(0.4)

        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
        return results

    def _parse_articles_xml(self, xml_text: str) -> list[dict[str, Any]]:
        """Parse PubMed efetch XML into rich article records."""
        results = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"PubMed XML parse error: {e}")
            return []

        for article in root.findall(".//PubmedArticle"):
            try:
                record = self._parse_single_article(article)
                if record:
                    results.append(record)
            except Exception as e:
                logger.debug(f"Failed to parse article: {e}")
        return results

    def _parse_single_article(self, article) -> dict[str, Any] | None:
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
            with contextlib.suppress(ValueError):
                year = int(year_el.text)
        if not year:
            medline_year = article.find(".//MedlineDate")
            if medline_year is not None and medline_year.text:
                m = re.search(r"(\d{4})", medline_year.text)
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
    def _classify_study_type(pub_types: list[str]) -> str:
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


def _dedup_and_filter(results: list[dict[str, str]]) -> list[dict[str, str]]:
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

    async def _extract_searxng_results(self, raw: list) -> list[dict[str, str]]:
        """Extract url/title/snippet from SearXNG raw results."""
        results = []
        for r in raw:
            url = r.get("url", "") if isinstance(r, dict) else getattr(r, "url", "")
            title = r.get("title", "") if isinstance(r, dict) else getattr(r, "title", "")
            snippet = r.get("content", "") if isinstance(r, dict) else getattr(r, "snippet", "")
            if url:
                results.append({"url": url, "title": title, "snippet": snippet})
        return results

    async def search(
        self, query: str, max_results: int = 10, min_academic: int = MIN_ACADEMIC_RESULTS
    ) -> list[dict[str, str]]:
        academic_results = []

        # Tier 1a: PubMed
        pubmed_results = await self.pubmed.search(query, max_results=max_results)
        academic_results.extend(pubmed_results)

        # Tier 1b: Google Scholar via SearXNG
        try:
            async with SearxngClient() as client:
                if await client.validate_connection():
                    raw = await client.search(query, engines=["google scholar"], num_results=max_results)
                    academic_results.extend(await self._extract_searxng_results(raw))
        except Exception as e:
            logger.error(f"Google Scholar search failed: {e}")

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
                    raw = await client.search(query, engines=["google", "bing", "brave"], num_results=max_results)
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
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        for r in data.get("web", {}).get("results", []):
                            general_results.append(
                                {
                                    "url": r.get("url", ""),
                                    "title": r.get("title", ""),
                                    "snippet": r.get("description", ""),
                                }
                            )
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
        # NOT the vLLM gate: this is an HTTP fetch, not an inference call. Holding a vLLM
        # slot for up to SCRAPING_TIMEOUT would starve the other track's model calls.
        async with self.semaphore:
            # SSRF guard — block private/link-local IPs
            if not is_safe_url(url):
                logger.warning(f"Blocked SSRF-unsafe URL: {url}")
                return FetchedPage(url=url, content="", title="", status_code=0)
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
                        soup.find("main")
                        or soup.find("article")
                        or soup.find("div", class_=re.compile(r"content|main-content|post-content|article"))
                        or soup.find("body")
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

    async def fetch_all(self, urls: list[str]) -> list[FetchedPage]:
        return await asyncio.gather(*[self.fetch_page(url) for url in urls])


class SummaryWorker:
    """Parallel page summarization + study-metadata extraction.

    Ran on the Fast model (qwen3.5:9b via Ollama) until 2026-08-10. Retargeted to
    the Smart model when the Fast model was removed: measured on this GB10 box the
    "fast" model decoded at 21 tok/s against the Smart model's 27 tok/s, because
    Ollama runs on CPU whenever vLLM holds the GPU. It was slower AND weaker, so
    the second endpoint bought nothing. The class survives the removal because the
    structured prompt, the concurrency semaphore and the batch gather are all still
    wanted — only the endpoint changed.
    """

    def __init__(self, client: AsyncOpenAI, model: str = SMART_MODEL):
        self.client = client
        self.model = model

    def _parse_metadata_from_response(self, raw_text: str) -> tuple[str, StudyMetadata | None]:
        """Parse FACTS and METADATA sections from the model response.

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

        json_part = raw_text[marker_idx + len(marker) :].strip()

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

        json_str = json_part[brace_start : brace_end + 1]

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
                url=page.url, title=page.title, summary="", query=query, goal=goal, error=page.error or "Empty content"
            )
        content = page.content[:_SMART_CONTENT_CHARS]
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
        try:
            resp = await gated_create(
                self.client,
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Source URL: {page.url}\n\nContent:\n{content}"},
                ],
                max_tokens=1536,
                temperature=0.1,
                timeout=180,
                extra_body=QWEN3_NO_THINK_EXTRA_BODY,
            )
            raw_text = safe_message_text(resp)
            facts_text, metadata = self._parse_metadata_from_response(raw_text)
            return SummarizedSource(
                url=page.url, title=page.title, summary=facts_text, query=query, goal=goal, metadata=metadata
            )
        except Exception as e:
            logger.warning(f"Summarization failed for {page.url}: {str(e)[:100]}")
            return SummarizedSource(
                url=page.url, title=page.title, summary="", query=query, goal=goal, error=str(e)[:200]
            )

    async def summarize_batch(self, pages: list[FetchedPage], goal: str, query: str) -> list[SummarizedSource]:
        return await asyncio.gather(*[self.summarize(page, goal, query) for page in pages])


# --- Smart Model: The Researcher Agent ---


def _keyword_domain_vocabulary(is_social: bool) -> dict:
    """Prompt fragments that differ between the PECO (social) and PICO (clinical) frames."""
    if is_social:
        return {
            "search_db": "cascading academic database search",
            "framework": "PECO",
            "exp_label": "exposure",
            "t1_exp": (
                "  Exposure: exact folk/common names for *this specific exposure/factor* only.\n"
                '  Example for screen time: ["screen time", "smartphone use", "social media use"]\n'
                "  Do NOT include broader digital media — that belongs in Tier 3.\n"
            ),
            "t3_exp": (
                "  Exposure: broader conceptual category (source ambiguity accepted).\n"
                '  Example: ["digital media", "technology use", "internet exposure", "media consumption"]\n'
                "  These results require inference (e.g., any digital media -> screen time effect) and "
                "will be flagged as speculative in the output.\n"
            ),
            "fwk_json": '  "peco": {"population": "...", "exposure": "...", "comparison": "...", "outcome": "..."},\n',
            "fwk_rule": "- Also produce a PECO summary.\n\n",
        }
    return {
        "search_db": "cascading PubMed search",
        "framework": "PICO",
        "exp_label": "intervention",
        "t1_exp": (
            "  Intervention: exact folk/common names for *this specific substance* only.\n"
            '  Example for coffee: ["coffee", "coffee drinking", "coffee consumption"]\n'
            "  Do NOT include caffeine — caffeine also comes from tea, energy drinks, etc.\n"
        ),
        "t3_exp": (
            "  Intervention: active compound class / mechanism (source ambiguity accepted).\n"
            '  Example: ["caffeine", "methylxanthine", "adenosine antagonist", "caffeinated beverage"]\n'
            "  These results require inference (e.g., caffeine from any source -> coffee effect) and "
            "will be flagged as speculative in the output.\n"
        ),
        "fwk_json": '  "pico": {"population": "...", "intervention": "...", "comparison": "...", "outcome": "..."},\n',
        "fwk_rule": "- Also produce a PICO summary.\n\n",
    }


def _format_evidence_blocks(good_summaries: list) -> list:
    """One markdown block per source, with its study metadata inline."""
    blocks = []
    for s in good_summaries:
        meta_line = ""
        if s.metadata:
            m = s.metadata
            fields = [
                ("Type", m.study_type),
                ("N", m.sample_size),
                ("Journal", m.journal_name),
                ("Year", m.publication_year),
                ("Effect", m.effect_size),
                ("Authors", m.authors),
                ("Pop", m.demographics),
                ("Funding", m.funding_source),
            ]
            parts = [f"{label}: {value}" for label, value in fields if value]
            if parts:
                meta_line = f"**Study Metadata:** {' | '.join(parts)}\n"
        blocks.append(
            f"### Source: {s.title or s.url}\n"
            f"**URL:** {s.url}\n"
            f"**Research Goal:** {s.goal}\n"
            f"{meta_line}"
            f"**Extracted Facts:**\n{s.summary}\n"
        )
    return blocks


@dataclass
class AgentDeps:
    """Collaborators a ResearchAgent works through.

    Both researchers are built from the same four, so they travel together.
    """

    smart_client: Any
    summary_worker: Any
    search_service: Any
    fetcher: Any

    def __post_init__(self):
        # summary_worker used to be optional (None when no Fast model was configured),
        # and _search_and_summarize guarded every use. The guards went with the Fast
        # model on 2026-08-10, so a None here is now an AttributeError deep inside a
        # gather — and inside _screen_abstracts it would be swallowed by the broad
        # except and returned as {}, i.e. silently empty typing. Fail at construction.
        if self.summary_worker is None:
            raise ValueError("AgentDeps.summary_worker is required (it is no longer optional)")


@dataclass
class ResearchConfig:
    """Model endpoints and search settings for one pipeline run."""

    brave_api_key: str = ""
    results_per_query: int = 5
    max_iterations: int = MAX_RESEARCH_ITERATIONS
    domain: str = "clinical"
    smart_base_url: str = SMART_BASE_URL
    smart_model: str = SMART_MODEL


@dataclass(frozen=True)
class _ScreenContext:
    """What Step 3 screening needs besides the records themselves."""

    pico_str: str
    max_select: int
    topic: str
    intervention_override: str = ""


@dataclass(frozen=True)
class _TrackSpec:
    """What distinguishes the affirmative track from the falsification one.

    The two Step 1-5 tracks were written out twice inline; everything except
    these five fields was identical.
    """

    researcher_attr: str  # "lead_researcher" | "counter_researcher"
    strategy_role: str  # role passed to _formulate_tiered_strategy
    case_role: str  # role passed to _build_case
    label: str  # "Affirmative" | "Falsification" — log suffix
    step_suffix: str  # "a" | "b" — log step numbering


_AFFIRMATIVE_TRACK = _TrackSpec(
    researcher_attr="lead_researcher",
    strategy_role="affirmative",
    case_role="affirmative",
    label="Affirmative",
    step_suffix="a",
)

_FALSIFICATION_TRACK = _TrackSpec(
    researcher_attr="counter_researcher",
    strategy_role="adversarial",
    case_role="falsification",
    label="Falsification",
    step_suffix="b",
)


@dataclass
class _TrackResult:
    """One track's outputs plus the metrics Step 7 sums across both."""

    plan: Any
    records: list
    top_records: list
    extractions: list
    case_report: Any
    highest_tier: int
    wide_net_total: int
    screened_in: int
    fulltext_ok: int
    fulltext_err: int


class _RecordPool:
    """Accumulates WideNetRecords, deduplicating as they arrive.

    A record is a duplicate if its PMID, its URL, or its title prefix has been
    seen before. Both Step 2 cascades (PubMed and OpenAlex/ERIC) used to carry
    these three sets and the same skip-then-add dance inline at four sites each.
    """

    TITLE_KEY_LEN = 80

    def __init__(self):
        self.records: list[WideNetRecord] = []
        self.seen_pmids: set = set()
        self.seen_urls: set = set()
        self.seen_titles: set = set()

    def is_duplicate(self, *, pmid=None, url=None, title_key=None) -> bool:
        return bool(
            (pmid and pmid in self.seen_pmids)
            or (url and url in self.seen_urls)
            or (title_key and title_key in self.seen_titles)
        )

    def add(self, record: WideNetRecord, *, title_key=None) -> None:
        """Record the identifiers, then append. Callers check is_duplicate first."""
        if record.pmid:
            self.seen_pmids.add(record.pmid)
        if record.url:
            self.seen_urls.add(record.url)
        if title_key:
            self.seen_titles.add(title_key)
        self.records.append(record)

    @classmethod
    def title_key(cls, title: str) -> str:
        return (title or "").lower().strip()[: cls.TITLE_KEY_LEN]


def _add_pubmed_articles(pool: _RecordPool, articles: list, tier_num: int, log) -> int:
    """Append the non-duplicate PubMed articles to the pool. Returns how many."""
    added = 0
    for art in articles:
        pmid = art.get("pmid", "")
        url = art.get("url", "")
        if pool.is_duplicate(pmid=pmid, url=url):
            continue
        pool.add(
            WideNetRecord(
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
            )
        )
        added += 1
        log("[STUDY_FOUND]")
    return added


class ResearchAgent:
    """
    A smart-model-driven researcher that iteratively delegates to workers.

    The agent:
    1. Plans what to search (based on its role and what's missing)
    2. Delegates search + summarization to SearchService + SummaryWorker
    3. Evaluates gathered evidence
    4. Identifies gaps and generates new queries
    5. Repeats until satisfied or max iterations reached
    6. Writes a final report
    """

    # Set by the Orchestrator once the domain is classified, and read through getattr in three
    # places. Declared here so it is a real attribute rather than one mypy has to guess at.
    _domain: str = "clinical"

    def __init__(
        self,
        deps: "AgentDeps",
        smart_model: str = SMART_MODEL,
        results_per_query: int = 5,
        max_iterations: int = MAX_RESEARCH_ITERATIONS,
    ):
        self.smart_client = deps.smart_client
        self.smart_model = smart_model
        self.summary_worker = deps.summary_worker
        self.search = deps.search_service
        self.fetcher = deps.fetcher
        self.results_per_query = results_per_query
        self.max_iterations = max_iterations

    async def _call_smart(self, system: str, user: str, max_tokens: int = 2048, temperature: float = 0.3) -> str:
        """Call the smart model with retry on transient failures.

        Delegates to the shared async_call_smart() helper in utils.py.
        """
        return await async_call_smart(
            self.smart_client,
            self.smart_model,
            system,
            user,
            SmartCallOptions(max_tokens=max_tokens, temperature=temperature),
        )

    def _parse_json_queries(self, raw: str) -> list[ResearchQuery]:
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

    def _format_evidence_so_far(self, summaries: list[SummarizedSource]) -> str:
        """Format collected evidence for the smart model to review."""
        good = [s for s in summaries if s.summary and s.summary != "NO RELEVANT DATA" and not s.error]
        if not good:
            return "No evidence collected yet."
        blocks = []
        for s in good:
            blocks.append(f"- [{s.title or 'Untitled'}]({s.url}): {s.summary[:300]}")
        return "\n".join(blocks)

    async def _search_and_summarize(
        self, queries: list[ResearchQuery], seen_urls: set, log
    ) -> tuple[list[SummarizedSource], int, int]:
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

            batch = await self.summary_worker.summarize_batch(good_pages, rq.goal, rq.query)

            good = sum(1 for s in batch if s.summary and not s.error)
            log(f"      [{rq.goal[:40]}] {good}/{len(good_pages)} summarized")
            all_summaries.extend(batch)

        return all_summaries, total_fetched, total_errors

    def _build_plan_prompt(self, role, role_instructions, topic, iteration, all_summaries) -> str:
        """Iteration 0 opens the search; later iterations ask for gap-filling queries."""
        if iteration == 0:
            return (
                f"You are a {role}. {role_instructions}\n\n"
                f"Topic: {topic}\n\n"
                f"Generate 5-7 specific search queries to begin your research.\n"
                f'Return ONLY a JSON array: [{{"query": "...", "goal": "..."}}]'
            )
        good = len([s for s in all_summaries if s.summary and not s.error])
        return (
            f"You are a {role}. {role_instructions}\n\n"
            f"Topic: {topic}\n\n"
            f"Evidence gathered so far ({good} sources):\n"
            f"{self._format_evidence_so_far(all_summaries)}\n\n"
            f"Based on what you have, identify 3-5 specific GAPS in your evidence.\n"
            f"Generate NEW targeted search queries to fill those gaps.\n"
            f"If evidence is sufficient, return an empty array: []\n\n"
            f'Return ONLY a JSON array: [{{"query": "...", "goal": "..."}}]'
        )

    async def research(self, topic: str, role: str, role_instructions: str, log=logger.info) -> ResearchReport:
        """
        Run iterative research as the given role.

        Args:
            topic: Research topic
            role: Role name (e.g. "Lead Researcher", "Counter Researcher")
            role_instructions: Specific instructions for this role
            log: Logging callback
        """
        start_time = time.time()
        all_summaries: list[SummarizedSource] = []
        seen_urls: set = set()
        total_fetched = 0
        total_errors = 0

        log(f"\n  {'─' * 60}")
        log(f"  {role.upper()}: Starting iterative research")
        log(f"  Topic: {topic}")
        log(f"  Max iterations: {self.max_iterations}")
        log(f"  {'─' * 60}")

        for iteration in range(self.max_iterations):
            log(f"\n  ── Iteration {iteration + 1}/{self.max_iterations} ──")

            plan_prompt = self._build_plan_prompt(role, role_instructions, topic, iteration, all_summaries)

            log("    Planning: asking smart model for queries...")
            raw_plan = await self._call_smart(
                "You are a research planning expert. Return ONLY valid JSON arrays.",
                plan_prompt,
                max_tokens=2048,
                temperature=0.3,
            )
            queries = self._parse_json_queries(raw_plan)

            if not queries:
                log("    Smart model returned no new queries — evidence deemed sufficient")
                break

            log(f"    Plan: {len(queries)} queries")
            for i, q in enumerate(queries, 1):
                log(f"      {i}. [{q.goal[:50]}] {q.query}")

            # Step 2: Delegate search + summarization to workers
            log("    Delegating to search + summarize workers...")
            batch_summaries, batch_fetched, batch_errors = await self._search_and_summarize(queries, seen_urls, log)
            all_summaries.extend(batch_summaries)
            total_fetched += batch_fetched
            total_errors += batch_errors

            good_count = len([s for s in all_summaries if s.summary and not s.error])
            log(f"    Iteration {iteration + 1} complete: {good_count} total good sources")

        # Step 3: Smart model writes final report
        good_summaries = [s for s in all_summaries if s.summary and s.summary != "NO RELEVANT DATA" and not s.error]
        log(f"\n  Writing final report from {len(good_summaries)} sources...")

        evidence_blocks = _format_evidence_blocks(good_summaries)
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
                max_tokens=8000,
                temperature=0.2,
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

    @staticmethod
    def _truncate_after_json(text: str) -> str:
        """Truncate trailing text after the outermost JSON object/array closes."""
        if not text:
            return text
        opener = text[0]
        closer = "}" if opener == "{" else "]" if opener == "[" else None
        if closer is None:
            return text
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if ch == "\\":
                if in_string:
                    escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    return text[: i + 1]
        return text  # unclosed — return as-is for downstream repair

    @staticmethod
    def _parse_json_response(raw: str) -> Any:
        """Parse JSON from smart model output, handling code blocks and LLM noise.

        Static because the Orchestrator needs the same repair logic for the GRADE record and
        borrowing a bound method chains into whatever else `self` happens to carry — which it did,
        and the tests caught it.
        """
        if not raw or not raw.strip():
            raise ValueError("Empty response from LLM")
        # Strip <think>...</think> blocks (Qwen3 thinking mode safety net)
        raw = strip_think_blocks(raw)
        # Strip code blocks
        if "```" in raw:
            match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
            if match:
                raw = match.group(1).strip()
        # Strip leading non-JSON text (e.g. "Here is the JSON:")
        first_brace = raw.find("{")
        first_bracket = raw.find("[")
        starts = [i for i in (first_brace, first_bracket) if i >= 0]
        if starts:
            raw = raw[min(starts) :]
        # Truncate trailing text after JSON object closes (handles "extra data" errors)
        raw = ResearchAgent._truncate_after_json(raw)
        # Try parsing as-is
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Repair: close unterminated string, then try appending closing braces
            stripped = raw.rstrip()
            # If last non-whitespace is inside a string (odd unescaped quotes),
            # close the string first
            quote_count = stripped.count('"') - stripped.count('\\"')
            if quote_count % 2 == 1:
                stripped += '"'
            for suffix in ["}", "}}", "]", "]}", ""]:
                try:
                    data = json.loads(stripped + suffix)
                    break
                except json.JSONDecodeError:
                    continue
            else:
                logger.warning("JSON parse failed, raw (first 500 chars): %s", raw[:500])
                raise
        # Detect template echo: if a value contains the example template text, null it out
        template_markers = ["parallel RCT | crossover RCT | meta-analysis"]
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, str):
                    for marker in template_markers:
                        if marker in v:
                            data[k] = None
        return data

    async def _decompose_topic(self, topic: str, framing_context: str = "") -> dict:
        """Pre-PICO: extract canonical scientific terms from a folk-language topic."""
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
        self,
        topic: str,
        role: str,
        framing_context: str,
        decomposition: dict | None,
        auditor_feedback: str = "",
        log=logger.info,
    ) -> TieredSearchPlan:
        """Generate three-tier keyword plan as plain lists — NO Boolean/MeSH syntax."""
        log(f"    [Step 1] Generating tiered keywords ({role})...")

        is_adversarial = role == "adversarial"
        is_social = getattr(self, "_domain", "clinical") == "social_science"

        framing_note = f"\n\nRESEARCH FRAMING:\n{framing_context[:3000]}" if framing_context else ""

        decomp_note = ""
        if decomposition and any(decomposition.values()):
            canonical = ", ".join(decomposition.get("canonical_terms", []))
            related = ", ".join(decomposition.get("related_concepts", []))
            decomp_note = f"\n\nCANONICAL SCIENTIFIC TERMS: {canonical}\nRELATED CONCEPTS: {related}"

        revision_note = ""
        if auditor_feedback:
            revision_note = (
                f"\n\n⚠ AUDITOR REJECTED YOUR PREVIOUS KEYWORDS with this feedback:\n"
                f"{auditor_feedback}\n"
                "You MUST revise your keywords to address this feedback."
            )

        if is_adversarial:
            role_header = (
                "You are a systematic review scientist generating search keyword tiers "
                "for the FALSIFICATION track of a systematic review.\n\n"
                "YOUR SOLE GOAL: surface studies that CONTRADICT, NULL, or HARM — "
                "NOT studies that support the intervention.\n"
                "Outcome terms MUST target adverse effects, null results, harms, tolerance, "
                "withdrawal, dose-response toxicity, methodological failures, and funding bias.\n"
                "Do NOT use benefit-oriented outcome terms (e.g., 'productivity', 'performance', "
                "'alertness'). Those belong in the affirmative track.\n\n"
            )
            outcome_ex_t1 = (
                "  Outcome: direct HARM or NULL outcome labels as they appear in clinical trial titles.\n"
                '  Example for coffee: ["sleep disruption", "anxiety", "caffeine dependence", '
                '"jitteriness", "heart palpitation"]\n'
            )
            outcome_ex_t2 = (
                "  Outcome: SUPERSET of Tier 1 harm outcomes — include ALL Tier 1 harm terms PLUS "
                "broader adverse-effect and null-result proxies.\n"
                '  Example: ["sleep disruption", "anxiety", "caffeine dependence", "jitteriness", '
                '"heart palpitation", "cardiovascular risk", "hypertension", "null result", '
                '"no significant effect", "withdrawal symptom"]\n'
            )
        else:
            role_header = "You are a systematic review scientist generating search keyword tiers.\n\n"
            outcome_ex_t1 = (
                "  Outcome: direct primary outcome labels as they appear in clinical trial titles.\n"
                '  Example: ["work productivity", "job performance", "occupational performance"]\n'
            )
            outcome_ex_t2 = (
                "  Outcome: SUPERSET of Tier 1 outcomes. Include ALL Tier 1 outcome terms PLUS broader\n"
                "    proxy outcomes and related clinical endpoints. Must be strictly broader, never narrower.\n"
                '  Example: ["work productivity", "job performance", "cognitive performance", '
                '"alertness", "executive function", "mental performance"]\n'
            )

        vocab = _keyword_domain_vocabulary(is_social)
        _search_db = vocab["search_db"]
        _t1_exp, _t3_exp = vocab["t1_exp"], vocab["t3_exp"]
        _fwk_json, _fwk_rule, _exp_label = vocab["fwk_json"], vocab["fwk_rule"], vocab["exp_label"]

        system = (
            f"{role_header}"
            f"TASK: Produce three keyword tiers for a {_search_db}.\n"
            "Each tier is a set of PLAIN KEYWORD LISTS — no Boolean operators, no MeSH notation, "
            "no brackets, no field tags. Just simple English phrases.\n\n"
            "TIER DEFINITIONS:\n\n"
            "TIER 1 — 'Established evidence' (strictest):\n"
            f"{_t1_exp}"
            f"{outcome_ex_t1}"
            "  Population: specific population relevant to the research question.\n"
            '  Example: ["working adults", "employees"]\n\n'
            "TIER 2 — 'Supporting evidence' (broadened scope):\n"
            f"  {_exp_label.capitalize()}: <<INHERITED FROM TIER 1 — do not generate, will be copied automatically>>\n"
            f"{outcome_ex_t2}"
            "  Population: BROADER than Tier 1. Widen to more general populations.\n"
            '  Example: ["adults", "healthy adults"]\n\n'
            "TIER 3 — 'Speculative extrapolation' (broader category):\n"
            f"{_t3_exp}"
            "  Outcome: <<INHERITED FROM TIER 2 — do not generate, will be copied automatically>>\n"
            "  Population: <<INHERITED FROM TIER 2 — do not generate, will be copied automatically>>\n\n"
            "RULES:\n"
            "- Keyword lists must contain plain phrases only — no AND, OR, NOT, [MeSH], [tiab], etc.\n"
            "- Each term should be 1-4 words max.\n"
            f"- Tier 1 {_exp_label} must NOT contain broad category terms.\n"
            "- Tier 2 outcome MUST include ALL Tier 1 outcome terms plus additional broader terms.\n"
            + ("- Outcome terms MUST be harm/null-focused — NOT benefit-oriented.\n" if is_adversarial else "")
            + f"{_fwk_rule}"
            "Return ONLY valid JSON (note: Tier 2 has no intervention/exposure, Tier 3 has no outcome/population):\n"
            "{\n"
            f"{_fwk_json}"
            '  "tier1": {\n'
            '    "intervention": ["term1", "term2"],\n'
            '    "outcome": ["term1", "term2"],\n'
            '    "population": ["term1", "term2"],\n'
            '    "rationale": "Why these exact terms belong at Tier 1"\n'
            "  },\n"
            '  "tier2": {"outcome": ["all tier1 outcomes + broader terms"], "population": ["broader"], "rationale": "..."},\n'
            '  "tier3": {"intervention": ["compound class terms"], "rationale": "..."}\n'
            "}"
            f"{framing_note}{decomp_note}{revision_note}"
        )

        user = f"Research topic: {topic}"

        raw = await self._call_smart(system, user, max_tokens=2048, temperature=0.2)
        try:
            data = self._parse_json_response(raw)

            # Map PECO → PICO for social science domain
            if is_social and "peco" in data:
                peco = data["peco"]
                data["pico"] = {
                    "population": peco.get("population", ""),
                    "intervention": peco.get("exposure", ""),
                    "comparison": peco.get("comparison", ""),
                    "outcome": peco.get("outcome", ""),
                }

            def parse_tier(d: dict) -> TierKeywords:
                # Accept both "intervention" and "exposure" keys
                intervention = d.get("intervention", []) or d.get("exposure", [])
                return TierKeywords(
                    intervention=intervention,
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

            # Guard: if Tier 3 intervention wasn't broadened (same as Tier 1),
            # use canonical_terms from topic decomposition as a broader fallback.
            t3_set = set(t.lower().strip() for t in plan.tier3.intervention)
            t1_set = set(t.lower().strip() for t in plan.tier1.intervention)
            if t3_set == t1_set or not plan.tier3.intervention:
                canonical = (decomposition or {}).get("canonical_terms", [])
                if canonical:
                    plan.tier3.intervention = list(canonical)
                    log(f"    [Step 1] Tier 3 intervention not broadened — using canonical terms: {canonical[:4]}")

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
                pico={
                    "population": "general",
                    "intervention": topic,
                    "comparison": "control",
                    "outcome": "primary outcome",
                },
                tier1=TierKeywords(intervention=fallback_terms, outcome=[], population=[], rationale="fallback"),
                tier2=TierKeywords(intervention=fallback_terms, outcome=[], population=[], rationale="fallback"),
                tier3=TierKeywords(intervention=fallback_terms, outcome=[], population=[], rationale="fallback"),
                role=role,
            )

    async def _audit_tier_plan(self, plan: TieredSearchPlan, topic: str, log=logger.info) -> tuple:
        """Auditor reviews all three tiers in one call. Returns (approved: bool, notes: str)."""
        log(f"    [Auditor] Reviewing tier keyword plan ({plan.role})...")

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

        adversarial_context = (
            "\n⚠ ADVERSARIAL TRACK: Outcome terms MUST be harm/null-focused "
            "(e.g., 'sleep disruption', 'anxiety', 'null result', 'cardiovascular risk'). "
            "Benefit-oriented outcomes (e.g., 'productivity', 'performance') are WRONG for this track. "
            "Check that outcomes target adverse effects and contradicting evidence.\n"
            if plan.role == "adversarial"
            else ""
        )

        user = (
            f"Research topic: {topic}\n"
            f"Track role: {plan.role.upper()}{adversarial_context}\n"
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
                log("    [Auditor] APPROVED")
            else:
                log(f"    [Auditor] REJECTED — {notes[:200]}")
            return approved, notes
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Audit response parse failed: {e} — rejecting (fail-closed)")
            return False, f"Rejected: auditor response could not be parsed ({type(e).__name__})"

    async def _formulate_tiered_strategy(
        self, topic: str, role: str, framing_context: str, decomposition: dict | None, log=logger.info
    ) -> TieredSearchPlan:
        """Step 1: Scientist generates tier keywords, Auditor reviews; loop until approved."""
        MAX_REVISIONS = 2
        feedback = ""

        for attempt in range(MAX_REVISIONS + 1):
            log(f"\n    [Step 1] Generating tier keywords (attempt {attempt + 1}/{MAX_REVISIONS + 1})...")
            plan = await self._generate_tiered_keywords(
                topic, role, framing_context, decomposition, auditor_feedback=feedback, log=log
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

        def group(terms: list[str]) -> str:
            quoted = [f'"{t}"[Title/Abstract]' for t in terms if t.strip()]
            return "(" + " OR ".join(quoted) + ")" if quoted else ""

        parts = [
            g
            for g in [
                group(tier.intervention),
                group(tier.outcome),
                group(tier.population),
            ]
            if g
        ]

        query = " AND ".join(parts)
        if extra_filters and query:
            query += f" AND {extra_filters}"
        return query

    # ------------------------------------------------------------------ #
    #  STEP 2 — Tiered cascade search                                      #
    # ------------------------------------------------------------------ #

    async def _tiered_search(self, plan: TieredSearchPlan, log=logger.info) -> tuple:
        """Step 2: Run tiered PubMed cascade. Stop when pool >= TIER_THRESHOLD.

        Returns:
            (List[WideNetRecord], int) — records and highest tier reached.
        """
        if getattr(self, "_domain", "clinical") == "social_science":
            return await self._tiered_search_social(plan, log)
        log(f"    [Step 2] Tiered cascade search ({plan.role})...")

        pool = _RecordPool()
        tier_configs = [
            (1, plan.tier1, "Humans[MeSH] AND English[la]"),
            (2, plan.tier2, "Humans[MeSH]"),
            (3, plan.tier3, ""),
        ]

        highest_tier = await self._run_pubmed_cascade(pool, tier_configs, log)
        await self._add_scholar_records(pool, plan, log)

        # Zero-result fallback: if all tiers returned nothing, retry with
        # intervention-only queries (no outcome/population AND clause) to cast a
        # wider net — let Step 3 screening do the filtering.
        if not pool.records:
            await self._run_pubmed_fallback(pool, tier_configs, log)

        log(f"    [Step 2] Total pool: {len(pool.records)} records (highest tier: {highest_tier})")
        await self._apply_study_typing(pool.records, log)
        return pool.records[:500], highest_tier

    async def _run_pubmed_cascade(self, pool, tier_configs, log) -> int:
        """Tier 1 -> 2 -> 3, stopping once the pool reaches the threshold."""
        highest_tier = 0
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
                articles = await self.search.pubmed.search_extended(query, max_results=200)
                added = _add_pubmed_articles(pool, articles, tier_num, log)
                log(f"    [Tier {tier_num}] +{added} new records (pool: {len(pool.records)})")
            except Exception as e:
                logger.error(f"Tier {tier_num} PubMed search failed: {e}")

            highest_tier = tier_num

            if len(pool.records) >= TIER_CASCADE_THRESHOLD:
                log(f"    [Tier {tier_num}] Threshold ({TIER_CASCADE_THRESHOLD}) reached — stopping cascade")
                break
        return highest_tier

    async def _run_pubmed_fallback(self, pool, tier_configs, log) -> None:
        """Intervention-only retry, used only when the cascade found nothing."""
        log("    [Step 2] Zero results from cascade — retrying with intervention-only queries")
        for tier_num, tier_kw, filters in tier_configs:
            if not tier_kw.intervention:
                continue
            intervention_only = TierKeywords(
                intervention=tier_kw.intervention,
                outcome=[],
                population=[],
                rationale="fallback-intervention-only",
            )
            query = self._build_tier_query(intervention_only, filters)
            if not query:
                continue
            log(f"    [Tier {tier_num} fallback] Query: {query[:140]}...")
            try:
                articles = await self.search.pubmed.search_extended(query, max_results=200)
                _add_pubmed_articles(pool, articles, tier_num, log)
            except Exception as e:
                logger.error(f"Tier {tier_num} fallback search failed: {e}")
            if len(pool.records) >= TIER_CASCADE_THRESHOLD:
                break
        log(f"    [Step 2] Fallback pool: {len(pool.records)} records")

    async def _add_scholar_records(self, pool, plan: TieredSearchPlan, log) -> None:
        """Google Scholar supplement — Tier 1 plain-text keywords always."""
        scholar_query = " ".join(plan.tier1.intervention + plan.tier1.outcome)
        if not scholar_query.strip():
            return
        try:
            async with SearxngClient() as client:
                if not await client.validate_connection():
                    return
                raw = await client.search(scholar_query, engines=["google scholar"], num_results=100)
                scholar_added = 0
                for r in raw:
                    url = r.get("url", "") if isinstance(r, dict) else getattr(r, "url", "")
                    if not url or url in pool.seen_urls or is_junk_url(url):
                        continue
                    title = r.get("title", "") if isinstance(r, dict) else getattr(r, "title", "")
                    snippet = r.get("content", "") if isinstance(r, dict) else getattr(r, "snippet", "")
                    pool.add(
                        WideNetRecord(
                            pmid=None,
                            doi=None,
                            title=title,
                            abstract=snippet,
                            study_type="other",
                            sample_size=None,
                            primary_objective=None,
                            year=None,
                            journal=None,
                            authors=None,
                            url=url,
                            source_db="scholar",
                            research_tier=1,
                        )
                    )
                    scholar_added += 1
                    log("[STUDY_FOUND]")
                log(f"    [Scholar] +{scholar_added} records (Tier 1 keywords)")
        except Exception as e:
            logger.error(f"Google Scholar search failed: {e}")

    async def _apply_study_typing(self, records: list, log) -> None:
        """Screen study_type / sample_size on "other" records.

        Renamed from _apply_fast_typing 2026-08-10 (Fast model removed). Note the old
        guard was `if not (needs_screening and self.fast_worker): return` — with no Fast
        model configured this silently SKIPPED typing, leaving records as "other" and
        quietly weakening tier-aware screening priority. Typing is now unconditional.
        """
        needs_screening = [r for r in records if r.study_type == "other" and r.abstract]
        if not needs_screening:
            return
        log(f"    [Step 2] Typing {len(needs_screening)} abstracts...")
        screened = await self._screen_abstracts(needs_screening)
        screening_map = {id(r): s for r, s in zip(needs_screening, screened, strict=True)}
        for r in records:
            if id(r) in screening_map:
                s = screening_map[id(r)]
                if s.get("study_type"):
                    r.study_type = s["study_type"]
                if s.get("sample_size"):
                    r.sample_size = s["sample_size"]
                if s.get("primary_objective"):
                    r.primary_objective = s["primary_objective"]

    async def _tiered_search_social(self, plan: TieredSearchPlan, log=logger.info) -> tuple:
        """Step 2: Search OpenAlex + ERIC + Scholar for social science topics."""
        log(f"    [Step 2] Social science search ({plan.role})...")

        pool = _RecordPool()
        highest_tier = 0

        for tier_num, tier_kw in [(1, plan.tier1), (2, plan.tier2), (3, plan.tier3)]:
            terms = tier_kw.intervention + tier_kw.outcome
            query = " ".join(terms[:5])
            if not query.strip():
                log(f"    [Tier {tier_num}] No terms — skipping")
                continue

            log(f"    [Tier {tier_num}] Query: {query[:140]}...")
            await self._add_openalex_records(pool, query, tier_num, log)
            await self._add_eric_records(pool, query, tier_num, log)

            highest_tier = tier_num
            if len(pool.records) >= TIER_CASCADE_THRESHOLD:
                log(f"    [Tier {tier_num}] Threshold ({TIER_CASCADE_THRESHOLD}) reached — stopping cascade")
                break

        await self._add_scholar_records(pool, plan, log)

        log(f"    [Step 2] Total pool: {len(pool.records)} records (highest tier: {highest_tier})")
        await self._apply_study_typing(pool.records, log)
        return pool.records[:500], highest_tier

    async def _add_openalex_records(self, pool, query: str, tier_num: int, log) -> None:
        """OpenAlex works for one tier. The record URL is derived from the DOI."""
        try:
            oa_results = await self._openalex.search_works(query, per_page=50)
            added = 0
            for oa in oa_results:
                title = oa.get("title", "") or ""
                oa_doi = oa.get("doi") or ""
                url = ("https://doi.org/" + oa_doi) if oa_doi else ""
                if not title and not url:
                    continue
                title_key = _RecordPool.title_key(title)
                if pool.is_duplicate(url=url, title_key=title_key):
                    continue
                pri_loc = oa.get("primary_location")
                journal_name = None
                if isinstance(pri_loc, dict):
                    src = pri_loc.get("source")
                    if isinstance(src, dict):
                        journal_name = src.get("display_name")
                pool.add(
                    WideNetRecord(
                        pmid=None,
                        doi=oa_doi or None,
                        title=title,
                        abstract=oa.get("abstract_text", "") or "",
                        study_type=oa.get("type", "other"),
                        sample_size=None,
                        primary_objective=None,
                        year=oa.get("publication_year"),
                        journal=journal_name,
                        authors=None,
                        url=url,
                        source_db="openalex",
                        research_tier=tier_num,
                    ),
                    title_key=title_key,
                )
                added += 1
                log("[STUDY_FOUND]")
            log(f"    [Tier {tier_num}] OpenAlex: +{added} records")
        except Exception as e:
            logger.error(f"Tier {tier_num} OpenAlex search failed: {e}")

    async def _add_eric_records(self, pool, query: str, tier_num: int, log) -> None:
        """ERIC (IES) results for one tier."""
        try:
            eric_results = await self._eric.search(query, max_results=30)
            added = 0
            for er in eric_results:
                title = er.get("title", "")
                url = er.get("url", "")
                title_key = _RecordPool.title_key(title)
                if pool.is_duplicate(url=url, title_key=title_key):
                    continue
                er_authors = er.get("author", [])
                pool.add(
                    WideNetRecord(
                        pmid=None,
                        doi=None,
                        title=title,
                        abstract=er.get("description", ""),
                        study_type="other",
                        sample_size=None,
                        primary_objective=None,
                        year=er.get("year"),
                        journal=er.get("source", ""),
                        authors=", ".join(er_authors) if er_authors else None,
                        url=url,
                        source_db="eric",
                        research_tier=tier_num,
                    ),
                    title_key=title_key,
                )
                added += 1
                log("[STUDY_FOUND]")
            log(f"    [Tier {tier_num}] ERIC: +{added} records")
        except Exception as e:
            logger.error(f"Tier {tier_num} ERIC search failed: {e}")

    async def _screen_abstracts(self, records: list[WideNetRecord]) -> list[dict]:
        """Extract study_type, sample_size, primary_objective from abstracts."""
        # Was Semaphore(2) to work around the Ollama-on-CPU footgun while vLLM held the
        # GPU. That constraint died with the Fast model; the replacement is the SHARED
        # gate in utils.gated_create, not a bigger local one.

        async def screen_one(record: WideNetRecord) -> dict:
            try:
                resp = await gated_create(
                    self.summary_worker.client,
                    model=self.summary_worker.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Extract from this abstract:\n"
                                "- study_type: RCT | meta-analysis | systematic-review | cohort | "
                                "case-control | cross-sectional | case-report | in-vitro | animal-model | "
                                "review | guideline | other\n"
                                '- sample_size: "n=X" or null\n'
                                "- primary_objective: one sentence or null\n"
                                'Return JSON only: {"study_type":"...","sample_size":"...","primary_objective":"..."}'
                            ),
                        },
                        {"role": "user", "content": f"Title: {record.title}\n\nAbstract: {record.abstract[:2000]}"},
                    ],
                    max_tokens=256,
                    temperature=0.1,
                    timeout=180,
                    extra_body=QWEN3_NO_THINK_EXTRA_BODY,
                )
                raw = safe_message_text(resp)
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
        self,
        records: list[WideNetRecord],
        strategy: TieredSearchPlan,
        max_select: int = 20,
        topic: str = "",
        log=logger.info,
    ) -> list[WideNetRecord]:
        """Step 3: Smart model screens wide net records → top 20 with tier-aware priority."""
        if not records:
            log("    [Step 3] No records to screen")
            return []

        pico_str = json.dumps(strategy.pico)

        # Group records by tier
        tier_groups: dict[int, list[WideNetRecord]] = {1: [], 2: [], 3: []}
        for r in records:
            tier = r.research_tier if r.research_tier in (1, 2, 3) else 3
            tier_groups[tier].append(r)

        log(f"    [Step 3] Pool by tier: T1={len(tier_groups[1])}, T2={len(tier_groups[2])}, T3={len(tier_groups[3])}")

        # Screen each tier independently with tier-appropriate intervention
        screened: dict[int, list[WideNetRecord]] = {}
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
                tier_records,
                0,
                _ScreenContext(
                    pico_str=pico_str,
                    max_select=max_select,
                    topic=topic,
                    intervention_override=tier_intervention,
                ),
                log,
            )
            log(f"    [Step 3] Tier {tier_num}: {len(screened[tier_num])} passed screening")

        # Priority fill: Tier 1 → Tier 2 → Tier 3 (with cap)
        t3_available = len(screened[3])
        min_t3 = min(MIN_TIER3_STUDIES, t3_available)
        tier3_cap = int(max_select * MAX_TIER3_RATIO)
        tier12_budget = max_select - min_t3

        selected: list[WideNetRecord] = list(screened[1][:tier12_budget])
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

        log(
            f"    [Step 3] Final selection: T1={sum(1 for s in selected if s.research_tier == 1)}, "
            f"T2={sum(1 for s in selected if s.research_tier == 2)}, "
            f"T3={sum(1 for s in selected if s.research_tier == 3)}, total={len(selected)}"
        )

        return selected

    async def _screen_chunk(
        self, records: list[WideNetRecord], offset: int, ctx: "_ScreenContext", log
    ) -> list[WideNetRecord]:
        """Screen a chunk of records with the smart model."""
        pico_str = ctx.pico_str
        max_select = ctx.max_select
        topic = ctx.topic
        intervention_override = ctx.intervention_override
        compact = []
        for i, r in enumerate(records):
            compact.append(
                {
                    "idx": offset + i,
                    "title": r.title[:150],
                    "type": r.study_type,
                    "n": r.sample_size,
                    "year": r.year,
                    "journal": r.journal,
                    "abstract": r.abstract[:300],
                }
            )

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
            logger.warning(f"Screening parse failed: {e}, applying fallback with keyword filter")
            # Fallback: keyword relevance filter + study type priority
            _kw_words = set(w.lower() for w in re.findall(r"\w{3,}", intervention_text)) - {
                "the",
                "and",
                "for",
                "with",
                "from",
                "pico",
                "intervention",
            }

            def _fallback_relevant(r):
                if not _kw_words:
                    return True
                text = f"{(r.title or '').lower()} {(r.abstract or '')[:500].lower()}"
                return any(w in text for w in _kw_words)

            relevant = [r for r in records if _fallback_relevant(r)]
            if len(relevant) < 5:
                relevant = records  # filter too aggressive, use all
            priority = {
                "meta-analysis": 0,
                "systematic-review": 1,
                "RCT": 2,
                "clinical-trial": 3,
                "cohort": 4,
                "observational": 5,
            }
            sorted_records = sorted(relevant, key=lambda r: priority.get(r.study_type, 99))
            return sorted_records[:max_select]

    @staticmethod
    def _load_extraction_cache(output_dir: str = None) -> dict:
        """Load PMID extraction cache from output_dir/meta/ (with root fallback)."""
        if output_dir:
            cache_path = Path(output_dir) / "meta" / "extraction_cache.json"
            if cache_path.exists():
                try:
                    with open(cache_path) as f:
                        return json.load(f)
                except (json.JSONDecodeError, OSError):
                    return {}
            # Backward compat: check old root-level location
            legacy_path = Path(output_dir) / "extraction_cache.json"
            if legacy_path.exists():
                try:
                    with open(legacy_path) as f:
                        return json.load(f)
                except (json.JSONDecodeError, OSError):
                    return {}
        else:
            # Anchor to project root, not CWD — CLI and web UI launch from different dirs
            cache_path = Path(__file__).resolve().parents[2] / "research_outputs" / "extraction_cache.json"
            if cache_path.exists():
                try:
                    with open(cache_path) as f:
                        return json.load(f)
                except (json.JSONDecodeError, OSError):
                    return {}
        return {}

    @staticmethod
    def _save_extraction_cache(cache: dict, output_dir: str = None):
        """Save PMID extraction cache to output_dir/meta/extraction_cache.json."""
        if output_dir:
            cache_path = Path(output_dir) / "meta" / "extraction_cache.json"
        else:
            # Anchor to project root, not CWD — CLI and web UI launch from different dirs
            cache_path = Path(__file__).resolve().parents[2] / "research_outputs" / "extraction_cache.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)

    @staticmethod
    def _build_findings(data: dict, source_text: str, artifact_id: str) -> list:
        """Turn the model's findings into records Python vouches for.

        Three things the model does NOT get to decide. The ``finding_key`` is computed here from the
        normalised identity tuple — a model-authored key makes replication grouping a semantic task
        again, which is the one thing the key exists to remove. The ``char_offset`` is found by
        searching the source, because a model asked for an offset would be asked to count, and an
        invented number satisfies the contract while pointing nowhere. And a quote that cannot be
        found in the source **drops the finding**: an unverifiable quote is not evidence.
        """
        from dr2_podcast.schemas import compute_finding_key, finding_errors

        built = []
        for raw in data.get("findings") or []:
            if not isinstance(raw, dict):
                continue
            identity = {
                "population": safe_str(raw.get("population")) or "",
                "intervention": safe_str(raw.get("intervention")) or "",
                "comparator": safe_str(raw.get("comparator")) or "",
                "endpoint": safe_str(raw.get("endpoint")) or "",
                "timepoint": safe_str(raw.get("timepoint")),
            }
            if not identity["endpoint"]:
                logger.debug("dropping a finding with no endpoint")
                continue
            quote = safe_str(raw.get("quote")) or ""
            identity_quote = safe_str(raw.get("identity_quote")) or ""
            result_hit = locate_span(source_text, quote) if quote else None
            identity_hit = locate_span(source_text, identity_quote) if identity_quote else None
            if result_hit is None or identity_hit is None:
                logger.warning(
                    "dropping finding %r: a quote is not in the source text", identity["endpoint"][:60]
                )
                continue

            values = {
                "direction": safe_str(raw.get("direction")) or "null_result",
                "value": safe_float(raw.get("value")),
                "unit": safe_str(raw.get("unit")),
                "ci_low": safe_float(raw.get("ci_low")),
                "ci_high": safe_float(raw.get("ci_high")),
                "p_value": safe_float(raw.get("p_value")),
                "control_event_rate": safe_float(raw.get("control_event_rate")),
                "experimental_event_rate": safe_float(raw.get("experimental_event_rate")),
                "outcome_is_adverse": safe_bool(raw.get("outcome_is_adverse")),
            }
            # Field-level coverage, which is why there are two quotes: every claim-bearing field the
            # record actually carries must be named by a locator, and one span cannot honestly
            # substantiate both who was studied and what happened to them.
            identity_fields = [name for name in ("population", "intervention", "comparator", "timepoint")
                               if identity.get(name)]
            result_fields = ["endpoint"] + [name for name, value in values.items() if value is not None]
            finding = Finding(
                **identity,
                **values,
                # safe_bool, not bool(): the model returns JSON, and a JSON string "false" is
                # truthy (prepush codex 2026-08-13). Getting this wrong picks the wrong finding as
                # the paper's primary result at clinical.py:2789, and its CER/EER are what the
                # deterministic ARR/NNT math is computed from. safe_bool refuses to coerce, so only
                # a real boolean true counts; anything else is "not stated", and said out loud
                # rather than silently demoting every finding a string-emitting model produced.
                is_primary=_primary_flag(raw.get("is_primary"), identity["endpoint"]),
                locators=[
                    {
                        "fields": identity_fields,
                        "source_artifact_id": artifact_id,
                        "char_offset": identity_hit[0],
                        # The LITERAL source substring, not the model's rendering of it: the source
                        # may be line-wrapped where the model used a space, and the locator contract
                        # is an exact-substring check at the offset.
                        "quoted_span": identity_hit[1],
                    },
                    {
                        "fields": result_fields,
                        "source_artifact_id": artifact_id,
                        "char_offset": result_hit[0],
                        "quoted_span": result_hit[1],
                    },
                ],
            )
            finding.finding_key = compute_finding_key(identity)
            # The contract is enforced HERE, on the production path, or it is not enforced at all
            # (prepush codex 2026-08-13): schemas/ was written and then only ever called by its own
            # tests, so an unknown `direction`, an empty population, one of CER/EER without the
            # other or a rate outside [0, 1] would flow into primary-result selection, the
            # deterministic ARR/NNT math and the v2 cache. A finding that fails is DROPPED, which is
            # what this function already does with a quote it cannot find in the source — one bad
            # record must not cost the paper its good ones.
            problems = finding_errors(finding.to_dict(), {artifact_id: source_text})
            if problems:
                logger.warning(
                    "dropping finding %r: it does not satisfy the finding contract (%s)",
                    identity["endpoint"][:60],
                    "; ".join(problems[:3]),
                )
                continue
            built.append(finding)
        return built

    @staticmethod
    def _build_funding(data: dict, record: WideNetRecord, source_text: str, artifact_id: str) -> FundingBlock:
        """The five-field funding block, with its provenance stated rather than assumed.

        The extractor already falls back to ``paper_metadata.funding_sources`` when the model finds
        nothing in the text. That fallback is real information and worth keeping — but it is API
        metadata, it appears nowhere in the paper, and it therefore cannot carry a locator. Saying
        which of the two a value came from is the whole point of the split.
        """
        from dr2_podcast.schemas import funding_errors

        raw = safe_str(data.get("funding_raw"))
        quote = safe_str(data.get("funding_quote")) or ""
        category = safe_str(data.get("funding_category")) or "unknown"
        disclosure = safe_str(data.get("funding_disclosure")) or "unknown"

        hit = locate_span(source_text, quote) if (raw and quote) else None
        if raw and hit is not None:
            block = FundingBlock(
                funding_raw=raw,
                funding_category=category if category not in ("undisclosed", "unknown") else "unknown",
                funding_disclosure="disclosed",
                funding_source_type="extracted_text",
                funding_locator={
                    "fields": ["funding_raw"],
                    "source_artifact_id": artifact_id,
                    "char_offset": hit[0],
                    "quoted_span": hit[1],
                },
            )
            # Same reason as the findings above: the contract has to run where the model's output
            # actually enters the pipeline. A block that fails it is not silently downgraded to
            # api_metadata — that would fabricate provenance — it falls through to "unknown".
            problems = funding_errors(block.to_dict(), {artifact_id: source_text})
            if not problems:
                return block
            logger.warning(
                "discarding extracted funding: it does not satisfy the funding contract (%s)",
                "; ".join(problems[:3]),
            )
        if raw:
            # The model produced a funder but no quote that is actually in the paper. It is NOT
            # api_metadata — nothing from an API produced it — and calling it that would fabricate
            # exactly the provenance this split exists to guarantee. The value is discarded; what
            # follows is the real API fallback, or nothing.
            logger.warning("discarding unverifiable funding text %r: its quote is not in the source", raw[:60])

        api = getattr(record.paper_metadata, "funding_sources", None) if record.paper_metadata else None
        if api:
            # The API names a funder, which settles DISCLOSURE. It does not settle CATEGORY, and the
            # model's guess about a statement it could not quote is not evidence of one, so an
            # unusable category stays 'unknown' — a state funding.schema.json now admits on this
            # branch specifically, because the two questions have different answers here.
            block = FundingBlock(
                funding_raw=", ".join(api[:3]),
                # ALWAYS unknown. The category the model returned describes a funding statement it
                # could not quote — the one discarded a few lines above — so carrying it here would
                # attach an unverifiable 'industry' or 'government' label, the exact conflict-of-
                # interest classification the episode reasons about, to a name an API supplied
                # (prepush codex 2026-08-13). The API gives names, not categories.
                funding_category="unknown",
                funding_disclosure="disclosed",
                funding_source_type="api_metadata",
                funding_locator=None,
            )
            problems = funding_errors(block.to_dict(), {artifact_id: source_text})
            if not problems:
                return block
            logger.warning(
                "discarding API funding metadata %r: it does not satisfy the funding contract (%s)",
                (block.funding_raw or "")[:60],
                "; ".join(problems[:3]),
            )
        if disclosure == "undisclosed":
            return FundingBlock(funding_category="undisclosed", funding_disclosure="undisclosed")
        return FundingBlock()

    @staticmethod
    def _cache_entry_still_verifies(cached: DeepExtraction, text: str, artifact_id: str) -> bool:
        """Whether everything a cache entry claims still holds against the text fetched this run.

        Findings AND funding: an ``extracted_text`` funding block carries a locator into the same
        source, and checking only the findings would let a stale, unquotable funder — the conflict
        of interest the episode reasons about — ride through on a cache hit.
        """
        from dr2_podcast.schemas import finding_errors, funding_errors

        if not text.strip():
            # Nothing to check it against. The cached record is all there is, and refusing it would
            # only trade a verified-when-written extraction for no extraction at all.
            return True
        artifacts = {artifact_id: text}
        if any(finding_errors(f.to_dict(), artifacts) for f in cached.findings):
            return False
        funding = cached.funding
        return not (funding is not None and funding_errors(funding.to_dict(), artifacts))

    @staticmethod
    def _extraction_from_cache(record: WideNetRecord, cached: dict) -> DeepExtraction:
        """Reconstruct a DeepExtraction from cached data."""
        return DeepExtraction(
            pmid=record.pmid,
            doi=record.doi,
            title=record.title,
            url=record.url,
            attrition_pct=cached.get("attrition_pct"),
            effect_size=cached.get("effect_size"),
            demographics=cached.get("demographics"),
            follow_up_period=cached.get("follow_up_period"),
            funding_source=cached.get("funding_source"),
            conflicts_of_interest=cached.get("conflicts_of_interest"),
            biological_mechanism=cached.get("biological_mechanism"),
            control_event_rate=cached.get("cer"),
            experimental_event_rate=cached.get("eer"),
            outcome_is_adverse=cached.get("outcome_is_adverse"),
            primary_outcome=cached.get("primary_outcome"),
            secondary_outcomes=cached.get("secondary_outcomes"),
            blinding=cached.get("blinding"),
            randomization_method=cached.get("randomization_method"),
            intention_to_treat=cached.get("intention_to_treat"),
            sample_size_total=cached.get("sample_size_total"),
            sample_size_intervention=cached.get("sample_size_intervention"),
            sample_size_control=cached.get("sample_size_control"),
            study_design=cached.get("study_design"),
            risk_of_bias=cached.get("risk_of_bias"),
            research_tier=record.research_tier,
            raw_facts=cached.get("raw_facts", ""),
            # Step 9a. Rebuilt, not skipped: a cache hit that dropped these would silently lose the
            # structured findings and provenance the first run paid to extract, and the paper would
            # contribute nothing on every subsequent run — the exact failure the v2 key was for.
            findings=[Finding.from_dict(f) for f in (cached.get("findings") or []) if isinstance(f, dict)],
            funding=FundingBlock.from_dict(cached.get("funding")),
            trial_registration=cached.get("trial_registration"),
            author_group=cached.get("author_group"),
            paper_metadata=(
                PaperMetadata.from_dict(cached["paper_metadata"])
                if isinstance(cached.get("paper_metadata"), dict)
                else record.paper_metadata
            ),
        )

    @staticmethod
    def _cache_extraction(extraction: DeepExtraction) -> dict:
        """Convert a DeepExtraction to cache-friendly dict."""
        d = extraction.to_dict()
        # Rename CER/EER keys for clarity in cache
        d["cer"] = d.pop("control_event_rate", None)
        d["eer"] = d.pop("experimental_event_rate", None)
        d["cached_at"] = datetime.datetime.now().strftime("%Y-%m-%d")
        # Remove identifiers (stored as cache key or reconstructed from record)
        for k in ("pmid", "doi", "title", "url"):
            d.pop(k, None)
        return d

    async def _deep_extract_batch(
        self, articles, records: list[WideNetRecord], pico: dict[str, str], log=logger.info, output_dir: str = None
    ) -> list[DeepExtraction]:
        """Step 4: Extract clinical variables from full-text articles using the Smart model.
        Uses PMID-keyed cache to ensure identical NNT across runs for the same paper."""
        log(f"    [Step 4] Deep extraction from {len(articles)} articles (Smart Model)...")
        semaphore = asyncio.Semaphore(3)  # Reduced from 6 to avoid overloading model server
        is_social = getattr(self, "_domain", "clinical") == "social_science"

        # Load extraction cache
        extraction_cache = self._load_extraction_cache(output_dir)
        cache_hits = 0

        async def extract_one(article, record: WideNetRecord) -> DeepExtraction:
            nonlocal cache_hits
            # Check cache first (PMID-keyed)
            # v2 — Step 9a. Without the prefix, a cached v1 entry deserialises into a record with
            # no findings[] and no funding block, silently, and the paper simply contributes nothing.
            cache_key = f"v2:{record.pmid or record.doi or ''}" if (record.pmid or record.doi) else ""
            text = getattr(article, "full_text", "") or ""
            # Fall back to abstract if full-text is empty or too short
            if len(text.strip()) < 200 and record.abstract:
                text = record.abstract
            artifact_id = f"pmid:{record.pmid}" if record.pmid else (record.doi or record.url or record.title)
            if cache_key and cache_key in extraction_cache:
                cached = self._extraction_from_cache(record, extraction_cache[cache_key])
                # A locator is a claim about THIS text. The cache is keyed by PMID/DOI, and the same
                # paper fetched from a different provider — PMC one run, a publisher scrape the next
                # — is a different string, so the stored offsets can point at the wrong words or
                # nowhere at all while their CER/EER keep feeding the ARR/NNT math (prepush codex
                # 2026-08-13). Re-checked against what this run actually fetched; an entry that no
                # longer verifies is not this paper's extraction, and the paper is re-extracted.
                if self._cache_entry_still_verifies(cached, text, artifact_id):
                    cache_hits += 1
                    log(f"    [Cache hit] {record.title[:50]}...")
                    return cached
                logger.warning(
                    "cache entry for %r no longer verifies against the text fetched this run; "
                    "re-extracting",
                    (record.title or record.pmid or "")[:60],
                )
            if not text.strip():
                return DeepExtraction(
                    pmid=record.pmid,
                    doi=record.doi,
                    title=record.title,
                    url=record.url,
                    research_tier=record.research_tier,
                    raw_facts="No content available",
                )

            async with semaphore:
                try:
                    content = text[:_SMART_CONTENT_CHARS]
                    if is_social:
                        system_prompt = (
                            "You are a social science data extraction specialist. Read this study and extract "
                            "ALL of the following variables. Use null for any field not found.\n\n"
                            "Return ONLY valid JSON:\n"
                            "{\n"
                            '  "effect_size": "0.45 (Cohen\'s d) or null",\n'
                            '  "effect_size_type": "Cohen\'s d | Hedges\' g | OR | r | eta-squared | null",\n'
                            '  "group_1_mean": 3.2,\n'
                            '  "group_1_sd": 1.1,\n'
                            '  "group_1_n": 150,\n'
                            '  "group_2_mean": 2.8,\n'
                            '  "group_2_sd": 1.0,\n'
                            '  "group_2_n": 148,\n'
                            '  "demographics": "age, sex, population or null",\n'
                            '  "follow_up_period": "duration or null",\n'
                            '  "funding_source": "source or null",\n'
                            '  "study_design": "RCT | quasi-experimental | cohort | cross-sectional | meta-analysis | etc.",\n'
                            '  "sample_size_total": 298,\n'
                            '  "primary_outcome": "exact outcome or null",\n'
                            '  "risk_of_bias": "low | some concerns | high | unclear",\n'
                            '  "raw_facts": "3-5 key findings"\n'
                            "}"
                        )
                    else:
                        system_prompt = (
                            "You are a clinical data extraction specialist. Read this study and extract "
                            "ALL of the following variables. Use null for any field not found.\n\n"
                            "IMPORTANT for control_event_rate / experimental_event_rate:\n"
                            "- The 'event' MUST be an ADVERSE or NEGATIVE outcome (e.g., disease incidence, "
                            "mortality, hospitalization, weight gain, metabolic worsening).\n"
                            "- If the study measures a POSITIVE outcome (e.g., weight loss, improvement, "
                            "remission), INVERT it: report the proportion who did NOT improve.\n"
                            "- Example: if 63% of experimental group lost weight, the adverse event "
                            "(no weight loss) rate is 0.37.\n"
                            "- Set outcome_is_adverse to false ONLY if you could not invert and are "
                            "reporting a beneficial event rate directly.\n\n"
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
                            '  "outcome_is_adverse": true,\n'
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
                            '  "raw_facts": "3-5 key findings as bullet points",\n'
                            '  "trial_registration": "NCT/UMIN identifier or null",\n'
                            '  "author_group": "first author + institution, e.g. Tanaka H; Osaka University",\n'
                            # One JSON string per line. Wrapping a description across source lines
                            # renders as two adjacent quoted fragments in the prompt the model
                            # actually reads — `"...or null" " if the paper is silent"` — which is
                            # not valid JSON, and a model copying the template's shape returns
                            # something the parser rejects (prepush codex 2026-08-13).
                            '  "funding_raw": "verbatim funding statement as printed, or null if silent",\n'
                            '  "funding_category": "industry | government | foundation | institutional '
                            '| mixed | none_declared | undisclosed | unknown",\n'
                            '  "funding_disclosure": "disclosed | undisclosed | unknown",\n'
                            '  "funding_quote": "the exact sentence the funding statement appears in, or null",\n'
                            '  "findings": [\n'
                            "    {\n"
                            '      "population": "who was studied",\n'
                            '      "intervention": "what they received",\n'
                            '      "comparator": "what it was compared against",\n'
                            '      "endpoint": "the outcome measured",\n'
                            '      "timepoint": "when it was measured, or null",\n'
                            '      "direction": "increase | decrease | null_result",\n'
                            '      "value": 5.0, "unit": "%", "ci_low": 2.0, "ci_high": 8.0, "p_value": 0.03,\n'
                            '      "is_primary": true,\n'
                            '      "control_event_rate": 0.15, "experimental_event_rate": 0.10,\n'
                            '      "outcome_is_adverse": true,\n'
                            '      "identity_quote": "the exact sentence establishing WHO was studied, '
                            'what they received and what it was compared against",\n'
                            '      "quote": "the exact sentence from the paper that states this result"\n'
                            "    }\n"
                            "  ]\n"
                            "}\n\n"
                            "ONE ENTRY IN findings PER (population, intervention, comparator, endpoint, timepoint).\n"
                            "A paper that reports benefit on one endpoint and no effect on another has TWO entries. "
                            "Do NOT collapse them, and do NOT invent an entry for an endpoint\n"
                            "the paper does not report.\n"
                            "BOTH quotes MUST be copied VERBATIM from the study text above. Each is checked "
                            "against the source, and a finding whose quotes cannot be found there is DISCARDED. "
                            "Two quotes because the facts live in two places: who was studied is in the "
                            "methods, what happened to them is in the results."
                        )

                    # Extraction has always run on the Smart model: a 9B was judged too
                    # small for reliable 19-field JSON extraction from full text. The Fast
                    # model was removed entirely 2026-08-10; this comment is kept because the
                    # *reason* still governs any future attempt to demote this call.
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Title: {record.title}\n\nContent:\n{content}"},
                    ]
                    try:
                        resp = await gated_create(
                            self.smart_client,
                            model=self.smart_model,
                            messages=messages,
                            max_tokens=2048,
                            temperature=0.1,
                            timeout=180,
                            extra_body=QWEN3_NO_THINK_EXTRA_BODY,
                        )
                    except openai.BadRequestError as ctx_err:
                        if "context length" not in str(ctx_err).lower():
                            raise
                        # Token-dense content (tables, citations) — retry with quarter
                        content = content[: len(content) // 4]
                        logger.info(
                            f"    Context length exceeded, retrying with {len(content)} chars for {record.title[:50]}"
                        )
                        messages[1]["content"] = f"Title: {record.title}\n\nContent:\n{content}"
                        resp = await gated_create(
                            self.smart_client,
                            model=self.smart_model,
                            messages=messages,
                            max_tokens=2048,
                            temperature=0.1,
                            timeout=180,
                            extra_body=QWEN3_NO_THINK_EXTRA_BODY,
                        )

                    raw = safe_message_text(resp)
                    data = self._parse_json_response(raw)

                    def safe_list(v):
                        if isinstance(v, list):
                            return v
                        return None

                    # Step 9a. findings[] is the real result set; the paper-level effect fields below
                    # are DERIVED from the primary finding so every consumer keeps working until it
                    # is migrated. Deriving rather than parsing them twice is what keeps the two from
                    # disagreeing about the same paper.
                    findings = self._build_findings(data, text, artifact_id)
                    funding = self._build_funding(data, record, text, artifact_id)
                    primary = next((f for f in findings if f.is_primary), findings[0] if findings else None)
                    # extraction.schema.json requires at least one finding and calls a paper with
                    # none an extraction FAILURE. That is the clinical contract; the social-science
                    # prompt does not ask for findings at all, so only the clinical path is judged
                    # by it. Not raising here — the record's narrative fields are still usable and
                    # dropping the paper outright would shrink the corpus on a model hiccup — but it
                    # is said out loud, it contributes no rates (above), and it is not cached, so a
                    # rerun retries instead of inheriting the failure.
                    if not is_social and not findings:
                        logger.warning(
                            "no verified finding survived extraction for %r; it contributes no "
                            "event rates to the clinical math",
                            (record.title or record.pmid or "")[:60],
                        )

                    extraction = DeepExtraction(
                        pmid=record.pmid,
                        doi=record.doi,
                        title=record.title,
                        url=record.url,
                        attrition_pct=safe_str(data.get("attrition_pct")),
                        effect_size=safe_str(data.get("effect_size")),
                        demographics=safe_str(data.get("demographics")),
                        follow_up_period=safe_str(data.get("follow_up_period")),
                        # From the VALIDATED block, never from data["funding_source"]. _build_case
                        # injects this legacy field into the synthesis prompt as "Funding", so
                        # reading the raw model response here handed the episode exactly the
                        # unverifiable claim _build_funding had just discarded (prepush codex
                        # 2026-08-13). Slice 2 removes the field; until then it mirrors the block.
                        funding_source=funding.funding_raw,
                        conflicts_of_interest=safe_str(data.get("conflicts_of_interest")),
                        biological_mechanism=safe_str(data.get("biological_mechanism")),
                        # ONLY from a verified finding. The fallback these three used to have read
                        # the model's paper-level numbers directly — no quote, no locator, nothing
                        # a Python check could refute — and fed them to the deterministic ARR/NNT
                        # math, which is the one place in this pipeline where an unverified number
                        # becomes a number the episode states out loud (prepush codex 2026-08-13).
                        # A paper whose every finding failed verification contributes no rates.
                        control_event_rate=primary.control_event_rate if primary else None,
                        experimental_event_rate=primary.experimental_event_rate if primary else None,
                        outcome_is_adverse=primary.outcome_is_adverse if primary else None,
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
                        research_tier=record.research_tier,
                        raw_facts=safe_str(data.get("raw_facts")) or "",
                        paper_metadata=record.paper_metadata,
                        findings=findings,
                        funding=funding,
                        trial_registration=safe_str(data.get("trial_registration")),
                        author_group=safe_str(data.get("author_group")),
                    )
                    return extraction
                except Exception as e:
                    logger.warning(f"Deep extraction failed for {record.title[:50]}: {e}")
                    return DeepExtraction(
                        pmid=record.pmid,
                        doi=record.doi,
                        title=record.title,
                        url=record.url,
                        research_tier=record.research_tier,
                        raw_facts=f"Extraction failed: {str(e)[:100]}",
                        paper_metadata=record.paper_metadata,
                    )

        results = await asyncio.gather(*[extract_one(art, rec) for art, rec in zip(articles, records, strict=True)])
        good = sum(1 for r in results if r.raw_facts and "failed" not in r.raw_facts.lower())
        log(f"    [Step 4] Extracted data from {good}/{len(results)} articles (cache hits: {cache_hits})")

        self._persist_new_extractions(
            list(results), extraction_cache, output_dir, log, findings_required=not is_social
        )
        return list(results)

    def _persist_new_extractions(
        self, results, extraction_cache: dict, output_dir, log, *, findings_required: bool
    ) -> None:
        """Remember the extractions worth remembering, and only those.

        ``findings_required`` is the contract the batch ran under — the clinical prompt asks for
        findings and the social-science prompt does not — and it arrives as an argument rather than
        riding on each record, because it describes the extraction, not the paper.
        """
        new_cache_entries: dict[str, Any] = {}
        for r in results:
            cache_key = f"v2:{r.pmid or r.doi or ''}" if (r.pmid or r.doi) else ""
            # A clinical extraction that produced no verified finding is not a result worth
            # remembering: caching it would make the next run inherit the failure without retrying.
            unverified = findings_required and not r.findings
            if (
                cache_key
                and cache_key not in extraction_cache
                and "failed" not in (r.raw_facts or "").lower()
                and not unverified
            ):
                new_cache_entries[cache_key] = self._cache_extraction(r)
        if new_cache_entries:
            extraction_cache.update(new_cache_entries)
            self._save_extraction_cache(extraction_cache, output_dir)
            log(f"    [Step 4] Saved {len(new_cache_entries)} new extractions to cache")

    async def _build_case(
        self, topic: str, strategy: TieredSearchPlan, extractions: list[DeepExtraction], case_type: str, log=logger.info
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
            # Per FINDING, not per paper. Serialising one CER/EER pair meant the case synthesis
            # only ever saw the primary endpoint: a study reporting benefit on fractures and no
            # effect on falls presented as unambiguous support, and the secondary result — the one
            # a falsification case most needs — never reached the model at all.
            block += _findings_block(ex)
            if ex.demographics:
                block += f"  Demographics: {ex.demographics}\n"
            if ex.follow_up_period:
                block += f"  Follow-up: {ex.follow_up_period}\n"
            if ex.blinding:
                block += f"  Blinding: {ex.blinding}\n"
            if ex.risk_of_bias:
                block += f"  Risk of bias: {ex.risk_of_bias}\n"
            block += _funding_line(ex)
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
                max_tokens=6000,
                temperature=0.2,
            )
            has_synthetic, flagged = _detect_synthetic_citations(report, extractions)
            if has_synthetic:
                warning = (
                    "\n\n---\n"
                    "⚠ **SYNTHETIC CITATION WARNING** — the following references were cited but "
                    f"not found in the {len(extractions)} retrieved studies. "
                    "They may be hallucinated:\n" + "\n".join(f"- {r}" for r in flagged) + "\n---\n"
                )
                report = warning + report
                if not extractions:
                    logger.critical(f"SYNTHETIC CITATIONS DETECTED (0 studies input): {flagged}")
            return report
        except Exception as e:
            logger.error(f"Build case ({case_type}) failed: {e}")
            return f"# {case_type.title()} Case: {topic}\n\n*Case synthesis failed ({e}).*\n\n{extractions_text}"


_CITATION_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+et\s+al\.)?)\s+\((\d{4})\)")


def _detect_synthetic_citations(report: str, extractions: list) -> tuple:
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
    known = " ".join(f"{ex.title or ''} {ex.raw_facts or ''}" for ex in extractions).lower()
    flagged = [f"{a} ({y})" for a, y in found if a.replace("et al.", "").strip().lower() not in known]
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

    #: The structured GRADE record, set at step 7. None until then, and None for social science,
    #: which has an evidence-quality ladder rather than GRADE's modifiers.
    grade_record: dict | None = None

    def __init__(self, config: "ResearchConfig | None" = None):
        config = config or ResearchConfig()
        smart_base_url = config.smart_base_url
        smart_model = config.smart_model
        brave_api_key = config.brave_api_key
        results_per_query = config.results_per_query
        max_iterations = config.max_iterations
        domain = config.domain

        self.domain = domain
        self.smart_client = AsyncOpenAI(base_url=smart_base_url, api_key="NA")
        self.smart_model = smart_model

        summary_worker = SummaryWorker(self.smart_client, smart_model)
        search_svc = SearchService(brave_api_key)
        self._page_cache = PageCache()
        fetcher = ContentFetcher(max_concurrent=15, cache=self._page_cache)

        deps = AgentDeps(
            smart_client=self.smart_client,
            summary_worker=summary_worker,
            search_service=search_svc,
            fetcher=fetcher,
        )
        self.lead_researcher = ResearchAgent(deps, smart_model, results_per_query, max_iterations)
        self.counter_researcher = ResearchAgent(deps, smart_model, results_per_query, max_iterations)

        # Set domain on researchers for Step 2 dispatch
        self.lead_researcher._domain = self.domain
        self.counter_researcher._domain = self.domain

        # Social science clients (OpenAlex + ERIC)
        if domain == "social_science":
            from dr2_podcast.research.metadata_clients import OpenAlexClient, ERICClient, MetadataCache

            self._metadata_cache = MetadataCache()
            self.openalex = OpenAlexClient(cache=self._metadata_cache)
            self.eric = ERICClient(cache=self._metadata_cache)
            self.lead_researcher._openalex = self.openalex
            self.lead_researcher._eric = self.eric
            self.counter_researcher._openalex = self.openalex
            self.counter_researcher._eric = self.eric

        # Full-text fetcher for Step 4
        from dr2_podcast.research.fulltext_fetcher import FullTextFetcher

        self.fulltext_fetcher = FullTextFetcher(max_concurrent=5, cache=self._page_cache)

    def _run_step6_math(self, all_extractions: list, log) -> tuple:
        """Step 6: deterministic effect sizes (social) or ARR/NNT (clinical).

        Returns (impacts, math_report). No LLM is involved either way.
        """
        from dr2_podcast.research import clinical_math

        rule = "=" * 70
        log(f"\n{rule}")
        if self.domain == "social_science":
            from dr2_podcast.research.effect_size_math import (
                batch_calculate as es_batch_calculate,
                format_effect_size_report,
            )

            log("STEP 6: DETERMINISTIC MATH (Effect Size)")
            log(rule)
            impacts = es_batch_calculate(all_extractions)
            math_report = format_effect_size_report(impacts)
            log(f"    Calculated effect sizes for {len(impacts)} studies")
            return impacts, math_report

        log("STEP 6: DETERMINISTIC MATH (ARR/NNT)")
        log(rule)
        impacts = clinical_math.batch_calculate(all_extractions)
        math_report = clinical_math.format_math_report(impacts)
        log(f"    Calculated clinical impact for {len(impacts)} studies with CER+EER data")
        for imp in impacts:
            log(f"      {imp.study_id}: NNT={imp.nnt:.1f} ({imp.direction})")
        return impacts, math_report

    def _databases_searched(self) -> list:
        if self.domain == "social_science":
            return ["OpenAlex", "ERIC", "Google Scholar"]
        return ["PubMed", "Google Scholar"]

    def _combined_metrics(self, search_date, db_list, aff: _TrackResult, fal: _TrackResult) -> SearchMetrics:
        """Both tracks' metrics summed, for the auditor's report."""
        total_wide = aff.wide_net_total + fal.wide_net_total
        total_ft_ok = aff.fulltext_ok + fal.fulltext_ok
        total_ft_err = aff.fulltext_err + fal.fulltext_err
        return SearchMetrics(
            search_date=search_date,
            databases_searched=db_list,
            total_identified=total_wide,
            total_after_dedup=total_wide,  # dedup happens inside PubMedClient
            total_fetched=total_ft_ok + total_ft_err,
            total_fetch_errors=total_ft_err,
            total_with_content=total_ft_ok,
            total_summarized=len(aff.extractions) + len(fal.extractions),
            academic_sources=total_wide,
            general_web_sources=0,
            wide_net_total=total_wide,
            screened_in=aff.screened_in + fal.screened_in,
            fulltext_retrieved=total_ft_ok,
            fulltext_errors=total_ft_err,
        )

    async def _run_step7_grade(self, topic, aff: _TrackResult, fal: _TrackResult, math_report, search_date, log):
        """Step 7: GRADE synthesis over both tracks' totals."""
        rule = "=" * 70
        log(f"\n{rule}")
        log("STEP 7: GRADE SYNTHESIS")
        log(rule)
        return await self._grade_synthesis(topic, aff, fal, math_report, search_date, log)

    def _log_run_summary(self, start_time, aff: _TrackResult, fal: _TrackResult, impacts, all_extractions, log):
        rule = "=" * 70
        log(f"\n{rule}")
        log(f"ALL RESEARCH COMPLETE in {time.time() - start_time:.0f}s")
        log(f"  Affirmative: {len(aff.extractions)} studies from {aff.wide_net_total} candidates")
        log(f"  Falsification: {len(fal.extractions)} studies from {fal.wide_net_total} candidates")
        math_label = "Effect size math" if self.domain == "social_science" else "Clinical math"
        math_detail = "effect size data" if self.domain == "social_science" else "NNT data"
        # "findings", not "studies": since slice 2 the math is computed per finding, so a paper
        # reporting two endpoints contributes two rows and this count is no longer a study count.
        log(f"  {math_label}: {len(impacts)} findings with {math_detail}")
        log(f"  Total articles analyzed: {len(all_extractions)}")
        log(f"{rule}\n")

    def _build_track_report(self, role: str, track: _TrackResult, search_date, db_list, duration) -> ResearchReport:
        """Wrap one track's outputs in the backward-compatible ResearchReport."""
        sources = self._extractions_to_sources(track.extractions, role.lower().split()[0])
        fetched = track.fulltext_ok + track.fulltext_err
        return ResearchReport(
            topic=self._current_topic,
            role=role,
            sources=sources,
            report=track.case_report,
            iterations_used=0,
            total_urls_fetched=fetched,
            total_summaries=len(track.extractions),
            total_errors=track.fulltext_err,
            duration_seconds=duration,
            search_metrics=SearchMetrics(
                search_date=search_date,
                databases_searched=db_list,
                total_identified=track.wide_net_total,
                total_after_dedup=track.wide_net_total,
                total_fetched=fetched,
                total_fetch_errors=track.fulltext_err,
                total_with_content=track.fulltext_ok,
                total_summarized=len(track.extractions),
                academic_sources=track.wide_net_total,
                general_web_sources=0,
                wide_net_total=track.wide_net_total,
                screened_in=track.screened_in,
                fulltext_retrieved=track.fulltext_ok,
                fulltext_errors=track.fulltext_err,
            ),
        )

    async def _run_research_track(
        self, spec: _TrackSpec, topic: str, framing_context: str, decomposition, output_dir, log
    ) -> _TrackResult:
        """Steps 1-5 for one track.

        The affirmative and falsification tracks are the same five steps; only
        the researcher, the two role strings and the log labels differ, which is
        what _TrackSpec carries.
        """
        researcher = getattr(self, spec.researcher_attr)
        n, label = spec.step_suffix, spec.label
        rule = "=" * 70

        log(f"\n{rule}")
        log(f"STEP 1{n}: TIERED KEYWORD GENERATION + AUDITOR GATE ({label})")
        log(rule)
        plan = await researcher._formulate_tiered_strategy(
            topic, spec.strategy_role, framing_context, decomposition, log=log
        )

        log(f"\n{rule}")
        log(f"STEP 2{n}: TIERED CASCADE SEARCH ({label})")
        log(rule)
        records, highest_tier = await researcher._tiered_search(plan, log)

        log(f"\n{rule}")
        log(f"STEP 3{n}: SCREENING ({len(records)} → top 20) ({label})")
        log(rule)
        top_records = await researcher._screen_and_prioritize(records, plan, topic=topic, log=log)
        screened_in = len(top_records)

        # Metadata enrichment (optional — degrades gracefully)
        top_records = await self._enrich_with_metadata(top_records, log)
        # Filter out retracted papers
        retracted = [r for r in top_records if r.paper_metadata and r.paper_metadata.is_retracted]
        if retracted:
            log(f"    ⚠ Filtering out {len(retracted)} retracted paper(s) from {label.lower()} track")
            top_records = [r for r in top_records if not (r.paper_metadata and r.paper_metadata.is_retracted)]

        log(f"\n{rule}")
        log(f"STEP 4{n}: DEEP EXTRACTION ({len(top_records)} articles) ({label})")
        log(rule)
        fulltexts = await self.fulltext_fetcher.fetch_all(top_records)
        fulltext_ok = sum(1 for ft in fulltexts if not ft.error)
        fulltext_err = sum(1 for ft in fulltexts if ft.error)
        log(f"    Full-text retrieved: {fulltext_ok}/{len(fulltexts)}")

        extractions = await researcher._deep_extract_batch(
            fulltexts, top_records, plan.pico, log, output_dir=output_dir
        )

        log(f"\n{rule}")
        log(f"STEP 5{n}: {label.upper()} CASE")
        log(rule)
        case_report = await researcher._build_case(topic, plan, extractions, spec.case_role, log)

        return _TrackResult(
            plan=plan,
            records=records,
            top_records=top_records,
            extractions=extractions,
            case_report=case_report,
            highest_tier=highest_tier,
            wide_net_total=len(records),
            screened_in=screened_in,
            fulltext_ok=fulltext_ok,
            fulltext_err=fulltext_err,
        )

    async def run(
        self, topic: str, framing_context: str = "", progress_callback=None, output_dir: str = None
    ) -> dict[str, ResearchReport]:
        """Run the full 7-step clinical research pipeline.

        Args:
            topic: Research topic
            framing_context: Optional research framing document to guide searches
            progress_callback: Optional callback for progress messages
            output_dir: Optional directory to save intermediate artifacts

        Returns:
            Dict[str, ResearchReport] with keys: "lead", "counter", "audit"
        """
        if not SMART_MODEL or not SMART_BASE_URL:
            raise RuntimeError(
                "MODEL_NAME and LLM_BASE_URL environment variables must be set before running the pipeline"
            )
        start_time = time.time()

        def log(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        log(f"\n{'=' * 70}")
        domain_label = "Social Science" if self.domain == "social_science" else "Clinical"
        log(f"DEEP RESEARCH AGENT - Evidence-Based {domain_label} Pipeline")
        log(f"{'=' * 70}")
        log(f"Topic: {topic}")
        if framing_context:
            log(f"Research framing provided: {len(framing_context)} chars")
        log(f"{'=' * 70}")

        # --- Phase 0: Concept Decomposition (C2) ---
        log(f"\n{'=' * 70}")
        log("PHASE 0: CONCEPT DECOMPOSITION")
        log(f"{'=' * 70}")
        decomposition = await self.lead_researcher._decompose_topic(topic, framing_context)
        if decomposition.get("canonical_terms"):
            log(f"  Canonical terms: {', '.join(decomposition['canonical_terms'])}")
        if decomposition.get("related_concepts"):
            log(f"  Related concepts: {', '.join(decomposition['related_concepts'])}")

        # --- Run both tracks in parallel ---
        log(f"\n{'=' * 70}")
        log("RUNNING AFFIRMATIVE & FALSIFICATION TRACKS IN PARALLEL")
        log(f"{'=' * 70}")

        aff, fal = await asyncio.gather(
            self._run_research_track(_AFFIRMATIVE_TRACK, topic, framing_context, decomposition, output_dir, log),
            self._run_research_track(_FALSIFICATION_TRACK, topic, framing_context, decomposition, output_dir, log),
        )

        aff_strategy, fal_strategy = aff.plan, fal.plan
        aff_top, fal_top = aff.top_records, fal.top_records
        aff_extractions, fal_extractions = aff.extractions, fal.extractions
        aff_fulltext_ok, aff_fulltext_err = aff.fulltext_ok, aff.fulltext_err
        fal_fulltext_ok, fal_fulltext_err = fal.fulltext_ok, fal.fulltext_err

        all_extractions = aff_extractions + fal_extractions
        impacts, math_report = self._run_step6_math(all_extractions, log)

        search_date = datetime.date.today().isoformat()
        audit_text = await self._run_step7_grade(topic, aff, fal, math_report, search_date, log)

        if output_dir:
            self._save_artifacts(output_dir, aff, fal, math_report)

        # --- Build backward-compatible return ---
        # Convert extractions to SummarizedSource for compatibility
        aff_sources = self._extractions_to_sources(aff_extractions, "affirmative")
        fal_sources = self._extractions_to_sources(fal_extractions, "falsification")

        db_list = self._databases_searched()
        combined_metrics = self._combined_metrics(search_date, db_list, aff, fal)

        lead_duration = time.time() - start_time
        self._current_topic = topic
        lead_report = self._build_track_report("Lead Researcher", aff, search_date, db_list, lead_duration)
        counter_report = self._build_track_report("Counter Researcher", fal, search_date, db_list, lead_duration)

        audit_report = ResearchReport(
            topic=topic,
            role="Auditor",
            sources=aff_sources + fal_sources,
            report=audit_text,
            iterations_used=0,
            total_urls_fetched=(aff_fulltext_ok + aff_fulltext_err + fal_fulltext_ok + fal_fulltext_err),
            total_summaries=len(aff_extractions) + len(fal_extractions),
            total_errors=aff_fulltext_err + fal_fulltext_err,
            duration_seconds=time.time() - start_time,
            search_metrics=combined_metrics,
        )

        self._log_run_summary(start_time, aff, fal, impacts, all_extractions, log)

        # Close social science clients if used
        if self.domain == "social_science":
            await self.openalex.close()
            await self.eric.close()
            self._metadata_cache.close()

        return {
            "lead": lead_report,
            "counter": counter_report,
            "audit": audit_report,
            # Raw pipeline data for IMRaD SOT assembly (additive — backward-compatible)
            "pipeline_data": {
                "domain": self.domain,
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
                # The structured record behind grade_synthesis.md. It travels with the prose, so a
                # consumer never has to scrape the prose to learn what the prose decided.
                "grade_record": self.grade_record,
                "aff_highest_tier": aff.highest_tier,
                "fal_highest_tier": fal.highest_tier,
                "metrics": {
                    "aff_wide_net_total": aff.wide_net_total,
                    "aff_screened_in": aff.screened_in,
                    "aff_fulltext_ok": aff.fulltext_ok,
                    "aff_fulltext_err": aff.fulltext_err,
                    "fal_wide_net_total": fal.wide_net_total,
                    "fal_screened_in": fal.screened_in,
                    "fal_fulltext_ok": fal.fulltext_ok,
                    "fal_fulltext_err": fal.fulltext_err,
                },
            },
        }

    async def _grade_synthesis(
        self,
        topic: str,
        aff: _TrackResult,
        fal: _TrackResult,
        math_report: str,
        search_date: str,
        log=logger.info,
    ) -> str:
        """Step 7: GRADE / Evidence Quality synthesis by the Auditor.

        Reads the two tracks' cases, plans, extractions and PRISMA counts off
        their _TrackResult rather than taking fourteen positional arguments.
        """
        aff_case, fal_case = aff.case_report, fal.case_report
        aff_strategy = aff.plan
        aff_extractions, fal_extractions = aff.extractions, fal.extractions
        total_wide = aff.wide_net_total + fal.wide_net_total
        total_screened = aff.screened_in + fal.screened_in
        total_ft_ok = aff.fulltext_ok + fal.fulltext_ok
        total_ft_err = aff.fulltext_err + fal.fulltext_err

        synthesis_label = "Evidence Quality" if self.domain == "social_science" else "GRADE"
        log(f"    [Step 7] {synthesis_label} synthesis...")

        pico_str = json.dumps(aff_strategy.pico)

        if self.domain == "social_science":
            audit_system = (
                "/no_think\n"
                "You are The Auditor — an independent scientific arbiter.\n\n"
                "You have received:\n"
                "1. The AFFIRMATIVE CASE (arguing FOR the exposure/factor)\n"
                "2. The FALSIFICATION CASE (arguing AGAINST the exposure/factor)\n"
                "3. DETERMINISTIC MATH (Python-calculated effect sizes — these numbers are EXACT, not LLM-generated)\n\n"
                f"PECO Framework: {pico_str}\n\n"
                "Your task: Issue an Evidence Quality synthesis.\n\n"
                "Evidence Quality Levels:\n"
                "- STRONG: Systematic reviews/meta-analyses of controlled studies\n"
                "- MODERATE_STRONG: RCTs (rare in social science)\n"
                "- MODERATE: Quasi-experimental (DiD, regression discontinuity)\n"
                "- MODERATE_WEAK: Cohort/longitudinal studies\n"
                "- WEAK: Cross-sectional/correlational\n"
                "- VERY_WEAK: Case studies/qualitative/expert opinion\n\n"
                "Structure:\n"
                "1. Evidence Quality Assessment\n"
                "2. Strength of Evidence\n"
                "3. Consistency Across Studies\n"
                "4. Effect Size Summary\n"
                "   - Include the effect size table directly (do NOT recalculate — use the exact numbers provided)\n"
                "5. Methodological Limitations\n"
                "6. Recommendations\n\n"
                "7. PRISMA Flow Diagram (text-based)\n"
                f"   Records identified: {total_wide}\n"
                f"   Screened (top studies): {total_screened}\n"
                f"   Full-text retrieved: {total_ft_ok}\n"
                f"   Full-text errors: {total_ft_err}\n"
                f"   Included in synthesis: {total_screened}\n\n"
                "8. Executive Summary\n"
                "9. Final Evidence Quality: [STRONG/MODERATE/WEAK/VERY_WEAK]\n\n"
                "CRITICAL RULES:\n"
                "- NEVER recalculate effect sizes — use the Python-provided numbers exactly\n"
                "- Be heavily caveated — acknowledge uncertainty\n"
                "- Flag any potential conflicts of interest\n"
                "- Distinguish between statistical significance and practical significance\n"
                "- Note that absence of evidence is not evidence of absence"
            )
        else:
            audit_system = (
                "/no_think\n"
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
                "   FINAL GRADE: HIGH | MODERATE | LOW | VERY LOW\n"
                "   Then close the section with this block, exactly in this form, listing every\n"
                "   modifier you APPLIED and nothing else. A domain you considered and did not apply\n"
                "   does not belong here. Steps are 1 (serious) or 2 (very serious).\n"
                "   APPLIED MODIFIERS:\n"
                "   - DOWNGRADE risk_of_bias 1 — reason\n"
                "   - UPGRADE large_effect 1 — reason\n"
                "   Write the single word NONE on its own line under APPLIED MODIFIERS if you applied\n"
                "   no modifier at all. Domains: risk_of_bias, inconsistency, indirectness,\n"
                "   imprecision, publication_bias, large_effect, dose_response, plausible_confounding.\n\n"
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

        db_names = (
            "OpenAlex, ERIC, Google Scholar"
            if self.domain == "social_science"
            else "PubMed (MeSH Boolean), Google Scholar"
        )
        combined_input = (
            f"TOPIC: {topic}\n\n"
            f"=== SEARCH METHODOLOGY ===\n"
            f"Search date: {search_date}\n"
            f"Databases: {db_names}\n"
            f"Records identified: {total_wide}\n"
            f"Screened to top studies: {total_screened}\n"
            f"Full-text retrieved: {total_ft_ok} (errors: {total_ft_err})\n\n"
            f"=== AFFIRMATIVE CASE ===\n{aff_case}\n\n"
            f"=== FALSIFICATION CASE ===\n{fal_case}\n\n"
            f"=== DETERMINISTIC MATH (Python-calculated, NOT LLM) ===\n{math_report}\n"
        )

        # Append external metadata signals if available
        metadata_summary = self._summarize_metadata_for_grade(aff_extractions or [], fal_extractions or [])
        if metadata_summary:
            combined_input += f"\n=== EXTERNAL METADATA (API-Sourced) ===\n{metadata_summary}\n"

        if len(combined_input) > 80000:
            combined_input = combined_input[:80000] + "\n\n[...truncated...]"

        try:
            resp = await gated_create(
                self.smart_client,
                model=self.smart_model,
                messages=[{"role": "system", "content": audit_system}, {"role": "user", "content": combined_input}],
                max_tokens=8000,
                temperature=0.2,
                timeout=300,
                extra_body=QWEN3_NO_THINK_EXTRA_BODY,
            )
            audit_text = safe_message_text(resp)
            log(f"    [Step 7] {synthesis_label} synthesis complete ({len(audit_text)} chars)")
        except Exception as e:
            logger.error(f"{synthesis_label} synthesis failed: {e}")
            return (
                f"# GRADE Synthesis: {topic}\n\n*GRADE synthesis failed ({e}). Raw inputs below.*\n\n{combined_input}"
            )

        # OUTSIDE the handler above, and that placement is the fail-closed contract (prepush codex
        # 2026-08-13). Inside it, a GRADE record that could not be grounded became fallback prose,
        # the adapter saw grade_record=None, treated grade_synthesis.json as an absent optional
        # output — it is optional for social science — and completed a clinical stage with no
        # grounded assessment at all. The prose call has a degraded mode; the record does not.
        self.grade_record = await self._grade_record(
            audit_text, {"case:affirmative": aff_case, "case:falsification": fal_case}, log
        )
        return audit_text


    #: How many times the auditor may be asked again for a record that will not validate. Bounded
    #: because the alternative to a bound is a forty-minute stage looping on a model that has
    #: decided it cannot ground its own reasoning.
    GRADE_RECORD_ATTEMPTS = 2

    async def _grade_record(self, prose: str, artifacts: dict[str, str], log=logger.info) -> dict | None:
        """The GRADE prose, read back as the structured record `grade.schema.json` describes.

        A SECOND pass over what the auditor just wrote, rather than asking for prose and JSON in one
        response: the prose is the human-readable artifact and the record is a structured reading of
        it, which is exactly what ``pipeline_sot.py``'s regex was doing — only complete, grounded,
        and fail-closed instead of defaulting to "Not Determined" when the pattern misses.

        Every modifier must quote the case report it comes from. The model supplies the span, Python
        finds the offset (asking a model to count characters produces a number that satisfies the
        contract while pointing nowhere), and a record whose quotes are not in the cases does not
        validate. Returns None only for the social-science domain, which has no GRADE ladder.
        """
        if self.domain == "social_science":
            return None

        from dr2_podcast.schemas import SCHEMA_VERSION, grade_errors

        instruction = (
            "/no_think\n"
            "Read the GRADE synthesis below — which you just wrote — and state its assessment as JSON.\n"
            "Report ONLY what the synthesis says. You are transcribing a judgement, not making a new one.\n\n"
            "{\n"
            '  "level": "high | moderate | low | very_low",\n'
            '  "downgrades": [{"domain": "risk_of_bias | inconsistency | indirectness | imprecision '
            '| publication_bias", "steps": 1, "reason": "why it applies", '
            '"artifact_id": "case:affirmative | case:falsification", "quote": "the exact sentence"}],\n'
            '  "upgrades": [{"domain": "large_effect | dose_response | plausible_confounding", '
            '"steps": 1, "reason": "why it applies", "artifact_id": "case:affirmative | '
            'case:falsification", "quote": "the exact sentence"}]\n'
            "}\n\n"
            "AT MOST ONE ENTRY PER DOMAIN. Two imprecision downgrades are one entry of 2 steps, never "
            "two entries — the steps are summed, and a repeated domain counts its evidence twice.\n"
            "steps is 1 (serious) or 2 (very serious). Nothing else is a GRADE step.\n"
            "Every quote MUST be copied VERBATIM from the case named in artifact_id. Each is checked "
            "against that text, and a record whose quotes cannot be found there is rejected.\n"
            "A modifier the synthesis does not apply is simply absent. Do not invent one to fill the list."
        )
        user = "\n\n".join(
            [f"=== GRADE SYNTHESIS ===\n{prose}"]
            + [f"=== {name} ===\n{text}" for name, text in artifacts.items()]
        )

        problems: list[str] = []
        for attempt in range(1, self.GRADE_RECORD_ATTEMPTS + 1):
            retry_note = (
                ""
                if not problems
                else "\n\nYour previous answer was rejected:\n" + "\n".join(f"- {p}" for p in problems[:6])
            )
            try:
                resp = await gated_create(
                    self.smart_client,
                    model=self.smart_model,
                    messages=[
                        {"role": "system", "content": instruction + retry_note},
                        {"role": "user", "content": user[:80000]},
                    ],
                    max_tokens=2000,
                    temperature=0.1,
                    timeout=180,
                    extra_body=QWEN3_NO_THINK_EXTRA_BODY,
                )
                raw = ResearchAgent._parse_json_response(safe_message_text(resp)) or {}
            except Exception as exc:
                problems = [f"the call itself failed: {exc}"]
                logger.warning("GRADE record attempt %d failed: %s", attempt, exc)
                continue

            record: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "level": safe_str(raw.get("level")) or "",
                "downgrades": self._grade_modifiers(raw.get("downgrades"), artifacts),
                "upgrades": self._grade_modifiers(raw.get("upgrades"), artifacts),
            }
            problems = grade_errors(record, artifacts)
            # Grounded is not the same as complete. grade_errors checks the modifiers that are
            # there; this checks that the ones the synthesis applied are all there, because a
            # dropped downgrade flips net_direction and moves the confidence the wrong way.
            problems += self._transcription_errors(record, prose) if not problems else []
            if not problems:
                log(
                    f"    [Step 7] GRADE record: {record['level']} "
                    f"({len(record['downgrades'])} down, {len(record['upgrades'])} up)"
                )
                return record
            logger.warning("GRADE record attempt %d did not validate: %s", attempt, "; ".join(problems[:3]))

        from dr2_podcast.artifacts import ArtifactError

        raise ArtifactError(
            "the GRADE assessment could not be stated as a record that validates after "
            f"{self.GRADE_RECORD_ATTEMPTS} attempts: {'; '.join(problems[:5])}. The regex scrape this "
            "replaces defaulted to 'Not Determined' and let the episode speak a confidence nobody computed."
        )

    #: The APPLIED MODIFIERS block the GRADE prompt asks for, one line per applied modifier.
    _APPLIED_LINE = re.compile(
        r"^\s*[-*]?\s*(DOWNGRADE|UPGRADE)\s+([a-z_]+)\s+([12])\b", re.IGNORECASE | re.MULTILINE
    )

    @staticmethod
    def declared_modifiers(prose: str) -> set[tuple[str, str, int]] | None:
        """What the synthesis SAYS it applied, as ``{(kind, domain, steps)}``.

        None means the prose never declared a block, which is not the same as declaring none — the
        first cannot be checked and the second can. The distinction is the whole point: a record
        that quietly drops a downgrade still validates, because ``grade_errors`` only checks the
        modifiers that ARE there, and the missing one changes ``net_direction`` and therefore which
        way the evidence moved the confidence (prepush codex 2026-08-13).
        """
        marker = re.search(r"APPLIED\s+MODIFIERS\s*:?", prose, re.IGNORECASE)
        if not marker:
            return None
        block = prose[marker.end() :]
        # Stop at the next numbered section heading, so a later section's prose cannot add modifiers.
        end = re.search(r"\n\s*(?:#{1,6}\s|\d+\.\s+[A-Z])", block)
        if end:
            block = block[: end.start()]
        if re.match(r"\s*NONE\b", block, re.IGNORECASE):
            return set()
        return {
            (kind.lower(), domain.lower(), int(steps))
            for kind, domain, steps in Orchestrator._APPLIED_LINE.findall(block)
        }

    @staticmethod
    def _record_modifiers(record: dict) -> set[tuple[str, str, int]]:
        return {("downgrade", e["domain"], e["steps"]) for e in record["downgrades"]} | {
            ("upgrade", e["domain"], e["steps"]) for e in record["upgrades"]
        }

    @classmethod
    def _transcription_errors(cls, record: dict, prose: str) -> list[str]:
        """Whether the record is the WHOLE of what the prose said it applied."""
        declared = cls.declared_modifiers(prose)
        if declared is None:
            return [
                "the synthesis did not close its GRADE Assessment with an APPLIED MODIFIERS block, "
                "so there is nothing to check the record against"
            ]
        transcribed = cls._record_modifiers(record)
        errors = []
        for kind, domain, steps in sorted(declared - transcribed):
            errors.append(f"the synthesis applied {kind} {domain} {steps}, and the record does not")
        for kind, domain, steps in sorted(transcribed - declared):
            errors.append(f"the record claims {kind} {domain} {steps}, which the synthesis did not apply")
        return errors

    @staticmethod
    def _grade_modifiers(raw_list: Any, artifacts: dict[str, str]) -> list[dict]:
        """Model-supplied modifiers with Python-found offsets. A quote that is not in its artifact
        keeps its (unfindable) locator, so validation rejects the record rather than the modifier
        disappearing — a dropped downgrade silently changes net_direction."""
        built = []
        for raw in raw_list or []:
            if not isinstance(raw, dict):
                continue
            artifact_id = safe_str(raw.get("artifact_id")) or ""
            quote = safe_str(raw.get("quote")) or ""
            hit = locate_span(artifacts.get(artifact_id, ""), quote)
            built.append(
                {
                    "domain": safe_str(raw.get("domain")) or "",
                    "steps": safe_int(raw.get("steps")) or 1,
                    "reason": safe_str(raw.get("reason")) or "",
                    "locator": {
                        "fields": ["reason"],
                        "source_artifact_id": artifact_id,
                        "char_offset": hit[0] if hit else -1,
                        "quoted_span": hit[1] if hit else quote,
                    },
                }
            )
        return built

    @staticmethod
    def _extractions_to_sources(extractions: list[DeepExtraction], role: str) -> list[SummarizedSource]:
        """Convert DeepExtraction list to SummarizedSource for backward compatibility."""
        sources = []
        original_count = len(extractions)
        for ex in extractions:
            # Filter out extractions with empty/missing URLs
            if not ex.url:
                continue
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
                research_tier=ex.research_tier,
            )
            sources.append(
                SummarizedSource(
                    url=ex.url,
                    title=ex.title,
                    summary=ex.raw_facts,
                    query=role,
                    goal=role,
                    metadata=metadata,
                )
            )
        if filtered_count := original_count - len(sources):
            logger.info(f"Filtered {filtered_count} {role} sources with empty URLs")
        return sources

    @staticmethod
    async def _enrich_with_metadata(records: list[WideNetRecord], log=logger.info) -> list[WideNetRecord]:
        """Enrich WideNetRecords with metadata from OpenAlex, Semantic Scholar, Crossref.

        Optional — returns records unchanged on failure. All API errors are caught.
        """
        try:
            from dr2_podcast.research.metadata_clients import (
                MetadataCache,
                OpenAlexClient,
                SemanticScholarClient,
                CrossrefClient,
                enrich_papers_metadata,
            )
        except ImportError:
            log("    [Metadata] metadata_clients not available — skipping enrichment")
            return records

        if not records:
            return records

        try:
            with MetadataCache() as cache:
                oa_client = OpenAlexClient(cache=cache)
                s2_client = SemanticScholarClient(cache=cache)
                cr_client = CrossrefClient(cache=cache)
                try:
                    papers = [{"doi": r.doi or "", "pmid": r.pmid or ""} for r in records]
                    enriched = await enrich_papers_metadata(
                        papers,
                        openalex_client=oa_client,
                        s2_client=s2_client,
                        crossref_client=cr_client,
                    )

                    retracted_titles = []
                    enriched_count = 0
                    for rec, ep in zip(records, enriched, strict=True):
                        if not ep.enrichment_sources:
                            continue
                        enriched_count += 1
                        pm = PaperMetadata(
                            citation_count=ep.best_citation_count,
                            influential_citation_count=ep.influential_citation_count,
                            fwci=ep.fwci,
                            funding_sources=ep.all_funding_sources or None,
                            is_retracted=ep.is_retracted,
                            is_corrected=ep.is_corrected,
                            has_clinical_trial_number=bool(ep.clinical_trial_numbers),
                            clinical_trial_numbers=ep.clinical_trial_numbers or None,
                            enrichment_sources=ep.enrichment_sources,
                        )
                        rec.paper_metadata = pm
                        if ep.is_retracted:
                            retracted_titles.append(rec.title)

                    log(f"    [Metadata] Enriched {enriched_count}/{len(records)} records")
                    for title in retracted_titles:
                        logger.warning(f"RETRACTED paper detected: {title}")
                        log(f"    ⚠ RETRACTED: {title[:80]}")
                finally:
                    await oa_client.close()
                    await s2_client.close()
                    await cr_client.close()
        except Exception as e:
            logger.warning(f"Metadata enrichment failed (non-fatal): {e}")
            log(f"    [Metadata] Enrichment failed (non-fatal): {e}")

        return records

    @staticmethod
    def _summarize_metadata_for_grade(
        aff_extractions: list[DeepExtraction],
        fal_extractions: list[DeepExtraction],
    ) -> str:
        """Produce a text block summarizing metadata signals for GRADE synthesis."""
        lines = []
        industry_funded = []
        trial_registered = []
        corrected = []

        for ext in aff_extractions + fal_extractions:
            pm = ext.paper_metadata
            if pm is None:
                continue
            label = ext.pmid or ext.title[:50]
            if pm.funding_sources:
                industry_keywords = {"pharma", "industry", "inc.", "corp.", "ltd.", "gmbh"}
                for src in pm.funding_sources:
                    if any(kw in src.lower() for kw in industry_keywords):
                        industry_funded.append(f"  - {label}: funded by {src}")
                        break
            if pm.has_clinical_trial_number and pm.clinical_trial_numbers:
                trial_registered.append(f"  - {label}: {', '.join(pm.clinical_trial_numbers)}")
            if pm.is_corrected:
                corrected.append(f"  - {label}")

        if industry_funded:
            lines.append("Industry-funded studies (consider risk of bias):")
            lines.extend(industry_funded)
        if trial_registered:
            lines.append("Studies with clinical trial registration (quality signal):")
            lines.extend(trial_registered)
        if corrected:
            lines.append("Corrected/erratum studies:")
            lines.extend(corrected)

        return "\n".join(lines) if lines else ""

    @staticmethod
    def _save_artifacts(output_dir: str, aff: _TrackResult, fal: _TrackResult, math_report: str):
        """Save intermediate pipeline artifacts to output directory.

        Writes into research/ subdirectory if it exists (M9 layout).
        Falls back to flat layout for backward compatibility.
        """
        aff_strategy, fal_strategy = aff.plan, fal.plan
        aff_records, fal_records = aff.records, fal.records
        aff_top, fal_top = aff.top_records, fal.top_records
        aff_highest_tier, fal_highest_tier = aff.highest_tier, fal.highest_tier

        import dataclasses
        from pathlib import Path

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Use research/ subdirectory if it exists (M9 layout)
        research_dir = out / "research"
        _out = research_dir if research_dir.is_dir() else out

        # write_atomic, not a bare open(): these are the `research` stage's DECLARED outputs, and a
        # rerun interrupted partway — Ctrl-C, SIGKILL, power loss — truncated the last coherent
        # result of a forty-minute stage with a half-written file that reads as finished (prepush
        # codex 2026-08-13). Fixed here rather than by staging the stage, because run_deep_research
        # also loads and saves the extraction cache from this directory, and staging that would
        # make every rerun re-extract every paper.
        from dr2_podcast.artifacts import write_atomic

        # Strategy files — TieredSearchPlan serialized via dataclasses.asdict
        write_atomic(_out / "search_strategy_aff.json", json.dumps(dataclasses.asdict(aff_strategy), indent=2))
        write_atomic(_out / "search_strategy_neg.json", json.dumps(dataclasses.asdict(fal_strategy), indent=2))

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
                "all_candidates": [_record_to_dict(r, id(r) in selected_set) for r in records],
            }

        write_atomic(
            _out / "screening_results_aff.json",
            json.dumps(_screening_payload(aff_records, aff_top, aff_highest_tier), indent=2, ensure_ascii=False),
        )
        write_atomic(
            _out / "screening_results_neg.json",
            json.dumps(_screening_payload(fal_records, fal_top, fal_highest_tier), indent=2, ensure_ascii=False),
        )
        # allow_empty: a run with no studies to compute on has an empty math report, and refusing to
        # write it would fail the stage over the honest answer.
        write_atomic(_out / "clinical_math.md", math_report, allow_empty=True)


# --- Convenience functions ---


async def run_deep_research(
    topic: str,
    config: "ResearchConfig | None" = None,
    framing_context: str = "",
    output_dir: str = None,
) -> "DeepResearchResult":
    """Entry point for the 7-step pipeline.

    Search/model settings travel as one ResearchConfig; the default is the
    module's configured smart model with a clinical domain.
    """
    orchestrator = Orchestrator(config or ResearchConfig())
    return await orchestrator.run(topic, framing_context=framing_context, output_dir=output_dir)


async def main():
    """Test the evidence-based clinical research pipeline."""
    import os

    topic = "does coffee intake improve cognitive performance and productivity?"
    brave_key = os.getenv("BRAVE_API_KEY", "")

    from pathlib import Path

    output_dir = Path("research_outputs/test_deep_agent")
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = await run_deep_research(
        topic=topic,
        config=ResearchConfig(brave_api_key=brave_key, results_per_query=5),
        output_dir=str(output_dir),
    )

    # Save reports
    report_filenames = {
        "lead": "affirmative_case.md",
        "counter": "falsification_case.md",
        "audit": "grade_synthesis.md",
    }
    for role, report in reports.items():
        filename = output_dir / report_filenames.get(role, f"{role}.md")
        with open(filename, "w") as f:
            f.write(report.report)
        logger.info(f"Saved {role} report: {filename} ({len(report.report)} chars)")

    logger.info(f"Total sources: {reports['audit'].total_summaries}")
    logger.info(f"Total time: {reports['audit'].duration_seconds:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
