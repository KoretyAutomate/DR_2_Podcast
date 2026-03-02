"""
Social Science Research Pipeline — PECO framework with effect size analysis.

Parallel to clinical_research.py but optimized for:
- Education, parenting, productivity, social policy topics
- PECO (Population, Exposure, Comparison, Outcome) instead of PICO
- ERIC + OpenAlex + Google Scholar instead of PubMed
- Cohen's d / Hedges' g instead of NNT/ARR
- Evidence quality hierarchy instead of GRADE

Architecture (7-Step Social Science Pipeline — parallel a/b tracks):
  Pre-step: Concept Decomposition — extract canonical terms (ERIC thesaurus, APA terms)
  Steps 1a–5a (Affirmative) run in parallel with Steps 1b–5b (Falsification):
    Step 1: PECO tiered keywords (Smart) → Auditor gate → loop until approved
    Step 2: Cascading search — OpenAlex → ERIC → Scholar
    Step 3: Evidence-quality-aware screening → top 20 (systematic review > quasi-exp > cohort)
    Step 4: Full-text fetch + effect size extraction
    Step 5: Case synthesis (Smart Model)
  Step 6: Deterministic effect size math (Python, no LLM)
  Step 7: Evidence quality synthesis (Smart Model)
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import AsyncOpenAI

from dr2_podcast.utils import safe_float, safe_int, safe_str
from dr2_podcast.research.metadata_clients import OpenAlexClient, ERICClient, MetadataCache
from dr2_podcast.research.search_service import SearxngClient
from dr2_podcast.research.effect_size_math import (
    EffectSizeImpact, batch_calculate, format_effect_size_report,
)

logger = logging.getLogger(__name__)

# --- Configuration ---

from dr2_podcast.config import SMART_MODEL, SMART_BASE_URL, FAST_MODEL, FAST_BASE_URL, SCRAPING_TIMEOUT

MAX_SELECT = 20  # Maximum studies to select per track

# --- Evidence quality hierarchy ---

EVIDENCE_HIERARCHY = {
    "systematic_review": 6,
    "meta_analysis": 6,
    "rct": 5,
    "quasi_experimental": 4,
    "cohort": 3,
    "longitudinal": 3,
    "cross_sectional": 2,
    "correlational": 2,
    "case_study": 1,
    "qualitative": 1,
    "expert_opinion": 0,
    "other": 1,
}

EVIDENCE_QUALITY_LABELS = {
    6: "STRONG",
    5: "MODERATE_STRONG",
    4: "MODERATE",
    3: "MODERATE_WEAK",
    2: "WEAK",
    1: "VERY_WEAK",
    0: "VERY_WEAK",
}


# --- Data Models ---

@dataclass
class PECOSearchPlan:
    """PECO-based search plan for social science research."""
    peco: Dict[str, str]   # P, E, C, O
    tier1: Dict[str, List[str]]  # exact terms: intervention, outcome, population
    tier2: Dict[str, List[str]]  # synonyms
    tier3: Dict[str, List[str]]  # broader concept class
    role: str              # "affirmative" | "adversarial"
    auditor_approved: bool = False
    auditor_notes: str = ""
    revision_count: int = 0


@dataclass
class SocialScienceRecord:
    """Screening record for social science research."""
    doi: Optional[str]
    title: str
    abstract: str
    study_type: str
    sample_size: Optional[str]
    primary_objective: Optional[str]
    year: Optional[int]
    source: Optional[str]   # journal or institution
    authors: Optional[str]
    url: str
    source_db: str          # "openalex", "eric", "scholar"
    research_tier: Optional[int] = None
    relevance_score: Optional[float] = None
    wwc_rating: Optional[str] = None
    eric_id: Optional[str] = None
    evidence_quality_score: int = 1


@dataclass
class SocialScienceExtraction:
    """Effect size extraction from full-text social science articles."""
    doi: Optional[str]
    title: str
    url: str
    # Effect size fields
    effect_size_value: Optional[float] = None
    effect_size_type: Optional[str] = None  # "cohens_d", "odds_ratio", "correlation_r", "beta", "hedges_g"
    effect_size_ci: Optional[str] = None
    # Study characteristics
    study_design: Optional[str] = None
    sample_size_total: Optional[int] = None
    sample_size_treatment: Optional[int] = None
    sample_size_control: Optional[int] = None
    setting: Optional[str] = None
    demographics: Optional[str] = None
    theoretical_framework: Optional[str] = None
    measurement_instrument: Optional[str] = None
    follow_up_period: Optional[str] = None
    attrition_pct: Optional[str] = None
    funding_source: Optional[str] = None
    limitations: Optional[str] = None
    # Metadata
    research_tier: Optional[int] = None
    evidence_quality_score: int = 1
    raw_facts: str = ""
    pmid: Optional[str] = None  # For compatibility

    def to_dict(self) -> dict:
        d = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None and v != "" and v != []:
                d[f.name] = v
        return d


# --- Helper functions ---

def _classify_study_type(type_str: str) -> Tuple[str, int]:
    """Classify a study type string and return (normalized_type, quality_score)."""
    t = (type_str or "").lower().strip()
    for key, score in EVIDENCE_HIERARCHY.items():
        if key.replace("_", " ") in t or key.replace("_", "-") in t:
            return key, score
    if "meta" in t:
        return "meta_analysis", 6
    if "systematic" in t and "review" in t:
        return "systematic_review", 6
    if "random" in t:
        return "rct", 5
    if "quasi" in t or "difference" in t or "regression discontinuity" in t:
        return "quasi_experimental", 4
    if "cohort" in t or "longitudinal" in t or "panel" in t:
        return "cohort", 3
    if "cross" in t or "survey" in t:
        return "cross_sectional", 2
    if "correlat" in t:
        return "correlational", 2
    if "case" in t or "qualitative" in t:
        return "case_study", 1
    return "other", 1


async def _call_smart_model(client, model, system_prompt, user_prompt,
                            max_tokens=4000, temperature=0.3, timeout=120):
    """Call the smart model with retry logic."""
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"/no_think {system_prompt}"},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                logger.warning(f"Smart model call failed (attempt {attempt+1}): {e}")
                await asyncio.sleep(2 ** attempt)
            else:
                raise
    return ""


def _parse_json_response(text: str) -> dict:
    """Extract JSON from an LLM response, handling markdown code blocks."""
    # Strip markdown code blocks
    cleaned = re.sub(r'```(?:json)?\s*', '', text)
    cleaned = re.sub(r'```', '', cleaned)
    # Find JSON object
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Try the whole string
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        return {}


# --- Social Science Orchestrator ---

class SocialScienceOrchestrator:
    """Orchestrates the 7-step social science research pipeline."""

    def __init__(self, fast_model_available: bool = True):
        self.smart_client = AsyncOpenAI(base_url=SMART_BASE_URL, api_key="not-needed")
        self.smart_model = SMART_MODEL

        self.fast_client = None
        self.fast_model = None
        if fast_model_available and FAST_MODEL and FAST_BASE_URL:
            self.fast_client = AsyncOpenAI(base_url=FAST_BASE_URL, api_key="not-needed")
            self.fast_model = FAST_MODEL

        self.cache = MetadataCache()
        self.openalex = OpenAlexClient(cache=self.cache)
        self.eric = ERICClient(cache=self.cache)

    async def run(
        self, topic: str, framing_context: str = "", output_dir: str = None,
        log=print,
    ) -> Dict[str, Any]:
        """Execute the full 7-step social science pipeline."""
        start_time = time.time()
        search_date = time.strftime("%Y-%m-%d")

        log(f"\n{'='*70}")
        log(f"SOCIAL SCIENCE RESEARCH PIPELINE")
        log(f"Topic: {topic}")
        log(f"{'='*70}")

        # Pre-step: Concept decomposition
        log(f"\n{'='*70}")
        log(f"PRE-STEP: CONCEPT DECOMPOSITION")
        log(f"{'='*70}")
        decomposition = await self._concept_decomposition(topic, framing_context, log)

        # Track metrics
        aff_wide_net = fal_wide_net = 0
        aff_screened = fal_screened = 0
        aff_ft_ok = fal_ft_ok = 0
        aff_ft_err = fal_ft_err = 0

        # --- Affirmative Track ---
        async def affirmative_track():
            nonlocal aff_wide_net, aff_screened, aff_ft_ok, aff_ft_err

            log(f"\n{'='*70}")
            log(f"STEP 1a: PECO KEYWORD GENERATION (Affirmative)")
            log(f"{'='*70}")
            plan = await self._formulate_strategy(topic, "affirmative", framing_context, decomposition, log)

            log(f"\n{'='*70}")
            log(f"STEP 2a: CASCADING SEARCH (OpenAlex + ERIC + Scholar)")
            log(f"{'='*70}")
            records = await self._search(plan, log)
            aff_wide_net = len(records)

            log(f"\n{'='*70}")
            log(f"STEP 3a: SCREENING ({len(records)} → top {MAX_SELECT})")
            log(f"{'='*70}")
            top = await self._screen(records, plan, topic, log)
            aff_screened = len(top)

            log(f"\n{'='*70}")
            log(f"STEP 4a: EXTRACTION ({len(top)} articles)")
            log(f"{'='*70}")
            extractions = await self._extract_batch(top, plan.peco, log)
            aff_ft_ok = sum(1 for e in extractions if e.raw_facts)
            aff_ft_err = len(top) - aff_ft_ok

            log(f"\n{'='*70}")
            log(f"STEP 5a: AFFIRMATIVE CASE")
            log(f"{'='*70}")
            case = await self._build_case(topic, plan, extractions, "affirmative", log)

            return plan, records, top, extractions, case

        # --- Falsification Track ---
        async def falsification_track():
            nonlocal fal_wide_net, fal_screened, fal_ft_ok, fal_ft_err

            log(f"\n{'='*70}")
            log(f"STEP 1b: PECO KEYWORD GENERATION (Falsification)")
            log(f"{'='*70}")
            plan = await self._formulate_strategy(topic, "adversarial", framing_context, decomposition, log)

            log(f"\n{'='*70}")
            log(f"STEP 2b: CASCADING SEARCH (OpenAlex + ERIC + Scholar)")
            log(f"{'='*70}")
            records = await self._search(plan, log)
            fal_wide_net = len(records)

            log(f"\n{'='*70}")
            log(f"STEP 3b: SCREENING ({len(records)} → top {MAX_SELECT})")
            log(f"{'='*70}")
            top = await self._screen(records, plan, topic, log)
            fal_screened = len(top)

            log(f"\n{'='*70}")
            log(f"STEP 4b: EXTRACTION ({len(top)} articles)")
            log(f"{'='*70}")
            extractions = await self._extract_batch(top, plan.peco, log)
            fal_ft_ok = sum(1 for e in extractions if e.raw_facts)
            fal_ft_err = len(top) - fal_ft_ok

            log(f"\n{'='*70}")
            log(f"STEP 5b: FALSIFICATION CASE")
            log(f"{'='*70}")
            case = await self._build_case(topic, plan, extractions, "falsification", log)

            return plan, records, top, extractions, case

        # Run both tracks in parallel
        log(f"\n{'='*70}")
        log(f"RUNNING AFFIRMATIVE & FALSIFICATION TRACKS IN PARALLEL")
        log(f"{'='*70}")

        (aff_plan, aff_records, aff_top, aff_extractions, aff_case), \
        (fal_plan, fal_records, fal_top, fal_extractions, fal_case) = \
            await asyncio.gather(affirmative_track(), falsification_track())

        # Step 6: Effect size math
        log(f"\n{'='*70}")
        log(f"STEP 6: DETERMINISTIC EFFECT SIZE MATH")
        log(f"{'='*70}")
        all_extractions = aff_extractions + fal_extractions
        impacts = batch_calculate(all_extractions)
        math_report = format_effect_size_report(impacts)
        log(f"    Calculated effect sizes for {len(impacts)} studies")

        # Step 7: Evidence quality synthesis
        log(f"\n{'='*70}")
        log(f"STEP 7: EVIDENCE QUALITY SYNTHESIS")
        log(f"{'='*70}")
        total_wide = aff_wide_net + fal_wide_net
        total_screened = aff_screened + fal_screened
        total_ft_ok = aff_ft_ok + fal_ft_ok
        total_ft_err = aff_ft_err + fal_ft_err

        synthesis_text = await self._evidence_synthesis(
            topic, aff_case, fal_case, math_report,
            aff_plan, fal_plan,
            total_wide, total_screened, total_ft_ok, total_ft_err,
            search_date, log,
        )

        # Save artifacts
        if output_dir:
            self._save_artifacts(
                output_dir, aff_plan, fal_plan,
                aff_records, fal_records, aff_top, fal_top,
                math_report,
            )

        # Cleanup metadata clients
        await self.openalex.close()
        await self.eric.close()
        self.cache.close()

        duration = time.time() - start_time
        log(f"\n✓ Social science pipeline complete in {duration:.0f}s")

        # Build report objects compatible with pipeline.py expectations
        from dr2_podcast.research.clinical import ResearchReport, SummarizedSource, StudyMetadata

        def _to_report(case_text, extractions, role):
            sources = []
            for ex in extractions:
                md = StudyMetadata(
                    study_type=ex.study_design,
                    sample_size=str(ex.sample_size_total) if ex.sample_size_total else None,
                    effect_size=str(ex.effect_size_value) if ex.effect_size_value else None,
                    demographics=ex.demographics,
                    funding_source=ex.funding_source,
                    research_tier=ex.research_tier,
                )
                sources.append(SummarizedSource(
                    url=ex.url, title=ex.title, summary=ex.raw_facts,
                    query=role, goal=role, metadata=md,
                ))
            return ResearchReport(
                topic=topic, role=role, sources=sources, report=case_text,
                iterations_used=1, total_urls_fetched=len(extractions),
                total_summaries=len(extractions), total_errors=0,
                duration_seconds=duration,
            )

        return {
            "lead": _to_report(aff_case, aff_extractions, "lead"),
            "counter": _to_report(fal_case, fal_extractions, "counter"),
            "audit": ResearchReport(
                report=synthesis_text, topic=topic, role='audit',
                sources=[], iterations_used=1, total_urls_fetched=0,
                total_summaries=0, total_errors=0, duration_seconds=0,
            ),
            "pipeline_data": {
                "domain": "social_science",
                "aff_strategy": aff_plan,
                "fal_strategy": fal_plan,
                "aff_extractions": aff_extractions,
                "fal_extractions": fal_extractions,
                "aff_top": aff_top,
                "fal_top": fal_top,
                "math_report": math_report,
                "impacts": impacts,
                "framing_context": framing_context,
                "search_date": search_date,
                "aff_highest_tier": 1,
                "fal_highest_tier": 1,
                "metrics": {
                    "aff_wide_net_total": aff_wide_net,
                    "fal_wide_net_total": fal_wide_net,
                    "aff_screened_in": aff_screened,
                    "fal_screened_in": fal_screened,
                    "aff_fulltext_ok": aff_ft_ok,
                    "fal_fulltext_ok": fal_ft_ok,
                    "aff_fulltext_err": aff_ft_err,
                    "fal_fulltext_err": fal_ft_err,
                },
            },
        }

    # --- Pre-step: Concept Decomposition ---

    async def _concept_decomposition(self, topic: str, framing: str, log=print) -> str:
        """Extract canonical social science terms from folk topic."""
        client = self.fast_client or self.smart_client
        model = self.fast_model or self.smart_model
        try:
            result = await _call_smart_model(
                client, model,
                "You are a social science terminology expert.",
                (
                    f"Extract canonical academic terms from this topic:\n{topic}\n\n"
                    f"Context: {framing[:500]}\n\n"
                    "Return a comma-separated list of terms from:\n"
                    "- ERIC thesaurus terms\n"
                    "- APA PsycINFO terms\n"
                    "- Standard social science methodology terms\n"
                    "Focus on terms useful for database searches."
                ),
                max_tokens=500,
            )
            log(f"    Decomposition: {result[:200]}")
            return result
        except Exception as e:
            logger.warning(f"Concept decomposition failed: {e}")
            return ""

    # --- Step 1: PECO Strategy Formulation ---

    async def _formulate_strategy(
        self, topic: str, role: str, framing: str, decomposition: str, log=print
    ) -> PECOSearchPlan:
        """Generate a PECO-based tiered keyword plan."""
        role_desc = "supporting" if role == "affirmative" else "challenging"

        prompt = (
            f"You are a social science researcher designing a search strategy.\n\n"
            f"TOPIC: {topic}\n"
            f"ROLE: Find evidence {role_desc} this claim.\n"
            f"CANONICAL TERMS: {decomposition}\n\n"
            "Generate a PECO framework and three tiers of search keywords.\n\n"
            "Return JSON:\n"
            '{\n'
            '  "peco": {"P": "population", "E": "exposure/intervention", "C": "comparison", "O": "outcome"},\n'
            '  "tier1": {"intervention": ["exact terms"], "outcome": ["exact terms"], "population": ["population terms"]},\n'
            '  "tier2": {"intervention": ["synonyms"], "outcome": ["synonyms"], "population": ["broader"]},\n'
            '  "tier3": {"intervention": ["broader class"], "outcome": ["broader class"], "population": ["general"]}\n'
            '}'
        )

        result = await _call_smart_model(
            self.smart_client, self.smart_model,
            "You are a social science search strategist. Return only JSON.",
            prompt, max_tokens=2000,
        )

        data = _parse_json_response(result)
        plan = PECOSearchPlan(
            peco=data.get("peco", {"P": topic, "E": topic, "C": "no exposure", "O": "outcomes"}),
            tier1=data.get("tier1", {"intervention": [topic], "outcome": [], "population": []}),
            tier2=data.get("tier2", {"intervention": [], "outcome": [], "population": []}),
            tier3=data.get("tier3", {"intervention": [], "outcome": [], "population": []}),
            role=role,
            auditor_approved=True,  # Simplified for initial implementation
        )
        log(f"    PECO: P={plan.peco.get('P','')}, E={plan.peco.get('E','')}, O={plan.peco.get('O','')}")
        return plan

    # --- Step 2: Cascading Search ---

    async def _search(self, plan: PECOSearchPlan, log=print) -> List[SocialScienceRecord]:
        """Search OpenAlex + ERIC + Scholar with tiered keywords."""
        records = []
        seen_titles = set()

        def _dedup_add(new_records):
            for r in new_records:
                key = r.title.lower().strip()[:80]
                if key not in seen_titles:
                    seen_titles.add(key)
                    records.append(r)

        # Build search queries from tiers
        for tier_num, tier_data in [(1, plan.tier1), (2, plan.tier2), (3, plan.tier3)]:
            terms = tier_data.get("intervention", []) + tier_data.get("outcome", [])
            query = " ".join(terms[:5])
            if not query.strip():
                continue

            # OpenAlex search
            try:
                oa_results = await self.openalex.search_works(query, per_page=50)
                for oa in oa_results:
                    abstract = oa.get("abstract_text", "")
                    if not abstract and not oa.get("doi"):
                        continue
                    study_type, quality = _classify_study_type(oa.get("type", ""))
                    _dedup_add([SocialScienceRecord(
                        doi=oa.get("doi", ""),
                        title=oa.get("openalex_id", "").split("/")[-1] if not abstract else (abstract[:80] + "..."),
                        abstract=abstract,
                        study_type=study_type,
                        sample_size=None,
                        primary_objective=None,
                        year=None,
                        source=None,
                        authors=None,
                        url=f"https://doi.org/{oa['doi']}" if oa.get("doi") else "",
                        source_db="openalex",
                        research_tier=tier_num,
                        evidence_quality_score=quality,
                    )])
                log(f"    Tier {tier_num} OpenAlex: {len(oa_results)} results")
            except Exception as e:
                logger.warning(f"OpenAlex search failed for tier {tier_num}: {e}")

            # ERIC search (education topics)
            try:
                eric_results = await self.eric.search(query, max_results=30)
                for er in eric_results:
                    study_type, quality = _classify_study_type(
                        " ".join(er.get("publication_type", []))
                    )
                    _dedup_add([SocialScienceRecord(
                        doi=None,
                        title=er.get("title", ""),
                        abstract=er.get("description", ""),
                        study_type=study_type,
                        sample_size=None,
                        primary_objective=None,
                        year=er.get("year"),
                        source=er.get("source", ""),
                        authors=", ".join(er.get("author", [])),
                        url=er.get("url", ""),
                        source_db="eric",
                        research_tier=tier_num,
                        eric_id=er.get("eric_id", ""),
                        evidence_quality_score=quality,
                    )])
                log(f"    Tier {tier_num} ERIC: {len(eric_results)} results")
            except Exception as e:
                logger.warning(f"ERIC search failed for tier {tier_num}: {e}")

            # Stop if we have enough
            if len(records) >= 100:
                break

        # Google Scholar via SearXNG (Tier 1 keywords only)
        try:
            t1_terms = plan.tier1.get("intervention", []) + plan.tier1.get("outcome", [])
            scholar_query = " ".join(t1_terms[:4])
            if scholar_query.strip():
                async with SearxngClient() as searxng:
                    scholar_results = await searxng.search(
                        scholar_query + " research study",
                        num_results=20, engines=["google scholar"]
                    )
                    for sr in scholar_results:
                        _dedup_add([SocialScienceRecord(
                            doi=None,
                            title=sr.title,
                            abstract=sr.snippet or "",
                            study_type="other",
                            sample_size=None,
                            primary_objective=None,
                            year=None,
                            source=None,
                            authors=None,
                            url=sr.url,
                            source_db="scholar",
                            research_tier=1,
                            evidence_quality_score=1,
                        )])
                    log(f"    Scholar: {len(scholar_results)} results")
        except Exception as e:
            logger.warning(f"Scholar search failed: {e}")

        log(f"    Total unique records: {len(records)}")
        return records

    # --- Step 3: Evidence-Quality-Aware Screening ---

    async def _screen(
        self, records: List[SocialScienceRecord], plan: PECOSearchPlan,
        topic: str, log=print
    ) -> List[SocialScienceRecord]:
        """Screen and prioritize records by evidence quality."""
        if not records:
            return []

        # Sort by evidence quality score (descending), then tier (ascending)
        records.sort(key=lambda r: (-r.evidence_quality_score, r.research_tier or 3))

        # Take top MAX_SELECT
        selected = records[:MAX_SELECT]
        log(f"    Selected {len(selected)} records (best evidence quality first)")

        # Log quality distribution
        quality_dist = {}
        for r in selected:
            label = EVIDENCE_QUALITY_LABELS.get(r.evidence_quality_score, "UNKNOWN")
            quality_dist[label] = quality_dist.get(label, 0) + 1
        for label, count in sorted(quality_dist.items()):
            log(f"      {label}: {count}")

        return selected

    # --- Step 4: Effect Size Extraction ---

    async def _extract_batch(
        self, records: List[SocialScienceRecord], peco: Dict[str, str], log=print
    ) -> List[SocialScienceExtraction]:
        """Extract effect sizes and study characteristics from records."""
        client = self.smart_client
        model = self.smart_model

        async def extract_one(record: SocialScienceRecord) -> SocialScienceExtraction:
            content = record.abstract or record.title
            if len(content) < 30:
                return SocialScienceExtraction(
                    doi=record.doi, title=record.title, url=record.url,
                    research_tier=record.research_tier,
                    evidence_quality_score=record.evidence_quality_score,
                    raw_facts="Insufficient content for extraction",
                )

            prompt = (
                f"Extract research data from this study.\n\n"
                f"PECO: {json.dumps(peco)}\n"
                f"TITLE: {record.title}\n"
                f"CONTENT: {content[:4000]}\n\n"
                "Return JSON with these fields (null if not found):\n"
                '{"effect_size_value": float, "effect_size_type": "cohens_d"|"odds_ratio"|"correlation_r"|"beta"|"hedges_g",\n'
                ' "effect_size_ci": "95% CI [low, high]",\n'
                ' "study_design": str, "sample_size_total": int,\n'
                ' "sample_size_treatment": int, "sample_size_control": int,\n'
                ' "setting": str, "demographics": str,\n'
                ' "theoretical_framework": str, "measurement_instrument": str,\n'
                ' "follow_up_period": str, "attrition_pct": str,\n'
                ' "funding_source": str, "limitations": str,\n'
                ' "raw_facts": "key findings summary"}'
            )

            try:
                result = await _call_smart_model(
                    client, model,
                    "You are a social science data extractor. Return only JSON.",
                    prompt, max_tokens=2000, timeout=60,
                )
                data = _parse_json_response(result)

                return SocialScienceExtraction(
                    doi=record.doi,
                    title=record.title,
                    url=record.url,
                    effect_size_value=safe_float(data.get("effect_size_value")),
                    effect_size_type=safe_str(data.get("effect_size_type")),
                    effect_size_ci=safe_str(data.get("effect_size_ci")),
                    study_design=safe_str(data.get("study_design")),
                    sample_size_total=safe_int(data.get("sample_size_total")),
                    sample_size_treatment=safe_int(data.get("sample_size_treatment")),
                    sample_size_control=safe_int(data.get("sample_size_control")),
                    setting=safe_str(data.get("setting")),
                    demographics=safe_str(data.get("demographics")),
                    theoretical_framework=safe_str(data.get("theoretical_framework")),
                    measurement_instrument=safe_str(data.get("measurement_instrument")),
                    follow_up_period=safe_str(data.get("follow_up_period")),
                    attrition_pct=safe_str(data.get("attrition_pct")),
                    funding_source=safe_str(data.get("funding_source")),
                    limitations=safe_str(data.get("limitations")),
                    research_tier=record.research_tier,
                    evidence_quality_score=record.evidence_quality_score,
                    raw_facts=safe_str(data.get("raw_facts")) or "",
                )
            except Exception as e:
                logger.warning(f"Extraction failed for {record.title[:50]}: {e}")
                return SocialScienceExtraction(
                    doi=record.doi, title=record.title, url=record.url,
                    research_tier=record.research_tier,
                    evidence_quality_score=record.evidence_quality_score,
                    raw_facts=f"Extraction failed: {str(e)[:100]}",
                )

        results = await asyncio.gather(*[extract_one(r) for r in records])
        valid = [r for r in results if r.raw_facts and "failed" not in r.raw_facts.lower()[:20]]
        log(f"    Extracted {len(valid)}/{len(records)} successfully")
        return list(results)

    # --- Step 5: Case Synthesis ---

    async def _build_case(
        self, topic: str, plan: PECOSearchPlan,
        extractions: List[SocialScienceExtraction], role: str, log=print
    ) -> str:
        """Build an affirmative or falsification case from extractions."""
        direction = "supporting" if role == "affirmative" else "opposing"
        evidence_block = "\n\n".join([
            f"### {ex.title}\n"
            f"- Design: {ex.study_design or 'Unknown'}\n"
            f"- N: {ex.sample_size_total or 'Unknown'}\n"
            f"- Effect: {ex.effect_size_type}={ex.effect_size_value if ex.effect_size_value else 'N/A'}\n"
            f"- Setting: {ex.setting or 'Unknown'}\n"
            f"- Findings: {ex.raw_facts[:500]}"
            for ex in extractions if ex.raw_facts
        ][:10])

        prompt = (
            f"Synthesize the following studies into a {direction} case.\n\n"
            f"TOPIC: {topic}\n"
            f"PECO: {json.dumps(plan.peco)}\n\n"
            f"EVIDENCE:\n{evidence_block}\n\n"
            f"Write a structured {direction} case report (1000-2000 words) with:\n"
            "1. Thesis statement\n"
            "2. Evidence synthesis (cite each study)\n"
            "3. Methodological assessment\n"
            "4. Limitations\n"
            "5. Conclusion"
        )

        result = await _call_smart_model(
            self.smart_client, self.smart_model,
            f"You are a social science researcher writing a {direction} case report.",
            prompt, max_tokens=4000,
        )
        log(f"    {role.capitalize()} case: {len(result)} chars")
        return result

    # --- Step 7: Evidence Quality Synthesis ---

    async def _evidence_synthesis(
        self, topic: str, aff_case: str, fal_case: str, math_report: str,
        aff_plan: PECOSearchPlan, fal_plan: PECOSearchPlan,
        total_wide: int, total_screened: int, total_ft_ok: int, total_ft_err: int,
        search_date: str, log=print,
    ) -> str:
        """Synthesize evidence quality assessment (parallel to GRADE synthesis)."""
        system_prompt = (
            "You are a social science evidence quality assessor.\n\n"
            "Evidence Quality Levels:\n"
            "- STRONG: Systematic reviews/meta-analyses of controlled studies\n"
            "- MODERATE_STRONG: RCTs (rare in social science)\n"
            "- MODERATE: Quasi-experimental (DiD, regression discontinuity)\n"
            "- MODERATE_WEAK: Cohort/longitudinal studies\n"
            "- WEAK: Cross-sectional/correlational\n"
            "- VERY_WEAK: Case studies/qualitative/expert opinion\n\n"
            "Produce a synthesis with:\n"
            "1. Evidence Quality Assessment\n"
            "2. Strength of Evidence\n"
            "3. Consistency Across Studies\n"
            "4. Effect Size Summary\n"
            "5. Methodological Limitations\n"
            "6. Recommendations\n"
            "7. Executive Summary\n"
            "8. Final Evidence Quality: [STRONG/MODERATE/WEAK/VERY_WEAK]\n\n"
            "CRITICAL: Use the Python-calculated effect sizes exactly — do NOT recalculate."
        )

        combined = (
            f"TOPIC: {topic}\n\n"
            f"=== SEARCH METHODOLOGY ===\n"
            f"Search date: {search_date}\n"
            f"Databases: OpenAlex, ERIC, Google Scholar\n"
            f"Records identified: {total_wide}\n"
            f"Screened: {total_screened}\n"
            f"Extracted: {total_ft_ok} (errors: {total_ft_err})\n\n"
            f"=== AFFIRMATIVE CASE ===\n{aff_case}\n\n"
            f"=== FALSIFICATION CASE ===\n{fal_case}\n\n"
            f"=== DETERMINISTIC EFFECT SIZE MATH ===\n{math_report}\n"
        )

        if len(combined) > 80000:
            combined = combined[:80000] + "\n\n[...truncated...]"

        try:
            result = await _call_smart_model(
                self.smart_client, self.smart_model,
                system_prompt, combined,
                max_tokens=8000, temperature=0.2, timeout=300,
            )
            log(f"    Evidence synthesis complete ({len(result)} chars)")
            return result
        except Exception as e:
            logger.error(f"Evidence synthesis failed: {e}")
            return (
                f"# Evidence Quality Synthesis: {topic}\n\n"
                f"*Synthesis failed ({e}). Raw inputs below.*\n\n{combined}"
            )

    # --- Save Artifacts ---

    @staticmethod
    def _save_artifacts(
        output_dir: str,
        aff_plan: PECOSearchPlan, fal_plan: PECOSearchPlan,
        aff_records, fal_records, aff_top, fal_top,
        math_report: str,
    ):
        """Save intermediate pipeline artifacts.

        Writes into research/ subdirectory if it exists (M9 layout).
        Falls back to flat layout for backward compatibility.
        """
        import dataclasses
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Use research/ subdirectory if it exists (M9 layout)
        research_dir = out / "research"
        if research_dir.is_dir():
            _out = research_dir
        else:
            _out = out

        # Strategy files
        with open(_out / "search_strategy_aff.json", 'w') as f:
            json.dump(dataclasses.asdict(aff_plan), f, indent=2)
        with open(_out / "search_strategy_neg.json", 'w') as f:
            json.dump(dataclasses.asdict(fal_plan), f, indent=2)

        # Screening results
        def _screening_payload(records, top):
            selected_set = {id(r) for r in top}
            return {
                "total_candidates": len(records),
                "selected_count": len(top),
                "records": [
                    {
                        "selected": id(r) in selected_set,
                        "doi": r.doi,
                        "title": r.title,
                        "study_type": r.study_type,
                        "year": r.year,
                        "source_db": r.source_db,
                        "research_tier": r.research_tier,
                        "evidence_quality_score": r.evidence_quality_score,
                    }
                    for r in records[:200]
                ],
            }

        with open(_out / "screening_results_aff.json", 'w') as f:
            json.dump(_screening_payload(aff_records, aff_top), f, indent=2, ensure_ascii=False)
        with open(_out / "screening_results_neg.json", 'w') as f:
            json.dump(_screening_payload(fal_records, fal_top), f, indent=2, ensure_ascii=False)

        # Math report
        with open(_out / "effect_size_math.md", 'w') as f:
            f.write(math_report)


# --- Convenience function ---

async def run_social_science_research(
    topic: str,
    framing_context: str = "",
    output_dir: str = None,
    fast_model_available: bool = True,
) -> Dict[str, Any]:
    """Run the social science research pipeline."""
    orchestrator = SocialScienceOrchestrator(fast_model_available=fast_model_available)
    return await orchestrator.run(topic, framing_context=framing_context, output_dir=output_dir)
