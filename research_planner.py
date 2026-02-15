"""
Research Planner & Iterative Search Engine for Deep Research Podcast
====================================================================

Implements deep research capabilities:
1. Structured research plan generation from user questions
2. Iterative search loops with evidence gap detection
3. Query diversification for comprehensive coverage
4. Source coverage tracking across research aspects
5. Dynamic result scaling based on evidence quality
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx
from search_agent import SearxngClient, DeepResearch, SearchResult, ScrapedContent, ResearchResult

logger = logging.getLogger(__name__)


# --- Data Models ---

@dataclass
class ResearchAspect:
    """A single sub-question/aspect of the research plan."""
    id: str
    question: str
    evidence_types_needed: List[str]  # e.g., ["RCT", "meta-analysis", "mechanism"]
    search_queries: List[str]
    sources_found: List[Dict[str, str]] = field(default_factory=list)
    evidence_level: str = "none"  # none, weak, moderate, strong
    notes: str = ""

    @property
    def source_count(self) -> int:
        return len(self.sources_found)

    def add_source(self, title: str, url: str, source_type: str, snippet: str = ""):
        """Add a source if not already tracked (by URL)."""
        existing_urls = {s["url"] for s in self.sources_found}
        if url not in existing_urls:
            self.sources_found.append({
                "title": title,
                "url": url,
                "type": source_type,
                "snippet": snippet[:300]
            })

    def assess_evidence_level(self) -> str:
        """Assess evidence level based on sources found."""
        if not self.sources_found:
            self.evidence_level = "none"
        elif len(self.sources_found) < 3:
            self.evidence_level = "weak"
        elif len(self.sources_found) < 5:
            self.evidence_level = "moderate"
        else:
            types = {s["type"] for s in self.sources_found}
            if types & {"RCT", "meta-analysis", "systematic_review"}:
                self.evidence_level = "strong"
            else:
                self.evidence_level = "moderate"
        return self.evidence_level


@dataclass
class ResearchPlan:
    """Complete research plan for a topic."""
    topic: str
    aspects: List[ResearchAspect]
    total_sources: int = 0
    total_unique_urls: set = field(default_factory=set)
    iteration_count: int = 0
    max_iterations: int = 5

    def get_coverage_summary(self) -> Dict[str, Any]:
        """Get a summary of research coverage."""
        coverage = {
            "topic": self.topic,
            "total_aspects": len(self.aspects),
            "total_sources": len(self.total_unique_urls),
            "iteration_count": self.iteration_count,
            "aspects": []
        }
        for aspect in self.aspects:
            aspect.assess_evidence_level()
            coverage["aspects"].append({
                "id": aspect.id,
                "question": aspect.question,
                "evidence_level": aspect.evidence_level,
                "source_count": aspect.source_count,
                "evidence_types": list({s["type"] for s in aspect.sources_found})
            })
        return coverage

    def get_weak_aspects(self) -> List[ResearchAspect]:
        """Get aspects with insufficient evidence."""
        weak = []
        for aspect in self.aspects:
            aspect.assess_evidence_level()
            if aspect.evidence_level in ("none", "weak"):
                weak.append(aspect)
        return weak

    def is_complete(self) -> bool:
        """Check if all aspects have at least moderate evidence."""
        if self.iteration_count >= self.max_iterations:
            return True
        for aspect in self.aspects:
            aspect.assess_evidence_level()
            if aspect.evidence_level in ("none", "weak"):
                return False
        return True

    def format_for_llm(self) -> str:
        """Format the plan for LLM consumption."""
        lines = [
            f"# Research Plan: {self.topic}",
            f"Total sources collected: {len(self.total_unique_urls)}",
            f"Research iterations completed: {self.iteration_count}/{self.max_iterations}",
            ""
        ]
        for aspect in self.aspects:
            aspect.assess_evidence_level()
            level_icon = {"none": "❌", "weak": "⚠️", "moderate": "✓", "strong": "✓✓"}
            lines.append(f"## {aspect.id}. {aspect.question}")
            lines.append(f"   Evidence Level: {level_icon.get(aspect.evidence_level, '?')} {aspect.evidence_level.upper()} ({aspect.source_count} sources)")
            if aspect.sources_found:
                for s in aspect.sources_found:
                    lines.append(f"   - [{s['type']}] {s['title'][:80]}")
                    lines.append(f"     URL: {s['url']}")
            lines.append("")
        return "\n".join(lines)


# --- Research Plan Builder ---

def build_research_plan(topic: str, llm_response: str = "") -> ResearchPlan:
    """
    Build a research plan from a topic. If an LLM response is provided with
    structured aspects, parse it. Otherwise, generate a default plan.

    Args:
        topic: The research topic
        llm_response: Optional LLM-generated plan with aspects

    Returns:
        ResearchPlan object
    """
    if llm_response:
        return _parse_llm_plan(topic, llm_response)
    return _generate_default_plan(topic)


def _generate_default_plan(topic: str) -> ResearchPlan:
    """Generate a default research plan with standard scientific aspects."""
    # Extract core keywords (remove question words, articles, etc.)
    stop_words = {"will", "does", "do", "can", "is", "are", "the", "a", "an", "of", "to", "in", "for", "on", "with", "by", "how", "what", "why"}
    words = topic.strip().rstrip("?").lower().split()
    keywords = [w for w in words if w not in stop_words]
    short_topic = " ".join(keywords[:8])  # Use max 8 content words

    aspects = [
        ResearchAspect(
            id="A1",
            question=f"What are the primary mechanisms by which {short_topic}?",
            evidence_types_needed=["mechanism", "review", "in_vitro"],
            search_queries=[
                f"{short_topic} mechanism of action",
                f"{short_topic} biochemical pathway",
                f"{short_topic} molecular mechanism review",
            ]
        ),
        ResearchAspect(
            id="A2",
            question=f"What do RCTs and clinical trials show about {short_topic}?",
            evidence_types_needed=["RCT", "clinical_trial", "meta-analysis"],
            search_queries=[
                f"{short_topic} randomized controlled trial",
                f"{short_topic} RCT results",
                f"{short_topic} clinical trial outcomes",
                f"{short_topic} meta-analysis",
            ]
        ),
        ResearchAspect(
            id="A3",
            question=f"What observational/epidemiological evidence exists for {short_topic}?",
            evidence_types_needed=["cohort", "observational", "epidemiological"],
            search_queries=[
                f"{short_topic} cohort study",
                f"{short_topic} observational study results",
                f"{short_topic} epidemiological evidence",
                f"{short_topic} population study",
            ]
        ),
        ResearchAspect(
            id="A4",
            question=f"What are the limitations, risks, or counter-evidence regarding {short_topic}?",
            evidence_types_needed=["systematic_review", "adverse_effects", "criticism"],
            search_queries=[
                f"{short_topic} limitations side effects",
                f"{short_topic} risks adverse effects",
                f"{short_topic} criticism negative findings",
                f"{short_topic} null results no effect",
            ]
        ),
        ResearchAspect(
            id="A5",
            question=f"What is the current scientific consensus on {short_topic}?",
            evidence_types_needed=["systematic_review", "guideline", "expert_opinion"],
            search_queries=[
                f"{short_topic} systematic review 2024",
                f"{short_topic} scientific consensus",
                f"{short_topic} expert guidelines recommendations",
            ]
        ),
    ]

    return ResearchPlan(topic=topic, aspects=aspects)


def _parse_llm_plan(topic: str, llm_response: str) -> ResearchPlan:
    """Parse an LLM-generated research plan."""
    aspects = []
    current_id = 0

    # Try to parse numbered sections
    sections = re.split(r'\n(?=\d+[\.\)]\s)', llm_response)
    for section in sections:
        section = section.strip()
        if not section:
            continue
        current_id += 1
        # Extract the main question/aspect from first line
        first_line = section.split('\n')[0].strip()
        # Remove numbering
        first_line = re.sub(r'^\d+[\.\)]\s*', '', first_line)

        # Generate search queries from the aspect
        clean_topic = topic.strip().rstrip("?").lower()
        aspect_keywords = first_line.lower().replace("?", "")

        queries = [
            f"{clean_topic} {aspect_keywords}",
            f"{aspect_keywords} research evidence",
            f"{aspect_keywords} scientific study",
        ]

        aspects.append(ResearchAspect(
            id=f"A{current_id}",
            question=first_line,
            evidence_types_needed=["RCT", "observational", "review"],
            search_queries=queries
        ))

    if not aspects:
        return _generate_default_plan(topic)

    return ResearchPlan(topic=topic, aspects=aspects)


# --- Iterative Search Engine ---

# Domains that are not useful for scientific research
JUNK_DOMAINS = {
    "merriam-webster.com", "dictionary.com", "oxfordlearnersdictionaries.com",
    "cambridge.org/dictionary", "does.dc.gov", "wiktionary.org",
    "thesaurus.com", "collinsdictionary.com", "urbandictionary.com",
}


def is_junk_url(url: str) -> bool:
    """Check if a URL is from a non-research domain."""
    url_lower = url.lower()
    return any(domain in url_lower for domain in JUNK_DOMAINS)


def classify_source_type(title: str, snippet: str, metadata_type: Optional[str] = None) -> str:
    """Classify a source based on its title and snippet.

    If metadata_type is provided (from structured extraction) and is not
    "general", it takes precedence over keyword matching.
    """
    if metadata_type and metadata_type != "general":
        return metadata_type
    text = f"{title} {snippet}".lower()
    url_lower = title.lower()

    # Check URL domain for academic sources
    if any(d in url_lower for d in ["pubmed", "ncbi.nlm.nih", "pmc"]):
        # PubMed sources — try to classify further
        pass
    if any(d in url_lower for d in ["frontiersin.org", "nature.com", "sciencedirect", "wiley.com", "springer.com", "thelancet.com"]):
        # Academic publisher — likely peer-reviewed
        pass

    if any(kw in text for kw in ["meta-analysis", "meta analysis", "systematic review", "umbrella review", "cochrane"]):
        return "meta-analysis"
    if any(kw in text for kw in ["randomized controlled", "randomised controlled", "rct", "double-blind", "placebo-controlled", "crossover trial"]):
        return "RCT"
    if any(kw in text for kw in ["clinical trial", "intervention study", "controlled trial"]):
        return "clinical_trial"
    if any(kw in text for kw in ["cohort study", "longitudinal study", "prospective study", "retrospective study", "follow-up study"]):
        return "cohort"
    if any(kw in text for kw in ["observational study", "observational evidence", "cross-sectional"]):
        return "observational"
    if any(kw in text for kw in ["survey study", "population study", "population-based", "epidemiolog", "prevalence", "incidence"]):
        return "epidemiological"
    if any(kw in text for kw in ["in vitro", "animal model", "mouse study", "rat study", "preclinical", "murine", "rodent"]):
        return "animal_model"
    if any(kw in text for kw in ["mechanism", "pathway", "receptor", "molecular", "biochem", "pharmacol", "adenosine", "dopamine", "neurotransmit"]):
        return "mechanism"
    if any(kw in text for kw in ["guideline", "recommendation", "position statement", "expert opinion", "consensus statement"]):
        return "guideline"
    if any(kw in text for kw in ["review", "overview", "narrative review", "scoping review", "state of the art"]):
        return "review"
    if any(kw in text for kw in ["adverse", "side effect", "toxicity", "risk", "harm", "negative effect", "contraindic"]):
        return "adverse_effects"
    # Check if it's from a scientific domain
    if any(d in text for d in ["pubmed", "ncbi", "pmc", "frontiersin", "nature.com", "sciencedirect", "wiley.com"]):
        return "peer_reviewed"
    return "general"


def diversify_queries(base_queries: List[str], iteration: int) -> List[str]:
    """
    Generate diversified query variants for deeper coverage.

    Each iteration adds different modifiers to find new sources.
    """
    modifiers_by_iteration = [
        # Iteration 0: original queries
        [],
        # Iteration 1: add specificity
        ["site:pubmed.ncbi.nlm.nih.gov", "peer-reviewed", "2023 OR 2024 OR 2025"],
        # Iteration 2: broaden
        ["systematic review", "dose-response", "long-term effects"],
        # Iteration 3: counter-evidence
        ["limitations", "conflicting evidence", "null results", "failed to replicate"],
        # Iteration 4: mechanisms and animal models
        ["in vivo", "in vitro", "animal model", "pharmacokinetics"],
    ]

    if iteration == 0:
        return base_queries

    idx = min(iteration, len(modifiers_by_iteration) - 1)
    modifiers = modifiers_by_iteration[idx]

    diversified = []
    for query in base_queries[:2]:  # Use first 2 base queries
        for mod in modifiers:
            diversified.append(f"{query} {mod}")

    return diversified


async def run_iterative_search(
    plan: ResearchPlan,
    use_searxng: bool = True,
    use_brave: bool = True,
    brave_api_key: str = "",
    max_results_per_query: int = 10,
    progress_callback=None
) -> ResearchPlan:
    """
    Execute iterative search loops across all aspects of a research plan.

    For each iteration:
    1. Identify aspects with weak/no evidence
    2. Generate diversified queries
    3. Search using SearXNG and/or BraveSearch
    4. Extract and classify sources
    5. Update coverage tracking
    6. Repeat until all aspects have adequate evidence or max iterations reached

    Args:
        plan: Research plan with aspects to investigate
        use_searxng: Whether to use SearXNG for full-content search
        use_brave: Whether to use BraveSearch as supplementary
        brave_api_key: Brave API key
        max_results_per_query: Max results to fetch per query
        progress_callback: Optional callback for progress updates

    Returns:
        Updated ResearchPlan with sources and coverage data
    """

    def log_progress(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
        print(msg)

    log_progress(f"\n{'='*60}")
    log_progress(f"ITERATIVE DEEP RESEARCH ENGINE")
    log_progress(f"{'='*60}")
    log_progress(f"Topic: {plan.topic}")
    log_progress(f"Aspects to research: {len(plan.aspects)}")
    log_progress(f"Max iterations: {plan.max_iterations}")
    log_progress(f"Search engines: {'SearXNG' if use_searxng else ''} {'BraveSearch' if use_brave else ''}")
    log_progress(f"{'='*60}\n")

    for iteration in range(plan.max_iterations):
        plan.iteration_count = iteration + 1

        # Check which aspects need more evidence
        weak_aspects = plan.get_weak_aspects()
        if not weak_aspects:
            log_progress(f"\n✓ All aspects have adequate evidence after {iteration + 1} iterations")
            break

        log_progress(f"\n--- Iteration {iteration + 1}/{plan.max_iterations} ---")
        log_progress(f"Aspects needing more evidence: {len(weak_aspects)}/{len(plan.aspects)}")

        for aspect in weak_aspects:
            # Generate diversified queries for this iteration
            queries = diversify_queries(aspect.search_queries, iteration)
            log_progress(f"\n  Aspect {aspect.id}: {aspect.question[:60]}...")
            log_progress(f"  Current evidence: {aspect.evidence_level} ({aspect.source_count} sources)")
            log_progress(f"  Queries to run: {len(queries)}")

            # Dynamic result scaling: fetch more results for weaker aspects
            results_per_query = max_results_per_query
            if aspect.evidence_level == "none":
                results_per_query = min(max_results_per_query * 2, 40)

            # Search with SearXNG (full content)
            if use_searxng:
                await _search_with_searxng(aspect, queries, results_per_query, plan, log_progress)

            # Supplement with BraveSearch
            if use_brave and brave_api_key:
                await _search_with_brave(aspect, queries[:3], brave_api_key, plan, log_progress)

            aspect.assess_evidence_level()
            log_progress(f"  Updated evidence: {aspect.evidence_level} ({aspect.source_count} sources)")

        # Print coverage summary after each iteration
        coverage = plan.get_coverage_summary()
        log_progress(f"\n  Coverage after iteration {iteration + 1}:")
        log_progress(f"  Total unique sources: {len(plan.total_unique_urls)}")
        for a in coverage["aspects"]:
            icon = {"none": "❌", "weak": "⚠️", "moderate": "✓", "strong": "✓✓"}.get(a["evidence_level"], "?")
            log_progress(f"    {a['id']}: {icon} {a['evidence_level']} ({a['source_count']} sources)")

    log_progress(f"\n{'='*60}")
    log_progress(f"RESEARCH COMPLETE")
    log_progress(f"Total iterations: {plan.iteration_count}")
    log_progress(f"Total unique sources: {len(plan.total_unique_urls)}")
    log_progress(f"{'='*60}\n")

    return plan


async def _search_with_searxng(
    aspect: ResearchAspect,
    queries: List[str],
    results_per_query: int,
    plan: ResearchPlan,
    log_progress
):
    """Search using SearXNG with full content extraction."""
    try:
        async with SearxngClient() as client:
            if not await client.validate_connection():
                log_progress("    SearXNG not available, skipping")
                return

            async with DeepResearch(client) as research:
                for query in queries:
                    try:
                        result = await research.deep_dive(
                            query=query,
                            top_n=results_per_query,
                            engines=['google', 'bing', 'brave']
                        )

                        for sr in result.search_results:
                            if is_junk_url(sr.url):
                                continue
                            source_type = classify_source_type(sr.title, sr.snippet)
                            aspect.add_source(
                                title=sr.title,
                                url=sr.url,
                                source_type=source_type,
                                snippet=sr.snippet
                            )
                            plan.total_unique_urls.add(sr.url)

                        # Also capture scraped content metadata
                        for page in result.scraped_pages:
                            if not page.error and page.url not in plan.total_unique_urls:
                                if is_junk_url(page.url):
                                    continue
                                source_type = classify_source_type(page.title, page.content[:500])
                                aspect.add_source(
                                    title=page.title,
                                    url=page.url,
                                    source_type=source_type,
                                    snippet=page.content[:300]
                                )
                                plan.total_unique_urls.add(page.url)

                    except Exception as e:
                        logger.warning(f"SearXNG search failed for query '{query[:50]}': {e}")

    except Exception as e:
        logger.warning(f"SearXNG connection failed: {e}")


async def _search_with_brave(
    aspect: ResearchAspect,
    queries: List[str],
    api_key: str,
    plan: ResearchPlan,
    log_progress
):
    """Search using BraveSearch API."""
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}

    async with httpx.AsyncClient(timeout=15.0) as client:
        for query in queries:
            try:
                params = {"q": query, "count": 10}
                response = await client.get(url, headers=headers, params=params)
                if response.status_code == 200:
                    results = response.json().get("web", {}).get("results", [])
                    for r in results:
                        r_url = r.get("url", "")
                        if is_junk_url(r_url):
                            continue
                        source_type = classify_source_type(
                            r.get("title", ""),
                            r.get("description", "")
                        )
                        aspect.add_source(
                            title=r.get("title", ""),
                            url=r_url,
                            source_type=source_type,
                            snippet=r.get("description", "")
                        )
                        plan.total_unique_urls.add(r_url)
            except Exception as e:
                logger.warning(f"BraveSearch failed for query '{query[:50]}': {e}")


# --- Plan Comparison ---

def compare_plan_vs_results(plan: ResearchPlan) -> str:
    """
    Compare the original research plan against actual findings.
    Returns a formatted report documenting coverage gaps and quality.
    """
    lines = [
        f"# Research Plan vs Results Comparison",
        f"## Topic: {plan.topic}",
        f"",
        f"**Total Sources Found:** {len(plan.total_unique_urls)}",
        f"**Research Iterations:** {plan.iteration_count}",
        f"",
        f"## Coverage Analysis",
        f""
    ]

    fully_covered = 0
    partially_covered = 0
    not_covered = 0

    for aspect in plan.aspects:
        aspect.assess_evidence_level()
        if aspect.evidence_level in ("strong",):
            fully_covered += 1
            status = "FULLY COVERED"
        elif aspect.evidence_level in ("moderate",):
            partially_covered += 1
            status = "PARTIALLY COVERED"
        else:
            not_covered += 1
            status = "INSUFFICIENT"

        lines.append(f"### {aspect.id}: {aspect.question}")
        lines.append(f"- **Status:** {status}")
        lines.append(f"- **Evidence Level:** {aspect.evidence_level}")
        lines.append(f"- **Sources Found:** {aspect.source_count}")

        if aspect.sources_found:
            types = {}
            for s in aspect.sources_found:
                t = s["type"]
                types[t] = types.get(t, 0) + 1
            lines.append(f"- **Source Types:** {', '.join(f'{t}({c})' for t, c in types.items())}")

        # Note what's missing
        found_types = {s["type"] for s in aspect.sources_found}
        needed = set(aspect.evidence_types_needed)
        missing = needed - found_types
        if missing:
            lines.append(f"- **Missing Evidence Types:** {', '.join(missing)}")

        lines.append("")

    lines.extend([
        f"## Summary",
        f"- Fully covered aspects: {fully_covered}/{len(plan.aspects)}",
        f"- Partially covered: {partially_covered}/{len(plan.aspects)}",
        f"- Insufficient evidence: {not_covered}/{len(plan.aspects)}",
        f"",
        f"## Recommendations for Future Plans",
    ])

    if not_covered > 0:
        lines.append("- Consider narrowing the topic to improve evidence density")
        lines.append("- Add more specific search queries for under-researched aspects")
    if partially_covered > 0:
        lines.append("- Increase max_iterations for deeper research on complex topics")
        lines.append("- Add academic database-specific queries (PubMed, Google Scholar)")

    return "\n".join(lines)


async def run_supplementary_research(
    plan: ResearchPlan,
    audit_text: str,
    use_searxng: bool = True,
    use_brave: bool = False,
    brave_api_key: str = "",
    max_results_per_query: int = 10
) -> tuple:
    """
    Analyze audit report for evidence gaps and run targeted supplementary searches.
    Returns (supplementary_report: str, new_source_count: int).
    """
    # Find aspects with weak or no evidence
    weak_aspects = [
        a for a in plan.aspects
        if a.evidence_level in ("none", "weak")
    ]

    if not weak_aspects:
        return "All research aspects have moderate or better evidence coverage. No supplementary research needed.", 0

    print(f"\n  Found {len(weak_aspects)} aspects with insufficient evidence:")
    for a in weak_aspects:
        print(f"    - {a.question} ({a.evidence_level}, {a.source_count} sources)")

    new_sources_total = 0
    report_lines = [
        "# Supplementary Research Report",
        f"## Triggered by evidence gaps in {len(weak_aspects)} research aspects",
        ""
    ]

    def log_supp(msg):
        print(msg)

    for aspect in weak_aspects:
        print(f"\n  Supplementary search for: {aspect.question}")
        queries = diversify_queries(aspect.search_queries, iteration=plan.iteration_count)
        before_count = aspect.source_count

        # Reuse the same search helpers as the main iterative engine
        if use_searxng:
            await _search_with_searxng(aspect, queries, max_results_per_query, plan, log_supp)

        if use_brave and brave_api_key:
            await _search_with_brave(aspect, queries[:3], brave_api_key, plan, log_supp)

        aspect.assess_evidence_level()
        aspect_new_sources = aspect.source_count - before_count
        new_sources_total += aspect_new_sources

        report_lines.append(f"### {aspect.question}")
        report_lines.append(f"- New sources found: {aspect_new_sources}")
        report_lines.append(f"- Updated evidence level: {aspect.evidence_level}")
        report_lines.append(f"- Total sources now: {aspect.source_count}")
        report_lines.append("")

    report_lines.extend([
        "## Summary",
        f"- Total new sources from supplementary research: {new_sources_total}",
        f"- Total unique sources across all research: {len(plan.total_unique_urls)}",
    ])

    return "\n".join(report_lines), new_sources_total


# --- Standalone Test ---

async def test_research_engine():
    """Test the research planner and iterative search engine."""
    topic = "does coffee intake improve cognitive performance and productivity?"

    print("Building research plan...")
    plan = build_research_plan(topic)

    print(f"\nResearch Plan:")
    print(f"  Topic: {plan.topic}")
    print(f"  Aspects: {len(plan.aspects)}")
    for aspect in plan.aspects:
        print(f"  {aspect.id}: {aspect.question[:70]}...")
        print(f"    Queries: {len(aspect.search_queries)}")

    brave_key = os.getenv("BRAVE_API_KEY", "")

    # Run iterative search (limit to 2 iterations for testing)
    plan.max_iterations = 2
    plan = await run_iterative_search(
        plan,
        use_searxng=True,
        use_brave=bool(brave_key),
        brave_api_key=brave_key,
        max_results_per_query=5
    )

    # Compare plan vs results
    comparison = compare_plan_vs_results(plan)
    print("\n" + comparison)

    return plan


if __name__ == "__main__":
    asyncio.run(test_research_engine())
