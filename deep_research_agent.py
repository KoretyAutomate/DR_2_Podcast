"""
Deep Research Agent - Dual-Model Iterative Delegation Architecture

Optimized for Nvidia DGX Spark (128GB Unified Memory):
- SMART MODEL (Qwen2.5-32B-Instruct-AWQ) on port 8000: Reasoning, planning, evaluation
- FAST MODEL (Phi-4 Mini via Ollama) on port 11434: Parallel content summarization

Architecture (Option B - Agent-driven iterative delegation):
  The Smart Model acts as a researcher with a specific role (lead/counter).
  It iteratively:
    1. Plans what to search next
    2. Delegates search + summarization to workers (SearXNG + Fast Model)
    3. Evaluates gathered evidence, identifies gaps
    4. Repeats until evidence is sufficient or max iterations reached
    5. Writes final report from all gathered evidence

Author: DR_2_Podcast Team
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from openai import AsyncOpenAI

from search_agent import SearxngClient, SearchResult

logger = logging.getLogger(__name__)

# --- Configuration ---

SMART_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
SMART_BASE_URL = "http://localhost:8000/v1"

FAST_MODEL = "phi4-mini"
FAST_BASE_URL = "http://localhost:11434/v1"

MAX_INPUT_TOKENS = 32000
MAX_CONCURRENT_SUMMARIES = 2
MAX_RESEARCH_ITERATIONS = 3
SCRAPING_TIMEOUT = 20.0
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

JUNK_DOMAINS = {
    "dictionary.com", "merriam-webster.com", "thefreedictionary.com",
    "cambridge.org", "wiktionary.org", "vocabulary.com",
    "thesaurus.com", "urbandictionary.com",
    "facebook.com", "fb.com", "twitter.com", "instagram.com", "tiktok.com",
    "pinterest.com", "reddit.com", "youtube.com",
    "starbucks.com", "amazon.com", "walmart.com",
    "dailythemedcrosswordanswers.com", "crosswordanswers.com",
}

def is_junk_url(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    return any(junk in domain for junk in JUNK_DOMAINS)


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
class SummarizedSource:
    url: str
    title: str
    summary: str
    query: str
    goal: str
    error: Optional[str] = None

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


# --- Worker Services (IO + Fast Model) ---

class SearchService:
    """Searches via SearXNG and BraveSearch."""

    def __init__(self, brave_api_key: str = ""):
        self.brave_api_key = brave_api_key

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        results = []
        try:
            async with SearxngClient() as client:
                if await client.validate_connection():
                    raw = await client.search(query, num_results=max_results)
                    for r in raw:
                        url = r.get("url", "") if isinstance(r, dict) else getattr(r, "url", "")
                        title = r.get("title", "") if isinstance(r, dict) else getattr(r, "title", "")
                        snippet = r.get("content", "") if isinstance(r, dict) else getattr(r, "snippet", "")
                        if url:
                            results.append({"url": url, "title": title, "snippet": snippet})
        except Exception as e:
            logger.warning(f"SearXNG search failed: {e}")

        if self.brave_api_key and len(results) < max_results:
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
                            results.append({
                                "url": r.get("url", ""),
                                "title": r.get("title", ""),
                                "snippet": r.get("description", "")
                            })
            except Exception as e:
                logger.warning(f"BraveSearch failed: {e}")

        seen = set()
        unique = []
        for r in results:
            url = r["url"]
            if url not in seen and not is_junk_url(url):
                seen.add(url)
                unique.append(r)
        return unique[:max_results]


class ContentFetcher:
    """Async parallel content fetcher."""

    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_page(self, url: str) -> FetchedPage:
        async with self.semaphore:
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
                    return FetchedPage(url=url, title=title, content=text, word_count=len(text.split()))
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

    async def summarize(self, page: FetchedPage, goal: str, query: str) -> SummarizedSource:
        if page.error or not page.content.strip():
            return SummarizedSource(
                url=page.url, title=page.title, summary="",
                query=query, goal=goal, error=page.error or "Empty content"
            )
        content = page.content[:MAX_INPUT_TOKENS * 4]
        system_prompt = (
            f"You are a precise data extractor. Extract facts relevant to: '{goal}'. "
            f"Output ONLY the facts as a bulleted list. Do not summarize unrelated sections. "
            f"Be extremely concise. If no relevant information is found, output 'NO RELEVANT DATA'."
        )
        async with self.semaphore:
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Source URL: {page.url}\n\nContent:\n{content}"}
                    ],
                    max_tokens=1024, temperature=0.1, timeout=180
                )
                summary = resp.choices[0].message.content.strip()
                return SummarizedSource(url=page.url, title=page.title, summary=summary, query=query, goal=goal)
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
        results_per_query: int = 8,
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
            evidence_blocks.append(
                f"### Source: {s.title or s.url}\n"
                f"**URL:** {s.url}\n"
                f"**Research Goal:** {s.goal}\n"
                f"**Extracted Facts:**\n{s.summary}\n"
            )
        aggregated = "\n---\n".join(evidence_blocks) if evidence_blocks else "No evidence gathered."
        if len(aggregated) > 80000:
            aggregated = aggregated[:80000] + "\n\n[...truncated...]"

        report_system = (
            f"You are a {role}. {role_instructions}\n\n"
            f"Write a comprehensive research report based ONLY on the evidence provided.\n"
            f"Structure:\n"
            f"1. Executive Summary\n"
            f"2. Key Findings (grouped by theme)\n"
            f"3. Evidence Quality Assessment\n"
            f"4. Gaps & Limitations\n"
            f"5. Bibliography (all source URLs)\n\n"
            f"Be factual. Cite sources by URL. Note evidence strength."
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

        return ResearchReport(
            topic=topic,
            role=role,
            sources=all_summaries,
            report=report_text,
            iterations_used=min(iteration + 1, self.max_iterations),
            total_urls_fetched=total_fetched,
            total_summaries=len(good_summaries),
            total_errors=total_errors,
            duration_seconds=duration
        )


# --- Orchestrator: Full Pipeline ---

class Orchestrator:
    """
    Runs the full DR_2_Podcast research pipeline with dual-model delegation.

    Pipeline:
    1. Lead Researcher: iterative search for supporting evidence
    2. Counter Researcher: iterative search for opposing evidence
    3. Auditor: evaluates both reports, identifies remaining gaps
    4. Final synthesis: combined research report
    """

    def __init__(
        self,
        smart_base_url: str = SMART_BASE_URL,
        fast_base_url: str = FAST_BASE_URL,
        smart_model: str = SMART_MODEL,
        fast_model: str = FAST_MODEL,
        brave_api_key: str = "",
        results_per_query: int = 8,
        max_iterations: int = MAX_RESEARCH_ITERATIONS,
        fast_model_available: bool = True
    ):
        self.smart_client = AsyncOpenAI(base_url=smart_base_url, api_key="NA")
        self.fast_client = AsyncOpenAI(base_url=fast_base_url, api_key="NA") if fast_model_available else None
        self.smart_model = smart_model

        fast_worker = FastWorker(self.fast_client, fast_model) if fast_model_available else None
        search_svc = SearchService(brave_api_key)
        fetcher = ContentFetcher(max_concurrent=15)

        self.lead_researcher = ResearchAgent(
            self.smart_client, fast_worker, search_svc, fetcher,
            smart_model, results_per_query, max_iterations
        )
        self.counter_researcher = ResearchAgent(
            self.smart_client, fast_worker, search_svc, fetcher,
            smart_model, results_per_query, max_iterations
        )
        self.fast_model_available = fast_model_available

    async def run(self, topic: str, framing_context: str = "", progress_callback=None) -> Dict[str, ResearchReport]:
        """Run the full research pipeline. Returns dict of role→report.

        Args:
            topic: Research topic
            framing_context: Optional research framing document to guide searches
            progress_callback: Optional callback for progress messages
        """
        start_time = time.time()

        def log(msg: str):
            logger.info(msg)
            print(msg)
            if progress_callback:
                progress_callback(msg)

        mode = "DUAL-MODEL" if self.fast_model_available else "SINGLE-MODEL"
        log(f"\n{'='*70}")
        log(f"DEEP RESEARCH AGENT - Iterative Delegation ({mode})")
        log(f"{'='*70}")
        log(f"Topic: {topic}")
        if framing_context:
            log(f"Research framing provided: {len(framing_context)} chars")
        log(f"{'='*70}")

        # Build framing prefix for role instructions
        framing_prefix = ""
        if framing_context:
            framing_prefix = (
                f"A Research Framing document has been prepared to guide your investigation. "
                f"Use the core questions, scope boundaries, and evidence criteria below to focus "
                f"your searches systematically:\n\n{framing_context}\n\n"
                f"--- END FRAMING ---\n\n"
            )

        # Phase 1: Lead Researcher
        log(f"\n{'='*70}")
        log(f"PHASE 1: LEAD RESEARCHER")
        log(f"{'='*70}")
        lead_report = await self.lead_researcher.research(
            topic=topic,
            role="Lead Researcher (Principal Investigator)",
            role_instructions=(
                f"{framing_prefix}"
                "Your job is to find SUPPORTING scientific evidence for the topic. "
                "Focus on: mechanisms of action, clinical trials (RCTs), meta-analyses, "
                "and expert consensus that SUPPORTS the claim. "
                "Prioritize peer-reviewed sources. Include specific data points, "
                "study sizes, and effect sizes when available."
            ),
            log=log
        )

        # Phase 2: Counter Researcher
        log(f"\n{'='*70}")
        log(f"PHASE 2: COUNTER RESEARCHER")
        log(f"{'='*70}")
        counter_report = await self.counter_researcher.research(
            topic=topic,
            role="Counter Researcher (The Skeptic)",
            role_instructions=(
                f"{framing_prefix}"
                "Your job is to find OPPOSING and CONTRADICTORY evidence for the topic. "
                "Focus on: studies showing null effects, negative outcomes, methodological "
                "flaws in supporting studies, and expert criticism. "
                "Search specifically for 'criticism of [topic]', 'limitations of [topic]', "
                "and 'no effect' findings. Be adversarial but evidence-based."
            ),
            log=log
        )

        # Phase 3: Auditor synthesis
        log(f"\n{'='*70}")
        log(f"PHASE 3: AUDITOR SYNTHESIS")
        log(f"{'='*70}")

        audit_system = (
            "You are a Scientific Auditor. Review both the supporting and opposing research reports. "
            "Write a balanced meta-audit that:\n"
            "1. Grades each major claim on a 1-10 evidence scale\n"
            "2. Creates a Reliability Scorecard\n"
            "3. Lists caveats (The Caveat Box): why findings might be wrong\n"
            "4. Identifies the overall scientific consensus\n"
            "5. Lists all cited URLs\n\n"
            "Be impartial. Evaluate evidence quality, not quantity."
        )

        combined_evidence = (
            f"TOPIC: {topic}\n\n"
            f"=== SUPPORTING EVIDENCE (Lead Researcher) ===\n"
            f"{lead_report.report}\n\n"
            f"=== OPPOSING EVIDENCE (Counter Researcher) ===\n"
            f"{counter_report.report}"
        )

        # Truncate if needed
        if len(combined_evidence) > 80000:
            combined_evidence = combined_evidence[:80000] + "\n\n[...truncated...]"

        try:
            resp = await self.smart_client.chat.completions.create(
                model=self.smart_model,
                messages=[
                    {"role": "system", "content": audit_system},
                    {"role": "user", "content": combined_evidence}
                ],
                max_tokens=8000, temperature=0.2, timeout=300
            )
            audit_text = resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Audit failed: {e}")
            audit_text = f"Audit synthesis failed: {e}\n\n{combined_evidence}"

        audit_report = ResearchReport(
            topic=topic, role="Auditor",
            sources=lead_report.sources + counter_report.sources,
            report=audit_text,
            iterations_used=0,
            total_urls_fetched=lead_report.total_urls_fetched + counter_report.total_urls_fetched,
            total_summaries=lead_report.total_summaries + counter_report.total_summaries,
            total_errors=lead_report.total_errors + counter_report.total_errors,
            duration_seconds=time.time() - start_time
        )

        total_time = time.time() - start_time
        log(f"\n{'='*70}")
        log(f"ALL RESEARCH COMPLETE in {total_time:.0f}s")
        log(f"  Lead: {lead_report.total_summaries} sources in {lead_report.duration_seconds:.0f}s")
        log(f"  Counter: {counter_report.total_summaries} sources in {counter_report.duration_seconds:.0f}s")
        log(f"  Total unique sources: {lead_report.total_summaries + counter_report.total_summaries}")
        log(f"{'='*70}\n")

        return {
            "lead": lead_report,
            "counter": counter_report,
            "audit": audit_report,
        }


# --- Convenience functions ---

async def run_deep_research(
    topic: str,
    brave_api_key: str = "",
    results_per_query: int = 8,
    max_iterations: int = MAX_RESEARCH_ITERATIONS,
    fast_model_available: bool = True,
    framing_context: str = ""
) -> Dict[str, ResearchReport]:
    orchestrator = Orchestrator(
        brave_api_key=brave_api_key,
        results_per_query=results_per_query,
        max_iterations=max_iterations,
        fast_model_available=fast_model_available
    )
    return await orchestrator.run(topic, framing_context=framing_context)


async def main():
    """Test the iterative delegation research agent."""
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

    reports = await run_deep_research(
        topic=topic,
        brave_api_key=brave_key,
        results_per_query=5,
        max_iterations=2,  # Limit for testing
        fast_model_available=fast_available
    )

    # Save reports
    from pathlib import Path
    output_dir = Path("research_outputs/test_deep_agent")
    output_dir.mkdir(parents=True, exist_ok=True)

    for role, report in reports.items():
        filename = output_dir / f"{role}_report.md"
        with open(filename, "w") as f:
            f.write(report.report)
        print(f"Saved {role} report: {filename} ({len(report.report)} chars)")

    print(f"\nTotal sources: {reports['audit'].total_summaries}")
    print(f"Total time: {reports['audit'].duration_seconds:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
