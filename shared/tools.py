"""
shared/tools.py — CrewAI tool functions for research library access.

Tools reference a FlowContext object that holds output_dir and deep_reports.
Each flow sets the context before creating agents/tasks.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Union

from crewai.tools import tool

from link_validator_tool import LinkValidatorTool
from search_agent import SearxngClient, DeepResearch


# ---------------------------------------------------------------------------
# Flow context (mutable singleton — set by each flow before running)
# ---------------------------------------------------------------------------
class FlowContext:
    """Holds runtime state that tools need."""
    output_dir: Path = Path(".")
    deep_reports: dict = None

    @classmethod
    def set(cls, output_dir: Path, deep_reports=None):
        cls.output_dir = output_dir
        cls.deep_reports = deep_reports


# Module-level list for gap-fill search requests
_pending_search_requests: list[dict] = []


def get_pending_search_requests() -> list[dict]:
    return _pending_search_requests


def clear_pending_search_requests():
    _pending_search_requests.clear()


# ---------------------------------------------------------------------------
# Link validator instance
# ---------------------------------------------------------------------------
link_validator = LinkValidatorTool()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool("ReadValidationResults")
def read_validation_results(url: str) -> str:
    """Look up the pre-validated status of a URL from batch validation results.

    Args:
        url: The URL to check.
    """
    validation_file = FlowContext.output_dir / "url_validation_results.json"
    if not validation_file.exists():
        return "No pre-validation data available. Use Link Validator to check this URL."
    try:
        data = json.loads(validation_file.read_text())
        return data.get(url, f"Not pre-validated. Use Link Validator to check: {url}")
    except Exception as e:
        return f"Error reading validation data: {e}"


@tool("ListResearchSources")
def list_research_sources(role: str) -> str:
    """List all available research sources from the deep research pre-scan.

    Args:
        role: Either "lead" (supporting evidence) or "counter" (opposing evidence)

    Returns a numbered index with title, URL, and research goal for each source.
    Use ReadResearchSource to read the full summary of any specific source.
    """
    sources_file = FlowContext.output_dir / "deep_research_sources.json"
    if not sources_file.exists():
        return "No research library available. Deep research pre-scan may not have run."
    try:
        data = json.loads(sources_file.read_text())
    except Exception as e:
        return f"Error reading research library: {e}"

    role_key = role.strip().lower()
    if role_key not in data:
        return f"Unknown role '{role}'. Available: {', '.join(data.keys())}"

    sources = data[role_key]
    header = f"=== {role_key.upper()} RESEARCHER SOURCES ({len(sources)} total) ===\n\n"
    lines = []
    for s in sources:
        meta = s.get("metadata")
        study_tag = f" [{meta.get('study_type', 'general')}]" if meta and meta.get("study_type") else ""
        lines.append(
            f"[{s['index']}]{study_tag} \"{s['title']}\"\n"
            f"    URL: {s['url']}\n"
            f"    Goal: {s['goal']}"
        )
    return header + "\n\n".join(lines)


@tool("ReadResearchSource")
def read_research_source(role_and_index: str) -> str:
    """Read the full summary of a specific research source from the deep research pre-scan.

    Args:
        role_and_index: Format "role:index" e.g. "lead:5" or "counter:12"
    """
    sources_file = FlowContext.output_dir / "deep_research_sources.json"
    if not sources_file.exists():
        return "No research library available."
    try:
        data = json.loads(sources_file.read_text())
    except Exception as e:
        return f"Error reading research library: {e}"

    parts = role_and_index.strip().split(":")
    if len(parts) != 2:
        return "Invalid format. Use 'role:index' e.g. 'lead:5' or 'counter:12'."
    role_key, idx_str = parts[0].strip().lower(), parts[1].strip()

    if role_key not in data:
        return f"Unknown role '{role_key}'. Available: {', '.join(data.keys())}"
    try:
        idx = int(idx_str)
    except ValueError:
        return f"Invalid index '{idx_str}'. Must be a number."

    sources = data[role_key]
    if idx < 0 or idx >= len(sources):
        return f"Index {idx} out of range. {role_key} has {len(sources)} sources (0-{len(sources)-1})."

    s = sources[idx]
    meta_section = ""
    meta = s.get("metadata")
    if meta:
        meta_lines = ["--- STUDY METADATA ---"]
        for key, label in [
            ("study_type", "Study Type"), ("sample_size", "Sample Size"),
            ("key_result", "Key Result"), ("journal_name", "Journal"),
            ("publication_year", "Year"), ("effect_size", "Effect Size"),
            ("authors", "Authors"), ("demographics", "Demographics"),
            ("limitations", "Limitations"),
        ]:
            if meta.get(key):
                meta_lines.append(f"{label}: {meta[key]}")
        meta_section = "\n".join(meta_lines) + "\n\n"

    return (
        f"=== SOURCE [{idx}] ===\n"
        f"Title: {s['title']}\n"
        f"URL: {s['url']}\n"
        f"Query: {s['query']}\n"
        f"Goal: {s['goal']}\n\n"
        f"{meta_section}"
        f"--- FULL SUMMARY ---\n{s['summary']}"
    )


@tool("ReadFullReport")
def read_full_report(report_name: str) -> str:
    """Read a full research report from disk. Available reports:
    - "lead": Full supporting evidence report
    - "counter": Full opposing evidence report
    - "audit": Full audit synthesis report
    - "framing": Research framing document

    WARNING: Reports can be very long. Prefer ListResearchSources + ReadResearchSource.
    """
    name_map = {
        "lead": "deep_research_lead.md",
        "counter": "deep_research_counter.md",
        "audit": "deep_research_audit.md",
        "framing": "RESEARCH_FRAMING.md",
    }
    key = report_name.strip().lower()
    if key not in name_map:
        return f"Unknown report '{report_name}'. Available: {', '.join(name_map.keys())}"

    report_path = FlowContext.output_dir / name_map[key]
    if not report_path.exists():
        return f"Report file not found: {report_path}"

    content = report_path.read_text()
    if len(content) > 15000:
        return (
            content[:15000]
            + f"\n\n... [TRUNCATED \u2014 full report is {len(content)} chars. "
            f"Use ListResearchSources + ReadResearchSource for targeted reading.] ..."
        )
    return content


@tool("RequestSearch")
def request_search(search_requests_json: Union[str, list]) -> str:
    """Request targeted searches to fill evidence gaps.

    Args:
        search_requests_json: JSON array of search requests.
            Each must have "query" and "goal".
    """
    try:
        if isinstance(search_requests_json, list):
            requests_list = search_requests_json
        else:
            requests_list = json.loads(search_requests_json)
        if not isinstance(requests_list, list):
            return "ERROR: Input must be a JSON array of {query, goal} objects."
        valid = []
        for req in requests_list:
            if isinstance(req, dict) and "query" in req and "goal" in req:
                valid.append({"query": req["query"], "goal": req["goal"]})
        if not valid:
            return "ERROR: No valid search requests. Each must have 'query' and 'goal' keys."
        _pending_search_requests.extend(valid)
        return (
            f"Queued {len(valid)} search request(s). These searches run ASYNCHRONOUSLY in a later phase \u2014 "
            f"results will NOT appear in ListResearchSources until a future round. "
            f"Do NOT call ListResearchSources again to check for new results. "
            f"IMMEDIATELY proceed to write your FINAL ANSWER using the evidence you have already gathered."
        )
    except (json.JSONDecodeError, TypeError):
        return 'ERROR: Invalid JSON. Provide a JSON array like: [{"query": "...", "goal": "..."}]'


@tool("BraveSearch")
def search_tool(search_query: str):
    """Search for scientific evidence using Brave Search API.

    PRIMARY SOURCES (Search First):
    1. Peer-reviewed journals: Nature, Science, Lancet, Cell, PNAS
    2. Recent data published after 2024
    3. RCTs and meta-analyses

    SECONDARY SOURCES (If primary insufficient):
    4. Observatory studies and cohort studies
    5. Cross-sectional population studies

    SUPPLEMENTARY EVIDENCE:
    7. Non-human RCTs (animal studies, in vitro)
    8. Mechanistic studies
    """
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return "Brave API Key missing. Use internal knowledge."

    import httpx
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
    params = {"q": search_query, "count": 5}

    try:
        response = httpx.get(url, headers=headers, params=params, timeout=15.0)
        if response.status_code == 200:
            results = response.json().get("web", {}).get("results", [])
            return "\n\n".join(
                [f"Title: {r['title']}\nURL: {r['url']}\nDesc: {r['description']}" for r in results]
            ) or "No results found."
        return "Search API error. Use internal knowledge."
    except Exception as e:
        return f"Search failed: {e}"


@tool("DeepSearch")
def deep_search_tool(search_query: str) -> str:
    """Deep research using self-hosted SearXNG with full content extraction.

    Provides FULL PAGE CONTENT (not just snippets) from top 5 results.
    Uses local SearXNG (no API key required).
    """
    async def perform_deep_search():
        try:
            async with SearxngClient() as client:
                if not await client.validate_connection():
                    return (
                        "SearXNG not accessible at http://localhost:8080\n"
                        "Falling back to internal knowledge or use BraveSearch."
                    )
                async with DeepResearch(client) as research:
                    results = await research.deep_dive(
                        query=search_query, top_n=5, engines=['google', 'bing', 'brave']
                    )
                    if not results.scraped_pages:
                        return "No results found. Use internal knowledge."
                    output = f"=== Deep Research Results for: {search_query} ===\n\n"
                    for i, content in enumerate(results.scraped_pages, 1):
                        if not content.error:
                            output += (
                                f"--- SOURCE {i}: {content.title} ---\n"
                                f"URL: {content.url}\n"
                                f"Content Length: {content.word_count} words\n\n"
                                f"{content.content}\n\n"
                                "=" * 80 + "\n\n"
                            )
                        else:
                            output += f"--- SOURCE {i}: [FAILED] {content.url} ---\nError: {content.error}\n\n"
                    if results.errors:
                        output += f"\nSome sources failed to load ({len(results.errors)} errors)\n"
                    return output
        except Exception as e:
            return f"Deep search failed: {e}\nTry BraveSearch as fallback."

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(perform_deep_search())


# ---------------------------------------------------------------------------
# Gap-fill search execution
# ---------------------------------------------------------------------------
async def execute_gap_fill_searches(
    pending_requests: list[dict],
    role: str,
    brave_api_key: str,
    fast_model_available: bool = True,
) -> list[dict]:
    """Execute gap-fill searches using deep_research_agent pipeline."""
    from deep_research_agent import (
        SearchService, ContentFetcher, FastWorker, ResearchAgent, ResearchQuery,
        SMART_MODEL, SMART_BASE_URL, FAST_MODEL, FAST_BASE_URL,
    )
    from openai import AsyncOpenAI

    smart_client = AsyncOpenAI(base_url=SMART_BASE_URL, api_key="NA")
    fast_client = AsyncOpenAI(base_url=FAST_BASE_URL, api_key="NA") if fast_model_available else None
    fast_worker = FastWorker(fast_client, FAST_MODEL) if fast_model_available else None
    search_svc = SearchService(brave_api_key)
    fetcher = ContentFetcher(max_concurrent=10)

    agent = ResearchAgent(
        smart_client=smart_client, fast_worker=fast_worker,
        search_service=search_svc, fetcher=fetcher,
        smart_model=SMART_MODEL, results_per_query=5, max_iterations=1,
    )

    queries = [ResearchQuery(query=r["query"], goal=r["goal"]) for r in pending_requests]
    summaries, _, _ = await agent._search_and_summarize(queries, set(), print)

    new_sources = []
    for s in summaries:
        if s.error or not s.summary or s.summary.strip().upper() == "NO RELEVANT DATA":
            continue
        meta = None
        if s.metadata:
            import dataclasses
            meta = {f.name: getattr(s.metadata, f.name) for f in dataclasses.fields(s.metadata)}
        new_sources.append({
            "url": s.url, "title": s.title, "query": s.query,
            "goal": s.goal, "summary": s.summary, "metadata": meta,
        })
    return new_sources


def append_sources_to_library(new_sources: list[dict], role: str, output_dir: Path = None):
    """Append new sources to deep_research_sources.json."""
    src_dir = output_dir or FlowContext.output_dir
    sources_file = Path(src_dir) / "deep_research_sources.json"
    if sources_file.exists():
        data = json.loads(sources_file.read_text())
    else:
        data = {"lead": [], "counter": []}
    role_key = role.strip().lower()
    if role_key not in data:
        data[role_key] = []
    start_idx = len(data[role_key])
    for i, src in enumerate(new_sources):
        src["index"] = start_idx + i
        data[role_key].append(src)
    with open(sources_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Appended {len(new_sources)} sources to {role_key} library (total: {len(data[role_key])})")
