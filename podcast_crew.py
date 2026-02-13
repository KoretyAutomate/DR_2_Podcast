import os
import platform
import re
import httpx
import time
import random
import sys
import json
import argparse
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from fpdf import FPDF
from link_validator_tool import LinkValidatorTool
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Literal
import soundfile as sf
import numpy as np
import wave
from audio_engine import generate_audio_from_script, clean_script_for_tts, post_process_audio
from search_agent import SearxngClient, DeepResearch
from research_planner import build_research_plan, run_iterative_search, compare_plan_vs_results, run_supplementary_research
from deep_research_agent import Orchestrator, run_deep_research

_pending_search_requests: list[dict] = []

# --- SOURCE TRACKING MODELS ---
class ScientificSource(BaseModel):
    """Structured scientific source."""
    title: str
    url: HttpUrl
    journal: Optional[str] = None
    publication_year: Optional[int] = None
    source_type: Literal["peer_reviewed", "preprint", "review", "meta_analysis", "web_article"]
    trust_level: Literal["high", "medium", "low"] = "medium"
    cited_by: str  # Which agent cited this
    key_finding: Optional[str] = None

class SourceBibliography(BaseModel):
    """Complete bibliography with categorization."""
    supporting_sources: List[ScientificSource] = []
    contradicting_sources: List[ScientificSource] = []

    def get_high_trust_sources(self) -> List[ScientificSource]:
        """Filter for high-trust peer-reviewed sources."""
        all_sources = self.supporting_sources + self.contradicting_sources
        return [s for s in all_sources if s.trust_level == "high" and s.source_type == "peer_reviewed"]

# Audio generation now uses Kokoro TTS (local, high-quality)
# MetaVoice-1B has been deprecated in favor of Kokoro-82M

# Setup logging (will be reconfigured after output_dir is created)
def setup_logging(output_dir: Path):
    """Configure logging with timestamped output directory"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'podcast_generation.log'),
            logging.StreamHandler()
        ],
        force=True
    )

# --- INITIALIZATION ---
load_dotenv()
# Override .env settings for model configuration
# Using vLLM with Qwen2.5-32B-Instruct-AWQ (supports function/tool calling)
os.environ["MODEL_NAME"] = "Qwen/Qwen2.5-32B-Instruct-AWQ"
os.environ["LLM_BASE_URL"] = "http://localhost:8000/v1"
script_dir = Path(__file__).parent.absolute()
base_output_dir = script_dir / "research_outputs"
base_output_dir.mkdir(exist_ok=True)

# --- TIMESTAMPED OUTPUT DIRECTORY ---
def create_timestamped_output_dir(base_dir: Path) -> Path:
    """
    Create a timestamped subfolder for this podcast generation run.
    Format: research_outputs/YYYY-MM-DD_HH-MM-SS/

    Args:
        base_dir: Base output directory (research_outputs)

    Returns:
        Path to timestamped directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamped_dir = base_dir / timestamp
    timestamped_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"OUTPUT DIRECTORY: {timestamped_dir}")
    print(f"{'='*60}\n")

    return timestamped_dir

# Create timestamped directory for this run
output_dir = create_timestamped_output_dir(base_output_dir)

# Configure logging with new output directory
setup_logging(output_dir)

# --- TOPIC CONFIGURATION ---
def get_topic():
    """
    Get podcast topic from multiple sources (priority order):
    1. Command-line argument (--topic)
    2. Environment variable (PODCAST_TOPIC)
    3. Default topic (for backward compatibility)
    """
    parser = argparse.ArgumentParser(
        description='Generate a research-driven debate podcast on any scientific topic.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python podcast_crew.py --topic "effects of meditation on brain plasticity"
  python podcast_crew.py --topic "climate change impact on marine ecosystems"

Environment variable:
  export PODCAST_TOPIC="your topic here"
  python podcast_crew.py
        """
    )
    parser.add_argument(
        '--topic',
        type=str,
        help='Scientific topic for podcast research and debate'
    )

    # Parse known args to avoid conflicts with other argument parsers (e.g., --language)
    args, _ = parser.parse_known_args()

    # Priority: CLI arg > env var > default
    if args.topic:
        topic = args.topic
        print(f"Using topic from command-line: {topic}")
    elif os.getenv("PODCAST_TOPIC"):
        topic = os.getenv("PODCAST_TOPIC")
        print(f"Using topic from environment: {topic}")
    else:
        topic = 'scientific benefit of coffee intake to increase productivity during the day'
        print(f"Using default topic: {topic}")

    return topic

topic_name = get_topic()

# --- CHARACTER CONFIGURATION ---
CHARACTERS = {
    "Kaz": {
        "gender": "male",
        "voice_model": "male_voice",  # TTS-specific, will update in #3
        "base_personality": "Enthusiastic science advocate, optimistic, data-driven"
    },
    "Erika": {
        "gender": "female",
        "voice_model": "female_voice",  # TTS-specific, will update in #3
        "base_personality": "Skeptical analyst, cautious, evidence-focused"
    }
}

# --- ROLE ASSIGNMENT (Dynamic per session) ---
def assign_roles() -> dict:
    """Randomly assign Kaz and Erika to pro/con roles for this session."""
    characters = list(CHARACTERS.keys())
    random.shuffle(characters)

    role_assignment = {
        "pro": {
            "character": characters[0],
            "stance": "supporting",
            "personality": CHARACTERS[characters[0]]["base_personality"]
        },
        "con": {
            "character": characters[1],
            "stance": "critical",
            "personality": CHARACTERS[characters[1]]["base_personality"]
        }
    }

    print(f"\n{'='*60}")
    print(f"SESSION ROLE ASSIGNMENT:")
    print(f"  Supporting: {role_assignment['pro']['character']} ({CHARACTERS[characters[0]]['gender']})")
    print(f"  Critical: {role_assignment['con']['character']} ({CHARACTERS[characters[1]]['gender']})")
    print(f"{'='*60}\n")

    return role_assignment

SESSION_ROLES = assign_roles()

# --- TTS DEPENDENCY CHECK ---
def check_tts_dependencies():
    """Verify Kokoro TTS is installed."""
    try:
        import kokoro
        print("✓ Kokoro TTS dependencies verified")
    except ImportError as e:
        print(f"WARNING: Kokoro TTS not installed: {e}")
        print("Install with: pip install kokoro>=0.9")
        print("Audio generation will fail without Kokoro.")
        # Don't exit - let it fail gracefully during audio generation

check_tts_dependencies()

# --- LANGUAGE CONFIGURATION ---
SUPPORTED_LANGUAGES = {
    'en': {
        'name': 'English',
        'tts_code': 'a',  # Kokoro American English
        'instruction': 'Write all content in English.',
        'pdf_font': 'Helvetica'  # Latin-1 compatible
    },
    'ja': {
        'name': '日本語 (Japanese)',
        'tts_code': 'j',  # Kokoro Japanese
        'instruction': 'すべてのコンテンツを日本語で書いてください。(Write all content in Japanese.)',
        'pdf_font': 'Arial Unicode MS'  # Unicode compatible for Japanese
    }
}

def get_language():
    """
    Get podcast language from multiple sources (priority order):
    1. Command-line argument (--language)
    2. Environment variable (PODCAST_LANGUAGE)
    3. Default language (English)
    """
    parser = argparse.ArgumentParser(
        description='Generate a research-driven debate podcast in English or Japanese.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Language Options:
  en    English (default)
  ja    日本語 (Japanese)

Examples:
  python podcast_crew.py --language ja
  python podcast_crew.py --language en

Environment variable:
  export PODCAST_LANGUAGE=ja
  python podcast_crew.py
        """
    )
    parser.add_argument(
        '--language',
        type=str,
        choices=['en', 'ja'],
        help='Language for podcast generation (en=English, ja=Japanese)'
    )

    # Parse known args to avoid conflicts with other argument parsers
    args, _ = parser.parse_known_args()

    # Priority: CLI arg > env var > default
    if args.language:
        lang_code = args.language
        print(f"Using language from command-line: {SUPPORTED_LANGUAGES[lang_code]['name']}")
    elif os.getenv("PODCAST_LANGUAGE") and os.getenv("PODCAST_LANGUAGE") in SUPPORTED_LANGUAGES:
        lang_code = os.getenv("PODCAST_LANGUAGE")
        print(f"Using language from environment: {SUPPORTED_LANGUAGES[lang_code]['name']}")
    else:
        lang_code = 'en'
        print(f"Using default language: {SUPPORTED_LANGUAGES[lang_code]['name']}")

    return lang_code

language = get_language()
language_config = SUPPORTED_LANGUAGES[language]
# English-first reasoning: all research/reasoning phases use English,
# only post-translation phases (polish, show notes, accuracy check) use target language.
english_instruction = "Write all content in English."
target_instruction = language_config['instruction']
# For backward compatibility and phases that always match the target language
language_instruction = language_config['instruction']

# --- ACCESSIBILITY LEVEL CONFIG ---
# Controls how aggressively scientific terms are simplified.
#   simple  – define every term inline, heavy use of analogies (default)
#   moderate – define key terms once, then use them normally
#   technical – minimal simplification, assume some science literacy
ACCESSIBILITY_LEVEL = os.getenv("ACCESSIBILITY_LEVEL", "simple").lower()
if ACCESSIBILITY_LEVEL not in ("simple", "moderate", "technical"):
    print(f"Warning: Unknown ACCESSIBILITY_LEVEL '{ACCESSIBILITY_LEVEL}', falling back to 'simple'")
    ACCESSIBILITY_LEVEL = "simple"
print(f"Accessibility level: {ACCESSIBILITY_LEVEL}")

ACCESSIBILITY_INSTRUCTIONS = {
    "simple": (
        "Explain every scientific term the first time it appears using a one-line plain-English definition. "
        "Use everyday analogies (e.g. 'blood sugar is like fuel in a car'). "
        "After defining a term once, you may use it freely."
    ),
    "moderate": (
        "Define key domain terms once when first introduced, then use them normally. "
        "Assume the listener can follow a simple cause-and-effect explanation. "
        "Use analogies sparingly — only for the most abstract concepts."
    ),
    "technical": (
        "Use standard scientific terminology without extensive definitions. "
        "Assume the listener has basic biology knowledge (high school AP level). "
        "Focus on depth and nuance rather than simplification."
    ),
}
accessibility_instruction = ACCESSIBILITY_INSTRUCTIONS[ACCESSIBILITY_LEVEL]

# --- MODEL DETECTION & CONFIG ---
# Using Ollama with DeepSeek-R1:32b (131k context, excellent research capabilities)
DEFAULT_MODEL = "deepseek-r1:32b"  # No prefix - CrewAI detects Ollama from base_url
DEFAULT_BASE_URL = "http://localhost:11434/v1"  # Ollama OpenAI-compatible endpoint

def get_final_model_string():
    model = os.getenv("MODEL_NAME", DEFAULT_MODEL)
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
    print(f"Connecting to Ollama server at {base_url}...")

    for i in range(10):
        try:
            response = httpx.get(f"{base_url}/models", timeout=5.0)
            if response.status_code == 200:
                print(f"✓ Ollama server online! Using model: {model}")
                return model
        except Exception as e:
            if i % 5 == 0:
                print(f"Waiting for Ollama server... ({i}s) - {e}")
            time.sleep(1)

    print("Error: Could not connect to Ollama server. Check if it is running.")
    print("Start Ollama with: ollama serve")
    sys.exit(1)

final_model_string = get_final_model_string()

# LLM Configuration for Qwen2.5-32B-Instruct (32k context window, function calling support)
dgx_llm_strict = LLM(
    model=final_model_string,
    base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
    api_key="NA",  # vLLM uses "NA"
    provider="openai",
    timeout=600,
    temperature=0.1,  # Strict mode for Researcher/Auditor
    max_tokens=16000,  # Safe limit for 32k context (leaves room for input)
    stop=["<|im_end|>", "<|endoftext|>"]
)

dgx_llm_creative = LLM(
    model=final_model_string,
    base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
    api_key="NA",  # vLLM uses "NA"
    provider="openai",
    timeout=600,
    temperature=0.7,  # Creative mode for Producer/Personality
    max_tokens=16000,  # Safe limit for 32k context (leaves room for input)
    stop=["<|im_end|>", "<|endoftext|>"]
)

# Legacy alias for backward compatibility
dgx_llm = dgx_llm_strict


def summarize_report_with_fast_model(report_text: str, role: str, topic: str) -> str:
    """Condense a deep research report using phi4-mini via Ollama.

    Returns a ~2000-word summary that preserves ALL key findings (not just
    the first N characters).  Falls back to [:6000] truncation on error.
    """
    try:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        response = client.chat.completions.create(
            model="phi4-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Condense this research report into a summary that preserves "
                        "ALL key findings, claim names, source URLs, and evidence "
                        "strength ratings. Target ~2000 words. Do not drop any "
                        "findings — compress descriptions instead."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Summarize this {role} research report on '{topic}':\n\n"
                        f"{report_text}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=4000,
            timeout=180,
        )
        summary = response.choices[0].message.content.strip()
        if len(summary) > 200:
            print(f"  ✓ {role} report summarized: {len(report_text)} → {len(summary)} chars")
            return summary
        # Summary too short — fall through to truncation
        print(f"  ⚠ {role} summary too short ({len(summary)} chars), falling back to truncation")
    except Exception as e:
        print(f"  ⚠ phi4-mini summarization failed for {role}: {e}")

    return report_text[:6000]


@tool("BraveSearch")
def search_tool(search_query: str):
    """
    Search for scientific evidence with hierarchical strategy:

    PRIMARY SOURCES (Search First):
    1. Peer-reviewed journals: Nature, Science, Lancet, Cell, PNAS
    2. Recent data published after 2024
    3. RCTs and meta-analyses

    SECONDARY SOURCES (If primary insufficient):
    4. Observatory studies and cohort studies
    5. Cross-sectional population studies
    6. Epidemiological data

    SUPPLEMENTARY EVIDENCE (To verify logic):
    7. Non-human RCTs (animal studies, in vitro)
    8. Mechanistic studies
    9. Preclinical research

    SEARCH STRATEGY:
    - Start with "[topic] RCT" or "[topic] meta-analysis"
    - If no strong evidence, expand to "[topic] observatory study"
    - Supplement with "[topic] animal study" or "[topic] mechanism"
    - Always prioritize peer-reviewed > preprint > news

    CRITICAL: Always search to obtain verifiable URLs for all citations.
    This enables source validation and provides readers with direct access to evidence.
    """
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return "Brave API Key missing. Use internal knowledge."

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
    params = {"q": search_query, "count": 5}

    try:
        response = httpx.get(url, headers=headers, params=params, timeout=15.0)
        if response.status_code == 200:
            results = response.json().get("web", {}).get("results", [])
            return "\n\n".join([f"Title: {r['title']}\nURL: {r['url']}\nDesc: {r['description']}" for r in results]) or "No results found."
        return "Search API error. Use internal knowledge."
    except Exception as e:
        return f"Search failed: {e}"

@tool("DeepSearch")
def deep_search_tool(search_query: str) -> str:
    """
    Deep research using self-hosted SearXNG with full content extraction.

    Search for scientific evidence with hierarchical strategy:

    PRIMARY SOURCES (Search First):
    1. Peer-reviewed journals: Nature, Science, Lancet, Cell, PNAS
    2. Recent data published after 2024
    3. RCTs and meta-analyses

    SECONDARY SOURCES (If primary insufficient):
    4. Observatory studies and cohort studies
    5. Cross-sectional population studies
    6. Epidemiological data

    SUPPLEMENTARY EVIDENCE (To verify logic):
    7. Non-human RCTs (animal studies, in vitro)
    8. Mechanistic studies
    9. Preclinical research

    SEARCH STRATEGY:
    - Start with "[topic] RCT" or "[topic] meta-analysis"
    - If no strong evidence, expand to "[topic] observatory study"
    - Supplement with "[topic] animal study" or "[topic] mechanism"
    - Always prioritize peer-reviewed > preprint > news

    ADVANTAGE: Provides FULL PAGE CONTENT (not just snippets) from top 5 results.
    Uses local SearXNG (no API key required).

    CRITICAL: Always search to obtain verifiable URLs and full article content for all citations.
    This enables thorough source validation and provides detailed evidence for research claims.
    """

    async def perform_deep_search():
        """Async wrapper for deep search."""
        try:
            async with SearxngClient() as client:
                # Validate connection
                if not await client.validate_connection():
                    return (
                        "❌ SearXNG not accessible at http://localhost:8080\n"
                        "Start with: docker run -d -p 8080:8080 searxng/searxng:latest\n"
                        "Falling back to internal knowledge or use BraveSearch."
                    )

                async with DeepResearch(client) as research:
                    # Perform deep research
                    results = await research.deep_dive(
                        query=search_query,
                        top_n=5,
                        engines=['google', 'bing', 'brave']
                    )

                    if not results.scraped_pages:
                        return "No results found. Use internal knowledge."

                    # Format results for scientific research
                    output = f"=== Deep Research Results for: {search_query} ===\n\n"

                    for i, content in enumerate(results.scraped_pages, 1):
                        if not content.error:
                            output += f"--- SOURCE {i}: {content.title} ---\n"
                            output += f"URL: {content.url}\n"
                            output += f"Content Length: {content.word_count} words\n\n"
                            output += f"{content.content}\n\n"
                            output += "=" * 80 + "\n\n"
                        else:
                            # Include failed URLs but mark them
                            output += f"--- SOURCE {i}: [FAILED] {content.url} ---\n"
                            output += f"Error: {content.error}\n\n"

                    if results.errors:
                        output += f"\n⚠️ Some sources failed to load ({len(results.errors)} errors)\n"

                    return output

        except Exception as e:
            return (
                f"Deep search failed: {e}\n"
                f"Try BraveSearch as fallback or use internal knowledge."
            )

    # Run async function in sync context (CrewAI uses sync tools)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If event loop is already running, create a new one
            import nest_asyncio
            nest_asyncio.apply()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(perform_deep_search())


@tool("RequestSearch")
def request_search(search_requests_json: str) -> str:
    """Request targeted searches to fill evidence gaps. Only use if you find a CRITICAL gap with ZERO coverage in the Research Library.

    Args:
        search_requests_json: JSON array of search requests. Each must have "query" and "goal".
            Example: [{"query": "creatine RCT cognitive function", "goal": "Find RCTs on creatine and cognition"}]
    """
    try:
        # Handle both JSON string and raw list (LLM sometimes passes list directly)
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
            f"Queued {len(valid)} search request(s). Results will be available via "
            f"ListResearchSources/ReadResearchSource after the search round completes. "
            f"Continue your analysis with existing sources."
        )
    except (json.JSONDecodeError, TypeError):
        return 'ERROR: Invalid JSON. Provide a JSON array like: [{"query": "...", "goal": "..."}]'


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
            meta = {f.name: getattr(s.metadata, f.name) for f in __import__('dataclasses').fields(s.metadata)}
        new_sources.append({
            "url": s.url, "title": s.title, "query": s.query,
            "goal": s.goal, "summary": s.summary, "metadata": meta,
        })
    return new_sources


def append_sources_to_library(new_sources: list[dict], role: str, output_dir_path=None):
    """Append new sources to deep_research_sources.json."""
    src_dir = output_dir_path or output_dir
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


# --- RESEARCH LIBRARY TOOLS ---
# These tools let agents browse and read individual sources from the
# deep research pre-scan that was saved to deep_research_sources.json.

@tool("ListResearchSources")
def list_research_sources(role: str) -> str:
    """List all available research sources from the deep research pre-scan.

    Args:
        role: Either "lead" (supporting evidence) or "counter" (opposing evidence)

    Returns a numbered index with title, URL, and research goal for each source.
    Use ReadResearchSource to read the full summary of any specific source.
    """
    sources_file = output_dir / "deep_research_sources.json"
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

    Returns the full extracted summary for that source, including URL, title,
    research goal, and all extracted facts.
    """
    sources_file = output_dir / "deep_research_sources.json"
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
        if meta.get("study_type"):
            meta_lines.append(f"Study Type: {meta['study_type']}")
        if meta.get("sample_size"):
            meta_lines.append(f"Sample Size: {meta['sample_size']}")
        if meta.get("key_result"):
            meta_lines.append(f"Key Result: {meta['key_result']}")
        if meta.get("journal_name"):
            meta_lines.append(f"Journal: {meta['journal_name']}")
        if meta.get("publication_year"):
            meta_lines.append(f"Year: {meta['publication_year']}")
        if meta.get("effect_size"):
            meta_lines.append(f"Effect Size: {meta['effect_size']}")
        if meta.get("authors"):
            meta_lines.append(f"Authors: {meta['authors']}")
        if meta.get("demographics"):
            meta_lines.append(f"Demographics: {meta['demographics']}")
        if meta.get("limitations"):
            meta_lines.append(f"Limitations: {meta['limitations']}")
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

    WARNING: Reports can be very long. Prefer ListResearchSources + ReadResearchSource
    to selectively read specific sources instead.
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

    report_path = output_dir / name_map[key]
    if not report_path.exists():
        return f"Report file not found: {report_path}"

    content = report_path.read_text()
    if len(content) > 15000:
        return (
            content[:15000]
            + f"\n\n... [TRUNCATED — full report is {len(content)} chars. "
            f"Use ListResearchSources + ReadResearchSource for targeted reading.] ..."
        )
    return content


# --- PDF GENERATOR UTILITY ---
class SciencePDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'DGX Spark Research Intelligence Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf(title, content, filename):
    """Create PDF with language-appropriate encoding"""
    pdf = SciencePDF()
    pdf.add_page()

    # Clean up markdown for PDF
    clean_content = re.sub(r'<think>.*?</think>', '', str(content), flags=re.DOTALL)

    # Handle encoding based on language
    if language == 'ja':
        # For Japanese, keep UTF-8 characters but warn about PDF limitations
        # FPDF has limited Unicode support - ideally would use fpdf2 or ReportLab
        clean_title = title.encode('latin-1', 'ignore').decode('latin-1')
        clean_content = clean_content.encode('latin-1', 'ignore').decode('latin-1')
        print("Warning: Japanese characters may not display correctly in PDF. Consider upgrading to fpdf2 for full Unicode support.")
    else:
        # English - use latin-1 encoding
        clean_title = title.encode('latin-1', 'ignore').decode('latin-1')
        clean_content = clean_content.encode('latin-1', 'ignore').decode('latin-1')

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, clean_title, 0, 1, 'L')
    pdf.ln(5)

    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 10, clean_content)

    file_path = output_dir / filename
    pdf.output(str(file_path))
    print(f"PDF Generated: {file_path}")
    return file_path

# --- AGENTS (Masters-Degree Level) ---
# Initialize Link Validator Tool
link_validator = LinkValidatorTool()

@tool("ReadValidationResults")
def read_validation_results(url: str) -> str:
    """Look up the pre-validated status of a URL from batch validation results.

    Args:
        url: The URL to check.
    """
    validation_file = output_dir / "url_validation_results.json"
    if not validation_file.exists():
        return "No pre-validation data available. Use Link Validator to check this URL."
    try:
        data = json.loads(validation_file.read_text())
        return data.get(url, f"Not pre-validated. Use Link Validator to check: {url}")
    except Exception as e:
        return f"Error reading validation data: {e}"

researcher = Agent(
    role='Principal Investigator (Lead Researcher)',
    goal=f'Find and document credible scientific signals about {topic_name}, organized by mechanism of action. {english_instruction}',
    backstory=(
        f'You are a desperate scientist looking for signals in the noise. '
        f'CONSTRAINT: If Human RCTs are unavailable, you are AUTHORIZED to use Animal Models or Mechanistic Studies, '
        f'but you MUST label them as "Early Signal" or "Animal Model". '
        f'\n\n'
        f'OUTPUT REQUIREMENT: Do not just summarize. Group findings by:\n'
        f'  1. "Mechanism of Action" (HOW it works biologically)\n'
        f'  2. "Clinical Evidence" (WHAT human studies show)\n'
        f'\n'
        f'Evidence hierarchy: (1) Human RCTs/meta-analyses from Nature/Science/Lancet, '
        f'(2) Observatory/cohort studies (label as "Observational"), '
        f'(3) Animal/in vitro studies (label as "Animal Model" or "Early Signal"). '
        f'\n'
        f'In this podcast, you will be portrayed by "{SESSION_ROLES["pro"]["character"]}" '
        f'who has a {SESSION_ROLES["pro"]["personality"]} approach. '
        f'\n\n'
        f'You have access to a Research Library containing all sources from the deep research pre-scan. '
        f'Use ListResearchSources to browse, ReadResearchSource to read specific ones. '
        f'If you find a CRITICAL gap with ZERO coverage, use RequestSearch to queue targeted searches. '
        f'Do NOT attempt BraveSearch or DeepSearch — they are not available.'
        f'{english_instruction}'
    ),
    tools=[list_research_sources, read_research_source, read_full_report, request_search],
    llm=dgx_llm_strict,
    verbose=True,
    max_iter=10,
)

auditor = Agent(
    role='Scientific Auditor (The Grader)',
    goal=f'Grade the research quality with a Reliability Scorecard. Do NOT write content - GRADE it. {english_instruction}',
    backstory=(
        f'You are a harsh peer reviewer. You do not write content; you GRADE it.\n\n'
        f'YOUR TASKS:\n'
        f'  1. Link Check: If a claim has no URL or a broken URL, REJECT it.\n'
        f'  2. Strength Rating: Assign a score (1-10) to the main claims:\n'
        f'       10 = Meta-analysis from top journal\n'
        f'       7-9 = Human RCT with good sample size\n'
        f'       4-6 = Observational/cohort study\n'
        f'       1-3 = Animal model or speculation\n'
        f'  3. The Caveat Box: Explicitly list why the findings might be wrong:\n'
        f'       (e.g., "Mouse study only", "Sample size n=12", "Conflicts of interest")\n'
        f'  4. Consensus Check: Verify consensus from pre-scanned sources in the Research Library.\n'
        f'  5. Source Validation: Use ReadResearchSource to read source content. Verify claims match sources. REJECT misrepresented sources.\n'
        f'\n'
        f'You have access to a Research Library containing all sources from the deep research pre-scan. '
        f'Use ListResearchSources to browse available sources, then ReadResearchSource to read specific ones in detail.\n\n'
        f'OUTPUT: A structured Markdown report with a "Reliability Scorecard". '
        f'{english_instruction}'
    ),
    tools=[list_research_sources, read_research_source, read_full_report, link_validator],
    llm=dgx_llm_strict,
    verbose=True,
    max_iter=15,
)

counter_researcher = Agent(
    role='Adversarial Researcher (The Skeptic)',
    goal=f'Systematically challenge and debunk specific claims about {topic_name}. {english_instruction}',
    backstory=(
        f'Skeptical meta-analyst who hunts for contradictory evidence and methodology flaws. '
        f'You actively search for "criticism of {topic_name}" and "limitations of [specific studies]".\n\n'
        f'COUNTER-EVIDENCE HIERARCHY:\n'
        f'  1. PRIMARY: Contradictory RCTs, systematic reviews showing null/negative effects\n'
        f'  2. SECONDARY: Observatory/cohort studies with null findings or adverse outcomes\n'
        f'  3. SUPPLEMENTARY: Animal studies contradicting proposed mechanisms\n'
        f'\n'
        f'Label all evidence appropriately (RCT, Observational, Animal Model). '
        f'Focus on WHY the original claims might be wrong (confounders, bias, small samples). '
        f'\n'
        f'In this podcast, you will be portrayed by "{SESSION_ROLES["con"]["character"]}" '
        f'who has a {SESSION_ROLES["con"]["personality"]} approach. '
        f'\n\n'
        f'You have access to a Research Library containing all sources from the deep research pre-scan. '
        f'Use ListResearchSources to browse, ReadResearchSource to read specific ones. '
        f'If you find a CRITICAL gap with ZERO coverage, use RequestSearch to queue targeted searches. '
        f'Do NOT attempt BraveSearch or DeepSearch — they are not available.'
        f'{english_instruction}'
    ),
    tools=[list_research_sources, read_research_source, read_full_report, request_search],
    llm=dgx_llm_strict,
    verbose=True,
    max_iter=10,
)

scriptwriter = Agent(
    role='Podcast Producer (The Showrunner)',
    goal=(
        f'Transform research into a Masters/PhD-level debate on "{topic_name}". '
        f'Target: Intellectual, curious, slightly skeptical professionals. {english_instruction}'
    ),
    backstory=(
        f'Science Communicator targeting Post-Graduate Professionals (Masters/PhD level). '
        f'Tone: Think "The Economist" or "Huberman Lab" - intellectual, curious, slightly skeptical.\n\n'
        f'CRITICAL RULES:\n'
        f'  1. NO BASICS: Do NOT define basic terms like "DNA", "inflation", "supply chain", '
        f'     "peer review", "RCT", or "meta-analysis". Assume the listener knows them.\n'
        f'  2. LENGTH: Generate exactly 1,500 words (approx 10 minutes).\n'
        f'  3. FORMAT: Script MUST use "Host 1:" (The Expert) and "Host 2:" (The Skeptic).\n'
        f'  4. DYNAMIC: Host 2 must ask hard questions based on the "Caveat Box" from the Auditor. '
        f'     Host 2 represents the listener\'s doubts.\n'
        f'\n'
        f'Your dialogue should dive into nuance, trade-offs, and disputed evidence. '
        f'The audience wants intellectual depth, not simplified explanations. '
        f'{english_instruction}'
    ),
    llm=dgx_llm_creative,
    verbose=True
)

personality = Agent(
    role='Podcast Personality (The Editor)',
    goal=(
        f'Polish the "{topic_name}" script for natural verbal delivery at Masters-level. '
        f'Target: Exactly 1,500 words (10 minutes). '
        f'{target_instruction}'
    ),
    backstory=(
        f'Editor for high-end intellectual podcasts (Huberman Lab, The Economist Audio). '
        f'Your audience has advanced degrees - they want depth, not hand-holding.\n\n'
        f'EDITING RULES:\n'
        f'  - Remove any definitions of basic scientific concepts\n'
        f'  - Ensure Host 2 challenges Host 1 on weak evidence (from Caveat Box)\n'
        f'  - Keep technical language intact (no dumbing down)\n'
        f'  - Target exactly 1,500 words for 10-minute runtime\n'
        f'\n'
        f'If script is too short, add nuance and disputed evidence. '
        f'If too long, cut repetition while preserving technical depth. '
        f'{target_instruction}'
    ),
    llm=dgx_llm_creative,
    verbose=True
)

source_verifier = Agent(
    role='Scientific Source Verifier',
    goal='Extract, validate, and categorize all scientific sources from research papers.',
    backstory=(
        'Librarian and bibliometrics expert specializing in source verification. '
        'Uses LinkValidatorTool to check every URL. '
        'Ensures citations come from reputable peer-reviewed journals. '
        'Prioritizes high-impact publications (Nature, Science, Lancet, Cell, PNAS).'
    ),
    tools=[link_validator, read_validation_results],
    llm=dgx_llm_strict,
    verbose=True
)

research_framer = Agent(
    role='Research Framing Specialist',
    goal=f'Define the research scope, core questions, and evidence criteria for investigating {topic_name}. {english_instruction}',
    backstory=(
        'You are a senior research methodologist who designs investigation frameworks. '
        'Before any evidence is gathered, you establish:\n'
        '  1. Core research questions that must be answered\n'
        '  2. Scope boundaries (what is in/out of scope)\n'
        '  3. Evidence criteria (what counts as strong/weak evidence)\n'
        '  4. Suggested search directions and keywords\n'
        '  5. Hypotheses to test\n\n'
        'Your framing document guides all downstream research, ensuring systematic '
        'coverage rather than ad-hoc searching. You do NOT search for evidence yourself — '
        'you define WHAT to look for and HOW to evaluate it.'
    ),
    llm=dgx_llm_strict,
    verbose=True
)

# --- TASKS ---
framing_task = Task(
    description=(
        f"Define the research framework for investigating: {topic_name}\n\n"
        f"Produce a structured framing document with:\n\n"
        f"## 1. Core Research Questions\n"
        f"List 5-8 specific questions that this research MUST answer. "
        f"These should cover mechanisms, clinical evidence, population effects, and limitations.\n\n"
        f"## 2. Scope Boundaries\n"
        f"Define what is IN SCOPE and OUT OF SCOPE. "
        f"E.g., 'In scope: human health effects. Out of scope: economic impact, agricultural methods.'\n\n"
        f"## 3. Evidence Criteria\n"
        f"Define what constitutes strong vs weak evidence for this topic:\n"
        f"  - What study types are most relevant? (RCT, cohort, meta-analysis, etc.)\n"
        f"  - What sample sizes would be convincing?\n"
        f"  - What confounders should researchers watch for?\n\n"
        f"## 4. Suggested Search Directions\n"
        f"Provide 8-12 specific search queries or keyword combinations that would "
        f"systematically cover the topic. Group by: supporting evidence, opposing evidence, "
        f"mechanistic evidence, and population-level data.\n\n"
        f"## 5. Hypotheses\n"
        f"State 3-5 testable hypotheses that the research should evaluate.\n\n"
        f"Do NOT search for evidence. Only define the framework. "
        f"{english_instruction}"
    ),
    expected_output=(
        f"Structured research framing document with core questions, scope boundaries, "
        f"evidence criteria, search directions, and hypotheses. {english_instruction}"
    ),
    agent=research_framer,
    output_file=str(output_dir / "RESEARCH_FRAMING.md")
)

research_task = Task(
    description=(
        f"Conduct exhaustive deep dive into {topic_name}, guided by the Research Framing document. "
        f"Draft condensed scientific paper (Nature style). "
        f"\n\nIMPORTANT: The Research Framing document defines the core questions, scope boundaries, "
        f"and evidence criteria for this investigation. Use it to guide your searches systematically — "
        f"ensure each core research question is addressed.\n\n"
        f"CRITICAL: Focus ONLY on the health topic itself. Include:\n"
        f"- Specific health effects and mechanisms\n"
        f"- Biochemical pathways and physiological impacts\n"
        f"- Clinical outcomes and disease relationships\n"
        f"- Concrete examples of health consequences\n\n"
        f"EVIDENCE HIERARCHY:\n"
        f"1. PRIMARY: RCTs and meta-analyses from Nature/Science/Lancet/Cell/PNAS\n"
        f"2. SECONDARY: Observatory studies, cohort studies, epidemiological data (when RCTs unavailable)\n"
        f"3. SUPPLEMENTARY: Non-human RCTs (animal studies, in vitro) to verify proposed mechanisms\n\n"
        f"RESEARCH LIBRARY: You have access to a pre-scanned Research Library with dozens of sources. "
        f"Use ListResearchSources('lead') and ReadResearchSource('lead:N') as your PRIMARY evidence source. "
        f"If a critical gap exists, use RequestSearch to queue targeted searches.\n\n"
        f"SEARCH STRATEGY: Start with the Research Library sources. If no strong evidence for a specific mechanism, "
        f"expand search using RequestSearch. Supplement with animal/mechanistic studies to validate logic.\n\n"
        f"Every citation in your bibliography MUST include a URL for source validation.\n"
        f"If time runs short, CONCLUDE with available evidence — present findings as hypotheses based on current knowledge rather than failing.\n"
        f"Include: Abstract, Introduction, 3 Biochemical Mechanisms with CONCRETE health impacts, Bibliography with URLs and study types noted. "
        f"{english_instruction}"
    ),
    expected_output=f"Scientific paper with SPECIFIC health mechanisms and effects, citations with URLs from RCTs, observatory studies, and non-human studies. Bibliography must include verifiable URLs for all sources. {english_instruction}",
    agent=researcher,
    context=[framing_task]
)

gap_analysis_task = Task(
    description=(
        f"RESEARCH GATE: Evaluate whether the initial research on {topic_name} adequately "
        f"addresses the core research questions defined in the framing document.\n\n"
        f"Compare the research output against the framing document's:\n"
        f"- Core research questions: Are they all addressed?\n"
        f"- Evidence criteria: Does the evidence meet the defined standards?\n"
        f"- Scope boundaries: Did the research stay in scope?\n\n"
        f"For EACH core research question, assess:\n"
        f"  - ADDRESSED: Question answered with adequate evidence\n"
        f"  - PARTIALLY ADDRESSED: Some evidence but significant gaps remain\n"
        f"  - NOT ADDRESSED: No meaningful evidence found\n\n"
        f"OUTPUT FORMAT:\n"
        f"## Research Gate Assessment\n\n"
        f"### Question Coverage\n"
        f"[For each core question: ADDRESSED / PARTIALLY / NOT ADDRESSED + brief justification]\n\n"
        f"### Identified Gaps\n"
        f"[Numbered list of specific gaps that need filling]\n\n"
        f"### Weak Points for Adversarial Review\n"
        f"[3-5 specific scientific weak points for the Counter-Researcher]\n\n"
        f"### VERDICT: [PASS or FAIL]\n"
        f"PASS = All core questions at least PARTIALLY addressed with credible evidence.\n"
        f"FAIL = One or more core questions NOT ADDRESSED, or evidence quality critically low.\n\n"
        f"{english_instruction}"
    ),
    expected_output=(
        f"Research gate assessment with question coverage, identified gaps, weak points, "
        f"and a clear VERDICT: PASS or VERDICT: FAIL. {english_instruction}"
    ),
    agent=auditor,
    context=[framing_task, research_task]
)

gap_fill_task = Task(
    description=(
        f"Conduct TARGETED supplementary research to fill specific gaps identified by the Research Gate.\n\n"
        f"The Gap Analysis has identified specific weaknesses in the initial research on {topic_name}. "
        f"Your job is to conduct focused searches ONLY for the missing evidence.\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Read the gap analysis carefully — it lists specific missing evidence\n"
        f"2. For EACH identified gap, conduct 1-2 targeted searches\n"
        f"3. Report findings organized by which gap they address\n"
        f"4. Do NOT repeat research already covered — only fill gaps\n\n"
        f"Use RequestSearch to queue targeted searches for missing evidence. Results will appear in the Research Library. "
        f"{english_instruction}"
    ),
    expected_output=(
        f"Targeted supplementary research addressing each identified gap, "
        f"with verifiable sources and URLs. {english_instruction}"
    ),
    agent=researcher,
    tools=[request_search, list_research_sources, read_research_source],
    context=[research_task, gap_analysis_task]
)

adversarial_task = Task(
    description=(
        f"Based on 'Supporting Paper' and 'Gap Analysis', draft 'Anti-Thesis' paper on {topic_name}. "
        f"Address and debunk the SPECIFIC health mechanisms proposed in initial research. "
        f"\n\nCRITICAL: Stay focused on the health topic. Debunk the specific biological and clinical claims. "
        f"Do NOT discuss research methodology or journal quality — challenge the SCIENCE ITSELF.\n\n"
        f"COUNTER-EVIDENCE HIERARCHY:\n"
        f"1. PRIMARY: Contradictory RCTs, systematic reviews showing null/negative effects\n"
        f"2. SECONDARY: Observatory/cohort studies with null findings or adverse outcomes\n"
        f"3. SUPPLEMENTARY: Animal studies contradicting proposed mechanisms\n\n"
        f"RESEARCH LIBRARY: You have access to a pre-scanned Research Library with dozens of sources. "
        f"Use ListResearchSources('counter') and ReadResearchSource('counter:N') as your PRIMARY evidence source. "
        f"If a critical gap exists, use RequestSearch to queue targeted searches.\n\n"
        f"SEARCH STRATEGY: Find contradictory RCTs first from the Research Library. If limited, use observatory studies with null findings.\n\n"
        f"Every citation in your bibliography MUST include a URL for source validation.\n"
        f"Include Bibliography with URLs and study types noted. "
        f"{english_instruction}"
    ),
    expected_output=f"Scientific paper challenging SPECIFIC health claims with contradictory evidence from RCTs, observatory studies, and animal studies. Bibliography must include verifiable URLs for all sources. {english_instruction}",
    agent=counter_researcher,
    context=[research_task, gap_analysis_task]
)

source_verification_task = Task(
    description=(
        f"Extract ALL sources from Supporting and Anti-Thesis papers. "
        f"For each source verify:\n"
        f"1. URL points to scientific content\n"
        f"2. Source type (peer-reviewed, preprint, review, meta-analysis)\n"
        f"3. Trust level: HIGH (Nature/Science/Lancet/Cell/PNAS), "
        f"MEDIUM (PubMed/arXiv), LOW (news/blogs)\n"
        f"4. Journal name and year if available\n\n"
        f"CLAIM-TO-SOURCE VERIFICATION (NEW):\n"
        f"For each major claim in the research papers, verify that:\n"
        f"  - The cited source ACTUALLY supports the claim as stated\n"
        f"  - The claim does not overstate what the source says\n"
        f"  - Hedging language (may, suggests, correlates) is preserved accurately\n"
        f"  - Flag any misrepresented sources as 'MISREPRESENTED: [explanation]'\n\n"
        f"Create structured bibliography JSON:\n"
        f'{{"supporting_sources": [{{title, url, journal, year, trust_level, source_type, claim_match: "VERIFIED/MISREPRESENTED"}}],\n'
        f' "contradicting_sources": [...],\n'
        f' "misrepresented_claims": ["claim X cites source Y but source actually says Z"],\n'
        f' "summary": "X high-trust, Y medium-trust sources, Z misrepresented"}}\n\n'
        f"REJECT non-scientific sources. Flag if <3 high-trust sources. "
        f"{english_instruction}"
    ),
    expected_output=(
        f"JSON bibliography with categorized, verified sources, claim-to-source match verification, "
        f"and quality summary. {english_instruction}"
    ),
    agent=source_verifier,
    context=[research_task, adversarial_task]
)

audit_task = Task(
    description=(
        f"Synthesize ALL research on {topic_name} into a single Source-of-Truth document.\n\n"
        f"This is NOT a grade — it is a SYNTHESIS. Combine the supporting evidence, opposing evidence, "
        f"and source verification results into one authoritative reference document that the script "
        f"writers will use as their ONLY source.\n\n"
        f"OUTPUT FORMAT (Markdown):\n"
        f"# Source of Truth: {topic_name}\n\n"
        f"## Executive Summary\n"
        f"[2-3 paragraph balanced summary of what the evidence shows]\n\n"
        f"## Key Claims with Confidence Levels\n"
        f"For EACH major claim, assign a confidence level:\n"
        f"  - **HIGH**: Multiple RCTs/meta-analyses agree, no significant contradictions\n"
        f"  - **MEDIUM**: Some RCT evidence but limited replication, or strong observational data\n"
        f"  - **LOW**: Only animal/mechanistic studies, or single small study\n"
        f"  - **CONTESTED**: Significant evidence on BOTH sides\n\n"
        f"Format each claim as:\n"
        f"### Claim: [statement]\n"
        f"- **Confidence**: HIGH/MEDIUM/LOW/CONTESTED\n"
        f"- **Supporting evidence**: [brief summary with citations]\n"
        f"- **Opposing evidence**: [brief summary with citations, or 'None found']\n"
        f"- **Key caveats**: [limitations]\n\n"
        f"## Settled Science vs Active Debate\n"
        f"### What the Evidence Broadly Agrees On:\n"
        f"[Claims with HIGH confidence where both sides agree]\n\n"
        f"### Where the Science is Still Debated:\n"
        f"[CONTESTED and LOW confidence claims with both sides presented]\n\n"
        f"## Reliability Scorecard\n"
        f"| Claim | Confidence | Evidence Type | Best Source |\n"
        f"| --- | --- | --- | --- |\n\n"
        f"## The Caveat Box\n"
        f"### Why These Findings Might Be Wrong:\n"
        f"- [List of limitations and concerns]\n\n"
        f"## Evidence Table\n"
        f"| Claim | Source | Study Type | Sample Size | Effect Size | Journal | Year | Demographics |\n"
        f"| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        f"[Fill from structured study metadata extracted during deep research]\n\n"
        f"## Complete Bibliography\n"
        f"[All verified sources with URLs, organized by claim]\n\n"
        f"The output MUST contain concrete health information, NOT a discussion about source quality. "
        f"This document is the SINGLE SOURCE OF TRUTH for all downstream script generation. "
        f"{english_instruction}"
    ),
    expected_output=(
        f"Structured Source-of-Truth document (SOURCE_OF_TRUTH.md) with:\n"
        f"- Executive Summary\n"
        f"- Key Claims with Confidence Levels (HIGH/MEDIUM/LOW/CONTESTED)\n"
        f"- Settled Science vs Active Debate sections\n"
        f"- Reliability Scorecard\n"
        f"- Caveat Box\n"
        f"- Evidence Table (study type, sample size, effect size, journal, year)\n"
        f"- Complete Bibliography\n"
        f"{english_instruction}"
    ),
    agent=auditor,
    context=[research_task, adversarial_task, source_verification_task],
    output_file=str(output_dir / "SOURCE_OF_TRUTH.md")
)

script_task = Task(
    description=(
        f"Using the audit report, write a podcast dialogue about \"{topic_name}\" "
        f"featuring {SESSION_ROLES['pro']['character']} vs {SESSION_ROLES['con']['character']}.\n\n"
        f"TOPIC FOCUS — every exchange must be about the health topic. Suggested structure:\n"
        f"  1. Open: What high GI food means in plain terms and why it matters\n"
        f"  2. Body: The main health risks (blood sugar spikes, insulin resistance, cardiovascular, etc.)\n"
        f"  3. Disagreement: Which risks are proven vs still debated\n"
        f"  4. Close: Practical takeaway for listeners\n\n"
        f"DO NOT discuss: peer review, journal quality, source trustworthiness, research methodology.\n"
        f"If the audit report contains that commentary, skip it. Extract only the health conclusions.\n\n"
        f"SIMPLIFY THE SCIENCE (not the research process):\n"
        f"- 'Glycemic index' → 'a score that measures how fast a food raises blood sugar'\n"
        f"- 'Insulin resistance' → 'when your body stops responding properly to insulin'\n"
        f"- 'Postprandial glucose spike' → 'a sharp rise in blood sugar after eating'\n\n"
        f"CHARACTER ROLES:\n"
        f"  - {SESSION_ROLES['pro']['character']}: argues the health risks ARE significant, "
        f"{SESSION_ROLES['pro']['personality']}\n"
        f"  - {SESSION_ROLES['con']['character']}: argues some risks are overstated or context-dependent, "
        f"{SESSION_ROLES['con']['personality']}\n\n"
        f"Format STRICTLY as:\n"
        f"{SESSION_ROLES['pro']['character']}: [dialogue]\n"
        f"{SESSION_ROLES['con']['character']}: [dialogue]\n\n"
        f"Maintain consistent roles throughout. NO role switching mid-conversation. "
        f"{english_instruction}"
    ),
    expected_output=(
        f"Dialogue about the health risks of {topic_name} between {SESSION_ROLES['pro']['character']} (risks are real) "
        f"and {SESSION_ROLES['con']['character']} (some risks are overstated). Every line discusses the topic. "
        f"{english_instruction}"
    ),
    agent=scriptwriter,
    context=[audit_task]
)

# --- TRANSLATION TASK (only when language != 'en') ---
translation_task = None
if language != 'en':
    translation_task = Task(
        description=(
            f"Translate the podcast script and key Source-of-Truth findings about "
            f"{topic_name} into {language_config['name']}.\n\n"
            f"RULES:\n"
            f"- Preserve Host 1: / Host 2: format exactly\n"
            f"- Preserve scientific terminology accuracy\n"
            f"- Translate for natural spoken delivery, not literal translation\n"
            f"- Keep proper nouns, study names, journal names in English\n"
            f"- Maintain debate structure and argumentation flow\n"
            f"{target_instruction}"
        ),
        expected_output=f"Complete translated script in {language_config['name']} with Host 1:/Host 2: format.",
        agent=scriptwriter,
        context=[script_task, audit_task],
    )

natural_language_task = Task(
    description=(
        f"Polish the \"{topic_name}\" dialogue for natural spoken delivery at Masters-level.\n\n"
        f"MASTERS-LEVEL REQUIREMENTS:\n"
        f"- Remove ALL definitions of basic scientific concepts (DNA, peer review, RCT, meta-analysis)\n"
        f"- Ensure Host 2 challenges Host 1 on weak evidence (refer to Caveat Box from audit)\n"
        f"- Keep technical language intact - NO dumbing down\n"
        f"- Target exactly 1,500 words (10 minutes at 150 wpm)\n\n"
        f"MAINTAIN ROLES:\n"
        f"  - Host 1 ({SESSION_ROLES['pro']['character']}): The Expert - presents evidence\n"
        f"  - Host 2 ({SESSION_ROLES['con']['character']}): The Skeptic - challenges weak claims\n\n"
        f"Format:\nHost 1: [dialogue]\n"
        f"Host 2: [dialogue]\n\n"
        f"Remove meta-tags, markdown, stage directions. Dialogue only. "
        f"{target_instruction}"
    ),
    expected_output=(
        f"Final Masters-level dialogue about {topic_name}, exactly 1,500 words. "
        f"No basic definitions. Host 2 challenges weak evidence. "
        f"{target_instruction}"
    ),
    agent=personality,
    context=[script_task, audit_task]
)

accuracy_check_task = Task(
    description=(
        f"Compare the POLISHED SCRIPT against the Source-of-Truth document for {topic_name}.\n\n"
        f"This is a POST-POLISH accuracy check. The script has been edited for natural delivery, "
        f"and scientific drift may have been introduced during polishing.\n\n"
        f"CHECK FOR THESE SPECIFIC DRIFT PATTERNS:\n"
        f"1. **Correlation → Causation drift**: Script says 'X causes Y' when source says 'X is associated with Y'\n"
        f"2. **Hedge removal**: Source says 'may' or 'suggests', script says 'does' or 'proves'\n"
        f"3. **Confidence inflation**: LOW confidence claims presented as settled fact\n"
        f"4. **Cherry-picking**: Only one side of CONTESTED claims presented\n"
        f"5. **Contested-as-settled**: Claims marked CONTESTED in source-of-truth presented as consensus\n\n"
        f"OUTPUT FORMAT:\n"
        f"# Accuracy Check: {topic_name}\n\n"
        f"## Overall Assessment\n"
        f"[PASS / PASS WITH WARNINGS / FAIL]\n\n"
        f"## Drift Instances Found\n"
        f"For each issue:\n"
        f"- **Script says**: [exact quote from script]\n"
        f"- **Source-of-truth says**: [what the evidence actually supports]\n"
        f"- **Drift type**: [one of the 5 patterns above]\n"
        f"- **Severity**: HIGH / MEDIUM / LOW\n\n"
        f"## Recommendations\n"
        f"[Specific line-level fixes if needed]\n\n"
        f"NOTE: This check is ADVISORY. It does NOT block audio generation. "
        f"{target_instruction}"
    ),
    expected_output=(
        f"Accuracy check report comparing polished script against source-of-truth, "
        f"listing any scientific drift with severity ratings. {target_instruction}"
    ),
    agent=auditor,
    context=[natural_language_task, audit_task],
    output_file=str(output_dir / "ACCURACY_CHECK.md")
)

show_notes_task = Task(
    description=(
        f"Generate comprehensive show notes (SHOW_NOTES.md) for the podcast episode on {topic_name}.\n\n"
        f"Using the Source-of-Truth document and its bibliography, create a bulleted list with:\n"
        f"1. Episode title and topic\n"
        f"2. Key takeaways (3-5 bullet points)\n"
        f"3. Full citation list with validity ratings:\n\n"
        f"FORMAT:\n"
        f"## Citations\n\n"
        f"### Supporting Evidence\n"
        f"- [Study Title] (Journal, Year) - [URL] - **Validity: ✓ High/Medium/Low**\n"
        f"  - Evidence Type: [RCT/Observational/Animal Model]\n"
        f"  - Key Finding: [One sentence summary]\n\n"
        f"### Contradicting Evidence\n"
        f"- [Study Title] (Journal, Year) - [URL] - **Validity: ✓ High/Medium/Low**\n"
        f"  - Evidence Type: [RCT/Observational/Animal Model]\n"
        f"  - Key Finding: [One sentence summary]\n\n"
        f"Include validity ratings from the Reliability Scorecard. "
        f"Mark broken links as '✗ Broken Link'. "
        f"{target_instruction}"
    ),
    expected_output=(
        f"Markdown show notes with:\n"
        f"- Episode title\n"
        f"- Key takeaways (3-5 bullets)\n"
        f"- Full citation list with validity ratings (✓ High/Medium/Low)\n"
        f"- Evidence type labels (RCT/Observational/Animal Model)\n"
        f"{target_instruction}"
    ),
    agent=scriptwriter,
    context=[audit_task],
    output_file=str(output_dir / "SHOW_NOTES.md")
)

# --- TRANSLATION PIPELINE: Update contexts when translating ---
if translation_task is not None:
    # Polish reads from translated script instead of English script
    natural_language_task.context = [translation_task, audit_task]
    # Show notes use translated SOT context
    show_notes_task.context = [translation_task, audit_task]
    # Accuracy check compares polished (target lang) against translated script
    accuracy_check_task.context = [natural_language_task, translation_task]

# --- TASK METADATA & WORKFLOW PLANNING ---
TASK_METADATA = {
    'framing_task': {
        'name': 'Research Framing & Hypothesis',
        'phase': '0',
        'estimated_duration_min': 2,
        'description': 'Research Framing Specialist defines scope, questions, and evidence criteria',
        'agent': 'Research Framing Specialist',
        'dependencies': [],
        'crew': 0
    },
    'research_task': {
        'name': 'Systematic Evidence Gathering',
        'phase': '1',
        'estimated_duration_min': 8,
        'description': 'Lead Researcher conducts deep dive guided by framing document',
        'agent': 'Principal Investigator',
        'dependencies': ['framing_task'],
        'crew': 1
    },
    'gap_analysis_task': {
        'name': 'Research Gate & Gap Analysis',
        'phase': '2',
        'estimated_duration_min': 3,
        'description': 'Scientific Auditor evaluates coverage with PASS/FAIL gate',
        'agent': 'Scientific Auditor',
        'dependencies': ['framing_task', 'research_task'],
        'crew': 1
    },
    'gap_fill_task': {
        'name': 'Gap-Fill Research (conditional)',
        'phase': '2b',
        'estimated_duration_min': 4,
        'description': 'Targeted supplementary research if gate FAILS',
        'agent': 'Principal Investigator',
        'dependencies': ['research_task', 'gap_analysis_task'],
        'crew': 'conditional',
        'conditional': True
    },
    'adversarial_task': {
        'name': 'Counter-Evidence Research',
        'phase': '3',
        'estimated_duration_min': 8,
        'description': 'Counter-Researcher challenges findings',
        'agent': 'Adversarial Researcher',
        'dependencies': ['research_task', 'gap_analysis_task'],
        'crew': 2
    },
    'source_verification_task': {
        'name': 'Source Validation & Claim Verification',
        'phase': '4',
        'estimated_duration_min': 5,
        'description': 'Source Verifier validates citations and claim-to-source accuracy',
        'agent': 'Scientific Source Verifier',
        'dependencies': ['research_task', 'adversarial_task'],
        'crew': 2
    },
    'audit_task': {
        'name': 'Source-of-Truth Synthesis',
        'phase': '5',
        'estimated_duration_min': 5,
        'description': 'Scientific Auditor synthesizes authoritative source-of-truth document',
        'agent': 'Scientific Auditor',
        'dependencies': ['research_task', 'adversarial_task', 'source_verification_task'],
        'crew': 2
    },
    'show_notes_task': {
        'name': 'Show Notes & Citations',
        'phase': '6a',
        'estimated_duration_min': 3,
        'description': 'Scriptwriter generates show notes from source-of-truth',
        'agent': 'Podcast Producer',
        'dependencies': ['audit_task'],
        'crew': 2
    },
    'script_task': {
        'name': 'Podcast Script Generation',
        'phase': '6b',
        'estimated_duration_min': 6,
        'description': 'Scriptwriter creates debate dialogue from source-of-truth',
        'agent': 'Podcast Producer',
        'dependencies': ['audit_task'],
        'crew': 2
    },
    'natural_language_task': {
        'name': 'Script Polishing & Editing',
        'phase': '7',
        'estimated_duration_min': 4,
        'description': 'Personality Agent refines for natural delivery',
        'agent': 'Podcast Personality',
        'dependencies': ['script_task', 'audit_task'],
        'crew': 2
    },
    'accuracy_check_task': {
        'name': 'Script Accuracy Check',
        'phase': '8',
        'estimated_duration_min': 3,
        'description': 'Scientific Auditor checks polished script for drift (advisory)',
        'agent': 'Scientific Auditor',
        'dependencies': ['natural_language_task', 'audit_task'],
        'crew': 2
    }
}

def display_workflow_plan():
    """
    Display detailed workflow plan before execution.
    Shows Phases 0-8 with durations, dependencies, and total time estimate.
    Phase 2b is marked as conditional.
    """
    print("\n" + "="*70)
    print(" "*20 + "PODCAST GENERATION WORKFLOW")
    print("="*70)
    print(f"\nTopic: {topic_name}")
    print(f"Language: {language_config['name']}")
    print(f"Output Directory: {output_dir}")
    print("\n" + "-"*70)
    print(f"{'PHASE':<6} {'TASK NAME':<40} {'EST TIME':<12} {'AGENT':<25}")
    print("-"*70)

    total_duration = 0
    for task_name, metadata in TASK_METADATA.items():
        phase = metadata['phase']
        name = metadata['name']
        duration = metadata['estimated_duration_min']
        agent = metadata['agent']
        is_conditional = metadata.get('conditional', False)

        if not is_conditional:
            total_duration += duration

        conditional_marker = " [CONDITIONAL]" if is_conditional else ""
        print(f"{phase:<6} {name:<40} {duration:>3} min{'':<6} {agent:<25}{conditional_marker}")
        print(f"{'':6} └─ {metadata['description']}")
        if metadata['dependencies']:
            deps_str = ', '.join([f"Phase {TASK_METADATA[d]['phase']}" for d in metadata['dependencies'] if d in TASK_METADATA])
            print(f"{'':6}    Dependencies: {deps_str}")
        print()

    print("-"*70)
    print(f"TOTAL ESTIMATED TIME: {total_duration} minutes (~{total_duration//60}h {total_duration%60}m)")
    print(f"  (+ up to 4 min if gap-fill triggers)")
    print("="*70 + "\n")

class ProgressTracker:
    """
    Real-time progress tracking for CrewAI task execution.
    Tracks current phase, elapsed time, and estimated remaining time.
    """
    def __init__(self, task_metadata: dict):
        self.task_metadata = task_metadata
        self.task_names = list(task_metadata.keys())
        self.current_task_index = 0
        # Exclude conditional tasks from total count initially
        self.total_phases = len([m for m in task_metadata.values() if not m.get('conditional', False)])
        self.start_time = None
        self.task_start_time = None
        self.completed_tasks = []
        self.gate_passed = True  # Updated after gate check

    def start_workflow(self):
        """Mark workflow start time"""
        self.start_time = time.time()
        print(f"\n{'='*70}")
        print("WORKFLOW EXECUTION STARTED")
        print(f"{'='*70}\n")

    def task_started(self, task_index: int):
        """Called when a task begins"""
        if task_index >= len(self.task_names):
            return

        task_name = self.task_names[task_index]
        self.current_task_index = task_index
        self.task_start_time = time.time()

        metadata = self.task_metadata[task_name]

        print(f"\n{'='*70}")
        print(f"PHASE {metadata['phase']}/{self.total_phases}: {metadata['name'].upper()}")
        print(f"{'='*70}")
        print(f"Agent: {metadata['agent']}")
        print(f"Description: {metadata['description']}")
        print(f"Estimated Duration: {metadata['estimated_duration_min']} minutes")
        if metadata['dependencies']:
            deps_str = ', '.join([self.task_metadata[d]['name'] for d in metadata['dependencies'] if d in self.task_metadata])
            print(f"Dependencies: {deps_str}")
        print("-"*70)

    def task_completed(self, task_index: int):
        """Called when a task completes"""
        if task_index >= len(self.task_names):
            return

        task_name = self.task_names[task_index]
        elapsed_task = time.time() - self.task_start_time
        self.completed_tasks.append({
            'name': task_name,
            'duration': elapsed_task
        })

        # Calculate progress
        progress_pct = (len(self.completed_tasks) / self.total_phases) * 100

        # Calculate time estimates
        elapsed_total = time.time() - self.start_time
        avg_time_per_task = elapsed_total / len(self.completed_tasks)
        remaining_tasks = self.total_phases - len(self.completed_tasks)
        estimated_remaining = avg_time_per_task * remaining_tasks

        metadata = self.task_metadata[task_name]

        print(f"\n{'='*70}")
        print(f"✓ PHASE {metadata['phase']}/{self.total_phases} COMPLETED")
        print(f"{'='*70}")
        print(f"Task Duration: {elapsed_task/60:.1f} minutes ({elapsed_task:.0f} seconds)")
        print(f"Total Elapsed: {elapsed_total/60:.1f} minutes")
        print(f"Progress: {progress_pct:.1f}% complete ({len(self.completed_tasks)}/{self.total_phases} tasks)")
        print(f"Estimated Remaining: {estimated_remaining/60:.1f} minutes")
        print(f"{'='*70}\n")

    def workflow_completed(self):
        """Called when entire workflow finishes"""
        total_time = time.time() - self.start_time

        print(f"\n{'='*70}")
        print(" "*22 + "WORKFLOW COMPLETED")
        print(f"{'='*70}")
        print(f"\nTotal Execution Time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        print(f"Tasks Completed: {len(self.completed_tasks)}/{self.total_phases}")

        print(f"\n{'Task Performance Summary':^70}")
        print("-"*70)
        for i, task_info in enumerate(self.completed_tasks, 1):
            task_name = task_info['name']
            duration = task_info['duration']
            estimated = self.task_metadata[task_name]['estimated_duration_min'] * 60
            variance = ((duration - estimated) / estimated) * 100 if estimated > 0 else 0

            print(f"{i}. {self.task_metadata[task_name]['name']:<40} "
                  f"{duration/60:>6.1f} min (est: {estimated/60:.1f} min, {variance:+.0f}%)")

        print(f"{'='*70}\n")

# --- EXECUTION (Multi-Crew Pipeline with Gate) ---
# Display workflow plan before execution
display_workflow_plan()

# Initialize progress tracker
progress_tracker = ProgressTracker(TASK_METADATA)
progress_tracker.start_workflow()

# Combined task list for tracking (will be updated if gap_fill runs)
all_task_list = [
    framing_task,
    research_task,
    gap_analysis_task,
    # gap_fill_task inserted here conditionally
    adversarial_task,
    source_verification_task,
    audit_task,
    show_notes_task,
    script_task,
    natural_language_task,
    accuracy_check_task,
]

print(f"\n--- Initiating Scientific Research Pipeline on DGX Spark ---")
print(f"Topic: {topic_name}")
print(f"Language: {language_config['name']} ({language})")
print("---\n")

# Start progress monitoring in background thread
import threading

class CrewMonitor(threading.Thread):
    """Background thread that monitors crew execution progress"""
    def __init__(self, task_list, progress_tracker):
        super().__init__(daemon=True)
        self.task_list = task_list
        self.progress_tracker = progress_tracker
        self.running = True
        self.last_completed = -1

    def run(self):
        """Monitor crew tasks in background"""
        while self.running:
            try:
                completed_count = 0
                for task in self.task_list:
                    if hasattr(task, 'output') and task.output is not None:
                        completed_count += 1
                    else:
                        break
                if completed_count > self.last_completed:
                    if self.last_completed >= 0:
                        self.progress_tracker.task_completed(self.last_completed)
                    if completed_count < len(self.task_list):
                        self.progress_tracker.task_started(completed_count)
                    self.last_completed = completed_count
                time.sleep(3)
            except Exception:
                pass

    def stop(self):
        """Stop monitoring"""
        self.running = False

# ================================================================
# PHASE 0: Research Framing & Hypothesis
# ================================================================
print(f"\n{'='*70}")
print(f"PHASE 0: RESEARCH FRAMING & HYPOTHESIS")
print(f"{'='*70}")

crew_0 = Crew(
    agents=[research_framer],
    tasks=[framing_task],
    verbose=True,
    process='sequential'
)

try:
    crew_0_result = crew_0.kickoff()
    framing_output = framing_task.output.raw if hasattr(framing_task, 'output') and framing_task.output else ""
    print(f"✓ Phase 0 complete: Research framing generated ({len(framing_output)} chars)")
except Exception as e:
    print(f"⚠ Phase 0 (Research Framing) failed: {e}")
    print("Continuing without framing context...")
    framing_output = ""

# ================================================================
# DEEP RESEARCH PRE-SCAN (Dual-Model Map-Reduce)
# ================================================================
brave_key = os.getenv("BRAVE_API_KEY", "")

# Check if fast model (Phi-4 Mini via Ollama) is available
fast_model_available = False
try:
    _resp = httpx.get("http://localhost:11434/v1/models", timeout=3)
    if _resp.status_code == 200:
        _models = [m.get("id", "") for m in _resp.json().get("data", [])]
        fast_model_available = any("phi" in m.lower() for m in _models)
        if fast_model_available:
            print("✓ Fast model (Phi-4 Mini) detected on Ollama")
        else:
            print(f"⚠ Ollama running but no phi model found. Available: {_models}")
except Exception:
    print("⚠ Fast model not available, using smart-only mode")

try:
    deep_reports = asyncio.run(run_deep_research(
        topic=topic_name,
        brave_api_key=brave_key,
        results_per_query=5,
        fast_model_available=fast_model_available,
        framing_context=framing_output
    ))

    # Save all reports (lead, counter, audit)
    for role_name, report in deep_reports.items():
        report_file = output_dir / f"deep_research_{role_name}.md"
        with open(report_file, 'w') as f:
            f.write(report.report)
        print(f"✓ {role_name.capitalize()} report saved: {report_file} ({report.total_summaries} sources)")

    # Save source-level data to JSON for the research library tools
    sources_json = {}
    for role_name in ("lead", "counter"):
        report = deep_reports[role_name]
        role_sources = []
        for idx, src in enumerate(report.sources):
            if src.error or not src.summary or src.summary.strip().upper() == "NO RELEVANT DATA":
                continue
            role_sources.append({
                "index": idx,
                "url": src.url,
                "title": src.title,
                "query": src.query,
                "goal": src.goal,
                "summary": src.summary,
                "metadata": src.metadata.to_dict() if src.metadata else None,
            })
        sources_json[role_name] = role_sources
    sources_file = output_dir / "deep_research_sources.json"
    with open(sources_file, 'w') as f:
        json.dump(sources_json, f, indent=2, ensure_ascii=False)
    print(f"✓ Research library saved: {sources_file} "
          f"(lead={len(sources_json['lead'])}, counter={len(sources_json['counter'])} sources)")

    # Use audit report (combined synthesis) for injection into CrewAI agents
    deep_audit_report = deep_reports["audit"]
    lead_report = deep_reports["lead"]
    counter_report = deep_reports["counter"]

    # Summarize reports with phi4-mini (preserves ALL findings, not just first 6000 chars)
    print("Summarizing deep research reports with phi4-mini...")
    lead_summary = summarize_report_with_fast_model(lead_report.report, "lead", topic_name)
    counter_summary = summarize_report_with_fast_model(counter_report.report, "counter", topic_name)

    # Inject summarized supporting evidence into lead research task
    lead_injection = (
        f"\n\nIMPORTANT: A deep research pre-scan has comprehensively analyzed "
        f"{lead_report.total_summaries} supporting sources in {lead_report.duration_seconds:.0f}s.\n\n"
        f"YOUR PRIMARY TASK: Synthesize and organize this pre-collected evidence. "
        f"Use ListResearchSources('lead') to browse all {lead_report.total_summaries} sources, "
        f"and ReadResearchSource('lead:N') to read any source in full.\n\n"
        f"SEARCH POLICY: Do NOT use RequestSearch unless you identify a CRITICAL "
        f"gap — a specific claim or mechanism that has ZERO coverage in the pre-scan. "
        f"The pre-scan already covers the major aspects of this topic.\n\n"
        f"PRE-COLLECTED SUPPORTING EVIDENCE (condensed):\n{lead_summary}"
    )
    research_task.description = f"{research_task.description}{lead_injection}"

    # Inject summarized opposing evidence into adversarial task
    counter_injection = (
        f"\n\nIMPORTANT: A deep research pre-scan has comprehensively analyzed "
        f"{counter_report.total_summaries} opposing sources in {counter_report.duration_seconds:.0f}s.\n\n"
        f"YOUR PRIMARY TASK: Synthesize and organize this pre-collected evidence. "
        f"Use ListResearchSources('counter') to browse all {counter_report.total_summaries} sources, "
        f"and ReadResearchSource('counter:N') to read any source in full.\n\n"
        f"SEARCH POLICY: Do NOT use RequestSearch unless you identify a CRITICAL "
        f"gap — a specific claim or mechanism that has ZERO coverage in the pre-scan. "
        f"The pre-scan already covers the major aspects of this topic.\n\n"
        f"PRE-COLLECTED OPPOSING EVIDENCE (condensed):\n{counter_summary}"
    )
    adversarial_task.description = f"{adversarial_task.description}{counter_injection}"

except Exception as e:
    print(f"⚠ Deep research pre-scan failed: {e}")
    print("Continuing with standard agent research...")
    deep_reports = None

# ================================================================
# Inject framing context into Crew 1 tasks (cross-crew context)
# ================================================================
if framing_output:
    framing_injection = (
        f"\n\nRESEARCH FRAMING CONTEXT (from Phase 0):\n"
        f"{framing_output}\n"
        f"--- END FRAMING CONTEXT ---\n"
    )
    research_task.description = f"{research_task.description}{framing_injection}"

# ================================================================
# CREW 1: Phases 1-2 (Research + Gate)
# ================================================================
print(f"\n{'='*70}")
print(f"CREW 1: PHASES 1-2 (RESEARCH + GATE)")
print(f"{'='*70}")

crew_1 = Crew(
    agents=[researcher, auditor],
    tasks=[research_task, gap_analysis_task],
    verbose=True,
    process='sequential'
)

try:
    crew_1_result = crew_1.kickoff()
except TimeoutError as e:
    print(f"\n{'='*70}")
    print("CREW 1: AGENT TIMED OUT — using partial results")
    print(f"{'='*70}")
    print(f"Timeout details: {str(e)[:200]}")
    # Extract whatever partial output exists from the tasks
    for t in [research_task, gap_analysis_task]:
        if hasattr(t, 'output') and t.output and hasattr(t.output, 'raw'):
            print(f"  Task '{t.description[:60]}...' has {len(t.output.raw)} chars of output")
        else:
            print(f"  Task '{t.description[:60]}...' has no output yet")
    # Continue with whatever we have — agents should draw conclusions from available evidence
except Exception as e:
    print(f"\n{'='*70}")
    print("CREW 1 FAILED")
    print(f"{'='*70}")
    print(f"Error: {e}")
    raise

# ================================================================
# GATE CHECK: Parse gap_analysis_task output for VERDICT
# ================================================================
print(f"\n{'='*70}")
print(f"RESEARCH GATE CHECK")
print(f"{'='*70}")

gate_passed = True
gap_fill_output = ""

gate_output = gap_analysis_task.output.raw if hasattr(gap_analysis_task, 'output') and gap_analysis_task.output else ""

# Parse for VERDICT: PASS or VERDICT: FAIL
verdict_match = re.search(r'VERDICT:\s*(PASS|FAIL)', gate_output, re.IGNORECASE)
if verdict_match:
    verdict = verdict_match.group(1).upper()
    gate_passed = (verdict == "PASS")
    print(f"Gate verdict: {verdict}")
else:
    # If no clear verdict found, default to PASS with warning
    print("⚠ No clear VERDICT found in gate output. Defaulting to PASS.")
    gate_passed = True

if not gate_passed:
    print(f"\n{'='*70}")
    print(f"PHASE 2b: GAP-FILL RESEARCH (Gate FAILED)")
    print(f"{'='*70}")

    gap_fill_task.description = (
        f"{gap_fill_task.description}\n\n"
        f"GAP ANALYSIS RESULTS (the gaps you need to fill):\n"
        f"{gate_output}\n"
        f"--- END GAP ANALYSIS ---"
    )

    MAX_SEARCH_ROUNDS = 3
    gap_fill_output = ""
    for search_round in range(MAX_SEARCH_ROUNDS):
        print(f"\n  --- Gap-Fill Round {search_round + 1}/{MAX_SEARCH_ROUNDS} ---")
        _pending_search_requests.clear()

        gap_fill_crew = Crew(
            agents=[researcher],
            tasks=[gap_fill_task],
            verbose=True,
            process='sequential'
        )

        try:
            gap_fill_crew.kickoff()
            gap_fill_output = gap_fill_task.output.raw if hasattr(gap_fill_task, 'output') and gap_fill_task.output else ""
            print(f"  Round {search_round + 1}: Agent produced {len(gap_fill_output)} chars")
        except Exception as e:
            print(f"  Round {search_round + 1}: Gap-fill failed: {e}")
            break

        if not _pending_search_requests:
            print(f"  No search requests queued — gap-fill complete")
            break

        print(f"  Executing {len(_pending_search_requests)} queued search requests...")
        new_sources = asyncio.run(execute_gap_fill_searches(
            pending_requests=list(_pending_search_requests),
            role="lead",
            brave_api_key=brave_key,
            fast_model_available=fast_model_available,
        ))

        if new_sources:
            append_sources_to_library(new_sources, "lead")
            print(f"  Added {len(new_sources)} new sources to library")
        else:
            print(f"  No new sources found — gap-fill complete")
            break

    # Insert gap_fill_task into tracking list
    idx = all_task_list.index(adversarial_task)
    all_task_list.insert(idx, gap_fill_task)
else:
    print("✓ Gate PASSED — skipping gap-fill research")
    gap_fill_output = ""

# Store gate status for progress tracker
progress_tracker.gate_passed = gate_passed

# ================================================================
# BATCH URL VALIDATION (parallel, outside agent loops)
# ================================================================
print(f"\n{'='*70}")
print(f"BATCH URL VALIDATION (parallel)")
print(f"{'='*70}")

from link_validator_tool import validate_multiple_urls_parallel

all_urls = set()
url_pattern = re.compile(r'https?://[^\s\)\]\"\'<>]+')
# Collect from research output
research_output = research_task.output.raw if hasattr(research_task, 'output') and research_task.output else ""
if research_output:
    all_urls.update(url_pattern.findall(research_output))
if gap_fill_output:
    all_urls.update(url_pattern.findall(gap_fill_output))

# From source library
sources_file = output_dir / "deep_research_sources.json"
if sources_file.exists():
    try:
        src_data = json.loads(sources_file.read_text())
        for role_sources in src_data.values():
            if isinstance(role_sources, list):
                for src in role_sources:
                    if src.get("url"):
                        all_urls.add(src["url"])
    except Exception:
        pass

print(f"  Found {len(all_urls)} unique URLs to validate")

if all_urls:
    validation_results = validate_multiple_urls_parallel(list(all_urls), max_workers=15)
    valid_count = sum(1 for v in validation_results.values() if "Valid" in v)
    broken_count = sum(1 for v in validation_results.values() if "Broken" in v or "Invalid" in v)
    print(f"  Results: {valid_count} valid, {broken_count} broken, "
          f"{len(validation_results) - valid_count - broken_count} other")

    validation_file = output_dir / "url_validation_results.json"
    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {validation_file}")

    validation_summary = "\n".join(
        f"  {url}: {status}" for url, status in sorted(validation_results.items())
    )
    source_verification_task.description = (
        f"{source_verification_task.description}\n\n"
        f"PRE-VALIDATED URL RESULTS ({len(validation_results)} URLs checked in parallel):\n"
        f"{validation_summary}\n"
        f"--- END PRE-VALIDATION ---\n"
        f"Use these results instead of checking URLs one by one. "
        f"Only use LinkValidator for any NEW URLs not in this list."
    )
else:
    print("  No URLs found to validate")

# ================================================================
# Inject cross-crew context into Crew 2 tasks
# ================================================================
# Inject research output into downstream tasks (research_output already set above)
gap_analysis_output = gap_analysis_task.output.raw if hasattr(gap_analysis_task, 'output') and gap_analysis_task.output else ""
gap_analysis_output = gap_analysis_task.output.raw if hasattr(gap_analysis_task, 'output') and gap_analysis_task.output else ""

# Build combined research context for adversarial task
# Keep injections small — the deep research pre-scan is already injected above
research_context_injection = (
    f"\n\nPRIOR RESEARCH CONTEXT (from Phases 1-2):\n\n"
    f"=== SUPPORTING RESEARCH (summary) ===\n{research_output[:4000]}\n\n"
    f"=== GAP ANALYSIS ===\n{gap_analysis_output[:2000]}\n"
)
if gap_fill_output:
    research_context_injection += (
        f"\n=== GAP-FILL RESEARCH (Phase 2b) ===\n{gap_fill_output[:2000]}\n"
    )
research_context_injection += f"--- END PRIOR CONTEXT ---\n"
adversarial_task.description = f"{adversarial_task.description}{research_context_injection}"

# Inject framing + research into source verification
if framing_output:
    source_verification_task.description = (
        f"{source_verification_task.description}\n\n"
        f"RESEARCH FRAMING (for scope reference):\n{framing_output[:2000]}\n"
        f"--- END FRAMING ---\n"
    )

# Inject research context into audit (source-of-truth) task
audit_context_injection = (
    f"\n\nPRIOR RESEARCH CONTEXT:\n\n"
    f"=== SUPPORTING RESEARCH ===\n{research_output[:4000]}\n\n"
    f"=== GAP ANALYSIS ===\n{gap_analysis_output[:2000]}\n"
)
if gap_fill_output:
    audit_context_injection += f"\n=== GAP-FILL ===\n{gap_fill_output[:2000]}\n"
audit_context_injection += f"--- END PRIOR CONTEXT ---\n"
audit_task.description = f"{audit_task.description}{audit_context_injection}"

# ================================================================
# CREW 2: Phases 3-8 (Adversarial through Accuracy Check)
# ================================================================
print(f"\n{'='*70}")
print(f"CREW 2: PHASES 3-8 (ADVERSARIAL → ACCURACY CHECK)")
print(f"{'='*70}")

# Build Crew 2 task list — order depends on whether translation is needed
if translation_task is not None:
    # Non-English: script → translate → polish → show notes → accuracy check
    print(f"\nTRANSLATION PHASE: Translating to {language_config['name']}")
    crew_2_tasks = [
        adversarial_task,
        source_verification_task,
        audit_task,
        script_task,
        translation_task,
        natural_language_task,
        show_notes_task,
        accuracy_check_task,
    ]
else:
    # English: original order (no translation)
    crew_2_tasks = [
        adversarial_task,
        source_verification_task,
        audit_task,
        show_notes_task,
        script_task,
        natural_language_task,
        accuracy_check_task,
    ]

crew_2 = Crew(
    agents=[counter_researcher, source_verifier, auditor, scriptwriter, personality],
    tasks=crew_2_tasks,
    verbose=True,
    process='sequential'
)

# Start background monitor for crew 2
monitor = CrewMonitor(all_task_list, progress_tracker)
monitor.start()

try:
    result = crew_2.kickoff()
except Exception as e:
    print(f"\n{'='*70}")
    print("CREW 2 FAILED")
    print(f"{'='*70}")
    print(f"Error: {e}")
    monitor.stop()
    raise
finally:
    monitor.stop()
    monitor.join(timeout=2)
    progress_tracker.workflow_completed()

# --- PDF GENERATION STEP ---
print("\n--- Generating Documentation PDFs ---")
pdf_tasks = [
    ("Research Framing", framing_output, "research_framing.pdf"),
    ("Supporting Scientific Paper", research_task, "supporting_paper.pdf"),
    ("Adversarial Anti-Thesis Paper", adversarial_task, "adversarial_paper.pdf"),
    ("Verified Source Bibliography", source_verification_task, "verified_sources_bibliography.pdf"),
    ("Source of Truth", audit_task, "source_of_truth.pdf"),
    ("Accuracy Check", accuracy_check_task, "accuracy_check.pdf"),
]
for title, source, filename in pdf_tasks:
    try:
        if isinstance(source, str):
            content = source
        elif hasattr(source, 'output') and source.output and hasattr(source.output, 'raw'):
            content = source.output.raw
        else:
            print(f"  Skipping {filename}: no output available")
            continue
        create_pdf(title, content, filename)
    except Exception as e:
        print(f"  Warning: Failed to create {filename}: {e}")

print("\n--- Saving Research Outputs (Markdown) ---")
markdown_outputs = [
    ("Research Framing", framing_output, "research_framing.md"),
    ("Supporting Research", research_task, "supporting_research.md"),
    ("Gap Analysis", gap_analysis_task, "gap_analysis.md"),
    ("Gap-Fill Research", gap_fill_output, "gap_fill_research.md"),
    ("Adversarial Research", adversarial_task, "adversarial_research.md"),
    ("Source Verification", source_verification_task, "source_verification.md"),
    ("Source of Truth", audit_task, "source_of_truth.md"),
    ("Accuracy Check", accuracy_check_task, "accuracy_check.md"),
    ("Show Notes", show_notes_task, "show_notes.md"),
    ("Podcast Script (Raw)", script_task, "podcast_script_raw.md"),
    ("Podcast Script (Polished)", natural_language_task, "podcast_script_polished.md"),
]
for label, source, filename in markdown_outputs:
    try:
        if isinstance(source, str):
            content = source
        elif hasattr(source, 'output') and source.output and hasattr(source.output, 'raw'):
            content = source.output.raw
        else:
            content = None
        if content and content.strip():
            outfile = output_dir / filename
            with open(outfile, 'w') as f:
                f.write(content)
            print(f"  Saved {filename} ({len(content)} chars)")
        else:
            print(f"  Skipping {filename}: no output available")
    except Exception as e:
        print(f"  Warning: Could not save {filename}: {e}")

# --- RESEARCH SUMMARY ---
if deep_reports is not None:
    deep_audit = deep_reports["audit"]
    print(f"\n--- Deep Research Summary ---")
    print(f"  Lead sources: {deep_reports['lead'].total_summaries}")
    print(f"  Counter sources: {deep_reports['counter'].total_summaries}")
    print(f"  Total sources: {deep_audit.total_summaries}")
    print(f"  Total URLs fetched: {deep_audit.total_urls_fetched}")
    print(f"  Duration: {deep_audit.duration_seconds:.0f}s")

# --- GATE SUMMARY ---
print(f"\n--- Gate Summary ---")
print(f"  Gate verdict: {'PASS' if gate_passed else 'FAIL'}")
if not gate_passed:
    print(f"  Gap-fill research: {'completed' if gap_fill_output else 'failed'}")

# --- SESSION METADATA ---
print("\n--- Documenting Session Metadata ---")
session_metadata = (
    f"PODCAST SESSION METADATA\n{'='*60}\n\n"
    f"Topic: {topic_name}\n\n"
    f"Language: {language_config['name']} ({language})\n\n"
    f"Character Assignments:\n"
    f"  {SESSION_ROLES['pro']['character']}: Supporting ({SESSION_ROLES['pro']['personality']})\n"
    f"  {SESSION_ROLES['con']['character']}: Critical ({SESSION_ROLES['con']['personality']})\n"
)
metadata_file = output_dir / "session_metadata.txt"
with open(metadata_file, 'w') as f:
    f.write(session_metadata)
print(f"Session metadata: {metadata_file}")

# --- SCRIPT PARSING (Deprecated - now handled by audio_engine.py) ---
# These functions are no longer needed as Kokoro's audio_engine handles
# script parsing internally

# def parse_script_to_segments(script_text: str, character_mapping: dict = None) -> list:
#     """DEPRECATED: Use audio_engine.generate_audio_from_script() instead"""
#     pass

# def save_parsed_segments(segments: list):
#     """DEPRECATED: No longer needed"""
#     pass


# --- LEGACY AUDIO GENERATION (Deprecated - kept for reference) ---
# MetaVoice-1B has been replaced by Kokoro TTS (audio_engine.py)
# The old functions below are commented out but kept for reference

# def generate_audio_metavoice(dialogue_segments: list, output_filename: str = "podcast_final_audio.wav"):
#     """DEPRECATED: Use audio_engine.generate_audio_from_script() instead"""
#     pass

# def generate_audio_gtts_fallback(dialogue_segments: list, output_filename: str = "podcast_final_audio.mp3"):
#     """DEPRECATED: No longer needed with Kokoro TTS"""
#     pass

# Generate audio with Kokoro TTS
print("\n--- Generating Multi-Voice Podcast Audio (Kokoro TTS) ---")

# Check script length before generation
# Get the polished script from natural_language_task (not the last crew result, which is accuracy_check)
script_text = natural_language_task.output.raw if hasattr(natural_language_task, 'output') and natural_language_task.output else result.raw
word_count = len(script_text.split())
estimated_duration_min = word_count / 150  # 150 words per minute

print(f"\n{'='*60}")
print(f"DURATION CHECK")
print(f"{'='*60}")
print(f"Script word count: {word_count}")
print(f"Estimated duration: {estimated_duration_min:.1f} minutes")
print(f"Target: 10 minutes (1,500 words)")

if word_count < 1350:
    print(f"⚠ WARNING: Script is SHORT ({word_count} words < 1,500 target)")
    print(f"  Estimated {estimated_duration_min:.1f} min")
elif word_count > 1650:
    print(f"⚠ WARNING: Script is LONG ({word_count} words > 1,500 target)")
    print(f"  Estimated {estimated_duration_min:.1f} min")
else:
    print(f"✓ Script length GOOD ({word_count} words)")
    print(f"  Estimated {estimated_duration_min:.1f} min")
print(f"{'='*60}\n")

# Clean script and generate audio with Kokoro
cleaned_script = clean_script_for_tts(script_text)

# Save podcast script for review
script_file = output_dir / "podcast_script.txt"
with open(script_file, 'w') as f:
    f.write(script_text)
print(f"Podcast script saved: {script_file} ({word_count} words)")

output_path = output_dir / "podcast_final_audio.wav"

audio_file = None
try:
    audio_file = generate_audio_from_script(cleaned_script, str(output_path), lang_code=language_config['tts_code'])
    if audio_file:
        audio_file = Path(audio_file)
        # Post-process: normalize loudness and optionally overlay background music
        mastered = post_process_audio(str(audio_file))
        if mastered:
            audio_file = Path(mastered)
except Exception as e:
    print(f"✗ ERROR: Kokoro TTS failed: {e}")
    print("  Ensure Kokoro is installed: pip install kokoro>=0.9")
    audio_file = None

# Check actual audio duration
if audio_file and audio_file.exists():
    try:
        with wave.open(str(audio_file), 'r') as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            duration_seconds = frames / float(rate)
            duration_minutes = duration_seconds / 60

        print(f"\n{'='*60}")
        print(f"AUDIO DURATION VERIFICATION")
        print(f"{'='*60}")
        print(f"Actual audio duration: {duration_minutes:.2f} minutes ({duration_seconds:.1f} seconds)")
        print(f"Target range: 9-11 minutes")

        if duration_minutes < 9.0:
            print(f"✗ FAILED: Audio is TOO SHORT ({duration_minutes:.2f} min < 9 min)")
            print(f"  ACTION: Re-run with longer script")
        elif duration_minutes > 11.0:
            print(f"✗ FAILED: Audio is TOO LONG ({duration_minutes:.2f} min > 11 min)")
            print(f"  ACTION: Re-run with shorter script")
        else:
            print(f"✓ SUCCESS: Audio duration within acceptable range")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"Warning: Could not verify audio duration: {e}")