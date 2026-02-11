import os
import platform
import re
import httpx
import time
import random
import sys
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
from audio_engine import generate_audio_from_script, clean_script_for_tts
from search_agent import SearxngClient, DeepResearch
from research_planner import build_research_plan, run_iterative_search, compare_plan_vs_results, run_supplementary_research
from deep_research_agent import Orchestrator, run_deep_research

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
        'tts_code': 'en',
        'instruction': 'Write all content in English.',
        'pdf_font': 'Helvetica'  # Latin-1 compatible
    },
    'ja': {
        'name': '日本語 (Japanese)',
        'tts_code': 'ja',
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

researcher = Agent(
    role='Principal Investigator (Lead Researcher)',
    goal=f'Find and document credible scientific signals about {topic_name}, organized by mechanism of action. {language_instruction}',
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
        f'{language_instruction}'
    ),
    tools=[search_tool, deep_search_tool],
    llm=dgx_llm_strict,
    verbose=True
)

auditor = Agent(
    role='Scientific Auditor (The Grader)',
    goal=f'Grade the research quality with a Reliability Scorecard. Do NOT write content - GRADE it. {language_instruction}',
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
        f'  4. Consensus Check: Search specifically for "criticism of [topic]" or "limitations of [study]".\n'
        f'  5. Source Validation: Use DeepSearch to scan through full article content from cited URLs.\n'
        f'     Verify that claims actually match what the source says. REJECT misrepresented sources.\n'
        f'\n'
        f'OUTPUT: A structured Markdown report with a "Reliability Scorecard". '
        f'{language_instruction}'
    ),
    tools=[search_tool, deep_search_tool, link_validator],
    llm=dgx_llm_strict,
    verbose=True
)

counter_researcher = Agent(
    role='Adversarial Researcher (The Skeptic)',
    goal=f'Systematically challenge and debunk specific claims about {topic_name}. {language_instruction}',
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
        f'{language_instruction}'
    ),
    tools=[search_tool, deep_search_tool],
    llm=dgx_llm_strict,
    verbose=True
)

scriptwriter = Agent(
    role='Podcast Producer (The Showrunner)',
    goal=(
        f'Transform research into a Masters/PhD-level debate on "{topic_name}". '
        f'Target: Intellectual, curious, slightly skeptical professionals. {language_instruction}'
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
        f'{language_instruction}'
    ),
    llm=dgx_llm_creative,
    verbose=True
)

personality = Agent(
    role='Podcast Personality (The Editor)',
    goal=(
        f'Polish the "{topic_name}" script for natural verbal delivery at Masters-level. '
        f'Target: Exactly 1,500 words (10 minutes). '
        f'{language_instruction}'
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
        f'{language_instruction}'
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
    tools=[link_validator],
    llm=dgx_llm_strict,
    verbose=True
)

# --- TASKS ---
research_task = Task(
    description=(
        f"Conduct exhaustive deep dive into {topic_name}. "
        f"Draft condensed scientific paper (Nature style). "
        f"\n\nCRITICAL: Focus ONLY on the health topic itself. Include:\n"
        f"- Specific health effects and mechanisms\n"
        f"- Biochemical pathways and physiological impacts\n"
        f"- Clinical outcomes and disease relationships\n"
        f"- Concrete examples of health consequences\n\n"
        f"EVIDENCE HIERARCHY:\n"
        f"1. PRIMARY: RCTs and meta-analyses from Nature/Science/Lancet/Cell/PNAS\n"
        f"2. SECONDARY: Observatory studies, cohort studies, epidemiological data (when RCTs unavailable)\n"
        f"3. SUPPLEMENTARY: Non-human RCTs (animal studies, in vitro) to verify proposed mechanisms\n\n"
        f"SEARCH STRATEGY: Start with RCT/meta-analysis search. If no strong evidence, "
        f"expand to observatory studies. Supplement with animal/mechanistic studies to validate logic.\n\n"
        f"CRITICAL: Use BraveSearch or DeepSearch to find and cite verifiable sources with URLs for ALL major claims. "
        f"Every citation in your bibliography MUST include a URL for source validation.\n"
        f"Include: Abstract, Introduction, 3 Biochemical Mechanisms with CONCRETE health impacts, Bibliography with URLs and study types noted. "
        f"{language_instruction}"
    ),
    expected_output=f"Scientific paper with SPECIFIC health mechanisms and effects, citations with URLs from RCTs, observatory studies, and non-human studies. Bibliography must include verifiable URLs for all sources. {language_instruction}",
    agent=researcher
)

gap_analysis_task = Task(
    description=(
        f"Review the Lead Scientist's Supporting Paper. Identify potential weaknesses "
        f"and suggest specific topics for the Adversarial Researcher to investigate. "
        f"{language_instruction}"
    ),
    expected_output=f"A list of 3-5 specific scientific 'weak points'. {language_instruction}",
    agent=auditor,
    context=[research_task]
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
        f"SEARCH STRATEGY: Find contradictory RCTs first. If limited, use observatory studies showing "
        f"no effect or harm. Include animal studies that disprove the mechanism.\n\n"
        f"CRITICAL: Use BraveSearch or DeepSearch to find and cite contradictory evidence with URLs. "
        f"Every citation in your bibliography MUST include a URL for source validation.\n"
        f"Include Bibliography with URLs and study types noted. "
        f"{language_instruction}"
    ),
    expected_output=f"Scientific paper challenging SPECIFIC health claims with contradictory evidence from RCTs, observatory studies, and animal studies. Bibliography must include verifiable URLs for all sources. {language_instruction}",
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
        f"Create structured bibliography JSON:\n"
        f'{{"supporting_sources": [{{title, url, journal, year, trust_level, source_type}}],\n'
        f' "contradicting_sources": [...],\n'
        f' "summary": "X high-trust, Y medium-trust sources"}}\n\n'
        f"REJECT non-scientific sources. Flag if <3 high-trust sources. "
        f"{language_instruction}"
    ),
    expected_output=f"JSON bibliography with categorized, verified sources and quality summary. {language_instruction}",
    agent=source_verifier,
    context=[research_task, adversarial_task]
)

audit_task = Task(
    description=(
        f"Review Supporting, Anti-Thesis papers on {topic_name} AND verified source bibliography. "
        f"Use LinkValidatorTool to verify every URL cited. If a URL is broken, REJECT that citation.\n\n"
        f"GRADING REQUIREMENTS:\n"
        f"1. Assign strength ratings (1-10) to each main claim\n"
        f"2. Create a 'Reliability Scorecard' with scores and justifications\n"
        f"3. Build 'The Caveat Box' - list why findings might be wrong:\n"
        f"   - Sample size issues (e.g., 'n=12 only')\n"
        f"   - Study limitations (e.g., 'Mouse study only')\n"
        f"   - Conflicts of interest\n"
        f"   - Contradictory findings from other studies\n"
        f"4. Search for criticism: 'criticism of {topic_name}' and 'limitations of [study]'\n\n"
        f"OUTPUT FORMAT (Markdown):\n"
        f"# Research Audit Report: {topic_name}\n\n"
        f"## Abstract\n"
        f"[Brief summary of findings]\n\n"
        f"## Evidence Block\n"
        f"[Key findings grouped by mechanism]\n\n"
        f"## Reliability Scorecard\n"
        f"| Claim | Strength (1-10) | Evidence Type | Justification |\n"
        f"| --- | --- | --- | --- |\n\n"
        f"## The Caveat Box\n"
        f"### Why These Findings Might Be Wrong:\n"
        f"- [List of limitations and concerns]\n\n"
        f"The output MUST contain concrete health information, NOT a discussion about source quality. "
        f"{language_instruction}"
    ),
    expected_output=(
        f"Structured Markdown report (RESEARCH_REPORT.md format) with:\n"
        f"- Abstract\n"
        f"- Evidence Block (mechanisms grouped)\n"
        f"- Reliability Scorecard (table with 1-10 ratings)\n"
        f"- Caveat Box (limitations list)\n"
        f"{language_instruction}"
    ),
    agent=auditor,
    context=[research_task, adversarial_task, source_verification_task],
    output_file=str(output_dir / "RESEARCH_REPORT.md")
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
        f"{language_instruction}"
    ),
    expected_output=(
        f"Dialogue about the health risks of {topic_name} between {SESSION_ROLES['pro']['character']} (risks are real) "
        f"and {SESSION_ROLES['con']['character']} (some risks are overstated). Every line discusses the topic. "
        f"{language_instruction}"
    ),
    agent=scriptwriter,
    context=[audit_task]
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
        f"{language_instruction}"
    ),
    expected_output=(
        f"Final Masters-level dialogue about {topic_name}, exactly 1,500 words. "
        f"No basic definitions. Host 2 challenges weak evidence. "
        f"{language_instruction}"
    ),
    agent=personality,
    context=[script_task, audit_task]
)

show_notes_task = Task(
    description=(
        f"Generate comprehensive show notes (SHOW_NOTES.md) for the podcast episode on {topic_name}.\n\n"
        f"Using the Research Report and verified sources, create a bulleted list with:\n"
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
        f"{language_instruction}"
    ),
    expected_output=(
        f"Markdown show notes with:\n"
        f"- Episode title\n"
        f"- Key takeaways (3-5 bullets)\n"
        f"- Full citation list with validity ratings (✓ High/Medium/Low)\n"
        f"- Evidence type labels (RCT/Observational/Animal Model)\n"
        f"{language_instruction}"
    ),
    agent=scriptwriter,
    context=[audit_task, source_verification_task],
    output_file=str(output_dir / "SHOW_NOTES.md")
)

# --- TASK METADATA & WORKFLOW PLANNING ---
TASK_METADATA = {
    'research_task': {
        'name': 'Initial Research & Evidence Gathering',
        'phase': 1,
        'estimated_duration_min': 8,
        'description': 'Lead Researcher conducts deep dive into topic',
        'agent': 'Principal Investigator',
        'dependencies': []
    },
    'gap_analysis_task': {
        'name': 'Research Quality Assessment',
        'phase': 2,
        'estimated_duration_min': 3,
        'description': 'Scientific Auditor identifies weak points',
        'agent': 'Scientific Auditor',
        'dependencies': ['research_task']
    },
    'adversarial_task': {
        'name': 'Counter-Evidence Research',
        'phase': 3,
        'estimated_duration_min': 8,
        'description': 'Counter-Researcher challenges findings',
        'agent': 'Adversarial Researcher',
        'dependencies': ['research_task', 'gap_analysis_task']
    },
    'source_verification_task': {
        'name': 'Source Validation & Bibliography',
        'phase': 4,
        'estimated_duration_min': 5,
        'description': 'Source Verifier validates all citations',
        'agent': 'Scientific Source Verifier',
        'dependencies': ['research_task', 'adversarial_task']
    },
    'audit_task': {
        'name': 'Final Meta-Audit & Grading',
        'phase': 5,
        'estimated_duration_min': 5,
        'description': 'Scientific Auditor grades research quality',
        'agent': 'Scientific Auditor',
        'dependencies': ['research_task', 'adversarial_task', 'source_verification_task']
    },
    'script_task': {
        'name': 'Podcast Script Generation',
        'phase': 6,
        'estimated_duration_min': 6,
        'description': 'Scriptwriter creates debate dialogue',
        'agent': 'Podcast Producer',
        'dependencies': ['audit_task']
    },
    'natural_language_task': {
        'name': 'Script Polishing & Editing',
        'phase': 7,
        'estimated_duration_min': 4,
        'description': 'Personality Agent refines for natural delivery',
        'agent': 'Podcast Personality',
        'dependencies': ['script_task', 'audit_task']
    },
    'show_notes_task': {
        'name': 'Show Notes & Citations',
        'phase': 8,
        'estimated_duration_min': 3,
        'description': 'Scriptwriter generates comprehensive show notes',
        'agent': 'Podcast Producer',
        'dependencies': ['audit_task', 'source_verification_task']
    }
}

def display_workflow_plan():
    """
    Display detailed workflow plan before execution.
    Shows all 8 phases with durations, dependencies, and total time estimate.
    """
    print("\n" + "="*70)
    print(" "*20 + "PODCAST GENERATION WORKFLOW")
    print("="*70)
    print(f"\nTopic: {topic_name}")
    print(f"Language: {language_config['name']}")
    print(f"Output Directory: {output_dir}")
    print("\n" + "-"*70)
    print(f"{'PHASE':<6} {'TASK NAME':<35} {'EST TIME':<12} {'AGENT':<20}")
    print("-"*70)

    total_duration = 0
    for task_name, metadata in TASK_METADATA.items():
        phase = metadata['phase']
        name = metadata['name']
        duration = metadata['estimated_duration_min']
        agent = metadata['agent']

        total_duration += duration

        print(f"{phase:<6} {name:<35} {duration:>3} min{'':<6} {agent:<20}")
        print(f"{'':6} └─ {metadata['description']}")
        if metadata['dependencies']:
            deps_str = ', '.join([f"Phase {TASK_METADATA[d]['phase']}" for d in metadata['dependencies']])
            print(f"{'':6}    Dependencies: {deps_str}")
        print()

    print("-"*70)
    print(f"TOTAL ESTIMATED TIME: {total_duration} minutes (~{total_duration//60}h {total_duration%60}m)")
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
        self.total_phases = len(task_metadata)
        self.start_time = None
        self.task_start_time = None
        self.completed_tasks = []

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
            deps_str = ', '.join([self.task_metadata[d]['name'] for d in metadata['dependencies']])
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

# --- EXECUTION ---
crew = Crew(
    agents=[researcher, auditor, counter_researcher, source_verifier, scriptwriter, personality],
    tasks=[
        research_task,
        gap_analysis_task,
        adversarial_task,
        source_verification_task,
        audit_task,
        script_task,
        natural_language_task,
        show_notes_task
    ],
    verbose=True,
    process='sequential'
)

# Display workflow plan before execution
display_workflow_plan()

# Initialize progress tracker
progress_tracker = ProgressTracker(TASK_METADATA)
progress_tracker.start_workflow()

# Get task list for tracking
task_list = [
    research_task,
    gap_analysis_task,
    adversarial_task,
    source_verification_task,
    audit_task,
    script_task,
    natural_language_task,
    show_notes_task
]

print(f"\n--- Initiating Scientific Research Pipeline on DGX Spark ---")
print(f"Topic: {topic_name}")
print(f"Language: {language_config['name']} ({language})")
print("---\n")

# --- PHASE 0: DEEP RESEARCH PRE-SCAN (Dual-Model Map-Reduce) ---
brave_key = os.getenv("BRAVE_API_KEY", "")

# Check if fast model is available
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
        results_per_query=10,
        fast_model_available=fast_model_available
    ))

    # Save all reports (lead, counter, audit)
    for role_name, report in deep_reports.items():
        report_file = output_dir / f"deep_research_{role_name}.md"
        with open(report_file, 'w') as f:
            f.write(report.report)
        print(f"✓ {role_name.capitalize()} report saved: {report_file} ({report.total_summaries} sources)")

    # Use audit report (combined synthesis) for injection into CrewAI agents
    audit_report = deep_reports["audit"]
    lead_report = deep_reports["lead"]
    counter_report = deep_reports["counter"]

    # Inject supporting evidence into lead research task
    lead_injection = (
        f"\n\nIMPORTANT: A deep research pre-scan has already analyzed {lead_report.total_summaries} "
        f"supporting sources in {lead_report.duration_seconds:.0f}s. Use the evidence below as a "
        f"starting point, then supplement with your own searches.\n\n"
        f"PRE-COLLECTED SUPPORTING EVIDENCE:\n{lead_report.report}"
    )
    research_task.description = f"{research_task.description}{lead_injection}"

    # Inject opposing evidence into adversarial task
    counter_injection = (
        f"\n\nIMPORTANT: A deep research pre-scan has already analyzed {counter_report.total_summaries} "
        f"opposing sources in {counter_report.duration_seconds:.0f}s. Use the evidence below as a "
        f"starting point, then supplement with your own searches.\n\n"
        f"PRE-COLLECTED OPPOSING EVIDENCE:\n{counter_report.report}"
    )
    adversarial_task.description = f"{adversarial_task.description}{counter_injection}"

except Exception as e:
    print(f"⚠ Deep research pre-scan failed: {e}")
    print("Continuing with standard agent research...")
    deep_reports = None

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
                # Check how many tasks have outputs (completed)
                completed_count = 0
                for task in self.task_list:
                    if hasattr(task, 'output') and task.output is not None:
                        completed_count += 1
                    else:
                        break  # Tasks complete sequentially

                # New task started or completed
                if completed_count > self.last_completed:
                    # If we detected completion of previous task
                    if self.last_completed >= 0:
                        self.progress_tracker.task_completed(self.last_completed)

                    # New task has started
                    if completed_count < len(self.task_list):
                        self.progress_tracker.task_started(completed_count)

                    self.last_completed = completed_count

                time.sleep(3)  # Check every 3 seconds
            except Exception:
                pass  # Silently continue monitoring

    def stop(self):
        """Stop monitoring"""
        self.running = False

# Start background monitor
monitor = CrewMonitor(task_list, progress_tracker)
monitor.start()

# Execute crew
try:
    result = crew.kickoff()
except Exception as e:
    print(f"\n{'='*70}")
    print("WORKFLOW FAILED")
    print(f"{'='*70}")
    print(f"Error: {e}")
    print(f"{'='*70}\n")
    monitor.stop()
    raise
finally:
    # Stop monitor and show final summary
    monitor.stop()
    monitor.join(timeout=2)
    progress_tracker.workflow_completed()

# --- PDF GENERATION STEP ---
print("\n--- Generating Documentation PDFs ---")
try:
    # Use task_outputs to get specific results
    create_pdf("Supporting Scientific Paper", research_task.output.raw, "supporting_paper.pdf")
    create_pdf("Adversarial Anti-Thesis Paper", adversarial_task.output.raw, "adversarial_paper.pdf")
    create_pdf("Verified Source Bibliography", source_verification_task.output.raw, "verified_sources_bibliography.pdf")
    create_pdf("Final Meta-Audit Verdict", audit_task.output.raw, "final_audit_report.pdf")
except Exception as e:
    print(f"Warning: PDF generation failed, but research is complete: {e}")

# --- RESEARCH SUMMARY ---
if deep_reports is not None:
    audit = deep_reports["audit"]
    print(f"\n--- Deep Research Summary ---")
    print(f"  Lead sources: {deep_reports['lead'].total_summaries}")
    print(f"  Counter sources: {deep_reports['counter'].total_summaries}")
    print(f"  Total sources: {audit.total_summaries}")
    print(f"  Total URLs fetched: {audit.total_urls_fetched}")
    print(f"  Duration: {audit.duration_seconds:.0f}s")

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
script_text = result.raw
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
output_path = output_dir / "podcast_final_audio.wav"

audio_file = None
try:
    audio_file = generate_audio_from_script(cleaned_script, str(output_path))
    if audio_file:
        audio_file = Path(audio_file)
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