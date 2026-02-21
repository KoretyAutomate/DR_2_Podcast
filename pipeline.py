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
import shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from markdown_it import MarkdownIt
import weasyprint
from link_validator import LinkValidatorTool
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Literal
import soundfile as sf
import numpy as np
import wave
from audio_engine import generate_audio_from_script, clean_script_for_tts, post_process_audio
from clinical_research import run_deep_research

# --- EVIDENCE QUALITY CONSTANTS ---
EVIDENCE_LIMITED_THRESHOLD = 30  # evidence_quality = "limited" if aff_candidates < this


class InsufficientEvidenceError(RuntimeError):
    """Raised when the affirmative research track finds zero candidates."""
    pass


def _write_insufficient_evidence_report(topic, aff_n, neg_n, output_dir_path):
    """Write a structured failure report when evidence is completely absent."""
    report_path = output_dir_path / "insufficient_evidence_report.md"
    strat_aff = strat_neg = "(not available)"
    strat_data = {}
    for fname, var in [("search_strategy_aff.json", "aff"), ("search_strategy_neg.json", "neg")]:
        p = output_dir_path / fname
        if p.exists():
            try:
                data = json.loads(p.read_text())
                strat_data[var] = json.dumps(data.get("search_strings", {}), indent=2)
            except Exception:
                pass
    strat_aff = strat_data.get("aff", strat_aff)
    strat_neg = strat_data.get("neg", strat_neg)
    content = f"""# Insufficient Evidence Report

**Topic:** {topic}
**Run:** {output_dir_path.name}

## Search Results
- Affirmative track candidates: {aff_n}
- Adversarial track candidates: {neg_n}

## Diagnosis
The affirmative research track found **zero candidate studies**. This typically means:
1. The search strings used folk-language phrasing instead of canonical MeSH terms.
2. The topic is too narrow — no studies directly test the exact intervention.

## Suggested Rephrasing
Try rephrasing the topic using canonical scientific terms such as:
- "chrono-nutrition", "early time-restricted feeding", "caloric front-loading"
- "circadian meal timing", "meal timing and metabolic outcomes"

## Search Strategies Used
### Affirmative
{strat_aff}

### Adversarial
{strat_neg}
"""
    report_path.write_text(content)
    print(f"✗ Insufficient evidence report written → {report_path.name}")


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
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )

    # Redirect stdout and stderr to logger
    class StreamToLogger(object):
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level
            self.linebuf = ''

        def write(self, buf):
            for line in buf.rstrip().splitlines():
                self.logger.log(self.level, line.rstrip())

        def flush(self):
            pass

    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

# --- INITIALIZATION ---
load_dotenv()
# Configuration loaded from .env
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
def parse_arguments():
    """Parse command-line arguments for topic and language."""
    parser = argparse.ArgumentParser(
        description='Generate a research-driven debate podcast on any scientific topic.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --topic "effects of meditation on brain plasticity" --language en
  python pipeline.py --topic "climate change impact on marine ecosystems" --language ja

Environment variables:
  export PODCAST_TOPIC="your topic here"
  export PODCAST_LANGUAGE=ja
  python pipeline.py
        """
    )
    parser.add_argument(
        '--topic',
        type=str,
        help='Scientific topic for podcast research and debate'
    )
    parser.add_argument(
        '--language',
        type=str,
        choices=['en', 'ja'],
        help='Language for podcast generation (en=English, ja=Japanese)'
    )
    parser.add_argument(
        '--reuse-dir',
        type=str,
        help='Previous run output directory to reuse research from'
    )
    parser.add_argument(
        '--crew3-only',
        action='store_true',
        help='Skip research phases, run Crew 3 (podcast production) only using reuse-dir research'
    )
    parser.add_argument(
        '--check-supplemental',
        action='store_true',
        help='LLM decides if supplemental research is needed for the reused topic'
    )
    return parser.parse_args()

def get_topic(args):
    """
    Get podcast topic from multiple sources (priority order):
    1. Command-line argument (--topic)
    2. Environment variable (PODCAST_TOPIC)
    3. Default topic
    """
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

def get_language(args):
    """
    Get podcast language from multiple sources (priority order):
    1. Command-line argument (--language)
    2. Environment variable (PODCAST_LANGUAGE)
    3. Default language (English)
    """
    lang_code = 'en' # Default
    if args.language:
        lang_code = args.language
        print(f"Using language from command-line: {SUPPORTED_LANGUAGES[lang_code]['name']}")
    elif os.getenv("PODCAST_LANGUAGE") and os.getenv("PODCAST_LANGUAGE") in SUPPORTED_LANGUAGES:
        lang_code = os.getenv("PODCAST_LANGUAGE")
        print(f"Using language from environment: {SUPPORTED_LANGUAGES[lang_code]['name']}")
    else:
        print(f"Using default language: {SUPPORTED_LANGUAGES[lang_code]['name']}")
    return lang_code

args = parse_arguments()
topic_name = get_topic(args)

# --- CHARACTER CONFIGURATION ---
CHARACTERS = {
    "Kaz": {
        "gender": "male",
        "host_label": "Host 1",       # male → Host 1 voice (always)
        "voice_model": "male_voice",  # TTS-specific, will update in #3
        "base_personality": "Enthusiastic science communicator, clear explainer, data-driven"
    },
    "Erika": {
        "gender": "female",
        "host_label": "Host 2",       # female → Host 2 voice (always)
        "voice_model": "female_voice",  # TTS-specific, will update in #3
        "base_personality": "Curious and sharp interviewer, asks what the audience is thinking"
    }
}

# --- ROLE ASSIGNMENT (Dynamic per session) ---
def assign_roles() -> dict:
    """
    Assign Kaz and Erika to presenter/questioner roles.
    Respects PODCAST_HOSTS env var if set (kaz_erika, erika_kaz).
    Otherwise defaults to random assignment.
    """
    characters = list(CHARACTERS.keys())
    
    # Check for manual override
    host_config = os.getenv("PODCAST_HOSTS", "random").lower()
    
    if host_config == "kaz_erika":
        # Kaz is Presenter, Erika is Questioner
        presenter_name = "Kaz"
        questioner_name = "Erika"
    elif host_config == "erika_kaz":
        # Erika is Presenter, Kaz is Questioner
        presenter_name = "Erika"
        questioner_name = "Kaz"
    else:
        # Random assignment
        random.shuffle(characters)
        presenter_name = characters[0]
        questioner_name = characters[1]

    role_assignment = {
        "presenter": {
            "character": presenter_name,
            "label": CHARACTERS[presenter_name]["host_label"],
            "stance": "teaching",
            "personality": CHARACTERS[presenter_name]["base_personality"]
        },
        "questioner": {
            "character": questioner_name,
            "label": CHARACTERS[questioner_name]["host_label"],
            "stance": "curious",
            "personality": CHARACTERS[questioner_name]["base_personality"]
        }
    }

    print(f"\n{'='*60}")
    print(f"SESSION ROLE ASSIGNMENT ({host_config}):")
    print(f"  Presenter: {role_assignment['presenter']['character']} ({CHARACTERS[presenter_name]['gender']})")
    print(f"  Questioner: {role_assignment['questioner']['character']} ({CHARACTERS[questioner_name]['gender']})")
    print(f"{'='*60}\n")

    return role_assignment

SESSION_ROLES = assign_roles()


# --- REUSE HELPER FUNCTIONS ---
RESEARCH_ARTIFACTS = [
    "source_of_truth.md", "SOURCE_OF_TRUTH.md",
    "research_sources.json",
    "research_framing.md", "research_framing.pdf",
    "source_of_truth.pdf",
    "source_of_truth_ja.md",
    "source_of_truth_ja.pdf",
    "url_validation_results.json",
    "affirmative_case.md", "falsification_case.md", "grade_synthesis.md",
    "clinical_math.md",
    "search_strategy_aff.json",
    "search_strategy_neg.json",
    "screening_results_aff.json",
    "screening_results_neg.json",
]

# Legacy artifact names for backward compatibility with old runs
LEGACY_ARTIFACT_NAMES = {
    "research_sources.json": "deep_research_sources.json",
    "affirmative_case.md": "deep_research_lead.md",
    "falsification_case.md": "deep_research_counter.md",
    "grade_synthesis.md": "deep_research_audit.md",
    "clinical_math.md": "deep_research_math.md",
    "search_strategy_aff.json": "deep_research_strategy_aff.json",
    "search_strategy_neg.json": "deep_research_strategy_neg.json",
    "screening_results_aff.json": "deep_research_screening.json",  # aff falls back to old unified file
    "screening_results_neg.json": "deep_research_screening.json",  # neg falls back to old unified file
    "show_outline.md": "show_notes.md",
    "SHOW_OUTLINE.md": "SHOW_NOTES.md",
    "accuracy_audit.md": "accuracy_check.md",
    "ACCURACY_AUDIT.md": "ACCURACY_CHECK.md",
    "accuracy_audit.pdf": "accuracy_check.pdf",
    "script_draft.md": "podcast_script_raw.md",
    "script_final.md": "podcast_script_polished.md",
    "audio.wav": "audio.wav",
    "script.txt": "script.txt",
}


def _copy_research_artifacts(src_dir: Path, dst_dir: Path):
    """Copy research-related files from a previous run to a new output directory."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in RESEARCH_ARTIFACTS:
        src = src_dir / name
        if not src.exists():
            # Fall back to legacy name from old runs
            legacy = LEGACY_ARTIFACT_NAMES.get(name)
            if legacy:
                src = src_dir / legacy
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            copied += 1
    print(f"  Copied {copied} research artifacts from {src_dir.name}")


def _copy_all_artifacts(src_dir: Path, dst_dir: Path):
    """Copy all files from a previous run to a new output directory."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in src_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, dst_dir / item.name)
            copied += 1
    print(f"  Copied {copied} total artifacts from {src_dir.name}")


def check_supplemental_needed(topic: str, reuse_dir: Path) -> dict:
    """Ask the LLM if the previous source_of_truth.md adequately covers the new topic."""
    sot_path = reuse_dir / "source_of_truth.md"
    if not sot_path.exists():
        sot_path = reuse_dir / "SOURCE_OF_TRUTH.md"
    if not sot_path.exists():
        return {"needs_supplement": True, "reason": "No source_of_truth.md found", "queries": []}

    sot_content = sot_path.read_text()[:8000]

    prompt = (
        f"You are a research completeness evaluator.\n\n"
        f"NEW TOPIC: {topic}\n\n"
        f"EXISTING RESEARCH REPORT (source_of_truth.md):\n{sot_content}\n\n"
        f"QUESTION: Does this existing report adequately cover the NEW TOPIC?\n"
        f"Consider: Are there significant gaps, missing perspectives, or outdated information?\n\n"
        f"Respond with a JSON object:\n"
        f'{{"needs_supplement": true/false, "reason": "brief explanation", '
        f'"queries": [{{"query": "search query", "goal": "what to find"}}]}}\n'
        f"If needs_supplement is false, queries should be an empty array.\n"
        f"If needs_supplement is true, provide 2-5 targeted search queries to fill gaps.\n"
        f"Return ONLY the JSON object."
    )

    try:
        resp = httpx.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-32B-Instruct-AWQ",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1024,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        # Extract JSON from response
        import re as _re
        json_match = _re.search(r'\{.*\}', content, _re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "needs_supplement": result.get("needs_supplement", True),
                "reason": result.get("reason", ""),
                "queries": result.get("queries", []),
            }
    except Exception as e:
        print(f"  Supplemental check failed: {e}")

    # Default to needing supplement if check fails
    return {"needs_supplement": True, "reason": "Check failed, running supplemental as precaution", "queries": []}


# --- TTS DEPENDENCY CHECK ---
def check_tts_dependencies():
    """Verify Kokoro TTS is installed."""
    try:
        import kokoro
        print("✓ Kokoro TTS dependencies verified")
    except ImportError as e:
        print(f"CRITICAL ERROR: Kokoro TTS not installed: {e}")
        print("Install with: pip install kokoro>=0.9")
        print("Audio generation cannot proceed without Kokoro.")
        sys.exit(1)

check_tts_dependencies()

# --- LANGUAGE CONFIGURATION ---
# speech_rate: units spoken per minute at a natural conversational pace
# length_unit: how script length is measured ('words' for space-delimited, 'chars' for character-based)
# prompt_unit: singular label used in LLM prompts (e.g. "word", "character")
SUPPORTED_LANGUAGES = {
    'en': {
        'name': 'English',
        'tts_code': 'a',            # Kokoro American English
        'instruction': 'Write all content in English.',
        'speech_rate': 150,         # ~150 words/min conversational pace
        'length_unit': 'words',
        'prompt_unit': 'word',
    },
    'ja': {
        'name': '日本語 (Japanese)',
        'tts_code': 'j',            # Kokoro Japanese / Qwen3-TTS
        'instruction': 'すべてのコンテンツを日本語で書いてください。(Write all content in Japanese.)',
        'speech_rate': 500,         # ~500 chars/min conversational pace
        'length_unit': 'chars',
        'prompt_unit': 'character',
    }
}

# Target episode duration per podcast_length mode (in minutes)
TARGET_MINUTES = {'short': 10, 'medium': 20, 'long': 30}
SCRIPT_TOLERANCE = 0.10  # ±10% around target length is acceptable

language = get_language(args)
language_config = SUPPORTED_LANGUAGES[language]
# English-first reasoning: all research/reasoning phases use English,
# only post-translation phases (polish, show notes, accuracy check) use target language.
english_instruction = "Write all content in English."
target_instruction = language_config['instruction']
# For backward compatibility and phases that always match the target language
language_instruction = language_config['instruction']

# --- DURATION TARGETS (computed once from speech rate × target minutes) ---
length_mode = os.getenv("PODCAST_LENGTH", "long").lower()
_speech_rate     = language_config['speech_rate']
_target_min      = TARGET_MINUTES.get(length_mode, TARGET_MINUTES['long'])
target_length_int    = _target_min * _speech_rate        # numeric, e.g. 4500
target_script        = f"{target_length_int:,}"          # formatted, e.g. "4,500"
target_unit_singular = language_config['prompt_unit']    # e.g. "word" / "character"
target_unit_plural   = language_config['length_unit']    # e.g. "words" / "chars"
duration_label       = f"{length_mode.capitalize()} ({_target_min} min)"

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
    max_tokens=8000,  # Researchers/auditors produce short structured outputs; leaves 24k for input
    stop=["<|im_end|>", "<|endoftext|>"]
)

dgx_llm_creative = LLM(
    model=final_model_string,
    base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
    api_key="NA",  # vLLM uses "NA"
    provider="openai",
    timeout=600,
    temperature=0.7,  # Creative mode for Producer/Personality
    max_tokens=12000,  # Scriptwriter needs more output for 4,500-word scripts; leaves 20k for input
    frequency_penalty=0.15,  # Prevent repetition loops while allowing long creative output
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
        client = OpenAI(base_url=os.getenv("FAST_LLM_BASE_URL", "http://localhost:11434/v1"), api_key="ollama")
        response = client.chat.completions.create(
            model=os.getenv("FAST_MODEL_NAME", "llama3.2:1b"),
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

def append_sources_to_library(new_sources: list[dict], role: str, output_dir_path=None):
    """Append new sources to research_sources.json."""
    src_dir = output_dir_path or output_dir
    sources_file = Path(src_dir) / "research_sources.json"
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
# deep research pre-scan that was saved to research_sources.json.

@tool("ListResearchSources")
def list_research_sources(role: str) -> str:
    """List all available research sources from the deep research pre-scan.

    Args:
        role: Either "lead" (supporting evidence) or "counter" (opposing evidence)

    Returns a numbered index with title, URL, and research goal for each source.
    Use ReadResearchSource to read the full summary of any specific source.
    """
    sources_file = output_dir / "research_sources.json"
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
    sources_file = output_dir / "research_sources.json"
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
        "lead": "affirmative_case.md",
        "counter": "falsification_case.md",
        "audit": "grade_synthesis.md",
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
_MD_PARSER = MarkdownIt().enable("table")

_PDF_CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: 'Noto Sans CJK JP', 'Noto Sans', Helvetica, sans-serif;
       font-size: 11pt; line-height: 1.5; color: #222; }
h1 { font-size: 18pt; border-bottom: 2px solid #333; padding-bottom: 4pt; }
h2 { font-size: 15pt; color: #1a5276; margin-top: 16pt; }
h3 { font-size: 13pt; color: #2e4053; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt; }
th, td { border: 1px solid #999; padding: 4pt 6pt; text-align: left; }
th { background: #d5dbdb; font-weight: bold; }
tr:nth-child(even) { background: #f2f3f4; }
code { font-family: monospace; background: #eee; padding: 1pt 3pt; }
.header { text-align: center; font-size: 10pt; color: #666; margin-bottom: 12pt; }
"""

def create_pdf(title, content, filename):
    """Convert markdown content to a styled PDF via weasyprint."""
    clean = re.sub(r'<think>.*?</think>', '', str(content), flags=re.DOTALL)
    body_html = _MD_PARSER.render(clean)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{_PDF_CSS}</style></head>
<body>
<div class="header">DGX Spark Research Intelligence Report</div>
<h1>{title}</h1>
{body_html}
</body></html>"""
    file_path = output_dir / filename
    weasyprint.HTML(string=html).write_pdf(str(file_path))
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

auditor_agent = Agent(
    role='Scientific Auditor',
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

producer_agent = Agent(
    role='Podcast Producer',
    goal=(
        f'Transform research into an engaging, in-depth teaching conversation on "{topic_name}". '
        f'Target: Intellectual, curious professionals who want to learn. {english_instruction}'
    ),
    backstory=(
        f'Science Communicator targeting Post-Graduate Professionals (Masters/PhD level). '
        f'Tone: Think "Huberman Lab" or "Lex Fridman" - intellectual, curious, deep-diving.\n\n'
        f'CRITICAL RULES:\n'
        f'  1. NO BASICS: Do NOT define basic terms like "DNA", "inflation", "supply chain", '
        f'     "peer review", "RCT", or "meta-analysis". Assume the listener knows them.\n'
        f'  2. LENGTH: Generate exactly {target_script} {target_unit_plural} (approx {_target_min} minutes at {_speech_rate} {target_unit_plural}/min). This is CRITICAL.\n'
        f'  3. FORMAT: Script MUST use "{SESSION_ROLES["presenter"]["label"]}:" (Presenter) '
        f'     and "{SESSION_ROLES["questioner"]["label"]}:" (Questioner).\n'
        f'  4. TEACHING STYLE: The Presenter explains the topic systematically. '
        f'     The Questioner asks bridging questions on behalf of the audience:\n'
        f'     - Clarify jargon or uncommon terms\n'
        f'     - Request real-world examples and analogies\n'
        f'     - Occasionally push back on weak or debated evidence\n'
        f'  5. DEPTH: Cover 3-4 main aspects of the topic thoroughly with mechanisms, evidence, and implications.\n'
        f'\n'
        f'Your dialogue should dive into nuance, trade-offs, and practical implications. '
        f'The questioner keeps it accessible without dumbing it down. '
        f'{english_instruction}'
        + (f'\n\nLANGUAGE WARNING: When generating Japanese (日本語) output, you MUST stay in Japanese throughout. '
           f'Do NOT switch to Chinese (中文). Use katakana for host names: カズ and エリカ (NOT 卡兹/埃里卡). '
           f'Avoid Kanji that is only used in Chinese (e.g., use 気 instead of 气, 楽 instead of 乐).'
           if language == 'ja' else '')
    ),
    llm=dgx_llm_creative,
    verbose=True
)

editor_agent = Agent(
    role='Podcast Editor',
    goal=(
        f'Polish the "{topic_name}" script for natural verbal delivery at Masters-level. '
        f'Target: Exactly 4,500 words (30 minutes). '
        f'{target_instruction}'
    ),
    backstory=(
        f'Editor for high-end intellectual podcasts (Huberman Lab, Lex Fridman). '
        f'Your audience has advanced degrees - they want depth, not hand-holding.\n\n'
        f'EDITING RULES:\n'
        f'  - Remove any definitions of basic scientific concepts\n'
        f'  - Ensure the questioner\'s questions feel natural and audience-aligned\n'
        f'  - Keep technical language intact (no dumbing down)\n'
        f'  - Target exactly 4,500 words for 30-minute runtime. If the script is too short, YOU MUST ADD DEPTH AND EXAMPLES TO REACH THE TARGET.\n'
        f'  - Ensure the opening follows the 3-part structure: welcome → hook question → topic shift\n'
        f'  - Teaching flow: presenter explains, questioner bridges gaps for listeners\n'
        f'\n'
        f'If script is too short, add more depth, examples, and practical implications. DO NOT CUT CONTENT IF THE SCRIPT IS UNDER THE WORD COUNT TARGET.\n'
        f'If too long, cut repetition while preserving teaching flow. '
        f'{target_instruction}'
    ),
    llm=dgx_llm_creative,
    verbose=True
)

framing_agent = Agent(
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
    agent=framing_agent,
    output_file=str(output_dir / "RESEARCH_FRAMING.md")
)

print(f"Podcast Length Mode: {duration_label}")

script_task = Task(
    description=(
        f"Using the audit report, write a comprehensive {target_script}-{target_unit_singular} podcast dialogue about \"{topic_name}\" "
        f"featuring {SESSION_ROLES['presenter']['character']} (presenter) and {SESSION_ROLES['questioner']['character']} (questioner).\n\n"
        f"STRUCTURE:\n"
        f"  1. OPENING (joint welcome):\n"
        f"     a) Both hosts greet listeners with a short, warm welcome to the channel\n"
        f"     b) One host hooks listeners with a relatable question (e.g., 'Have you ever wondered why...?' "
        f"or 'Have you experienced...?') — the other responds naturally\n"
        f"     c) Transition into the topic: 'Today, we're going to explore...'\n\n"
        f"  2. BODY — write EXACTLY 6 segments, each 500-600 words:\n"
        f"     SEGMENT 1: First main aspect — the core mechanism (how/why it works scientifically)\n"
        f"     SEGMENT 2: First aspect — evidence, studies, data\n"
        f"     SEGMENT 3: Second main aspect — mechanism\n"
        f"     SEGMENT 4: Second aspect — real-world implications and counter-arguments\n"
        f"     SEGMENT 5: Third main aspect — practical advice for listeners\n"
        f"     SEGMENT 6: Fourth main aspect OR synthesis — what it all means\n"
        f"     Each segment: Presenter explains (5-8 sentences) → Questioner asks 2-3 bridging questions → deeper dive → analogy → real example\n"
        f"     IMPORTANT: Each host turn must be at least 3-5 sentences. No one-line replies.\n\n"
        f"  3. CLOSING:\n"
        f"     - Summarize key takeaways\n"
        f"     - Practical advice for listeners\n"
        f"     - Sign off together\n\n"
        f"SIMPLIFY THE SCIENCE (not the research process):\n"
        f"- 'Glycemic index' → 'a score that measures how fast a food raises blood sugar'\n"
        f"- 'Insulin resistance' → 'when your body stops responding properly to insulin'\n"
        f"- 'Postprandial glucose spike' → 'a sharp rise in blood sugar after eating'\n\n"
        f"CHARACTER ROLES:\n"
        f"  - {SESSION_ROLES['presenter']['character']} (Presenter): presents evidence and explains the topic, "
        f"{SESSION_ROLES['presenter']['personality']}\n"
        f"  - {SESSION_ROLES['questioner']['character']} (Questioner): asks questions the audience would ask, bridges gaps, "
        f"{SESSION_ROLES['questioner']['personality']}\n\n"
        f"Format STRICTLY as:\n"
        f"{SESSION_ROLES['presenter']['label']}: [dialogue]\n"
        f"{SESSION_ROLES['questioner']['label']}: [dialogue]\n\n"
        f"TARGET LENGTH: {target_script} {target_unit_plural}. This is CRITICAL — do not write less. The podcast MUST last {_target_min} minutes. If you are too brief, the production will fail. SEGMENT CHECKLIST: You must write all 6 body segments. Count them as you write.\n"
        f"TO REACH THIS LENGTH: You must be extremely detailed and conversational. For every single claim or mechanism, you MUST provide:\n"
        f"  1. A deep-dive explanation of the specific scientific mechanism (how it works at a molecular/cellular level)\n"
        f"  2. A real-world analogy or metaphor that lasts several lines\n"
        f"  3. A practical, relatable example or case study\n"
        f"  4. A potential counter-argument or nuance followed by a rebuttal\n"
        f"  5. Interactive host dialogue (e.g., 'Wait, let me make sure I've got this right...', 'That's fascinating, tell me more about...')\n"
        f"Expand the conversation. Do not just list facts. Have the hosts explore the 'So what?' and 'What now?' for the audience.\n"
        f"Maintain consistent roles throughout. NO role switching mid-conversation. "
        f"{target_instruction}"
    ),
    expected_output=(
        f"A {target_script}-{target_unit_singular} teaching-style dialogue about {topic_name} between "
        f"{SESSION_ROLES['presenter']['character']} (presents and explains) "
        f"and {SESSION_ROLES['questioner']['character']} (asks bridging questions). "
        f"Opens with welcome → hook → topic shift. Every line discusses the topic. "
        f"{target_instruction}"
    ),
    agent=producer_agent,
    context=[]
)

# --- SOT TRANSLATION TASK (only when language != 'en') ---
# Translates the English Source-of-Truth into the target language BEFORE script writing.
# This ensures the podcast script is derived from translated research, not translated afterwards.
translation_task = None
if language != 'en':
    translation_task = Task(
        description=(
            f"Translate the entire Source-of-Truth document about {topic_name} into {language_config['name']}.\n\n"
            f"TRANSLATION RULES:\n"
            f"- Translate ALL sections faithfully: Executive Summary, Key Claims, Evidence, Bibliography\n"
            f"- Preserve scientific accuracy — translate meaning, not word-for-word\n"
            f"- Keep confidence labels (HIGH/MEDIUM/LOW/CONTESTED) intact\n"
            f"- Keep study names, journal names, and URLs in English\n"
            f"- Maintain all markdown formatting (headers, tables, bullet points)\n"
            + (f"- CRITICAL: Output MUST be in Japanese (日本語) only. Do NOT switch to Chinese (中文).\n"
               f"  Use standard Japanese kanji (e.g., 気 not 气, 楽 not 乐).\n"
               if language == 'ja' else '')
            + f"{target_instruction}"
        ),
        expected_output=(
            f"Complete {language_config['name']} translation of the Source-of-Truth document, "
            f"preserving all sections, claims, confidence levels, and evidence citations."
        ),
        agent=producer_agent,
        context=[],
    )

polish_task = Task(
    description=(
        f"Polish the \"{topic_name}\" dialogue for natural spoken delivery at Masters-level.\n\n"
        f"MASTERS-LEVEL REQUIREMENTS:\n"
        f"- Remove ALL definitions of basic scientific concepts (DNA, peer review, RCT, meta-analysis)\n"
        f"- Ensure the questioner's questions feel natural and audience-aligned\n"
        f"- Keep technical language intact - NO dumbing down\n"
        f"- Ensure the questioner's questions feel natural and audience-aligned\n"
        f"- Keep technical language intact - NO dumbing down\n"
        f"- Target exactly {target_script} {target_unit_plural}\n\n"
        f"MAINTAIN ROLES:\n"
        f"  - {SESSION_ROLES['presenter']['character']} (Presenter): explains and teaches the topic\n"
        f"  - {SESSION_ROLES['questioner']['character']} (Questioner): asks bridging questions, occasionally pushes back\n\n"
        f"OPENING STRUCTURE (verify this is intact):\n"
        f"  1. Both hosts greet listeners warmly\n"
        f"  2. Hook question to engage the audience\n"
        f"  3. Transition into the topic\n\n"
        f"Format:\n{SESSION_ROLES['presenter']['label']}: [dialogue]\n"
        f"{SESSION_ROLES['questioner']['label']}: [dialogue]\n\n"
        f"Remove meta-tags, markdown, stage directions. Dialogue only.\n"
        f"- CRITICAL: Do NOT shorten or summarize. Output MUST be at least as long as the input. Add depth where possible.\n"
        + (f"\nCRITICAL: Output MUST be in Japanese (日本語) only. Do NOT switch to Chinese (中文). "
           f"Keep speaker labels exactly as 'Host 1:' and 'Host 2:' — do NOT replace them with Japanese names. "
           f"Avoid Kanji that is only used in Chinese (e.g., use 気 instead of 气, 楽 instead of 乐). "
           if language == 'ja' else '')
        + f"{target_instruction}"
    ),
    expected_output=(
        f"Final Masters-level dialogue about {topic_name}, exactly {target_script} {target_unit_plural}. "
        f"No basic definitions. Teaching style with engaging 3-part opening. "
        f"{target_instruction}"
    ),
    agent=editor_agent,
    context=[script_task]
)

audit_task = Task(
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
    agent=auditor_agent,
    context=[polish_task],
    output_file=str(output_dir / "ACCURACY_AUDIT.md")
)

outline_task = Task(
    description=(
        f"Generate comprehensive show outline (SHOW_OUTLINE.md) for the podcast episode on {topic_name}.\n\n"
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
    agent=producer_agent,
    context=[],
    output_file=str(output_dir / "SHOW_OUTLINE.md")
)

# --- SOT TRANSLATION PIPELINE: Update contexts when translating ---
if translation_task is not None:
    # Recording task writes script directly in target language using the translated SOT
    script_task.context = [translation_task]
    # Show notes use the translated SOT as their reference
    outline_task.context = [translation_task]
    # Polish reads from the target-language script with translated SOT as reference
    polish_task.context = [script_task, translation_task]
    # Accuracy check compares polished script against translated SOT
    audit_task.context = [polish_task, translation_task]

# --- PHASE MARKERS FOR PROGRESS TRACKING ---
PHASE_MARKERS = [
    ("PHASE 0: RESEARCH FRAMING", "Research Framing", 5),
    ("PHASE 1: CLINICAL RESEARCH", "Clinical Research", 10),
    ("PHASE 2: SOURCE VALIDATION", "Source Validation", 50),
    ("PHASE 3: REPORT TRANSLATION", "Report Translation", 55),
    ("PHASE 4: SHOW OUTLINE", "Show Outline", 60),
    ("PHASE 5: SCRIPT WRITING", "Script Writing", 75),
    ("PHASE 6: SCRIPT POLISH", "Script Polish", 90),
    ("PHASE 7: ACCURACY AUDIT", "Accuracy Audit", 95),
    ("PHASE 8: AUDIO PRODUCTION", "Audio Production", 98),
]

# --- TASK METADATA & WORKFLOW PLANNING ---
TASK_METADATA = {
    'framing_task': {
        'name': 'Research Framing',
        'phase': '0',
        'estimated_duration_min': 2,
        'description': 'Defining scope, questions, and evidence criteria',
        'agent': 'Research Framing Specialist',
        'dependencies': [],
        'crew': 1
    },
    'clinical_research': {
        'name': 'Clinical Research (7-Step Pipeline)',
        'phase': '1',
        'estimated_duration_min': 6,
        'description': 'PICO strategy, wide net, screening, extraction, cases, math, GRADE synthesis',
        'agent': 'Dual-Model Pipeline',
        'dependencies': ['framing_task'],
        'crew': 'procedural'
    },
    'source_validation': {
        'name': 'Source Validation',
        'phase': '2',
        'estimated_duration_min': 1,
        'description': 'Batch HEAD requests to validate all cited URLs',
        'agent': 'Automated',
        'dependencies': ['clinical_research'],
        'crew': 'procedural'
    },
    'translation_task': {
        'name': 'Report Translation',
        'phase': '3',
        'estimated_duration_min': 3,
        'description': 'Translate SOT to target language (conditional)',
        'agent': 'Podcast Producer',
        'dependencies': ['source_validation'],
        'crew': 2
    },
    'outline_task': {
        'name': 'Show Outline',
        'phase': '4',
        'estimated_duration_min': 3,
        'description': 'Developing show outline, citations, and narrative arc',
        'agent': 'Podcast Producer',
        'dependencies': ['translation_task'],
        'crew': 3
    },
    'script_task': {
        'name': 'Script Writing',
        'phase': '5',
        'estimated_duration_min': 6,
        'description': 'Script writing and conversation generation',
        'agent': 'Podcast Producer',
        'dependencies': ['outline_task'],
        'crew': 3
    },
    'polish_task': {
        'name': 'Script Polish',
        'phase': '6',
        'estimated_duration_min': 5,
        'description': 'Script polishing for natural verbal delivery',
        'agent': 'Podcast Editor',
        'dependencies': ['script_task'],
        'crew': 3
    },
    'audit_task': {
        'name': 'Accuracy Audit',
        'phase': '7',
        'estimated_duration_min': 3,
        'description': 'Advisory drift detection against Source-of-Truth',
        'agent': 'Scientific Auditor',
        'dependencies': ['polish_task'],
        'crew': 3
    },
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

# ================================================================
# REUSE MODE BRANCHING
# ================================================================
# If --reuse-dir is specified, skip the normal pipeline and run the
# appropriate reuse mode instead. This exits early via sys.exit(0).

if args.reuse_dir:
    reuse_dir = Path(args.reuse_dir)
    print(f"\n{'='*70}")
    print(f"REUSE MODE: Reusing research from {reuse_dir.name}")
    print(f"{'='*70}")

    if args.crew3_only:
        # --- CREW 3 ONLY: Skip research, run podcast production ---
        print(f"\nMode: Crew 3 Only (podcast production)")

        # Create new output dir
        new_output_dir = create_timestamped_output_dir(base_output_dir)

        # Copy research artifacts
        _copy_research_artifacts(reuse_dir, new_output_dir)

        # Load source_of_truth.md content for context injection
        sot_path = new_output_dir / "source_of_truth.md"
        if not sot_path.exists():
            sot_path = new_output_dir / "SOURCE_OF_TRUTH.md"
        if not sot_path.exists():
            print("ERROR: No source_of_truth.md found in reuse directory")
            sys.exit(1)
        sot_content = sot_path.read_text()

        # Inject source_of_truth content into Crew 3 task descriptions
        sot_injection = (
            f"\n\nPREVIOUS RESEARCH (Source of Truth):\n"
            f"{sot_content[:8000]}\n"
            f"--- END PREVIOUS RESEARCH ---\n"
        )
        script_task.description = f"{script_task.description}{sot_injection}"
        outline_task.description = f"{outline_task.description}{sot_injection}"
        audit_task.description = f"{audit_task.description}{sot_injection}"

        # Update output_dir for file outputs
        # Reassign global output_dir so output_file paths work
        import builtins
        # Update task output_file paths to new dir
        for task_obj in [audit_task, outline_task]:
            if hasattr(task_obj, '_original_output_file') or hasattr(task_obj, 'output_file'):
                old_path = getattr(task_obj, 'output_file', '')
                if old_path:
                    filename = Path(old_path).name
                    task_obj.output_file = str(new_output_dir / filename)

        print(f"\nCREW 3: PODCAST PRODUCTION")

        if translation_task is not None:
            print(f"\nPHASE 3: REPORT TRANSLATION")
            crew_2 = Crew(
                agents=[producer_agent],
                tasks=[translation_task],
                verbose=True,
                process='sequential'
            )
            crew_2.kickoff()

        crew_3_tasks = [outline_task, script_task, polish_task, audit_task]

        crew_3 = Crew(
            agents=[producer_agent, editor_agent, auditor_agent],
            tasks=crew_3_tasks,
            verbose=True,
            process='sequential'
        )

        result = crew_3.kickoff()

        # Save markdown outputs
        print("\n--- Saving Outputs ---")
        script_text = polish_task.output.raw if hasattr(polish_task, 'output') and polish_task.output else result.raw
        for label, source, filename in [
            ("Source of Truth (Translated)", translation_task, "source_of_truth.md"),
            ("Show Outline", outline_task, "show_outline.md"),
            ("Script Draft", script_task, "script_draft.md"),
            ("Script Final", polish_task, "script_final.md"),
            ("Accuracy Audit", audit_task, "accuracy_audit.md"),
        ]:
            try:
                if isinstance(source, str):
                    content = source
                elif hasattr(source, 'output') and source.output and hasattr(source.output, 'raw'):
                    content = source.output.raw
                else:
                    content = None
                if content and content.strip():
                    outfile = new_output_dir / filename
                    with open(outfile, 'w') as f:
                        f.write(content)
                    print(f"  Saved {filename} ({len(content)} chars)")
            except Exception as e:
                print(f"  Warning: Could not save {filename}: {e}")

        # Generate PDF for accuracy audit
        try:
            acc_content = audit_task.output.raw if hasattr(audit_task, 'output') and audit_task.output else ""
            if acc_content:
                create_pdf("Accuracy Audit", acc_content, "accuracy_audit.pdf")
        except Exception:
            pass

        # TTS + BGM
        print("\n--- Generating Multi-Voice Podcast Audio (Kokoro TTS) ---")
        cleaned_script = clean_script_for_tts(script_text)
        script_file = new_output_dir / "script.txt"
        with open(script_file, 'w') as f:
            f.write(script_text)

        audio_output_path = new_output_dir / "audio.wav"
        audio_file = generate_audio_from_script(cleaned_script, str(audio_output_path), lang_code=language_config['tts_code'])
        if audio_file:
            audio_file = Path(audio_file)
            print(f"Audio generation complete: {audio_file}")
            print(f"Starting BGM Merging Phase...")
            try:
                mastered = post_process_audio(str(audio_file), bgm_target="Interesting BGM.wav")
                if mastered and os.path.exists(mastered) and mastered != str(audio_file):
                    audio_file = Path(mastered)
                    print(f"✓ BGM Merging Complete: {audio_file}")
            except Exception as e:
                print(f"⚠ BGM merging warning: {e}")

            # Duration check
            try:
                import wave
                with wave.open(str(audio_file), 'r') as wav:
                    frames = wav.getnframes()
                    rate = wav.getframerate()
                    duration_seconds = frames / float(rate)
                    duration_minutes = duration_seconds / 60
                print(f"SUCCESS: Audio duration {duration_minutes:.2f} minutes")
            except Exception:
                pass

        # Session metadata
        session_metadata = (
            f"PODCAST SESSION METADATA (REUSE: crew3_only)\n{'='*60}\n\n"
            f"Topic: {topic_name}\n"
            f"Language: {language_config['name']} ({language})\n"
            f"Reused from: {reuse_dir}\n"
        )
        with open(new_output_dir / "session_metadata.txt", 'w') as f:
            f.write(session_metadata)

        print(f"\n{'='*70}")
        print("REUSE_COMPLETE: CREW3_ONLY")
        print(f"{'='*70}")
        sys.exit(0)

    elif args.check_supplemental:
        # --- CHECK SUPPLEMENTAL: LLM decides if supplement needed ---
        print(f"\nMode: Check Supplemental")

        result = check_supplemental_needed(topic_name, reuse_dir)
        print(f"  Needs supplement: {result['needs_supplement']}")
        print(f"  Reason: {result['reason']}")

        if not result['needs_supplement']:
            # Full reuse — copy everything
            new_output_dir = create_timestamped_output_dir(base_output_dir)
            _copy_all_artifacts(reuse_dir, new_output_dir)

            # Session metadata
            session_metadata = (
                f"PODCAST SESSION METADATA (REUSE: full_reuse)\n{'='*60}\n\n"
                f"Topic: {topic_name}\n"
                f"Language: {language_config['name']} ({language})\n"
                f"Reused from: {reuse_dir}\n"
                f"Reason: {result['reason']}\n"
            )
            with open(new_output_dir / "session_metadata.txt", 'w') as f:
                f.write(session_metadata)

            print(f"\n{'='*70}")
            print("REUSE_COMPLETE: NO_CHANGES")
            print(f"{'='*70}")
            sys.exit(0)

        else:
            # Supplemental research needed
            print(f"\nSUPPLEMENTAL RESEARCH needed: {result['reason']}")
            print(f"  Running {len(result['queries'])} supplemental searches...")

            new_output_dir = create_timestamped_output_dir(base_output_dir)
            _copy_research_artifacts(reuse_dir, new_output_dir)

            # Run supplemental research with BraveSearch
            supp_text = ""
            if result['queries']:
                brave_api_key = os.getenv("BRAVE_API_KEY", "")
                if brave_api_key:
                    supp_parts = []
                    for q in result['queries']:
                        query_str = q.get("query", q) if isinstance(q, dict) else str(q)
                        try:
                            resp = httpx.get(
                                "https://api.search.brave.com/res/v1/web/search",
                                headers={"Accept": "application/json", "X-Subscription-Token": brave_api_key},
                                params={"q": query_str, "count": 5},
                                timeout=15.0,
                            )
                            if resp.status_code == 200:
                                results = resp.json().get("web", {}).get("results", [])
                                for r in results:
                                    supp_parts.append(f"### {r.get('title', 'N/A')}\nURL: {r.get('url', '')}\n{r.get('description', '')}")
                        except Exception as e:
                            print(f"  Supplemental search failed for '{query_str}': {e}")
                    supp_text = "\n\n".join(supp_parts)
                    if supp_text:
                        print(f"  Found supplemental evidence ({len(supp_parts)} results)")
                    else:
                        print("  No supplemental results found")
                else:
                    print("  No BRAVE_API_KEY set, skipping supplemental search")

            # Load existing source_of_truth for context
            sot_path = new_output_dir / "source_of_truth.md"
            if not sot_path.exists():
                sot_path = new_output_dir / "SOURCE_OF_TRUTH.md"
            sot_content = sot_path.read_text() if sot_path.exists() else ""

            # Append supplemental findings to SOT
            if supp_text:
                sot_content += (
                    f"\n\n## Supplemental Research Findings\n\n"
                    f"{supp_text}\n"
                )
                with open(new_output_dir / "source_of_truth.md", 'w') as f:
                    f.write(sot_content)
                print(f"  Updated source_of_truth.md with supplemental findings ({len(sot_content)} chars)")

            # Now run Crew 3 with updated research
            sot_injection = (
                f"\n\nSOURCE OF TRUTH (from previous research + supplemental):\n"
                f"{sot_content[:8000]}\n"
                f"--- END SOURCE OF TRUTH ---\n"
            )
            script_task.description = f"{script_task.description}{sot_injection}"
            outline_task.description = f"{outline_task.description}{sot_injection}"
            audit_task.description = f"{audit_task.description}{sot_injection}"

            # Update output_file paths
            for task_obj in [audit_task, outline_task]:
                old_path = getattr(task_obj, 'output_file', '')
                if old_path:
                    filename = Path(old_path).name
                    task_obj.output_file = str(new_output_dir / filename)

            print(f"\nCREW 3: PODCAST PRODUCTION")

            if translation_task is not None:
                print(f"\nPHASE 3: REPORT TRANSLATION")
                crew_2 = Crew(
                    agents=[producer_agent],
                    tasks=[translation_task],
                    verbose=True,
                    process='sequential'
                )
                crew_2.kickoff()

            crew_3_tasks = [outline_task, script_task, polish_task, audit_task]

            crew_3 = Crew(
                agents=[producer_agent, editor_agent, auditor_agent],
                tasks=crew_3_tasks,
                verbose=True,
                process='sequential'
            )

            result = crew_3.kickoff()

            # Save outputs
            print("\n--- Saving Outputs ---")
            script_text = polish_task.output.raw if hasattr(polish_task, 'output') and polish_task.output else result.raw
            for label, source, filename in [
                ("Source of Truth (Translated)", translation_task, "source_of_truth.md"),
                ("Show Outline", outline_task, "show_outline.md"),
                ("Script Draft", script_task, "script_draft.md"),
                ("Script Final", polish_task, "script_final.md"),
                ("Accuracy Audit", audit_task, "accuracy_audit.md"),
            ]:
                try:
                    if isinstance(source, str):
                        content = source
                    elif hasattr(source, 'output') and source.output and hasattr(source.output, 'raw'):
                        content = source.output.raw
                    else:
                        content = None
                    if content and content.strip():
                        outfile = new_output_dir / filename
                        with open(outfile, 'w') as f:
                            f.write(content)
                        print(f"  Saved {filename} ({len(content)} chars)")
                except Exception as e:
                    print(f"  Warning: Could not save {filename}: {e}")

            # TTS + BGM
            print("\n--- Generating Multi-Voice Podcast Audio (Kokoro TTS) ---")
            cleaned_script = clean_script_for_tts(script_text)
            script_file = new_output_dir / "script.txt"
            with open(script_file, 'w') as f:
                f.write(script_text)

            audio_output_path = new_output_dir / "audio.wav"
            audio_file = generate_audio_from_script(cleaned_script, str(audio_output_path), lang_code=language_config['tts_code'])
            if audio_file:
                audio_file = Path(audio_file)
                print(f"Audio generation complete: {audio_file}")
                print(f"Starting BGM Merging Phase...")
                try:
                    mastered = post_process_audio(str(audio_file), bgm_target="Interesting BGM.wav")
                    if mastered and os.path.exists(mastered) and mastered != str(audio_file):
                        audio_file = Path(mastered)
                        print(f"✓ BGM Merging Complete: {audio_file}")
                except Exception as e:
                    print(f"⚠ BGM merging warning: {e}")

                try:
                    import wave
                    with wave.open(str(audio_file), 'r') as wav:
                        frames = wav.getnframes()
                        rate = wav.getframerate()
                        duration_seconds = frames / float(rate)
                        duration_minutes = duration_seconds / 60
                    print(f"SUCCESS: Audio duration {duration_minutes:.2f} minutes")
                except Exception:
                    pass

            # Session metadata
            session_metadata = (
                f"PODCAST SESSION METADATA (REUSE: supplemental)\n{'='*60}\n\n"
                f"Topic: {topic_name}\n"
                f"Language: {language_config['name']} ({language})\n"
                f"Reused from: {reuse_dir}\n"
                f"Supplemental reason: {result['reason'] if isinstance(result, dict) else 'N/A'}\n"
            )
            with open(new_output_dir / "session_metadata.txt", 'w') as f:
                f.write(session_metadata)

            print(f"\n{'='*70}")
            print("REUSE_COMPLETE: SUPPLEMENTAL")
            print(f"{'='*70}")
            sys.exit(0)

# ================================================================
# NORMAL PIPELINE (no --reuse-dir)
# ================================================================

# --- EXECUTION (Streamlined Pipeline) ---
# Display workflow plan before execution
display_workflow_plan()

# Initialize progress tracker
progress_tracker = ProgressTracker(TASK_METADATA)
progress_tracker.start_workflow()

# Combined task list for tracking
all_task_list = [
    framing_task,
    outline_task,
    script_task,
    polish_task,
    audit_task,
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
# PHASE 0: Research Framing
# ================================================================
print(f"\n{'='*70}")
print(f"PHASE 0: RESEARCH FRAMING")
print(f"{'='*70}")

crew_1 = Crew(
    agents=[framing_agent],
    tasks=[framing_task],
    verbose=True,
    process='sequential'
)

try:
    crew_1_result = crew_1.kickoff()
    framing_output = framing_task.output.raw if hasattr(framing_task, 'output') and framing_task.output else ""
    print(f"✓ Phase 0 complete: Research framing generated ({len(framing_output)} chars)")
except Exception as e:
    print(f"⚠ Phase 0 (Research Framing) failed: {e}")
    print("Continuing without framing context...")
    framing_output = ""

# ================================================================
# PHASE 1: CLINICAL RESEARCH (7-Step Pipeline)
# ================================================================
print(f"\n{'='*70}")
print(f"PHASE 1: CLINICAL RESEARCH")
print(f"{'='*70}")

brave_key = os.getenv("BRAVE_API_KEY", "")

# Check if fast model (Phi-4 Mini via Ollama) is available
fast_model_available = False
try:
    _resp = httpx.get("http://localhost:11434/v1/models", timeout=3)
    if _resp.status_code == 200:
        _models = [m.get("id", "") for m in _resp.json().get("data", [])]
        fast_model_available = any(any(k in m.lower() for k in ["phi", "llama", "qwen", "mistral"]) for m in _models)
        if fast_model_available:
            print(f"✓ Fast model detected on Ollama (available: {_models})")
        else:
            print(f"⚠ Ollama running but no suitable fast model found (phi/llama/qwen). Available: {_models}")
except Exception:
    print("⚠ Fast model not available, using smart-only mode")

sot_content = ""  # Will hold the synthesized Source-of-Truth
aff_candidates = 0
neg_candidates = 0
evidence_quality = "sufficient"

try:
    deep_reports = asyncio.run(run_deep_research(
        topic=topic_name,
        brave_api_key=brave_key,
        results_per_query=15,
        fast_model_available=fast_model_available,
        framing_context=framing_output,
        output_dir=str(output_dir)
    ))

    # C5: Gate — abort if affirmative track found zero candidates
    for fname, varname in [("screening_results_aff.json", "aff"), ("screening_results_neg.json", "neg")]:
        p = output_dir / fname
        if p.exists():
            try:
                val = json.loads(p.read_text()).get("total_candidates", 0)
                if varname == "aff":
                    aff_candidates = val
                else:
                    neg_candidates = val
            except Exception:
                pass
    if aff_candidates == 0:
        _write_insufficient_evidence_report(topic_name, 0, neg_candidates, output_dir)
        raise InsufficientEvidenceError(
            f"Affirmative track: 0 candidates for '{topic_name}'. "
            f"Adversarial found {neg_candidates}. "
            "See insufficient_evidence_report.md for suggested rephrasing."
        )

    # C6: Evidence quality flag
    if 0 < aff_candidates < EVIDENCE_LIMITED_THRESHOLD:
        evidence_quality = "limited"

    # Save all reports (lead, counter, audit)
    REPORT_FILENAMES = {"lead": "affirmative_case.md", "counter": "falsification_case.md", "audit": "grade_synthesis.md"}
    for role_name, report in deep_reports.items():
        report_file = output_dir / REPORT_FILENAMES.get(role_name, f"{role_name}.md")
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
    sources_file = output_dir / "research_sources.json"
    with open(sources_file, 'w') as f:
        json.dump(sources_json, f, indent=2, ensure_ascii=False)
    print(f"✓ Research library saved: {sources_file} "
          f"(lead={len(sources_json['lead'])}, counter={len(sources_json['counter'])} sources)")

    deep_audit_report = deep_reports["audit"]
    lead_report = deep_reports["lead"]
    counter_report = deep_reports["counter"]

    # ================================================================
    # Build Source-of-Truth from deep research outputs
    # ================================================================
    def _extract_conclusion_status(grade_report: str) -> tuple:
        """Extract GRADE level, conclusion status, and executive summary."""
        m = re.search(
            r'Final\s+(?:GRADE|Grade)[:\s]*\*{0,2}(High|Moderate|Low|Very\s+Low)\*{0,2}',
            grade_report, re.IGNORECASE)
        grade = m.group(1).strip() if m else "Not Determined"

        status_map = {
            "High": "Scientifically Supported",
            "Moderate": "Partially Supported — Further Research Recommended",
            "Low": "Insufficient Evidence — More Research Needed",
            "Very Low": "Not Supported by Current Evidence",
        }
        status = status_map.get(grade, "Under Evaluation")

        m2 = re.search(r'Executive\s+Summary[#\s:]*\n+(.+?)(?:\n\n|\n#)',
                       grade_report, re.DOTALL)
        summary = m2.group(1).strip() if m2 else ""

        return grade, status, summary

    grade_level, conclusion_status, exec_summary = _extract_conclusion_status(
        deep_audit_report.report)

    sot_content = f"# Source of Truth: {topic_name}\n\n"
    sot_content += f"## Research Conclusion\n\n"
    sot_content += f"**Evidence Quality (GRADE): {grade_level}**\n\n"
    sot_content += f"**Conclusion: {conclusion_status}**\n\n"
    if exec_summary:
        sot_content += f"{exec_summary}\n\n"
    sot_content += f"## Evidence Quality Assessment (GRADE)\n\n{deep_audit_report.report}\n\n"
    math_file = output_dir / "clinical_math.md"
    if math_file.exists():
        sot_content += f"## Clinical Impact (Deterministic Math)\n\n{math_file.read_text()}\n\n"
    sot_content += (
        "---\n\n"
        "*Detailed case reports available in affirmative_case.md and falsification_case.md*\n"
    )

    # C6: Prepend evidence quality banner if limited
    if evidence_quality == "limited":
        sot_content = (
            "## ⚠ Evidence Quality Notice\n\n"
            f"The affirmative research track retrieved only **{aff_candidates} candidate studies** "
            f"(threshold: {EVIDENCE_LIMITED_THRESHOLD}). "
            "The following synthesis is based on limited direct evidence. "
            "Claims should be interpreted cautiously.\n\n"
        ) + sot_content

    # Save as source_of_truth.md
    sot_file = output_dir / "source_of_truth.md"
    with open(sot_file, 'w') as f:
        f.write(sot_content)
    print(f"✓ Source of Truth generated from deep research ({len(sot_content)} chars)")

    # Summarize for injection into Crew 3 task descriptions
    print("Summarizing Source-of-Truth with fast model...")
    sot_summary = summarize_report_with_fast_model(sot_content, "sot", topic_name)

except InsufficientEvidenceError:
    raise
except Exception as e:
    print(f"⚠ Deep research pre-scan failed: {e}")
    print("Continuing without deep research...")
    deep_reports = None
    sot_summary = ""

# ================================================================
# PHASE 2: SOURCE VALIDATION (batch, parallel)
# ================================================================
print(f"\n{'='*70}")
print(f"PHASE 2: SOURCE VALIDATION")
print(f"{'='*70}")

from link_validator import validate_multiple_urls_parallel

all_urls = set()
url_pattern = re.compile(r'https?://[^\s\)\]\"\'<>]+')

# Collect URLs from source library
sources_file = output_dir / "research_sources.json"
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
else:
    print("  No URLs found to validate")

# ================================================================
# Inject SOT into Crew 3 task descriptions
# ================================================================
if sot_summary:
    sot_injection = (
        f"\n\nSOURCE OF TRUTH (from deep research pipeline):\n"
        f"Use this as your authoritative reference for all evidence and claims.\n\n"
        f"{sot_summary}\n\n"
        f"For detailed sources, use ListResearchSources('lead') and ListResearchSources('counter').\n"
        f"--- END SOURCE OF TRUTH ---\n"
    )
    script_task.description += sot_injection
    outline_task.description += sot_injection
    audit_task.description += sot_injection

# For translation: inject full SOT (not summary) since translation needs complete text
if translation_task is not None and sot_content:
    translation_task.description += f"\n\nSOURCE OF TRUTH TO TRANSLATE:\n{sot_content}\n--- END ---\n"

# C6: Check GRADE file for LOW quality (may upgrade evidence_quality even if aff_candidates >= threshold)
grade_file = output_dir / "grade_synthesis.md"
if grade_file.exists():
    try:
        if "GRADE: LOW" in grade_file.read_text():
            evidence_quality = "limited"
    except Exception:
        pass

# C6: Inject evidence disclosure into tasks if limited
if evidence_quality == "limited":
    script_task.description += (
        "\n\nEVIDENCE QUALITY NOTE — READ CAREFULLY:\n"
        "The systematic review found limited direct scientific evidence for this question.\n"
        "Your script MUST:\n"
        "1. Acknowledge this in the OPENING: "
        "   'While direct studies on this are limited, related research gives us clues...'\n"
        "2. In each body segment, distinguish: "
        "(a) what limited direct evidence shows, "
        "(b) what related evidence suggests, "
        "(c) what remains unknown.\n"
        "3. Frame CLOSING takeaways as 'based on current evidence' — not 'proven'.\n"
        "4. Do NOT invent citations. If few studies exist, say so in the dialogue.\n"
        "Example dialogue:\n"
        "  Presenter: 'Direct studies on this exact pattern are surprisingly rare. "
        "But here's what the broader science on meal timing tells us...'\n"
        "  Questioner: 'So we're working with partial evidence here. "
        "What do we actually know for certain?'\n"
    )
    outline_task.description += (
        "\n\nEVIDENCE NOTE: Research was limited. "
        "Mark citations with [LIMITED EVIDENCE] where the research base is sparse. "
        "Add a 'Research Limitations' section to the outline."
    )

# ================================================================
# CREW 3: Podcast Production
# ================================================================
print(f"\n{'='*70}")
print(f"CREW 3: PODCAST PRODUCTION")
print(f"{'='*70}")

translated_sot = None  # set below if translation runs
if translation_task is not None:
    print(f"\nPHASE 3: REPORT TRANSLATION")
    print(f"Translating Source-of-Truth to {language_config['name']}")
    crew_2 = Crew(
        agents=[producer_agent],
        tasks=[translation_task],
        verbose=True,
        process='sequential'
    )
    crew_2.kickoff()

    # Save translated SOT to disk
    if hasattr(translation_task, 'output') and translation_task.output and \
            hasattr(translation_task.output, 'raw') and translation_task.output.raw:
        translated_sot = translation_task.output.raw
        lang_suffix = language  # e.g. "ja"
        sot_translated_file = output_dir / f"source_of_truth_{lang_suffix}.md"
        with open(sot_translated_file, 'w', encoding='utf-8') as f:
            f.write(translated_sot)
        print(f"✓ Translated SOT saved ({len(translated_sot)} chars) → {sot_translated_file.name}")
    else:
        print(f"  Warning: Translation task produced no output — translated SOT not saved")
        translated_sot = None

crew_3_tasks = [outline_task, script_task, polish_task, audit_task]

crew_3 = Crew(
    agents=[producer_agent, editor_agent, auditor_agent],
    tasks=crew_3_tasks,
    verbose=True,
    process='sequential'
)

# Start background monitor for crew 3
monitor = CrewMonitor(all_task_list, progress_tracker)
monitor.start()

try:
    result = crew_3.kickoff()
except Exception as e:
    print(f"\n{'='*70}")
    print("CREW 3 FAILED")
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
    ("Source of Truth", sot_content, "source_of_truth.pdf"),
    ("Accuracy Audit", audit_task, "accuracy_audit.pdf"),
    # Translated SOT PDF — only generated when translation ran and produced output
    *([(f"Source of Truth ({language_config['name']})",
        translated_sot,
        f"source_of_truth_{language}.pdf")]
      if translation_task is not None and translated_sot else []),
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
    # source_of_truth.md already saved from deep research outputs
    ("Accuracy Audit", audit_task, "accuracy_audit.md"),
    ("Show Outline", outline_task, "show_outline.md"),
    ("Script Draft", script_task, "script_draft.md"),
    ("Script Final", polish_task, "script_final.md"),
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

# --- SESSION METADATA ---
print("\n--- Documenting Session Metadata ---")
session_metadata = (
    f"PODCAST SESSION METADATA\n{'='*60}\n\n"
    f"Topic: {topic_name}\n\n"
    f"Language: {language_config['name']} ({language})\n\n"
    f"Character Assignments:\n"
    f"  {SESSION_ROLES['presenter']['character']}: Presenter ({SESSION_ROLES['presenter']['personality']})\n"
    f"  {SESSION_ROLES['questioner']['character']}: Questioner ({SESSION_ROLES['questioner']['personality']})\n"
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

# def generate_audio_metavoice(dialogue_segments: list, output_filename: str = "audio.wav"):
#     """DEPRECATED: Use audio_engine.generate_audio_from_script() instead"""
#     pass

# def generate_audio_gtts_fallback(dialogue_segments: list, output_filename: str = "podcast_final_audio.mp3"):
#     """DEPRECATED: No longer needed with Kokoro TTS"""
#     pass

# Generate audio with Kokoro TTS
print("\n--- Generating Multi-Voice Podcast Audio (Kokoro TTS) ---")

# Check script length before generation
# Get the polished script from polish_task (not the last crew result, which is audit_task)
script_text = polish_task.output.raw if hasattr(polish_task, 'output') and polish_task.output else result.raw

# Language-aware script length measurement — rates and targets derived from SUPPORTED_LANGUAGES
speech_rate  = language_config['speech_rate']
length_unit  = language_config['length_unit']
if length_unit == 'chars':
    script_length = len(re.sub(r'[\s\n\r\t　：:「」、。・（）\-\—\*#]', '', script_text))
else:
    content_only = re.sub(r'^[A-Za-z0-9_ ]+:\s*', '', script_text, flags=re.MULTILINE)
    script_length = len(content_only.split())
    length_unit = "words (net)"
estimated_duration_min = script_length / speech_rate
target_length = target_length_int
target_low    = int(target_length * (1 - SCRIPT_TOLERANCE))
target_high   = int(target_length * (1 + SCRIPT_TOLERANCE))

# Expansion retry — triggered if script is below target for any language
if script_length < target_low:
    print(f"⚠ WARNING: Script too short ({script_length} {length_unit} < {target_low} target) — running expansion pass")
    expansion_task = Task(
        description=(
            f"The following podcast script is too short ({script_length} {length_unit}, target: {target_length}).\n"
            f"{target_instruction}\n"
            f"Expand it to reach {target_length} {length_unit} by:\n"
            f"  1. For each topic segment, add deeper explanation of the scientific mechanism\n"
            f"  2. Add one more real-world example or analogy per segment\n"
            f"  3. Add more back-and-forth host dialogue — questioner should ask 'Why?' and 'What does that mean for listeners?'\n"
            f"  4. Never cut or reorder existing content — only add.\n"
            f"  5. Respond entirely in the same language as the input script.\n\n"
            f"SCRIPT TO EXPAND:\n{script_text}"
        ),
        expected_output=(
            f"Expanded podcast script of at least {target_length} {length_unit}. "
            f"Respond entirely in the same language as the input script. "
            f"Same speaker label: dialogue format as input. No summaries, no truncation."
        ),
        agent=producer_agent,
    )
    from crewai import Crew as _Crew
    expansion_result = _Crew(agents=[producer_agent], tasks=[expansion_task], verbose=False).kickoff()
    expanded = expansion_result.raw if hasattr(expansion_result, 'raw') else str(expansion_result)
    if length_unit == 'chars' or length_unit.startswith('char'):
        expanded_length = len(re.sub(r'[\s\n\r\t　：:「」、。・（）\-\—\*#]', '', expanded))
    else:
        expanded_length = len(expanded.split())
    if expanded_length > script_length:
        print(f"Expansion pass: {script_length} → {expanded_length} {length_unit}")
        script_text = expanded
        script_length = expanded_length
        estimated_duration_min = script_length / speech_rate
    else:
        print(f"⚠ WARNING: Expansion pass did not improve length — using original")

print(f"\n{'='*60}")
print(f"DURATION CHECK")
print(f"{'='*60}")
print(f"Script length: {script_length} {length_unit}")
print(f"Estimated duration: {estimated_duration_min:.1f} minutes")
print(f"Target: 30 minutes ({target_length} {length_unit})")

if script_length < target_low:
    print(f"⚠ WARNING: Script is SHORT ({script_length} {length_unit} < {target_length} target)")
    print(f"  Estimated {estimated_duration_min:.1f} min")
elif script_length > target_high:
    print(f"⚠ WARNING: Script is LONG ({script_length} {length_unit} > {target_length} target)")
    print(f"  Estimated {estimated_duration_min:.1f} min")
else:
    print(f"✓ Script length GOOD ({script_length} {length_unit})")
    print(f"  Estimated {estimated_duration_min:.1f} min")
print(f"{'='*60}\n")

# Clean script and generate audio with Kokoro
cleaned_script = clean_script_for_tts(script_text)

# Save podcast script for review
script_file = output_dir / "script.txt"
with open(script_file, 'w') as f:
    f.write(script_text)
print(f"Podcast script saved: {script_file} ({script_length} {length_unit})")

output_path = output_dir / "audio.wav"

audio_file = None

audio_file = None
try:
    print(f"Starting audio generation with script length: {len(cleaned_script)} chars")
    audio_file = generate_audio_from_script(cleaned_script, str(output_path), lang_code=language_config['tts_code'])
    if audio_file:
        audio_file = Path(audio_file)
        print(f"Audio generation complete: {audio_file}")
except Exception as e:
    print(f"✗ ERROR: Kokoro TTS failed with exception: {e}")
    import traceback
    traceback.print_exc()
    print("  Ensure Kokoro is installed: pip install kokoro>=0.9")
    audio_file = None

# --- BGM MERGING ---
if audio_file and audio_file.exists():
    print(f"\n{PHASE_MARKERS[-1][0]} {PHASE_MARKERS[-1][1]} ({PHASE_MARKERS[-1][2]}%)\n")
    print(f"Starting BGM Merging Phase...")
    
    try:
        # Default to "Interesting BGM.wav" as per user request
        # This function now checks the 'Podcast BGM' library first
        mastered = post_process_audio(str(audio_file), bgm_target="Interesting BGM.wav")
        if mastered and os.path.exists(mastered) and mastered != str(audio_file):
            audio_file = Path(mastered)
            print(f"✓ BGM Merging Complete: {audio_file}")
            print(f"[SOURCE] {audio_file}") # Mark final file as source
        else:
             print(f"⚠ BGM Merging skipped or failed (original audio preserved)")
    except Exception as e:
        print(f"⚠ Warning: BGM merging process encountered an error: {e}")


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
        print(f"Target range: 27-33 minutes")

        if duration_minutes < 27.0:
            print(f"✗ FAILED: Audio is TOO SHORT ({duration_minutes:.2f} min < {target_low/150 if language!='ja' else target_low/500:.1f} min)")
            print(f"  ACTION: Re-run with longer script")
        elif duration_minutes > (target_high/150 if language!='ja' else target_high/500) * 1.2: # Allow some buffer
            print(f"✗ FAILED: Audio is TOO LONG ({duration_minutes:.2f} min > {target_high/150 if language!='ja' else target_high/500:.1f} min)")
            print(f"  ACTION: Re-run with shorter script")
        else:
            print(f"✓ SUCCESS: Audio duration within acceptable range")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"Warning: Could not verify audio duration: {e}")