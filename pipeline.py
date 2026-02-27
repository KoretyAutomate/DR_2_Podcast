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
load_dotenv()  # Load .env BEFORE any module-level imports that read env vars (e.g. audio_engine)
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

# --- HOST CONFIGURATION ---
HOSTS = {
    "Host 1": {"gender": "male"},
    "Host 2": {"gender": "female"},
}

# Role-based personality (assigned by role, not character)
ROLE_PERSONALITIES = {
    "presenter": (
        "Enthusiastic science communicator who gets genuinely excited about breakthroughs, "
        "uses vivid metaphors, occasionally laughs at surprising findings, and makes complex "
        "mechanisms feel like detective stories"
    ),
    "questioner": (
        "Curious and sharp interviewer who reacts with genuine surprise, playful skepticism, "
        "and humor — calls out when something sounds too good to be true, shares personal "
        "anecdotes, and advocates for the listener"
    ),
}

# --- ROLE ASSIGNMENT (Dynamic per session) ---
def assign_roles() -> dict:
    """
    Assign Host 1 / Host 2 to presenter/questioner roles.
    Respects PODCAST_HOSTS env var: host1_leads, host2_leads, or random.
    """
    host_labels = list(HOSTS.keys())
    host_config = os.getenv("PODCAST_HOSTS", "random").lower()

    if host_config == "host1_leads":
        presenter_label, questioner_label = "Host 1", "Host 2"
    elif host_config == "host2_leads":
        presenter_label, questioner_label = "Host 2", "Host 1"
    else:
        random.shuffle(host_labels)
        presenter_label, questioner_label = host_labels[0], host_labels[1]

    role_assignment = {
        "presenter": {
            "label": presenter_label,
            "stance": "teaching",
            "personality": ROLE_PERSONALITIES["presenter"],
        },
        "questioner": {
            "label": questioner_label,
            "stance": "curious",
            "personality": ROLE_PERSONALITIES["questioner"],
        },
    }

    print(f"\n{'='*60}")
    print(f"SESSION ROLE ASSIGNMENT ({host_config}):")
    print(f"  Presenter: {presenter_label} ({HOSTS[presenter_label]['gender']})")
    print(f"  Questioner: {questioner_label} ({HOSTS[questioner_label]['gender']})")
    print(f"{'='*60}\n")

    return role_assignment

SESSION_ROLES = assign_roles()


def _truncate_at_boundary(text: str, max_len: int) -> str:
    """Truncate text at the last paragraph boundary before max_len."""
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rfind('\n\n')
    if cut > 0:
        return text[:cut]
    cut = text[:max_len].rfind('\n')
    if cut > 0:
        return text[:cut]
    return text[:max_len]


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

    sot_content = _truncate_at_boundary(sot_path.read_text(), 8000)

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
            f"{SMART_BASE_URL}/chat/completions",
            json={
                "model": SMART_MODEL,
                "messages": [{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1024,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip <think> blocks (Qwen3 safety net)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

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

# --- CHANNEL INTRO (fixed text, spoken every episode) ---
channel_intro = os.getenv("PODCAST_CHANNEL_INTRO", "").strip()
core_target = os.getenv("PODCAST_CORE_TARGET", "").strip()
channel_mission = os.getenv("PODCAST_CHANNEL_MISSION", "").strip()

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
# All model config comes from .env (loaded by dotenv above)
SMART_MODEL = os.environ["MODEL_NAME"]
SMART_BASE_URL = os.environ["LLM_BASE_URL"]
MID_MODEL = os.environ.get("MID_MODEL_NAME", "qwen2.5:7b")
MID_BASE_URL = os.environ.get("MID_LLM_BASE_URL", os.environ.get("FAST_LLM_BASE_URL", "http://localhost:11434/v1"))

def get_final_model_string():
    model = SMART_MODEL
    base_url = SMART_BASE_URL
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

# LLM Configuration for Qwen3-32B-AWQ (32k context window)
dgx_llm_strict = LLM(
    model=final_model_string,
    base_url=SMART_BASE_URL,
    api_key="NA",  # vLLM uses "NA"
    provider="openai",
    timeout=600,
    temperature=0.1,  # Strict mode for Researcher/Auditor
    max_tokens=8000,  # Researchers/auditors produce short structured outputs; leaves 24k for input
    stop=["<|im_end|>", "<|endoftext|>"],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

dgx_llm_creative = LLM(
    model=final_model_string,
    base_url=SMART_BASE_URL,
    api_key="NA",  # vLLM uses "NA"
    provider="openai",
    timeout=600,
    temperature=0.7,  # Creative mode for Producer/Personality
    max_tokens=16000,  # Increased for expansion; scriptwriter needs room for 4,500-word scripts
    frequency_penalty=0.15,  # Prevent repetition loops while allowing long creative output
    stop=["<|im_end|>", "<|endoftext|>"],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
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
        client = OpenAI(base_url=os.environ["FAST_LLM_BASE_URL"], api_key="ollama")
        response = client.chat.completions.create(
            model=os.environ["FAST_MODEL_NAME"],
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


def _estimate_task_tokens(task, translation_task_obj=None, language='en'):
    """Rough estimate of input tokens for a CrewAI task (description + context chain outputs).

    Japanese/Chinese: ~2 chars/token. Other languages: ~4 chars/token.
    Adds 2000-token buffer for agent system prompt overhead.
    """
    chars_per_tok = 2 if language in ('ja', 'zh') else 4
    total_chars = len(task.description or '')
    for ctx_task in (task.context or []):
        raw = getattr(getattr(ctx_task, 'output', None), 'raw', '') or ''
        total_chars += len(raw)
    return total_chars // chars_per_tok + 2000


def _build_sot_injection_for_stage(stage, sot_file, translated_sot_file,
                                    sot_summary, translated_sot_summary,
                                    grade_numbers_text, language_config):
    """Return SOT injection text for a context-degradation stage.

    Stage 1: Full target-language fast-model summary + file path    (~3K tokens)
    Stage 2: IMRaD Abstract + GRADE section from file + path         (~1.5K tokens)
    Stage 3: File path + pre-extracted GRADE/clinical numbers only   (~300 tokens)
    """
    target_file = str(translated_sot_file or sot_file or '')
    lang_name = language_config.get('name', 'target language') if isinstance(language_config, dict) else str(language_config)

    if stage == 1:
        summary = (translated_sot_summary or sot_summary or '')
        return (
            f"\n\nSOURCE OF TRUTH SUMMARY ({lang_name}):\n"
            f"Use this as your primary research reference.\n\n"
            f"{summary}\n\n"
            f"Full research file: {target_file}\n"
            f"--- END SOT ---\n"
        )
    elif stage == 2:
        abstract_text = ''
        grade_text = ''
        if target_file and Path(target_file).exists():
            try:
                raw = Path(target_file).read_text(encoding='utf-8')
                m = re.search(r'(?:## 1\.|##\s*Abstract)(.*?)(?=\n## |\Z)', raw, re.DOTALL | re.IGNORECASE)
                if m:
                    abstract_text = m.group(1).strip()[:2000]
                # Try GRADE (clinical) or Evidence Quality Synthesis (social science)
                m = re.search(r'(?:### 4\.3|###\s*GRADE|##\s*GRADE|###\s*Evidence\s+Quality\s+Synthesis)(.*?)(?=\n### |\n## |\Z)', raw, re.DOTALL | re.IGNORECASE)
                if m:
                    grade_text = m.group(1).strip()[:1000]
            except Exception:
                pass
        evidence_label = "EVIDENCE ASSESSMENT"
        return (
            f"\n\n[SOT Stage 2 — reduced for context budget]\n"
            f"RESEARCH ABSTRACT:\n{abstract_text or '(not available)'}\n\n"
            f"{evidence_label}:\n{grade_text or '(not available)'}\n\n"
            f"Full research file: {target_file}\n"
            f"--- END SOT ---\n"
        )
    else:  # stage 3
        return (
            f"\n\n[SOT Stage 3 — minimal context; use research file for details]\n"
            f"Full research file: {target_file}\n"
            f"{grade_numbers_text or ''}\n"
            f"--- END SOT ---\n"
        )


_SOT_BLOCK_RE = re.compile(
    r'\n\nSOURCE OF TRUTH SUMMARY[^\n]*\n.*?--- END SOT ---\n'
    r'|\n\n\[SOT Stage \d[^\n]*\n.*?--- END SOT ---\n',
    re.DOTALL
)


def _crew_kickoff_guarded(crew_factory_fn, task, translation_task_obj, language,
                           sot_file, translated_sot_file, sot_summary, translated_sot_summary,
                           grade_numbers_text, language_config, crew_name,
                           ctx_window=32768, max_tokens=16000):
    """Run a crew kickoff with pre-emptive 3-stage context-budget check.

    Before kickoff, estimates input tokens. If over budget, degrades the SOT
    injection to the next stage (summary → abstract+GRADE+path → path only).
    Selects the lowest stage that fits; runs the crew exactly once.

    Stages:
      1 — Full target-language summary inline        (~3K tokens, default)
      2 — Abstract + GRADE sections + file path      (~1.5K tokens)
      3 — File path + clinical numbers only           (~300 tokens)
    """
    budget = ctx_window - max_tokens - 2000  # 2000-token system-prompt buffer

    for stage in range(1, 4):
        est = _estimate_task_tokens(task, translation_task_obj, language)
        if est <= budget or stage == 3:
            if stage > 1:
                print(f"  ⚠ {crew_name}: SOT stage {stage} selected "
                      f"(est {est:,} tokens, budget {budget:,})")
            crew_factory_fn().kickoff()
            return
        # Over budget — degrade to next stage
        print(f"  ⚠ {crew_name}: Stage {stage} est {est:,} tokens > budget {budget:,}. "
              f"Degrading to stage {stage + 1}...")
        base_desc = _SOT_BLOCK_RE.sub('', task.description)
        task.description = base_desc + _build_sot_injection_for_stage(
            stage + 1, sot_file, translated_sot_file,
            sot_summary, translated_sot_summary, grade_numbers_text, language_config
        )


def _call_smart_model(system: str, user: str, max_tokens: int = 4000, temperature: float = 0.1, timeout: int = 0) -> str:
    """Call the Smart Model (vLLM) directly via OpenAI API. Returns response text.
    timeout: seconds to wait. 0 = auto-scale based on max_tokens (~10 tok/s + 60s buffer).
    """
    from openai import OpenAI
    # Disable Qwen3 thinking mode to avoid wasting tokens on <think> blocks
    system = "/no_think\n" + system
    if timeout <= 0:
        # Auto-scale: ~10 tok/s generation speed + 60s buffer for prompt processing
        timeout = max(300, int(max_tokens / 10) + 60)
    client = OpenAI(
        base_url=SMART_BASE_URL,
        api_key=os.getenv("LLM_API_KEY", "NA"),
    )
    resp = client.chat.completions.create(
        model=SMART_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )
    text = resp.choices[0].message.content.strip()
    # Strip <think>...</think> blocks (Qwen3 thinking mode safety net)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def _call_mid_model(system: str, user: str, max_tokens: int = 4000, temperature: float = 0.1, timeout: int = 0) -> str:
    """Call the Mid-Tier Model (Ollama qwen2.5:7b) for translation.
    Falls back to Smart Model if mid-tier is unavailable.
    timeout: seconds to wait. 0 = auto-scale based on max_tokens (~40 tok/s + 60s buffer).
    """
    from openai import OpenAI
    if timeout <= 0:
        # Auto-scale: ~40 tok/s generation speed + 60s buffer
        timeout = max(180, int(max_tokens / 40) + 60)
    try:
        client = OpenAI(
            base_url=MID_BASE_URL,
            api_key=os.getenv("LLM_API_KEY", "NA"),
        )
        resp = client.chat.completions.create(
            model=MID_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        text = resp.choices[0].message.content.strip()
        return text
    except Exception as e:
        print(f"  ⚠ Mid-tier model ({MID_MODEL}) unavailable: {e} — falling back to Smart Model")
        return _call_smart_model(system, user, max_tokens=max_tokens, temperature=temperature, timeout=0)


# --- IMRaD-aware SOT splitting for translation prefix caching ---

_IMRAD_HEADERS = {
    "## Abstract",
    "## 1. Introduction",
    "## 2. Methods",
    "## 3. Results",
    "## 4. Discussion",
    "## 5. References",
}


def _split_sot_imrad(sot_content: str) -> list:
    """Split SOT at top-level IMRaD boundaries only.
    Returns [(header, body), ...]. First entry may have header="" (preamble).
    Embedded ## headers (case reports, framing doc) stay inside their parent section.
    """
    sections = []
    current_header = ""
    current_lines = []

    for line in sot_content.splitlines(keepends=True):
        stripped = line.rstrip()
        if stripped in _IMRAD_HEADERS:
            # Flush previous section
            sections.append((current_header, "".join(current_lines)))
            current_header = stripped
            current_lines = []
        else:
            current_lines.append(line)

    # Flush last section
    sections.append((current_header, "".join(current_lines)))
    return sections


def _estimate_translation_tokens(char_count: int) -> int:
    """Dynamic max_tokens based on input size. Japanese expands ~1.5x vs English in tokens."""
    if char_count < 5000:
        return 4000
    elif char_count <= 15000:
        return 6000
    else:
        return 10000


def _split_at_subheaders(body: str) -> list:
    """Split Discussion body at numbered ### 4.N boundaries only.
    Returns [(sub_header, sub_body), ...]. First entry may have sub_header="" (preamble).
    Embedded ### headers (e.g. ### **Study Designs**) stay inside their parent subsection.
    """
    sections = []
    current_header = ""
    current_lines = []

    for line in body.splitlines(keepends=True):
        stripped = line.rstrip()
        # Only split at numbered Discussion subsections: ### 4.1, ### 4.2, etc.
        if re.match(r"^### 4\.\d", stripped):
            sections.append((current_header, "".join(current_lines)))
            current_header = stripped
            current_lines = []
        else:
            current_lines.append(line)

    sections.append((current_header, "".join(current_lines)))
    return sections


def _translate_sot_pipelined(sot_content: str, language: str, language_config: dict) -> str:
    """Pipelined: translate on mid-tier model, audit on smart model.
    Translation and audit overlap via asyncio producer-consumer pattern.
    Falls back to sequential smart-model-only if mid-tier is unavailable.
    """
    # Strip leftover <think> blocks from Qwen3 thinking mode that may be embedded in the SOT
    sot_content = re.sub(r"<think>.*?</think>", "", sot_content, flags=re.DOTALL).strip()

    imrad_sections = _split_sot_imrad(sot_content)

    lang_name = language_config['name']
    chinese_ban = ""
    if language == 'ja':
        chinese_ban = (
            "ABSOLUTE RULE: Output MUST be in Japanese (\u65e5\u672c\u8a9e) ONLY. NEVER use Chinese (\u4e2d\u6587).\n"
            "WRONG: \u6267\u884c\u529f\u80fd \u2192 CORRECT: \u5b9f\u884c\u6a5f\u80fd; WRONG: \u8865\u5145 \u2192 CORRECT: \u88dc\u5145\n"
            "If unsure of the Japanese term, keep the English term \u2014 NEVER use Chinese.\n"
            "Common errors to avoid:\n"
            "\u6267\u884c\u2192\u5b9f\u884c, \u8865\u5145\u2192\u88dc\u5145, \u8ba4\u77e5\u2192\u8a8d\u77e5, \u6548\u679c\u2192\u52b9\u679c, \u8425\u517b\u2192\u6804\u990a,\n"
            "\u7ef4\u751f\u7d20\u2192\u30d3\u30bf\u30df\u30f3, \u5242\u91cf\u2192\u7528\u91cf, \u8bc1\u636e\u2192\u30a8\u30d3\u30c7\u30f3\u30b9, \u7ed3\u8bba\u2192\u7d50\u8ad6,\n"
            "\u663e\u8457\u2192\u986f\u8457, \u5206\u6790\u2192\u5206\u6790, \u4e34\u5e8a\u2192\u81e8\u5e8a, \u6444\u5165\u2192\u6442\u53d6, \u5065\u5eb7\u2192\u5065\u5eb7\n"
            "Double-check: NO Simplified Chinese characters in your output.\n\n"
        )

    translate_system = (
        "{chinese_ban}"
        "You are a medical translation specialist. Translate the following section into {lang_name}.\n"
        "RULES:\n"
        "- Preserve ALL markdown formatting (headers, tables, bullet points, bold, italic)\n"
        "- Keep study names, journal names, and URLs in English\n"
        "- Keep clinical abbreviations in English: ARR, NNT, GRADE, CER, EER, RCT, RRR, CI, OR, HR\n"
        "- Preserve ALL numerical values exactly (percentages, CI ranges, p-values, sample sizes)\n"
        "- Keep confidence labels (HIGH/MEDIUM/LOW/CONTESTED) in English\n"
        "- Translate meaning, not word-for-word\n"
        "- Output ONLY the translated text, no commentary"
    ).format(chinese_ban=chinese_ban, lang_name=lang_name)

    chinese_rules = ""
    if language == 'ja':
        chinese_rules = (
            "CRITICAL FOCUS \u2014 Chinese contamination:\n"
            "- Replace ANY Simplified Chinese characters with their Japanese equivalents\n"
            "- Common errors: \u6267\u884c\u2192\u5b9f\u884c, \u8865\u5145\u2192\u88dc\u5145, \u8ba4\u77e5\u2192\u8a8d\u77e5, \u7814\u7a76\u2192\u7814\u7a76, \u6548\u679c\u2192\u52b9\u679c, "
            "\u8425\u517b\u2192\u6804\u990a, \u7ef4\u751f\u7d20\u2192\u30d3\u30bf\u30df\u30f3, \u5242\u91cf\u2192\u7528\u91cf, \u8bc1\u636e\u2192\u30a8\u30d3\u30c7\u30f3\u30b9, \u7ed3\u8bba\u2192\u7d50\u8ad6, "
            "\u663e\u8457\u2192\u986f\u8457, \u5206\u6790\u2192\u5206\u6790, \u4e34\u5e8a\u2192\u81e8\u5e8a, \u6444\u5165\u2192\u6442\u53d6, \u5065\u5eb7\u2192\u5065\u5eb7\n"
            "- If you find Chinese sentences, translate them to Japanese\n\n"
        )

    audit_system = (
        "You are a {lang_name} language quality auditor for medical documents.\n\n"
        "{chinese_rules}"
        "Your task:\n"
        "1. Find and fix any non-{lang_name} text (Chinese or untranslated English sentences)\n"
        "2. Fix garbled or unnatural transliterations\n"
        "3. Ensure medical terminology is correct in {lang_name}\n"
        "4. KEEP in English: study names, journal names, URLs, clinical abbreviations "
        "(ARR, NNT, GRADE, CER, EER, RCT, RRR, CI, OR, HR), confidence labels\n"
        "5. Preserve ALL markdown formatting and numerical values exactly\n\n"
        "Return the COMPLETE corrected section. If no issues found, return the section unchanged.\n"
        "IMPORTANT: Output ONLY the corrected text. Do NOT include any commentary, explanation, or preamble."
    ).format(lang_name=lang_name, chinese_rules=chinese_rules)

    # --- Flatten sections into ordered chunk list ---
    # Each chunk: (index, header, body, passthrough_flag)
    # passthrough_flag: True = skip translate/audit, False = needs translate+audit,
    #                   "discussion_marker" = reassembly marker for Discussion section
    chunks = []
    for sec_idx, (header, body) in enumerate(imrad_sections):
        if header == "## 5. References":
            chunks.append((len(chunks), header, body, True))
            continue
        if (not body.strip() and not header) or (len(body.strip()) < 10 and not header):
            chunks.append((len(chunks), header, body, True))
            continue
        if header == "## 4. Discussion":
            sub_chunks = _split_at_subheaders(body)
            disc_parts = []
            for sub_idx, (sub_hdr, sub_body) in enumerate(sub_chunks):
                if (not sub_body.strip() and not sub_hdr) or (len(sub_body.strip()) < 10 and not sub_hdr):
                    disc_parts.append((len(chunks), sub_hdr, sub_body, True))
                else:
                    disc_parts.append((len(chunks), sub_hdr, sub_body, False))
                chunks.append(disc_parts[-1])
            # Store Discussion metadata for reassembly
            chunks.append((len(chunks), header, disc_parts, "discussion_marker"))
            continue
        chunks.append((len(chunks), header, body, False))

    # --- Build reassembly plan ---
    output_plan = []
    i = 0
    while i < len(chunks):
        idx, header, body_or_parts, flag = chunks[i]
        if flag == "discussion_marker":
            disc_indices = [p[0] for p in body_or_parts]
            output_plan.append((header, "discussion", disc_indices))
        else:
            output_plan.append((header, "single", [idx]))
        i += 1

    # --- Identify translatable chunks ---
    translatable = [(c[0], c[1], c[2]) for c in chunks if c[3] is False]
    total_translatable = len(translatable)
    if total_translatable == 0:
        print("  No translatable sections found")
        return sot_content

    # --- Results storage (indexed by chunk index) ---
    results = {}

    # --- Probe mid-tier model with first chunk ---
    mid_tier_available = True
    first_chunk_idx, first_header, first_body = translatable[0]
    first_label = first_header if first_header else "preamble"
    max_tok = _estimate_translation_tokens(len(first_body))
    use_smart_first = len(first_body) > 8000
    try:
        if use_smart_first:
            translated = _call_smart_model(
                system=translate_system,
                user="Translate this section:\n\n" + first_body,
                max_tokens=max_tok, temperature=0.1,
            )
        else:
            from openai import OpenAI as _MidOpenAI
            _mid_client = _MidOpenAI(base_url=MID_BASE_URL, api_key=os.getenv("LLM_API_KEY", "NA"))
            _mid_resp = _mid_client.chat.completions.create(
                model=MID_MODEL,
                messages=[
                    {"role": "system", "content": translate_system},
                    {"role": "user", "content": "Translate this section:\n\n" + first_body},
                ],
                max_tokens=max_tok, temperature=0.1,
                timeout=max(180, int(max_tok / 40) + 60),
            )
            translated = _mid_resp.choices[0].message.content.strip()
        results[first_chunk_idx] = translated
        model_tag = "smart" if use_smart_first else "mid"
        print("  \u2713 Translated {} ({} \u2192 {} chars) [{}, 1/{}]".format(
            first_label, len(first_body), len(translated), model_tag, total_translatable))
    except Exception as e:
        if not use_smart_first:
            print("  \u26a0 Mid-tier model ({}) unavailable: {} \u2014 falling back to smart-model-only".format(MID_MODEL, e))
            mid_tier_available = False
            try:
                translated = _call_smart_model(
                    system=translate_system,
                    user="Translate this section:\n\n" + first_body,
                    max_tokens=max_tok, temperature=0.1,
                )
                results[first_chunk_idx] = translated
                print("  \u2713 Translated {} ({} \u2192 {} chars) [smart fallback, 1/{}]".format(
                    first_label, len(first_body), len(translated), total_translatable))
            except Exception as e2:
                print("  \u26a0 Translation failed for {}: {} \u2014 keeping original".format(first_label, e2))
                results[first_chunk_idx] = first_body
        else:
            print("  \u26a0 Translation failed for {}: {} \u2014 keeping original".format(first_label, e))
            results[first_chunk_idx] = first_body

    remaining = translatable[1:]

    if not mid_tier_available:
        # --- FALLBACK: Sequential smart-model-only (old behavior) ---
        print("  Running sequential translate+audit on Smart Model (no pipeline)")
        translate_count = 1
        for chunk_idx, header, body in remaining:
            label = header if header else "preamble"
            max_tok = _estimate_translation_tokens(len(body))
            try:
                translated = _call_smart_model(
                    system=translate_system,
                    user="Translate this section:\n\n" + body,
                    max_tokens=max_tok, temperature=0.1,
                )
                translate_count += 1
                results[chunk_idx] = translated
                print("  \u2713 Translated {} ({} \u2192 {} chars) [smart, {}/{}]".format(
                    label, len(body), len(translated), translate_count, total_translatable))
            except Exception as e:
                print("  \u26a0 Translation failed for {}: {} \u2014 keeping original".format(label, e))
                results[chunk_idx] = body

        # Sequential audit pass
        audit_count = 0
        for chunk_idx, header, body in translatable:
            label = header if header else "preamble"
            translated_body = results.get(chunk_idx, body)
            if not translated_body.strip():
                continue
            max_tok = _estimate_translation_tokens(len(translated_body))
            try:
                audited = _call_smart_model(
                    system=audit_system,
                    user="Audit and correct this {} medical document section:\n\n".format(lang_name) + translated_body,
                    max_tokens=max_tok, temperature=0.1,
                )
                audit_count += 1
                if len(audited) < len(translated_body) * 0.5:
                    print("  \u26a0 Audit output too short for {} ({} vs {}) \u2014 keeping translation".format(
                        label, len(audited), len(translated_body)))
                else:
                    results[chunk_idx] = audited
                    print("  \u2713 Audited {} ({} \u2192 {} chars) [smart, {}/{}]".format(
                        label, len(translated_body), len(audited), audit_count, total_translatable))
            except Exception as e:
                print("  \u26a0 Audit failed for {}: {} \u2014 keeping translation".format(label, e))

        print("  \u2713 Translation+audit complete: {} translate + {} audit calls (sequential fallback)".format(
            translate_count, audit_count))
    else:
        # --- PIPELINED: Mid-tier translates, smart model audits concurrently ---
        print("  Running pipelined translate (mid-tier) + audit (smart) [{}]".format(MID_MODEL))

        async def _run_pipeline():
            queue = asyncio.Queue()
            translate_count = 1  # first chunk already done
            audit_count = 0

            async def producer():
                nonlocal translate_count
                for chunk_idx, header, body in remaining:
                    label = header if header else "preamble"
                    max_tok = _estimate_translation_tokens(len(body))
                    use_smart_for_chunk = len(body) > 8000
                    try:
                        if use_smart_for_chunk:
                            translated = await asyncio.to_thread(
                                _call_smart_model,
                                translate_system,
                                "Translate this section:\n\n" + body,
                                max_tok, 0.1,
                            )
                        else:
                            translated = await asyncio.to_thread(
                                _call_mid_model,
                                translate_system,
                                "Translate this section:\n\n" + body,
                                max_tok, 0.1,
                            )
                        translate_count += 1
                        results[chunk_idx] = translated
                        model_tag = "smart" if use_smart_for_chunk else "mid"
                        print("  \u2713 Translated {} ({} \u2192 {} chars) [{}, {}/{}]".format(
                            label, len(body), len(translated), model_tag, translate_count, total_translatable))
                    except Exception as e:
                        print("  \u26a0 Translation failed for {}: {} \u2014 keeping original".format(label, e))
                        results[chunk_idx] = body
                    await queue.put(chunk_idx)
                # Signal done
                await queue.put(None)

            async def consumer():
                nonlocal audit_count
                # First: audit the already-translated first chunk
                first_translated = results.get(first_chunk_idx, first_body)
                if first_translated.strip():
                    max_tok = _estimate_translation_tokens(len(first_translated))
                    try:
                        audited = await asyncio.to_thread(
                            _call_smart_model,
                            audit_system,
                            "Audit and correct this {} medical document section:\n\n".format(lang_name) + first_translated,
                            max_tok, 0.1,
                        )
                        audit_count += 1
                        if len(audited) >= len(first_translated) * 0.5:
                            results[first_chunk_idx] = audited
                            flabel = first_header if first_header else "preamble"
                            print("  \u2713 Audited {} ({} \u2192 {} chars) [smart, {}/{}]".format(
                                flabel, len(first_translated), len(audited), audit_count, total_translatable))
                        else:
                            flabel = first_header if first_header else "preamble"
                            print("  \u26a0 Audit output too short for {} \u2014 keeping translation".format(flabel))
                    except Exception as e:
                        flabel = first_header if first_header else "preamble"
                        print("  \u26a0 Audit failed for {}: {} \u2014 keeping translation".format(flabel, e))

                # Then: audit chunks as they arrive from producer
                while True:
                    chunk_idx = await queue.get()
                    if chunk_idx is None:
                        break
                    translated_body = results.get(chunk_idx, "")
                    if not translated_body.strip():
                        continue
                    # Find label for this chunk
                    label = "section"
                    for c in translatable:
                        if c[0] == chunk_idx:
                            label = c[1] if c[1] else "preamble"
                            break
                    max_tok = _estimate_translation_tokens(len(translated_body))
                    try:
                        audited = await asyncio.to_thread(
                            _call_smart_model,
                            audit_system,
                            "Audit and correct this {} medical document section:\n\n".format(lang_name) + translated_body,
                            max_tok, 0.1,
                        )
                        audit_count += 1
                        if len(audited) >= len(translated_body) * 0.5:
                            results[chunk_idx] = audited
                            print("  \u2713 Audited {} ({} \u2192 {} chars) [smart, {}/{}]".format(
                                label, len(translated_body), len(audited), audit_count, total_translatable))
                        else:
                            print("  \u26a0 Audit output too short for {} \u2014 keeping translation".format(label))
                    except Exception as e:
                        print("  \u26a0 Audit failed for {}: {} \u2014 keeping translation".format(label, e))

            await asyncio.gather(producer(), consumer())
            return translate_count, audit_count

        # Run the async pipeline
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                t_count, a_count = pool.submit(lambda: asyncio.run(_run_pipeline())).result()
        else:
            t_count, a_count = asyncio.run(_run_pipeline())
        print("  \u2713 Pipelined translation+audit complete: {} translate + {} audit calls".format(t_count, a_count))

    # --- Reassemble output in original section order ---
    assembled_parts = []
    for plan_entry in output_plan:
        header = plan_entry[0]
        plan_type = plan_entry[1]

        if plan_type == "discussion":
            disc_indices = plan_entry[2]
            disc_sub_parts = []
            for didx in disc_indices:
                for c in chunks:
                    if c[0] == didx:
                        sub_hdr = c[1]
                        if c[3] is True:
                            piece = (sub_hdr + "\n" + c[2]) if sub_hdr else c[2]
                        else:
                            body = results.get(didx, c[2])
                            piece = (sub_hdr + "\n" + body) if sub_hdr else body
                        disc_sub_parts.append(piece)
                        break
            discussion_body = "\n".join(disc_sub_parts)
            assembled_parts.append(header + "\n" + discussion_body)
        elif plan_type == "single":
            cidx = plan_entry[2][0]
            for c in chunks:
                if c[0] == cidx:
                    if c[3] is True:
                        piece = (c[1] + "\n" + c[2]) if c[1] else c[2]
                    elif c[3] == "discussion_marker":
                        continue
                    else:
                        body = results.get(cidx, c[2])
                        piece = (c[1] + "\n" + body) if c[1] else body
                    assembled_parts.append(piece)
                    break

    result_text = "\n".join(assembled_parts)

    # Completeness check: compare ## header counts
    source_headers = sot_content.count("\n## ")
    result_headers = result_text.count("\n## ")
    if result_headers < source_headers:
        missing = source_headers - result_headers
        print("  \u26a0 Translation missing {} section(s): source has {} ## headers, result has {}".format(
            missing, source_headers, result_headers))
    # Length sanity check
    if len(result_text) < len(sot_content) * 0.5:
        print("  \u26a0 Translated SOT suspiciously short: {} chars vs {} chars original".format(
            len(result_text), len(sot_content)))

    return result_text


def _translate_prompt(prompt_text: str, language: str, language_config: dict) -> str:
    """Translate a task prompt/instruction to the target language. Preserves structure."""
    lang_name = language_config['name']
    chinese_ban = ""
    if language == 'ja':
        chinese_ban = (
            "ABSOLUTE RULE: Translate to Japanese (日本語) ONLY. NEVER use Chinese.\n"
            "WRONG: 执行功能 → CORRECT: 実行機能\n\n"
        )
    system = (
        f"{chinese_ban}"
        f"Translate these podcast production instructions to {lang_name}.\n"
        f"KEEP intact: all markdown formatting (##, ###, numbered lists, bold), "
        f"variable placeholders, technical abbreviations (ARR, NNT, GRADE, RCT, CI, HR, OR), "
        f"speaker labels (Host 1:, Host 2:), [TRANSITION] markers.\n"
        f"This is an instruction template, not content — translate the instructional language only."
    )
    try:
        result = _call_smart_model(
            system=system,
            user=f"Translate:\n\n{prompt_text}",
            max_tokens=6000,
            temperature=0.1,
        )
        if len(result) < len(prompt_text) * 0.3:
            print(f"  ⚠ Prompt translation too short — keeping original")
            return prompt_text
        return result
    except Exception as e:
        print(f"  ⚠ Prompt translation failed: {e} — keeping original")
        return prompt_text


def _audit_script_language(script_text: str, language: str, language_config: dict) -> str:
    """Post-Crew 3 audit: ensure script is consistently in the target language."""
    if language == 'en':
        return script_text
    lang_name = language_config['name']
    chinese_ban = ""
    if language == 'ja':
        chinese_ban = (
            "Also fix any Chinese characters — replace with Japanese equivalents.\n"
        )
    system = (
        f"You are a {lang_name} language consistency auditor for a podcast script.\n"
        f"Find any non-{lang_name} sentences (English or Chinese) and translate them to natural {lang_name}.\n"
        f"{chinese_ban}"
        f"KEEP in English: scientific terms (ARR, NNT, GRADE, RCT, CI, HR, OR), "
        f"study abbreviations, URLs, speaker labels (Host 1:, Host 2:).\n"
        f"Return the COMPLETE corrected script preserving ALL [TRANSITION] markers "
        f"and Host 1/Host 2 labels.\n"
        f"CRITICAL: Do NOT add any preamble, notes, or commentary before or after the script. "
        f"Output ONLY the corrected script lines. "
        f"If you must include any notes, prefix each note line with '## '."
    )
    try:
        result = _call_smart_model(
            system=system,
            user=f"Audit this podcast script for language consistency:\n\n{script_text}",
            max_tokens=12000,
            temperature=0.1,
        )
        # Sanity checks
        if len(result) < len(script_text) * 0.5:
            print(f"  ⚠ Script audit output too short — keeping original")
            return script_text
        orig_transitions = script_text.count('[TRANSITION]')
        result_transitions = result.count('[TRANSITION]')
        if orig_transitions > 0 and result_transitions < orig_transitions:
            print(f"  ⚠ Script audit lost [TRANSITION] markers ({orig_transitions}→{result_transitions}) — keeping original")
            return script_text
        print(f"  ✓ Script language audit complete ({len(script_text)} → {len(result)} chars)")
        return result
    except Exception as e:
        print(f"  ⚠ Script language audit failed: {e} — keeping original")
        return script_text


def _quick_content_audit(script_text: str, sot_content: str) -> str | None:
    """Check script for top-3 drift patterns against SOT. Returns issue string or None."""
    sot_excerpt = _truncate_at_boundary(sot_content, 4000)
    script_excerpt = _truncate_at_boundary(script_text, 4000)
    system = (
        "You are a scientific accuracy auditor. Compare the podcast script excerpt against "
        "the source-of-truth research. Check for these 3 drift patterns ONLY:\n"
        "1. CAUSATION INFLATION: Script states causation where source only shows correlation\n"
        "2. HEDGE REMOVAL: Script presents uncertain findings as settled fact\n"
        "3. CHERRY-PICKING: Script omits contradicting evidence that the source includes\n\n"
        "If the script is faithful to the source, respond with exactly: CLEAN\n"
        "If you find drift, respond with a 1-2 sentence description of the issue."
    )
    try:
        result = _call_smart_model(
            system=system,
            user=(
                f"SOURCE OF TRUTH (excerpt):\n{sot_excerpt}\n\n"
                f"PODCAST SCRIPT (excerpt):\n{script_excerpt}"
            ),
            max_tokens=200,
            temperature=0.1,
        )
        if result.strip().upper() == "CLEAN":
            return None
        return result.strip()
    except Exception as e:
        print(f"  ⚠ Content audit call failed: {e} — skipping")
        return None


def _validate_script(script_text: str, target_length: int, tolerance: float,
                     language_config: dict, sot_content: str, stage: str) -> dict:
    """
    Validate script for length, structure, repetition, and content accuracy.
    Returns: {'pass': bool, 'feedback': str, 'word_count': int, 'issues': list}
    """
    issues = []

    # Strip <think> blocks before measuring (Qwen3 safety net)
    script_text = re.sub(r'<think>.*?</think>', '', script_text, flags=re.DOTALL).strip()

    length_unit = language_config['length_unit']

    # 1. Measure word/char count
    if length_unit == 'chars':
        count = len(re.sub(r'[\s\n\r\t\u3000\uff1a:\u300c\u300d\u3001\u3002\u30fb\uff08\uff09\-\u2014*#]', '', script_text))
    else:
        content_only = re.sub(r'^[A-Za-z0-9_ ]+:\s*', '', script_text, flags=re.MULTILINE)
        count = len(content_only.split())

    low = int(target_length * (1 - tolerance))
    high = int(target_length * (1 + tolerance))

    if count < low:
        issues.append(f"TOO SHORT: {count} {length_unit} (need \u2265{low})")
    elif count > high:
        issues.append(f"TOO LONG: {count} {length_unit} (need \u2264{high})")

    # 2. Degenerate repetition detection
    words = script_text.lower().split()
    consecutive = 1
    for i in range(1, len(words)):
        if words[i] == words[i-1] and len(words[i]) > 2:
            consecutive += 1
            if consecutive >= 4:
                issues.append(f"DEGENERATE REPETITION: '{words[i]}' repeated {consecutive}+ times consecutively")
                break
        else:
            consecutive = 1

    # 3. Structure check (for polish stage)
    if stage == 'polish':
        transition_count = script_text.count('[TRANSITION]')
        if transition_count < 3:
            issues.append(f"MISSING TRANSITIONS: only {transition_count} [TRANSITION] markers (need \u22653)")

    # 4. LLM content audit (only if Python checks pass — saves tokens)
    if not issues and sot_content:
        content_audit = _quick_content_audit(script_text, sot_content)
        if content_audit:
            issues.append(f"CONTENT: {content_audit}")

    return {
        'pass': len(issues) == 0,
        'feedback': '\n'.join(f"- {i}" for i in issues) if issues else 'PASS',
        'word_count': count,
        'issues': issues
    }


# --- SCRIPT EXPANSION HELPERS ---
# Default per-act word allocation (fraction of total target)
ACT_ALLOCATIONS = {1: 0.20, 2: 0.35, 3: 0.25, 4: 0.20}


def _count_words(text: str, language_config: dict) -> int:
    """Count words (English) or content characters (Japanese) in text."""
    if language_config['length_unit'] == 'chars':
        return len(re.sub(r'[\s\n\r\t\u3000\uff1a:\u300c\u300d\u3001\u3002\u30fb\uff08\uff09\-\u2014*#]', '', text))
    else:
        content_only = re.sub(r'^[A-Za-z0-9_ ]+:\s*', '', text, flags=re.MULTILINE)
        return len(content_only.split())


def _analyze_acts(script_text: str, language_config: dict, target_length: int) -> list:
    """
    Parse script into acts and compute per-act word counts, targets, and deficits.
    Returns list of dicts: [{'num': 1, 'text': '...', 'count': N, 'target': N, 'deficit': N}, ...]
    Also returns preamble and postamble as special entries with num=0 and num=99.
    """
    acts = []
    length_unit = language_config['length_unit']

    # English: split on "### **ACT N" headers
    act_pattern = re.compile(r'(###\s*\*?\*?ACT\s+(\d+)[\s\S]*?)(?=###\s*\*?\*?ACT\s+\d+|$)', re.IGNORECASE)
    matches = list(act_pattern.finditer(script_text))

    if matches:
        # Preamble: everything before first act
        preamble_text = script_text[:matches[0].start()].strip()
        preamble_count = _count_words(preamble_text, language_config)
        acts.append({'num': 0, 'text': preamble_text, 'count': preamble_count, 'target': 0, 'deficit': 0, 'label': 'preamble'})

        for m in matches:
            act_num = int(m.group(2))
            act_text = m.group(1).strip()
            act_count = _count_words(act_text, language_config)
            alloc = ACT_ALLOCATIONS.get(act_num, 0.20)
            act_target = int(target_length * alloc)
            deficit = act_target - act_count
            acts.append({
                'num': act_num,
                'text': act_text,
                'count': act_count,
                'target': act_target,
                'deficit': deficit,
                'label': f'ACT {act_num}',
            })

        # Postamble: everything after last act (wrap-up, one-action)
        last_end = matches[-1].end()
        postamble_text = script_text[last_end:].strip()
        if postamble_text:
            postamble_count = _count_words(postamble_text, language_config)
            acts.append({'num': 99, 'text': postamble_text, 'count': postamble_count, 'target': 0, 'deficit': 0, 'label': 'postamble'})
    else:
        # Japanese or unstructured: split on [TRANSITION] markers
        sections = re.split(r'\[TRANSITION\]', script_text)
        if len(sections) >= 3:
            # Distribute evenly across sections, treating first as preamble and last as postamble
            n_acts = max(len(sections) - 2, 1)
            per_act_target = target_length // n_acts if n_acts > 0 else target_length
            acts.append({'num': 0, 'text': sections[0].strip(), 'count': _count_words(sections[0], language_config), 'target': 0, 'deficit': 0, 'label': 'preamble'})
            for i, sec in enumerate(sections[1:-1], start=1):
                sec_text = sec.strip()
                sec_count = _count_words(sec_text, language_config)
                acts.append({
                    'num': i,
                    'text': sec_text,
                    'count': sec_count,
                    'target': per_act_target,
                    'deficit': per_act_target - sec_count,
                    'label': f'Section {i}',
                })
            acts.append({'num': 99, 'text': sections[-1].strip(), 'count': _count_words(sections[-1], language_config), 'target': 0, 'deficit': 0, 'label': 'postamble'})
        else:
            # Cannot parse — return single block
            total = _count_words(script_text, language_config)
            acts.append({'num': 1, 'text': script_text, 'count': total, 'target': target_length, 'deficit': target_length - total, 'label': 'full script'})

    return acts


def _expand_act(act_info: dict, sot_content: str, language_config: dict,
                session_roles: dict, topic_name: str, target_instruction: str) -> str:
    """
    Expand a single act that is under its word target using _call_smart_model().
    Returns the expanded act text.
    """
    act_num = act_info['num']
    act_text = act_info['text']
    act_target = act_info['target']
    current_count = act_info['count']
    length_unit = language_config['length_unit']

    presenter = session_roles['presenter']['label']
    questioner = session_roles['questioner']['label']

    # Truncate SOT to a manageable excerpt for the expansion prompt
    sot_excerpt = sot_content[:4000] if len(sot_content) > 4000 else sot_content

    system_prompt = (
        f"You are expanding ACT {act_num} of a two-host science podcast about \"{topic_name}\".\n"
        f"Hosts: {presenter} (presenter) and {questioner} (questioner).\n"
        f"This act currently has {current_count} {length_unit} but needs ~{act_target} {length_unit}.\n"
        f"RULES:\n"
        f"- Keep ALL existing dialogue lines intact — do NOT remove or rephrase them.\n"
        f"- ADD new exchanges between the hosts that deepen the discussion.\n"
        f"- Add concrete examples, study details, listener-relevant implications.\n"
        f"- Maintain the natural conversational tone with reactions, follow-up questions, humor.\n"
        f"- Use speaker labels: **{presenter}:** and **{questioner}:**\n"
        f"- Do NOT add section headers or act labels — just return the dialogue content.\n"
        f"- {target_instruction}\n"
    )

    user_prompt = (
        f"SOURCE OF TRUTH (key findings):\n{sot_excerpt}\n\n"
        f"CURRENT ACT TEXT (expand this):\n{act_text}\n\n"
        f"Return the COMPLETE expanded act with ~{act_target} {length_unit}."
    )

    try:
        expanded = _call_smart_model(
            system=system_prompt,
            user=user_prompt,
            max_tokens=12000,  # Increased from 6000: Japanese ~1.5 chars/token → need 10K tokens for 15K chars
            temperature=0.7,
        )
        expanded_count = _count_words(expanded, language_config)
        # Only use expansion if it's actually longer
        if expanded_count > current_count:
            print(f"    ACT {act_num}: {current_count} → {expanded_count} {length_unit}")
            return expanded
        else:
            print(f"    ACT {act_num}: expansion did not increase length ({expanded_count} vs {current_count}) — keeping original")
            return act_text
    except Exception as e:
        print(f"    ACT {act_num}: expansion failed ({e}) — keeping original")
        return act_text


def _run_script_expansion(draft_text: str, sot_content: str, target_length: int,
                          language_config: dict, session_roles: dict,
                          topic_name: str, target_instruction: str) -> tuple:
    """
    Orchestrator: analyze acts, expand any under 75% of target, reassemble.
    Returns (expanded_text, was_expanded).
    """
    EXPANSION_THRESHOLD = 0.75  # expand acts under 75% of their target

    acts = _analyze_acts(draft_text, language_config, target_length)
    length_unit = language_config['length_unit']
    was_expanded = False

    # Log per-act analysis
    print("  Per-act analysis:")
    for a in acts:
        if a['num'] in (0, 99):
            print(f"    {a['label']}: {a['count']} {length_unit}")
        else:
            pct = (a['count'] / a['target'] * 100) if a['target'] > 0 else 100
            marker = " ← EXPAND" if pct < EXPANSION_THRESHOLD * 100 else ""
            print(f"    {a['label']}: {a['count']}/{a['target']} {length_unit} ({pct:.0f}%){marker}")

    # Expand under-target acts
    for i, a in enumerate(acts):
        if a['num'] in (0, 99):
            continue  # skip preamble/postamble
        if a['target'] <= 0:
            continue
        pct = a['count'] / a['target']
        if pct < EXPANSION_THRESHOLD:
            expanded_text = _expand_act(a, sot_content, language_config,
                                        session_roles, topic_name, target_instruction)
            acts[i]['text'] = expanded_text
            acts[i]['count'] = _count_words(expanded_text, language_config)
            was_expanded = True

    if not was_expanded:
        return draft_text, False

    # Reassemble: detect whether original used [TRANSITION] or ACT headers
    has_act_headers = bool(re.search(r'###\s*\*?\*?ACT\s+\d+', draft_text, re.IGNORECASE))

    if has_act_headers:
        # Re-join with the original act header lines preserved in act text
        parts = []
        for a in acts:
            parts.append(a['text'])
        reassembled = '\n\n'.join(parts)
    else:
        # Re-join with [TRANSITION] markers
        parts = []
        for a in acts:
            parts.append(a['text'])
        reassembled = '\n\n[TRANSITION]\n\n'.join(parts)

    total = _count_words(reassembled, language_config)
    print(f"  Pass 1 complete: {total} {length_unit} total")

    # Second pass: if still below 85% of target, expand again on the combined result
    if total < target_length * 0.85:
        print(f"  Still short ({total}/{target_length} {length_unit}) — running pass 2...")
        pass2_act = {
            'num': 1, 'text': reassembled, 'count': total,
            'target': target_length, 'deficit': target_length - total,
            'label': 'full script (pass 2)',
        }
        pass2_result = _expand_act(pass2_act, sot_content, language_config,
                                   session_roles, topic_name, target_instruction)
        pass2_count = _count_words(pass2_result, language_config)
        if pass2_count > total:
            reassembled = pass2_result
            total = pass2_count
        print(f"  Pass 2 complete: {total} {length_unit} total")

    print(f"  Expansion complete: {total} {length_unit} total")
    return reassembled, True


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
        f'{english_instruction}\n\n'
        f'DIALOGUE RULE: Hosts must NEVER address each other by name inside dialogue — '
        f'no personal names, no "Host 1", no "Host 2" spoken aloud. '
        f'Names are only used as speaker LABELS before the colon, never within the dialogue itself.'
        + (f'\n\nLANGUAGE WARNING: When generating Japanese (日本語) output, you MUST stay in Japanese throughout. '
           f'Do NOT switch to Chinese (中文). '
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
        f'Target: Exactly {target_script} {target_unit_plural} ({_target_min} minutes). '
        f'{target_instruction}'
    ),
    backstory=(
        f'Editor for high-end intellectual podcasts (Huberman Lab, Lex Fridman). '
        f'Your audience has advanced degrees - they want depth, not hand-holding.\n\n'
        f'EDITING RULES:\n'
        f'  - Remove any definitions of basic scientific concepts\n'
        f'  - Ensure the questioner\'s questions feel natural and audience-aligned\n'
        f'  - Keep technical language intact (no dumbing down)\n'
        f'  - Target exactly {target_script} {target_unit_plural} for {_target_min}-minute runtime. If the script is too short, YOU MUST ADD DEPTH AND EXAMPLES TO REACH THE TARGET.\n'
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

# Build channel intro directive for script
if channel_intro:
    _channel_intro_directive = (
        f"  1. CHANNEL INTRO (~25 {target_unit_plural}, ~10 seconds):\n"
        f"     {SESSION_ROLES['presenter']['label']}: {channel_intro}\n"
        f"     CRITICAL: Use this text EXACTLY as written above. Do NOT rephrase, summarize, or modify it.\n\n"
    )
else:
    _channel_intro_directive = (
        f"  1. CHANNEL INTRO (~25 {target_unit_plural}, ~10 seconds):\n"
        f"     Both hosts briefly introduce the show and today's topic.\n"
        f"     {SESSION_ROLES['presenter']['label']}: Welcome to Deep Research Podcast. Today we're diving deep into {topic_name}.\n\n"
    )

# Compute approximate word allocations per act
_act1_target = int(target_length_int * 0.20)
_act2_target = int(target_length_int * 0.35)
_act3_target = int(target_length_int * 0.25)
_act4_target = int(target_length_int * 0.20)

script_task = Task(
    description=(
        f"Using the Episode Blueprint, write a comprehensive {target_script}-{target_unit_singular} podcast dialogue about \"{topic_name}\" "
        f"featuring {SESSION_ROLES['presenter']['label']} (presenter) and {SESSION_ROLES['questioner']['label']} (questioner).\n\n"
        f"SCRIPT STRUCTURE (follow this EXACTLY):\n\n"
        + _channel_intro_directive +
        f"  2. THE HOOK (~40 {target_unit_plural}, ~15 seconds):\n"
        f"     Based on the hook question from the Episode Blueprint.\n"
        f"     {SESSION_ROLES['presenter']['label']}: [Provocative question from Blueprint — must be a question, NOT a statement]\n"
        f"     {SESSION_ROLES['questioner']['label']}: [Engaged reaction: 'Oh, that's a great question!' or 'Hmm, I actually have no idea...']\n\n"
        f"  3. ACT 1 — THE CLAIM (~{_act1_target:,} {target_unit_plural}):\n"
        f"     What people believe. The folk wisdom. Why this matters personally.\n"
        f"     - Presenter sets up the common belief or question\n"
        f"     - Questioner validates: 'Right, I've heard that too' / 'That's what everyone says'\n"
        f"     - Establish emotional stakes: why should the listener care?\n\n"
        f"  4. ACT 2 — THE EVIDENCE (~{_act2_target:,} {target_unit_plural}):\n"
        f"     What science actually says. Use BOTH supporting and contradicting evidence from the Blueprint.\n"
        f"     - Present key studies with GRADE-informed framing from the Blueprint's Section 6\n"
        f"     - Include specific numbers (NNT, ARR, sample sizes) where available\n"
        f"     - Questioner challenges: 'But how strong is that evidence?' / 'What about the studies that say otherwise?'\n"
        f"     - Address contradicting evidence honestly — do NOT cherry-pick\n\n"
        f"  5. ACT 3 — THE NUANCE (~{_act3_target:,} {target_unit_plural}):\n"
        f"     Where it gets complicated.\n"
        f"     - GRADE confidence level and what it means for the listener\n"
        f"     - Population differences, dose-response relationships, timing factors\n"
        f"     - Questioner pushes: 'So it's not as simple as people think?'\n"
        f"     - Acknowledge what we DON'T know — science is honest about its limits\n\n"
        f"  6. ACT 4 — THE PROTOCOL (~{_act4_target:,} {target_unit_plural}):\n"
        f"     Translate science into daily life.\n"
        f"     - Specific, practical recommendations\n"
        f"     - 'In practical terms, this means...'\n"
        f"{'     - Tailor recommendations specifically to ' + core_target + chr(10) if core_target else '     - Who should pay attention vs. who can safely ignore this' + chr(10)}"
        f"     - Questioner: 'So what should our listeners actually DO with this?'\n\n"
        f"  7. WRAP-UP (~60 {target_unit_plural}, ~25 seconds):\n"
        f"     Three-sentence summary of the most important takeaways.\n\n"
        f"  8. THE 'ONE ACTION' ENDING (~40 {target_unit_plural}, ~15 seconds):\n"
        f"     {SESSION_ROLES['presenter']['label']}: 'If you take ONE thing from today — [action{' tailored to ' + core_target if core_target else ' to try this week'}].'\n"
        f"     {SESSION_ROLES['questioner']['label']}: [Brief agreement + sign-off]\n\n"
        f"PERSONALITY DIRECTIVES:\n"
        f"- ENERGY: Vary vocal energy — excited for surprising findings, thoughtful pauses for nuance, urgency for practical advice\n"
        f"- REACTIONS: Questioner reacts authentically — genuine surprise ('Wait, seriously?!'), skepticism ('Hmm, that sounds too good to be true...'), humor ('Okay, so basically I've been doing this all wrong')\n"
        f"- BANTER: Include brief moments of friendly banter between hosts — a shared laugh, a playful jab, a relatable personal admission\n"
        f"- FILLERS: Natural conversational fillers: 'Hm, that's interesting', 'Right, right', 'Oh wow', 'Okay so let me get this straight...'\n"
        f"- EMPHASIS: Dramatic pauses via ellipses: 'And here's where it gets interesting...'\n"
        f"- STORYTELLING: After each key finding, paint a picture: 'Imagine you're...' or 'Think about your morning routine...'\n"
        f"- PERSONAL: Brief personal connections: 'I actually tried this myself and...' or 'My partner always says...'\n"
        f"- MOMENTUM: Each act builds energy — start curious, peak at the most surprising finding, resolve with practical clarity\n\n"
        f"CHARACTER ROLES:\n"
        f"  - {SESSION_ROLES['presenter']['label']} (Presenter): presents evidence and explains the topic, "
        f"{SESSION_ROLES['presenter']['personality']}\n"
        f"  - {SESSION_ROLES['questioner']['label']} (Questioner): asks questions the audience would ask, bridges gaps, "
        f"{SESSION_ROLES['questioner']['personality']}\n\n"
        f"Format STRICTLY as:\n"
        f"{SESSION_ROLES['presenter']['label']}: [dialogue]\n"
        f"{SESSION_ROLES['questioner']['label']}: [dialogue]\n\n"
        f"TARGET LENGTH: {target_script} {target_unit_plural}. This is CRITICAL — do not write less. The podcast MUST last {_target_min} minutes. "
        f"If you are too brief, the production will fail.\n"
        f"ACT CHECKLIST: You must write all 4 acts plus Hook, Channel Intro, Wrap-up, and One Action. Count them as you write.\n"
        f"TO REACH THIS LENGTH: You must be extremely detailed and conversational. For every single claim or mechanism, you MUST provide:\n"
        f"  1. A deep-dive explanation of the specific scientific mechanism\n"
        f"  2. A real-world analogy or metaphor that lasts several lines\n"
        f"  3. A practical, relatable example or case study\n"
        f"  4. A counter-argument or nuance followed by a rebuttal\n"
        f"  5. Interactive host dialogue (e.g., 'Wait, let me make sure I've got this right...', 'That's fascinating, tell me more about...')\n"
        f"Expand the conversation. Do not just list facts. Have the hosts explore the 'So what?' and 'What now?' for the audience.\n"
        f"Maintain consistent roles throughout. NO role switching mid-conversation. "
        + (f"\nCRITICAL LANGUAGE RULE: You are writing in Japanese (日本語). "
           f"Do NOT use Chinese (中文) at any point. Every sentence must be in Japanese. "
           f"Use standard Japanese kanji only (気 not 气, 楽 not 乐).\n"
           if language == 'ja' else '')
        + f"{target_instruction}"
    ),
    expected_output=(
        f"A {target_script}-{target_unit_singular} podcast dialogue about {topic_name} between "
        f"{SESSION_ROLES['presenter']['label']} (presents and explains) "
        f"and {SESSION_ROLES['questioner']['label']} (asks bridging questions). "
        f"Follows 8-part structure: Hook, Channel Intro, 4 Acts (Claim, Evidence, Nuance, Protocol), Wrap-up, One Action. "
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
            (f"ABSOLUTE RULE: Output MUST be in Japanese (日本語) ONLY. NEVER use Chinese (中文) at any point.\n"
             f"WRONG: 执行功能 → CORRECT: 実行機能; WRONG: 补充 → CORRECT: 補充; WRONG: 认知 → CORRECT: 認知\n"
             f"If unsure of the Japanese term, keep the English term — NEVER use Chinese.\n\n"
             if language == 'ja' else '')
            + f"Translate the entire Source-of-Truth document about {topic_name} into {language_config['name']}.\n\n"
            f"TRANSLATION RULES:\n"
            f"- Translate ALL sections faithfully: Executive Summary, Key Claims, Evidence, Bibliography\n"
            f"- Preserve scientific accuracy — translate meaning, not word-for-word\n"
            f"- Keep confidence labels (HIGH/MEDIUM/LOW/CONTESTED) intact\n"
            f"- Keep study names, journal names, and URLs in English\n"
            f"- Keep clinical abbreviations in English: ARR, NNT, GRADE, CER, EER, RCT, RRR, CI, OR, HR\n"
            f"- Maintain all markdown formatting (headers, tables, bullet points)\n"
            f"- Preserve ALL numerical values exactly (percentages, CI ranges, p-values, sample sizes) — do NOT convert or round\n"
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
        f"- Target exactly {target_script} {target_unit_plural}\n\n"
        f"MAINTAIN ROLES:\n"
        f"  - {SESSION_ROLES['presenter']['label']} (Presenter): explains and teaches the topic\n"
        f"  - {SESSION_ROLES['questioner']['label']} (Questioner): asks bridging questions, occasionally pushes back\n\n"
        f"VERIFY 8-PART STRUCTURE (all must be present):\n"
        f"  1. Channel Intro\n"
        f"  2. Hook (provocative question)\n"
        f"  3. Act 1 — The Claim\n"
        f"  4. Act 2 — The Evidence\n"
        f"  5. Act 3 — The Nuance\n"
        f"  6. Act 4 — The Protocol\n"
        f"  7. Wrap-up\n"
        f"  8. One Action Ending\n\n"
        + (f"CHANNEL INTRO VERIFICATION:\n"
           f"The Channel Intro MUST contain this EXACT text: \"{channel_intro}\"\n"
           f"Do NOT rephrase, modify, or remove it.\n\n"
           if channel_intro else '')
        + f"TRANSITION MARKERS:\n"
        f"Insert [TRANSITION] on its own line between major sections:\n"
        f"  - After Channel Intro, before Act 1\n"
        f"  - Between Act 1 and Act 2\n"
        f"  - Between Act 2 and Act 3\n"
        f"  - Between Act 3 and Act 4\n"
        f"  - After Act 4, before Wrap-up\n"
        f"These markers create musical transition moments in the final audio. Do NOT speak them.\n"
        f"Format: place [TRANSITION] on a line by itself between the last line of one act and the first line of the next.\n\n"
        f"ONE ACTION ENDING CHECK:\n"
        f"Verify the script ends with a single, specific, actionable recommendation.\n"
        f"If missing, add one based on the Protocol section (Act 4).\n\n"
        f"GRADE FRAMING CHECK:\n"
        f"Verify that claims use appropriate hedging language per confidence level.\n"
        f"Do NOT present LOW-confidence claims as settled fact.\n\n"
        f"Format:\n{SESSION_ROLES['presenter']['label']}: [dialogue]\n"
        f"{SESSION_ROLES['questioner']['label']}: [dialogue]\n\n"
        f"Remove meta-tags, markdown, stage directions. Dialogue only (plus [TRANSITION] markers).\n"
        f"- CRITICAL: Do NOT shorten or summarize. Output MUST be at least as long as the input. Add depth where possible.\n"
        + (f"\nCRITICAL: Output MUST be in Japanese (日本語) only. Do NOT switch to Chinese (中文). "
           f"Keep speaker labels exactly as 'Host 1:' and 'Host 2:' — do NOT replace them with Japanese names. "
           f"Avoid Kanji that is only used in Chinese (e.g., use 気 instead of 气, 楽 instead of 乐). "
           if language == 'ja' else '')
        + f"{target_instruction}"
    ),
    expected_output=(
        f"Final Masters-level dialogue about {topic_name}, exactly {target_script} {target_unit_plural}. "
        f"8-part structure with [TRANSITION] markers between acts. One Action ending present. "
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
        f"5. **Contested-as-settled**: Claims marked CONTESTED in source-of-truth presented as consensus\n"
        + (f"6. **Language consistency**: Flag any non-{language_config['name']} sentences that should be in {language_config['name']}. "
           f"(Exclude scientific abbreviations: ARR, NNT, GRADE, RCT, CI, HR, OR)\n\n"
           if language != 'en' else '\n')
        + f"OUTPUT FORMAT:\n"
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
        f"NOTE: HIGH-severity drift will trigger a script correction pass before audio generation. "
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

# --- Audience context for blueprint & script prompts ---
_audience_context = ""
if core_target:
    _audience_context += f"TARGET AUDIENCE: {core_target}\n"
if channel_mission:
    _audience_context += f"CHANNEL MISSION: {channel_mission}\n"

# Pre-built block for Listener Value Proposition section (avoids backslash in f-string expr)
_audience_context_block = (
    _audience_context + "Tailor the value proposition to this specific audience.\n"
) if _audience_context else ""

# Content framework hint based on channel mission
_framework_hint = ""
if channel_mission and any(kw in channel_mission.lower() for kw in ("actionable", "practical", "protocol", "how-to", "how to")):
    _framework_hint = "Note: The channel mission suggests PPP (Problem-Proof-Protocol) may be a good fit.\n"

blueprint_task = Task(
    description=(
        f"Create an Episode Blueprint for the podcast episode on \"{topic_name}\".\n\n"
        f"This is a CONTENT STRATEGY document that guides the script writer. It defines what the episode "
        f"will say, why listeners should care, and how to structure the narrative.\n\n"
        f"OUTPUT FORMAT — produce ALL 7 sections:\n\n"
        f"# Episode Blueprint: {topic_name}\n\n"
        f"## 1. Episode Thesis\n"
        f"One sentence: what this episode will prove or explore.\n\n"
        f"## 2. Listener Value Proposition\n"
        f"{_audience_context_block}"
        f"- What will the listener GAIN from this episode?\n"
        f"- Why should they listen to THIS episode instead of reading an article?\n"
        f"- What will they be able to DO differently after listening?\n\n"
        f"## 3. Hook\n"
        f"A provocative QUESTION for listeners based on the most surprising finding from the research.\n"
        f"The question should make listeners want to know the answer and feel personally relevant.\n"
        f"BAD: 'Have you ever wondered about coffee?' (too vague)\n"
        f"GOOD: 'What if your morning coffee habit was actually adding years to your life — but only if you drink exactly the right amount?'\n\n"
        f"## 4. Content Framework\n"
        f"{_framework_hint}"
        f"Choose ONE:\n"
        f"- [PPP] Problem-Proof-Protocol — if the topic has a clear actionable outcome\n"
        f"- [QEI] Question-Evidence-Insight — if the topic is exploratory with no single recommendation\n\n"
        f"## 5. Narrative Arc (4 Acts)\n"
        f"### Act 1 — The Claim (~20% of episode)\n"
        f"What people believe. The folk wisdom or common assumption. Why this matters personally.\n"
        f"Key points to cover: [3-4 bullets]\n\n"
        f"### Act 2 — The Evidence (~35% of episode)\n"
        f"What science actually says. Key studies from BOTH supporting and contradicting evidence.\n"
        f"Supporting evidence: [2-3 key studies with how to frame them]\n"
        f"Contradicting evidence: [1-2 key studies]\n"
        f"Key numbers to cite: [NNT, ARR, sample sizes if available]\n\n"
        f"### Act 3 — The Nuance (~25% of episode)\n"
        f"Where it gets complicated. Contested findings, population differences, dose-response, limitations.\n"
        f"Key nuance points: [2-3 bullets]\n\n"
        f"### Act 4 — The Protocol (~20% of episode)\n"
        f"Actionable translation to daily life.\n"
        f"'One Action' for the ending: [specific, memorable, doable this week]\n\n"
        f"## 6. GRADE-Informed Framing Guide\n"
        f"For each major claim in the episode, specify the appropriate framing language.\n"
        f"Use this mapping based on the evidence confidence:\n"
        f"- HIGH confidence → 'Research clearly demonstrates...'\n"
        f"- MODERATE confidence → 'Evidence suggests...'\n"
        f"- LOW confidence → 'Emerging research indicates...'\n"
        f"- VERY LOW confidence → 'Preliminary findings hint at...'\n"
        f"List each major claim with its recommended framing.\n\n"
        f"## 7. Citations\n"
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
        f"Episode Blueprint with all 7 sections: thesis, listener value proposition, hook, "
        f"content framework (PPP or QEI), 4-act narrative arc, GRADE framing guide, and citations. "
        f"{target_instruction}"
    ),
    agent=producer_agent,
    context=[],
    output_file=str(output_dir / "EPISODE_BLUEPRINT.md")
)

# --- CONTEXT CHAIN: script_task always depends on blueprint_task ---
script_task.context = [blueprint_task]

# --- SOT TRANSLATION PIPELINE: Update contexts when translating ---
if translation_task is not None:
    # Recording task writes script directly in target language using the translated SOT
    script_task.context = [blueprint_task, translation_task]
    # Blueprint uses the translated SOT as reference
    blueprint_task.context = [translation_task]
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
    'blueprint_task': {
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
        'dependencies': ['blueprint_task'],
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

        # Warn about missing critical clinical artifacts
        _critical_artifacts = ["grade_synthesis.md", "affirmative_case.md",
                               "falsification_case.md", "research_sources.json"]
        for _art in _critical_artifacts:
            if not (new_output_dir / _art).exists():
                print(f"  ⚠ Missing clinical artifact '{_art}' — pipeline will rely on SOT content only")

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
            f"{_truncate_at_boundary(sot_content, 8000)}\n"
            f"--- END PREVIOUS RESEARCH ---\n"
        )
        script_task.description = f"{script_task.description}{sot_injection}"
        blueprint_task.description = f"{blueprint_task.description}{sot_injection}"
        audit_task.description = f"{audit_task.description}{sot_injection}"

        # Update output_dir for file outputs
        # Reassign global output_dir so output_file paths work
        import builtins
        # Update task output_file paths to new dir
        for task_obj in [audit_task, blueprint_task]:
            if hasattr(task_obj, '_original_output_file') or hasattr(task_obj, 'output_file'):
                old_path = getattr(task_obj, 'output_file', '')
                if old_path:
                    filename = Path(old_path).name
                    task_obj.output_file = str(new_output_dir / filename)

        print(f"\nCREW 3: PODCAST PRODUCTION")

        _r_tl_summary = ""
        _r_sot_translated_file = None

        if translation_task is not None and sot_content:
            print(f"\nPHASE 3: REPORT TRANSLATION (pipelined)")
            translated_sot = _translate_sot_pipelined(sot_content, language, language_config)
            if translated_sot:
                _r_sot_translated_file = new_output_dir / f"source_of_truth_{language}.md"
                with open(_r_sot_translated_file, 'w', encoding='utf-8') as f:
                    f.write(translated_sot)
                print(f"✓ Translated SOT saved ({len(translated_sot)} chars)")
                # Generate compact summary of translated SOT for task description injection
                print("  Summarizing translated SOT for Crew 3 context injection...")
                _r_tl_summary = summarize_report_with_fast_model(translated_sot, "sot_translated", topic_name)
                if _r_tl_summary:
                    _r_tl_inj = _build_sot_injection_for_stage(
                        1, sot_path, _r_sot_translated_file,
                        _truncate_at_boundary(sot_content, 8000), _r_tl_summary, "", language_config
                    )
                    blueprint_task.description += _r_tl_inj
                    script_task.description += _r_tl_inj
                    audit_task.description += _r_tl_inj
                # CRITICAL: compact reference only — full 84KB SOT as context causes 27K+ tokens
                # → CrewAI context overflow → infinite summarizer loop (observed: 36 cycles, 9.6h wasted)
                from types import SimpleNamespace
                translation_task.output = SimpleNamespace(raw=(
                    f"[Translation complete — {len(translated_sot):,} chars]\n"
                    f"Translated SOT saved: {_r_sot_translated_file}\n"
                    f"Key research summary injected into task descriptions."
                ))
                sot_translated_file = _r_sot_translated_file  # keep for downstream artifact tracking

        # Extract base descriptions before translation for audit-loop feedback
        _reuse_script_base_desc = script_task.description
        _reuse_script_expected = script_task.expected_output
        _reuse_polish_base_desc = polish_task.description
        _reuse_polish_expected = polish_task.expected_output

        # Translate Crew 3 task prompts for non-English runs
        if language != 'en':
            for _task, _name in [
                (blueprint_task, "blueprint"), (script_task, "script"),
                (polish_task, "polish"), (audit_task, "audit"),
            ]:
                _task.description = _translate_prompt(_task.description, language, language_config)
            _reuse_script_base_desc = script_task.description
            _reuse_polish_base_desc = polish_task.description

        _REUSE_MAX_ATTEMPTS = 3

        # Phase 4: Blueprint
        print(f"\n  PHASE 4: EPISODE BLUEPRINT")
        _crew_kickoff_guarded(
            lambda: Crew(agents=[producer_agent], tasks=[blueprint_task], verbose=True),
            blueprint_task, translation_task, language,
            sot_path, _r_sot_translated_file,
            _truncate_at_boundary(sot_content, 8000), _r_tl_summary,
            "", language_config, "Phase 4 Blueprint"
        )

        # Phase 5: Script Draft + Expansion
        print(f"\n  PHASE 5: SCRIPT DRAFT + EXPANSION")
        Crew(agents=[producer_agent], tasks=[script_task], verbose=True).kickoff()
        _r_draft_text = re.sub(r'<think>.*?</think>', '', script_task.output.raw, flags=re.DOTALL).strip()
        _r_val = _validate_script(_r_draft_text, target_length_int, SCRIPT_TOLERANCE,
                                   language_config, sot_content, stage='draft')
        print(f"    Draft: {_r_val['word_count']} {language_config['length_unit']} — "
              f"{'PASS' if _r_val['pass'] else 'NEEDS EXPANSION'}")
        if not _r_val['pass'] and any('TOO SHORT' in i for i in _r_val['issues']):
            print("    Running per-act expansion...")
            _r_draft_text, _r_expanded = _run_script_expansion(
                _r_draft_text, sot_content, target_length_int,
                language_config, SESSION_ROLES, topic_name, target_instruction)
            _r_val = _validate_script(_r_draft_text, target_length_int, SCRIPT_TOLERANCE,
                                       language_config, sot_content, stage='draft')
            print(f"    Post-expansion: {_r_val['word_count']} {language_config['length_unit']} — "
                  f"{'PASS' if _r_val['pass'] else 'FAIL'}")
            if not _r_val['pass'] and any('TOO SHORT' in i for i in _r_val['issues']):
                _r_acts = _analyze_acts(_r_draft_text, language_config, target_length_int)
                _r_per_act_fb = "Per-act word counts:\n"
                for _ra in _r_acts:
                    if _ra['num'] not in (0, 99) and _ra['target'] > 0:
                        _r_per_act_fb += f"  {_ra['label']}: {_ra['count']}/{_ra['target']} {language_config['length_unit']}\n"
                _r_fb = (
                    f"\n\nPREVIOUS ATTEMPT FEEDBACK:\n{_r_val['feedback']}\n"
                    f"{_r_per_act_fb}\nExpand the shortest acts to hit their targets.\n"
                )
                _r_retry = Task(
                    description=_reuse_script_base_desc + _r_fb,
                    expected_output=_reuse_script_expected,
                    agent=producer_agent,
                    context=[blueprint_task],
                )
                if translation_task is not None:
                    _r_retry.context = [blueprint_task, translation_task]
                Crew(agents=[producer_agent], tasks=[_r_retry], verbose=True).kickoff()
                _r_retry_text = re.sub(r'<think>.*?</think>', '', _r_retry.output.raw, flags=re.DOTALL).strip()
                _r_retry_val = _validate_script(_r_retry_text, target_length_int, SCRIPT_TOLERANCE,
                                                 language_config, sot_content, stage='draft')
                if _r_retry_val['word_count'] > _r_val['word_count']:
                    _r_draft_text = _r_retry_text
                    script_task = _r_retry
        _r_expanded_count = _count_words(_r_draft_text, language_config)
        _r_current_script = script_task

        # Phase 6: Polish + Shrinkage Guard
        print(f"\n  PHASE 6: SCRIPT POLISH (audit loop)")
        _r_polish_feedback = ""
        _r_current_polish = polish_task
        _r_current_polish.context = [_r_current_script]
        if translation_task is not None:
            _r_current_polish.context = [_r_current_script, translation_task]
        for _r_attempt in range(1, _REUSE_MAX_ATTEMPTS + 1):
            if _r_polish_feedback:
                _r_fb = (
                    f"\n\nPREVIOUS ATTEMPT FEEDBACK (attempt {_r_attempt-1}):\n{_r_polish_feedback}\n"
                    f"Fix ALL issues listed above.\n"
                )
                _r_current_polish = Task(
                    description=_reuse_polish_base_desc + _r_fb,
                    expected_output=_reuse_polish_expected,
                    agent=editor_agent,
                    context=[_r_current_script],
                )
                if translation_task is not None:
                    _r_current_polish.context = [_r_current_script, translation_task]
            Crew(agents=[editor_agent], tasks=[_r_current_polish], verbose=True).kickoff()
            _r_polished = re.sub(r'<think>.*?</think>', '', _r_current_polish.output.raw, flags=re.DOTALL).strip()
            _r_val = _validate_script(_r_polished, target_length_int, SCRIPT_TOLERANCE,
                                       language_config, sot_content, stage='polish')
            print(f"    Polish attempt {_r_attempt}: {_r_val['word_count']} {language_config['length_unit']} — "
                  f"{'PASS' if _r_val['pass'] else 'FAIL'}")
            if _r_val['pass']:
                break
            _r_polish_feedback = _r_val['feedback']
        # Shrinkage guard
        _r_polished_count = _count_words(_r_polished, language_config)
        if _r_expanded_count > 0 and _r_polished_count < _r_expanded_count * 0.90:
            print(f"    ⚠ Polish shrunk script ({_r_expanded_count} → {_r_polished_count}) — using expanded draft")
            if _r_draft_text.count('[TRANSITION]') < 3:
                _r_draft_text = re.sub(r'(---\s*\n###\s*\*?\*?ACT)', r'[TRANSITION]\n\n\1', _r_draft_text)
            _r_polished = _r_draft_text
        polish_task = _r_current_polish

        # Phase 7: Accuracy Audit
        print(f"\n  PHASE 7: ACCURACY AUDIT")
        audit_task.context = [polish_task]
        if translation_task is not None:
            audit_task.context = [polish_task, translation_task]
        Crew(agents=[auditor_agent], tasks=[audit_task], verbose=True).kickoff()

        # Save markdown outputs
        print("\n--- Saving Outputs ---")
        # Use _r_polished (may be expanded draft if shrinkage guard fired)
        script_text = _r_polished if _r_polished else (
            polish_task.output.raw if hasattr(polish_task, 'output') and polish_task.output else "")
        # Post-script language audit for reuse mode
        if language != 'en':
            script_text = _audit_script_language(script_text, language, language_config)

        # Save script_final.md (authoritative for TTS)
        with open(new_output_dir / "script_final.md", 'w', encoding='utf-8') as f:
            f.write(script_text)

        for label, source, filename in [
            ("Source of Truth (Translated)", translation_task, "source_of_truth.md"),
            ("Episode Blueprint", blueprint_task, "EPISODE_BLUEPRINT.md"),
            ("Script Draft", script_task, "script_draft.md"),
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
            f.write(cleaned_script)

        audio_output_path = new_output_dir / "audio.wav"
        tts_result = generate_audio_from_script(cleaned_script, str(audio_output_path), lang_code=language_config['tts_code'])
        if isinstance(tts_result, tuple):
            audio_file_path, transition_positions = tts_result
        else:
            audio_file_path, transition_positions = tts_result, []
        if audio_file_path:
            audio_file = Path(audio_file_path)
            print(f"Audio generation complete: {audio_file}")
            print(f"Starting BGM Merging Phase...")
            try:
                mastered = post_process_audio(str(audio_file), bgm_target="Interesting BGM.wav",
                                              transition_positions_ms=transition_positions)
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
                f"{_truncate_at_boundary(sot_content, 8000)}\n"
                f"--- END SOURCE OF TRUTH ---\n"
            )
            script_task.description = f"{script_task.description}{sot_injection}"
            blueprint_task.description = f"{blueprint_task.description}{sot_injection}"
            audit_task.description = f"{audit_task.description}{sot_injection}"

            # Update output_file paths
            for task_obj in [audit_task, blueprint_task]:
                old_path = getattr(task_obj, 'output_file', '')
                if old_path:
                    filename = Path(old_path).name
                    task_obj.output_file = str(new_output_dir / filename)

            print(f"\nCREW 3: PODCAST PRODUCTION")

            _s_tl_summary = ""
            _s_sot_translated_file = None

            if translation_task is not None and sot_content:
                print(f"\nPHASE 3: REPORT TRANSLATION (pipelined)")
                translated_sot = _translate_sot_pipelined(sot_content, language, language_config)
                if translated_sot:
                    _s_sot_translated_file = new_output_dir / f"source_of_truth_{language}.md"
                    with open(_s_sot_translated_file, 'w', encoding='utf-8') as f:
                        f.write(translated_sot)
                    print(f"✓ Translated SOT saved ({len(translated_sot)} chars)")
                    # Generate compact summary of translated SOT for task description injection
                    print("  Summarizing translated SOT for Crew 3 context injection...")
                    _s_tl_summary = summarize_report_with_fast_model(translated_sot, "sot_translated", topic_name)
                    if _s_tl_summary:
                        _s_tl_inj = _build_sot_injection_for_stage(
                            1, sot_path, _s_sot_translated_file,
                            _truncate_at_boundary(sot_content, 8000), _s_tl_summary, "", language_config
                        )
                        blueprint_task.description += _s_tl_inj
                        script_task.description += _s_tl_inj
                        audit_task.description += _s_tl_inj
                    # CRITICAL: compact reference only — full SOT as context causes 27K+ token overflow
                    from types import SimpleNamespace
                    translation_task.output = SimpleNamespace(raw=(
                        f"[Translation complete — {len(translated_sot):,} chars]\n"
                        f"Translated SOT saved: {_s_sot_translated_file}\n"
                        f"Key research summary injected into task descriptions."
                    ))
                    sot_translated_file = _s_sot_translated_file  # keep for artifact tracking

            # Extract base descriptions before translation for audit-loop feedback
            _supp_script_base_desc = script_task.description
            _supp_script_expected = script_task.expected_output
            _supp_polish_base_desc = polish_task.description
            _supp_polish_expected = polish_task.expected_output

            # Translate Crew 3 task prompts for non-English runs
            if language != 'en':
                for _task, _name in [
                    (blueprint_task, "blueprint"), (script_task, "script"),
                    (polish_task, "polish"), (audit_task, "audit"),
                ]:
                    _task.description = _translate_prompt(_task.description, language, language_config)
                _supp_script_base_desc = script_task.description
                _supp_polish_base_desc = polish_task.description

            _SUPP_MAX_ATTEMPTS = 3

            # Phase 4: Blueprint
            print(f"\n  PHASE 4: EPISODE BLUEPRINT")
            _crew_kickoff_guarded(
                lambda: Crew(agents=[producer_agent], tasks=[blueprint_task], verbose=True),
                blueprint_task, translation_task, language,
                sot_path, _s_sot_translated_file,
                _truncate_at_boundary(sot_content, 8000), _s_tl_summary,
                "", language_config, "Phase 4 Blueprint"
            )

            # Phase 5: Script Draft + Expansion
            print(f"\n  PHASE 5: SCRIPT DRAFT + EXPANSION")
            Crew(agents=[producer_agent], tasks=[script_task], verbose=True).kickoff()
            _s_draft_text = re.sub(r'<think>.*?</think>', '', script_task.output.raw, flags=re.DOTALL).strip()
            _s_val = _validate_script(_s_draft_text, target_length_int, SCRIPT_TOLERANCE,
                                       language_config, sot_content, stage='draft')
            print(f"    Draft: {_s_val['word_count']} {language_config['length_unit']} — "
                  f"{'PASS' if _s_val['pass'] else 'NEEDS EXPANSION'}")
            if not _s_val['pass'] and any('TOO SHORT' in i for i in _s_val['issues']):
                print("    Running per-act expansion...")
                _s_draft_text, _s_expanded = _run_script_expansion(
                    _s_draft_text, sot_content, target_length_int,
                    language_config, SESSION_ROLES, topic_name, target_instruction)
                _s_val = _validate_script(_s_draft_text, target_length_int, SCRIPT_TOLERANCE,
                                           language_config, sot_content, stage='draft')
                print(f"    Post-expansion: {_s_val['word_count']} {language_config['length_unit']} — "
                      f"{'PASS' if _s_val['pass'] else 'FAIL'}")
                if not _s_val['pass'] and any('TOO SHORT' in i for i in _s_val['issues']):
                    _s_acts = _analyze_acts(_s_draft_text, language_config, target_length_int)
                    _s_per_act_fb = "Per-act word counts:\n"
                    for _sa in _s_acts:
                        if _sa['num'] not in (0, 99) and _sa['target'] > 0:
                            _s_per_act_fb += f"  {_sa['label']}: {_sa['count']}/{_sa['target']} {language_config['length_unit']}\n"
                    _s_fb = (
                        f"\n\nPREVIOUS ATTEMPT FEEDBACK:\n{_s_val['feedback']}\n"
                        f"{_s_per_act_fb}\nExpand the shortest acts to hit their targets.\n"
                    )
                    _s_retry = Task(
                        description=_supp_script_base_desc + _s_fb,
                        expected_output=_supp_script_expected,
                        agent=producer_agent,
                        context=[blueprint_task],
                    )
                    if translation_task is not None:
                        _s_retry.context = [blueprint_task, translation_task]
                    Crew(agents=[producer_agent], tasks=[_s_retry], verbose=True).kickoff()
                    _s_retry_text = re.sub(r'<think>.*?</think>', '', _s_retry.output.raw, flags=re.DOTALL).strip()
                    _s_retry_val = _validate_script(_s_retry_text, target_length_int, SCRIPT_TOLERANCE,
                                                     language_config, sot_content, stage='draft')
                    if _s_retry_val['word_count'] > _s_val['word_count']:
                        _s_draft_text = _s_retry_text
                        script_task = _s_retry
            _s_expanded_count = _count_words(_s_draft_text, language_config)
            _s_cur_script = script_task

            # Phase 6: Polish + Shrinkage Guard
            print(f"\n  PHASE 6: SCRIPT POLISH (audit loop)")
            _s_polish_fb = ""
            _s_cur_polish = polish_task
            _s_cur_polish.context = [_s_cur_script]
            if translation_task is not None:
                _s_cur_polish.context = [_s_cur_script, translation_task]
            for _s_att in range(1, _SUPP_MAX_ATTEMPTS + 1):
                if _s_polish_fb:
                    _s_fb = (
                        f"\n\nPREVIOUS ATTEMPT FEEDBACK (attempt {_s_att-1}):\n{_s_polish_fb}\n"
                        f"Fix ALL issues listed above.\n"
                    )
                    _s_cur_polish = Task(
                        description=_supp_polish_base_desc + _s_fb,
                        expected_output=_supp_polish_expected,
                        agent=editor_agent,
                        context=[_s_cur_script],
                    )
                    if translation_task is not None:
                        _s_cur_polish.context = [_s_cur_script, translation_task]
                Crew(agents=[editor_agent], tasks=[_s_cur_polish], verbose=True).kickoff()
                _s_polished = re.sub(r'<think>.*?</think>', '', _s_cur_polish.output.raw, flags=re.DOTALL).strip()
                _s_val = _validate_script(_s_polished, target_length_int, SCRIPT_TOLERANCE,
                                           language_config, sot_content, stage='polish')
                print(f"    Polish attempt {_s_att}: {_s_val['word_count']} {language_config['length_unit']} — "
                      f"{'PASS' if _s_val['pass'] else 'FAIL'}")
                if _s_val['pass']:
                    break
                _s_polish_fb = _s_val['feedback']
            # Shrinkage guard
            _s_polished_count = _count_words(_s_polished, language_config)
            if _s_expanded_count > 0 and _s_polished_count < _s_expanded_count * 0.90:
                print(f"    ⚠ Polish shrunk script ({_s_expanded_count} → {_s_polished_count}) — using expanded draft")
                if _s_draft_text.count('[TRANSITION]') < 3:
                    _s_draft_text = re.sub(r'(---\s*\n###\s*\*?\*?ACT)', r'[TRANSITION]\n\n\1', _s_draft_text)
                _s_polished = _s_draft_text
            polish_task = _s_cur_polish

            # Phase 7: Accuracy Audit
            print(f"\n  PHASE 7: ACCURACY AUDIT")
            audit_task.context = [polish_task]
            if translation_task is not None:
                audit_task.context = [polish_task, translation_task]
            Crew(agents=[auditor_agent], tasks=[audit_task], verbose=True).kickoff()

            # Save outputs
            print("\n--- Saving Outputs ---")
            # Use _s_polished (may be expanded draft if shrinkage guard fired)
            script_text = _s_polished if _s_polished else (
                polish_task.output.raw if hasattr(polish_task, 'output') and polish_task.output else "")
            # Post-script language audit for supplemental reuse mode
            if language != 'en':
                script_text = _audit_script_language(script_text, language, language_config)

            # Save script_final.md (authoritative for TTS)
            with open(new_output_dir / "script_final.md", 'w', encoding='utf-8') as f:
                f.write(script_text)

            for label, source, filename in [
                ("Source of Truth (Translated)", translation_task, "source_of_truth.md"),
                ("Episode Blueprint", blueprint_task, "EPISODE_BLUEPRINT.md"),
                ("Script Draft", script_task, "script_draft.md"),
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
                f.write(cleaned_script)

            audio_output_path = new_output_dir / "audio.wav"
            tts_result = generate_audio_from_script(cleaned_script, str(audio_output_path), lang_code=language_config['tts_code'])
            if isinstance(tts_result, tuple):
                audio_file_path, transition_positions = tts_result
            else:
                audio_file_path, transition_positions = tts_result, []
            if audio_file_path:
                audio_file = Path(audio_file_path)
                print(f"Audio generation complete: {audio_file}")
                print(f"Starting BGM Merging Phase...")
                try:
                    mastered = post_process_audio(str(audio_file), bgm_target="Interesting BGM.wav",
                                                  transition_positions_ms=transition_positions)
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
    blueprint_task,
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
# PHASE 0: RESEARCH FRAMING (includes domain classification)
# ================================================================
print(f"\n{'='*70}")
print(f"PHASE 0: RESEARCH FRAMING")
print(f"{'='*70}")

# Step 0a: classify domain first (fast, mostly deterministic) so framing is domain-aware
from domain_classifier import classify_topic, ResearchDomain
_smart_base = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
_smart_model = os.environ.get("MODEL_NAME", "")
try:
    from openai import AsyncOpenAI as _AOAIClassify
    _classify_client = _AOAIClassify(base_url=_smart_base, api_key="not-needed")
except Exception:
    _classify_client = None
domain_classification = asyncio.run(classify_topic(
    topic=topic_name,
    smart_client=_classify_client,
    smart_model=_smart_model,
))
print(f"  Domain: {domain_classification.domain.value} "
      f"(confidence={domain_classification.confidence:.2f}, framework={domain_classification.suggested_framework})")
_dc_path = output_dir / "domain_classification.json"
_dc_path.write_text(json.dumps({
    "domain": domain_classification.domain.value,
    "confidence": domain_classification.confidence,
    "reasoning": domain_classification.reasoning,
    "framework": domain_classification.suggested_framework,
    "databases": domain_classification.primary_databases,
}, indent=2))

# Step 0b: run framing agent with domain context injected
if domain_classification.domain == ResearchDomain.SOCIAL_SCIENCE:
    _domain_framing_note = (
        f"\n\nDOMAIN CONTEXT: This is a SOCIAL SCIENCE topic. "
        f"Use PECO framework (Population, Exposure, Comparison, Outcome). "
        f"Prioritise effect sizes (Cohen's d, Hedges' g), quasi-experimental designs, "
        f"and databases such as {', '.join(domain_classification.primary_databases)}. "
        f"Do NOT use clinical terminology (NNT, ARR, GRADE, MeSH terms)."
    )
else:
    _domain_framing_note = (
        f"\n\nDOMAIN CONTEXT: This is a CLINICAL/HEALTH topic. "
        f"Use PICO framework (Population, Intervention, Comparison, Outcome). "
        f"Prioritise RCTs, systematic reviews, GRADE evidence levels, NNT/ARR statistics, "
        f"and databases such as {', '.join(domain_classification.primary_databases)}."
    )
framing_task.description += _domain_framing_note

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
# PHASE 1: RESEARCH PIPELINE (domain-routed)
# ================================================================
_research_domain = domain_classification.domain.value  # "clinical" | "social_science" | "general"

if _research_domain in ("social_science",):
    # Social science pipeline (Phase E implementation)
    print(f"\n{'='*70}")
    print(f"PHASE 1: SOCIAL SCIENCE RESEARCH (PECO Pipeline)")
    print(f"{'='*70}")
    try:
        from social_science_research import run_social_science_research
        deep_reports = asyncio.run(run_social_science_research(
            topic=topic_name,
            framing_context=framing_output,
            output_dir=str(output_dir),
        ))
    except ImportError:
        print("⚠ social_science_research module not yet available — falling back to clinical pipeline")
        _research_domain = "clinical"  # fall through to clinical below
    except Exception as e:
        print(f"⚠ Social science pipeline failed: {e} — falling back to clinical pipeline")
        _research_domain = "clinical"

if _research_domain in ("clinical", "general"):
    print(f"\n{'='*70}")
    print(f"PHASE 1: CLINICAL RESEARCH")
    print(f"{'='*70}")

brave_key = os.getenv("BRAVE_API_KEY", "")

# Check if the configured fast model (FAST_MODEL_NAME from .env) is available
_fast_model_name = os.environ.get("FAST_MODEL_NAME", "")
_fast_base_url = os.environ.get("FAST_LLM_BASE_URL", "http://localhost:11434/v1")
fast_model_available = False
try:
    _resp = httpx.get(f"{_fast_base_url}/models", timeout=3)
    if _resp.status_code == 200:
        _models = [m.get("id", "") for m in _resp.json().get("data", [])]
        fast_model_available = _fast_model_name in _models
        if fast_model_available:
            print(f"✓ Fast model ready: {_fast_model_name}")
        else:
            print(f"⚠ Fast model '{_fast_model_name}' not found in Ollama. Available: {_models}")
            print(f"  Falling back to smart-only mode. Run: ollama pull {_fast_model_name}")
except Exception:
    print(f"⚠ Fast model not available (Ollama unreachable at {_fast_base_url}). Running in smart-only mode.")

sot_content = ""  # Will hold the synthesized Source-of-Truth
sot_file = None   # Path to source_of_truth.md (set after deep research completes)
sot_summary = ""  # Fast-model summary of sot_content (set after deep research completes)
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
    for role_name, filename in REPORT_FILENAMES.items():
        report = deep_reports.get(role_name)
        if not report:
            print(f"  ⚠ {role_name.capitalize()} report missing — skipping save")
            continue
        report_file = output_dir / filename
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

    deep_audit_report = deep_reports.get("audit")
    lead_report = deep_reports.get("lead")
    counter_report = deep_reports.get("counter")

    # ================================================================
    # Build Source-of-Truth in IMRaD format from deep research outputs
    # ================================================================
    def _extract_conclusion_status(grade_report: str, domain: str = "clinical") -> tuple:
        """Extract evidence level, conclusion status, and executive summary.

        Supports both GRADE (clinical) and Evidence Quality (social science) levels.
        """
        if domain == "social_science":
            # Social science evidence quality levels
            m = re.search(
                r'Final\s+Evidence\s+Quality[:\s]*\*{0,2}(STRONG|MODERATE_STRONG|MODERATE_WEAK|MODERATE|WEAK|VERY_WEAK)\*{0,2}',
                grade_report, re.IGNORECASE)
            grade = m.group(1).strip().upper() if m else "Not Determined"
            status_map = {
                "STRONG": "Scientifically Supported",
                "MODERATE_STRONG": "Well Supported — Robust Evidence",
                "MODERATE": "Partially Supported — Further Research Recommended",
                "MODERATE_WEAK": "Tentatively Supported — More Rigorous Studies Needed",
                "WEAK": "Insufficient Evidence — More Research Needed",
                "VERY_WEAK": "Not Supported by Current Evidence",
            }
        else:
            # Clinical GRADE levels
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

    def _parse_grade_sections(audit_text: str) -> dict:
        """Split GRADE synthesis text into named subsections by ### headers."""
        sections = {}
        current_key = None
        current_lines = []
        for line in audit_text.split('\n'):
            if line.startswith('### '):
                if current_key is not None:
                    sections[current_key] = '\n'.join(current_lines).strip()
                current_key = line.lstrip('#').strip().lower()
                current_lines = []
            else:
                current_lines.append(line)
        if current_key is not None:
            sections[current_key] = '\n'.join(current_lines).strip()
        return sections

    def _format_study_characteristics_table(extractions: list) -> str:
        """Build a study characteristics table from DeepExtraction objects."""
        if not extractions:
            return "*No studies with full extraction data available.*\n"
        # Check if any extraction has enrichment metadata
        has_metadata = any(getattr(ext, 'paper_metadata', None) for ext in extractions)
        if has_metadata:
            rows = ["| # | Study | Design | N | Demographics | Follow-up | Funding | Bias Risk | Citations | FWCI | Tier |",
                    "|---|-------|--------|---|--------------|-----------|---------|-----------|-----------|------|------|"]
        else:
            rows = ["| # | Study | Design | N | Demographics | Follow-up | Funding | Bias Risk | Tier |",
                    "|---|-------|--------|---|--------------|-----------|---------|-----------|------|"]
        seen = set()
        idx = 0
        for ext in extractions:
            key = ext.pmid or ext.doi or ext.title
            if key in seen:
                continue
            seen.add(key)
            idx += 1
            label = f"{ext.title[:50]}{'…' if len(ext.title) > 50 else ''}"
            if ext.pmid:
                label += f" ([PMID:{ext.pmid}](https://pubmed.ncbi.nlm.nih.gov/{ext.pmid}/))"
            tier_label = f"T{ext.research_tier}" if getattr(ext, 'research_tier', None) else "N/A"
            base = (
                f"| {idx} "
                f"| {label} "
                f"| {ext.study_design or 'N/A'} "
                f"| {ext.sample_size_total or 'N/A'} "
                f"| {(ext.demographics or 'N/A')[:40]} "
                f"| {ext.follow_up_period or 'N/A'} "
                f"| {(ext.funding_source or 'N/A')[:30]} "
                f"| {ext.risk_of_bias or 'N/A'} "
            )
            if has_metadata:
                pm = getattr(ext, 'paper_metadata', None)
                cite_str = str(pm.citation_count) if pm and pm.citation_count is not None else "N/A"
                fwci_str = f"{pm.fwci:.1f}" if pm and pm.fwci is not None else "N/A"
                base += f"| {cite_str} | {fwci_str} "
            base += f"| {tier_label} |"
            rows.append(base)
        return '\n'.join(rows) + '\n'

    def _format_references(extractions: list, wide_net_records: list) -> str:
        """Build a numbered reference list from extraction metadata enriched by WideNetRecords."""
        wnr_by_pmid = {r.pmid: r for r in wide_net_records if r.pmid}
        wnr_by_title = {r.title.lower().strip(): r for r in wide_net_records if r.title}
        refs = []
        seen = set()
        idx = 0
        for ext in extractions:
            key = ext.pmid or ext.doi or ext.title
            if key in seen:
                continue
            seen.add(key)
            idx += 1
            wnr = wnr_by_pmid.get(ext.pmid) or wnr_by_title.get((ext.title or "").lower().strip())
            authors = (wnr.authors if wnr and wnr.authors else "").strip() or "Unknown authors"
            journal = (wnr.journal if wnr and wnr.journal else "").strip()
            year = wnr.year if wnr and wnr.year else ""
            title = ext.title or "Untitled"
            parts = [f"{idx}. {authors}."]
            parts.append(f"*{title}*.")
            if journal:
                parts.append(f"{journal}.")
            if year:
                parts.append(f"({year}).")
            if ext.pmid:
                parts.append(f"PMID: [{ext.pmid}](https://pubmed.ncbi.nlm.nih.gov/{ext.pmid}/).")
            if ext.doi:
                parts.append(f"DOI: {ext.doi}.")
            refs.append(" ".join(parts))
        return '\n'.join(refs) + '\n' if refs else "*No references available.*\n"

    def _build_social_science_sot(
        topic, pd, audit_text, aff_case_text, fal_case_text,
        ev_quality, aff_cand, all_extractions, all_wide, impacts, metrics,
        framing, search_date, aff_strategy, fal_strategy,
    ) -> str:
        """Build IMRaD SOT for social science topics (PECO, effect sizes, evidence quality)."""
        from effect_size_math import EffectSizeImpact

        grade_level, conclusion_status, exec_summary = _extract_conclusion_status(
            audit_text, domain="social_science"
        )

        # Extract metrics
        aff_wide = metrics.get("aff_wide_net_total", 0)
        fal_wide = metrics.get("fal_wide_net_total", 0)
        total_wide = aff_wide + fal_wide
        total_screened = metrics.get("aff_screened_in", 0) + metrics.get("fal_screened_in", 0)
        total_ft_ok = metrics.get("aff_fulltext_ok", 0) + metrics.get("fal_fulltext_ok", 0)

        out = []

        # --- Abstract ---
        out.append(f"# Source of Truth: {topic}\n")
        out.append("## 1. Abstract\n")

        # Research question (PECO)
        peco = {}
        if aff_strategy and hasattr(aff_strategy, 'peco'):
            peco = aff_strategy.peco if isinstance(aff_strategy.peco, dict) else getattr(aff_strategy, 'peco', {})
        elif isinstance(aff_strategy, dict):
            peco = aff_strategy.get("peco", {})
        if peco:
            out.append(f"**Research Question (PECO):** In {peco.get('P', 'the target population')}, "
                       f"does exposure to {peco.get('E', 'the intervention')} compared to "
                       f"{peco.get('C', 'no exposure')} affect {peco.get('O', 'outcomes')}?\n")

        out.append(f"**Methods:** Systematic search of OpenAlex, ERIC, and Google Scholar identified "
                   f"{total_wide} records. After screening, {total_screened} studies were selected and "
                   f"{total_ft_ok} were fully extracted using the PECO framework.\n")

        # Key finding (effect sizes)
        if impacts:
            es_list = [i for i in impacts if isinstance(i, EffectSizeImpact)]
            if es_list:
                avg_d = sum(abs(i.cohens_d or 0) for i in es_list) / len(es_list)
                magnitude = "negligible" if avg_d < 0.2 else "small" if avg_d < 0.5 else "medium" if avg_d < 0.8 else "large"
                out.append(f"**Key Finding:** Across {len(es_list)} studies with reported effect sizes, "
                           f"the average magnitude was {magnitude} (mean |d| = {avg_d:.3f}).\n")

        out.append(f"**Evidence Quality:** {grade_level} — {conclusion_status}\n")
        if exec_summary:
            out.append(f"**Executive Summary:** {exec_summary}\n")

        # --- Introduction ---
        out.append("\n## 2. Introduction\n")
        if framing:
            out.append(f"{framing}\n")
        else:
            out.append(f"This review examines the evidence for: *{topic}*.\n")
        out.append("This review employs a dual-hypothesis design with parallel affirmative and "
                   "falsification research tracks, using the PECO (Population, Exposure, Comparison, Outcome) framework.\n")

        # --- Methods ---
        out.append("\n## 3. Methods\n")
        out.append("### 3.1 Search Strategy\n")
        out.append(f"**Framework:** PECO (Population, Exposure, Comparison, Outcome)\n")
        if peco:
            out.append(f"- **P (Population):** {peco.get('P', 'Not specified')}\n")
            out.append(f"- **E (Exposure):** {peco.get('E', 'Not specified')}\n")
            out.append(f"- **C (Comparison):** {peco.get('C', 'Not specified')}\n")
            out.append(f"- **O (Outcome):** {peco.get('O', 'Not specified')}\n")

        out.append(f"\n### 3.2 Data Collection\n")
        out.append(f"**Databases:** OpenAlex, ERIC (IES), Google Scholar\n")
        out.append(f"**Search date:** {search_date}\n")
        out.append(f"**Records identified:** {total_wide}\n")
        out.append(f"**Screened:** {total_screened}\n")
        out.append(f"**Extracted:** {total_ft_ok}\n")

        out.append(f"\n### 3.3 Statistical Analysis\n")
        out.append("Effect sizes were standardized to Cohen's d using deterministic Python calculations. "
                   "Hedges' g correction was applied where sample sizes were available. "
                   "Odds ratios and correlation coefficients were converted to d for comparability.\n")

        # --- Results ---
        out.append("\n## 4. Results\n")
        out.append("### 4.1 Study Characteristics\n")
        if all_extractions:
            has_metadata = any(getattr(ext, 'paper_metadata', None) for ext in all_extractions)
            rows = ["| # | Study | Design | N | Setting | Demographics | Effect Size | Follow-up | Tier |",
                    "|---|-------|--------|---|---------|--------------|-------------|-----------|------|"]
            seen = set()
            idx = 0
            for ext in all_extractions:
                key = getattr(ext, 'doi', None) or getattr(ext, 'title', '')
                if key in seen:
                    continue
                seen.add(key)
                idx += 1
                title_str = (getattr(ext, 'title', '') or '')[:50]
                es_val = getattr(ext, 'effect_size_value', None)
                es_type = getattr(ext, 'effect_size_type', None)
                es_str = f"{es_type}={es_val}" if es_val is not None else "N/A"
                setting = (getattr(ext, 'setting', None) or "N/A")[:30]
                demo = (getattr(ext, 'demographics', None) or "N/A")[:30]
                design = getattr(ext, 'study_design', None) or "N/A"
                n = getattr(ext, 'sample_size_total', None) or "N/A"
                fu = getattr(ext, 'follow_up_period', None) or "N/A"
                tier = f"T{getattr(ext, 'research_tier', 'N/A')}" if getattr(ext, 'research_tier', None) else "N/A"
                rows.append(f"| {idx} | {title_str} | {design} | {n} | {setting} | {demo} | {es_str} | {fu} | {tier} |")
            out.append('\n'.join(rows) + '\n')
        else:
            out.append("*No studies with full extraction data available.*\n")

        out.append("\n### 4.2 Effect Size Analysis\n")
        math_report = pd.get("math_report", "")
        if math_report:
            out.append(f"{math_report}\n")
        else:
            out.append("*No effect sizes calculated.*\n")

        # --- Discussion ---
        out.append("\n## 5. Discussion\n")
        out.append("### 5.1 Affirmative Case\n")
        if aff_case_text:
            out.append(f"{aff_case_text}\n")
        out.append("\n### 5.2 Falsification Case\n")
        if fal_case_text:
            out.append(f"{fal_case_text}\n")

        out.append("\n### 5.3 Evidence Quality Synthesis\n")
        if audit_text:
            out.append(f"{audit_text}\n")

        # --- References ---
        out.append("\n## 6. References\n")
        if all_extractions:
            seen = set()
            idx = 0
            for ext in all_extractions:
                key = getattr(ext, 'doi', None) or getattr(ext, 'title', '')
                if key in seen:
                    continue
                seen.add(key)
                idx += 1
                title = getattr(ext, 'title', 'Untitled') or 'Untitled'
                doi = getattr(ext, 'doi', None)
                parts = [f"{idx}. *{title}*."]
                if doi:
                    parts.append(f"DOI: {doi}.")
                url = getattr(ext, 'url', None)
                if url:
                    parts.append(f"URL: {url}")
                out.append(" ".join(parts))
            out.append("")
        else:
            out.append("*No references available.*\n")

        return '\n'.join(out)

    def build_imrad_sot(
        topic: str,
        reports: dict,
        ev_quality: str,
        aff_cand: int,
        domain: str = "clinical",
    ) -> str:
        """Assemble the Source of Truth document in IMRaD scientific paper format.

        Args:
            domain: "clinical" or "social_science" — controls framework terminology
        """
        pd = reports.get("pipeline_data", {})
        # Auto-detect domain from pipeline_data if not explicitly set
        if pd.get("domain") == "social_science":
            domain = "social_science"
        aff_strategy = pd.get("aff_strategy")
        fal_strategy = pd.get("fal_strategy")
        aff_extractions = pd.get("aff_extractions", [])
        fal_extractions = pd.get("fal_extractions", [])
        aff_top = pd.get("aff_top", [])
        fal_top = pd.get("fal_top", [])
        impacts = pd.get("impacts", [])
        framing = pd.get("framing_context", "")
        search_date = pd.get("search_date", "")
        metrics = pd.get("metrics", {})
        all_extractions = aff_extractions + fal_extractions
        all_wide = aff_top + fal_top

        _empty_rpt = type('_E', (), {'report': '', 'total_summaries': 0, 'total_urls_fetched': 0, 'duration_seconds': 0, 'sources': []})()
        audit_text = reports.get("audit", _empty_rpt).report
        aff_case_text = reports.get("lead", _empty_rpt).report
        fal_case_text = reports.get("counter", _empty_rpt).report

        # Dispatch to domain-specific SOT builder
        if domain == "social_science":
            return _build_social_science_sot(
                topic, pd, audit_text, aff_case_text, fal_case_text,
                ev_quality, aff_cand, all_extractions, all_wide, impacts, metrics,
                framing, search_date, aff_strategy, fal_strategy,
            )

        grade_level, conclusion_status, exec_summary = _extract_conclusion_status(audit_text)
        grade_sections = _parse_grade_sections(audit_text)

        m = metrics
        aff_wide = m.get("aff_wide_net_total", 0)
        fal_wide = m.get("fal_wide_net_total", 0)
        aff_screened = m.get("aff_screened_in", 0)
        fal_screened = m.get("fal_screened_in", 0)
        aff_ft_ok = m.get("aff_fulltext_ok", 0)
        fal_ft_ok = m.get("fal_fulltext_ok", 0)
        aff_ft_err = m.get("aff_fulltext_err", 0)
        fal_ft_err = m.get("fal_fulltext_err", 0)
        total_wide = aff_wide + fal_wide
        total_screened = aff_screened + fal_screened
        total_ft_ok = aff_ft_ok + fal_ft_ok
        total_ft_err = aff_ft_err + fal_ft_err

        # Summarize PICO for abstract
        pico_summary = ""
        if aff_strategy and hasattr(aff_strategy, 'pico'):
            p = aff_strategy.pico
            pico_summary = (
                f"**P** (Population): {p.get('population', 'N/A')}  \n"
                f"**I** (Intervention): {p.get('intervention', 'N/A')}  \n"
                f"**C** (Comparison): {p.get('comparison', 'N/A')}  \n"
                f"**O** (Outcome): {p.get('outcome', 'N/A')}"
            )

        # Determine representative NNT for abstract
        nnt_summary = ""
        if impacts:
            benefit = [i for i in impacts if i.direction == "benefit"]
            ref_impact = benefit[0] if benefit else impacts[0]
            nnt_summary = (
                f"The primary quantitative finding is NNT = **{ref_impact.nnt:.1f}** "
                f"({ref_impact.direction}; ARR = {ref_impact.arr:+.4f})."
            )

        # ── ABSTRACT ─────────────────────────────────────────────────────────
        out = [f"# Source of Truth: {topic}\n"]
        out.append("## Abstract\n")
        if pico_summary:
            out.append(f"**Clinical Question (PICO):**  \n{pico_summary}\n")
        out.append(
            f"**Methods:** Dual-track systematic review using parallel Affirmative and "
            f"Falsification search strategies. A total of {total_wide} records were identified "
            f"across PubMed and Google Scholar; {total_screened} were screened to top candidates "
            f"per track; {total_ft_ok} full-text articles were retrieved and deeply extracted. "
            f"Absolute Risk Reduction (ARR) and Number Needed to Treat (NNT) were calculated "
            f"deterministically in Python (no LLM). Evidence quality was assessed using the "
            f"GRADE framework.\n"
        )
        if nnt_summary:
            out.append(f"**Key Finding:** {nnt_summary}\n")
        out.append(
            f"**Evidence Quality (GRADE):** {grade_level}  \n"
            f"**Conclusion:** {conclusion_status}\n"
        )
        if exec_summary:
            out.append(f"\n{exec_summary}\n")

        # ── 1. INTRODUCTION ───────────────────────────────────────────────────
        out.append("\n---\n\n## 1. Introduction\n")
        if framing:
            out.append(framing.strip() + "\n")
        else:
            out.append(
                f"This systematic review was initiated to evaluate the scientific evidence "
                f"for and against the following research question: **{topic}**\n"
            )
        out.append(
            "\nThis review employs a dual-hypothesis design. Two parallel search tracks "
            "were run simultaneously:\n\n"
            "- **Affirmative Track**: Seeks evidence supporting the hypothesis.\n"
            "- **Falsification Track**: Adversarially seeks evidence of null results, harms, "
            "methodological flaws, and confounders.\n"
        )
        if aff_strategy and hasattr(aff_strategy, 'pico'):
            p = aff_strategy.pico
            out.append(
                f"\n**Affirmative Hypothesis:** In {p.get('population', 'the target population')}, "
                f"does {p.get('intervention', 'the intervention')} improve "
                f"{p.get('outcome', 'the primary outcome')} compared to "
                f"{p.get('comparison', 'control')}?\n"
            )
        if fal_strategy and hasattr(fal_strategy, 'pico'):
            fp = fal_strategy.pico
            out.append(
                f"\n**Falsification Hypothesis:** Does {fp.get('intervention', 'the intervention')} "
                f"fail to improve, or actively harm, {fp.get('outcome', 'the primary outcome')} "
                f"in {fp.get('population', 'the target population')}?\n"
            )

        # ── 2. METHODS ────────────────────────────────────────────────────────
        out.append("\n---\n\n## 2. Methods\n")

        # 2.1 Search Strategy
        out.append("### 2.1 Search Strategy\n")
        for label, strategy in [("Affirmative", aff_strategy), ("Falsification", fal_strategy)]:
            if not strategy or not hasattr(strategy, 'pico'):
                continue
            out.append(f"#### {label} Track\n")
            p = strategy.pico
            out.append(
                f"**PICO Framework:**  \n"
                f"- **P** (Population): {p.get('population', 'N/A')}  \n"
                f"- **I** (Intervention): {p.get('intervention', 'N/A')}  \n"
                f"- **C** (Comparison): {p.get('comparison', 'N/A')}  \n"
                f"- **O** (Outcome): {p.get('outcome', 'N/A')}\n"
            )
            # Tiered keyword plan (new architecture)
            if hasattr(strategy, 'tier1'):
                tier_map = [
                    ("Tier 1 — Established evidence (exact folk terms)", strategy.tier1),
                    ("Tier 2 — Supporting evidence (canonical synonyms)", strategy.tier2),
                    ("Tier 3 — Speculative extrapolation (compound class)", strategy.tier3),
                ]
                out.append("\n**Three-Tier Keyword Plan:**\n")
                for tier_label, tier_kw in tier_map:
                    if hasattr(tier_kw, 'intervention') and tier_kw.intervention:
                        out.append(f"\n*{tier_label}*\n")
                        out.append(f"- Intervention: {', '.join(tier_kw.intervention)}\n")
                        out.append(f"- Outcome: {', '.join(tier_kw.outcome)}\n")
                        if tier_kw.population:
                            out.append(f"- Population: {', '.join(tier_kw.population)}\n")
                        out.append(f"- *Rationale: {tier_kw.rationale}*\n")
                if strategy.auditor_approved:
                    out.append(f"\n✅ *Auditor approved after {strategy.revision_count} revision(s).*\n")
                else:
                    out.append(f"\n⚠ *Auditor not approved (proceeded after max revisions). Notes: {strategy.auditor_notes[:200]}*\n")
            # Legacy: Boolean search strings (old architecture — kept for backward compat)
            elif hasattr(strategy, 'mesh_terms') and strategy.mesh_terms:
                mt = strategy.mesh_terms
                out.append("\n**MeSH Terms:**\n")
                for cat, terms in mt.items():
                    if terms:
                        out.append(f"- *{cat.capitalize()}*: {', '.join(terms)}\n")
            if hasattr(strategy, 'search_strings') and strategy.search_strings:
                ss = strategy.search_strings
                out.append("\n**Boolean Search Strings:**\n")
                for db, query in ss.items():
                    if query:
                        out.append(f"- **{db.replace('_', ' ').title()}**: `{query}`\n")
            out.append("\n")

        # 2.2 Data Collection
        out.append("### 2.2 Data Collection\n")
        aff_tier = pd.get("aff_highest_tier", 1)
        fal_tier = pd.get("fal_highest_tier", 1)
        tier_labels = {1: "Tier 1 (established — exact folk terms)",
                       2: "Tier 2 (supporting — canonical synonyms)",
                       3: "Tier 3 (speculative — compound class)"}
        out.append(
            f"- **Databases searched:** PubMed (NCBI E-utilities), Google Scholar (via SearXNG)\n"
            f"- **Search date:** {search_date}\n"
            f"- **Search architecture:** Three-tier cascading keyword search. "
            f"Tier 1 runs first (exact folk terms). If pool < 50 records, Tier 2 runs "
            f"(canonical synonyms). If still < 50, Tier 3 runs (compound class/mechanism — "
            f"results require inference and are flagged as speculative extrapolation).\n"
            f"- **Affirmative track cascade reached:** {tier_labels.get(aff_tier, str(aff_tier))}\n"
            f"- **Falsification track cascade reached:** {tier_labels.get(fal_tier, str(fal_tier))}\n"
        )
        if aff_tier == 3 or fal_tier == 3:
            out.append(
                f"- ⚠ **Note:** One or both tracks reached Tier 3. Tier 3 evidence involves the "
                f"active compound class (e.g., caffeine from any source, not coffee specifically). "
                f"These results require an inference step to apply to the original substance and "
                f"are presented as speculative extrapolation in this review.\n"
            )
        out.append(
            f"- **Affirmative track records identified:** {aff_wide}\n"
            f"- **Falsification track records identified:** {fal_wide}\n"
            f"- **Total records identified:** {total_wide}\n"
        )

        # 2.3 Screening & Selection
        out.append("\n### 2.3 Screening & Selection\n")
        out.append(
            "Title and abstract screening was performed by the Smart Model (Qwen3-32B-AWQ) "
            "using structured inclusion/exclusion criteria:\n\n"
            "**Inclusion criteria:** Human clinical studies (RCTs, meta-analyses, systematic reviews, "
            "large cohort studies); sample size ≥ 30 participants; published in peer-reviewed journals; "
            "directly relevant to the PICO question.\n\n"
            "**Exclusion criteria:** Animal models; in vitro studies; case reports (n < 5); "
            "conference abstracts without full data; non-English publications; retracted publications.\n\n"
            f"- **Affirmative track screened to top candidates:** {aff_screened}\n"
            f"- **Falsification track screened to top candidates:** {fal_screened}\n"
            f"- **Total articles selected for full-text retrieval:** {total_screened}\n"
        )

        # 2.4 Data Extraction
        out.append("\n### 2.4 Data Extraction\n")
        out.append(
            "Full-text articles were retrieved using a 4-tier cascade:\n\n"
            "1. **PMC EFetch** (NCBI EUtils `elink` + `efetch`): Open-access full XML\n"
            "2. **Europe PMC REST API**: Full-text XML for OA articles\n"
            "3. **Unpaywall API**: Open-access PDF/HTML links via DOI\n"
            "4. **NCBI Abstract EFetch**: Official abstract XML (fallback for paywalled articles)\n\n"
            f"- **Affirmative track full-text retrieved:** {aff_ft_ok} "
            f"(errors: {aff_ft_err})\n"
            f"- **Falsification track full-text retrieved:** {fal_ft_ok} "
            f"(errors: {fal_ft_err})\n"
            f"- **Total full texts successfully retrieved:** {total_ft_ok}\n\n"
            "Clinical variables were extracted from each full text by the Fast Model "
            "(llama3.2:1b) using a structured extraction template capturing: "
            "study design, sample sizes, demographics, follow-up period, "
            "Control Event Rate (CER), Experimental Event Rate (EER), "
            "effect size with confidence intervals, blinding, randomization method, "
            "intention-to-treat analysis, funding source, and risk of bias.\n"
        )

        # 2.5 Statistical Analysis
        out.append("\n### 2.5 Statistical Analysis\n")
        out.append(
            "**Deterministic Clinical Math (Step 6):** ARR, RRR, and NNT were calculated "
            "using pure Python arithmetic from extracted CER and EER values — no LLM involvement:\n\n"
            "- ARR (Absolute Risk Reduction) = CER − EER\n"
            "- RRR (Relative Risk Reduction) = ARR / CER\n"
            "- NNT (Number Needed to Treat) = 1 / |ARR|\n\n"
            "**GRADE Framework (Step 7):** Evidence quality was assessed by the Smart Model "
            "using the Grading of Recommendations, Assessment, Development, and Evaluations "
            "(GRADE) framework. Starting quality was HIGH for RCTs and LOW for observational "
            "studies, then adjusted for: risk of bias, inconsistency, indirectness, imprecision, "
            "and publication bias (downgrade factors); and large effect size, dose-response "
            "gradient, and plausible confounders (upgrade factors).\n"
        )

        # ── 3. RESULTS ────────────────────────────────────────────────────────
        out.append("\n---\n\n## 3. Results\n")

        # 3.1 Study Selection (PRISMA)
        out.append("### 3.1 Study Selection\n")
        prisma_from_grade = grade_sections.get("prisma flow diagram", "")
        out.append(
            "**PRISMA Flow:**\n\n"
            f"| Stage | Affirmative | Falsification | Total |\n"
            f"|-------|-------------|---------------|-------|\n"
            f"| Records identified | {aff_wide} | {fal_wide} | {total_wide} |\n"
            f"| Screened (top candidates) | {aff_screened} | {fal_screened} | {total_screened} |\n"
            f"| Full-text retrieved | {aff_ft_ok} | {fal_ft_ok} | {total_ft_ok} |\n"
            f"| Full-text errors | {aff_ft_err} | {fal_ft_err} | {total_ft_err} |\n"
            f"| Included in synthesis | {len(aff_extractions)} | {len(fal_extractions)} | {len(all_extractions)} |\n"
        )
        if prisma_from_grade:
            out.append(f"\n{prisma_from_grade}\n")

        # 3.2 Study Characteristics
        out.append("\n### 3.2 Study Characteristics\n")
        out.append(_format_study_characteristics_table(all_extractions))

        # 3.3 Clinical Impact
        out.append("\n### 3.3 Clinical Impact (Deterministic Math)\n")
        math_file_path = output_dir / "clinical_math.md"
        if math_file_path.exists():
            out.append(math_file_path.read_text().strip() + "\n")
        elif impacts:
            rows = ["| Study | CER | EER | ARR | RRR | NNT | Direction |",
                    "|-------|-----|-----|-----|-----|-----|-----------|"]
            for i in impacts:
                rows.append(
                    f"| {i.study_id} | {i.cer:.3f} | {i.eer:.3f} | "
                    f"{i.arr:+.4f} | {i.rrr:+.2%} | {i.nnt:.1f} | {i.direction} |"
                )
            out.append('\n'.join(rows) + "\n\n")
            for i in impacts:
                out.append(f"- **{i.study_id}**: {i.nnt_interpretation}\n")
        else:
            out.append("*No studies provided both CER and EER — NNT calculation not available.*\n")

        # ── 4. DISCUSSION ─────────────────────────────────────────────────────
        out.append("\n---\n\n## 4. Discussion\n")

        # 4.1 Affirmative Case
        out.append("### 4.1 Affirmative Case\n")
        out.append(aff_case_text.strip() + "\n")

        # 4.2 Falsification Case
        out.append("\n### 4.2 Falsification Case\n")
        out.append(fal_case_text.strip() + "\n")

        # 4.3 GRADE Evidence Assessment
        out.append("\n### 4.3 GRADE Evidence Assessment\n")
        ep = grade_sections.get("evidence profile", "")
        ga = grade_sections.get("grade assessment", "")
        if ep:
            out.append(f"**Evidence Profile:**\n\n{ep}\n")
        if ga:
            out.append(f"\n**GRADE Assessment:**\n\n{ga}\n")
        if not ep and not ga:
            # Fallback: include the full audit text minus already-extracted sections
            out.append(audit_text.strip() + "\n")

        # 4.4 Balanced Verdict
        out.append("\n### 4.4 Balanced Verdict\n")
        bv = grade_sections.get("balanced verdict", "")
        if bv:
            out.append(bv + "\n")
        else:
            out.append(
                f"**Evidence Quality (GRADE):** {grade_level}  \n"
                f"**Conclusion:** {conclusion_status}\n"
            )

        # 4.5 Limitations
        out.append("\n### 4.5 Limitations\n")
        out.append(
            "The following pipeline-specific limitations apply to this synthesis:\n\n"
            "- The Fast Model (llama3.2:1b) may have misclassified study designs or "
            "misextracted clinical variables from complex full-text articles.\n"
            "- Articles not available via PMC, Europe PMC, or Unpaywall were reduced to "
            "abstract-level data, limiting extraction depth for paywalled literature.\n"
            "- The 500-result cap per track may exclude relevant studies not retrieved "
            "in the top results from PubMed or Google Scholar.\n"
            "- CER/EER extraction relies on the Fast Model correctly identifying "
            "event rates in text; studies reporting outcomes without explicit event "
            "rates could not be included in NNT calculations.\n"
            "- Non-English language publications were excluded.\n"
        )

        # 4.6 Recommendations
        out.append("\n### 4.6 Recommendations for Further Research\n")
        recs = grade_sections.get("recommendations for further research", "")
        if recs:
            out.append(recs + "\n")
        else:
            out.append(
                "- Conduct large-scale, long-term randomized controlled trials with "
                "rigorous methodologies to address identified evidence gaps.\n"
                "- Investigate outcomes in under-represented populations.\n"
                "- Address potential biases and ensure transparency in study design "
                "and reporting.\n"
            )

        # ── 5. REFERENCES ──────────────────────────────────────────────────────
        out.append("\n---\n\n## 5. References\n")
        out.append(_format_references(all_extractions, all_wide))

        return '\n'.join(out)

    sot_content = build_imrad_sot(
        topic=topic_name,
        reports=deep_reports,
        ev_quality=evidence_quality,
        aff_cand=aff_candidates,
        domain=_research_domain,
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
    print(f"✓ Source of Truth (IMRaD) generated from deep research ({len(sot_content)} chars)")

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
    blueprint_task.description += sot_injection
    audit_task.description += sot_injection

# Inject evidence level and key numbers into blueprint for informed framing
_grade_injection = ""  # may be populated below; referenced by _crew_kickoff_guarded
if deep_reports and deep_reports.get("audit"):
    _audit_obj = deep_reports.get("audit")
    _audit_text = _audit_obj.report if hasattr(_audit_obj, 'report') else str(_audit_obj)
    _pd = deep_reports.get("pipeline_data", {}) if isinstance(deep_reports, dict) else {}
    _impacts = _pd.get("impacts", [])

    if _research_domain == "social_science":
        # Social science: Evidence Quality levels + effect sizes
        _eq_m = re.search(r'Final\s+Evidence\s+Quality[:\s]*\*{0,2}(STRONG|MODERATE_STRONG|MODERATE_WEAK|MODERATE|WEAK|VERY_WEAK)\*{0,2}', _audit_text, re.IGNORECASE)
        _eq_level = _eq_m.group(1).strip().upper() if _eq_m else "Not Determined"
        _grade_injection = f"\n\nEVIDENCE QUALITY LEVEL: {_eq_level}\n"
        _grade_injection += "Use this to calibrate your framing language:\n"
        _grade_injection += "- STRONG → 'Rigorous research clearly demonstrates...'\n"
        _grade_injection += "- MODERATE → 'Evidence suggests...'\n"
        _grade_injection += "- WEAK → 'Preliminary findings indicate...'\n"
        _grade_injection += "- VERY_WEAK → 'Limited evidence hints at...'\n"
        if _impacts:
            _grade_injection += "\nKEY EFFECT SIZE NUMBERS:\n"
            for _imp in _impacts[:5]:
                _label = getattr(_imp, 'study_id', 'Study')
                _d = getattr(_imp, 'cohens_d', None)
                _g = getattr(_imp, 'hedges_g', None)
                _mag = getattr(_imp, 'magnitude', 'unknown')
                if _d is not None:
                    _g_str = f", g={_g:.3f}" if _g is not None else ""
                    _grade_injection += f"- {_label}: d={_d:.3f}{_g_str} ({_mag})\n"
    else:
        # Clinical: GRADE levels + NNT/ARR
        _grade_m = re.search(r'Final\s+(?:GRADE|Grade)[:\s]*\*{0,2}(High|Moderate|Low|Very\s+Low)\*{0,2}', _audit_text, re.IGNORECASE)
        _grade_level = _grade_m.group(1).strip() if _grade_m else "Not Determined"
        _grade_injection = f"\n\nGRADE EVIDENCE LEVEL: {_grade_level}\n"
        _grade_injection += "Use this to calibrate your framing language in Section 6 (GRADE-Informed Framing Guide).\n"
        if _impacts:
            _grade_injection += "\nKEY CLINICAL IMPACT NUMBERS:\n"
            for _imp in _impacts[:5]:
                _label = getattr(_imp, 'study_id', 'Study')
                _nnt = getattr(_imp, 'nnt', None)
                _arr = getattr(_imp, 'arr', None)
                _dir = getattr(_imp, 'direction', 'unknown')
                if _nnt is not None:
                    _grade_injection += f"- {_label}: NNT={_nnt:.0f} (ARR={_arr:.3f}, direction: {_dir})\n"
    blueprint_task.description += _grade_injection

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
        "1. Acknowledge this in the HOOK or Act 1: "
        "   'While direct studies on this are limited, related research gives us clues...'\n"
        "2. In Act 2 (Evidence) and Act 3 (Nuance), distinguish: "
        "(a) what limited direct evidence shows, "
        "(b) what related evidence suggests, "
        "(c) what remains unknown.\n"
        "3. In Act 4 (Protocol), frame recommendations as 'based on current evidence' — not 'proven'.\n"
        "4. Do NOT invent citations. If few studies exist, say so in the dialogue.\n"
        "Example dialogue:\n"
        "  Presenter: 'Direct studies on this exact pattern are surprisingly rare. "
        "But here's what the broader science on meal timing tells us...'\n"
        "  Questioner: 'So we're working with partial evidence here. "
        "What do we actually know for certain?'\n"
    )
    blueprint_task.description += (
        "\n\nEVIDENCE NOTE: Research was limited. "
        "Mark citations with [LIMITED EVIDENCE] where the research base is sparse. "
        "In the Narrative Arc, ensure Act 3 (Nuance) heavily emphasizes what is NOT known. "
        "The Hook should acknowledge the evidence gap (e.g., 'Despite X million people doing Y, "
        "we have surprisingly few studies on...')."
    )

# Domain-aware terminology injection for Crew 3 tasks
if _research_domain == "social_science":
    _domain_note = (
        "\n\nDOMAIN NOTE: This is a SOCIAL SCIENCE topic (not clinical/health).\n"
        "Use effect sizes (Cohen's d, Hedges' g) instead of NNT/ARR.\n"
        "Use Evidence Quality levels (STRONG/MODERATE/WEAK/VERY_WEAK) instead of GRADE.\n"
        "Cite study designs as: systematic review, quasi-experimental, cohort, cross-sectional.\n"
        "Do NOT use clinical terminology (NNT, ARR, CER, EER, RCT) unless the study is actually clinical.\n"
    )
    script_task.description += _domain_note
    blueprint_task.description += _domain_note
    audit_task.description += _domain_note

# ================================================================
# CREW 3: Podcast Production
# ================================================================
print(f"\n{'='*70}")
print(f"CREW 3: PODCAST PRODUCTION")
print(f"{'='*70}")

translated_sot = None  # set below if translation runs
translated_sot_summary = ""
sot_translated_file = None
if translation_task is not None and sot_content:
    print(f"\nPHASE 3: REPORT TRANSLATION (pipelined)")
    print(f"Translating Source-of-Truth to {language_config['name']} — pipelined mid-tier translate + smart audit")

    # Pipelined translation: mid-tier model translates, smart model audits concurrently
    translated_sot = _translate_sot_pipelined(sot_content, language, language_config)

    if translated_sot:
        lang_suffix = language  # e.g. "ja"
        sot_translated_file = output_dir / f"source_of_truth_{lang_suffix}.md"
        with open(sot_translated_file, 'w', encoding='utf-8') as f:
            f.write(translated_sot)
        print(f"✓ Translated SOT saved ({len(translated_sot)} chars) → {sot_translated_file.name}")
        # Generate compact summary of translated SOT for task description injection
        print("  Summarizing translated SOT for Crew 3 context injection...")
        translated_sot_summary = summarize_report_with_fast_model(translated_sot, "sot_translated", topic_name)
        if translated_sot_summary:
            _tl_injection = _build_sot_injection_for_stage(
                1, sot_file, sot_translated_file,
                sot_summary, translated_sot_summary, _grade_injection, language_config
            )
            blueprint_task.description += _tl_injection
            script_task.description += _tl_injection
            audit_task.description += _tl_injection
        # CRITICAL: compact reference only — full 84KB SOT as context causes 27K+ input tokens
        # → CrewAI context overflow → infinite summarizer loop (observed: 36 cycles, 9.6h wasted)
        from types import SimpleNamespace
        translation_task.output = SimpleNamespace(raw=(
            f"[Translation complete — {len(translated_sot):,} chars]\n"
            f"Translated SOT saved: {sot_translated_file}\n"
            f"Key research summary injected into task descriptions."
        ))
    else:
        print(f"  Warning: Chunked translation produced no output — translated SOT not saved")

# --- Extract base task descriptions for audit-loop feedback injection ---
script_task_base_description = script_task.description
script_task_expected_output = script_task.expected_output
polish_task_base_description = polish_task.description
polish_task_expected_output = polish_task.expected_output

# Bug 5: Translate task descriptions for non-English runs
if language != 'en':
    print(f"\nTranslating Crew 3 task prompts to {language_config['name']}...")
    for _task, _name in [
        (blueprint_task, "blueprint"),
        (script_task, "script"),
        (polish_task, "polish"),
        (audit_task, "audit"),
    ]:
        _task.description = _translate_prompt(_task.description, language, language_config)
        print(f"  ✓ {_name} task prompt translated")
    # Keep expected_output in English (CrewAI internals)
    # Update base descriptions with translated versions
    script_task_base_description = script_task.description
    polish_task_base_description = polish_task.description

# Start background monitor for crew 3
monitor = CrewMonitor(all_task_list, progress_tracker)
monitor.start()

MAX_SCRIPT_ATTEMPTS = 3

try:
    # === PHASE 4: BLUEPRINT (single task, no loop) ===
    print(f"\n{'='*60}")
    print("PHASE 4: EPISODE BLUEPRINT")
    print(f"{'='*60}")
    _crew_kickoff_guarded(
        lambda: Crew(agents=[producer_agent], tasks=[blueprint_task], verbose=True),
        blueprint_task, translation_task, language,
        sot_file, sot_translated_file,
        sot_summary, translated_sot_summary,
        _grade_injection, language_config, "Phase 4 Blueprint"
    )
    print("  ✓ Blueprint complete")

    # === PHASE 5: SCRIPT DRAFT + EXPANSION ===
    print(f"\n{'='*60}")
    print("PHASE 5: SCRIPT DRAFT + EXPANSION")
    print(f"{'='*60}")

    # 5a. Generate initial draft (single CrewAI call)
    script_crew = Crew(agents=[producer_agent], tasks=[script_task], verbose=True)
    script_crew.kickoff()
    script_draft_text = script_task.output.raw

    # 5b. Strip <think> blocks from draft output
    script_draft_text = re.sub(r'<think>.*?</think>', '', script_draft_text, flags=re.DOTALL).strip()

    # 5c. Validate raw draft
    draft_validation = _validate_script(script_draft_text, target_length_int, SCRIPT_TOLERANCE,
                                         language_config, sot_content, stage='draft')
    print(f"  Draft: {draft_validation['word_count']} {language_config['length_unit']} — "
          f"{'PASS' if draft_validation['pass'] else 'NEEDS EXPANSION'}")

    # 5d. Run per-act expansion if draft is short
    if not draft_validation['pass'] and any('TOO SHORT' in i for i in draft_validation['issues']):
        print("  Running per-act expansion...")
        script_draft_text, was_expanded = _run_script_expansion(
            script_draft_text, sot_content, target_length_int,
            language_config, SESSION_ROLES, topic_name, target_instruction)

        # Re-validate after expansion
        exp_validation = _validate_script(script_draft_text, target_length_int, SCRIPT_TOLERANCE,
                                           language_config, sot_content, stage='draft')
        print(f"  Post-expansion: {exp_validation['word_count']} {language_config['length_unit']} — "
              f"{'PASS' if exp_validation['pass'] else 'FAIL'}")

        # 5e. If still short after expansion, ONE retry with per-act feedback
        if not exp_validation['pass'] and any('TOO SHORT' in i for i in exp_validation['issues']):
            acts_analysis = _analyze_acts(script_draft_text, language_config, target_length_int)
            per_act_feedback = "Per-act word counts:\n"
            for a in acts_analysis:
                if a['num'] not in (0, 99) and a['target'] > 0:
                    per_act_feedback += f"  {a['label']}: {a['count']}/{a['target']} {language_config['length_unit']}\n"
            _feedback_block = (
                f"\n\nPREVIOUS ATTEMPT FEEDBACK:\n{exp_validation['feedback']}\n"
                f"{per_act_feedback}\n"
                f"Expand the shortest acts to hit their targets.\n"
            )
            retry_script_task = Task(
                description=script_task_base_description + _feedback_block,
                expected_output=script_task_expected_output,
                agent=producer_agent,
                context=[blueprint_task],
            )
            if translation_task is not None:
                retry_script_task.context = [blueprint_task, translation_task]
            Crew(agents=[producer_agent], tasks=[retry_script_task], verbose=True).kickoff()
            retry_text = re.sub(r'<think>.*?</think>', '', retry_script_task.output.raw, flags=re.DOTALL).strip()
            retry_val = _validate_script(retry_text, target_length_int, SCRIPT_TOLERANCE,
                                          language_config, sot_content, stage='draft')
            print(f"  Retry: {retry_val['word_count']} {language_config['length_unit']} — "
                  f"{'PASS' if retry_val['pass'] else 'FAIL'}")
            # Use whichever is longer
            if retry_val['word_count'] > exp_validation['word_count']:
                script_draft_text = retry_text
                script_task = retry_script_task

    # Store expanded draft word count for shrinkage guard in Phase 6
    _expanded_draft_count = _count_words(script_draft_text, language_config)
    # Update script_task output with expanded text if we modified it outside CrewAI
    current_script_task = script_task

    # === PHASE 6: POLISH + SHRINKAGE GUARD ===
    print(f"\n{'='*60}")
    print("PHASE 6: SCRIPT POLISH (audit loop, max {0} attempts)".format(MAX_SCRIPT_ATTEMPTS))
    print(f"{'='*60}")
    polish_feedback = ""
    current_polish_task = polish_task  # first iteration uses original task
    # Update context to point at the latest draft
    polish_task.context = [current_script_task]
    if translation_task is not None:
        polish_task.context = [current_script_task, translation_task]

    for attempt in range(1, MAX_SCRIPT_ATTEMPTS + 1):
        if polish_feedback:
            _feedback_block = (
                f"\n\nPREVIOUS ATTEMPT FEEDBACK (attempt {attempt-1}):\n{polish_feedback}\n"
                f"Fix ALL issues listed above.\n"
            )
            current_polish_task = Task(
                description=polish_task_base_description + _feedback_block,
                expected_output=polish_task_expected_output,
                agent=editor_agent,
                context=[current_script_task],
            )
            if translation_task is not None:
                current_polish_task.context = [current_script_task, translation_task]

        polish_crew = Crew(agents=[editor_agent], tasks=[current_polish_task], verbose=True)
        polish_crew.kickoff()
        polished_text = re.sub(r'<think>.*?</think>', '', current_polish_task.output.raw, flags=re.DOTALL).strip()

        validation = _validate_script(polished_text, target_length_int, SCRIPT_TOLERANCE,
                                       language_config, sot_content, stage='polish')

        print(f"  Polish attempt {attempt}: {validation['word_count']} {language_config['length_unit']} — "
              f"{'PASS' if validation['pass'] else 'FAIL'}")

        if validation['pass']:
            break

        polish_feedback = validation['feedback']
        print(f"  Issues:\n{polish_feedback}")

    # Shrinkage guard: if polish shrunk by >10% vs expanded draft, discard polish
    _polished_count = _count_words(polished_text, language_config)
    if _expanded_draft_count > 0 and _polished_count < _expanded_draft_count * 0.90:
        print(f"  ⚠ Polish shrunk script by {100 - _polished_count * 100 // _expanded_draft_count}% "
              f"({_expanded_draft_count} → {_polished_count} {language_config['length_unit']}) — "
              f"using expanded draft instead")
        # Use the expanded draft but ensure [TRANSITION] markers are present
        if script_draft_text.count('[TRANSITION]') < 3:
            # Inject [TRANSITION] markers at act boundaries
            script_draft_text = re.sub(
                r'(---\s*\n###\s*\*?\*?ACT)',
                r'[TRANSITION]\n\n\1',
                script_draft_text
            )
        polished_text = script_draft_text
    polish_task = current_polish_task

    # === PHASE 7: ACCURACY AUDIT (advisory, existing logic) ===
    print(f"\n{'='*60}")
    print("PHASE 7: ACCURACY AUDIT")
    print(f"{'='*60}")
    # Update audit context to latest polish output
    audit_task.context = [polish_task]
    if translation_task is not None:
        audit_task.context = [polish_task, translation_task]
    audit_crew = Crew(agents=[auditor_agent], tasks=[audit_task], verbose=True)
    audit_crew.kickoff()

except Exception as e:
    print(f"\n{'='*70}")
    print("CREW 3 FAILED — saving partial outputs")
    print(f"{'='*70}")
    print(f"Error: {e}")
    _partial_tasks = [
        ("blueprint_partial.md", blueprint_task),
        ("script_draft_partial.md", script_task),
        ("script_polished_partial.md", polish_task),
        ("audit_partial.md", audit_task),
    ]
    for _fname, _task in _partial_tasks:
        try:
            if hasattr(_task, 'output') and _task.output and hasattr(_task.output, 'raw') and _task.output.raw:
                with open(output_dir / _fname, 'w', encoding='utf-8') as _f:
                    _f.write(_task.output.raw)
                print(f"  ✓ Partial output saved: {_fname}")
        except Exception:
            pass
    monitor.stop()
    raise
finally:
    monitor.stop()
    monitor.join(timeout=2)
    progress_tracker.workflow_completed()

# --- CONDITIONAL SCRIPT CORRECTION (Phase 7 blocking on HIGH-severity drift) ---
audit_output = audit_task.output.raw if hasattr(audit_task, 'output') and audit_task.output else ""
high_severity_found = bool(re.search(r'\*\*Severity\*\*:\s*HIGH', audit_output, re.IGNORECASE))

if high_severity_found and audit_output:
    print(f"\n{'='*60}")
    print("HIGH-SEVERITY DRIFT DETECTED — RUNNING SCRIPT CORRECTION")
    print(f"{'='*60}")
    # Use polished_text which may be expanded draft if shrinkage guard fired
    polished_script_raw = polished_text if polished_text else (
        polish_task.output.raw if hasattr(polish_task, 'output') and polish_task.output else "")
    correction_task = Task(
        description=(
            f"The accuracy audit found HIGH-severity scientific drift in the podcast script.\n\n"
            f"AUDIT REPORT:\n{audit_output}\n\n"
            f"POLISHED SCRIPT:\n{polished_script_raw}\n\n"
            f"Fix ONLY the specific lines cited in the audit's 'Drift Instances Found' section.\n"
            f"For each HIGH-severity issue:\n"
            f"  - Find the exact quote from 'Script says'\n"
            f"  - Replace it with language consistent with 'Source-of-truth says'\n"
            f"Do NOT rewrite the entire script. Only fix cited drift instances.\n"
            f"Preserve all [TRANSITION] markers, speaker labels, and overall structure.\n"
            f"{target_instruction}"
        ),
        expected_output="Corrected podcast script with HIGH-severity drift fixed.",
        agent=editor_agent,
    )
    try:
        correction_crew = Crew(agents=[editor_agent], tasks=[correction_task], verbose=False)
        correction_result = correction_crew.kickoff()
        corrected = correction_result.raw if hasattr(correction_result, 'raw') else str(correction_result)
        orig_transitions = polished_script_raw.count('[TRANSITION]')
        corrected_transitions = corrected.count('[TRANSITION]')
        if len(corrected) < len(polished_script_raw) * 0.5:
            print("⚠ Correction output too short — using original polished script")
            _corrected_script_text = None
        elif orig_transitions > 0 and corrected_transitions < orig_transitions:
            print(f"⚠ Correction lost [TRANSITION] markers ({orig_transitions}→{corrected_transitions}) — using original polished script")
            _corrected_script_text = None
        else:
            with open(output_dir / "ACCURACY_CORRECTIONS.md", 'w') as f:
                f.write("# Script Corrections Applied\n\n")
                f.write("HIGH-severity drift instances were corrected before audio generation.\n\n")
                f.write(f"## Original Audit\n{audit_output}\n")
            print("✓ Script correction applied — using corrected script for audio")
            _corrected_script_text = corrected
    except Exception as e:
        print(f"⚠ Script correction failed: {e} — using original polished script")
        _corrected_script_text = None
else:
    _corrected_script_text = None
    if audit_output:
        print("✓ Accuracy audit: No HIGH-severity drift — proceeding to audio")

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
    ("Episode Blueprint", blueprint_task, "EPISODE_BLUEPRINT.md"),
    ("Script Draft", script_task, "script_draft.md"),
    # script_final.md saved in Phase 8 (from polished/corrected script before TTS)
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
    deep_audit = deep_reports.get("audit")
    if deep_audit:
        print(f"\n--- Deep Research Summary ---")
        _lead = deep_reports.get("lead")
        _counter = deep_reports.get("counter")
        if _lead:
            print(f"  Lead sources: {_lead.total_summaries}")
        if _counter:
            print(f"  Counter sources: {_counter.total_summaries}")
        print(f"  Total sources: {deep_audit.total_summaries}")
        print(f"  Total URLs fetched: {deep_audit.total_urls_fetched}")
        print(f"  Duration: {deep_audit.duration_seconds:.0f}s")

# --- SESSION METADATA ---
print("\n--- Documenting Session Metadata ---")
session_metadata = (
    f"PODCAST SESSION METADATA\n{'='*60}\n\n"
    f"Topic: {topic_name}\n\n"
    f"Language: {language_config['name']} ({language})\n\n"
    f"Role Assignments:\n"
    f"  {SESSION_ROLES['presenter']['label']}: Presenter ({SESSION_ROLES['presenter']['personality']})\n"
    f"  {SESSION_ROLES['questioner']['label']}: Questioner ({SESSION_ROLES['questioner']['personality']})\n"
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

# --- PHASE 8: AUDIO GENERATION ---
print("\n--- Generating Multi-Voice Podcast Audio (Kokoro TTS) ---")

# Use corrected script if HIGH-severity drift was fixed, otherwise use polished script
if _corrected_script_text:
    script_text = _corrected_script_text
    print("Using drift-corrected script for audio generation")
else:
    # Use polished_text (may be expanded draft if shrinkage guard fired)
    script_text = polished_text if polished_text else (
        polish_task.output.raw if hasattr(polish_task, 'output') and polish_task.output else "")

# Post-Crew 3 language audit — fix Chinese contamination + English leakage for non-English runs
if language != 'en':
    print(f"\nRunning post-script language audit ({language_config['name']})...")
    script_text = _audit_script_language(script_text, language, language_config)

# Save script_final.md (the authoritative script for TTS)
with open(output_dir / "script_final.md", 'w', encoding='utf-8') as f:
    f.write(script_text)

# Language-aware script length measurement
speech_rate  = language_config['speech_rate']
length_unit  = language_config['length_unit']
if length_unit == 'chars':
    script_length = len(re.sub(r'[\s\n\r\t\u3000\uff1a:\u300c\u300d\u3001\u3002\u30fb\uff08\uff09\-\u2014*#]', '', script_text))
else:
    content_only = re.sub(r'^[A-Za-z0-9_ ]+:\s*', '', script_text, flags=re.MULTILINE)
    script_length = len(content_only.split())
    length_unit = "words (net)"
estimated_duration_min = script_length / speech_rate
target_length = target_length_int
target_low    = int(target_length * (1 - SCRIPT_TOLERANCE))
target_high   = int(target_length * (1 + SCRIPT_TOLERANCE))

print(f"\n{'='*60}")
print(f"DURATION CHECK")
print(f"{'='*60}")
print(f"Script length: {script_length} {length_unit}")
print(f"Estimated duration: {estimated_duration_min:.1f} minutes")
print(f"Target: {_target_min} minutes ({target_length} {length_unit})")

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

# Clean script for TTS and save .txt copy for debugging
cleaned_script = clean_script_for_tts(script_text)

script_file = output_dir / "script.txt"
with open(script_file, 'w') as f:
    f.write(cleaned_script)
print(f"Cleaned script saved: {script_file} ({script_length} {length_unit})")

output_path = output_dir / "audio.wav"

audio_file = None
transition_positions = []

audio_file = None
try:
    print(f"Starting audio generation with script length: {len(cleaned_script)} chars")
    tts_result = generate_audio_from_script(cleaned_script, str(output_path), lang_code=language_config['tts_code'])
    if isinstance(tts_result, tuple):
        audio_file_path, transition_positions = tts_result
    else:
        audio_file_path, transition_positions = tts_result, []
    if audio_file_path:
        audio_file = Path(audio_file_path)
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
        mastered = post_process_audio(str(audio_file), bgm_target="Interesting BGM.wav",
                                      transition_positions_ms=transition_positions)
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