"""
shared/config.py — Central configuration for DR_2_Podcast pipeline.

All environment variables, LLM instances, language settings, character
definitions, and output directory management live here.
"""

import os
import sys
import re
import time
import json
import random
import shutil
import httpx
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.parent.absolute()  # DR_2_Podcast root
BASE_OUTPUT_DIR = SCRIPT_DIR / "research_outputs"
BASE_OUTPUT_DIR.mkdir(exist_ok=True)

TOPIC_INDEX_FILE = BASE_OUTPUT_DIR / "topic_index.json"

# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES = {
    'en': {
        'name': 'English',
        'tts_code': 'a',  # Kokoro American English
        'instruction': 'Write all content in English.',
        'pdf_font': 'Helvetica',
    },
    'ja': {
        'name': '\u65e5\u672c\u8a9e (Japanese)',
        'tts_code': 'j',  # Kokoro Japanese
        'instruction': '\u3059\u3079\u3066\u306e\u30b3\u30f3\u30c6\u30f3\u30c4\u3092\u65e5\u672c\u8a9e\u3067\u66f8\u3044\u3066\u304f\u3060\u3055\u3044\u3002(Write all content in Japanese.)',
        'pdf_font': 'Arial Unicode MS',
    },
}

# ---------------------------------------------------------------------------
# Characters
# ---------------------------------------------------------------------------
CHARACTERS = {
    "Kaz": {
        "gender": "male",
        "voice_model": "male_voice",
        "base_personality": "Enthusiastic science communicator, clear explainer, data-driven",
    },
    "Erika": {
        "gender": "female",
        "voice_model": "female_voice",
        "base_personality": "Curious and sharp interviewer, asks what the audience is thinking",
    },
}


def assign_roles(host_config: str = "random") -> dict:
    """Assign Kaz and Erika to presenter/questioner roles."""
    characters = list(CHARACTERS.keys())
    host_config = host_config.lower()

    if host_config == "kaz_erika":
        presenter_name, questioner_name = "Kaz", "Erika"
    elif host_config == "erika_kaz":
        presenter_name, questioner_name = "Erika", "Kaz"
    else:
        random.shuffle(characters)
        presenter_name, questioner_name = characters[0], characters[1]

    return {
        "presenter": {
            "character": presenter_name,
            "stance": "teaching",
            "personality": CHARACTERS[presenter_name]["base_personality"],
        },
        "questioner": {
            "character": questioner_name,
            "stance": "curious",
            "personality": CHARACTERS[questioner_name]["base_personality"],
        },
    }


# ---------------------------------------------------------------------------
# Accessibility
# ---------------------------------------------------------------------------
ACCESSIBILITY_INSTRUCTIONS = {
    "simple": (
        "Explain every scientific term the first time it appears using a one-line plain-English definition. "
        "Use everyday analogies (e.g. 'blood sugar is like fuel in a car'). "
        "After defining a term once, you may use it freely."
    ),
    "moderate": (
        "Define key domain terms once when first introduced, then use them normally. "
        "Assume the listener can follow a simple cause-and-effect explanation. "
        "Use analogies sparingly \u2014 only for the most abstract concepts."
    ),
    "technical": (
        "Use standard scientific terminology without extensive definitions. "
        "Assume the listener has basic biology knowledge (high school AP level). "
        "Focus on depth and nuance rather than simplification."
    ),
}

# ---------------------------------------------------------------------------
# Research artifacts (for reuse-mode copy)
# ---------------------------------------------------------------------------
RESEARCH_ARTIFACTS = [
    "source_of_truth.md", "SOURCE_OF_TRUTH.md",
    "supporting_research.md", "adversarial_research.md",
    "source_verification.md", "deep_research_sources.json",
    "research_framing.md", "gap_analysis.md",
    "research_framing.pdf", "supporting_paper.pdf",
    "adversarial_paper.pdf", "verified_sources_bibliography.pdf",
    "source_of_truth.pdf", "gap_fill_research.md",
    "url_validation_results.json",
]


def copy_research_artifacts(src_dir: Path, dst_dir: Path):
    """Copy research-related files from a previous run to a new output directory."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in RESEARCH_ARTIFACTS:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            copied += 1
    print(f"  Copied {copied} research artifacts from {src_dir.name}")


def copy_all_artifacts(src_dir: Path, dst_dir: Path):
    """Copy all files from a previous run to a new output directory."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in src_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, dst_dir / item.name)
            copied += 1
    print(f"  Copied {copied} total artifacts from {src_dir.name}")


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
def create_timestamped_output_dir(base_dir: Path = BASE_OUTPUT_DIR) -> Path:
    """Create research_outputs/YYYY-MM-DD_HH-MM-SS/."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = base_dir / timestamp
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"OUTPUT DIRECTORY: {out}")
    print(f"{'='*60}\n")
    return out


# ---------------------------------------------------------------------------
# LLM instances
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "deepseek-r1:32b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"


def _wait_for_model_server() -> str:
    """Wait for the LLM server to come online and return the model name."""
    model = os.getenv("MODEL_NAME", DEFAULT_MODEL)
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
    print(f"Connecting to LLM server at {base_url}...")

    for i in range(10):
        try:
            response = httpx.get(f"{base_url}/models", timeout=5.0)
            if response.status_code == 200:
                print(f"\u2713 LLM server online! Using model: {model}")
                return model
        except Exception as e:
            if i % 5 == 0:
                print(f"Waiting for LLM server... ({i}s) \u2014 {e}")
            time.sleep(1)

    print("Error: Could not connect to LLM server.")
    sys.exit(1)


def build_llm_instances() -> tuple:
    """Return (strict_llm, creative_llm) configured from env."""
    model_name = _wait_for_model_server()
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)

    strict = LLM(
        model=model_name,
        base_url=base_url,
        api_key="NA",
        provider="openai",
        timeout=600,
        temperature=0.1,
        max_tokens=8000,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    creative = LLM(
        model=model_name,
        base_url=base_url,
        api_key="NA",
        provider="openai",
        timeout=600,
        temperature=0.7,
        max_tokens=12000,
        frequency_penalty=0.15,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    return strict, creative


# ---------------------------------------------------------------------------
# Fast model helper
# ---------------------------------------------------------------------------
def summarize_report_with_fast_model(report_text: str, role: str, topic: str) -> str:
    """Condense a deep research report using the fast model via Ollama."""
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=os.getenv("FAST_LLM_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
        )
        response = client.chat.completions.create(
            model=os.getenv("FAST_MODEL_NAME", "llama3.2:1b"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Condense this research report into a summary that preserves "
                        "ALL key findings, claim names, source URLs, and evidence "
                        "strength ratings. Target ~2000 words. Do not drop any "
                        "findings \u2014 compress descriptions instead."
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
            print(f"  \u2713 {role} report summarized: {len(report_text)} \u2192 {len(summary)} chars")
            return summary
        print(f"  \u26a0 {role} summary too short ({len(summary)} chars), falling back to truncation")
    except Exception as e:
        print(f"  \u26a0 Fast-model summarization failed for {role}: {e}")

    return report_text[:6000]


def check_fast_model_available() -> bool:
    """Check if Ollama fast model is reachable."""
    try:
        resp = httpx.get(f"{os.getenv('FAST_LLM_BASE_URL', 'http://localhost:11434/v1')}/models", timeout=3)
        if resp.status_code == 200:
            models = [m.get("id", "") for m in resp.json().get("data", [])]
            available = any(
                any(k in m.lower() for k in ["phi", "llama", "qwen", "mistral"])
                for m in models
            )
            if available:
                print(f"\u2713 Fast model detected on Ollama (available: {models})")
            return available
    except Exception:
        pass
    print("\u26a0 Fast model not available, using smart-only mode")
    return False


# ---------------------------------------------------------------------------
# Supplemental check (used by f02_approach_definition)
# ---------------------------------------------------------------------------
def check_supplemental_needed(topic: str, reuse_dir: Path) -> dict:
    """Ask the LLM if the previous source_of_truth.md adequately covers the topic."""
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
            f"{os.getenv('LLM_BASE_URL', 'http://localhost:8000/v1')}/chat/completions",
            json={
                "model": os.getenv("MODEL_NAME", DEFAULT_MODEL),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1024,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "needs_supplement": result.get("needs_supplement", True),
                "reason": result.get("reason", ""),
                "queries": result.get("queries", []),
            }
    except Exception as e:
        print(f"  Supplemental check failed: {e}")

    return {"needs_supplement": True, "reason": "Check failed, running supplemental as precaution", "queries": []}


# ---------------------------------------------------------------------------
# TTS dependency check
# ---------------------------------------------------------------------------
def check_tts_dependencies():
    """Verify Kokoro TTS is installed."""
    try:
        import kokoro  # noqa: F401
        print("\u2713 Kokoro TTS dependencies verified")
    except ImportError as e:
        print(f"CRITICAL ERROR: Kokoro TTS not installed: {e}")
        print("Install with: pip install kokoro>=0.9")
        print("Audio generation cannot proceed without Kokoro.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Length targets
# ---------------------------------------------------------------------------
def get_length_targets(language: str, length_mode: str = "long") -> dict:
    """Return word/char targets for the given language and length mode."""
    length_mode = length_mode.lower()
    if language == "ja":
        targets = {
            "short":  {"target": 5000,  "low": 4000,  "high": 6000,  "unit": "chars", "label": "Short (10-15 min)"},
            "medium": {"target": 10000, "low": 8500,  "high": 11500, "unit": "chars", "label": "Medium (20-25 min)"},
            "long":   {"target": 15000, "low": 13500, "high": 16500, "unit": "chars", "label": "Long (30+ min)"},
        }
    else:
        targets = {
            "short":  {"target": 1500, "low": 1200, "high": 1800, "unit": "words (net)", "label": "Short (10-15 min)"},
            "medium": {"target": 3000, "low": 2500, "high": 3500, "unit": "words (net)", "label": "Medium (20-25 min)"},
            "long":   {"target": 4500, "low": 4050, "high": 4950, "unit": "words (net)", "label": "Long (30+ min)"},
        }
    return targets.get(length_mode, targets["long"])
