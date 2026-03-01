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
# load_dotenv() — called in __main__ block (avoid side effects on import)
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from markdown_it import MarkdownIt
import weasyprint
from dr2_podcast.tools.link_validator import LinkValidatorTool
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Literal
import soundfile as sf
import numpy as np
import wave
from dr2_podcast.audio.engine import generate_audio_from_script, clean_script_for_tts, post_process_audio
from dr2_podcast.research.clinical import run_deep_research
from dr2_podcast.utils import strip_think_blocks
from dataclasses import fields as dc_fields

from dr2_podcast.config import EVIDENCE_LIMITED_THRESHOLD

# --- Extracted modules (T4.1) ---
from dr2_podcast.pipeline_sot import (
    build_imrad_sot as _build_imrad_sot_impl,
    _extract_conclusion_status,
    _parse_grade_sections,
    _format_study_characteristics_table,
    _format_references,
    _build_social_science_sot,
)
from dr2_podcast.pipeline_script import (
    _count_words,
    _deduplicate_script,
    _parse_blueprint_inventory,
    _validate_script as _validate_script_impl,
    _add_reaction_guidance as _add_reaction_guidance_impl,
    _quick_content_audit as _quick_content_audit_impl,
    _run_trim_pass as _run_trim_pass_impl,
    SCRIPT_TOLERANCE,
)
from dr2_podcast.pipeline_translation import (
    _split_sot_imrad,
    _estimate_translation_tokens,
    _split_at_subheaders,
    _translate_sot_pipelined as _translate_sot_pipelined_impl,
    _translate_prompt as _translate_prompt_impl,
    _audit_script_language as _audit_script_language_impl,
)
from dr2_podcast.pipeline_crew import (
    _estimate_task_tokens,
    _build_sot_injection_for_stage,
    _crew_kickoff_guarded,
    _SOT_BLOCK_RE,
    create_agents_and_tasks,
    PHASE_MARKERS,
    TASK_METADATA,
    display_workflow_plan,
    ProgressTracker,
)


logger = logging.getLogger(__name__)

class InsufficientEvidenceError(RuntimeError):
    """Raised when the affirmative research track finds zero candidates."""
    pass


def _write_insufficient_evidence_report(topic, aff_n, neg_n, output_dir_path):
    """Write a structured failure report when evidence is completely absent."""
    report_path = output_path(output_dir_path, "insufficient_evidence_report.md")
    strat_aff = strat_neg = "(not available)"
    strat_data = {}
    for fname, var in [("search_strategy_aff.json", "aff"), ("search_strategy_neg.json", "neg")]:
        p = output_path(output_dir_path, fname)
        if p.exists():
            try:
                data = json.loads(p.read_text())
                strat_data[var] = json.dumps(data.get("search_strings", {}), indent=2)
            except Exception as e:
                logger.warning(f"Failed to read search strategy {fname}: {e}")
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
    logger.warning(f"✗ Insufficient evidence report written → {report_path.name}")


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
            logging.FileHandler(output_path(output_dir, 'podcast_generation.log')),
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
# load_dotenv() — called in __main__ block
# Configuration loaded from .env
script_dir = Path(__file__).resolve().parent.parent  # project root (one level up from package)
base_output_dir = script_dir / "research_outputs"
# base_output_dir.mkdir(exist_ok=True) — called in __main__ block

# --- TIMESTAMPED OUTPUT DIRECTORY ---
def create_timestamped_output_dir(base_dir: Path) -> Path:
    """
    Create a timestamped subfolder for this podcast generation run.
    Format: research_outputs/YYYY-MM-DD_HH-MM-SS/

    Creates four subdirectories: research/, scripts/, audio/, meta/

    Args:
        base_dir: Base output directory (research_outputs)

    Returns:
        Path to timestamped directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamped_dir = base_dir / timestamp
    timestamped_dir.mkdir(parents=True, exist_ok=True)

    # Create per-phase subdirectories
    for subdir in OUTPUT_SUBDIRS:
        (timestamped_dir / subdir).mkdir(exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"OUTPUT DIRECTORY: {timestamped_dir}")
    logger.info(f"{'='*60}\n")

    return timestamped_dir


# --- OUTPUT SUBDIRECTORY MAPPING ---
# Maps file suffixes/patterns to subdirectories within each run folder.
OUTPUT_SUBDIRS = ("research", "scripts", "audio", "meta")

# Mapping from filename to subdirectory. Files not listed here stay in root.
_FILE_SUBDIR_MAP = {
    # research/ — research artifacts
    "source_of_truth.md": "research",
    "SOURCE_OF_TRUTH.md": "research",
    "research_sources.json": "research",
    "research_framing.md": "research",
    "affirmative_case.md": "research",
    "falsification_case.md": "research",
    "grade_synthesis.md": "research",
    "clinical_math.md": "research",
    "search_strategy_aff.json": "research",
    "search_strategy_neg.json": "research",
    "screening_results_aff.json": "research",
    "screening_results_neg.json": "research",
    "url_validation_results.json": "research",
    "domain_classification.json": "research",
    "RESEARCH_FRAMING.md": "research",
    "EPISODE_BLUEPRINT.md": "research",
    "ACCURACY_AUDIT.md": "research",
    "accuracy_audit.md": "research",
    "ACCURACY_CORRECTIONS.md": "research",
    # scripts/ — podcast scripts
    "script_draft.md": "scripts",
    "script_polished.md": "scripts",
    "script_final.md": "scripts",
    "script.txt": "scripts",
    # audio/ — audio files
    "audio.wav": "audio",
    "audio_mixed.wav": "audio",
    # meta/ — session metadata, logs, PDFs, checkpoints
    "session_metadata.txt": "meta",
    "podcast_generation.log": "meta",
    "checkpoint.json": "meta",
    "research_framing.pdf": "meta",
    "source_of_truth.pdf": "meta",
    "accuracy_audit.pdf": "meta",
}


def output_path(run_dir: Path, filename: str) -> Path:
    """Return the full path for a file within a run directory, using subdirectories.

    Checks _FILE_SUBDIR_MAP for known filenames. For dynamic filenames
    (e.g. source_of_truth_ja.md, podcast_*.wav), uses suffix-based heuristics.
    Falls back to run_dir root for unknown files.

    Also handles legacy flat directories: if the subdirectory does not exist
    (old run), falls back to run_dir / filename.
    """
    # Check explicit map first
    subdir = _FILE_SUBDIR_MAP.get(filename)

    # Heuristic fallbacks for dynamic filenames
    if subdir is None:
        lower = filename.lower()
        if lower.startswith("source_of_truth_") and lower.endswith(".md"):
            subdir = "research"
        elif lower.startswith("source_of_truth_") and lower.endswith(".pdf"):
            subdir = "meta"
        elif lower.startswith("podcast_") and (lower.endswith(".wav") or lower.endswith(".mp3")):
            subdir = "audio"
        elif lower.endswith(".pdf"):
            subdir = "meta"

    if subdir is None:
        return run_dir / filename

    # If subdirectory exists, use it; otherwise fall back to flat (legacy runs)
    subdir_path = run_dir / subdir
    if subdir_path.is_dir():
        return subdir_path / filename
    return run_dir / filename

# ================================================================
# CHECKPOINT / RESUME UTILITIES
# ================================================================
CHECKPOINT_FILE = "checkpoint.json"


def _serialize_dataclass(obj):
    """Recursively serialize a dataclass (or list/dict of dataclasses) to JSON-safe dicts."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_serialize_dataclass(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize_dataclass(v) for k, v in obj.items()}
    # Dataclass instances — use dc_fields to convert
    if hasattr(obj, '__dataclass_fields__'):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        result = {}
        for f in dc_fields(obj):
            val = getattr(obj, f.name)
            result[f.name] = _serialize_dataclass(val)
        return result
    # Fallback: try str()
    return str(obj)


def _deserialize_pipeline_data(pd_dict):
    """Deserialize pipeline_data dict back into dataclass objects.

    Imports clinical_research dataclasses lazily to avoid circular imports
    when pipeline.py is loaded as a module.
    """
    if not pd_dict:
        return pd_dict
    from dr2_podcast.research.clinical import (
        TieredSearchPlan, TierKeywords, WideNetRecord,
        DeepExtraction, PaperMetadata,
    )
    from dr2_podcast.research.clinical_math import ClinicalImpact

    def _restore_tier_keywords(d):
        if d is None or not isinstance(d, dict):
            return d
        return TierKeywords(
            intervention=d.get("intervention", []),
            outcome=d.get("outcome", []),
            population=d.get("population", []),
            rationale=d.get("rationale", ""),
        )

    def _restore_tiered_search_plan(d):
        if d is None or not isinstance(d, dict):
            return d
        return TieredSearchPlan(
            pico=d.get("pico", {}),
            tier1=_restore_tier_keywords(d.get("tier1")),
            tier2=_restore_tier_keywords(d.get("tier2")),
            tier3=_restore_tier_keywords(d.get("tier3")),
            role=d.get("role", ""),
            auditor_approved=d.get("auditor_approved", False),
            auditor_notes=d.get("auditor_notes", ""),
            revision_count=d.get("revision_count", 0),
        )

    def _restore_paper_metadata(d):
        if d is None or not isinstance(d, dict):
            return None
        return PaperMetadata.from_dict(d)

    def _restore_wide_net_record(d):
        if not isinstance(d, dict):
            return d
        pm = _restore_paper_metadata(d.get("paper_metadata"))
        valid_keys = {f.name for f in dc_fields(WideNetRecord)}
        kwargs = {k: v for k, v in d.items() if k in valid_keys and k != "paper_metadata"}
        # Supply defaults for required positional args that serializer may have dropped
        for req_opt in ("pmid", "doi", "sample_size", "primary_objective", "year", "journal", "authors"):
            kwargs.setdefault(req_opt, None)
        for req_str in ("title", "abstract", "study_type", "url", "source_db"):
            kwargs.setdefault(req_str, "")
        return WideNetRecord(**kwargs, paper_metadata=pm)

    def _restore_deep_extraction(d):
        if not isinstance(d, dict):
            return d
        pm = _restore_paper_metadata(d.get("paper_metadata"))
        valid_keys = {f.name for f in dc_fields(DeepExtraction)}
        kwargs = {k: v for k, v in d.items() if k in valid_keys and k != "paper_metadata"}
        # Required positional args that to_dict() may have dropped when None/empty
        kwargs.setdefault("pmid", None)
        kwargs.setdefault("doi", None)
        kwargs.setdefault("title", "")
        kwargs.setdefault("url", "")
        return DeepExtraction(**kwargs, paper_metadata=pm)

    def _restore_clinical_impact(d):
        if not isinstance(d, dict):
            return d
        valid_keys = {f.name for f in dc_fields(ClinicalImpact)}
        return ClinicalImpact(**{k: v for k, v in d.items() if k in valid_keys})

    restored = dict(pd_dict)  # shallow copy

    for key in ("aff_strategy", "fal_strategy"):
        if key in restored:
            restored[key] = _restore_tiered_search_plan(restored[key])

    for key in ("aff_extractions", "fal_extractions"):
        if key in restored and isinstance(restored[key], list):
            restored[key] = [_restore_deep_extraction(x) for x in restored[key]]

    for key in ("aff_top", "fal_top"):
        if key in restored and isinstance(restored[key], list):
            restored[key] = [_restore_wide_net_record(x) for x in restored[key]]

    if "impacts" in restored and isinstance(restored["impacts"], list):
        restored["impacts"] = [_restore_clinical_impact(x) for x in restored["impacts"]]

    return restored


def save_checkpoint(output_dir_path, phase_num, topic, language, pipeline_state):
    """Save a checkpoint after a phase completes successfully.

    Args:
        output_dir_path: Path to the output directory.
        phase_num: Phase number that just completed (0-8).
        topic: The topic string.
        language: Language code ('en', 'ja').
        pipeline_state: Dict of state to persist between phases.
            Keys may include: framing_output, deep_reports, sot_content,
            sot_file, sot_summary, evidence_quality, aff_candidates,
            neg_candidates, domain_classification, _research_domain, etc.
    """
    ckpt_path = output_path(Path(output_dir_path), CHECKPOINT_FILE)

    # Load existing checkpoint to preserve completed_phases list
    existing = {}
    if ckpt_path.exists():
        try:
            existing = json.loads(ckpt_path.read_text())
        except Exception:
            pass

    completed = existing.get("completed_phases", [])
    if phase_num not in completed:
        completed.append(phase_num)
        completed.sort()

    # Serialize pipeline_state — handle dataclass objects
    serialized_state = {}
    for key, val in pipeline_state.items():
        try:
            serialized_state[key] = _serialize_dataclass(val)
        except Exception as e:
            logger.warning(f"  Checkpoint: skipping non-serializable key '{key}': {e}")

    checkpoint = {
        "topic": topic,
        "language": language,
        "completed_phases": completed,
        "timestamp": datetime.now().isoformat(),
        "pipeline_state": serialized_state,
    }

    # Validate it's JSON-serializable before writing
    try:
        ckpt_json = json.dumps(checkpoint, indent=2, ensure_ascii=False, default=str)
        ckpt_path.write_text(ckpt_json)
        logger.info(f"  Checkpoint saved: phases {completed} -> {ckpt_path.name}")
    except Exception as e:
        logger.warning(f"  WARNING: Failed to save checkpoint: {e}")


def load_checkpoint(output_dir_path):
    """Load a checkpoint from a previous run.

    Returns:
        dict with keys: topic, language, completed_phases, timestamp, pipeline_state.
        Returns None if no checkpoint exists.
    """
    ckpt_path = output_path(Path(output_dir_path), CHECKPOINT_FILE)
    if not ckpt_path.exists():
        return None

    try:
        data = json.loads(ckpt_path.read_text())
    except Exception as e:
        logger.warning(f"  WARNING: Failed to load checkpoint: {e}")
        return None

    # Validate required keys
    required_keys = {"topic", "language", "completed_phases", "timestamp"}
    if not required_keys.issubset(data.keys()):
        missing = required_keys - data.keys()
        logger.warning(f"  WARNING: Checkpoint missing keys: {missing}")
        return None

    # Deserialize pipeline_data within pipeline_state if present
    ps = data.get("pipeline_state", {})
    if "deep_reports" in ps and isinstance(ps["deep_reports"], dict):
        if "pipeline_data" in ps["deep_reports"]:
            ps["deep_reports"]["pipeline_data"] = _deserialize_pipeline_data(
                ps["deep_reports"]["pipeline_data"]
            )

    return data


# ================================================================

# output_dir initialized in __main__ block (avoids creating directories on import)
output_dir = Path(".")  # sentinel — reassigned in __main__

# setup_logging(output_dir) — called in __main__ block

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
    parser.add_argument(
        '--resume',
        type=str,
        metavar='OUTPUT_DIR',
        help='Resume a previously failed pipeline run from the last completed phase'
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
        logger.info(f"Using topic from command-line: {topic}")
    elif os.getenv("PODCAST_TOPIC"):
        topic = os.getenv("PODCAST_TOPIC")
        logger.info(f"Using topic from environment: {topic}")
    else:
        topic = 'scientific benefit of coffee intake to increase productivity during the day'
        logger.info(f"Using default topic: {topic}")
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
        logger.info(f"Using language from command-line: {SUPPORTED_LANGUAGES[lang_code]['name']}")
    elif os.getenv("PODCAST_LANGUAGE") and os.getenv("PODCAST_LANGUAGE") in SUPPORTED_LANGUAGES:
        lang_code = os.getenv("PODCAST_LANGUAGE")
        logger.info(f"Using language from environment: {SUPPORTED_LANGUAGES[lang_code]['name']}")
    else:
        logger.info(f"Using default language: {SUPPORTED_LANGUAGES[lang_code]['name']}")
    return lang_code

args = None  # initialized in __main__
topic_name = None  # initialized in __main__

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

    logger.info(f"\n{'='*60}")
    logger.info(f"SESSION ROLE ASSIGNMENT ({host_config}):")
    logger.info(f"  Presenter: {presenter_label} ({HOSTS[presenter_label]['gender']})")
    logger.info(f"  Questioner: {questioner_label} ({HOSTS[questioner_label]['gender']})")
    logger.info(f"{'='*60}\n")

    return role_assignment

# SESSION_ROLES initialized in __main__ block (assign_roles() prints to stdout)
SESSION_ROLES = {
    "presenter": {"label": "", "stance": "", "personality": ""},
    "questioner": {"label": "", "stance": "", "personality": ""},
}  # sentinel — reassigned in __main__


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


def _find_artifact(src_dir: Path, name: str) -> Path:
    """Locate an artifact in src_dir, checking subdirectories then flat layout then legacy names."""
    # Check subdirectory layout first
    candidate = output_path(src_dir, name)
    if candidate.exists():
        return candidate
    # Flat layout fallback
    flat = src_dir / name
    if flat.exists():
        return flat
    # Legacy name fallback
    legacy = LEGACY_ARTIFACT_NAMES.get(name)
    if legacy:
        legacy_sub = output_path(src_dir, legacy)
        if legacy_sub.exists():
            return legacy_sub
        legacy_flat = src_dir / legacy
        if legacy_flat.exists():
            return legacy_flat
    return Path("")  # non-existent sentinel


def _copy_research_artifacts(src_dir: Path, dst_dir: Path):
    """Copy research-related files from a previous run to a new output directory.

    Handles source directories with flat layout, subdirectory layout, or legacy names.
    Always writes into subdirectory layout in dst_dir.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for subdir in OUTPUT_SUBDIRS:
        (dst_dir / subdir).mkdir(exist_ok=True)
    copied = 0
    for name in RESEARCH_ARTIFACTS:
        src = _find_artifact(src_dir, name)
        if src.exists():
            dst = output_path(dst_dir, name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    logger.info(f"  Copied {copied} research artifacts from {src_dir.name}")


def _copy_all_artifacts(src_dir: Path, dst_dir: Path):
    """Copy all files from a previous run to a new output directory.

    Handles both flat and subdirectory source layouts.
    Writes into subdirectory layout in dst_dir.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for subdir in OUTPUT_SUBDIRS:
        (dst_dir / subdir).mkdir(exist_ok=True)
    copied = 0
    # Copy files from root of src_dir
    for item in src_dir.iterdir():
        if item.is_file():
            dst = output_path(dst_dir, item.name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst)
            copied += 1
    # Copy files from subdirectories
    for subdir in OUTPUT_SUBDIRS:
        sub = src_dir / subdir
        if sub.is_dir():
            for item in sub.iterdir():
                if item.is_file():
                    dst = output_path(dst_dir, item.name)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.exists():
                        shutil.copy2(item, dst)
                        copied += 1
    logger.info(f"  Copied {copied} total artifacts from {src_dir.name}")


def check_supplemental_needed(topic: str, reuse_dir: Path) -> dict:
    """Ask the LLM if the previous source_of_truth.md adequately covers the new topic."""
    sot_path = _find_artifact(reuse_dir, "source_of_truth.md")
    if not sot_path.exists():
        sot_path = _find_artifact(reuse_dir, "SOURCE_OF_TRUTH.md")
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
        content = strip_think_blocks(content)

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
        logger.warning(f"  Supplemental check failed: {e}")

    # Default to needing supplement if check fails
    return {"needs_supplement": True, "reason": "Check failed, running supplemental as precaution", "queries": []}


# --- TTS DEPENDENCY CHECK ---
def check_tts_dependencies():
    """Verify Kokoro TTS is installed."""
    try:
        import kokoro
        logger.info("✓ Kokoro TTS dependencies verified")
    except ImportError as e:
        logger.error(f"CRITICAL ERROR: Kokoro TTS not installed: {e}")
        logger.error("Install with: pip install kokoro>=0.9")
        logger.error("Audio generation cannot proceed without Kokoro.")
        sys.exit(1)

# check_tts_dependencies() — called in __main__ block

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
# SCRIPT_TOLERANCE imported from pipeline_script

# --- All language/duration/channel/accessibility vars initialized in __main__ block ---
language = None  # initialized in __main__
language_config = {'name': '', 'tts_code': '', 'instruction': '', 'speech_rate': 0, 'length_unit': '', 'prompt_unit': ''}  # sentinel
english_instruction = "Write all content in English."
target_instruction = ""  # initialized in __main__
language_instruction = ""  # initialized in __main__
length_mode = ""  # initialized in __main__
_speech_rate = 0  # initialized in __main__
_target_min = 0  # initialized in __main__
target_length_int = 0  # initialized in __main__
target_script = "0"  # initialized in __main__
target_unit_singular = ""  # initialized in __main__
target_unit_plural = ""  # initialized in __main__
duration_label = ""  # initialized in __main__
channel_intro = ""  # initialized in __main__
core_target = ""  # initialized in __main__
channel_mission = ""  # initialized in __main__
ACCESSIBILITY_LEVEL = "simple"  # initialized in __main__

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
# All model config comes from .env — initialized in __main__ block
SMART_MODEL = None  # initialized in __main__
SMART_BASE_URL = None  # initialized in __main__
MID_MODEL = None  # initialized in __main__
MID_BASE_URL = None  # initialized in __main__

def get_final_model_string():
    model = SMART_MODEL
    base_url = SMART_BASE_URL
    logger.info(f"Connecting to Ollama server at {base_url}...")

    for i in range(10):
        try:
            response = httpx.get(f"{base_url}/models", timeout=5.0)
            if response.status_code == 200:
                logger.info(f"✓ Ollama server online! Using model: {model}")
                return model
        except Exception as e:
            if i % 5 == 0:
                logger.warning(f"Waiting for Ollama server... ({i}s) - {e}")
            time.sleep(1)

    logger.error("Error: Could not connect to Ollama server. Check if it is running.")
    logger.error("Start Ollama with: ollama serve")
    sys.exit(1)

final_model_string = None  # initialized in __main__

# LLM objects initialized in __main__ block (require network connection)
dgx_llm_strict = None  # initialized in __main__
dgx_llm_creative = None  # initialized in __main__
dgx_llm = None  # initialized in __main__


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
            logger.info(f"  ✓ {role} report summarized: {len(report_text)} → {len(summary)} chars")
            return summary
        # Summary too short — fall through to truncation
        logger.warning(f"  ⚠ {role} summary too short ({len(summary)} chars), falling back to truncation")
    except Exception as e:
        logger.warning(f"  ⚠ phi4-mini summarization failed for {role}: {e}")

    return report_text[:6000]


# _estimate_task_tokens, _build_sot_injection_for_stage, _crew_kickoff_guarded,
# _SOT_BLOCK_RE — imported from pipeline_crew


def _call_smart_model(system: str, user: str, max_tokens: int = 4000,
                      temperature: float = 0.1, timeout: int = 0,
                      frequency_penalty: float = 0.0) -> str:
    """Call the Smart Model (vLLM) directly via OpenAI API. Returns response text.

    Retries up to 3 times (4 total attempts) with exponential backoff + jitter.
    Fast-fails on non-transient errors (BadRequestError, AuthenticationError).
    timeout: seconds to wait. 0 = auto-scale based on max_tokens (~10 tok/s + 60s buffer).
    frequency_penalty: penalize repeated tokens (0.0 = off, 0.3 = moderate anti-repetition).
    """
    import openai
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
    create_kwargs = dict(
        model=SMART_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )
    if frequency_penalty > 0:
        create_kwargs["frequency_penalty"] = frequency_penalty
    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(**create_kwargs)
            text = resp.choices[0].message.content.strip()
            # Strip <think>...</think> blocks (Qwen3 thinking mode safety net)
            text = strip_think_blocks(text)
            return text
        except (openai.BadRequestError, openai.AuthenticationError):
            # Non-transient errors — fast-fail, no retry
            raise
        except (ConnectionError, TimeoutError, OSError,
                openai.APIConnectionError, openai.APITimeoutError,
                openai.InternalServerError) as e:
            if attempt < max_retries:
                base_wait = 5 * (2 ** attempt)  # 5, 10, 20
                jitter = random.uniform(-base_wait * 0.3, base_wait * 0.3)
                wait = base_wait + jitter
                logger.warning(
                    f"  WARNING: _call_smart_model() attempt {attempt+1}/{max_retries+1} "
                    f"failed ({type(e).__name__}), retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
            else:
                logger.error(f"  ERROR: _call_smart_model() failed after {max_retries+1} attempts: {e}")
                raise


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
        logger.warning(f"  ⚠ Mid-tier model ({MID_MODEL}) unavailable: {e} — falling back to Smart Model")
        return _call_smart_model(system, user, max_tokens=max_tokens, temperature=temperature, timeout=0)


# --- Translation functions (extracted to pipeline_translation.py) ---
# Thin wrappers that inject module-level globals into the extracted implementations.



def _translate_sot_pipelined(sot_content, language, language_config):
    """Wrapper — delegates to pipeline_translation with module-level model config."""
    return _translate_sot_pipelined_impl(
        sot_content, language, language_config,
        _call_smart_model=_call_smart_model,
        _call_mid_model=_call_mid_model,
        SMART_BASE_URL=SMART_BASE_URL, SMART_MODEL=SMART_MODEL,
        MID_BASE_URL=MID_BASE_URL, MID_MODEL=MID_MODEL,
    )


def _translate_prompt(prompt_text, language, language_config):
    """Wrapper — delegates to pipeline_translation with _call_smart_model."""
    return _translate_prompt_impl(
        prompt_text, language, language_config,
        _call_smart_model=_call_smart_model,
    )


def _audit_script_language(script_text, language, language_config):
    """Wrapper — delegates to pipeline_translation with _call_smart_model."""
    return _audit_script_language_impl(
        script_text, language, language_config,
        _call_smart_model=_call_smart_model,
    )



# --- Script functions (extracted to pipeline_script.py) ---
# Thin wrappers that inject module-level globals into the extracted implementations.
# _count_words, _deduplicate_script, _parse_blueprint_inventory — imported directly.


def _add_reaction_guidance(script_text, language_config):
    """Wrapper — delegates to pipeline_script with _call_smart_model."""
    return _add_reaction_guidance_impl(
        script_text, language_config,
        _call_smart_model=_call_smart_model,
    )


def _quick_content_audit(script_text, sot_content):
    """Wrapper — delegates to pipeline_script with _call_smart_model."""
    return _quick_content_audit_impl(
        script_text, sot_content,
        _call_smart_model=_call_smart_model,
        _truncate_at_boundary=_truncate_at_boundary,
    )


def _validate_script(script_text, target_length, tolerance, language_config, sot_content, stage):
    """Wrapper — delegates to pipeline_script with _call_smart_model."""
    return _validate_script_impl(
        script_text, target_length, tolerance, language_config, sot_content, stage,
        _call_smart_model=_call_smart_model,
        _truncate_at_boundary=_truncate_at_boundary,
    )


def _run_trim_pass(script_text, inventory, target_length, language_config,
                   session_roles, topic_name, target_instruction):
    """Wrapper — delegates to pipeline_script with _call_smart_model."""
    return _run_trim_pass_impl(
        script_text, inventory, target_length, language_config,
        session_roles, topic_name, target_instruction,
        _call_smart_model=_call_smart_model,
    )



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
    src_dir = Path(output_dir_path) if output_dir_path else output_dir
    sources_file = output_path(src_dir, "research_sources.json")
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
    logger.info(f"  Appended {len(new_sources)} sources to {role_key} library (total: {len(data[role_key])})")


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
    sources_file = output_path(output_dir, "research_sources.json")
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
    sources_file = output_path(output_dir, "research_sources.json")
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

    report_path = output_path(output_dir, name_map[key])
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
    clean = strip_think_blocks(str(content))
    body_html = _MD_PARSER.render(clean)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{_PDF_CSS}</style></head>
<body>
<div class="header">DGX Spark Research Intelligence Report</div>
<h1>{title}</h1>
{body_html}
</body></html>"""
    file_path = output_path(output_dir, filename)
    weasyprint.HTML(string=html).write_pdf(str(file_path))
    logger.info(f"PDF Generated: {file_path}")
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
    validation_file = output_path(output_dir, "url_validation_results.json")
    if not validation_file.exists():
        return "No pre-validation data available. Use Link Validator to check this URL."
    try:
        data = json.loads(validation_file.read_text())
        return data.get(url, f"Not pre-validated. Use Link Validator to check: {url}")
    except Exception as e:
        return f"Error reading validation data: {e}"


# Agent/Task objects — sentinel declarations. Properly constructed in _create_agents_and_tasks().
auditor_agent = None
producer_agent = None
editor_agent = None
framing_agent = None
framing_task = None
script_task = None
translation_task = None
polish_task = None
audit_task = None
blueprint_task = None


def _create_agents_and_tasks():
    """Construct all Agents and Tasks using current global variables.

    Delegates to pipeline_crew.create_agents_and_tasks(), passing all required
    module-level state. Sets module globals from the returned dict.
    """
    global auditor_agent, producer_agent, editor_agent, framing_agent
    global framing_task, script_task, translation_task, polish_task, audit_task, blueprint_task

    result = create_agents_and_tasks(
        topic_name=topic_name,
        language=language,
        language_config=language_config,
        english_instruction=english_instruction,
        target_instruction=target_instruction,
        target_script=target_script,
        target_unit_singular=target_unit_singular,
        target_unit_plural=target_unit_plural,
        _target_min=_target_min,
        target_length_int=target_length_int,
        SESSION_ROLES=SESSION_ROLES,
        channel_intro=channel_intro,
        core_target=core_target,
        channel_mission=channel_mission,
        dgx_llm_strict=dgx_llm_strict,
        dgx_llm_creative=dgx_llm_creative,
        SCRIPT_TOLERANCE=SCRIPT_TOLERANCE,
        output_dir=output_dir,
        output_path_fn=output_path,
        list_research_sources=list_research_sources,
        read_research_source=read_research_source,
        read_full_report=read_full_report,
        link_validator=link_validator,
    )

    auditor_agent = result['auditor_agent']
    producer_agent = result['producer_agent']
    editor_agent = result['editor_agent']
    framing_agent = result['framing_agent']
    framing_task = result['framing_task']
    script_task = result['script_task']
    translation_task = result['translation_task']
    polish_task = result['polish_task']
    audit_task = result['audit_task']
    blueprint_task = result['blueprint_task']



# ================================================================
# Build Source-of-Truth — extracted to pipeline_sot.py
# _extract_conclusion_status, _parse_grade_sections,
# _format_study_characteristics_table, _format_references,
# _build_social_science_sot — imported from pipeline_sot.
# ================================================================


def build_imrad_sot(topic, reports, ev_quality, aff_cand, domain="clinical"):
    """Wrapper — delegates to pipeline_sot with output_dir and output_path."""
    return _build_imrad_sot_impl(
        topic, reports, ev_quality, aff_cand,
        domain=domain,
        output_dir=output_dir,
        output_path_fn=output_path,
    )



# ════════════════════════════════════════════════════════════════════════
# SHARED PIPELINE FUNCTIONS  (T2.3 — deduplicated from 3 code paths)
# ════════════════════════════════════════════════════════════════════════

def _inject_blueprint_checklist(blueprint_task, script_task, length_mode, script_base_desc):
    """Parse Section 8 inventory from blueprint and inject coverage checklist into script task.

    Returns (inventory_dict, updated_script_base_desc).
    """
    blueprint_raw = strip_think_blocks(blueprint_task.output.raw)
    inventory = _parse_blueprint_inventory(blueprint_raw)
    if inventory:
        tier_filter = {
            'short': {'Basic'},
            'medium': {'Basic', 'Context'},
            'long': {'Basic', 'Context', 'Deep-dive'},
        }
        allowed_tiers = tier_filter.get(length_mode, {'Basic', 'Context', 'Deep-dive'})
        checklist_lines = ["\n\nCOVERAGE CHECKLIST — discuss EACH item below in its Act:"]
        for act_label, items in inventory.items():
            filtered = [it for it in items if it['tier'] in allowed_tiers]
            if filtered:
                checklist_lines.append(f"\n{act_label}:")
                for it in filtered:
                    checklist_lines.append(f"  [{it['tier']}] {it['question']}")
                    checklist_lines.append(f"    \u2192 {it['answer'][:120]}...")
        checklist_block = '\n'.join(checklist_lines)
        script_task.description = script_base_desc + checklist_block
        script_base_desc = script_task.description  # CRITICAL: update base for retry tasks
        logger.info(f"  Coverage checklist injected: "
              f"{sum(len(v) for v in inventory.values())} items "
              f"filtered to {len(allowed_tiers)} tiers")
    return inventory or {}, script_base_desc


def _run_script_draft(producer_agent, script_task, target_length_int, language_config, sot_content):
    """Phase 5: Generate script draft and validate.

    Returns (draft_text, draft_count).
    """
    Crew(agents=[producer_agent], tasks=[script_task], verbose=True).kickoff()
    draft_text = strip_think_blocks(script_task.output.raw)
    val = _validate_script(draft_text, target_length_int, SCRIPT_TOLERANCE,
                           language_config, sot_content, stage='draft')
    logger.info(f"    Draft: {val['word_count']} {language_config['length_unit']} — "
          f"{'PASS' if val['pass'] else 'NEEDS WORK'}")
    if not val['pass'] and any('TOO SHORT' in i for i in val['issues']):
        logger.warning(f"    \u26a0 Draft short ({val['word_count']} {language_config['length_unit']}) — "
              f"proceeding to polish (expansion removed; coverage checklist should prevent this)")
    draft_count = _count_words(draft_text, language_config)
    return draft_text, draft_count


def _run_polish_loop(draft_text, draft_count, inventory, target_length_int,
                     language_config, sot_content, script_task, polish_task,
                     editor_agent, translation_task, polish_base_desc,
                     polish_expected, max_attempts=3, *,
                     session_roles=None, topic_name=None, target_instruction=None):
    """Phase 6: Pre-polish trim, polish loop with feedback, shrinkage guard.

    Returns (polished_text, final_polish_task).
    """
    # Pre-polish trim: if over-target, reduce before polish to prevent poor cuts
    if draft_count > target_length_int * (1 + SCRIPT_TOLERANCE) and inventory:
        logger.info(f"  Draft over target ({draft_count}/{target_length_int}) — running inventory trim pass...")
        draft_text = _run_trim_pass(draft_text, inventory, target_length_int,
                                    language_config, session_roles, topic_name, target_instruction)
        draft_text = _deduplicate_script(draft_text, language_config)
        draft_count = _count_words(draft_text, language_config)
        logger.info(f"  Post-trim: {draft_count} {language_config['length_unit']}")
    elif draft_count < target_length_int * (1 - SCRIPT_TOLERANCE):
        logger.warning(f"  \u26a0 Draft still under target ({draft_count}/{target_length_int}) — proceeding to polish anyway")

    polish_feedback = ""
    current_polish = polish_task
    current_polish.context = [script_task]
    if translation_task is not None:
        current_polish.context = [script_task, translation_task]

    polished = ""
    for attempt in range(1, max_attempts + 1):
        if polish_feedback:
            fb = (
                f"\n\nPREVIOUS ATTEMPT FEEDBACK (attempt {attempt-1}):\n{polish_feedback}\n"
                f"Fix ALL issues listed above.\n"
            )
            current_polish = Task(
                description=polish_base_desc + fb,
                expected_output=polish_expected,
                agent=editor_agent,
                context=[script_task],
            )
            if translation_task is not None:
                current_polish.context = [script_task, translation_task]
        Crew(agents=[editor_agent], tasks=[current_polish], verbose=True).kickoff()
        polished = strip_think_blocks(current_polish.output.raw)
        val = _validate_script(polished, target_length_int, SCRIPT_TOLERANCE,
                               language_config, sot_content, stage='polish')
        logger.info(f"    Polish attempt {attempt}: {val['word_count']} {language_config['length_unit']} — "
              f"{'PASS' if val['pass'] else 'FAIL'}")
        if val['pass']:
            break
        polish_feedback = val['feedback']

    # Shrinkage guard
    polished_count = _count_words(polished, language_config)
    min_acceptable = int(target_length_int * (1 - SCRIPT_TOLERANCE))
    if polished_count < min_acceptable:
        logger.warning(f"    \u26a0 Polish shrunk script below minimum ({polished_count} < {min_acceptable}) — using draft")
        if draft_text.count('[TRANSITION]') < 3:
            draft_text = re.sub(r'(---\s*\n###\s*\*?\*?ACT)', r'[TRANSITION]\n\n\1', draft_text)
        polished = draft_text

    return polished, current_polish


def _run_accuracy_audit(audit_task, polish_task, auditor_agent, translation_task):
    """Phase 7: Run accuracy audit."""
    logger.info(f"\n  PHASE 7: ACCURACY AUDIT")
    audit_task.context = [polish_task]
    if translation_task is not None:
        audit_task.context = [polish_task, translation_task]
    Crew(agents=[auditor_agent], tasks=[audit_task], verbose=True).kickoff()


def _finalize_script(polished_text, polish_task, language, language_config, output_dir,
                     corrected_text=None):
    """Post-script: apply corrections, language audit, reaction guidance, save script_final.md.

    Returns final script_text ready for TTS.
    """
    if corrected_text:
        script_text = corrected_text
        logger.info("Using drift-corrected script for audio generation")
    else:
        script_text = polished_text if polished_text else (
            polish_task.output.raw if hasattr(polish_task, 'output') and polish_task.output else "")

    if language != 'en':
        script_text = _audit_script_language(script_text, language, language_config)

    logger.info("\nAdding reaction/emotion guidance to script...")
    script_text = _add_reaction_guidance(script_text, language_config)

    with open(output_path(output_dir, "script_final.md"), 'w', encoding='utf-8') as f:
        f.write(script_text)

    return script_text


def _save_task_outputs(output_dir, task_output_list):
    """Save markdown outputs from a list of (label, source, filename) tuples."""
    for label, source, filename in task_output_list:
        try:
            if isinstance(source, str):
                content = source
            elif hasattr(source, 'output') and source.output and hasattr(source.output, 'raw'):
                content = source.output.raw
            else:
                content = None
            if content and content.strip():
                outfile = output_path(output_dir, filename)
                with open(outfile, 'w') as f:
                    f.write(content)
                logger.info(f"  Saved {filename} ({len(content)} chars)")
        except Exception as e:
            logger.warning(f"  Warning: Could not save {filename}: {e}")


def _run_audio_pipeline(script_text, output_dir, language_config):
    """Phase 8: Clean script → TTS → BGM merge → duration check.

    Returns (audio_file_path_or_None, duration_minutes_or_None).
    """
    logger.info("\n--- Generating Multi-Voice Podcast Audio (Kokoro TTS) ---")
    cleaned_script = clean_script_for_tts(script_text)
    script_file = output_path(output_dir, "script.txt")
    with open(script_file, 'w') as f:
        f.write(cleaned_script)

    audio_output_path = output_path(output_dir, "audio.wav")
    audio_file = None
    duration_minutes = None

    try:
        tts_result = generate_audio_from_script(
            cleaned_script, str(audio_output_path),
            lang_code=language_config['tts_code'])
        if isinstance(tts_result, tuple):
            audio_file_path, transition_positions = tts_result
        else:
            audio_file_path, transition_positions = tts_result, []
        if audio_file_path:
            audio_file = Path(audio_file_path)
            logger.info(f"Audio generation complete: {audio_file}")
    except Exception as e:
        logger.error(f"\u2717 ERROR: Kokoro TTS failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    if audio_file:
        logger.info(f"Starting BGM Merging Phase...")
        try:
            mastered = post_process_audio(
                str(audio_file), bgm_target="Interesting BGM.wav",
                transition_positions_ms=transition_positions)
            if mastered and os.path.exists(mastered) and mastered != str(audio_file):
                audio_file = Path(mastered)
                logger.info(f"\u2713 BGM Merging Complete: {audio_file}")
        except Exception as e:
            logger.warning(f"\u26a0 BGM merging warning: {e}")

        try:
            with wave.open(str(audio_file), 'r') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                duration_seconds = frames / float(rate)
                duration_minutes = duration_seconds / 60
            logger.info(f"SUCCESS: Audio duration {duration_minutes:.2f} minutes")
        except Exception:
            pass

    return audio_file, duration_minutes


def _translate_and_inject_sot(sot_content, language, language_config, topic_name,
                              output_dir, sot_path, sot_summary, grade_injection,
                              blueprint_task, script_task, audit_task, translation_task):
    """Translate SOT and inject summary into Crew 3 task descriptions.

    Returns (translated_sot_text, sot_translated_file, translated_summary).
    """
    logger.info(f"\nPHASE 3: REPORT TRANSLATION (pipelined)")
    translated_sot = _translate_sot_pipelined(sot_content, language, language_config)
    sot_translated_file = None
    translated_summary = ""

    if translated_sot:
        sot_translated_file = output_path(output_dir, f"source_of_truth_{language}.md")
        with open(sot_translated_file, 'w', encoding='utf-8') as f:
            f.write(translated_sot)
        logger.info(f"\u2713 Translated SOT saved ({len(translated_sot)} chars)")
        logger.info("  Summarizing translated SOT for Crew 3 context injection...")
        translated_summary = summarize_report_with_fast_model(translated_sot, "sot_translated", topic_name)
        if translated_summary:
            tl_injection = _build_sot_injection_for_stage(
                1, sot_path, sot_translated_file,
                sot_summary, translated_summary, grade_injection, language_config
            )
            blueprint_task.description += tl_injection
            script_task.description += tl_injection
            audit_task.description += tl_injection
        # CRITICAL: compact reference only — full 84KB SOT as context causes 27K+ tokens
        # → CrewAI context overflow → infinite summarizer loop (observed: 36 cycles, 9.6h wasted)
        from types import SimpleNamespace
        translation_task.output = SimpleNamespace(raw=(
            f"[Translation complete \u2014 {len(translated_sot):,} chars]\n"
            f"Translated SOT saved: {sot_translated_file}\n"
            f"Key research summary injected into task descriptions."
        ))
    else:
        logger.warning(f"  Warning: Chunked translation produced no output \u2014 translated SOT not saved")

    return translated_sot, sot_translated_file, translated_summary


if __name__ == "__main__":
    # --- Runtime initialization (only when running as main script) ---
    load_dotenv()
    base_output_dir.mkdir(exist_ok=True)

    args = parse_arguments()

    # --- Resume mode: reuse existing output directory ---
    _resume_checkpoint = None
    if args.resume:
        _resume_dir = Path(args.resume)
        if not _resume_dir.is_absolute():
            _resume_dir = base_output_dir / args.resume
        if not _resume_dir.exists():
            print(f"ERROR: Resume directory does not exist: {_resume_dir}")
            sys.exit(1)
        _resume_checkpoint = load_checkpoint(_resume_dir)
        if _resume_checkpoint is None:
            print(f"ERROR: No valid {CHECKPOINT_FILE} found in {_resume_dir}")
            sys.exit(1)
        output_dir = _resume_dir
        print(f"\n{'='*70}")
        print(f"RESUME MODE: Resuming from {_resume_dir.name}")
        print(f"  Completed phases: {_resume_checkpoint['completed_phases']}")
        print(f"  Topic: {_resume_checkpoint['topic']}")
        print(f"  Last checkpoint: {_resume_checkpoint['timestamp']}")
        print(f"{'='*70}\n")
    else:
        output_dir = create_timestamped_output_dir(base_output_dir)
    setup_logging(output_dir)

    # Override topic/language from checkpoint if resuming
    if _resume_checkpoint:
        args.topic = _resume_checkpoint["topic"]
        if not args.language:
            args.language = _resume_checkpoint["language"]
    topic_name = get_topic(args)
    SESSION_ROLES = assign_roles()
    check_tts_dependencies()

    language = get_language(args)
    language_config = SUPPORTED_LANGUAGES[language]
    english_instruction = "Write all content in English."
    target_instruction = language_config['instruction']
    language_instruction = language_config['instruction']

    length_mode = os.getenv("PODCAST_LENGTH", "long").lower()
    _speech_rate = language_config['speech_rate']
    _target_min = TARGET_MINUTES.get(length_mode, TARGET_MINUTES['long'])
    target_length_int = _target_min * _speech_rate
    target_script = f"{target_length_int:,}"
    target_unit_singular = language_config['prompt_unit']
    target_unit_plural = language_config['length_unit']
    duration_label = f"{length_mode.capitalize()} ({_target_min} min)"

    channel_intro = os.getenv("PODCAST_CHANNEL_INTRO", "").strip()
    core_target = os.getenv("PODCAST_CORE_TARGET", "").strip()
    channel_mission = os.getenv("PODCAST_CHANNEL_MISSION", "").strip()

    ACCESSIBILITY_LEVEL = os.getenv("ACCESSIBILITY_LEVEL", "simple").lower()
    if ACCESSIBILITY_LEVEL not in ("simple", "moderate", "technical"):
        logger.warning(f"Warning: Unknown ACCESSIBILITY_LEVEL '{ACCESSIBILITY_LEVEL}', falling back to 'simple'")
        ACCESSIBILITY_LEVEL = "simple"
    logger.info(f"Accessibility level: {ACCESSIBILITY_LEVEL}")
    accessibility_instruction = ACCESSIBILITY_INSTRUCTIONS[ACCESSIBILITY_LEVEL]

    SMART_MODEL = os.environ["MODEL_NAME"]
    SMART_BASE_URL = os.environ["LLM_BASE_URL"]
    MID_MODEL = os.environ.get("MID_MODEL_NAME", "qwen2.5:7b")
    MID_BASE_URL = os.environ.get("MID_LLM_BASE_URL", os.environ.get("FAST_LLM_BASE_URL", "http://localhost:11434/v1"))

    final_model_string = get_final_model_string()

    dgx_llm_strict = LLM(
        model=final_model_string, base_url=SMART_BASE_URL, api_key="NA",
        provider="openai", timeout=600, temperature=0.1, max_tokens=8000,
        stop=["<|im_end|>", "<|endoftext|>"],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    dgx_llm_creative = LLM(
        model=final_model_string, base_url=SMART_BASE_URL, api_key="NA",
        provider="openai", timeout=600, temperature=0.7, max_tokens=16000,
        frequency_penalty=0.15, stop=["<|im_end|>", "<|endoftext|>"],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    dgx_llm = dgx_llm_strict

    # Construct all Agent/Task objects with correct runtime values
    logger.info(f"Podcast Length Mode: {duration_label}")
    _create_agents_and_tasks()


    # ================================================================
    # REUSE MODE BRANCHING
    # ================================================================
    # If --reuse-dir is specified, skip the normal pipeline and run the
    # appropriate reuse mode instead. This exits early via sys.exit(0).

    if args.reuse_dir:
        reuse_dir = Path(args.reuse_dir)
        logger.info(f"\n{'='*70}")
        logger.info(f"REUSE MODE: Reusing research from {reuse_dir.name}")
        logger.info(f"{'='*70}")

        if args.crew3_only:
            # --- CREW 3 ONLY: Skip research, run podcast production ---
            logger.info(f"\nMode: Crew 3 Only (podcast production)")

            # Create new output dir
            new_output_dir = create_timestamped_output_dir(base_output_dir)

            # Copy research artifacts
            _copy_research_artifacts(reuse_dir, new_output_dir)

            # Warn about missing critical clinical artifacts
            _critical_artifacts = ["grade_synthesis.md", "affirmative_case.md",
                                   "falsification_case.md", "research_sources.json"]
            for _art in _critical_artifacts:
                if not output_path(new_output_dir, _art).exists():
                    logger.warning(f"  ⚠ Missing clinical artifact '{_art}' — pipeline will rely on SOT content only")

            # Load source_of_truth.md content for context injection
            sot_path = output_path(new_output_dir, "source_of_truth.md")
            if not sot_path.exists():
                sot_path = output_path(new_output_dir, "SOURCE_OF_TRUTH.md")
            if not sot_path.exists():
                logger.error("ERROR: No source_of_truth.md found in reuse directory")
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
                        task_obj.output_file = str(output_path(new_output_dir, filename))

            logger.info(f"\nCREW 3: PODCAST PRODUCTION")

            _r_tl_summary = ""
            _r_sot_translated_file = None

            if translation_task is not None and sot_content:
                _r_translated, _r_sot_translated_file, _r_tl_summary = _translate_and_inject_sot(
                    sot_content, language, language_config, topic_name,
                    new_output_dir, sot_path,
                    _truncate_at_boundary(sot_content, 8000), "",
                    blueprint_task, script_task, audit_task, translation_task
                )
                if _r_sot_translated_file:
                    sot_translated_file = _r_sot_translated_file

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
            logger.info(f"\n  PHASE 4: EPISODE BLUEPRINT")
            _crew_kickoff_guarded(
                lambda: Crew(agents=[producer_agent], tasks=[blueprint_task], verbose=True),
                blueprint_task, translation_task, language,
                sot_path, _r_sot_translated_file,
                _truncate_at_boundary(sot_content, 8000), _r_tl_summary,
                "", language_config, "Phase 4 Blueprint"
            )

            # Inject blueprint inventory into script task
            _r_inventory, _reuse_script_base_desc = _inject_blueprint_checklist(
                blueprint_task, script_task, length_mode, _reuse_script_base_desc)

            # Phase 5: Script Draft
            logger.info(f"\n  PHASE 5: SCRIPT DRAFT")
            _r_draft_text, _r_draft_count = _run_script_draft(
                producer_agent, script_task, target_length_int, language_config, sot_content)

            # Phase 6: Script Polish
            logger.info(f"\n  PHASE 6: SCRIPT POLISH (audit loop)")
            _r_polished, polish_task = _run_polish_loop(
                _r_draft_text, _r_draft_count, _r_inventory, target_length_int,
                language_config, sot_content, script_task, polish_task,
                editor_agent, translation_task, _reuse_polish_base_desc,
                _reuse_polish_expected, _REUSE_MAX_ATTEMPTS,
                session_roles=SESSION_ROLES, topic_name=topic_name,
                target_instruction=target_instruction)

            # Phase 7: Accuracy Audit
            _run_accuracy_audit(audit_task, polish_task, auditor_agent, translation_task)

            # Finalize script (language audit, reaction guidance, save script_final.md)
            logger.info("\n--- Saving Outputs ---")
            script_text = _finalize_script(
                _r_polished, polish_task, language, language_config, new_output_dir)

            _save_task_outputs(new_output_dir, [
                ("Source of Truth (Translated)", translation_task, "source_of_truth.md"),
                ("Episode Blueprint", blueprint_task, "EPISODE_BLUEPRINT.md"),
                ("Script Draft", script_task, "script_draft.md"),
                ("Accuracy Audit", audit_task, "accuracy_audit.md"),
            ])

            # Generate PDF for accuracy audit
            try:
                acc_content = audit_task.output.raw if hasattr(audit_task, 'output') and audit_task.output else ""
                if acc_content:
                    create_pdf("Accuracy Audit", acc_content, "accuracy_audit.pdf")
            except Exception:
                pass

            # TTS + BGM
            _run_audio_pipeline(script_text, new_output_dir, language_config)

            # Session metadata
            session_metadata = (
                f"PODCAST SESSION METADATA (REUSE: crew3_only)\n{'='*60}\n\n"
                f"Topic: {topic_name}\n"
                f"Language: {language_config['name']} ({language})\n"
                f"Reused from: {reuse_dir}\n"
            )
            with open(output_path(new_output_dir, "session_metadata.txt"), 'w') as f:
                f.write(session_metadata)

            logger.info(f"\n{'='*70}")
            logger.info("REUSE_COMPLETE: CREW3_ONLY")
            logger.info(f"{'='*70}")
            sys.exit(0)

        elif args.check_supplemental:
            # --- CHECK SUPPLEMENTAL: LLM decides if supplement needed ---
            logger.info(f"\nMode: Check Supplemental")

            result = check_supplemental_needed(topic_name, reuse_dir)
            logger.info(f"  Needs supplement: {result['needs_supplement']}")
            logger.info(f"  Reason: {result['reason']}")

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
                with open(output_path(new_output_dir, "session_metadata.txt"), 'w') as f:
                    f.write(session_metadata)

                logger.info(f"\n{'='*70}")
                logger.info("REUSE_COMPLETE: NO_CHANGES")
                logger.info(f"{'='*70}")
                sys.exit(0)

            else:
                # Supplemental research needed
                logger.info(f"\nSUPPLEMENTAL RESEARCH needed: {result['reason']}")
                logger.info(f"  Running {len(result['queries'])} supplemental searches...")

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
                                logger.warning(f"  Supplemental search failed for '{query_str}': {e}")
                        supp_text = "\n\n".join(supp_parts)
                        if supp_text:
                            logger.info(f"  Found supplemental evidence ({len(supp_parts)} results)")
                        else:
                            logger.warning("  No supplemental results found")
                    else:
                        logger.warning("  No BRAVE_API_KEY set, skipping supplemental search")

                # Load existing source_of_truth for context
                sot_path = output_path(new_output_dir, "source_of_truth.md")
                if not sot_path.exists():
                    sot_path = output_path(new_output_dir, "SOURCE_OF_TRUTH.md")
                sot_content = sot_path.read_text() if sot_path.exists() else ""

                # Append supplemental findings to SOT
                if supp_text:
                    sot_content += (
                        f"\n\n## Supplemental Research Findings\n\n"
                        f"{supp_text}\n"
                    )
                    with open(output_path(new_output_dir, "source_of_truth.md"), 'w') as f:
                        f.write(sot_content)
                    logger.info(f"  Updated source_of_truth.md with supplemental findings ({len(sot_content)} chars)")

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
                        task_obj.output_file = str(output_path(new_output_dir, filename))

                logger.info(f"\nCREW 3: PODCAST PRODUCTION")

                _s_tl_summary = ""
                _s_sot_translated_file = None

                if translation_task is not None and sot_content:
                    _s_translated, _s_sot_translated_file, _s_tl_summary = _translate_and_inject_sot(
                        sot_content, language, language_config, topic_name,
                        new_output_dir, sot_path,
                        _truncate_at_boundary(sot_content, 8000), "",
                        blueprint_task, script_task, audit_task, translation_task
                    )
                    if _s_sot_translated_file:
                        sot_translated_file = _s_sot_translated_file

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
                logger.info(f"\n  PHASE 4: EPISODE BLUEPRINT")
                _crew_kickoff_guarded(
                    lambda: Crew(agents=[producer_agent], tasks=[blueprint_task], verbose=True),
                    blueprint_task, translation_task, language,
                    sot_path, _s_sot_translated_file,
                    _truncate_at_boundary(sot_content, 8000), _s_tl_summary,
                    "", language_config, "Phase 4 Blueprint"
                )

                # Inject blueprint inventory into script task
                _s_inventory, _supp_script_base_desc = _inject_blueprint_checklist(
                    blueprint_task, script_task, length_mode, _supp_script_base_desc)

                # Phase 5: Script Draft
                logger.info(f"\n  PHASE 5: SCRIPT DRAFT")
                _s_draft_text, _s_draft_count = _run_script_draft(
                    producer_agent, script_task, target_length_int, language_config, sot_content)

                # Phase 6: Script Polish
                logger.info(f"\n  PHASE 6: SCRIPT POLISH (audit loop)")
                _s_polished, polish_task = _run_polish_loop(
                    _s_draft_text, _s_draft_count, _s_inventory, target_length_int,
                    language_config, sot_content, script_task, polish_task,
                    editor_agent, translation_task, _supp_polish_base_desc,
                    _supp_polish_expected, _SUPP_MAX_ATTEMPTS,
                    session_roles=SESSION_ROLES, topic_name=topic_name,
                    target_instruction=target_instruction)

                # Phase 7: Accuracy Audit
                _run_accuracy_audit(audit_task, polish_task, auditor_agent, translation_task)

                # Finalize script (language audit, reaction guidance, save script_final.md)
                logger.info("\n--- Saving Outputs ---")
                script_text = _finalize_script(
                    _s_polished, polish_task, language, language_config, new_output_dir)

                _save_task_outputs(new_output_dir, [
                    ("Source of Truth (Translated)", translation_task, "source_of_truth.md"),
                    ("Episode Blueprint", blueprint_task, "EPISODE_BLUEPRINT.md"),
                    ("Script Draft", script_task, "script_draft.md"),
                    ("Accuracy Audit", audit_task, "accuracy_audit.md"),
                ])

                # TTS + BGM
                _run_audio_pipeline(script_text, new_output_dir, language_config)

                # Session metadata
                session_metadata = (
                    f"PODCAST SESSION METADATA (REUSE: supplemental)\n{'='*60}\n\n"
                    f"Topic: {topic_name}\n"
                    f"Language: {language_config['name']} ({language})\n"
                    f"Reused from: {reuse_dir}\n"
                    f"Supplemental reason: {result['reason'] if isinstance(result, dict) else 'N/A'}\n"
                )
                with open(output_path(new_output_dir, "session_metadata.txt"), 'w') as f:
                    f.write(session_metadata)

                logger.info(f"\n{'='*70}")
                logger.info("REUSE_COMPLETE: SUPPLEMENTAL")
                logger.info(f"{'='*70}")
                sys.exit(0)

    # ================================================================
    # NORMAL PIPELINE (no --reuse-dir)
    # ================================================================

    # --- Checkpoint/Resume state ---
    _completed_phases = set()
    _pipeline_state = {}
    if _resume_checkpoint:
        _completed_phases = set(_resume_checkpoint.get("completed_phases", []))
        _pipeline_state = _resume_checkpoint.get("pipeline_state", {})

    def _phase_done(phase_num):
        """Check if a phase was already completed (for resume)."""
        return phase_num in _completed_phases

    def _save_phase(phase_num, extra_state=None):
        """Save checkpoint after a phase completes."""
        if extra_state:
            _pipeline_state.update(extra_state)
        save_checkpoint(output_dir, phase_num, topic_name, language, _pipeline_state)

    # --- EXECUTION (Streamlined Pipeline) ---
    # Display workflow plan before execution
    display_workflow_plan(topic_name, language_config, output_dir)

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

    logger.info(f"\n--- Initiating Scientific Research Pipeline on DGX Spark ---")
    logger.info(f"Topic: {topic_name}")
    logger.info(f"Language: {language_config['name']} ({language})")
    if _completed_phases:
        logger.info(f"Resuming: phases {sorted(_completed_phases)} already complete")
    logger.info("---\n")

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
    from dr2_podcast.research.domain_classifier import classify_topic, ResearchDomain

    if _phase_done(0):
        logger.info(f"\n{'='*70}")
        logger.info(f"PHASE 0: RESEARCH FRAMING — already complete, skipping")
        logger.info(f"{'='*70}")
        # Restore framing_output from disk or checkpoint
        _framing_path = output_path(output_dir, "research_framing.md")
        framing_output = _pipeline_state.get("framing_output", "")
        if not framing_output and _framing_path.exists():
            framing_output = _framing_path.read_text()
        # Restore domain_classification from disk
        _dc_path = output_path(output_dir, "domain_classification.json")
        if _dc_path.exists():
            _dc_data = json.loads(_dc_path.read_text())
            _dc_domain_val = _dc_data.get("domain", "clinical")
            # Build a simple namespace to carry domain info
            class _DCProxy:
                pass
            domain_classification = _DCProxy()
            domain_classification.domain = ResearchDomain(_dc_domain_val)
            domain_classification.confidence = _dc_data.get("confidence", 0.0)
            domain_classification.reasoning = _dc_data.get("reasoning", "")
            domain_classification.suggested_framework = _dc_data.get("framework", "")
            domain_classification.primary_databases = _dc_data.get("databases", [])
        else:
            # Fallback: re-run classification (fast, deterministic)
            _smart_base = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
            _smart_model = os.environ.get("MODEL_NAME", "")
            try:
                from openai import AsyncOpenAI as _AOAIClassify
                _classify_client = _AOAIClassify(base_url=_smart_base, api_key="not-needed")
            except Exception:
                _classify_client = None
            domain_classification = asyncio.run(classify_topic(
                topic=topic_name, smart_client=_classify_client, smart_model=_smart_model,
            ))
        logger.info(f"  Restored: framing_output={len(framing_output)} chars, "
              f"domain={domain_classification.domain.value}")
    else:
        logger.info(f"\n{'='*70}")
        logger.info(f"PHASE 0: RESEARCH FRAMING")
        logger.info(f"{'='*70}")

        # Step 0a: classify domain first (fast, mostly deterministic) so framing is domain-aware
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
        logger.info(f"  Domain: {domain_classification.domain.value} "
              f"(confidence={domain_classification.confidence:.2f}, framework={domain_classification.suggested_framework})")
        _dc_path = output_path(output_dir, "domain_classification.json")
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
            logger.info(f"✓ Phase 0 complete: Research framing generated ({len(framing_output)} chars)")
        except Exception as e:
            logger.warning(f"⚠ Phase 0 (Research Framing) failed: {e}")
            logger.info("Continuing without framing context...")
            framing_output = ""

        # Save Phase 0 checkpoint
        _save_phase(0, {"framing_output": framing_output})

    # ================================================================
    # PHASE 1: RESEARCH PIPELINE (domain-routed)
    # ================================================================
    _research_domain = domain_classification.domain.value  # "clinical" | "social_science" | "general"

    sot_content = ""  # Will hold the synthesized Source-of-Truth
    sot_file = None   # Path to source_of_truth.md (set after deep research completes)
    sot_summary = ""  # Fast-model summary of sot_content (set after deep research completes)
    aff_candidates = 0
    neg_candidates = 0
    evidence_quality = "sufficient"
    deep_reports = None

    if _phase_done(1):
        logger.info(f"\n{'='*70}")
        logger.info(f"PHASE 1: RESEARCH PIPELINE — already complete, skipping")
        logger.info(f"{'='*70}")
        # Restore key variables from checkpoint state
        evidence_quality = _pipeline_state.get("evidence_quality", "sufficient")
        aff_candidates = _pipeline_state.get("aff_candidates", 0)
        neg_candidates = _pipeline_state.get("neg_candidates", 0)
        _research_domain = _pipeline_state.get("_research_domain", _research_domain)
        # Restore deep_reports from checkpoint (with deserialized pipeline_data)
        _ckpt_dr = _pipeline_state.get("deep_reports")
        if _ckpt_dr and isinstance(_ckpt_dr, dict):
            deep_reports = _ckpt_dr
        # Restore sot_content from disk
        _sot_path = output_path(output_dir, "source_of_truth.md")
        if _sot_path.exists():
            sot_content = _sot_path.read_text()
            sot_file = _sot_path
        # Restore sot_summary from checkpoint or re-generate
        sot_summary = _pipeline_state.get("sot_summary", "")
        if not sot_summary and sot_content:
            logger.info("  Re-summarizing Source-of-Truth with fast model...")
            sot_summary = summarize_report_with_fast_model(sot_content, "sot", topic_name)
        logger.info(f"  Restored: sot={len(sot_content)} chars, evidence={evidence_quality}, "
              f"aff={aff_candidates}, neg={neg_candidates}")
    else:
        if _research_domain in ("social_science",):
            # Social science pipeline (Phase E implementation)
            logger.info(f"\n{'='*70}")
            logger.info(f"PHASE 1: SOCIAL SCIENCE RESEARCH (PECO Pipeline)")
            logger.info(f"{'='*70}")
            try:
                from dr2_podcast.research.social_science import run_social_science_research
                deep_reports = asyncio.run(run_social_science_research(
                    topic=topic_name,
                    framing_context=framing_output,
                    output_dir=str(output_dir),
                ))
            except ImportError:
                logger.warning("⚠ social_science_research module not yet available — falling back to clinical pipeline")
                _research_domain = "clinical"  # fall through to clinical below
            except Exception as e:
                logger.warning(f"⚠ Social science pipeline failed: {e} — falling back to clinical pipeline")
                _research_domain = "clinical"

        if _research_domain in ("clinical", "general"):
            logger.info(f"\n{'='*70}")
            logger.info(f"PHASE 1: CLINICAL RESEARCH")
            logger.info(f"{'='*70}")

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
                    logger.info(f"✓ Fast model ready: {_fast_model_name}")
                else:
                    logger.warning(f"⚠ Fast model '{_fast_model_name}' not found in Ollama. Available: {_models}")
                    logger.warning(f"  Falling back to smart-only mode. Run: ollama pull {_fast_model_name}")
        except Exception:
            logger.warning(f"⚠ Fast model not available (Ollama unreachable at {_fast_base_url}). Running in smart-only mode.")

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
                p = output_path(output_dir, fname)
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
                    logger.warning(f"  ⚠ {role_name.capitalize()} report missing — skipping save")
                    continue
                report_file = output_path(output_dir, filename)
                with open(report_file, 'w') as f:
                    f.write(report.report)
                logger.info(f"✓ {role_name.capitalize()} report saved: {report_file} ({report.total_summaries} sources)")

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
            sources_file = output_path(output_dir, "research_sources.json")
            with open(sources_file, 'w') as f:
                json.dump(sources_json, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Research library saved: {sources_file} "
                  f"(lead={len(sources_json['lead'])}, counter={len(sources_json['counter'])} sources)")

            deep_audit_report = deep_reports.get("audit")
            lead_report = deep_reports.get("lead")
            counter_report = deep_reports.get("counter")


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
            sot_file = output_path(output_dir, "source_of_truth.md")
            with open(sot_file, 'w') as f:
                f.write(sot_content)
            logger.info(f"✓ Source of Truth (IMRaD) generated from deep research ({len(sot_content)} chars)")

            # Summarize for injection into Crew 3 task descriptions
            logger.info("Summarizing Source-of-Truth with fast model...")
            sot_summary = summarize_report_with_fast_model(sot_content, "sot", topic_name)

        except InsufficientEvidenceError:
            raise
        except Exception as e:
            logger.warning(f"⚠ Deep research pre-scan failed: {e}")
            logger.info("Continuing without deep research...")
            deep_reports = None
            sot_summary = ""

        # Save Phase 1 checkpoint
        _save_phase(1, {
            "deep_reports": deep_reports,
            "sot_summary": sot_summary,
            "evidence_quality": evidence_quality,
            "aff_candidates": aff_candidates,
            "neg_candidates": neg_candidates,
            "_research_domain": _research_domain,
        })

    # ================================================================
    # PHASE 2: SOURCE VALIDATION (batch, parallel)
    # ================================================================
    if _phase_done(2):
        logger.info(f"\n{'='*70}")
        logger.info(f"PHASE 2: SOURCE VALIDATION — already complete, skipping")
        logger.info(f"{'='*70}")
    else:
        logger.info(f"\n{'='*70}")
        logger.info(f"PHASE 2: SOURCE VALIDATION")
        logger.info(f"{'='*70}")

        from dr2_podcast.tools.link_validator import validate_multiple_urls_parallel

        all_urls = set()
        url_pattern = re.compile(r'https?://[^\s\)\]\"\'<>]+')

        # Collect URLs from source library
        sources_file = output_path(output_dir, "research_sources.json")
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

        logger.info(f"  Found {len(all_urls)} unique URLs to validate")

        if all_urls:
            validation_results = validate_multiple_urls_parallel(list(all_urls), max_workers=15)
            valid_count = sum(1 for v in validation_results.values() if "Valid" in v)
            broken_count = sum(1 for v in validation_results.values() if "Broken" in v or "Invalid" in v)
            logger.info(f"  Results: {valid_count} valid, {broken_count} broken, "
                  f"{len(validation_results) - valid_count - broken_count} other")

            validation_file = output_path(output_dir, "url_validation_results.json")
            with open(validation_file, 'w') as f:
                json.dump(validation_results, f, indent=2, ensure_ascii=False)
            logger.info(f"  Saved to {validation_file}")
        else:
            logger.warning("  No URLs found to validate")

        # Save Phase 2 checkpoint
        _save_phase(2)

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
    grade_file = output_path(output_dir, "grade_synthesis.md")
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
    logger.info(f"\n{'='*70}")
    logger.info(f"CREW 3: PODCAST PRODUCTION")
    logger.info(f"{'='*70}")

    translated_sot = None  # set below if translation runs
    translated_sot_summary = ""
    sot_translated_file = None
    if _phase_done(3):
        logger.info(f"  Phase 3 (Translation) — already complete, skipping LLM translation")
        # Restore translated SOT from disk if it exists
        _tl_path = output_path(output_dir, f"source_of_truth_{language}.md")
        if _tl_path.exists():
            translated_sot = _tl_path.read_text()
            sot_translated_file = _tl_path
            translated_sot_summary = _pipeline_state.get("translated_sot_summary", "")
            if not translated_sot_summary and translated_sot:
                translated_sot_summary = summarize_report_with_fast_model(translated_sot, "sot_tl", topic_name)
            # Re-inject into tasks (needed since task objects are fresh)
            if sot_translated_file and translation_task is not None:
                _tl_injection = (
                    f"\n\nTRANSLATED SOURCE OF TRUTH:\n"
                    f"{translated_sot_summary if translated_sot_summary else translated_sot[:8000]}\n"
                    f"--- END TRANSLATED SOT ---\n"
                )
                script_task.description += _tl_injection
                blueprint_task.description += _tl_injection
                audit_task.description += _tl_injection
    elif translation_task is not None and sot_content:
        translated_sot, sot_translated_file, translated_sot_summary = _translate_and_inject_sot(
            sot_content, language, language_config, topic_name,
            output_dir, sot_file,
            sot_summary, _grade_injection,
            blueprint_task, script_task, audit_task, translation_task
        )
        # Save Phase 3 checkpoint
        _save_phase(3, {"translated_sot_summary": translated_sot_summary})
    else:
        # No translation needed (English) — mark Phase 3 done
        if not _phase_done(3):
            _save_phase(3)

    # --- Extract base task descriptions for audit-loop feedback injection ---
    script_task_base_description = script_task.description
    script_task_expected_output = script_task.expected_output
    polish_task_base_description = polish_task.description
    polish_task_expected_output = polish_task.expected_output

    # Bug 5: Translate task descriptions for non-English runs
    if language != 'en':
        logger.info(f"\nTranslating Crew 3 task prompts to {language_config['name']}...")
        for _task, _name in [
            (blueprint_task, "blueprint"),
            (script_task, "script"),
            (polish_task, "polish"),
            (audit_task, "audit"),
        ]:
            _task.description = _translate_prompt(_task.description, language, language_config)
            logger.info(f"  ✓ {_name} task prompt translated")
        # Keep expected_output in English (CrewAI internals)
        # Update base descriptions with translated versions
        script_task_base_description = script_task.description
        polish_task_base_description = polish_task.description

    # Start background monitor for crew 3
    monitor = CrewMonitor(all_task_list, progress_tracker)
    monitor.start()

    MAX_SCRIPT_ATTEMPTS = 3

    # Variables that may be restored from checkpoint for later phases
    script_draft_text = ""
    _draft_count = 0
    polished_text = ""
    _bp_inventory = {}

    try:
        # === PHASE 4: BLUEPRINT (single task, no loop) ===
        if _phase_done(4):
            logger.info(f"\n{'='*60}")
            logger.info("PHASE 4: EPISODE BLUEPRINT — already complete, skipping")
            logger.info(f"{'='*60}")
            # Restore blueprint output from disk to inject into script task
            _bp_path = output_path(output_dir, "EPISODE_BLUEPRINT.md")
            if _bp_path.exists():
                _bp_text = _bp_path.read_text()
                # Simulate blueprint_task.output for downstream code
                class _FakeOutput:
                    def __init__(self, raw):
                        self.raw = raw
                blueprint_task.output = _FakeOutput(_bp_text)
            # Re-inject blueprint inventory into script task
            _bp_inventory, script_task_base_description = _inject_blueprint_checklist(
                blueprint_task, script_task, length_mode, script_task_base_description)
        else:
            logger.info(f"\n{'='*60}")
            logger.info("PHASE 4: EPISODE BLUEPRINT")
            logger.info(f"{'='*60}")
            _crew_kickoff_guarded(
                lambda: Crew(agents=[producer_agent], tasks=[blueprint_task], verbose=True),
                blueprint_task, translation_task, language,
                sot_file, sot_translated_file,
                sot_summary, translated_sot_summary,
                _grade_injection, language_config, "Phase 4 Blueprint"
            )
            logger.info("  ✓ Blueprint complete")

            # Inject blueprint inventory into script task
            _bp_inventory, script_task_base_description = _inject_blueprint_checklist(
                blueprint_task, script_task, length_mode, script_task_base_description)

            # Save Phase 4 checkpoint
            _save_phase(4)

        # === PHASE 5: SCRIPT DRAFT ===
        if _phase_done(5):
            logger.info(f"\n{'='*60}")
            logger.info("PHASE 5: SCRIPT DRAFT — already complete, skipping")
            logger.info(f"{'='*60}")
            # Restore script draft from disk
            _sd_path = output_path(output_dir, "script_draft.md")
            if _sd_path.exists():
                script_draft_text = _sd_path.read_text()
                _draft_count = _count_words(script_draft_text, language_config)
                logger.info(f"  Restored: {_draft_count} {language_config['length_unit']}")
            else:
                logger.warning("  WARNING: script_draft.md not found — re-running Phase 5")
                script_draft_text, _draft_count = _run_script_draft(
                    producer_agent, script_task, target_length_int, language_config, sot_content)
                _save_phase(5)
        else:
            logger.info(f"\n{'='*60}")
            logger.info("PHASE 5: SCRIPT DRAFT")
            logger.info(f"{'='*60}")
            script_draft_text, _draft_count = _run_script_draft(
                producer_agent, script_task, target_length_int, language_config, sot_content)

            # Save Phase 5 checkpoint (also save draft to disk for resume)
            _sd_path = output_path(output_dir, "script_draft.md")
            with open(_sd_path, 'w', encoding='utf-8') as _f:
                _f.write(script_draft_text)
            _save_phase(5)

        # === PHASE 6: POLISH + SHRINKAGE GUARD ===
        if _phase_done(6):
            logger.info(f"\n{'='*60}")
            logger.info("PHASE 6: SCRIPT POLISH — already complete, skipping")
            logger.info(f"{'='*60}")
            # Restore polished script from disk
            _pol_path = output_path(output_dir, "script_polished.md")
            if not _pol_path.exists():
                _pol_path = output_path(output_dir, "script_final.md")
            if _pol_path.exists():
                polished_text = _pol_path.read_text()
                _pol_count = _count_words(polished_text, language_config)
                logger.info(f"  Restored: {_pol_count} {language_config['length_unit']}")
                # Simulate polish_task.output for downstream code
                class _FakeOutput:
                    def __init__(self, raw):
                        self.raw = raw
                polish_task.output = _FakeOutput(polished_text)
            else:
                logger.warning("  WARNING: polished script not found — re-running Phase 6")
                polished_text, polish_task = _run_polish_loop(
                    script_draft_text, _draft_count, _bp_inventory, target_length_int,
                    language_config, sot_content, script_task, polish_task,
                    editor_agent, translation_task, polish_task_base_description,
                    polish_task_expected_output, MAX_SCRIPT_ATTEMPTS,
                    session_roles=SESSION_ROLES, topic_name=topic_name,
                    target_instruction=target_instruction)
                _save_phase(6)
        else:
            logger.info(f"\n{'='*60}")
            logger.info("PHASE 6: SCRIPT POLISH (audit loop, max {0} attempts)".format(MAX_SCRIPT_ATTEMPTS))
            logger.info(f"{'='*60}")
            polished_text, polish_task = _run_polish_loop(
                script_draft_text, _draft_count, _bp_inventory, target_length_int,
                language_config, sot_content, script_task, polish_task,
                editor_agent, translation_task, polish_task_base_description,
                polish_task_expected_output, MAX_SCRIPT_ATTEMPTS,
                session_roles=SESSION_ROLES, topic_name=topic_name,
                target_instruction=target_instruction)

            # Save polished script to disk for resume
            _pol_path = output_path(output_dir, "script_polished.md")
            with open(_pol_path, 'w', encoding='utf-8') as _f:
                _f.write(polished_text)
            _save_phase(6)

        # === PHASE 7: ACCURACY AUDIT (advisory, existing logic) ===
        if _phase_done(7):
            logger.info(f"\n{'='*60}")
            logger.info("PHASE 7: ACCURACY AUDIT — already complete, skipping")
            logger.info(f"{'='*60}")
            # Restore audit output from disk
            _aud_path = output_path(output_dir, "accuracy_audit.md")
            if _aud_path.exists():
                _aud_text = _aud_path.read_text()
                class _FakeOutput:
                    def __init__(self, raw):
                        self.raw = raw
                audit_task.output = _FakeOutput(_aud_text)
        else:
            logger.info(f"\n{'='*60}")
            logger.info("PHASE 7: ACCURACY AUDIT")
            logger.info(f"{'='*60}")
            _run_accuracy_audit(audit_task, polish_task, auditor_agent, translation_task)

            # Save Phase 7 checkpoint
            _save_phase(7)

    except Exception as e:
        logger.error(f"\n{'='*70}")
        logger.error("CREW 3 FAILED — saving partial outputs")
        logger.error(f"{'='*70}")
        logger.error(f"Error: {e}")
        _partial_tasks = [
            ("blueprint_partial.md", blueprint_task),
            ("script_draft_partial.md", script_task),
            ("script_polished_partial.md", polish_task),
            ("audit_partial.md", audit_task),
        ]
        for _fname, _task in _partial_tasks:
            try:
                if hasattr(_task, 'output') and _task.output and hasattr(_task.output, 'raw') and _task.output.raw:
                    with open(output_path(output_dir, _fname), 'w', encoding='utf-8') as _f:
                        _f.write(_task.output.raw)
                    logger.info(f"  ✓ Partial output saved: {_fname}")
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
        logger.info(f"\n{'='*60}")
        logger.info("HIGH-SEVERITY DRIFT DETECTED — RUNNING SCRIPT CORRECTION")
        logger.info(f"{'='*60}")
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
                logger.warning("⚠ Correction output too short — using original polished script")
                _corrected_script_text = None
            elif orig_transitions > 0 and corrected_transitions < orig_transitions:
                logger.warning(f"⚠ Correction lost [TRANSITION] markers ({orig_transitions}→{corrected_transitions}) — using original polished script")
                _corrected_script_text = None
            else:
                with open(output_path(output_dir, "ACCURACY_CORRECTIONS.md"), 'w') as f:
                    f.write("# Script Corrections Applied\n\n")
                    f.write("HIGH-severity drift instances were corrected before audio generation.\n\n")
                    f.write(f"## Original Audit\n{audit_output}\n")
                logger.info("✓ Script correction applied — using corrected script for audio")
                _corrected_script_text = corrected
        except Exception as e:
            logger.warning(f"⚠ Script correction failed: {e} — using original polished script")
            _corrected_script_text = None
    else:
        _corrected_script_text = None
        if audit_output:
            logger.info("✓ Accuracy audit: No HIGH-severity drift — proceeding to audio")

    # --- PDF GENERATION STEP ---
    logger.info("\n--- Generating Documentation PDFs ---")
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
                logger.warning(f"  Skipping {filename}: no output available")
                continue
            create_pdf(title, content, filename)
        except Exception as e:
            logger.warning(f"  Warning: Failed to create {filename}: {e}")

    logger.info("\n--- Saving Research Outputs (Markdown) ---")
    _save_task_outputs(output_dir, [
        ("Research Framing", framing_output, "research_framing.md"),
        # source_of_truth.md already saved from deep research outputs
        ("Accuracy Audit", audit_task, "accuracy_audit.md"),
        ("Episode Blueprint", blueprint_task, "EPISODE_BLUEPRINT.md"),
        ("Script Draft", script_task, "script_draft.md"),
        # script_final.md saved in Phase 8 (from polished/corrected script before TTS)
    ])

    # --- RESEARCH SUMMARY ---
    if deep_reports is not None:
        deep_audit = deep_reports.get("audit")
        if deep_audit:
            logger.info(f"\n--- Deep Research Summary ---")
            _lead = deep_reports.get("lead")
            _counter = deep_reports.get("counter")
            if _lead:
                logger.info(f"  Lead sources: {_lead.total_summaries}")
            if _counter:
                logger.info(f"  Counter sources: {_counter.total_summaries}")
            logger.info(f"  Total sources: {deep_audit.total_summaries}")
            logger.info(f"  Total URLs fetched: {deep_audit.total_urls_fetched}")
            logger.info(f"  Duration: {deep_audit.duration_seconds:.0f}s")

    # --- SESSION METADATA ---
    logger.info("\n--- Documenting Session Metadata ---")
    session_metadata = (
        f"PODCAST SESSION METADATA\n{'='*60}\n\n"
        f"Topic: {topic_name}\n\n"
        f"Language: {language_config['name']} ({language})\n\n"
        f"Role Assignments:\n"
        f"  {SESSION_ROLES['presenter']['label']}: Presenter ({SESSION_ROLES['presenter']['personality']})\n"
        f"  {SESSION_ROLES['questioner']['label']}: Questioner ({SESSION_ROLES['questioner']['personality']})\n"
    )
    metadata_file = output_path(output_dir, "session_metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write(session_metadata)
    logger.info(f"Session metadata: {metadata_file}")

    # --- PHASE 8: AUDIO GENERATION ---
    # Finalize script (corrections, language audit, reaction guidance, save script_final.md)
    script_text = _finalize_script(
        polished_text, polish_task, language, language_config, output_dir,
        corrected_text=_corrected_script_text)

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

    logger.info(f"\n{'='*60}")
    logger.info(f"DURATION CHECK")
    logger.info(f"{'='*60}")
    logger.info(f"Script length: {script_length} {length_unit}")
    logger.info(f"Estimated duration: {estimated_duration_min:.1f} minutes")
    logger.info(f"Target: {_target_min} minutes ({target_length} {length_unit})")

    if script_length < target_low:
        logger.warning(f"\u26a0 WARNING: Script is SHORT ({script_length} {length_unit} < {target_length} target)")
        logger.info(f"  Estimated {estimated_duration_min:.1f} min")
    elif script_length > target_high:
        logger.warning(f"\u26a0 WARNING: Script is LONG ({script_length} {length_unit} > {target_length} target)")
        logger.info(f"  Estimated {estimated_duration_min:.1f} min")
    else:
        logger.info(f"\u2713 Script length GOOD ({script_length} {length_unit})")
        logger.info(f"  Estimated {estimated_duration_min:.1f} min")
    logger.info(f"{'='*60}\n")

    # TTS + BGM
    if not _phase_done(8):
        _run_audio_pipeline(script_text, output_dir, language_config)
        # Save final checkpoint — pipeline complete
        _save_phase(8)
    else:
        logger.info("PHASE 8: AUDIO — already complete, skipping")
        # Check if audio file exists
        _audio_dir = output_dir / "audio"
        if _audio_dir.is_dir():
            _audio_files = list(_audio_dir.glob("podcast_*.wav")) + list(_audio_dir.glob("podcast_*.mp3"))
        else:
            _audio_files = list(output_dir.glob("podcast_*.wav")) + list(output_dir.glob("podcast_*.mp3"))
        if _audio_files:
            logger.info(f"  Audio file: {_audio_files[0].name}")
        else:
            logger.warning("  WARNING: No audio file found — re-running audio generation")
            _run_audio_pipeline(script_text, output_dir, language_config)
            _save_phase(8)