"""Fix 14: Lesson Reviewer — threshold-triggered promotion & cleanup.

When ``lessons_pending.json`` reaches 10 entries, automatically runs:
1. Expire entries older than 90 days
2. Check if observations are already implemented in ``prompt_strings.py``
3. Deduplicate via one Smart Model call (cluster similar, keep best)
4. Promote survivors to ``lessons_promoted.json``
5. Notify via Telegram

Also provides a CLI entry point for reviewing promoted lessons.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from openai import AsyncOpenAI

from dr2_podcast.config import SMART_MODEL, SMART_BASE_URL
from dr2_podcast.utils import async_call_smart

logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).resolve().parent
_PENDING_PATH = _EVAL_DIR / "lessons_pending.json"
_PROMOTED_PATH = _EVAL_DIR / "lessons_promoted.json"
_PROMPT_STRINGS_PATH = _EVAL_DIR.parent / "prompt_strings.py"

_PHASE_TO_PROMPT_SECTION = {
    "research": ["keyword", "search", "screen", "extract", "pico"],
    "content": ["blueprint", "script", "polish", "condense", "section_gen", "audit"],
    "audio": ["tts", "audio", "voice", "bgm"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pending() -> list[dict]:
    if _PENDING_PATH.exists():
        try:
            data = json.loads(_PENDING_PATH.read_text())
            return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def _save_pending(lessons: list[dict]) -> None:
    _PENDING_PATH.write_text(json.dumps(lessons, indent=2, ensure_ascii=False))


def _load_promoted() -> list[dict]:
    if _PROMOTED_PATH.exists():
        try:
            data = json.loads(_PROMOTED_PATH.read_text())
            return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def _save_promoted(lessons: list[dict]) -> None:
    _PROMOTED_PATH.write_text(json.dumps(lessons, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Step 1: Expire old lessons (>90 days)
# ---------------------------------------------------------------------------

def _expire_old(lessons: list[dict], max_age_days: int = 90) -> tuple[list[dict], int]:
    """Remove lessons older than max_age_days.  Returns (remaining, expired_count)."""
    cutoff = datetime.now() - timedelta(days=max_age_days)
    remaining = []
    expired = 0
    for lesson in lessons:
        try:
            ts = datetime.fromisoformat(lesson["timestamp"])
            if ts < cutoff:
                expired += 1
                continue
        except (KeyError, ValueError):
            pass
        remaining.append(lesson)
    return remaining, expired


# ---------------------------------------------------------------------------
# Step 2: Check already-implemented
# ---------------------------------------------------------------------------

def _check_already_implemented(lessons: list[dict]) -> tuple[list[dict], int]:
    """Remove lessons whose observations are already reflected in prompt_strings.py."""
    if not _PROMPT_STRINGS_PATH.exists():
        return lessons, 0

    prompt_text = _PROMPT_STRINGS_PATH.read_text(errors="replace").lower()

    remaining = []
    removed = 0
    for lesson in lessons:
        obs = lesson.get("observation", "").lower()
        phase = lesson.get("phase_group", "")

        # Extract key terms from observation (words > 4 chars)
        key_terms = [w for w in obs.split() if len(w) > 4]
        if not key_terms:
            remaining.append(lesson)
            continue

        # Check if a meaningful fraction of key terms appear in the relevant
        # prompt section area
        matches = sum(1 for term in key_terms if term in prompt_text)
        match_ratio = matches / len(key_terms) if key_terms else 0

        if match_ratio >= 0.6:
            removed += 1
            logger.info("Lesson already implemented: %s", lesson.get("id", "?"))
        else:
            remaining.append(lesson)

    return remaining, removed


# ---------------------------------------------------------------------------
# Step 3: Deduplicate via Smart Model
# ---------------------------------------------------------------------------

_DEDUP_SYSTEM = """\
You are a deduplication assistant.  You receive a JSON array of lesson objects.
Each has "id", "phase_group", "observation", and "evidence".

Cluster lessons that describe the SAME underlying issue (same phase_group and
semantically overlapping observation).  From each cluster, keep the entry with
the most specific / actionable observation.

Respond with ONLY a JSON array of the IDs to KEEP.  No prose, no markdown fences.
Example: ["lesson_1", "lesson_5", "lesson_8"]
"""


async def _deduplicate_async(lessons: list[dict]) -> tuple[list[dict], int]:
    """Use one Smart Model call to cluster and deduplicate."""
    if len(lessons) <= 1:
        return lessons, 0

    client = AsyncOpenAI(base_url=SMART_BASE_URL, api_key="not-needed")

    input_data = [
        {
            "id": l["id"],
            "phase_group": l.get("phase_group", ""),
            "observation": l.get("observation", ""),
            "evidence": l.get("evidence", ""),
        }
        for l in lessons
    ]

    try:
        raw = await async_call_smart(
            client=client,
            model=SMART_MODEL,
            system=_DEDUP_SYSTEM,
            user=json.dumps(input_data, ensure_ascii=False),
            max_tokens=512,
            temperature=0.1,
        )

        text = raw.strip()
        # Handle markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        keep_ids = set(json.loads(text))
    except Exception as e:
        logger.warning("Dedup LLM call failed, keeping all: %s", e)
        return lessons, 0

    remaining = [l for l in lessons if l["id"] in keep_ids]
    removed = len(lessons) - len(remaining)
    return remaining, removed


# ---------------------------------------------------------------------------
# Step 4: Promote
# ---------------------------------------------------------------------------

def _promote(lessons: list[dict]) -> list[dict]:
    """Move surviving lessons to promoted format with suggested edit locations."""
    promoted = []
    for lesson in lessons:
        phase = lesson.get("phase_group", "")
        sections = _PHASE_TO_PROMPT_SECTION.get(phase, [])
        suggested_key = f"{phase}.{sections[0]}" if sections else f"{phase}.general"

        entry = {
            "id": lesson["id"],
            "phase_group": phase,
            "observation": lesson.get("observation", ""),
            "evidence": lesson.get("evidence", ""),
            "seen_count": 1,
            "first_seen": lesson.get("timestamp", ""),
            "last_seen": lesson.get("timestamp", ""),
            "suggested_prompt_key": suggested_key,
            "suggested_edit": lesson.get("observation", ""),
        }
        promoted.append(entry)
    return promoted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_threshold() -> bool:
    """Return True if pending lesson count >= 10."""
    pending = _load_pending()
    return len(pending) >= 10


def run_review() -> dict:
    """Execute the full expire/dedup/promote pipeline.

    Returns a summary dict with counts for each step.
    """
    pending = _load_pending()
    initial_count = len(pending)

    # Step 1: Expire
    pending, expired_count = _expire_old(pending)
    logger.info("Review: expired %d lessons (>90 days)", expired_count)

    # Step 2: Already implemented
    pending, impl_count = _check_already_implemented(pending)
    logger.info("Review: removed %d already-implemented lessons", impl_count)

    # Step 3: Deduplicate
    pending, dedup_count = asyncio.run(_deduplicate_async(pending))
    logger.info("Review: deduplicated %d lessons", dedup_count)

    # Step 4: Promote survivors
    promoted_entries = _promote(pending)
    existing_promoted = _load_promoted()
    existing_promoted.extend(promoted_entries)
    _save_promoted(existing_promoted)
    logger.info("Review: promoted %d lessons", len(promoted_entries))

    # Clear pending
    _save_pending([])

    summary = {
        "initial_count": initial_count,
        "expired": expired_count,
        "already_implemented": impl_count,
        "deduplicated": dedup_count,
        "promoted": len(promoted_entries),
        "remaining_pending": 0,
        "promoted_lessons": promoted_entries,
    }

    # Step 5: Telegram notification
    try:
        from dr2_podcast.evaluation.telegram_report import send_review_report
        send_review_report(summary)
    except Exception as e:
        logger.warning("Review Telegram notification failed: %s", e)

    return summary


# ---------------------------------------------------------------------------
# CLI entry point: python -m dr2_podcast.evaluation.lesson_reviewer review
# ---------------------------------------------------------------------------

def _cli_review():
    """Print promoted lessons for manual review."""
    promoted = _load_promoted()
    if not promoted:
        print("No promoted lessons.")
        return

    print(f"\n{'='*60}")
    print(f"PROMOTED LESSONS ({len(promoted)})")
    print(f"{'='*60}\n")

    for i, entry in enumerate(promoted, 1):
        print(f"  {i}. [{entry.get('phase_group', '?')}] {entry.get('observation', '')}")
        print(f"     Evidence: {entry.get('evidence', '-')}")
        print(f"     Suggested key: {entry.get('suggested_prompt_key', '-')}")
        print(f"     First seen: {entry.get('first_seen', '-')}")
        print()


def _cli_pending():
    """Print pending lessons."""
    pending = _load_pending()
    if not pending:
        print("No pending lessons.")
        return

    print(f"\n{'='*60}")
    print(f"PENDING LESSONS ({len(pending)})")
    print(f"{'='*60}\n")

    for i, entry in enumerate(pending, 1):
        print(f"  {i}. [{entry.get('phase_group', '?')}] {entry.get('observation', '')}")
        print(f"     Run: {entry.get('run_id', '-')} | {entry.get('timestamp', '-')}")
        print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "review"

    if cmd == "review":
        _cli_review()
    elif cmd == "pending":
        _cli_pending()
    elif cmd == "run":
        if check_threshold():
            summary = run_review()
            print(f"Review complete: {summary}")
        else:
            pending = _load_pending()
            print(f"Threshold not reached ({len(pending)}/10 pending)")
    else:
        print(f"Usage: python -m dr2_podcast.evaluation.lesson_reviewer [review|pending|run]")
