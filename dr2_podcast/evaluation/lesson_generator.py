"""Fix 12: Lesson Generator — LLM-assisted observation extraction.

After scorecard generation, makes one Smart Model call with the scorecard +
regressions as input.  Returns 1-3 short, actionable observations tagged by
phase group.  Appends results to ``lessons_pending.json``.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI

from dr2_podcast.config import SMART_MODEL, SMART_BASE_URL
from dr2_podcast.utils import async_call_smart

logger = logging.getLogger(__name__)

_LESSONS_DIR = Path(__file__).resolve().parent
_PENDING_PATH = _LESSONS_DIR / "lessons_pending.json"

_SYSTEM_PROMPT = """\
You are a pipeline-improvement analyst for a research podcast generator.
You receive a run scorecard (JSON) with deterministic metrics.

Respond with ONLY a JSON array — no markdown, no prose, no explanation.
Each element must be an object with exactly these keys:
  - "phase_group": one of "research", "content", "audio"
  - "observation": a single actionable sentence tied to a specific metric
  - "evidence": the metric name and value that supports the observation

Rules:
- Return 1 to 3 lessons maximum
- Each observation must be actionable (suggest a direction, not just state a fact)
- Tie every observation to a concrete metric from the scorecard
- If all metrics look healthy and there are no regressions, return an empty array []
"""


def _load_pending() -> list[dict]:
    if _PENDING_PATH.exists():
        try:
            data = json.loads(_PENDING_PATH.read_text())
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def _save_pending(lessons: list[dict]) -> None:
    _PENDING_PATH.write_text(json.dumps(lessons, indent=2, ensure_ascii=False))


async def _generate_lessons_async(scorecard: dict) -> list[dict]:
    """Call Smart Model once, return parsed lesson list."""
    client = AsyncOpenAI(base_url=SMART_BASE_URL, api_key="not-needed")

    user_msg = json.dumps(scorecard, indent=2, ensure_ascii=False)

    raw = await async_call_smart(
        client=client,
        model=SMART_MODEL,
        system=_SYSTEM_PROMPT,
        user=user_msg,
        max_tokens=1024,
        temperature=0.3,
    )

    # Parse JSON from response — handle markdown fences
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        lessons = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Lesson generator returned non-JSON: %s", text[:200])
        return []

    if not isinstance(lessons, list):
        logger.warning("Lesson generator returned non-list: %s", type(lessons))
        return []

    # Validate and cap at 3
    valid = []
    for item in lessons[:3]:
        if not isinstance(item, dict):
            continue
        if item.get("phase_group") not in ("research", "content", "audio"):
            continue
        if not item.get("observation"):
            continue
        valid.append(item)

    return valid


def generate_lessons(scorecard: dict, output_dir: str) -> list[dict]:
    """Generate lessons from scorecard, append to pending store.

    Args:
        scorecard: The run scorecard dict from ``generate_scorecard()``.
        output_dir: Run output directory path (used for run_id).

    Returns:
        List of newly generated lesson dicts.
    """
    run_id = scorecard.get("run_id", Path(output_dir).name)
    now = datetime.now().isoformat()

    try:
        raw_lessons = asyncio.run(_generate_lessons_async(scorecard))
    except Exception as e:
        logger.warning("Lesson generation LLM call failed: %s", e)
        return []

    # Stamp each lesson
    new_lessons = []
    for i, lesson in enumerate(raw_lessons, 1):
        entry = {
            "id": f"lesson_{run_id.replace('-', '').replace('_', '')}_{i}",
            "run_id": run_id,
            "timestamp": now,
            "phase_group": lesson["phase_group"],
            "observation": lesson["observation"],
            "evidence": lesson.get("evidence", ""),
            "status": "pending",
        }
        new_lessons.append(entry)

    # Append to pending store
    if new_lessons:
        pending = _load_pending()
        pending.extend(new_lessons)
        _save_pending(pending)
        logger.info("Added %d lesson(s) to pending (%d total)", len(new_lessons), len(pending))

    return new_lessons
