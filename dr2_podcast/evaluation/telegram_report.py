"""Fix 13: Telegram Report — post-run notification via stdlib urllib.

Sends a formatted scorecard + lesson summary to Telegram after each run.
Reads bot token and chat ID from ``~/.claude/telegram-secrets.json``.
Zero new dependencies.  Never crashes the pipeline.
"""

import json
import logging
import urllib.request
import urllib.error
from pathlib import Path

logger = logging.getLogger(__name__)

_SECRETS_PATH = Path.home() / ".claude" / "telegram-secrets.json"


def _load_credentials() -> tuple[str, str] | None:
    """Return (bot_token, chat_id) or None if unavailable."""
    if not _SECRETS_PATH.exists():
        logger.warning("Telegram secrets not found at %s", _SECRETS_PATH)
        return None
    try:
        data = json.loads(_SECRETS_PATH.read_text())
        token = data.get("BOT_TOKEN", "")
        chat_id = data.get("CHAT_ID", "")
        if not token or not chat_id:
            logger.warning("Telegram secrets missing BOT_TOKEN or CHAT_ID")
            return None
        return token, chat_id
    except Exception as e:
        logger.warning("Failed to read Telegram secrets: %s", e)
        return None


def _send_message(token: str, chat_id: str, text: str) -> bool:
    """Send a message via Telegram Bot API.  Returns True on success."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError) as e:
        logger.warning("Telegram send failed: %s", e)
        return False


def _status_icon(value: float | None, target: float | None, higher_better: bool = True) -> str:
    """Return check or warning icon based on 90% threshold."""
    if value is None or target is None or target == 0:
        return "\u2753"  # ❓
    ratio = value / target if higher_better else target / value
    return "\u2705" if ratio >= 0.9 else "\u26a0\ufe0f"


def _format_run_report(scorecard: dict, lessons: list[dict]) -> str:
    """Format the run report message."""
    m = scorecard.get("metrics", {})
    r = m.get("research", {})
    s = m.get("script", {})
    a = m.get("audio", {})
    p = m.get("pipeline", {})

    topic = scorecard.get("topic", "Unknown")
    lang = scorecard.get("language", "?")
    pipeline_min = p.get("total_duration_min") or "?"
    audio_min = a.get("actual_duration_min") or "?"
    audio_target = a.get("target_duration_min")
    audio_pct = a.get("adherence_pct")
    audio_pct_str = f"{audio_pct*100:.0f}%" if audio_pct else "?"

    # Studies count
    tiers = r.get("studies_found", {})
    total_studies = sum(tiers.values()) if isinstance(tiers, dict) else 0
    timeout_rate = r.get("extraction_timeout_rate", 0)
    timeout_pct = f"{timeout_rate*100:.0f}%"

    # Script
    script_adh = s.get("adherence_pct", 0)
    script_pct = f"{script_adh*100:.0f}%"
    audit_issues = s.get("content_audit_issues", 0)

    # Status icons
    research_icon = _status_icon(1 - timeout_rate, 0.9, higher_better=True)
    script_icon = _status_icon(script_adh, 0.9)
    audio_icon = _status_icon(
        a.get("actual_duration_min"), a.get("target_duration_min")
    )

    lines = [
        f"\U0001f4cb <b>Podcast Complete:</b> \"{topic}\"",
        f"Language: {lang} | Pipeline: {pipeline_min}min",
        f"Audio: {audio_min}min (target {audio_target or '?'}, {audio_pct_str})",
        "",
        "<b>Scorecard:</b>",
        f"{research_icon} Research: {total_studies} studies, {timeout_pct} timeout",
        f"{script_icon} Script: {script_pct} of target, {audit_issues} audit issues",
        f"{audio_icon} Audio: {audio_min}min (target {audio_target or '?'})",
    ]

    # Regressions
    regressions = scorecard.get("regressions", [])
    if regressions:
        lines.append("")
        lines.append("\u26a0\ufe0f <b>Regressions:</b>")
        for reg in regressions[:5]:
            lines.append(f"\u2022 {reg}")

    # New lessons
    if lessons:
        lines.append("")
        lines.append(f"<b>New Lessons ({len(lessons)}):</b>")
        for lesson in lessons[:3]:
            phase = lesson.get("phase_group", "?")
            obs = lesson.get("observation", "")
            lines.append(f"\u2022 [{phase}] {obs}")

    # Pending count
    pending_path = Path(__file__).resolve().parent / "lessons_pending.json"
    pending_count = 0
    if pending_path.exists():
        try:
            pending_count = len(json.loads(pending_path.read_text()))
        except Exception:
            pass
    lines.append(f"\nPending lessons: {pending_count}/10")

    return "\n".join(lines)


def _format_review_report(summary: dict) -> str:
    """Format the lesson review report message."""
    expired = summary.get("expired", 0)
    already_impl = summary.get("already_implemented", 0)
    deduped = summary.get("deduplicated", 0)
    promoted = summary.get("promoted", 0)
    remaining = summary.get("remaining_pending", 0)

    lines = [
        "\U0001f50d <b>Lesson Review Complete</b>",
        "",
        f"\u2022 Expired (>90d): {expired}",
        f"\u2022 Already implemented: {already_impl}",
        f"\u2022 Deduplicated: {deduped}",
        f"\u2022 <b>Promoted: {promoted}</b>",
        f"\u2022 Remaining pending: {remaining}",
    ]

    promoted_lessons = summary.get("promoted_lessons", [])
    if promoted_lessons:
        lines.append("")
        lines.append("<b>Promoted:</b>")
        for pl in promoted_lessons[:5]:
            obs = pl.get("observation", "")
            lines.append(f"\u2022 {obs}")

    return "\n".join(lines)


def send_run_report(scorecard: dict, lessons: list[dict]) -> bool:
    """Send a run completion report to Telegram.

    Returns True if sent successfully, False otherwise.
    Never raises — all errors are logged and swallowed.
    """
    try:
        creds = _load_credentials()
        if not creds:
            return False

        token, chat_id = creds
        message = _format_run_report(scorecard, lessons)

        # Telegram limit is 4096 chars
        if len(message) > 4000:
            message = message[:3990] + "\n..."

        return _send_message(token, chat_id, message)
    except Exception as e:
        logger.warning("Telegram run report failed: %s", e)
        return False


def send_review_report(summary: dict) -> bool:
    """Send a lesson review summary to Telegram.

    Returns True if sent successfully, False otherwise.
    Never raises.
    """
    try:
        creds = _load_credentials()
        if not creds:
            return False

        token, chat_id = creds
        message = _format_review_report(summary)

        if len(message) > 4000:
            message = message[:3990] + "\n..."

        return _send_message(token, chat_id, message)
    except Exception as e:
        logger.warning("Telegram review report failed: %s", e)
        return False
