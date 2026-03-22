"""Fix 11: Run Scorecard — deterministic metric collection from existing artifacts.

Collects metrics from log files, JSON artifacts, and markdown files after a
pipeline run completes.  No LLM calls — pure file parsing.

Writes ``run_scorecard.json`` into the run's output directory and flags
regressions vs the rolling average of the last 5 scorecards.
"""

import json
import logging
import re
import struct
import wave
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric parsers — each returns a partial dict merged into the scorecard
# ---------------------------------------------------------------------------

def _parse_screening_tiers(research_dir: Path) -> dict:
    """Count studies per tier from screening_results_aff/neg JSON files."""
    tiers = {"tier1": 0, "tier2": 0, "tier3": 0}
    for suffix in ("_aff", "_neg"):
        path = research_dir / f"screening_results{suffix}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            # Format is a dict with "selected_records" list
            if isinstance(data, dict):
                records = data.get("selected_records", [])
            elif isinstance(data, list):
                records = data
            else:
                continue
            for rec in records:
                tier = rec.get("research_tier") or rec.get("tier") or rec.get("search_tier", 0)
                tier_int = int(tier) if tier else 0
                if tier_int == 1:
                    tiers["tier1"] += 1
                elif tier_int == 2:
                    tiers["tier2"] += 1
                elif tier_int >= 3:
                    tiers["tier3"] += 1
        except Exception:
            continue
    return tiers


def _parse_log_metrics(log_path: Path) -> dict:
    """Extract metrics from podcast_generation.log via regex."""
    metrics: dict = {}
    if not log_path.exists():
        return metrics

    text = log_path.read_text(errors="replace")

    # Extraction timeout rate
    timeouts = len(re.findall(r"timed out", text, re.IGNORECASE))
    metrics["extraction_timeouts"] = timeouts

    # Articles extracted / attempted  ("Extracted data from X/Y articles")
    m = re.findall(r"Extracted data from (\d+)/(\d+) articles", text)
    if m:
        extracted = sum(int(x) for x, _ in m)
        attempted = sum(int(y) for _, y in m)
        metrics["articles_extracted"] = extracted
        metrics["articles_attempted"] = attempted
        if attempted > 0:
            metrics["extraction_timeout_rate"] = round(
                1 - extracted / attempted, 3
            )
    else:
        metrics["articles_extracted"] = 0
        metrics["articles_attempted"] = 0

    # Section budget adherence ("Section <name>: XXXX/YYYY chars")
    section_hits = re.findall(
        r"Section (\w+):\s+(\d+)/(\d+) chars", text
    )
    if section_hits:
        adherences = []
        for _name, actual, budget in section_hits:
            a, b = int(actual), int(budget)
            adherences.append(round(a / b, 3) if b > 0 else 0.0)
        metrics["section_adherence"] = adherences

    # Max deficit cascade ratio ("budget adjusted .* -> .* chars")
    deficit_ratios = re.findall(
        r"budget adjusted (\d+)\s*->\s*(\d+) chars", text
    )
    if deficit_ratios:
        ratios = [int(new) / int(orig) for orig, new in deficit_ratios if int(orig) > 0]
        metrics["max_deficit_ratio"] = round(max(ratios), 3) if ratios else 1.0

    # Assembled draft length ("Assembled draft: XXXX chars")
    m_draft = re.search(r"Assembled draft:\s*(\d+)\s*chars", text)
    if m_draft:
        metrics["draft_char_count"] = int(m_draft.group(1))

    # Script target length ("Sectional draft: N sections, total budget XXXX chars")
    m_budget = re.search(r"total budget\s+(\d+)\s+(?:chars|words)", text)
    if m_budget:
        metrics["script_target_length"] = int(m_budget.group(1))

    # Degenerate repetition %
    m_degen = re.search(r"[Dd]egenerate.*?(\d+(?:\.\d+)?)\s*%", text)
    if m_degen:
        metrics["degenerate_repetition_pct"] = float(m_degen.group(1))
    else:
        metrics["degenerate_repetition_pct"] = 0.0

    # Content audit issues ("CLEAN" or issue text)
    if re.search(r"content audit.*CLEAN", text, re.IGNORECASE):
        metrics["content_audit_issues"] = 0
    else:
        issues = re.findall(r"content audit.*issue", text, re.IGNORECASE)
        metrics["content_audit_issues"] = len(issues)

    # Pipeline start/end timestamps for total duration
    timestamps = re.findall(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text, re.MULTILINE)
    if len(timestamps) >= 2:
        fmt = "%Y-%m-%d %H:%M:%S"
        try:
            t0 = datetime.strptime(timestamps[0], fmt)
            t1 = datetime.strptime(timestamps[-1], fmt)
            metrics["total_duration_min"] = round((t1 - t0).total_seconds() / 60, 1)
        except ValueError:
            pass

    # Audio duration ("Audio duration X minutes" or WAV-based)
    m_audio = re.search(r"[Aa]udio duration[:\s]+(\d+(?:\.\d+)?)\s*min", text)
    if m_audio:
        metrics["audio_duration_min"] = float(m_audio.group(1))

    # Podcast Length Mode ("Podcast Length Mode: Long (30 min)")
    m_target = re.search(r"Podcast Length Mode:\s*\w+\s*\((\d+)\s*min\)", text)
    if m_target:
        metrics["target_duration_min"] = int(m_target.group(1))

    # Language — capture the 2-letter code: "Language: 日本語 (Japanese) (ja)" or "Language: English (en)"
    m_lang = re.search(r"Language:.*\(([a-z]{2})\)", text)
    if m_lang:
        metrics["language"] = m_lang.group(1)
    else:
        # Fallback: any parenthetical word
        m_lang2 = re.search(r"Language:.*\((\w+)\)", text)
        if m_lang2:
            lang_raw = m_lang2.group(1).lower()
            metrics["language"] = {"japanese": "ja", "english": "en"}.get(lang_raw, lang_raw)

    # Topic from log
    m_topic = re.search(r"^Topic:\s*(.+)$", text, re.MULTILINE)
    if m_topic:
        metrics["topic"] = m_topic.group(1).strip()

    return metrics


def _parse_url_validation(research_dir: Path) -> dict:
    """Parse URL validation pass rate from url_validation_results.json."""
    path = research_dir / "url_validation_results.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            # Format: {url: status_string, ...}
            # "✓ Valid Link ..." = valid, "✗ ..." = broken, "⚠ ..." = other
            if all(isinstance(v, str) for v in data.values()):
                total = len(data)
                valid = sum(1 for v in data.values() if "\u2713" in v or "valid" in v.lower())
                return {
                    "url_validation_total": total,
                    "url_validation_valid": valid,
                    "url_validation_pass_rate": round(valid / total, 3) if total > 0 else 0.0,
                }
            # Fallback: nested dict with "results" key
            results = data.get("results", data)
            if isinstance(results, dict):
                total = sum(v for v in results.values() if isinstance(v, (int, float)))
                valid = results.get("valid", 0)
                return {
                    "url_validation_total": total,
                    "url_validation_valid": valid,
                    "url_validation_pass_rate": round(valid / total, 3) if total > 0 else 0.0,
                }
        elif isinstance(data, list):
            total = len(data)
            valid = sum(
                1 for r in data
                if isinstance(r, dict) and r.get("status") in ("valid", "ok", True)
            )
            return {
                "url_validation_total": total,
                "url_validation_valid": valid,
                "url_validation_pass_rate": round(valid / total, 3) if total > 0 else 0.0,
            }
        return {}
    except Exception:
        return {}


def _parse_accuracy_audit(research_dir: Path) -> dict:
    """Count accuracy audit findings from ACCURACY_AUDIT.md or accuracy_audit.md."""
    for name in ("ACCURACY_AUDIT.md", "accuracy_audit.md"):
        path = research_dir / name
        if path.exists():
            text = path.read_text(errors="replace")
            findings = len(re.findall(r"^###\s+", text, re.MULTILINE))
            return {"accuracy_audit_findings": findings}
    return {"accuracy_audit_findings": 0}


def _parse_script_length(scripts_dir: Path, research_dir: Path) -> dict:
    """Get script word/char count from script_final.md."""
    for parent in (scripts_dir, research_dir):
        path = parent / "script_final.md"
        if path.exists():
            text = path.read_text(errors="replace")
            return {
                "script_char_count": len(text),
                "script_word_count": len(text.split()),
            }
    return {}


def _get_audio_duration_from_wav(output_dir: Path) -> float | None:
    """Read WAV duration directly from file header."""
    for candidate in (
        output_dir / "audio" / "audio.wav",
        output_dir / "audio" / "audio_mixed.wav",
    ):
        if candidate.exists():
            try:
                with wave.open(str(candidate), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    if rate > 0:
                        return round(frames / rate / 60, 2)
            except Exception:
                continue
    return None


def _load_recent_scorecards(output_base: Path, exclude_run: str, limit: int = 5) -> list[dict]:
    """Load the last N scorecards from sibling output directories."""
    scorecards = []
    if not output_base.exists():
        return scorecards
    for d in sorted(output_base.iterdir(), reverse=True):
        if not d.is_dir() or d.name == exclude_run:
            continue
        sc_path = d / "run_scorecard.json"
        if sc_path.exists():
            try:
                scorecards.append(json.loads(sc_path.read_text()))
            except Exception:
                continue
        if len(scorecards) >= limit:
            break
    return scorecards


def _detect_regressions(current: dict, history: list[dict]) -> list[str]:
    """Flag metrics that worsened by >20% vs rolling average."""
    if not history:
        return []

    # Flatten current metrics
    cm = current.get("metrics", {})
    regressions = []

    # Define checks: (display_name, metric_path, higher_is_better)
    checks = [
        ("extraction_timeout_rate", ("research", "extraction_timeout_rate"), False),
        ("url_validation_pass_rate", ("research", "url_validation_pass_rate"), True),
        ("script_adherence_pct", ("script", "adherence_pct"), True),
        ("audio_adherence_pct", ("audio", "adherence_pct"), True),
        ("accuracy_audit_findings", ("script", "accuracy_audit_findings"), False),
        ("degenerate_repetition_pct", ("script", "degenerate_repetition_pct"), False),
    ]

    for name, path, higher_better in checks:
        # Get current value
        val = cm
        for key in path:
            if isinstance(val, dict):
                val = val.get(key)
            else:
                val = None
                break
        if val is None:
            continue

        # Get historical values
        hist_vals = []
        for sc in history:
            hv = sc.get("metrics", {})
            for key in path:
                if isinstance(hv, dict):
                    hv = hv.get(key)
                else:
                    hv = None
                    break
            if hv is not None:
                hist_vals.append(hv)

        if not hist_vals:
            continue

        avg = sum(hist_vals) / len(hist_vals)
        if avg == 0:
            continue

        if higher_better:
            # Regression if current is >20% below average
            if val < avg * 0.8:
                regressions.append(
                    f"{name} dropped to {val:.3f} (avg {avg:.3f}, -{(1 - val/avg)*100:.0f}%)"
                )
        else:
            # Regression if current is >20% above average
            if val > avg * 1.2:
                regressions.append(
                    f"{name} increased to {val:.3f} (avg {avg:.3f}, +{(val/avg - 1)*100:.0f}%)"
                )

    return regressions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_scorecard(output_dir: str) -> dict:
    """Generate a deterministic run scorecard from pipeline artifacts.

    Args:
        output_dir: Path to the timestamped run output directory.

    Returns:
        The scorecard dict (also written to ``output_dir/run_scorecard.json``).
    """
    od = Path(output_dir)
    run_id = od.name

    # Locate subdirectories (handle both flat and nested layouts)
    research_dir = od / "research" if (od / "research").is_dir() else od
    scripts_dir = od / "scripts" if (od / "scripts").is_dir() else od
    meta_dir = od / "meta" if (od / "meta").is_dir() else od
    log_path = meta_dir / "podcast_generation.log"

    # Collect metrics
    tiers = _parse_screening_tiers(research_dir)
    log_m = _parse_log_metrics(log_path)
    url_m = _parse_url_validation(research_dir)
    audit_m = _parse_accuracy_audit(research_dir)
    script_m = _parse_script_length(scripts_dir, research_dir)

    # Audio duration: prefer log, fallback to WAV header
    audio_dur = log_m.pop("audio_duration_min", None)
    if audio_dur is None:
        audio_dur = _get_audio_duration_from_wav(od)
    target_dur = log_m.pop("target_duration_min", None)

    topic = log_m.pop("topic", "")
    language = log_m.pop("language", "en")

    # Script adherence
    draft_chars = log_m.get("draft_char_count", script_m.get("script_char_count", 0))
    # Prefer log-derived target ("Sectional draft: N sections, total budget XXXX chars")
    target_length = log_m.pop("script_target_length", None)
    if target_length is None:
        target_length = 15000 if language == "ja" else 4000  # char vs word default

    scorecard: dict = {
        "run_id": run_id,
        "topic": topic,
        "language": language,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "research": {
                "studies_found": tiers,
                "extraction_timeout_rate": log_m.get("extraction_timeout_rate", 0.0),
                "extraction_timeouts": log_m.get("extraction_timeouts", 0),
                "articles_extracted": log_m.get("articles_extracted", 0),
                "articles_attempted": log_m.get("articles_attempted", 0),
                "url_validation_pass_rate": url_m.get("url_validation_pass_rate", 0.0),
            },
            "script": {
                "target_length": target_length,
                "actual_length": draft_chars,
                "adherence_pct": round(draft_chars / target_length, 3) if target_length > 0 else 0.0,
                "section_adherence": log_m.get("section_adherence", []),
                "max_deficit_ratio": log_m.get("max_deficit_ratio", 1.0),
                "degenerate_repetition_pct": log_m.get("degenerate_repetition_pct", 0.0),
                "content_audit_issues": log_m.get("content_audit_issues", 0),
                "accuracy_audit_findings": audit_m.get("accuracy_audit_findings", 0),
            },
            "audio": {
                "target_duration_min": target_dur,
                "actual_duration_min": audio_dur,
                "adherence_pct": (
                    round(audio_dur / target_dur, 3)
                    if audio_dur and target_dur and target_dur > 0
                    else None
                ),
            },
            "pipeline": {
                "total_duration_min": log_m.get("total_duration_min"),
            },
        },
        "regressions": [],
    }

    # Run-over-run comparison
    output_base = od.parent
    history = _load_recent_scorecards(output_base, run_id)
    regressions = _detect_regressions(scorecard, history)
    scorecard["regressions"] = regressions

    # Write scorecard
    sc_path = od / "run_scorecard.json"
    sc_path.write_text(json.dumps(scorecard, indent=2, ensure_ascii=False))
    logger.info("Scorecard written to %s", sc_path)

    if regressions:
        for r in regressions:
            logger.warning("Regression: %s", r)

    return scorecard
