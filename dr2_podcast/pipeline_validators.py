"""Deterministic, LLM-free post-generation validators for podcast scripts.

Root-cause context (2026-07-05 investigation of the sleep-week fabrications):
the pipeline's LLM Scientific Auditor DOES detect science drift — it graded the
sleep episodes FAIL and even wrote exact corrections — but (a) its verdict was
never enforced (`_finalize_script(corrected_text=None)` always), and (b) it does
not check label formatting or citation existence deterministically at all. These
gates are the reliable backstop: they never miss the failure classes they cover,
and they catch the formatting defects the science audit ignores.

Covered failure classes:
  1. Speaker-label corruption  -> normalize_speaker_labels / check_speaker_alternation
  2. Fabricated / misattributed citations -> validate_citations
  3. Duplicate endings / repeated lines -> detect_duplicate_blocks

All functions are pure (no I/O except the optional source-file reads in
validate_citations) and safe to call anywhere in the pipeline.
"""

from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------- #
# 1. Speaker labels
# --------------------------------------------------------------------------- #
# Matches any speaker-label variant at line start: Host 1:, Host 1：, **Host 1**:,
# host_1：, ホスト1：, ホスト 1:, Speaker 1:  (case-insensitive, underscore/space tolerant)
_SPK_ANY = re.compile(
    r"^[ \t]*\*{0,2}\s*(?:host|ホスト|speaker)[ _]*([12])[ _]*\*{0,2}\s*[:：]",
    re.IGNORECASE,
)
_SPK_CANON = re.compile(r"^Host ([12]):")


def normalize_speaker_labels(text: str) -> tuple[str, int]:
    """Canonicalize every speaker-label variant to 'Host N: '.

    Fixes the pipeline defects observed in the sleep week: lowercase-underscore
    ``host_1：`` (which ``clean_script_for_tts`` matched case-sensitively and so
    left un-normalized -> leaked into audio) and full-width ``Host 1：``.

    Returns (normalized_text, n_lines_changed).
    """
    out, fixed = [], 0
    for line in text.split("\n"):
        m = _SPK_ANY.match(line)
        if m:
            rest = line[m.end() :].lstrip()
            newline = f"Host {m.group(1)}: {rest}".rstrip() if rest else f"Host {m.group(1)}:"
            if newline != line:
                fixed += 1
            out.append(newline)
        else:
            out.append(line)
    return "\n".join(out), fixed


def check_speaker_alternation(text: str, max_run: int = 6, single_voice_ratio: float = 0.85) -> list[str]:
    """Flag structural speaker defects. Run AFTER normalize_speaker_labels.

    - SPEAKER_RUN: more than ``max_run`` consecutive same-speaker turns (a
      mislabeled block, e.g. the Saturday sleep episode's 40 Host-1 turns).
    - SINGLE_VOICE: one speaker owns more than ``single_voice_ratio`` of all turns.
    """
    turns = [m.group(1) for line in text.split("\n") if (m := _SPK_CANON.match(line))]
    if not turns:
        return ["NO_SPEAKER_LABELS: no canonical 'Host N:' turns found"]
    issues = []
    run = maxrun = 1
    for a, b in zip(turns, turns[1:], strict=False):
        run = run + 1 if a == b else 1
        maxrun = max(maxrun, run)
    if maxrun > max_run:
        issues.append(
            f"SPEAKER_RUN: {maxrun} consecutive same-speaker turns (max {max_run}) — likely a mislabeled block"
        )
    dom = max(Counter(turns).values()) / len(turns)
    if dom > single_voice_ratio:
        issues.append(
            f"SINGLE_VOICE: {int(dom * 100)}% of {len(turns)} turns are one "
            f"speaker (max {int(single_voice_ratio * 100)}%)"
        )
    return issues


# --------------------------------------------------------------------------- #
# 2. Citations
# --------------------------------------------------------------------------- #
_YEAR = r"(?:18|19|20)\d\d"
# Author-year in a script: "Vandewalle et al. (2007)", "Zhong et al., 2022",
# "Leproult & Van Cauter 2011", "Winer (2019)". Surname is Title-case (first
# upper, then lowercase) so ALLCAPS acronyms / trial names (RCT, PREDIMED, VITAL)
# do NOT match.
_CITE = re.compile(
    r"\b([A-Z][a-zÀ-ſ]{2,}(?:\s+[A-Z][a-zÀ-ſ]+)?)"  # surname (opt. "Van Cauter")
    r"(?:\s*(?:et\s*al\.?|&\s*[A-Z][a-zÀ-ſ]+|and\s+[A-Z][a-zÀ-ſ]+))?"
    r"\s*[,\s]?\s*[\(（]?(" + _YEAR + r")[\)）]?"
)

# Title-case English words that legitimately precede a year but are NOT authors
# (journals, orgs, trial names, generic nouns). Kept lowercase for comparison.
_NONAUTHOR = {
    "nature",
    "science",
    "cell",
    "lancet",
    "cochrane",
    "harvard",
    "oxford",
    "day",
    "week",
    "act",
    "study",
    "table",
    "figure",
    "since",
    "before",
    "after",
    "around",
    "about",
    "the",
    "in",
    "by",
    "december",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
}


def _load_source_whitelist(
    sot_path: str | None, sources_json_path: str | None, sot_text: str | None = None
) -> tuple[set, set, set]:
    """Build (author_surnames, (surname, year) pairs, PMIDs) from the retrieved
    sources. Authoritative source = the SOT reference list, whose format is:
    ``N. Surname et al.. *Title*. Journal. (YEAR). PMID: [12345](url).``

    The SOT may be supplied as a file path (``sot_path``) or as an already-loaded
    string (``sot_text``) — the latter avoids a re-read inside the pipeline flow.
    """
    surnames: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    pmids: set[str] = set()

    _sot = sot_text
    if _sot is None and sot_path and Path(sot_path).exists():
        _sot = Path(sot_path).read_text(encoding="utf-8")
    if _sot:
        for line in _sot.split("\n"):
            ref = re.match(r"\s*\d+\.\s+([A-Z][^*.]+?)\.{1,2}\s*\*", line)
            if not ref:
                continue
            year = re.search(r"[\(（](\d{4})[\)）]", line)
            pm = re.search(r"PMID:\s*\[?(\d+)", line)
            first = re.match(r"([A-Z][a-zA-ZÀ-ſ\-]{2,})", ref.group(1).strip())
            if first:
                sn = first.group(1).lower()
                surnames.add(sn)
                if year:
                    pairs.add((sn, year.group(1)))
            if pm:
                pmids.add(pm.group(1))

    if sources_json_path and Path(sources_json_path).exists():
        try:
            data = json.loads(Path(sources_json_path).read_text(encoding="utf-8"))
        except Exception:
            data = None
        entries = []
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    entries += v
        elif isinstance(data, list):
            entries = data
        for e in entries:
            if not isinstance(e, dict):
                continue
            pm = re.search(r"/(\d{6,9})/?", str(e.get("url", "")))
            if pm:
                pmids.add(pm.group(1))
    return surnames, pairs, pmids


def validate_citations(
    script_text: str, sot_path: str | None = None, sources_json_path: str | None = None, sot_text: str | None = None
) -> list[str]:
    """Flag Author-year citations in the script that are not backed by any
    retrieved source. Catches PURE fabrications (surname absent from the source
    corpus, e.g. 'Vandewalle 2007', 'Pillai 2014') deterministically. Author
    present-but-misused (e.g. real 'Zhong 2022' cited for the wrong finding) is
    left to the LLM fidelity audit, since the surname IS in the corpus.

    Returns [] when no whitelist can be built (never false-flags blindly).

    Only PURE fabrication (surname absent from the corpus) is flagged. Year /
    misattribution mismatches are intentionally NOT flagged here — an author with
    multiple papers or a co-authored citation (e.g. "Leproult & Van Cauter 2011"
    where Van Cauter also appears for another year) would false-positive. Those
    belong to the LLM fidelity audit, which reads the study content.
    """
    surnames, _pairs, _pmids = _load_source_whitelist(sot_path, sources_json_path, sot_text)
    if not surnames:
        return []
    issues, seen = [], set()
    for m in _CITE.finditer(script_text):
        sn = re.match(r"[A-Za-zÀ-ſ]+", m.group(1)).group(0).lower()
        yr = m.group(2)
        if sn in _NONAUTHOR or sn in seen:
            continue
        seen.add(sn)
        if sn not in surnames:
            issues.append(
                f"FABRICATED_CITATION: '{m.group(1).strip()} ({yr})' — surname not found in any retrieved source"
            )
    return issues


# --------------------------------------------------------------------------- #
# 3. Duplicate blocks / repeated endings
# --------------------------------------------------------------------------- #
def deduplicate_lines(text: str, min_len: int = 25) -> tuple[str, int]:
    """Remove repeated dialogue lines, keeping the first occurrence. Catches the
    un-deduplicated repeated closings (the Monday sleep episode 'ended' 3-4 times
    with the same sign-off). Only substantial Host lines (>= min_len chars) are
    de-duplicated; short back-channels ('はい。') and ##/[..] lines are untouched.
    Returns (deduped_text, n_removed).
    """
    seen: set[str] = set()
    out, removed = [], 0
    for line in text.split("\n"):
        m = _SPK_CANON.match(line.strip())
        if m:
            body = _SPK_CANON.sub("", line.strip()).strip()
            if len(body) >= min_len:
                if body in seen:
                    removed += 1
                    continue
                seen.add(body)
        out.append(line)
    return "\n".join(out), removed


def detect_duplicate_blocks(text: str, min_len: int = 25) -> list[str]:
    """Flag identical dialogue lines that appear more than once — catches the
    un-deduplicated repeated closings (the Monday sleep episode 'ended' 3-4
    times with an identical sign-off line). Ignores short/structural lines.
    """
    counts: Counter[str] = Counter()
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith(("#", "[", "---")) or not _SPK_CANON.match(s):
            continue
        body = _SPK_CANON.sub("", s).strip()
        if len(body) >= min_len:
            counts[body] += 1
    return [f'DUPLICATE_LINE (x{n}): "{body[:50]}…"' for body, n in counts.items() if n > 1]


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def validate_script_structure(
    script_text: str, sot_path: str | None = None, sources_json_path: str | None = None, sot_text: str | None = None
) -> dict:
    """Run all deterministic gates. Returns
    {'pass': bool, 'issues': [str], 'normalized_text': str, 'labels_fixed': int}.
    ``normalized_text`` has speaker labels canonicalized and should be used
    downstream. Citation checks run only when a source (path or text) is supplied.
    """
    normalized, fixed = normalize_speaker_labels(script_text)
    issues = check_speaker_alternation(normalized)
    issues += detect_duplicate_blocks(normalized)
    if sot_path or sources_json_path or sot_text:
        issues += validate_citations(normalized, sot_path, sources_json_path, sot_text)
    return {
        "pass": len(issues) == 0,
        "issues": issues,
        "normalized_text": normalized,
        "labels_fixed": fixed,
    }


# --------------------------------------------------------------------------- #
# 4. TTS context-dependent reading hazards (Layer 3 backstop) — WARN-only
# --------------------------------------------------------------------------- #
# Layer 1 (engine.apply_tts_glossary) deterministically fixes CONTEXT-FREE
# misreadings. This flags CONTEXT-DEPENDENT ones the editor prompt (Layer 2)
# may miss — the correct reading depends on MEANING, so it can't be blindly
# substituted (表=おもて/ひょう, 辛い=からい/つらい, の方=ほう/かた, 大あり
# segmentation). Non-blocking: surfaces to the editor / human pre-render scan.
# See PLAN.md "TTS glossary + style-rules pipeline enforcement".
_TTS_READING_HAZARDS = [
    (
        re.compile(r"表[がにを](?:出|裏)|表と裏|表向き|コイン[^。\n]{0,8}表"),
        "TABLE_FRONT: 表 may misread ひょう vs おもて (heads/surface context) — verify",
    ),
    (re.compile(r"大あり"), "OOARI: 大あり segmentation unreliable in TTS — consider rephrasing"),
    (
        re.compile(r"(?:料理|味|スパイス|香辛料|カレー|激|唐辛子)[^。\n]{0,4}辛い|辛い(?:料理|もの|味|食)"),
        "SPICY_KARAI: 辛い may misread つらい vs からい (spicy context) — verify",
    ),
    (
        re.compile(
            r"(?:こちら|そちら|あちら|どちら|上|下|右|左|前|後|奥|外|内|東|西|南|北)の方"
            r"|[一-鿿ぁ-ん]の方(?=[はをがにでもへとの])(?![法向面針])"
        ),
        "NO_HOU_KATA: 〜の方 may misread ほう vs かた — verify reading",
    ),
]


def validate_tts_readings(text: str, max_report: int = 20) -> list[str]:
    """Flag CONTEXT-DEPENDENT TTS reading hazards for review (warn, non-blocking).

    Complements engine.apply_tts_glossary (Layer 1, deterministic context-free
    fixes): these readings depend on meaning and cannot be auto-substituted, so
    they surface as warnings for the editor and human pre-render scan.
    """
    issues: list[str] = []
    for lineno, line in enumerate(text.split("\n"), 1):
        for pat, msg in _TTS_READING_HAZARDS:
            m = pat.search(line)
            if m:
                issues.append(f"{msg} [L{lineno}: …{m.group(0)}…]")
                if len(issues) >= max_report:
                    issues.append(f"TTS_READINGS_TRUNCATED: >{max_report} hazards; showing first {max_report}")
                    return issues
    return issues


# ---------------------------------------------------------------- GRADE / NNT
# The final GRADE level and the ARR/NNT figures are produced by the Auditor and
# by deterministic Python math (research/clinical_math.py) — NOT written by the
# script LLM. So when the script disagrees with them, the script is wrong.
# Observed 2026-05-05 sleep episode: the LLM auditor graded the script FAIL yet
# missed every GRADE/NNT inversion (script claimed "中程度〜高い" against a basis
# of LOW, and projected "年間数万単位" of prevented cases against NNT = inf).
# Detection is a string comparison; it does not need an LLM.

_GRADE_FINAL = re.compile(
    r"(?:FINAL\s+GRADE|最終\s*GRADE|Overall\s+GRADE)\s*[:：]?\s*\**\s*"
    r"(VERY\s+LOW|LOW|MODERATE|HIGH|非常に低い|低い|中程度|高い)",
    re.I,
)

# Level tokens as they appear in a Japanese script, mapped to canonical grades.
_GRADE_TOKENS = [
    (re.compile(r"非常に低い|VERY\s+LOW", re.I), "VERY LOW"),
    (re.compile(r"中程度|中等度|MODERATE", re.I), "MODERATE"),
    (re.compile(r"高い|HIGH", re.I), "HIGH"),
    (re.compile(r"低い|LOW", re.I), "LOW"),
]
_GRADE_MENTION = re.compile(r"GRADE|グレード")

# "no measurable difference" signatures from the deterministic math.
_NNT_NULL = re.compile(r"NNT[^\n]{0,40}\binf\b|no_effect|ARR\s*=\s*\+?0\.0000")
# Script turning that null into a positive population-level benefit.
_NNT_PROJECTION = re.compile(
    r"(?:NNT|治療必要数)[^\n]{0,80}?(?:防げ|予防でき|減らせ)"
    r"|(?:年間|試算)[^\n]{0,30}?[0-9０-９万千百]+\s*(?:人|単位)[^\n]{0,20}?(?:防げ|予防でき|減らせ)"
)


def _canonical_grade(text: str) -> str | None:
    """Extract the basis's final GRADE level, or None if not stated."""
    m = _GRADE_FINAL.search(text or "")
    if not m:
        return None
    raw = m.group(1).upper().strip()
    return {"非常に低い": "VERY LOW", "低い": "LOW", "中程度": "MODERATE", "高い": "HIGH"}.get(m.group(1), raw)


def validate_grade_consistency(
    script_text: str, grade_path: str | None = None, sot_path: str | None = None, basis_text: str | None = None
) -> list[str]:
    """Flag script claims that contradict the basis's GRADE level or NNT result.

    Deterministic: compares what the script says about certainty against values
    the pipeline itself computed. Fail-safe — returns [] when the basis states
    no final GRADE (never guesses a grade it cannot read).

    Warn-level, like validate_tts_readings: a hit means a human/LLM must look,
    not that the run halts.
    """
    basis = basis_text or ""
    if not basis:
        for p in (grade_path, sot_path):
            if not p:
                continue
            try:
                basis += Path(p).read_text(encoding="utf-8") + "\n"
            except (OSError, UnicodeDecodeError):
                continue
    if not basis:
        return []

    issues: list[str] = []
    grade = _canonical_grade(basis)

    if grade:
        for lineno, line in enumerate(script_text.split("\n"), 1):
            if not _GRADE_MENTION.search(line):
                continue
            for pat, level in _GRADE_TOKENS:
                m = pat.search(line)
                if m and level != grade:
                    issues.append(
                        f"GRADE_CONTRADICTION: script says '{m.group(0)}' "
                        f"({level}) but basis final GRADE is {grade} [L{lineno}]"
                    )
                    break

    if _NNT_NULL.search(basis):
        for lineno, line in enumerate(script_text.split("\n"), 1):
            m = _NNT_PROJECTION.search(line)
            if m:
                issues.append(
                    f"NNT_NULL_CONTRADICTION: basis computed NNT=inf/no_effect but "
                    f"script projects a benefit [L{lineno}: …{m.group(0)[:40]}…]"
                )
    return issues
