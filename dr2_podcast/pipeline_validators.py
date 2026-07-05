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
    r'^[ \t]*\*{0,2}\s*(?:host|ホスト|speaker)[ _]*([12])[ _]*\*{0,2}\s*[:：]',
    re.IGNORECASE,
)
_SPK_CANON = re.compile(r'^Host ([12]):')


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
            rest = line[m.end():].lstrip()
            newline = f"Host {m.group(1)}: {rest}".rstrip() if rest else f"Host {m.group(1)}:"
            if newline != line:
                fixed += 1
            out.append(newline)
        else:
            out.append(line)
    return "\n".join(out), fixed


def check_speaker_alternation(text: str, max_run: int = 6,
                              single_voice_ratio: float = 0.85) -> list[str]:
    """Flag structural speaker defects. Run AFTER normalize_speaker_labels.

    - SPEAKER_RUN: more than ``max_run`` consecutive same-speaker turns (a
      mislabeled block, e.g. the Saturday sleep episode's 40 Host-1 turns).
    - SINGLE_VOICE: one speaker owns more than ``single_voice_ratio`` of all turns.
    """
    turns = [m.group(1) for line in text.split("\n")
             if (m := _SPK_CANON.match(line))]
    if not turns:
        return ["NO_SPEAKER_LABELS: no canonical 'Host N:' turns found"]
    issues = []
    run = maxrun = 1
    for a, b in zip(turns, turns[1:]):
        run = run + 1 if a == b else 1
        maxrun = max(maxrun, run)
    if maxrun > max_run:
        issues.append(
            f"SPEAKER_RUN: {maxrun} consecutive same-speaker turns "
            f"(max {max_run}) — likely a mislabeled block")
    dom = max(Counter(turns).values()) / len(turns)
    if dom > single_voice_ratio:
        issues.append(
            f"SINGLE_VOICE: {int(dom * 100)}% of {len(turns)} turns are one "
            f"speaker (max {int(single_voice_ratio * 100)}%)")
    return issues


# --------------------------------------------------------------------------- #
# 2. Citations
# --------------------------------------------------------------------------- #
_YEAR = r'(?:18|19|20)\d\d'
# Author-year in a script: "Vandewalle et al. (2007)", "Zhong et al., 2022",
# "Leproult & Van Cauter 2011", "Winer (2019)". Surname is Title-case (first
# upper, then lowercase) so ALLCAPS acronyms / trial names (RCT, PREDIMED, VITAL)
# do NOT match.
_CITE = re.compile(
    r'\b([A-Z][a-zÀ-ſ]{2,}(?:\s+[A-Z][a-zÀ-ſ]+)?)'  # surname (opt. "Van Cauter")
    r'(?:\s*(?:et\s*al\.?|&\s*[A-Z][a-zÀ-ſ]+|and\s+[A-Z][a-zÀ-ſ]+))?'
    r'\s*[,\s]?\s*[\(（]?(' + _YEAR + r')[\)）]?'
)

# Title-case English words that legitimately precede a year but are NOT authors
# (journals, orgs, trial names, generic nouns). Kept lowercase for comparison.
_NONAUTHOR = {
    "nature", "science", "cell", "lancet", "cochrane", "harvard", "oxford",
    "day", "week", "act", "study", "table", "figure", "since", "before",
    "after", "around", "about", "the", "in", "by", "december", "january",
    "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november",
}


def _load_source_whitelist(sot_path: str | None,
                           sources_json_path: str | None,
                           sot_text: str | None = None) -> tuple[set, set, set]:
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
            ref = re.match(r'\s*\d+\.\s+([A-Z][^*.]+?)\.{1,2}\s*\*', line)
            if not ref:
                continue
            year = re.search(r'[\(（](\d{4})[\)）]', line)
            pm = re.search(r'PMID:\s*\[?(\d+)', line)
            first = re.match(r'([A-Z][a-zA-ZÀ-ſ\-]{2,})', ref.group(1).strip())
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
            pm = re.search(r'/(\d{6,9})/?', str(e.get("url", "")))
            if pm:
                pmids.add(pm.group(1))
    return surnames, pairs, pmids


def validate_citations(script_text: str, sot_path: str | None = None,
                       sources_json_path: str | None = None,
                       sot_text: str | None = None) -> list[str]:
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
        sn = re.match(r'[A-Za-zÀ-ſ]+', m.group(1)).group(0).lower()
        yr = m.group(2)
        if sn in _NONAUTHOR or sn in seen:
            continue
        seen.add(sn)
        if sn not in surnames:
            issues.append(
                f"FABRICATED_CITATION: '{m.group(1).strip()} ({yr})' — "
                f"surname not found in any retrieved source")
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
    return [f"DUPLICATE_LINE (x{n}): \"{body[:50]}…\""
            for body, n in counts.items() if n > 1]


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def validate_script_structure(script_text: str, sot_path: str | None = None,
                              sources_json_path: str | None = None,
                              sot_text: str | None = None) -> dict:
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
