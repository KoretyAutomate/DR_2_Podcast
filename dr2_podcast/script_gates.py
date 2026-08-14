"""The mechanical tiers of the draft and polish loops, and the bound that stops them.

PLAN.md Steps 3, 4 and 6. Three tiers, cheapest first: tier 0 is shape, tier 1 is lexical, tier 2 is
meaning. Only the first two live here — **tier 1 is a pre-filter that keeps Claude's reads down, not
a correctness gate**, and the distinction is the point of the design:

* A draft can reuse an ALLOWED number and attach it to the wrong endpoint, arm, population or
  timepoint. Every numeral passes; the claim is false.
* 「効果が確認された」 and 「効果が確認されなかった」 carry the same hedge token. The negation is the
  whole meaning, and no lexical rule separates them.

So anything about meaning belongs to tier 2, which is Claude's. What is here is what Python can
prove, and its limits are stated rather than papered over.

**The polish gate preserves the claim SET, not its wording.** Requiring the approved sentences to
survive verbatim would forbid the work polishing exists to do — the claim sentences are exactly the
ones most likely to need natural Japanese. What may not change is what a claim asserts.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

#: Revision rounds per section, for both loops. One bound, named once. `MAX_AUDITOR_REVISIONS` in
#: config.py is dead — clinical.py hardcodes its own local MAX_REVISIONS and never imports it — so
#: this deliberately does not reuse that name.
MAX_REVISION_ROUNDS = 3

#: Rounds of an identical finding set before the loop is declared stuck. Comparing the finding SET
#: rather than the count is what makes this work: equal counts can mean one defect fixed and another
#: introduced, and a falling count can cycle among blockers.
THRASH_REPEATS = 2

#: A sentence carries the claim it serves, or `none` for connective tissue. An unannotated draft is
#: malformed — the annotation is what makes every later check possible.
_ANNOTATION = re.compile(r"\[\[(?P<claim>[A-Za-z0-9_none-]+)\]\]\s*$")
_SPEAKER = re.compile(r"^\s*(?:Host|ホスト)\s*(?P<n>[12])\s*[:：]")

#: Numerals that are not claims. Counting expressions, ordinals and dosing frequencies appear in
#: natural Japanese constantly, and rejecting them would make tier 1 fire on every well-written
#: section — a gate that fires on correct work gets switched off.
_NON_CLAIM_NUMERAL = re.compile(
    r"\d+\s*(?:つ|個|回目|番目|人目|つ目|点目|段階|ステップ|章|話|分間|秒|時間目)"
    r"|(?:1日|一日)\s*\d+\s*(?:回|錠|回分)"
    r"|第\s*\d+"
)


@dataclass(frozen=True)
class Sentence:
    """One annotated line of a draft."""

    index: int
    speaker: int | None
    text: str
    claim_id: str | None  # None for connective tissue


@dataclass(frozen=True)
class Finding:
    """One gate failure, normalised so two rounds can be compared.

    Identity is ``(claim_id, rule_id, location)`` and NOT the message: a message that mentions a
    count or an index changes between rounds while naming the same defect, and thrash detection
    would never fire.
    """

    rule_id: str
    claim_id: str | None
    location: str
    message: str

    def identity(self) -> tuple[str, str | None, str]:
        return (self.rule_id, self.claim_id, self.location)


@dataclass
class LoopState:
    """One section's journey through a bounded loop."""

    section: str
    rounds: list[list[Finding]] = field(default_factory=list)

    def record(self, findings: list[Finding]) -> None:
        self.rounds.append(list(findings))

    @property
    def exhausted(self) -> bool:
        return len(self.rounds) >= MAX_REVISION_ROUNDS

    @property
    def thrashing(self) -> bool:
        """The same finding set, twice running. Rewriting is not converging on anything."""
        if len(self.rounds) < THRASH_REPEATS:
            return False
        recent = [frozenset(f.identity() for f in round_) for round_ in self.rounds[-THRASH_REPEATS:]]
        return len(set(recent)) == 1 and bool(recent[0])

    @property
    def should_stop(self) -> bool:
        return self.exhausted or self.thrashing or not self.rounds[-1]

    def event(self) -> dict[str, Any]:
        """What gets written to meta/loop_events.json — a section type that repeatedly escapes is a
        blueprint-template bug, and only the record makes that visible."""
        surviving = self.rounds[-1] if self.rounds else []
        return {
            "section": self.section,
            "rounds": len(self.rounds),
            "outcome": "converged" if not surviving else ("thrashed" if self.thrashing else "exhausted"),
            "surviving_findings": [
                {"rule_id": f.rule_id, "claim_id": f.claim_id, "location": f.location, "message": f.message}
                for f in surviving
            ],
        }


def parse_draft(text: str) -> list[Sentence]:
    """Split a draft into annotated sentences.

    One line is one sentence here. The annotation is a trailing ``[[claim_id]]``, and a line without
    one is reported by tier 0 rather than guessed at.
    """
    sentences = []
    for index, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        speaker_match = _SPEAKER.match(line)
        annotation = _ANNOTATION.search(line)
        claim = annotation.group("claim") if annotation else None
        # The speaker prefix is stripped from the body, or "Host 1:" contributes a numeral to every
        # single line and tier 1 fires on the whole script.
        body = _SPEAKER.sub("", _ANNOTATION.sub("", line)).strip()
        sentences.append(
            Sentence(
                index=index,
                speaker=int(speaker_match.group("n")) if speaker_match else None,
                text=body,
                claim_id=None if claim in (None, "none") else claim,
            )
        )
    return sentences


def tier0_errors(text: str, known_claim_ids: set[str]) -> list[Finding]:
    """Shape. Every sentence annotated, every claim id one the blueprint declared.

    A tier 0 failure is malformed output, not a wrong answer — the caller must treat it as a
    transport retry and NOT spend a revision round on it.
    """
    findings: list[Finding] = []
    sentences = parse_draft(text)
    if not sentences:
        return [Finding("tier0.empty", None, "section", "the section is empty")]

    annotated = re.compile(r"\[\[[A-Za-z0-9_none-]+\]\]\s*$")
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if raw.strip() and not annotated.search(raw.strip()):
            findings.append(
                Finding("tier0.unannotated", None, f"line {line_no}",
                        "every sentence carries the claim_id it serves, or [[none]] for connective tissue")
            )
    for sentence in sentences:
        if sentence.claim_id and sentence.claim_id not in known_claim_ids:
            findings.append(
                Finding("tier0.unknown_claim", sentence.claim_id, f"line {sentence.index}",
                        f"{sentence.claim_id!r} is not a claim this section's blueprint declares")
            )
    return findings


def _numerals(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKC", text)
    without_counts = _NON_CLAIM_NUMERAL.sub(" ", folded)
    return re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|％)?", without_counts)


def _normalise_numeral(value: str) -> str:
    cleaned = unicodedata.normalize("NFKC", value).replace(",", "").replace(" ", "")
    return cleaned.rstrip("%") + ("%" if cleaned.endswith("%") else "")


def _allowed_set(step: dict[str, Any]) -> set[str]:
    """The allowed numbers, normalised the way a drafted numeral will be.

    A blueprint writes 「12週」 and 「n=1,447」 — value and unit together, because that is what the
    claim is about — while extraction sees the numeral alone. Normalising only one side means an
    allowed number never matches itself, which is a gate that rejects every correct draft.
    """
    allowed: set[str] = set()
    for claim in step.get("claims", []):
        for entry in claim["allowed_numbers"]:
            allowed.update(_normalise_numeral(found) for found in _numerals(entry))
    return allowed


def tier1_errors(text: str, step: dict[str, Any]) -> list[Finding]:
    """Lexical. Numerals, forbidden phrasings, speaker alternation.

    Cheap on purpose, and it must not fire on correct work: counting expressions like 「3つ」 and
    「1日2回」 are not claims, and a gate that rejects them is one nobody keeps switched on.
    """
    findings: list[Finding] = []
    allowed = _allowed_set(step)
    forbidden = {phrase for claim in step.get("claims", []) for phrase in claim["must_not_say"]}

    sentences = parse_draft(text)
    for sentence in sentences:
        for numeral in _numerals(sentence.text):
            value = _normalise_numeral(numeral)
            if value and value not in allowed:
                findings.append(
                    Finding("tier1.numeral", sentence.claim_id, f"line {sentence.index}",
                            f"{numeral!r} is not among this section's allowed numbers")
                )
        for phrase in forbidden:
            if phrase in sentence.text:
                findings.append(
                    Finding("tier1.must_not_say", sentence.claim_id, f"line {sentence.index}",
                            f"{phrase!r} is on this claim's must_not_say list")
                )

    speakers = [s.speaker for s in sentences if s.speaker is not None]
    for position, (first, second) in enumerate(zip(speakers, speakers[1:], strict=False)):
        if first == second:
            findings.append(
                Finding("tier1.alternation", None, f"turn {position + 2}",
                        f"speaker {first} speaks twice in a row")
            )
    return findings


def claim_fingerprints(text: str) -> dict[str, Counter]:
    """Per claim, the normalised numerals it carries. The invariance gate's unit of comparison.

    Wording is deliberately absent: what may not change across a polish is what a claim ASSERTS, and
    a literal-preservation rule would forbid the rewriting the phase exists to do.
    """
    fingerprints: dict[str, Counter] = {}
    for sentence in parse_draft(text):
        if not sentence.claim_id:
            continue
        counts = fingerprints.setdefault(sentence.claim_id, Counter())
        counts.update(_normalise_numeral(n) for n in _numerals(sentence.text))
    return fingerprints


def invariance_errors(approved: str, polished: str, step: dict[str, Any]) -> list[Finding]:
    """Step 4's gate: the claim set and its numbers survive; the wording need not.

    Runs before Claude looks, at zero Claude cost. What it cannot see is the same thing tier 1
    cannot: a polish that keeps every number and reverses the meaning around them.
    """
    findings: list[Finding] = []
    before, after = claim_fingerprints(approved), claim_fingerprints(polished)

    for claim_id in sorted(set(before) - set(after)):
        findings.append(
            Finding("polish.claim_dropped", claim_id, "section",
                    f"{claim_id!r} was in the approved draft and is not in the polished one")
        )
    for claim_id in sorted(set(after) - set(before)):
        findings.append(
            Finding("polish.claim_added", claim_id, "section",
                    f"{claim_id!r} appears in the polish and was in no approved draft")
        )
    for claim_id in sorted(set(before) & set(after)):
        if before[claim_id] != after[claim_id]:
            missing = before[claim_id] - after[claim_id]
            added = after[claim_id] - before[claim_id]
            findings.append(
                Finding("polish.numbers_moved", claim_id, "section",
                        f"{claim_id!r} changed its numbers"
                        + (f"; lost {sorted(missing)}" if missing else "")
                        + (f"; gained {sorted(added)}" if added else ""))
            )
    findings.extend(tier1_errors(polished, step))
    return findings


def write_loop_events(run_dir, events: list[dict[str, Any]]) -> None:
    """Append this run's loop outcomes to meta/loop_events.json."""
    from dr2_podcast.artifacts import read_json_strict, write_json_atomic

    path = run_dir / "meta/loop_events.json"
    existing = read_json_strict(path).get("events", []) if path.exists() else []
    write_json_atomic(path, {"schema_version": 1, "events": existing + events})


# --------------------------------------------------------------------------- #
# Banned phrases — PLAN.md Step 12
# --------------------------------------------------------------------------- #
#: Where the ban list lives. It is enforced today only by
#: `regen_edu_aivis_from_scripttxt.py`, which is the educational-series render path — so the MAIN
#: pipeline can ship a banned phrase to audio, which is the hole Step 12 names.
BANNED_PHRASES_FILE = "educational_series/banned_phrases.json"


def load_banned_phrases(root=None) -> list[dict[str, Any]]:
    """The ban list, or a failure. Never an empty list standing in for one.

    A gate that degrades to "no phrases are banned" when its list will not load is a gate that
    passes everything on the day it breaks — and the reason this file exists is that a prose rule
    was not enough: 「今日の核心」 was abolished series-wide and shipped to audio twice anyway,
    caught by ear three days later.
    """
    import json
    from pathlib import Path as _Path

    from dr2_podcast.artifacts import ArtifactError

    base = _Path(root) if root else _Path(__file__).resolve().parent.parent
    path = base / BANNED_PHRASES_FILE
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            f"the banned-phrase list at {path} could not be read ({exc}), so nothing can say whether "
            f"this script is publishable. Refusing rather than treating an unreadable ban list as an "
            f"empty one."
        ) from exc
    entries = document.get("banned")
    if not isinstance(entries, list) or not entries:
        raise ArtifactError(f"{path} carries no banned phrases; an empty gate is not a gate")
    return entries


def banned_phrase_findings(script: str, root=None) -> list[Finding]:
    """Every banned phrase the script says, with the reason it was banned.

    Substring matching, deliberately: 「核心」 was banned as the literal 「今日の核心」 first, and
    「ベイズ思考の核心」 sailed through for months because it was a different compound. The ban is on
    the word wherever it appears.
    """
    findings = []
    for line_no, line in enumerate(script.splitlines(), start=1):
        for entry in load_banned_phrases(root):
            pattern = entry.get("pattern", "")
            if pattern and pattern in line:
                findings.append(
                    Finding(
                        "banned_phrase",
                        None,
                        f"line {line_no}",
                        f"{pattern!r} is banned ({entry.get('since', 'date unrecorded')}). "
                        f"Use instead: {entry.get('use_instead', '(no alternative recorded)')}",
                    )
                )
    return findings
