"""Does the translated source of truth still say what the source of truth said?

PLAN.md Step 11. Smart translates the SOT and nothing checks it, yet the translated SOT is the basis
for every claim in a Japanese episode — a mistranslated finding poisons everything downstream, in
silence.

**Ordered records, not sets.** Set equality was the first draft and it is too weak in two exact
ways. It ignores multiplicity, so a duplicated value can vanish and the set is unchanged. And it
ignores association, so a number can be detached from one claim and attached to another — every
number still present, every set identical, every claim now wrong. PMID and DOI sets have the same
attribution hole.

**Ordered by BLOCK, and unordered within one.** Two earlier versions were both too strict for the
language this exists to check. The flat token sequence failed a faithful translation outright, since
Japanese puts the timepoint before the value: "5.0% ... at 12 months" becomes 「12か月時点…5.0%」.
Grouping by physical line failed the next case up — nothing makes a translator preserve line
wrapping, so a reflowed paragraph read as a moved claim (prepush codex 2026-08-13).

So the boundary is a block: a table row is its own block, because a table row IS a claim and the
per-row association is what catches a value swapped between studies; everything else groups into
blank-line-separated paragraphs, which is the unit a translator may reflow inside but does not
reorder. Blocks are compared in sequence, tokens within a block as a multiset.

The cost is stated rather than hidden: two claims that share one paragraph can have their numbers
exchanged without this noticing. Tightening that would mean failing legitimate translations, and a
check that cries wolf gets removed rather than satisfied.

**What this deliberately does not check.** A dropped negation and an inflated hedge
(「示唆される」 becoming 「示された」) change no number and no citation, so nothing here can see them.
That is not a gap to be patched with a keyword list — it is the boundary between what Python can
prove and what needs a reader. Claude's half of Step 11 covers it; this half must not pretend to.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

#: Identifiers whose text must survive translation unchanged. A citation that moves from one claim
#: to another is the attribution failure a set comparison cannot see.
_CITATION = re.compile(r"\b(?:PMID:?\s*(\d{6,9})|10\.\d{4,9}/[-._;()/:a-z0-9A-Z]+)")

#: A DOI's character class legitimately contains dots and brackets, so the match runs into whatever
#: punctuation ends the sentence. Japanese ends it with 。or follows the DOI with a particle, and the
#: token would differ for a reason that is not a mistranslation (prepush codex 2026-08-13).
_DOI_TRAILING = ".,;:)]}>'\"、。"

#: Numbers, including decimals and thousands separators. Signs are kept: −0.05 and 0.05 are
#: different claims, and a lost minus sign flips a direction.
_NUMBER = re.compile(r"[-−–+]?\d+(?:,\d{3})*(?:\.\d+)?")

#: Markdown scaffolding that carries no claim. Section numbers appear on both sides and would
#: otherwise dominate the comparison with noise nobody translated.
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_TABLE_RULE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")


@dataclass(frozen=True)
class ClaimToken:
    """One checkable token, and enough of its line to say where it was."""

    kind: str  # "number" | "citation"
    value: str  # normalised
    line: int
    context: str

    def where(self) -> str:
        excerpt = self.context if len(self.context) <= 60 else self.context[:57] + "…"
        return f"line {self.line}: {excerpt!r}"


def normalise(text: str) -> str:
    """Fold the differences Japanese typesetting introduces and meaning does not.

    NFKC turns full-width digits and ％ into their ASCII forms, which is most of it; the rest is
    thousands separators and the several dash characters that all mean minus. Without this the
    comparison false-fails on formatting, and a check that cries wolf is a check that gets removed.
    """
    folded = unicodedata.normalize("NFKC", text)
    return folded.replace("−", "-").replace("–", "-").replace("−", "-")


def _normalise_number(raw: str) -> str:
    cleaned = raw.replace(",", "").lstrip("+")
    try:
        # Through float and back, so 5 and 5.0 and 5.00 are one value rather than three.
        return f"{float(cleaned):g}"
    except ValueError:
        return cleaned


def claim_tokens(text: str) -> list[ClaimToken]:
    """Every number and citation in the document, in the order it appears.

    Headings and table rules are skipped: their numbers are scaffolding that appears identically on
    both sides, and letting them in would bury a real mismatch in noise.
    """
    tokens: list[ClaimToken] = []
    for index, raw_line in enumerate(normalise(text).splitlines(), start=1):
        line = raw_line.strip()
        if not line or _HEADING.match(raw_line) or _TABLE_RULE.match(raw_line):
            continue
        for match in _CITATION.finditer(line):
            value = match.group(0).replace(" ", "").rstrip(_DOI_TRAILING)
            tokens.append(ClaimToken("citation", value, index, line))
        # Citations contain digits; blank them out so a PMID is not also read as a number.
        without_citations = _CITATION.sub(lambda m: " " * len(m.group(0)), line)
        for match in _NUMBER.finditer(without_citations):
            tokens.append(ClaimToken("number", _normalise_number(match.group(0)), index, line))
    return tokens


def _block_index(text: str) -> dict[int, int]:
    """Which block each 1-based line belongs to.

    A table row is its own block — the row IS a claim, and per-row association is what catches a
    value swapped between two studies. Everything else runs together until a blank line, because a
    paragraph is the unit a translator may reflow inside.
    """
    mapping: dict[int, int] = {}
    block = 0
    in_paragraph = False
    for index, raw_line in enumerate(normalise(text).splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            in_paragraph = False
            continue
        if line.startswith("|"):
            block += 1
            in_paragraph = False
        elif not in_paragraph:
            block += 1
            in_paragraph = True
        mapping[index] = block
    return mapping


def claim_lines(text: str) -> list[tuple[ClaimToken, Counter]]:
    """The claim-bearing blocks in order, each with the multiset of tokens it carries.

    Blocks with no number and no citation are dropped from both sides: they carry nothing this half
    can check, and keeping them would make the comparison depend on the translator's paragraphing.
    """
    blocks = _block_index(text)
    grouped: dict[int, list[ClaimToken]] = {}
    for token in claim_tokens(text):
        grouped.setdefault(blocks.get(token.line, token.line), []).append(token)
    return [
        (tokens[0], Counter((t.kind, t.value) for t in tokens))
        for _block, tokens in sorted(grouped.items())
    ]


def translation_errors(source: str, translated: str) -> list[str]:
    """Everything the comparison can prove is wrong with a translation.

    Reported as the first divergence plus the totals, rather than a full diff: the first place two
    sequences part company is where a reader has to look, and a hundred cascading mismatches after
    it are the same defect said a hundred times.
    """
    before, after = claim_lines(source), claim_lines(translated)
    errors: list[str] = []

    for position, ((was_token, was), (now_token, now)) in enumerate(zip(before, after, strict=False)):
        if was == now:
            continue
        missing, added = was - now, now - was
        errors.append(
            f"claim {position + 1} differs — source at {was_token.where()}; "
            f"translation at {now_token.where()}"
            + (f"; missing {_render(missing)}" if missing else "")
            + (f"; added {_render(added)}" if added else "")
        )
        break

    if len(before) != len(after):
        source_total, translated_total = _totals(before), _totals(after)
        missing, added = source_total - translated_total, translated_total - source_total
        errors.append(
            f"the source carries {len(before)} claim-bearing line(s) and the translation "
            f"{len(after)}"
            + (f"; missing {_render(missing)}" if missing else "")
            + (f"; added {_render(added)}" if added else "")
        )
    return errors


def _totals(lines):
    total: Counter = Counter()
    for _token, counts in lines:
        total.update(counts)
    return total


def _render(counter) -> str:
    return ", ".join(f"{value} ({kind})" + (f" ×{n}" if n > 1 else "") for (kind, value), n in sorted(counter.items()))


def audit_translation(source: str, translated: str) -> dict:
    """A verdict a caller can log, and a list a person can act on."""
    errors = translation_errors(source, translated)
    return {
        "checked_tokens": len(claim_tokens(source)),
        "checked_claims": len(claim_lines(source)),
        "ok": not errors,
        "errors": errors,
        # Said out loud on every run, because a check whose limits are only written in its docstring
        # gets read as a guarantee it never made.
        "not_checked": (
            "negation, hedge strength and direction wording — no number moves when those do; and two "
            "claims sharing one paragraph can exchange numbers, which is the price of tolerating the "
            "line reflow a faithful translation is entitled to"
        ),
    }
