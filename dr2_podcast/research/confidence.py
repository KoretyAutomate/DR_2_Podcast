"""確信度: the one word the episode says about how sure we are, and where it comes from.

PLAN.md Step 9b item 4 asks for a deterministic GRADE + evidence-staircase lookup, and Step S left
the staircase axis undefined on purpose — "inventing one would be exactly the false precision Step
9's rewrite removed". Defining it was handed to sequencing item 3, which is this file.

The definition is deliberately narrow, and it is built out of something the pipeline already
extracts rather than a new judgement:

* **The staircase is the study-design hierarchy** the episodes already teach as エビデンスの階段 —
  meta-analysis above RCT above cohort above case-control above cross-sectional above case report.
  Its position for a body of evidence is the highest rung any included study occupies, read from
  ``DeepExtraction.study_design``. No counting, no weighting, no threshold: those would be numbers
  nobody derived.
* **The staircase can only LOWER the confidence, never raise it.** GRADE already begins at HIGH for
  randomised evidence and LOW for observational, so multiplying the two would count study design
  twice — the error the plan's own critique of `verdict_contribution` names. What the staircase adds
  is the thing GRADE's level alone cannot express: you may not speak 高い confidence off a base
  whose best design is cross-sectional, whatever the modifiers did.

The model never picks the word. That is the whole point of the lookup: a synthesised sentence can
say "the evidence is quite strong" in a hundred registers, and the episode's 確信度 has to be one of
five values that mean the same thing every week.
"""

from __future__ import annotations

import re

from dr2_podcast.schemas import CONFIDENCE_LADDER

__all__ = [
    "CONFIDENCE_LADDER",
    "DESIGN_STAIRCASE",
    "GRADE_TO_CONFIDENCE",
    "STAIRCASE_CAP",
    "confidence_level",
    "design_rung",
    "staircase_position",
]

#: The rungs, lowest first. Position is an index into this tuple; higher is stronger.
DESIGN_STAIRCASE: tuple[str, ...] = (
    "expert_opinion",
    "case_report",
    "cross_sectional",
    "case_control",
    "cohort",
    "quasi_experimental",
    "rct",
    "meta_analysis",
)

#: Substrings that identify a rung in the free text ``study_design`` carries. Ordered from the
#: highest rung down, because "meta-analysis of RCTs" names two rungs and means the higher one.
_DESIGN_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("meta_analysis", ("meta-analysis", "meta analysis", "metaanalysis", "systematic review")),
    ("rct", ("rct", "randomized controlled", "randomised controlled", "randomized trial", "randomised trial")),
    ("quasi_experimental", ("quasi-experimental", "quasi experimental", "difference-in-differences",
                            "regression discontinuity", "interrupted time series")),
    ("cohort", ("cohort", "longitudinal", "prospective study")),
    ("case_control", ("case-control", "case control")),
    ("cross_sectional", ("cross-sectional", "cross sectional", "survey", "correlational")),
    ("case_report", ("case report", "case series")),
    ("expert_opinion", ("expert opinion", "editorial", "narrative review", "commentary")),
)

#: GRADE's four levels as ladder words, before the staircase cap.
GRADE_TO_CONFIDENCE: dict[str, str] = {
    "high": "ほぼ確実",
    "moderate": "高い",
    "low": "中程度",
    "very_low": "低い",
}

#: The highest 確信度 a body of evidence may reach given its best study design. Observational
#: designs cap below 高い because no amount of GRADE upgrading makes a cross-sectional base a
#: confident answer to a causal question — which is the claim the episode is making when it speaks.
STAIRCASE_CAP: dict[str, str] = {
    "meta_analysis": "ほぼ確実",
    "rct": "ほぼ確実",
    "quasi_experimental": "高い",
    "cohort": "中程度",
    "case_control": "中程度",
    "cross_sectional": "低い",
    "case_report": "低い",
    "expert_opinion": "まだ分からない",
}

_UNKNOWN = "まだ分からない"


def design_rung(study_design: str | None) -> str | None:
    """Which rung a study's design text names, or None when it names none.

    None is not a rung: a design the extractor could not read is absent from the staircase rather
    than sitting at its bottom. Putting unreadable designs on the lowest rung would let a failed
    extraction cap the confidence of a body of evidence it says nothing about.
    """
    if not study_design:
        return None
    text = re.sub(r"\s+", " ", study_design.strip().lower())
    for rung, patterns in _DESIGN_PATTERNS:
        if any(pattern in text for pattern in patterns):
            return rung
    return None


def staircase_position(extractions) -> str | None:
    """The highest rung any included study occupies, or None when none is readable."""
    best = None
    best_index = -1
    for extraction in extractions or []:
        rung = design_rung(getattr(extraction, "study_design", None))
        if rung is None:
            continue
        index = DESIGN_STAIRCASE.index(rung)
        if index > best_index:
            best, best_index = rung, index
    return best


def confidence_level(grade_level: str | None, extractions) -> str:
    """The episode's 確信度, from the structured GRADE level and the evidence staircase.

    Returns a value from :data:`CONFIDENCE_LADDER`. With no GRADE level there is nothing to speak —
    that is まだ分からない, and it is a real answer rather than a missing one.
    """
    word = GRADE_TO_CONFIDENCE.get((grade_level or "").strip().lower())
    if word is None:
        return _UNKNOWN
    rung = staircase_position(extractions)
    if rung is None:
        # GRADE assessed something, but no study's design was readable. The cap cannot be applied,
        # so the GRADE word stands — saying まだ分からない here would discard an assessment that was
        # made, and inventing a cap from an unread design would be the false precision this avoids.
        return word
    cap = STAIRCASE_CAP[rung]
    return word if CONFIDENCE_LADDER.index(word) <= CONFIDENCE_LADDER.index(cap) else cap
