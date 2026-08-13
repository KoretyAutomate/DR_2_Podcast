"""The 確信度 lookup: GRADE level, capped by the evidence staircase, never picked by a model.

PLAN.md Step 9b item 4. Step S shipped CONFIDENCE_LADDER and the monotonicity check but left the
staircase axis undefined, because inventing one would be false precision; sequencing item 3 defines
it, and these tests are what "defined" means.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dr2_podcast.research.confidence import (
    CONFIDENCE_LADDER,
    DESIGN_STAIRCASE,
    STAIRCASE_CAP,
    confidence_level,
    design_rung,
    staircase_position,
)


def _study(design):
    return SimpleNamespace(study_design=design)


# --------------------------------------------------------------------------- #
# Reading a rung off the design text
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "rung"),
    [
        ("meta-analysis of 14 RCTs", "meta_analysis"),
        ("Systematic Review", "meta_analysis"),
        ("parallel RCT", "rct"),
        ("randomised controlled trial", "rct"),
        ("difference-in-differences", "quasi_experimental"),
        ("prospective cohort", "cohort"),
        ("case-control", "case_control"),
        ("cross-sectional survey", "cross_sectional"),
        ("case series", "case_report"),
        ("editorial", "expert_opinion"),
    ],
)
def test_a_design_names_its_rung(text: str, rung: str) -> None:
    assert design_rung(text) == rung


def test_a_meta_analysis_of_rcts_is_read_as_the_higher_rung() -> None:
    """The text names two rungs and means the higher one, so pattern order is load-bearing."""
    assert design_rung("meta-analysis of randomized controlled trials") == "meta_analysis"


@pytest.mark.parametrize("text", [None, "", "   ", "something the extractor could not classify"])
def test_an_unreadable_design_is_not_a_rung(text) -> None:
    """Absent from the staircase, not sitting at its bottom: a failed extraction must not cap the
    confidence of a body of evidence it says nothing about."""
    assert design_rung(text) is None


# --------------------------------------------------------------------------- #
# The position of a body of evidence
# --------------------------------------------------------------------------- #
def test_the_position_is_the_highest_rung_any_study_occupies() -> None:
    studies = [_study("cross-sectional"), _study("prospective cohort"), _study("parallel RCT")]
    assert staircase_position(studies) == "rct"


def test_an_unreadable_design_does_not_lower_the_position() -> None:
    assert staircase_position([_study("parallel RCT"), _study("unclear")]) == "rct"


def test_no_readable_design_is_no_position() -> None:
    assert staircase_position([_study(None), _study("unclear")]) is None
    assert staircase_position([]) is None


# --------------------------------------------------------------------------- #
# The lookup
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("grade", "word"),
    [("high", "ほぼ確実"), ("moderate", "高い"), ("low", "中程度"), ("very_low", "低い")],
)
def test_randomised_evidence_speaks_its_grade_word_unchanged(grade: str, word: str) -> None:
    assert confidence_level(grade, [_study("parallel RCT")]) == word


def test_the_staircase_caps_a_confident_grade_on_observational_evidence() -> None:
    """The point of the second axis. GRADE can upgrade observational evidence to HIGH for a large
    effect; the episode still may not say ほぼ確実 off a cross-sectional base."""
    assert confidence_level("high", [_study("cross-sectional survey")]) == "低い"
    assert confidence_level("high", [_study("prospective cohort")]) == "中程度"


def test_the_staircase_never_raises() -> None:
    """It is a cap, not a second score. GRADE already starts HIGH for randomised evidence, so
    letting the staircase lift a low GRADE would count study design twice."""
    assert confidence_level("very_low", [_study("meta-analysis of RCTs")]) == "低い"


def test_no_grade_level_is_a_real_answer_not_a_missing_one() -> None:
    for level in (None, "", "Not Determined", "excellent"):
        assert confidence_level(level, [_study("parallel RCT")]) == "まだ分からない"


def test_an_assessment_with_no_readable_design_keeps_its_grade_word() -> None:
    """The cap cannot be applied, and discarding an assessment that was made would be worse than
    not applying a cap that nothing supports."""
    assert confidence_level("high", [_study("unclear")]) == "ほぼ確実"


# --------------------------------------------------------------------------- #
# The tables themselves
# --------------------------------------------------------------------------- #
def test_every_rung_has_a_cap() -> None:
    """A rung without a cap raises KeyError inside the lookup — on a live run, at step 7."""
    assert set(STAIRCASE_CAP) == set(DESIGN_STAIRCASE)


def test_every_cap_is_a_ladder_word() -> None:
    for rung, cap in STAIRCASE_CAP.items():
        assert cap in CONFIDENCE_LADDER, rung


def test_the_caps_do_not_fall_as_the_designs_get_stronger() -> None:
    """A cap that dipped in the middle would mean a stronger design permitted less confidence."""
    caps = [CONFIDENCE_LADDER.index(STAIRCASE_CAP[rung]) for rung in DESIGN_STAIRCASE]
    assert caps == sorted(caps)
