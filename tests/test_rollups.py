"""Replication, funding and bias counted from the records — PLAN.md Step 9b items 2 and 3.

The exit criterion says these must be *correct*, not merely non-null: a fabricated answer passes a
presence check. So every case here is one whose right answer is countable by hand from the fixture.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dr2_podcast.research.clinical import Finding, FundingBlock
from dr2_podcast.research.rollups import (
    bias_rollup,
    design_rollup,
    funding_rollup,
    replication_groups,
    replication_rollup,
)


def _finding(endpoint="hip fracture", direction="decrease", **kw):
    return Finding(
        population="adults",
        intervention="drug",
        comparator="placebo",
        endpoint=endpoint,
        direction=direction,
        finding_key=f"{endpoint}|{direction}",
        **kw,
    )


def _study(*, group=None, registration=None, category="government", design="parallel RCT",
           bias="low", findings=None):
    return SimpleNamespace(
        pmid="1",
        title="a study",
        study_design=design,
        author_group=group,
        trial_registration=registration,
        risk_of_bias=bias,
        funding=FundingBlock(
            funding_raw="a funder",
            funding_category=category,
            funding_disclosure="disclosed",
            funding_source_type="api_metadata",
        ),
        findings=findings if findings is not None else [_finding()],
    )


# --------------------------------------------------------------------------- #
# Replication — the question is whether someone ELSE saw it, in someone ELSE's participants
# --------------------------------------------------------------------------- #
def test_two_groups_on_two_cohorts_is_a_replication() -> None:
    studies = [
        _study(group="Tanaka H; Osaka", registration="NCT01"),
        _study(group="Smith J; Yale", registration="NCT02"),
    ]
    [group] = replication_groups(studies)
    assert group.independent_groups == 2
    assert group.distinct_cohorts == 2
    assert group.is_replicated


def test_one_group_reporting_twice_is_not_a_replication() -> None:
    studies = [
        _study(group="Tanaka H; Osaka", registration="NCT01"),
        _study(group="Tanaka H; Osaka", registration="NCT02"),
    ]
    [group] = replication_groups(studies)
    assert group.independent_groups == 1
    assert not group.is_replicated


def test_two_groups_reporting_the_same_trial_is_not_a_replication() -> None:
    """A trial reported twice is one trial. Counting it as two is how a single result becomes
    'consistently found across studies'."""
    studies = [
        _study(group="Tanaka H; Osaka", registration="NCT01"),
        _study(group="Smith J; Yale", registration="NCT01"),
    ]
    [group] = replication_groups(studies)
    assert group.distinct_cohorts == 1
    assert not group.is_replicated


def test_agreeing_on_the_endpoint_but_not_the_direction_is_not_a_replication() -> None:
    """It is the disagreement the falsification track exists to surface, and folding the two into
    one group would report it as corroboration."""
    studies = [
        _study(group="Tanaka H; Osaka", registration="NCT01", findings=[_finding(direction="decrease")]),
        _study(group="Smith J; Yale", registration="NCT02", findings=[_finding(direction="increase")]),
    ]
    groups = replication_groups(studies)
    assert len(groups) == 2
    assert not any(g.is_replicated for g in groups)


# prepush codex 2026-08-13 [P2]: with no trial registration anywhere, distinct_cohorts is 0, and a
# `!= 1` test read that as "not overlapping" — so two groups reporting the same finding with no
# registration data counted as replicated. Observational studies routinely have no registration, so
# that was the common case, and it turned missing data into positive evidence.
def test_two_groups_with_no_registrations_is_unknown_not_replicated() -> None:
    studies = [
        _study(group="Tanaka H; Osaka", registration=None),
        _study(group="Smith J; Yale", registration=None),
    ]
    [group] = replication_groups(studies)
    assert group.independent_groups == 2
    assert group.distinct_cohorts == 0
    assert group.status == "cohorts_unknown"
    assert not group.is_replicated

    rollup = replication_rollup(studies)
    assert rollup["findings_cohorts_unknown"] == 1
    assert rollup["findings_replicated"] == 0


def test_one_group_with_no_registration_is_simply_not_replicated() -> None:
    """The cohort question only arises once two groups have reported it."""
    [group] = replication_groups([_study(group="Tanaka H; Osaka", registration=None)])
    assert group.status == "not_replicated"


# prepush codex 2026-08-13 [P2], the other half: with one paper registered and one not,
# distinct_cohorts is 1, and reading that as "they all named the same trial" states a negative the
# records do not support. Overlap is unknown unless EVERY report names a registration.
def test_one_registered_report_and_one_unregistered_is_unknown_not_a_negative() -> None:
    studies = [
        _study(group="Tanaka H; Osaka", registration="NCT01"),
        _study(group="Smith J; Yale", registration=None),
    ]
    [group] = replication_groups(studies)
    assert group.distinct_cohorts == 1
    assert group.status == "cohorts_unknown"


def test_every_report_naming_the_same_trial_is_still_a_negative() -> None:
    """The control: when all of them name it, one trial reported twice really is one trial."""
    studies = [
        _study(group="Tanaka H; Osaka", registration="NCT01"),
        _study(group="Smith J; Yale", registration="NCT01"),
    ]
    [group] = replication_groups(studies)
    assert group.status == "not_replicated"


def test_a_paper_with_no_author_group_is_counted_but_not_as_independent() -> None:
    """'We could not tell' is not 'it was not replicated', and the rollup names it separately."""
    studies = [_study(group=None, registration="NCT01"), _study(group=None, registration="NCT02")]
    [group] = replication_groups(studies)
    assert group.independent_groups == 0
    assert group.unattributed == 2
    assert not group.is_replicated

    rollup = replication_rollup(studies)
    assert rollup["findings_unattributable"] == 1
    assert rollup["findings_replicated"] == 0


def test_the_rollup_counts_findings_not_papers() -> None:
    """One paper reporting two endpoints is two findings, each with its own replication status."""
    two = [_finding(endpoint="hip fracture"), _finding(endpoint="falls", direction="null_result")]
    studies = [
        _study(group="Tanaka H; Osaka", registration="NCT01", findings=two),
        _study(group="Smith J; Yale", registration="NCT02", findings=[_finding(endpoint="hip fracture")]),
    ]
    rollup = replication_rollup(studies)
    assert rollup["findings_total"] == 2
    assert rollup["findings_replicated"] == 1, "the fracture finding; falls was reported once"
    assert rollup["findings_single_group"] == 1


# --------------------------------------------------------------------------- #
# Funding — the denominator never silently shrinks
# --------------------------------------------------------------------------- #
def test_the_categories_sum_to_the_number_of_studies() -> None:
    studies = [
        _study(category="industry"),
        _study(category="industry"),
        _study(category="government"),
        _study(category="undisclosed"),
        _study(category="unknown"),
    ]
    rollup = funding_rollup(studies)
    assert rollup["studies_total"] == 5
    assert sum(rollup["by_category"].values()) == 5
    assert rollup["by_category"]["industry"] == 2


def test_undisclosed_and_unknown_are_counted_apart() -> None:
    """The paper being silent is a finding; our failing to extract is a gap. Ep09's thesis is built
    on the difference, so a rollup that merges them erases the thesis."""
    rollup = funding_rollup([_study(category="undisclosed"), _study(category="unknown")])
    assert rollup["undisclosed"] == 1
    assert rollup["unknown"] == 1


def test_a_study_with_no_funding_block_counts_as_unknown_not_as_nothing() -> None:
    bare = _study()
    bare.funding = None
    rollup = funding_rollup([bare])
    assert rollup["studies_total"] == 1
    assert rollup["unknown"] == 1


def test_the_rollup_says_how_much_of_itself_is_unverifiable() -> None:
    """API-derived funding exists nowhere in the paper and carries no locator."""
    quoted = _study(category="government")
    quoted.funding = FundingBlock(
        funding_raw="NIA",
        funding_category="government",
        funding_disclosure="disclosed",
        funding_source_type="extracted_text",
        funding_locator={"fields": ["funding_raw"], "source_artifact_id": "a", "char_offset": 0,
                         "quoted_span": "NIA"},
    )
    rollup = funding_rollup([quoted, _study(category="industry")])
    assert rollup["from_api_metadata_unverified"] == 1


def test_mixed_is_its_own_category_and_is_not_double_counted() -> None:
    rollup = funding_rollup([_study(category="mixed")])
    assert rollup["by_category"]["mixed"] == 1
    assert rollup["by_category"]["industry"] == 0
    assert rollup["by_category"]["government"] == 0


# --------------------------------------------------------------------------- #
# Bias — per-study ratings, and what GRADE actually downgraded for
# --------------------------------------------------------------------------- #
def test_the_bias_rollup_counts_ratings_and_grade_downgrades() -> None:
    record = {
        "schema_version": 1,
        "level": "low",
        "downgrades": [
            {"domain": "imprecision", "steps": 2, "reason": "r",
             "locator": {"fields": ["reason"], "source_artifact_id": "a", "char_offset": 0, "quoted_span": "x"}},
            {"domain": "risk_of_bias", "steps": 1, "reason": "r",
             "locator": {"fields": ["reason"], "source_artifact_id": "a", "char_offset": 0, "quoted_span": "x"}},
        ],
        "upgrades": [],
    }
    studies = [_study(bias="low"), _study(bias="high"), _study(bias="not a rating")]
    rollup = bias_rollup(studies, record)

    assert rollup["risk_of_bias"] == {"low": 1, "some concerns": 0, "high": 1, "unclear": 1}
    assert rollup["grade_downgrades"] == {"imprecision": 2, "risk_of_bias": 1}
    assert rollup["grade_downgrade_steps"] == 3


def test_no_grade_record_means_no_downgrades_not_a_crash() -> None:
    rollup = bias_rollup([_study()], None)
    assert rollup["grade_downgrades"] == {}
    assert rollup["grade_downgrade_steps"] == 0


# --------------------------------------------------------------------------- #
# Design
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("design,rung", [("parallel RCT", "rct"), ("prospective cohort", "cohort")])
def test_the_design_rollup_counts_rungs(design: str, rung: str) -> None:
    answer = design_rollup([_study(design=design), _study(design=design)])
    assert answer[rung] == 2
    assert answer["highest_rung"] == rung
    assert answer["studies_total"] == 2


def test_an_unreadable_design_is_counted_as_unreadable() -> None:
    answer = design_rollup([_study(design="parallel RCT"), _study(design="???")])
    assert answer["unreadable"] == 1
    assert answer["highest_rung"] == "rct", "one unreadable design does not lower the base"


# --------------------------------------------------------------------------- #
# What the SOT says out loud
# --------------------------------------------------------------------------- #
# PLAN.md Step 9b item 3: §4.1's per-study Funding and Bias Risk columns have always been there;
# "14 of 20 industry-funded, 5 undisclosed" is a different fact, and it is the one steps 5 and 8 ask
# for. Rendered from the same functions the step pack projects, so document and projection cannot
# disagree about what they counted.
def test_the_sot_states_the_aggregates_with_their_denominators() -> None:
    from dr2_podcast.pipeline_sot import _format_rollups

    studies = [
        _study(group="Tanaka H; Osaka", registration="NCT01", category="industry"),
        _study(group="Smith J; Yale", registration="NCT02", category="undisclosed", bias="high"),
    ]
    rendered = _format_rollups(studies, None)

    assert "n=2" in rendered
    assert "industry 1" in rendered and "undisclosed 1" in rendered
    assert "undisclosed 1, unknown 0" in rendered, "the two states are stated apart"
    assert "1 reproduced by two or more independent groups" in rendered
    assert "low 1, some concerns 0, high 1, unclear 0" in rendered


def test_the_sot_names_the_unverifiable_share_of_its_funding_rollup() -> None:
    from dr2_podcast.pipeline_sot import _format_rollups

    rendered = _format_rollups([_study(group="A", registration="NCT01")], None)
    assert "API-sourced and unverifiable against the paper 1" in rendered


def test_the_sot_spells_out_what_grade_downgraded_for() -> None:
    from dr2_podcast.pipeline_sot import _format_rollups

    record = {
        "schema_version": 1, "level": "low", "upgrades": [],
        "downgrades": [{"domain": "imprecision", "steps": 2, "reason": "r",
                        "locator": {"fields": ["reason"], "source_artifact_id": "a",
                                    "char_offset": 0, "quoted_span": "x"}}],
    }
    rendered = _format_rollups([_study(group="A", registration="NCT01")], record)
    assert "GRADE downgraded for: imprecision" in rendered


def test_no_studies_means_no_rollup_block_at_all() -> None:
    """Rather than a section of zeroes, which reads as a measured result."""
    from dr2_podcast.pipeline_sot import _format_rollups

    assert _format_rollups([], None) == ""
