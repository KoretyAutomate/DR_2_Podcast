"""The step pack — PLAN.md Step 9b. A projection of the SOT, computed and never authored.

The schema cannot check that an answer was derived rather than written: handed a finished pack, a
validator cannot tell a computed count from a plausible one. So the generator is what guarantees it,
and these tests are the guarantee — every answer here is one whose value is countable by hand from
the fixture, and the pack is regenerated from the same inputs and compared.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dr2_podcast.research.clinical import Finding, FundingBlock
from dr2_podcast.research.step_pack import QUESTIONS_JA, SOT_SECTIONS, build_step_pack
from dr2_podcast.schemas import step_pack_errors, validate_step_pack

SOT = (
    "# Source of Truth\n\n"
    "## 1. Abstract\nWhat this episode is about.\n\n"
    "## 2. Methods\nHow the search ran.\n\n"
    "### 3.3 Clinical Impact (Deterministic Math)\n| Study | CER |\n\n"
    "### 4.1 Study Characteristics\n| # | Study | Design |\n\n"
    "## 5. Discussion\nWhat it means.\n"
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


def _study(*, group="Tanaka H; Osaka", registration="NCT01", category="government",
           design="parallel RCT", bias="low", findings=None):
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


GRADE = {
    "schema_version": 1,
    "level": "moderate",
    "downgrades": [
        {"domain": "imprecision", "steps": 1, "reason": "wide interval",
         "locator": {"fields": ["reason"], "source_artifact_id": "a", "char_offset": 0, "quoted_span": "x"}}
    ],
    "upgrades": [],
}

PIPELINE_DATA = {
    "metrics": {
        "aff_wide_net_total": 40, "fal_wide_net_total": 20,
        "aff_screened_in": 10, "fal_screened_in": 5,
        "aff_fulltext_ok": 8, "fal_fulltext_ok": 4,
        "aff_fulltext_err": 2, "fal_fulltext_err": 1,
    },
    "impacts": [SimpleNamespace(direction="benefit"), SimpleNamespace(direction="no_effect")],
    "grade_record": GRADE,
    "aff_highest_tier": 1,
    "fal_highest_tier": 2,
}


def _pack(extractions=None, sot=SOT, domain="clinical", pipeline_data=None):
    return build_step_pack(
        pipeline_data=pipeline_data if pipeline_data is not None else PIPELINE_DATA,
        extractions=extractions if extractions is not None else [_study(), _study(group="Smith J; Yale",
                                                                                 registration="NCT02")],
        sot=sot,
        domain=domain,
    )


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #
def test_a_generated_pack_validates_against_the_sot_it_projects() -> None:
    validate_step_pack(_pack(), {"research/source_of_truth.md": SOT})


def test_every_step_key_agrees_with_the_step_it_stores() -> None:
    for key, step in _pack()["steps"].items():
        assert str(step["step"]) == key


def test_the_mandatory_steps_are_always_present() -> None:
    """Prior, numbers, update, verdict. A run that cannot compute one still carries it, marked
    absent — silence would read as 'not applicable' for a step that always applies."""
    steps = _pack(extractions=[])["steps"]
    for number in ("1", "4", "9", "10"):
        assert number in steps


def test_step_seven_is_absent_by_design() -> None:
    """This pipeline reads primary literature, not press coverage of it, so メディア歪曲チェック has
    no subject and no evidence artifact."""
    assert 7 not in QUESTIONS_JA
    assert "7" not in _pack()["steps"]


# --------------------------------------------------------------------------- #
# Derived, not authored: every answer is countable by hand
# --------------------------------------------------------------------------- #
def test_the_search_answer_is_the_sum_of_both_tracks() -> None:
    answer = _pack()["steps"]["2"]["answer"]
    assert answer["records_identified"] == 60
    assert answer["screened_in"] == 15
    assert answer["full_text_retrieved"] == 12
    assert answer["full_text_errors"] == 3


def test_the_design_answer_counts_the_studies() -> None:
    answer = _pack(extractions=[_study(design="parallel RCT"), _study(design="prospective cohort")])["steps"]["3"]
    assert answer["answer"]["rct"] == 1
    assert answer["answer"]["cohort"] == 1
    assert answer["answer"]["highest_rung"] == "rct"


def test_the_numbers_answer_counts_the_computed_impacts() -> None:
    answer = _pack()["steps"]["4"]["answer"]
    assert answer["findings_with_event_rates"] == 2
    assert answer["benefit"] == 1
    assert answer["no_effect"] == 1


def test_the_replication_answer_is_the_hand_countable_one() -> None:
    """Two groups, two cohorts, same endpoint and direction — one replicated finding."""
    answer = _pack()["steps"]["6"]["answer"]
    assert answer["findings_total"] == 1
    assert answer["findings_replicated"] == 1


def test_the_verdict_answer_comes_from_the_lookup_not_from_a_model() -> None:
    answer = _pack()["steps"]["10"]["answer"]
    assert answer["grade_level"] == "moderate"
    assert answer["confidence_ja"] == "高い", "moderate GRADE on an RCT base, uncapped"


def test_the_staircase_caps_the_verdict_the_pack_states() -> None:
    answer = _pack(extractions=[_study(design="cross-sectional survey")])["steps"]["10"]["answer"]
    assert answer["confidence_ja"] == "低い", "the same GRADE level, on a base that cannot carry it"


def test_regenerating_from_the_same_inputs_gives_the_same_pack() -> None:
    """The only honest check that an answer was derived: a written one would not survive this."""
    assert _pack() == _pack()


# --------------------------------------------------------------------------- #
# Sufficiency is load-bearing
# --------------------------------------------------------------------------- #
def test_a_step_with_nothing_to_project_is_absent_with_a_reason() -> None:
    step = _pack(extractions=[], pipeline_data={**PIPELINE_DATA, "impacts": []})["steps"]["4"]
    assert step["sufficiency"] == "absent"
    assert step["answer"]["unavailable"]
    assert step["locators"] == []


def test_partial_coverage_is_said_rather_than_rounded_up() -> None:
    """Funding unknown on half the studies is step 5's honest answer, and the episode says so."""
    studies = [_study(category="government"), _study(category="unknown")]
    step = _pack(extractions=studies)["steps"]["5"]
    assert step["sufficiency"] == "partial"
    assert step["answer"]["unknown"] == 1
    assert step["answer"]["studies_total"] == 2


def test_the_prior_step_is_absent_and_says_why() -> None:
    """It is Claude's to author at stage 1, and no stage is Claude-authored yet. Absent with the
    reason, never invented — a prior computed after the evidence is hindsight in costume."""
    step = _pack()["steps"]["1"]
    assert step["sufficiency"] == "absent"
    assert "prior" in step["answer"]["unavailable"]


def test_the_update_step_waits_for_the_prior() -> None:
    step = _pack()["steps"]["9"]
    assert step["sufficiency"] == "absent"
    assert "prior" in step["answer"]["unavailable"]


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def test_every_non_absent_step_carries_a_locator_into_the_sot() -> None:
    for key, step in _pack()["steps"].items():
        if step["sufficiency"] == "absent":
            continue
        assert step["locators"], key
        for locator in step["locators"]:
            offset, span = locator["char_offset"], locator["quoted_span"]
            assert SOT[offset : offset + len(span)] == span, key


def test_a_step_whose_section_is_missing_from_the_sot_reports_absent() -> None:
    """Rather than a locator quoting a heading that was never written, which would verify against
    nothing and pass a presence check."""
    thin = "# Source of Truth\n\n## 2. Methods\nonly this.\n"
    pack = _pack(sot=thin)
    assert pack["steps"]["3"]["sufficiency"] == "absent"
    assert pack["steps"]["2"]["sufficiency"] != "absent"
    assert step_pack_errors(pack, {"research/source_of_truth.md": thin}) == []


# --------------------------------------------------------------------------- #
# The two SOT builders number their sections differently
# --------------------------------------------------------------------------- #
def test_the_section_references_are_domain_aware() -> None:
    """PLAN.md names this trap explicitly: clinical is 4.x, social science is 3.x, and a
    domain-blind reference cites the wrong section of the wrong document."""
    assert SOT_SECTIONS["clinical"][3] != SOT_SECTIONS["social_science"][3]


def test_a_social_science_pack_records_which_builder_it_projects() -> None:
    social_sot = SOT.replace("### 4.1 Study Characteristics", "### 3.1 Study Characteristics")
    pack = _pack(sot=social_sot, domain="social_science")
    assert pack["sot_domain"] == "social_science"
    assert pack["steps"]["3"]["sot_sections"] == ["3.1"]
    validate_step_pack(pack, {"research/source_of_truth.md": social_sot})


@pytest.mark.parametrize("domain", ["clinical", "social_science"])
def test_every_question_has_a_section_mapping_in_every_domain(domain: str) -> None:
    """A step with no mapping silently gets no locator and reports absent, which would look like
    missing evidence rather than a missing table entry."""
    mapped = set(SOT_SECTIONS[domain]) | {1, 9}  # 1 and 9 wait on the prior
    assert set(QUESTIONS_JA) <= mapped
