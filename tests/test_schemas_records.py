"""Funding, the paper-level extraction record, structured GRADE, and the step pack.

Part of the schema mutation matrix; see test_schemas.py for what that means.
"""

from __future__ import annotations

from typing import Any

import pytest

from dr2_podcast.schemas import (
    CONFIDENCE_LADDER,
    SchemaValidationError,
    funding_errors,
    grade_errors,
    load_example,
    net_direction,
    ordinal_monotonicity_errors,
    schema_errors,
    validate_grade,
)

from tests._schema_fixtures import Mutation, _drop_field_from_coverage, _mutated, artifacts_for, errors_for


# --------------------------------------------------------------------------- #
# Funding — the legal-combination table
# --------------------------------------------------------------------------- #
LEGAL_FUNDING: list[tuple[str, dict[str, Any]]] = [
    ("disclosed_extracted", load_example("funding")),
    (
        "disclosed_api_metadata",
        {
            "funding_raw": "National Institute on Aging, Acme Pharma",
            "funding_category": "mixed",
            "funding_disclosure": "disclosed",
            "funding_source_type": "api_metadata",
            "funding_locator": None,
        },
    ),
    (
        "paper_is_silent",
        {
            "funding_raw": None,
            "funding_category": "undisclosed",
            "funding_disclosure": "undisclosed",
            "funding_source_type": "none",
            "funding_locator": None,
        },
    ),
    (
        "extraction_failed",
        {
            "funding_raw": None,
            "funding_category": "unknown",
            "funding_disclosure": "unknown",
            "funding_source_type": "none",
            "funding_locator": None,
        },
    ),
]


@pytest.mark.parametrize(("case", "block"), LEGAL_FUNDING, ids=[case for case, _ in LEGAL_FUNDING])
def test_legal_funding_combination_passes(case: str, block: dict[str, Any]) -> None:
    assert funding_errors(block, artifacts_for(block)) == [], case


def _silent_paper(block: dict[str, Any]) -> None:
    block.update(
        funding_raw=None,
        funding_locator=None,
        funding_disclosure="undisclosed",
        funding_source_type="none",
        funding_category="undisclosed",
    )


def _unknown_with_raw_text(block: dict[str, Any]) -> None:
    """Extraction failed, yet a funder string survived — nothing may attribute it."""
    _silent_paper(block)
    block.update(funding_disclosure="unknown", funding_category="unknown", funding_raw="NIA")


def _undisclosed_but_industry(block: dict[str, Any]) -> None:
    _silent_paper(block)
    block["funding_category"] = "industry"


def _silent_paper_that_quotes(block: dict[str, Any]) -> None:
    _silent_paper(block)
    block["funding_source_type"] = "extracted_text"


FUNDING_MUTATIONS: list[tuple[str, Mutation]] = [
    # iteration 5, finding 3 — without the category constraint, undisclosed+industry is legal and
    # silently poisons the step-5 aggregate.
    ("undisclosed_but_industry", _undisclosed_but_industry),
    ("silent_paper_that_somehow_quotes_a_funder", _silent_paper_that_quotes),
    ("unknown_but_raw_text_present", _unknown_with_raw_text),
    ("extracted_text_without_locator", lambda b: b.__setitem__("funding_locator", None)),
    ("api_metadata_with_a_locator", lambda b: b.__setitem__("funding_source_type", "api_metadata")),
    ("disclosed_but_category_unknown", lambda b: b.__setitem__("funding_category", "unknown")),
    ("category_outside_enum", lambda b: b.__setitem__("funding_category", "charity")),
    ("missing_disclosure_field", lambda b: b.pop("funding_disclosure")),
    ("unknown_property", lambda b: b.__setitem__("funding_source", "NIA")),
]


@pytest.mark.parametrize(("case", "mutation"), FUNDING_MUTATIONS, ids=[case for case, _ in FUNDING_MUTATIONS])
def test_illegal_funding_combination_is_rejected(case: str, mutation: Mutation) -> None:
    assert errors_for("funding", _mutated("funding", mutation)), f"{case} was accepted"


def test_funding_locator_must_source_funding_raw() -> None:
    block = load_example("funding")
    block["funding_locator"]["fields"] = ["funding_category"]
    errors = errors_for("funding", block)
    assert any("must name 'funding_raw'" in error for error in errors), errors


def test_illegal_combination_error_points_at_the_table() -> None:
    block = load_example("funding")
    block["funding_disclosure"] = "undisclosed"
    assert any("oneOf table in funding.schema.json" in error for error in errors_for("funding", block))


# --------------------------------------------------------------------------- #
# The paper-level extraction record
# --------------------------------------------------------------------------- #
# codex review 2026-08-12, finding 1: with no parent schema, `findings[]`, the funding block,
# `trial_registration` and `author_group` could all simply be absent with nothing to reject it.
def test_canonical_extraction_carries_two_findings_from_one_paper() -> None:
    """The 'a paper is not a finding' case, in the fixture itself: benefit on one endpoint, null on another."""
    extraction = load_example("extraction")
    assert len(extraction["findings"]) == 2
    assert {f["direction"] for f in extraction["findings"]} == {"decrease", "null_result"}


EXTRACTION_MUTATIONS: list[tuple[str, Mutation, str]] = [
    ("no_findings_at_all", lambda e: e.__setitem__("findings", []), "findings"),
    ("findings_key_missing", lambda e: e.pop("findings"), "findings"),
    ("trial_registration_missing", lambda e: e.pop("trial_registration"), "trial_registration"),
    ("author_group_missing", lambda e: e.pop("author_group"), "author_group"),
    ("funding_block_missing", lambda e: e.pop("funding"), "funding"),
    # The three fields 9a MOVES must not survive on the paper-level record.
    ("legacy_funding_source_field", lambda e: e.__setitem__("funding_source", "NIA"), "funding_source"),
    ("legacy_paper_level_cer", lambda e: e.__setitem__("control_event_rate", 0.15), "control_event_rate"),
    ("legacy_paper_level_polarity", lambda e: e.__setitem__("outcome_is_adverse", True), "outcome_is_adverse"),
    ("risk_of_bias_as_prose", lambda e: e.__setitem__("risk_of_bias", "probably fine"), "probably fine"),
    ("research_tier_out_of_range", lambda e: e.__setitem__("research_tier", 4), "research_tier"),
    (
        "duplicate_finding_key_in_one_paper",
        lambda e: e["findings"].__setitem__(1, dict(e["findings"][0])),
        "appears 2 times",
    ),
    (
        "nested_finding_with_broken_coverage",
        lambda e: _drop_field_from_coverage(e["findings"][0], "p_value"),
        "/findings/0/locators",
    ),
    (
        "nested_funding_illegal_combination",
        lambda e: e["funding"].__setitem__("funding_disclosure", "undisclosed"),
        "funding",
    ),
    (
        "nested_finding_rates_without_polarity",
        lambda e: e["findings"][0].__setitem__("outcome_is_adverse", None),
        "outcome_is_adverse",
    ),
]


@pytest.mark.parametrize(
    ("case", "mutation", "expected_fragment"),
    EXTRACTION_MUTATIONS,
    ids=[case for case, _, _ in EXTRACTION_MUTATIONS],
)
def test_extraction_mutation_is_rejected(case: str, mutation: Mutation, expected_fragment: str) -> None:
    errors = errors_for("extraction", _mutated("extraction", mutation))
    assert errors, f"{case} was accepted"
    assert any(expected_fragment in error for error in errors), errors


@pytest.mark.parametrize("value", ["low", "some concerns", "high", "unclear", "unknown"])
def test_every_prompt_constrained_risk_of_bias_value_is_accepted(value: str) -> None:
    extraction = load_example("extraction")
    extraction["risk_of_bias"] = value
    assert errors_for("extraction", extraction) == []


def test_an_observational_paper_may_have_no_trial_registration() -> None:
    extraction = load_example("extraction")
    extraction.update(trial_registration=None, study_design="prospective cohort", randomization_method=None)
    assert errors_for("extraction", extraction) == []


def test_paper_metadata_is_accepted_and_constrained() -> None:
    """codex review 2026-08-12, finding 2: every successfully built record carries it
    (clinical.py:2527) and to_dict() serialises it, so additionalProperties:false was rejecting
    real records. It is accepted — but as external API metadata, not as claim material."""
    extraction = load_example("extraction")
    assert errors_for("extraction", extraction) == []
    extraction["paper_metadata"]["citation_cnt"] = 3
    assert errors_for("extraction", extraction)


def test_paper_metadata_may_be_absent_or_null() -> None:
    extraction = load_example("extraction")
    extraction.pop("paper_metadata")
    assert errors_for("extraction", extraction) == []
    extraction["paper_metadata"] = None
    assert errors_for("extraction", extraction) == []


# --------------------------------------------------------------------------- #
# Migration guard: what today's DeepExtraction still has to change
# --------------------------------------------------------------------------- #
def _todays_extraction_dict() -> dict[str, Any]:
    from dr2_podcast.research.clinical import DeepExtraction, PaperMetadata

    return DeepExtraction(
        pmid="12345678",
        doi="10.1000/example.2026.001",
        title="Vitamin D supplementation and hip fracture",
        url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
        funding_source="National Institute on Aging",
        control_event_rate=0.15,
        experimental_event_rate=0.1,
        outcome_is_adverse=True,
        study_design="randomised controlled trial",
        risk_of_bias="some concerns",
        research_tier=1,
        raw_facts="…",
        paper_metadata=PaperMetadata(citation_count=214),
    ).to_dict()


def test_step_9a_added_every_field_the_contract_requires() -> None:
    """Inverted 2026-08-13, as its predecessor said to do. The four fields the contract required and
    DeepExtraction lacked — trial_registration, author_group, funding, findings — are all present."""
    errors = schema_errors("extraction", _todays_extraction_dict())
    for field in ("trial_registration", "author_group", "funding", "findings"):
        assert not any(f"'{field}' is a required property" in error for error in errors), field


def test_the_paper_level_effect_fields_are_what_step_9a_still_has_to_remove() -> None:
    """The remaining gap, kept as a checklist rather than a discovery.

    findings[] is populated and every consumer still reads the paper-level CER/EER/polarity, which
    slice 1 DERIVES from the primary finding so nothing breaks. They come out when clinical_math,
    _build_case and the two SOT renderers read findings[] instead — the second half of 9a. Until
    then the record does not satisfy the contract, and this says exactly why.
    """
    errors = schema_errors("extraction", _todays_extraction_dict())
    for field in ("funding_source", "control_event_rate", "experimental_event_rate", "outcome_is_adverse"):
        assert any(field in error and "not allowed" in error for error in errors), field


def test_to_dict_now_emits_explicit_nulls() -> None:
    """Inverted 2026-08-13. It used to drop every None, and absent cannot distinguish "we looked and
    the paper does not say" from "this producer version does not set the field"."""
    from dr2_podcast.research.clinical import DeepExtraction

    record = DeepExtraction(pmid="1", doi=None, title="t", url="u").to_dict()
    assert "doi" in record and record["doi"] is None
    assert record["findings"] == []
    assert record["funding"]["funding_disclosure"] == "unknown"
    assert not any("'doi' is a required property" in error for error in schema_errors("extraction", record))


# --------------------------------------------------------------------------- #
# Structured GRADE
# --------------------------------------------------------------------------- #
GRADE_MUTATIONS: list[tuple[str, Mutation, str]] = [
    # iteration 5, finding 4 — the formula sums every entry, so a repeated domain double-counts
    # and net_direction comes out wrong. "Deduplicated by construction" was unearned.
    (
        "repeated_downgrade_domain",
        lambda g: g["downgrades"].append(dict(g["downgrades"][0], reason="again")),
        "appears 2 times",
    ),
    (
        "repeated_upgrade_domain",
        lambda g: g["upgrades"].append(dict(g["upgrades"][0], reason="again")),
        "appears 2 times",
    ),
    ("three_step_downgrade", lambda g: g["downgrades"][0].__setitem__("steps", 3), "steps"),
    ("zero_step_downgrade", lambda g: g["downgrades"][0].__setitem__("steps", 0), "steps"),
    ("unknown_domain", lambda g: g["downgrades"][0].__setitem__("domain", "vibes"), "vibes"),
    (
        "upgrade_domain_used_as_downgrade",
        lambda g: g["downgrades"][0].__setitem__("domain", "large_effect"),
        "large_effect",
    ),
    # The permissive regex scrape at pipeline_sot.py:43 defaults to "Not Determined"; the
    # structured record must refuse it rather than carry it forward.
    ("not_determined_level", lambda g: g.__setitem__("level", "Not Determined"), "Not Determined"),
    ("missing_schema_version", lambda g: g.pop("schema_version"), "schema_version"),
    ("missing_locator", lambda g: g["upgrades"][0].pop("locator"), "locator"),
    ("empty_reason", lambda g: g["upgrades"][0].__setitem__("reason", ""), "reason"),
    (
        "derived_locator_without_inputs",
        lambda g: g["downgrades"][1]["locator"].__setitem__("inputs", []),
        "inputs",
    ),
    ("unknown_property", lambda g: g.__setitem__("verdict", "works"), "verdict"),
]


@pytest.mark.parametrize(
    ("case", "mutation", "expected_fragment"),
    GRADE_MUTATIONS,
    ids=[case for case, _, _ in GRADE_MUTATIONS],
)
def test_grade_mutation_is_rejected(case: str, mutation: Mutation, expected_fragment: str) -> None:
    errors = errors_for("grade", _mutated("grade", mutation))
    assert errors, f"{case} was accepted"
    assert any(expected_fragment in error for error in errors), errors


def test_grade_accepts_a_derived_modifier_and_a_quoted_one() -> None:
    record = load_example("grade")
    assert errors_for("grade", record) == []
    assert record["downgrades"][1]["locator"]["kind"] == "derived"


def test_grade_with_no_modifiers_passes() -> None:
    assert grade_errors({"schema_version": 1, "level": "high", "downgrades": [], "upgrades": []}, {}) == []


# --------------------------------------------------------------------------- #
# net_direction
# --------------------------------------------------------------------------- #
def _modifier(domain: str, steps: int) -> dict[str, Any]:
    return {
        "domain": domain,
        "steps": steps,
        "reason": "…",
        "locator": {"fields": ["reason"], "source_artifact_id": "a", "char_offset": 0, "quoted_span": "x"},
    }


@pytest.mark.parametrize(
    ("downgrades", "upgrades", "expected"),
    [
        ([], [], 0),
        ([("imprecision", 1)], [], -1),
        ([], [("large_effect", 2)], 1),
        ([("imprecision", 1), ("risk_of_bias", 1)], [("large_effect", 2)], 0),
        ([("imprecision", 2)], [("large_effect", 1)], -1),
    ],
)
def test_net_direction(downgrades: list[tuple[str, int]], upgrades: list[tuple[str, int]], expected: int) -> None:
    record = {
        "schema_version": 1,
        "level": "moderate",
        "downgrades": [_modifier(d, s) for d, s in downgrades],
        "upgrades": [_modifier(u, s) for u, s in upgrades],
    }
    assert net_direction(record) == expected


def test_net_direction_fails_closed_on_an_invalid_record() -> None:
    with pytest.raises(SchemaValidationError):
        net_direction({"schema_version": 1, "level": "moderate", "downgrades": [], "upgrades": [{"domain": "x"}]})


def test_net_direction_fails_closed_on_a_repeated_domain() -> None:
    """The sum is exactly what a duplicate corrupts, so this cannot be left to the caller."""
    record = {
        "schema_version": 1,
        "level": "low",
        "downgrades": [_modifier("imprecision", 1), _modifier("imprecision", 1)],
        "upgrades": [],
    }
    with pytest.raises(SchemaValidationError):
        net_direction(record)


def test_canonical_grade_example_nets_negative() -> None:
    assert net_direction(load_example("grade")) == -1


def test_validate_grade_raises() -> None:
    with pytest.raises(SchemaValidationError):
        validate_grade({"schema_version": 1, "level": "moderate"}, {})


# --------------------------------------------------------------------------- #
# Ordinal monotonicity
# --------------------------------------------------------------------------- #
def test_confidence_ladder_is_the_five_level_scale() -> None:
    assert CONFIDENCE_LADDER == ("まだ分からない", "低い", "中程度", "高い", "ほぼ確実")


@pytest.mark.parametrize(
    ("prior", "posterior", "net", "reason", "ok"),
    [
        ("中程度", "高い", 1, None, True),
        ("中程度", "中程度", 1, None, True),
        ("中程度", "低い", 1, None, False),
        ("中程度", "低い", -1, None, True),
        ("中程度", "高い", -1, None, False),
        ("中程度", "ほぼ確実", 0, None, True),
        ("まだ分からない", "ほぼ確実", 1, None, False),
        ("まだ分からない", "ほぼ確実", 1, "3件の大規模RCTが一致", True),
        ("中程度", "とても高い", 1, None, False),
    ],
)
def test_ordinal_monotonicity(prior: str, posterior: str, net: int, reason: str | None, ok: bool) -> None:
    errors = ordinal_monotonicity_errors(prior, posterior, net, reason)
    assert (errors == []) is ok, errors


# --------------------------------------------------------------------------- #
# Step pack
# --------------------------------------------------------------------------- #
def test_step_pack_json_pointer_resolves_by_step_number_not_by_position() -> None:
    """The blueprint references 'step_pack.json#/steps/3'. Against an array that pointer resolves
    to the FOURTH element, and the step numbers skip 7 — so the mapping must be keyed, not listed."""
    pack = load_example("step_pack")
    assert isinstance(pack["steps"], dict)
    assert pack["steps"]["3"]["step"] == 3


STEP_PACK_MUTATIONS: list[tuple[str, Mutation, str]] = [
    ("missing_mandatory_step_9", lambda p: p["steps"].pop("9"), "9"),
    ("missing_mandatory_step_1", lambda p: p["steps"].pop("1"), "1"),
    ("dropped_step_7_reintroduced", lambda p: p["steps"].__setitem__("7", dict(p["steps"]["3"], step=7)), "7"),
    ("key_disagrees_with_step_number", lambda p: p["steps"]["3"].__setitem__("step", 4), "stored under key"),
    ("answer_without_provenance", lambda p: p["steps"]["3"].__setitem__("locators", []), "no provenance"),
    ("empty_answer", lambda p: p["steps"]["3"].__setitem__("answer", {}), "answer"),
    ("bad_sufficiency", lambda p: p["steps"]["3"].__setitem__("sufficiency", "mostly"), "mostly"),
    (
        "verdict_contribution_outside_enum",
        lambda p: p["steps"]["4"].__setitem__("verdict_contribution", "strong"),
        "strong",
    ),
    ("missing_sot_domain", lambda p: p.pop("sot_domain"), "sot_domain"),
    ("bad_sot_domain", lambda p: p.__setitem__("sot_domain", "clinical_v2"), "clinical_v2"),
    ("unknown_property_in_step", lambda p: p["steps"]["4"].__setitem__("script", "…"), "script"),
    ("missing_schema_version", lambda p: p.pop("schema_version"), "schema_version"),
]


@pytest.mark.parametrize(
    ("case", "mutation", "expected_fragment"),
    STEP_PACK_MUTATIONS,
    ids=[case for case, _, _ in STEP_PACK_MUTATIONS],
)
def test_step_pack_mutation_is_rejected(case: str, mutation: Mutation, expected_fragment: str) -> None:
    errors = errors_for("step_pack", _mutated("step_pack", mutation))
    assert errors, f"{case} was accepted"
    assert any(expected_fragment in error for error in errors), errors


def test_absent_sufficiency_may_have_no_locators() -> None:
    """Where the absence IS the finding, the step still runs — and it has nothing to quote."""
    pack = load_example("step_pack")
    pack["steps"]["5"]["sufficiency"] = "absent"
    pack["steps"]["5"]["locators"] = []
    assert errors_for("step_pack", pack) == []
