"""Mutation matrix for the four schema artifacts.

PLAN.md, "Where the loop stands": rounds 4 and 5 of the Codex plan review found *only* missing
fields in schemas written as Markdown prose, and the stated terminal move was to make a missing
field a test failure rather than a reviewer's catch. This file is that test. Every case below
that carries a `# iteration N` comment is a defect a human reviewer actually had to find.

Structure: for each schema, the canonical instance must PASS, and one mutation per promised
failure class must be REJECTED. A mutation matrix, not a spot check — a validator that rejects
everything passes a spot check.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from dr2_podcast.schemas import (
    CONFIDENCE_LADDER,
    EXAMPLE_NAMES,
    SCHEMA_NAMES,
    SchemaValidationError,
    compute_finding_key,
    example_path,
    finding_errors,
    funding_errors,
    grade_errors,
    load_example,
    load_schema,
    net_direction,
    ordinal_monotonicity_errors,
    schema_path,
    step_pack_errors,
    validate_finding,
    validate_grade,
    verify_locator_span,
)

Mutation = Callable[[dict[str, Any]], None]


def _mutated(name: str, mutation: Mutation) -> dict[str, Any]:
    """A fresh copy of the canonical instance with one defect injected."""
    instance = load_example(name)
    mutation(instance)
    return instance


def _artifact_text(locators: list[dict[str, Any]]) -> dict[str, str]:
    """Synthesise source artifacts that literally contain every span at its declared offset."""
    by_artifact: dict[str, list[dict[str, Any]]] = {}
    for locator in locators:
        by_artifact.setdefault(locator["source_artifact_id"], []).append(locator)
    artifacts: dict[str, str] = {}
    for artifact_id, entries in by_artifact.items():
        size = max(entry["char_offset"] + len(entry["quoted_span"]) for entry in entries)
        buffer = ["."] * size
        for entry in entries:
            for index, char in enumerate(entry["quoted_span"]):
                buffer[entry["char_offset"] + index] = char
        artifacts[artifact_id] = "".join(buffer)
    return artifacts


# --------------------------------------------------------------------------- #
# The schema files themselves
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_schema_file_is_itself_a_valid_2020_12_schema(name: str) -> None:
    schema = load_schema(name)
    Draft202012Validator.check_schema(schema)
    assert schema["$id"].endswith(f"/{name}.schema.json")
    assert schema["description"], "every schema states WHY it is shaped the way it is"


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_canonical_example_round_trips_on_disk(name: str) -> None:
    """File -> parse -> dump -> parse is identity, and the parsed instance validates clean."""
    raw = example_path(name).read_text(encoding="utf-8")
    instance = json.loads(raw)
    assert json.loads(json.dumps(instance, ensure_ascii=False)) == instance
    assert VALIDATORS[name](instance) == []


def test_schema_and_example_files_are_utf8_and_newline_terminated() -> None:
    for path in [schema_path(n) for n in SCHEMA_NAMES] + [example_path(n) for n in EXAMPLE_NAMES]:
        assert path.read_bytes().endswith(b"\n"), f"{path.name} must end with a newline"


# --------------------------------------------------------------------------- #
# finding_key
# --------------------------------------------------------------------------- #
def test_finding_key_is_stable_under_normalisation() -> None:
    """Full-width vs half-width, case and whitespace must not split one finding into two."""
    base = load_example("finding")
    variant = dict(base)
    variant["intervention"] = "ビタミンD  800 ＩＵ/日"  # doubled space + full-width IU
    variant["comparator"] = "プラセボ "
    assert compute_finding_key(variant) == compute_finding_key(base)


def test_finding_key_separates_endpoints_of_the_same_paper() -> None:
    """A paper is not a finding: benefit on one endpoint and a null result on another are two keys."""
    base = load_example("finding")
    other = dict(base, endpoint="転倒")
    assert compute_finding_key(other) != compute_finding_key(base)


def test_null_timepoint_is_the_same_identity_as_empty() -> None:
    base = load_example("finding")
    assert compute_finding_key(dict(base, timepoint=None)) == compute_finding_key(dict(base, timepoint=""))


# --------------------------------------------------------------------------- #
# Findings — must pass
# --------------------------------------------------------------------------- #
def test_canonical_finding_passes_including_span_verification() -> None:
    finding = load_example("finding")
    artifacts = _artifact_text(finding["locators"])
    assert finding_errors(finding, artifacts) == []


def test_finding_with_only_identity_and_direction_passes() -> None:
    """Coverage is required for claim-bearing fields that are PRESENT, not for every field."""
    finding = load_example("finding")
    for field in ("value", "unit", "ci_low", "ci_high", "p_value"):
        finding[field] = None
    for field in ("control_event_rate", "experimental_event_rate", "outcome_is_adverse"):
        finding[field] = None
    finding["locators"] = [
        {
            "fields": ["population", "intervention", "comparator", "endpoint", "timepoint", "direction"],
            "source_artifact_id": "pmid:12345678#abstract",
            "char_offset": 0,
            "quoted_span": "no numeric result was reported for this endpoint",
        }
    ]
    assert finding_errors(finding) == []


def test_null_timepoint_needs_no_locator() -> None:
    finding = load_example("finding")
    finding["timepoint"] = None
    finding["finding_key"] = compute_finding_key(finding)
    finding["locators"][0]["fields"] = ["population", "intervention", "comparator", "endpoint"]
    finding["locators"][1]["fields"] = ["direction", "value", "unit", "ci_low", "ci_high", "p_value"]
    assert finding_errors(finding) == []


# --------------------------------------------------------------------------- #
# Findings — mutation matrix
# --------------------------------------------------------------------------- #
def _drop_field_from_coverage(finding: dict[str, Any], field: str) -> None:
    for locator in finding["locators"]:
        if field in locator["fields"]:
            locator["fields"] = [f for f in locator["fields"] if f != field]
            if not locator["fields"]:
                locator["fields"] = ["endpoint"]


FINDING_MUTATIONS: list[tuple[str, Mutation, str]] = [
    (
        "model_authored_key",
        lambda f: f.__setitem__("finding_key", "0" * 40),
        "finding_key",
    ),
    (
        "identity_changed_without_rekey",
        lambda f: f.__setitem__("endpoint", "転倒"),
        "finding_key",
    ),
    (
        "uncovered_p_value",
        lambda f: _drop_field_from_coverage(f, "p_value"),
        "no locator names 'p_value'",
    ),
    # iteration 5, finding 2 — intervention and comparator are two of the five finding_key inputs,
    # so an unsourced arm corrupts finding identity and therefore replication grouping.
    (
        "uncovered_intervention",
        lambda f: _drop_field_from_coverage(f, "intervention"),
        "no locator names 'intervention'",
    ),
    (
        "uncovered_comparator",
        lambda f: _drop_field_from_coverage(f, "comparator"),
        "no locator names 'comparator'",
    ),
    (
        # p_value goes null while a locator still claims to source it.
        "locator_names_a_null_field",
        lambda f: f.__setitem__("p_value", None),
        "is null on this finding",
    ),
    (
        "locator_names_a_non_claim_field",
        lambda f: f["locators"][0]["fields"].append("title"),
        "not a claim-bearing field",
    ),
    (
        "cer_without_eer",
        lambda f: f.__setitem__("experimental_event_rate", None),
        "must both be present or both null",
    ),
    # iteration 5, finding 1 — clinical_math.py:39 reads a missing flag as "adverse" and would
    # report a directionally wrong NNT for a beneficial endpoint.
    (
        "rates_without_polarity",
        lambda f: f.__setitem__("outcome_is_adverse", None),
        "outcome_is_adverse",
    ),
    (
        "inverted_confidence_interval",
        lambda f: f.__setitem__("ci_low", 9.0),
        "greater than ci_high",
    ),
    (
        "half_a_confidence_interval",
        lambda f: f.__setitem__("ci_high", None),
        "ci_low and ci_high",
    ),
    (
        "impossible_p_value",
        lambda f: f.__setitem__("p_value", 1.5),
        "1.5",
    ),
    (
        "direction_outside_enum",
        lambda f: f.__setitem__("direction", "improved"),
        "improved",
    ),
    (
        "unknown_property",
        lambda f: f.__setitem__("conclusion", "vitamin D works"),
        "conclusion",
    ),
    (
        "no_locators_at_all",
        lambda f: f.__setitem__("locators", []),
        "locators",
    ),
    (
        "missing_required_identity_field",
        lambda f: f.pop("comparator"),
        "comparator",
    ),
]


@pytest.mark.parametrize(
    ("case", "mutation", "expected_fragment"),
    FINDING_MUTATIONS,
    ids=[case for case, _, _ in FINDING_MUTATIONS],
)
def test_finding_mutation_is_rejected(case: str, mutation: Mutation, expected_fragment: str) -> None:
    errors = finding_errors(_mutated("finding", mutation))
    assert errors, f"{case} was accepted"
    assert any(expected_fragment in error for error in errors), errors


def test_span_that_is_not_in_the_artifact_is_rejected() -> None:
    finding = load_example("finding")
    artifacts = _artifact_text(finding["locators"])
    finding["locators"][1]["quoted_span"] = "absolute risk reduction 50.0% (95% CI 2.0 to 8.0), p=0.03"
    errors = finding_errors(finding, artifacts)
    assert any("not a literal substring" in error for error in errors), errors


def test_span_against_an_unknown_artifact_is_rejected() -> None:
    finding = load_example("finding")
    errors = finding_errors(finding, {"pmid:99999999#fulltext": "unrelated text"})
    assert all("unknown artifact" in error for error in errors)
    assert len(errors) == len(finding["locators"])


def test_span_offset_shifted_by_one_is_rejected() -> None:
    finding = load_example("finding")
    artifacts = _artifact_text(finding["locators"])
    finding["locators"][0]["char_offset"] += 1
    assert any("not a literal substring" in error for error in finding_errors(finding, artifacts))


def test_verify_locator_span_is_offset_sensitive() -> None:
    locator = {"fields": ["endpoint"], "source_artifact_id": "a", "char_offset": 5, "quoted_span": "beta"}
    assert verify_locator_span(locator, "alphabetagamma")
    assert not verify_locator_span(locator, "alphaXbetagamma")


def test_validate_finding_raises_and_carries_every_error() -> None:
    finding = load_example("finding")
    finding["finding_key"] = "0" * 40
    finding["ci_low"] = 9.0
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_finding(finding)
    assert len(excinfo.value.errors) == 2
    assert excinfo.value.artifact == "finding"


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
    assert funding_errors(block) == [], case


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


# Each mutation changes exactly ONE axis away from a legal row, so the case cannot pass for a
# reason other than the one it names.
FUNDING_MUTATIONS: list[tuple[str, Mutation]] = [
    # iteration 5, finding 3 — without the category constraint, undisclosed+industry is legal and
    # silently poisons the step-5 aggregate.
    (
        "undisclosed_but_industry",
        lambda b: (_silent_paper(b), b.__setitem__("funding_category", "industry"))[1],
    ),
    (
        "silent_paper_that_somehow_quotes_a_funder",
        lambda b: (_silent_paper(b), b.__setitem__("funding_source_type", "extracted_text"))[1],
    ),
    ("unknown_but_raw_text_present", _unknown_with_raw_text),
    ("extracted_text_without_locator", lambda b: b.__setitem__("funding_locator", None)),
    ("api_metadata_with_a_locator", lambda b: b.__setitem__("funding_source_type", "api_metadata")),
    ("disclosed_but_category_unknown", lambda b: b.__setitem__("funding_category", "unknown")),
    ("category_outside_enum", lambda b: b.__setitem__("funding_category", "charity")),
    ("missing_disclosure_field", lambda b: b.pop("funding_disclosure")),
    ("unknown_property", lambda b: b.__setitem__("funding_source", "NIA")),
]


@pytest.mark.parametrize(
    ("case", "mutation"), FUNDING_MUTATIONS, ids=[case for case, _ in FUNDING_MUTATIONS]
)
def test_illegal_funding_combination_is_rejected(case: str, mutation: Mutation) -> None:
    assert funding_errors(_mutated("funding", mutation)), f"{case} was accepted"


def test_funding_locator_must_source_funding_raw() -> None:
    block = load_example("funding")
    block["funding_locator"]["fields"] = ["funding_category"]
    errors = funding_errors(block)
    assert any("must name 'funding_raw'" in error for error in errors), errors


def test_illegal_combination_error_points_at_the_table() -> None:
    block = load_example("funding")
    block["funding_disclosure"] = "undisclosed"
    assert any("oneOf table in funding.schema.json" in error for error in funding_errors(block))


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
    errors = grade_errors(_mutated("grade", mutation))
    assert errors, f"{case} was accepted"
    assert any(expected_fragment in error for error in errors), errors


def test_grade_accepts_a_derived_modifier_and_a_quoted_one() -> None:
    record = load_example("grade")
    assert grade_errors(record) == []
    assert record["downgrades"][1]["locator"]["kind"] == "derived"


def test_grade_with_no_modifiers_passes() -> None:
    assert grade_errors({"schema_version": 1, "level": "high", "downgrades": [], "upgrades": []}) == []


# --------------------------------------------------------------------------- #
# net_direction
# --------------------------------------------------------------------------- #
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
    def entry(domain: str, steps: int) -> dict[str, Any]:
        return {
            "domain": domain,
            "steps": steps,
            "reason": "…",
            "locator": {
                "fields": ["reason"],
                "source_artifact_id": "a",
                "char_offset": 0,
                "quoted_span": "x",
            },
        }

    record = {
        "schema_version": 1,
        "level": "moderate",
        "downgrades": [entry(d, s) for d, s in downgrades],
        "upgrades": [entry(u, s) for u, s in upgrades],
    }
    assert net_direction(record) == expected


def test_net_direction_fails_closed_on_an_invalid_record() -> None:
    with pytest.raises(SchemaValidationError):
        net_direction({"schema_version": 1, "level": "moderate", "downgrades": [], "upgrades": [{"domain": "x"}]})


def test_canonical_grade_example_nets_negative() -> None:
    assert net_direction(load_example("grade")) == -1


def test_validate_grade_raises() -> None:
    with pytest.raises(SchemaValidationError):
        validate_grade({"schema_version": 1, "level": "moderate"})


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
def test_canonical_step_pack_passes() -> None:
    assert step_pack_errors(load_example("step_pack")) == []


def test_step_pack_json_pointer_resolves_by_step_number_not_by_position() -> None:
    """The blueprint references 'step_pack.json#/steps/3'. Against an array that pointer resolves
    to the FOURTH element, and the step numbers skip 7 — so the mapping must be keyed, not listed."""
    pack = load_example("step_pack")
    assert isinstance(pack["steps"], dict)
    assert pack["steps"]["3"]["step"] == 3


STEP_PACK_MUTATIONS: list[tuple[str, Mutation, str]] = [
    ("missing_mandatory_step_9", lambda p: p["steps"].pop("9"), "9"),
    ("missing_mandatory_step_1", lambda p: p["steps"].pop("1"), "1"),
    (
        "dropped_step_7_reintroduced",
        lambda p: p["steps"].__setitem__("7", dict(p["steps"]["3"], step=7)),
        "7",
    ),
    (
        "key_disagrees_with_step_number",
        lambda p: p["steps"]["3"].__setitem__("step", 4),
        "stored under key",
    ),
    (
        "answer_without_provenance",
        lambda p: p["steps"]["3"].__setitem__("locators", []),
        "no provenance",
    ),
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
    errors = step_pack_errors(_mutated("step_pack", mutation))
    assert errors, f"{case} was accepted"
    assert any(expected_fragment in error for error in errors), errors


def test_absent_sufficiency_may_have_no_locators() -> None:
    """Where the absence IS the finding, the step still runs — and it has nothing to quote."""
    pack = load_example("step_pack")
    pack["steps"]["5"]["sufficiency"] = "absent"
    pack["steps"]["5"]["locators"] = []
    assert step_pack_errors(pack) == []


VALIDATORS: dict[str, Callable[[dict[str, Any]], list[str]]] = {
    "finding": finding_errors,
    "funding": funding_errors,
    "grade": grade_errors,
    "step_pack": step_pack_errors,
}
