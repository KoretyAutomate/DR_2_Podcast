"""Mutation matrix for the schema artifacts.

PLAN.md, "Where the loop stands": rounds 4 and 5 of the Codex plan review found *only* missing
fields in schemas written as Markdown prose, and the stated terminal move was to make a missing
field a test failure rather than a reviewer's catch. This file is that test. Every case below
that carries an `# iteration N` or `# codex review` comment is a defect a reviewer actually had
to find.

Structure: for each schema, the canonical instance must PASS, and one mutation per promised
failure class must be REJECTED. A mutation matrix, not a spot check — a validator that rejects
everything passes a spot check. Each mutation moves exactly one axis away from a legal instance,
so a case cannot pass for a reason other than the one it names.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from dr2_podcast.research.clinical_math import calculate_impact
from dr2_podcast.research.effect_size_math import d_to_r, hedges_g_correction, odds_ratio_to_d, r_to_d
from dr2_podcast.schemas import (
    CONFIDENCE_LADDER,
    DERIVED_OPERATIONS,
    EXAMPLE_NAMES,
    SCHEMA_NAMES,
    SchemaValidationError,
    agrees_at_producer_precision,
    compute_finding_key,
    example_path,
    extraction_errors,
    finding_errors,
    funding_errors,
    grade_errors,
    iter_locators,
    load_example,
    load_schema,
    net_direction,
    ordinal_monotonicity_errors,
    recompute_derived,
    schema_errors,
    schema_path,
    span_errors,
    step_pack_errors,
    validate_finding,
    validate_grade,
    verify_locator_span,
)

Mutation = Callable[[dict[str, Any]], None]


def artifacts_for(instance: Any) -> dict[str, str]:
    """Synthesise source artifacts that literally contain every span at its declared offset.

    Real runs pass the fetched documents. The tests need the *provenance* check to pass so that a
    mutation elsewhere is what fails; the span-specific cases below break these deliberately.
    """
    by_artifact: dict[str, list[dict[str, Any]]] = {}
    for _, locator in iter_locators(instance):
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


def errors_for(name: str, instance: dict[str, Any]) -> list[str]:
    """Full validation of an instance against artifacts built to satisfy its own locators."""
    return VALIDATORS[name](instance, artifacts_for(instance))


def _mutated(name: str, mutation: Mutation) -> dict[str, Any]:
    """A fresh copy of the canonical instance with one defect injected."""
    instance = load_example(name)
    mutation(instance)
    return instance


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
    assert errors_for(name, instance) == []


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
    assert errors_for("finding", finding) == []


def test_null_timepoint_needs_no_locator() -> None:
    finding = load_example("finding")
    finding["timepoint"] = None
    finding["finding_key"] = compute_finding_key(finding)
    finding["locators"][0]["fields"] = ["population", "intervention", "comparator", "endpoint"]
    finding["locators"][1]["fields"] = ["direction", "value", "unit", "ci_low", "ci_high", "p_value"]
    assert errors_for("finding", finding) == []


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
    ("model_authored_key", lambda f: f.__setitem__("finding_key", "0" * 40), "finding_key"),
    ("identity_changed_without_rekey", lambda f: f.__setitem__("endpoint", "転倒"), "finding_key"),
    ("uncovered_p_value", lambda f: _drop_field_from_coverage(f, "p_value"), "no locator names 'p_value'"),
    # iteration 5, finding 2 — intervention and comparator are two of the five finding_key inputs,
    # so an unsourced arm corrupts finding identity and therefore replication grouping.
    (
        "uncovered_intervention",
        lambda f: _drop_field_from_coverage(f, "intervention"),
        "no locator names 'intervention'",
    ),
    ("uncovered_comparator", lambda f: _drop_field_from_coverage(f, "comparator"), "no locator names 'comparator'"),
    # p_value goes null while a locator still claims to source it.
    ("locator_names_a_null_field", lambda f: f.__setitem__("p_value", None), "is null on this finding"),
    (
        "locator_names_a_non_claim_field",
        lambda f: f["locators"][0]["fields"].append("title"),
        "not a claim-bearing field",
    ),
    ("cer_without_eer", lambda f: f.__setitem__("experimental_event_rate", None), "must both be present or both null"),
    # iteration 5, finding 1 — clinical_math.py:39 reads a missing flag as "adverse" and would
    # report a directionally wrong NNT for a beneficial endpoint.
    ("rates_without_polarity", lambda f: f.__setitem__("outcome_is_adverse", None), "outcome_is_adverse"),
    ("inverted_confidence_interval", lambda f: f.__setitem__("ci_low", 9.0), "greater than ci_high"),
    ("half_a_confidence_interval", lambda f: f.__setitem__("ci_high", None), "ci_low and ci_high"),
    ("impossible_p_value", lambda f: f.__setitem__("p_value", 1.5), "1.5"),
    ("direction_outside_enum", lambda f: f.__setitem__("direction", "improved"), "improved"),
    ("unknown_property", lambda f: f.__setitem__("conclusion", "vitamin D works"), "conclusion"),
    ("no_locators_at_all", lambda f: f.__setitem__("locators", []), "locators"),
    ("missing_required_identity_field", lambda f: f.pop("comparator"), "comparator"),
]


@pytest.mark.parametrize(
    ("case", "mutation", "expected_fragment"),
    FINDING_MUTATIONS,
    ids=[case for case, _, _ in FINDING_MUTATIONS],
)
def test_finding_mutation_is_rejected(case: str, mutation: Mutation, expected_fragment: str) -> None:
    errors = errors_for("finding", _mutated("finding", mutation))
    assert errors, f"{case} was accepted"
    assert any(expected_fragment in error for error in errors), errors


# --------------------------------------------------------------------------- #
# Locator spans — the provenance check, on every schema that can carry one
# --------------------------------------------------------------------------- #
# codex review 2026-08-12, finding 2: span verification used to be opt-in (`artifacts=None`) and
# funding/GRADE/step-pack locators were never resolved at all, so fabricated provenance passed.
SPAN_TARGETS: list[tuple[str, Callable[[dict[str, Any]], dict[str, Any]]]] = [
    ("finding", lambda i: i["locators"][1]),
    ("funding", lambda i: i["funding_locator"]),
    ("extraction", lambda i: i["findings"][0]["locators"][0]),
    ("grade", lambda i: i["downgrades"][0]["locator"]),
    ("grade_derived_input", lambda i: i["downgrades"][1]["locator"]["operands"]["ci_low"]["quoted"]),
    ("step_pack", lambda i: i["steps"]["4"]["locators"][0]),
]


@pytest.mark.parametrize(("case", "pick"), SPAN_TARGETS, ids=[case for case, _ in SPAN_TARGETS])
def test_a_span_absent_from_its_artifact_is_rejected(
    case: str, pick: Callable[[dict[str, Any]], dict[str, Any]]
) -> None:
    name = case.split("_derived")[0]
    instance = load_example(name)
    artifacts = artifacts_for(instance)
    pick(instance)["char_offset"] += 3
    errors = VALIDATORS[name](instance, artifacts)
    assert any("not a literal substring" in error for error in errors), errors


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_validation_against_no_artifacts_at_all_fails(name: str) -> None:
    """An empty artifact map is not a free pass — every locator becomes unresolvable."""
    instance = load_example(name)
    errors = VALIDATORS[name](instance, {})
    assert errors, f"{name} validated with nothing to check its provenance against"
    assert all("unknown artifact" in error for error in errors), errors


def test_a_rewritten_quote_is_rejected() -> None:
    finding = load_example("finding")
    artifacts = artifacts_for(finding)
    finding["locators"][1]["quoted_span"] = "absolute risk reduction 50.0% (95% CI 2.0 to 8.0), p=0.03"
    assert any("not a literal substring" in error for error in finding_errors(finding, artifacts))


def test_verify_locator_span_is_offset_sensitive() -> None:
    locator = {"fields": ["endpoint"], "source_artifact_id": "a", "char_offset": 5, "quoted_span": "beta"}
    assert verify_locator_span(locator, "alphabetagamma")
    assert not verify_locator_span(locator, "alphaXbetagamma")


def test_iter_locators_reaches_every_nesting_depth() -> None:
    """Including the locators quoting a derived value's operands, four levels down."""
    pointers = [pointer for pointer, _ in iter_locators(load_example("grade"))]
    assert "/downgrades/0/locator" in pointers
    assert "/downgrades/1/locator/operands/ci_low/quoted" in pointers
    assert "/downgrades/1/locator/operands/ci_high/quoted" in pointers
    assert "/upgrades/0/locator" in pointers


def test_span_errors_is_usable_standalone() -> None:
    assert span_errors(load_example("grade"), artifacts_for(load_example("grade"))) == []


def test_validate_finding_raises_and_carries_every_error() -> None:
    finding = load_example("finding")
    artifacts = artifacts_for(finding)
    finding["finding_key"] = "0" * 40
    finding["ci_low"] = 9.0
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_finding(finding, artifacts)
    assert len(excinfo.value.errors) == 2
    assert excinfo.value.artifact == "finding"


# --------------------------------------------------------------------------- #
# Derived values — recomputation, not shape
# --------------------------------------------------------------------------- #
def _quoted(field: str) -> dict[str, Any]:
    return {"fields": [field], "source_artifact_id": "a", "char_offset": 0, "quoted_span": "x"}


def _derived(
    operation: str,
    values: dict[str, float],
    result: Any,
    *,
    constants: tuple[str, ...] = (),
    computed: dict[str, dict[str, Any]] | None = None,
    unsourced: tuple[str, ...] = (),
) -> dict[str, Any]:
    computed = computed or {}
    operands: dict[str, Any] = {}
    for name, value in values.items():
        if name in constants or name in unsourced:
            operands[name] = {"value": value, "constant": True}
        elif name in computed:
            operands[name] = {"value": value, "computed": computed[name]}
        else:
            operands[name] = {"value": value, "quoted": _quoted(name)}
    return {"kind": "derived", "operation": operation, "operands": operands, "result": result}


def _misquoted(
    operation: str, values: dict[str, float], result: Any, operand: str, fields: list[str]
) -> dict[str, Any]:
    """A record whose `operand` is quoted by a locator naming `fields` instead of itself."""
    record = _derived(operation, values, result)
    record["operands"][operand]["quoted"]["fields"] = fields
    return record


def _correct_derived_records() -> list[tuple[str, dict[str, Any]]]:
    """One record per operation, with the result taken from the PRODUCTION calculator.

    Hardcoding the expected numbers here would only prove this file agrees with itself. Taking
    them from `clinical_math` / `effect_size_math` pins the schema's recomputation to the
    functions that actually produce these values — if either drifts, this fails.
    """
    return [
        ("difference", _derived("difference", {"minuend": 0.15, "subtrahend": 0.1}, 0.05)),
        ("negate", _derived("negate", {"value": 0.05}, -0.05)),
        ("ratio", _derived("ratio", {"numerator": 0.05, "denominator": 0.15}, 0.3333333333333333)),
        ("reciprocal_abs", _derived("reciprocal_abs", {"value": 0.05}, 20.0)),
        ("hedges_g", _derived("hedges_g", {"cohens_d": 0.5, "sample_size": 20}, hedges_g_correction(0.5, 20))),
        (
            "hedges_g_below_the_correction_floor",
            _derived("hedges_g", {"cohens_d": 0.5, "sample_size": 3}, hedges_g_correction(0.5, 3)),
        ),
        ("odds_ratio_to_d", _derived("odds_ratio_to_d", {"odds_ratio": 2.0}, odds_ratio_to_d(2.0))),
        ("r_to_d", _derived("r_to_d", {"r": 0.3}, r_to_d(0.3))),
        ("d_to_r", _derived("d_to_r", {"cohens_d": 0.5}, d_to_r(0.5))),
        (
            "ci_includes_null",
            _derived(
                "ci_includes_null", {"ci_low": -0.4, "ci_high": 2.8, "null_value": 0}, True, constants=("null_value",)
            ),
        ),
        (
            "ci_excludes_null",
            _derived(
                "ci_excludes_null", {"ci_low": 2.0, "ci_high": 8.0, "null_value": 0}, True, constants=("null_value",)
            ),
        ),
    ]


@pytest.mark.parametrize(
    ("case", "record"), _correct_derived_records(), ids=[case for case, _ in _correct_derived_records()]
)
def test_correct_derived_value_passes(case: str, record: dict[str, Any]) -> None:
    assert recompute_derived(record) == [], case


def test_every_operation_has_a_correct_case() -> None:
    """A closed enum with an untested member is a contract nobody has checked."""
    assert {case.split("_below")[0] for case, _ in _correct_derived_records()} == set(DERIVED_OPERATIONS)


# codex review 2026-08-12, finding 1 (second round): operands could only be quoted from a paper,
# which cannot express the pipeline's own arithmetic — RRR consumes a computed ARR, and NNT
# consumes 1/|ARR|. Neither has a span in any paper.
def test_the_real_calculate_impact_chain_validates_end_to_end() -> None:
    impact = calculate_impact("pmid:12345678", cer=0.15, eer=0.1, outcome_is_adverse=True)
    assert impact is not None
    arr = _derived("difference", {"minuend": 0.15, "subtrahend": 0.1}, impact.arr)
    rrr = _derived(
        "ratio",
        {"numerator": impact.arr, "denominator": 0.15},
        impact.rrr,
        computed={"numerator": arr},
    )
    nnt = _derived("reciprocal_abs", {"value": impact.arr}, impact.nnt, computed={"value": arr})
    for record in (arr, rrr, nnt):
        assert schema_errors("derived", record) == [], record
        assert recompute_derived(record) == [], record


def test_a_rounded_result_is_accepted_but_a_wrong_one_is_not() -> None:
    """calculate_impact rounds RRR to 4 decimals and NNT to 1; full-precision equality would
    reject its own correct output. The tolerance is the PRODUCER's, per operation."""
    assert agrees_at_producer_precision("ratio", 1 / 3, 0.3333)
    assert agrees_at_producer_precision("reciprocal_abs", 20.04, 20.0)
    assert not agrees_at_producer_precision("ratio", 1 / 3, 0.3433)
    assert not agrees_at_producer_precision("difference", 0.05, 0.5)


# codex review 2026-08-12, finding 1 (third round): inferring the tolerance from how many decimals
# the STATED value was written at let a record buy its own tolerance — stating 0.0 bought ±0.05,
# so a recomputed 0.049 passed, and effect_size_math.py:137 calls anything above 0.01 a non-null
# direction. That is a flipped verdict slipping through the check whose job is to catch one.
@pytest.mark.parametrize(
    ("operation", "expected", "stated"),
    [
        ("negate", -0.049, 0.0),
        ("difference", 0.049, 0.0),
        ("d_to_r", 0.0123, 0.0),
        ("ratio", 0.004, 0.0),
    ],
)
def test_a_result_cannot_widen_its_own_tolerance_by_being_vague(
    operation: str, expected: float, stated: float
) -> None:
    assert not agrees_at_producer_precision(operation, expected, stated)


def test_the_near_zero_case_is_rejected_end_to_end() -> None:
    record = _derived("negate", {"value": 0.049}, 0.0)
    assert any("recomputed" in error for error in recompute_derived(record))


def test_a_computed_operand_must_equal_the_derivation_it_names() -> None:
    arr = _derived("difference", {"minuend": 0.15, "subtrahend": 0.1}, 0.05)
    nnt = _derived("reciprocal_abs", {"value": 0.02}, 50.0, computed={"value": arr})
    errors = recompute_derived(nnt)
    assert any("does not equal the result" in error for error in errors), errors


def test_a_defect_in_a_nested_derivation_is_caught() -> None:
    """The walk recomputes the operand's own derivation, not just the top-level one."""
    arr = _derived("difference", {"minuend": 0.15, "subtrahend": 0.1}, 0.02)
    nnt = _derived("reciprocal_abs", {"value": 0.02}, 50.0, computed={"value": arr})
    errors = recompute_derived(nnt)
    assert any(error.startswith("/operands/value/computed/result: states 0.02") for error in errors), errors


DERIVED_MUTATIONS: list[tuple[str, dict[str, Any], str]] = [
    ("wrong_arithmetic_result", _derived("difference", {"minuend": 0.15, "subtrahend": 0.1}, 0.5), "recomputed"),
    (
        "ci_verdict_inverted",
        _derived("ci_includes_null", {"ci_low": 2.0, "ci_high": 8.0, "null_value": 0}, True, constants=("null_value",)),
        "recomputed",
    ),
    (
        "boolean_where_a_number_belongs",
        _derived("difference", {"minuend": 0.15, "subtrahend": 0.1}, True),
        "yields float",
    ),
    ("division_by_zero", _derived("ratio", {"numerator": 0.05, "denominator": 0.0}, 0.0), "undefined"),
    ("odds_ratio_out_of_domain", _derived("odds_ratio_to_d", {"odds_ratio": -1.0}, 0.0), "undefined"),
    ("correlation_out_of_domain", _derived("r_to_d", {"r": 1.0}, 0.0), "undefined"),
    (
        "missing_operand",
        _derived("ci_includes_null", {"ci_low": -0.4, "ci_high": 2.8}, True),
        "takes",
    ),
    ("extra_operand", _derived("reciprocal_abs", {"value": 0.05, "fudge": 1.0}, 20.0), "takes"),
    # A measurement laundered as a constant is the way round the provenance requirement.
    (
        "measurement_declared_a_constant",
        _derived("difference", {"minuend": 0.15, "subtrahend": 0.1}, 0.05, unsourced=("subtrahend",)),
        "is a measurement, not a constant",
    ),
    # codex review 2026-08-12, finding 2 (third round): the operand schema required a
    # locator-SHAPED object but nothing checked that the locator named this operand, which
    # reopened for derived values the field-level hole that findings close.
    (
        "quoted_span_attached_to_the_wrong_operand",
        _misquoted("difference", {"minuend": 0.15, "subtrahend": 0.1}, 0.05, "minuend", ["subtrahend"]),
        "does not name 'minuend'",
    ),
    (
        "quoted_span_naming_something_that_is_not_an_operand",
        _misquoted("negate", {"value": 0.05}, -0.05, "value", ["value", "vibes"]),
        "are not operands of negate",
    ),
    (
        "fractional_sample_size",
        _derived("hedges_g", {"cohens_d": 0.5, "sample_size": 20.5}, hedges_g_correction(0.5, 20.5)),
        "non-negative whole number",
    ),
]


@pytest.mark.parametrize(
    ("case", "record", "expected_fragment"),
    DERIVED_MUTATIONS,
    ids=[case for case, _, _ in DERIVED_MUTATIONS],
)
def test_derived_mutation_is_rejected(case: str, record: dict[str, Any], expected_fragment: str) -> None:
    errors = recompute_derived(record)
    assert errors, f"{case} was accepted"
    assert any(expected_fragment in error for error in errors), errors


def test_an_operand_with_no_account_of_itself_is_structurally_invalid() -> None:
    """Three ways in — quoted, computed, constant — and no fourth."""
    record = _derived("negate", {"value": 0.05}, -0.05)
    del record["operands"]["value"]["quoted"]
    assert schema_errors("derived", record)


def test_derived_defect_inside_a_grade_record_is_rejected() -> None:
    """The walk reaches a derived value wherever it is nested, not only at the top level."""
    record = load_example("grade")
    artifacts = artifacts_for(record)
    record["downgrades"][1]["locator"]["result"] = False
    errors = grade_errors(record, artifacts)
    assert any("recomputed" in error for error in errors), errors


def test_free_text_formula_is_no_longer_accepted() -> None:
    """PLAN.md sketched `formula` as prose; a prose formula cannot be recomputed, so it is gone."""
    record = load_example("grade")
    record["downgrades"][1]["locator"] = {
        "kind": "derived",
        "formula": "the moon says so",
        "inputs": [_quoted("ci_low")],
    }
    assert grade_errors(record, artifacts_for(record))


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


def test_todays_deepextraction_names_exactly_the_step_9a_migration() -> None:
    """This test is expected to FAIL-to-validate until Step 9a lands, and it says why.

    It exists so the migration is a checklist rather than a discovery: when someone changes
    DeepExtraction, the diff between this and the contract is what is left to do. When 9a is
    complete this test flips to asserting the record validates — do not delete it, invert it.
    """
    errors = schema_errors("extraction", _todays_extraction_dict())
    missing = {"trial_registration", "author_group", "funding", "findings"}
    moved_away = {"funding_source", "control_event_rate", "experimental_event_rate", "outcome_is_adverse"}
    for field in missing:
        assert any(f"'{field}' is a required property" in error for error in errors), field
    for field in moved_away:
        assert any(field in error and "not allowed" in error for error in errors), field


def test_to_dict_dropping_nulls_is_itself_part_of_the_migration() -> None:
    """DeepExtraction.to_dict() omits any field that is None (clinical.py:288), but the contract
    requires the key to be present and explicitly null: absent cannot distinguish 'we looked and
    found nothing' from 'this producer version does not set it'. 9a's to_dict must emit nulls."""
    from dr2_podcast.research.clinical import DeepExtraction

    record = DeepExtraction(pmid="1", doi=None, title="t", url="u").to_dict()
    assert "doi" not in record
    assert any("'doi' is a required property" in error for error in schema_errors("extraction", record))


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


VALIDATORS: dict[str, Callable[[dict[str, Any], dict[str, str]], list[str]]] = {
    "finding": finding_errors,
    "funding": funding_errors,
    "extraction": extraction_errors,
    "grade": grade_errors,
    "step_pack": step_pack_errors,
}
