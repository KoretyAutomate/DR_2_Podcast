"""Derived values: recomputation, operand provenance, composition, and producer precision.

Part of the schema mutation matrix; see test_schemas.py for what that means.
"""

from __future__ import annotations

from typing import Any

import pytest

from dr2_podcast.research.clinical_math import calculate_impact
from dr2_podcast.research.effect_size_math import d_to_r, hedges_g_correction, odds_ratio_to_d, r_to_d
from dr2_podcast.schemas import (
    DERIVED_OPERATIONS,
    agrees_at_producer_precision,
    grade_errors,
    load_example,
    recompute_derived,
    schema_errors,
)

from tests._schema_fixtures import artifacts_for


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


# prepush codex 2026-08-12 [P2]: the producer computes RRR and NNT from the FULL-PRECISION arr and
# rounds all three only at the end, so requiring a computed operand to equal the rounded ARR then
# recomputing from that rounded value rejects the calculator's own output. These CER values are
# small enough that the 6-decimal ARR rounding is amplified past the 4-decimal RRR tolerance.
@pytest.mark.parametrize(("cer", "eer"), [(0.15, 0.1), (0.0100005, 0.005), (0.0025003, 0.0005), (0.008, 0.0030004)])
def test_the_calculate_impact_chain_validates_at_small_event_rates(cer: float, eer: float) -> None:
    impact = calculate_impact("pmid:12345678", cer=cer, eer=eer, outcome_is_adverse=True)
    assert impact is not None
    unrounded_arr = cer - eer
    arr = _derived("difference", {"minuend": cer, "subtrahend": eer}, impact.arr)
    rrr = _derived(
        "ratio", {"numerator": unrounded_arr, "denominator": cer}, impact.rrr, computed={"numerator": arr}
    )
    nnt = _derived("reciprocal_abs", {"value": unrounded_arr}, impact.nnt, computed={"value": arr})
    for record in (arr, rrr, nnt):
        assert schema_errors("derived", record) == [], record
        assert recompute_derived(record) == [], record


# prepush codex 2026-08-12 [P1]: calculate_impact returns nnt = inf when CER equals EER, and JSON
# cannot carry infinity — so the contract could not encode the calculator's own no-effect output.
def test_the_no_effect_case_is_representable() -> None:
    impact = calculate_impact("pmid:12345678", cer=0.1, eer=0.1, outcome_is_adverse=True)
    assert impact is not None
    assert impact.nnt == float("inf") and impact.direction == "no_effect"
    arr = _derived("difference", {"minuend": 0.1, "subtrahend": 0.1}, impact.arr)
    nnt = _derived("reciprocal_abs", {"value": 0.0}, None, computed={"value": arr})
    rrr = _derived("ratio", {"numerator": 0.0, "denominator": 0.1}, impact.rrr, computed={"numerator": arr})
    for record in (arr, nnt, rrr):
        assert schema_errors("derived", record) == [], record
        assert recompute_derived(record) == [], record


def test_the_producer_and_the_contract_agree_that_a_zero_rate_ratio_is_undefined() -> None:
    """prepush codex 2026-08-12 [P2]. Rather than blessing the producer's 0.0, the producer was
    changed: RRR over a zero control-event rate is undefined, and calculate_impact now says None.
    A zero-event control arm is real, so this had to agree in both directions."""
    impact = calculate_impact("pmid:12345678", cer=0.0, eer=0.05, outcome_is_adverse=True)
    assert impact is not None
    assert impact.rrr is None
    rrr = _derived("ratio", {"numerator": -0.05, "denominator": 0.0}, None)
    assert schema_errors("derived", rrr) == []
    assert recompute_derived(rrr) == []


def test_an_infinite_result_cannot_be_fed_into_another_computation() -> None:
    nnt = _derived("reciprocal_abs", {"value": 0.0}, None)
    onward = _derived("negate", {"value": 0.0}, -0.0, computed={"value": nnt})
    assert any("no finite result" in error for error in recompute_derived(onward))


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


def test_a_computed_operand_must_agree_with_the_derivation_it_names() -> None:
    """Agreement is at the nested operation's reported precision — a different number, not a
    differently-rounded one, is still rejected."""
    arr = _derived("difference", {"minuend": 0.15, "subtrahend": 0.1}, 0.05)
    nnt = _derived("reciprocal_abs", {"value": 0.02}, 50.0, computed={"value": arr})
    errors = recompute_derived(nnt)
    assert any("does not agree with the result" in error for error in errors), errors


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
    # clinical_math.py:69 emits rrr=0.0 when CER is zero. RRR is undefined there, not zero — see
    # the migration note in PLAN.md Step S.
    (
        "ratio_over_a_zero_rate_stated_as_zero",
        _derived("ratio", {"numerator": 0.05, "denominator": 0.0}, 0.0),
        "must be",
    ),
    ("null_result_where_the_operation_is_defined", _derived("negate", {"value": 0.05}, None), "is defined"),
    ("odds_ratio_out_of_domain", _derived("odds_ratio_to_d", {"odds_ratio": -1.0}, 0.0), "outside its domain"),
    ("correlation_out_of_domain", _derived("r_to_d", {"r": 1.0}, 0.0), "outside its domain"),
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
        # The result is the correct one for n=20.5, so the fractional n is the only defect.
        "fractional_sample_size",
        _derived("hedges_g", {"cohens_d": 0.5, "sample_size": 20.5}, 0.5 * (1 - 3 / (4 * 20.5 - 9))),
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
