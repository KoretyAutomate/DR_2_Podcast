"""Derived values: recomputation, operand provenance, and producer precision.

PLAN.md's carve-out sentence is "the check recomputes the arithmetic". That is why ``operation``
is a closed enum rather than the free-text ``formula`` the plan sketched — prose can be validated
for shape and nothing more, which would leave the sentence unearned.
"""

from __future__ import annotations

import math
from typing import Any

from dr2_podcast.schemas._locators import span_errors


DERIVED_OPERATIONS: dict[str, tuple[str, ...]] = {
    "difference": ("minuend", "subtrahend"),
    "negate": ("value",),
    "ratio": ("numerator", "denominator"),
    "reciprocal_abs": ("value",),
    "hedges_g": ("cohens_d", "sample_size"),
    "odds_ratio_to_d": ("odds_ratio",),
    "r_to_d": ("r",),
    "d_to_r": ("cohens_d",),
    "ci_includes_null": ("ci_low", "ci_high", "null_value"),
    "ci_excludes_null": ("ci_low", "ci_high", "null_value"),
}

#: Which operands may be declared ``constant`` rather than quoted or computed. Only the null a
#: confidence interval is compared against — everything else is a measurement and needs an account
#: of where it came from, or ``constant`` becomes the way to launder an unsourced number.
CONSTANT_OPERANDS: dict[str, frozenset[str]] = {
    "ci_includes_null": frozenset({"null_value"}),
    "ci_excludes_null": frozenset({"null_value"}),
}


def _evaluate_arithmetic(operation: str, values: dict[str, float]) -> float | None:
    if operation == "difference":
        return values["minuend"] - values["subtrahend"]
    if operation == "negate":
        return -values["value"]
    if operation == "ratio":
        return values["numerator"] / values["denominator"] if values["denominator"] else None
    if operation == "reciprocal_abs":
        return 1.0 / abs(values["value"]) if values["value"] else None
    if operation == "hedges_g":
        # Mirrors effect_size_math.hedges_g_correction, including its n<4 no-correction branch.
        sample_size = values["sample_size"]
        if sample_size < 4:
            return values["cohens_d"]
        return values["cohens_d"] * (1 - 3 / (4 * sample_size - 9))
    if operation == "odds_ratio_to_d":
        odds_ratio = values["odds_ratio"]
        return math.log(odds_ratio) * math.sqrt(3) / math.pi if odds_ratio > 0 else None
    if operation == "r_to_d":
        r = values["r"]
        return 2 * r / math.sqrt(1 - r * r) if abs(r) < 1.0 else None
    return values["cohens_d"] / math.sqrt(values["cohens_d"] ** 2 + 4)


def _evaluate_derived(operation: str, values: dict[str, float]) -> float | bool | None:
    if operation in ("ci_includes_null", "ci_excludes_null"):
        if values["ci_low"] > values["ci_high"]:
            # A transposed interval makes `ci_low <= null <= ci_high` false whatever the null is,
            # which would certify `ci_excludes_null: true` off malformed bounds. The interval has
            # to be rejected rather than silently answered.
            return None
        inside = values["ci_low"] <= values["null_value"] <= values["ci_high"]
        return inside if operation == "ci_includes_null" else not inside
    return _evaluate_arithmetic(operation, values)


#: operation -> the number of decimals the PRODUCER of that quantity rounds to.
#: ``clinical_math.calculate_impact`` rounds ARR to 6, RRR to 4 and NNT to 1
#: (``clinical_math.py:71-74``); ``effect_size_math.calculate_effect`` rounds every conversion to 4
#: (``effect_size_math.py:153-155``). Full-precision equality would reject those producers' own
#: correct output, so the tolerance has to accommodate their rounding — but it is THEIR rounding,
#: fixed here, not a precision the submitted record gets to choose.
DERIVED_RESULT_DECIMALS: dict[str, int] = {
    "difference": 6,
    "negate": 6,
    "ratio": 4,
    "reciprocal_abs": 1,
    "hedges_g": 4,
    "odds_ratio_to_d": 4,
    "r_to_d": 4,
    "d_to_r": 4,
}


def agrees_at_producer_precision(operation: str, expected: float, stated: float) -> bool:
    """True iff ``stated`` is ``expected`` at the precision this operation's producer rounds to.

    An earlier version inferred the tolerance from how many decimals the *stated* value was
    written at. That let the record choose its own tolerance: a result stated as ``0.0`` bought a
    window of ±0.05, so a recomputed ``0.049`` passed — and ``effect_size_math.py:137`` classifies
    anything above 0.01 as a non-null direction, so that is a flipped verdict slipping through a
    check whose entire job is to catch one. The tolerance is now a property of the operation.
    """
    if math.isclose(expected, stated, rel_tol=1e-9, abs_tol=1e-12):
        return True
    return abs(expected - stated) <= 0.5 * 10.0 ** -DERIVED_RESULT_DECIMALS[operation]


#: The threshold ``clinical_math`` itself treats as zero (``clinical_math.py:49,69``). Mirrored
#: exactly, so the two agree on where the degenerate branch begins rather than nearly agreeing.
ZERO_THRESHOLD = 1e-10


def _is_degenerate(operation: str, values: dict[str, float]) -> bool:
    """True where the quantity has no finite value: NNT of a zero difference, RRR over a zero rate."""
    if operation == "reciprocal_abs":
        return abs(values["value"]) < ZERO_THRESHOLD
    if operation == "ratio":
        return not abs(values["denominator"]) > ZERO_THRESHOLD
    return False


def _degenerate_result_errors(operation: str, stated: Any, pointer: str) -> list[str]:
    if stated is None:
        return []
    return [
        f"{pointer}/result: {operation} has no finite value for these operands, so the result must be "
        f"null, not {stated!r} — null is how an infinite NNT or an undefined RRR is written down"
    ]


def _derived_result_errors(record: dict[str, Any], pointer: str) -> list[str]:
    operation = record["operation"]
    values = {name: operand["value"] for name, operand in record["operands"].items()}
    stated = record["result"]
    if _is_degenerate(operation, values):
        return _degenerate_result_errors(operation, stated, pointer)
    if stated is None:
        return [f"{pointer}/result: null, but {operation} is defined for these operands"]
    expected = _evaluate_derived(operation, values)
    if expected is None:
        return [f"{pointer}/operands: {operation} is outside its domain for these operands"]
    if isinstance(expected, bool) != isinstance(stated, bool):
        return [f"{pointer}/result: {operation} yields {type(expected).__name__}, but the record states {stated!r}"]
    if isinstance(expected, bool):
        return [] if expected == stated else [f"{pointer}/result: states {stated!r}, recomputed {expected!r}"]
    if not agrees_at_producer_precision(operation, expected, float(stated)):
        return [f"{pointer}/result: states {stated!r}, recomputed {expected!r} from {values!r}"]
    return []


def _quoted_operand_errors(name: str, operand: dict[str, Any], operation: str, pointer: str) -> list[str]:
    """The same field-level agreement findings get: a locator must source what it is attached to."""
    quoted = operand.get("quoted")
    if quoted is None:
        return []
    errors: list[str] = []
    if name not in quoted["fields"]:
        errors.append(
            f"{pointer}/operands/{name}/quoted/fields: does not name {name!r} — a span attached to an "
            f"operand has to be the span that states it"
        )
    strays = sorted(set(quoted["fields"]) - set(DERIVED_OPERATIONS[operation]))
    if strays:
        errors.append(f"{pointer}/operands/{name}/quoted/fields: {strays} are not operands of {operation}")
    return errors


def _computed_operand_errors(name: str, operand: dict[str, Any], pointer: str) -> list[str]:
    """A computed operand is what was FED IN, which is not always what was reported out.

    ``calculate_impact`` computes RRR and NNT from the full-precision ARR and rounds all three
    only at the end (``clinical_math.py:66-74``). So the value fed into the ratio is the unrounded
    intermediate while the ARR record reports the 6-decimal one. Requiring exact equality rejects
    the calculator's own output: at cer=0.0025003, recomputing RRR from the rounded ARR misses by
    9.6e-05 against a 4-decimal tolerance of 5e-05. The two must agree at the precision the nested
    operation is reported to, and no more.
    """
    nested = operand.get("computed")
    if nested is None:
        return []
    reported = nested["result"]
    if isinstance(reported, bool):
        return [f"{pointer}/operands/{name}/computed: a boolean verdict cannot be a numeric operand"]
    if reported is None:
        return [
            f"{pointer}/operands/{name}/computed: the derivation it names has no finite result, "
            f"so it cannot be fed into another computation"
        ]
    if not agrees_at_producer_precision(nested["operation"], operand["value"], float(reported)):
        return [
            f"{pointer}/operands/{name}/value: {operand['value']!r} does not agree with the result "
            f"{reported!r} of the derivation it names, at {nested['operation']}'s reported precision"
        ]
    return []


def _operand_provenance_errors(record: dict[str, Any], pointer: str) -> list[str]:
    """Every operand must actually account for itself, whichever of the three ways it claims."""
    operation = record["operation"]
    allowed_constants = CONSTANT_OPERANDS.get(operation, frozenset())
    errors: list[str] = []
    for name, operand in sorted(record["operands"].items()):
        if "constant" in operand and name not in allowed_constants:
            errors.append(
                f"{pointer}/operands/{name}: {name!r} is a measurement, not a constant of {operation} — "
                f"it needs a quoted span or a computed derivation"
            )
        errors.extend(_quoted_operand_errors(name, operand, operation, pointer))
        errors.extend(_computed_operand_errors(name, operand, pointer))
    return errors


def _operand_domain_errors(record: dict[str, Any], pointer: str) -> list[str]:
    """Domain constraints the producing function's signature implies but JSON Schema cannot see."""
    if record["operation"] != "hedges_g":
        return []
    sample_size = record["operands"]["sample_size"]["value"]
    if sample_size < 0 or sample_size != int(sample_size):
        return [
            f"{pointer}/operands/sample_size/value: {sample_size!r} is not a non-negative whole number — "
            f"effect_size_math.hedges_g_correction takes n: int"
        ]
    return []


def _derived_errors(record: dict[str, Any], pointer: str) -> list[str]:
    accepted = DERIVED_OPERATIONS[record["operation"]]
    operands = record["operands"]
    if set(operands) != set(accepted):
        return [f"{pointer}/operands: {record['operation']} takes {list(accepted)}, got {sorted(operands)}"]
    errors = _operand_provenance_errors(record, pointer) + _operand_domain_errors(record, pointer)
    return errors + _derived_result_errors(record, pointer)


def recompute_derived(instance: Any) -> list[str]:
    """Re-evaluate every derived value in an instance and reject any stated result that disagrees.

    PLAN.md's carve-out is "the check recomputes the arithmetic". Validating only the shape of a
    derived record would leave that sentence unearned — which is why ``operation`` is a closed
    enum rather than the free-text ``formula`` the plan sketched.
    """
    errors: list[str] = []
    for pointer, record in _iter_derived(instance):
        errors.extend(_derived_errors(record, pointer))
    return errors


def _iter_derived(instance: Any, pointer: str = "") -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(instance, dict):
        if instance.get("kind") == "derived" and instance.get("operation") in DERIVED_OPERATIONS:
            found.append((pointer or "<root>", instance))
        for key, value in instance.items():
            found.extend(_iter_derived(value, f"{pointer}/{key}"))
    elif isinstance(instance, list):
        for index, value in enumerate(instance):
            found.extend(_iter_derived(value, f"{pointer}/{index}"))
    return found


def _provenance_errors(instance: Any, artifacts: dict[str, str]) -> list[str]:
    """Every check that applies to any instance regardless of which schema it is."""
    return span_errors(instance, artifacts) + recompute_derived(instance)


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #
