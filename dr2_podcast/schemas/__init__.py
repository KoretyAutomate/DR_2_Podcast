"""JSON Schema artifacts for the four-role pipeline, plus the checks JSON Schema cannot express.

Why these are files and not prose (PLAN.md, "Where the loop stands"): five rounds of Codex
review on the plan found, in rounds 4 and 5, *only* missing fields in schemas written as
Markdown tables — `outcome_is_adverse` left paper-level while CER/EER moved per-finding,
`intervention`/`comparator` missing from mandatory locator coverage, a funding table that could
not represent its own `undisclosed` state, a GRADE schema that permitted repeated domains while
the formula summed every entry. That defect class keeps appearing until a missing field is a
test failure rather than a reviewer's catch. These files are that move.

Split of responsibility:

* **JSON Schema** owns structure, enums, types, required keys, and the funding legal-combination
  table (expressible as a four-branch ``oneOf``, so the file is self-describing to any consumer).
* **Python, here** owns what JSON Schema genuinely cannot state: ``finding_key`` computation and
  agreement, field-level locator coverage in both directions, at-most-one-entry-per-GRADE-domain,
  CER/EER pairing and its polarity requirement, key↔``step`` agreement in the step pack, literal
  span verification against a source artifact, recomputation of every derived value, and ordinal
  monotonicity.

Everything here is fail-closed by default: ``validate_*`` raises :class:`SchemaValidationError`.
The ``*_errors`` variants return the full list instead, which is what the mutation-matrix tests
assert against.

**``artifacts`` is a required argument, not an optional one.** Every entry point that can contain
a locator takes ``artifacts`` (source_artifact_id -> text) and verifies every span in the instance,
however deeply nested. An optional artifact map is a bypass: the caller who omits it gets a green
result over unverified provenance, which is indistinguishable from a checked one. A caller that
genuinely wants shape alone calls :func:`schema_errors`, which is named for what it does.

Nothing in this module calls an LLM, and nothing in it is authored by one.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

SCHEMA_DIR = Path(__file__).parent
EXAMPLE_DIR = SCHEMA_DIR / "examples"

#: Every schema shipped here. Order is load order only; refs resolve by ``$id``.
SCHEMA_NAMES: tuple[str, ...] = (
    "locator",
    "derived",
    "finding",
    "funding",
    "extraction",
    "grade",
    "step_pack",
)

#: Schemas that ship a canonical valid instance under ``examples/``. These are the fixtures the
#: mutation-matrix tests start from, and the worked example an implementer reads first.
EXAMPLE_NAMES: tuple[str, ...] = ("finding", "funding", "extraction", "grade", "step_pack")

#: Bumped when any on-disk artifact shape changes. Also the value the extraction cache keys on —
#: without a bump, cached entries deserialize into records with silently missing ``findings[]``.
SCHEMA_VERSION = 1

#: The five-level 確信度 ladder, ordinal 0..4. Both the conclusion-first opening and step 10
#: speak a value from this list, and the model never picks the word.
CONFIDENCE_LADDER: tuple[str, ...] = ("まだ分からない", "低い", "中程度", "高い", "ほぼ確実")

#: The tuple ``finding_key`` hashes. Identity of a finding, not of a paper.
FINDING_KEY_FIELDS: tuple[str, ...] = ("population", "intervention", "comparator", "endpoint", "timepoint")

#: Every field of a finding that carries a claim and therefore needs provenance. `intervention`
#: and `comparator` are here because they are two of the five ``finding_key`` inputs — leaving
#: them unsourced would let the model invent an arm while passing validation, corrupting finding
#: identity and therefore replication grouping, which is the whole reason the key exists.
CLAIM_BEARING_FIELDS: tuple[str, ...] = (
    "population",
    "intervention",
    "comparator",
    "endpoint",
    "timepoint",
    "direction",
    "value",
    "unit",
    "ci_low",
    "ci_high",
    "p_value",
    "control_event_rate",
    "experimental_event_rate",
    "outcome_is_adverse",
)


class SchemaValidationError(ValueError):
    """Raised by every ``validate_*`` entry point. Carries the complete error list."""

    def __init__(self, artifact: str, errors: list[str]) -> None:
        self.artifact = artifact
        self.errors = list(errors)
        super().__init__(f"{artifact}: " + "; ".join(self.errors))


# --------------------------------------------------------------------------- #
# Schema loading
# --------------------------------------------------------------------------- #
def schema_path(name: str) -> Path:
    """Absolute path of a shipped schema file."""
    if name not in SCHEMA_NAMES:
        raise KeyError(f"unknown schema {name!r}; known: {', '.join(SCHEMA_NAMES)}")
    return SCHEMA_DIR / f"{name}.schema.json"


def load_schema(name: str) -> dict[str, Any]:
    """Return a fresh copy of a schema document, safe for the caller to mutate."""
    return json.loads(schema_path(name).read_text(encoding="utf-8"))


def example_path(name: str) -> Path:
    """Absolute path of a shipped canonical instance."""
    if name not in EXAMPLE_NAMES:
        raise KeyError(f"no example for {name!r}; known: {', '.join(EXAMPLE_NAMES)}")
    return EXAMPLE_DIR / f"{name}.json"


def load_example(name: str) -> dict[str, Any]:
    """Return a fresh copy of a canonical instance, safe for the caller to mutate."""
    return json.loads(example_path(name).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _registry() -> Registry:
    registry: Registry = Registry()
    for path in sorted(SCHEMA_DIR.glob("*.schema.json")):
        contents = json.loads(path.read_text(encoding="utf-8"))
        registry = registry.with_resource(contents["$id"], Resource.from_contents(contents))
    return registry


@lru_cache(maxsize=len(SCHEMA_NAMES))
def _validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(load_schema(name), registry=_registry())


def _pointer(error: Any) -> str:
    return "/" + "/".join(str(p) for p in error.absolute_path) if error.absolute_path else "<root>"


def schema_errors(name: str, instance: Any) -> list[str]:
    """Structural errors only. Returns ``[]`` when the instance is structurally valid."""
    found = sorted(_validator(name).iter_errors(instance), key=lambda e: list(e.absolute_path))
    return [f"{_pointer(e)}: {e.message}" for e in found]


def _raise(artifact: str, errors: list[str]) -> None:
    if errors:
        raise SchemaValidationError(artifact, errors)


# --------------------------------------------------------------------------- #
# Locators
# --------------------------------------------------------------------------- #
def verify_locator_span(locator: dict[str, Any], artifact_text: str) -> bool:
    """True iff ``quoted_span`` is the literal substring of ``artifact_text`` at ``char_offset``.

    Provenance only. It says the text exists in a document we fetched; it says nothing about
    whether the span was paired with the right PMID, the right arm or the right direction.
    """
    offset = locator["char_offset"]
    span = locator["quoted_span"]
    return artifact_text[offset : offset + len(span)] == span


_LOCATOR_KEYS = frozenset({"fields", "source_artifact_id", "char_offset", "quoted_span"})


def iter_locators(instance: Any, pointer: str = "") -> list[tuple[str, dict[str, Any]]]:
    """Every locator anywhere in an instance, as ``(json_pointer, locator)``.

    Walks the whole document rather than the places a locator is *expected*, so a locator nested
    inside a GRADE modifier, inside a derived value's inputs, or inside a step-pack entry is
    verified by the same code path that verifies a finding's own.
    """
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(instance, dict):
        if _LOCATOR_KEYS.issubset(instance.keys()):
            found.append((pointer or "<root>", instance))
        for key, value in instance.items():
            found.extend(iter_locators(value, f"{pointer}/{key}"))
    elif isinstance(instance, list):
        for index, value in enumerate(instance):
            found.extend(iter_locators(value, f"{pointer}/{index}"))
    return found


def span_errors(instance: Any, artifacts: dict[str, str]) -> list[str]:
    """Verify every locator in an instance against the artifact it names.

    An unknown ``source_artifact_id`` is an error, not a skip: a span nobody can resolve has not
    been checked, and treating it as checked is how unverified provenance passes for verified.
    """
    errors: list[str] = []
    for pointer, locator in iter_locators(instance):
        artifact_id = locator["source_artifact_id"]
        text = artifacts.get(artifact_id)
        if text is None:
            errors.append(f"{pointer}/source_artifact_id: unknown artifact {artifact_id!r}")
        elif not verify_locator_span(locator, text):
            errors.append(
                f"{pointer}/quoted_span: not a literal substring of {artifact_id!r} "
                f"at offset {locator['char_offset']}"
            )
    return errors


# --------------------------------------------------------------------------- #
# Derived values
# --------------------------------------------------------------------------- #
#: operation -> the operand names it takes. Exactly the arithmetic this repo's two deterministic
#: calculators compute (``clinical_math.py``, ``effect_size_math.py``) plus the GRADE imprecision
#: judgement. Nothing speculative: an operation with no producer would be a contract nobody meets,
#: and an operation a producer needs but the enum lacks would reject real data.
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


def _derived_result_errors(record: dict[str, Any], pointer: str) -> list[str]:
    operation = record["operation"]
    values = {name: operand["value"] for name, operand in record["operands"].items()}
    expected = _evaluate_derived(operation, values)
    stated = record["result"]
    if expected is None:
        return [f"{pointer}/operands: {operation} is undefined for these operands"]
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
    nested = operand.get("computed")
    if nested is None:
        return []
    if isinstance(nested["result"], bool):
        return [f"{pointer}/operands/{name}/computed: a boolean verdict cannot be a numeric operand"]
    if not math.isclose(operand["value"], float(nested["result"]), rel_tol=1e-9, abs_tol=1e-12):
        return [
            f"{pointer}/operands/{name}/value: {operand['value']!r} does not equal the result "
            f"{nested['result']!r} of the derivation it names"
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
def _normalise_key_part(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def compute_finding_key(finding: dict[str, Any]) -> str:
    """sha1 of the normalised (population, intervention, comparator, endpoint, timepoint) tuple.

    The ONLY producer of ``finding_key``. A model-authored key would make replication grouping a
    semantic task again, which is exactly what the key exists to remove.
    """
    joined = "\x1f".join(_normalise_key_part(finding.get(field)) for field in FINDING_KEY_FIELDS)
    return hashlib.sha1(joined.encode("utf-8"), usedforsecurity=False).hexdigest()


def _coverage_errors(finding: dict[str, Any]) -> list[str]:
    present = {field for field in CLAIM_BEARING_FIELDS if finding.get(field) is not None}
    covered: set[str] = set()
    errors: list[str] = []
    for index, locator in enumerate(finding["locators"]):
        for field in locator["fields"]:
            covered.add(field)
            if field not in CLAIM_BEARING_FIELDS:
                errors.append(f"/locators/{index}/fields: {field!r} is not a claim-bearing field")
            elif field not in present:
                errors.append(f"/locators/{index}/fields: {field!r} is null on this finding, so nothing sources it")
    errors.extend(
        f"/locators: no locator names {field!r}, which is claim-bearing and non-null"
        for field in sorted(present - covered)
    )
    return errors


def _rate_errors(finding: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    cer, eer = finding.get("control_event_rate"), finding.get("experimental_event_rate")
    if (cer is None) != (eer is None):
        errors.append("<root>: control_event_rate and experimental_event_rate must both be present or both null")
    if cer is not None and finding.get("outcome_is_adverse") is None:
        errors.append(
            "/outcome_is_adverse: required when CER/EER are present — clinical_math.py:39 flips the ARR "
            "interpretation on this flag, and a missing one silently reads as 'adverse', producing a "
            "directionally wrong NNT for a beneficial endpoint"
        )
    low, high = finding.get("ci_low"), finding.get("ci_high")
    if (low is None) != (high is None):
        errors.append("<root>: ci_low and ci_high must both be present or both null")
    elif low is not None and high is not None and low > high:
        errors.append(f"/ci_low: {low} is greater than ci_high {high}")
    return errors


def finding_errors(finding: dict[str, Any], artifacts: dict[str, str]) -> list[str]:
    """All errors for one finding, spans included. ``artifacts`` maps artifact id -> text."""
    errors = schema_errors("finding", finding)
    if errors:
        return errors
    expected = compute_finding_key(finding)
    if finding["finding_key"] != expected:
        errors.append(
            f"/finding_key: {finding['finding_key']!r} does not match the key computed from this "
            f"finding's identity tuple ({expected!r}); the key is Python's to produce, never the model's"
        )
    errors.extend(_coverage_errors(finding))
    errors.extend(_rate_errors(finding))
    errors.extend(_provenance_errors(finding, artifacts))
    return errors


def validate_finding(finding: dict[str, Any], artifacts: dict[str, str]) -> None:
    """Fail closed on one finding."""
    _raise("finding", finding_errors(finding, artifacts))


# --------------------------------------------------------------------------- #
# The paper-level record
# --------------------------------------------------------------------------- #
def extraction_errors(extraction: dict[str, Any], artifacts: dict[str, str]) -> list[str]:
    """All errors for one paper-level extraction, including every nested finding and the funding block."""
    errors = schema_errors("extraction", extraction)
    if errors:
        return errors
    for index, finding in enumerate(extraction["findings"]):
        expected = compute_finding_key(finding)
        if finding["finding_key"] != expected:
            errors.append(f"/findings/{index}/finding_key: does not match this finding's identity tuple ({expected!r})")
        errors.extend(f"/findings/{index}{error}" for error in _coverage_errors(finding))
        errors.extend(f"/findings/{index}{error}" for error in _rate_errors(finding))
    errors.extend(f"/funding{error}" for error in _funding_locator_errors(extraction["funding"]))
    errors.extend(_duplicate_finding_key_errors(extraction["findings"]))
    errors.extend(_provenance_errors(extraction, artifacts))
    return errors


def _duplicate_finding_key_errors(findings: list[dict[str, Any]]) -> list[str]:
    counts = Counter(finding["finding_key"] for finding in findings)
    return [
        f"/findings: finding_key {key!r} appears {count} times in one paper; two records with the same "
        f"(population, intervention, comparator, endpoint, timepoint) are the same finding and must be merged"
        for key, count in sorted(counts.items())
        if count > 1
    ]


def validate_extraction(extraction: dict[str, Any], artifacts: dict[str, str]) -> None:
    """Fail closed on a paper-level extraction."""
    _raise("extraction", extraction_errors(extraction, artifacts))


# --------------------------------------------------------------------------- #
# Funding
# --------------------------------------------------------------------------- #
def _funding_locator_errors(block: dict[str, Any]) -> list[str]:
    locator = block["funding_locator"]
    if locator is not None and "funding_raw" not in locator["fields"]:
        return [
            "/funding_locator/fields: must name 'funding_raw' — a locator has to source the field it substantiates"
        ]
    return []


def funding_errors(block: dict[str, Any], artifacts: dict[str, str]) -> list[str]:
    """All errors for a funding block, including the legal-combination table and its locator's span."""
    errors = schema_errors("funding", block)
    if errors:
        if any("is not valid under any of the given schemas" in error for error in errors):
            errors.append(
                "<root>: no legal (funding_disclosure, funding_source_type, funding_raw, funding_locator, "
                "funding_category) combination matched — see the oneOf table in funding.schema.json"
            )
        return errors
    errors.extend(_funding_locator_errors(block))
    errors.extend(_provenance_errors(block, artifacts))
    return errors


def validate_funding(block: dict[str, Any], artifacts: dict[str, str]) -> None:
    """Fail closed on a funding block."""
    _raise("funding", funding_errors(block, artifacts))


# --------------------------------------------------------------------------- #
# Structured GRADE
# --------------------------------------------------------------------------- #
def _duplicate_domain_errors(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("downgrades", "upgrades"):
        counts = Counter(entry["domain"] for entry in record[key])
        errors.extend(
            f"/{key}: domain {domain!r} appears {count} times; at most one entry per domain, "
            f"aggregated before writing — otherwise sum(steps) double-counts and net_direction is wrong"
            for domain, count in sorted(counts.items())
            if count > 1
        )
    return errors


def grade_errors(record: dict[str, Any], artifacts: dict[str, str]) -> list[str]:
    """All errors for a structured GRADE record, including every modifier's evidence."""
    errors = schema_errors("grade", record)
    if errors:
        return errors
    errors.extend(_duplicate_domain_errors(record))
    errors.extend(_provenance_errors(record, artifacts))
    return errors


def validate_grade(record: dict[str, Any], artifacts: dict[str, str]) -> None:
    """Fail closed on a GRADE record. A record that will not parse stops the run — it never
    defaults to 'Not Determined', which is what the regex scrape at pipeline_sot.py:43 does today."""
    _raise("grade", grade_errors(record, artifacts))


def net_direction(record: dict[str, Any]) -> int:
    """``sign(sum(upgrades[].steps) - sum(downgrades[].steps))``: -1, 0 or +1.

    Derived from the evidence layer, never from the step pack. Counting step-pack rows would
    weight a finding by how often the episode mentions it, and would let the narrative layer
    constrain a posterior that is supposed to come from the evidence.

    Fails closed on the two properties the sum actually depends on — structure and one entry per
    domain. It does NOT verify spans, because it does not need artifacts to add integers; the
    caller that writes the record is the one that must have run :func:`validate_grade`.
    """
    _raise("grade", schema_errors("grade", record) + _duplicate_domain_errors(record))
    total = sum(entry["steps"] for entry in record["upgrades"]) - sum(entry["steps"] for entry in record["downgrades"])
    return (total > 0) - (total < 0)


# --------------------------------------------------------------------------- #
# Ordinal monotonicity (the one mechanical residue of step 9)
# --------------------------------------------------------------------------- #
def confidence_index(level: str) -> int:
    """Ordinal position on :data:`CONFIDENCE_LADDER`."""
    if level not in CONFIDENCE_LADDER:
        raise SchemaValidationError("confidence", [f"{level!r} is not on CONFIDENCE_LADDER"])
    return CONFIDENCE_LADDER.index(level)


def ordinal_monotonicity_errors(
    prior_level: str,
    posterior_level: str,
    net: int,
    jump_reason: str | None = None,
) -> list[str]:
    """Direction-only check on the prior -> posterior update.

    Coarse on purpose: it checks the *direction* of the update, never its magnitude, and claims
    nothing more. Step 9 proper is a qualitative reconciliation audited by Claude — requiring the
    posterior to equal the GRADE-derived 確信度 would assert one value twice and verify nothing,
    since step 10's 確信度 is the same lookup.
    """
    errors: list[str] = []
    for label, level in (("prior_level", prior_level), ("posterior_level", posterior_level)):
        if level not in CONFIDENCE_LADDER:
            errors.append(f"/{label}: {level!r} is not on CONFIDENCE_LADDER")
    if errors:
        return errors
    prior_index = CONFIDENCE_LADDER.index(prior_level)
    posterior_index = CONFIDENCE_LADDER.index(posterior_level)
    if net > 0 and posterior_index < prior_index:
        errors.append(f"/posterior_level: net-supporting evidence (net={net}) must not move confidence down the ladder")
    if net < 0 and posterior_index > prior_index:
        errors.append(f"/posterior_level: net-undermining evidence (net={net}) must not move confidence up the ladder")
    if abs(posterior_index - prior_index) > 2 and not (jump_reason or "").strip():
        errors.append(
            f"/jump_reason: a move of {abs(posterior_index - prior_index)} ladder steps requires a stated reason"
        )
    return errors


def validate_ordinal_monotonicity(
    prior_level: str,
    posterior_level: str,
    net: int,
    jump_reason: str | None = None,
) -> None:
    """Fail closed on the prior -> posterior update direction."""
    _raise("confidence", ordinal_monotonicity_errors(prior_level, posterior_level, net, jump_reason))


# --------------------------------------------------------------------------- #
# Step pack
# --------------------------------------------------------------------------- #
def step_pack_errors(pack: dict[str, Any], artifacts: dict[str, str]) -> list[str]:
    """All errors for a step pack: key/step agreement, the provenance rule, and every span.

    What this does NOT check is that an ``answer`` was derived rather than authored. The only
    honest check for that is regenerating the pack from ``pipeline_data`` + extractions + GRADE
    and comparing, which belongs to the Step 9b generator; a validator handed a finished pack
    cannot tell a computed count from a plausible one. Recorded as open work in PLAN.md Step S.
    """
    errors = schema_errors("step_pack", pack)
    if errors:
        return errors
    errors.extend(_provenance_errors(pack, artifacts))
    for key, step in sorted(pack["steps"].items()):
        if str(step["step"]) != key:
            errors.append(f"/steps/{key}/step: says {step['step']} but is stored under key {key!r}")
        if step["sufficiency"] != "absent" and not step["locators"]:
            errors.append(
                f"/steps/{key}/locators: empty while sufficiency is {step['sufficiency']!r} — an answer with "
                f"no provenance passes a presence check while being fabricated"
            )
    return errors


def validate_step_pack(pack: dict[str, Any], artifacts: dict[str, str]) -> None:
    """Fail closed on a step pack."""
    _raise("step_pack", step_pack_errors(pack, artifacts))
