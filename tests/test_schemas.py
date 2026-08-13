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

from dr2_podcast.schemas import (
    EXAMPLE_NAMES,
    SCHEMA_NAMES,
    SchemaValidationError,
    compute_finding_key,
    example_path,
    finding_errors,
    iter_locators,
    load_example,
    load_schema,
    schema_path,
    span_errors,
    validate_finding,
    verify_locator_span,
)

from tests._schema_fixtures import (
    VALIDATORS,
    Mutation,
    _drop_field_from_coverage,
    _mutated,
    artifacts_for,
    errors_for,
)


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
