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
  span verification against a source artifact, and ordinal monotonicity.

Everything here is fail-closed by default: ``validate_*`` raises :class:`SchemaValidationError`.
The ``*_errors`` variants return the full list instead, which is what the mutation-matrix tests
assert against.

Nothing in this module calls an LLM, and nothing in it is authored by one.
"""

from __future__ import annotations

import hashlib
import json
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
SCHEMA_NAMES: tuple[str, ...] = ("locator", "derived", "finding", "funding", "grade", "step_pack")

#: Schemas that ship a canonical valid instance under ``examples/``. These are the fixtures the
#: mutation-matrix tests start from, and the worked example an implementer reads first.
EXAMPLE_NAMES: tuple[str, ...] = ("finding", "funding", "grade", "step_pack")

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


def _span_errors(locators: list[dict[str, Any]], artifacts: dict[str, str], prefix: str = "") -> list[str]:
    errors: list[str] = []
    for index, locator in enumerate(locators):
        artifact_id = locator["source_artifact_id"]
        text = artifacts.get(artifact_id)
        if text is None:
            errors.append(f"{prefix}/locators/{index}/source_artifact_id: unknown artifact {artifact_id!r}")
        elif not verify_locator_span(locator, text):
            errors.append(
                f"{prefix}/locators/{index}/quoted_span: not a literal substring of {artifact_id!r} "
                f"at offset {locator['char_offset']}"
            )
    return errors


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


def finding_errors(finding: dict[str, Any], artifacts: dict[str, str] | None = None) -> list[str]:
    """All errors for one finding. Pass ``artifacts`` (id -> text) to also verify literal spans."""
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
    if artifacts is not None:
        errors.extend(_span_errors(finding["locators"], artifacts))
    return errors


def validate_finding(finding: dict[str, Any], artifacts: dict[str, str] | None = None) -> None:
    """Fail closed on one finding."""
    _raise("finding", finding_errors(finding, artifacts))


def validate_findings(findings: list[dict[str, Any]], artifacts: dict[str, str] | None = None) -> None:
    """Fail closed on a paper's whole ``findings[]`` list, reporting every record's errors at once."""
    errors: list[str] = []
    for index, finding in enumerate(findings):
        errors.extend(f"[{index}]{error}" for error in finding_errors(finding, artifacts))
    _raise("findings", errors)


# --------------------------------------------------------------------------- #
# Funding
# --------------------------------------------------------------------------- #
def funding_errors(block: dict[str, Any]) -> list[str]:
    """All errors for a funding block, including the legal-combination table."""
    errors = schema_errors("funding", block)
    if errors:
        if any("is not valid under any of the given schemas" in error for error in errors):
            errors.append(
                "<root>: no legal (funding_disclosure, funding_source_type, funding_raw, funding_locator, "
                "funding_category) combination matched — see the oneOf table in funding.schema.json"
            )
        return errors
    locator = block["funding_locator"]
    if locator is not None and "funding_raw" not in locator["fields"]:
        errors.append(
            "/funding_locator/fields: must name 'funding_raw' — a locator has to source the field it substantiates"
        )
    return errors


def validate_funding(block: dict[str, Any]) -> None:
    """Fail closed on a funding block."""
    _raise("funding", funding_errors(block))


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


def grade_errors(record: dict[str, Any]) -> list[str]:
    """All errors for a structured GRADE record."""
    errors = schema_errors("grade", record)
    if errors:
        return errors
    return _duplicate_domain_errors(record)


def validate_grade(record: dict[str, Any]) -> None:
    """Fail closed on a GRADE record. A record that will not parse stops the run — it never
    defaults to 'Not Determined', which is what the regex scrape at pipeline_sot.py:43 does today."""
    _raise("grade", grade_errors(record))


def net_direction(record: dict[str, Any]) -> int:
    """``sign(sum(upgrades[].steps) - sum(downgrades[].steps))``: -1, 0 or +1.

    Derived from the evidence layer, never from the step pack. Counting step-pack rows would
    weight a finding by how often the episode mentions it, and would let the narrative layer
    constrain a posterior that is supposed to come from the evidence.
    """
    validate_grade(record)
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
def step_pack_errors(pack: dict[str, Any]) -> list[str]:
    """All errors for a step pack, including key/step agreement and the provenance rule."""
    errors = schema_errors("step_pack", pack)
    if errors:
        return errors
    for key, step in sorted(pack["steps"].items()):
        if str(step["step"]) != key:
            errors.append(f"/steps/{key}/step: says {step['step']} but is stored under key {key!r}")
        if step["sufficiency"] != "absent" and not step["locators"]:
            errors.append(
                f"/steps/{key}/locators: empty while sufficiency is {step['sufficiency']!r} — an answer with "
                f"no provenance passes a presence check while being fabricated"
            )
    return errors


def validate_step_pack(pack: dict[str, Any]) -> None:
    """Fail closed on a step pack."""
    _raise("step_pack", step_pack_errors(pack))
