"""Findings and the paper-level extraction record."""

from __future__ import annotations

import hashlib
import unicodedata
from collections import Counter
from typing import Any

from dr2_podcast.schemas._derived import _provenance_errors
from dr2_podcast.schemas._loading import _raise, structural_errors
from dr2_podcast.schemas._records import _funding_locator_errors


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
    errors = structural_errors("finding", finding)
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
    errors = structural_errors("extraction", extraction)
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
