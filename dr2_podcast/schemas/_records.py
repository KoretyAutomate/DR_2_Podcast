"""Funding, structured GRADE, the confidence ladder, and the step pack."""

from __future__ import annotations

from collections import Counter
from typing import Any

from dr2_podcast.schemas._derived import _provenance_errors
from dr2_podcast.schemas._loading import SchemaValidationError, _raise, structural_errors


#: The five-level 確信度 ladder, ordinal 0..4. Both the conclusion-first opening and step 10
#: speak a value from this list, and the model never picks the word.
CONFIDENCE_LADDER: tuple[str, ...] = ("まだ分からない", "低い", "中程度", "高い", "ほぼ確実")

#: The tuple ``finding_key`` hashes. Identity of a finding, not of a paper.


def _funding_locator_errors(block: dict[str, Any]) -> list[str]:
    locator = block["funding_locator"]
    if locator is not None and "funding_raw" not in locator["fields"]:
        return [
            "/funding_locator/fields: must name 'funding_raw' — a locator has to source the field it substantiates"
        ]
    return []


def funding_errors(block: dict[str, Any], artifacts: dict[str, str]) -> list[str]:
    """All errors for a funding block, including the legal-combination table and its locator's span."""
    errors = structural_errors("funding", block)
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
    errors = structural_errors("grade", record)
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
    _raise("grade", structural_errors("grade", record) + _duplicate_domain_errors(record))
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
    errors = structural_errors("step_pack", pack)
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


def _freeze_time_errors(record: dict[str, Any]) -> list[str]:
    """`frozen_at` must name a real instant, not merely digits in the right places.

    The pattern accepts 2026-99-99 and 2026-02-31 — layout is not a date (prepush codex 2026-08-17).
    Parsed here instead, because "written before the search" is the one property this artifact has
    and a value nothing can order against the run is no evidence of it.
    """
    from datetime import datetime

    stamp = record.get("frozen_at", "")
    try:
        datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return [f"/frozen_at: {stamp!r} is not a real date or time"]
    return []


def framing_prior_errors(record: dict[str, Any]) -> list[str]:
    """Shape only. The prior is a JUDGEMENT — nothing can check it against the literature, and
    checking it against THIS run's literature would defeat the point of freezing it beforehand.
    What is checkable is that every component states its basis."""
    errors = structural_errors("framing_prior", record)
    return errors or _freeze_time_errors(record)


def validate_framing_prior(record: dict[str, Any]) -> None:
    """Fail closed on a prior. Step 9 does ordinal arithmetic over prior_level and the episode
    states the result, so a prior that will not parse is worse than no prior at all."""
    _raise("framing_prior", framing_prior_errors(record))
