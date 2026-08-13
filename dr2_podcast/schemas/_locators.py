"""Locators: the literal-substring provenance check, at any nesting depth."""

from __future__ import annotations

from typing import Any


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
