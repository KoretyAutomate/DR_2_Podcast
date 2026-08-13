"""Shared fixtures for the schema mutation matrices.

Split out of ``test_schemas.py`` so each module stays under the repo's file-size ceiling. What
these helpers buy is honest cases: every mutation starts from a canonical instance AND from
artifacts built to satisfy that instance's own locators, so the only thing wrong with a mutated
record is the thing its case names.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dr2_podcast.manifest import manifest_errors
from dr2_podcast.schemas import (
    extraction_errors,
    finding_errors,
    funding_errors,
    grade_errors,
    iter_locators,
    load_example,
    schema_errors,
    step_pack_errors,
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


def _drop_field_from_coverage(finding: dict[str, Any], field: str) -> None:
    for locator in finding["locators"]:
        if field in locator["fields"]:
            locator["fields"] = [f for f in locator["fields"] if f != field]
            if not locator["fields"]:
                locator["fields"] = ["endpoint"]


VALIDATORS: dict[str, Callable[[dict[str, Any], dict[str, str]], list[str]]] = {
    "finding": finding_errors,
    "funding": funding_errors,
    "extraction": extraction_errors,
    "grade": grade_errors,
    "step_pack": step_pack_errors,
    # A manifest and a run config carry no locators — hashes ARE their provenance —
    # so they take the artifacts argument only to keep one shape for every validator.
    "manifest": lambda instance, _artifacts: manifest_errors(instance),
    "run_config": lambda instance, _artifacts: schema_errors("run_config", instance),
}
