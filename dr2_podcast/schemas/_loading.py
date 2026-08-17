"""Schema files on disk, and the structural-only validator.

``schema_errors`` is deliberately the one public entry point that checks shape ALONE. Everything
that also verifies provenance takes an ``artifacts`` map and lives in the sibling modules; a
caller who wants only structure has to say so by name.
"""

from __future__ import annotations

import json
import math
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
    "blueprint",
    "manifest",
    "run_config",
)

#: Schemas that ship a canonical valid instance under ``examples/``. These are the fixtures the
#: mutation-matrix tests start from, and the worked example an implementer reads first.
EXAMPLE_NAMES: tuple[str, ...] = ("finding", "funding", "extraction", "grade", "step_pack", "manifest", "run_config")

#: Bumped when any on-disk artifact shape changes. Also the value the extraction cache keys on —
#: without a bump, cached entries deserialize into records with silently missing ``findings[]``.
SCHEMA_VERSION = 1

class SchemaValidationError(ValueError):
    """Raised by every ``validate_*`` entry point. Carries the complete error list."""

    def __init__(self, artifact: str, errors: list[str]) -> None:
        self.artifact = artifact
        self.errors = list(errors)
        super().__init__(f"{artifact}: " + "; ".join(self.errors))


# --------------------------------------------------------------------------- #
# Schema loading
# --------------------------------------------------------------------------- #


def _reject_constant(name: str) -> float:
    raise ValueError(f"{name} is not valid JSON — the file uses Python's non-standard extension")


def loads_strict(text: str) -> Any:
    """``json.loads`` with NaN/Infinity refused rather than silently accepted."""
    return json.loads(text, parse_constant=_reject_constant)


def schema_path(name: str) -> Path:
    """Absolute path of a shipped schema file."""
    if name not in SCHEMA_NAMES:
        raise KeyError(f"unknown schema {name!r}; known: {', '.join(SCHEMA_NAMES)}")
    return SCHEMA_DIR / f"{name}.schema.json"


def load_schema(name: str) -> dict[str, Any]:
    """Return a fresh copy of a schema document, safe for the caller to mutate."""
    return loads_strict(schema_path(name).read_text(encoding="utf-8"))


def example_path(name: str) -> Path:
    """Absolute path of a shipped canonical instance."""
    if name not in EXAMPLE_NAMES:
        raise KeyError(f"no example for {name!r}; known: {', '.join(EXAMPLE_NAMES)}")
    return EXAMPLE_DIR / f"{name}.json"


def load_example(name: str) -> dict[str, Any]:
    """Return a fresh copy of a canonical instance, safe for the caller to mutate."""
    return loads_strict(example_path(name).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _registry() -> Registry:
    registry: Registry = Registry()
    for path in sorted(SCHEMA_DIR.glob("*.schema.json")):
        contents = loads_strict(path.read_text(encoding="utf-8"))
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


def non_finite_errors(instance: Any, pointer: str = "") -> list[str]:
    """Reject NaN and ±Infinity anywhere in an instance.

    Not a theoretical concern. Python's ``json`` emits and accepts ``NaN`` / ``Infinity`` as a
    non-standard extension, and JSON Schema's ``minimum`` / ``maximum`` are comparisons — every
    comparison against NaN is False, so a NaN slips through its own bounds. Worse, it slips
    through the semantic checks the same way: ``ci_low = NaN`` makes ``ci_low <= null <= ci_high``
    False, which validates an asserted ``ci_excludes_null: true``. A value that defeats every
    comparison has to be rejected before anything compares it.
    """
    if isinstance(instance, bool):
        return []
    if isinstance(instance, float) and not math.isfinite(instance):
        return [f"{pointer or '<root>'}: {instance!r} is not a finite number"]
    if isinstance(instance, dict):
        return [e for key, value in instance.items() for e in non_finite_errors(value, f"{pointer}/{key}")]
    if isinstance(instance, list):
        return [e for index, value in enumerate(instance) for e in non_finite_errors(value, f"{pointer}/{index}")]
    return []


def structural_errors(name: str, instance: Any) -> list[str]:
    """Shape, plus the numeric sanity JSON Schema cannot express. What every entry point starts with."""
    return schema_errors(name, instance) or non_finite_errors(instance)


def _raise(artifact: str, errors: list[str]) -> None:
    if errors:
        raise SchemaValidationError(artifact, errors)


# --------------------------------------------------------------------------- #
# Locators
# --------------------------------------------------------------------------- #
