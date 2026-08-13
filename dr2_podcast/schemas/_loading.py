"""Schema files on disk, and the structural-only validator.

``schema_errors`` is deliberately the one public entry point that checks shape ALONE. Everything
that also verifies provenance takes an ``artifacts`` map and lives in the sibling modules; a
caller who wants only structure has to say so by name.
"""

from __future__ import annotations

import json
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
