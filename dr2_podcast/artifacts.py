"""Atomic artifact writes and fail-closed reads.

PLAN.md Step 8. Two rules, both of which the repo currently breaks somewhere:

**Atomic writes.** Candidate → validate → rename. An interrupted write must never leave a
half-artifact that the next stage reads as complete, and empty or malformed output must never
overwrite the last approved version. Today artifacts are written with a bare ``write_text`` at a
dozen call sites, so a crash mid-write leaves a truncated file that looks finished.

**Fail-closed parsing.** A parse failure stops the stage. The repo's habit is the opposite, and
that is why this module exists: ``clinical.py:705`` returns ``metadata=None`` on parse failure and
the text flows on as usable facts; ``clinical.py:1092`` turns a query-JSON parse failure into an
empty list; ``clinical.py:2654`` returns a failure document with the extraction text concatenated
into it, which then looks like a synthesis. None of those may become stage contracts.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dr2_podcast.schemas import SchemaValidationError, schema_errors
from dr2_podcast.schemas._loading import loads_strict

#: Suffix for the not-yet-committed candidate. Written beside the target so the final rename is on
#: the same filesystem, which is what makes it atomic.
CANDIDATE_SUFFIX = ".candidate"


class ArtifactError(RuntimeError):
    """Raised instead of degrading. Every path in this module fails closed."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file's bytes. Raises if it does not exist — an unhashable input is not a current one."""
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise ArtifactError(f"cannot hash {path}: {exc}") from exc


def write_atomic(
    path: Path,
    data: str | bytes,
    *,
    validate: Callable[[bytes], None] | None = None,
    allow_empty: bool = False,
) -> str:
    """Write ``data`` to ``path`` atomically, returning its sha256.

    The candidate is fsynced before the rename, so a crash leaves either the previous version or
    the complete new one — never a truncated file that the next stage reads as finished.

    ``validate`` receives the candidate's bytes and raises to abort; the target is left untouched.
    Empty content is refused unless ``allow_empty``, because an empty artifact overwriting a good
    one is the silent-degradation failure this module exists to prevent.
    """
    payload = data.encode("utf-8") if isinstance(data, str) else data
    if not payload.strip() and not allow_empty:
        raise ArtifactError(f"refusing to write empty content to {path}; pass allow_empty to mean it")
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(path.name + CANDIDATE_SUFFIX)
    try:
        with open(candidate, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if validate is not None:
            validate(payload)
        os.replace(candidate, path)
    except Exception:
        candidate.unlink(missing_ok=True)
        raise
    return sha256_bytes(payload)


def write_json_atomic(path: Path, value: Any, *, schema: str | None = None) -> str:
    """Serialise and write JSON atomically, optionally validating the candidate against a schema.

    Validation happens on the candidate, before the rename: an instance that does not satisfy its
    contract never becomes the artifact other stages read.
    """
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"

    def _validate(payload: bytes) -> None:
        if schema is None:
            return
        errors = schema_errors(schema, loads_strict(payload.decode("utf-8")))
        if errors:
            raise SchemaValidationError(schema, errors)

    return write_atomic(path, text, validate=_validate)


def read_json_strict(path: Path, *, schema: str | None = None) -> Any:
    """Read JSON, refusing NaN/Infinity, and optionally requiring it to satisfy a schema.

    Every failure raises. There is no ``default=`` parameter and there will not be one: the point
    of the module is that a stage stops rather than continuing on a value it invented.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"cannot read {path}: {exc}") from exc
    try:
        value = loads_strict(text)
    except ValueError as exc:
        raise ArtifactError(f"{path} is not valid JSON: {exc}") from exc
    if schema is not None:
        errors = schema_errors(schema, value)
        if errors:
            raise SchemaValidationError(f"{path}", errors)
    return value


def read_text_strict(path: Path) -> str:
    """Read a text artifact, refusing an empty or whitespace-only one."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"cannot read {path}: {exc}") from exc
    if not text.strip():
        raise ArtifactError(f"{path} is empty — an empty artifact is a failed stage, not a result")
    return text


def clear_candidates(run_dir: Path) -> list[Path]:
    """Delete leftover candidates from an interrupted write. Returns what was removed.

    A candidate on disk means a run died between writing and renaming. It is never a valid
    artifact, so it is removed rather than recovered — the stage that wrote it re-runs.
    """
    removed = [p for p in run_dir.rglob(f"*{CANDIDATE_SUFFIX}") if p.is_file()]
    for path in removed:
        path.unlink()
    return removed
