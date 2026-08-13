"""Atomic artifact writes and fail-closed reads — PLAN.md Step 8."""

from __future__ import annotations

from pathlib import Path

import pytest

from dr2_podcast.artifacts import (
    ArtifactError,
    clear_candidates,
    read_json_strict,
    read_text_strict,
    sha256_file,
    write_atomic,
    write_json_atomic,
)
from dr2_podcast.schemas import SchemaValidationError


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    for sub in ("research", "scripts", "audio", "meta"):
        (tmp_path / sub).mkdir()
    return tmp_path


# --------------------------------------------------------------------------- #
# Atomic writes
# --------------------------------------------------------------------------- #
def test_write_atomic_returns_the_hash_it_wrote(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    digest = write_atomic(path, "hello")
    assert digest == sha256_file(path)
    assert path.read_text() == "hello"


def test_an_empty_write_is_refused(tmp_path: Path) -> None:
    """An empty artifact overwriting a good one is the silent degradation this prevents."""
    path = tmp_path / "a.md"
    write_atomic(path, "the good version")
    with pytest.raises(ArtifactError, match="empty"):
        write_atomic(path, "   \n")
    assert path.read_text() == "the good version"


def test_a_failed_validation_leaves_the_previous_version_intact(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    write_atomic(path, '{"good": true}')

    def _reject(payload: bytes) -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        write_atomic(path, '{"bad": true}', validate=_reject)
    assert path.read_text() == '{"good": true}'
    assert not list(tmp_path.glob("*.candidate")), "the candidate must be cleaned up"


def test_a_schema_violating_json_never_becomes_the_artifact(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    with pytest.raises(SchemaValidationError):
        write_json_atomic(path, {"schema_version": 1, "mode": "nonsense", "stages": {}}, schema="manifest")
    assert not path.exists()


def test_json_writes_refuse_nan(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_json_atomic(tmp_path / "x.json", {"x": float("nan")})


def test_read_json_strict_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "gone.json"
    with pytest.raises(ArtifactError, match="cannot read"):
        read_json_strict(missing)
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(ArtifactError, match="not valid JSON"):
        read_json_strict(broken)
    nan = tmp_path / "nan.json"
    nan.write_text('{"x": NaN}')
    with pytest.raises(ArtifactError, match="not valid JSON"):
        read_json_strict(nan)


def test_read_text_strict_refuses_an_empty_artifact(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("  \n ")
    with pytest.raises(ArtifactError, match="empty"):
        read_text_strict(path)


# prepush codex 2026-08-12 [P2]: fsyncing the candidate persists its CONTENTS; the directory entry
# created by os.replace is separate metadata, so without a directory fsync a power loss right after
# the rename can leave the target missing — making the stated crash-safety contract untrue.
def test_the_rename_itself_is_made_durable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os as os_module

    synced: list[str] = []
    real_fsync = os_module.fsync

    def _record(fd: int) -> None:
        synced.append("dir" if os_module.fstat(fd).st_mode & 0o040000 else "file")
        real_fsync(fd)

    monkeypatch.setattr(os_module, "fsync", _record)
    write_atomic(tmp_path / "a.md", "durable")
    assert synced == ["file", "dir"], "the contents and the directory entry both have to be synced"


def test_leftover_candidates_are_removed_not_recovered(run_dir: Path) -> None:
    """A candidate means a run died between writing and renaming. It is never a valid artifact."""
    stray = run_dir / "research" / "sot.md.candidate"
    stray.write_text("half a file")
    assert clear_candidates(run_dir) == [stray]
    assert not stray.exists()
