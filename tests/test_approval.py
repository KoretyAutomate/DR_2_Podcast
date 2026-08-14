"""The strategy approval bundle — PLAN.md Step 10.

The exit criterion is explicit about what a strategy-only hash would have let through: approve a
plan, then confirm the search refuses after each of a mutated strategy file, a mutated
`research_framing.md`, and a mutated frozen prior. All three are here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dr2_podcast.approval import (
    APPROVAL_ARTIFACT,
    APPROVAL_INPUTS,
    approval_errors,
    bundle_hash,
    bundle_inputs,
    require_approval,
    write_approval,
)
from dr2_podcast.artifacts import ArtifactError


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    for sub in ("research", "meta", "scripts", "audio"):
        (tmp_path / sub).mkdir()
    (tmp_path / "research/research_framing.md").write_text("# Framing\n\nWho, what, compared to what.\n")
    (tmp_path / "research/search_strategy_aff.json").write_text('{"tier1": {"intervention": ["vitamin D"]}}')
    (tmp_path / "research/search_strategy_neg.json").write_text('{"tier1": {"intervention": ["vitamin D harm"]}}')
    return tmp_path


def _approve(run_dir: Path):
    return write_approval(run_dir, approver="claude", approved_at="2026-08-13T09:00:00+09:00")


# --------------------------------------------------------------------------- #
# The bundle itself
# --------------------------------------------------------------------------- #
def test_an_approval_verifies_against_the_artifacts_it_was_made_against(run_dir: Path) -> None:
    _approve(run_dir)
    assert approval_errors(run_dir) == []
    assert require_approval(run_dir)["approver"] == "claude"


def test_the_bundle_hash_is_stable_across_processes(run_dir: Path) -> None:
    """It has to be recomputable on another machine months later, so nothing about how the JSON
    happens to be written may reach the hash."""
    inputs = bundle_inputs(run_dir)
    assert bundle_hash(inputs) == bundle_hash(json.loads(json.dumps(inputs)))
    assert bundle_hash(inputs) == bundle_hash([dict(reversed(list(entry.items()))) for entry in inputs])


def test_the_artifact_order_is_fixed(run_dir: Path) -> None:
    """A set has no order, and a hash over an unordered thing is not reproducible."""
    assert [entry["artifact"] for entry in bundle_inputs(run_dir)] == list(APPROVAL_INPUTS)


# --------------------------------------------------------------------------- #
# What the approval refuses
# --------------------------------------------------------------------------- #
def test_no_approval_means_no_search(run_dir: Path) -> None:
    with pytest.raises(ArtifactError, match="not approved"):
        require_approval(run_dir)


@pytest.mark.parametrize(
    "artifact",
    ["research/search_strategy_aff.json", "research/search_strategy_neg.json"],
)
def test_a_strategy_edited_after_approval_fails_closed(run_dir: Path, artifact: str) -> None:
    _approve(run_dir)
    (run_dir / artifact).write_text('{"tier1": {"intervention": ["something else entirely"]}}')
    with pytest.raises(ArtifactError, match="has changed since"):
        require_approval(run_dir)


def test_a_framing_edited_after_approval_fails_closed(run_dir: Path) -> None:
    """The first of the two a strategy-only hash would have let through. The strategies are approved
    BY COMPARISON with the framing, so a changed framing invalidates the comparison."""
    _approve(run_dir)
    (run_dir / "research/research_framing.md").write_text("# Framing\n\nA different question entirely.\n")
    with pytest.raises(ArtifactError, match="research_framing.md has changed"):
        require_approval(run_dir)


def test_a_prior_appearing_after_approval_fails_closed(run_dir: Path) -> None:
    """The second. framing_prior.json does not exist yet — no stage authors it — so it is recorded
    as absent rather than skipped. Skipping would mean the day it appears, a stale approval still
    verifies against a bundle that never noticed it."""
    _approve(run_dir)
    assert not (run_dir / "research/framing_prior.json").exists()

    (run_dir / "research/framing_prior.json").write_text('{"prior_level": "低い"}')
    with pytest.raises(ArtifactError, match="did not exist when the strategies were approved"):
        require_approval(run_dir)


def test_a_prior_removed_after_approval_fails_closed(run_dir: Path) -> None:
    (run_dir / "research/framing_prior.json").write_text('{"prior_level": "低い"}')
    _approve(run_dir)
    (run_dir / "research/framing_prior.json").unlink()
    with pytest.raises(ArtifactError, match="existed when the strategies were approved"):
        require_approval(run_dir)


def test_an_approval_edited_after_the_fact_fails_closed(run_dir: Path) -> None:
    """Rewriting the recorded hashes to match a mutated strategy is the obvious way to defeat this,
    so the record is checked against its own bundle hash first."""
    _approve(run_dir)
    path = run_dir / APPROVAL_ARTIFACT
    document = json.loads(path.read_text())
    document["inputs"][2]["sha256"] = "0" * 64
    path.write_text(json.dumps(document))

    with pytest.raises(ArtifactError, match="edited since it was written"):
        require_approval(run_dir)


def test_an_approval_naming_different_artifacts_fails_closed(run_dir: Path) -> None:
    _approve(run_dir)
    path = run_dir / APPROVAL_ARTIFACT
    document = json.loads(path.read_text())
    document["inputs"][0]["artifact"] = "research/something_else.md"
    document["bundle_sha256"] = bundle_hash(document["inputs"])
    path.write_text(json.dumps(document))

    with pytest.raises(ArtifactError, match="different set of artifacts"):
        require_approval(run_dir)


def test_an_unreadable_approval_fails_closed(run_dir: Path) -> None:
    (run_dir / APPROVAL_ARTIFACT).write_text("{not json")
    with pytest.raises(ArtifactError):
        require_approval(run_dir)


def test_re_approving_after_a_change_makes_it_valid_again(run_dir: Path) -> None:
    """The gate has to be passable, or it gets bypassed instead of satisfied."""
    _approve(run_dir)
    (run_dir / "research/search_strategy_aff.json").write_text('{"tier1": {"intervention": ["revised"]}}')
    with pytest.raises(ArtifactError):
        require_approval(run_dir)

    _approve(run_dir)
    assert approval_errors(run_dir) == []
