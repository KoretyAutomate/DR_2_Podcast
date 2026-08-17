"""The approval bundle: what a reviewer approved, hashed so it cannot change underneath them.

PLAN.md Step 10. A search strategy is approved *by comparison with* the framing it is supposed to
serve and the prior it is supposed to test — so hashing only the strategy files would leave an
approval valid after the framing changed, which is the one thing the approval was for.

Two properties do the work:

* **Fixed artifact order and canonical JSON.** The bundle hash has to be reproducible on a different
  machine, in a different process, months later. Dict ordering, indentation and unicode escaping all
  change bytes without changing meaning, so none of them are allowed to reach the hash.
* **An absent artifact is recorded as absent, not skipped.** `framing_prior.json` does not exist yet
  — no stage authors it (PLAN.md Step 2). Skipping it would mean the day it appears, the bundle is
  unchanged and a stale approval still verifies. Recording `null` makes its arrival a mismatch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dr2_podcast.artifacts import ArtifactError, read_json_strict, write_json_atomic

#: Everything a strategy approval is made against, in the order the bundle hashes them. Fixed
#: because a set has no order and a hash over an unordered thing is not reproducible.
APPROVAL_INPUTS: tuple[str, ...] = (
    "research/research_framing.md",
    "research/framing_prior.json",
    "research/search_strategy_aff.json",
    "research/search_strategy_neg.json",
)

APPROVAL_ARTIFACT = "meta/strategy_approval.json"


def _artifact_hash(path: Path) -> str | None:
    """sha256 of an artifact's bytes, or None when it is not there.

    Bytes, not parsed content: a JSON file that round-trips differently is a different file, and the
    approval is over what the reviewer actually read.
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def bundle_inputs(run_dir: Path) -> list[dict[str, Any]]:
    """The current state of every input an approval covers, in bundle order."""
    return [{"artifact": name, "sha256": _artifact_hash(run_dir / name)} for name in APPROVAL_INPUTS]


def bundle_hash(inputs: list[dict[str, Any]]) -> str:
    """One hash over the ordered inputs, canonically serialised.

    ``sort_keys`` and ``separators`` pin the bytes; ``ensure_ascii`` pins the escaping. Without all
    three the same bundle hashes differently depending on how it was written, and an approval that
    cannot be recomputed is an approval that always fails — which is a gate nobody keeps.
    """
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_approval(run_dir: Path, *, approver: str, approved_at: str, note: str = "") -> dict[str, Any]:
    """Record that ``approver`` approved the strategies as they stand right now."""
    inputs = bundle_inputs(run_dir)
    document = {
        "approved_at": approved_at,
        "approver": approver,
        "inputs": inputs,
        "bundle_sha256": bundle_hash(inputs),
    }
    if note:
        document["note"] = note
    write_json_atomic(run_dir / APPROVAL_ARTIFACT, document)
    return document


def approval_errors(run_dir: Path) -> list[str]:
    """Everything wrong with the approval on disk, as reasons a person can act on."""
    path = run_dir / APPROVAL_ARTIFACT
    if not path.exists():
        return [
            f"no {APPROVAL_ARTIFACT}: the search strategies have not been approved, and the yield gate "
            f"downstream can only catch a strategy that is wrong in QUANTITY — not one that searches "
            f"for the wrong population, or a falsification track that is not actually adversarial"
        ]
    try:
        document = read_json_strict(path)
    except (ArtifactError, ValueError) as exc:
        return [f"{APPROVAL_ARTIFACT} could not be read: {exc}"]

    recorded = document.get("inputs")
    if not isinstance(recorded, list):
        return [f"{APPROVAL_ARTIFACT} has no inputs list, so there is nothing to check it against"]
    if bundle_hash(recorded) != document.get("bundle_sha256"):
        return [
            f"{APPROVAL_ARTIFACT}: bundle_sha256 does not match its own inputs list — the record has "
            f"been edited since it was written"
        ]

    current = bundle_inputs(run_dir)
    errors = []
    for was, now in zip(recorded, current, strict=False):
        if was.get("artifact") != now["artifact"]:
            errors.append(
                f"{APPROVAL_ARTIFACT}: approved input order does not match "
                f"({was.get('artifact')!r} vs {now['artifact']!r}) — this approval was made against a "
                f"different set of artifacts"
            )
            continue
        if was.get("sha256") != now["sha256"]:
            errors.append(_changed_message(now["artifact"], was.get("sha256"), now["sha256"]))
    if len(recorded) != len(current):
        errors.append(
            f"{APPROVAL_ARTIFACT}: approval covers {len(recorded)} artifact(s), this run has "
            f"{len(current)} — the bundle definition changed, so the approval cannot be checked"
        )
    return errors


def _changed_message(artifact: str, was: str | None, now: str | None) -> str:
    if was is None:
        return f"{artifact} did not exist when the strategies were approved, and now it does"
    if now is None:
        return f"{artifact} existed when the strategies were approved, and now it does not"
    return f"{artifact} has changed since the strategies were approved"


def require_approval(run_dir: Path) -> dict[str, Any]:
    """The approval, or refuse to search.

    Fails closed on every shape of drift: no approval, an approval edited after the fact, a strategy
    changed after approval, and — the two a strategy-only hash would have let through — a framing or
    a prior changed after approval.
    """
    errors = approval_errors(run_dir)
    if errors:
        raise ArtifactError(
            "the search strategies are not approved for this run: "
            + "; ".join(errors)
            + f". Re-approve by writing {APPROVAL_ARTIFACT} against the artifacts as they stand."
        )
    return read_json_strict(run_dir / APPROVAL_ARTIFACT)
