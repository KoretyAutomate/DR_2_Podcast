"""``python -m dr2_podcast.stage <name> --run <dir>`` — run one stage and exit.

PLAN.md Step 1. Claude cannot orchestrate a single 87-minute subprocess, so the run becomes stages
whose contract is files on disk: each one reads what it needs from the run directory, writes its
artifacts atomically, and records itself in the manifest. No stage calls Claude; Claude calls stages.

This module owns the *orchestration* — resolving a stage, refusing one that is not separable yet,
skipping one that is already current, guarding its inputs, recording attempts, and reporting what
its completion made stale. Executing a stage is an **adapter**, registered in :data:`ADAPTERS`.

**The adapters are the remaining half of Step 1, and they are a refactor, not a wiring exercise.**
Every phase today takes live Python objects from the phase before it — ``phase_0_framing`` receives
``framing_task_ref`` / ``framing_agent_ref`` and four more CrewAI refs the flow constructed,
``phase_5_script_draft`` receives ``bp_inventory``, ``phase_6_polish`` receives
``script_draft_text`` — and ``phase_7_audit`` takes task refs outright
(``pipeline_flow.py:808``). None of that survives a process boundary. Turning each phase into a
function of ``(run_dir, run_config)`` is what makes the stage CLI real, and it is deliberately not
faked here: an unregistered stage says exactly what it needs rather than pretending to run.
"""

from __future__ import annotations

import argparse
import fcntl
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from dr2_podcast.artifacts import ArtifactError, clear_candidates, read_json_strict, write_json_atomic
from dr2_podcast.manifest import Manifest, config_fingerprint
from dr2_podcast.schemas import SchemaValidationError
from dr2_podcast.stages import AVAILABLE_STAGE_NAMES, get_stage

RUN_CONFIG_ARTIFACT = "meta/run_config.json"

#: stage name -> callable(run_dir, run_config) -> None. A stage writes its own artifacts; the
#: runner hashes and records them afterwards from the graph's declaration.
ADAPTERS: dict[str, Callable[[Path, dict[str, Any]], None]] = {}


class StageError(RuntimeError):
    """Raised instead of proceeding. Every guard in this module fails closed."""


LOCK_ARTIFACT = "meta/.stage.lock"


@contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    """Serialise everything that touches one run's manifest.

    The manifest is a read-modify-write of a single file. Two stages running concurrently against
    one run — `sot` and `url_validation` are independent branches, so this is a real shape, not a
    hypothetical — would each load it, each save their private copy, and the later save would erase
    the other's status and attempts. They would also share `manifest.json.candidate`, and one
    process's `clear_candidates()` would delete the other's live candidate mid-write.

    Non-blocking on purpose: waiting silently on a lock held by a run that may last 40 minutes is
    worse than saying so.
    """
    lock_path = run_dir / LOCK_ARTIFACT
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise StageError(
                f"another stage is already running for {run_dir} (lock {lock_path}). "
                f"Stages of one run are serialised because they share the manifest."
            ) from exc
        yield


def register(name: str) -> Callable[[Callable[[Path, dict[str, Any]], None]], Callable[..., None]]:
    """Decorator registering a stage adapter."""
    get_stage(name)

    def _wrap(func: Callable[[Path, dict[str, Any]], None]) -> Callable[..., None]:
        ADAPTERS[name] = func
        return func

    return _wrap


def write_run_config(run_dir: Path, *, topic: str, language: str, target_length_minutes: int) -> dict[str, Any]:
    """Create the run's parameter artifact. A staged run has no caller holding these in memory."""
    config = {
        "schema_version": 1,
        "topic": topic,
        "language": language,
        "target_length_minutes": target_length_minutes,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "notes": None,
    }
    write_json_atomic(run_dir / RUN_CONFIG_ARTIFACT, config, schema="run_config")
    return config


def load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / RUN_CONFIG_ARTIFACT
    if not path.exists():
        raise StageError(
            f"{path} is missing. A staged run reads its parameters from disk — create it by passing "
            f"--topic/--language on the first stage invocation."
        )
    return read_json_strict(path, schema="run_config")


def _guard_inputs(run_dir: Path, name: str, manifest: Manifest, fingerprint: str, *, force: bool) -> None:
    """Inputs must exist AND the stages that wrote them must be current.

    Existence alone is not enough, and the gap is not hypothetical: change the model and every
    upstream record stops being current *without any file disappearing*, so a downstream stage
    would consume artifacts built under the old configuration and then record itself complete
    under the new one — a run whose manifest says it is coherent when it is not.

    ``--force`` bypasses the currency half, because the honest reading of "these inputs are what I
    want" is a decision a human can make; it does not bypass existence.
    """
    from dr2_podcast.stages import direct_producers, producer_of

    missing = [a for a in get_stage(name).consumes if not (run_dir / a).exists()]
    if missing:
        detail = ", ".join(f"{a} (run stage {producer_of(a)!r})" for a in missing)
        raise StageError(f"stage {name!r} cannot run: missing input(s) {detail}")
    if force:
        return
    stale = [
        producer
        for producer in direct_producers(name)
        if not manifest.is_current(producer, config_sha256=fingerprint)
    ]
    if stale:
        raise StageError(
            f"stage {name!r} cannot run: producer stage(s) {', '.join(sorted(stale))} are not current, "
            f"so their artifacts on disk are not what this configuration would produce. Re-run them, "
            f"or pass --force to consume the artifacts as they stand."
        )


def _resolve(name: str) -> None:
    stage = get_stage(name)
    if not stage.available:
        raise StageError(f"stage {name!r} is not separable yet — {stage.unavailable_reason}")
    if name not in ADAPTERS:
        raise StageError(
            f"stage {name!r} has no adapter yet. Its phase still takes live objects from the phase "
            f"before it, so it cannot be driven from disk; see the module docstring."
        )


def run_stage(
    run_dir: Path,
    name: str,
    *,
    force: bool = False,
    new_config: dict[str, Any] | None = None,
) -> str:
    """Run one stage. Returns a human-readable outcome line.

    Order matters and each step is a guard: resolve, then skip-if-current, then check inputs, then
    run. A stage that is already current is not re-run without ``--force``, because re-running it
    would stale everything downstream of it for no reason.

    ``new_config`` writes ``meta/run_config.json`` **inside the same lock as the run**. Writing it
    outside would let an invocation rewrite the topic of a run that is already executing: the
    running stage would carry on with the old parameters in memory while the run directory
    described the new ones, and both processes would write through the same ``.candidate`` path.
    """
    _resolve(name)
    with run_lock(run_dir):
        return _run_stage_locked(run_dir, name, force=force, new_config=new_config)


def _run_stage_locked(run_dir: Path, name: str, *, force: bool, new_config: dict[str, Any] | None = None) -> str:
    if new_config is not None:
        write_run_config(run_dir, **new_config)
    removed = clear_candidates(run_dir)
    manifest = Manifest.load(run_dir)
    # The run config is read BEFORE the currency check because it is part of currency: a stage
    # completed for a different topic is not current for this one.
    run_config = load_run_config(run_dir)
    fingerprint = config_fingerprint(run_config=run_config)

    if manifest.is_current(name, config_sha256=fingerprint) and not force:
        return f"{name}: already current, skipped (use --force to re-run)"

    _guard_inputs(run_dir, name, manifest, fingerprint, force=force)

    from dr2_podcast import config as app_config

    manifest.start(name, model=getattr(app_config, "SMART_MODEL", "unknown"), config_sha256=fingerprint)
    manifest.save()
    try:
        ADAPTERS[name](run_dir, run_config)
        manifest.record_attempt(name, "complete")
        # Output hashing is inside the try on purpose: an adapter that returns normally without
        # writing what it declared raises here, and if that escaped the handler the manifest left
        # on disk would still say "running" — a stage reported as live after the process exited.
        staled = manifest.complete(name)
    except Exception as exc:
        manifest.record_attempt(name, "failed", str(exc)[:200])
        manifest.fail(name, str(exc)[:200])
        # A failed rerun may already have rewritten some outputs, so everything behind it has to be
        # invalidated too — otherwise a descendant whose own inputs happen not to have moved stays
        # falsely current behind a stage that is known to be broken.
        manifest.invalidate_downstream(name)
        manifest.save()
        raise
    manifest.save()

    note = f" (cleared {len(removed)} stale candidate(s))" if removed else ""
    if staled:
        return f"{name}: complete{note}; now stale: {', '.join(staled)}"
    return f"{name}: complete{note}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dr2_podcast.stage",
        description="Run one pipeline stage against a run directory.",
    )
    parser.add_argument("stage", help=f"stage to run; available: {', '.join(AVAILABLE_STAGE_NAMES)}")
    parser.add_argument("--run", required=True, type=Path, help="run directory")
    parser.add_argument("--force", action="store_true", help="re-run even if the stage is current")
    parser.add_argument("--topic", help="create meta/run_config.json with this topic")
    parser.add_argument("--language", default="ja", help="episode language (default: ja)")
    parser.add_argument("--target-length", type=int, default=25, help="target minutes (default: 25)")
    parser.add_argument("--status", action="store_true", help="print every stage's status and exit")
    return parser


def _print_status(run_dir: Path) -> int:
    manifest = Manifest.load(run_dir)
    path = run_dir / RUN_CONFIG_ARTIFACT
    run_config = read_json_strict(path, schema="run_config") if path.exists() else None
    fingerprint = config_fingerprint(run_config=run_config)
    for name in AVAILABLE_STAGE_NAMES:
        current = "current" if manifest.is_current(name, config_sha256=fingerprint) else "not current"
        reason = manifest.record_for(name).get("stale_reason") or ""
        print(f"  {name:<14} {manifest.status(name):<9} {current}{'  — ' + reason if reason else ''}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir: Path = args.run
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory", file=sys.stderr)
        return 2
    if args.status:
        return _print_status(run_dir)
    # `is not None`, not truthiness: `--topic ""` is an invalid request, not an omitted option, and
    # silently falling back to the previous topic would run the stage against parameters nobody asked
    # for. An empty topic reaches the schema and is rejected there.
    new_config = (
        None
        if args.topic is None
        else {"topic": args.topic, "language": args.language, "target_length_minutes": args.target_length}
    )
    try:
        print(run_stage(run_dir, args.stage, force=args.force, new_config=new_config))
    except (StageError, ArtifactError, SchemaValidationError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
