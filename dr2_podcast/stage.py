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
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from dr2_podcast.artifacts import ArtifactError, clear_candidates, read_json_strict, write_json_atomic
from dr2_podcast.manifest import Manifest, config_fingerprint
from dr2_podcast.stages import AVAILABLE_STAGE_NAMES, get_stage

RUN_CONFIG_ARTIFACT = "meta/run_config.json"

#: stage name -> callable(run_dir, run_config) -> None. A stage writes its own artifacts; the
#: runner hashes and records them afterwards from the graph's declaration.
ADAPTERS: dict[str, Callable[[Path, dict[str, Any]], None]] = {}


class StageError(RuntimeError):
    """Raised instead of proceeding. Every guard in this module fails closed."""


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


def _guard_inputs(run_dir: Path, name: str) -> None:
    missing = [a for a in get_stage(name).consumes if not (run_dir / a).exists()]
    if missing:
        from dr2_podcast.stages import producer_of

        detail = ", ".join(f"{a} (run stage {producer_of(a)!r})" for a in missing)
        raise StageError(f"stage {name!r} cannot run: missing input(s) {detail}")


def _resolve(name: str) -> None:
    stage = get_stage(name)
    if not stage.available:
        raise StageError(f"stage {name!r} is not separable yet — {stage.unavailable_reason}")
    if name not in ADAPTERS:
        raise StageError(
            f"stage {name!r} has no adapter yet. Its phase still takes live objects from the phase "
            f"before it, so it cannot be driven from disk; see the module docstring."
        )


def run_stage(run_dir: Path, name: str, *, force: bool = False) -> str:
    """Run one stage. Returns a human-readable outcome line.

    Order matters and each step is a guard: resolve, then skip-if-current, then check inputs, then
    run. A stage that is already current is not re-run without ``--force``, because re-running it
    would stale everything downstream of it for no reason.
    """
    _resolve(name)
    removed = clear_candidates(run_dir)
    manifest = Manifest.load(run_dir)
    fingerprint = config_fingerprint()

    if manifest.is_current(name, config_sha256=fingerprint) and not force:
        return f"{name}: already current, skipped (use --force to re-run)"

    _guard_inputs(run_dir, name)
    run_config = load_run_config(run_dir)

    from dr2_podcast import config as app_config

    manifest.start(name, model=getattr(app_config, "SMART_MODEL", "unknown"), config_sha256=fingerprint)
    manifest.save()
    try:
        ADAPTERS[name](run_dir, run_config)
    except Exception as exc:
        manifest.record_attempt(name, "failed", str(exc)[:200])
        manifest.fail(name, str(exc)[:200])
        manifest.save()
        raise
    manifest.record_attempt(name, "complete")
    staled = manifest.complete(name)
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
    fingerprint = config_fingerprint()
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
    try:
        if args.topic:
            write_run_config(
                run_dir, topic=args.topic, language=args.language, target_length_minutes=args.target_length
            )
        print(run_stage(run_dir, args.stage, force=args.force))
    except (StageError, ArtifactError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
