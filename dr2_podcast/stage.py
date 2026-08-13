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
import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from dr2_podcast.artifacts import clear_candidates, read_json_strict, write_json_atomic
from dr2_podcast.manifest import Manifest, config_fingerprint
from dr2_podcast.stages import ADAPTERS, AVAILABLE_STAGE_NAMES, get_stage

RUN_CONFIG_ARTIFACT = "meta/run_config.json"

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


def load_adapters() -> None:
    """Import the adapter module for its registration side effects.

    By name rather than as a plain top-level import, because the import exists only for its side
    effects and a bare one reads as unused to a linter — and inline suppressions are banned
    house-wide.
    """
    importlib.import_module("dr2_podcast.adapters")


def runnable_stage_names() -> tuple[str, ...]:
    """Stages that are both separable and have an adapter — i.e. that would actually run."""
    load_adapters()
    return tuple(name for name in AVAILABLE_STAGE_NAMES if name in ADAPTERS)


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


def _direct_producers(run_dir: Path, name: str, substitutions: dict[str, str]) -> set[str]:
    """The stages that wrote what ``name`` will ACTUALLY read.

    An optional input that is not on disk is not read, so requiring its producer would make an
    English episode unable to run `blueprint` at all — `translate` produces the translated SOT that
    no English run has.
    """
    from dr2_podcast.stages import producer_of, resolve

    stage = get_stage(name)
    optional = resolve(stage.optional_consumes, substitutions)
    reading = list(stage.consumes) + [a for a in optional if (run_dir / a).exists()]
    return {producer for artifact in reading if (producer := producer_of(artifact))}


def _stale_upstream(
    run_dir: Path, name: str, manifest: Manifest, run_config: dict[str, Any], substitutions: dict[str, str]
) -> list[str]:
    """Every stage anywhere upstream of ``name`` that is not current, nearest first.

    TRANSITIVE, and the direct-producers-only version was wrong in a way no direct check can see
    (prepush codex 2026-08-13): change a research-scoped setting and `research` goes stale, which
    makes `blueprint` unusable — but blueprint's OWN fingerprint and files are untouched, so `draft`
    asking only about blueprint is told everything is fine. Nobody ever asks about research unless
    they invoke blueprint, and the whole point of a skip is that they do not.

    Each stage is judged by ITS OWN fingerprint: identity is scoped per stage, so comparing research
    against audio's settings would answer a question nobody asked.
    """
    stale: list[str] = []
    seen: set[str] = {name}
    frontier = sorted(_direct_producers(run_dir, name, substitutions))
    while frontier:
        producer = frontier.pop(0)
        if producer in seen:
            continue
        seen.add(producer)
        if not manifest.is_current(
            producer, config_sha256=config_fingerprint(run_config=run_config, stage=producer)
        ):
            stale.append(producer)
        frontier.extend(sorted(_direct_producers(run_dir, producer, substitutions) - seen))
    return stale


def _guard_inputs(
    run_dir: Path,
    name: str,
    manifest: Manifest,
    run_config: dict[str, Any],
    *,
    force: bool,
    substitutions: dict[str, str],
) -> None:
    """Inputs must exist AND the stages that wrote them must be current.

    Existence alone is not enough, and the gap is not hypothetical: change the model and every
    upstream record stops being current *without any file disappearing*, so a downstream stage
    would consume artifacts built under the old configuration and then record itself complete
    under the new one — a run whose manifest says it is coherent when it is not.

    ``--force`` bypasses the currency half, because the honest reading of "these inputs are what I
    want" is a decision a human can make; it does not bypass existence.
    """
    from dr2_podcast.stages import producer_of, resolve

    stage = get_stage(name)
    # A `{language}` input is optional only because an ENGLISH run has no translated SOT. For any
    # other language it is the evidence the stage is supposed to read, and treating it as optional
    # let a Japanese run reach `blueprint` — and complete — with translate never having run
    # (prepush codex 2026-08-13). The blueprint adapter continues with an empty translated summary,
    # so nothing else would have said a word about it.
    required = list(stage.consumes)
    if substitutions.get("language", "en") != "en":
        required += [
            resolved
            for pattern, resolved in zip(
                stage.optional_consumes, resolve(stage.optional_consumes, substitutions), strict=True
            )
            if "{language}" in pattern
        ]
    missing = [a for a in required if not (run_dir / a).exists()]
    if missing:
        detail = ", ".join(f"{a} (run stage {producer_of(a)!r})" for a in missing)
        raise StageError(f"stage {name!r} cannot run: missing input(s) {detail}")
    if force:
        return
    stale = _stale_upstream(run_dir, name, manifest, run_config, substitutions)
    if stale:
        raise StageError(
            f"stage {name!r} cannot run: producer stage(s) {', '.join(sorted(stale))} are not current, "
            f"so their artifacts on disk are not what this configuration would produce. Re-run them, "
            f"or pass --force to consume the artifacts as they stand."
        )


def _resolve(name: str) -> None:
    load_adapters()
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
    removed = clear_candidates(run_dir)
    # The manifest is loaded BEFORE the config is replaced. Writing the config first meant that a
    # corrupt manifest left the run described by parameters its artifacts were not generated from —
    # the command reported failure having already changed the run's source of truth.
    manifest = Manifest.load(run_dir)

    # And the config is not COMMITTED until every guard has passed. Committing it first meant that
    # `--topic X` on a stage whose producers are stale rewrote meta/run_config.json, then refused —
    # leaving the run renamed, every completed stage non-current, and nothing actually run. The
    # prospective config drives the checks; only a stage that is really going to run writes it.
    config_path = run_dir / RUN_CONFIG_ARTIFACT
    existing = read_json_strict(config_path, schema="run_config") if config_path.exists() else None
    if existing is None and new_config is None:
        load_run_config(run_dir)  # raises with the message that says how to create one
    prospective = {**(existing or {}), **(new_config or {})}
    fingerprint = config_fingerprint(run_config=prospective, stage=name)

    substitutions = {"language": str(prospective.get("language", ""))}
    if manifest.is_current(name, config_sha256=fingerprint) and not force:
        # Guarded BEFORE the skip, not only before a run. A stage's own record saying "current" is
        # not enough: identity is scoped per stage, so a setting that only url_validation reads
        # leaves blueprint's fingerprint and its recorded input hashes untouched while its producer
        # goes stale — and the skip reported a stage as current on top of a stale producer, which
        # is precisely the incoherent manifest the guard exists to prevent (prepush codex
        # 2026-08-13). The guard walks the whole chain, not just the direct producers: an earlier
        # version of this comment claimed a stale producer's own consumers would be guarded when
        # they were asked for, which is exactly what a skip means nobody does.
        _guard_inputs(run_dir, name, manifest, prospective, force=force, substitutions=substitutions)
        return f"{name}: already current, skipped (use --force to re-run)"

    _guard_inputs(run_dir, name, manifest, prospective, force=force, substitutions=substitutions)

    if new_config is not None:
        write_run_config(run_dir, **new_config)
    run_config = load_run_config(run_dir)

    from dr2_podcast import config as app_config

    # `or`, not a getattr default: config.py defines SMART_MODEL as "" when MODEL_NAME is unset
    # (config.py:9), so the attribute exists and is empty — which the manifest schema rejects at
    # minLength 1, aborting the save before the adapter ever runs.
    manifest.start(name, model=getattr(app_config, "SMART_MODEL", "") or "unknown", config_sha256=fingerprint)
    manifest.save()
    try:
        ADAPTERS[name](run_dir, run_config)
        # Output hashing is inside the try on purpose: an adapter that returns normally without
        # writing what it declared raises here, and if that escaped the handler the manifest left
        # on disk would still say "running" — a stage reported as live after the process exited.
        # The attempt is recorded only once that succeeds, so one execution never leaves both a
        # "complete" and a "failed" attempt in the history.
        staled = manifest.complete(name, substitutions)
        manifest.record_attempt(name, "complete")
    except BaseException as exc:
        # BaseException, not Exception: Ctrl-C during a 40-minute stage would otherwise leave the
        # persisted manifest saying "running" forever and skip downstream invalidation, even though
        # the adapter may already have rewritten outputs. The interrupt is re-raised unchanged, so
        # a deliberate stop still stops — it is just recorded on the way past.
        detail = str(exc)[:200] or type(exc).__name__
        manifest.record_attempt(name, "failed", detail)
        manifest.fail(name, detail)
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
    runnable = runnable_stage_names()
    parser.add_argument(
        "stage",
        nargs="?",
        help=(
            f"stage to run. Separable: {', '.join(AVAILABLE_STAGE_NAMES)}. "
            + (
                f"Runnable now: {', '.join(runnable)}."
                if runnable
                else "Runnable now: NONE — no stage adapter is registered yet, so every stage refuses; "
                "see the module docstring in dr2_podcast/stage.py."
            )
        ),
    )
    parser.add_argument("--run", required=True, type=Path, help="run directory")
    parser.add_argument("--force", action="store_true", help="re-run even if the stage is current")
    parser.add_argument("--topic", help="create or update meta/run_config.json with this topic")
    parser.add_argument("--language", help="episode language (default: ja, for a NEW run config)")
    parser.add_argument("--target-length", type=int, help="target minutes (default: 25, for a NEW run config)")
    parser.add_argument("--status", action="store_true", help="print every stage's status and exit")
    return parser


def _merged_run_config(run_dir: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    """The run config to write, or None to leave the existing one alone.

    Omitted options keep whatever the run already has. Copying the parser defaults in unconditionally
    meant that changing the topic of an English 60-minute run silently turned it into a Japanese
    25-minute one — and since those fields are part of stage identity, it would also invalidate every
    completed stage on the way past. Defaults apply only when there is no run config yet.
    """
    supplied = (args.topic, args.language, args.target_length)
    if all(value is None for value in supplied):
        return None
    path = run_dir / RUN_CONFIG_ARTIFACT
    existing = read_json_strict(path, schema="run_config") if path.exists() else {}
    topic = args.topic if args.topic is not None else existing.get("topic")
    if topic is None:
        raise StageError("--language/--target-length update an existing run config; this run has none. Pass --topic.")
    # `is not None`, never truthiness: `--target-length 0` is an invalid request, not an absent one,
    # and it has to reach the schema that rejects it rather than being replaced by a default.
    return {
        "topic": topic,
        "language": args.language if args.language is not None else existing.get("language", "ja"),
        "target_length_minutes": (
            args.target_length if args.target_length is not None else existing.get("target_length_minutes", 25)
        ),
    }


def _stale_producers(run_dir: Path, name: str, manifest: Manifest, run_config: dict[str, Any] | None) -> list[str]:
    """What the runner would refuse this stage for — the same transitive walk, for the status view."""
    substitutions = {"language": str((run_config or {}).get("language", ""))}
    return sorted(_stale_upstream(run_dir, name, manifest, run_config or {}, substitutions))


def _print_status(run_dir: Path) -> int:
    manifest = Manifest.load(run_dir)
    path = run_dir / RUN_CONFIG_ARTIFACT
    run_config = read_json_strict(path, schema="run_config") if path.exists() else None
    load_adapters()
    for name in AVAILABLE_STAGE_NAMES:
        fingerprint = config_fingerprint(run_config=run_config, stage=name)
        # A stage's own hashes and fingerprint are not the whole answer. A producer can go stale on
        # its own scoped configuration without its artifact bytes moving, and printing the consumer
        # as "current" then contradicts what running it actually does — _guard_inputs refuses it
        # (prepush codex 2026-08-13). The status view and the runner must agree, or the status view
        # is worse than none.
        blocked = _stale_producers(run_dir, name, manifest, run_config)
        if manifest.is_current(name, config_sha256=fingerprint) and blocked:
            current = f"not current (producer: {', '.join(blocked)})"
        else:
            current = "current" if manifest.is_current(name, config_sha256=fingerprint) else "not current"
        reason = manifest.record_for(name).get("stale_reason") or ""
        adapter = "" if name in ADAPTERS else "  [no adapter]"
        print(f"  {name:<14} {manifest.status(name):<9} {current}{adapter}{'  — ' + reason if reason else ''}")
    if not runnable_stage_names():
        print("\nNo stage adapter is registered, so no stage can run yet. See dr2_podcast/stage.py.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir: Path = args.run
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory", file=sys.stderr)
        return 2
    # `is not None`, not truthiness: `--topic ""` is an invalid request, not an omitted option, and
    # silently falling back to the previous topic would run the stage against parameters nobody asked
    # for. An empty topic reaches the schema and is rejected there.
    try:
        # Inside the handler: it reads and schema-checks the existing run config, so a corrupt one
        # owes the same ERROR line and exit code as any other artifact failure.
        new_config = _merged_run_config(run_dir, args)
        # --status shares this handler: it reads the manifest and the run config, so it has exactly
        # the same failure modes as running a stage and owes the same ERROR line and exit code.
        if args.status:
            return _print_status(run_dir)
        if args.stage is None:
            raise StageError("no stage given. Pass a stage name, or --status to see them all.")
        print(run_stage(run_dir, args.stage, force=args.force, new_config=new_config))
    except Exception as exc:
        # Broad on purpose, and only at the command-line boundary. A backend that is down, a CrewAI
        # error, an HTTP failure — these are ordinary operational outcomes for a stage, and the
        # manifest has already recorded the failure by the time it reaches here, so a traceback adds
        # nothing a user can act on. KeyboardInterrupt and SystemExit derive from BaseException and
        # still pass through, so a deliberate stop stays a stop.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
