"""Running a stage whose author is Claude.

PLAN.md "Runner decision — `claude -p`". Six pieces of the four-role allocation are built, tested
and waiting for exactly one thing: a way to hand a stage to Claude and get its artifacts back. The
frozen prior, step 9's Bayesian update, the derived blueprint for a real episode, and the three
tier-2 semantic reads are all blocked on this file and nothing else.

Three constraints shape it, and all three are recorded decisions rather than preferences:

* **A plain-text prompt, not a slash command.** `claude -p "<text>"` treats the message as literal
  text, so `/skillname` never reaches the skill resolver. The skill has to be model-invocable and
  the prompt has to be prose that triggers it.
* **No permission prompt can ever fire.** An unattended run has nobody to answer at minute 40, so
  the tool list is explicit and closed. A stage that needs a tool outside it fails loudly instead of
  hanging forever on a question nobody will read.
* **Outcome comes from the turn's completion, never from the spawn.** `web_ui.py` marks a task
  running the moment `Popen` returns; a run that no-ops on its first turn would log as success.
  MulmoTerminal shipped exactly that bug. Here, success means the declared artifacts exist and
  changed — the same rule every other stage in this pipeline is judged by.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dr2_podcast.artifacts import ArtifactError

#: The closed tool list an unattended authoring turn gets. Read/Write/Edit because it writes an
#: artifact; Glob/Grep because it has to find the evidence it is writing about. NOT Bash: a stage
#: that needs to run a command is a stage Python should be running, and an unattended shell is the
#: one permission nobody can take back.
DEFAULT_ALLOWED_TOOLS: tuple[str, ...] = ("Read", "Write", "Edit", "Glob", "Grep")

#: Wall-clock ceiling for one authoring turn. Generous, because reading a source of truth and
#: writing a judgement about it is not fast — but finite, because a hung turn in an unattended run
#: is indistinguishable from a slow one until someone looks in the morning.
DEFAULT_TIMEOUT_SECONDS = 1800

#: The CLI to invoke. Defaults to `claude` on PATH, which is where it normally is — REQUIRING the
#: variable would fail on every machine that never set it, and .env is the single source of truth
#: for model and backend configuration, not for the location of a tool. What is not acceptable is
#: doing it SILENTLY (prepush codex 2026-08-17): an unattended run picking up whatever `claude`
#: happens to be first on PATH should say which one, so resolve() reports the absolute path and
#: every failure names it.
logger = logging.getLogger(__name__)

CLAUDE_BINARY = os.environ.get("DR2_CLAUDE_BINARY", "claude")

#: The model the authoring turn runs on, pinned rather than inherited. Without this the stage used
#: whichever default the CLI or the account currently selects — so a changed default would change
#: the authored prior while the manifest, fingerprinting only Smart/vLLM settings, still called the
#: old output current (prepush codex 2026-08-17). It IS part of stage identity: see
#: manifest.CONTENT_ENV_KEYS. Unset means "whatever the CLI picks", which is honest for a laptop and
#: wrong for a pipeline, so an unset value is hashed as unset and a later pin invalidates.
CLAUDE_MODEL = os.environ.get("DR2_CLAUDE_MODEL", "")


def resolve_binary() -> str:
    """The absolute path of the CLI that will be invoked, or a failure naming what was looked for."""
    import shutil

    found = shutil.which(CLAUDE_BINARY)
    if not found:
        raise ClaudeUnavailable(
            f"{CLAUDE_BINARY!r} is not on PATH, so no Claude-authored stage can run. Set "
            f"DR2_CLAUDE_BINARY to its absolute path if it lives somewhere else."
        )
    return found


class ClaudeUnavailable(ArtifactError):
    """The CLI could not be run at all — not installed, not on PATH, or it never answered."""


@dataclass(frozen=True)
class ClaudeTurn:
    """What one `claude -p` invocation did."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def spoke(self) -> bool:
        """Whether the turn produced any output at all. Not the same as having done the work."""
        return bool(self.stdout.strip())


def _command(prompt: str, allowed_tools: tuple[str, ...], binary: str | None = None) -> list[str]:
    """The argv. A list, never a shell string — the prompt carries a topic a user typed.

    ``binary`` is the RESOLVED path when there is one. It matters because the turn runs with cwd set
    to the run directory: a relative DR2_CLAUDE_BINARY would resolve against the pipeline's cwd and
    then be launched from somewhere else entirely, so the executable validated and logged would not
    be the one that ran (prepush codex 2026-08-17).
    """
    argv = [binary or CLAUDE_BINARY, "-p", prompt, "--allowedTools", ",".join(allowed_tools)]
    if CLAUDE_MODEL:
        argv += ["--model", CLAUDE_MODEL]
    return argv


def run_turn(
    prompt: str,
    *,
    cwd: Path,
    allowed_tools: tuple[str, ...] = DEFAULT_ALLOWED_TOOLS,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> ClaudeTurn:
    """One authoring turn. Returns what happened; decides nothing about whether it worked."""
    if not prompt.strip():
        raise ClaudeUnavailable("refusing to spawn an authoring turn with an empty prompt")
    binary = resolve_binary()
    logger.info("authoring turn via %s", binary)
    try:
        completed = subprocess.run(  # noqa: S603 - argv list, shell=False, binary is a module constant
            _command(prompt, allowed_tools, binary),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ClaudeUnavailable(f"{binary!r} vanished between resolving it and running it") from exc
    except subprocess.TimeoutExpired as exc:
        return ClaudeTurn(returncode=-1, stdout=exc.stdout or "", stderr=exc.stderr or "", timed_out=True)
    return ClaudeTurn(completed.returncode, completed.stdout or "", completed.stderr or "")


def author_artifacts(
    prompt: str,
    *,
    run_dir: Path,
    expected: tuple[str, ...],
    allowed_tools: tuple[str, ...] = DEFAULT_ALLOWED_TOOLS,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> ClaudeTurn:
    """Ask Claude to write ``expected`` into ``run_dir``, and verify it actually did.

    ``expected`` is what makes this safe to run unattended. The exit code says the process ended;
    it does not say the work happened, and a turn that reads for ten minutes and writes nothing
    exits 0. So the artifacts are snapshotted before and compared after: each one must exist, be
    non-empty, and — if it was already there — have changed. Anything less is a failure with a
    reason, never a silent pass.
    """
    before = {name: _fingerprint(run_dir / name) for name in expected}
    turn = run_turn(prompt, cwd=run_dir, allowed_tools=allowed_tools, timeout=timeout)

    if turn.timed_out:
        raise ClaudeUnavailable(
            f"the authoring turn did not finish within {timeout}s. Nothing it may have half-written "
            f"is trusted; re-run the stage."
        )

    problems = []
    for name in expected:
        path = run_dir / name
        if not path.exists():
            problems.append(f"{name} was never written")
        elif path.stat().st_size == 0:
            problems.append(f"{name} is empty")
        elif before[name] is not None and _fingerprint(path) == before[name]:
            # The case the exit code cannot see: the turn ran, said something plausible, and left
            # the previous run's artifact exactly where it was.
            problems.append(f"{name} is unchanged from before the turn, so nothing authored it")
    if problems:
        raise ArtifactError(
            "the Claude-authored stage did not produce what it declared: "
            + "; ".join(problems)
            + (f". The turn exited {turn.returncode} and said: {turn.stdout.strip()[:300]}" if turn.spoke
               else f". The turn exited {turn.returncode} and said nothing at all.")
        )
    return turn


def _fingerprint(path: Path) -> tuple[int, int] | None:
    """Size and mtime, or None when the file is not there.

    Cheap on purpose — this runs either side of a turn that may take twenty minutes, and the
    question it answers is "did this change", not "how".
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_size, stat.st_mtime_ns)
