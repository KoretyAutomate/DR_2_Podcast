"""Running a stage whose author is Claude.

PLAN.md "Runner decision — `claude -p`". Six pieces of the four-role allocation are built, tested
and waiting for exactly one thing: a way to hand a stage to Claude and get its artifacts back. The
frozen prior, step 9's Bayesian update, the derived blueprint for a real episode, and the three
tier-2 semantic reads are all blocked on this file and nothing else.

Three constraints shape it, and all three are recorded decisions rather than preferences:

* **A plain-text prompt, not a slash command.** `claude -p "<text>"` treats the message as literal
  text, so `/skillname` never reaches the skill resolver. The skill has to be model-invocable and
  the prompt has to be prose that triggers it.
* **No permission prompt can ever fire, and the tool list is genuinely closed.** An unattended
  run has nobody to answer at minute 40, so every turn names its tools explicitly. That takes two
  flags, not one: `--allowedTools` pre-approves, which is what keeps a turn from stopping to ask,
  but it does not take anything away — Read, Glob and Grep never ask in the first place, so a turn
  granted only `Write` still held them, absolute paths included (prepush codex 2026-08-20).
  `--tools` is the half that removes: a tool outside that list is not available to the turn at all.
  Nor is the built-in set the whole set — this machine's user config carries MCP servers (codex,
  codebase-memory, localcrew, claude-memory), and a turn restricted to `Write` still asked one of
  them to read a file for it, verified. So every turn also starts with MCP emptied. A stage that
  needs a tool outside its list fails loudly instead of hanging forever on a question nobody reads.
* **A turn that only has to judge holds no tools at all.** `Write` takes ABSOLUTE paths, so the
  scratch cwd a stage runs in confines nothing — and these prompts carry a topic somebody typed
  into the Web UI plus text an LLM generated from it, either of which can carry an instruction
  (prepush codex 2026-08-20). So the default shape here is `ask_for_json`: the turn ANSWERS, Python
  parses and validates the answer, and Python decides what path it lands on. Granting `Write` is
  for a stage that genuinely must produce files, and it grants writing ANYWHERE this process can
  reach.
* **The model is configuration, never an inheritance.** A judgement's identity includes who made
  it, so `DR2_CLAUDE_MODEL` is required and every turn passes `--model`. Letting it fall back to
  the CLI's current default records the same empty value before and after that default changes,
  which is a fingerprint that cannot see the one thing it claims to (prepush codex 2026-08-20).
* **Outcome comes from the turn's completion, never from the spawn.** `web_ui.py` marks a task
  running the moment `Popen` returns; a run that no-ops on its first turn would log as success.
  MulmoTerminal shipped exactly that bug. Here, success means the declared artifacts exist and
  changed — the same rule every other stage in this pipeline is judged by.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dr2_podcast.artifacts import ArtifactError

#: The closed tool list an unattended authoring turn gets. Read/Write/Edit because it writes an
#: artifact; Glob/Grep because it has to find the evidence it is writing about. NOT Bash: a stage
#: that needs to run a command is a stage Python should be running, and an unattended shell is the
#: one permission nobody can take back.
DEFAULT_ALLOWED_TOOLS: tuple[str, ...] = ("Read", "Write", "Edit", "Glob", "Grep")

#: The tool list for a turn that only has to ANSWER — empty, and `--tools ""` is the CLI's way of
#: saying "no tools at all" (`claude --help`: `Use "" to disable all tools`). Verified against the
#: real CLI on 2026-08-20: a turn under these flags returns its JSON and holds no filesystem
#: capability whatsoever, so there is no path by which an instruction hidden in the prompt reaches
#: a file. This is what a judgement stage gets.
NO_TOOLS: tuple[str, ...] = ()

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

#: The environment variable naming the model an authoring turn runs on. REQUIRED — see
#: :func:`resolve_model` for why there is deliberately no fallback.
CLAUDE_MODEL_ENV = "DR2_CLAUDE_MODEL"

#: The MCP configuration an authoring turn is launched with: none. Passed with --strict-mcp-config
#: so the user's own servers are ignored rather than merged — see _command().
_NO_MCP_SERVERS = '{"mcpServers": {}}'


def resolve_model() -> str:
    """The model this turn is pinned to, or a failure saying so — never an inherited default.

    Two rounds of review landed on requiring it. The first pinned the model when the variable
    happened to be set and passed no ``--model`` when it did not (prepush codex 2026-08-17); the
    second pointed out that this only looks like a fix (prepush codex 2026-08-20). An unset value
    means the frozen prior is authored by whichever model the CLI or the account currently
    defaults to, and the manifest fingerprints the variable — so it records the same empty string
    before and after a CLI upgrade, and a judgement made by a different model reads as current.

    A stage whose output is a JUDGEMENT cannot have "whichever model happened to be default" in
    its identity. So this is configuration, it is required, and its absence is loud: `.env.example`
    carries it, and a machine that has not set it fails here rather than authoring something no
    later run can reproduce.
    """
    model = os.environ.get(CLAUDE_MODEL_ENV, "").strip()
    if not model:
        raise ClaudeUnavailable(
            f"{CLAUDE_MODEL_ENV} is not set, so an authoring turn would run on whichever model the "
            f"Claude CLI currently defaults to. The stage's output is a judgement and the run "
            f"manifest fingerprints the model that made it, so an inherited default would let a CLI "
            f"or account change author a different judgement while existing artifacts still read as "
            f"current. Set {CLAUDE_MODEL_ENV} in .env (see .env.example)."
        )
    return model


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


def _command(
    prompt: str, allowed_tools: tuple[str, ...], model: str, binary: str | None = None
) -> list[str]:
    """The argv. A list, never a shell string — the prompt carries a topic a user typed.

    ``model`` is required rather than defaulted, which is the whole point: there is no way to build
    an argv here that leaves the model to the CLI. :func:`resolve_model` is where an unset value
    becomes a failure, and this refuses an empty one so a future caller cannot route around it.

    ``binary`` is the RESOLVED path when there is one. It matters because the turn runs with cwd set
    to the run directory: a relative DR2_CLAUDE_BINARY would resolve against the pipeline's cwd and
    then be launched from somewhere else entirely, so the executable validated and logged would not
    be the one that ran (prepush codex 2026-08-17).
    """
    if not model.strip():
        raise ClaudeUnavailable("refusing to build an authoring turn with no model pinned")
    tools = ",".join(allowed_tools)
    # Both flags carry the SAME closed list, because they answer different questions and only one
    # of them is a guarantee: --tools decides what exists for this turn, --allowedTools decides
    # what runs without asking. Availability without approval would hang; approval without
    # availability limiting is what let a write-only turn read (prepush codex 2026-08-20).
    argv = [binary or CLAUDE_BINARY, "-p", prompt, "--tools", tools, "--allowedTools", tools]
    # And the closed list is only closed if it is the whole list. Measured on this machine the same
    # day: a turn given --tools Write reached the codex MCP server and had it read the file the
    # stage was supposed to be blind to. An empty --mcp-config plus --strict-mcp-config is what
    # actually leaves the turn holding nothing but the tools named above.
    argv += ["--strict-mcp-config", "--mcp-config", _NO_MCP_SERVERS]
    argv += ["--model", model]
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
    model = resolve_model()
    binary = resolve_binary()
    logger.info("authoring turn via %s on %s", binary, model)
    try:
        completed = subprocess.run(
            _command(prompt, allowed_tools, model, binary),
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

    **Granting ``Write`` grants writing ANYWHERE this process can reach.** The CLI's `Write` takes
    absolute paths, so neither ``run_dir`` nor a scratch cwd bounds it, and the prompt of a stage
    like this one carries text a user typed and text a model generated (prepush codex 2026-08-20).
    Use :func:`ask_for_json` unless the stage genuinely has to put files on disk itself; that one
    hands the answer back through stdout and lets Python choose the path.

    ``expected`` is what makes this safe to run unattended. The exit code says the process ended;
    it does not say the work happened, and a turn that reads for ten minutes and writes nothing
    exits 0. So the artifacts are snapshotted before and compared after: each one must exist, be
    non-empty, and — if it was already there — have changed. Anything less is a failure with a
    reason, never a silent pass.
    """
    if not any(tool in allowed_tools for tool in ("Write", "Edit")):
        # An empty list means `--tools ""`, which disables everything; a list without a writing tool
        # is the same outcome by a different route. Either way the turn cannot produce what it is
        # being spawned to produce, so say so now rather than after the wall-clock ceiling.
        raise ClaudeUnavailable(
            f"refusing to spawn an artifact-authoring turn holding no writing tool: {allowed_tools!r}"
        )
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


def ask_for_json(
    prompt: str,
    *,
    cwd: Path,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    """Ask Claude a question whose answer is a JSON document, and get it back through stdout.

    The turn holds NO tools (see :data:`NO_TOOLS`), which is the whole point: a judgement stage
    needs no filesystem capability, and a capability it does not hold is the only kind an
    instruction smuggled into the prompt cannot use. Python parses what comes back, the caller
    validates it against a schema, and the caller — never the model — decides where it is written.

    The same rule as :func:`author_artifacts` decides the outcome, in the only form available to a
    turn that writes nothing: the answer must BE a JSON document. A turn that exits 0 having said
    nothing, or having said something plausible in prose, did not do the work.
    """
    turn = run_turn(prompt, cwd=cwd, allowed_tools=NO_TOOLS, timeout=timeout)
    if turn.timed_out:
        raise ClaudeUnavailable(
            f"the authoring turn did not finish within {timeout}s. Nothing it may have half-said is "
            f"trusted; re-run the stage."
        )
    if not turn.spoke:
        raise ArtifactError(
            f"the Claude-authored stage said nothing at all (exit {turn.returncode}), so there is no "
            f"judgement to record."
        )
    try:
        return json.loads(_json_payload(turn.stdout))
    except ValueError as exc:
        raise ArtifactError(
            f"the Claude-authored stage did not answer with JSON ({exc}). It exited "
            f"{turn.returncode} and said: {turn.stdout.strip()[:300]}"
        ) from exc


#: A fenced block, which is how the CLI actually replies — measured 2026-08-20, the answer came back
#: as ```json …``` even under "output the JSON and nothing else".
_FENCED = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _json_payload(stdout: str) -> str:
    """The JSON document inside a reply that may also carry a fence or a sentence around it."""
    text = stdout.strip()
    fenced = _FENCED.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in the reply")
    return text[start : end + 1]


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
