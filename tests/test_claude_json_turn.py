"""The toolless authoring turn — `claude_runner.ask_for_json`.

prepush codex 2026-08-20 [P1]. The frozen prior ran with a pre-approved `Write`, and `Write` takes
ABSOLUTE paths: the scratch cwd the stage ran in constrained nothing, while the prompt carried a
topic somebody typed into the Web UI and a framing an LLM generated from it. The answer is not a
tighter path check — it is that a judgement stage needs no filesystem capability at all. The turn
ANSWERS, Python parses, the caller validates, and Python decides what path the result lands on.

What is pinned here: the turn is spawned holding nothing, and the same rule that governs a writing
turn still decides the outcome — a turn that exits 0 having said nothing, or having said something
plausible in prose, did not do the work.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from dr2_podcast.artifacts import ArtifactError
from dr2_podcast.claude_runner import CLAUDE_MODEL_ENV, ClaudeUnavailable, ask_for_json

ANSWER = {"prior_level": "低い", "topic": "ビタミンDと骨折"}


@pytest.fixture(autouse=True)
def _configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The authoring model is required configuration — a judgement records who made it."""
    monkeypatch.setenv(CLAUDE_MODEL_ENV, "claude-opus-5")


def _replying(monkeypatch: pytest.MonkeyPatch, stdout: str, *, returncode: int = 0) -> dict:
    seen: dict = {}

    def _run(argv, **kwargs):
        seen["argv"] = argv
        seen["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(argv, returncode, stdout, "")

    monkeypatch.setattr(subprocess, "run", _run)
    return seen


# --------------------------------------------------------------------------- #
# The turn holds nothing
# --------------------------------------------------------------------------- #
def test_the_turn_is_spawned_with_no_tools_at_all(tmp_path: Path, monkeypatch) -> None:
    seen = _replying(monkeypatch, json.dumps(ANSWER))
    ask_for_json("judge this", cwd=tmp_path)

    argv = seen["argv"]
    for flag in ("--tools", "--allowedTools"):
        assert argv[argv.index(flag) + 1] == "", flag
    # And the built-in set is not the whole set: a turn restricted to nothing still reached this
    # machine's MCP servers until --strict-mcp-config was added (PLAN.md, 2026-08-20).
    assert "--strict-mcp-config" in argv
    assert json.loads(argv[argv.index("--mcp-config") + 1]) == {"mcpServers": {}}


def test_nothing_is_written_anywhere(tmp_path: Path, monkeypatch) -> None:
    """The property the finding asks for, stated directly: this call writes no file."""
    _replying(monkeypatch, json.dumps(ANSWER))
    ask_for_json("judge this", cwd=tmp_path)
    assert list(tmp_path.rglob("*")) == []


# --------------------------------------------------------------------------- #
# Getting the document out of the reply
# --------------------------------------------------------------------------- #
# Measured against the real CLI on 2026-08-20: asked for "the JSON and nothing else", it answered
# inside a ```json fence. A parser that only accepts a bare object would fail on every real run.
@pytest.mark.parametrize(
    "reply",
    [
        '{"prior_level": "低い", "topic": "ビタミンDと骨折"}',
        '```json\n{"prior_level": "低い", "topic": "ビタミンDと骨折"}\n```',
        '```\n{"prior_level": "低い", "topic": "ビタミンDと骨折"}\n```',
        'Here is the prior:\n{"prior_level": "低い", "topic": "ビタミンDと骨折"}\nLet me know.',
    ],
)
def test_the_json_is_taken_out_of_whatever_the_reply_wraps_it_in(tmp_path: Path, monkeypatch, reply) -> None:
    _replying(monkeypatch, reply)
    assert ask_for_json("judge this", cwd=tmp_path) == ANSWER


def test_the_reply_is_returned_as_data_not_text(tmp_path: Path, monkeypatch) -> None:
    """So the caller validates a record against a schema rather than pattern-matching prose."""
    _replying(monkeypatch, json.dumps(ANSWER))
    assert ask_for_json("judge this", cwd=tmp_path)["prior_level"] == "低い"


# --------------------------------------------------------------------------- #
# Outcome comes from the answer, never from the exit code
# --------------------------------------------------------------------------- #
def test_a_turn_that_answers_in_prose_is_a_failure(tmp_path: Path, monkeypatch) -> None:
    """The MulmoTerminal rule in the only form available to a turn that writes nothing: exiting 0
    while saying something plausible is not doing the work."""
    _replying(monkeypatch, "I have considered the topic and set the prior to 低い.")
    with pytest.raises(ArtifactError, match="did not answer with JSON"):
        ask_for_json("judge this", cwd=tmp_path)


def test_the_failure_quotes_what_the_turn_actually_said(tmp_path: Path, monkeypatch) -> None:
    _replying(monkeypatch, "I could not find the research framing.")
    with pytest.raises(ArtifactError, match="could not find the research framing"):
        ask_for_json("judge this", cwd=tmp_path)


def test_a_silent_turn_says_so(tmp_path: Path, monkeypatch) -> None:
    _replying(monkeypatch, "   ")
    with pytest.raises(ArtifactError, match="said nothing at all"):
        ask_for_json("judge this", cwd=tmp_path)


def test_broken_json_is_a_failure_not_a_partial_record(tmp_path: Path, monkeypatch) -> None:
    _replying(monkeypatch, '{"prior_level": "低い", ')
    with pytest.raises(ArtifactError, match="did not answer with JSON"):
        ask_for_json("judge this", cwd=tmp_path)


def test_a_nonzero_exit_with_a_real_answer_still_succeeds(tmp_path: Path, monkeypatch) -> None:
    """Symmetry with the writing path: if the exit code cannot prove success it cannot prove
    failure either. The answer decides."""
    _replying(monkeypatch, json.dumps(ANSWER), returncode=1)
    assert ask_for_json("judge this", cwd=tmp_path) == ANSWER


def test_a_hung_turn_is_bounded_and_its_output_is_not_trusted(tmp_path: Path, monkeypatch) -> None:
    def _hangs(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 0), output='{"prior', stderr="")

    monkeypatch.setattr(subprocess, "run", _hangs)
    with pytest.raises(ClaudeUnavailable, match="did not finish within"):
        ask_for_json("judge this", cwd=tmp_path, timeout=5)
