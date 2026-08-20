"""The Claude authoring runner — PLAN.md "Runner decision".

The property under test throughout is the one MulmoTerminal got wrong and PLAN.md wrote down
afterwards: **a turn that exits 0 having done nothing is a failure.** Every test here is either
that, or one of the two CLI constraints that keep an unattended run from hanging on a question.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dr2_podcast.artifacts import ArtifactError
from dr2_podcast.claude_runner import (
    DEFAULT_ALLOWED_TOOLS,
    ClaudeTurn,
    ClaudeUnavailable,
    _command,
    author_artifacts,
    run_turn,
)


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    for sub in ("research", "meta", "scripts", "audio"):
        (tmp_path / sub).mkdir()
    return tmp_path


def _spawning(monkeypatch: pytest.MonkeyPatch, *, writes=None, returncode=0, stdout="done", stderr=""):
    """Stand in for the CLI: optionally write files, then return."""
    seen: dict = {}

    def _run(argv, **kwargs):
        seen["argv"] = argv
        seen["cwd"] = kwargs.get("cwd")
        seen["timeout"] = kwargs.get("timeout")
        for relative, content in (writes or {}).items():
            path = Path(kwargs["cwd"]) / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    monkeypatch.setattr(subprocess, "run", _run)
    return seen


# --------------------------------------------------------------------------- #
# The two CLI constraints
# --------------------------------------------------------------------------- #
def test_the_prompt_is_plain_text_not_a_slash_command() -> None:
    """`claude -p "<text>"` treats the message as literal text, so a slash command never reaches
    the skill resolver. The skill is model-invocable and the prompt is prose."""
    argv = _command("Write the framing prior for this run.", DEFAULT_ALLOWED_TOOLS)
    assert argv[1] == "-p"
    assert not argv[2].startswith("/"), argv[2]


def test_the_tool_list_is_explicit_and_closed() -> None:
    """An unattended run has nobody to answer a permission prompt at minute 40."""
    argv = _command("anything", DEFAULT_ALLOWED_TOOLS)
    assert "--allowedTools" in argv
    granted = set(argv[argv.index("--allowedTools") + 1].split(","))
    assert granted == set(DEFAULT_ALLOWED_TOOLS)


def test_bash_is_not_granted() -> None:
    """A stage that needs to run a command is a stage Python should be running, and an unattended
    shell is the one permission nobody can take back."""
    assert "Bash" not in DEFAULT_ALLOWED_TOOLS


def test_the_prompt_is_an_argv_element_never_a_shell_string() -> None:
    """It carries a topic a user typed."""
    argv = _command("vitamin D; rm -rf ~", DEFAULT_ALLOWED_TOOLS)
    assert "vitamin D; rm -rf ~" in argv


# --------------------------------------------------------------------------- #
# Outcome comes from the artifacts, never from the exit code
# --------------------------------------------------------------------------- #
def test_a_turn_that_writes_what_it_promised_succeeds(run_dir: Path, monkeypatch) -> None:
    _spawning(monkeypatch, writes={"research/framing_prior.json": '{"prior_level": "低い"}'})
    turn = author_artifacts("write the prior", run_dir=run_dir, expected=("research/framing_prior.json",))
    assert turn.returncode == 0


def test_a_turn_that_exits_zero_having_done_nothing_is_a_failure(run_dir: Path, monkeypatch) -> None:
    """The bug MulmoTerminal shipped and PLAN.md wrote down: the process ended, so it looked fine."""
    _spawning(monkeypatch, writes={}, returncode=0, stdout="I have written the prior.")
    with pytest.raises(ArtifactError, match="was never written"):
        author_artifacts("write the prior", run_dir=run_dir, expected=("research/framing_prior.json",))


def test_an_empty_artifact_is_a_failure(run_dir: Path, monkeypatch) -> None:
    _spawning(monkeypatch, writes={"research/framing_prior.json": ""})
    with pytest.raises(ArtifactError, match="is empty"):
        author_artifacts("write the prior", run_dir=run_dir, expected=("research/framing_prior.json",))


def test_an_unchanged_artifact_is_a_failure(run_dir: Path, monkeypatch) -> None:
    """A turn that read for ten minutes and left the PREVIOUS run's file exactly where it was.
    Existence alone cannot tell that from success."""
    existing = run_dir / "research/framing_prior.json"
    existing.write_text('{"prior_level": "中程度"}')
    _spawning(monkeypatch, writes={}, stdout="the existing prior still looks right to me")

    with pytest.raises(ArtifactError, match="unchanged from before the turn"):
        author_artifacts("write the prior", run_dir=run_dir, expected=("research/framing_prior.json",))


def test_rewriting_an_existing_artifact_succeeds(run_dir: Path, monkeypatch) -> None:
    """The control for the test above — a re-run that genuinely reconsiders must not be rejected."""
    existing = run_dir / "research/framing_prior.json"
    existing.write_text('{"prior_level": "中程度"}')
    _spawning(monkeypatch, writes={"research/framing_prior.json": '{"prior_level": "低い"}'})

    author_artifacts("write the prior", run_dir=run_dir, expected=("research/framing_prior.json",))
    assert "低い" in existing.read_text()


def test_a_nonzero_exit_with_the_artifact_written_still_succeeds(run_dir: Path, monkeypatch) -> None:
    """Symmetry: if the exit code cannot prove success, it cannot prove failure either. The
    artifacts decide, and a turn that wrote what it promised did the work."""
    _spawning(monkeypatch, writes={"research/framing_prior.json": "{}"}, returncode=1)
    turn = author_artifacts("write the prior", run_dir=run_dir, expected=("research/framing_prior.json",))
    assert turn.returncode == 1


def test_the_failure_says_what_the_turn_actually_said(run_dir: Path, monkeypatch) -> None:
    """Debugging an unattended run at 3am is reading one error message."""
    _spawning(monkeypatch, writes={}, stdout="I could not find the source of truth.")
    with pytest.raises(ArtifactError, match="could not find the source of truth"):
        author_artifacts("write the prior", run_dir=run_dir, expected=("research/framing_prior.json",))


def test_a_silent_turn_says_so(run_dir: Path, monkeypatch) -> None:
    _spawning(monkeypatch, writes={}, stdout="")
    with pytest.raises(ArtifactError, match="said nothing at all"):
        author_artifacts("write the prior", run_dir=run_dir, expected=("research/framing_prior.json",))


# --------------------------------------------------------------------------- #
# When the CLI itself is the problem
# --------------------------------------------------------------------------- #
def test_a_missing_cli_says_which_binary_and_how_to_point_at_it(run_dir: Path, monkeypatch) -> None:
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ClaudeUnavailable, match="DR2_CLAUDE_BINARY"):
        run_turn("anything", cwd=run_dir)


# prepush codex 2026-08-17: defaulting to PATH is right — requiring the variable would fail on every
# machine that never set it — but doing it SILENTLY is not. An unattended run picking up whichever
# `claude` is first on PATH should say which one it picked.
def test_the_resolved_binary_is_an_absolute_path() -> None:
    from dr2_podcast.claude_runner import resolve_binary

    resolved = resolve_binary()
    assert Path(resolved).is_absolute(), resolved


def test_a_hung_turn_is_bounded_and_its_output_is_not_trusted(run_dir: Path, monkeypatch) -> None:
    """A hung turn in an unattended run is indistinguishable from a slow one until morning."""
    def _hangs(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 0), output="half a", stderr="")

    monkeypatch.setattr(subprocess, "run", _hangs)
    with pytest.raises(ClaudeUnavailable, match="did not finish within"):
        author_artifacts("write the prior", run_dir=run_dir, expected=("research/framing_prior.json",), timeout=5)


def test_an_empty_prompt_is_refused_before_spawning(run_dir: Path, monkeypatch) -> None:
    def _must_not_run(argv, **kwargs):
        raise AssertionError("nothing should have been spawned")

    monkeypatch.setattr(subprocess, "run", _must_not_run)
    with pytest.raises(ClaudeUnavailable, match="empty prompt"):
        run_turn("   ", cwd=run_dir)


def test_the_turn_runs_in_the_run_directory(run_dir: Path, monkeypatch) -> None:
    seen = _spawning(monkeypatch, writes={"research/framing_prior.json": "{}"})
    author_artifacts("write the prior", run_dir=run_dir, expected=("research/framing_prior.json",))
    assert seen["cwd"] == str(run_dir)


def test_a_turn_reports_whether_it_spoke() -> None:
    assert ClaudeTurn(0, "something", "").spoke
    assert not ClaudeTurn(0, "   ", "").spoke


# --------------------------------------------------------------------------- #
# The frozen prior, the runner's first caller
# --------------------------------------------------------------------------- #
# prepush codex 2026-08-17: the prompt asked for topic and frozen_at, the schema did not require
# them, so they could be omitted silently — losing what proposition was judged and when.
def _prior(**overrides) -> dict:
    record = {
        "schema_version": 1, "authored_by": "claude", "prior_level": "低い",
        "plausibility": {"stated": "small at best", "basis": "no mechanism links the two"},
        "known_mechanism": {"stated": None, "basis": "none described in reviews"},
        "class_effect": {"stated": None, "basis": "the class is heterogeneous"},
        "base_rate": {"stated": "~10% replicate", "basis": "Ioannidis 2005"},
        "topic": "ビタミンDと骨折", "frozen_at": "2026-08-17",
    }
    record.update(overrides)
    return record


@pytest.mark.parametrize("missing", ["topic", "frozen_at"])
def test_a_prior_without_its_question_or_its_time_is_rejected(missing: str) -> None:
    from dr2_podcast.schemas import framing_prior_errors

    record = _prior()
    del record[missing]
    assert framing_prior_errors(record), f"{missing} must be required"


def test_a_complete_prior_validates() -> None:
    from dr2_podcast.schemas import framing_prior_errors

    assert framing_prior_errors(_prior()) == []


def test_every_component_must_state_its_basis() -> None:
    """A prior without a basis is a number somebody made up, and step 9 would be arithmetic over it."""
    from dr2_podcast.schemas import framing_prior_errors

    assert framing_prior_errors(_prior(base_rate={"stated": "10%", "basis": ""}))


def test_nothing_is_known_is_a_real_answer() -> None:
    from dr2_podcast.schemas import framing_prior_errors

    assert framing_prior_errors(_prior(plausibility={"stated": None, "basis": "no prior literature"})) == []


def test_a_prior_about_a_different_question_is_refused(run_dir: Path, monkeypatch) -> None:
    """Step 9 would update it against THIS run's evidence and the episode would state the result."""
    import json

    from dr2_podcast.adapters import research_stages

    (run_dir / "research/research_framing.md").write_text("# Framing\n\nQuestions.\n")
    _spawning(monkeypatch, writes={"research/framing_prior.json": json.dumps(_prior(topic="something else"))})

    with pytest.raises(ArtifactError, match="but this run is about"):
        research_stages.framing_prior(run_dir, {"topic": "ビタミンDと骨折", "language": "ja"})


def test_a_matching_prior_is_accepted(run_dir: Path, monkeypatch) -> None:
    import json

    from dr2_podcast.adapters import research_stages

    (run_dir / "research/research_framing.md").write_text("# Framing\n\nQuestions.\n")
    _spawning(monkeypatch, writes={"research/framing_prior.json": json.dumps(_prior())})
    research_stages.framing_prior(run_dir, {"topic": "ビタミンDと骨折", "language": "ja"})
    assert (run_dir / "research/framing_prior.json").exists()


def test_the_prompt_forbids_searching() -> None:
    """The prior's one property is that it was written BEFORE the evidence."""
    from dr2_podcast.adapters.research_stages import _PRIOR_PROMPT

    assert "do NOT search" in _PRIOR_PROMPT


# prepush codex 2026-08-17: frozen_at only had to be non-empty, so "today" and "unknown" validated —
# and "written before the search" is the ONE property this artifact has.
@pytest.mark.parametrize(
    "stamp",
    # The last two are the ones a pattern cannot catch: digits in the right places, no such day.
    ["today", "unknown", "", "2026", "soon", "2026-99-99", "2026-02-31"],
)
def test_a_freeze_time_that_names_no_instant_is_rejected(stamp: str) -> None:
    from dr2_podcast.schemas import framing_prior_errors

    assert framing_prior_errors(_prior(frozen_at=stamp)), stamp


@pytest.mark.parametrize(
    "stamp", ["2026-08-17", "2026-08-17T09:30", "2026-08-17T09:30:00Z", "2026-08-17 09:30:00+09:00"]
)
def test_a_real_timestamp_is_accepted(stamp: str) -> None:
    from dr2_podcast.schemas import framing_prior_errors

    assert framing_prior_errors(_prior(frozen_at=stamp)) == [], stamp


# prepush codex 2026-08-17: the turn passed no model, so it used whichever default the CLI or the
# account currently selects — and a changed default would change the authored prior while the
# manifest, fingerprinting only Smart/vLLM settings, still called the old output current.
def test_the_model_is_pinned_when_configured(monkeypatch) -> None:
    from dr2_podcast import claude_runner

    monkeypatch.setattr(claude_runner, "CLAUDE_MODEL", "claude-opus-5")
    argv = claude_runner._command("anything", DEFAULT_ALLOWED_TOOLS)
    assert argv[argv.index("--model") + 1] == "claude-opus-5"


def test_an_unpinned_model_passes_no_flag(monkeypatch) -> None:
    """Honest for a laptop, wrong for a pipeline — so it is hashed as unset and a later pin
    invalidates the stage rather than silently changing what it authored."""
    from dr2_podcast import claude_runner

    monkeypatch.setattr(claude_runner, "CLAUDE_MODEL", "")
    assert "--model" not in claude_runner._command("anything", DEFAULT_ALLOWED_TOOLS)


def test_the_smart_backend_does_not_restale_a_claude_authored_stage() -> None:
    """framing_prior never touches vLLM, so changing MODEL_NAME must not restale a frozen prior —
    and with it the whole research chain behind it (prepush codex 2026-08-20)."""
    from dr2_podcast.manifest import config_fingerprint

    values = {"env:DR2_CLAUDE_MODEL": "claude-opus-5", "env:MODEL_NAME": "m", "SMART_MODEL": "m"}
    assert config_fingerprint(values, None, "framing_prior") == config_fingerprint(
        {**values, "env:MODEL_NAME": "another", "SMART_MODEL": "another"}, None, "framing_prior"
    )


def test_the_smart_backend_still_restales_a_smart_stage() -> None:
    """The control: the group split must not make the Smart stages blind to their own model."""
    from dr2_podcast.manifest import config_fingerprint

    values = {"env:MODEL_NAME": "m", "SMART_MODEL": "m"}
    assert config_fingerprint(values, None, "research") != config_fingerprint(
        {**values, "SMART_MODEL": "another"}, None, "research"
    )


def test_the_authoring_model_is_part_of_stage_identity() -> None:
    from dr2_podcast.manifest import CONTENT_ENV_KEYS, config_fingerprint

    assert "DR2_CLAUDE_MODEL" in CONTENT_ENV_KEYS
    values = {"env:DR2_CLAUDE_MODEL": "claude-opus-5", "env:MODEL_NAME": "m"}
    assert config_fingerprint(values, None, "framing_prior") != config_fingerprint(
        {**values, "env:DR2_CLAUDE_MODEL": "something-else"}, None, "framing_prior"
    )


def test_the_binary_that_runs_is_the_one_that_was_validated(run_dir: Path, monkeypatch) -> None:
    """The turn runs with cwd set to the RUN directory, so a relative binary would resolve against
    the pipeline's cwd and then be launched from somewhere else."""
    import shutil

    from dr2_podcast import claude_runner

    monkeypatch.setattr(claude_runner, "CLAUDE_BINARY", "./bin/claude")
    monkeypatch.setattr(shutil, "which", lambda name: "/opt/tools/claude")
    seen = _spawning(monkeypatch, writes={"research/framing_prior.json": "{}"})

    author_artifacts("write it", run_dir=run_dir, expected=("research/framing_prior.json",))
    assert seen["argv"][0] == "/opt/tools/claude"


# prepush codex 2026-08-17: the framing was passed as framing[:4000], so questions and scope
# constraints past that boundary could not influence the judgement — while the prior was still
# recorded as applying to the whole topic. The same silent-truncation class Step 0 made loud.
def test_the_whole_framing_reaches_the_authoring_turn(run_dir: Path, monkeypatch) -> None:
    import json

    from dr2_podcast.adapters import research_stages

    tail = "SCOPE: adults over 75 are out of scope for this episode."
    (run_dir / "research/research_framing.md").write_text("padding. " * 900 + tail)
    seen = _spawning(monkeypatch, writes={"research/framing_prior.json": json.dumps(_prior())})

    research_stages.framing_prior(run_dir, {"topic": "ビタミンDと骨折", "language": "ja"})
    assert tail in seen["argv"][2], "the constraint at the end never reached the judgement"


# prepush codex 2026-08-20, and it is the deepest one: the prompt says not to look at the evidence,
# but the turn holds Read, Glob and Grep — and on a re-run the run directory is full of search
# results. A prior that COULD have read the findings is not demonstrably a pre-search prior,
# whatever it says. So the turn gets a directory containing only the framing.
def test_the_prior_is_authored_where_the_evidence_is_not(run_dir: Path, monkeypatch) -> None:
    import json

    from dr2_podcast.adapters import research_stages

    (run_dir / "research/research_framing.md").write_text("# Framing\n\nQuestions.\n")
    # A re-run: the previous search is still sitting there.
    (run_dir / "research/source_of_truth.md").write_text("ARR was 5.0% for hip fracture.\n")
    (run_dir / "research/research_sources.json").write_text('{"lead": []}')

    saw: dict = {}

    def _record_and_write(argv, **kwargs):
        cwd = Path(kwargs["cwd"])
        saw["visible"] = sorted(p.name for p in cwd.rglob("*") if p.is_file())
        (cwd / "research").mkdir(parents=True, exist_ok=True)
        (cwd / "research/framing_prior.json").write_text(json.dumps(_prior()))
        return subprocess.CompletedProcess(argv, 0, "written", "")

    monkeypatch.setattr(subprocess, "run", _record_and_write)
    research_stages.framing_prior(run_dir, {"topic": "ビタミンDと骨折", "language": "ja"})

    assert saw["visible"] == ["research_framing.md"], saw["visible"]
    assert (run_dir / "research/framing_prior.json").exists(), "and the prior still lands in the run"


def test_the_authored_prior_reaches_the_run_directory(run_dir: Path, monkeypatch) -> None:
    import json

    from dr2_podcast.adapters import research_stages

    (run_dir / "research/research_framing.md").write_text("# Framing\n")

    def _write_in_scratch(argv, **kwargs):
        cwd = Path(kwargs["cwd"])
        (cwd / "research").mkdir(parents=True, exist_ok=True)
        (cwd / "research/framing_prior.json").write_text(json.dumps(_prior(prior_level="中程度")))
        return subprocess.CompletedProcess(argv, 0, "written", "")

    monkeypatch.setattr(subprocess, "run", _write_in_scratch)
    research_stages.framing_prior(run_dir, {"topic": "ビタミンDと骨折", "language": "ja"})

    assert json.loads((run_dir / "research/framing_prior.json").read_text())["prior_level"] == "中程度"


# prepush codex 2026-08-20 [P1]: a scratch cwd does not sandbox Read, Glob or Grep — they take
# absolute paths — so isolating the prior by directory was isolating it by hope. The framing is
# already in the prompt, so the turn needs to read nothing, and a capability it does not hold is
# the only kind it cannot use.
def test_the_prior_turn_can_only_write(run_dir: Path, monkeypatch) -> None:
    import json

    from dr2_podcast.adapters import research_stages

    (run_dir / "research/research_framing.md").write_text("# Framing\n")
    seen = _spawning(monkeypatch, writes={})

    def _write_then(argv, **kwargs):
        cwd = Path(kwargs["cwd"])
        (cwd / "research").mkdir(parents=True, exist_ok=True)
        (cwd / "research/framing_prior.json").write_text(json.dumps(_prior()))
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, "written", "")

    monkeypatch.setattr(subprocess, "run", _write_then)
    research_stages.framing_prior(run_dir, {"topic": "ビタミンDと骨折", "language": "ja"})

    granted = set(seen["argv"][seen["argv"].index("--allowedTools") + 1].split(","))
    assert granted == {"Write"}, granted
    for reader in ("Read", "Glob", "Grep"):
        assert reader not in granted


def test_the_framing_is_in_the_prompt_so_nothing_needs_reading() -> None:
    """The reason write-only is sufficient rather than merely strict."""
    from dr2_podcast.adapters.research_stages import _PRIOR_PROMPT

    assert "{framing}" in _PRIOR_PROMPT
