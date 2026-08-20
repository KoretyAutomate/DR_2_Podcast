"""Tests for the helpers extracted out of web_ui's generation workers.

run_podcast_generation took 14 parameters and ran 77 statements;
run_podcast_reuse ran 65 and duplicated its mark-running and
topic-registration blocks. Both now share GenerationRequest and a handful of
helpers. These tests cover the extracted units — the request mapping, the
subprocess environment, and the error-output cleaner — none of which had any
coverage before.
"""

import subprocess

import pytest

from dr2_podcast.web import web_ui


class TestGenerationRequestFromTaskData:
    def test_maps_every_field(self):
        req = web_ui.GenerationRequest.from_task_data(
            {
                "topic": "Coffee",
                "language": "ja",
                "accessibility_level": "expert",
                "podcast_length": "short",
                "podcast_hosts": "fixed",
                "channel_intro": "intro text",
                "core_target": "target",
                "channel_mission": "mission",
            }
        )
        assert req.topic == "Coffee"
        assert req.language == "ja"
        assert req.accessibility_level == "expert"
        assert req.podcast_length == "short"
        assert req.podcast_hosts == "fixed"
        assert req.channel_intro == "intro text"
        assert req.core_target == "target"
        assert req.channel_mission == "mission"

    def test_optional_fields_default_when_absent(self):
        req = web_ui.GenerationRequest.from_task_data(
            {
                "topic": "T",
                "language": "en",
                "accessibility_level": "simple",
                "podcast_length": "long",
                "podcast_hosts": "random",
            }
        )
        assert req.channel_intro == ""
        assert req.core_target == ""
        assert req.channel_mission == ""

    def test_a_missing_required_key_is_an_error_not_a_silent_default(self):
        with pytest.raises(KeyError):
            web_ui.GenerationRequest.from_task_data({"language": "en"})


class TestBuildGenerationEnv:
    def test_always_sets_the_three_core_vars(self):
        env = web_ui._build_generation_env(
            web_ui.GenerationRequest(topic="T", language="en", accessibility_level="expert")
        )
        assert env["ACCESSIBILITY_LEVEL"] == "expert"
        assert env["PODCAST_LENGTH"] == "long"
        assert env["PODCAST_HOSTS"] == "random"

    def test_optional_vars_are_omitted_when_empty(self):
        env = web_ui._build_generation_env(web_ui.GenerationRequest(topic="T", language="en"))
        assert "PODCAST_CHANNEL_INTRO" not in env
        assert "PODCAST_CORE_TARGET" not in env
        assert "PODCAST_CHANNEL_MISSION" not in env

    def test_optional_vars_are_set_when_present(self):
        env = web_ui._build_generation_env(
            web_ui.GenerationRequest(
                topic="T",
                language="en",
                channel_intro="hello",
                core_target="curious people",
                channel_mission="explain science",
            )
        )
        assert env["PODCAST_CHANNEL_INTRO"] == "hello"
        assert env["PODCAST_CORE_TARGET"] == "curious people"
        assert env["PODCAST_CHANNEL_MISSION"] == "explain science"

    def test_inherits_the_parent_environment(self, monkeypatch):
        monkeypatch.setenv("SOME_UNRELATED_VAR", "kept")
        env = web_ui._build_generation_env(web_ui.GenerationRequest(topic="T", language="en"))
        assert env["SOME_UNRELATED_VAR"] == "kept"


class TestCleanErrorOutput:
    def test_strips_the_log_prefix(self):
        out = web_ui._clean_error_output(["2026-08-03 10:00:00 - root - ERROR - the real message\n"])
        assert out == "the real message"

    def test_drops_caret_only_pydantic_lines(self):
        out = web_ui._clean_error_output(["real failure\n", "    ^^^^^^\n", "   \n", "second line\n"])
        assert out == "real failure\nsecond line"

    def test_keeps_only_the_last_50_meaningful_lines(self):
        out = web_ui._clean_error_output([f"line {i}\n" for i in range(200)])
        lines = out.split("\n")
        assert len(lines) == 50
        assert lines[-1] == "line 199"

    def test_considers_only_the_last_100_raw_lines(self):
        """A meaningful line older than the 100-line window must not surface."""
        raw = ["NEEDLE\n"] + ["    ^^^\n"] * 150
        assert web_ui._clean_error_output(raw) == ""

    def test_empty_input_gives_empty_string(self):
        assert web_ui._clean_error_output([]) == ""


class TestSpawnAndStream:
    """The one place web_ui spawns the pipeline.

    It was two copies — full generation and subprocess reuse — each spawning, registering the pid,
    streaming and unregistering in the same order. What the copies did not do is survive a stream
    that raises: the pid stayed registered (so /api/stop would signal a pid the OS may have
    reassigned) and the child kept running with nobody holding it — a 40-minute generation still
    on the GPU, still writing into a run the UI had already marked failed.
    """

    class _FakeProc:
        def __init__(self, returncode=0, stubborn=False):
            self.pid = 4242
            self.returncode = returncode
            self.terminated = False
            self.killed = False
            self.reaped = False
            self._alive = True
            self._stubborn = stubborn

        def poll(self):
            return None if self._alive else self.returncode

        def wait(self, timeout=None):
            if self._stubborn and timeout is not None and not self.killed:
                raise subprocess.TimeoutExpired("pipeline", timeout)
            self._alive = False
            self.reaped = True
            return self.returncode

        def terminate(self):
            self.terminated = True
            if not self._stubborn:
                self._alive = False

        def kill(self):
            self.killed = True
            self._alive = False

    def _spawning(self, monkeypatch, proc, streamer):
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: proc)
        monkeypatch.setattr(web_ui, "_stream_process_output", streamer)
        web_ui._running_pids.pop("t1", None)

    def test_it_returns_the_outcome_and_drops_the_pid(self, monkeypatch):
        proc = self._FakeProc(returncode=0)
        seen = {}

        def _stream(p, task_id):
            seen["pid_while_running"] = web_ui._running_pids.get(task_id)
            return ["line\n"]

        self._spawning(monkeypatch, proc, _stream)
        returncode, lines = web_ui._spawn_and_stream("t1", ["echo", "hi"], {})

        assert (returncode, lines) == (0, ["line\n"])
        assert seen["pid_while_running"] == 4242, "/api/stop could not have found it"
        assert "t1" not in web_ui._running_pids

    def test_a_stream_that_raises_kills_the_child_and_drops_the_pid(self, monkeypatch):
        proc = self._FakeProc()

        def _explodes(p, task_id):
            raise RuntimeError("the log line was unparseable")

        self._spawning(monkeypatch, proc, _explodes)
        with pytest.raises(RuntimeError, match="unparseable"):
            web_ui._spawn_and_stream("t1", ["echo", "hi"], {})

        assert proc.terminated, "a 40-minute generation would have kept running unwatched"
        assert "t1" not in web_ui._running_pids, "/api/stop would signal a reassigned pid"

    def test_a_child_that_ignores_sigterm_is_killed(self, monkeypatch):
        proc = self._FakeProc(stubborn=True)

        def _explodes(p, task_id):
            raise RuntimeError("boom")

        self._spawning(monkeypatch, proc, _explodes)
        with pytest.raises(RuntimeError):
            web_ui._spawn_and_stream("t1", ["echo", "hi"], {})

        assert proc.terminated and proc.killed and proc.reaped
