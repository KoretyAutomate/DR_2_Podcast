"""Tests for the helpers extracted out of web_ui's generation workers.

run_podcast_generation took 14 parameters and ran 77 statements;
run_podcast_reuse ran 65 and duplicated its upload, mark-running and
topic-registration blocks. Both now share GenerationRequest and a handful of
helpers. These tests cover the extracted units — the request mapping, the
subprocess environment, and the error-output cleaner — none of which had any
coverage before.
"""

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
                "upload_buzzsprout": True,
                "upload_youtube": True,
                "core_target": "target",
                "channel_mission": "mission",
                "buzzsprout_api_key": "k",
                "buzzsprout_account_id": "a",
                "youtube_secret_path": "/tmp/secret.json",
            }
        )
        assert req.topic == "Coffee"
        assert req.language == "ja"
        assert req.accessibility_level == "expert"
        assert req.podcast_length == "short"
        assert req.podcast_hosts == "fixed"
        assert req.channel_intro == "intro text"
        assert req.upload_buzzsprout is True
        assert req.upload_youtube is True
        assert req.core_target == "target"
        assert req.channel_mission == "mission"
        assert req.buzzsprout_api_key == "k"
        assert req.buzzsprout_account_id == "a"
        assert req.youtube_secret_path == "/tmp/secret.json"

    def test_optional_fields_default_when_absent(self):
        req = web_ui.GenerationRequest.from_task_data(
            {
                "topic": "T",
                "language": "en",
                "accessibility_level": "simple",
                "podcast_length": "long",
                "podcast_hosts": "random",
                "upload_buzzsprout": False,
                "upload_youtube": False,
            }
        )
        assert req.channel_intro == ""
        assert req.core_target == ""
        assert req.channel_mission == ""
        assert req.buzzsprout_api_key is None
        assert req.youtube_secret_path is None

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
