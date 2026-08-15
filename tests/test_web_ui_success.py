"""The Web UI reports success from artifacts, not from an exit code — PLAN.md Step 7.

`returncode == 0` was treated as success, and when the [OUTPUT_DIR] marker never appeared the UI
attached the task to whatever directory was newest. So a run that did nothing was reported as a
completed episode, showing a PREVIOUS run's artifacts. A failure is visible; that is not.
"""

from __future__ import annotations

import inspect

from dr2_podcast.web import web_ui


def test_the_mtime_fallback_is_gone_from_the_completion_path() -> None:
    source = inspect.getsource(web_ui)
    assert "_find_latest_output_dir()\n            if output_dir:" not in source


def test_a_run_that_names_no_output_directory_is_failed_not_completed() -> None:
    source = inspect.getsource(web_ui)
    marker = source.index("the run exited without naming an output directory")
    assert "_fail_task(" in source[marker - 300 : marker]


def test_the_required_deliverables_are_the_ones_an_episode_is_made_of() -> None:
    assert set(web_ui.REQUIRED_DELIVERABLES) == {
        "scripts/script_final.md",
        "research/source_of_truth.md",
        "audio/audio.wav",
    }


def test_a_missing_or_empty_deliverable_is_reported(tmp_path) -> None:
    for sub in ("scripts", "research", "audio"):
        (tmp_path / sub).mkdir()
    assert len(web_ui._incomplete_deliverables(tmp_path)) == 3

    (tmp_path / "scripts/script_final.md").write_text("Host 1: hello\n")
    (tmp_path / "research/source_of_truth.md").write_text("# SOT\n")
    (tmp_path / "audio/audio.wav").write_bytes(b"")
    problems = web_ui._incomplete_deliverables(tmp_path)
    assert problems == ["audio/audio.wav is empty"], problems


def test_a_complete_run_reports_no_problems(tmp_path) -> None:
    for sub in ("scripts", "research", "audio"):
        (tmp_path / sub).mkdir()
    (tmp_path / "scripts/script_final.md").write_text("Host 1: hello\n")
    (tmp_path / "research/source_of_truth.md").write_text("# SOT\n")
    (tmp_path / "audio/audio.wav").write_bytes(b"RIFF")
    assert web_ui._incomplete_deliverables(tmp_path) == []


# prepush codex 2026-08-13: removing upload support took the credential filter with it, while
# podcast_tasks.json is persistent — records written by the previous version can still carry
# buzzsprout_api_key and youtube_secret_path, and the status API returned whatever the record held.
def test_a_legacy_task_never_returns_its_stored_credentials() -> None:
    legacy = {
        "id": "old",
        "status": "completed",
        "buzzsprout_api_key": "bz_live_should_never_be_returned",
        "buzzsprout_account_id": "12345",
        "youtube_secret_path": "/home/korety/client_secret.json",
    }
    sanitised = web_ui._without_credentials(legacy)
    assert sanitised == {"id": "old", "status": "completed"}


def test_the_filter_names_every_field_the_retired_feature_stored() -> None:
    assert {
        "buzzsprout_api_key",
        "buzzsprout_account_id",
        "youtube_secret_path",
    } == web_ui._CREDENTIAL_FIELDS


def test_both_task_endpoints_go_through_the_filter() -> None:
    """One sanitised endpoint and one raw one is the same leak with extra steps."""
    import inspect

    source = inspect.getsource(web_ui)
    assert "response = dict(task)" not in source
    assert "return [dict(t) for t in sorted_tasks" not in source
