"""Tests for dr2_podcast/web/web_ui.py -- artifact counting, output dir, API routes."""

import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Ensure env var is set before any import of web_ui (module-level side effects)
os.environ.setdefault("PODCAST_WEB_PASSWORD", "testpass123")


# ---------------------------------------------------------------------------
# count_artifacts
# ---------------------------------------------------------------------------

class TestCountArtifacts:

    def test_none_directory(self):
        from dr2_podcast.web.web_ui import count_artifacts
        found, total = count_artifacts(None)
        assert found == 0
        assert total > 0

    def test_nonexistent_directory(self):
        from dr2_podcast.web.web_ui import count_artifacts
        found, total = count_artifacts("/nonexistent/path")
        assert found == 0
        assert total > 0

    def test_directory_with_some_files(self, tmp_path):
        from dr2_podcast.web.web_ui import count_artifacts
        (tmp_path / "research_framing.md").write_text("test")
        (tmp_path / "source_of_truth.md").write_text("test")
        found, total = count_artifacts(str(tmp_path))
        assert found == 2
        assert total > 2

    def test_subdirectory_layout(self, tmp_path):
        from dr2_podcast.web.web_ui import count_artifacts
        (tmp_path / "research").mkdir()
        (tmp_path / "research" / "research_framing.md").write_text("test")
        found, total = count_artifacts(str(tmp_path))
        assert found == 1

    def test_language_adds_extra_artifacts(self, tmp_path):
        from dr2_podcast.web.web_ui import count_artifacts
        # Japanese adds extra expected artifacts
        _, total_en = count_artifacts(str(tmp_path), language="en")
        _, total_ja = count_artifacts(str(tmp_path), language="ja")
        assert total_ja > total_en

    def test_all_files_found(self, tmp_path):
        from dr2_podcast.web.web_ui import count_artifacts, EXPECTED_ARTIFACTS
        for f in EXPECTED_ARTIFACTS:
            (tmp_path / f).write_text("content")
        found, total = count_artifacts(str(tmp_path))
        assert found == total


# ---------------------------------------------------------------------------
# _find_latest_output_dir
# ---------------------------------------------------------------------------

class TestFindLatestOutputDir:

    def test_no_output_dir(self):
        from dr2_podcast.web.web_ui import _find_latest_output_dir
        with patch("dr2_podcast.web.web_ui.OUTPUT_DIR", Path("/nonexistent")):
            result = _find_latest_output_dir()
            assert result is None

    def test_empty_output_dir(self, tmp_path):
        from dr2_podcast.web.web_ui import _find_latest_output_dir
        with patch("dr2_podcast.web.web_ui.OUTPUT_DIR", tmp_path):
            result = _find_latest_output_dir()
            assert result is None

    def test_multiple_timestamped_dirs(self, tmp_path):
        from dr2_podcast.web.web_ui import _find_latest_output_dir
        d1 = tmp_path / "2026-02-28_120000_topic1"
        d1.mkdir()
        time.sleep(0.05)
        d2 = tmp_path / "2026-03-01_150000_topic2"
        d2.mkdir()
        with patch("dr2_podcast.web.web_ui.OUTPUT_DIR", tmp_path):
            result = _find_latest_output_dir()
            assert result == d2

    def test_ignores_non_timestamped_dirs(self, tmp_path):
        from dr2_podcast.web.web_ui import _find_latest_output_dir
        (tmp_path / "random_dir").mkdir()
        (tmp_path / "__pycache__").mkdir()
        with patch("dr2_podcast.web.web_ui.OUTPUT_DIR", tmp_path):
            result = _find_latest_output_dir()
            assert result is None


# ---------------------------------------------------------------------------
# API routes (FastAPI TestClient)
# ---------------------------------------------------------------------------

class TestAPIRoutes:

    def test_wrong_password_returns_401(self):
        from dr2_podcast.web.web_ui import app
        from fastapi.testclient import TestClient
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/status/nonexistent", auth=("admin", "wrongpassword"))
        assert resp.status_code == 401

    def test_download_path_traversal_blocked(self):
        """Verify that path traversal via ../../ is blocked with 403 or 404."""
        from dr2_podcast.web.web_ui import app, USERNAME, PASSWORD, tasks_db, tasks_lock
        from fastapi.testclient import TestClient
        client = TestClient(app, raise_server_exceptions=False)
        # Create a fake task entry with an output_dir
        with tasks_lock:
            tasks_db["test-task-traversal"] = {
                "status": "completed",
                "output_dir": "/tmp/safe_dir",
                "topic": "test",
            }
        try:
            resp = client.get(
                "/api/download/test-task-traversal/../../etc/passwd",
                auth=(USERNAME, PASSWORD),
            )
            # Should be blocked (403 or 404, not 200)
            assert resp.status_code in (403, 404)
        finally:
            with tasks_lock:
                tasks_db.pop("test-task-traversal", None)
