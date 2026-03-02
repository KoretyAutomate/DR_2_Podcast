"""Tests for dr2_podcast/tools -- link_validator and upload_utils."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import httpx
from dr2_podcast.tools.link_validator import LinkValidatorTool
from dr2_podcast.tools.upload_utils import validate_upload_config, upload_to_buzzsprout


# ---------------------------------------------------------------------------
# LinkValidatorTool._run
# ---------------------------------------------------------------------------

class TestLinkValidator:

    def test_valid_link_200(self):
        validator = LinkValidatorTool()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch.object(httpx, "head", return_value=mock_resp):
            result = validator._run("https://example.com")
        assert "Valid Link" in result

    def test_not_found_404(self):
        validator = LinkValidatorTool()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch.object(httpx, "head", return_value=mock_resp):
            result = validator._run("https://example.com/missing")
        assert "Broken Link" in result or "404" in result

    def test_forbidden_403(self):
        validator = LinkValidatorTool()
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        with patch.object(httpx, "head", return_value=mock_resp):
            result = validator._run("https://example.com/protected")
        assert "protected" in result.lower() or "403" in result

    def test_server_error_500(self):
        validator = LinkValidatorTool()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch.object(httpx, "head", return_value=mock_resp):
            result = validator._run("https://example.com/error")
        assert "Server error" in result or "500" in result

    def test_timeout(self):
        validator = LinkValidatorTool()
        with patch.object(httpx, "head", side_effect=httpx.TimeoutException("timeout")):
            result = validator._run("https://slow.example.com")
        assert "Timeout" in result

    def test_too_many_redirects(self):
        validator = LinkValidatorTool()
        with patch.object(httpx, "head", side_effect=httpx.TooManyRedirects("loop")):
            result = validator._run("https://redirect.example.com")
        assert "redirect" in result.lower() or "Invalid" in result


# ---------------------------------------------------------------------------
# validate_upload_config
# ---------------------------------------------------------------------------

class TestValidateUploadConfig:

    def test_valid_buzzsprout_config(self):
        result = validate_upload_config(
            True, False,
            buzzsprout_api_key="key", buzzsprout_account_id="123",
        )
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_missing_buzzsprout_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            result = validate_upload_config(True, False)
        assert result["valid"] is False
        assert len(result["errors"]) >= 1

    def test_valid_youtube_config(self, tmp_path):
        secret_file = tmp_path / "client_secret.json"
        secret_file.write_text("{}")
        with patch("dr2_podcast.tools.upload_utils.PROJECT_ROOT", tmp_path):
            result = validate_upload_config(
                False, True,
                youtube_secret_path="client_secret.json",
            )
        assert result["valid"] is True

    def test_missing_youtube_secret(self):
        with patch("dr2_podcast.tools.upload_utils.PROJECT_ROOT", Path("/nonexistent")):
            result = validate_upload_config(False, True)
        assert result["valid"] is False
        assert any("client_secret.json" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# upload_to_buzzsprout
# ---------------------------------------------------------------------------

class TestUploadToBuzzsprout:

    def test_successful_upload(self, tmp_path):
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"fake audio data")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": 42}
        mock_resp.raise_for_status = MagicMock()
        with patch("dr2_podcast.tools.upload_utils.httpx.post", return_value=mock_resp):
            result = upload_to_buzzsprout(str(audio), "Test Episode",
                                          api_key="k", account_id="1")
        assert result["success"] is True
        assert result["episode_id"] == "42"

    def test_upload_error(self, tmp_path):
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"fake audio data")
        with patch("dr2_podcast.tools.upload_utils.httpx.post",
                    side_effect=Exception("Network error")):
            result = upload_to_buzzsprout(str(audio), "Test",
                                          api_key="k", account_id="1")
        assert result["success"] is False
        assert "Network error" in result["error"]
