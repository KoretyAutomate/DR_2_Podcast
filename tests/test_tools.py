"""Tests for dr2_podcast/tools -- link_validator."""

from unittest.mock import patch, MagicMock

import httpx
from dr2_podcast.tools.link_validator import LinkValidatorTool


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
