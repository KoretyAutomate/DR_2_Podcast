"""
Test T3.3 — LLM Retry Logic with Exponential Backoff + Jitter

Tests for both:
  - _call_smart() in clinical_research.py (async, on ResearchAgent class)
  - _call_smart_model() in pipeline.py (sync, module-level function)

Verifies:
  1. Retries on transient ConnectionError with increasing delays
  2. Does NOT retry on openai.BadRequestError (immediate raise)
  3. After 4 total attempts (1 initial + 3 retries), raises original exception
  4. Backoff values: ~5s, ~10s, ~20s (with jitter ±30%)
"""

import asyncio
import os
import sys
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Set env vars before importing modules
os.environ.setdefault("MODEL_NAME", "test-model")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:9999/v1")
os.environ.setdefault("FAST_MODEL_NAME", "test-fast")
os.environ.setdefault("FAST_LLM_BASE_URL", "http://localhost:9999/v1")



class TestCallSmartRetry(unittest.TestCase):
    """Tests for _call_smart() in clinical_research.py (async, ResearchAgent)."""

    def setUp(self):
        """Create a ResearchAgent instance with mocked clients."""
        from dr2_podcast.research.clinical import ResearchAgent
        self.agent = ResearchAgent.__new__(ResearchAgent)
        self.agent.smart_client = AsyncMock()
        self.agent.smart_model = "test-model"

    def _run(self, coro):
        """Helper to run async code."""
        return asyncio.run(coro)

    def test_retries_on_connection_error_with_escalating_delays(self):
        """_call_smart retries on ConnectionError with increasing backoff."""
        sleep_times = []

        async def mock_sleep(t):
            sleep_times.append(t)

        # Fail 3 times with ConnectionError, succeed on 4th
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "success response"

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise ConnectionError("Connection refused")
            return mock_resp

        self.agent.smart_client.chat.completions.create = AsyncMock(side_effect=side_effect)

        with patch("dr2_podcast.research.clinical.asyncio.sleep", side_effect=mock_sleep):
            result = self._run(self.agent._call_smart("system", "user"))

        self.assertEqual(result, "success response")
        self.assertEqual(call_count, 4, "Should have attempted 4 times total")
        self.assertEqual(len(sleep_times), 3, "Should have slept 3 times")

        # Verify escalating backoff: ~5s, ~10s, ~20s (±30% jitter)
        self.assertGreaterEqual(sleep_times[0], 5 * 0.7, f"First wait {sleep_times[0]:.1f}s too low")
        self.assertLessEqual(sleep_times[0], 5 * 1.3, f"First wait {sleep_times[0]:.1f}s too high")

        self.assertGreaterEqual(sleep_times[1], 10 * 0.7, f"Second wait {sleep_times[1]:.1f}s too low")
        self.assertLessEqual(sleep_times[1], 10 * 1.3, f"Second wait {sleep_times[1]:.1f}s too high")

        self.assertGreaterEqual(sleep_times[2], 20 * 0.7, f"Third wait {sleep_times[2]:.1f}s too low")
        self.assertLessEqual(sleep_times[2], 20 * 1.3, f"Third wait {sleep_times[2]:.1f}s too high")

        print(f"  PASS: Escalating delays: {[f'{t:.1f}s' for t in sleep_times]}")

    def test_no_retry_on_bad_request_error(self):
        """_call_smart does NOT retry on openai.BadRequestError."""
        import openai

        self.agent.smart_client.chat.completions.create = AsyncMock(
            side_effect=openai.BadRequestError(
                message="Invalid request",
                response=MagicMock(status_code=400),
                body=None,
            )
        )

        with self.assertRaises(openai.BadRequestError):
            self._run(self.agent._call_smart("system", "user"))

        # Should have been called exactly once (no retries)
        self.assertEqual(
            self.agent.smart_client.chat.completions.create.call_count,
            1,
            "Should NOT retry on BadRequestError"
        )
        print("  PASS: BadRequestError raises immediately without retry")

    def test_no_retry_on_authentication_error(self):
        """_call_smart does NOT retry on openai.AuthenticationError."""
        import openai

        self.agent.smart_client.chat.completions.create = AsyncMock(
            side_effect=openai.AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401),
                body=None,
            )
        )

        with self.assertRaises(openai.AuthenticationError):
            self._run(self.agent._call_smart("system", "user"))

        self.assertEqual(
            self.agent.smart_client.chat.completions.create.call_count,
            1,
            "Should NOT retry on AuthenticationError"
        )
        print("  PASS: AuthenticationError raises immediately without retry")

    def test_raises_after_all_retries_exhausted(self):
        """_call_smart raises after 4 total attempts (1 + 3 retries)."""
        self.agent.smart_client.chat.completions.create = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )

        async def mock_sleep(t):
            pass  # Skip actual sleeping

        with patch("dr2_podcast.research.clinical.asyncio.sleep", side_effect=mock_sleep):
            with self.assertRaises(ConnectionError) as ctx:
                self._run(self.agent._call_smart("system", "user"))

        self.assertEqual(
            self.agent.smart_client.chat.completions.create.call_count,
            4,
            "Should attempt exactly 4 times total"
        )
        self.assertIn("Connection refused", str(ctx.exception))
        print("  PASS: Raises ConnectionError after 4 attempts")

    def test_retries_on_openai_transient_errors(self):
        """_call_smart retries on openai.APIConnectionError, APITimeoutError, InternalServerError."""
        import openai

        transient_errors = [
            (openai.APIConnectionError, {"request": MagicMock()}, "APIConnectionError"),
            (openai.APITimeoutError, {"request": MagicMock()}, "APITimeoutError"),
            (openai.InternalServerError,
             {"message": "Internal server error", "response": MagicMock(status_code=500), "body": None},
             "InternalServerError"),
        ]

        for error_class, error_kwargs, error_name in transient_errors:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "ok"

            call_count = 0

            async def make_side_effect(err_cls, err_kw):
                nonlocal call_count
                call_count = 0

                async def side_effect(**kwargs):
                    nonlocal call_count
                    call_count += 1
                    if call_count == 1:
                        raise err_cls(**err_kw)
                    return mock_resp
                return side_effect

            side_fn = self._run(make_side_effect(error_class, error_kwargs))
            self.agent.smart_client.chat.completions.create = AsyncMock(side_effect=side_fn)

            async def mock_sleep(t):
                pass

            with patch("dr2_podcast.research.clinical.asyncio.sleep", side_effect=mock_sleep):
                result = self._run(self.agent._call_smart("system", "user"))

            self.assertEqual(result, "ok")
            self.assertEqual(call_count, 2, f"{error_name}: should retry once then succeed")
            print(f"  PASS: Retries on {error_name}")


class TestCallSmartModelRetry(unittest.TestCase):
    """Tests for _call_smart_model() in pipeline.py (sync)."""

    def setUp(self):
        """Set module globals so _call_smart_model doesn't raise RuntimeError."""
        import dr2_podcast.pipeline as _p
        self._orig_model = _p.SMART_MODEL
        self._orig_url = _p.SMART_BASE_URL
        _p.SMART_MODEL = "test-model"
        _p.SMART_BASE_URL = "http://localhost:9999/v1"

    def tearDown(self):
        import dr2_podcast.pipeline as _p
        _p.SMART_MODEL = self._orig_model
        _p.SMART_BASE_URL = self._orig_url

    def test_retries_on_connection_error_with_escalating_delays(self):
        """_call_smart_model retries on ConnectionError with escalating backoff."""
        sleep_times = []

        def mock_sleep(t):
            sleep_times.append(t)

        call_count = 0
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "success"

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise ConnectionError("Connection refused")
            return mock_resp

        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create

        with patch("time.sleep", side_effect=mock_sleep), \
             patch("openai.OpenAI", return_value=mock_client):
            from dr2_podcast.pipeline import _call_smart_model
            result = _call_smart_model("system", "user")

        self.assertEqual(result, "success")
        self.assertEqual(call_count, 4)
        self.assertEqual(len(sleep_times), 3)

        # Verify escalating backoff: ~5s, ~10s, ~20s (±30% jitter)
        self.assertGreaterEqual(sleep_times[0], 5 * 0.7)
        self.assertLessEqual(sleep_times[0], 5 * 1.3)
        self.assertGreaterEqual(sleep_times[1], 10 * 0.7)
        self.assertLessEqual(sleep_times[1], 10 * 1.3)
        self.assertGreaterEqual(sleep_times[2], 20 * 0.7)
        self.assertLessEqual(sleep_times[2], 20 * 1.3)

        print(f"  PASS: Escalating delays: {[f'{t:.1f}s' for t in sleep_times]}")

    def test_no_retry_on_bad_request_error(self):
        """_call_smart_model does NOT retry on openai.BadRequestError."""
        import openai

        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            raise openai.BadRequestError(
                message="Invalid request",
                response=MagicMock(status_code=400),
                body=None,
            )

        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create

        with patch("openai.OpenAI", return_value=mock_client):
            from dr2_podcast.pipeline import _call_smart_model
            with self.assertRaises(openai.BadRequestError):
                _call_smart_model("system", "user")

        self.assertEqual(call_count, 1, "Should NOT retry on BadRequestError")
        print("  PASS: BadRequestError raises immediately without retry")

    def test_no_retry_on_authentication_error(self):
        """_call_smart_model does NOT retry on openai.AuthenticationError."""
        import openai

        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            raise openai.AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401),
                body=None,
            )

        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create

        with patch("openai.OpenAI", return_value=mock_client):
            from dr2_podcast.pipeline import _call_smart_model
            with self.assertRaises(openai.AuthenticationError):
                _call_smart_model("system", "user")

        self.assertEqual(call_count, 1, "Should NOT retry on AuthenticationError")
        print("  PASS: AuthenticationError raises immediately without retry")

    def test_raises_after_all_retries_exhausted(self):
        """_call_smart_model raises after 4 total attempts."""
        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            raise TimeoutError("Timed out")

        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create

        def mock_sleep(t):
            pass

        with patch("time.sleep", side_effect=mock_sleep), \
             patch("openai.OpenAI", return_value=mock_client):
            from dr2_podcast.pipeline import _call_smart_model
            with self.assertRaises(TimeoutError):
                _call_smart_model("system", "user")

        self.assertEqual(call_count, 4, "Should attempt exactly 4 times total")
        print("  PASS: Raises TimeoutError after 4 attempts")


if __name__ == "__main__":
    unittest.main(verbosity=2)
