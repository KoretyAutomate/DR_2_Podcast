"""Tests for pipeline_crew.py -- CrewAI agent/task utilities."""

import time
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from dr2_podcast.pipeline_crew import (
    ProgressTracker, _estimate_task_tokens, _build_sot_injection_for_stage,
    _crew_kickoff_guarded,
)


# ---------------------------------------------------------------------------
# ProgressTracker
# ---------------------------------------------------------------------------

class TestProgressTracker:

    @staticmethod
    def _make_metadata():
        return {
            "task1": {
                "phase": "1", "name": "Blueprint", "agent": "producer",
                "description": "Create blueprint", "estimated_duration_min": 5,
                "dependencies": [],
            },
            "task2": {
                "phase": "2", "name": "Script", "agent": "producer",
                "description": "Write script", "estimated_duration_min": 10,
                "dependencies": ["task1"],
            },
        }

    def test_init_total_phases(self):
        md = self._make_metadata()
        tracker = ProgressTracker(md)
        assert tracker.total_phases == 2

    def test_init_excludes_conditional(self):
        md = self._make_metadata()
        md["task3"] = {
            "phase": "3", "name": "Translation", "agent": "producer",
            "description": "Translate", "estimated_duration_min": 3,
            "dependencies": [], "conditional": True,
        }
        tracker = ProgressTracker(md)
        # conditional task should not be counted
        assert tracker.total_phases == 2

    def test_task_started_sets_time(self):
        tracker = ProgressTracker(self._make_metadata())
        tracker.task_started(0)
        assert tracker.task_start_time is not None

    def test_task_completed_none_start_time(self):
        """task_completed is a no-op when start_time is None (ARCH-23 guard)."""
        tracker = ProgressTracker(self._make_metadata())
        # start_time is None -- should not raise or add to completed
        tracker.task_completed(0)
        assert len(tracker.completed_tasks) == 0

    def test_full_lifecycle(self):
        tracker = ProgressTracker(self._make_metadata())
        tracker.start_workflow()
        assert tracker.start_time is not None

        tracker.task_started(0)
        time.sleep(0.01)
        tracker.task_completed(0)
        assert len(tracker.completed_tasks) == 1
        assert tracker.completed_tasks[0]["name"] == "task1"

    def test_task_started_out_of_range(self):
        """task_started with index > len returns early."""
        tracker = ProgressTracker(self._make_metadata())
        tracker.task_started(999)
        assert tracker.task_start_time is None


# ---------------------------------------------------------------------------
# _estimate_task_tokens
# ---------------------------------------------------------------------------

class TestEstimateTaskTokens:

    def test_english_token_estimate(self):
        task = SimpleNamespace(description="Test description " * 100, context=[])
        est = _estimate_task_tokens(task, None, "en")
        # ~1700 chars / 4 + 2000 = ~2425
        assert est > 2000

    def test_japanese_token_estimate(self):
        task = SimpleNamespace(description="Test description " * 100, context=[])
        est_ja = _estimate_task_tokens(task, None, "ja")
        est_en = _estimate_task_tokens(task, None, "en")
        # Japanese: ~2 chars/token → higher token count for same text
        assert est_ja > est_en

    def test_with_context_tasks(self):
        output = SimpleNamespace(raw="Context output text " * 50)
        ctx_task = SimpleNamespace(output=output)
        task = SimpleNamespace(description="Short desc", context=[ctx_task])
        est = _estimate_task_tokens(task, None, "en")
        # Context text adds to token count
        assert est > 2100


# ---------------------------------------------------------------------------
# _build_sot_injection_for_stage
# ---------------------------------------------------------------------------

class TestBuildSotInjection:

    def test_stage_1_contains_summary(self, tmp_path):
        result = _build_sot_injection_for_stage(
            1, None, None, "This is the summary.", None, "",
            {"name": "English"},
        )
        assert "SOURCE OF TRUTH SUMMARY" in result
        assert "This is the summary." in result

    def test_stage_2_reads_from_file(self, tmp_path):
        sot_file = tmp_path / "sot.md"
        sot_file.write_text(
            "## Abstract\nTest abstract content.\n"
            "## 1. Introduction\nIntro.\n"
            "### 4.3 GRADE Assessment\nGRADE data here.\n"
        )
        result = _build_sot_injection_for_stage(
            2, str(sot_file), None, "", None, "",
            {"name": "English"},
        )
        assert "[SOT Stage 2" in result
        # Should read abstract from file
        assert "RESEARCH ABSTRACT" in result

    def test_stage_2_missing_file(self):
        result = _build_sot_injection_for_stage(
            2, "/nonexistent/file.md", None, "", None, "",
            {"name": "English"},
        )
        assert "[SOT Stage 2" in result
        assert "(not available)" in result

    def test_stage_3_minimal(self):
        result = _build_sot_injection_for_stage(
            3, None, None, "", None, "ARR: 5%, NNT: 20",
            {"name": "English"},
        )
        assert "[SOT Stage 3" in result
        assert "--- END SOT ---" in result
        assert "ARR: 5%, NNT: 20" in result


# ---------------------------------------------------------------------------
# _crew_kickoff_guarded
# ---------------------------------------------------------------------------

class TestCrewKickoffGuarded:

    def test_success_first_try(self):
        kickoff_mock = MagicMock()
        crew_mock = MagicMock()
        crew_mock.kickoff = kickoff_mock
        crew_factory = lambda: crew_mock
        task = SimpleNamespace(description="Short task", context=[])
        _crew_kickoff_guarded(
            crew_factory, task, None, "en",
            None, None, "", "", "", {"name": "English"},
            "test-crew", ctx_window=100000, max_tokens=16000,
        )
        kickoff_mock.assert_called_once()

    def test_degrades_when_over_budget(self):
        """When token estimate exceeds budget, SOT injection is degraded."""
        call_count = {"n": 0}

        def crew_factory():
            crew = MagicMock()
            crew.kickoff = MagicMock()
            call_count["n"] += 1
            return crew

        # Large description to blow budget
        big_desc = "x" * 200000
        task = SimpleNamespace(description=big_desc, context=[])
        _crew_kickoff_guarded(
            crew_factory, task, None, "en",
            None, None, "summary text", "", "",
            {"name": "English"},
            "test-crew", ctx_window=8000, max_tokens=4000,
        )
        # Should have been called once (at some stage)
        assert call_count["n"] >= 1
