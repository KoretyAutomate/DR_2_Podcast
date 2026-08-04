"""Characterization tests for scorecard._parse_log_metrics.

Written before splitting that 70-statement function into per-concern parsers.
The scorecard module had no tests at all; every metric below is one the run
report displays, so a regex that quietly stops matching is a silent quality
regression rather than a crash.
"""

import pytest

from dr2_podcast.evaluation.scorecard import _parse_log_metrics


FULL_LOG = """\
Topic: Coffee and productivity
2026-08-03 10:00:00 INFO starting run
2026-08-03 10:00:01 INFO Language: 日本語 (Japanese) (ja)
2026-08-03 10:00:02 INFO Podcast Length Mode: Long (30 min)
2026-08-03 10:01:00 INFO     [Step 4] Extracted data from 18/20 articles (cache hits: 3)
2026-08-03 10:02:00 INFO     [Step 4] Extracted data from 9/10 articles (cache hits: 0)
2026-08-03 10:03:00 WARNING Request timed out after 180s
2026-08-03 10:03:30 WARNING Request timed out after 180s
2026-08-03 10:04:00 INFO Sectional draft: 4 sections, total budget 10500 chars
2026-08-03 10:05:00 INFO   Section opening: 1000/1200 chars
2026-08-03 10:06:00 INFO   Section evidence: 5000/5000 chars
2026-08-03 10:07:00 INFO   evidence: budget adjusted 5000 -> 4000 chars
2026-08-03 10:07:30 INFO   closing: budget adjusted 2000 -> 1000 chars
2026-08-03 10:08:00 INFO Assembled draft: 10240 chars
2026-08-03 10:09:00 INFO Degenerate repetition detected: 3.5 %
2026-08-03 10:10:00 INFO   Quick content audit: CLEAN
2026-08-03 10:30:00 INFO SUCCESS: Audio duration 28.4 minutes
"""


class TestParseLogMetrics:
    @pytest.fixture
    def metrics(self, tmp_path):
        log = tmp_path / "podcast_generation.log"
        log.write_text(FULL_LOG)
        return _parse_log_metrics(log)

    def test_missing_log_gives_empty_metrics(self, tmp_path):
        assert _parse_log_metrics(tmp_path / "nope.log") == {}

    def test_extraction_counts_are_summed_across_tracks(self, metrics):
        assert metrics["articles_extracted"] == 27
        assert metrics["articles_attempted"] == 30
        assert metrics["extraction_timeout_rate"] == 0.1

    def test_timeouts_are_counted(self, metrics):
        assert metrics["extraction_timeouts"] == 2

    def test_no_extraction_lines_zeroes_the_counts(self, tmp_path):
        log = tmp_path / "l.log"
        log.write_text("2026-08-03 10:00:00 INFO nothing here\n")
        m = _parse_log_metrics(log)
        assert m["articles_extracted"] == 0
        assert m["articles_attempted"] == 0
        assert "extraction_timeout_rate" not in m

    def test_section_adherence_is_actual_over_budget(self, metrics):
        assert metrics["section_adherence"] == [0.833, 1.0]

    def test_deficit_ratio_is_the_max_observed(self, metrics):
        """Two adjustments present (0.8 and 0.5) — the metric reports the max."""
        assert metrics["max_deficit_ratio"] == 0.8

    def test_draft_length_and_unit(self, metrics):
        assert metrics["draft_char_count"] == 10240
        assert metrics["draft_length_unit"] == "chars"

    def test_english_draft_reports_words(self, tmp_path):
        log = tmp_path / "l.log"
        log.write_text("Assembled draft: 4200 words\n")
        assert _parse_log_metrics(log)["draft_length_unit"] == "words"

    def test_script_target_length(self, metrics):
        assert metrics["script_target_length"] == 10500

    def test_degenerate_repetition_percentage(self, metrics):
        assert metrics["degenerate_repetition_pct"] == 3.5

    def test_degenerate_defaults_to_zero_when_absent(self, tmp_path):
        log = tmp_path / "l.log"
        log.write_text("nothing\n")
        assert _parse_log_metrics(log)["degenerate_repetition_pct"] == 0.0

    def test_clean_content_audit_is_zero_issues(self, metrics):
        assert metrics["content_audit_issues"] == 0

    def test_content_audit_issues_are_counted_when_not_clean(self, tmp_path):
        log = tmp_path / "l.log"
        log.write_text("content audit issue one\ncontent audit issue two\n")
        assert _parse_log_metrics(log)["content_audit_issues"] == 2

    def test_total_duration_from_first_and_last_timestamp(self, metrics):
        assert metrics["total_duration_min"] == 30.0

    def test_single_timestamp_yields_no_duration(self, tmp_path):
        log = tmp_path / "l.log"
        log.write_text("2026-08-03 10:00:00 INFO only one\n")
        assert "total_duration_min" not in _parse_log_metrics(log)

    def test_audio_and_target_duration(self, metrics):
        assert metrics["audio_duration_min"] == 28.4
        assert metrics["target_duration_min"] == 30

    def test_language_code_is_taken_from_the_parenthetical(self, metrics):
        assert metrics["language"] == "ja"

    def test_language_falls_back_to_the_spelled_out_name(self, tmp_path):
        log = tmp_path / "l.log"
        log.write_text("Language: English\nLanguage: (English)\n")
        assert _parse_log_metrics(log)["language"] == "en"

    def test_topic_is_captured(self, metrics):
        assert metrics["topic"] == "Coffee and productivity"

    def test_topic_must_start_its_line(self, tmp_path):
        """The pattern is ^Topic: — a log-prefixed line is deliberately not matched."""
        log = tmp_path / "l.log"
        log.write_text("2026-08-03 10:00:00 INFO Topic: Coffee\n")
        assert "topic" not in _parse_log_metrics(log)
