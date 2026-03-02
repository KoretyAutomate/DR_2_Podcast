"""Tests for pipeline_script.py — script validation, trimming, and deduplication."""

import pytest
from dr2_podcast.pipeline_script import (
    _count_words, _deduplicate_script, _parse_blueprint_inventory,
    _validate_script, _add_reaction_guidance,
)


# ---------------------------------------------------------------------------
# _count_words
# ---------------------------------------------------------------------------

class TestCountWords:

    def test_english_word_count(self, english_lang_config):
        text = "The quick brown fox jumps over the lazy dog"
        assert _count_words(text, english_lang_config) == 9

    def test_english_strips_speaker_labels(self, english_lang_config):
        text = "Host 1: Hello there\nHost 2: Hi back"
        count = _count_words(text, english_lang_config)
        # Speaker labels "Host 1:" stripped, counts "Hello there Hi back" = 4
        assert count == 4

    def test_japanese_char_count(self, japanese_lang_config):
        text = "これはテストです。"
        count = _count_words(text, japanese_lang_config)
        # Punctuation stripped, count remaining chars
        assert count > 0

    def test_empty_string(self, english_lang_config):
        assert _count_words("", english_lang_config) == 0

    def test_mixed_content(self, english_lang_config):
        text = "Word1 word2 word3"
        assert _count_words(text, english_lang_config) == 3


# ---------------------------------------------------------------------------
# _deduplicate_script
# ---------------------------------------------------------------------------

class TestDeduplicateScript:

    def test_removes_duplicated_block(self, english_lang_config):
        lines = [
            "Host 1: Line A content here",
            "Host 2: Line B content here",
            "Host 1: Line C content here",
            "",
            "Host 1: Line A content here",
            "Host 2: Line B content here",
            "Host 1: Line C content here",
        ]
        script = "\n".join(lines)
        result = _deduplicate_script(script, english_lang_config)
        # The duplicate block should be removed
        assert result.count("Line A content here") == 1

    def test_transition_lines_preserved(self, english_lang_config):
        lines = [
            "[TRANSITION]",
            "[TRANSITION]",
            "[TRANSITION]",
        ]
        script = "\n".join(lines)
        result = _deduplicate_script(script, english_lang_config)
        # Blocks of only markers are skipped in dedup
        assert result.count("[TRANSITION]") == 3

    def test_no_duplicates_unchanged(self, english_lang_config):
        script = "Line A\nLine B\nLine C\nLine D"
        result = _deduplicate_script(script, english_lang_config)
        assert result == script

    def test_excessive_blank_lines_cleaned(self, english_lang_config):
        lines = [
            "Host 1: Line A content here",
            "Host 2: Line B content here",
            "Host 1: Line C content here",
            "",
            "",
            "",
            "",
            "Host 1: Line A content here",
            "Host 2: Line B content here",
            "Host 1: Line C content here",
        ]
        script = "\n".join(lines)
        result = _deduplicate_script(script, english_lang_config)
        # Excessive blanks (> 2 consecutive) should be trimmed
        assert "\n\n\n\n" not in result


# ---------------------------------------------------------------------------
# _parse_blueprint_inventory
# ---------------------------------------------------------------------------

class TestParseBlueprintInventory:

    def test_valid_section8(self, sample_blueprint):
        inv = _parse_blueprint_inventory(sample_blueprint)
        assert len(inv) > 0
        # Should have Act 1 through Act 4
        act_keys = list(inv.keys())
        assert any("Act 1" in k for k in act_keys)
        assert any("Act 4" in k for k in act_keys)

    def test_items_have_tier_question_answer(self, sample_blueprint):
        inv = _parse_blueprint_inventory(sample_blueprint)
        for act_label, items in inv.items():
            for it in items:
                assert "tier" in it
                assert "question" in it
                assert "answer" in it

    def test_missing_section8_returns_empty(self):
        text = "# Blueprint\n## 1. Thesis\nSome content.\n## 7. Citations\nRefs here."
        inv = _parse_blueprint_inventory(text)
        assert inv == {}

    def test_tier_values_normalized(self, sample_blueprint):
        inv = _parse_blueprint_inventory(sample_blueprint)
        valid_tiers = {"Basic", "Context", "Deep-dive", "Unknown"}
        for items in inv.values():
            for it in items:
                assert it["tier"] in valid_tiers


# ---------------------------------------------------------------------------
# _validate_script
# ---------------------------------------------------------------------------

class TestValidateScript:

    def test_over_length_fails(self, english_lang_config):
        script = " ".join(["word"] * 200)
        result = _validate_script(script, 100, 0.10, english_lang_config,
                                  "", "draft")
        assert not result["pass"]
        assert any("TOO LONG" in i for i in result["issues"])

    def test_under_length_fails(self, english_lang_config):
        script = "short script"
        result = _validate_script(script, 1000, 0.10, english_lang_config,
                                  "", "draft")
        assert not result["pass"]
        assert any("TOO SHORT" in i for i in result["issues"])

    def test_missing_transitions_polish_stage(self, english_lang_config):
        # 100 words, target 100, tolerance 0.10, but only 1 transition
        words = " ".join(["word"] * 100)
        script = words + "\n[TRANSITION]\n" + words
        result = _validate_script(script, 200, 0.10, english_lang_config,
                                  "", "polish")
        assert not result["pass"]
        assert any("MISSING TRANSITIONS" in i for i in result["issues"])

    def test_degenerate_repetition(self, english_lang_config):
        script = "the " * 5 + " ".join(["normal"] * 95)
        result = _validate_script(script, 100, 0.10, english_lang_config,
                                  "", "draft")
        assert not result["pass"]
        assert any("DEGENERATE REPETITION" in i for i in result["issues"])

    def test_pass_no_llm(self, english_lang_config):
        # Use varied words to avoid degenerate repetition detection
        words = ["alpha", "beta", "gamma", "delta", "epsilon"] * 20
        script = " ".join(words)
        result = _validate_script(script, 100, 0.10, english_lang_config,
                                  "", "draft")
        assert result["pass"]
        assert result["feedback"] == "PASS"

    def test_llm_clean_passes(self, english_lang_config):
        words = ["alpha", "beta", "gamma", "delta", "epsilon"] * 20
        script = " ".join(words)
        result = _validate_script(
            script, 100, 0.10, english_lang_config, "SOT content", "draft",
            _call_smart_model=lambda **kw: "CLEAN",
            _truncate_at_boundary=lambda text, n: text[:n],
        )
        assert result["pass"]

    def test_llm_drift_fails(self, english_lang_config):
        words = ["alpha", "beta", "gamma", "delta", "epsilon"] * 20
        script = " ".join(words)
        result = _validate_script(
            script, 100, 0.10, english_lang_config, "SOT content", "draft",
            _call_smart_model=lambda **kw: "Script overstates causation",
            _truncate_at_boundary=lambda text, n: text[:n],
        )
        assert not result["pass"]
        assert any("CONTENT:" in i for i in result["issues"])


# ---------------------------------------------------------------------------
# _add_reaction_guidance
# ---------------------------------------------------------------------------

class TestAddReactionGuidance:

    def test_no_host_lines_returns_original(self, english_lang_config):
        script = "No host lines here.\nJust plain text."
        result = _add_reaction_guidance(
            script, english_lang_config,
            _call_smart_model=lambda **kw: ""
        )
        assert result == script

    def test_successful_annotation(self, english_lang_config):
        script = (
            "Host 1: Welcome to the show.\n"
            "Host 2: Let's dive in.\n"
            "Host 1: The evidence shows improvement.\n"
        )

        def mock_smart(**kw):
            return "1: [intrigued, building suspense]\n3: [authoritative, measured pace]"

        result = _add_reaction_guidance(
            script, english_lang_config,
            _call_smart_model=mock_smart
        )
        assert "## [intrigued, building suspense]" in result
        assert "## [authoritative, measured pace]" in result

    def test_unparseable_llm_output_returns_original(self, english_lang_config):
        script = "Host 1: Hello.\nHost 2: Hi."
        result = _add_reaction_guidance(
            script, english_lang_config,
            _call_smart_model=lambda **kw: "No annotations to provide."
        )
        assert result == script

    def test_llm_exception_returns_original(self, english_lang_config):
        script = "Host 1: Hello.\nHost 2: Hi."

        def mock_fail(**kw):
            raise Exception("API error")

        result = _add_reaction_guidance(
            script, english_lang_config,
            _call_smart_model=mock_fail
        )
        assert result == script
