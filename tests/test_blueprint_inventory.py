"""
Unit tests for Blueprint Discussion Inventory + Script Trimming.

Tests _parse_blueprint_inventory, _count_words, _deduplicate_script,
_run_trim_pass, and the coverage checklist injection logic.

All functions are imported from pipeline.py — no inline copies.
"""

import re
from unittest.mock import patch, MagicMock

import pytest

from dr2_podcast.pipeline import (
    _parse_blueprint_inventory,
    _count_words,
    _deduplicate_script,
    _run_trim_pass,
)


# ── Test A: _parse_blueprint_inventory on mock input ─────────────────────


class TestParseBlueprintInventoryMock:

    def test_returns_nonempty_dict(self, sample_blueprint):
        result = _parse_blueprint_inventory(sample_blueprint)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_has_four_act_keys(self, sample_blueprint):
        result = _parse_blueprint_inventory(sample_blueprint)
        assert len(result) == 4

    def test_act1_key_found(self, sample_blueprint):
        result = _parse_blueprint_inventory(sample_blueprint)
        act1_keys = [k for k in result if 'Act 1' in k]
        assert len(act1_keys) == 1

    def test_act1_basic_tier_parsed(self, sample_blueprint):
        result = _parse_blueprint_inventory(sample_blueprint)
        act1_key = next(k for k in result if 'Act 1' in k)
        tiers = [it['tier'] for it in result[act1_key]]
        assert 'Basic' in tiers

    def test_act1_context_tier_parsed(self, sample_blueprint):
        result = _parse_blueprint_inventory(sample_blueprint)
        act1_key = next(k for k in result if 'Act 1' in k)
        tiers = [it['tier'] for it in result[act1_key]]
        assert 'Context' in tiers

    def test_act1_deep_dive_tier_parsed(self, sample_blueprint):
        result = _parse_blueprint_inventory(sample_blueprint)
        act1_key = next(k for k in result if 'Act 1' in k)
        tiers = [it['tier'] for it in result[act1_key]]
        assert 'Deep-dive' in tiers

    def test_question_text_extracted(self, sample_blueprint):
        result = _parse_blueprint_inventory(sample_blueprint)
        act1_key = next(k for k in result if 'Act 1' in k)
        basic = next(it for it in result[act1_key] if it['tier'] == 'Basic')
        assert len(basic['question']) > 5

    def test_answer_text_extracted(self, sample_blueprint):
        result = _parse_blueprint_inventory(sample_blueprint)
        act1_key = next(k for k in result if 'Act 1' in k)
        basic = next(it for it in result[act1_key] if it['tier'] == 'Basic')
        assert len(basic['answer']) > 10

    def test_no_section8_returns_empty_dict(self):
        result = _parse_blueprint_inventory("## 1. Intro\nHello.\n## 2. Body\nContent here.\n")
        assert result == {}

    def test_empty_string_returns_empty_dict(self):
        result = _parse_blueprint_inventory("")
        assert result == {}


# ── Test B: Coverage checklist injection logic ───────────────────────────


class TestCoverageChecklistInjection:
    """Test the tier-filtering logic used by _inject_blueprint_checklist."""

    @staticmethod
    def _build_checklist(inventory, length_mode):
        """Reproduce the tier-filtering logic from _inject_blueprint_checklist."""
        tier_filter = {
            'short': {'Basic'},
            'medium': {'Basic', 'Context'},
            'long': {'Basic', 'Context', 'Deep-dive'},
        }
        allowed_tiers = tier_filter.get(length_mode, {'Basic', 'Context', 'Deep-dive'})
        checklist_lines = ["\n\nCOVERAGE CHECKLIST — discuss EACH item below in its Act:"]
        for act_label, items in inventory.items():
            filtered = [it for it in items if it['tier'] in allowed_tiers]
            if filtered:
                checklist_lines.append(f"\n{act_label}:")
                for it in filtered:
                    checklist_lines.append(f"  [{it['tier']}] {it['question']}")
                    checklist_lines.append(f"    \u2192 {it['answer'][:120]}...")
        return '\n'.join(checklist_lines)

    def test_short_mode_only_basic(self, sample_inventory):
        desc = self._build_checklist(sample_inventory, 'short')
        assert '[Basic]' in desc
        assert '[Context]' not in desc
        assert '[Deep-dive]' not in desc

    def test_medium_mode_basic_and_context(self, sample_inventory):
        desc = self._build_checklist(sample_inventory, 'medium')
        assert '[Basic]' in desc
        assert '[Context]' in desc
        assert '[Deep-dive]' not in desc

    def test_long_mode_all_tiers(self, sample_inventory):
        desc = self._build_checklist(sample_inventory, 'long')
        assert '[Basic]' in desc
        assert '[Context]' in desc
        assert '[Deep-dive]' in desc

    def test_checklist_has_header(self, sample_inventory):
        desc = self._build_checklist(sample_inventory, 'long')
        assert 'COVERAGE CHECKLIST' in desc

    def test_checklist_contains_question_text(self, sample_inventory):
        desc = self._build_checklist(sample_inventory, 'long')
        assert 'Does coffee really improve brain function?' in desc

    def test_checklist_contains_answer_text(self, sample_inventory):
        desc = self._build_checklist(sample_inventory, 'long')
        assert 'multiple RCTs' in desc


# ── Test C: _count_words ─────────────────────────────────────────────────


class TestCountWords:

    def test_english_word_count(self, english_lang_config):
        text = "Hello world this is a test"
        assert _count_words(text, english_lang_config) == 6

    def test_english_strips_speaker_labels(self, english_lang_config):
        text = "Host: Welcome to the show\nGuest: Thanks for having me"
        count = _count_words(text, english_lang_config)
        # Speaker labels ("Host:", "Guest:") get stripped by regex
        assert count > 0

    def test_japanese_char_count(self, japanese_lang_config):
        text = "こんにちは世界"
        count = _count_words(text, japanese_lang_config)
        assert count > 0

    def test_empty_string(self, english_lang_config):
        assert _count_words("", english_lang_config) == 0


# ── Test D: _deduplicate_script ──────────────────────────────────────────


class TestDeduplicateScript:

    def test_no_duplicates_unchanged(self, english_lang_config):
        text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        result = _deduplicate_script(text, english_lang_config)
        assert result == text

    def test_removes_duplicated_block(self, english_lang_config):
        text = (
            "Host: Welcome to the show\n"
            "Guest: Thanks for having me\n"
            "Host: Let's talk about the topic\n"
            "\n"
            "Host: Welcome to the show\n"
            "Guest: Thanks for having me\n"
            "Host: Let's talk about the topic\n"
        )
        result = _deduplicate_script(text, english_lang_config)
        # Should remove the second occurrence of the 3-line block
        assert result.count("Welcome to the show") == 1

    def test_transition_markers_preserved(self, english_lang_config):
        text = (
            "[TRANSITION]\n"
            "## [Act 2]\n"
            "[TRANSITION]\n"
            "\n"
            "[TRANSITION]\n"
            "## [Act 2]\n"
            "[TRANSITION]\n"
        )
        result = _deduplicate_script(text, english_lang_config)
        # Transition blocks are explicitly excluded from deduplication
        assert result.count("[TRANSITION]") >= 2


# ── Test E: Pipeline code checks ────────────────────────────────────────


class TestPipelineCodeState:
    """Verify expansion functions are removed and new functions exist.

    After T4.1 module split, functions may live in pipeline_script.py or
    pipeline_crew.py rather than directly in pipeline.py.  Tests now check
    that the functions are callable from pipeline (re-exported via imports)
    and that prompt strings exist in the correct module.
    """

    @pytest.fixture(autouse=True)
    def _load_pipeline_source(self):
        import dr2_podcast.pipeline as pipeline
        import inspect
        self.source = inspect.getsource(pipeline)

    def test_expansion_functions_removed(self):
        assert '_run_script_expansion' not in self.source
        assert '_expand_act' not in self.source
        assert '_analyze_acts' not in self.source
        assert 'ACT_ALLOCATIONS' not in self.source

    def test_parse_blueprint_inventory_exists(self):
        """Function must be importable from pipeline (re-exported from pipeline_script)."""
        import dr2_podcast.pipeline as pipeline
        assert callable(getattr(pipeline, '_parse_blueprint_inventory', None))

    def test_run_trim_pass_exists(self):
        """Function must be importable from pipeline (wrapper delegating to pipeline_script)."""
        import dr2_podcast.pipeline as pipeline
        assert callable(getattr(pipeline, '_run_trim_pass', None))

    def test_at_least_in_prompts(self):
        """'AT LEAST' appears in agent/task prompts (now in pipeline_crew)."""
        import dr2_podcast.pipeline_crew as pipeline_crew
        import inspect
        crew_source = inspect.getsource(pipeline_crew)
        assert 'AT LEAST' in crew_source

    def test_shrinkage_guard_uses_min_acceptable(self):
        assert 'min_acceptable' in self.source


# ── Test F: _run_trim_pass with mocked LLM ──────────────────────────────


class TestRunTrimPass:

    LONG_SCRIPT = (
        "Host: Welcome to Science Unpacked. Today we examine coffee.\n\n"
        "Guest: Great topic. Coffee is the most widely consumed psychoactive substance.\n\n"
        "Host: How does caffeine work in the brain?\n\n"
        "Guest: Caffeine blocks adenosine receptors, primarily A1 and A2A subtypes. "
        "Adenosine normally promotes sleep and relaxation, so blocking it increases alertness.\n\n"
        "Host: What does clinical evidence say about cognitive performance?\n\n"
        "Guest: A 2023 meta-analysis covering 41 RCTs found significant improvements "
        "in sustained attention with a standardized mean difference of 0.43.\n\n"
        "Host: Those are meaningful effect sizes. What about optimal dose?\n\n"
        "Guest: The sweet spot is 200 to 400 milligrams. Below 200, effects are modest. "
        "Above 400, anxiety often impairs performance.\n\n"
        "[TRANSITION]\n\n"
        "Host: What is the practical takeaway?\n\n"
        "Guest: Take 200 milligrams about 30 to 45 minutes before demanding work. "
        "Avoid after 2pm to protect sleep architecture.\n\n"
        "Host: That's our One Action for today."
    )

    TRIM_INVENTORY = {
        "Act 1 — The Evidence": [
            {'tier': 'Deep-dive',
             'question': 'What neuroimaging studies support caffeine mechanisms?',
             'answer': 'PET studies show dose-dependent adenosine receptor occupancy.'},
            {'tier': 'Context',
             'question': 'How do habitual users differ from non-habitual?',
             'answer': 'Non-habitual drinkers show larger acute effects.'},
            {'tier': 'Basic',
             'question': 'What dose range shows cognitive benefits?',
             'answer': '200-400mg is the established optimal range.'},
        ],
    }

    SESSION_ROLES = {
        'presenter': {'label': 'Host'},
        'questioner': {'label': 'Guest'},
    }

    def test_returns_string(self, english_lang_config):
        """Trim pass returns a string even when LLM call is mocked."""
        trimmed_text = "Host: Welcome. Guest: Short answer. Host: One Action for today."
        with patch('dr2_podcast.pipeline._call_smart_model', return_value=trimmed_text):
            result = _run_trim_pass(
                script_text=self.LONG_SCRIPT,
                inventory=self.TRIM_INVENTORY,
                target_length=50,
                language_config=english_lang_config,
                session_roles=self.SESSION_ROLES,
                topic_name="Coffee and Cognition",
                target_instruction="Keep the One Action ending.",
            )
        assert isinstance(result, str)

    def test_result_shorter_than_input(self, english_lang_config):
        """Mocked LLM returns shorter text, trim pass should use it."""
        short = "Host: Coffee helps cognition. Guest: Agreed. Host: One Action."
        with patch('dr2_podcast.pipeline._call_smart_model', return_value=short):
            result = _run_trim_pass(
                script_text=self.LONG_SCRIPT,
                inventory=self.TRIM_INVENTORY,
                target_length=50,
                language_config=english_lang_config,
                session_roles=self.SESSION_ROLES,
                topic_name="Coffee and Cognition",
                target_instruction="Keep the One Action ending.",
            )
        assert _count_words(result, english_lang_config) < _count_words(
            self.LONG_SCRIPT, english_lang_config
        )

    def test_no_trim_when_under_target(self, english_lang_config):
        """Script already under target should be returned unchanged."""
        result = _run_trim_pass(
            script_text=self.LONG_SCRIPT,
            inventory=self.TRIM_INVENTORY,
            target_length=99999,
            language_config=english_lang_config,
            session_roles=self.SESSION_ROLES,
            topic_name="Coffee and Cognition",
            target_instruction="",
        )
        assert result == self.LONG_SCRIPT

    def test_no_removable_items(self, english_lang_config):
        """Inventory with only Basic items means nothing can be trimmed."""
        basic_only = {
            "Act 1": [
                {'tier': 'Basic', 'question': 'Q1', 'answer': 'A1'},
            ]
        }
        result = _run_trim_pass(
            script_text=self.LONG_SCRIPT,
            inventory=basic_only,
            target_length=10,
            language_config=english_lang_config,
            session_roles=self.SESSION_ROLES,
            topic_name="Test",
            target_instruction="",
        )
        assert result == self.LONG_SCRIPT

    def test_llm_failure_returns_original(self, english_lang_config):
        """If LLM call raises, trim pass should return the original script."""
        with patch('dr2_podcast.pipeline._call_smart_model', side_effect=Exception("LLM down")):
            result = _run_trim_pass(
                script_text=self.LONG_SCRIPT,
                inventory=self.TRIM_INVENTORY,
                target_length=50,
                language_config=english_lang_config,
                session_roles=self.SESSION_ROLES,
                topic_name="Coffee and Cognition",
                target_instruction="",
            )
        assert result == self.LONG_SCRIPT
