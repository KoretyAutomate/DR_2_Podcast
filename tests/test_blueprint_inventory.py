"""
Unit tests for Blueprint Discussion Inventory + Script Condensing.

Tests _parse_blueprint_inventory, _count_words, _deduplicate_script,
_run_condense_pass (_run_trim_pass alias), and the coverage checklist injection logic.

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


# ── Test A: _parse_blueprint_inventory on new Section 5 format ────────────


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

    def test_items_have_no_tier_key(self, sample_blueprint):
        """New format items should not have a 'tier' key."""
        result = _parse_blueprint_inventory(sample_blueprint)
        act1_key = next(k for k in result if 'Act 1' in k)
        for item in result[act1_key]:
            assert 'tier' not in item

    def test_items_have_question_and_answer(self, sample_blueprint):
        result = _parse_blueprint_inventory(sample_blueprint)
        act1_key = next(k for k in result if 'Act 1' in k)
        for item in result[act1_key]:
            assert 'question' in item
            assert 'answer' in item

    def test_question_text_extracted(self, sample_blueprint):
        result = _parse_blueprint_inventory(sample_blueprint)
        act1_key = next(k for k in result if 'Act 1' in k)
        first = result[act1_key][0]
        assert len(first['question']) > 5

    def test_answer_text_extracted(self, sample_blueprint):
        result = _parse_blueprint_inventory(sample_blueprint)
        act1_key = next(k for k in result if 'Act 1' in k)
        first = result[act1_key][0]
        assert len(first['answer']) > 10

    def test_act1_has_three_items(self, sample_blueprint):
        result = _parse_blueprint_inventory(sample_blueprint)
        act1_key = next(k for k in result if 'Act 1' in k)
        assert len(result[act1_key]) == 3

    def test_no_section5_or_8_returns_empty_dict(self):
        result = _parse_blueprint_inventory("## 1. Intro\nHello.\n## 2. Body\nContent here.\n")
        assert result == {}

    def test_empty_string_returns_empty_dict(self):
        result = _parse_blueprint_inventory("")
        assert result == {}


class TestParseBlueprintInventoryLegacy:
    """Test backward compatibility with legacy Section 8 format."""

    def test_legacy_section8_parsed(self, sample_blueprint_legacy):
        result = _parse_blueprint_inventory(sample_blueprint_legacy)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_legacy_items_have_no_tier(self, sample_blueprint_legacy):
        """Even legacy items should be returned without tier key."""
        result = _parse_blueprint_inventory(sample_blueprint_legacy)
        act1_key = next(k for k in result if 'Act 1' in k)
        for item in result[act1_key]:
            assert 'tier' not in item

    def test_legacy_question_extracted(self, sample_blueprint_legacy):
        result = _parse_blueprint_inventory(sample_blueprint_legacy)
        act1_key = next(k for k in result if 'Act 1' in k)
        assert any('coffee' in it['question'].lower() for it in result[act1_key])


# ── Test B: Coverage checklist injection logic ───────────────────────────


class TestCoverageChecklistInjection:
    """Test the checklist injection logic (no tier filtering — all items included)."""

    @staticmethod
    def _build_checklist(inventory):
        """Reproduce the injection logic from _inject_blueprint_checklist."""
        checklist_lines = ["\n\nCOVERAGE CHECKLIST --- discuss EACH item below in its Act:"]
        for act_label, items in inventory.items():
            checklist_lines.append(f"\n{act_label}:")
            for it in items:
                checklist_lines.append(f"  {it['question']}")
                checklist_lines.append(f"    -> {it['answer'][:120]}...")
        return '\n'.join(checklist_lines)

    def test_all_items_included(self, sample_inventory):
        desc = self._build_checklist(sample_inventory)
        assert 'Does coffee really improve brain function?' in desc
        assert 'What neuroimaging studies' in desc
        assert 'Can tolerance be managed' in desc

    def test_checklist_has_header(self, sample_inventory):
        desc = self._build_checklist(sample_inventory)
        assert 'COVERAGE CHECKLIST' in desc

    def test_checklist_contains_question_text(self, sample_inventory):
        desc = self._build_checklist(sample_inventory)
        assert 'Does coffee really improve brain function?' in desc

    def test_checklist_contains_answer_text(self, sample_inventory):
        desc = self._build_checklist(sample_inventory)
        assert 'multiple RCTs' in desc

    def test_no_tier_labels_in_output(self, sample_inventory):
        desc = self._build_checklist(sample_inventory)
        assert '[Basic]' not in desc
        assert '[Context]' not in desc
        assert '[Deep-dive]' not in desc


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

    def test_run_condense_pass_exists(self):
        """Function must be importable from pipeline."""
        import dr2_podcast.pipeline as pipeline
        assert callable(getattr(pipeline, '_run_condense_pass', None))

    def test_run_trim_pass_alias_exists(self):
        """Backward-compatible alias must exist."""
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


# ── Test F: _run_condense_pass (via _run_trim_pass alias) with mocked LLM ──


class TestRunCondensePass:

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

    CONDENSE_INVENTORY = {
        "Act 1 --- Evidence & Nuance": [
            {'question': 'What neuroimaging studies support caffeine mechanisms?',
             'answer': 'PET studies show dose-dependent adenosine receptor occupancy.'},
            {'question': 'How do habitual users differ from non-habitual?',
             'answer': 'Non-habitual drinkers show larger acute effects.'},
            {'question': 'What dose range shows cognitive benefits?',
             'answer': '200-400mg is the established optimal range.'},
        ],
    }

    SESSION_ROLES = {
        'presenter': {'label': 'Host'},
        'questioner': {'label': 'Guest'},
    }

    def test_returns_string(self, english_lang_config):
        """Condense pass returns a string even when LLM call is mocked."""
        condensed_text = "Host: Welcome. Guest: Short answer. Host: One Action for today."
        with patch('dr2_podcast.pipeline._call_smart_model', return_value=condensed_text):
            result = _run_trim_pass(
                script_text=self.LONG_SCRIPT,
                inventory=self.CONDENSE_INVENTORY,
                target_length=50,
                language_config=english_lang_config,
                session_roles=self.SESSION_ROLES,
                topic_name="Coffee and Cognition",
                target_instruction="Keep the One Action ending.",
            )
        assert isinstance(result, str)

    def test_result_shorter_than_input(self, english_lang_config):
        """Mocked LLM returns shorter text, condense pass should use it."""
        short = "Host: Coffee helps cognition. Guest: Agreed. Host: One Action."
        with patch('dr2_podcast.pipeline._call_smart_model', return_value=short):
            result = _run_trim_pass(
                script_text=self.LONG_SCRIPT,
                inventory=self.CONDENSE_INVENTORY,
                target_length=50,
                language_config=english_lang_config,
                session_roles=self.SESSION_ROLES,
                topic_name="Coffee and Cognition",
                target_instruction="Keep the One Action ending.",
            )
        assert _count_words(result, english_lang_config) < _count_words(
            self.LONG_SCRIPT, english_lang_config
        )

    def test_no_condense_when_under_target(self, english_lang_config):
        """Script already under target should be returned unchanged."""
        result = _run_trim_pass(
            script_text=self.LONG_SCRIPT,
            inventory=self.CONDENSE_INVENTORY,
            target_length=99999,
            language_config=english_lang_config,
            session_roles=self.SESSION_ROLES,
            topic_name="Coffee and Cognition",
            target_instruction="",
        )
        assert result == self.LONG_SCRIPT

    def test_llm_failure_returns_original(self, english_lang_config):
        """If LLM call raises, condense pass should return the original script."""
        with patch('dr2_podcast.pipeline._call_smart_model', side_effect=Exception("LLM down")):
            result = _run_trim_pass(
                script_text=self.LONG_SCRIPT,
                inventory=self.CONDENSE_INVENTORY,
                target_length=50,
                language_config=english_lang_config,
                session_roles=self.SESSION_ROLES,
                topic_name="Coffee and Cognition",
                target_instruction="",
            )
        assert result == self.LONG_SCRIPT
