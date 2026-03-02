"""Tests for pipeline_translation.py — SOT splitting and script language audit."""

import pytest
from dr2_podcast.pipeline_translation import (
    _split_sot_imrad, _audit_script_language,
)


# ---------------------------------------------------------------------------
# _split_sot_imrad
# ---------------------------------------------------------------------------

class TestSplitSotImrad:

    def test_clinical_imrad_headers(self):
        sot = (
            "# Source of Truth: Test\n\n"
            "## Abstract\nAbstract text here.\n"
            "## 1. Introduction\nIntro text.\n"
            "## 2. Methods\nMethods text.\n"
            "## 3. Results\nResults text.\n"
            "## 4. Discussion\nDiscussion text.\n"
            "## 5. References\nRefs here.\n"
        )
        sections = _split_sot_imrad(sot)
        headers = [h for h, _ in sections if h]
        assert "## Abstract" in headers
        assert "## 1. Introduction" in headers
        assert "## 5. References" in headers

    def test_social_science_numbered_headers(self):
        sot = (
            "## 1. Abstract\nAbstract.\n"
            "## 2. Introduction\nIntro.\n"
            "## 3. Methods\nMethods.\n"
            "## 4. Results\nResults.\n"
            "## 5. Discussion\nDisc.\n"
            "## 6. References\nRefs.\n"
        )
        sections = _split_sot_imrad(sot)
        headers = [h for h, _ in sections if h]
        assert "## 1. Abstract" in headers
        assert "## 6. References" in headers

    def test_preamble_before_first_header(self):
        sot = "Some preamble text\n\n## Abstract\nAbstract body.\n"
        sections = _split_sot_imrad(sot)
        assert sections[0][0] == ""  # first entry has no header
        assert "preamble" in sections[0][1].lower()

    def test_embedded_subsections_preserved(self):
        sot = (
            "## 2. Methods\n"
            "### 2.1 Search Strategy\nSearch details.\n"
            "### 2.2 Data Collection\nData details.\n"
            "## 3. Results\nResults.\n"
        )
        sections = _split_sot_imrad(sot)
        # ### subsections should stay inside ## 2. Methods, not split
        methods_sections = [b for h, b in sections if h == "## 2. Methods"]
        assert len(methods_sections) == 1
        assert "### 2.1 Search Strategy" in methods_sections[0]
        assert "### 2.2 Data Collection" in methods_sections[0]

    def test_empty_input(self):
        sections = _split_sot_imrad("")
        assert len(sections) == 1
        assert sections[0] == ("", "")


# ---------------------------------------------------------------------------
# _audit_script_language
# ---------------------------------------------------------------------------

class TestAuditScriptLanguage:

    def test_english_returns_unchanged(self):
        script = "Host 1: Welcome to the show.\nHost 2: Great topic."
        result = _audit_script_language(
            script, "en", {"name": "English"},
            _call_smart_model=lambda **kw: "should not be called"
        )
        assert result == script

    def test_short_output_fallback(self):
        script = "Host 1: Long script text " * 20 + "\n[TRANSITION]\nHost 2: More text."
        result = _audit_script_language(
            script, "ja", {"name": "Japanese"},
            _call_smart_model=lambda **kw: "short"
        )
        # "short" is < 50% of original → fallback to original
        assert result == script

    def test_lost_transition_markers_fallback(self):
        script = (
            "Host 1: Some text.\n"
            "[TRANSITION]\n"
            "Host 2: More text.\n"
            "[TRANSITION]\n"
            "Host 1: Final text.\n"
        )
        # LLM returns text without [TRANSITION]
        def mock_smart(**kw):
            return "Host 1: Some text.\nHost 2: More text.\nHost 1: Final text."

        result = _audit_script_language(
            script, "ja", {"name": "Japanese"},
            _call_smart_model=mock_smart
        )
        assert result == script

    def test_llm_exception_returns_original(self):
        script = "Host 1: Test.\n[TRANSITION]\nHost 2: Test."

        def mock_fail(**kw):
            raise Exception("API error")

        result = _audit_script_language(
            script, "ja", {"name": "Japanese"},
            _call_smart_model=mock_fail
        )
        assert result == script

    def test_successful_audit(self):
        script = "Host 1: Test line.\n[TRANSITION]\nHost 2: Another line."
        corrected = "Host 1: Corrected line.\n[TRANSITION]\nHost 2: Another corrected line."

        result = _audit_script_language(
            script, "ja", {"name": "Japanese"},
            _call_smart_model=lambda **kw: corrected
        )
        assert result == corrected
