"""Tests for Source-of-Truth (IMRaD) builder functions.

File under test: dr2_podcast/pipeline_sot.py
"""

from types import SimpleNamespace

import pytest

from dr2_podcast.pipeline_sot import (
    _extract_conclusion_status,
    _parse_grade_sections,
    _format_study_characteristics_table,
    _format_references,
    build_imrad_sot,
)


# ---------------------------------------------------------------------------
# _extract_conclusion_status
# ---------------------------------------------------------------------------

class TestExtractConclusionStatus:

    # --- Clinical domain (GRADE) ---

    def test_clinical_high(self):
        grade, status, _ = _extract_conclusion_status("Final GRADE: High", "clinical")
        assert grade == "High"
        assert status == "Scientifically Supported"

    def test_clinical_moderate(self):
        grade, status, _ = _extract_conclusion_status("Final GRADE: Moderate", "clinical")
        assert grade == "Moderate"
        assert status == "Partially Supported \u2014 Further Research Recommended"

    def test_clinical_low(self):
        grade, status, _ = _extract_conclusion_status("Final GRADE: Low", "clinical")
        assert grade == "Low"
        assert status == "Insufficient Evidence \u2014 More Research Needed"

    def test_clinical_very_low(self):
        grade, status, _ = _extract_conclusion_status("Final GRADE: Very Low", "clinical")
        assert grade == "Very Low"
        assert status == "Not Supported by Current Evidence"

    def test_clinical_no_match(self):
        grade, status, _ = _extract_conclusion_status("No GRADE info here.", "clinical")
        assert grade == "Not Determined"
        assert status == "Under Evaluation"

    # --- Social science domain ---

    def test_social_science_strong(self):
        grade, status, _ = _extract_conclusion_status(
            "Final Evidence Quality: STRONG", "social_science"
        )
        assert grade == "STRONG"
        assert status == "Scientifically Supported"

    def test_social_science_moderate(self):
        grade, status, _ = _extract_conclusion_status(
            "Final Evidence Quality: MODERATE", "social_science"
        )
        assert grade == "MODERATE"
        assert status == "Partially Supported \u2014 Further Research Recommended"

    def test_social_science_weak(self):
        grade, status, _ = _extract_conclusion_status(
            "Final Evidence Quality: WEAK", "social_science"
        )
        assert grade == "WEAK"
        assert status == "Insufficient Evidence \u2014 More Research Needed"

    # --- Executive summary ---

    def test_executive_summary_extracted(self):
        text = (
            "Final GRADE: High\n"
            "Executive Summary:\n"
            "The evidence strongly supports the intervention.\n"
            "\n"
            "## Next Section"
        )
        _, _, summary = _extract_conclusion_status(text, "clinical")
        assert "strongly supports" in summary

    def test_no_executive_summary(self):
        _, _, summary = _extract_conclusion_status("Final GRADE: Low\nSome text.", "clinical")
        assert summary == ""


# ---------------------------------------------------------------------------
# _parse_grade_sections
# ---------------------------------------------------------------------------

class TestParseGradeSections:

    def test_standard_sections(self):
        text = "### Evidence Profile\ntext1\n### GRADE Assessment\ntext2"
        result = _parse_grade_sections(text)
        assert "evidence profile" in result
        assert result["evidence profile"] == "text1"
        assert "grade assessment" in result
        assert result["grade assessment"] == "text2"

    def test_empty_string(self):
        assert _parse_grade_sections("") == {}

    def test_no_headers(self):
        result = _parse_grade_sections("Just plain text without any headers.")
        assert result == {}

    def test_section_with_no_content(self):
        text = "### Empty Section\n### Another Section\ncontent here"
        result = _parse_grade_sections(text)
        assert result["empty section"] == ""
        assert result["another section"] == "content here"


# ---------------------------------------------------------------------------
# _format_study_characteristics_table
# ---------------------------------------------------------------------------

def _make_extraction(**overrides):
    """Create a minimal DeepExtraction-like object for table formatting."""
    defaults = dict(
        pmid="12345678", doi="10.1000/test", title="Test Study Title",
        study_design="RCT", sample_size_total=100,
        demographics="Adults aged 30-60", follow_up_period="12 weeks",
        funding_source="NIH", risk_of_bias="Low", research_tier=1,
        paper_metadata=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestFormatStudyCharacteristicsTable:

    def test_empty_list(self):
        result = _format_study_characteristics_table([])
        assert "No studies" in result

    def test_single_extraction(self):
        ext = _make_extraction()
        result = _format_study_characteristics_table([ext])
        assert "| 1 " in result
        assert "RCT" in result
        assert "PMID:12345678" in result

    def test_deduplication_by_pmid(self):
        ext1 = _make_extraction(pmid="111", title="Study A")
        ext2 = _make_extraction(pmid="111", title="Study A duplicate")
        result = _format_study_characteristics_table([ext1, ext2])
        # Only 1 data row (header + separator + 1 row)
        lines = [l for l in result.strip().split("\n") if l.startswith("|")]
        assert len(lines) == 3  # header, separator, 1 data row


# ---------------------------------------------------------------------------
# _format_references
# ---------------------------------------------------------------------------

def _make_wnr(**overrides):
    """Create a minimal WideNetRecord-like object for reference formatting."""
    defaults = dict(
        pmid="12345678", title="Test Study Title",
        authors="Smith J et al.", journal="Test Journal", year=2023,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestFormatReferences:

    def test_empty_lists(self):
        result = _format_references([], [])
        assert "No references available" in result

    def test_single_extraction_with_matching_wnr(self):
        ext = _make_extraction(doi="10.1000/test")
        wnr = _make_wnr()
        result = _format_references([ext], [wnr])
        assert "1." in result
        assert "Smith J et al." in result
        assert "Test Journal" in result
        assert "(2023)" in result

    def test_deduplication(self):
        ext1 = _make_extraction(pmid="111")
        ext2 = _make_extraction(pmid="111")
        wnr = _make_wnr(pmid="111")
        result = _format_references([ext1, ext2], [wnr])
        # Should contain only one numbered reference
        assert "1." in result
        assert "2." not in result

    def test_missing_wnr_metadata(self):
        ext = _make_extraction(pmid="999", doi=None)
        result = _format_references([ext], [])
        assert "Unknown authors" in result


# ---------------------------------------------------------------------------
# build_imrad_sot
# ---------------------------------------------------------------------------

def _make_report(report_text=""):
    """Create a minimal report-like object with .report attribute."""
    return SimpleNamespace(
        report=report_text, total_summaries=0,
        total_urls_fetched=0, duration_seconds=0, sources=[],
    )


class TestBuildImradSot:

    def test_clinical_domain_sections(self):
        reports = {
            "pipeline_data": {
                "aff_extractions": [], "fal_extractions": [],
                "aff_top": [], "fal_top": [],
                "impacts": [], "metrics": {},
                "framing_context": "Test topic",
                "search_date": "2026-01-01",
            },
            "audit": _make_report("Final GRADE: Moderate\nSome audit text."),
            "lead": _make_report("Affirmative case."),
            "counter": _make_report("Falsification case."),
        }
        result = build_imrad_sot("Test Topic", reports, "sufficient", 5,
                                 domain="clinical")
        assert "## Abstract" in result
        assert "## 1. Introduction" in result
        assert "## 2. Methods" in result
        assert "## 3. Results" in result
        assert "## 4. Discussion" in result
        assert "## 5. References" in result

    def test_social_science_domain_sections(self):
        reports = {
            "pipeline_data": {
                "domain": "social_science",
                "aff_extractions": [], "fal_extractions": [],
                "aff_top": [], "fal_top": [],
                "impacts": [], "metrics": {},
                "framing_context": "Education topic",
                "search_date": "2026-01-01",
            },
            "audit": _make_report("Final Evidence Quality: STRONG\nSome text."),
            "lead": _make_report("Affirmative case."),
            "counter": _make_report("Falsification case."),
        }
        result = build_imrad_sot("Education Topic", reports, "sufficient", 5,
                                 domain="clinical")  # domain auto-detected from pipeline_data
        assert "## 1. Abstract" in result
        assert "## 2. Introduction" in result
        assert "## 3. Methods" in result
        assert "## 4. Results" in result
        assert "## 5. Discussion" in result
        assert "## 6. References" in result

    def test_missing_pipeline_data_keys_no_crash(self):
        """build_imrad_sot should handle missing keys gracefully."""
        reports = {
            "pipeline_data": {},
            "audit": _make_report(""),
            "lead": _make_report(""),
            "counter": _make_report(""),
        }
        result = build_imrad_sot("Missing Keys Topic", reports, "unknown", 0)
        assert isinstance(result, str)
        assert "Missing Keys Topic" in result
