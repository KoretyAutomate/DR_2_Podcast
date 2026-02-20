"""Unit tests for clinical_math.py — deterministic ARR/NNT calculator."""

import math
import pytest
from clinical_math import ClinicalImpact, calculate_impact, batch_calculate, format_math_report


class TestCalculateImpact:
    def test_benefit_basic(self):
        """Standard case: treatment reduces event rate."""
        result = calculate_impact("Study-1", cer=0.20, eer=0.10)
        assert result is not None
        assert result.direction == "benefit"
        assert result.arr == pytest.approx(0.10, abs=1e-6)
        assert result.rrr == pytest.approx(0.50, abs=1e-4)
        assert result.nnt == pytest.approx(10.0, abs=0.1)
        assert "prevent" in result.nnt_interpretation

    def test_harm(self):
        """EER > CER: treatment causes harm."""
        result = calculate_impact("Study-2", cer=0.10, eer=0.20)
        assert result is not None
        assert result.direction == "harm"
        assert result.arr == pytest.approx(-0.10, abs=1e-6)
        assert result.nnt == pytest.approx(10.0, abs=0.1)
        assert "cause" in result.nnt_interpretation

    def test_no_effect(self):
        """CER == EER: no measurable effect."""
        result = calculate_impact("Study-3", cer=0.15, eer=0.15)
        assert result is not None
        assert result.direction == "no_effect"
        assert result.arr == 0.0
        assert result.rrr == 0.0
        assert result.nnt == float('inf')

    def test_null_cer(self):
        """None CER should return None."""
        result = calculate_impact("Study-4", cer=None, eer=0.10)
        assert result is None

    def test_null_eer(self):
        """None EER should return None."""
        result = calculate_impact("Study-5", cer=0.20, eer=None)
        assert result is None

    def test_both_null(self):
        """Both None should return None."""
        result = calculate_impact("Study-6", cer=None, eer=None)
        assert result is None

    def test_zero_cer(self):
        """CER=0 edge case: avoid division by zero in RRR."""
        result = calculate_impact("Study-7", cer=0.0, eer=0.05)
        assert result is not None
        assert result.direction == "harm"
        assert result.rrr == 0.0  # CER is 0, so RRR = ARR/0 = 0 by guard

    def test_very_small_effect(self):
        """Very small effect near epsilon threshold."""
        result = calculate_impact("Study-8", cer=0.100, eer=0.099)
        assert result is not None
        assert result.direction == "benefit"
        assert result.nnt == pytest.approx(1000.0, abs=1.0)

    def test_large_effect(self):
        """Large treatment effect."""
        result = calculate_impact("Study-9", cer=0.50, eer=0.10)
        assert result is not None
        assert result.arr == pytest.approx(0.40, abs=1e-6)
        assert result.nnt == pytest.approx(2.5, abs=0.1)

    def test_study_id_preserved(self):
        """Study ID should be preserved in output."""
        result = calculate_impact("PMID:12345678", cer=0.20, eer=0.10)
        assert result.study_id == "PMID:12345678"


class TestBatchCalculate:
    def _make_extraction(self, pmid, title, cer, eer):
        """Helper to create a mock DeepExtraction-like object."""
        from types import SimpleNamespace
        return SimpleNamespace(
            pmid=pmid, title=title,
            control_event_rate=cer,
            experimental_event_rate=eer
        )

    def test_mixed_batch(self):
        """Batch with some having CER/EER and some not."""
        extractions = [
            self._make_extraction("1", "Study A", 0.20, 0.10),
            self._make_extraction("2", "Study B", None, 0.10),
            self._make_extraction("3", "Study C", 0.30, None),
            self._make_extraction("4", "Study D", 0.15, 0.05),
        ]
        results = batch_calculate(extractions)
        assert len(results) == 2
        assert results[0].study_id == "1"
        assert results[1].study_id == "4"

    def test_empty_batch(self):
        """Empty list should return empty results."""
        results = batch_calculate([])
        assert results == []

    def test_all_null(self):
        """No studies with both CER and EER."""
        extractions = [
            self._make_extraction("1", "Study A", None, None),
            self._make_extraction("2", "Study B", 0.10, None),
        ]
        results = batch_calculate(extractions)
        assert results == []


class TestFormatMathReport:
    def test_empty_report(self):
        """No impacts produces informative message."""
        report = format_math_report([])
        assert "No studies" in report

    def test_single_impact(self):
        """Single impact formats correctly."""
        impact = ClinicalImpact(
            study_id="Test-1", cer=0.20, eer=0.10,
            arr=0.10, rrr=0.50, nnt=10.0,
            nnt_interpretation="Treat 10 patients to prevent 1 additional event",
            direction="benefit"
        )
        report = format_math_report([impact])
        assert "Test-1" in report
        assert "0.200" in report
        assert "0.100" in report
        assert "10.0" in report
        assert "benefit" in report
        assert "Treat 10" in report

    def test_multiple_impacts(self):
        """Multiple impacts produce table rows."""
        impacts = [
            ClinicalImpact("A", 0.20, 0.10, 0.10, 0.50, 10.0, "Treat 10...", "benefit"),
            ClinicalImpact("B", 0.10, 0.20, -0.10, -1.00, 10.0, "Treat 10...", "harm"),
        ]
        report = format_math_report(impacts)
        assert report.count("|") > 10  # Has table structure
        assert "A" in report
        assert "B" in report

    def test_report_has_header(self):
        """Report includes header."""
        impacts = [ClinicalImpact("X", 0.1, 0.05, 0.05, 0.5, 20.0, "...", "benefit")]
        report = format_math_report(impacts)
        assert "Deterministic Clinical Impact" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
