"""
Unit tests for effect_size_math.py.
"""

import math

import pytest

from dr2_podcast.research.effect_size_math import (
    classify_magnitude_d,
    hedges_g_correction,
    odds_ratio_to_d,
    r_to_d,
    d_to_r,
    calculate_effect,
    batch_calculate,
    format_effect_size_report,
    EffectSizeImpact,
)


class TestClassifyMagnitude:

    def test_negligible(self):
        assert classify_magnitude_d(0.0) == "negligible"
        assert classify_magnitude_d(0.1) == "negligible"
        assert classify_magnitude_d(0.19) == "negligible"
        assert classify_magnitude_d(-0.15) == "negligible"

    def test_small(self):
        assert classify_magnitude_d(0.2) == "small"
        assert classify_magnitude_d(0.35) == "small"
        assert classify_magnitude_d(0.49) == "small"
        assert classify_magnitude_d(-0.3) == "small"

    def test_medium(self):
        assert classify_magnitude_d(0.5) == "medium"
        assert classify_magnitude_d(0.65) == "medium"
        assert classify_magnitude_d(0.79) == "medium"
        assert classify_magnitude_d(-0.6) == "medium"

    def test_large(self):
        assert classify_magnitude_d(0.8) == "large"
        assert classify_magnitude_d(1.5) == "large"
        assert classify_magnitude_d(-1.0) == "large"


class TestHedgesGCorrection:

    def test_large_sample(self):
        """For large N, g ≈ d."""
        d = 0.5
        g = hedges_g_correction(d, 1000)
        assert abs(g - d) < 0.01

    def test_small_sample(self):
        """For small N, correction is meaningful."""
        d = 0.5
        g = hedges_g_correction(d, 10)
        assert g < d  # Correction shrinks d
        assert g > 0  # But keeps sign

    def test_formula_exact(self):
        """Verify exact formula: g = d * (1 - 3/(4n-9))."""
        d, n = 0.6, 20
        expected = d * (1 - 3 / (4 * n - 9))
        assert hedges_g_correction(d, n) == pytest.approx(expected)

    def test_very_small_n(self):
        """n < 4: returns d unchanged (correction undefined)."""
        assert hedges_g_correction(0.5, 3) == 0.5
        assert hedges_g_correction(0.5, 2) == 0.5

    def test_n_equals_4(self):
        """n = 4: correction factor = 1 - 3/7 ≈ 0.571."""
        d = 1.0
        g = hedges_g_correction(d, 4)
        assert g == pytest.approx(1.0 * (1 - 3/7))

    def test_negative_d(self):
        d = -0.5
        g = hedges_g_correction(d, 50)
        assert g < 0  # Preserves sign
        assert abs(g) < abs(d)  # Shrinks magnitude


class TestOddsRatioToD:

    def test_or_1(self):
        """OR = 1 means no effect → d = 0."""
        assert odds_ratio_to_d(1.0) == pytest.approx(0.0)

    def test_or_positive_effect(self):
        """OR > 1 → positive d."""
        d = odds_ratio_to_d(2.0)
        assert d is not None
        assert d > 0
        # ln(2) * sqrt(3) / pi ≈ 0.3830
        assert d == pytest.approx(math.log(2) * math.sqrt(3) / math.pi, rel=0.01)

    def test_or_negative_effect(self):
        """OR < 1 → negative d."""
        d = odds_ratio_to_d(0.5)
        assert d is not None
        assert d < 0

    def test_or_invalid(self):
        assert odds_ratio_to_d(0) is None
        assert odds_ratio_to_d(-1) is None
        assert odds_ratio_to_d(None) is None


class TestRToD:

    def test_r_zero(self):
        """r = 0 → d = 0."""
        assert r_to_d(0.0) == pytest.approx(0.0)

    def test_r_positive(self):
        """r = 0.5 → d > 0."""
        d = r_to_d(0.5)
        assert d is not None
        assert d > 0
        # d = 2*0.5 / sqrt(1 - 0.25) = 1/sqrt(0.75) ≈ 1.1547
        assert d == pytest.approx(2 * 0.5 / math.sqrt(0.75), rel=0.01)

    def test_r_negative(self):
        d = r_to_d(-0.3)
        assert d is not None
        assert d < 0

    def test_r_boundary(self):
        """r = ±1 is undefined."""
        assert r_to_d(1.0) is None
        assert r_to_d(-1.0) is None

    def test_r_none(self):
        assert r_to_d(None) is None


class TestDToR:

    def test_d_zero(self):
        assert d_to_r(0.0) == pytest.approx(0.0)

    def test_roundtrip(self):
        """d → r → d should be approximately identity."""
        original_d = 0.5
        r = d_to_r(original_d)
        recovered_d = r_to_d(r)
        assert recovered_d == pytest.approx(original_d, rel=0.01)


class TestCalculateEffect:

    def test_cohens_d(self):
        impact = calculate_effect("Study1", "cohens_d", 0.5, sample_size=100)
        assert impact is not None
        assert impact.cohens_d == pytest.approx(0.5)
        assert impact.magnitude == "medium"
        assert impact.direction == "positive"

    def test_cohens_d_negative(self):
        impact = calculate_effect("Study2", "cohens_d", -0.3)
        assert impact is not None
        assert impact.magnitude == "small"
        assert impact.direction == "negative"

    def test_odds_ratio(self):
        impact = calculate_effect("Study3", "odds_ratio", 2.0, sample_size=50)
        assert impact is not None
        assert impact.cohens_d is not None
        assert impact.cohens_d > 0
        assert impact.hedges_g is not None

    def test_correlation_r(self):
        impact = calculate_effect("Study4", "correlation_r", 0.3)
        assert impact is not None
        assert impact.cohens_d is not None
        assert impact.direction == "positive"

    def test_beta(self):
        impact = calculate_effect("Study5", "beta", 0.4)
        assert impact is not None
        assert impact.cohens_d is not None

    def test_invalid_type(self):
        impact = calculate_effect("Study6", "unknown_type", 0.5)
        assert impact is None

    def test_none_value(self):
        impact = calculate_effect("Study7", "cohens_d", None)
        assert impact is None

    def test_hedges_g_applied(self):
        """When sample_size is provided, Hedges' g should be calculated."""
        impact = calculate_effect("Study8", "cohens_d", 0.5, sample_size=20)
        assert impact is not None
        assert impact.hedges_g is not None
        assert abs(impact.hedges_g) < abs(impact.cohens_d)

    def test_null_effect(self):
        impact = calculate_effect("Study9", "cohens_d", 0.005)
        assert impact is not None
        assert impact.direction == "null"
        assert impact.magnitude == "negligible"


class TestBatchCalculate:

    def test_empty_list(self):
        assert batch_calculate([]) == []

    def test_with_mock_extractions(self):
        """Test with objects that have the expected attributes."""

        class MockExtraction:
            def __init__(self, title, es_value, es_type, n=None):
                self.title = title
                self.pmid = None
                self.effect_size_value = es_value
                self.effect_size_type = es_type
                self.sample_size_total = n

        extractions = [
            MockExtraction("Study A", 0.5, "cohens_d", 100),
            MockExtraction("Study B", 2.0, "odds_ratio", 50),
            MockExtraction("Study C", None, None),  # No data
            MockExtraction("Study D", 0.3, "correlation_r"),
        ]

        results = batch_calculate(extractions)
        assert len(results) == 3  # Study C excluded
        assert results[0].study_id == "Study A"
        assert results[1].study_id == "Study B"
        assert results[2].study_id == "Study D"


class TestFormatReport:

    def test_empty_impacts(self):
        report = format_effect_size_report([])
        assert "not possible" in report.lower()

    def test_with_impacts(self):
        impacts = [
            EffectSizeImpact(
                study_id="Study1",
                effect_type="cohens_d",
                raw_value=0.5,
                cohens_d=0.5,
                hedges_g=0.48,
                magnitude="medium",
                direction="positive",
                interpretation="A medium positive effect (d = 0.500)",
                sample_size=100,
            ),
        ]
        report = format_effect_size_report(impacts)
        assert "Study1" in report
        assert "0.500" in report
        assert "0.480" in report
        assert "medium" in report
        assert "## Deterministic Effect Size Calculations" in report
