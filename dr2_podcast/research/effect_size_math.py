"""
Deterministic effect size calculator for social science research.
No LLM involvement — pure arithmetic to prevent hallucinated statistics.

Parallel to clinical_math.py but for social science effect measures:
- Cohen's d (standardized mean difference)
- Hedges' g (small-sample corrected d)
- Odds ratio to d conversion
- Pearson r to d conversion
- Magnitude classification (Cohen's conventions)
"""

import math
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Social science extraction types defined in social_science_research.py


@dataclass
class EffectSizeImpact:
    study_id: str              # ID or title
    effect_type: str           # "cohens_d", "hedges_g", "odds_ratio", "correlation_r", "beta"
    raw_value: float           # Original reported value
    cohens_d: Optional[float]  # Converted/normalized to Cohen's d
    hedges_g: Optional[float]  # Small-sample corrected (if n available)
    magnitude: str             # "negligible", "small", "medium", "large"
    direction: str             # "positive", "negative", "null"
    interpretation: str        # Human-readable interpretation
    sample_size: Optional[int] = None


def classify_magnitude_d(d: float) -> str:
    """Classify effect size magnitude using Cohen's conventions.

    |d| < 0.2  → negligible
    0.2 ≤ |d| < 0.5  → small
    0.5 ≤ |d| < 0.8  → medium
    |d| ≥ 0.8  → large
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def hedges_g_correction(d: float, n: int) -> float:
    """Apply Hedges' g small-sample correction to Cohen's d.

    g = d * (1 - 3/(4n - 9))

    For n < 10, the correction is substantial.
    For n > 50, the correction is negligible.
    """
    if n < 4:
        return d  # Correction undefined for very small samples
    correction = 1 - 3 / (4 * n - 9)
    return d * correction


def odds_ratio_to_d(or_val: float) -> Optional[float]:
    """Convert odds ratio to Cohen's d using the log-odds method.

    d = ln(OR) * sqrt(3) / pi

    Returns None if OR is invalid (≤ 0).
    """
    if or_val is None or or_val <= 0:
        return None
    return math.log(or_val) * math.sqrt(3) / math.pi


def r_to_d(r: float) -> Optional[float]:
    """Convert Pearson correlation r to Cohen's d.

    d = 2r / sqrt(1 - r^2)

    Returns None if |r| ≥ 1 (undefined).
    """
    if r is None or abs(r) >= 1.0:
        return None
    return 2 * r / math.sqrt(1 - r * r)


def d_to_r(d: float) -> float:
    """Convert Cohen's d to Pearson r.

    r = d / sqrt(d^2 + 4)
    """
    return d / math.sqrt(d * d + 4)


def calculate_effect(
    study_id: str,
    effect_type: str,
    raw_value: float,
    sample_size: Optional[int] = None,
) -> Optional[EffectSizeImpact]:
    """Calculate standardized effect size from a reported statistic.

    effect_type: "cohens_d", "hedges_g", "odds_ratio", "correlation_r", "beta"
    """
    if raw_value is None:
        return None

    cohens_d = None
    hedges_g_val = None

    if effect_type == "cohens_d":
        cohens_d = raw_value
    elif effect_type == "hedges_g":
        # Hedges' g is already corrected; approximate d by reversing correction
        cohens_d = raw_value  # Close enough for classification
        hedges_g_val = raw_value
    elif effect_type == "odds_ratio":
        cohens_d = odds_ratio_to_d(raw_value)
        if cohens_d is None:
            return None
    elif effect_type == "correlation_r":
        cohens_d = r_to_d(raw_value)
        if cohens_d is None:
            return None
    elif effect_type == "beta":
        # Standardized regression coefficient ~= correlation r for bivariate case
        cohens_d = r_to_d(raw_value) if abs(raw_value) < 1.0 else None
        if cohens_d is None:
            return None
    else:
        return None

    # Apply Hedges' g correction if sample size available
    if hedges_g_val is None and sample_size and sample_size >= 4:
        hedges_g_val = hedges_g_correction(cohens_d, sample_size)

    magnitude = classify_magnitude_d(cohens_d)
    direction = "positive" if cohens_d > 0.01 else ("negative" if cohens_d < -0.01 else "null")

    # Human-readable interpretation
    abs_d = abs(cohens_d)
    if direction == "null":
        interp = f"No meaningful effect detected (d = {cohens_d:.3f})"
    else:
        dir_word = "positive" if direction == "positive" else "negative"
        interp = f"A {magnitude} {dir_word} effect (d = {cohens_d:.3f})"
        if hedges_g_val is not None:
            interp += f", Hedges' g = {hedges_g_val:.3f}"
        if sample_size:
            interp += f", N = {sample_size}"

    return EffectSizeImpact(
        study_id=study_id,
        effect_type=effect_type,
        raw_value=round(raw_value, 4),
        cohens_d=round(cohens_d, 4) if cohens_d is not None else None,
        hedges_g=round(hedges_g_val, 4) if hedges_g_val is not None else None,
        magnitude=magnitude,
        direction=direction,
        interpretation=interp,
        sample_size=sample_size,
    )


def batch_calculate(extractions: list) -> List[EffectSizeImpact]:
    """Calculate effect sizes for all studies that have reported statistics.

    Each extraction should have:
        - effect_size_value (float): The reported effect size
        - effect_size_type (str): "cohens_d", "odds_ratio", "correlation_r", "beta", "hedges_g"
        - sample_size_total (int, optional)
        - pmid or title (str): study identifier
    """
    results = []
    for ex in extractions:
        es_value = getattr(ex, 'effect_size_value', None)
        es_type = getattr(ex, 'effect_size_type', None)
        if es_value is None or es_type is None:
            continue

        study_id = getattr(ex, 'pmid', None) or getattr(ex, 'title', "Unknown")
        sample_size = getattr(ex, 'sample_size_total', None)

        impact = calculate_effect(
            study_id=study_id,
            effect_type=es_type,
            raw_value=es_value,
            sample_size=sample_size,
        )
        if impact:
            results.append(impact)

    return results


def format_effect_size_report(impacts: List[EffectSizeImpact]) -> str:
    """Format a deterministic effect size report."""
    if not impacts:
        return "No studies provided effect sizes. Effect size calculation not possible.\n"

    lines = [
        "## Deterministic Effect Size Calculations\n",
        "| Study | Type | Raw Value | Cohen's d | Hedges' g | Magnitude | Direction |",
        "|-------|------|-----------|-----------|-----------|-----------|-----------|",
    ]
    for i in impacts:
        d_str = f"{i.cohens_d:.3f}" if i.cohens_d is not None else "N/A"
        g_str = f"{i.hedges_g:.3f}" if i.hedges_g is not None else "N/A"
        lines.append(
            f"| {i.study_id} | {i.effect_type} | {i.raw_value:.4f} | "
            f"{d_str} | {g_str} | {i.magnitude} | {i.direction} |"
        )
    lines.append("")
    for i in impacts:
        lines.append(f"- **{i.study_id}**: {i.interpretation}")
    return "\n".join(lines)
