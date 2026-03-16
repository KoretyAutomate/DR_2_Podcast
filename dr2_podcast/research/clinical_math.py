"""
Deterministic clinical statistics calculator.
No LLM involvement — pure arithmetic to prevent hallucinated math.

Calculates ARR (Absolute Risk Reduction), RRR (Relative Risk Reduction),
and NNT (Number Needed to Treat) from CER/EER values extracted by the
deep research pipeline.
"""

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dr2_podcast.research.clinical import DeepExtraction


@dataclass
class ClinicalImpact:
    study_id: str           # PMID or title
    cer: float              # Control Event Rate
    eer: float              # Experimental Event Rate
    arr: float              # Absolute Risk Reduction = CER - EER
    rrr: float              # Relative Risk Reduction = ARR / CER
    nnt: float              # Number Needed to Treat = 1 / |ARR|
    nnt_interpretation: str # "Treat 10 patients to prevent 1 event"
    direction: str          # "benefit" | "harm" | "no_effect"


def calculate_impact(study_id: str, cer: float, eer: float,
                     outcome_is_adverse: Optional[bool] = None) -> Optional[ClinicalImpact]:
    """
    Calculate ARR, RRR, NNT from CER and EER.

    ARR = CER - EER        (positive = benefit, negative = harm)
    RRR = ARR / CER        (relative measure)
    NNT = 1 / |ARR|        (patients needed to treat for one outcome)

    If outcome_is_adverse is False, the "event" is beneficial (e.g. weight loss),
    so the direction interpretation is flipped: EER > CER means benefit.
    """
    if cer is None or eer is None:
        return None

    arr = cer - eer
    # If the event is beneficial (not adverse), flip ARR so that
    # EER > CER (more beneficial events in experimental) = positive ARR = benefit
    if outcome_is_adverse is False:
        arr = -arr
    if abs(arr) < 1e-10:
        return ClinicalImpact(
            study_id=study_id, cer=cer, eer=eer,
            arr=0.0, rrr=0.0, nnt=float('inf'),
            nnt_interpretation="No measurable difference between groups",
            direction="no_effect"
        )

    rrr = arr / cer if abs(cer) > 1e-10 else 0.0
    nnt = 1.0 / abs(arr)
    direction = "benefit" if arr > 0 else "harm"
    verb = "prevent" if direction == "benefit" else "cause"
    interp = f"Treat {nnt:.0f} patients to {verb} 1 additional event"

    return ClinicalImpact(
        study_id=study_id, cer=cer, eer=eer,
        arr=round(arr, 6), rrr=round(rrr, 4), nnt=round(nnt, 1),
        nnt_interpretation=interp, direction=direction
    )


def batch_calculate(extractions: List["DeepExtraction"]) -> List[ClinicalImpact]:
    """Calculate clinical impact for all studies that have CER and EER."""
    results = []
    for ex in extractions:
        if ex.control_event_rate is not None and ex.experimental_event_rate is not None:
            impact = calculate_impact(
                study_id=ex.pmid or ex.title,
                cer=ex.control_event_rate,
                eer=ex.experimental_event_rate,
                outcome_is_adverse=getattr(ex, "outcome_is_adverse", None),
            )
            if impact:
                results.append(impact)
    return results


def format_math_report(impacts: List[ClinicalImpact]) -> str:
    """Format a deterministic math report for the Auditor."""
    if not impacts:
        return (
            "## Deterministic Clinical Impact Calculations\n\n"
            "**Status:** Data Insufficient\n\n"
            "NNT (Number Needed to Treat) calculation requires both Control Event Rate (CER) "
            "and Experimental Event Rate (EER) from extracted studies. "
            "The screening and extraction phases did not identify studies with both metrics available.\n\n"
            "**Why NNT is not calculated:**\n"
            "- Many studies report only one outcome (e.g., EER without CER)\n"
            "- Some studies report qualitative outcomes or continuous measures (not binary event rates)\n"
            "- Extraction may have incomplete data from full-text access constraints\n\n"
            "**Impact:** Clinical effect sizes cannot be quantified numerically. "
            "Evidence quality assessment and narrative synthesis remain unaffected.\n"
        )

    lines = [
        "## Deterministic Clinical Impact Calculations\n",
        "| Study | CER | EER | ARR | RRR | NNT | Direction |",
        "|-------|-----|-----|-----|-----|-----|-----------|",
    ]
    for i in impacts:
        lines.append(
            f"| {i.study_id} | {i.cer:.3f} | {i.eer:.3f} | "
            f"{i.arr:+.4f} | {i.rrr:+.2%} | {i.nnt:.1f} | {i.direction} |"
        )
    lines.append("")
    for i in impacts:
        lines.append(f"- **{i.study_id}**: {i.nnt_interpretation}")
    return "\n".join(lines)
