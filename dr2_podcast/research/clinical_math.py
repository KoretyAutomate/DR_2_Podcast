"""
Deterministic clinical statistics calculator.
No LLM involvement — pure arithmetic to prevent hallucinated math.

Calculates ARR (Absolute Risk Reduction), RRR (Relative Risk Reduction),
and NNT (Number Needed to Treat) from CER/EER values extracted by the
deep research pipeline.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dr2_podcast.research.clinical import DeepExtraction, Finding


@dataclass
class ClinicalImpact:
    study_id: str  # PMID or title
    cer: float  # Control Event Rate
    eer: float  # Experimental Event Rate
    arr: float  # Absolute Risk Reduction = CER - EER
    rrr: float | None  # Relative Risk Reduction = ARR / CER; None when CER is 0 and it is undefined
    nnt: float  # Number Needed to Treat = 1 / |ARR|
    nnt_interpretation: str  # "Treat 10 patients to prevent 1 event"
    direction: str  # "benefit" | "harm" | "no_effect"
    # Step 9a slice 2. A paper is not a finding: one study reporting benefit on one endpoint and no
    # effect on another produced two rows keyed by the same study_id, which rendered as ambiguous
    # duplicates and gave replication grouping nothing to group by. Optional and last so the
    # positional construction in the checkpoint tests keeps working.
    finding_key: str | None = None
    endpoint: str | None = None
    timepoint: str | None = None

    @property
    def row_label(self) -> str:
        """How this row names itself: the study, and which of its findings this is."""
        if not self.endpoint:
            return self.study_id
        at = f" @ {self.timepoint}" if self.timepoint else ""
        return f"{self.study_id} — {self.endpoint}{at}"


def calculate_impact(
    study_id: str,
    # Optional in the annotation because it always was in fact: the first thing the body does is
    # return None when either rate is missing, and both the legacy fallback and a finding without
    # event rates reach it that way.
    cer: float | None,
    eer: float | None,
    outcome_is_adverse: bool | None = None,
    finding: "Finding | None" = None,
) -> ClinicalImpact | None:
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
            study_id=study_id,
            cer=cer,
            eer=eer,
            arr=0.0,
            rrr=0.0 if abs(cer) > 1e-10 else None,
            nnt=float("inf"),
            nnt_interpretation="No measurable difference between groups",
            direction="no_effect",
            finding_key=finding.finding_key if finding else None,
            endpoint=finding.endpoint if finding else None,
            timepoint=finding.timepoint if finding else None,
        )

    # RRR over a zero control-event rate is UNDEFINED, not zero. Reporting 0.0 there states
    # "no relative reduction", which is a quantitative claim the data does not support — a
    # zero-event control arm is a known and real situation, not a rounding artefact. None is the
    # honest value, and dr2_podcast/schemas requires it of any derived record built from this.
    rrr = arr / cer if abs(cer) > 1e-10 else None
    nnt = 1.0 / abs(arr)
    direction = "benefit" if arr > 0 else "harm"
    verb = "prevent" if direction == "benefit" else "cause"
    interp = f"Treat {nnt:.0f} patients to {verb} 1 additional event"

    return ClinicalImpact(
        study_id=study_id,
        cer=cer,
        eer=eer,
        arr=round(arr, 6),
        rrr=round(rrr, 4) if rrr is not None else None,
        nnt=round(nnt, 1),
        nnt_interpretation=interp,
        direction=direction,
        finding_key=finding.finding_key if finding else None,
        endpoint=finding.endpoint if finding else None,
        timepoint=finding.timepoint if finding else None,
    )


def batch_calculate(extractions: list["DeepExtraction"]) -> list[ClinicalImpact]:
    """Calculate clinical impact for every FINDING that has CER and EER.

    Per finding, not per paper. A study reporting a benefit on its primary endpoint and a null
    result on a secondary one contributes two rows, each with its own polarity — and
    ``outcome_is_adverse`` is per-finding precisely because applying one paper's polarity to both
    flips the ARR interpretation on one of them and produces a directionally wrong NNT.
    """
    results = []
    for ex in extractions:
        study_id = ex.pmid or ex.title
        findings = getattr(ex, "findings", None) or []
        if not findings:
            # A checkpoint written before findings[] existed carries its rates at the paper level,
            # and dropping it would replace previously computed ARR/NNT with "Data Insufficient" on
            # a resumed run (prepush codex 2026-08-13). This does NOT reopen the door slice 1
            # closed: an extraction produced since then has no verified finding only when its rates
            # are None, so it still contributes nothing here.
            legacy = calculate_impact(
                study_id=study_id,
                cer=getattr(ex, "control_event_rate", None),
                eer=getattr(ex, "experimental_event_rate", None),
                outcome_is_adverse=getattr(ex, "outcome_is_adverse", None),
            )
            if legacy:
                results.append(legacy)
            continue
        for finding in findings:
            if finding.control_event_rate is None or finding.experimental_event_rate is None:
                continue
            impact = calculate_impact(
                study_id=study_id,
                cer=finding.control_event_rate,
                eer=finding.experimental_event_rate,
                outcome_is_adverse=finding.outcome_is_adverse,
                finding=finding,
            )
            if impact:
                results.append(impact)
    return results


def format_rrr(rrr: float | None) -> str:
    """RRR is undefined when the control-event rate is zero; say so rather than printing 0%."""
    return f"{rrr:+.2%}" if rrr is not None else "n/a (CER=0)"


def format_math_report(impacts: list[ClinicalImpact]) -> str:
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
            f"| {i.row_label} | {i.cer:.3f} | {i.eer:.3f} | {i.arr:+.4f} | "
            f"{format_rrr(i.rrr)} | {i.nnt:.1f} | {i.direction} |"
        )
    lines.append("")
    for i in impacts:
        lines.append(f"- **{i.row_label}**: {i.nnt_interpretation}")
    return "\n".join(lines)
