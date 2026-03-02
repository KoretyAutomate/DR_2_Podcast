"""Type definitions for the DR_2_Podcast pipeline.

Provides TypedDict interfaces for the return values of run_deep_research()
and related functions, enabling static type checking and documenting the
data contracts between pipeline phases.
"""

from typing import TypedDict, List, Optional, Dict, Any


class PipelineMetrics(TypedDict, total=False):
    """PRISMA-style search flow metrics for both research tracks."""
    aff_wide_net_total: int
    aff_screened_in: int
    aff_fulltext_ok: int
    aff_fulltext_err: int
    fal_wide_net_total: int
    fal_screened_in: int
    fal_fulltext_ok: int
    fal_fulltext_err: int


class PipelineData(TypedDict, total=False):
    """Raw data from the clinical/social-science research pipeline.

    Passed to build_imrad_sot() for Source-of-Truth assembly.
    """
    domain: str                          # "clinical" or "social_science"
    aff_strategy: Any                    # TieredSearchPlan (affirmative)
    fal_strategy: Any                    # TieredSearchPlan (falsification)
    aff_extractions: list                # List[DeepExtraction]
    fal_extractions: list                # List[DeepExtraction]
    aff_top: list                        # List[WideNetRecord] — top screened papers
    fal_top: list                        # List[WideNetRecord]
    math_report: str                     # Deterministic ARR/NNT report
    impacts: list                        # List[ClinicalImpact] or List[EffectSizeImpact]
    framing_context: str                 # Research framing from Phase 0
    search_date: str                     # ISO date of search
    aff_highest_tier: int                # 1-3: highest tier reached
    fal_highest_tier: int
    metrics: PipelineMetrics


class DeepResearchResult(TypedDict):
    """Return type of run_deep_research() in both clinical and social science pipelines.

    Keys:
        lead:          ResearchReport for the affirmative case
        counter:       ResearchReport for the falsification case
        audit:         ResearchReport for the GRADE synthesis / audit
        pipeline_data: Raw pipeline data for SOT assembly
    """
    lead: Any              # ResearchReport dataclass
    counter: Any           # ResearchReport dataclass
    audit: Any             # ResearchReport dataclass
    pipeline_data: PipelineData
