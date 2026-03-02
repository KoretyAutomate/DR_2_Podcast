"""Type definitions for the DR_2_Podcast pipeline.

Provides TypedDict interfaces for the return values of run_deep_research()
and related functions, enabling static type checking and documenting the
data contracts between pipeline phases.
"""

from dataclasses import dataclass, field, fields
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


@dataclass
class StudyMetadata:
    """Structured metadata extracted from a scientific source."""
    study_type: Optional[str] = None       # RCT, meta-analysis, cohort, observational, etc.
    sample_size: Optional[str] = None      # "n=1234" or None
    key_result: Optional[str] = None       # Main quantitative finding
    publication_year: Optional[int] = None
    journal_name: Optional[str] = None
    authors: Optional[str] = None          # "First Author et al."
    effect_size: Optional[str] = None      # "HR 0.82", "OR 1.5", "d=0.3"
    limitations: Optional[str] = None      # Author-stated limitations
    demographics: Optional[str] = None     # "age 25-45, 60% female, healthy adults"
    funding_source: Optional[str] = None   # "Industry-funded", "NIH grant", "Independent", etc.
    research_tier: Optional[int] = None    # 1=folk 2=synonym 3=compound

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "StudyMetadata":
        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})

@dataclass
class SummarizedSource:
    url: str
    title: str
    summary: str
    query: str
    goal: str
    error: Optional[str] = None
    metadata: Optional[StudyMetadata] = None

@dataclass
class SearchMetrics:
    """PRISMA-style search flow metrics for auto-generated methodology sections."""
    search_date: str                    # ISO date
    databases_searched: List[str]       # ["PubMed", "Google Scholar", "Google", "Bing", "Brave"]
    total_identified: int               # raw results before dedup
    total_after_dedup: int              # after dedup
    total_fetched: int                  # pages fetched
    total_fetch_errors: int             # fetch failures
    total_with_content: int             # pages with extractable content
    total_summarized: int               # successfully summarized
    academic_sources: int               # pubmed + scholar count
    general_web_sources: int            # general web count
    tier1_sufficient_count: int = 0     # queries where Tier 1 was sufficient
    tier3_expanded_count: int = 0       # queries that needed Tier 3
    wide_net_total: int = 0             # total records from wide net search (Step 2)
    screened_in: int = 0                # records selected after screening (Step 3)
    fulltext_retrieved: int = 0         # full-text articles successfully retrieved (Step 4)
    fulltext_errors: int = 0            # full-text retrieval failures

@dataclass
class ResearchReport:
    topic: str
    role: str
    sources: List[SummarizedSource]
    report: str
    iterations_used: int
    total_urls_fetched: int
    total_summaries: int
    total_errors: int
    duration_seconds: float
    search_metrics: Optional[SearchMetrics] = None


class DeepResearchResult(TypedDict):
    """Return type of run_deep_research() in both clinical and social science pipelines.

    Keys:
        lead:          ResearchReport for the affirmative case
        counter:       ResearchReport for the falsification case
        audit:         ResearchReport for the GRADE synthesis / audit
        pipeline_data: Raw pipeline data for SOT assembly
    """
    lead: ResearchReport
    counter: ResearchReport
    audit: ResearchReport
    pipeline_data: PipelineData
