"""
shared/models.py — Pydantic models and dataclasses for inter-flow communication.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal

from pydantic import BaseModel, HttpUrl, Field


# ---------------------------------------------------------------------------
# Source tracking (used by validation flow)
# ---------------------------------------------------------------------------
class ScientificSource(BaseModel):
    """Structured scientific source."""
    title: str
    url: HttpUrl
    journal: Optional[str] = None
    publication_year: Optional[int] = None
    source_type: Literal["peer_reviewed", "preprint", "review", "meta_analysis", "web_article"]
    trust_level: Literal["high", "medium", "low"] = "medium"
    cited_by: str
    key_finding: Optional[str] = None


class SourceBibliography(BaseModel):
    """Complete bibliography with categorization."""
    supporting_sources: List[ScientificSource] = []
    contradicting_sources: List[ScientificSource] = []

    def get_high_trust_sources(self) -> List[ScientificSource]:
        all_sources = self.supporting_sources + self.contradicting_sources
        return [s for s in all_sources if s.trust_level == "high" and s.source_type == "peer_reviewed"]


# ---------------------------------------------------------------------------
# Pipeline parameter / result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class PipelineParams:
    """Parameters gathered from Web UI or CLI."""
    topic: str
    language: str = "en"
    host_order: str = "random"
    accessibility_level: str = "simple"
    reuse_dir: Optional[str] = None
    crew3_only: bool = False
    check_supplemental: bool = False
    upload_buzzsprout: bool = False
    upload_youtube: bool = False
    podcast_length: str = "long"


@dataclass
class PipelineApproach:
    """Decisions about which sub-flows to run."""
    run_research: bool = True
    run_validation: bool = True
    run_translation: bool = False
    run_podcast_planning: bool = True
    run_audio: bool = True
    run_upload: bool = False
    reuse_dir: Optional[Path] = None
    supplemental_needed: bool = False
    output_dir: Path = field(default_factory=Path)


@dataclass
class EvidenceGatheringResult:
    """Output of f03_evidence_gathering."""
    framing_output: str = ""
    deep_reports: Optional[dict] = None
    deep_sources_json: Optional[dict] = None
    supporting_research: str = ""
    gap_analysis: str = ""
    gap_fill_output: str = ""
    gate_passed: bool = True
    output_dir: Path = field(default_factory=Path)


@dataclass
class ValidationResult:
    """Output of f04_evidence_validation."""
    adversarial_research: str = ""
    url_validation: Optional[dict] = None
    source_of_truth: str = ""
    source_verification: str = ""
    output_dir: Path = field(default_factory=Path)


@dataclass
class TranslationResult:
    """Output of f05_translation."""
    translated_source_of_truth: str = ""
    translated_supporting: str = ""
    output_dir: Path = field(default_factory=Path)


@dataclass
class PodcastPlanningResult:
    """Output of f06_podcast_planning."""
    show_notes: str = ""
    script_raw: str = ""
    script_polished: str = ""
    accuracy_check: str = ""
    output_dir: Path = field(default_factory=Path)


@dataclass
class AudioResult:
    """Output of f07_audio_generation."""
    audio_path: Optional[Path] = None
    duration_minutes: Optional[float] = None
    script_length: int = 0
    output_dir: Path = field(default_factory=Path)


@dataclass
class UploadResult:
    """Output of f08_upload."""
    buzzsprout: Optional[dict] = None
    youtube: Optional[dict] = None
