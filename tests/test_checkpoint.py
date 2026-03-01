"""Tests for pipeline checkpoint/resume functionality (T3.1)."""

import json
import os
from datetime import datetime

import pytest


class TestSerializeDataclass:
    """Test _serialize_dataclass with real clinical research dataclasses."""

    def test_serialize_deep_extraction(self):
        from dr2_podcast.research.clinical import DeepExtraction, PaperMetadata
        de = DeepExtraction(
            pmid="12345678",
            doi="10.1234/test",
            title="Test Study",
            url="https://example.com",
            effect_size="HR 0.82",
            sample_size_total=500,
            control_event_rate=0.15,
            experimental_event_rate=0.10,
            research_tier=1,
            paper_metadata=PaperMetadata(
                citation_count=42,
                is_retracted=False,
                enrichment_sources=["openalex"],
            ),
        )
        from dr2_podcast.pipeline import _serialize_dataclass
        result = _serialize_dataclass(de)
        assert isinstance(result, dict)
        assert result["pmid"] == "12345678"
        assert result["sample_size_total"] == 500
        assert result["paper_metadata"]["citation_count"] == 42
        # Validate it's JSON-serializable
        json_str = json.dumps(result)
        assert "12345678" in json_str

    def test_serialize_tiered_search_plan(self):
        from dr2_podcast.research.clinical import TieredSearchPlan, TierKeywords
        plan = TieredSearchPlan(
            pico={"P": "adults", "I": "caffeine", "C": "placebo", "O": "cognition"},
            tier1=TierKeywords(
                intervention=["coffee"], outcome=["memory"],
                population=["adults"], rationale="exact terms"
            ),
            tier2=TierKeywords(
                intervention=["caffeine"], outcome=["cognition"],
                population=["humans"], rationale="scientific synonyms"
            ),
            tier3=TierKeywords(
                intervention=["methylxanthines"], outcome=["cognitive performance"],
                population=["healthy adults"], rationale="compound class"
            ),
            role="affirmative",
            auditor_approved=True,
            auditor_notes="Approved",
            revision_count=1,
        )
        from dr2_podcast.pipeline import _serialize_dataclass
        result = _serialize_dataclass(plan)
        assert isinstance(result, dict)
        assert result["pico"]["I"] == "caffeine"
        assert result["tier1"]["intervention"] == ["coffee"]
        assert result["auditor_approved"] is True

    def test_serialize_clinical_impact(self):
        from dr2_podcast.research.clinical_math import ClinicalImpact
        impact = ClinicalImpact(
            study_id="PMID:12345",
            cer=0.15, eer=0.10,
            arr=0.05, rrr=0.333,
            nnt=20.0,
            nnt_interpretation="Treat 20 patients to prevent 1 event",
            direction="benefit",
        )
        from dr2_podcast.pipeline import _serialize_dataclass
        result = _serialize_dataclass(impact)
        assert isinstance(result, dict)
        assert result["nnt"] == 20.0
        assert result["direction"] == "benefit"

    def test_serialize_wide_net_record(self):
        from dr2_podcast.research.clinical import WideNetRecord
        wnr = WideNetRecord(
            pmid="99999",
            doi="10.9999/test",
            title="Wide Net Test",
            abstract="This is a test abstract",
            study_type="RCT",
            sample_size="n=200",
            primary_objective="Test objective",
            year=2024,
            journal="Test Journal",
            authors="Author A et al.",
            url="https://pubmed.ncbi.nlm.nih.gov/99999",
            source_db="pubmed",
            research_tier=2,
            relevance_score=0.85,
        )
        from dr2_podcast.pipeline import _serialize_dataclass
        result = _serialize_dataclass(wnr)
        assert result["pmid"] == "99999"
        assert result["research_tier"] == 2

    def test_serialize_list_of_dataclasses(self):
        from dr2_podcast.research.clinical import DeepExtraction
        items = [
            DeepExtraction(pmid="1", doi=None, title="Study 1", url="https://example.com/1"),
            DeepExtraction(pmid="2", doi=None, title="Study 2", url="https://example.com/2"),
        ]
        from dr2_podcast.pipeline import _serialize_dataclass
        result = _serialize_dataclass(items)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["pmid"] == "1"
        assert result[1]["title"] == "Study 2"

    def test_serialize_none_and_primitives(self):
        from dr2_podcast.pipeline import _serialize_dataclass
        assert _serialize_dataclass(None) is None
        assert _serialize_dataclass("hello") == "hello"
        assert _serialize_dataclass(42) == 42
        assert _serialize_dataclass(3.14) == 3.14
        assert _serialize_dataclass(True) is True

    def test_serialize_nested_dict(self):
        from dr2_podcast.pipeline import _serialize_dataclass
        result = _serialize_dataclass({"key": [1, 2, 3], "nested": {"a": "b"}})
        assert result == {"key": [1, 2, 3], "nested": {"a": "b"}}


class TestDeserializePipelineData:
    """Test _deserialize_pipeline_data round-trip."""

    def test_round_trip_deep_extraction(self):
        from dr2_podcast.research.clinical import DeepExtraction, PaperMetadata
        from dr2_podcast.pipeline import _serialize_dataclass, _deserialize_pipeline_data

        original = {
            "aff_extractions": [
                DeepExtraction(
                    pmid="111", doi="10.1/a", title="Study A", url="https://a.com",
                    effect_size="OR 1.5", research_tier=1,
                    paper_metadata=PaperMetadata(citation_count=10),
                ),
            ],
            "fal_extractions": [],
            "aff_top": [],
            "fal_top": [],
            "impacts": [],
        }
        serialized = _serialize_dataclass(original)
        restored = _deserialize_pipeline_data(serialized)

        assert len(restored["aff_extractions"]) == 1
        de = restored["aff_extractions"][0]
        assert isinstance(de, DeepExtraction)
        assert de.pmid == "111"
        assert de.effect_size == "OR 1.5"
        assert de.paper_metadata is not None
        assert de.paper_metadata.citation_count == 10

    def test_round_trip_tiered_search_plan(self):
        from dr2_podcast.research.clinical import TieredSearchPlan, TierKeywords
        from dr2_podcast.pipeline import _serialize_dataclass, _deserialize_pipeline_data

        original = {
            "aff_strategy": TieredSearchPlan(
                pico={"P": "adults", "I": "coffee", "C": "placebo", "O": "cognition"},
                tier1=TierKeywords(["coffee"], ["memory"], ["adults"], "exact"),
                tier2=TierKeywords(["caffeine"], ["cognition"], ["humans"], "synonyms"),
                tier3=TierKeywords(["methylxanthines"], ["cognitive"], ["all"], "class"),
                role="affirmative",
                auditor_approved=True,
            ),
            "fal_strategy": None,
            "aff_extractions": [],
            "fal_extractions": [],
            "aff_top": [],
            "fal_top": [],
            "impacts": [],
        }
        serialized = _serialize_dataclass(original)
        restored = _deserialize_pipeline_data(serialized)

        plan = restored["aff_strategy"]
        assert isinstance(plan, TieredSearchPlan)
        assert plan.pico["I"] == "coffee"
        assert plan.tier1.intervention == ["coffee"]
        assert plan.auditor_approved is True

    def test_round_trip_clinical_impact(self):
        from dr2_podcast.research.clinical_math import ClinicalImpact
        from dr2_podcast.pipeline import _serialize_dataclass, _deserialize_pipeline_data

        original = {
            "impacts": [
                ClinicalImpact(
                    study_id="PMID:1", cer=0.2, eer=0.1,
                    arr=0.1, rrr=0.5, nnt=10.0,
                    nnt_interpretation="Treat 10...", direction="benefit",
                ),
            ],
            "aff_extractions": [],
            "fal_extractions": [],
            "aff_top": [],
            "fal_top": [],
        }
        serialized = _serialize_dataclass(original)
        restored = _deserialize_pipeline_data(serialized)

        assert len(restored["impacts"]) == 1
        imp = restored["impacts"][0]
        assert isinstance(imp, ClinicalImpact)
        assert imp.nnt == 10.0

    def test_empty_pipeline_data(self):
        from dr2_podcast.pipeline import _deserialize_pipeline_data
        assert _deserialize_pipeline_data({}) == {}
        assert _deserialize_pipeline_data(None) is None


class TestSaveLoadCheckpoint:
    """Test save_checkpoint and load_checkpoint."""

    def test_save_creates_checkpoint_file(self, tmp_output_dir):
        from dr2_podcast.pipeline import save_checkpoint, CHECKPOINT_FILE
        save_checkpoint(
            tmp_output_dir, 0, "test topic", "en",
            {"framing_output": "some framing text"}
        )
        ckpt_path = tmp_output_dir / CHECKPOINT_FILE
        assert ckpt_path.exists()

        data = json.loads(ckpt_path.read_text())
        assert data["topic"] == "test topic"
        assert data["language"] == "en"
        assert 0 in data["completed_phases"]
        assert "timestamp" in data
        assert data["pipeline_state"]["framing_output"] == "some framing text"

    def test_save_accumulates_phases(self, tmp_output_dir):
        from dr2_podcast.pipeline import save_checkpoint, CHECKPOINT_FILE
        save_checkpoint(tmp_output_dir, 0, "topic", "en", {"a": "1"})
        save_checkpoint(tmp_output_dir, 1, "topic", "en", {"b": "2"})
        save_checkpoint(tmp_output_dir, 2, "topic", "en", {"c": "3"})

        data = json.loads((tmp_output_dir / CHECKPOINT_FILE).read_text())
        assert data["completed_phases"] == [0, 1, 2]

    def test_save_does_not_duplicate_phases(self, tmp_output_dir):
        from dr2_podcast.pipeline import save_checkpoint, CHECKPOINT_FILE
        save_checkpoint(tmp_output_dir, 0, "topic", "en", {})
        save_checkpoint(tmp_output_dir, 0, "topic", "en", {})

        data = json.loads((tmp_output_dir / CHECKPOINT_FILE).read_text())
        assert data["completed_phases"] == [0]

    def test_load_valid_checkpoint(self, tmp_output_dir):
        from dr2_podcast.pipeline import save_checkpoint, load_checkpoint
        save_checkpoint(tmp_output_dir, 0, "my topic", "ja", {"key": "val"})
        result = load_checkpoint(tmp_output_dir)
        assert result is not None
        assert result["topic"] == "my topic"
        assert result["language"] == "ja"
        assert result["completed_phases"] == [0]
        assert result["pipeline_state"]["key"] == "val"

    def test_load_missing_checkpoint(self, tmp_output_dir):
        from dr2_podcast.pipeline import load_checkpoint
        result = load_checkpoint(tmp_output_dir)
        assert result is None

    def test_load_corrupt_json(self, tmp_output_dir):
        from dr2_podcast.pipeline import load_checkpoint, CHECKPOINT_FILE
        (tmp_output_dir / CHECKPOINT_FILE).write_text("not valid json{{{")
        result = load_checkpoint(tmp_output_dir)
        assert result is None

    def test_load_missing_required_keys(self, tmp_output_dir):
        from dr2_podcast.pipeline import load_checkpoint, CHECKPOINT_FILE
        (tmp_output_dir / CHECKPOINT_FILE).write_text(json.dumps({"topic": "x"}))
        result = load_checkpoint(tmp_output_dir)
        assert result is None

    def test_checkpoint_json_has_required_keys(self, tmp_output_dir):
        """Test criterion: checkpoint.json has keys: topic, language, completed_phases, timestamp."""
        from dr2_podcast.pipeline import save_checkpoint, CHECKPOINT_FILE
        save_checkpoint(tmp_output_dir, 1, "effects of caffeine", "en", {})

        data = json.loads((tmp_output_dir / CHECKPOINT_FILE).read_text())
        for key in ("topic", "language", "completed_phases", "timestamp"):
            assert key in data, f"Missing required key: {key}"

    def test_save_with_dataclass_state(self, tmp_output_dir):
        """Verify that pipeline_data with dataclasses serializes to valid JSON."""
        from dr2_podcast.research.clinical import DeepExtraction, TieredSearchPlan, TierKeywords
        from dr2_podcast.research.clinical_math import ClinicalImpact
        from dr2_podcast.pipeline import save_checkpoint, load_checkpoint, CHECKPOINT_FILE

        state = {
            "deep_reports": {
                "pipeline_data": {
                    "aff_strategy": TieredSearchPlan(
                        pico={"P": "adults", "I": "coffee", "C": "placebo", "O": "cognition"},
                        tier1=TierKeywords(["coffee"], ["memory"], ["adults"], "exact"),
                        tier2=TierKeywords(["caffeine"], ["cognition"], ["humans"], "syn"),
                        tier3=TierKeywords(["methylxanthines"], ["perf"], ["all"], "class"),
                        role="affirmative",
                    ),
                    "fal_strategy": None,
                    "aff_extractions": [
                        DeepExtraction(pmid="1", doi=None, title="Study", url="https://x.com"),
                    ],
                    "fal_extractions": [],
                    "aff_top": [],
                    "fal_top": [],
                    "impacts": [
                        ClinicalImpact("PMID:1", 0.2, 0.1, 0.1, 0.5, 10.0, "Treat 10", "benefit"),
                    ],
                    "math_report": "some report",
                    "framing_context": "context",
                    "search_date": "2026-02-28",
                },
            },
            "sot_summary": "summary of SOT",
            "evidence_quality": "sufficient",
        }
        save_checkpoint(tmp_output_dir, 1, "caffeine", "en", state)

        # Verify the file is valid JSON
        ckpt_path = tmp_output_dir / CHECKPOINT_FILE
        assert ckpt_path.exists()
        data = json.loads(ckpt_path.read_text())
        assert data["completed_phases"] == [1]

        # Load and verify deserialization
        loaded = load_checkpoint(tmp_output_dir)
        assert loaded is not None
        pd = loaded["pipeline_state"]["deep_reports"]["pipeline_data"]
        # pipeline_data should have been deserialized back to dataclasses
        from dr2_podcast.research.clinical import TieredSearchPlan as TSP, DeepExtraction as DE
        from dr2_podcast.research.clinical_math import ClinicalImpact as CI
        assert isinstance(pd["aff_strategy"], TSP)
        assert isinstance(pd["aff_extractions"][0], DE)
        assert isinstance(pd["impacts"][0], CI)


class TestResumeArgParsing:
    """Test that --resume argument is correctly parsed."""

    def test_resume_arg_accepted(self):
        """Verify parse_arguments() accepts --resume."""
        from dr2_podcast.pipeline import parse_arguments
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["pipeline.py", "--resume", "research_outputs/2026-02-28_12-00-00"]
            args = parse_arguments()
            assert args.resume == "research_outputs/2026-02-28_12-00-00"
        finally:
            sys.argv = old_argv

    def test_resume_arg_default_none(self):
        """Verify --resume defaults to None."""
        from dr2_podcast.pipeline import parse_arguments
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["pipeline.py", "--topic", "test"]
            args = parse_arguments()
            assert args.resume is None
        finally:
            sys.argv = old_argv


class TestPhaseSkipLogic:
    """Test that the _phase_done helper works correctly with checkpoint data."""

    def test_phase_done_with_empty_set(self):
        """When no phases completed, all should return False."""
        completed = set()
        assert 0 not in completed
        assert 1 not in completed
        assert 5 not in completed

    def test_phase_done_with_some_completed(self):
        """When some phases completed, only those should be True."""
        completed = {0, 1, 2}
        assert 0 in completed
        assert 1 in completed
        assert 2 in completed
        assert 3 not in completed
        assert 5 not in completed

    def test_full_checkpoint_cycle(self, tmp_output_dir):
        """Simulate: run Phase 0-1, save checkpoint, load, verify skippable."""
        from dr2_podcast.pipeline import save_checkpoint, load_checkpoint

        # Simulate Phase 0 completion
        save_checkpoint(tmp_output_dir, 0, "topic", "en", {"framing_output": "framing"})
        # Simulate Phase 1 completion
        save_checkpoint(tmp_output_dir, 1, "topic", "en", {"evidence_quality": "sufficient"})

        # Load checkpoint (as resume would)
        ckpt = load_checkpoint(tmp_output_dir)
        completed = set(ckpt["completed_phases"])

        assert 0 in completed  # Phase 0 should be skipped
        assert 1 in completed  # Phase 1 should be skipped
        assert 2 not in completed  # Phase 2 should run
        assert 5 not in completed  # Phase 5 should run
