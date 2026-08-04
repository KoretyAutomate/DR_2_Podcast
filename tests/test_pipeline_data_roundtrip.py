"""Round-trip tests for pipeline_data serialization.

_deserialize_pipeline_data rebuilds the research dataclasses that
_serialize_dataclass flattened, so --resume can restore a run's state. The real
invariant is the round trip: serialize → deserialize → same objects.

Written before splitting that complexity-23 function into module-level
restorers. tests/test_checkpoint.py covered _serialize_dataclass only; nothing
covered the way back.
"""

import pytest

from dr2_podcast.pipeline import _deserialize_pipeline_data, _serialize_dataclass
from dr2_podcast.research.clinical import (
    DeepExtraction,
    PaperMetadata,
    TieredSearchPlan,
    TierKeywords,
    WideNetRecord,
)
from dr2_podcast.research.clinical_math import ClinicalImpact


def _tier(n):
    return TierKeywords(intervention=[f"i{n}"], outcome=[f"o{n}"], population=[f"p{n}"], rationale=f"r{n}")


def _plan(role="affirmative"):
    return TieredSearchPlan(
        pico={"population": "adults", "intervention": "drug", "comparison": "placebo", "outcome": "death"},
        tier1=_tier(1),
        tier2=_tier(2),
        tier3=_tier(3),
        role=role,
        auditor_approved=True,
        auditor_notes="notes",
        revision_count=3,
    )


def _extraction():
    return DeepExtraction(
        pmid="123",
        doi="10.1/x",
        title="A study",
        url="https://example.org/x",
        sample_size_total=500,
        study_design="RCT",
        research_tier=2,
        control_event_rate=0.2,
        experimental_event_rate=0.1,
        outcome_is_adverse=True,
        paper_metadata=PaperMetadata(citation_count=10, fwci=1.5, enrichment_sources=["openalex"]),
    )


def _wide():
    return WideNetRecord(
        pmid="456",
        doi="10.2/y",
        title="Wide record",
        abstract="abs",
        study_type="RCT",
        sample_size="100",
        primary_objective="obj",
        year=2024,
        journal="J",
        authors="A",
        url="https://example.org/y",
        source_db="pubmed",
        research_tier=1,
        paper_metadata=PaperMetadata(citation_count=4),
    )


def _impact():
    return ClinicalImpact(
        study_id="123",
        cer=0.2,
        eer=0.1,
        arr=0.1,
        rrr=0.5,
        nnt=10.0,
        nnt_interpretation="Treat 10",
        direction="benefit",
    )


def _full_pipeline_data():
    return {
        "domain": "clinical",
        "aff_strategy": _plan("affirmative"),
        "fal_strategy": _plan("adversarial"),
        "aff_extractions": [_extraction()],
        "fal_extractions": [_extraction()],
        "aff_top": [_wide()],
        "fal_top": [_wide()],
        "impacts": [_impact()],
        "framing_context": "framing",
        "search_date": "2026-08-03",
        "metrics": {"aff_wide_net_total": 10},
    }


class TestRoundTrip:
    @pytest.fixture
    def restored(self):
        return _deserialize_pipeline_data(_serialize_dataclass(_full_pipeline_data()))

    def test_strategies_become_tiered_search_plans(self, restored):
        for key in ("aff_strategy", "fal_strategy"):
            plan = restored[key]
            assert isinstance(plan, TieredSearchPlan)
            assert isinstance(plan.tier1, TierKeywords)
            assert plan.tier1.intervention == ["i1"]
            assert plan.tier3.rationale == "r3"
            assert plan.auditor_approved is True
            assert plan.revision_count == 3
            assert plan.pico["outcome"] == "death"

    def test_extractions_become_deep_extractions_with_metadata(self, restored):
        for key in ("aff_extractions", "fal_extractions"):
            [ext] = restored[key]
            assert isinstance(ext, DeepExtraction)
            assert ext.pmid == "123"
            assert ext.sample_size_total == 500
            assert ext.control_event_rate == 0.2
            assert ext.outcome_is_adverse is True
            assert isinstance(ext.paper_metadata, PaperMetadata)
            assert ext.paper_metadata.citation_count == 10

    def test_wide_records_become_wide_net_records(self, restored):
        for key in ("aff_top", "fal_top"):
            [rec] = restored[key]
            assert isinstance(rec, WideNetRecord)
            assert rec.pmid == "456"
            assert rec.year == 2024
            assert rec.source_db == "pubmed"
            assert isinstance(rec.paper_metadata, PaperMetadata)

    def test_impacts_become_clinical_impacts(self, restored):
        [imp] = restored["impacts"]
        assert isinstance(imp, ClinicalImpact)
        assert imp.nnt == 10.0
        assert imp.direction == "benefit"

    def test_plain_values_pass_through_untouched(self, restored):
        assert restored["domain"] == "clinical"
        assert restored["framing_context"] == "framing"
        assert restored["search_date"] == "2026-08-03"
        assert restored["metrics"] == {"aff_wide_net_total": 10}

    def test_the_input_dict_is_not_mutated(self):
        serialized = _serialize_dataclass(_full_pipeline_data())
        before = serialized["aff_strategy"]
        _deserialize_pipeline_data(serialized)
        assert serialized["aff_strategy"] is before, "deserialize must not mutate its input"
        assert isinstance(serialized["aff_strategy"], dict)


class TestEdgeCases:
    def test_empty_input_returns_input(self):
        assert _deserialize_pipeline_data({}) == {}
        assert _deserialize_pipeline_data(None) is None

    def test_missing_keys_are_not_invented(self):
        assert _deserialize_pipeline_data({"domain": "clinical"}) == {"domain": "clinical"}

    def test_non_dict_strategy_passes_through(self):
        out = _deserialize_pipeline_data({"aff_strategy": "not-a-dict"})
        assert out["aff_strategy"] == "not-a-dict"

    def test_none_strategy_passes_through(self):
        assert _deserialize_pipeline_data({"aff_strategy": None})["aff_strategy"] is None

    def test_non_list_extractions_pass_through(self):
        out = _deserialize_pipeline_data({"aff_extractions": "oops"})
        assert out["aff_extractions"] == "oops"

    def test_extraction_missing_required_fields_gets_defaults(self):
        """to_dict() drops None/empty values, so the restorer must supply them."""
        [ext] = _deserialize_pipeline_data({"aff_extractions": [{"study_design": "RCT"}]})["aff_extractions"]
        assert isinstance(ext, DeepExtraction)
        assert ext.pmid is None and ext.doi is None
        assert ext.title == "" and ext.url == ""
        assert ext.study_design == "RCT"

    def test_wide_record_missing_required_fields_gets_defaults(self):
        [rec] = _deserialize_pipeline_data({"aff_top": [{"study_type": "RCT"}]})["aff_top"]
        assert isinstance(rec, WideNetRecord)
        assert rec.pmid is None and rec.year is None
        assert rec.title == "" and rec.source_db == ""

    def test_unknown_keys_in_a_record_are_dropped_not_passed_to_the_constructor(self):
        [rec] = _deserialize_pipeline_data({"aff_top": [{"title": "T", "bogus_field": 1}]})["aff_top"]
        assert rec.title == "T"
        assert not hasattr(rec, "bogus_field")
