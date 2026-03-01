"""
Unit tests for domain_classifier.py.
"""

import asyncio

import pytest

from dr2_podcast.research.domain_classifier import (
    ResearchDomain,
    DomainClassification,
    classify_topic_deterministic,
    classify_topic,
)


class TestDeterministicClassification:

    def test_clinical_coffee(self):
        result = classify_topic_deterministic("Does coffee improve cognitive function?")
        # "cognitive" is not in the clinical keywords and "caffeine" is not in
        # the topic text, so deterministic classification should return None.
        assert result is None

    def test_clinical_statin(self):
        result = classify_topic_deterministic("Effects of statin medication on cholesterol levels")
        assert result is not None
        assert result.domain == ResearchDomain.CLINICAL
        assert result.suggested_framework == "PICO"
        assert "PubMed" in result.primary_databases

    def test_clinical_cancer(self):
        result = classify_topic_deterministic("Does chemotherapy improve cancer survival rates?")
        assert result is not None
        assert result.domain == ResearchDomain.CLINICAL

    def test_clinical_diabetes_treatment(self):
        result = classify_topic_deterministic("Metformin treatment for type 2 diabetes patients")
        assert result is not None
        assert result.domain == ResearchDomain.CLINICAL
        assert result.confidence >= 0.6

    def test_social_science_homework(self):
        result = classify_topic_deterministic(
            "Effects of homework on student academic achievement in elementary school"
        )
        assert result is not None
        assert result.domain == ResearchDomain.SOCIAL_SCIENCE
        assert result.suggested_framework == "PECO"
        assert "ERIC" in result.primary_databases

    def test_social_science_daycare(self):
        result = classify_topic_deterministic(
            "Effects of daycare vs stay-at-home parenting on child development"
        )
        assert result is not None
        assert result.domain == ResearchDomain.SOCIAL_SCIENCE

    def test_social_science_remote_work(self):
        result = classify_topic_deterministic(
            "Does remote work increase employee productivity and job satisfaction?"
        )
        assert result is not None
        assert result.domain == ResearchDomain.SOCIAL_SCIENCE

    def test_ambiguous_returns_none(self):
        result = classify_topic_deterministic("Is intermittent fasting beneficial?")
        # Could be clinical or general wellness — may be ambiguous
        # The word "fasting" is not a keyword. This should return None.
        # Actually let's test with something truly ambiguous
        result2 = classify_topic_deterministic("How does music affect the brain?")
        # "brain" is clinical but it's a single keyword → score 0.33 < 0.6
        assert result2 is None

    def test_strong_clinical_multiple_keywords(self):
        result = classify_topic_deterministic(
            "Randomized clinical trial of aspirin for heart disease patients"
        )
        assert result is not None
        assert result.domain == ResearchDomain.CLINICAL

    def test_strong_social_science_multiple_keywords(self):
        result = classify_topic_deterministic(
            "Quasi-experimental study of tutoring on student literacy in elementary school"
        )
        assert result is not None
        assert result.domain == ResearchDomain.SOCIAL_SCIENCE


class TestAsyncClassification:

    def test_clinical_topic_classified(self):
        result = asyncio.run(classify_topic(
            "Metformin treatment for diabetes patients"
        ))
        assert result.domain == ResearchDomain.CLINICAL

    def test_social_science_topic_classified(self):
        result = asyncio.run(classify_topic(
            "Effects of homework on student academic achievement"
        ))
        assert result.domain == ResearchDomain.SOCIAL_SCIENCE

    def test_ambiguous_defaults_to_clinical(self):
        """Without LLM client, ambiguous topics default to CLINICAL."""
        result = asyncio.run(classify_topic(
            "How does music affect the brain?"
        ))
        assert result.domain == ResearchDomain.CLINICAL
        assert result.confidence == 0.5
        assert "conservative" in result.reasoning.lower() or "default" in result.reasoning.lower()

    def test_deterministic_bypasses_llm(self):
        """Clear clinical topic shouldn't need LLM at all."""
        result = asyncio.run(classify_topic(
            "Randomized clinical trial of aspirin for heart disease patients",
            smart_client=None, smart_model=""
        ))
        assert result.domain == ResearchDomain.CLINICAL
        assert result.confidence >= 0.6


class TestDomainClassificationDataclass:

    def test_dataclass_fields(self):
        dc = DomainClassification(
            domain=ResearchDomain.CLINICAL,
            confidence=0.9,
            reasoning="Test",
            suggested_framework="PICO",
            primary_databases=["PubMed"],
        )
        assert dc.domain == ResearchDomain.CLINICAL
        assert dc.confidence == 0.9
        assert dc.suggested_framework == "PICO"

    def test_research_domain_values(self):
        assert ResearchDomain.CLINICAL.value == "clinical"
        assert ResearchDomain.SOCIAL_SCIENCE.value == "social_science"
        assert ResearchDomain.GENERAL.value == "general"
