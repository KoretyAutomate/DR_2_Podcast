"""
Domain classifier for routing topics to the appropriate research pipeline.

Classifies topics into:
- CLINICAL: Health/medical topics → existing 7-step clinical pipeline (PubMed, PICO, GRADE)
- SOCIAL_SCIENCE: Education, parenting, productivity → social science pipeline (ERIC, OpenAlex, PECO)
- GENERAL: Ambiguous topics → defaults to CLINICAL (conservative fallback)

Classification uses deterministic keyword rules first, LLM only for ambiguous cases.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ResearchDomain(Enum):
    CLINICAL = "clinical"
    SOCIAL_SCIENCE = "social_science"
    GENERAL = "general"


@dataclass
class DomainClassification:
    domain: ResearchDomain
    confidence: float           # 0.0-1.0
    reasoning: str
    suggested_framework: str    # "PICO" | "PECO"
    primary_databases: list     # e.g. ["PubMed", "Google Scholar"] or ["OpenAlex", "ERIC"]


# --- Deterministic keyword rules ---

CLINICAL_KEYWORDS = {
    # Conditions & anatomy
    "disease", "disorder", "syndrome", "cancer", "tumor", "diabetes", "hypertension",
    "cholesterol", "obesity", "bmi", "blood pressure", "heart", "cardiac", "liver",
    "kidney", "lung", "brain", "neural", "alzheimer", "parkinson", "dementia",
    "depression", "anxiety", "adhd", "autism spectrum",
    # Interventions
    "drug", "medication", "supplement", "vitamin", "dosage", "prescription",
    "surgery", "chemotherapy", "immunotherapy", "vaccine", "antibiotic",
    "statin", "insulin", "aspirin", "metformin",
    # Clinical methodology
    "clinical trial", "rct", "randomized", "placebo", "double-blind",
    "patient", "diagnosis", "treatment", "therapy", "prognosis",
    "mortality", "morbidity", "incidence", "prevalence",
    # Biomarkers & mechanisms
    "biomarker", "gene expression", "receptor", "enzyme", "protein",
    "inflammation", "oxidative stress", "metabolic",
    # Substances with health effects
    "caffeine", "alcohol", "nicotine", "cannabis", "thc", "cbd",
}

SOCIAL_SCIENCE_KEYWORDS = {
    # Education
    "education", "school", "classroom", "teacher", "student", "curriculum",
    "homework", "reading intervention", "math instruction", "stem education",
    "special education", "gifted", "kindergarten", "elementary", "middle school",
    "high school", "university", "college", "academic achievement", "test scores",
    "standardized test", "literacy", "numeracy", "tutoring", "pedagogy",
    "online learning", "distance learning", "homeschool",
    # Parenting & child development
    "parenting", "child development", "daycare", "childcare", "preschool",
    "attachment", "screen time", "sibling", "family structure", "divorce",
    "co-parenting", "breastfeeding duration", "toilet training",
    "developmental milestone", "socialization",
    # Productivity & workplace
    "productivity", "remote work", "work from home", "telecommuting",
    "4-day work week", "open office", "workplace", "employee",
    "job satisfaction", "burnout", "work-life balance",
    # Social science methodology
    "quasi-experimental", "difference-in-differences", "regression discontinuity",
    "cohen's d", "effect size", "longitudinal study",
}

# Patterns that strongly indicate a domain (compiled once)
_CLINICAL_PATTERNS = [
    re.compile(r"\b(?:mg|mcg|iu)\s*/\s*(?:day|kg)\b", re.IGNORECASE),
    re.compile(r"\brisk\s+(?:of|for)\s+\w+\s+(?:disease|cancer|mortality)\b", re.IGNORECASE),
    re.compile(r"\bdietary\s+(?:supplement|intake)\b", re.IGNORECASE),
]

_SOCIAL_SCIENCE_PATTERNS = [
    re.compile(r"\beffects?\s+(?:of|on)\s+\w+\s+(?:achievement|performance|development)\b", re.IGNORECASE),
    re.compile(r"\bchild(?:ren)?(?:'s)?\s+(?:development|behavior|outcomes?)\b", re.IGNORECASE),
    re.compile(r"\bstudent\s+(?:achievement|outcomes?|performance)\b", re.IGNORECASE),
]


def _keyword_score(topic_lower: str, keywords: set) -> float:
    """Count keyword matches, normalized to 0-1 range."""
    matches = sum(1 for kw in keywords if kw in topic_lower)
    return min(matches / 3.0, 1.0)  # 3+ matches = max confidence


def _pattern_matches(topic: str, patterns: list) -> int:
    return sum(1 for p in patterns if p.search(topic))


def classify_topic_deterministic(topic: str) -> Optional[DomainClassification]:
    """Deterministic classification using keyword rules.

    Returns None if topic is ambiguous (requires LLM fallback).
    """
    topic_lower = topic.lower()

    clinical_score = _keyword_score(topic_lower, CLINICAL_KEYWORDS)
    social_score = _keyword_score(topic_lower, SOCIAL_SCIENCE_KEYWORDS)

    clinical_score += _pattern_matches(topic, _CLINICAL_PATTERNS) * 0.3
    social_score += _pattern_matches(topic, _SOCIAL_SCIENCE_PATTERNS) * 0.3

    # Clear winner with sufficient margin
    if clinical_score >= 0.6 and clinical_score > social_score + 0.2:
        return DomainClassification(
            domain=ResearchDomain.CLINICAL,
            confidence=min(clinical_score, 1.0),
            reasoning=f"Topic contains clinical/health keywords (score={clinical_score:.2f})",
            suggested_framework="PICO",
            primary_databases=["PubMed", "Google Scholar"],
        )

    if social_score >= 0.6 and social_score > clinical_score + 0.2:
        return DomainClassification(
            domain=ResearchDomain.SOCIAL_SCIENCE,
            confidence=min(social_score, 1.0),
            reasoning=f"Topic contains social science keywords (score={social_score:.2f})",
            suggested_framework="PECO",
            primary_databases=["OpenAlex", "ERIC", "Google Scholar"],
        )

    # Ambiguous — needs LLM
    return None


async def classify_topic(
    topic: str,
    smart_client=None,
    smart_model: str = "",
) -> DomainClassification:
    """Classify a research topic into a domain.

    Tries deterministic rules first, falls back to LLM for ambiguous topics.
    Falls back to CLINICAL if LLM classification confidence < 0.6.

    Args:
        topic: The research topic string
        smart_client: AsyncOpenAI client (optional, for LLM fallback)
        smart_model: Model name for LLM classification
    """
    # Deterministic rules
    det = classify_topic_deterministic(topic)
    if det is not None:
        return det

    # LLM classification for ambiguous topics
    if smart_client and smart_model:
        try:
            return await _classify_with_llm(topic, smart_client, smart_model)
        except Exception as e:
            logger.warning(f"LLM domain classification failed: {e}")

    # Conservative fallback: CLINICAL (preserves existing pipeline behavior)
    return DomainClassification(
        domain=ResearchDomain.CLINICAL,
        confidence=0.5,
        reasoning="Ambiguous topic — defaulting to clinical pipeline (conservative fallback)",
        suggested_framework="PICO",
        primary_databases=["PubMed", "Google Scholar"],
    )


async def _classify_with_llm(
    topic: str, smart_client, smart_model: str
) -> DomainClassification:
    """Use LLM to classify ambiguous topics."""
    prompt = (
        "Classify the following research topic into exactly one domain.\n\n"
        f"TOPIC: {topic}\n"
    )
    prompt += (
        "\nDOMAINS:\n"
        "1. CLINICAL — Health, medicine, nutrition, pharmacology, disease, fitness.\n"
        "   Uses: PubMed, PICO framework, GRADE evidence levels, NNT/ARR statistics.\n"
        "2. SOCIAL_SCIENCE — Education, parenting, child development, productivity, workplace, social policy.\n"
        "   Uses: ERIC, OpenAlex, PECO framework, Cohen's d effect sizes.\n"
        "3. GENERAL — Does not fit either domain well.\n\n"
        "Respond with ONLY a JSON object (no markdown):\n"
        '{"domain": "CLINICAL" or "SOCIAL_SCIENCE" or "GENERAL", '
        '"confidence": 0.0-1.0, "reasoning": "brief explanation"}'
    )

    resp = await smart_client.chat.completions.create(
        model=smart_model,
        messages=[
            {"role": "system", "content": "/no_think You are a research methodology classifier. Respond only with JSON."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        temperature=0.1,
        timeout=30,
    )
    raw = resp.choices[0].message.content.strip()

    # Parse JSON from response (handle markdown code blocks)
    json_match = re.search(r'\{[^}]+\}', raw)
    if not json_match:
        raise ValueError(f"No JSON found in LLM response: {raw[:200]}")

    data = json.loads(json_match.group())
    domain_str = data.get("domain", "GENERAL").upper()
    confidence = float(data.get("confidence", 0.5))
    reasoning = data.get("reasoning", "LLM classification")

    domain_map = {
        "CLINICAL": ResearchDomain.CLINICAL,
        "SOCIAL_SCIENCE": ResearchDomain.SOCIAL_SCIENCE,
        "GENERAL": ResearchDomain.GENERAL,
    }
    domain = domain_map.get(domain_str, ResearchDomain.GENERAL)

    # Conservative: if low confidence, default to CLINICAL
    if confidence < 0.6 or domain == ResearchDomain.GENERAL:
        return DomainClassification(
            domain=ResearchDomain.CLINICAL,
            confidence=confidence,
            reasoning=f"LLM classified as {domain_str} with low confidence ({confidence:.2f}) — defaulting to CLINICAL. {reasoning}",
            suggested_framework="PICO",
            primary_databases=["PubMed", "Google Scholar"],
        )

    framework = "PECO" if domain == ResearchDomain.SOCIAL_SCIENCE else "PICO"
    databases = (
        ["OpenAlex", "ERIC", "Google Scholar"]
        if domain == ResearchDomain.SOCIAL_SCIENCE
        else ["PubMed", "Google Scholar"]
    )

    return DomainClassification(
        domain=domain,
        confidence=confidence,
        reasoning=f"LLM: {reasoning}",
        suggested_framework=framework,
        primary_databases=databases,
    )
