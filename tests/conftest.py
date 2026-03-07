"""Shared pytest fixtures for DR_2_Podcast test suite."""

import pytest
import os


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Set minimum env vars so modules can be imported without real services."""
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("LLM_API_KEY", "NA")
    monkeypatch.setenv("FAST_MODEL_NAME", "test-fast")
    monkeypatch.setenv("FAST_LLM_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("MID_MODEL_NAME", "test-mid")
    monkeypatch.setenv("MID_LLM_BASE_URL", "http://localhost:9999/v1")


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory for test artifacts."""
    d = tmp_path / "test_output"
    d.mkdir()
    return d


@pytest.fixture
def sample_blueprint():
    """Sample blueprint text with Section 5 inline Discussion Points (new format)."""
    return """# Episode Blueprint: Coffee and Cognition

## 1. Thesis
Coffee improves cognition through caffeine's adenosine antagonism.

## 5. Narrative Arc (4 Acts) with Discussion Points

### Act 1 --- The Claim
What people believe. The folk wisdom or common assumption.
Key points to cover: caffeine myths, cultural significance, personal relevance

**Discussion Points (generate 5-8):**
- Q: Does coffee really improve brain function?
  A: Yes, multiple RCTs show acute cognitive improvements from 200-400mg caffeine. The effect is most robust for alertness and reaction time, with smaller effects on working memory.
- Q: How long have humans been using coffee for mental performance?
  A: Coffee consumption dates to 15th century Ethiopia. Historical use for alertness predates any scientific understanding of its mechanism, suggesting strong empirical awareness.
- Q: What is the specific receptor subtype responsible for caffeine's effects?
  A: Adenosine A1 and A2A receptors are the primary targets. A2A receptors in the striatum modulate dopaminergic signaling, which may explain mood enhancement separate from pure alertness effects.

### Act 2 --- Evidence & Nuance (above 50% of the episode)
**Structure:** Start by stating the overall conclusion upfront.

**Discussion Points (generate 5-8, at least 1 per study):**
- Q: What does the clinical evidence show?
  A: A 2023 meta-analysis of 41 RCTs found significant improvements in attention (SMD=0.43) and reaction time (SMD=0.38) with 200-300mg caffeine doses in healthy adults.
- Q: Are there differences between habitual and non-habitual coffee drinkers?
  A: Non-habitual drinkers show larger acute effects, but habitual users maintain baseline performance better. Withdrawal reversal explains about 30% of apparent benefits in chronic users.
- Q: What neuroimaging studies have examined caffeine's mechanisms?
  A: PET studies using [11C]DPCPX show dose-dependent adenosine receptor occupancy. fMRI studies demonstrate increased prefrontal activation correlating with improved executive function tasks.

### Act 3 --- Holistic Conclusion
Synthesize ALL evidence into a unified takeaway.

**Discussion Points (generate 3-5):**
- Q: Are there downsides to coffee consumption?
  A: Anxiety, insomnia, and cardiovascular effects are well-documented. The optimal dose window is 200-400mg; higher doses often impair performance through anxiety and tremor.
- Q: How does individual genetics affect coffee response?
  A: CYP1A2 polymorphisms affect caffeine metabolism speed. Fast metabolizers (AA genotype) show different risk profiles than slow metabolizers (AC/CC), particularly for cardiovascular outcomes.

### Act 4 --- The Protocol
Actionable translation to daily life.

**Discussion Points (generate 5-8):**
- Q: What is the practical recommendation for cognitive performance?
  A: 200mg caffeine (about 2 cups) consumed 30-45 minutes before cognitively demanding tasks. Avoid after 2pm to protect sleep, which is essential for memory consolidation.
- Q: Can tolerance be managed strategically?
  A: Caffeine cycling protocols suggest 5 days on, 2 days off. Some researchers advocate minimum effective dose strategies to preserve receptor sensitivity while maintaining cognitive benefits.

## 6. GRADE-Informed Framing Guide
...

## 7. Citations
...
"""


@pytest.fixture
def sample_blueprint_legacy():
    """Sample blueprint text with legacy Section 8 Discussion Inventory."""
    return """# Episode Blueprint: Coffee and Cognition

## 1. Thesis
Coffee improves cognition through caffeine's adenosine antagonism.

## 7. Citations
...

## 8. Discussion Inventory
For each of the 4 Acts, list 3-4 discussion items.

### Act 1 — The Claim
- [Basic] Q: Does coffee really improve brain function?
  A: Yes, multiple RCTs show acute cognitive improvements from 200-400mg caffeine.
- [Context] Q: How long have humans been using coffee for mental performance?
  A: Coffee consumption dates to 15th century Ethiopia.

### Act 2 — The Evidence
- [Basic] Q: What does the clinical evidence show?
  A: A 2023 meta-analysis of 41 RCTs found significant improvements.

### Act 3 — The Nuance
- [Basic] Q: Are there downsides to coffee consumption?
  A: Anxiety, insomnia, and cardiovascular effects are well-documented.

### Act 4 — The Protocol
- [Basic] Q: What is the practical recommendation?
  A: 200mg caffeine consumed 30-45 minutes before demanding tasks.

## 9. Appendix
Extra stuff here.
"""


@pytest.fixture
def sample_inventory():
    """Pre-parsed inventory dict matching the sample_blueprint Section 5."""
    return {
        "Act 1 --- The Claim": [
            {'question': 'Does coffee really improve brain function?',
             'answer': 'Yes, multiple RCTs show acute cognitive improvements from 200-400mg caffeine. The effect is most robust for alertness and reaction time, with smaller effects on working memory.'},
            {'question': 'How long have humans been using coffee for mental performance?',
             'answer': 'Coffee consumption dates to 15th century Ethiopia. Historical use for alertness predates any scientific understanding of its mechanism, suggesting strong empirical awareness.'},
            {'question': 'What is the specific receptor subtype responsible for caffeine\'s effects?',
             'answer': 'Adenosine A1 and A2A receptors are the primary targets. A2A receptors in the striatum modulate dopaminergic signaling, which may explain mood enhancement separate from pure alertness effects.'},
        ],
        "Act 2 --- Evidence & Nuance (above 50% of the episode)": [
            {'question': 'What does the clinical evidence show?',
             'answer': 'A 2023 meta-analysis of 41 RCTs found significant improvements in attention (SMD=0.43) and reaction time (SMD=0.38) with 200-300mg caffeine doses in healthy adults.'},
            {'question': 'Are there differences between habitual and non-habitual coffee drinkers?',
             'answer': 'Non-habitual drinkers show larger acute effects, but habitual users maintain baseline performance better. Withdrawal reversal explains about 30% of apparent benefits in chronic users.'},
            {'question': 'What neuroimaging studies have examined caffeine\'s mechanisms?',
             'answer': 'PET studies using [11C]DPCPX show dose-dependent adenosine receptor occupancy. fMRI studies demonstrate increased prefrontal activation correlating with improved executive function tasks.'},
        ],
        "Act 3 --- Holistic Conclusion": [
            {'question': 'Are there downsides to coffee consumption?',
             'answer': 'Anxiety, insomnia, and cardiovascular effects are well-documented. The optimal dose window is 200-400mg; higher doses often impair performance through anxiety and tremor.'},
            {'question': 'How does individual genetics affect coffee response?',
             'answer': 'CYP1A2 polymorphisms affect caffeine metabolism speed. Fast metabolizers (AA genotype) show different risk profiles than slow metabolizers (AC/CC), particularly for cardiovascular outcomes.'},
        ],
        "Act 4 --- The Protocol": [
            {'question': 'What is the practical recommendation for cognitive performance?',
             'answer': '200mg caffeine (about 2 cups) consumed 30-45 minutes before cognitively demanding tasks. Avoid after 2pm to protect sleep, which is essential for memory consolidation.'},
            {'question': 'Can tolerance be managed strategically?',
             'answer': 'Caffeine cycling protocols suggest 5 days on, 2 days off. Some researchers advocate minimum effective dose strategies to preserve receptor sensitivity while maintaining cognitive benefits.'},
        ],
    }


@pytest.fixture
def english_lang_config():
    """Language config for English (word-based counting)."""
    return {'length_unit': 'words'}


@pytest.fixture
def japanese_lang_config():
    """Language config for Japanese (character-based counting)."""
    return {'length_unit': 'chars'}


@pytest.fixture
def mock_llm_response():
    """Factory fixture for mock OpenAI LLM responses."""
    class MockChoice:
        def __init__(self, content):
            self.message = type('obj', (object,), {'content': content})()

    class MockResponse:
        def __init__(self, content):
            self.choices = [MockChoice(content)]

    def _make(content="test response"):
        return MockResponse(content)

    return _make


@pytest.fixture
def make_deep_extraction():
    """Factory for DeepExtraction-like dicts."""
    def _make(**overrides):
        base = {"pmid": "12345678", "doi": "10.1000/test", "title": "Test Study",
                "authors": "Smith J et al.", "year": "2023", "journal": "Test J",
                "study_type": "RCT", "sample_size": "100", "population": "Adults",
                "intervention": "Drug A", "comparator": "Placebo",
                "primary_outcome": "Recovery", "effect_size": "0.5",
                "ci_lower": "0.2", "ci_upper": "0.8", "p_value": "0.01",
                "risk_of_bias": "Low", "key_findings": "Significant improvement",
                "limitations": "Small sample", "abstract": "Test abstract."}
        base.update(overrides)
        return base
    return _make


@pytest.fixture
def make_wide_net_record():
    """Factory for WideNetRecord-like dicts."""
    def _make(**overrides):
        base = {"pmid": "12345678", "doi": "10.1000/test", "title": "Test Study",
                "citation_count": 50, "fwci": 1.2, "source": "pubmed"}
        base.update(overrides)
        return base
    return _make


@pytest.fixture
def make_pipeline_data(make_deep_extraction, make_wide_net_record):
    """Factory for minimal pipeline_data dict (input to build_imrad_sot)."""
    def _make(**overrides):
        ext = make_deep_extraction()
        base = {
            "domain": "clinical", "framing_context": "Test topic framing",
            "search_date": "2026-03-01",
            "aff_strategy": {"tiers": []}, "fal_strategy": {"tiers": []},
            "aff_extractions": [ext], "fal_extractions": [ext],
            "aff_top": [make_wide_net_record()], "fal_top": [make_wide_net_record()],
            "math_report": "No math applicable.", "impacts": [],
            "aff_case": "Affirmative case text.", "fal_case": "Falsification case text.",
            "grade_synthesis": "### Overall\nGRADE: Moderate\nVerdict: Supported",
            "aff_metrics": {"candidates": 10}, "fal_metrics": {"candidates": 8},
        }
        base.update(overrides)
        return base
    return _make
