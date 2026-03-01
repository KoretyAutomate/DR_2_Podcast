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
    """Sample blueprint text with Section 8 Discussion Inventory."""
    return """# Episode Blueprint: Coffee and Cognition

## 1. Thesis
Coffee improves cognition through caffeine's adenosine antagonism.

## 7. Citations
...

## 8. Discussion Inventory
For each of the 4 Acts, list 3-4 discussion items.

### Act 1 — The Claim
- [Basic] Q: Does coffee really improve brain function?
  A: Yes, multiple RCTs show acute cognitive improvements from 200-400mg caffeine. The effect is most robust for alertness and reaction time, with smaller effects on working memory.
- [Context] Q: How long have humans been using coffee for mental performance?
  A: Coffee consumption dates to 15th century Ethiopia. Historical use for alertness predates any scientific understanding of its mechanism, suggesting strong empirical awareness.
- [Deep-dive] Q: What is the specific receptor subtype responsible for caffeine's effects?
  A: Adenosine A1 and A2A receptors are the primary targets. A2A receptors in the striatum modulate dopaminergic signaling, which may explain mood enhancement separate from pure alertness effects.

### Act 2 — The Evidence
- [Basic] Q: What does the clinical evidence show?
  A: A 2023 meta-analysis of 41 RCTs found significant improvements in attention (SMD=0.43) and reaction time (SMD=0.38) with 200-300mg caffeine doses in healthy adults.
- [Context] Q: Are there differences between habitual and non-habitual coffee drinkers?
  A: Non-habitual drinkers show larger acute effects, but habitual users maintain baseline performance better. Withdrawal reversal explains about 30% of apparent benefits in chronic users.
- [Deep-dive] Q: What neuroimaging studies have examined caffeine's mechanisms?
  A: PET studies using [11C]DPCPX show dose-dependent adenosine receptor occupancy. fMRI studies demonstrate increased prefrontal activation correlating with improved executive function tasks.

### Act 3 — The Nuance
- [Basic] Q: Are there downsides to coffee consumption?
  A: Anxiety, insomnia, and cardiovascular effects are well-documented. The optimal dose window is 200-400mg; higher doses often impair performance through anxiety and tremor.
- [Context] Q: How does individual genetics affect coffee response?
  A: CYP1A2 polymorphisms affect caffeine metabolism speed. Fast metabolizers (AA genotype) show different risk profiles than slow metabolizers (AC/CC), particularly for cardiovascular outcomes.

### Act 4 — The Protocol
- [Basic] Q: What is the practical recommendation for cognitive performance?
  A: 200mg caffeine (about 2 cups) consumed 30-45 minutes before cognitively demanding tasks. Avoid after 2pm to protect sleep, which is essential for memory consolidation.
- [Deep-dive] Q: Can tolerance be managed strategically?
  A: Caffeine cycling protocols suggest 5 days on, 2 days off. Some researchers advocate minimum effective dose strategies to preserve receptor sensitivity while maintaining cognitive benefits.

## 9. Appendix
Extra stuff here.
"""


@pytest.fixture
def sample_inventory():
    """Pre-parsed inventory dict matching the sample_blueprint Section 8."""
    return {
        "Act 1 — The Claim": [
            {'tier': 'Basic', 'question': 'Does coffee really improve brain function?',
             'answer': 'Yes, multiple RCTs show acute cognitive improvements from 200-400mg caffeine. The effect is most robust for alertness and reaction time, with smaller effects on working memory.'},
            {'tier': 'Context', 'question': 'How long have humans been using coffee for mental performance?',
             'answer': 'Coffee consumption dates to 15th century Ethiopia. Historical use for alertness predates any scientific understanding of its mechanism, suggesting strong empirical awareness.'},
            {'tier': 'Deep-dive', 'question': 'What is the specific receptor subtype responsible for caffeine\'s effects?',
             'answer': 'Adenosine A1 and A2A receptors are the primary targets. A2A receptors in the striatum modulate dopaminergic signaling, which may explain mood enhancement separate from pure alertness effects.'},
        ],
        "Act 2 — The Evidence": [
            {'tier': 'Basic', 'question': 'What does the clinical evidence show?',
             'answer': 'A 2023 meta-analysis of 41 RCTs found significant improvements in attention (SMD=0.43) and reaction time (SMD=0.38) with 200-300mg caffeine doses in healthy adults.'},
            {'tier': 'Context', 'question': 'Are there differences between habitual and non-habitual coffee drinkers?',
             'answer': 'Non-habitual drinkers show larger acute effects, but habitual users maintain baseline performance better. Withdrawal reversal explains about 30% of apparent benefits in chronic users.'},
            {'tier': 'Deep-dive', 'question': 'What neuroimaging studies have examined caffeine\'s mechanisms?',
             'answer': 'PET studies using [11C]DPCPX show dose-dependent adenosine receptor occupancy. fMRI studies demonstrate increased prefrontal activation correlating with improved executive function tasks.'},
        ],
        "Act 3 — The Nuance": [
            {'tier': 'Basic', 'question': 'Are there downsides to coffee consumption?',
             'answer': 'Anxiety, insomnia, and cardiovascular effects are well-documented. The optimal dose window is 200-400mg; higher doses often impair performance through anxiety and tremor.'},
            {'tier': 'Context', 'question': 'How does individual genetics affect coffee response?',
             'answer': 'CYP1A2 polymorphisms affect caffeine metabolism speed. Fast metabolizers (AA genotype) show different risk profiles than slow metabolizers (AC/CC), particularly for cardiovascular outcomes.'},
        ],
        "Act 4 — The Protocol": [
            {'tier': 'Basic', 'question': 'What is the practical recommendation for cognitive performance?',
             'answer': '200mg caffeine (about 2 cups) consumed 30-45 minutes before cognitively demanding tasks. Avoid after 2pm to protect sleep, which is essential for memory consolidation.'},
            {'tier': 'Deep-dive', 'question': 'Can tolerance be managed strategically?',
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
