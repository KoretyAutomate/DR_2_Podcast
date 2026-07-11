"""Tests for the 3-layer TTS reading enforcement system.

Layer 1: engine.apply_tts_glossary (deterministic context-free substitution)
Layer 2: prompt_strings reading_tone_rules (editor prompt — smoke-checked)
Layer 3: pipeline_validators.validate_tts_readings (context-dependent hazard warn)

See PLAN.md "TTS glossary + style-rules pipeline enforcement".
"""
import pytest

from dr2_podcast.audio import engine
from dr2_podcast.audio.engine import (
    apply_tts_glossary, clean_script_for_tts, _load_tts_glossary,
)
from dr2_podcast.pipeline_validators import validate_tts_readings
from dr2_podcast.prompt_strings import get_prompt


@pytest.fixture(autouse=True)
def _reset_glossary_cache():
    """Isolate the module-level glossary cache between tests."""
    engine._tts_glossary_cache = None
    yield
    engine._tts_glossary_cache = None


# --------------------------------------------------------------------------- #
# Layer 1 — deterministic glossary
# --------------------------------------------------------------------------- #
class TestApplyGlossary:
    def test_standalone_substitution(self):
        assert apply_tts_glossary("母数と強さ") == "ぼすうとつよさ"

    @pytest.mark.parametrize("compound", [
        "酵母数", "手強さ", "力強さ", "心強さ", "粘り強さ",
        "意見方針", "奉仕方法", "一番下手", "一番下品",
    ])
    def test_boundary_safety(self, compound):
        # Embeddable keys (強さ/母数/見方/仕方/一番下) must NOT be substituted
        # inside a correctly-read compound — this is the regression the review
        # (B2) flagged and the reason deterministic substitution was risky.
        assert apply_tts_glossary(compound) == compound

    def test_guarded_standalone_still_fires(self):
        assert apply_tts_glossary("階段の一番下です") == "階段の一番したです"
        assert apply_tts_glossary("この強さ") == "このつよさ"

    def test_safe_full_phrase(self):
        assert apply_tts_glossary("考え方と読み方") == "かんがえかたとよみかた"

    def test_non_key_untouched(self):
        # 作り方/選び方 are NOT keys; a pass that converts 考え方 must leave them alone.
        out = apply_tts_glossary("考え方と作り方と選び方")
        assert "作り方" in out and "選び方" in out and "かんがえかた" in out

    def test_idempotent(self):
        s = "母数と強さと考え方、酵母数と手強さ、一番下と一番下手、捕食者、〇〇"
        once = apply_tts_glossary(s)
        assert apply_tts_glossary(once) == once

    def test_acronyms(self):
        assert apply_tts_glossary("NMNとBMI") == "エヌエムエヌとボディマスインデックス"


class TestGlossaryLoad:
    def test_invariant_holds(self):
        gl = _load_tts_glossary()
        keys = list(gl["safe"]) + list(gl["guarded"])
        vals = list(gl["safe"].values()) + [g["to"] for g in gl["guarded"].values()]
        for v in vals:
            for k in keys:
                assert k not in v, f"idempotency invariant broken: {v!r} contains {k!r}"

    def test_failsafe_missing_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(engine, "_TTS_GLOSSARY_PATH", tmp_path / "nope.json")
        # Missing file must NOT crash the audio path — glossary becomes a no-op.
        assert apply_tts_glossary("母数") == "母数"

    def test_disable_flag(self, monkeypatch):
        monkeypatch.setenv("TTS_GLOSSARY_ENABLED", "0")
        assert apply_tts_glossary("母数") == "母数"


class TestCleanScriptIntegration:
    def test_furigana_then_glossary(self):
        # 母数（ぼすう） -> furigana-strip -> 母数 -> glossary -> ぼすう (single, no double read)
        assert clean_script_for_tts("母数（ぼすう）") == "ぼすう"

    def test_glossary_applied_in_clean(self):
        out = clean_script_for_tts("Speaker 1: 他の薬と強さ")
        assert "ほかの" in out and "つよさ" in out


# --------------------------------------------------------------------------- #
# Layer 3 — context-dependent hazard validator
# --------------------------------------------------------------------------- #
class TestValidateReadings:
    @pytest.mark.parametrize("text", [
        "コインを投げて表が出た", "表と裏の関係", "リスクは大ありです",
        "この料理は辛いです", "下の方を見てください", "あの方は医師です",
    ])
    def test_flags_hazards(self, text):
        assert validate_tts_readings(text), f"should flag: {text}"

    @pytest.mark.parametrize("text", [
        "研究が発表された", "代表的な例です", "この方法が良い",
        "両方とも正しい", "辛い経験でした",
    ])
    def test_skips_safe(self, text):
        assert validate_tts_readings(text) == [], f"should NOT flag: {text}"

    def test_truncation_cap(self):
        text = "\n".join("大あり" for _ in range(50))
        issues = validate_tts_readings(text, max_report=10)
        assert len(issues) == 11 and "TRUNCATED" in issues[-1]


# --------------------------------------------------------------------------- #
# Layer 3 — WIRED INTO THE LIVE PATH (proves the validator actually runs)
# --------------------------------------------------------------------------- #
class TestValidatorWiredIntoFinalize:
    def test_finalize_script_emits_tts_reading_warning(self, monkeypatch, tmp_path, caplog):
        import logging
        from dr2_podcast import pipeline
        # Patch the LLM-backed neighbours to passthrough so _finalize_script runs
        # offline; the validate_tts_readings loop sits between them in the JA branch.
        monkeypatch.setattr(pipeline, "_audit_script_language",
                            lambda s, lang, cfg: s)
        monkeypatch.setattr(pipeline, "_add_reaction_guidance",
                            lambda s, cfg: s)
        script = ("Host 1: コインを投げると表が出ることがあります。\n"
                  "Host 2: なるほど、確率の話ですね。\n")
        with caplog.at_level(logging.WARNING, logger=pipeline.logger.name):
            pipeline._finalize_script(
                polished_text=script, polish_task=None, language="ja",
                language_config={}, output_dir=tmp_path)
        assert any("TTS_READING" in r.message for r in caplog.records), \
            "validate_tts_readings did not run inside _finalize_script"


# --------------------------------------------------------------------------- #
# Layer 2 — editor prompt smoke check
# --------------------------------------------------------------------------- #
class TestEditorPromptRules:
    def test_reading_tone_block_present(self):
        ja = get_prompt("polish", "reading_tone_rules", "ja")
        assert "ほう" in ja and "からい" in ja and "超" in ja
        en = get_prompt("polish", "reading_tone_rules", "en")
        assert "READING & TONE" in en
