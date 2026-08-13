"""Tests for the 3-layer TTS reading enforcement system.

Layer 1: engine.apply_tts_glossary (deterministic context-free substitution)
Layer 2: prompt_strings reading_tone_rules (editor prompt — smoke-checked)
Layer 3: pipeline_validators.validate_tts_readings (context-dependent hazard warn)

See PLAN.md "TTS glossary + style-rules pipeline enforcement".
"""

import pytest

from dr2_podcast.audio import engine
from dr2_podcast.audio.engine import (
    apply_tts_glossary,
    clean_script_for_tts,
    _load_tts_glossary,
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
        assert apply_tts_glossary("母数と仕方") == "ぼすうとしかた"

    @pytest.mark.parametrize(
        "compound",
        [
            "酵母数",
            "奉仕方法",
        ],
    )
    def test_boundary_safety(self, compound):
        # Embeddable keys (母数/仕方) must NOT be substituted
        # inside a correctly-read compound — this is the regression the review
        # (B2) flagged and the reason deterministic substitution was risky.
        assert apply_tts_glossary(compound) == compound

    def test_guarded_standalone_still_fires(self):
        assert apply_tts_glossary("この母数です") == "このぼすうです"
        assert apply_tts_glossary("その仕方") == "そのしかた"

    def test_safe_full_phrase(self):
        assert apply_tts_glossary("建前と捕食者") == "たてまえとほしょくしゃ"

    def test_non_key_untouched(self):
        # 働き者/人気者 are NOT keys; a pass that converts 笑い者 must leave them alone.
        out = apply_tts_glossary("笑い者と働き者と人気者")
        assert "働き者" in out and "人気者" in out and "わらいもの" in out

    def test_idempotent(self):
        s = "母数と仕方、酵母数と奉仕方法、建前、捕食者、〇〇"
        once = apply_tts_glossary(s)
        assert apply_tts_glossary(once) == once

    def test_acronyms(self):
        assert apply_tts_glossary("NMNとNIH") == "エヌエムエヌとアメリカ国立衛生研究所"

    def test_bmi_is_left_alone(self):
        # 2026-08-12 (ep14 listening): BMI→ボディマスインデックス was an expansion,
        # not a misreading fix. AivisSpeech says ビイエムアイ for a bare BMI, which
        # is what the user wants spoken.
        assert apply_tts_glossary("たとえばBMIが正常な人") == "たとえばBMIが正常な人"

    # Ep09 listening round 2026-07-24: 五つ目→「ごつめ」, 建前→「けんまえ」,
    # 放っておけない→「はなっておけない」. Same family as the 2026-07-24 ep08
    # finding 4つ目→「よんつめ」. All context-free → glossary, not per-script kana.
    @pytest.mark.parametrize(
        "src,want",
        [
            ("五つ目のパターン", "いつつめのパターン"),
            ("五つ、高額の掲載料", "いつつ、高額の掲載料"),
            ("六つ目、特許取得", "むっつめ、特許取得"),
            ("5つ目のフレーズ", "いつつめのフレーズ"),
            ("6つのフレーズ", "むっつのフレーズ"),
            ("4つ目の選択肢", "よっつめの選択肢"),
            ("建前としては", "たてまえとしては"),
            ("放っておけない病気", "ほうっておけない病気"),
            ("放っておくと悪化する", "ほうっておくと悪化する"),
        ],
    )
    def test_ep09_round_readings(self, src, want):
        assert apply_tts_glossary(src) == want

    # Ep05 listening round 2026-08-03: 「何人いれば十分か」→ イレバジュップンカ.
    # STRICTLY context-dependent — every other 十分 surface reads ジュウブン correctly
    # both in isolation and in its real line, so the key is the 3-char 十分か, NOT 十分.
    # Widening it to 十分 would flatten the 21 correctly-read occurrences in the corpus
    # and degrade their pitch accent for no gain.
    @pytest.mark.parametrize(
        "src,want",
        [
            ("「何人いれば十分か」に決まった線はない", "「何人いればじゅうぶんか」に決まった線はない"),
            ("効果量や信頼区間は十分か、再現性は", "効果量や信頼区間はじゅうぶんか、再現性は"),
        ],
    )
    def test_ep05_juubun_ka(self, src, want):
        assert apply_tts_glossary(src) == want

    @pytest.mark.parametrize(
        "src",
        [
            "それだけでは不十分。",
            "基本はこれで十分です。",
            "まだデータが十分に集まっていない",
            "生物学的妥当性は「十分条件」ではありません",
            "用量反応は週百五十分くらいまで",
        ],
    )
    def test_juubun_ka_does_not_overreach(self, src):
        # No 十分か substring -> the entry must not fire. 週百五十分 is 150 MINUTES
        # (correctly ジュップン) and must never be rewritten.
        assert apply_tts_glossary(src) == src

    def test_ordinal_longest_first(self):
        # 五つ目/六つ目 must win over the bare 五つ/六つ keys, and the guarded
        # 目の rule (which fires first) must still land on いつつめの.
        # 2026-07-28: 五つめ->ゴツメ / 五つめの->ゴツメノ confirmed broken via /audio_query,
        # so this family is load-bearing and must NOT be pruned.
        assert apply_tts_glossary("五つ目のパターンと六つ目") == "いつつめのパターンとむっつめ"

    # 2026-07-26: the 目の->めの rule was itself CAUSING the reported 五つ目->「ごつめ」.
    # Engine readings (verified via /audio_query moras):
    #   一つ目の -> ヒトツメノ ok | 一つめの -> イチツメノ BROKEN
    #   五つ目の -> イツツメノ ok | 五つめの -> ゴツメノ  BROKEN
    #   六つ目の -> ムッツメノ ok | 六つめの -> ロクツメノ BROKEN
    # So the ordinal family must be guarded against that rule.
    @pytest.mark.parametrize(
        "src",
        [
            "一つ目のパターン",
            "二つ目のパターン",
            "三つ目のパターン",
            "四つ目のパターン",
            "七つ目の場所",
            "3つ目の質問",
        ],
    )
    def test_ordinal_plus_no_is_guarded(self, src):
        # The hazard is a KANJI numeral left in front of めの — that is what misreads
        # (五つめの -> ゴツメノ). A fully-kana ordinal is fine: verified 2026-07-28 via
        # /audio_query, みっつめの -> ミッツメノ and よっつめの -> ヨッツメノ are both
        # CORRECT. So the invariant is "no kanji numeral + めの", not "no めの at all" —
        # the broader form failed once 三つ目/四つ目 gained their own safe keys.
        import re

        out = apply_tts_glossary(src)
        assert not re.search(r"[一二三四五六七八九十]つ?めの?", out), f"{src} -> {out}"

    @pytest.mark.parametrize(
        "src,want",
        [
            ("五つ目のパターン", "いつつめのパターン"),  # safe key wins, kana is unambiguous
            ("六つ目の場所", "むっつめの場所"),
            ("4つ目の質問", "よっつめの質問"),
            ("一つ目のパターン", "一つ目のパターン"),  # untouched — reads ヒトツメノ correctly
        ],
    )
    def test_ordinal_readings_after_guard(self, src, want):
        assert apply_tts_glossary(src) == want

    def test_me_no_rule_still_fires_where_intended(self):
        # the guard must not disable the rule for its original targets
        assert apply_tts_glossary("目のつけどころ") == "めのつけどころ"

    def test_ordinals_idempotent(self):
        s = "五つ目、六つ目、5つ目、4つ目、建前、放っておけない"
        once = apply_tts_glossary(s)
        assert apply_tts_glossary(once) == once


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
        out = clean_script_for_tts("Speaker 1: 他の薬と母数")
        assert "ほかの" in out and "ぼすう" in out


# --------------------------------------------------------------------------- #
# Layer 3 — context-dependent hazard validator
# --------------------------------------------------------------------------- #
class TestValidateReadings:
    @pytest.mark.parametrize(
        "text",
        [
            "コインを投げて表が出た",
            "表と裏の関係",
            "リスクは大ありです",
            "この料理は辛いです",
            "下の方を見てください",
            "あの方は医師です",
        ],
    )
    def test_flags_hazards(self, text):
        assert validate_tts_readings(text), f"should flag: {text}"

    @pytest.mark.parametrize(
        "text",
        [
            "研究が発表された",
            "代表的な例です",
            "この方法が良い",
            "両方とも正しい",
            "辛い経験でした",
        ],
    )
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
        monkeypatch.setattr(pipeline, "_audit_script_language", lambda s, lang, cfg: s)
        monkeypatch.setattr(pipeline, "_add_reaction_guidance", lambda s, cfg: s)
        script = "Host 1: コインを投げると表が出ることがあります。\nHost 2: なるほど、確率の話ですね。\n"
        with caplog.at_level(logging.WARNING, logger=pipeline.logger.name):
            pipeline._finalize_script(
                polished_text=script, polish_task=None, language="ja", language_config={}, output_dir=tmp_path
            )
        assert any("TTS_READING" in r.message for r in caplog.records), (
            "validate_tts_readings did not run inside _finalize_script"
        )


# --------------------------------------------------------------------------- #
# Layer 2 — editor prompt smoke check
# --------------------------------------------------------------------------- #
class TestEditorPromptRules:
    def test_reading_tone_block_present(self):
        ja = get_prompt("polish", "reading_tone_rules", "ja")
        assert "ほう" in ja and "からい" in ja and "超" in ja
        en = get_prompt("polish", "reading_tone_rules", "en")
        assert "READING & TONE" in en
