"""Tests for the TTS misreading detector (PLAN.md Step 7, Layers 0/1/2).

These lock in two contracts that each caused a round of false findings on 2026-07-28:
  1. `_norm` must treat every katakana long-vowel SPELLING as equivalent. Getting this
     wrong made Layer 1 flag 93/93 lines, then 222, before settling at 139.
  2. A hazard is only a misreading when the SOURCE contains the hazardous form. Matching
     on the reading alone produced 13 false positives (コワサ matched the correct
     壊さなかった/怖さ, ゼロゼロ matched a correct 0.001).
"""

import pytest

from dr2_podcast.tools.tts_reading_check import (
    HAZARD_READINGS,
    MARKER_RE,
    SPEAKER_RE,
    _norm,
    check_line,
    spoken_lines,
)


class TestNorm:
    @pytest.mark.parametrize(
        "a,b",
        [
            ("デエタ", "データ"),  # doubled vowel vs ー
            ("コオヒイ", "コーヒー"),
            ("メエカア", "メーカー"),
            ("ホオホオ", "ホウホウ"),  # o-row + ウ is a long o
            ("エエヨオガク", "エイヨウガク"),  # e-row + イ is a long e, plus ヨウ
            ("コオコク", "コウコク"),
            ("ケエケン", "ケイケン"),
            ("ヘエキン", "ヘイキン"),
            ("ニュウス", "ニュース"),
            ("ソオゾオ", "ソウゾウ"),
        ],
    )
    def test_long_vowel_spellings_are_equivalent(self, a, b):
        assert _norm(a) == _norm(b), f"{a} should normalise equal to {b}"

    @pytest.mark.parametrize(
        "a,b",
        [
            ("ヲ", "オ"),
            ("ヅ", "ズ"),
            ("ヂ", "ジ"),
        ],
    )
    def test_kana_variants_collapse(self, a, b):
        assert _norm(a) == _norm(b)

    def test_punctuation_and_non_kana_dropped(self):
        assert _norm("ソオデス.ハイ,ソオ!") == _norm("ソオデスハイソオ")
        assert _norm("エイブラハム・ウォールド") == _norm("エイブラハムウォールド")

    def test_genuinely_different_readings_stay_different(self):
        # the whole point — normalisation must not erase real misreadings
        assert _norm("ツヨサ") != _norm("コワサ")  # 強さ vs 怖さ
        assert _norm("タテマエ") != _norm("ケンマエ")  # 建前
        assert _norm("イツツメ") != _norm("ゴツメ")  # 五つ目
        assert _norm("ヒト") != _norm("ジン")  # 人


class TestSpokenLines:
    def test_speaker_prefix_stripped(self):
        # engine.py:325 sends only group(3) to the engine, never the prefix
        assert spoken_lines("Speaker 1: こんにちは") == ["こんにちは"]
        assert spoken_lines("Speaker 2：こんにちは") == ["こんにちは"]

    def test_markers_not_spoken(self):
        # engine.py:386 treats [TRANSITION] as a BGM cue
        assert spoken_lines("[TRANSITION]") == []
        assert spoken_lines("Speaker 1: あ\n[TRANSITION]\nSpeaker 2: い") == ["あ", "い"]

    def test_blank_lines_dropped(self):
        assert spoken_lines("\n\nSpeaker 1: あ\n\n") == ["あ"]

    def test_regexes_match_engine(self):
        assert SPEAKER_RE.match("Speaker 1: x").group(3) == "x"
        assert MARKER_RE.match("[TRANSITION]")
        assert not MARKER_RE.match("[not a marker]")


class TestHazardGating:
    def test_every_hazard_pairs_source_with_reading(self):
        for src, val in HAZARD_READINGS.items():
            assert isinstance(val, tuple) and len(val) == 2, f"{src} must be (reading, why)"

    def test_hazard_requires_source_form(self, monkeypatch):
        """怖さ/壊さ legitimately read コワサ — flagging on the reading alone is the bug."""
        import dr2_podcast.tools.tts_reading_check as mod

        monkeypatch.setattr(mod, "engine_reading", lambda t, s, sess: "コワサ")
        monkeypatch.setattr(mod, "openjtalk_reading", lambda t: "コワサ")
        # source has no 強さ -> not a hazard
        assert check_line("怖さ", 1, None) is None
        # source has 強さ AND the engine says コワサ -> real hazard
        f = check_line("強さ", 1, None)
        assert f and f["priority"] == "HIGH"

    def test_empty_reading_is_flagged(self, monkeypatch):
        """A line that produces no audio is silent content loss — worse than a misreading."""
        import dr2_podcast.tools.tts_reading_check as mod

        monkeypatch.setattr(mod, "engine_reading", lambda t, s, sess: "")
        f = check_line("△△", 1, None)
        assert f and f["reason"] == "empty_reading"
