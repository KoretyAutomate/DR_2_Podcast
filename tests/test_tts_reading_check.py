"""Tests for the TTS misreading detector (PLAN.md Step 7, Layers 0/1/2).

These lock in two contracts that each caused a round of false findings on 2026-07-28:
  1. `_norm` must treat every katakana long-vowel SPELLING as equivalent. Getting this
     wrong made Layer 1 flag 93/93 lines, then 222, before settling at 139.
  2. A hazard is only a misreading when the SOURCE contains the hazardous form. Matching
     on the reading alone produced 13 false positives (コワサ matched the correct
     壊さなかった/怖さ, ゼロゼロ matched a correct 0.001).

and, from 2026-08-13:
  3. Every line is checked at the voice that will actually SPEAK it (the module used to
     hardcode one voice that an episode may never use).
  4. `--changed-vs` reads back only what an edit changed, because the misreadings that
     survive to the audio are positional and no rule in HAZARD_READINGS can express them.
"""

import pytest
import requests

from dr2_podcast.tools.tts_reading_check import (
    HAZARD_READINGS,
    MARKER_RE,
    SPEAKER_RE,
    EngineUnavailable,
    VoiceAssignment,
    _norm,
    boundary_conflicts,
    changed_turns,
    check_line,
    check_script,
    preflight,
    resolve_voices,
    spoken_lines,
    spoken_turns,
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


# ---------------------------------------------------------------------------
# Defect A (2026-08-13): check the voice that actually speaks the line
# ---------------------------------------------------------------------------

HOST1, HOST2 = 1111111111, 2222222222


@pytest.fixture
def pinned_ids(monkeypatch):
    """Pretend TTS_HOST1_ID/TTS_HOST2_ID are HOST1/HOST2."""
    import dr2_podcast.tools.tts_reading_check as mod

    monkeypatch.setattr(mod, "_get_tts_speaker_ids_int", lambda: (HOST1, HOST2))
    return HOST1, HOST2


class TestVoiceResolution:
    def test_pinned_when_random_voice_off(self, monkeypatch, pinned_ids):
        """TTS_RANDOM_VOICE=0 is how every published episode so far was rendered."""
        import dr2_podcast.audio.engine as engine

        monkeypatch.setattr(engine, "TTS_RANDOM_VOICE", False)
        v = resolve_voices("Speaker 1: あ\nSpeaker 2: い\n")
        assert (v.speaker1, v.speaker2, v.swapped) == (HOST1, HOST2, False)

    def test_matches_the_renderer_seed(self, monkeypatch, pinned_ids):
        """Same script, same voices as audio/engine.assign_seeded_voices would pick."""
        import dr2_podcast.audio.engine as engine

        monkeypatch.setattr(engine, "TTS_RANDOM_VOICE", True)
        for script in ("Speaker 1: あ\n", "Speaker 1: い\n", "Speaker 1: 今日は天気がいい\n"):
            expect = engine.assign_seeded_voices(script, HOST1, HOST2)
            got = resolve_voices(script)
            assert (got.speaker1, got.speaker2, got.swapped) == expect

    def test_seed_actually_swaps_for_some_scripts(self, monkeypatch, pinned_ids):
        """A resolver that never swaps would pass every other test here."""
        import dr2_podcast.audio.engine as engine

        monkeypatch.setattr(engine, "TTS_RANDOM_VOICE", True)
        swaps = {resolve_voices(f"Speaker 1: 行{i}\n").swapped for i in range(20)}
        assert swaps == {True, False}

    def test_for_speaker_mirrors_flush_turn(self):
        # engine.py:_AivisTimeline.flush_turn — host1 for Speaker 1, host2 for anything else
        v = VoiceAssignment(speaker1=7, speaker2=9, swapped=False, random_voice=False)
        assert (v.for_speaker(1), v.for_speaker(2)) == (7, 9)

    def test_non_numeric_ids_fail_loudly(self, monkeypatch):
        import dr2_podcast.tools.tts_reading_check as mod

        monkeypatch.setattr(mod, "_get_tts_speaker_ids_int", lambda: (None, None))
        with pytest.raises(mod.VoiceResolutionError):
            resolve_voices("Speaker 1: あ\n")


class TestSpokenTurns:
    def test_speaker_attribution(self):
        turns = spoken_turns("Speaker 1: あ\nSpeaker 2: い\nSpeaker 1: う")
        assert [(t.speaker, t.text) for t in turns] == [(1, "あ"), (2, "い"), (1, "う")]

    def test_continuation_line_joins_the_open_turn(self):
        """engine.py buffers an unlabelled line into the CURRENT turn and sends one query.

        Asking about the fragment alone is asking about text that is never synthesized —
        and the misreadings this tool exists for are context-dependent.
        """
        turns = spoken_turns("Speaker 2: い\nつづき")
        assert [(t.speaker, t.text) for t in turns] == [(2, "い つづき")]

    def test_marker_flushes_the_turn_without_changing_speaker(self):
        turns = spoken_turns("Speaker 1: あ\n[PAUSE]\nい")
        assert [(t.speaker, t.text) for t in turns] == [(1, "あ"), (1, "い")]

    def test_unknown_bracket_line_is_spoken_not_dropped(self):
        # only MARKER_SILENCE members are markers to the renderer; [FOO] is speech
        assert spoken_turns("Speaker 1: あ\n[FOO]") == [(1, "あ [FOO]")]

    def test_heading_and_rule_lines_are_skipped(self):
        assert spoken_lines("Speaker 1: あ\n---\n## note\nい") == ["あ い"]

    def test_speaker_zero_is_never_spoken(self):
        """flush_turn's `if not (text and speaker)` drops it — reporting it would be fiction."""
        assert spoken_turns("Speaker 0: あ\nSpeaker 1: い") == [(1, "い")]

    def test_channel_intro_is_speaker_two(self):
        # engine.py: unlabelled prose before any Speaker prefix is spoken by Speaker 2
        turns = spoken_turns("チャンネル紹介です\nSpeaker 1: あ")
        assert [(t.speaker, t.text) for t in turns] == [(2, "チャンネル紹介です"), (1, "あ")]

    def test_bracket_line_before_any_speaker_is_dropped(self):
        assert spoken_turns("[TRANSITION]\n[FOO]\nSpeaker 1: あ") == [(1, "あ")]

    def test_spoken_lines_matches_the_turns(self):
        script = "Speaker 1: あ\n[TRANSITION]\n\nSpeaker 2: い\nつづき\n"
        assert spoken_lines(script) == ["あ", "い つづき"]
        assert spoken_lines(script) == [t.text for t in spoken_turns(script)]


class TestEngineInputs:
    def test_long_turn_is_queried_in_the_chunks_the_renderer_sends(self):
        """engine.py:_synthesize_turn splits a turn before /audio_query."""
        from dr2_podcast.audio.engine import _chunk_japanese_text
        from dr2_podcast.tools.tts_reading_check import Turn, engine_inputs

        long_turn = "".join(f"これは{i}番目の文です。" for i in range(12))
        chunks = engine_inputs(Turn(1, long_turn))
        assert len(chunks) > 1
        assert chunks == _chunk_japanese_text(long_turn)
        assert "".join(chunks).replace(" ", "") == long_turn.replace(" ", "")

    def test_short_turn_is_one_query(self):
        from dr2_podcast.tools.tts_reading_check import Turn, engine_inputs

        assert engine_inputs(Turn(1, "みじかい。")) == ["みじかい。"]


class TestPerLineVoice:
    """check_script must ask the engine about each line at the voice that will speak it."""

    def _script(self, tmp_path):
        p = tmp_path / "script.txt"
        p.write_text("Speaker 1: あ\nSpeaker 2: い\nSpeaker 1: う\n", encoding="utf-8")
        return p

    def _stub_engine(self, monkeypatch, calls):
        import dr2_podcast.tools.tts_reading_check as mod

        def fake_phrases(text, speaker, session):
            calls.append((text, speaker))
            return ["ヨミ"]

        monkeypatch.setattr(mod, "engine_phrases", fake_phrases)
        monkeypatch.setattr(mod, "openjtalk_reading", lambda t: "ヨミ")
        monkeypatch.setattr(mod, "openjtalk_phrases", lambda t: ["ヨミ"])
        monkeypatch.setattr(mod, "preflight", lambda session, speaker: None)

    def test_each_speaker_checked_with_its_own_voice(self, tmp_path, monkeypatch, pinned_ids):
        import dr2_podcast.audio.engine as engine

        monkeypatch.setattr(engine, "TTS_RANDOM_VOICE", False)
        calls = []
        self._stub_engine(monkeypatch, calls)
        rep = check_script(self._script(tmp_path), session=object())
        assert calls == [("あ", HOST1), ("い", HOST2), ("う", HOST1)]
        assert rep["voices"]["speaker1"] == HOST1 and rep["voices"]["speaker2"] == HOST2

    def test_swap_follows_the_lines(self, tmp_path, monkeypatch, pinned_ids):
        """When the seed swaps the voices, Speaker 1's lines must move with it."""
        import dr2_podcast.audio.engine as engine
        import dr2_podcast.tools.tts_reading_check as mod

        monkeypatch.setattr(mod, "assign_seeded_voices", lambda text, a, b: (b, a, True))
        monkeypatch.setattr(engine, "TTS_RANDOM_VOICE", True)
        calls = []
        self._stub_engine(monkeypatch, calls)
        check_script(self._script(tmp_path), session=object())
        assert calls == [("あ", HOST2), ("い", HOST1), ("う", HOST2)]

    def test_a_long_turn_is_queried_chunk_by_chunk_at_one_voice(self, tmp_path, monkeypatch, pinned_ids):
        """The renderer chunks a turn before /audio_query; the readback must match it."""
        import dr2_podcast.audio.engine as engine
        from dr2_podcast.audio.engine import _chunk_japanese_text

        monkeypatch.setattr(engine, "TTS_RANDOM_VOICE", False)
        long_turn = "".join(f"これは{i}番目の文です。" for i in range(12))
        p = tmp_path / "script.txt"
        p.write_text(f"Speaker 2: {long_turn}\n", encoding="utf-8")

        calls = []
        self._stub_engine(monkeypatch, calls)
        rep = check_script(p, session=object())
        assert [t for t, _ in calls] == _chunk_japanese_text(long_turn)
        assert {spk for _, spk in calls} == {HOST2}
        assert rep["lines_checked"] == 1  # one turn, several engine queries

    def test_explicit_speaker_forces_one_voice(self, tmp_path, monkeypatch, pinned_ids):
        calls = []
        self._stub_engine(monkeypatch, calls)
        rep = check_script(self._script(tmp_path), 4242, session=object())
        assert {spk for _, spk in calls} == {4242}
        assert rep["voices"] == {"forced": 4242}


# ---------------------------------------------------------------------------
# Defect B (2026-08-13): the misreadings that occur are positional
# ---------------------------------------------------------------------------


class TestChangedTurns:
    def test_identical_scripts_have_no_changed_lines(self):
        s = "Speaker 1: あ\nSpeaker 2: い\n"
        assert changed_turns(s, s) == []

    def test_insert_does_not_shift_everything_after_it(self):
        """The whole point of diffing content instead of line numbers."""
        prev = "Speaker 1: あ\nSpeaker 2: い\nSpeaker 1: う\n"
        cur = "Speaker 1: あ\nSpeaker 2: にゅう\nSpeaker 2: い\nSpeaker 1: う\n"
        assert [(i, t.text) for i, t in changed_turns(prev, cur)] == [(1, "にゅう")]

    def test_edited_line_is_reported_with_its_new_index(self):
        prev = "Speaker 1: あ\nSpeaker 2: い\n"
        cur = "Speaker 1: あ\nSpeaker 2: いい\n"
        assert [(i, t.text) for i, t in changed_turns(prev, cur)] == [(1, "いい")]

    def test_moving_a_line_to_the_other_speaker_is_a_change(self):
        """Same words, other voice — the reading is a different engine answer."""
        got = changed_turns("Speaker 1: あ\n", "Speaker 2: あ\n")
        assert [(i, t.speaker, t.text) for i, t in got] == [(0, 2, "あ")]

    def test_deletion_alone_changes_nothing_to_check(self):
        prev = "Speaker 1: あ\nSpeaker 2: い\n"
        cur = "Speaker 1: あ\n"
        assert changed_turns(prev, cur) == []

    def test_repeated_short_turns_do_not_confuse_the_alignment(self):
        """autojunk would treat a turn repeated in >1% of a long script as junk."""
        prev = "".join(f"Speaker 1: そうですね。\nSpeaker 2: 行{i}\n" for i in range(120))
        cur = prev.replace("行7\n", "行7です\n")
        assert [t.text for _, t in changed_turns(prev, cur)] == ["行7です"]


class TestBoundaryConflicts:
    # Real readings measured against AivisSpeech on 2026-08-13. The engine opens an accent
    # phrase inside 大きさ, so 「その大きさ自体が」 is spoken ソノ / オオキ / サジタイガ.
    ENG_OOKISA = ["デモ,", "ソノ", "オオキ", "サジタイガ", "ワカラナイコトモ", "アリマスヨネ."]
    OJT_OOKISA = ["デモ", "ソノ", "オーキサ", "ジタイガ", "ワカラナイ", "コトモ", "アリマス’ヨネ"]

    def test_flat_readings_are_identical_so_layer1_alone_is_blind(self):
        assert _norm("".join(self.ENG_OOKISA)) == _norm("".join(self.OJT_OOKISA))

    def test_engine_boundary_inside_a_word_is_flagged(self):
        conflicts = boundary_conflicts(self.ENG_OOKISA, self.OJT_OOKISA)
        assert len(conflicts) == 1 and "オオキ/サジタイ" in conflicts[0]

    def test_finer_pyopenjtalk_granularity_is_not_a_conflict(self):
        # measured: 83 of 95 lines differ this way — flagging them makes the signal useless
        assert boundary_conflicts(["ケッカワドオダッタンデスカ"], ["ケッカワ", "ドオダッタ", "ンデスカ"]) == []

    def test_incomparable_when_the_readings_differ(self):
        assert boundary_conflicts(["エピソオドワン"], ["イーピーアイエスオーディーイー", "ワン"]) == []

    def test_no_pyopenjtalk_no_claim(self):
        assert boundary_conflicts(self.ENG_OOKISA, None) == []

    def test_offsets_survive_a_phrase_that_opens_on_a_long_vowel_mark(self):
        """_norm expands ー from the PRECEDING mora, so per-phrase lengths do not add up.

        Summing them instead of normalising each prefix invents a boundary that is not
        there (here: a second, spurious conflict after タ).
        """
        assert boundary_conflicts(["デ", "ータ", "ホン"], ["デエタ", "ホン"]) == ["…デ/エタホン…"]


class TestChangedLineMode:
    def _stub(self, monkeypatch, reading_phrases, ojt_phrases):
        import dr2_podcast.tools.tts_reading_check as mod

        monkeypatch.setattr(mod, "engine_phrases", lambda t, s, sess: reading_phrases)
        monkeypatch.setattr(mod, "openjtalk_reading", lambda t: "".join(ojt_phrases))
        monkeypatch.setattr(mod, "openjtalk_phrases", lambda t: ojt_phrases)

    def test_clean_changed_line_is_still_read_back(self, monkeypatch):
        """No rule predicted the 2026-08-13 misreadings — the reading itself is the output."""
        self._stub(monkeypatch, ["ソノトオリデス"], ["ソノトオリデス"])
        f = check_line("その通りです", 1, None, segmentation=True, always_report=True)
        assert f["priority"] == "READBACK"
        assert f["engine_reading_segmented"] == "ソノトオリデス"

    def test_clean_line_stays_silent_without_always_report(self, monkeypatch):
        self._stub(monkeypatch, ["ソノトオリデス"], ["ソノトオリデス"])
        assert check_line("その通りです", 1, None, segmentation=True) is None

    def test_disagreement_is_elevated_on_a_changed_line(self, monkeypatch):
        self._stub(monkeypatch, ["ツヨサ"], ["コワサ"])
        assert check_line("強い", 1, None, elevate=True)["priority"] == "ELEVATED"
        assert check_line("強い", 1, None, elevate=False)["priority"] == "REVIEW"

    def test_segmentation_conflict_is_a_finding_on_its_own(self, monkeypatch):
        """The flat readings agree — only the boundaries differ, and that is the bug."""
        self._stub(monkeypatch, TestBoundaryConflicts.ENG_OOKISA, TestBoundaryConflicts.OJT_OOKISA)
        f = check_line("その大きさ自体が", 1, None, segmentation=True, elevate=True)
        assert f["reason"] == "segmentation_disagreement"
        assert f["priority"] == "ELEVATED"
        assert f["boundary_conflicts"]

    def test_only_changed_lines_are_sent_to_the_engine(self, tmp_path, monkeypatch, pinned_ids):
        import dr2_podcast.audio.engine as engine
        import dr2_podcast.tools.tts_reading_check as mod

        monkeypatch.setattr(engine, "TTS_RANDOM_VOICE", False)
        calls = []

        def fake_phrases(text, speaker, session):
            calls.append(text)
            return ["ヨミ"]

        monkeypatch.setattr(mod, "engine_phrases", fake_phrases)
        monkeypatch.setattr(mod, "openjtalk_reading", lambda t: "ヨミ")
        monkeypatch.setattr(mod, "openjtalk_phrases", lambda t: ["ヨミ"])
        monkeypatch.setattr(mod, "preflight", lambda session, speaker: None)

        prev = tmp_path / "prev.txt"
        cur = tmp_path / "script.txt"
        prev.write_text("Speaker 1: あ\nSpeaker 2: い\n", encoding="utf-8")
        cur.write_text("Speaker 1: あ\nSpeaker 2: にゅう\nSpeaker 2: い\n", encoding="utf-8")

        rep = check_script(cur, baseline=prev, session=object())
        assert calls == ["にゅう"]
        assert rep["mode"] == "changed" and rep["lines_checked"] == 1 and rep["lines_total"] == 3
        assert rep["findings"][0]["speaker"] == 2 and rep["findings"][0]["voice"] == HOST2


# ---------------------------------------------------------------------------
# The check must never look clean because it never read anything
# ---------------------------------------------------------------------------


class _DeadSession:
    def get(self, *a, **kw):
        raise requests.exceptions.ConnectionError("refused")

    def post(self, *a, **kw):
        raise requests.exceptions.ConnectionError("refused")


class TestFailsLoudly:
    def test_preflight_raises_when_the_engine_is_down(self):
        with pytest.raises(EngineUnavailable):
            preflight(_DeadSession(), 1)

    def test_preflight_raises_when_the_engine_answers_with_nothing(self, monkeypatch):
        """Reachable but returning no reading would make every line look clean."""
        import dr2_podcast.tools.tts_reading_check as mod

        class _Ok:
            def raise_for_status(self):
                return None

        session = type("S", (), {"get": lambda self, *a, **kw: _Ok()})()
        monkeypatch.setattr(mod, "engine_phrases", lambda t, s, sess: [])
        with pytest.raises(EngineUnavailable):
            preflight(session, 1)

    def test_check_script_does_not_report_clean_when_the_engine_is_down(
        self, tmp_path, monkeypatch, pinned_ids
    ):
        import dr2_podcast.audio.engine as engine

        monkeypatch.setattr(engine, "TTS_RANDOM_VOICE", False)
        p = tmp_path / "script.txt"
        p.write_text("Speaker 1: あ\n", encoding="utf-8")
        with pytest.raises(EngineUnavailable):
            check_script(p, session=_DeadSession())

    def test_engine_error_on_one_line_is_counted(self, tmp_path, monkeypatch, pinned_ids):
        import dr2_podcast.audio.engine as engine
        import dr2_podcast.tools.tts_reading_check as mod

        monkeypatch.setattr(engine, "TTS_RANDOM_VOICE", False)
        monkeypatch.setattr(mod, "preflight", lambda session, speaker: None)

        def boom(text, speaker, session):
            raise requests.exceptions.Timeout("slow")

        monkeypatch.setattr(mod, "engine_phrases", boom)
        p = tmp_path / "script.txt"
        p.write_text("Speaker 1: あ\n", encoding="utf-8")
        rep = check_script(p, session=object())
        assert rep["errors"] == 1 and rep["findings"][0]["priority"] == "ERROR"

    def test_cli_exit_code_is_2_when_a_line_errored(self, tmp_path, monkeypatch, pinned_ids):
        import dr2_podcast.audio.engine as engine
        import dr2_podcast.tools.tts_reading_check as mod

        monkeypatch.setattr(engine, "TTS_RANDOM_VOICE", False)
        monkeypatch.setattr(mod, "preflight", lambda session, speaker: None)

        def boom(text, speaker, session):
            raise requests.exceptions.Timeout("slow")

        monkeypatch.setattr(mod, "engine_phrases", boom)
        p = tmp_path / "script.txt"
        p.write_text("Speaker 1: あ\n", encoding="utf-8")
        assert mod.main([str(p)]) == 2

    def test_cli_exit_code_is_2_for_a_missing_script(self, tmp_path):
        import dr2_podcast.tools.tts_reading_check as mod

        assert mod.main([str(tmp_path / "nope.txt")]) == 2
