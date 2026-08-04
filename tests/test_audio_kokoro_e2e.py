"""End-to-end tests for the Kokoro (English) path of generate_audio_from_script.

Like the AivisSpeech tests, these drive the real engine rather than a fake:
Kokoro runs in-process on CPU, so a short script renders in a few seconds.

Written before splitting that 134-statement, complexity-28 function. They also
cover the engine-selection dispatch at the top of the function, which is shared
by both languages.

Kokoro model weights are downloaded on first use; the module-level probe skips
the whole file if it cannot initialise.
"""

import wave

import pytest

from dr2_podcast.audio.engine import generate_audio_from_script


def _kokoro_up() -> bool:
    try:
        from kokoro import KPipeline

        KPipeline(lang_code="a", device="cpu")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _kokoro_up(), reason="Kokoro pipeline unavailable")


def _wav_seconds(path):
    with wave.open(str(path)) as f:
        return f.getnframes() / f.getframerate()


class TestKokoroPath:
    def test_two_speaker_script_renders(self, tmp_path):
        out = tmp_path / "a.wav"
        result = generate_audio_from_script("Speaker 1: Hello there.\nSpeaker 2: Good to be here.\n", str(out), "a")
        assert result is not None
        path, transitions = result
        assert path == str(out)
        assert out.exists() and out.stat().st_size > 1000
        assert 0.5 < _wav_seconds(out) < 30
        assert transitions == []

    def test_transition_marker_is_recorded(self, tmp_path):
        out = tmp_path / "b.wav"
        _, transitions = generate_audio_from_script(
            "Speaker 1: First point.\n\n[TRANSITION]\n\nSpeaker 2: Second point.\n", str(out), "a"
        )
        assert len(transitions) == 1
        assert transitions[0] > 0

    def test_headings_and_rules_are_not_spoken(self, tmp_path):
        plain, marked = tmp_path / "c1.wav", tmp_path / "c2.wav"
        generate_audio_from_script("Speaker 1: Hello there.\n", str(plain), "a")
        generate_audio_from_script("## Heading\n\n---\n\nSpeaker 1: Hello there.\n", str(marked), "a")
        assert _wav_seconds(marked) == pytest.approx(_wav_seconds(plain), abs=0.5)

    def test_continuation_lines_join_the_current_turn(self, tmp_path):
        one, two = tmp_path / "d1.wav", tmp_path / "d2.wav"
        generate_audio_from_script("Speaker 1: Hello there.\n", str(one), "a")
        generate_audio_from_script("Speaker 1: Hello there.\nAnd welcome along.\n", str(two), "a")
        assert _wav_seconds(two) > _wav_seconds(one) + 0.3

    def test_script_with_no_speaker_lines_produces_nothing(self, tmp_path):
        out = tmp_path / "e.wav"
        assert generate_audio_from_script("## Only a heading\n\n---\n", str(out), "a") is None
        assert not out.exists()

    def test_unknown_engine_name_returns_none(self, tmp_path, monkeypatch):
        """Engine dispatch: an unregistered name must fail loudly, not fall
        through to Kokoro."""
        from dr2_podcast.audio import engine

        monkeypatch.setattr(engine, "TTS_ENGINE_JA", "no-such-engine")
        assert generate_audio_from_script("Speaker 1: Hi.\n", str(tmp_path / "f.wav"), "j") is None

    def test_lang_code_j_dispatches_away_from_kokoro(self, tmp_path, monkeypatch):
        from dr2_podcast.audio import engine

        seen = {}

        def fake_adapter(script_text, output_filename):
            seen["called"] = (script_text, output_filename)
            return (output_filename, [])

        monkeypatch.setattr(engine, "TTS_ENGINE_JA", "fake")
        monkeypatch.setitem(engine._TTS_ENGINES, "fake", fake_adapter)
        out = tmp_path / "g.wav"
        assert generate_audio_from_script("Speaker 1: こんにちは。\n", str(out), "j") == (str(out), [])
        assert seen["called"][1] == str(out)
