"""End-to-end test for the AivisSpeech generator, against the real engine.

_generate_audio_aivisspeech is I/O-heavy — it parses a script, calls the
AivisSpeech HTTP API per chunk, and concatenates the result — so a golden over
a fake does not prove much. This drives the real Docker engine on :10101 with a
deliberately tiny script, and skips when it is not running.

Written before splitting that 128-statement, complexity-26 function, so the
split can be shown to preserve what the function actually produces: a WAV whose
duration is plausible, the transition-marker positions, and the speaker mapping.
"""

import wave

import httpx
import pytest

from dr2_podcast.audio.engine import _generate_audio_aivisspeech, _get_tts_speaker_ids_int

# The engine matches "Speaker N:" only — clean_script_for_tts is what rewrites
# the pipeline's "Host N:"/"ホストN:" labels into this form. Fixtures use the
# post-clean format because that is what reaches the generator.


def _aivisspeech_up() -> bool:
    try:
        return httpx.get("http://localhost:10101/version", timeout=3).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _aivisspeech_up(), reason="AivisSpeech (localhost:10101) is not running")


SHORT_SCRIPT = """\
Speaker 1: こんにちは。

Speaker 2: よろしくお願いします。
"""

SCRIPT_WITH_TRANSITION = """\
Speaker 1: 最初の話です。

[TRANSITION]

Speaker 2: 次の話に移ります。
"""


def _wav_seconds(path):
    with wave.open(str(path)) as f:
        return f.getnframes() / f.getframerate()


class TestGenerateAudioAivisspeech:
    def test_speaker_ids_are_configured(self):
        h1, h2 = _get_tts_speaker_ids_int()
        assert h1 is not None and h2 is not None

    def test_two_speaker_script_produces_a_playable_wav(self, tmp_path):
        out = tmp_path / "a.wav"
        result = _generate_audio_aivisspeech(SHORT_SCRIPT, str(out))
        assert result is not None, "generation returned None"
        path, transitions = result
        assert path == str(out)
        assert out.exists() and out.stat().st_size > 1000
        assert 0.5 < _wav_seconds(out) < 30, "duration is implausible for two short lines"
        assert transitions == []

    def test_transition_marker_records_a_position(self, tmp_path):
        out = tmp_path / "b.wav"
        result = _generate_audio_aivisspeech(SCRIPT_WITH_TRANSITION, str(out))
        assert result is not None
        _, transitions = result
        assert len(transitions) == 1
        assert transitions[0] > 0, "the marker must land after the first speaker's audio"

    def test_same_script_renders_to_the_same_length(self, tmp_path):
        """Voice swap is seeded off the script, so the same script picks the same
        two voices every time.

        Note the engine is NOT byte-deterministic — Style-Bert-VITS2 sampling
        varies run to run — so this asserts duration, not bytes. The assignment
        itself is pinned exactly in test_audio_engine's unit test of the seed.
        """
        a, b = tmp_path / "c1.wav", tmp_path / "c2.wav"
        _generate_audio_aivisspeech(SHORT_SCRIPT, str(a))
        _generate_audio_aivisspeech(SHORT_SCRIPT, str(b))
        assert _wav_seconds(a) == pytest.approx(_wav_seconds(b), abs=0.5)

    def test_script_with_no_speaker_lines_produces_nothing(self, tmp_path):
        out = tmp_path / "d.wav"
        assert _generate_audio_aivisspeech("## Just a heading\n\n---\n", str(out)) is None
        assert not out.exists()

    def test_continuation_lines_join_the_current_speaker_turn(self, tmp_path):
        """An unlabeled line after a speaker line belongs to that turn, and the
        final buffered turn must still be flushed at end of script."""
        one = tmp_path / "f1.wav"
        two = tmp_path / "f2.wav"
        _generate_audio_aivisspeech("Speaker 1: こんにちは。\n", str(one))
        _generate_audio_aivisspeech("Speaker 1: こんにちは。\nそしてこんばんは。\n", str(two))
        assert _wav_seconds(two) > _wav_seconds(one) + 0.3

    def test_headings_and_rules_are_not_spoken(self, tmp_path):
        """Markup around one line must not add speech.

        Tolerance is 0.5s: synthesis is not byte- or duration-deterministic, but
        speaking the heading would add well over a second.
        """
        plain = tmp_path / "e1.wav"
        marked = tmp_path / "e2.wav"
        _generate_audio_aivisspeech("Speaker 1: こんにちは。\n", str(plain))
        _generate_audio_aivisspeech("## 見出し\n\n---\n\nSpeaker 1: こんにちは。\n", str(marked))
        assert _wav_seconds(marked) == pytest.approx(_wav_seconds(plain), abs=0.5)
