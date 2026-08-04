"""Characterization tests for AudioMixer.mix_podcast_pro.

Written before collapsing its six tuning parameters into a MixSettings object.
These run the REAL mixer over synthetic WAVs — no ffmpeg needed, pydub reads
plain PCM WAV directly — so they pin the actual output duration arithmetic
rather than a mocked stand-in.

Duration is the load-bearing property: the mix is pre_roll + voice + post_roll,
and getting it wrong means the episode's BGM intro or outro is silently the
wrong length.
"""

import math
import struct
import wave

import pytest

from dr2_podcast.audio.engine import AudioMixer, MixSettings


def _write_wav(path, seconds, freq=440, rate=24000):
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            b"".join(
                struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * t / rate)))
                for t in range(int(rate * seconds))
            )
        )


def _duration_ms(path):
    with wave.open(str(path)) as f:
        return round(f.getnframes() / f.getframerate() * 1000)


@pytest.fixture
def tracks(tmp_path):
    voice = tmp_path / "voice.wav"
    music = tmp_path / "music.wav"
    _write_wav(voice, 3.0, 300)
    _write_wav(music, 2.0, 200)
    return voice, music, tmp_path / "out.wav"


class TestMixPodcastPro:
    def test_default_mix_is_preroll_plus_voice_plus_postroll(self, tracks):
        voice, music, out = tracks
        assert AudioMixer().mix_podcast_pro(str(voice), str(music), str(out)) is True
        # defaults: 4000ms pre-roll + 3000ms voice + 6000ms post-roll
        assert _duration_ms(out) == pytest.approx(13000, abs=50)

    def test_roll_lengths_are_honoured(self, tracks):
        voice, music, out = tracks
        AudioMixer().mix_podcast_pro(str(voice), str(music), str(out), MixSettings(pre_roll_ms=1000, post_roll_ms=2000))
        assert _duration_ms(out) == pytest.approx(6000, abs=50)

    def test_zero_rolls_currently_fall_back_to_the_basic_mixer(self, tracks, caplog):
        """LATENT EDGE CASE, pinned as-is rather than fixed here.

        post_roll_ms=0 makes pydub's .fade(duration=0) raise, the broad except
        catches it, and the mix silently degrades to mix_podcast(). It still
        returns True, so a caller cannot tell. Not reachable in production: the
        one call site never overrides the 4000/6000 defaults.
        """
        voice, music, out = tracks
        assert (
            AudioMixer().mix_podcast_pro(str(voice), str(music), str(out), MixSettings(pre_roll_ms=0, post_roll_ms=0))
            is True
        )
        assert "Pro mixing failed" in caplog.text
        assert "falling back to basic mix" in caplog.text

    def test_music_shorter_than_the_mix_is_looped_not_truncated(self, tracks):
        """Music is 2s and the mix is 13s — the output must still be full length."""
        voice, music, out = tracks
        AudioMixer().mix_podcast_pro(str(voice), str(music), str(out))
        assert _duration_ms(out) == pytest.approx(13000, abs=50)

    def test_transition_bumps_do_not_change_duration(self, tracks):
        voice, music, out = tracks
        AudioMixer().mix_podcast_pro(
            str(voice), str(music), str(out), MixSettings(transition_positions_ms=[500, 1500, 2500])
        )
        assert _duration_ms(out) == pytest.approx(13000, abs=50)

    def test_transition_position_past_the_end_is_tolerated(self, tracks):
        voice, music, out = tracks
        assert (
            AudioMixer().mix_podcast_pro(
                str(voice), str(music), str(out), MixSettings(transition_positions_ms=[999_000])
            )
            is True
        )

    def test_ducking_level_changes_the_audio_but_not_the_length(self, tracks):
        voice, music, out = tracks
        AudioMixer().mix_podcast_pro(str(voice), str(music), str(out), MixSettings(voice_ducking_db=-30))
        loud = out.read_bytes()
        AudioMixer().mix_podcast_pro(str(voice), str(music), str(out), MixSettings(voice_ducking_db=-5))
        quiet = out.read_bytes()
        assert len(loud) == len(quiet)
        assert loud != quiet, "ducking level had no audible effect"

    def test_missing_voice_file_returns_false(self, tmp_path, tracks):
        _, music, out = tracks
        assert AudioMixer().mix_podcast_pro(str(tmp_path / "nope.wav"), str(music), str(out)) is False

    def test_missing_music_file_returns_false(self, tmp_path, tracks):
        voice, _, out = tracks
        assert AudioMixer().mix_podcast_pro(str(voice), str(tmp_path / "nope.wav"), str(out)) is False

    def test_output_keeps_the_source_sample_rate(self, tracks):
        voice, music, out = tracks
        AudioMixer().mix_podcast_pro(str(voice), str(music), str(out))
        with wave.open(str(out)) as f:
            assert f.getframerate() == 24000
