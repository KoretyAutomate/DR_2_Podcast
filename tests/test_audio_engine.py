"""Tests for dr2_podcast/audio/engine.py -- TTS cleaning, chunking, and mixing."""

import pytest
import re
import numpy as np
from unittest.mock import patch, MagicMock

from dr2_podcast.audio.engine import clean_script_for_tts, _chunk_japanese_text, MARKER_SILENCE, _TARGET_RMS


# ---------------------------------------------------------------------------
# clean_script_for_tts
# ---------------------------------------------------------------------------

class TestCleanScriptForTTS:

    def test_strips_cue_lines(self):
        text = "## [intrigued, leaning in]\nHost 1: Welcome."
        result = clean_script_for_tts(text)
        assert "[intrigued, leaning in]" not in result
        assert "Welcome" in result

    def test_preserves_transition(self):
        text = "Host 1: Before.\n[TRANSITION]\nHost 2: After."
        result = clean_script_for_tts(text)
        assert "[TRANSITION]" in result

    def test_preserves_pause(self):
        text = "Host 1: Wait...\n[PAUSE]\nHost 1: Okay."
        result = clean_script_for_tts(text)
        assert "[PAUSE]" in result

    def test_preserves_beat(self):
        text = "[BEAT]\nHost 1: Then..."
        result = clean_script_for_tts(text)
        assert "[BEAT]" in result

    def test_strips_bold_markdown(self):
        text = "Host 1: This is **important** evidence."
        result = clean_script_for_tts(text)
        assert "**" not in result
        assert "important" in result

    def test_renames_host_to_speaker(self):
        text = "Host 1: Hello.\nHost 2: Hi there."
        result = clean_script_for_tts(text)
        assert "Speaker 1:" in result
        assert "Speaker 2:" in result
        assert "Host 1:" not in result

    def test_strips_think_blocks(self):
        text = "<think>internal thoughts</think>\nHost 1: Visible line."
        result = clean_script_for_tts(text)
        assert "internal thoughts" not in result
        assert "Visible line" in result


# ---------------------------------------------------------------------------
# _chunk_japanese_text
# ---------------------------------------------------------------------------

class TestChunkJapaneseText:

    def test_splits_at_punctuation(self):
        text = "first sentence here.second one."
        # Use Japanese punctuation
        text_jp = "first sentence here\u3002second one\u3002"
        chunks = _chunk_japanese_text(text_jp, max_chars=80)
        assert len(chunks) >= 1

    def test_short_text_single_chunk(self):
        text = "short"
        chunks = _chunk_japanese_text(text, max_chars=80)
        assert len(chunks) == 1
        assert chunks[0] == "short"

    def test_long_text_no_punctuation(self):
        """Without punctuation, text stays as a single chunk (no split points)."""
        text = "a" * 200
        chunks = _chunk_japanese_text(text, max_chars=80)
        # No punctuation means no split points -- single chunk returned
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_string(self):
        chunks = _chunk_japanese_text("", max_chars=80)
        assert chunks == []

    def test_multiple_sentences_merged(self):
        # Two short sentences that together fit under max_chars
        text = "a\u3002b\u3002"
        chunks = _chunk_japanese_text(text, max_chars=80)
        # Should merge into single chunk since combined < 80
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# AudioMixer
# ---------------------------------------------------------------------------

class TestAudioMixer:

    def test_mix_podcast_basic(self):
        from dr2_podcast.audio.engine import AudioMixer

        mixer = AudioMixer()
        mock_voice = MagicMock()
        mock_voice.__len__ = lambda self: 10000
        mock_music = MagicMock()
        mock_music.__len__ = lambda self: 5000
        mock_music.__mul__ = lambda self, n: mock_music
        mock_music.__getitem__ = lambda self, s: mock_music
        mock_music.__sub__ = lambda self, n: mock_music
        mock_music.overlay = MagicMock(return_value=mock_music)
        mock_music.fade_in = MagicMock(return_value=mock_music)
        mock_music.fade_out = MagicMock(return_value=mock_music)
        mock_music.export = MagicMock()

        with patch("dr2_podcast.audio.engine.AudioSegment") as MockAS, \
             patch("dr2_podcast.audio.engine.effects") as mock_effects:
            MockAS.from_wav.side_effect = [mock_voice, mock_music]
            mock_effects.normalize.side_effect = lambda x: x
            result = mixer.mix_podcast("voice.wav", "music.wav", "out.wav")

        assert result is True


# ---------------------------------------------------------------------------
# Per-speaker RMS normalization logic
# ---------------------------------------------------------------------------

class TestRMSNormalization:
    """Test the RMS normalization math used in both TTS _flush_buffer paths."""

    @staticmethod
    def _normalize(audio: np.ndarray) -> np.ndarray:
        """Replicate the normalization logic from engine.py."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 1e-6:
            audio = audio * (_TARGET_RMS / rms)
            audio = np.clip(audio, -1.0, 1.0)
        return audio

    def test_known_rms_normalized_to_target(self):
        # Create array with known RMS of 0.02
        audio = np.full(1000, 0.02, dtype=np.float32)
        result = self._normalize(audio)
        result_rms = np.sqrt(np.mean(result ** 2))
        assert abs(result_rms - _TARGET_RMS) < 0.001

    def test_zero_array_skipped(self):
        audio = np.zeros(1000, dtype=np.float32)
        result = self._normalize(audio)
        assert np.all(result == 0.0)

    def test_very_quiet_array_clipped(self):
        # RMS=0.001 needs 80x boost — peaks will clip to [-1.0, 1.0]
        rng = np.random.default_rng(42)
        audio = rng.normal(0, 0.001, 10000).astype(np.float32)
        result = self._normalize(audio)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_already_at_target_unchanged(self):
        audio = np.full(1000, _TARGET_RMS, dtype=np.float32)
        result = self._normalize(audio)
        result_rms = np.sqrt(np.mean(result ** 2))
        assert abs(result_rms - _TARGET_RMS) < 0.001
