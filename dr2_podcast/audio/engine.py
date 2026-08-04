"""
TTS Audio Engine for Deep Research Podcast
===========================================

Generates high-quality, multi-speaker podcast audio with automatic TTS engine selection:

  - English:  Kokoro TTS (local, CPU, proven quality)
  - Japanese: AivisSpeech (Docker container, Style-Bert-VITS2, natural/emotional)

Features:
- Dual-voice system with speaker detection
- Automatic language routing (lang_code='a' → Kokoro, 'j' → AivisSpeech)
- Script parsing and audio stitching
- WAV export with BGM support
"""

import logging
from dataclasses import dataclass

import soundfile as sf
from kokoro import KPipeline
import torch
import numpy as np
import re
import os
import json
import random
import hashlib
from pathlib import Path
from dr2_podcast.utils import strip_think_blocks
from dr2_podcast.config import (
    TTS_API_URL,
    TTS_ENGINE_EN,
    TTS_ENGINE_JA,
    TTS_HOST1_ID,
    TTS_HOST2_ID,
    TTS_RANDOM_VOICE,
    TTS_SPEED_SCALE,
    TTS_SPEED_OVERRIDES,
    TTS_INTONATION_SCALE,
    TTS_INTONATION_OVERRIDES,
)

from pydub import AudioSegment, effects

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AudioMixer — merged from audio_mixer.py (T4.1)
# ---------------------------------------------------------------------------


@dataclass
class MixSettings:
    """Tuning knobs for mix_podcast_pro.

    NOTE: pre_roll_ms=0 or post_roll_ms=0 makes pydub's .fade(duration=0)
    raise, and mix_podcast_pro's except falls back to the basic mixer while
    still returning True. Production never sets them to zero.
    """

    pre_roll_ms: int = 4000
    post_roll_ms: int = 6000
    transition_positions_ms: list | None = None
    voice_ducking_db: int = -20
    transition_bump_db: int = -10
    transition_duration_ms: int = 1500


class AudioMixer:
    """
    Mixes voice and background music with ducking capabilities.
    """

    def __init__(self):
        pass

    def mix_podcast(self, voice_path: str, music_path: str, output_path: str) -> bool:
        """
        Mixes voice and background music. Loop music and duck it.
        """
        try:
            logger.info(f"Mixing voice: {voice_path} with music: {music_path}")

            voice = AudioSegment.from_wav(voice_path)
            music = AudioSegment.from_wav(music_path)

            # 1. Loop music to match or exceed voice duration
            if len(music) < len(voice):
                repeats = (len(voice) // len(music)) + 1
                music = music * repeats

            # Trim music to exact length of voice (plus maybe a small fade out tail)
            music = music[: len(voice) + 2000]  # + 2s tail

            # 2. Lower volume of music ("ducking")
            voice = effects.normalize(voice)
            music = effects.normalize(music)

            # Reduce music volume significantly
            music = music - 20  # Reduce by 20dB

            # 3. Overlay
            final_mix = music.overlay(voice, position=0)

            # 4. Fade in/out music
            final_mix = final_mix.fade_in(2000).fade_out(3000)

            # Export
            final_mix.export(output_path, format="wav")
            logger.info(f"Mixed audio saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to mix audio: {e}")
            return False

    def mix_podcast_pro(
        self,
        voice_path: str,
        music_path: str,
        output_path: str,
        settings: "MixSettings | None" = None,
    ) -> bool:
        """
        Pro-grade podcast mixing with BGM-only intro/outro and transition bumps.

        Sections:
        1. PRE-ROLL: BGM only at full volume, fading down to ducked level
        2. VOICE+BGM: Main content with BGM ducked
        3. TRANSITIONS: Brief BGM volume bumps at marked positions
        4. POST-ROLL: BGM fading up from ducked to full, then fading out
        """
        s = settings or MixSettings()
        pre_roll_ms = s.pre_roll_ms
        post_roll_ms = s.post_roll_ms
        transition_positions_ms = s.transition_positions_ms
        voice_ducking_db = s.voice_ducking_db
        transition_bump_db = s.transition_bump_db
        transition_duration_ms = s.transition_duration_ms

        try:
            logger.info(f"Pro mixing: {voice_path} with {music_path}")
            logger.info(f"  Pre-roll: {pre_roll_ms}ms, Post-roll: {post_roll_ms}ms")
            if transition_positions_ms:
                logger.info(f"  Transition bumps at: {transition_positions_ms}")

            voice = AudioSegment.from_wav(voice_path)
            music = AudioSegment.from_wav(music_path)

            total_duration = len(voice) + pre_roll_ms + post_roll_ms

            # Loop and trim music to cover total duration
            if len(music) < total_duration + 2000:
                repeats = ((total_duration + 2000) // len(music)) + 1
                music = music * repeats
            music = music[: total_duration + 2000]

            # Normalize both tracks
            voice = effects.normalize(voice)
            music = effects.normalize(music)

            # --- 1. PRE-ROLL: BGM only, fading from full volume down to ducked ---
            pre_roll = music[:pre_roll_ms].fade(to_gain=voice_ducking_db, start=0, duration=pre_roll_ms)

            # --- 2. MAIN SECTION: ducked BGM under voice ---
            main_music = music[pre_roll_ms : pre_roll_ms + len(voice)] + voice_ducking_db

            # --- 3. TRANSITION BUMPS: brief volume increases at marked positions ---
            if transition_positions_ms:
                for pos_ms in transition_positions_ms:
                    bump_start = pos_ms
                    bump_end = min(bump_start + transition_duration_ms, len(main_music))
                    if 0 <= bump_start < len(main_music):
                        bump_gain = abs(voice_ducking_db - transition_bump_db)
                        bump_section = main_music[bump_start:bump_end] + bump_gain
                        bump_section = bump_section.fade_in(300).fade_out(300)
                        main_music = main_music[:bump_start] + bump_section + main_music[bump_end:]

            # --- 4. POST-ROLL: BGM fading up from ducked to full, then fading out ---
            post_start = pre_roll_ms + len(voice)
            post_music = music[post_start : post_start + post_roll_ms] + voice_ducking_db
            # Fade up from ducked to full volume over first half
            fade_up_duration = min(post_roll_ms // 2, 3000)
            post_music = post_music.fade(from_gain=0, to_gain=abs(voice_ducking_db), start=0, duration=fade_up_duration)
            # Fade out over second half
            post_music = post_music.fade_out(post_roll_ms - fade_up_duration)

            # --- ASSEMBLE: pre-roll + (main BGM overlaid with voice) + post-roll ---
            silence_pre = AudioSegment.silent(duration=pre_roll_ms, frame_rate=voice.frame_rate)
            silence_post = AudioSegment.silent(duration=post_roll_ms, frame_rate=voice.frame_rate)
            full_voice = silence_pre + voice + silence_post

            full_bgm = pre_roll + main_music + post_music

            # Ensure same length (trim to shorter)
            min_len = min(len(full_bgm), len(full_voice))
            full_bgm = full_bgm[:min_len]
            full_voice = full_voice[:min_len]

            # Overlay voice on top of BGM
            final_mix = full_bgm.overlay(full_voice)

            # Gentle global fade in/out
            final_mix = final_mix.fade_in(1500).fade_out(2000)

            final_mix.export(output_path, format="wav")
            logger.info(f"Pro-mixed audio saved to: {output_path}")
            logger.info(
                f"  Total duration: {len(final_mix) / 1000:.1f}s "
                f"(pre-roll {pre_roll_ms / 1000:.1f}s + voice {len(voice) / 1000:.1f}s + "
                f"post-roll {post_roll_ms / 1000:.1f}s)"
            )
            return True

        except Exception as e:
            logger.error(f"Pro mixing failed: {e}, falling back to basic mix")
            return self.mix_podcast(voice_path, music_path, output_path)


# Voice Configuration
VOICE_HOST_1 = "am_fenrir"  # American Male (The Expert) - default English
VOICE_HOST_2 = "af_heart"  # American Female (The Skeptic) - default English
LANG_CODE = "a"  # American English (default)

# Per-speaker RMS normalization target (~-22dBFS)
# Controls inter-speaker volume balance; downstream pydub normalize handles absolute level
_TARGET_RMS = 0.08

# Per-language voice mapping for Kokoro TTS
VOICE_MAP = {
    "a": {"host1": "am_fenrir", "host2": "af_heart"},  # English
    "j": {"host1": "jm_kumo", "host2": "jf_alpha"},  # Japanese
}

# ---------------------------------------------------------------------------
# AivisSpeech Adapter (Japanese TTS, HTTP/Docker)
# ---------------------------------------------------------------------------
# AivisSpeech-Engine is API-compatible with VOICEVOX (same /audio_query → /synthesis,
# /version). Default port 10101. Voices are Style-Bert-VITS2 models (natural/emotional).
# URL aliased to preserve existing call-site names; single source of truth is config.TTS_API_URL.
_AIVISSPEECH_API_URL = TTS_API_URL


def _get_tts_speaker_ids_int():
    """Speaker (style) IDs as integers — for engines that use numeric IDs (AivisSpeech, VOICEVOX et al.).
    Defaults: 1937616896 (にせ ノーマル) and 1717361472 (みちのくあいり 標準).
    Returns (None, None) if the configured IDs are not numeric (e.g. cloud-TTS voice names)."""
    try:
        return int(TTS_HOST1_ID), int(TTS_HOST2_ID)
    except (TypeError, ValueError):
        logger.error(
            f"TTS_HOST1_ID/TTS_HOST2_ID must be integer style IDs for this engine "
            f"(got {TTS_HOST1_ID!r}, {TTS_HOST2_ID!r})"
        )
        return None, None


def _aivisspeech_available():
    """Check if the AivisSpeech engine is reachable."""
    try:
        import requests

        resp = requests.get(f"{_AIVISSPEECH_API_URL}/version", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _call_aivisspeech_segment(
    text: str, speaker_id: int, speed_scale: float = None, intonation_scale: float = None
) -> tuple:
    """Call AivisSpeech API (two-step: audio_query → synthesis). Returns (audio_np, sample_rate) or (None, None).

    speed_scale overrides the global TTS_SPEED_SCALE for this segment (per-voice cadence).
    intonation_scale likewise overrides TTS_INTONATION_SCALE (per-voice pitch swing)."""
    try:
        import requests
        import io as _io
    except ImportError as e:
        logger.error(f"Missing dependency for AivisSpeech: {e}")
        return None, None

    try:
        # Step 1: Create audio query
        q_resp = requests.post(
            f"{_AIVISSPEECH_API_URL}/audio_query",
            params={"text": text, "speaker": speaker_id},
            timeout=30,
        )
        q_resp.raise_for_status()
        query_data = q_resp.json()
        query_data["speedScale"] = TTS_SPEED_SCALE if speed_scale is None else speed_scale
        query_data["intonationScale"] = TTS_INTONATION_SCALE if intonation_scale is None else intonation_scale

        # Step 2: Synthesize audio
        synth_resp = requests.post(
            f"{_AIVISSPEECH_API_URL}/synthesis",
            params={"speaker": speaker_id},
            json=query_data,
            timeout=60,
        )
        synth_resp.raise_for_status()

        audio, sr = sf.read(_io.BytesIO(synth_resp.content))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), sr
    except requests.exceptions.ConnectionError:
        logger.error(
            f"AivisSpeech API unreachable at {_AIVISSPEECH_API_URL}. "
            f"Start it: docker run -d --name aivisspeech -p 10101:10101 ghcr.io/aivis-project/aivisspeech-engine:cpu-latest"
        )
        return None, None
    except Exception as e:
        logger.error(f"AivisSpeech API error: {e}")
        return None, None


def assign_seeded_voices(script_text: str, host1_id: int, host2_id: int) -> tuple:
    """Pick which configured voice speaks Speaker 1, seeded off the script.

    Deterministic per episode (same script → same voices) but varies across
    episodes, so the explainer (Speaker 1) is not always Host 1. Only the two
    configured voices are ever used. Disable with TTS_RANDOM_VOICE=0.

    Returns (host1_id, host2_id, swapped).
    """
    if not TTS_RANDOM_VOICE or host1_id == host2_id:
        return host1_id, host2_id, False
    seed = int(hashlib.sha256(script_text.encode("utf-8")).hexdigest(), 16)
    if seed & 1:
        return host2_id, host1_id, True
    return host1_id, host2_id, False


def _marker_silence(marker: str, sample_rate, cumulative_samples: int):
    """Silence for a pause marker, plus its timeline position if it is a transition.

    Returns (silence_array, position_ms) where position_ms is None for markers
    other than [TRANSITION].
    """
    sr = sample_rate or 24000
    silence = np.zeros(int(MARKER_SILENCE[marker] * sr), dtype=np.float32)
    position_ms = None
    if marker == "[TRANSITION]":
        position_ms = int((cumulative_samples / sr) * 1000)
        logger.info(f"  [TRANSITION] marker at {position_ms}ms")
    return silence, position_ms


def _aivisspeech_preflight(host1_id: int, host2_id: int, voice_swapped: bool) -> bool:
    """Log the run banner and confirm the engine is reachable."""
    logger.info("=" * 60)
    logger.info("AIVISSPEECH — JAPANESE AUDIO GENERATION")
    logger.info("=" * 60)
    logger.info(f"API endpoint: {_AIVISSPEECH_API_URL}")
    rand_note = (
        f" [random-voice: {'SWAPPED' if voice_swapped else 'no-swap'}]" if TTS_RANDOM_VOICE else " [random-voice: off]"
    )
    logger.info(f"Voices: Speaker1 → speaker_id={host1_id}, Speaker2 → speaker_id={host2_id}{rand_note}")

    if not _aivisspeech_available():
        logger.error(
            f"✗ AivisSpeech API not reachable at {_AIVISSPEECH_API_URL}\n"
            f"  Start it: docker run -d --name aivisspeech -p 10101:10101 ghcr.io/aivis-project/aivisspeech-engine:cpu-latest"
        )
        return False

    logger.info("✓ AivisSpeech API is healthy")
    return True


def _synthesize_turn(text: str, spk_id: int, sample_rate):
    """Synthesize one speaker turn via AivisSpeech, RMS-normalised.

    Returns (audio, sample_rate), or (None, sample_rate) if every chunk failed.
    A chunk that fails individually becomes silence proportional to its length
    rather than dropping the words silently from the timeline.
    """
    chunks = _chunk_japanese_text(text)
    spk_speed = TTS_SPEED_OVERRIDES.get(spk_id, TTS_SPEED_SCALE)
    spk_inton = TTS_INTONATION_OVERRIDES.get(spk_id, TTS_INTONATION_SCALE)

    chunk_audios = []
    sr_chunk = None
    for chunk in chunks:
        a, sr = _call_aivisspeech_segment(chunk, spk_id, spk_speed, spk_inton)
        if a is not None:
            sr_chunk = sr
            chunk_audios.append(a)
        else:
            sr_fallback = sample_rate or 24000
            silence_secs = max(0.5, len(chunk) / 8.0)
            chunk_audios.append(np.zeros(int(silence_secs * sr_fallback), dtype=np.float32))
            logger.warning(f"  Chunk failed — inserted {silence_secs:.1f}s silence")

    if not chunk_audios:
        return None, sample_rate

    segment_audio = np.concatenate(chunk_audios)
    rms = np.sqrt(np.mean(segment_audio**2))
    if rms > 1e-6:
        segment_audio = np.clip(segment_audio * (_TARGET_RMS / rms), -1.0, 1.0)
    return segment_audio, (sample_rate if sample_rate is not None else (sr_chunk or 24000))


def _write_generated_audio(audio_segments: list, sample_rate, output_filename: str, engine_label: str) -> bool:
    """Concatenate segments and write the WAV. False if there is nothing to write."""
    if not (audio_segments and sample_rate):
        logger.error("✗ ERROR: No audio segments generated")
        return False
    try:
        final_audio = np.concatenate(audio_segments)
        sf.write(output_filename, final_audio, sample_rate)

        file_size = Path(output_filename).stat().st_size
        duration_sec = len(final_audio) / sample_rate

        logger.info(f"\n✓ Audio generated successfully ({engine_label}):")
        logger.info(f"  File: {output_filename}")
        logger.info(f"  Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        logger.info(f"  Duration: {duration_sec / 60:.2f} minutes ({duration_sec:.1f} seconds)")
        logger.info(f"  Sample rate: {sample_rate} Hz")
        logger.info("=" * 60 + "\n")
        return True
    except Exception as e:
        logger.error(f"✗ ERROR: Failed to save audio: {e}")
        return False


class _AivisTimeline:
    """Accumulates synthesized segments, silences and transition positions.

    Holds the state the generation loop used to carry as `nonlocal` locals:
    the segment list, the sample rate (discovered from the first successful
    chunk), the running sample count, and the transition marker positions.
    """

    def __init__(self, host1_id: int, host2_id: int):
        self.host1_id = host1_id
        self.host2_id = host2_id
        self.segments = []
        self.sample_rate = None
        self.transition_positions_ms = []
        self.cumulative_samples = 0
        self.segment_count = 0

    def flush_turn(self, text: str, speaker) -> None:
        """Synthesize a buffered speaker turn and append it."""
        if not (text and speaker):
            return
        logger.info(f"  Segment {self.segment_count + 1} (Speaker {speaker}): {text[:50]}...")
        spk_id = self.host1_id if speaker == 1 else self.host2_id
        segment_audio, seg_rate = _synthesize_turn(text, spk_id, self.sample_rate)
        if segment_audio is None:
            logger.warning(f"  ⚠ Segment {self.segment_count + 1} failed — skipping")
            return
        if self.sample_rate is None:
            self.sample_rate = seg_rate
            logger.info(f"  Sample rate: {self.sample_rate} Hz")
        self._append(segment_audio)
        self.segment_count += 1

    def add_marker(self, marker: str) -> None:
        silence, position_ms = _marker_silence(marker, self.sample_rate, self.cumulative_samples)
        if position_ms is not None:
            self.transition_positions_ms.append(position_ms)
        self._append(silence)

    def add_speaker_gap(self) -> None:
        """0.3s between different speakers — only once the rate is known."""
        if self.sample_rate:
            self._append(np.zeros(int(0.3 * self.sample_rate), dtype=np.float32))

    def _append(self, audio) -> None:
        self.segments.append(audio)
        self.cumulative_samples += len(audio)


def _generate_audio_aivisspeech(script_text: str, output_filename: str) -> str:
    """
    Japanese TTS via AivisSpeech API.

    Parses multi-speaker script and calls the AivisSpeech REST API for synthesis.
    AivisSpeech (Style-Bert-VITS2) provides natural, emotional Japanese speech and
    accurate kanji reading via OpenJTalk, with a VOICEVOX-compatible HTTP API.
    """
    host1_id, host2_id = _get_tts_speaker_ids_int()
    if host1_id is None or host2_id is None:
        return None

    host1_id, host2_id, voice_swapped = assign_seeded_voices(script_text, host1_id, host2_id)
    if not _aivisspeech_preflight(host1_id, host2_id, voice_swapped):
        return None

    # Parse script — strict Speaker N: pattern only
    speaker_pattern = re.compile(r"^(Speaker\s*(\d+))\s*[:：]\s*(.*)", re.IGNORECASE)
    speaker_map = {}
    tl = _AivisTimeline(host1_id, host2_id)

    current_speaker = None
    buffer_text = ""

    for raw_line in script_text.split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("##") or re.match(r"^-{3,}$", line):
            continue

        if line in MARKER_SILENCE:
            tl.flush_turn(buffer_text, current_speaker)
            buffer_text = ""
            tl.add_marker(line)
            continue

        match = speaker_pattern.match(line)
        if match:
            name = match.group(1).strip()
            if name not in speaker_map:
                speaker_map[name] = int(match.group(2))
                logger.info(f"  Speaker detected: '{name}' → Host {speaker_map[name]}")
            new_speaker = speaker_map[name]

            tl.flush_turn(buffer_text, current_speaker)
            buffer_text = ""
            if current_speaker is not None and current_speaker != new_speaker:
                tl.add_speaker_gap()

            current_speaker = new_speaker
            buffer_text = match.group(3).strip()
        elif current_speaker is None:
            # Unlabeled leading prose is the channel intro; spoken by Speaker 2.
            if line and not line.startswith("["):
                logger.info(f"  Channel intro (Speaker 2): {line[:60]}...")
                tl.flush_turn(line, 2)
            else:
                logger.debug(f"  Skipping unlabeled line: {line[:60]}...")
        else:
            buffer_text = f"{buffer_text} {line}".strip()

    tl.flush_turn(buffer_text, current_speaker)

    logger.info(f"Generated {tl.segment_count} audio segments")
    if tl.transition_positions_ms:
        logger.info(f"Transition positions: {tl.transition_positions_ms}")

    if not _write_generated_audio(tl.segments, tl.sample_rate, output_filename, "AivisSpeech"):
        return None
    return (output_filename, tl.transition_positions_ms)


# ---------------------------------------------------------------------------
# TTS Engine Registry — maps engine name → adapter fn(script_text, output_filename)
# Adapters must accept (str, str) and return the output path (or None on failure).
# To add a new engine: implement _generate_audio_<name> above and add an entry here.
# "kokoro" is not in this dict — it runs inline in generate_audio_from_script because
# it shares the function's local state (chunking, mixing) and is the EN default.
# ---------------------------------------------------------------------------
_TTS_ENGINES = {
    "aivisspeech": _generate_audio_aivisspeech,
}


def _chunk_japanese_text(text: str, max_chars: int = 80) -> list:
    """Split Japanese text at sentence-end punctuation to keep each TTS call under max_chars."""
    sentences = re.split(r"(?<=[。！？\n])", text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) > max_chars and current:
            chunks.append(current.strip())
            current = s
        else:
            current += s
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def generate_audio_from_script(script_text: str, output_filename: str = "final_podcast.wav", lang_code: str = "a"):
    """
    Parses a script looking for 'Speaker 1:' and 'Speaker 2:' lines,
    generates audio segments, and stitches them together.

    TTS Engine selection:
      - English (lang_code='a'): Kokoro TTS (local, CPU, proven)
      - Japanese (lang_code='j'): AivisSpeech API (Docker container)

    Args:
        script_text: Full podcast script with "Host 1:" / "Host 2:" labels (renamed to "Speaker N:" before parsing)
        output_filename: Output WAV file name (default: "final_podcast.wav")
        lang_code: Language code ('a' for English, 'j' for Japanese, etc.)

    Returns:
        Tuple of (path_to_audio_file, transition_positions_ms) or None if failed.
        transition_positions_ms is a list of millisecond positions where [TRANSITION]
        markers were found, used by the pro mixer for BGM volume bumps.

    Example Script Format (input uses Host N:, cleaned to Speaker N: internally):
        Host 1: Welcome to the show. Today we're discussing coffee.
        Host 2: But is coffee actually good for you? Let's examine the evidence.
        [TRANSITION]
        Host 1: Studies show that moderate coffee intake...
    """
    # Select engine by language. Engine registry dispatches HTTP-based engines;
    # Kokoro runs inline below (in-process, shares this function's state).
    engine_name = TTS_ENGINE_JA if lang_code == "j" else TTS_ENGINE_EN
    logger.info(f"Selected TTS engine: {engine_name} (lang_code={lang_code})")

    if engine_name != "kokoro":
        adapter = _TTS_ENGINES.get(engine_name)
        if adapter is None:
            logger.error(f"Unknown TTS engine '{engine_name}'. Registered: {sorted(_TTS_ENGINES.keys()) + ['kokoro']}")
            return None
        return adapter(script_text, output_filename)

    return _generate_audio_kokoro(script_text, output_filename, lang_code)


def _init_kokoro_pipeline(lang_code: str):
    """Build the Kokoro pipeline, falling back to CPU if CUDA cannot run.

    Returns the pipeline, or None if it cannot be initialised at all.
    """
    device = "cpu"
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            device = "cuda"
        except RuntimeError:
            logger.warning("  CUDA reported available but kernel execution failed, falling back to CPU")
    logger.info(f"Device: {device}")
    logger.info(f"Language code: {lang_code}")

    try:
        pipeline = KPipeline(lang_code=lang_code, device=device)
        logger.info("✓ Kokoro pipeline initialized")
        return pipeline
    except RuntimeError as e:
        if "CUDA" in str(e) and device == "cuda":
            logger.warning(f"  CUDA init failed, retrying on CPU: {e}")
            pipeline = KPipeline(lang_code=lang_code, device="cpu")
            logger.info("✓ Kokoro pipeline initialized (CPU fallback)")
            return pipeline
        logger.error(f"✗ ERROR: Failed to initialize Kokoro: {e}")
        return None
    except Exception as e:
        logger.error(f"✗ ERROR: Failed to initialize Kokoro: {e}")
        return None


class _KokoroTimeline:
    """Kokoro counterpart of _AivisTimeline. Sample rate is fixed at 24 kHz."""

    SAMPLE_RATE = 24000

    def __init__(self, pipeline, voice_host_1, voice_host_2):
        self.pipeline = pipeline
        self.voice_host_1 = voice_host_1
        self.voice_host_2 = voice_host_2
        self.segments = []
        self.transition_positions_ms = []
        self.cumulative_samples = 0
        self.segment_count = 0

    def flush_turn(self, text: str, speaker) -> None:
        if not (text and speaker):
            return
        voice = self.voice_host_1 if speaker == 1 else self.voice_host_2
        try:
            chunk_list = []
            for _, _, audio in self.pipeline(text, voice=voice, speed=1.0, split_pattern=r"\n+"):
                chunk_list.append(audio)
                self.segment_count += 1
            if chunk_list:
                segment_audio = np.concatenate(chunk_list)
                # Per-speaker RMS normalization
                rms = np.sqrt(np.mean(segment_audio**2))
                if rms > 1e-6:
                    segment_audio = np.clip(segment_audio * (_TARGET_RMS / rms), -1.0, 1.0)
                self._append(segment_audio)
        except Exception as e:
            logger.warning(f"  ⚠ Warning: Failed to generate segment {self.segment_count}: {e}")

    def add_marker(self, marker: str) -> None:
        silence, position_ms = _marker_silence(marker, self.SAMPLE_RATE, self.cumulative_samples)
        if position_ms is not None:
            self.transition_positions_ms.append(position_ms)
        self._append(silence)

    def add_speaker_gap(self) -> None:
        self._append(np.zeros(int(0.3 * self.SAMPLE_RATE), dtype=np.float32))

    def _append(self, audio) -> None:
        self.segments.append(audio)
        self.cumulative_samples += len(audio)


def _generate_audio_kokoro(script_text: str, output_filename: str, lang_code: str):
    """Kokoro TTS path (English default, and any language where TTS_ENGINE_* = kokoro)."""
    logger.info("=" * 60)
    logger.info("KOKORO TTS AUDIO GENERATION")
    logger.info("=" * 60)

    voices = VOICE_MAP.get(lang_code, VOICE_MAP["a"])
    logger.info(f"Voices: Host 1 ({voices['host1']}), Host 2 ({voices['host2']})")

    pipeline = _init_kokoro_pipeline(lang_code)
    if pipeline is None:
        return None

    # Parse Script — strict Speaker N: pattern only (no greedy name matching)
    speaker_pattern = re.compile(r"^(Speaker\s*(\d+))\s*[:：]\s*(.*)", re.IGNORECASE)
    speaker_map = {}  # "Speaker 1" → 1, "Speaker 2" → 2
    tl = _KokoroTimeline(pipeline, voices["host1"], voices["host2"])

    current_speaker = None
    buffer_text = ""

    for raw_line in script_text.split("\n"):
        line = raw_line.strip()
        # Skip blanks, ## guidance/metadata comments, and --- topic separators
        if not line or line.startswith("##") or re.match(r"^-{3,}$", line):
            continue

        # Audio markers ([TRANSITION], [PAUSE], [BEAT])
        if line in MARKER_SILENCE:
            tl.flush_turn(buffer_text, current_speaker)
            buffer_text = ""
            tl.add_marker(line)
            continue

        match = speaker_pattern.match(line)
        if match:
            name = match.group(1).strip()  # "Speaker 1" or "Speaker 2"
            if name not in speaker_map:
                speaker_map[name] = int(match.group(2))
                logger.info(f"  Speaker detected: '{name}' → Host {speaker_map[name]}")
            new_speaker = speaker_map[name]

            tl.flush_turn(buffer_text, current_speaker)
            buffer_text = ""
            # Silence gap on speaker change (not before the first speaker)
            if current_speaker is not None and current_speaker != new_speaker:
                tl.add_speaker_gap()

            current_speaker = new_speaker
            buffer_text = match.group(3).strip()
        elif current_speaker is None:
            # Channel intro line (before any Speaker label) — synthesize as
            # Speaker 2 (the presenter/narrator role) instead of skipping.
            if line and not line.startswith("["):
                logger.info(f"  Channel intro (Speaker 2): {line[:60]}...")
                tl.flush_turn(line, 2)
            else:
                logger.debug(f"  Skipping unlabeled line before first speaker: {line[:60]}...")
        else:
            buffer_text = f"{buffer_text} {line}".strip()

    tl.flush_turn(buffer_text, current_speaker)

    logger.info(f"Generated {tl.segment_count} audio segments")
    if tl.transition_positions_ms:
        logger.info(f"Transition positions: {tl.transition_positions_ms}")

    if not _write_generated_audio(tl.segments, tl.SAMPLE_RATE, output_filename, "Kokoro"):
        return None
    return (output_filename, tl.transition_positions_ms)


def post_process_audio(
    wav_path: str, bgm_target: str = "Interesting BGM.wav", transition_positions_ms: list = None
) -> str:
    """
    Post-process raw TTS output: select background music from library or generate it, then mix.

    Args:
        wav_path: Path to the raw WAV file (24kHz, mono)
        bgm_target: Filename in 'Podcast BGM' folder OR 'random' OR music description for generation.
                    Defaults to "Interesting BGM.wav".
        transition_positions_ms: List of millisecond positions for BGM volume bumps (from TTS markers).

    Returns:
        Path to the mastered WAV file, or None if processing failed
    """
    try:
        logger.info(f"Post-processing audio: {wav_path}")

        # BGM library lives at project root, not next to this file
        _project_root = Path(__file__).resolve().parent.parent.parent
        BGM_LIBRARY_DIR = _project_root / "Podcast BGM"
        if not BGM_LIBRARY_DIR.exists():
            BGM_LIBRARY_DIR = Path.cwd() / "Podcast BGM"
        music_path = None

        # 1. Select Music from Library
        if BGM_LIBRARY_DIR.exists():
            if bgm_target == "random":
                # Pick random .wav file
                files = list(BGM_LIBRARY_DIR.glob("*.wav"))
                if files:
                    selected = random.choice(files)
                    music_path = str(selected)
                    logger.info(f"Selected random BGM from library: {selected.name}")
                else:
                    logger.warning("BGM Library is empty.")

            elif (BGM_LIBRARY_DIR / bgm_target).exists():
                # Specific file found
                music_path = str(BGM_LIBRARY_DIR / bgm_target)
                logger.info(f"Selected specific BGM from library: {bgm_target}")

            elif bgm_target.endswith(".wav"):
                # Requested specific file but not found
                logger.warning(f"Requested BGM '{bgm_target}' not found in library.")
                default_bgm = BGM_LIBRARY_DIR / "Interesting BGM.wav"
                if default_bgm.exists():
                    music_path = str(default_bgm)
                    logger.warning("Falling back to default: Interesting BGM.wav")

        # 2. No BGM available — return voice-only
        if not music_path:
            logger.warning("No BGM available (library empty or file not found), returning voice-only audio.")
            return wav_path

        # 3. Mix
        # AudioMixer is defined above (merged from audio_mixer.py in T4.1)
        mixer = AudioMixer()
        mixed_path = wav_path.replace(".wav", "_mixed.wav")

        # Try pro mixing with pre/post roll and transition bumps
        from dr2_podcast.config import VOICE_DUCKING_DB

        success = mixer.mix_podcast_pro(
            wav_path,
            music_path,
            mixed_path,
            MixSettings(
                transition_positions_ms=transition_positions_ms or [],
                voice_ducking_db=VOICE_DUCKING_DB,
            ),
        )

        if success:
            logger.info(f"Mastered audio saved: {mixed_path}")
            return mixed_path
        else:
            return wav_path

    except Exception as e:
        logger.error(f"Audio post-processing failed: {e}")
        return wav_path


# Audio markers recognized by TTS engine — inserted by editor in Phase 6
AUDIO_MARKERS = {
    "[TRANSITION]": "___TRANSITION___",
    "[INTRO_END]": "___INTRO_END___",
    "[PAUSE]": "___PAUSE___",
    "[BEAT]": "___BEAT___",
}

# Silence duration (seconds) for each marker type
MARKER_SILENCE = {
    "[TRANSITION]": 1.5,
    "[INTRO_END]": 2.5,
    "[PAUSE]": 0.8,
    "[BEAT]": 0.3,
}


# ---------------------------------------------------------------------------
# TTS reading glossary (Layer 1) — deterministic pre-render substitution of
# CONFIRMED misread words to hiragana. Data: dr2_podcast/data/tts_glossary.json.
# Applied inside clean_script_for_tts (after furigana-strip, before markdown
# cleanup) so it reaches every render path. Context-DEPENDENT readings
# (方/表/辛い/大あり) live in the editor prompt + validator, NOT here.
# See PLAN.md "TTS glossary + style-rules pipeline enforcement".
# ---------------------------------------------------------------------------
_TTS_GLOSSARY_PATH = Path(__file__).resolve().parent.parent / "data" / "tts_glossary.json"
_tts_glossary_cache = None  # None = unloaded; dict once loaded (empty dicts = no-op)
_KANJI_CLASS = r"々一-鿿㐀-䶿"  # CJK ideographs + 々

# Stage directions / performance cues: annotations written for a human reader
# that TTS otherwise speaks aloud as words ("（笑）" is heard as "わらい").
# CURATED, never a blanket parenthetical strip — a corpus scan of every script
# shows the large majority of parentheticals are content glosses that MUST be
# spoken (（治療必要数）, （アブストラクト）, （Cohen's d）, （例えば 9 時間以上）).
# Only whole-parenthetical matches of these cues, plus 〜ながら action cues, go.
_STAGE_DIRECTION_CUES = [
    "笑",
    "笑い",
    "苦笑",
    "微笑",
    "爆笑",
    "失笑",
    "ため息",
    "ためいき",
    "咳払い",
    "間",
    "沈黙",
    "拍手",
    "拍手音",
    "効果音",
    "BGM",
    "会話再開",
]
_STAGE_DIRECTION_RE = re.compile(
    r"[（(]\s*(?:" + "|".join(re.escape(c) for c in _STAGE_DIRECTION_CUES) + r"|[^）)（(]{1,8}ながら)\s*[）)]"
)


def _load_tts_glossary():
    """Load + validate the glossary once (cached). Fail-safe: any problem →
    empty maps (glossary becomes a no-op; the audio path must never crash on it)."""
    global _tts_glossary_cache
    if _tts_glossary_cache is not None:
        return _tts_glossary_cache
    if os.environ.get("TTS_GLOSSARY_ENABLED", "1").lower() not in ("1", "true", "yes"):
        logger.info("TTS glossary disabled via TTS_GLOSSARY_ENABLED")
        _tts_glossary_cache = {"safe": {}, "guarded": {}}
        return _tts_glossary_cache
    try:
        with open(_TTS_GLOSSARY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        safe = {k: v for k, v in data.get("safe", {}).items() if not k.startswith("_")}
        guarded = {k: v for k, v in data.get("guarded", {}).items() if not k.startswith("_")}
        # Idempotency invariant: no output value may contain any key, else a
        # second pass would re-substitute (edu re-renders rely on idempotency).
        all_keys = list(safe) + list(guarded)
        for val in list(safe.values()) + [g["to"] for g in guarded.values()]:
            for key in all_keys:
                if key in val:
                    raise ValueError(f"glossary not idempotent: value {val!r} contains key {key!r}")
        _tts_glossary_cache = {"safe": safe, "guarded": guarded}
        logger.info(f"TTS glossary loaded: {len(safe)} safe + {len(guarded)} guarded entries")
    except Exception as e:
        logger.warning(f"TTS glossary unavailable ({type(e).__name__}: {e}); skipping substitution")
        _tts_glossary_cache = {"safe": {}, "guarded": {}}
    return _tts_glossary_cache


def apply_tts_glossary(text: str) -> str:
    """Substitute confirmed-misread words to hiragana (Layer 1).

    'safe' keys → plain longest-first replace (non-embeddable words).
    'guarded' keys → protect listed superstrings, optionally skip when the key
    is preceded by a kanji (compound boundary), then substitute — avoids
    corrupting correctly-read compounds like 手強さ/酵母数/意見方針/一番下手.
    """
    gl = _load_tts_glossary()
    if not gl["safe"] and not gl["guarded"]:
        return text
    fired = {}

    # Guarded first: protect superstrings, substitute, restore.
    for i, (key, spec) in enumerate(gl["guarded"].items()):
        placeholders = {}
        for j, g in enumerate(spec.get("guards", [])):
            if g in text:
                ph = f"\x00G{i}_{j}\x00"
                text = text.replace(g, ph)
                placeholders[ph] = g
        if spec.get("no_kanji_prefix"):
            text, n = re.subn(rf"(?<![{_KANJI_CLASS}]){re.escape(key)}", spec["to"], text)
        else:
            n = text.count(key)
            text = text.replace(key, spec["to"])
        if n:
            fired[key] = n
        for ph, g in placeholders.items():
            text = text.replace(ph, g)

    # Safe: longest-first plain replace.
    for key in sorted(gl["safe"], key=len, reverse=True):
        n = text.count(key)
        if n:
            text = text.replace(key, gl["safe"][key])
            fired[key] = n

    if fired:
        logger.info("TTS glossary applied: " + ", ".join(f"{k}×{v}" for k, v in fired.items()))
    return text


def clean_script_for_tts(script_text: str) -> str:
    """
    Clean script text for TTS processing by removing markdown and LLM artifacts.
    Preserves [TRANSITION], [INTRO_END], [PAUSE], and [BEAT] audio markers.

    Args:
        script_text: Raw script text with potential markdown and tags

    Returns:
        Cleaned script text ready for TTS
    """
    # Protect audio markers before cleaning
    for marker, placeholder in AUDIO_MARKERS.items():
        script_text = script_text.replace(marker, placeholder)

    # Remove thinking tags
    clean = strip_think_blocks(script_text)

    # Strip ## comment lines (guidance, metadata, LLM preamble) — must happen
    # BEFORE markdown # removal below, which would strip the ## prefix leaving bare text
    clean = re.sub(r"^##.*$", "", clean, flags=re.MULTILINE)

    # Drop furigana reading annotations: 漢字（かな） is a pronunciation guide for a
    # reader, but TTS pronounces the kanji correctly on its own — reading the
    # hiragana too produces an audible duplicate (e.g. "更年期（こうねんき）" is
    # heard as "こうねんき、こうねんき"). Only strip when the parenthetical is
    # PURE hiragana directly after a kanji run — katakana parentheticals (e.g.
    # "要約（アブストラクト）") are glosses/synonyms carrying new information and
    # must be read aloud, not furigana, so they're left untouched.
    clean = re.sub(r"([一-鿿㐀-䶿々]+)（[぀-ゟー]+）", r"\1", clean)

    # Drop stage directions / performance cues (（笑）（拍手）（頷きながら）) — these
    # are reader annotations, not dialogue, and the furigana strip above does not
    # catch them (it only matches kanji-then-pure-hiragana). Curated cue list, so
    # content glosses like （治療必要数）/（アブストラクト） are left to be spoken.
    clean = _STAGE_DIRECTION_RE.sub("", clean)

    # Layer 1: deterministic reading glossary — force CONFIRMED-misread words to
    # hiragana (after furigana-strip above so 漢字（かな） pairs collapse first).
    clean = apply_tts_glossary(clean)

    # Remove markdown formatting
    clean = re.sub(r"\*\*", "", clean)  # Bold
    clean = re.sub(
        r"[*#\[\]]", "", clean
    )  # Italics, headers, brackets (NOT underscores — protects ___TRANSITION___ placeholders)

    # Rename Japanese speaker labels "ホストN:" → "Speaker N:" (before English conversion)
    clean = re.sub(r"^\*{0,2}ホスト\s*(\d)\s*\*{0,2}\s*[:：]", r"Speaker \1:", clean, flags=re.MULTILINE)

    # Rename speaker labels "Host N:" → "Speaker N:" so that any remaining
    # "Host 1" / "Host 2" in dialogue text can be safely stripped.
    # Case/underscore-insensitive: also catches malformed 'host_1：' / 'HOST 1:'
    # labels that the pipeline normalizer is the first line of defense against
    # (the sleep-week Tue episode leaked lowercase host_1： into TTS).
    clean = re.sub(
        r"^\*{0,2}host[ _]*(\d)[ _]*\*{0,2}\s*[:：]", r"Speaker \1:", clean, flags=re.MULTILINE | re.IGNORECASE
    )
    clean = re.sub(r"Host [12]", "", clean, flags=re.IGNORECASE)

    # Normalize unicode punctuation to ASCII
    unicode_map = {
        "\u2018": "'",
        "\u2019": "'",  # Smart quotes
        "\u201c": '"',
        "\u201d": '"',  # Smart double quotes
        "\u2014": " - ",
        "\u2013": " - ",  # Em/en dash
        "\u2026": "...",  # Ellipsis
    }
    for old, new in unicode_map.items():
        clean = clean.replace(old, new)

    # Normalize whitespace within lines, but preserve line breaks
    clean = re.sub(r"[^\S\n]+", " ", clean)  # collapse spaces/tabs but keep \n
    clean = re.sub(r"\n{3,}", "\n\n", clean)  # collapse excessive blank lines

    # Strip spaces at Latin↔Japanese boundaries (the JA engine otherwise inserts an
    # audible pause between e.g. "AI ラボ員達" or "DHA 配合").
    _JP = r"\u3005\u3040-\u309F\u30A0-\u30FF\u3400-\u4DBF\u4E00-\u9FFF"
    clean = re.sub(rf"(?<=[A-Za-z0-9])[ \t]+(?=[{_JP}])", "", clean)
    clean = re.sub(rf"(?<=[{_JP}])[ \t]+(?=[A-Za-z0-9])", "", clean)

    clean = clean.strip()

    # Restore audio markers
    for marker, placeholder in AUDIO_MARKERS.items():
        clean = clean.replace(placeholder, marker)

    return clean


# Test function for standalone usage
if __name__ == "__main__":
    test_script = """
    Host 1: Welcome to Deep Research Podcast. Today we're exploring the scientific evidence behind coffee consumption and productivity.

    Host 2: That's an interesting topic. But we need to be careful about the claims. What does the evidence actually say?

    Host 1: Studies show that caffeine blocks adenosine receptors in the brain, which reduces fatigue and increases alertness. This mechanism is well-documented in neuroscience literature.

    Host 2: True, but that's just the mechanism. Does it actually translate to measurable productivity gains in real-world settings?

    Host 1: Meta-analyses of randomized controlled trials show a modest but consistent improvement in cognitive performance tasks, particularly for sustained attention and reaction time.

    Host 2: Modest is the key word there. And we should note that these effects plateau quickly. More coffee doesn't mean more productivity after a certain point.
    """

    print("Testing Kokoro TTS Engine...")
    cleaned_script = clean_script_for_tts(test_script)
    result = generate_audio_from_script(cleaned_script, "test_podcast.wav")

    if result:
        print(f"✓ Test successful! Audio saved to: {result}")
    else:
        print("✗ Test failed!")
