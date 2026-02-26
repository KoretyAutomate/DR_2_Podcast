"""
TTS Audio Engine for Deep Research Podcast
===========================================

Generates high-quality, multi-speaker podcast audio with automatic TTS engine selection:

  - English:  Kokoro TTS (local, CPU, proven quality)
  - Japanese: Qwen3-TTS CustomVoice (GPU via Docker, built-in preset voices)
              Voices: Kaz → Aiden (male), Erika → Ono_Anna (native Japanese female)

Features:
- Dual-voice system with speaker detection
- Automatic language routing (lang_code='a' → Kokoro, 'j' → Qwen3-TTS)
- Script parsing and audio stitching
- WAV export with BGM support
"""

import logging
import soundfile as sf
from kokoro import KPipeline
import torch
import numpy as np
import re
import os
import random
from pathlib import Path

from pydub import AudioSegment

logger = logging.getLogger(__name__)

# Voice Configuration
VOICE_HOST_1 = 'am_fenrir'  # American Male (The Expert) - default English
VOICE_HOST_2 = 'af_heart'   # American Female (The Skeptic) - default English
LANG_CODE = 'a'  # American English (default)

# Per-language voice mapping for Kokoro TTS
VOICE_MAP = {
    'a': {'host1': 'am_fenrir', 'host2': 'af_heart'},     # English
    'j': {'host1': 'jm_kumo',   'host2': 'jf_alpha'},    # Japanese
}

# Qwen3-TTS Configuration (for Japanese only — English uses Kokoro)
QWEN3_TTS_API_URL = os.getenv("QWEN3_TTS_API_URL")  # Set in .env; required for Japanese TTS


def _chunk_japanese_text(text: str, max_chars: int = 80) -> list:
    """Split Japanese text at sentence-end punctuation to keep each TTS call under max_chars."""
    sentences = re.split(r'(?<=[。！？\n])', text)
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


def _call_qwen3_tts_segment(text: str, speaker: int) -> tuple:
    """Call Qwen3-TTS API. speaker: 1=Kaz(Aiden), 2=Erika(Ono_Anna). Returns (audio, sr) or (None, None)."""
    try:
        import requests
        import io as _io
    except ImportError as e:
        logger.error(f"Missing dependency for Qwen3-TTS: {e}")
        return None, None

    speaker_name = "Kaz" if speaker == 1 else "Erika"
    try:
        resp = requests.post(
            f"{QWEN3_TTS_API_URL}/tts",
            json={"text": text, "speaker": speaker_name, "language": "Japanese"},
        )
        resp.raise_for_status()
        audio, sr = sf.read(_io.BytesIO(resp.content))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), sr
    except requests.exceptions.ConnectionError:
        logger.error(
            f"Qwen3-TTS API unreachable at {QWEN3_TTS_API_URL}. "
            f"Start it: docker compose -f docker/qwen3-tts/docker-compose.yml up -d"
        )
        return None, None
    except Exception as e:
        logger.error(f"Qwen3-TTS API error: {e}")
        return None, None


def _generate_audio_qwen3_tts(script_text: str, output_filename: str) -> str:
    """
    Japanese TTS via Qwen3-TTS API.

    Handles multi-speaker script parsing identically to the Kokoro path,
    but calls Qwen3-TTS REST API instead of the local Kokoro pipeline.

    Args:
        script_text: Full podcast script with speaker labels
        output_filename: Output WAV file path

    Returns:
        Path to generated audio file, or None if generation failed
    """
    logger.info("=" * 60)
    logger.info("QWEN3-TTS — JAPANESE AUDIO GENERATION")
    logger.info("=" * 60)
    logger.info(f"API endpoint: {QWEN3_TTS_API_URL}")
    logger.info("Voices: Kaz → Aiden (male), Erika → Ono_Anna (Japanese female)")

    # Health check
    try:
        import requests
        health = requests.get(f"{QWEN3_TTS_API_URL}/health", timeout=5)
        if health.status_code == 200:
            logger.info("✓ Qwen3-TTS API is healthy")
        else:
            logger.warning(f"  Qwen3-TTS API health check returned {health.status_code}")
    except Exception:
        logger.error(
            f"✗ Qwen3-TTS API not reachable at {QWEN3_TTS_API_URL}\n"
            f"  Start it: docker compose -f docker/qwen3-tts/docker-compose.yml up -d"
        )
        return None

    # Parse script — strict Host N: pattern only (no greedy name matching)
    speaker_pattern = re.compile(r'^(Host\s*(\d+))\s*[:：]\s*(.*)', re.IGNORECASE)
    speaker_map = {}  # "Host 1" → 1, "Host 2" → 2

    lines = script_text.split('\n')
    audio_segments = []
    sample_rate = None  # determined from first API response
    transition_positions_ms = []  # Track [TRANSITION] positions for pro mixer
    cumulative_samples = 0

    current_speaker = None
    buffer_text = ""
    segment_count = 0

    def _flush_buffer():
        nonlocal buffer_text, segment_count, sample_rate, cumulative_samples
        if buffer_text and current_speaker:
            logger.info(f"  Segment {segment_count + 1} (Speaker {current_speaker}): {buffer_text[:50]}...")
            chunks = _chunk_japanese_text(buffer_text)
            chunk_audios = []
            for chunk in chunks:
                a, sr_chunk = _call_qwen3_tts_segment(chunk, current_speaker)
                if a is not None:
                    chunk_audios.append(a)
                else:
                    sr_fallback = sample_rate or 24000
                    silence_secs = max(0.5, len(chunk) / 8.0)
                    chunk_audios.append(np.zeros(int(silence_secs * sr_fallback), dtype=np.float32))
                    logger.warning(f"  Chunk failed — inserted {silence_secs:.1f}s silence")
            if chunk_audios:
                if sample_rate is None:
                    sample_rate = sr_chunk if sr_chunk is not None else 24000
                    logger.info(f"  Sample rate: {sample_rate} Hz")
                segment_audio = np.concatenate(chunk_audios)
                audio_segments.append(segment_audio)
                cumulative_samples += len(segment_audio)
                segment_count += 1
            else:
                logger.warning(f"  ⚠ Segment {segment_count + 1} failed — skipping")
        buffer_text = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip ## comment lines (guidance, metadata)
        if line.startswith('##'):
            continue

        # Skip section separators
        if re.match(r'^-{3,}$', line):
            continue

        # Check for audio markers ([TRANSITION], [PAUSE], [BEAT])
        if line in MARKER_SILENCE:
            _flush_buffer()
            # Use 24000 as fallback sample rate if not yet determined
            _sr = sample_rate or 24000
            silence_sec = MARKER_SILENCE[line]
            silence_samples = int(silence_sec * _sr)
            audio_segments.append(np.zeros(silence_samples, dtype=np.float32))
            if line == '[TRANSITION]':
                position_ms = int((cumulative_samples / _sr) * 1000)
                transition_positions_ms.append(position_ms)
                logger.info(f"  [TRANSITION] marker at {position_ms}ms")
            cumulative_samples += silence_samples
            continue

        # Check for speaker switch — strict "Host N:" pattern only
        match = speaker_pattern.match(line)
        if match:
            name = match.group(1).strip()    # "Host 1" or "Host 2"
            slot = int(match.group(2))        # 1 or 2
            text_after = match.group(3).strip()

            if name not in speaker_map:
                speaker_map[name] = slot
                logger.info(f"  Speaker detected: '{name}' → Host {slot}")

            new_speaker = speaker_map[name]
            _flush_buffer()

            # Insert silence gap on speaker change
            if current_speaker is not None and current_speaker != new_speaker and sample_rate:
                silence = np.zeros(int(0.3 * sample_rate), dtype=np.float32)
                audio_segments.append(silence)
                cumulative_samples += len(silence)

            current_speaker = new_speaker
            buffer_text = text_after
        else:
            # Continuation of current speaker's dialogue
            if current_speaker is None:
                # Skip lines before the first Host label (preamble, metadata)
                logger.debug(f"  Skipping unlabeled line before first speaker: {line[:60]}...")
                continue
            buffer_text = f"{buffer_text} {line}".strip()

    # Process final buffer
    _flush_buffer()

    logger.info(f"Generated {segment_count} audio segments")
    if transition_positions_ms:
        logger.info(f"Transition positions: {transition_positions_ms}")

    # Stitch and save
    if audio_segments and sample_rate:
        try:
            final_audio = np.concatenate(audio_segments)
            sf.write(output_filename, final_audio, sample_rate)

            file_size = Path(output_filename).stat().st_size
            duration_sec = len(final_audio) / sample_rate
            duration_min = duration_sec / 60

            logger.info(f"\n✓ Audio generated successfully (Qwen3-TTS):")
            logger.info(f"  File: {output_filename}")
            logger.info(f"  Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
            logger.info(f"  Duration: {duration_min:.2f} minutes ({duration_sec:.1f} seconds)")
            logger.info(f"  Sample rate: {sample_rate} Hz")
            logger.info("=" * 60 + "\n")

            return (output_filename, transition_positions_ms)
        except Exception as e:
            logger.error(f"✗ ERROR: Failed to save audio: {e}")
            return None
    else:
        logger.error("✗ ERROR: No audio segments generated")
        return None


def generate_audio_from_script(script_text: str, output_filename: str = "final_podcast.wav", lang_code: str = 'a'):
    """
    Parses a script looking for 'Host 1:' and 'Host 2:' lines,
    generates audio segments, and stitches them together.

    TTS Engine selection:
      - English (lang_code='a'): Kokoro TTS (local, CPU, proven)
      - Japanese (lang_code='j'): Qwen3-TTS API (GPU via Docker, distinct preset voices)

    Args:
        script_text: Full podcast script with "Host 1:" and "Host 2:" labels
        output_filename: Output WAV file name (default: "final_podcast.wav")
        lang_code: Language code ('a' for English, 'j' for Japanese, etc.)

    Returns:
        Tuple of (path_to_audio_file, transition_positions_ms) or None if failed.
        transition_positions_ms is a list of millisecond positions where [TRANSITION]
        markers were found, used by the pro mixer for BGM volume bumps.

    Example Script Format:
        Host 1: Welcome to the show. Today we're discussing coffee.
        Host 2: But is coffee actually good for you? Let's examine the evidence.
        [TRANSITION]
        Host 1: Studies show that moderate coffee intake...
    """
    # Route to Qwen3-TTS for Japanese (GPU-accelerated, distinct preset voices)
    if lang_code == 'j':
        return _generate_audio_qwen3_tts(script_text, output_filename)

    # English and all other languages use Kokoro TTS
    logger.info("=" * 60)
    logger.info("KOKORO TTS AUDIO GENERATION")
    logger.info("=" * 60)

    # Resolve voices for this language
    voices = VOICE_MAP.get(lang_code, VOICE_MAP['a'])
    voice_host_1 = voices['host1']
    voice_host_2 = voices['host2']

    # 1. Initialize Pipeline
    device = 'cpu'
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            device = 'cuda'
        except RuntimeError:
            logger.warning("  CUDA reported available but kernel execution failed, falling back to CPU")
    logger.info(f"Device: {device}")
    logger.info(f"Language code: {lang_code}")
    logger.info(f"Voices: Host 1 ({voice_host_1}), Host 2 ({voice_host_2})")

    try:
        pipeline = KPipeline(lang_code=lang_code, device=device)
        logger.info("✓ Kokoro pipeline initialized")
    except RuntimeError as e:
        if 'CUDA' in str(e) and device == 'cuda':
            logger.warning(f"  CUDA init failed, retrying on CPU: {e}")
            device = 'cpu'
            pipeline = KPipeline(lang_code=lang_code, device=device)
            logger.info("✓ Kokoro pipeline initialized (CPU fallback)")
        else:
            logger.error(f"✗ ERROR: Failed to initialize Kokoro: {e}")
            return None
    except Exception as e:
        logger.error(f"✗ ERROR: Failed to initialize Kokoro: {e}")
        return None

    # 2. Parse Script — strict Host N: pattern only (no greedy name matching)
    speaker_pattern = re.compile(r'^(Host\s*(\d+))\s*[:：]\s*(.*)', re.IGNORECASE)
    speaker_map = {}  # "Host 1" → 1, "Host 2" → 2

    sample_rate = 24000  # Kokoro standard
    lines = script_text.split('\n')
    audio_segments = []
    silence_gap = np.zeros(int(0.3 * sample_rate), dtype=np.float32)  # 300ms silence
    transition_positions_ms = []  # Track [TRANSITION] positions for pro mixer
    cumulative_samples = 0  # Running total for position tracking

    current_speaker = None
    buffer_text = ""
    segment_count = 0

    def _flush_buffer():
        """Flush the current text buffer into audio segments."""
        nonlocal buffer_text, segment_count, cumulative_samples
        if buffer_text and current_speaker:
            voice = voice_host_1 if current_speaker == 1 else voice_host_2
            try:
                generator = pipeline(buffer_text, voice=voice, speed=1.0, split_pattern=r'\n+')
                for _, _, audio in generator:
                    audio_segments.append(audio)
                    cumulative_samples += len(audio)
                    segment_count += 1
            except Exception as e:
                logger.warning(f"  ⚠ Warning: Failed to generate segment {segment_count}: {e}")
        buffer_text = ""

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        # Skip ## comment lines (guidance, metadata)
        if line.startswith('##'):
            continue

        # Skip section separators (--- between topics) — not end of dialogue
        if re.match(r'^-{3,}$', line):
            continue

        # Check for audio markers ([TRANSITION], [PAUSE], [BEAT])
        if line in MARKER_SILENCE:
            _flush_buffer()
            silence_sec = MARKER_SILENCE[line]
            silence_samples = int(silence_sec * sample_rate)
            audio_segments.append(np.zeros(silence_samples, dtype=np.float32))
            if line == '[TRANSITION]':
                position_ms = int((cumulative_samples / sample_rate) * 1000)
                transition_positions_ms.append(position_ms)
                logger.info(f"  [TRANSITION] marker at {position_ms}ms")
            cumulative_samples += silence_samples
            continue

        # Check for Speaker Switch — strict "Host N:" pattern only
        match = speaker_pattern.match(line)
        if match:
            name = match.group(1).strip()    # "Host 1" or "Host 2"
            slot = int(match.group(2))        # 1 or 2
            text_after = match.group(3).strip()

            if name not in speaker_map:
                speaker_map[name] = slot
                logger.info(f"  Speaker detected: '{name}' → Host {slot}")

            new_speaker = speaker_map[name]

            # Flush previous buffer before switching
            _flush_buffer()

            # Insert silence gap on speaker change (not before first speaker)
            if current_speaker is not None and current_speaker != new_speaker:
                audio_segments.append(silence_gap)
                cumulative_samples += len(silence_gap)

            current_speaker = new_speaker
            buffer_text = text_after

        else:
            # Continuation of current speaker's dialogue
            if current_speaker is None:
                # Skip lines before the first Host label (preamble, metadata)
                logger.debug(f"  Skipping unlabeled line before first speaker: {line[:60]}...")
                continue
            buffer_text = f"{buffer_text} {line}".strip()

    # Process final buffer
    _flush_buffer()

    logger.info(f"Generated {segment_count} audio segments")
    if transition_positions_ms:
        logger.info(f"Transition positions: {transition_positions_ms}")

    # 3. Stitch and Save
    if audio_segments:
        try:
            final_audio = np.concatenate(audio_segments)
            sf.write(output_filename, final_audio, sample_rate)

            file_size = Path(output_filename).stat().st_size
            duration_sec = len(final_audio) / sample_rate
            duration_min = duration_sec / 60

            logger.info(f"\n✓ Audio generated successfully:")
            logger.info(f"  File: {output_filename}")
            logger.info(f"  Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
            logger.info(f"  Duration: {duration_min:.2f} minutes ({duration_sec:.1f} seconds)")
            logger.info("=" * 60 + "\n")

            return (output_filename, transition_positions_ms)
        except Exception as e:
            logger.error(f"✗ ERROR: Failed to save audio: {e}")
            return None
    else:
        logger.error("✗ ERROR: No audio segments generated")
        return None


def post_process_audio(wav_path: str, bgm_target: str = "Interesting BGM.wav",
                       transition_positions_ms: list = None) -> str:
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
        # Ensure archived_scripts is importable (music_engine, audio_mixer live there)
        import sys
        archived = str(Path(__file__).parent / "archived_scripts")
        if archived not in sys.path:
            sys.path.insert(0, archived)

        logger.info(f"Post-processing audio: {wav_path}")

        BGM_LIBRARY_DIR = Path(__file__).parent / "Podcast BGM"
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
                    logger.warning("BGM Library is empty. Falling back to generation.")

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
                     logger.warning(f"Falling back to default: Interesting BGM.wav")

        # 2. Fallback to MusicGen if no music selected yet
        if not music_path:
            logger.info("Generating new BGM (Library file not found or empty)...")
            try:
                from music_engine import MusicGenerator
            except ImportError as e:
                logger.warning(f"MusicGenerator unavailable ({e}), skipping BGM generation.")
                return wav_path
            music_gen = MusicGenerator()

            # Use bgm_target as prompt if it doesn't look like a filename, otherwise default prompt
            prompt = bgm_target if " " in bgm_target and not bgm_target.endswith(".wav") else "lofi hip hop beat, chill, study"

            music_path = str(Path(wav_path).parent / "bgm_generated.wav")
            if not os.path.exists(music_path):
                generated_music = music_gen.generate_music(prompt, duration=30, output_filename=music_path)
                if not generated_music:
                    logger.warning("Music generation failed. Skipping BGM.")
                    return wav_path
                music_path = generated_music

        # 3. Mix
        try:
            from audio_mixer import AudioMixer
        except ImportError as e:
            logger.error(f"AudioMixer unavailable ({e}), returning original audio.")
            return wav_path

        mixer = AudioMixer()
        mixed_path = wav_path.replace(".wav", "_mixed.wav")

        # Try pro mixing with pre/post roll and transition bumps
        success = mixer.mix_podcast_pro(
            wav_path, music_path, mixed_path,
            transition_positions_ms=transition_positions_ms or []
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
AUDIO_MARKERS = {'[TRANSITION]': '___TRANSITION___', '[PAUSE]': '___PAUSE___', '[BEAT]': '___BEAT___'}

# Silence duration (seconds) for each marker type
MARKER_SILENCE = {
    '[TRANSITION]': 1.5,
    '[PAUSE]': 0.8,
    '[BEAT]': 0.3,
}


def clean_script_for_tts(script_text: str) -> str:
    """
    Clean script text for TTS processing by removing markdown and LLM artifacts.
    Preserves [TRANSITION], [PAUSE], and [BEAT] audio markers.

    Args:
        script_text: Raw script text with potential markdown and tags

    Returns:
        Cleaned script text ready for TTS
    """
    # Protect audio markers before cleaning
    for marker, placeholder in AUDIO_MARKERS.items():
        script_text = script_text.replace(marker, placeholder)

    # Remove thinking tags
    clean = re.sub(r'<think>.*?</think>', '', script_text, flags=re.DOTALL)

    # Strip ## comment lines (guidance, metadata, LLM preamble) — must happen
    # BEFORE markdown # removal below, which would strip the ## prefix leaving bare text
    clean = re.sub(r'^##.*$', '', clean, flags=re.MULTILINE)

    # Remove markdown formatting
    clean = re.sub(r'\*\*', '', clean)  # Bold
    clean = re.sub(r'[*#\[\]]', '', clean)  # Italics, headers, brackets (NOT underscores — protects ___TRANSITION___ placeholders)

    # Normalize unicode punctuation to ASCII
    unicode_map = {
        '\u2018': "'", '\u2019': "'",  # Smart quotes
        '\u201c': '"', '\u201d': '"',  # Smart double quotes
        '\u2014': ' - ', '\u2013': ' - ',  # Em/en dash
        '\u2026': '...',  # Ellipsis
    }
    for old, new in unicode_map.items():
        clean = clean.replace(old, new)

    # Normalize whitespace within lines, but preserve line breaks
    clean = re.sub(r'[^\S\n]+', ' ', clean)  # collapse spaces/tabs but keep \n
    clean = re.sub(r'\n{3,}', '\n\n', clean)  # collapse excessive blank lines
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
