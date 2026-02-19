"""
Kokoro TTS Audio Engine for Deep Research Podcast
==================================================

Generates high-quality, multi-speaker podcast audio using Kokoro-82M (local TTS).

Features:
- Dual-voice system: Host 1 (am_fenrir - American Male Expert)
                     Host 2 (af_heart - American Female Skeptic)
- Script parsing with speaker detection
- Audio stitching and WAV export
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

def generate_audio_from_script(script_text: str, output_filename: str = "final_podcast.wav", lang_code: str = 'a') -> str:
    """
    Parses a script looking for 'Host 1:' and 'Host 2:' lines,
    generates audio segments using Kokoro, and stitches them together.

    Args:
        script_text: Full podcast script with "Host 1:" and "Host 2:" labels
        output_filename: Output WAV file name (default: "final_podcast.wav")
        lang_code: Kokoro language code ('a' for English, 'j' for Japanese, etc.)

    Returns:
        Path to generated audio file, or None if generation failed

    Example Script Format:
        Host 1: Welcome to the show. Today we're discussing coffee.
        Host 2: But is coffee actually good for you? Let's examine the evidence.
        Host 1: Studies show that moderate coffee intake...
    """
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

    # 2. Parse Script
    # Generic speaker detection: any line starting with "Name:" or "Name："
    # First unique name → Host 1 voice, second unique name → Host 2 voice
    speaker_pattern = re.compile(r'^(.+?)[:：]\s*(.*)')
    speaker_map = {}  # name → speaker number (1 or 2)

    lines = script_text.split('\n')
    audio_segments = []
    silence_gap = np.zeros(int(0.3 * 24000), dtype=np.float32)  # 300ms silence at 24kHz

    current_speaker = None
    buffer_text = ""
    segment_count = 0

    def _flush_buffer():
        """Flush the current text buffer into audio segments."""
        nonlocal buffer_text, segment_count
        if buffer_text and current_speaker:
            voice = voice_host_1 if current_speaker == 1 else voice_host_2
            try:
                generator = pipeline(buffer_text, voice=voice, speed=1.0, split_pattern=r'\n+')
                for _, _, audio in generator:
                    audio_segments.append(audio)
                    segment_count += 1
            except Exception as e:
                logger.warning(f"  ⚠ Warning: Failed to generate segment {segment_count}: {e}")
        buffer_text = ""

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        # Skip section separators (--- between topics) — not end of dialogue
        if re.match(r'^-{3,}$', line):
            continue

        # Check for Speaker Switch: any "Name:" or "Name：" prefix
        match = speaker_pattern.match(line)
        if match:
            name = match.group(1).strip()
            text_after = match.group(2).strip()

            # Assign speaker number on first occurrence (max 2 speakers)
            if name not in speaker_map:
                if len(speaker_map) < 2:
                    speaker_map[name] = len(speaker_map) + 1
                    logger.info(f"  Speaker detected: '{name}' → Host {speaker_map[name]}")
                else:
                    # More than 2 unique names — treat as continuation text
                    if current_speaker is None:
                        current_speaker = 1
                    buffer_text = f"{buffer_text} {line}".strip()
                    continue

            new_speaker = speaker_map[name]

            # Flush previous buffer before switching
            _flush_buffer()

            # Insert silence gap on speaker change (not before first speaker)
            if current_speaker is not None and current_speaker != new_speaker:
                audio_segments.append(silence_gap)

            current_speaker = new_speaker
            buffer_text = text_after

        else:
            # Continuation of current speaker (or unlabeled opening — default to Host 1)
            if current_speaker is None:
                current_speaker = 1
            buffer_text = f"{buffer_text} {line}".strip()

    # Process final buffer
    _flush_buffer()

    logger.info(f"Generated {segment_count} audio segments")

    # 3. Stitch and Save
    if audio_segments:
        try:
            final_audio = np.concatenate(audio_segments)
            sf.write(output_filename, final_audio, 24000)  # Kokoro standard: 24kHz

            file_size = Path(output_filename).stat().st_size
            duration_sec = len(final_audio) / 24000
            duration_min = duration_sec / 60

            logger.info(f"\n✓ Audio generated successfully:")
            logger.info(f"  File: {output_filename}")
            logger.info(f"  Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
            logger.info(f"  Duration: {duration_min:.2f} minutes ({duration_sec:.1f} seconds)")
            logger.info("=" * 60 + "\n")

            return output_filename
        except Exception as e:
            logger.error(f"✗ ERROR: Failed to save audio: {e}")
            return None
    else:
        logger.error("✗ ERROR: No audio segments generated")
        return None


def post_process_audio(wav_path: str, bgm_target: str = "Interesting BGM.wav") -> str:
    """
    Post-process raw Kokoro TTS output: select background music from library or generate it, then mix.

    Args:
        wav_path: Path to the raw WAV file (24kHz, mono)
        bgm_target: Filename in 'Podcast BGM' folder OR 'random' OR music description for generation.
                    Defaults to "Interesting BGM.wav".

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

        success = mixer.mix_podcast(wav_path, music_path, mixed_path)

        if success:
            logger.info(f"Mastered audio saved: {mixed_path}")
            return mixed_path
        else:
            return wav_path

    except Exception as e:
        logger.error(f"Audio post-processing failed: {e}")
        return wav_path


def clean_script_for_tts(script_text: str) -> str:
    """
    Clean script text for TTS processing by removing markdown and LLM artifacts.

    Args:
        script_text: Raw script text with potential markdown and tags

    Returns:
        Cleaned script text ready for TTS
    """
    # Remove thinking tags
    clean = re.sub(r'<think>.*?</think>', '', script_text, flags=re.DOTALL)

    # Remove markdown formatting
    clean = re.sub(r'\*\*', '', clean)  # Bold
    clean = re.sub(r'[*#_\[\]]', '', clean)  # Italics, headers, underscores, brackets

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
