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
    print("\n" + "="*60)
    print("KOKORO TTS AUDIO GENERATION")
    print("="*60)

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
            print("  CUDA reported available but kernel execution failed, falling back to CPU")
    print(f"Device: {device}")
    print(f"Language code: {lang_code}")
    print(f"Voices: Host 1 ({voice_host_1}), Host 2 ({voice_host_2})")

    try:
        pipeline = KPipeline(lang_code=lang_code, device=device)
        print("✓ Kokoro pipeline initialized")
    except RuntimeError as e:
        if 'CUDA' in str(e) and device == 'cuda':
            print(f"  CUDA init failed, retrying on CPU: {e}")
            device = 'cpu'
            pipeline = KPipeline(lang_code=lang_code, device=device)
            print("✓ Kokoro pipeline initialized (CPU fallback)")
        else:
            print(f"✗ ERROR: Failed to initialize Kokoro: {e}")
            return None
    except Exception as e:
        print(f"✗ ERROR: Failed to initialize Kokoro: {e}")
        return None

    # 2. Parse Script
    lines = script_text.split('\n')
    audio_segments = []
    silence_gap = np.zeros(int(0.3 * 24000), dtype=np.float32)  # 300ms silence at 24kHz

    current_speaker = None
    buffer_text = ""
    segment_count = 0

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        # Stop at section separator (--- marks end of dialogue, start of appendix/notes)
        if re.match(r'^-{3,}$', line):
            break

        # Check for Speaker Switch
        if line.startswith("Host 1:") or line.startswith("Dr. Data:") or line.startswith("Kaz:"):
            # Process previous buffer
            if buffer_text and current_speaker:
                voice = voice_host_1 if current_speaker == 1 else voice_host_2
                try:
                    generator = pipeline(buffer_text, voice=voice, speed=1.0, split_pattern=r'\n+')
                    for _, _, audio in generator:
                        audio_segments.append(audio)
                        segment_count += 1
                except Exception as e:
                    print(f"  ⚠ Warning: Failed to generate segment {segment_count}: {e}")

            # Insert silence gap between speaker switches (not before first speaker)
            if current_speaker is not None and current_speaker != 1:
                audio_segments.append(silence_gap)

            current_speaker = 1
            buffer_text = line.split(":", 1)[1].strip() if ":" in line else ""

        elif line.startswith("Host 2:") or line.startswith("Dr. Doubt:") or line.startswith("Erika:"):
            # Process previous buffer
            if buffer_text and current_speaker:
                voice = voice_host_1 if current_speaker == 1 else voice_host_2
                try:
                    generator = pipeline(buffer_text, voice=voice, speed=1.0, split_pattern=r'\n+')
                    for _, _, audio in generator:
                        audio_segments.append(audio)
                        segment_count += 1
                except Exception as e:
                    print(f"  ⚠ Warning: Failed to generate segment {segment_count}: {e}")

            # Insert silence gap between speaker switches (not before first speaker)
            if current_speaker is not None and current_speaker != 2:
                audio_segments.append(silence_gap)

            current_speaker = 2
            buffer_text = line.split(":", 1)[1].strip() if ":" in line else ""

        else:
            # Continuation of current speaker (or unlabeled opening — default to Host 1)
            if current_speaker is None:
                current_speaker = 1
            if buffer_text:
                buffer_text += " " + line
            else:
                buffer_text = line

    # Process final buffer
    if buffer_text and current_speaker:
        voice = voice_host_1 if current_speaker == 1 else voice_host_2
        try:
            generator = pipeline(buffer_text, voice=voice, speed=1.0, split_pattern=r'\n+')
            for _, _, audio in generator:
                audio_segments.append(audio)
                segment_count += 1
        except Exception as e:
            print(f"  ⚠ Warning: Failed to generate final segment: {e}")

    print(f"Generated {segment_count} audio segments")

    # 3. Stitch and Save
    if audio_segments:
        try:
            final_audio = np.concatenate(audio_segments)
            sf.write(output_filename, final_audio, 24000)  # Kokoro standard: 24kHz

            file_size = Path(output_filename).stat().st_size
            duration_sec = len(final_audio) / 24000
            duration_min = duration_sec / 60

            print(f"\n✓ Audio generated successfully:")
            print(f"  File: {output_filename}")
            print(f"  Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
            print(f"  Duration: {duration_min:.2f} minutes ({duration_sec:.1f} seconds)")
            print("="*60 + "\n")

            return output_filename
        except Exception as e:
            print(f"✗ ERROR: Failed to save audio: {e}")
            return None
    else:
        print("✗ ERROR: No audio segments generated")
        return None


def post_process_audio(wav_path: str) -> str:
    """
    Post-process raw Kokoro TTS output: normalize loudness and optionally overlay background music.

    Args:
        wav_path: Path to the raw WAV file (24kHz, mono)

    Returns:
        Path to the mastered WAV file, or None if processing failed
    """
    try:
        audio = AudioSegment.from_wav(wav_path)

        # Normalize loudness to -16 dBFS (podcast standard)
        target_dBFS = -16.0
        change = target_dBFS - audio.dBFS
        audio = audio.apply_gain(change)
        print(f"  Normalized loudness: {audio.dBFS:.1f} dBFS (target: {target_dBFS})")

        # Optional: overlay background music if available
        script_dir = Path(__file__).parent
        bg_music_path = script_dir / "asset" / "background_music.mp3"
        if bg_music_path.exists():
            try:
                bg_music = AudioSegment.from_mp3(str(bg_music_path))
                # Loop background music to match speech duration
                while len(bg_music) < len(audio):
                    bg_music = bg_music + bg_music
                bg_music = bg_music[:len(audio)]
                # Duck background music to -25 dB relative
                bg_music = bg_music.apply_gain(-25 - bg_music.dBFS)
                audio = audio.overlay(bg_music)
                print(f"  Background music overlaid from: {bg_music_path}")
            except Exception as e:
                logger.warning(f"Failed to overlay background music: {e}")
                print(f"  ⚠ Background music overlay failed: {e}")

        # Export mastered file
        mastered_path = wav_path.replace(".wav", "_mastered.wav")
        if mastered_path == wav_path:
            mastered_path = wav_path + "_mastered.wav"
        audio.export(mastered_path, format="wav")
        print(f"  Mastered audio saved: {mastered_path}")
        return mastered_path

    except Exception as e:
        logger.warning(f"Audio post-processing failed: {e}")
        print(f"  ⚠ Audio post-processing failed (using raw audio): {e}")
        return None


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
