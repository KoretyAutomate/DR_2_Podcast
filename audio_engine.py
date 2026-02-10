"""
Kokoro TTS Audio Engine for Deep Research Podcast
==================================================

Generates high-quality, multi-speaker podcast audio using Kokoro-82M (local TTS).

Features:
- Dual-voice system: Host 1 (bm_george - British Male Expert)
                     Host 2 (af_nicole - American Female Skeptic)
- Script parsing with speaker detection
- Audio stitching and WAV export
"""

import soundfile as sf
from kokoro import KPipeline
import torch
import numpy as np
import re
from pathlib import Path

# Voice Configuration
VOICE_HOST_1 = 'bm_george'  # British Male (The Expert)
VOICE_HOST_2 = 'af_nicole'  # American Female (The Skeptic)
LANG_CODE = 'a'  # American English

def generate_audio_from_script(script_text: str, output_filename: str = "final_podcast.wav") -> str:
    """
    Parses a script looking for 'Host 1:' and 'Host 2:' lines,
    generates audio segments using Kokoro, and stitches them together.

    Args:
        script_text: Full podcast script with "Host 1:" and "Host 2:" labels
        output_filename: Output WAV file name (default: "final_podcast.wav")

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

    # 1. Initialize Pipeline
    device = 'cpu'
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            device = 'cuda'
        except RuntimeError:
            print("  CUDA reported available but kernel execution failed, falling back to CPU")
    print(f"Device: {device}")
    print(f"Voices: Host 1 ({VOICE_HOST_1}), Host 2 ({VOICE_HOST_2})")

    try:
        pipeline = KPipeline(lang_code=LANG_CODE, device=device)
        print("✓ Kokoro pipeline initialized")
    except RuntimeError as e:
        if 'CUDA' in str(e) and device == 'cuda':
            print(f"  CUDA init failed, retrying on CPU: {e}")
            device = 'cpu'
            pipeline = KPipeline(lang_code=LANG_CODE, device=device)
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

    current_speaker = None
    buffer_text = ""
    segment_count = 0

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        # Check for Speaker Switch
        if line.startswith("Host 1:") or line.startswith("Dr. Data:") or line.startswith("Kaz:"):
            # Process previous buffer
            if buffer_text and current_speaker:
                voice = VOICE_HOST_1 if current_speaker == 1 else VOICE_HOST_2
                try:
                    generator = pipeline(buffer_text, voice=voice, speed=1.0, split_pattern=r'\n+')
                    for _, _, audio in generator:
                        audio_segments.append(audio)
                        segment_count += 1
                except Exception as e:
                    print(f"  ⚠ Warning: Failed to generate segment {segment_count}: {e}")

            current_speaker = 1
            buffer_text = line.split(":", 1)[1].strip() if ":" in line else ""

        elif line.startswith("Host 2:") or line.startswith("Dr. Doubt:") or line.startswith("Erika:"):
            # Process previous buffer
            if buffer_text and current_speaker:
                voice = VOICE_HOST_1 if current_speaker == 1 else VOICE_HOST_2
                try:
                    generator = pipeline(buffer_text, voice=voice, speed=1.0, split_pattern=r'\n+')
                    for _, _, audio in generator:
                        audio_segments.append(audio)
                        segment_count += 1
                except Exception as e:
                    print(f"  ⚠ Warning: Failed to generate segment {segment_count}: {e}")

            current_speaker = 2
            buffer_text = line.split(":", 1)[1].strip() if ":" in line else ""

        else:
            # Continuation of current speaker
            if buffer_text:
                buffer_text += " " + line
            else:
                buffer_text = line

    # Process final buffer
    if buffer_text and current_speaker:
        voice = VOICE_HOST_1 if current_speaker == 1 else VOICE_HOST_2
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

    # Normalize whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()

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
