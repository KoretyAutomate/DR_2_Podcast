"""
flows/f07_audio_generation.py — TTS audio generation + BGM merge.

Generates podcast audio from polished script using Kokoro TTS,
then mixes in background music.
"""

import os
import re
import wave
from pathlib import Path

from shared.models import PipelineParams, AudioResult
from shared.config import SUPPORTED_LANGUAGES, get_length_targets
from audio_engine import generate_audio_from_script, clean_script_for_tts, post_process_audio


def run_audio_generation(params: PipelineParams, script_polished: str, output_dir: Path) -> AudioResult:
    """Generate TTS audio and merge BGM.

    1. Clean script for TTS
    2. Duration / length check
    3. Kokoro TTS generation
    4. BGM mixing
    5. Duration verification
    """
    result = AudioResult(output_dir=output_dir)
    language = params.language
    language_config = SUPPORTED_LANGUAGES[language]
    length_targets = get_length_targets(language, params.podcast_length)

    # --- Measure script length ---
    if language == "ja":
        char_count = len(re.sub(r'[\s\n\r\t\u3000\uff1a:\u300c\u300d\u3001\u3002\u30fb\uff08\uff09\-\u2014\*#]', '', script_polished))
        script_length = char_count
        estimated_duration_min = char_count / 500
    else:
        content_only = re.sub(r'^[A-Za-z0-9_ ]+:\s*', '', script_polished, flags=re.MULTILINE)
        script_length = len(content_only.split())
        estimated_duration_min = script_length / 150

    result.script_length = script_length

    print(f"\n{'='*60}")
    print("DURATION CHECK")
    print(f"{'='*60}")
    print(f"Script length: {script_length} {length_targets['unit']}")
    print(f"Estimated duration: {estimated_duration_min:.1f} minutes")
    print(f"Target: {length_targets['target']} {length_targets['unit']}")

    if script_length < length_targets["low"]:
        print(f"WARNING: Script is SHORT ({script_length} < {length_targets['target']})")
    elif script_length > length_targets["high"]:
        print(f"WARNING: Script is LONG ({script_length} > {length_targets['target']})")
    else:
        print(f"Script length GOOD ({script_length} {length_targets['unit']})")
    print(f"{'='*60}\n")

    # --- Clean and save script ---
    cleaned_script = clean_script_for_tts(script_polished)

    script_file = output_dir / "podcast_script.txt"
    with open(script_file, 'w') as f:
        f.write(script_polished)
    print(f"Podcast script saved: {script_file} ({script_length} {length_targets['unit']})")

    # --- TTS generation ---
    print("\n--- Generating Multi-Voice Podcast Audio (Kokoro TTS) ---")
    output_path = output_dir / "podcast_final_audio.wav"
    audio_file = None

    try:
        print(f"Starting audio generation with script length: {len(cleaned_script)} chars")
        audio_file = generate_audio_from_script(
            cleaned_script, str(output_path), lang_code=language_config['tts_code']
        )
        if audio_file:
            audio_file = Path(audio_file)
            print(f"Audio generation complete: {audio_file}")
    except Exception as e:
        print(f"ERROR: Kokoro TTS failed: {e}")
        import traceback
        traceback.print_exc()
        print("  Ensure Kokoro is installed: pip install kokoro>=0.9")
        return result

    # --- BGM mixing ---
    if audio_file and audio_file.exists():
        print("Starting BGM Merging Phase...")
        try:
            mastered = post_process_audio(str(audio_file), bgm_target="Interesting BGM.wav")
            if mastered and os.path.exists(mastered) and mastered != str(audio_file):
                audio_file = Path(mastered)
                print(f"BGM Merging Complete: {audio_file}")
        except Exception as e:
            print(f"BGM merging warning: {e}")

    # --- Duration verification ---
    if audio_file and audio_file.exists():
        result.audio_path = audio_file
        try:
            with wave.open(str(audio_file), 'r') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                duration_seconds = frames / float(rate)
                duration_minutes = duration_seconds / 60

            result.duration_minutes = duration_minutes

            print(f"\n{'='*60}")
            print("AUDIO DURATION VERIFICATION")
            print(f"{'='*60}")
            print(f"Actual audio duration: {duration_minutes:.2f} minutes ({duration_seconds:.1f} seconds)")

            target_min = length_targets["low"] / (500 if language == "ja" else 150)
            target_max = length_targets["high"] / (500 if language == "ja" else 150) * 1.2
            if duration_minutes < target_min:
                print(f"FAILED: Audio TOO SHORT ({duration_minutes:.2f} min)")
            elif duration_minutes > target_max:
                print(f"FAILED: Audio TOO LONG ({duration_minutes:.2f} min)")
            else:
                print("SUCCESS: Audio duration within acceptable range")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"Duration check warning: {e}")

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m flows.f07_audio_generation <script_file> [language]")
        sys.exit(1)

    script_path = Path(sys.argv[1])
    lang = sys.argv[2] if len(sys.argv) > 2 else "en"

    from shared.config import check_tts_dependencies, create_timestamped_output_dir
    check_tts_dependencies()

    params = PipelineParams(topic="test", language=lang)
    out = create_timestamped_output_dir()
    result = run_audio_generation(params, script_path.read_text(), out)
    print(f"\nResult: audio_path={result.audio_path}, duration={result.duration_minutes}")
