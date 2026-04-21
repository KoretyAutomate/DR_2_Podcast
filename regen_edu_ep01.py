"""Regeneration script for Educational Episode 1.

Loads the manually written script, runs language audit + reaction guidance
via _finalize_script(), then generates audio via _run_audio_pipeline()
using Qwen3-TTS for Japanese.
"""
from pathlib import Path

RUN_DIR = Path("/home/korety/Project/DR_2_Podcast/research_outputs/edu_ep01_final")

from dr2_podcast import pipeline as P
from dr2_podcast.pipeline import SUPPORTED_LANGUAGES, _finalize_script, _run_audio_pipeline

# Set module-level language config
P.language = "ja"
P.language_config = SUPPORTED_LANGUAGES["ja"]

# Load the manually written script
draft_path = RUN_DIR / "scripts" / "script_draft.md"
draft_text = draft_path.read_text(encoding="utf-8")
print(f"Draft loaded: {len(draft_text)} chars, "
      f"{draft_text.count('Host 1:')} Host 1 / {draft_text.count('Host 2:')} Host 2 turns")

# Save as script_polished.md (bypassing CrewAI polish loop)
polished_path = RUN_DIR / "scripts" / "script_polished.md"
polished_path.write_text(draft_text, encoding="utf-8")
print(f"Saved polished copy: {polished_path}")

# Phase 7: finalize (language audit + reaction guidance + save script_final.md)
print("\n--- Finalizing script (language audit + reaction guidance) ---")
script_text = _finalize_script(
    polished_text=draft_text,
    polish_task=None,
    language="ja",
    language_config=SUPPORTED_LANGUAGES["ja"],
    output_dir=RUN_DIR,
    corrected_text=None,
)
print(f"Finalized script: {len(script_text)} chars, "
      f"{script_text.count('Host 1:')} Host 1 / {script_text.count('Host 2:')} Host 2 turns")

# Phase 8: audio generation (TTS + BGM mixing)
print("\n--- Generating audio ---")
audio_file, duration_min = _run_audio_pipeline(
    script_text, RUN_DIR, SUPPORTED_LANGUAGES["ja"]
)
if duration_min:
    print(f"\nAudio file: {audio_file}")
    print(f"Duration: {duration_min:.2f} min")
else:
    print(f"\nAudio file: {audio_file} (duration unknown)")
