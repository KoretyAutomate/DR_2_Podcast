"""Regenerate Episode 1 audio using VOICEVOX TTS.

Uses the same manually written script from edu_ep01_final,
but routes through VOICEVOX instead of Qwen3-TTS.
"""
from pathlib import Path

SRC_DIR = Path("/home/korety/Project/DR_2_Podcast/research_outputs/edu_ep01_final")
RUN_DIR = Path("/home/korety/Project/DR_2_Podcast/research_outputs/edu_ep01_voicevox")

from dr2_podcast import pipeline as P
from dr2_podcast.pipeline import SUPPORTED_LANGUAGES, _finalize_script, _run_audio_pipeline

P.language = "ja"
P.language_config = SUPPORTED_LANGUAGES["ja"]

# Load the original script draft
draft_path = SRC_DIR / "scripts" / "script_draft.md"
draft_text = draft_path.read_text(encoding="utf-8")
print(f"Draft loaded: {len(draft_text)} chars, "
      f"{draft_text.count('Host 1:')} Host 1 / {draft_text.count('Host 2:')} Host 2 turns")

# Copy to new output dir
for fname in ("script_draft.md", "script_polished.md"):
    (RUN_DIR / "scripts" / fname).write_text(draft_text, encoding="utf-8")

# Phase 7: finalize
print("\n--- Finalizing script ---")
script_text = _finalize_script(
    polished_text=draft_text,
    polish_task=None,
    language="ja",
    language_config=SUPPORTED_LANGUAGES["ja"],
    output_dir=RUN_DIR,
    corrected_text=None,
)
print(f"Finalized: {len(script_text)} chars")

# Phase 8: audio (will auto-detect VOICEVOX and use it)
print("\n--- Generating audio (VOICEVOX) ---")
audio_file, duration_min = _run_audio_pipeline(
    script_text, RUN_DIR, SUPPORTED_LANGUAGES["ja"]
)
if duration_min:
    print(f"\nAudio: {audio_file}")
    print(f"Duration: {duration_min:.2f} min")
else:
    print(f"\nAudio: {audio_file} (duration unknown)")
