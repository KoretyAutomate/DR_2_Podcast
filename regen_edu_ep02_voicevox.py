"""Regenerate Episode 2 audio using VOICEVOX TTS.

Educational series Day 2: The Evidence Hierarchy (証拠の階段).
Loads the manually written script, runs _finalize_script() to add
reaction guidance tags, then _run_audio_pipeline() for VOICEVOX + BGM.
"""
from pathlib import Path

SCRIPT_PATH = Path("/home/korety/Project/DR_2_Podcast/educational_series/ep02_script_draft.md")
RUN_DIR = Path("/home/korety/Project/DR_2_Podcast/research_outputs/edu_ep02_final")

from dr2_podcast import pipeline as P
from dr2_podcast.pipeline import SUPPORTED_LANGUAGES, _finalize_script, _run_audio_pipeline

P.language = "ja"
P.language_config = SUPPORTED_LANGUAGES["ja"]

draft_text = SCRIPT_PATH.read_text(encoding="utf-8")
print(f"Draft loaded: {len(draft_text)} chars, "
      f"{draft_text.count('Host 1:')} Host 1 / {draft_text.count('Host 2:')} Host 2 turns")

for fname in ("script_draft.md", "script_polished.md"):
    (RUN_DIR / "scripts" / fname).write_text(draft_text, encoding="utf-8")

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

print("\n--- Generating audio (VOICEVOX) ---")
audio_file, duration_min = _run_audio_pipeline(
    script_text, RUN_DIR, SUPPORTED_LANGUAGES["ja"]
)
if duration_min:
    print(f"\nAudio: {audio_file}")
    print(f"Duration: {duration_min:.2f} min")
else:
    print(f"\nAudio: {audio_file} (duration unknown)")
