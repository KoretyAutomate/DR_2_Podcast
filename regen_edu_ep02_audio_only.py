"""Regenerate Episode 2 audio — SKIPS _finalize_script().

The manually-written ep02 script already contains ## [emotion] cues, so we
bypass _finalize_script() (which would call the Smart Model via _call_smart_model
to add MORE cues) and go straight to _run_audio_pipeline(). clean_script_for_tts()
strips ## lines before TTS anyway.
"""
from pathlib import Path

SCRIPT_PATH = Path("/home/korety/Project/DR_2_Podcast/educational_series/ep02_script_draft.md")
RUN_DIR = Path("/home/korety/Project/DR_2_Podcast/research_outputs/edu_ep02_final")

from dr2_podcast import pipeline as P
from dr2_podcast.pipeline import SUPPORTED_LANGUAGES, _run_audio_pipeline, output_path

P.language = "ja"
P.language_config = SUPPORTED_LANGUAGES["ja"]

draft_text = SCRIPT_PATH.read_text(encoding="utf-8")
print(f"Draft loaded: {len(draft_text)} chars, "
      f"{draft_text.count('Host 1:')} Host 1 / {draft_text.count('Host 2:')} Host 2 turns")

for fname in ("script_draft.md", "script_polished.md", "script_final.md"):
    (RUN_DIR / "scripts" / fname).write_text(draft_text, encoding="utf-8")
print("Scripts mirrored to scripts/ (draft, polished, final)")

print("\n--- Generating audio (VOICEVOX) ---")
audio_file, duration_min = _run_audio_pipeline(
    draft_text, RUN_DIR, SUPPORTED_LANGUAGES["ja"]
)
if duration_min:
    print(f"\nAudio: {audio_file}")
    print(f"Duration: {duration_min:.2f} min")
else:
    print(f"\nAudio: {audio_file} (duration unknown)")
