"""One-shot regeneration helper for run 2026-04-12_11-37-35.

Uses the existing script_draft.md (which has correct Host 1:/Host 2: labels)
as the polished text, runs language audit + reaction guidance, then regenerates
audio via the TTS pipeline. This is the same flow that _run_polish_loop's new
speaker-label guard would trigger automatically on future runs.
"""
import sys
from pathlib import Path

RUN_DIR = Path("/home/korety/Project/DR_2_Podcast/research_outputs/2026-04-12_11-37-35")

from dr2_podcast import pipeline as P
from dr2_podcast.pipeline import SUPPORTED_LANGUAGES, _finalize_script, _run_audio_pipeline

# The finalize/audio helpers expect module-level `language` / `language_config`
P.language = "ja"
P.language_config = SUPPORTED_LANGUAGES["ja"]

draft_path = RUN_DIR / "scripts" / "script_draft.md"
draft_text = draft_path.read_text(encoding="utf-8")
print(f"Draft loaded: {len(draft_text)} chars, "
      f"{draft_text.count('Host 1:')} Host 1 / {draft_text.count('Host 2:')} Host 2 labels")

# Re-save script_polished.md using the draft (bypassing the busted polish output).
polished_path = RUN_DIR / "scripts" / "script_polished.md"
polished_path.write_text(draft_text, encoding="utf-8")
print(f"Overwrote {polished_path}")

# Phase: finalize (language audit + reaction guidance + save script_final.md)
script_text = _finalize_script(
    polished_text=draft_text,
    polish_task=None,
    language="ja",
    language_config=SUPPORTED_LANGUAGES["ja"],
    output_dir=RUN_DIR,
    corrected_text=None,
)
print(f"Finalized script: {len(script_text)} chars, "
      f"{script_text.count('Host 1:')} Host 1 / {script_text.count('Host 2:')} Host 2 labels")

# Phase 8: audio regeneration
audio_file, duration_min = _run_audio_pipeline(
    script_text, RUN_DIR, SUPPORTED_LANGUAGES["ja"]
)
print(f"Audio file: {audio_file}  duration: {duration_min:.2f} min"
      if duration_min else f"Audio file: {audio_file} (duration unknown)")
