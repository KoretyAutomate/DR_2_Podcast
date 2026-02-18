#!/usr/bin/env python3
"""
orchestrator.py — Central pipeline controller for DR_2_Podcast.

Chains sub-flows based on the approach defined by f02.
Backward-compatible CLI entry point.

Usage:
  python orchestrator.py --topic "effects of coffee" --language en
  python orchestrator.py --topic "..." --reuse-dir DIR --crew3-only
  python orchestrator.py --topic "..." --reuse-dir DIR --check-supplemental
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from shared.models import PipelineParams
from shared.config import (
    SUPPORTED_LANGUAGES,
    assign_roles,
    check_tts_dependencies,
    SCRIPT_DIR,
)
from shared.logging_setup import setup_logging
from shared.progress import (
    ProgressTracker,
    TASK_METADATA,
    display_workflow_plan,
)
from shared.pdf_utils import save_pdf_safe

from flows.f01_parameter_gathering import gather_params_from_cli
from flows.f02_approach_definition import define_approach
from flows.f03_evidence_gathering import run_evidence_gathering
from flows.f04_evidence_validation import run_evidence_validation
from flows.f05_translation import run_translation
from flows.f06_podcast_planning import run_podcast_planning
from flows.f07_audio_generation import run_audio_generation
from flows.f08_upload import run_upload


def run_pipeline(params: PipelineParams) -> dict:
    """Execute the full podcast generation pipeline.

    Returns a summary dict with output_dir and key results.
    """
    language = params.language
    language_config = SUPPORTED_LANGUAGES[language]
    session_roles = assign_roles(params.host_order)

    print(f"\n{'='*70}")
    print(" " * 15 + "DR_2_PODCAST PIPELINE")
    print(f"{'='*70}")
    print(f"Topic:    {params.topic}")
    print(f"Language: {language_config['name']} ({language})")
    print(f"Hosts:    {session_roles['presenter']['character']} (Presenter), "
          f"{session_roles['questioner']['character']} (Questioner)")
    print(f"{'='*70}\n")

    # --- Step 1: TTS check ---
    check_tts_dependencies()

    # --- Step 2: Define approach ---
    approach = define_approach(params)
    output_dir = approach.output_dir

    # Setup logging
    setup_logging(output_dir)

    # Display workflow plan
    display_workflow_plan(TASK_METADATA, params.topic, language_config, output_dir)

    # Initialize progress tracker
    progress_tracker = ProgressTracker(TASK_METADATA)
    progress_tracker.start_workflow()

    # Track results for summary
    summary = {
        "topic": params.topic,
        "language": language,
        "output_dir": str(output_dir),
        "approach": {
            "run_research": approach.run_research,
            "run_validation": approach.run_validation,
            "run_translation": approach.run_translation,
            "run_podcast_planning": approach.run_podcast_planning,
            "run_audio": approach.run_audio,
            "run_upload": approach.run_upload,
        },
    }

    evidence = None
    validation = None
    source_of_truth = ""
    supporting_research = ""
    adversarial_research = ""

    # --- Step 3: Evidence Gathering (f03) ---
    if approach.run_research:
        evidence = run_evidence_gathering(params, approach)
        supporting_research = evidence.supporting_research
        summary["gate_passed"] = evidence.gate_passed
    else:
        # Load source_of_truth from reused artifacts
        sot_path = output_dir / "source_of_truth.md"
        if not sot_path.exists():
            sot_path = output_dir / "SOURCE_OF_TRUTH.md"
        if sot_path.exists():
            source_of_truth = sot_path.read_text()
        print(f"Research skipped (reuse mode). Source-of-truth: {len(source_of_truth)} chars")

    # --- Step 4: Evidence Validation (f04) ---
    if approach.run_validation and evidence:
        validation = run_evidence_validation(params, evidence)
        source_of_truth = validation.source_of_truth
        adversarial_research = validation.adversarial_research
    elif evidence and not approach.run_validation:
        # If we did research but not validation (unusual), use framing as SOT
        source_of_truth = evidence.supporting_research

    # --- Step 5: Translation (f05) ---
    if approach.run_translation:
        translation = run_translation(
            params,
            source_of_truth=source_of_truth,
            supporting_research=supporting_research,
            output_dir=output_dir,
            session_roles=session_roles,
        )
        source_of_truth = translation.translated_source_of_truth
        supporting_research = translation.translated_supporting

    # --- Step 6: Podcast Planning (f06) ---
    planning = None
    if approach.run_podcast_planning:
        planning = run_podcast_planning(
            params,
            source_of_truth=source_of_truth,
            supporting_research=supporting_research,
            adversarial_research=adversarial_research,
            output_dir=output_dir,
        )
        summary["script_length"] = len(planning.script_polished) if planning.script_polished else 0

    # --- Step 7: Audio Generation (f07) ---
    audio = None
    if approach.run_audio and planning and planning.script_polished:
        audio = run_audio_generation(params, planning.script_polished, output_dir)
        summary["audio_path"] = str(audio.audio_path) if audio.audio_path else None
        summary["duration_minutes"] = audio.duration_minutes

    # --- Step 8: Upload (f08) ---
    if approach.run_upload and audio and audio.audio_path:
        upload_title = f"DR2 Podcast: {params.topic}"
        upload_result = run_upload(params, audio.audio_path, upload_title)
        summary["upload"] = {
            "buzzsprout": upload_result.buzzsprout,
            "youtube": upload_result.youtube,
        }

    # --- Session metadata ---
    print("\n--- Documenting Session Metadata ---")
    reuse_info = ""
    if approach.reuse_dir:
        reuse_info = f"Reused from: {approach.reuse_dir}\n"
    session_metadata = (
        f"PODCAST SESSION METADATA\n{'='*60}\n\n"
        f"Topic: {params.topic}\n\n"
        f"Language: {language_config['name']} ({language})\n\n"
        f"{reuse_info}"
        f"Character Assignments:\n"
        f"  {session_roles['presenter']['character']}: Presenter "
        f"({session_roles['presenter']['personality']})\n"
        f"  {session_roles['questioner']['character']}: Questioner "
        f"({session_roles['questioner']['personality']})\n"
    )
    metadata_file = output_dir / "session_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write(session_metadata)
    print(f"Session metadata: {metadata_file}")

    # --- Deep research summary ---
    if evidence and evidence.deep_reports:
        dr = evidence.deep_reports
        print(f"\n--- Deep Research Summary ---")
        if "lead" in dr:
            print(f"  Lead sources: {dr['lead'].total_summaries}")
        if "counter" in dr:
            print(f"  Counter sources: {dr['counter'].total_summaries}")
        if "audit" in dr:
            print(f"  Audit sources: {dr['audit'].total_summaries}")

    # --- Final summary ---
    progress_tracker.workflow_completed()

    print(f"\n{'='*70}")
    print(" " * 15 + "PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    if audio and audio.audio_path:
        print(f"Audio file: {audio.audio_path}")
        if audio.duration_minutes:
            print(f"Duration: {audio.duration_minutes:.1f} minutes")
    print(f"{'='*70}\n")

    return summary


if __name__ == "__main__":
    params = gather_params_from_cli()
    result = run_pipeline(params)
    print(json.dumps(result, indent=2, default=str))
