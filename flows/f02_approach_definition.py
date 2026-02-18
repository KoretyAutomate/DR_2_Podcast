"""
flows/f02_approach_definition.py — Decide which sub-flows to activate.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path (for direct execution)
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from shared.models import PipelineParams, PipelineApproach
from shared.config import (
    create_timestamped_output_dir,
    copy_research_artifacts,
    copy_all_artifacts,
    check_supplemental_needed,
)


def define_approach(params: PipelineParams) -> PipelineApproach:
    """Given pipeline params, decide which sub-flows to run.

    Returns a PipelineApproach with boolean flags and output_dir.
    """
    approach = PipelineApproach()
    approach.output_dir = create_timestamped_output_dir()

    if params.crew3_only and params.reuse_dir:
        # --- Crew 3 only: skip research + validation ---
        print("Approach: CREW3_ONLY (skip research, reuse artifacts)")
        approach.run_research = False
        approach.run_validation = False
        approach.reuse_dir = Path(params.reuse_dir)
        copy_research_artifacts(approach.reuse_dir, approach.output_dir)

    elif params.check_supplemental and params.reuse_dir:
        # --- Check supplemental: LLM decides ---
        print("Approach: CHECK_SUPPLEMENTAL")
        approach.reuse_dir = Path(params.reuse_dir)
        result = check_supplemental_needed(params.topic, approach.reuse_dir)
        approach.supplemental_needed = result["needs_supplement"]
        print(f"  Needs supplement: {result['needs_supplement']}")
        print(f"  Reason: {result['reason']}")

        if not result["needs_supplement"]:
            approach.run_research = False
            approach.run_validation = False
            copy_all_artifacts(approach.reuse_dir, approach.output_dir)
        else:
            copy_research_artifacts(approach.reuse_dir, approach.output_dir)
            # Stash queries for f03 to use
            approach._supplemental_queries = result.get("queries", [])

    else:
        # --- Full pipeline from scratch ---
        print("Approach: FULL PIPELINE (all phases)")

    # Translation only for non-English
    approach.run_translation = (params.language != "en")

    # Upload only if flags set
    approach.run_upload = (params.upload_buzzsprout or params.upload_youtube)

    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE APPROACH")
    print(f"{'='*60}")
    print(f"  Research:        {'YES' if approach.run_research else 'SKIP'}")
    print(f"  Validation:      {'YES' if approach.run_validation else 'SKIP'}")
    print(f"  Translation:     {'YES' if approach.run_translation else 'SKIP'}")
    print(f"  Podcast Planning:{'YES' if approach.run_podcast_planning else 'SKIP'}")
    print(f"  Audio:           {'YES' if approach.run_audio else 'SKIP'}")
    print(f"  Upload:          {'YES' if approach.run_upload else 'SKIP'}")
    print(f"  Output dir:      {approach.output_dir}")
    if approach.reuse_dir:
        print(f"  Reuse from:      {approach.reuse_dir}")
    print(f"{'='*60}\n")

    return approach


if __name__ == "__main__":
    params = PipelineParams(topic="test topic")
    approach = define_approach(params)
    print(f"Approach created: {approach}")
