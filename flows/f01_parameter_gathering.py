"""
flows/f01_parameter_gathering.py — Parameter collection from CLI or Web UI.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path (for direct execution)
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import os
import argparse
from typing import Optional

from shared.models import PipelineParams


def gather_params_from_cli() -> PipelineParams:
    """Parse CLI arguments + environment variables into PipelineParams."""
    parser = argparse.ArgumentParser(
        description='Generate a research-driven debate podcast on any scientific topic.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orchestrator.py --topic "effects of meditation on brain plasticity" --language en
  python orchestrator.py --topic "climate change impact on marine ecosystems" --language ja

Environment variables:
  PODCAST_TOPIC, PODCAST_LANGUAGE, PODCAST_HOSTS, ACCESSIBILITY_LEVEL, PODCAST_LENGTH
        """,
    )
    parser.add_argument('--topic', type=str, help='Scientific topic for podcast research')
    parser.add_argument('--language', type=str, choices=['en', 'ja'], help='Language (en/ja)')
    parser.add_argument('--reuse-dir', type=str, help='Previous run directory to reuse research from')
    parser.add_argument('--crew3-only', action='store_true', help='Skip research, run podcast production only')
    parser.add_argument('--check-supplemental', action='store_true', help='LLM decides if supplemental research needed')
    parser.add_argument('--upload-buzzsprout', action='store_true', help='Upload to Buzzsprout')
    parser.add_argument('--upload-youtube', action='store_true', help='Upload to YouTube')
    args = parser.parse_args()

    # Topic: CLI > env > default
    topic = args.topic or os.getenv("PODCAST_TOPIC", "scientific benefit of coffee intake to increase productivity during the day")

    # Language: CLI > env > default
    language = args.language or os.getenv("PODCAST_LANGUAGE", "en")
    if language not in ("en", "ja"):
        language = "en"

    # Host order
    host_order = os.getenv("PODCAST_HOSTS", "random")

    # Accessibility
    accessibility = os.getenv("ACCESSIBILITY_LEVEL", "simple").lower()
    if accessibility not in ("simple", "moderate", "technical"):
        accessibility = "simple"

    # Length
    podcast_length = os.getenv("PODCAST_LENGTH", "long").lower()
    if podcast_length not in ("short", "medium", "long"):
        podcast_length = "long"

    return PipelineParams(
        topic=topic,
        language=language,
        host_order=host_order,
        accessibility_level=accessibility,
        reuse_dir=args.reuse_dir,
        crew3_only=args.crew3_only,
        check_supplemental=args.check_supplemental,
        upload_buzzsprout=args.upload_buzzsprout,
        upload_youtube=args.upload_youtube,
        podcast_length=podcast_length,
    )


def gather_params_from_web(request_data: dict) -> PipelineParams:
    """Build PipelineParams from a Web UI POST body (dict)."""
    return PipelineParams(
        topic=request_data.get("topic", ""),
        language=request_data.get("language", "en"),
        host_order=request_data.get("host_order", "random"),
        accessibility_level=request_data.get("accessibility_level", "simple"),
        reuse_dir=request_data.get("reuse_dir"),
        crew3_only=request_data.get("crew3_only", False),
        check_supplemental=request_data.get("check_supplemental", False),
        upload_buzzsprout=request_data.get("upload_buzzsprout", False),
        upload_youtube=request_data.get("upload_youtube", False),
        podcast_length=request_data.get("podcast_length", "long"),
    )


if __name__ == "__main__":
    params = gather_params_from_cli()
    print(f"Topic: {params.topic}")
    print(f"Language: {params.language}")
    print(f"Host order: {params.host_order}")
    print(f"Accessibility: {params.accessibility_level}")
    print(f"Reuse dir: {params.reuse_dir}")
    print(f"Crew3 only: {params.crew3_only}")
    print(f"Check supplemental: {params.check_supplemental}")
    print(f"Upload BZ: {params.upload_buzzsprout}, YT: {params.upload_youtube}")
    print(f"Length: {params.podcast_length}")
