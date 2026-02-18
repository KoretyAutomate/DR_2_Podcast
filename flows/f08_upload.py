"""
flows/f08_upload.py — Upload to Buzzsprout / YouTube.

Wraps upload_utils.py with validation and result reporting.
"""

from pathlib import Path

from shared.models import PipelineParams, UploadResult
from upload_utils import validate_upload_config, upload_to_buzzsprout, upload_to_youtube


def run_upload(params: PipelineParams, audio_path: Path, title: str) -> UploadResult:
    """Upload the generated audio to configured platforms.

    Returns an UploadResult with per-platform success/failure info.
    """
    result = UploadResult()

    if not audio_path or not audio_path.exists():
        print("Upload skipped: no audio file available")
        return result

    # --- Validate config ---
    validation = validate_upload_config(
        buzzsprout=params.upload_buzzsprout,
        youtube=params.upload_youtube,
    )
    if not validation["valid"]:
        for err in validation["errors"]:
            print(f"Upload config error: {err}")
        return result

    # --- Buzzsprout ---
    if params.upload_buzzsprout:
        print(f"\n--- Uploading to Buzzsprout ---")
        bz_result = upload_to_buzzsprout(str(audio_path), title)
        result.buzzsprout = bz_result
        if bz_result["success"]:
            print(f"Buzzsprout upload OK: {bz_result['url']}")
        else:
            print(f"Buzzsprout upload FAILED: {bz_result['error']}")

    # --- YouTube ---
    if params.upload_youtube:
        print(f"\n--- Uploading to YouTube ---")
        yt_result = upload_to_youtube(str(audio_path), title, privacy="private")
        result.youtube = yt_result
        if yt_result["success"]:
            print(f"YouTube upload OK: {yt_result['url']}")
        else:
            print(f"YouTube upload FAILED: {yt_result['error']}")

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m flows.f08_upload <audio_file> <title> [--buzzsprout] [--youtube]")
        sys.exit(1)

    audio = Path(sys.argv[1])
    ep_title = sys.argv[2]
    params = PipelineParams(
        topic=ep_title,
        upload_buzzsprout="--buzzsprout" in sys.argv,
        upload_youtube="--youtube" in sys.argv,
    )
    r = run_upload(params, audio, ep_title)
    print(f"\nResult: buzzsprout={r.buzzsprout}, youtube={r.youtube}")
