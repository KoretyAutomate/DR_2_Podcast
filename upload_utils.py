"""
Upload utilities for DR_2_Podcast — Buzzsprout and YouTube draft uploads.

YOUTUBE SETUP
-------------
1. Go to https://console.cloud.google.com
2. Create a project (or use an existing one)
3. Enable "YouTube Data API v3":
       APIs & Services -> Library -> search "YouTube Data API v3" -> Enable
4. Create an OAuth 2.0 Client ID:
       APIs & Services -> Credentials -> Create Credentials
       -> OAuth 2.0 Client ID -> Desktop application
5. Download the client_secret.json and place it in this project's root directory
6. Add to .env:
       YOUTUBE_CLIENT_SECRET_PATH=./client_secret.json
7. First upload will open a browser window for Google consent.
   After you grant access, the token is cached to youtube_token.pickle
   and subsequent uploads use it automatically.
"""

import os
import pickle
from pathlib import Path

import httpx

SCRIPT_DIR = Path(__file__).parent.absolute()

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def validate_upload_config(buzzsprout: bool, youtube: bool) -> dict:
    """Check that env vars and files exist for the requested platforms."""
    errors = []

    if buzzsprout:
        if not os.getenv("BUZZSPROUT_API_KEY"):
            errors.append("BUZZSPROUT_API_KEY is not set in .env")
        if not os.getenv("BUZZSPROUT_ACCOUNT_ID"):
            errors.append("BUZZSPROUT_ACCOUNT_ID is not set in .env")

    if youtube:
        secret_path = os.getenv("YOUTUBE_CLIENT_SECRET_PATH", "./client_secret.json")
        if not (SCRIPT_DIR / secret_path).exists():
            errors.append(
                f"YouTube client_secret.json not found at {SCRIPT_DIR / secret_path}. "
                "See YOUTUBE SETUP instructions in upload_utils.py"
            )

    return {"valid": len(errors) == 0, "errors": errors}

# ---------------------------------------------------------------------------
# Buzzsprout
# ---------------------------------------------------------------------------

def upload_to_buzzsprout(audio_path: str, title: str) -> dict:
    """Upload audio as an unpublished (draft) episode to Buzzsprout."""
    api_key = os.getenv("BUZZSPROUT_API_KEY")
    account_id = os.getenv("BUZZSPROUT_ACCOUNT_ID")

    url = f"https://api.buzzsprout.com/v1/podcasts/{account_id}/episodes"
    headers = {"Authorization": f"Token token={api_key}"}

    try:
        with open(audio_path, "rb") as f:
            files = {"audio_file": (Path(audio_path).name, f, "audio/mpeg")}
            data = {"title": title}
            # No published_at → episode stays as draft
            response = httpx.post(url, headers=headers, files=files, data=data, timeout=120)

        response.raise_for_status()
        body = response.json()
        episode_id = body.get("id")
        episode_url = f"https://www.buzzsprout.com/podcasts/{account_id}/episodes/{episode_id}"
        return {"success": True, "episode_id": str(episode_id), "url": episode_url, "error": None}

    except Exception as e:
        return {"success": False, "episode_id": None, "url": None, "error": str(e)}

# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------

_YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
_TOKEN_PATH = SCRIPT_DIR / "youtube_token.pickle"


def get_youtube_credentials():
    """
    Return cached or freshly-obtained YouTube OAuth credentials.
    May open a browser on first run — call from a request thread, not a daemon thread.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.oauth2.credentials import Credentials

    secret_path = os.getenv("YOUTUBE_CLIENT_SECRET_PATH", "./client_secret.json")
    secret_full = SCRIPT_DIR / secret_path

    # Try loading cached token first
    if _TOKEN_PATH.exists():
        with open(_TOKEN_PATH, "rb") as f:
            creds = pickle.load(f)
        if creds and not creds.expired:
            return creds
        # If expired, we could refresh here, but let's just re-flow for simplicity
        if creds and creds.refresh_token:
            import google.auth.transport.requests
            creds.refresh(google.auth.transport.requests.Request())
            with open(_TOKEN_PATH, "wb") as f:
                pickle.dump(creds, f)
            return creds

    # No valid cached creds — run consent flow
    flow = InstalledAppFlow.from_client_secrets_file(str(secret_full), scopes=_YOUTUBE_SCOPES)
    creds = flow.run_local_server(port=0)

    with open(_TOKEN_PATH, "wb") as f:
        pickle.dump(creds, f)

    return creds


def upload_to_youtube(audio_path: str, title: str, privacy: str = "private") -> dict:
    """Upload audio as a private (or unlisted) YouTube video."""
    if privacy not in ("private", "unlisted"):
        return {"success": False, "video_id": None, "url": None,
                "error": "privacy must be 'private' or 'unlisted'"}

    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        creds = get_youtube_credentials()
        youtube = build("youtube", "v3", credentials=creds)

        media = MediaFileUpload(audio_path, mimetype="audio/mpeg", resumable=True)

        request = youtube.videos().insert(
            part="snippet,status",
            body={
                "snippet": {
                    "title": title,
                    "description": f"Auto-generated podcast: {title}",
                },
                "status": {
                    "privacyStatus": privacy,
                },
            },
            media_body=media,
        )

        response = request.execute()
        video_id = response.get("id")
        return {
            "success": True,
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "error": None,
        }

    except Exception as e:
        return {"success": False, "video_id": None, "url": None, "error": str(e)}
