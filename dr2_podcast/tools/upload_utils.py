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
   After you grant access, the token is cached to youtube_token.json
   and subsequent uploads use it automatically.
"""

import json
import logging
import os
from pathlib import Path

import httpx

SCRIPT_DIR = Path(__file__).parent.absolute()

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def validate_upload_config(buzzsprout: bool, youtube: bool,
                           buzzsprout_api_key: str = None,
                           buzzsprout_account_id: str = None,
                           youtube_secret_path: str = None) -> dict:
    """Check that credentials exist for the requested platforms.

    Credentials can be passed explicitly or fall back to env vars.
    """
    errors = []

    if buzzsprout:
        if not (buzzsprout_api_key or os.getenv("BUZZSPROUT_API_KEY")):
            errors.append("BUZZSPROUT_API_KEY is not set in .env")
        if not (buzzsprout_account_id or os.getenv("BUZZSPROUT_ACCOUNT_ID")):
            errors.append("BUZZSPROUT_ACCOUNT_ID is not set in .env")

    if youtube:
        secret_path = youtube_secret_path or os.getenv("YOUTUBE_CLIENT_SECRET_PATH", "./client_secret.json")
        if not (SCRIPT_DIR / secret_path).exists():
            errors.append(
                f"YouTube client_secret.json not found at {SCRIPT_DIR / secret_path}. "
                "See YOUTUBE SETUP instructions in upload_utils.py"
            )

    return {"valid": len(errors) == 0, "errors": errors}

# ---------------------------------------------------------------------------
# Buzzsprout
# ---------------------------------------------------------------------------

def upload_to_buzzsprout(audio_path: str, title: str,
                         api_key: str = None, account_id: str = None) -> dict:
    """Upload audio as an unpublished (draft) episode to Buzzsprout.

    Credentials can be passed explicitly or fall back to env vars.
    """
    api_key = api_key or os.getenv("BUZZSPROUT_API_KEY")
    account_id = account_id or os.getenv("BUZZSPROUT_ACCOUNT_ID")

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
_TOKEN_PATH = SCRIPT_DIR / "youtube_token.json"
_LEGACY_TOKEN_PATH = SCRIPT_DIR / "youtube_token.pickle"

logger = logging.getLogger(__name__)


def _save_credentials_json(creds) -> None:
    """Persist OAuth credentials to the JSON token file."""
    creds_data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or []),
    }
    with open(_TOKEN_PATH, "w") as f:
        json.dump(creds_data, f)


def get_youtube_credentials(youtube_secret_path: str = None):
    """
    Return cached or freshly-obtained YouTube OAuth credentials.
    May open a browser on first run — call from a request thread, not a daemon thread.

    youtube_secret_path can be passed explicitly or falls back to env var.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.oauth2.credentials import Credentials

    secret_path = youtube_secret_path or os.getenv("YOUTUBE_CLIENT_SECRET_PATH", "./client_secret.json")
    secret_full = SCRIPT_DIR / secret_path

    # One-time migration: convert legacy pickle token to JSON
    if _LEGACY_TOKEN_PATH.exists() and not _TOKEN_PATH.exists():
        logger.info("Migrating YouTube token from pickle to JSON format...")
        try:
            import pickle
            with open(_LEGACY_TOKEN_PATH, "rb") as f:
                legacy_creds = pickle.load(f)
            _save_credentials_json(legacy_creds)
            _LEGACY_TOKEN_PATH.unlink()
            logger.info("Migration complete — youtube_token.pickle removed, youtube_token.json written.")
        except Exception as exc:
            logger.warning("Failed to migrate legacy pickle token: %s. Will proceed to re-auth.", exc)

    # Try loading cached token first
    if _TOKEN_PATH.exists():
        with open(_TOKEN_PATH, "r") as f:
            creds_data = json.load(f)
        creds = Credentials.from_authorized_user_info(creds_data, scopes=_YOUTUBE_SCOPES)
        if creds and not creds.expired:
            return creds
        # Refresh expired token
        if creds and creds.refresh_token:
            import google.auth.transport.requests
            creds.refresh(google.auth.transport.requests.Request())
            _save_credentials_json(creds)
            return creds

    # No valid cached creds — run consent flow
    flow = InstalledAppFlow.from_client_secrets_file(str(secret_full), scopes=_YOUTUBE_SCOPES)
    creds = flow.run_local_server(port=0)

    _save_credentials_json(creds)

    return creds


def upload_to_youtube(audio_path: str, title: str, privacy: str = "private",
                      youtube_secret_path: str = None) -> dict:
    """Upload audio as a private (or unlisted) YouTube video.

    youtube_secret_path can be passed explicitly or falls back to env var.
    """
    if privacy not in ("private", "unlisted"):
        return {"success": False, "video_id": None, "url": None,
                "error": "privacy must be 'private' or 'unlisted'"}

    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        creds = get_youtube_credentials(youtube_secret_path=youtube_secret_path)
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
