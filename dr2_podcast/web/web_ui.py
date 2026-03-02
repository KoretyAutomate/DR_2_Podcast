#!/usr/bin/env python3
"""
Web-based UI for DR_2_Podcast Generation
Provides a user-friendly interface for generating research-driven debate podcasts
"""
import os
import sys
import subprocess
import threading
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
import queue
import secrets
import shutil
import base64

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import uvicorn
import httpx
from dotenv import load_dotenv

from dr2_podcast.tools.upload_utils import validate_upload_config, upload_to_buzzsprout, upload_to_youtube

load_dotenv()

# Configuration — project root is two levels up from dr2_podcast/web/
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = SCRIPT_DIR / "research_outputs"
TASKS_FILE = SCRIPT_DIR / "podcast_tasks.json"
TOPIC_INDEX_FILE = OUTPUT_DIR / "topic_index.json"

# Use podcast_env Python interpreter if available
PODCAST_ENV_PYTHON = Path.home() / "miniconda3" / "envs" / "podcast_flow" / "bin" / "python3"
if not PODCAST_ENV_PYTHON.exists():
    PODCAST_ENV_PYTHON = sys.executable  # Fallback to current Python

# Simple authentication (optional - can be disabled)
USERNAME = os.getenv("PODCAST_WEB_USER", "admin")
PASSWORD = os.getenv("PODCAST_WEB_PASSWORD", secrets.token_urlsafe(16))

_password_is_custom = bool(os.getenv("PODCAST_WEB_PASSWORD"))
print("="*60, file=sys.stderr)
print("PODCAST WEB UI CREDENTIALS", file=sys.stderr)
print("="*60, file=sys.stderr)
print(f"Username: {USERNAME}", file=sys.stderr)
if _password_is_custom:
    print("Password: (set via PODCAST_WEB_PASSWORD)", file=sys.stderr)
else:
    print(f"Password: {PASSWORD}", file=sys.stderr)
print("\nSet custom credentials with:", file=sys.stderr)
print("export PODCAST_WEB_USER=your_username", file=sys.stderr)
print("export PODCAST_WEB_PASSWORD=your_password", file=sys.stderr)
print("="*60, file=sys.stderr)

app = FastAPI(title="DR_2_Podcast Generator")
security = HTTPBasic()
# Credential fields stripped from API responses (SEC-04)
_CREDENTIAL_FIELDS = {"buzzsprout_api_key", "buzzsprout_account_id", "youtube_secret_path"}

# Task storage
tasks_db: Dict[str, Dict] = {}
tasks_lock = threading.RLock()

# Task Queue
task_queue = queue.Queue()
# Expected artifacts for progress tracking
EXPECTED_ARTIFACTS = [
    "research_framing.md", "research_framing.pdf",
    "affirmative_case.md", "falsification_case.md", "grade_synthesis.md",
    "research_sources.json", "clinical_math.md",
    "search_strategy_aff.json", "search_strategy_neg.json",
    "screening_results_aff.json", "screening_results_neg.json",
    "source_of_truth.md", "source_of_truth.pdf",
    "url_validation_results.json",
    "script_draft.md", "script_final.md", "script.txt",
    "EPISODE_BLUEPRINT.md", "accuracy_audit.md", "accuracy_audit.pdf",
    "podcast_generation.log", "session_metadata.txt",
    "audio.wav"
]

# Language-specific extra artifacts (added on top of EXPECTED_ARTIFACTS)
EXPECTED_ARTIFACTS_EXTRA = {
    "ja": ["source_of_truth_ja.md", "source_of_truth_ja.pdf"],
}

_ARTIFACT_SUBDIRS = ("research", "scripts", "audio", "meta")

def count_artifacts(directory: Optional[str], language: str = "en") -> tuple[int, int]:
    """Count generated artifacts vs expected total.

    Checks both flat (legacy) and subdirectory layouts.
    """
    all_artifacts = EXPECTED_ARTIFACTS + EXPECTED_ARTIFACTS_EXTRA.get(language, [])
    if not directory or not os.path.exists(directory):
        return 0, len(all_artifacts)

    d = Path(directory)
    found = 0
    for f in all_artifacts:
        if (d / f).exists():
            found += 1
        else:
            for subdir in _ARTIFACT_SUBDIRS:
                if (d / subdir / f).exists():
                    found += 1
                    break
    return found, len(all_artifacts)

current_task_id = None

def load_tasks():
    """Load tasks from file and clean up interrupted runs."""
    global tasks_db
    with tasks_lock:
        if TASKS_FILE.exists():
            try:
                with open(TASKS_FILE, 'r') as f:
                    tasks_db = json.load(f)

                # CLEANUP: Mark any interrupted "running" tasks as "cancelled" on startup
                dirty = False
                for tid, task in tasks_db.items():
                    if task["status"] in ["running", "uploading", "queued"]:
                        task["status"] = "cancelled"
                        task["error"] = "Server restarted during execution"
                        dirty = True
                if dirty:
                    save_tasks()
                    print("Cleaned up interrupted tasks from previous run.")
            except Exception as e:
                print(f"Error loading tasks: {e}")
                tasks_db = {}

def save_tasks():
    with tasks_lock:
        with open(TASKS_FILE, 'w') as f:
            json.dump(tasks_db, f, indent=2)


# --- Topic Index: central registry of completed research runs ---

def _sot_exists(output_dir: str) -> bool:
    """Return True if source_of_truth.md exists in the given output directory."""
    d = Path(output_dir)
    return (d / "source_of_truth.md").exists() or (d / "SOURCE_OF_TRUTH.md").exists()

def load_topic_index() -> list:
    """Load the topic index from disk."""
    if TOPIC_INDEX_FILE.exists():
        try:
            return json.loads(TOPIC_INDEX_FILE.read_text())
        except Exception:
            pass
    return []

def save_topic_index(index: list):
    """Save the topic index to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TOPIC_INDEX_FILE, 'w') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

def prune_topic_index() -> int:
    """Remove index entries whose output_dir no longer exists on disk.

    Returns the number of entries removed.
    """
    index = load_topic_index()
    before = len(index)
    index = [e for e in index if Path(e["output_dir"]).exists()]
    removed = before - len(index)
    if removed > 0:
        save_topic_index(index)
        print(f"Topic index: pruned {removed} stale entries (remaining: {len(index)})")
    return removed

def register_topic(topic: str, output_dir: str, language: str = "en",
                    accessibility_level: str = "simple", podcast_length: str = "long",
                    podcast_hosts: str = "random"):
    """Add a completed research run to the topic index.

    Prunes stale entries first, then upserts by output_dir (adds new entry or
    updates has_sot on an existing one).
    """
    prune_topic_index()
    index = load_topic_index()
    has_sot = _sot_exists(output_dir)

    # Update existing entry if already registered (e.g. has_sot may have changed)
    for entry in index:
        if entry["output_dir"] == output_dir:
            if entry.get("has_sot") != has_sot:
                entry["has_sot"] = has_sot
                save_topic_index(index)
            return

    index.append({
        "topic": topic,
        "output_dir": output_dir,
        "language": language,
        "accessibility_level": accessibility_level,
        "podcast_length": podcast_length,
        "podcast_hosts": podcast_hosts,
        "created_at": Path(output_dir).name,  # e.g. "2026-02-17_02-03-50"
        "has_sot": has_sot,
    })
    save_topic_index(index)

def backfill_topic_index():
    """Prune stale entries, then scan research_outputs/ to register any directories
    not yet in the index and refresh has_sot on all existing entries."""
    import re as _re

    prune_topic_index()
    index = load_topic_index()
    known_dirs = {entry["output_dir"] for entry in index}
    added = 0
    refreshed = 0

    # Refresh has_sot on all existing entries (field may be missing on older entries)
    for entry in index:
        current_sot = _sot_exists(entry["output_dir"])
        if entry.get("has_sot") != current_sot:
            entry["has_sot"] = current_sot
            refreshed += 1

    if not OUTPUT_DIR.exists():
        if refreshed > 0:
            save_topic_index(index)
        return

    for d in OUTPUT_DIR.iterdir():
        if not d.is_dir() or not d.name[:4].isdigit():
            continue
        if str(d) in known_dirs:
            continue

        # Extract topic + language from session_metadata.txt
        topic = ""
        lang = "en"
        meta_path = d / "session_metadata.txt"
        if meta_path.exists():
            meta_text = meta_path.read_text()
            m = _re.search(r'^Topic:\s*(.+)$', meta_text, _re.MULTILINE)
            if m:
                topic = m.group(1).strip()
            lm = _re.search(r'^Language:\s*\S+\s*\((\w+)\)', meta_text, _re.MULTILINE)
            if lm:
                lang = lm.group(1).strip()

        if not topic:
            # Fallback: check tasks_db
            for tid, task in tasks_db.items():
                if task.get("output_dir") == str(d):
                    topic = task.get("topic", "")
                    lang = task.get("language", "en")
                    break

        if not topic:
            continue

        # Look up other params from tasks_db
        access_level = "simple"
        pod_length = "long"
        pod_hosts = "random"
        for tid, task in tasks_db.items():
            if task.get("output_dir") == str(d):
                access_level = task.get("accessibility_level", "simple")
                pod_length = task.get("podcast_length", "long")
                pod_hosts = task.get("podcast_hosts", "random")
                break

        index.append({
            "topic": topic,
            "output_dir": str(d),
            "language": lang,
            "accessibility_level": access_level,
            "podcast_length": pod_length,
            "podcast_hosts": pod_hosts,
            "created_at": d.name,
            "has_sot": _sot_exists(str(d)),
        })
        added += 1

    if added > 0 or refreshed > 0:
        save_topic_index(index)
        print(f"Topic index: backfilled {added} entries, refreshed has_sot on {refreshed} (total: {len(index)})")


def worker_thread():
    """Background thread to process the task queue sequentially."""
    global current_task_id
    print("Worker thread started, waiting for tasks...")
    while True:
        try:
            # Block until a task is available
            task_data = task_queue.get()
            task_id = task_data["task_id"]
            current_task_id = task_id
            
            print(f"Starting task {task_id} from queue...")

            if task_data.get("reuse_mode"):
                # Reuse path
                run_podcast_reuse(task_data)
            else:
                # Normal full pipeline
                run_podcast_generation(
                    task_id,
                    task_data["topic"],
                    task_data["language"],
                    task_data["accessibility_level"],
                    task_data["podcast_length"],
                    task_data["podcast_hosts"],
                    task_data.get("channel_intro", ""),
                    task_data["upload_buzzsprout"],
                    task_data["upload_youtube"],
                    task_data.get("core_target", ""),
                    task_data.get("channel_mission", ""),
                    buzzsprout_api_key=task_data.get("buzzsprout_api_key"),
                    buzzsprout_account_id=task_data.get("buzzsprout_account_id"),
                    youtube_secret_path=task_data.get("youtube_secret_path"),
                )
            
        except Exception as e:
            print(f"Worker thread error: {e}")
            if current_task_id:
                tasks_db[current_task_id]["status"] = "failed"
                tasks_db[current_task_id]["error"] = f"Worker error: {str(e)}"
                save_tasks()
        finally:
            current_task_id = None
            task_queue.task_done()

# Start worker thread on import (or main)
threading.Thread(target=worker_thread, daemon=True).start()

# Load existing tasks on startup
load_tasks()

# Sync topic index: prune deleted dirs, then pick up any unregistered runs
prune_topic_index()
backfill_topic_index()

class PodcastRequest(BaseModel):
    topic: str
    language: str = "en"
    accessibility_level: str = "simple"
    podcast_length: str = "long"
    podcast_hosts: str = "random"
    channel_intro: str = ""
    core_target: str = ""
    channel_mission: str = ""
    upload_to_buzzsprout: bool = False
    upload_to_youtube: bool = False
    buzzsprout_api_key: str = ""
    buzzsprout_account_id: str = ""
    youtube_secret_path: str = ""

class GenerateIntroRequest(BaseModel):
    channel_name: str = ""
    core_target: str = ""
    channel_mission: str = ""
    language: str = "en"

class ReuseCheckRequest(BaseModel):
    topic: str

class ReuseGenerateRequest(BaseModel):
    topic: str
    reuse_task_id: str = ""
    reuse_output_dir: str = ""
    language: str = "en"
    accessibility_level: str = "simple"
    podcast_length: str = "long"
    podcast_hosts: str = "random"
    upload_to_buzzsprout: bool = False
    upload_to_youtube: bool = False
    buzzsprout_api_key: str = ""
    buzzsprout_account_id: str = ""
    youtube_secret_path: str = ""

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Simple authentication"""
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/", response_class=HTMLResponse)
def home(username: str = Depends(verify_credentials)):
    """Main page with podcast generation UI"""
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DR_2_Podcast Generator</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg-color: #0f172a;
                --card-bg: rgba(30, 41, 59, 0.7);
                --text-primary: #f8fafc;
                --text-secondary: #94a3b8;
                --accent-primary: #8b5cf6;
                --accent-secondary: #06b6d4;
                --border-color: rgba(255, 255, 255, 0.1);
                --success-color: #10b981;
                --warning-color: #f59e0b;
                --error-color: #ef4444;
            }}

            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
                background-color: var(--bg-color);
                background-image: 
                    radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
                    radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%), 
                    radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%);
                color: var(--text-primary);
                min-height: 100vh;
                padding: 40px 20px;
                line-height: 1.6;
            }}

            .container {{
                max-width: 900px;
                margin: 0 auto;
            }}

            /* Glassmorphism Card Style */
            .glass-card {{
                background: var(--card-bg);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border: 1px solid var(--border-color);
                border-radius: 16px;
                padding: 40px;
                margin-bottom: 30px;
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            }}

            .header {{
                text-align: center;
                margin-bottom: 40px;
            }}

            h1 {{
                font-size: 2.5rem;
                font-weight: 700;
                background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 10px;
                letter-spacing: -0.02em;
            }}

            .subtitle {{
                color: var(--text-secondary);
                font-size: 1.1rem;
                font-weight: 300;
            }}

            h2 {{
                color: var(--text-primary);
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 24px;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 12px;
            }}

            /* Form Elements */
            label {{
                display: block;
                font-size: 0.9rem;
                font-weight: 500;
                color: var(--text-secondary);
                margin-bottom: 8px;
                margin-top: 20px;
            }}

            input[type="text"], select, textarea {{
                width: 100%;
                background: rgba(15, 23, 42, 0.6);
                border: 1px solid var(--border-color);
                color: var(--text-primary);
                padding: 12px 16px;
                border-radius: 8px;
                font-size: 1rem;
                font-family: inherit;
                transition: all 0.3s ease;
                box-sizing: border-box;
            }}

            input[type="text"]:focus, select:focus, textarea:focus {{
                outline: none;
                border-color: var(--accent-primary);
                box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2);
            }}
            
            select {{
                appearance: none;
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
                background-repeat: no-repeat;
                background-position: right 12px center;
                background-size: 16px;
            }}

            button {{
                background: linear-gradient(135deg, var(--accent-primary) 0%, #7c3aed 100%);
                color: white;
                border: none;
                padding: 16px 32px;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                margin-top: 30px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 14px 0 rgba(124, 58, 237, 0.39);
            }}

            button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(124, 58, 237, 0.23);
            }}

            button:disabled {{
                background: linear-gradient(135deg, #475569 0%, #334155 100%);
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
                opacity: 0.7;
            }}

            /* Custom Checkbox */
            .checkbox-wrapper {{
                display: flex;
                align-items: center;
                gap: 12px;
                cursor: pointer;
                margin-top: 8px;
                color: var(--text-primary);
                font-weight: 400;
            }}

            input[type="checkbox"] {{
                appearance: none;
                background-color: rgba(15, 23, 42, 0.6);
                margin: 0;
                font: inherit;
                color: currentColor;
                width: 20px;
                height: 20px;
                border: 1px solid var(--text-secondary);
                border-radius: 4px;
                display: grid;
                place-content: center;
                transition: 0.2s ease-in-out;
            }}

            input[type="checkbox"]::before {{
                content: "";
                width: 10px;
                height: 10px;
                transform: scale(0);
                transition: 0.2s ease-in-out;
                box-shadow: inset 1em 1em var(--text-primary);
                transform-origin: center;
                clip-path: polygon(14% 44%, 0 65%, 50% 100%, 100% 16%, 80% 0%, 43% 62%);
            }}

            input[type="checkbox"]:checked {{
                background-color: var(--accent-primary);
                border-color: var(--accent-primary);
            }}
            
            input[type="checkbox"]:checked::before {{
                transform: scale(1);
            }}

            /* Config Sections */
            .config-section {{
                background: rgba(15, 23, 42, 0.4);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 20px;
                margin-top: 10px;
            }}

            /* Status Box */
            .status-box {{
                margin-top: 30px;
                display: none;
                border-top: 1px solid var(--border-color);
                padding-top: 30px;
            }}

            .status-box.show {{
                display: block;
                animation: fadeIn 0.5s ease-out;
            }}

            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}

            .status-header {{
                font-size: 1.1rem;
                font-weight: 600;
                color: var(--text-primary);
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                gap: 10px;
            }}

            .progress {{
                width: 100%;
                height: 6px;
                background: rgba(255,255,255,0.1);
                border-radius: 3px;
                overflow: hidden;
                margin-bottom: 20px;
            }}

            .progress-bar {{
                height: 100%;
                background: linear-gradient(90deg, var(--accent-secondary), var(--accent-primary));
                width: 0%;
                transition: width 0.4s ease;
                box-shadow: 0 0 10px rgba(139, 92, 246, 0.5);
            }}

            /* History List */
            .history {{
                list-style: none;
            }}

            .history-item {{
                background: rgba(255, 255, 255, 0.03);
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 15px;
                border: 1px solid transparent;
                transition: all 0.2s ease;
                display: flex;
                flex-direction: column;
                gap: 5px;
            }}

            .history-item:hover {{
                border-color: var(--border-color);
                background: rgba(255, 255, 255, 0.06);
            }}

            .history-topic {{
                font-weight: 600;
                color: var(--text-primary);
                font-size: 1.05rem;
            }}

            .history-meta {{
                font-size: 0.85rem;
                color: var(--text-secondary);
            }}

            .history-summary {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                cursor: pointer;
            }}

            .history-details {{
                display: none;
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1px dashed var(--border-color);
                font-size: 0.9rem;
            }}
            
            .history-details.open {{
                display: block;
            }}
            
            .artifact-pill {{
                display: inline-flex;
                align-items: center;
                padding: 4px 10px;
                background: rgba(99, 102, 241, 0.1);
                color: var(--accent-primary);
                border-radius: 12px;
                font-size: 0.8rem;
                margin: 2px;
                text-decoration: none;
                border: 1px solid rgba(99, 102, 241, 0.2);
            }}
            
            .artifact-pill:hover {{
                background: rgba(99, 102, 241, 0.2);
            }}

            /* Buttons inside sections */
            .auth-btn {{
                background: transparent;
                border: 1px solid var(--accent-primary);
                color: var(--accent-primary);
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 0.9rem;
                font-weight: 500;
                margin-top: 10px;
                width: auto;
                box-shadow: none;
            }}
            
            .auth-btn:hover {{
                background: rgba(139, 92, 246, 0.1);
                transform: none;
                box-shadow: none;
            }}

            /* Status Colors */
            .status-pending {{ color: var(--warning-color); }}
            .status-running {{ color: var(--accent-secondary); }}
            .status-completed {{ color: var(--success-color); }}
            .status-failed {{ color: var(--error-color); }}

            /* Download Button */
            .download-link {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid var(--success-color);
                color: var(--success-color);
                padding: 10px 20px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 500;
                margin-top: 15px;
                transition: all 0.2s;
            }}

            .download-link:hover {{
                background: rgba(16, 185, 129, 0.2);
            }}
            
            .error {{
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid var(--error-color);
                color: #fca5a5;
                padding: 15px;
                border-radius: 8px;
                margin-top: 15px;
                font-size: 0.9rem;
            }}

            /* Visualizer Grid */
            .research-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(24px, 1fr));
                gap: 8px;
                margin-top: 15px;
                max-height: 120px;
                overflow-y: auto;
                padding: 10px;
                background: rgba(0, 0, 0, 0.2);
                border-radius: 8px;
            }}

            .source-icon {{
                width: 24px;
                height: 24px;
                border-radius: 4px;
                background: #1e293b;
                border: 1px solid var(--border-color);
                transition: transform 0.2s;
            }}
            
            .source-icon:hover {{
                transform: scale(1.2);
                z-index: 10;
                border-color: var(--accent-secondary);
            }}

            .counter-box {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 15px;
                background: rgba(139, 92, 246, 0.1);
                border: 1px solid rgba(139, 92, 246, 0.2);
                padding: 15px;
                border-radius: 12px;
            }}

            .counter-label {{ font-size: 0.9rem; color: var(--accent-primary); font-weight: 600; }}
            .counter-value {{ font-size: 1.5rem; color: #fff; font-weight: 700; }}
            
            .eta-box {{
                font-family: monospace;
                font-size: 0.9rem;
                color: var(--text-secondary);
                text-align: right;
                margin-top: 5px;
            }}

        </style>
    </head>
    
    <body>
        <div class="container">
            <div class="header">
                <h1>🎙️ DR_2_Podcast</h1>
                <div class="subtitle">Next-Generation AI Research & Debate Engineer</div>
                <div id="gitStatus" class="git-status" style="display:none; margin-top:10px; font-size: 0.8rem; padding: 4px 8px; border-radius: 4px; background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; display: inline-block;"></div>
            </div>

            <div class="glass-card">
                <h2>Create New Episode</h2>
                <form id="podcastForm">
                    <label for="topic">Research Question</label>
                    <input
                        type="text"
                        id="topic"
                        name="topic"
                        placeholder="e.g., Effects of intermittent fasting on neuroplasticity"
                        required
                    />

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <label for="language">Language</label>
                            <select id="language" name="language">
                                <option value="en">English (US)</option>
                                <option value="ja">日本語 (Japanese)</option>
                            </select>
                        </div>
                        <div>
                            <label for="accessibility">Target Audience</label>
                            <select id="accessibility" name="accessibility_level">
                                <option value="simple">General Audience (Simple)</option>
                                <option value="moderate">Enthusiast (Moderate)</option>
                                <option value="technical">Researcher (Technical)</option>
                            </select>
                        </div>
                    </div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <label for="length">Key Duration</label>
                            <select id="length" name="podcast_length">
                                <option value="short">Briefing (~10-15 min)</option>
                                <option value="medium">Standard (~20-25 min)</option>
                                <option value="long" selected>Deep Dive (30+ min)</option>
                            </select>
                        </div>
                        <div>
                            <label for="hosts">Host Dynamic</label>
                            <select id="hosts" name="podcast_hosts">
                                <option value="random" selected>AI Choice (Random)</option>
                                <option value="host1_leads">Host 1 Leads (Male)</option>
                                <option value="host2_leads">Host 2 Leads (Female)</option>
                            </select>
                        </div>
                    </div>

                    <div style="margin-top: 20px;">
                        <label style="margin-bottom: 12px;">Channel Identity <span style="color: var(--text-secondary); font-size: 12px;">(used to auto-generate the Channel Intro)</span></label>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 8px;">
                            <div>
                                <label for="channelName" style="font-size: 13px; font-weight: 500;">Channel Name</label>
                                <input
                                    type="text"
                                    id="channelName"
                                    placeholder="e.g., Deep Research Podcast"
                                />
                            </div>
                            <div>
                                <label for="coreTarget" style="font-size: 13px; font-weight: 500;">Core Target</label>
                                <input
                                    type="text"
                                    id="coreTarget"
                                    placeholder="e.g., science enthusiasts seeking actionable insights"
                                />
                            </div>
                        </div>
                        <div style="margin-top: 12px;">
                            <label for="channelMission" style="font-size: 13px; font-weight: 500;">Channel Mission</label>
                            <input
                                type="text"
                                id="channelMission"
                                placeholder="e.g., we turn cutting-edge science into everyday wisdom"
                            />
                        </div>
                    </div>

                    <div style="margin-top: 20px;">
                        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
                            <label for="channelIntro" style="margin: 0;">Channel Intro <span style="color: var(--text-secondary); font-size: 12px;">(spoken every episode, editable)</span></label>
                            <button type="button" id="regenIntroBtn" style="font-size: 11px; padding: 2px 10px; background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.4); border-radius: 4px; color: var(--accent); cursor: pointer;">✨ Auto Generate</button>
                        </div>
                        <textarea
                            id="channelIntro"
                            name="channel_intro"
                            rows="5"
                            placeholder="Fill in Channel Name, Core Target, and Channel Mission above to auto-generate — or type your intro here."
                            style="resize: vertical; min-height: 110px;"
                        ></textarea>
                    </div>

                    <div style="margin-top: 30px;">
                        <label style="margin-bottom: 12px;">Research Options</label>
                        <div style="display: flex; gap: 24px; margin-bottom: 15px;">
                            <label class="checkbox-wrapper">
                                <input type="checkbox" id="leveragePast" />
                                Leverage previous research from the past 7 days if available
                            </label>
                        </div>
                    </div>

                    <div style="margin-top: 30px;">
                        <label style="margin-bottom: 12px;">Publishing Options</label>
                        
                        <div style="display: flex; gap: 24px; margin-bottom: 15px;">
                            <label class="checkbox-wrapper" id="buzzsproutLabel">
                                <input type="checkbox" id="uploadBuzzsprout" />
                                Buzzsprout (Draft)
                            </label>
                            <label class="checkbox-wrapper" id="youtubeLabel">
                                <input type="checkbox" id="uploadYoutube" />
                                YouTube (Private)
                            </label>
                        </div>

                        <div id="buzzsproutFields" class="config-section" style="display: none;">
                            <label style="margin-top: 0;">API Key</label>
                            <input type="text" id="buzzsproutApiKey" placeholder="Your Buzzsprout API key" style="margin-bottom: 10px;" />
                            <label>Account ID</label>
                            <input type="text" id="buzzsproutAccountId" placeholder="Your Buzzsprout account ID" />
                        </div>

                        <div id="youtubeFields" class="config-section" style="display: none;">
                            <label style="margin-top: 0;">Client Secret JSON Path</label>
                            <input type="text" id="youtubeSecretPath" placeholder="./client_secret.json" value="./client_secret.json" />
                            <button type="button" id="ytAuthBtn" class="auth-btn">Authorize YouTube</button>
                            <span id="ytAuthStatus" style="font-size: 12px; margin-left: 8px; color: var(--text-secondary);"></span>
                        </div>
                    </div>

                    <button type="submit" id="generateBtn">Initiate Production Sequence</button>
                </form>

                <div id="statusBox" class="status-box">
                    <div class="status-header">
                        <span id="statusIcon">⚡</span> 
                        <span id="statusText">System Ready</span>
                    </div>
                    
                    <div class="progress">
                        <div class="progress-bar" id="progressBar"></div>
                    </div>
                    
                    <div class="eta-box" id="etaDisplay"></div>
                    
                    <!-- Artifact Progress -->
                    <div id="artifactProgress" style="margin-top: 10px; font-size: 0.9rem; color: var(--text-secondary); display: none;">
                        📦 Artifacts: <span id="artifactCount" style="color: var(--text-primary); font-weight: bold;">0/24</span>
                    </div>

                    <!-- Step Durations -->
                    <div id="stepDurations" style="margin-top: 15px; border-top: 1px solid var(--border-color); padding-top: 10px; display: none;">
                        <h4 style="margin: 0 0 10px 0; font-size: 0.9rem; color: var(--text-secondary);">Step Timings</h4>
                        <table style="width: 100%; font-size: 0.85rem; border-collapse: collapse;">
                            <tbody id="stepDurationList">
                                <!-- Steps injected here -->
                            </tbody>
                        </table>
                        <div style="margin-top: 5px; font-size: 0.85rem; color: var(--accent-primary);">
                            Current Step: <span id="currentStepTimer">0m 0s</span>
                        </div>
                    </div>

                    <!-- Research Visualizer -->
                    <div id="researchViz" style="display:none; margin-top: 20px;">
                        <div class="counter-box">
                            <div>
                                <div class="counter-label">SOURCES REFERRED</div>
                                <div class="counter-value" id="sourceCount">0</div>
                            </div>
                            <div style="text-align:right;">
                                <div class="counter-label">DEEP RESEARCH</div>
                                <div style="font-size:0.8rem; color:var(--text-secondary);">Active Protocol</div>
                            </div>
                        </div>
                        <div class="research-grid" id="sourceGrid">
                            <!-- Icons injected here -->
                        </div>
                    </div>

                    <div id="statusDetails" style="color: var(--text-secondary); font-size: 0.9rem; font-family: monospace; margin-top: 15px;"></div>
                    <div id="downloads" class="downloads"></div>
                    <div id="error" class="error" style="display: none;"></div>
                </div>
            </div>

            <div class="glass-card">
                <h2 id="historyToggle" style="cursor:pointer;user-select:none;display:flex;justify-content:space-between;align-items:center;">
                    Production History <span id="historyArrow">&#9654;</span>
                </h2>
                <ul id="history" class="history" style="display:none;">
                    <li style="color: var(--text-secondary); text-align: center; padding: 20px;">No episodes generated yet</li>
                </ul>
            </div>
        </div>

        <!-- Reuse Modal -->
        <div id="reuseModal" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.7); backdrop-filter:blur(4px); z-index:1000; align-items:center; justify-content:center;">
            <div style="background:var(--card-bg); border:1px solid var(--border-color); border-radius:16px; padding:32px; max-width:600px; width:90%; max-height:80vh; overflow-y:auto; box-shadow:0 20px 60px rgba(0,0,0,0.5);">
                <h2 style="margin-bottom:16px; border:none; padding:0; font-size:1.3rem;">Similar Topic Found</h2>
                <div id="reuseMatchInfo" style="margin-bottom:20px;"></div>
                <div id="reuseParamDiff" style="margin-bottom:20px;"></div>
                <div style="display:flex; gap:12px; justify-content:flex-end; flex-wrap:wrap;">
                    <button type="button" id="reuseCancelBtn" style="background:transparent; border:1px solid var(--border-color); color:var(--text-secondary); padding:10px 20px; width:auto; margin:0; box-shadow:none;">Cancel</button>
                    <button type="button" id="reuseFreshBtn" style="background:transparent; border:1px solid var(--warning-color); color:var(--warning-color); padding:10px 20px; width:auto; margin:0; box-shadow:none;">Start Fresh</button>
                    <button type="button" id="reuseUseBtn" style="padding:10px 20px; width:auto; margin:0;">Use Previous Research</button>
                </div>
            </div>
        </div>

        <script>
            let currentTaskId = null;
            let statusInterval = null;
            let pendingTaskIds = [];  // queued tasks waiting to be tracked

            // Production History toggle
            document.getElementById('historyToggle').addEventListener('click', function() {{
                const el = document.getElementById('history');
                const arrow = document.getElementById('historyArrow');
                if (el.style.display === 'none') {{
                    el.style.display = 'block';
                    arrow.innerHTML = '&#9660;';
                }} else {{
                    el.style.display = 'none';
                    arrow.innerHTML = '&#9654;';
                }}
            }});

            // Load history on page load
            loadHistory();
            
            // Check System Status (Git)
            (async () => {{
                try {{
                    const res = await fetch('/api/system_status');
                    const status = await res.json();
                    const el = document.getElementById('gitStatus');
                    if (!status.git_clean) {{
                        el.textContent = status.message;
                        el.style.display = 'inline-block';
                    }}
                }} catch(e) {{ console.warn('Git status check failed'); }}
            }})();

            // Pre-fill fields if server already has config
            (async () => {{
                try {{
                    const cfg = await (await fetch('/api/upload_config')).json();
                    if (cfg.buzzsprout_configured) {{
                        document.getElementById('buzzsproutApiKey').placeholder = '(already configured on server)';
                        document.getElementById('buzzsproutAccountId').placeholder = '(already configured on server)';
                    }}
                    if (cfg.youtube_configured) {{
                        document.getElementById('ytAuthStatus').textContent = 'Client secret found on server';
                        document.getElementById('ytAuthStatus').style.color = '#10b981';
                    }}
                }} catch(e) {{ console.warn('upload_config check failed', e); }}
            }})();

            // Toggle Buzzsprout fields
            document.getElementById('uploadBuzzsprout').addEventListener('change', function() {{
                document.getElementById('buzzsproutFields').style.display = this.checked ? 'block' : 'none';
            }});

            // Toggle YouTube fields
            document.getElementById('uploadYoutube').addEventListener('change', function() {{
                document.getElementById('youtubeFields').style.display = this.checked ? 'block' : 'none';
            }});

            // --- Channel Intro — LLM-powered Auto Generate ---
            document.getElementById('regenIntroBtn').addEventListener('click', async function() {{
                const btn     = this;
                const el      = document.getElementById('channelIntro');
                const name    = (document.getElementById('channelName').value || '').trim();
                const target  = (document.getElementById('coreTarget').value || '').trim();
                const mission = (document.getElementById('channelMission').value || '').trim();
                const lang    = document.getElementById('language').value;

                btn.disabled = true;
                btn.textContent = '⏳ Generating...';

                try {{
                    const res = await fetch('/api/generate-intro', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            channel_name: name,
                            core_target: target,
                            channel_mission: mission,
                            language: lang
                        }})
                    }});
                    if (res.ok) {{
                        const data = await res.json();
                        el.value = data.intro || '';
                    }} else {{
                        const err = await res.json().catch(() => ({{}}));
                        alert('Could not generate intro: ' + (err.detail || res.statusText));
                    }}
                }} catch (e) {{
                    alert('Network error: ' + e.message);
                }} finally {{
                    btn.disabled = false;
                    btn.textContent = '✨ Auto Generate';
                }}
            }});

            // YouTube OAuth preflight
            document.getElementById('ytAuthBtn').addEventListener('click', async function() {{
                this.textContent = 'Authorizing...';
                this.disabled = true;
                const secretPath = document.getElementById('youtubeSecretPath').value;
                try {{
                    const res = await fetch('/api/youtube/preflight', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ secret_path: secretPath }})
                    }});
                    const data = await res.json();
                    const status = document.getElementById('ytAuthStatus');
                    if (data.ready) {{
                        this.textContent = 'Authorized';
                        this.style.background = 'rgba(16, 185, 129, 0.1)';
                        this.style.color = '#10b981';
                        this.style.borderColor = '#10b981';
                        status.textContent = 'Authorized';
                        status.style.color = '#10b981';
                    }} else {{
                        this.textContent = 'Retry';
                        status.textContent = 'Failed: ' + data.error;
                        status.style.color = '#ef4444';
                    }}
                }} catch(e) {{
                    this.textContent = 'Error';
                    document.getElementById('ytAuthStatus').textContent = 'Network error';
                }}
                this.disabled = false;
            }});

            // Show reuse modal and return user decision as a Promise
            function showReuseModal(match) {{
                return new Promise((resolve) => {{
                    const modal = document.getElementById('reuseModal');
                    const infoDiv = document.getElementById('reuseMatchInfo');
                    const diffDiv = document.getElementById('reuseParamDiff');

                    // Populate match info
                    const date = new Date(match.created_at).toLocaleDateString();
                    infoDiv.innerHTML = `
                        <div style="background:rgba(139,92,246,0.1); border:1px solid rgba(139,92,246,0.3); border-radius:8px; padding:16px;">
                            <div style="font-weight:600; color:var(--text-primary); margin-bottom:8px;">Previous Topic</div>
                            <div style="color:var(--text-secondary); margin-bottom:12px;">${{match.topic}}</div>
                            <div style="display:flex; gap:16px; font-size:0.85rem;">
                                <span style="color:var(--accent-primary);">Similarity: ${{match.similarity_score}}%</span>
                                <span style="color:var(--text-secondary);">Completed: ${{date}}</span>
                            </div>
                        </div>
                    `;

                    // Build parameter diff table
                    const newLang = document.getElementById('language').value;
                    const newAccess = document.getElementById('accessibility').value;
                    const newLength = document.getElementById('length').value;
                    const newHosts = document.getElementById('hosts').value;

                    const params = [
                        ['Language', match.language, newLang],
                        ['Audience', match.accessibility_level, newAccess],
                        ['Duration', match.podcast_length, newLength],
                        ['Hosts', match.podcast_hosts, newHosts],
                    ];

                    let tableRows = '';
                    for (const [name, old, cur] of params) {{
                        const changed = old !== cur;
                        const color = changed ? 'var(--warning-color)' : 'var(--text-secondary)';
                        const badge = changed ? ' <span style="color:var(--warning-color); font-size:0.75rem;">(changed)</span>' : '';
                        tableRows += `<tr>
                            <td style="padding:6px 12px; color:var(--text-secondary);">${{name}}</td>
                            <td style="padding:6px 12px; color:${{color}};">${{old}}</td>
                            <td style="padding:6px 12px; color:${{color}};">${{cur}}${{badge}}</td>
                        </tr>`;
                    }}

                    diffDiv.innerHTML = `
                        <table style="width:100%; font-size:0.85rem; border-collapse:collapse;">
                            <thead><tr style="border-bottom:1px solid var(--border-color);">
                                <th style="padding:6px 12px; text-align:left; color:var(--text-secondary); font-weight:500;">Parameter</th>
                                <th style="padding:6px 12px; text-align:left; color:var(--text-secondary); font-weight:500;">Previous</th>
                                <th style="padding:6px 12px; text-align:left; color:var(--text-secondary); font-weight:500;">Current</th>
                            </tr></thead>
                            <tbody>${{tableRows}}</tbody>
                        </table>
                    `;

                    modal.style.display = 'flex';

                    function cleanup() {{
                        modal.style.display = 'none';
                        document.getElementById('reuseCancelBtn').removeEventListener('click', onCancel);
                        document.getElementById('reuseFreshBtn').removeEventListener('click', onFresh);
                        document.getElementById('reuseUseBtn').removeEventListener('click', onReuse);
                    }}
                    function onCancel() {{ cleanup(); resolve({{ action: 'cancel' }}); }}
                    function onFresh() {{ cleanup(); resolve({{ action: 'fresh' }}); }}
                    function onReuse() {{ cleanup(); resolve({{ action: 'reuse', match }}); }}

                    document.getElementById('reuseCancelBtn').addEventListener('click', onCancel);
                    document.getElementById('reuseFreshBtn').addEventListener('click', onFresh);
                    document.getElementById('reuseUseBtn').addEventListener('click', onReuse);
                }});
            }}

            // Helper: start tracking a task after submission
            function startTracking(taskId) {{
                const statusBox = document.getElementById('statusBox');
                if (currentTaskId && statusInterval) {{
                    pendingTaskIds.push(taskId);
                    showQueuedToast(taskId, pendingTaskIds.length);
                }} else {{
                    currentTaskId = taskId;
                    statusBox.classList.add('show');
                    statusInterval = setInterval(checkStatus, 2000);
                }}
            }}

            // Helper: get current form params as object
            function getFormParams() {{
                return {{
                    topic: document.getElementById('topic').value,
                    language: document.getElementById('language').value,
                    accessibility_level: document.getElementById('accessibility').value,
                    podcast_length: document.getElementById('length').value,
                    podcast_hosts: document.getElementById('hosts').value,
                    channel_intro: document.getElementById('channelIntro').value || '',
                    core_target: document.getElementById('coreTarget').value || '',
                    channel_mission: document.getElementById('channelMission').value || '',
                    leverage_past: document.getElementById('leveragePast').checked,
                    upload_to_buzzsprout: document.getElementById('uploadBuzzsprout').checked,
                    upload_to_youtube: document.getElementById('uploadYoutube').checked,
                    buzzsprout_api_key: document.getElementById('buzzsproutApiKey').value || '',
                    buzzsprout_account_id: document.getElementById('buzzsproutAccountId').value || '',
                    youtube_secret_path: document.getElementById('youtubeSecretPath').value || ''
                }};
            }}

            // Form submission
            document.getElementById('podcastForm').addEventListener('submit', async (e) => {{
                e.preventDefault();

                const topic = document.getElementById('topic').value;
                const button = document.getElementById('generateBtn');

                document.getElementById('error').style.display = 'none';

                // Step 1: Check for similar previous runs (only if "leverage past" is checked)
                const leveragePast = document.getElementById('leveragePast').checked;
                if (leveragePast) {{
                    try {{
                        button.disabled = true;
                        button.textContent = 'Checking for similar topics...';
                        const reuseRes = await fetch('/api/check-reuse', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{ topic }})
                        }});
                        const reuseData = await reuseRes.json();

                        if (reuseData.has_match && reuseData.matches.length > 0) {{
                            // Auto-reuse: skip modal, go directly to reuse endpoint
                            button.disabled = true;
                            button.textContent = 'Reusing previous research...';
                            const params = getFormParams();
                            const match = reuseData.matches[0];
                            const reuseResp = await fetch('/api/generate-reuse', {{
                                method: 'POST',
                                headers: {{ 'Content-Type': 'application/json' }},
                                body: JSON.stringify({{
                                    ...params,
                                    reuse_task_id: match.task_id || '',
                                    reuse_output_dir: match.output_dir || ''
                                }})
                            }});
                            if (!reuseResp.ok) throw new Error('Failed to start reuse generation');
                            const reuseResult = await reuseResp.json();
                            button.disabled = false;
                            button.textContent = 'Initiate Production Sequence';
                            startTracking(reuseResult.task_id);
                            return;
                        }} else {{
                            // No match found — show brief toast and proceed with fresh generation
                            const toast = document.createElement('div');
                            toast.textContent = 'No similar past research found. Running fresh generation.';
                            toast.style.cssText = 'position:fixed;top:20px;right:20px;background:#334155;color:#e2e8f0;padding:12px 20px;border-radius:8px;z-index:9999;font-size:14px;';
                            document.body.appendChild(toast);
                            setTimeout(() => toast.remove(), 4000);
                        }}
                    }} catch (err) {{
                        console.warn('Reuse check failed, proceeding with fresh generation:', err);
                    }}
                }}

                // Step 2: Queue status check
                try {{
                    const qRes = await fetch('/api/queue-info');
                    const qInfo = await qRes.json();
                    const busy = qInfo.running + qInfo.queued;

                    if (busy > 0) {{
                        let msg = '';
                        if (qInfo.running > 0 && qInfo.queued > 0) {{
                            msg = `There is 1 task running and ${{qInfo.queued}} queued request${{qInfo.queued > 1 ? 's' : ''}}. Your request will be added to the queue.`;
                        }} else if (qInfo.running > 0) {{
                            msg = `There is 1 task currently running. Your request will be queued next.`;
                        }} else {{
                            msg = `There ${{qInfo.queued === 1 ? 'is' : 'are'}} ${{qInfo.queued}} queued request${{qInfo.queued > 1 ? 's' : ''}} ahead. Your request will be added to the queue.`;
                        }}
                        if (!confirm(msg + '\\n\\nProceed?')) {{
                            button.disabled = false;
                            button.textContent = 'Initiate Production Sequence';
                            return;
                        }}
                    }}
                }} catch (err) {{
                    console.warn('Queue check failed, proceeding anyway:', err);
                }}

                // Step 3: Normal generation
                button.disabled = true;
                button.textContent = 'Submitting...';

                try {{
                    const params = getFormParams();
                    const response = await fetch('/api/generate', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify(params)
                    }});

                    if (!response.ok) {{
                        throw new Error('Failed to start generation');
                    }}

                    const data = await response.json();
                    button.disabled = false;
                    button.textContent = 'Initiate Production Sequence';
                    startTracking(data.task_id);

                }} catch (error) {{
                    showError(error.message);
                    button.disabled = false;
                    button.textContent = 'Initiate Production Sequence';
                }}
            }});

            async function checkStatus() {{
                if (!currentTaskId) return;

                try {{
                    const response = await fetch(`/api/status/${{currentTaskId}}`);
                    const data = await response.json();

                    updateStatus(data);

                    if (data.status === 'completed' || data.status === 'failed') {{
                        clearInterval(statusInterval);
                        statusInterval = null;
                        loadHistory();

                        // Pick up next pending task if any
                        if (pendingTaskIds.length > 0) {{
                            currentTaskId = pendingTaskIds.shift();
                            statusInterval = setInterval(checkStatus, 2000);
                        }} else {{
                            currentTaskId = null;
                        }}
                    }}
                }} catch (error) {{
                    console.error('Status check failed:', error);
                }}
            }}

            function updateStatus(data) {{
                const statusText = document.getElementById('statusText');
                const statusDetails = document.getElementById('statusDetails');
                const progressBar = document.getElementById('progressBar');
                const downloads = document.getElementById('downloads');
                const statusIcon = document.getElementById('statusIcon');

                let displayStatus = data.status.charAt(0).toUpperCase() + data.status.slice(1);
                statusText.textContent = displayStatus;
                statusText.className = `status-${{data.status}}`;

                if (data.status === 'running' || data.status === 'queued') {{
                    statusBox.style.display = 'block';

                    if (data.status === 'queued') {{
                        statusText.textContent = 'Queued for Production...';
                        statusDetails.textContent = 'Waiting for previous task to finish.';
                        statusIcon.textContent = '⏳';
                        progressBar.style.width = '100%';
                        progressBar.style.background = '#334155'; // Grey for queued
                        return;
                    }}

                    statusText.textContent = 'Production In Progress...';
                    statusIcon.textContent = '⚡';
                    
                    const pct = data.progress || 0;
                    progressBar.style.width = pct + '%';
                    const step = data.current_step || data.phase || 'Initializing...';
                    statusDetails.textContent = `Phase: ${{step}} >> Progress: ${{pct}}%`;
                    statusIcon.textContent = '⚙️';
                    
                    // Update numeric ETA
                    if (data.estimated_remaining) {{
                         const mins = Math.floor(data.estimated_remaining / 60);
                         const secs = Math.floor(data.estimated_remaining % 60);
                         document.getElementById('etaDisplay').textContent = `⏱️ Approx. Remaining: ${{mins}}m ${{secs}}s`;
                    }}
                    
                    // Update Artifact Count
                    const artifactNav = document.getElementById('artifactProgress');
                    const artifactCount = document.getElementById('artifactCount');
                    if (data.artifacts_total) {{
                        artifactNav.style.display = 'block';
                        artifactCount.textContent = `${{data.artifacts_created}}/${{data.artifacts_total}}`;
                    }}
                    
                    // Update Step Durations
                    const durationBox = document.getElementById('stepDurations');
                    const durationList = document.getElementById('stepDurationList');
                    const currentTimer = document.getElementById('currentStepTimer');
                    
                    const PHASE_GROUPS = {{
                        'Research Framing': 'Scientific Fact Finding',
                        'Clinical Research': 'Scientific Fact Finding',
                        'Clinical Research Complete': 'Scientific Fact Finding',
                        'Source of Truth': 'Scientific Fact Finding',
                        'Source Validation': 'Scientific Fact Finding',
                        'Report Translation': 'Translation',
                        'Podcast Production': 'Podcast Planning',
                        'Show Outline': 'Podcast Planning',
                        'Script Writing': 'Podcast Planning',
                        'Script Polish': 'Podcast Planning',
                        'Accuracy Audit': 'Podcast Planning',
                        'Audio Production': 'Podcast Recording',
                        'Reuse Analysis': 'Scientific Fact Finding',
                        'Supplemental Research': 'Scientific Fact Finding',
                    }};
                    const GROUP_ORDER = ['Scientific Fact Finding', 'Translation', 'Podcast Planning', 'Podcast Recording'];

                    if ((data.step_durations && data.step_durations.length > 0) || data.current_step_duration_seconds > 0) {{
                        durationBox.style.display = 'block';

                        // Sum completed phase durations into groups
                        const groupTotals = {{}};
                        GROUP_ORDER.forEach(g => groupTotals[g] = 0);
                        if (data.step_durations) {{
                            data.step_durations.forEach(s => {{
                                const group = PHASE_GROUPS[s.phase];
                                if (group) groupTotals[group] += s.duration;
                            }});
                        }}

                        // Add running current step duration to its group
                        const currentGroup = PHASE_GROUPS[data.phase] || null;
                        if (currentGroup && data.current_step_duration_seconds > 0) {{
                            groupTotals[currentGroup] += data.current_step_duration_seconds;
                        }}

                        // Render only groups that have time > 0
                        durationList.innerHTML = GROUP_ORDER
                            .filter(g => groupTotals[g] > 0)
                            .map(g => {{
                                const secs = groupTotals[g];
                                const fmt = `${{Math.floor(secs / 60)}}m ${{Math.floor(secs % 60)}}s`;
                                const isCurrent = (g === currentGroup && data.status === 'running');
                                return `<tr style="border-bottom: 1px dashed var(--border-color);">
                                    <td style="padding: 4px 0;${{isCurrent ? ' font-weight: bold;' : ''}}">${{g}}${{isCurrent ? ' ⏱' : ''}}</td>
                                    <td style="text-align: right; color: var(--text-secondary);">${{fmt}}</td>
                                </tr>`;
                            }}).join('');

                        // Current Step timer — use the same group total for consistency
                        if (currentGroup && data.status === 'running') {{
                            const secs = groupTotals[currentGroup] || 0;
                            const fmt = `${{Math.floor(secs / 60)}}m ${{Math.floor(secs % 60)}}s`;
                            currentTimer.textContent = `${{currentGroup}} — ${{fmt}}`;
                        }} else {{
                            currentTimer.textContent = '';
                        }}
                    }} else {{
                         durationBox.style.display = 'none';
                    }}

                    // Update Research Visualizer
                    if (data.sources && data.sources.length > 0) {{
                        document.getElementById('researchViz').style.display = 'block';
                        document.getElementById('sourceCount').textContent = data.sources.length;
                        
                        const grid = document.getElementById('sourceGrid');
                        grid.innerHTML = data.sources.map(url => {{
                            let domain = '';
                            try {{ domain = new URL(url).hostname; }} catch(e) {{ return ''; }}
                            return `<img src="https://www.google.com/s2/favicons?domain=${{domain}}&sz=64" class="source-icon" title="${{domain}}" />`;
                        }}).join('');
                    }}

                }} else if (data.status === 'uploading') {{
                    progressBar.style.width = '75%';
                    statusDetails.textContent = 'Uploading to external platforms...';
                    statusIcon.textContent = '☁️';
                }} else if (data.status === 'completed') {{
                    progressBar.style.width = '100%';
                    statusDetails.textContent = 'Production Cycle Complete.';
                    statusIcon.textContent = '✅';

                    // Show download links
                    let uploadsHtml = '';
                    if (data.upload_results) {{
                        if (data.upload_results.buzzsprout) {{
                            const bs = data.upload_results.buzzsprout;
                            uploadsHtml += bs.success
                                ? `<a href="${{bs.url}}" target="_blank" class="download-link" style="border-color:#f59e0b; color:#f59e0b; background:rgba(245, 158, 11, 0.1);">🎙️ Buzzsprout Draft (ep #${{bs.episode_id}})</a>`
                                : `<span style="color:#ef4444; font-size:0.9rem;">Buzzsprout upload failed: ${{bs.error}}</span>`;
                        }}
                        if (data.upload_results.youtube) {{
                            const yt = data.upload_results.youtube;
                            uploadsHtml += yt.success
                                ? `<a href="${{yt.url}}" target="_blank" class="download-link" style="border-color:#ef4444; color:#ef4444; background:rgba(239, 68, 68, 0.1);">▶️ YouTube (private)</a>`
                                : `<span style="color:#ef4444; font-size:0.9rem;">YouTube upload failed: ${{yt.error}}</span>`;
                        }}
                    }}

                    downloads.innerHTML = `
                        <h3 style="margin-top:20px; font-size:1.1rem; color:var(--text-primary);">Artifacts Generated:</h3>
                        <div style="display:flex; flex-wrap:wrap; gap:10px;">
                        <a href="/api/download/${{data.task_id}}/audio.wav" class="download-link">🎵 Audio (WAV)</a>
                        <a href="/api/download/${{data.task_id}}/source_of_truth.md" class="download-link">📋 Source of Truth</a>
                        <a href="/api/download/${{data.task_id}}/show_outline.md" class="download-link">📝 Show Outline</a>
                        <a href="/api/download/${{data.task_id}}/accuracy_audit.md" class="download-link">✅ Accuracy Audit</a>
                        <a href="/api/download/${{data.task_id}}/affirmative_case.md" class="download-link">📄 Affirmative Case</a>
                        <a href="/api/download/${{data.task_id}}/grade_synthesis.md" class="download-link">📄 GRADE Synthesis</a>
                        <a href="/api/download/${{data.task_id}}/source_of_truth.pdf" class="download-link">📄 Source of Truth PDF</a>
                        ${{uploadsHtml}}
                        </div>
                    `;
                }} else if (data.status === 'failed') {{
                    progressBar.style.width = '100%';
                    progressBar.style.background = '#ef4444';
                    statusIcon.textContent = '❌';
                    statusDetails.textContent = 'Production failed.';
                    // Extract last meaningful error line instead of full traceback
                    const rawError = data.error || 'Generation failed';
                    const lines = rawError.split('\\n').filter(l => l.trim());
                    const lastLine = lines[lines.length - 1] || rawError;
                    showError(lastLine);
                }}
            }}

            function showError(message) {{
                const error = document.getElementById('error');
                error.textContent = 'SYSTEM ERROR: ' + message;
                error.style.display = 'block';
            }}

            function showQueuedToast(taskId, position) {{
                const toast = document.createElement('div');
                toast.style.cssText = 'position:fixed;top:20px;right:20px;background:#1e293b;border:1px solid #f59e0b;color:#f59e0b;padding:12px 20px;border-radius:8px;z-index:9999;font-size:0.9rem;box-shadow:0 4px 12px rgba(0,0,0,0.3);';
                toast.textContent = `Request queued (#${{position}} in queue). Current task progress continues below.`;
                document.body.appendChild(toast);
                setTimeout(() => toast.remove(), 4000);
            }}

                        async function loadHistory() {{
                try {{
                    const response = await fetch('/api/history');
                    const tasks = await response.json();

                    const historyList = document.getElementById('history');

                    if (tasks.length === 0) {{
                        historyList.innerHTML = '<li style="color: var(--text-secondary); text-align: center; padding: 20px;">No episodes generated yet</li>';
                        return;
                    }}

                    historyList.innerHTML = tasks.map((task, index) => {{
                         const date = new Date(task.created_at).toLocaleString();
                         const artifacts = [
                             {{ name: "Audio (WAV)", file: "audio.wav", icon: "🎵" }},
                             {{ name: "Source of Truth", file: "source_of_truth.md", icon: "📋" }},
                             {{ name: "Show Outline", file: "show_outline.md", icon: "📝" }},
                             {{ name: "Accuracy Audit", file: "accuracy_audit.md", icon: "✅" }},
                             {{ name: "Affirmative Case", file: "affirmative_case.md", icon: "📄" }},
                             {{ name: "GRADE Synthesis", file: "grade_synthesis.md", icon: "⚖️" }}
                         ];
                         
                         const artifactLinks = artifacts.map(a => 
                             `<a href="/api/download/${{task.task_id}}/${{a.file}}" class="artifact-pill" target="_blank">${{a.icon}} ${{a.name}}</a>`
                         ).join('');

                         return `
                        <li class="history-item">
                            <div class="history-summary" onclick="toggleDetails('details-${{task.task_id}}')">
                                <div>
                                    <div class="history-topic">${{task.topic}}</div>
                                    <div class="history-meta">
                                        ${{task.language === 'en' ? 'English' : '日本語'}} • 
                                        ${{date}} • 
                                        Sources: ${{task.sources ? task.sources.length : 0}} • 
                                        <span class="status-${{task.status}}">${{task.status}}</span>
                                    </div>
                                </div>
                                <div>
                                    <span style="font-size: 1.2rem;">${{task.status === 'completed' ? '✅' : '⚙️'}}</span>
                                    <span style="margin-left: 10px; color: var(--text-secondary);">▼</span>
                                </div>
                            </div>
                            <div id="details-${{task.task_id}}" class="history-details">
                                <div style="margin-bottom: 10px; font-weight: 600;">Artifacts Generated (${{task.artifacts_created || 0}}/${{task.artifacts_total || 24}}):</div>
                                <div style="display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px;">
                                    ${{task.status === 'completed' ? artifactLinks : '<span style="color:var(--text-secondary)">Processing...</span>'}}
                                </div>
                                <div style="font-size: 0.8rem; color: var(--text-secondary);">
                                    Task ID: ${{task.task_id}}<br>
                                    Duration: ${{task.step_durations ? Math.round(task.step_durations.reduce((a,b)=>a+b.duration,0)/60) : 0}} min
                                </div>
                            </div>
                        </li>
                    `}}).join('');
                }} catch (error) {{
                    console.error('Failed to load history:', error);
                }}
            }}
            
            window.toggleDetails = function(id) {{
                const el = document.getElementById(id);
                if (el.style.display === 'block') {{
                    el.style.display = 'none';
                    el.classList.remove('open');
                }} else {{
                    el.style.display = 'block';
                    el.classList.add('open');
                }}
            }};

            // Auto-refresh history every 10 seconds
            setInterval(loadHistory, 10000);
        </script>
    </body>
    </html>
    """
    return html

@app.get("/api/system_status")
def get_system_status(username: str = Depends(verify_credentials)):
    """Check git status for unpushed changes."""
    status = {"git_clean": True, "message": "Up to date"}
    try:
        # Check for uncommitted changes (exclude untracked files — they don't affect run integrity)
        proc = subprocess.run(["git", "status", "--porcelain"], cwd=SCRIPT_DIR, capture_output=True, text=True)
        tracked_changes = [l for l in proc.stdout.splitlines() if not l.startswith("??")]
        if tracked_changes:
            status["git_clean"] = False
            status["message"] = "⚠️ Uncommitted changes detected"
            return status

        # Check for unpushed commits
        proc = subprocess.run(["git", "log", "@{u}..HEAD"], cwd=SCRIPT_DIR, capture_output=True, text=True)
        if proc.stdout.strip():
            status["git_clean"] = False
            status["message"] = "⚠️ Local commits not pushed"
            return status

    except Exception as e:
        status["git_clean"] = False
        status["message"] = f"Git check failed: {str(e)}"
    
    return status

@app.get("/api/queue-info")
async def get_queue_info(username: str = Depends(verify_credentials)):
    """Return count of running and queued tasks."""
    with tasks_lock:
        running = sum(1 for t in tasks_db.values() if t["status"] == "running")
        queued = sum(1 for t in tasks_db.values() if t["status"] == "queued")
    return {"running": running, "queued": queued}

@app.post("/api/check-reuse")
async def check_reuse(request: ReuseCheckRequest, username: str = Depends(verify_credentials)):
    """Check if a similar topic has been researched recently.

    Reads the central topic_index.json for candidates from the past 7 days,
    then asks vLLM to score similarity.
    """
    try:
        import re as _re
        from datetime import timedelta

        cutoff_str = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Read the topic index and filter to past 7 days with existing dirs
        index = load_topic_index()
        candidates = []
        for entry in index:
            dir_date = entry.get("created_at", "")[:10]
            if dir_date < cutoff_str:
                continue
            # Verify directory and source_of_truth still exist
            od = Path(entry["output_dir"])
            if not od.exists():
                continue
            sot = od / "source_of_truth.md"
            if not sot.exists():
                sot = od / "SOURCE_OF_TRUTH.md"
            if not sot.exists():
                continue
            candidates.append(entry)

        if not candidates:
            return {"has_match": False, "matches": []}

        # Build LLM prompt to score similarity
        candidate_topics = [c["topic"] for c in candidates]
        prompt = (
            f"You are a topic similarity scorer. Compare the NEW topic against each PREVIOUS topic.\n"
            f"Return a JSON array of integers (0-100) representing similarity scores.\n"
            f"100 = identical topic, 70+ = very similar/overlapping, 50-69 = somewhat related, <50 = different.\n\n"
            f"NEW TOPIC: {request.topic}\n\n"
            f"PREVIOUS TOPICS:\n"
        )
        for i, t in enumerate(candidate_topics):
            prompt += f"  {i+1}. {t}\n"
        prompt += f"\nReturn ONLY a JSON array of {len(candidate_topics)} integers. Example: [85, 30, 72]"

        # Discover the model name from the vLLM server
        llm_base = os.environ["LLM_BASE_URL"]
        models_resp = httpx.get(f"{llm_base}/models", timeout=10.0)
        models_resp.raise_for_status()
        model_name = models_resp.json()["data"][0]["id"]

        resp = httpx.post(
            f"{llm_base}/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 512,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()

        # Parse scores from LLM response
        array_match = _re.search(r'\[[\d\s,]+\]', content)
        if not array_match:
            return {"has_match": False, "matches": []}
        scores = json.loads(array_match.group())

        # Pair scores with candidates, filter >= 70, take top 3
        matches = []
        for i, candidate in enumerate(candidates):
            if i < len(scores) and scores[i] >= 70:
                candidate["similarity_score"] = scores[i]
                matches.append(candidate)
        matches.sort(key=lambda m: m["similarity_score"], reverse=True)
        matches = matches[:3]

        return {"has_match": len(matches) > 0, "matches": matches}

    except Exception as e:
        print(f"check-reuse error (graceful fallback): {e}")
        return {"has_match": False, "matches": []}


@app.post("/api/generate-reuse")
async def generate_reuse(request: ReuseGenerateRequest, username: str = Depends(verify_credentials)):
    """Start podcast generation reusing a previous run's research."""
    import re as _re

    # Resolve reuse_dir from output_dir or task_id
    reuse_dir = None
    if request.reuse_output_dir:
        reuse_dir = Path(request.reuse_output_dir)
    elif request.reuse_task_id:
        prev_task = tasks_db.get(request.reuse_task_id)
        if prev_task and prev_task.get("output_dir"):
            reuse_dir = Path(prev_task["output_dir"])

    if not reuse_dir or not reuse_dir.exists():
        raise HTTPException(status_code=400, detail="Previous output directory not found or no longer exists")

    # Check it has source_of_truth.md
    sot = reuse_dir / "source_of_truth.md"
    if not sot.exists():
        sot = reuse_dir / "SOURCE_OF_TRUTH.md"
    if not sot.exists():
        raise HTTPException(status_code=400, detail="Previous output directory has no source_of_truth.md")

    # Determine old params: try tasks_db first, then session_metadata.txt
    old_lang = "en"
    old_access = "simple"
    old_length = "long"
    old_hosts = "random"

    # Check tasks_db
    for tid, task in tasks_db.items():
        if task.get("output_dir") == str(reuse_dir):
            old_lang = task.get("language", "en")
            old_access = task.get("accessibility_level", "simple")
            old_length = task.get("podcast_length", "long")
            old_hosts = task.get("podcast_hosts", "random")
            break
    else:
        # Fallback: parse session_metadata.txt for language
        meta_path = reuse_dir / "session_metadata.txt"
        if meta_path.exists():
            meta_text = meta_path.read_text()
            lang_match = _re.search(r'^Language:\s*\S+\s*\((\w+)\)', meta_text, _re.MULTILINE)
            if lang_match:
                old_lang = lang_match.group(1).strip()

    params_match = (
        old_lang == request.language
        and old_access == request.accessibility_level
        and old_length == request.podcast_length
    )
    hosts_match = old_hosts == request.podcast_hosts

    if params_match and hosts_match:
        reuse_mode = "check_supplemental"  # LLM decides full_reuse vs supplemental
    elif params_match and not hosts_match:
        reuse_mode = "tts_only"
    else:
        reuse_mode = "crew3_reuse"

    task_id = secrets.token_hex(8)
    task = {
        "task_id": task_id,
        "topic": request.topic,
        "language": request.language,
        "accessibility_level": request.accessibility_level,
        "podcast_length": request.podcast_length,
        "podcast_hosts": request.podcast_hosts,
        "status": "queued",
        "progress": 0,
        "phase": "Queued (Reuse)",
        "created_at": datetime.now().isoformat(),
        "error": None,
        "output_dir": None,
        "upload_buzzsprout": request.upload_to_buzzsprout,
        "upload_youtube": request.upload_to_youtube,
        "upload_results": {},
        "sources": [],
        "start_time": time.time(),
        "estimated_remaining": None,
        "reuse_from": request.reuse_task_id,
        "reuse_dir": str(reuse_dir),
        "reuse_mode": reuse_mode,
    }

    with tasks_lock:
        tasks_db[task_id] = task
        save_tasks()

    # Store upload credentials in task data for explicit passing (not os.environ)
    task["buzzsprout_api_key"] = request.buzzsprout_api_key or None
    task["buzzsprout_account_id"] = request.buzzsprout_account_id or None
    task["youtube_secret_path"] = request.youtube_secret_path or None

    task_queue.put(task)
    return {"task_id": task_id, "status": "queued", "reuse_mode": reuse_mode}


@app.post("/api/generate-intro")
async def generate_intro(request: GenerateIntroRequest, username: str = Depends(verify_credentials)):
    """Generate a channel intro using the LLM (smart model with fast-model fallback)."""
    import re
    lang_label = "Japanese" if request.language == "ja" else "English"
    name    = request.channel_name    or "our podcast"
    target  = request.core_target     or "curious listeners"
    mission = request.channel_mission or "turning science into everyday wisdom"

    if request.language == "ja":
        prompt = (
            f"以下の情報をもとに、ポッドキャストの自然なチャンネル紹介文を日本語で書いてください。\n\n"
            f"チャンネル名: {name}\n"
            f"ターゲットリスナー: {target}\n"
            f"チャンネルのミッション: {mission}\n\n"
            f"2〜3文、約60〜80文字で、ホストが話しているような自然な口語体で書いてください。"
            f"スピーカーラベルは不要です。紹介文のテキストのみ出力してください。"
        )
    else:
        prompt = (
            f"Write a natural, engaging channel introduction for a podcast.\n\n"
            f"Channel Name: {name}\n"
            f"Target Audience: {target}\n"
            f"Channel Mission: {mission}\n\n"
            f"Write 2-3 sentences (~30-40 words) in first person, as if the host is speaking aloud. "
            f"Sound warm and human — not corporate. Do NOT include speaker labels or quotes. "
            f"Output ONLY the intro text, nothing else."
        )

    payload = {
        "model": "",  # filled per endpoint below
        "messages": [{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}],
        "max_tokens": 120,
        "temperature": 0.8,
    }

    llm_base       = os.environ["LLM_BASE_URL"].rstrip("/")
    model_name     = os.environ["MODEL_NAME"]
    fast_base      = os.environ["FAST_LLM_BASE_URL"].rstrip("/")
    fast_model     = os.environ["FAST_MODEL_NAME"]

    async with httpx.AsyncClient(timeout=30) as client:
        # Primary: MODEL_NAME / LLM_BASE_URL from .env
        try:
            payload["model"] = model_name
            r = await client.post(f"{llm_base}/chat/completions", json=payload)
            if r.status_code == 200:
                intro = r.json()["choices"][0]["message"]["content"].strip()
                intro = re.sub(r"<think>.*?</think>", "", intro, flags=re.DOTALL).strip()
                return {"intro": intro}
            print(f"[generate-intro] primary non-200: {r.status_code} {r.text[:200]}")
        except Exception as e:
            print(f"[generate-intro] primary error: {e}")

        # Fallback: FAST_MODEL_NAME / FAST_LLM_BASE_URL from .env
        if fast_base != llm_base or fast_model != model_name:
            try:
                payload["model"] = fast_model
                r = await client.post(f"{fast_base}/chat/completions", json=payload)
                if r.status_code == 200:
                    intro = r.json()["choices"][0]["message"]["content"].strip()
                    intro = re.sub(r"<think>.*?</think>", "", intro, flags=re.DOTALL).strip()
                    return {"intro": intro}
                print(f"[generate-intro] fallback non-200: {r.status_code} {r.text[:200]}")
            except Exception as e:
                print(f"[generate-intro] fallback error: {e}")

    raise HTTPException(status_code=503, detail="LLM unavailable — fill in the intro manually.")


@app.post("/api/generate")
async def generate_podcast(request: PodcastRequest, username: str = Depends(verify_credentials)):
    """Start podcast generation"""
    task_id = secrets.token_hex(8)

    # Create task record
    task = {
        "task_id": task_id,
        "topic": request.topic,
        "language": request.language,
        "accessibility_level": request.accessibility_level,
        "podcast_length": request.podcast_length,
        "podcast_hosts": request.podcast_hosts,
        "channel_intro": request.channel_intro,
        "core_target": request.core_target,
        "channel_mission": request.channel_mission,

        "status": "queued", # Start as queued
        "progress": 0,
        "phase": "Queued",
        "created_at": datetime.now().isoformat(),
        "error": None,
        "output_dir": None,
        "upload_buzzsprout": request.upload_to_buzzsprout,
        "upload_youtube": request.upload_to_youtube,
        "upload_results": {},
        "sources": [],
        "start_time": time.time(),
        "estimated_remaining": None
    }

    with tasks_lock:
        tasks_db[task_id] = task
        save_tasks()

    # Store upload credentials in task data for explicit passing (not os.environ)
    task["buzzsprout_api_key"] = request.buzzsprout_api_key or None
    task["buzzsprout_account_id"] = request.buzzsprout_account_id or None
    task["youtube_secret_path"] = request.youtube_secret_path or None

    # Add to queue instead of starting thread immediately
    task_queue.put(task)

    return {"task_id": task_id, "status": "queued"}

# Phase markers parsed from pipeline.py stdout
# Each entry: (log_marker, display_name, progress_pct, is_phase)
# is_phase=True  → opens/closes the phase timer, recorded in step_durations
# is_phase=False → updates the current-step label and progress bar only
PHASE_MARKERS = [
    ("PHASE 0: RESEARCH FRAMING", "Research Framing", 5, True),
    # Clinical sub-steps (parallel a/b) — progress only, no phase timer reset
    ("STEP 1a:", "Step 1a: Search Strategy (Aff)", 8, False),
    ("STEP 1b:", "Step 1b: Search Strategy (Fal)", 8, False),
    ("STEP 2a:", "Step 2a: Wide Net (Aff)", 10, False),
    ("STEP 2b:", "Step 2b: Wide Net (Fal)", 10, False),
    ("STEP 3a:", "Step 3a: Screening (Aff)", 13, False),
    ("STEP 3b:", "Step 3b: Screening (Fal)", 13, False),
    ("STEP 4a:", "Step 4a: Deep Extraction (Aff)", 15, False),
    ("STEP 4b:", "Step 4b: Deep Extraction (Fal)", 15, False),
    ("STEP 5a:", "Step 5a: Affirmative Case", 17, False),
    ("STEP 5b:", "Step 5b: Falsification Case", 17, False),
    ("STEP 6:", "Step 6: Deterministic Math", 18, False),
    ("STEP 7:", "Step 7: GRADE Synthesis", 19, False),
    # Phase-level markers resume
    ("PHASE 1: CLINICAL RESEARCH", "Clinical Research", 15, True),
    ("ALL RESEARCH COMPLETE", "Clinical Research Complete", 40, True),
    ("Source of Truth generated", "Source of Truth", 45, True),
    # Post-research phases
    ("PHASE 2: SOURCE VALIDATION", "Source Validation", 50, True),
    ("PHASE 3: REPORT TRANSLATION", "Report Translation", 55, True),
    ("CREW 3:", "Podcast Production", 58, True),
    ("PHASE 4: SHOW OUTLINE", "Show Outline", 60, True),
    ("PHASE 5: SCRIPT WRITING", "Script Writing", 70, True),
    ("PHASE 6: SCRIPT POLISH", "Script Polish", 85, True),
    ("PHASE 7: ACCURACY AUDIT", "Accuracy Audit", 90, True),
    ("Starting BGM Merging Phase", "Audio Production", 95, True),
    ("SUCCESS: Audio duration", "Complete", 100, True),
    # Reuse-mode markers
    ("REUSE MODE:", "Reuse Analysis", 5, True),
    ("SUPPLEMENTAL RESEARCH", "Supplemental Research", 20, True),
    ("REUSE_COMPLETE:", "Complete (Cached)", 100, True),
]


def _close_phase(task_id: str):
    """Record the elapsed time of the current phase into step_durations."""
    task = tasks_db.get(task_id, {})
    prev_phase = task.get("phase", "")
    prev_start = task.get("phase_start_time")
    if not prev_start or not prev_phase:
        return
    duration = time.time() - prev_start
    step_durations = task.setdefault("step_durations", [])
    existing = next((s for s in step_durations if s["phase"] == prev_phase), None)
    if existing:
        existing["duration"] += duration
        total = existing["duration"]
        existing["duration_formatted"] = f"{int(total // 60)}m {int(total % 60)}s"
    else:
        step_durations.append({
            "phase": prev_phase,
            "duration": duration,
            "duration_formatted": f"{int(duration // 60)}m {int(duration % 60)}s",
        })

def _preflight_check() -> Optional[str]:
    """Test vLLM and Ollama reachability. Returns error message or None."""
    errors = []
    for name, url in [("vLLM", "http://localhost:8000/v1/models"),
                      ("Ollama", "http://localhost:11434/api/tags")]:
        try:
            resp = httpx.get(url, timeout=3.0)
            resp.raise_for_status()
        except Exception as e:
            errors.append(f"{name} unreachable at {url}: {e}")
    return "; ".join(errors) if errors else None


def _stream_process_output(proc, task_id: str) -> list:
    """Stream subprocess stdout, parse phase markers and sources, discover output_dir."""
    output_lines = []
    start_time = tasks_db[task_id]["start_time"]
    output_dir_discovered = False

    for line in proc.stdout:
        output_lines.append(line)

        # 0. Discover output_dir from [OUTPUT_DIR] marker or fallback
        if not output_dir_discovered:
            if "[OUTPUT_DIR]" in line:
                try:
                    discovered = line.split("[OUTPUT_DIR]")[1].strip()
                    if discovered:
                        tasks_db[task_id]["output_dir"] = discovered
                        output_dir_discovered = True
                except Exception:
                    pass
            if not output_dir_discovered:
                found_dir = _find_latest_output_dir()
                if found_dir:
                    tasks_db[task_id]["output_dir"] = str(found_dir)
                    output_dir_discovered = True

        # 1. Parse Phase Markers
        for marker, phase_name, progress_pct, is_phase in PHASE_MARKERS:
            if marker in line:
                current_time = time.time()
                tasks_db[task_id]["progress"] = progress_pct
                tasks_db[task_id]["current_step"] = phase_name

                if is_phase:
                    _close_phase(task_id)
                    tasks_db[task_id]["phase"] = phase_name
                    tasks_db[task_id]["phase_start_time"] = current_time

                elapsed = current_time - start_time
                if progress_pct > 5:
                    total_est = (elapsed / progress_pct) * 100
                    remaining = total_est - elapsed
                    tasks_db[task_id]["estimated_remaining"] = max(0, remaining)

                save_tasks()
                break

        # 2. Parse Sources
        if "[SOURCE]" in line:
            try:
                url = line.split("[SOURCE]")[1].strip()
                if url not in tasks_db[task_id]["sources"]:
                    tasks_db[task_id]["sources"].append(url)
                    save_tasks()
            except Exception:
                pass

    return output_lines


def run_podcast_generation(task_id: str, topic: str, language: str,
                           accessibility_level: str = "simple",
                           podcast_length: str = "long", podcast_hosts: str = "random",
                           channel_intro: str = "",
                           upload_buzzsprout: bool = False, upload_youtube: bool = False,
                           core_target: str = "", channel_mission: str = "",
                           buzzsprout_api_key: str = None,
                           buzzsprout_account_id: str = None,
                           youtube_secret_path: str = None):
    """Run pipeline.py in background with real-time phase tracking."""
    try:
        preflight_err = _preflight_check()
        if preflight_err:
            tasks_db[task_id]["status"] = "failed"
            tasks_db[task_id]["error"] = f"Pre-flight check failed: {preflight_err}"
            save_tasks()
            return

        tasks_db[task_id]["status"] = "running"
        tasks_db[task_id]["progress"] = 0
        tasks_db[task_id]["phase"] = ""        # set by first is_phase marker
        tasks_db[task_id]["current_step"] = "Initializing..."
        tasks_db[task_id]["phase_start_time"] = time.time()
        tasks_db[task_id]["step_durations"] = []
        save_tasks()

        env = os.environ.copy()
        env["ACCESSIBILITY_LEVEL"] = accessibility_level
        env["PODCAST_LENGTH"] = podcast_length
        env["PODCAST_HOSTS"] = podcast_hosts
        if channel_intro:
            env["PODCAST_CHANNEL_INTRO"] = channel_intro
        if core_target:
            env["PODCAST_CORE_TARGET"] = core_target
        if channel_mission:
            env["PODCAST_CHANNEL_MISSION"] = channel_mission

        proc = subprocess.Popen(
            [str(PODCAST_ENV_PYTHON), "-m", "dr2_podcast.pipeline", "--topic", topic, "--language", language],
            cwd=SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        output_lines = _stream_process_output(proc, task_id)

        proc.wait()

        if proc.returncode != 0:
            tasks_db[task_id]["status"] = "failed"
            # Filter out pydantic caret-only lines and log prefixes for cleaner error display
            raw_lines = output_lines[-100:]
            clean_lines = []
            for l in raw_lines:
                # Strip log prefix (timestamp + level) to get the actual message
                stripped = l.rstrip('\n')
                # Skip lines that are only carets/whitespace (pydantic error formatting)
                msg = stripped.split(' - ERROR - ', 1)[-1] if ' - ERROR - ' in stripped else stripped
                if msg.strip() and msg.strip().replace('^', '').strip():
                    clean_lines.append(msg)
            error_text = "\n".join(clean_lines[-50:])  # Last 50 meaningful lines
            tasks_db[task_id]["error"] = error_text or f"Process exited with code {proc.returncode}"
            save_tasks()
            return

        # Find the most recent timestamped output directory
        output_dir = _find_latest_output_dir()
        if output_dir:
            tasks_db[task_id]["output_dir"] = str(output_dir)
        save_tasks()

        # Generation succeeded — run uploads if requested
        resolved_dir = Path(tasks_db[task_id].get("output_dir") or str(OUTPUT_DIR))
        audio_path = str(resolved_dir / "audio.wav")
        title = topic.strip()

        if upload_buzzsprout or upload_youtube:
            tasks_db[task_id]["status"] = "uploading"
            tasks_db[task_id]["phase"] = "Uploading"
            tasks_db[task_id]["progress"] = 95
            save_tasks()

            if upload_buzzsprout:
                tasks_db[task_id]["upload_results"]["buzzsprout"] = upload_to_buzzsprout(
                    audio_path, title,
                    api_key=buzzsprout_api_key, account_id=buzzsprout_account_id)
                save_tasks()

            if upload_youtube:
                tasks_db[task_id]["upload_results"]["youtube"] = upload_to_youtube(
                    audio_path, title, youtube_secret_path=youtube_secret_path)
                save_tasks()

        # Record the final phase's duration before marking complete
        _close_phase(task_id)
        tasks_db[task_id]["status"] = "completed"
        tasks_db[task_id]["progress"] = 100
        tasks_db[task_id]["phase"] = "Complete"
        tasks_db[task_id]["current_step"] = "Complete"
        tasks_db[task_id]["completed_at"] = datetime.now().isoformat()

        # Register in topic index
        if tasks_db[task_id].get("output_dir"):
            register_topic(
                topic=topic, output_dir=tasks_db[task_id]["output_dir"],
                language=language, accessibility_level=accessibility_level,
                podcast_length=podcast_length, podcast_hosts=podcast_hosts,
            )

    except Exception as e:
        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["error"] = str(e)
        # Still register if research completed (source_of_truth.md exists)
        od = tasks_db[task_id].get("output_dir")
        if od:
            sot = Path(od) / "source_of_truth.md"
            if not sot.exists():
                sot = Path(od) / "SOURCE_OF_TRUTH.md"
            if sot.exists():
                register_topic(
                    topic=topic, output_dir=od,
                    language=language, accessibility_level=accessibility_level,
                    podcast_length=podcast_length, podcast_hosts=podcast_hosts,
                )
    finally:
        save_tasks()

def _find_latest_output_dir() -> Optional[Path]:
    """Find the most recently created timestamped subdirectory in research_outputs/."""
    if not OUTPUT_DIR.exists():
        return None
    subdirs = [d for d in OUTPUT_DIR.iterdir() if d.is_dir() and d.name[0:4].isdigit()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda d: d.stat().st_mtime)


def run_podcast_reuse(task_data: dict):
    """Run podcast generation in reuse mode (tts_only, crew3_reuse, or check_supplemental)."""
    task_id = task_data["task_id"]
    reuse_mode = task_data["reuse_mode"]
    reuse_dir = Path(task_data["reuse_dir"])
    topic = task_data["topic"]
    language = task_data.get("language", "en")
    podcast_hosts = task_data.get("podcast_hosts", "random")
    upload_buzzsprout = task_data.get("upload_buzzsprout", False)
    upload_youtube = task_data.get("upload_youtube", False)
    buzzsprout_api_key = task_data.get("buzzsprout_api_key")
    buzzsprout_account_id = task_data.get("buzzsprout_account_id")
    youtube_secret_path = task_data.get("youtube_secret_path")

    try:
        if reuse_mode != "tts_only":
            preflight_err = _preflight_check()
            if preflight_err:
                tasks_db[task_id]["status"] = "failed"
                tasks_db[task_id]["error"] = f"Pre-flight check failed: {preflight_err}"
                save_tasks()
                return

        tasks_db[task_id]["status"] = "running"
        tasks_db[task_id]["progress"] = 0
        tasks_db[task_id]["phase"] = ""
        tasks_db[task_id]["current_step"] = "Initializing..."
        tasks_db[task_id]["phase_start_time"] = time.time()
        tasks_db[task_id]["step_durations"] = []
        save_tasks()

        if reuse_mode == "tts_only":
            _run_tts_only_reuse(task_id, task_data, reuse_dir)
        elif reuse_mode in ("crew3_reuse", "check_supplemental"):
            _run_subprocess_reuse(task_id, task_data, reuse_dir)
        else:
            raise ValueError(f"Unknown reuse_mode: {reuse_mode}")

        # Handle uploads
        resolved_dir = Path(tasks_db[task_id].get("output_dir") or str(OUTPUT_DIR))
        audio_path = str(resolved_dir / "audio.wav")

        if upload_buzzsprout or upload_youtube:
            tasks_db[task_id]["status"] = "uploading"
            tasks_db[task_id]["phase"] = "Uploading"
            tasks_db[task_id]["progress"] = 95
            save_tasks()

            if upload_buzzsprout:
                tasks_db[task_id]["upload_results"]["buzzsprout"] = upload_to_buzzsprout(
                    audio_path, topic,
                    api_key=buzzsprout_api_key, account_id=buzzsprout_account_id)
                save_tasks()
            if upload_youtube:
                tasks_db[task_id]["upload_results"]["youtube"] = upload_to_youtube(
                    audio_path, topic, youtube_secret_path=youtube_secret_path)
                save_tasks()

        _close_phase(task_id)
        tasks_db[task_id]["status"] = "completed"
        tasks_db[task_id]["progress"] = 100
        tasks_db[task_id]["phase"] = "Complete"
        tasks_db[task_id]["current_step"] = "Complete"
        tasks_db[task_id]["completed_at"] = datetime.now().isoformat()

        # Register in topic index
        if tasks_db[task_id].get("output_dir"):
            register_topic(
                topic=topic, output_dir=tasks_db[task_id]["output_dir"],
                language=language,
                accessibility_level=task_data.get("accessibility_level", "simple"),
                podcast_length=task_data.get("podcast_length", "long"),
                podcast_hosts=podcast_hosts,
            )

    except Exception as e:
        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["error"] = str(e)
    finally:
        save_tasks()


def _run_tts_only_reuse(task_id: str, task_data: dict, reuse_dir: Path):
    """TTS-only reuse: copy research + script, regenerate audio with new hosts."""
    from dr2_podcast.audio.engine import generate_audio_from_script, clean_script_for_tts, post_process_audio

    # Create new timestamped output dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_output_dir = OUTPUT_DIR / timestamp
    new_output_dir.mkdir(parents=True, exist_ok=True)
    tasks_db[task_id]["output_dir"] = str(new_output_dir)

    # Phase: Copying artifacts
    tasks_db[task_id]["phase"] = "Copying Artifacts"
    tasks_db[task_id]["progress"] = 10
    save_tasks()

    # Copy all artifacts from previous run
    for item in reuse_dir.iterdir():
        if item.is_file():
            dest = new_output_dir / item.name
            # Skip audio files — we'll regenerate them
            if item.suffix in ('.wav', '.mp3'):
                continue
            shutil.copy2(item, dest)

    # Load the polished script
    script_path = new_output_dir / "script_final.md"
    if not script_path.exists():
        script_path = new_output_dir / "PODCAST_SCRIPT_POLISHED.md"
    if not script_path.exists():
        raise FileNotFoundError(f"No polished script found in {reuse_dir}")

    script_text = script_path.read_text()

    # Phase: TTS Generation
    tasks_db[task_id]["phase"] = "TTS Generation"
    tasks_db[task_id]["progress"] = 50
    save_tasks()

    # Determine language TTS code
    lang_code = "a" if task_data.get("language", "en") == "en" else "j"

    cleaned_script = clean_script_for_tts(script_text)
    output_path = new_output_dir / "audio.wav"

    tts_result = generate_audio_from_script(cleaned_script, str(output_path), lang_code=lang_code)
    if not tts_result:
        raise RuntimeError("TTS generation failed")
    audio_file_path, positions = tts_result if isinstance(tts_result, tuple) else (tts_result, [])
    audio_file = Path(audio_file_path)

    # Phase: BGM Merging
    tasks_db[task_id]["phase"] = "BGM Merging"
    tasks_db[task_id]["progress"] = 85
    save_tasks()

    if audio_file.exists():
        try:
            mastered = post_process_audio(str(audio_file), bgm_target="Interesting BGM.wav")
            if mastered and os.path.exists(mastered) and mastered != str(audio_file):
                audio_file = Path(mastered)
        except Exception as e:
            print(f"BGM merging warning: {e}")

    tasks_db[task_id]["progress"] = 100
    save_tasks()


def _run_subprocess_reuse(task_id: str, task_data: dict, reuse_dir: Path):
    """Run crew3_reuse or check_supplemental via pipeline.py subprocess."""
    reuse_mode = task_data["reuse_mode"]
    topic = task_data["topic"]
    language = task_data.get("language", "en")

    env = os.environ.copy()
    env["ACCESSIBILITY_LEVEL"] = task_data.get("accessibility_level", "simple")
    env["PODCAST_LENGTH"] = task_data.get("podcast_length", "long")
    env["PODCAST_HOSTS"] = task_data.get("podcast_hosts", "random")

    cmd = [
        str(PODCAST_ENV_PYTHON), "-m", "dr2_podcast.pipeline",
        "--topic", topic,
        "--language", language,
        "--reuse-dir", str(reuse_dir),
    ]
    if reuse_mode == "crew3_reuse":
        cmd.append("--crew3-only")
    elif reuse_mode == "check_supplemental":
        cmd.append("--check-supplemental")

    proc = subprocess.Popen(
        cmd, cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )

    output_lines = _stream_process_output(proc, task_id)

    proc.wait()

    if proc.returncode != 0:
        raw_lines = output_lines[-100:]
        clean_lines = []
        for l in raw_lines:
            stripped = l.rstrip('\n')
            msg = stripped.split(' - ERROR - ', 1)[-1] if ' - ERROR - ' in stripped else stripped
            if msg.strip() and msg.strip().replace('^', '').strip():
                clean_lines.append(msg)
        error_text = "\n".join(clean_lines[-50:])
        raise RuntimeError(error_text or f"Process exited with code {proc.returncode}")

    # Find output dir
    output_dir = _find_latest_output_dir()
    if output_dir:
        tasks_db[task_id]["output_dir"] = str(output_dir)
    save_tasks()


@app.get("/api/status/{task_id}")
async def get_status(task_id: str, username: str = Depends(verify_credentials)):
    """Get status of a specific task"""
    with tasks_lock:
        if task_id not in tasks_db:
            raise HTTPException(status_code=404, detail="Task not found")

        task = tasks_db[task_id]

        # Calculate artifact counts on-the-fly
        if task.get("output_dir"):
            created, total = count_artifacts(task["output_dir"], task.get("language", "en"))
            task["artifacts_created"] = created
            task["artifacts_total"] = total
        else:
            task["artifacts_created"] = 0
            task["artifacts_total"] = len(EXPECTED_ARTIFACTS + EXPECTED_ARTIFACTS_EXTRA.get(task.get("language", "en"), []))

        # Calculate current step duration
        current_step_duration = 0
        if task["status"] == "running" and task.get("phase_start_time"):
            current_step_duration = time.time() - task["phase_start_time"]

        response = {k: v for k, v in task.items() if k not in _CREDENTIAL_FIELDS}
        response["current_step_duration"] = f"{int(current_step_duration // 60)}m {int(current_step_duration % 60)}s"
        response["current_step_duration_seconds"] = current_step_duration

    return response

@app.get("/api/history")
async def get_history(username: str = Depends(verify_credentials)):
    """Get list of past production runs"""
    with tasks_lock:
        # Sort by created_at desc
        sorted_tasks = sorted(
            tasks_db.values(),
            key=lambda x: x["created_at"],
            reverse=True
        )

        # Calculate artifact counts for history items
        for task in sorted_tasks:
            if task.get("output_dir"):
                created, total = count_artifacts(task["output_dir"], task.get("language", "en"))
                task["artifacts_created"] = created
                task["artifacts_total"] = total
            else:
                task["artifacts_created"] = 0
                task["artifacts_total"] = len(EXPECTED_ARTIFACTS + EXPECTED_ARTIFACTS_EXTRA.get(task.get("language", "en"), []))

    # Return last 20 tasks, newest first — strip credential fields
    return [{k: v for k, v in t.items() if k not in _CREDENTIAL_FIELDS} for t in sorted_tasks[:20]]

@app.get("/api/download/{task_id}/{filename}")
def download_file(task_id: str, filename: str, username: str = Depends(verify_credentials)):
    """Download generated file"""
    with tasks_lock:
        if task_id not in tasks_db:
            raise HTTPException(status_code=404, detail="Task not found")
        task = tasks_db[task_id]
        output_dir = Path(task["output_dir"]) if task.get("output_dir") else OUTPUT_DIR
    file_path = (output_dir / filename).resolve()
    if not file_path.is_relative_to(output_dir.resolve()):
        raise HTTPException(status_code=403, detail="Access denied")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

class YoutubePreflightRequest(BaseModel):
    secret_path: str = "./client_secret.json"

@app.post("/api/youtube/preflight")
async def youtube_preflight(request: YoutubePreflightRequest = YoutubePreflightRequest(), username: str = Depends(verify_credentials)):
    """Run YouTube OAuth consent flow (may open browser). Must be called before generation."""
    from dr2_podcast.tools.upload_utils import get_youtube_credentials
    try:
        if request.secret_path:
            os.environ["YOUTUBE_CLIENT_SECRET_PATH"] = request.secret_path
        get_youtube_credentials()
        return {"ready": True}
    except Exception as e:
        return {"ready": False, "error": str(e)}

@app.get("/api/upload_config")
async def upload_config(username: str = Depends(verify_credentials)):
    """Report which upload platforms have valid credentials configured."""
    buzzsprout_ok = bool(os.getenv("BUZZSPROUT_API_KEY")) and bool(os.getenv("BUZZSPROUT_ACCOUNT_ID"))

    secret_path = os.getenv("YOUTUBE_CLIENT_SECRET_PATH", "./client_secret.json")
    youtube_ok = (SCRIPT_DIR / secret_path).exists()

    return {"buzzsprout_configured": buzzsprout_ok, "youtube_configured": youtube_ok}

if __name__ == "__main__":
    port = int(os.getenv("PODCAST_WEB_PORT", 8501))

    print(f"\nStarting DR_2_Podcast Web UI on http://0.0.0.0:{port}")
    print(f"Access from browser: http://localhost:{port}")
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=port)
