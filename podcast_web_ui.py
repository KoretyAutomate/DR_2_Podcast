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

from upload_utils import validate_upload_config, upload_to_buzzsprout, upload_to_youtube

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
OUTPUT_DIR = SCRIPT_DIR / "research_outputs"
TASKS_FILE = SCRIPT_DIR / "podcast_tasks.json"

# Use podcast_env Python interpreter if available
PODCAST_ENV_PYTHON = Path.home() / "miniconda3" / "envs" / "podcast_flow" / "bin" / "python3"
if not PODCAST_ENV_PYTHON.exists():
    PODCAST_ENV_PYTHON = sys.executable  # Fallback to current Python

# Simple authentication (optional - can be disabled)
USERNAME = os.getenv("PODCAST_WEB_USER", "admin")
PASSWORD = os.getenv("PODCAST_WEB_PASSWORD", secrets.token_urlsafe(16))

print("="*60)
print("PODCAST WEB UI CREDENTIALS")
print("="*60)
print(f"Username: {USERNAME}")
print(f"Password: {PASSWORD}")
print("\nSet custom credentials with:")
print("export PODCAST_WEB_USER=your_username")
print("export PODCAST_WEB_PASSWORD=your_password")
print("="*60)

app = FastAPI(title="DR_2_Podcast Generator")
security = HTTPBasic()

# Task storage
tasks_db: Dict[str, Dict] = {}

# Task Queue
task_queue = queue.Queue()
# Expected artifacts for progress tracking
EXPECTED_ARTIFACTS = [
    "research_framing.md", "research_framing.pdf",
    "deep_research_lead.md", "deep_research_counter.md", "deep_research_audit.md",
    "deep_research_sources.json",
    "gap_analysis.md",
    "adversarial_research.md", "adversarial_paper.pdf",
    "supporting_research.md", "supporting_paper.pdf",
    "source_verification.md", "verified_sources_bibliography.pdf",
    "source_of_truth.md", "source_of_truth.pdf",
    "podcast_script_raw.md", "podcast_script_polished.md", "podcast_script.txt",
    "show_notes.md", "accuracy_check.md", "accuracy_check.pdf",
    "podcast_generation.log", "session_metadata.txt",
    "podcast_final_audio.wav"
]

def count_artifacts(directory: Optional[str]) -> tuple[int, int]:
    """Count generated artifacts vs expected total."""
    if not directory or not os.path.exists(directory):
        return 0, len(EXPECTED_ARTIFACTS)
    
    found = 0
    for filename in EXPECTED_ARTIFACTS:
        if (Path(directory) / filename).exists():
            found += 1
            
    return found, len(EXPECTED_ARTIFACTS)

current_task_id = None

def load_tasks():
    """Load tasks from file and clean up interrupted runs."""
    global tasks_db
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
    with open(TASKS_FILE, 'w') as f:
        json.dump(tasks_db, f, indent=2)

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
            
            # Execute the generation
            run_podcast_generation(
                task_id, 
                task_data["topic"],
                task_data["language"],
                task_data["accessibility_level"],
                task_data["podcast_length"],
                task_data["podcast_hosts"],
                task_data["upload_buzzsprout"],
                task_data["upload_youtube"]
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

class PodcastRequest(BaseModel):
    topic: str
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

            input[type="text"], select {{
                width: 100%;
                background: rgba(15, 23, 42, 0.6);
                border: 1px solid var(--border-color);
                color: var(--text-primary);
                padding: 12px 16px;
                border-radius: 8px;
                font-size: 1rem;
                font-family: inherit;
                transition: all 0.3s ease;
            }}

            input[type="text"]:focus, select:focus {{
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
                    <label for="topic">Scientific Topic</label>
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
                                <option value="kaz_erika">Kaz Leads (Teaching)</option>
                                <option value="erika_kaz">Erika Leads (Teaching)</option>
                            </select>
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
                <h2>Production History</h2>
                <ul id="history" class="history">
                    <li style="color: var(--text-secondary); text-align: center; padding: 20px;">No episodes generated yet</li>
                </ul>
            </div>
        </div>

        <script>
            let currentTaskId = null;
            let statusInterval = null;

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

            // Form submission
            document.getElementById('podcastForm').addEventListener('submit', async (e) => {{
                e.preventDefault();

                const topic = document.getElementById('topic').value;
                const language = document.getElementById('language').value;
                const accessibility = document.getElementById('accessibility').value;
                const button = document.getElementById('generateBtn');
                const statusBox = document.getElementById('statusBox');

                button.disabled = true;
                button.textContent = 'Initializing Core Systems...';
                statusBox.classList.add('show');
                document.getElementById('error').style.display = 'none';

                try {{
                    const response = await fetch('/api/generate', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            topic, language,
                            accessibility_level: accessibility,
                            podcast_length: document.getElementById('length').value,
                            podcast_hosts: document.getElementById('hosts').value,
                            upload_to_buzzsprout: document.getElementById('uploadBuzzsprout').checked,
                            upload_to_youtube: document.getElementById('uploadYoutube').checked,
                            buzzsprout_api_key: document.getElementById('buzzsproutApiKey').value || '',
                            buzzsprout_account_id: document.getElementById('buzzsproutAccountId').value || '',
                            youtube_secret_path: document.getElementById('youtubeSecretPath').value || ''
                        }})
                    }});

                    if (!response.ok) {{
                        throw new Error('Failed to start generation');
                    }}

                    const data = await response.json();
                    currentTaskId = data.task_id;

                    // Start status polling
                    statusInterval = setInterval(checkStatus, 2000);

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
                        document.getElementById('generateBtn').disabled = false;
                        document.getElementById('generateBtn').textContent = 'Initiate Production Sequence';
                        loadHistory();
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
                    const phase = data.phase || 'Starting...';
                    statusDetails.textContent = `Phase: ${{phase}} >> Progress: ${{pct}}%`;
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
                    
                    if (data.step_durations && data.step_durations.length > 0) {{
                        durationBox.style.display = 'block';
                        durationList.innerHTML = data.step_durations.map(s => `
                            <tr style="border-bottom: 1px dashed var(--border-color);">
                                <td style="padding: 4px 0;">${{s.phase}}</td>
                                <td style="text-align: right; color: var(--text-secondary);">${{s.duration_formatted}}</td>
                            </tr>
                        `).join('');
                    }} else {{
                         durationBox.style.display = 'none';
                    }}
                    
                    if (data.current_step_duration) {{
                        durationBox.style.display = 'block';
                        currentTimer.textContent = data.current_step_duration;
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
                        <a href="/api/download/${{data.task_id}}/podcast_final_audio.wav" class="download-link">🎵 Audio (WAV)</a>
                        <a href="/api/download/${{data.task_id}}/source_of_truth.md" class="download-link">📋 Source of Truth</a>
                        <a href="/api/download/${{data.task_id}}/show_notes.md" class="download-link">📝 Show Notes</a>
                        <a href="/api/download/${{data.task_id}}/accuracy_check.md" class="download-link">✅ Accuracy Check</a>
                        <a href="/api/download/${{data.task_id}}/supporting_paper.pdf" class="download-link">📄 Supporting Paper</a>
                        <a href="/api/download/${{data.task_id}}/adversarial_paper.pdf" class="download-link">📄 Adversarial Paper</a>
                        <a href="/api/download/${{data.task_id}}/source_of_truth.pdf" class="download-link">📄 Source of Truth PDF</a>
                        ${{uploadsHtml}}
                        </div>
                    `;
                }} else if (data.status === 'failed') {{
                    progressBar.style.width = '100%';
                    progressBar.style.background = '#ef4444';
                    statusIcon.textContent = '❌';
                    showError(data.error || 'Generation failed');
                }}
            }}

            function showError(message) {{
                const error = document.getElementById('error');
                error.textContent = 'SYSTEM ERROR: ' + message;
                error.style.display = 'block';
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
                             {{ name: "Audio (WAV)", file: "podcast_final_audio.wav", icon: "🎵" }},
                             {{ name: "Source of Truth", file: "source_of_truth.md", icon: "📋" }},
                             {{ name: "Show Notes", file: "show_notes.md", icon: "📝" }},
                             {{ name: "Accuracy Check", file: "accuracy_check.md", icon: "✅" }},
                             {{ name: "Script PDF", file: "supporting_paper.pdf", icon: "📄" }},
                             {{ name: "Adversarial Check", file: "adversarial_paper.pdf", icon: "⚖️" }}
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
        # Check for uncommitted changes
        proc = subprocess.run(["git", "status", "--porcelain"], cwd=SCRIPT_DIR, capture_output=True, text=True)
        if proc.stdout.strip():
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

    tasks_db[task_id] = task
    save_tasks()

    # Set upload credentials as env vars if provided by the form
    if request.buzzsprout_api_key:
        os.environ["BUZZSPROUT_API_KEY"] = request.buzzsprout_api_key
    if request.buzzsprout_account_id:
        os.environ["BUZZSPROUT_ACCOUNT_ID"] = request.buzzsprout_account_id
    if request.youtube_secret_path:
        os.environ["YOUTUBE_CLIENT_SECRET_PATH"] = request.youtube_secret_path

    # Add to queue instead of starting thread immediately
    task_queue.put(task)

    return {"task_id": task_id, "status": "queued"}

# Phase markers parsed from podcast_crew.py stdout
# Phase markers parsed from podcast_crew.py stdout
PHASE_MARKERS = [
    ("PHASE 0: RESEARCH FRAMING", "Research Framing", 5),
    ("PHASE 1: DEEP RESEARCH", "Deep Research Execution", 10),
    ("Lead report saved", "Lead Researcher Report", 20),
    ("Gate verdict:", "Research Gate Check", 25),
    ("PHASE 2b: GAP-FILL", "Gap-Fill Research", 30),
    ("PHASE 3: ADVERSARIAL", "Adversarial Research", 40),
    ("PHASE 4: FAIRNESS", "Fairness & Accuracy Audit", 50),
    ("PHASE 6a: PODCAST PLANNING", "Podcast Planning", 60),
    ("PHASE 6b: PODCAST RECORDING", "Podcast Recording", 70),
    ("PHASE 7: POST-PROCESSING", "Post-Processing", 85),
    ("Starting BGM Merging Phase", "BGM Merging", 90),
    ("SUCCESS: Audio duration", "Complete", 100),
]

def run_podcast_generation(task_id: str, topic: str, language: str,
                           accessibility_level: str = "simple",
                           podcast_length: str = "long", podcast_hosts: str = "random",
                           upload_buzzsprout: bool = False, upload_youtube: bool = False):
    """Run podcast_crew.py in background with real-time phase tracking."""
    try:
        tasks_db[task_id]["status"] = "running"
        tasks_db[task_id]["progress"] = 0
        tasks_db[task_id]["phase"] = "Starting"
        tasks_db[task_id]["phase_start_time"] = time.time()
        tasks_db[task_id]["step_durations"] = []
        save_tasks()

        env = os.environ.copy()
        env["ACCESSIBILITY_LEVEL"] = accessibility_level
        env["PODCAST_LENGTH"] = podcast_length
        env["PODCAST_HOSTS"] = podcast_hosts

        proc = subprocess.Popen(
            [str(PODCAST_ENV_PYTHON), "podcast_crew.py", "--topic", topic, "--language", language],
            cwd=SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        # Stream stdout and parse phase markers
        output_lines = []
        # No timeout limit
        start_time = tasks_db[task_id]["start_time"]
        output_dir_discovered = False

        for line in proc.stdout:
            output_lines.append(line)

            # 0. Discover output_dir early so artifact counting works during execution
            if not output_dir_discovered:
                found_dir = _find_latest_output_dir()
                if found_dir:
                    tasks_db[task_id]["output_dir"] = str(found_dir)
                    output_dir_discovered = True

            # 1. Parse Phase Markers
            for marker, phase_name, progress_pct in PHASE_MARKERS:
                if marker in line:
                    # Calculate duration of previous phase
                    current_time = time.time()
                    if "phase_start_time" in tasks_db[task_id]:
                         prev_start = tasks_db[task_id]["phase_start_time"]
                         duration = current_time - prev_start
                         prev_phase = tasks_db[task_id]["phase"]
                         
                         # Add to step durations if not already recorded (avoid duplicates)
                         if "step_durations" not in tasks_db[task_id]:
                             tasks_db[task_id]["step_durations"] = []
                         
                         # Check if we already have this phase (some markers might repeat or be close)
                         if not any(s["phase"] == prev_phase for s in tasks_db[task_id]["step_durations"]):
                            tasks_db[task_id]["step_durations"].append({
                                "phase": prev_phase,
                                "duration": duration,
                                "duration_formatted": f"{int(duration // 60)}m {int(duration % 60)}s"
                            })

                    # Update to new phase
                    tasks_db[task_id]["phase"] = phase_name
                    tasks_db[task_id]["progress"] = progress_pct
                    tasks_db[task_id]["phase_start_time"] = current_time
                    
                    # Calculate ETA
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



        proc.wait()

        if proc.returncode != 0:
            tasks_db[task_id]["status"] = "failed"
            error_text = "".join(output_lines[-100:])  # Last 100 lines
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
        audio_path = str(resolved_dir / "podcast_final_audio.wav")
        title = topic.strip()

        if upload_buzzsprout or upload_youtube:
            tasks_db[task_id]["status"] = "uploading"
            tasks_db[task_id]["phase"] = "Uploading"
            tasks_db[task_id]["progress"] = 95
            save_tasks()

            if upload_buzzsprout:
                tasks_db[task_id]["upload_results"]["buzzsprout"] = upload_to_buzzsprout(audio_path, title)
                save_tasks()

            if upload_youtube:
                tasks_db[task_id]["upload_results"]["youtube"] = upload_to_youtube(audio_path, title)
                save_tasks()

        tasks_db[task_id]["status"] = "completed"
        tasks_db[task_id]["progress"] = 100
        tasks_db[task_id]["phase"] = "Complete"
        tasks_db[task_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["error"] = str(e)
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

@app.get("/api/status/{task_id}")
async def get_status(task_id: str, username: str = Depends(verify_credentials)):
    """Get status of a specific task"""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks_db[task_id]
    
    # Calculate artifact counts on-the-fly
    if task.get("output_dir"):
        created, total = count_artifacts(task["output_dir"])
        task["artifacts_created"] = created
        task["artifacts_total"] = total
    else:
        task["artifacts_created"] = 0
        task["artifacts_total"] = len(EXPECTED_ARTIFACTS)

    # Calculate current step duration
    current_step_duration = 0
    if task["status"] == "running" and task.get("phase_start_time"):
        current_step_duration = time.time() - task["phase_start_time"]
    
    response = task.copy()
    response["current_step_duration"] = f"{int(current_step_duration // 60)}m {int(current_step_duration % 60)}s"
    
    return response

@app.get("/api/history")
async def get_history(username: str = Depends(verify_credentials)):
    """Get list of past production runs"""
    # Sort by created_at desc
    sorted_tasks = sorted(
        tasks_db.values(),
        key=lambda x: x["created_at"],
        reverse=True
    )
    
    # Calculate artifact counts for history items
    for task in sorted_tasks:
        if task.get("output_dir"):
            created, total = count_artifacts(task["output_dir"])
            task["artifacts_created"] = created
            task["artifacts_total"] = total
        else:
            task["artifacts_created"] = 0
            task["artifacts_total"] = len(EXPECTED_ARTIFACTS)
            
    # Return last 20 tasks, newest first
    return sorted_tasks[:20]

@app.get("/api/download/{task_id}/{filename}")
def download_file(task_id: str, filename: str, username: str = Depends(verify_credentials)):
    """Download generated file"""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks_db[task_id]
    output_dir = Path(task["output_dir"]) if task.get("output_dir") else OUTPUT_DIR
    file_path = output_dir / filename

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
    from upload_utils import get_youtube_credentials
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
