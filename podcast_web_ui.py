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
import secrets

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

def load_tasks():
    """Load tasks from file"""
    global tasks_db
    if TASKS_FILE.exists():
        with open(TASKS_FILE, 'r') as f:
            tasks_db = json.load(f)

def save_tasks():
    """Save tasks to file"""
    with open(TASKS_FILE, 'w') as f:
        json.dump(tasks_db, f, indent=2)

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
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}

            .container {{
                max-width: 900px;
                margin: 0 auto;
            }}

            .header {{
                background: white;
                border-radius: 12px;
                padding: 30px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}

            h1 {{
                color: #667eea;
                font-size: 32px;
                margin-bottom: 8px;
            }}

            .subtitle {{
                color: #666;
                font-size: 14px;
            }}

            .card {{
                background: white;
                border-radius: 12px;
                padding: 30px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}

            h2 {{
                color: #333;
                font-size: 20px;
                margin-bottom: 20px;
            }}

            label {{
                display: block;
                font-weight: 600;
                color: #333;
                margin-bottom: 8px;
            }}

            input[type="text"], select {{
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 16px;
                margin-bottom: 20px;
            }}

            input[type="text"]:focus, select:focus {{
                outline: none;
                border-color: #667eea;
            }}

            button {{
                background: #667eea;
                color: white;
                border: none;
                padding: 14px 24px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                transition: background 0.3s;
            }}

            button:hover {{
                background: #5568d3;
            }}

            button:disabled {{
                background: #ccc;
                cursor: not-allowed;
            }}

            .status-box {{
                background: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                margin-top: 20px;
                display: none;
            }}

            .status-box.show {{
                display: block;
            }}

            .status-header {{
                font-weight: 600;
                margin-bottom: 10px;
            }}

            .status-pending {{
                color: #fbbf24;
            }}

            .status-running {{
                color: #3b82f6;
            }}

            .status-completed {{
                color: #10b981;
            }}

            .status-failed {{
                color: #ef4444;
            }}

            .progress {{
                width: 100%;
                height: 8px;
                background: #e0e0e0;
                border-radius: 4px;
                overflow: hidden;
                margin: 10px 0;
            }}

            .progress-bar {{
                height: 100%;
                background: #667eea;
                width: 0%;
                transition: width 0.3s;
                animation: pulse 2s infinite;
            }}

            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.7; }}
            }}

            .downloads {{
                margin-top: 20px;
            }}

            .download-link {{
                display: inline-block;
                background: #10b981;
                color: white;
                padding: 10px 20px;
                border-radius: 6px;
                text-decoration: none;
                margin-right: 10px;
                margin-top: 10px;
            }}

            .download-link:hover {{
                background: #059669;
            }}

            .history {{
                list-style: none;
            }}

            .history-item {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 10px;
                border-left: 4px solid #667eea;
            }}

            .history-topic {{
                font-weight: 600;
                margin-bottom: 5px;
            }}

            .history-meta {{
                font-size: 12px;
                color: #666;
            }}

            .error {{
                background: #fee2e2;
                color: #991b1b;
                padding: 15px;
                border-radius: 8px;
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎙️ DR_2_Podcast Generator</h1>
                <div class="subtitle">Generate AI-powered research debates on any scientific topic</div>
            </div>

            <div class="card">
                <h2>Generate New Podcast</h2>
                <form id="podcastForm">
                    <label for="topic">Scientific Topic</label>
                    <input
                        type="text"
                        id="topic"
                        name="topic"
                        placeholder="e.g., effects of meditation on brain plasticity"
                        required
                    />

                    <label for="language">Language</label>
                    <select id="language" name="language">
                        <option value="en">English</option>
                        <option value="ja">日本語 (Japanese)</option>
                    </select>

                    <label for="accessibility">Accessibility Level</label>
                    <select id="accessibility" name="accessibility_level">
                        <option value="simple">Simple (general audience)</option>
                        <option value="moderate">Moderate (some background)</option>
                        <option value="technical">Technical (expert audience)</option>
                    </select>

                    <div style="display: flex; gap: 20px;">
                        <div style="flex: 1;">
                            <label for="length">Duration</label>
                            <select id="length" name="podcast_length">
                                <option value="short">Short (10-15 min)</option>
                                <option value="medium">Medium (20-25 min)</option>
                                <option value="long" selected>Long (30+ min)</option>
                            </select>
                        </div>
                        <div style="flex: 1;">
                            <label for="hosts">Hosts</label>
                            <select id="hosts" name="podcast_hosts">
                                <option value="random" selected>Random Assignment</option>
                                <option value="kaz_erika">Kaz (Presenter) & Erika</option>
                                <option value="erika_kaz">Erika (Presenter) & Kaz</option>
                            </select>
                        </div>
                    </div>

                    <div style="margin-bottom: 20px;">
                        <label style="font-weight: 500; color: #555; font-size: 14px; margin-bottom: 10px;">Auto-upload (as draft)</label>
                        <div style="display: flex; gap: 24px; margin-bottom: 10px;">
                            <label style="font-weight: normal; display: flex; align-items: center; gap: 8px; cursor: pointer;" id="buzzsproutLabel">
                                <input type="checkbox" id="uploadBuzzsprout" style="width: 18px; height: 18px; cursor: pointer;" />
                                Buzzsprout (podcast)
                            </label>
                            <label style="font-weight: normal; display: flex; align-items: center; gap: 8px; cursor: pointer;" id="youtubeLabel">
                                <input type="checkbox" id="uploadYoutube" style="width: 18px; height: 18px; cursor: pointer;" />
                                YouTube (private)
                            </label>
                        </div>
                        <div id="buzzsproutFields" style="display: none; background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                            <label style="font-size: 13px; color: #555;">Buzzsprout API Key</label>
                            <input type="text" id="buzzsproutApiKey" placeholder="Your Buzzsprout API key" style="margin-bottom: 10px;" />
                            <label style="font-size: 13px; color: #555;">Buzzsprout Account ID</label>
                            <input type="text" id="buzzsproutAccountId" placeholder="Your Buzzsprout account ID" />
                        </div>
                        <div id="youtubeFields" style="display: none; background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                            <label style="font-size: 13px; color: #555;">YouTube Client Secret JSON Path</label>
                            <input type="text" id="youtubeSecretPath" placeholder="./client_secret.json" value="./client_secret.json" style="margin-bottom: 10px;" />
                            <button type="button" id="ytAuthBtn" style="background: #fff; color: #667eea; border: 1px solid #667eea; padding: 6px 16px; border-radius: 6px; font-size: 13px; cursor: pointer; width: auto;">Authorize YouTube</button>
                            <span id="ytAuthStatus" style="font-size: 12px; margin-left: 8px; color: #666;"></span>
                        </div>
                    </div>

                    <button type="submit" id="generateBtn">Generate Podcast</button>
                </form>

                <div id="statusBox" class="status-box">
                    <div class="status-header">
                        Status: <span id="statusText">Initializing...</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar" id="progressBar"></div>
                    </div>
                    <div id="statusDetails"></div>
                    <div id="downloads" class="downloads"></div>
                    <div id="error" class="error" style="display: none;"></div>
                </div>
            </div>

            <div class="card">
                <h2>Recent Podcasts</h2>
                <ul id="history" class="history">
                    <li style="color: #999;">No podcasts generated yet</li>
                </ul>
            </div>
        </div>

        <script>
            let currentTaskId = null;
            let statusInterval = null;

            // Load history on page load
            loadHistory();

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
                        this.style.background = '#10b981';
                        this.style.color = 'white';
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
                button.textContent = 'Generating...';
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
                    button.textContent = 'Generate Podcast';
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
                        document.getElementById('generateBtn').textContent = 'Generate Podcast';
                        loadHistory();
                    }}
                    // "uploading" keeps the interval alive — do nothing here
                }} catch (error) {{
                    console.error('Status check failed:', error);
                }}
            }}

            function updateStatus(data) {{
                const statusText = document.getElementById('statusText');
                const statusDetails = document.getElementById('statusDetails');
                const progressBar = document.getElementById('progressBar');
                const downloads = document.getElementById('downloads');

                statusText.textContent = data.status;
                statusText.className = `status-${{data.status}}`;

                if (data.status === 'running') {{
                    const pct = data.progress || 0;
                    progressBar.style.width = pct + '%';
                    const phase = data.phase || 'Starting...';
                    statusDetails.textContent = phase + ' (' + pct + '%)';
                }} else if (data.status === 'uploading') {{
                    progressBar.style.width = '75%';
                    statusDetails.textContent = 'Uploading to platforms...';
                }} else if (data.status === 'completed') {{
                    progressBar.style.width = '100%';
                    statusDetails.textContent = 'Podcast generated successfully!';

                    // Show download links
                    let uploadsHtml = '';
                    if (data.upload_results) {{
                        if (data.upload_results.buzzsprout) {{
                            const bs = data.upload_results.buzzsprout;
                            uploadsHtml += bs.success
                                ? `<a href="${{bs.url}}" target="_blank" class="download-link" style="background:#f59e0b;">🎙️ Buzzsprout Draft (ep #${{bs.episode_id}})</a>`
                                : `<span style="color:#ef4444; font-size:14px;">Buzzsprout upload failed: ${{bs.error}}</span>`;
                        }}
                        if (data.upload_results.youtube) {{
                            const yt = data.upload_results.youtube;
                            uploadsHtml += yt.success
                                ? `<a href="${{yt.url}}" target="_blank" class="download-link" style="background:#dc2626;">▶️ YouTube (private)</a>`
                                : `<span style="color:#ef4444; font-size:14px;">YouTube upload failed: ${{yt.error}}</span>`;
                        }}
                    }}

                    downloads.innerHTML = `
                        <h3>Download Results:</h3>
                        <a href="/api/download/${{data.task_id}}/podcast_final_audio.wav" class="download-link">🎵 Audio (WAV)</a>
                        <a href="/api/download/${{data.task_id}}/SOURCE_OF_TRUTH.md" class="download-link">📋 Source of Truth</a>
                        <a href="/api/download/${{data.task_id}}/SHOW_NOTES.md" class="download-link">📝 Show Notes</a>
                        <a href="/api/download/${{data.task_id}}/ACCURACY_CHECK.md" class="download-link">✅ Accuracy Check</a>
                        <a href="/api/download/${{data.task_id}}/supporting_paper.pdf" class="download-link">📄 Supporting Paper</a>
                        <a href="/api/download/${{data.task_id}}/adversarial_paper.pdf" class="download-link">📄 Adversarial Paper</a>
                        <a href="/api/download/${{data.task_id}}/source_of_truth.pdf" class="download-link">📄 Source of Truth PDF</a>
                        ${{uploadsHtml}}
                    `;
                }} else if (data.status === 'failed') {{
                    progressBar.style.width = '100%';
                    progressBar.style.background = '#ef4444';
                    showError(data.error || 'Generation failed');
                }}
            }}

            function showError(message) {{
                const error = document.getElementById('error');
                error.textContent = message;
                error.style.display = 'block';
            }}

            async function loadHistory() {{
                try {{
                    const response = await fetch('/api/history');
                    const tasks = await response.json();

                    const historyList = document.getElementById('history');

                    if (tasks.length === 0) {{
                        historyList.innerHTML = '<li style="color: #999;">No podcasts generated yet</li>';
                        return;
                    }}

                    historyList.innerHTML = tasks.map(task => `
                        <li class="history-item">
                            <div class="history-topic">${{task.topic}}</div>
                            <div class="history-meta">
                                Language: ${{task.language === 'en' ? 'English' : '日本語'}} |
                                Status: ${{task.status}} |
                                ${{new Date(task.created_at).toLocaleString()}}
                            </div>
                        </li>
                    `).join('');
                }} catch (error) {{
                    console.error('Failed to load history:', error);
                }}
            }}

            // Auto-refresh history every 10 seconds
            setInterval(loadHistory, 10000);
        </script>
    </body>
    </html>
    """
    return html

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
        "status": "pending",
        "progress": 0,
        "phase": "",
        "created_at": datetime.now().isoformat(),
        "error": None,
        "output_dir": None,
        "upload_buzzsprout": request.upload_to_buzzsprout,
        "upload_youtube": request.upload_to_youtube,
        "upload_results": {}
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

    # Start generation in background thread
    thread = threading.Thread(
        target=run_podcast_generation,
        args=(task_id, request.topic, request.language,
              request.accessibility_level, request.podcast_length, request.podcast_hosts,
              request.upload_to_buzzsprout, request.upload_to_youtube)
    )
    thread.daemon = True
    thread.start()

    return {"task_id": task_id, "status": "pending"}

# Phase markers parsed from podcast_crew.py stdout
PHASE_MARKERS = [
    ("PHASE 0: RESEARCH FRAMING", "Research Framing", 5),
    ("Fast model (Phi-4 Mini) detected", "Deep Research", 10),
    ("Research library saved", "Deep Research Complete", 25),
    ("CREW 1: PHASES 1-2", "Evidence Gathering", 30),
    ("Gate verdict:", "Gate Check", 45),
    ("PHASE 2b: GAP-FILL", "Gap-Fill Research", 50),
    ("CREW 2: PHASES 3-8", "Validation & Production", 55),
    ("TRANSLATION PHASE", "Translating to target language", 75),
    ("Generating Documentation PDFs", "Generating PDFs", 85),
    ("Generating Multi-Voice Podcast Audio", "Generating Audio", 90),
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
        tasks_db[task_id]["phase"] = "Starting..."
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
        deadline = time.time() + 3600  # 60 minute timeout
        for line in proc.stdout:
            output_lines.append(line)
            for marker, phase_name, progress_pct in PHASE_MARKERS:
                if marker in line:
                    tasks_db[task_id]["phase"] = phase_name
                    tasks_db[task_id]["progress"] = progress_pct
                    save_tasks()
                    break
            if time.time() > deadline:
                proc.kill()
                tasks_db[task_id]["status"] = "failed"
                tasks_db[task_id]["error"] = "Generation timed out after 60 minutes"
                save_tasks()
                return

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
def get_status(task_id: str, username: str = Depends(verify_credentials)):
    """Get task status"""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")

    return tasks_db[task_id]

@app.get("/api/history")
def get_history(username: str = Depends(verify_credentials)):
    """Get generation history"""
    # Return last 20 tasks, newest first
    sorted_tasks = sorted(
        tasks_db.values(),
        key=lambda x: x["created_at"],
        reverse=True
    )
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
