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

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
OUTPUT_DIR = SCRIPT_DIR / "research_outputs"
TASKS_FILE = SCRIPT_DIR / "podcast_tasks.json"

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

            // Form submission
            document.getElementById('podcastForm').addEventListener('submit', async (e) => {{
                e.preventDefault();

                const topic = document.getElementById('topic').value;
                const language = document.getElementById('language').value;
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
                        body: JSON.stringify({{ topic, language }})
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
                    progressBar.style.width = '50%';
                    statusDetails.textContent = 'Agents are researching and debating...';
                }} else if (data.status === 'completed') {{
                    progressBar.style.width = '100%';
                    statusDetails.textContent = 'Podcast generated successfully!';

                    // Show download links
                    downloads.innerHTML = `
                        <h3>Download Results:</h3>
                        <a href="/api/download/${{data.task_id}}/podcast_final_audio.mp3" class="download-link">🎵 Audio (MP3)</a>
                        <a href="/api/download/${{data.task_id}}/supporting_paper.pdf" class="download-link">📄 Supporting Paper</a>
                        <a href="/api/download/${{data.task_id}}/adversarial_paper.pdf" class="download-link">📄 Adversarial Paper</a>
                        <a href="/api/download/${{data.task_id}}/final_audit_report.pdf" class="download-link">📄 Final Report</a>
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
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "error": None
    }

    tasks_db[task_id] = task
    save_tasks()

    # Start generation in background thread
    thread = threading.Thread(target=run_podcast_generation, args=(task_id, request.topic, request.language))
    thread.daemon = True
    thread.start()

    return {"task_id": task_id, "status": "pending"}

def run_podcast_generation(task_id: str, topic: str, language: str):
    """Run podcast_crew.py in background"""
    try:
        tasks_db[task_id]["status"] = "running"
        save_tasks()

        # Run podcast_crew.py with topic and language
        result = subprocess.run(
            [sys.executable, "podcast_crew.py", "--topic", topic, "--language", language],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )

        if result.returncode == 0:
            tasks_db[task_id]["status"] = "completed"
            tasks_db[task_id]["completed_at"] = datetime.now().isoformat()
        else:
            tasks_db[task_id]["status"] = "failed"
            tasks_db[task_id]["error"] = result.stderr or "Unknown error"

    except subprocess.TimeoutExpired:
        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["error"] = "Generation timed out after 30 minutes"
    except Exception as e:
        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["error"] = str(e)
    finally:
        save_tasks()

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

    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

if __name__ == "__main__":
    port = int(os.getenv("PODCAST_WEB_PORT", 8000))

    print(f"\nStarting DR_2_Podcast Web UI on http://0.0.0.0:{port}")
    print(f"Access from browser: http://localhost:{port}")
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=port)
