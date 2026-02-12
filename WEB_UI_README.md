# DR_2_Podcast Web UI - Setup Guide

## Overview

A user-friendly web interface for generating AI-powered research debate podcasts on any scientific topic.

## Prerequisites

- Conda environment `podcast_flow` (Python 3.11) with all dependencies installed
- vLLM running on port 8000 (Qwen2.5-32B-Instruct-AWQ)
- Ollama running on port 11434 (phi4-mini)

## Required Dependencies

```bash
conda activate podcast_flow
pip install fastapi uvicorn python-multipart
```

All other dependencies come from the main podcast_crew.py requirements.

## Quick Start

### 1. Start the Web UI

```bash
cd /home/korety/Project/DR_2_Podcast
./start_podcast_web_ui.sh
```

Or with custom port:
```bash
./start_podcast_web_ui.sh 9000
```

### 2. Access from Browser

- **Local access**: http://localhost:8501
- **Network access**: http://YOUR_IP:8501 (from any device on your network)

### 3. Login

Default credentials are randomly generated on each start and displayed in the terminal.

To set custom credentials:
```bash
export PODCAST_WEB_USER=your_username
export PODCAST_WEB_PASSWORD=your_secure_password
./start_podcast_web_ui.sh
```

## Features

- **Topic Input** - Enter any scientific topic for debate
- **Language Selection** - Generate podcasts in English or Japanese
- **Accessibility Level** - Choose simple, moderate, or technical depth
- **Real-time Progress** - Phase-level tracking with progress percentage
- **Download Results** - Get audio WAV, markdown docs, and PDF reports
- **Auto-Upload** - Optional upload to Buzzsprout and YouTube
- **Generation History** - View past podcast generations
- **Mobile Responsive** - Works on phones and tablets
- **Secure Access** - Username/password authentication

## How to Use

1. **Open the web interface** in your browser
2. **Enter a scientific topic** (e.g., "effects of meditation on brain plasticity")
3. **Select language** (English or Japanese)
4. **Select accessibility level** (simple, moderate, or technical)
5. **Click "Generate Podcast"**
6. **Monitor progress** - the UI shows the current pipeline phase and percentage
7. **Download results:**
   - Audio file (WAV)
   - SOURCE_OF_TRUTH.md (key findings)
   - SHOW_NOTES.md (episode notes)
   - ACCURACY_CHECK.md (drift check)
   - Supporting Paper (PDF)
   - Adversarial Paper (PDF)
   - Source of Truth (PDF)

## Pipeline Phases

The web UI tracks these phases in real-time:

| Phase | Description | Progress |
|-------|-------------|----------|
| Research Framing | Topic scoping and query planning | 5% |
| Deep Research | Parallel web search + summarization | 10-25% |
| Evidence Gathering | Crew 1: supporting & adversarial research | 30% |
| Gate Check | Quality gate verification | 45% |
| Gap-Fill Research | Additional research if needed | 50% |
| Validation & Production | Crew 2: audit, script, polish | 55% |
| Generating PDFs | Documentation output | 85% |
| Generating Audio | Multi-voice Kokoro TTS | 90% |
| Complete | All outputs ready | 100% |

## Output Files

After generation, files are saved to timestamped directories:
```
research_outputs/
└── 2026-02-11_14-30-00/
    ├── podcast_final_audio.wav     # Final podcast audio
    ├── SOURCE_OF_TRUTH.md          # Key findings document
    ├── SHOW_NOTES.md               # Episode show notes
    ├── ACCURACY_CHECK.md           # Drift check report
    ├── supporting_paper.pdf        # Pro-argument research paper
    ├── adversarial_paper.pdf       # Counter-argument paper
    └── source_of_truth.pdf         # Source of truth PDF
```

## Configuration

### Port
Default: 8501 (avoids conflict with vLLM on port 8000)

Change with:
```bash
export PODCAST_WEB_PORT=9000
./start_podcast_web_ui.sh
```

### Authentication
Set custom credentials:
```bash
export PODCAST_WEB_USER=admin
export PODCAST_WEB_PASSWORD=your_secure_password
```

### Timeout
Default: 60 minutes per generation. Edit `podcast_web_ui.py` to change.

## Troubleshooting

### Generation fails?
- Check if podcast_crew.py works standalone:
  ```bash
  conda activate podcast_flow
  python podcast_crew.py --topic "test topic" --language en
  ```
- Verify vLLM is running on port 8000
- Verify Ollama is running on port 11434

### Port already in use?
```bash
pkill -f podcast_web_ui.py
./start_podcast_web_ui.sh 9000
```

## API Endpoints

### Generate Podcast
```bash
curl -u username:password -X POST http://localhost:8501/api/generate \
  -H "Content-Type: application/json" \
  -d '{"topic": "coffee and productivity", "language": "en", "accessibility_level": "simple"}'
```

### Check Status
```bash
curl -u username:password http://localhost:8501/api/status/TASK_ID
```

### Download File
```bash
curl -u username:password http://localhost:8501/api/download/TASK_ID/podcast_final_audio.wav \
  -o podcast.wav
```

### View History
```bash
curl -u username:password http://localhost:8501/api/history
```
