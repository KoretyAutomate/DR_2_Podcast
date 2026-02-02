# DR_2_Podcast Web UI - Setup Guide

## Overview

A user-friendly web interface for generating AI-powered research debate podcasts on any scientific topic.

## Prerequisites

**IMPORTANT:** This web UI requires the dynamic topic and language selection features to be merged first:
- PR #1: Dynamic topic feature (--topic argument)
- PR #2: Language selection feature (--language argument)

Make sure these PRs are merged before using the web UI, or merge this branch last.

## Required Dependencies

```bash
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

- **Local access**: http://localhost:8000
- **Network access**: http://YOUR_IP:8000 (from any device on your network)
- **Mobile access**: http://YOUR_IP:8000 (from your phone/tablet)

### 3. Login

Default credentials are randomly generated on each start and displayed in the terminal.

To set custom credentials:
```bash
export PODCAST_WEB_USER=your_username
export PODCAST_WEB_PASSWORD=your_secure_password
./start_podcast_web_ui.sh
```

Or add to your `.env` file:
```bash
PODCAST_WEB_USER=your_username
PODCAST_WEB_PASSWORD=your_secure_password
```

## Features

- ✅ **Topic Input** - Enter any scientific topic for debate
- ✅ **Language Selection** - Generate podcasts in English or Japanese
- ✅ **Real-time Status** - Monitor generation progress
- ✅ **Download Results** - Get audio MP3 and PDF reports
- ✅ **Generation History** - View past podcast generations
- ✅ **Mobile Responsive** - Works perfectly on phones and tablets
- ✅ **Secure Access** - Username/password authentication

## How to Use

1. **Open the web interface** in your browser
2. **Enter a scientific topic** (e.g., "effects of meditation on brain plasticity")
3. **Select language** (English or Japanese)
4. **Click "Generate Podcast"**
5. **Wait for generation** (typically 10-30 minutes depending on topic complexity)
6. **Download results:**
   - 🎵 Audio file (MP3)
   - 📄 Supporting Paper (PDF)
   - 📄 Adversarial Paper (PDF)
   - 📄 Final Audit Report (PDF)

## What Happens During Generation

The web UI runs the full podcast generation pipeline:

1. **Research Phase** - Lead scientist researches the topic
2. **Gap Analysis** - Auditor identifies weak points
3. **Counter-Research** - Adversarial researcher challenges claims
4. **Meta-Audit** - Final verdict weighing both sides
5. **Script Writing** - Podcast producer creates dialogue
6. **Polish** - Personality agent refines for verbal delivery
7. **PDF Generation** - Creates research papers
8. **Audio Synthesis** - Generates speech with gTTS

Total time: **10-30 minutes** depending on topic complexity and search requirements.

## File Structure

After generation, files are saved to:
```
research_outputs/
├── podcast_final_audio.mp3    # Final podcast audio
├── supporting_paper.pdf        # Pro-argument research paper
├── adversarial_paper.pdf       # Counter-argument paper
└── final_audit_report.pdf      # Meta-analysis verdict
```

## Task Tracking

Generation tasks are stored in `podcast_tasks.json` for persistence across server restarts.

View active tasks:
```bash
cat podcast_tasks.json | python3 -m json.tool
```

## Configuration

### Port
Default: 8000

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

To disable authentication (not recommended), modify `podcast_web_ui.py` and remove the `Depends(verify_credentials)` parameter from routes.

### Timeout
Default: 30 minutes per generation

To change, edit `podcast_web_ui.py` line with `timeout=1800`.

## Troubleshooting

### Can't access from mobile?
- Ensure mobile is on same network
- Check firewall: `sudo ufw allow 8000`
- Verify server is running: `ps aux | grep podcast_web_ui`

### Generation fails?
- Check if podcast_crew.py works standalone:
  ```bash
  python3 podcast_crew.py --topic "test topic" --language en
  ```
- Check that all dependencies are installed
- Review error in the web UI error message

### Timeout issues?
- For complex topics, increase timeout in podcast_web_ui.py
- Monitor progress with: `tail -f research_outputs/podcast_generation.log`

### Port already in use?
```bash
# Kill existing process
pkill -f podcast_web_ui.py

# Or use different port
./start_podcast_web_ui.sh 9000
```

## Security Notes

1. **Authentication Required** - All routes require basic auth
2. **Network Binding** - Binds to 0.0.0.0 for network access
3. **HTTPS Recommended** - For production, use nginx with SSL
4. **Password Protection** - Change default credentials before public deployment
5. **Firewall** - Only allow trusted networks

## API Endpoints

For programmatic access:

### Generate Podcast
```bash
curl -u username:password -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"topic": "coffee and productivity", "language": "en"}'
```

### Check Status
```bash
curl -u username:password http://localhost:8000/api/status/TASK_ID
```

### Download File
```bash
curl -u username:password http://localhost:8000/api/download/TASK_ID/podcast_final_audio.mp3 \
  -o podcast.mp3
```

### View History
```bash
curl -u username:password http://localhost:8000/api/history
```

## Mobile App Development

This web UI serves as the foundation for a future mobile app. The API is designed to be mobile-friendly:

- RESTful API design
- JSON responses
- File download support
- Status polling for progress
- Authentication ready

## Development

To modify the UI:
1. Edit `podcast_web_ui.py`
2. HTML/CSS/JavaScript are inline for easy customization
3. Restart server to see changes: `pkill -f podcast_web_ui.py && ./start_podcast_web_ui.sh`

## Performance

- **Concurrent generations**: Limited by system resources
- **File storage**: Each podcast ~5-20MB (audio + PDFs)
- **RAM usage**: ~2-4GB per generation (LLM + TTS)
- **CPU usage**: High during audio synthesis

## Future Enhancements

- [ ] Progress bar with detailed stages
- [ ] Email notification when complete
- [ ] Podcast library with search
- [ ] Batch generation queue
- [ ] Multi-user support with separate workspaces
- [ ] Mobile app (iOS/Android)
- [ ] Social sharing features
- [ ] Custom voice selection
- [ ] Character personality customization

## Support

For issues or questions:
1. Check `podcast_tasks.json` for error details
2. Review logs in `research_outputs/podcast_generation.log`
3. Test podcast_crew.py directly to isolate issues
4. Verify all dependencies are installed

---

**Server Status**: Ready for testing after PRs #1 and #2 are merged
**Port**: 8000 (default)
**Authentication**: Basic HTTP Auth
