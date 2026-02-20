#!/bin/bash
# Start DR_2_Podcast Web UI

cd "$(dirname "$0")"

# Set default port (8501 to avoid conflict with vLLM on 8000)
PORT=${1:-8501}
export PODCAST_WEB_PORT=$PORT

# Optional: Set custom credentials
# export PODCAST_WEB_USER=admin
# export PODCAST_WEB_PASSWORD=your_secure_password

echo "Starting DR_2_Podcast Web UI..."
echo "Port: $PORT"
echo ""

# Use podcast_flow conda env if available
CONDA_PYTHON="$HOME/miniconda3/envs/podcast_flow/bin/python3"
if [ -x "$CONDA_PYTHON" ]; then
    "$CONDA_PYTHON" web_ui.py
else
    python3 web_ui.py
fi
