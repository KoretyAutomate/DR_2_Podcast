#!/bin/bash
# Start DR_2_Podcast Web UI

cd "$(dirname "$0")"

# Set default port
PORT=${1:-8000}

# Optional: Set custom credentials
# export PODCAST_WEB_USER=admin
# export PODCAST_WEB_PASSWORD=your_secure_password

echo "Starting DR_2_Podcast Web UI..."
echo "Port: $PORT"
echo ""

python3 podcast_web_ui.py
