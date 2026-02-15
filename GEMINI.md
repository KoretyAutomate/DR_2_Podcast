# Gemini CLI Project Configuration for Deep-Research Podcast Crew

This file provides context to the Gemini CLI about the project structure, commands, and important files.

## Project Overview

An AI-powered pipeline that deeply researches any scientific topic, conducts adversarial peer review across 10 structured phases, and produces a broadcast-ready podcast with Kokoro TTS audio — all running on local models. The system uses a dual-model architecture with a "smart" model (like Qwen2.5-32B-Instruct) on vLLM and a "fast" model (like phi4-mini) on Ollama. The pipeline is orchestrated by 7 specialized CrewAI agents.

## Prerequisites

### 1. Python Environment

This project uses Python 3.11 with a Conda environment.

```bash
# It is recommended to use the podcast_flow conda environment
conda activate podcast_flow
```

### 2. Required Services

The following services must be running before starting the application.

**Smart Model (vLLM):**
```bash
docker run --gpus all -p 8000:8000 
  --name vllm-final 
  vllm/vllm-openai:latest 
  python -m vllm.entrypoints.openai.api_server 
  --model Qwen/Qwen2.5-32B-Instruct-AWQ 
  --max-model-len 32768
```

**Fast Model (Ollama):**
```bash
ollama serve
ollama pull phi4-mini
```

**Search Backend (SearXNG - Optional):**
```bash
docker run -d -p 8080:8080 searxng/searxng:latest
```

## Installation

Once the prerequisites are met, install the required Python packages:

```bash
pip install -r requirements.txt
```

## Important Files

*   `podcast_crew.py`: Main pipeline orchestration.
*   `deep_research_agent.py`: Dual-model map-reduce deep research engine.
*   `search_agent.py`: SearXNG client and web scraper.
*   `research_planner.py`: Structured research planning.
*   `audio_engine.py`: Kokoro TTS rendering.
*   `podcast_web_ui.py`: Gradio Web UI frontend.
*   `requirements.txt`: Python dependencies.
*   `README.md`: Detailed project documentation.
*   `test_installation.py`: Installation tests.
*   `test_integration.py`: Integration tests.

## Commands

### Run the Main Podcast Generation Pipeline

This command runs the main pipeline to research a topic and generate a podcast.

```bash
python podcast_crew.py --topic "your topic here"
```

### Launch the Web UI

This command starts the Gradio web interface.

```bash
python podcast_web_ui.py
```

### Run Tests

This project uses pytest for testing. Run tests from the project root directory.

```bash
pytest
```

## Environment Variables

The application is configured via environment variables. Key variables are managed in the `.env` file.

**`.env` file:**
This file is used to store secrets and environment-specific configurations. Make sure your `.env` file is populated with the necessary keys, like `BRAVE_API_KEY`.

**Shell Exports:**
You can also set the following optional variables in your shell to override the defaults for a session:

```bash
# Optional: Specify the topic and language for the podcast
export PODCAST_TOPIC="effects of intermittent fasting on cognitive performance"
export PODCAST_LANGUAGE="en" # en or ja

# Optional: Control the simplification of scientific terminology
export ACCESSIBILITY_LEVEL="simple" # simple | moderate | technical
```
