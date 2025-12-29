# Deep-Research Podcast Crew 🎙️

An AI-powered research team that investigates a scientific topic, conducts adversarial peer review, and generates a technical podcast script with text-to-speech audio.

## 🧠 System Architecture

The system utilizes **CrewAI** to orchestrate four specialized agents:
1. **Lead Researcher**: Identifies biochemical mechanisms.
2. **Scientific Auditor**: Validates data soundness.
3. **Adversarial Specialist**: Finds conflicting studies.
4. **Podcast Producer**: Scripts a dialogue between "Dr. Data" and "Dr. Doubt".



## 🚀 Setup Instructions

### 1. Prerequisites
- **Python 3.10+**
- **Local LLM Server**: This project is configured to connect to a "DGX Brain" (OpenAI-compatible API) running on `localhost:8000`.
- **Brave Search API Key**: Required for web-search capabilities.

### 2. Installation
```bash
pip install -r requirements.txt
