"""Centralized configuration for DR_2_Podcast pipeline."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Model Configuration ---
SMART_MODEL = os.environ.get("MODEL_NAME", "")
SMART_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
FAST_MODEL = os.environ.get("FAST_MODEL_NAME", "")
FAST_BASE_URL = os.environ.get("FAST_LLM_BASE_URL", "http://localhost:11434/v1")
MID_MODEL = os.environ.get("MID_MODEL_NAME", "qwen2.5:7b")
MID_BASE_URL = os.environ.get("MID_LLM_BASE_URL", os.environ.get("FAST_LLM_BASE_URL", "http://localhost:11434/v1"))

# --- Service URLs ---
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8080")
QWEN3_TTS_URL = os.environ.get("QWEN3_TTS_API_URL", "http://localhost:8082/tts")

# --- Timeouts (seconds) ---
LLM_TIMEOUT = 300
SCRAPING_TIMEOUT = 25.0
PUBMED_TIMEOUT = 15.0
VALIDATION_TIMEOUT = 10.0
UPLOAD_TIMEOUT = 120.0

# --- HTTP ---
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# --- Clinical Pipeline Thresholds ---
TIER_CASCADE_THRESHOLD = 50
MIN_TIER3_STUDIES = 3
MAX_TIER3_RATIO = 0.5
SCREENING_TOP_N = 20
MAX_AUDITOR_REVISIONS = 2

# --- Evidence Thresholds ---
EVIDENCE_LIMITED_THRESHOLD = 30
