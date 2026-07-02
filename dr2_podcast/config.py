"""Centralized configuration for DR_2_Podcast pipeline."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Model Configuration ---
SMART_MODEL = os.environ.get("MODEL_NAME", "")
SMART_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
FAST_MODEL = os.environ.get("FAST_MODEL_NAME", "")
FAST_BASE_URL = os.environ.get("FAST_LLM_BASE_URL", "http://localhost:11434/v1")

# --- Output Directory Override ---
OUTPUT_DIR_OVERRIDE = os.environ.get("OUTPUT_DIR")

# --- Service URLs ---
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8080")

# --- TTS Configuration ---
# Engine selectors per language. Known engines: "aivisspeech" (JA, HTTP/Docker), "kokoro" (EN, in-process).
# To add a new engine: define a _generate_audio_<name> adapter in audio/engine.py and register it.
TTS_ENGINE_JA = os.environ.get("TTS_ENGINE_JA", "aivisspeech")
TTS_ENGINE_EN = os.environ.get("TTS_ENGINE_EN", "kokoro")

# Generic HTTP endpoint — used by any HTTP-based TTS engine (AivisSpeech and future engines).
# AivisSpeech-Engine default port is 10101. Kokoro runs in-process and ignores this.
TTS_API_URL = os.environ.get("TTS_API_URL", "http://localhost:10101")

# Speaker identifiers — string form so different engines can interpret as needed
# (AivisSpeech/VOICEVOX: integer speaker/style ID; cloud TTS: voice name/UUID). Engine adapter casts.
# AivisSpeech style IDs (values below verified live via GET /speakers on 2026-06-28).
# Host 1 (male)   = にせ ノーマル          (AivisHub model 6d11c6c2-f4a4-4435-887e-23dd60f8b8dd)
# Host 2 (female) = みちのくあいり 標準     (AivisHub model 1b2830f4-8cf1-4184-a0d9-3a1bace3a844)
# Both models are ACML 1.0 licensed and must be installed into AivisSpeech first.
# If reinstalled/another engine: re-verify with `curl http://localhost:10101/speakers`.
TTS_HOST1_ID = os.environ.get("TTS_HOST1_ID", "1937616896")
TTS_HOST2_ID = os.environ.get("TTS_HOST2_ID", "1717361472")

# --- Timeouts (seconds) ---
LLM_TIMEOUT = 300
SCRAPING_TIMEOUT = 25.0
PUBMED_TIMEOUT = 15.0
VALIDATION_TIMEOUT = 10.0
UPLOAD_TIMEOUT = 120.0

# --- Audio Mixing ---
try:
    VOICE_DUCKING_DB = int(os.environ.get("VOICE_DUCKING_DB", "-20"))
except ValueError:
    VOICE_DUCKING_DB = -20

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
