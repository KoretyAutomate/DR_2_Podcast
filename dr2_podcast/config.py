"""Centralized configuration for DR_2_Podcast pipeline."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Model Configuration ---
SMART_MODEL = os.environ.get("MODEL_NAME", "")
SMART_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
# Fast model (qwen3.5:9b via Ollama) removed 2026-08-10 — it measured SLOWER than the
# Smart model on this GB10 box (21 vs 27 tok/s) because Ollama runs on CPU while vLLM
# holds the GPU. Every LLM call now goes to the Smart endpoint above.

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
# AivisSpeech style IDs (Host 1 verified 2026-06-28; Host 2 verified live 2026-07-12).
# Host 1 (male)   = にせ ノーマル          (AivisHub model 6d11c6c2-f4a4-4435-887e-23dd60f8b8dd)
# Host 2 (female) = わかな ノーマル          (AivisHub model f83c385c-829b-40c4-8c11-639027e61636)
#   ↳ real-voice (リアボVC公式モデル), same creator family as the prior ほのか. Swapped
#     ほのか (808373280) → わかな (1138003200) on 2026-07-12 at creator/user request.
# Both models are ACML 1.0 licensed (commercial use permitted) and must be installed into
# AivisSpeech first. If reinstalled/another engine: re-verify with `curl http://localhost:10101/speakers`.
TTS_HOST1_ID = os.environ.get("TTS_HOST1_ID", "1937616896")
TTS_HOST2_ID = os.environ.get("TTS_HOST2_ID", "1138003200")

# When enabled, per-episode randomly swap which of the two configured host voices speaks
# Speaker 1 (the explainer) vs Speaker 2. Seeded deterministically off the script content:
# the same script always renders the same assignment, but different episodes vary — giving
# ~50/50 male/female explainer across the series instead of Speaker 1 always being Host 1.
# Only the two configured voices (TTS_HOST1_ID/TTS_HOST2_ID) are used. Disable with =0.
TTS_RANDOM_VOICE = os.environ.get("TTS_RANDOM_VOICE", "1") not in ("0", "false", "False", "")

# Speech rate multiplier for HTTP-based engines that support VOICEVOX-style
# audio_query's "speedScale" field (AivisSpeech and future engines). 1.0 = engine
# default. AivisSpeech's default cadence reads slower than VOICEVOX did; 1.1 was
# chosen after a direct A/B listen (2026-07-03).
TTS_SPEED_SCALE = float(os.environ.get("TTS_SPEED_SCALE", "1.1"))


# Per-voice speedScale overrides, keyed by integer voice/style ID. Any voice not
# listed falls back to TTS_SPEED_SCALE. わかな (1138003200) reads rushed at the
# global 1.1, so she renders at 1.0 (user request, 2026-07-14). Format: "id:speed,id:speed".
def _parse_speed_overrides(raw: str) -> dict:
    out = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        vid, sep, spd = pair.partition(":")
        if not sep:
            continue
        try:
            out[int(vid.strip())] = float(spd.strip())
        except ValueError:
            continue
    return out


TTS_SPEED_OVERRIDES = _parse_speed_overrides(os.environ.get("TTS_SPEED_OVERRIDES", "1138003200:1.0"))

# Per-voice intonationScale overrides — same "id:value" format and same fallback rule as
# TTS_SPEED_OVERRIDES. AivisSpeech's default is 1.0. わかな (1138003200) reads with too much
# pitch swing at the default: user feedback 2026-07-31 on ep02 was
# 「わかなの声のトーンの起伏が少し激し過ぎる」— "a LITTLE too much", so she renders at 0.85,
# a noticeable flattening rather than a monotone.
TTS_INTONATION_SCALE = float(os.environ.get("TTS_INTONATION_SCALE", "1.0"))
TTS_INTONATION_OVERRIDES = _parse_speed_overrides(os.environ.get("TTS_INTONATION_OVERRIDES", "1138003200:0.85"))

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
