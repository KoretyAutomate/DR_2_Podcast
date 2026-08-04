"""
AivisSpeech two-voice smoke test for DR_2_Podcast.

Renders a short Japanese 2-speaker dialogue through the real engine path:
  Speaker 1 -> TTS_HOST1_ID (にせ, male)
  Speaker 2 -> TTS_HOST2_ID (みちのくあいり, female)

Config is forced here so the test is deterministic regardless of .env state.
(load_dotenv default override=False, so these win over .env.)
"""

import os
import sys

os.environ["TTS_ENGINE_JA"] = "aivisspeech"
os.environ["TTS_API_URL"] = "http://localhost:10101"
os.environ["TTS_HOST1_ID"] = "1937616896"  # にせ ノーマル (male)
os.environ["TTS_HOST2_ID"] = "1717361472"  # みちのくあいり 標準 (female)

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

from pathlib import Path
import soundfile as sf
from dr2_podcast.audio.engine import (
    generate_audio_from_script,
    clean_script_for_tts,
    _aivisspeech_available,
    _AIVISSPEECH_API_URL,
)

OUT = sys.argv[1] if len(sys.argv) > 1 else "aivisspeech_smoke.wav"

SCRIPT = """\
Host 1: こんにちは。今日は新しい音声エンジンのテストをします。
Host 2: わたしの声、ちゃんと聞こえていますか。自然に話せていると嬉しいです。
Host 1: ばっちりだよ。男性と女性、ふたつの声を確認しています。
Host 2: それでは、本番のエピソードでもよろしくお願いします。
"""

print("=" * 60)
print("AivisSpeech smoke test")
print(f"Endpoint: {_AIVISSPEECH_API_URL}")
print(f"Host1 (Speaker 1, にせ):       {os.environ['TTS_HOST1_ID']}")
print(f"Host2 (Speaker 2, みちのくあいり): {os.environ['TTS_HOST2_ID']}")
print("=" * 60)

if not _aivisspeech_available():
    print(f"FAIL: AivisSpeech engine not reachable at {_AIVISSPEECH_API_URL}")
    sys.exit(2)
print("Engine reachable: OK\n")

cleaned = clean_script_for_tts(SCRIPT)
result = generate_audio_from_script(cleaned, OUT, lang_code="j")

if not result:
    print("\nFAIL: generate_audio_from_script returned None")
    sys.exit(1)

path = result[0] if isinstance(result, tuple) else result
p = Path(path)
if not p.exists() or p.stat().st_size == 0:
    print(f"\nFAIL: output missing/empty: {path}")
    sys.exit(1)

info = sf.info(str(p))
print("\n" + "=" * 60)
print("PASS")
print(f"  File:        {p.resolve()}")
print(f"  Size:        {p.stat().st_size:,} bytes")
print(f"  Duration:    {info.duration:.2f} s")
print(f"  Sample rate: {info.samplerate} Hz")
print("=" * 60)
