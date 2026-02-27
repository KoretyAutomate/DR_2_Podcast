"""
Qwen3-TTS FastAPI Server
POST /tts  {"text": "...", "speaker": "Host1"}  -> WAV bytes
GET  /health -> {"status": "ok"}
Speaker map: Host1 -> Aiden (male), Host2 -> Ono_Anna (Japanese female)
"""
import io, os, logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import soundfile as sf
import numpy as np
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Qwen3-TTS Server")

SPEAKER_VOICE_MAP = {
    "Host1": "Aiden", "host1": "Aiden",
    "Host2": "Ono_Anna", "host2": "Ono_Anna",
}
CHECKPOINTS_PATH = os.getenv("CHECKPOINTS_PATH", os.path.join(os.path.dirname(__file__), "checkpoints"))
_model = None


def _load_model():
    global _model
    from qwen_tts import Qwen3TTSModel
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Qwen3-TTS from {CHECKPOINTS_PATH} (device_map={device_map})...")
    _model = Qwen3TTSModel.from_pretrained(
        CHECKPOINTS_PATH,
        device_map=device_map,
        dtype=torch.bfloat16,
    )
    logger.info(f"✓ Model loaded on {device_map}")


@app.on_event("startup")
def startup_event():
    _load_model()


class TTSRequest(BaseModel):
    text: str
    speaker: str
    language: str = "Japanese"


@app.get("/health")
def health():
    return {"status": "ok", "model": "Qwen3-TTS-CustomVoice"}


@app.post("/tts")
def synthesize(req: TTSRequest) -> Response:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    voice = SPEAKER_VOICE_MAP.get(req.speaker)
    if voice is None:
        raise HTTPException(status_code=400, detail=f"Unknown speaker: {req.speaker!r}. Valid: {list(SPEAKER_VOICE_MAP.keys())}")
    logger.info(f"TTS: speaker={req.speaker} -> voice={voice}, text={req.text[:60]!r}")
    try:
        # generate_custom_voice returns (List[np.ndarray], sample_rate)
        audio_list, sample_rate = _model.generate_custom_voice(
            text=req.text, language=req.language, speaker=voice,
        )
        # Concatenate list of audio segments into single array
        if isinstance(audio_list, list):
            audio_array = np.concatenate([np.asarray(a, dtype=np.float32) for a in audio_list])
        else:
            audio_array = np.asarray(audio_list, dtype=np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        buf = io.BytesIO()
        sf.write(buf, audio_array, sample_rate, format="WAV")
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
