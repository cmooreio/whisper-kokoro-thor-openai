import io
import os
import tempfile
from pathlib import Path
from typing import Optional

import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

# ---- Model init -------------------------------------------------------------

whisper_model_id = os.getenv("WHISPER_MODEL", "Systran/faster-distil-whisper-large-v3")

# Kokoro model paths - can be explicit paths or will look in /models/kokoro/
kokoro_onnx_path = os.getenv("KOKORO_ONNX_PATH", "/models/kokoro/kokoro-v1.0.onnx")
kokoro_voices_path = os.getenv("KOKORO_VOICES_PATH", "/models/kokoro/voices-v1.0.bin")

# faster-whisper: auto selects CUDA if available
print(f"[init] Loading faster-whisper model '{whisper_model_id}'")
whisper_model = WhisperModel(whisper_model_id, device="cuda", compute_type="float16")

# kokoro-onnx: requires model.onnx and voices.bin paths
print(f"[init] Loading Kokoro ONNX model from '{kokoro_onnx_path}'")
kokoro = Kokoro(kokoro_onnx_path, kokoro_voices_path)
print(f"[init] Kokoro loaded successfully")

default_voice = os.getenv("KOKORO_VOICE", "af_heart")
kokoro_model_ids_env = os.getenv(
    "KOKORO_MODEL_IDS", "kokoro-tts,speaches-ai/Kokoro-82M-v1.0-ONNX"
)
KOKORO_MODEL_IDS = {m.strip() for m in kokoro_model_ids_env.split(",") if m.strip()}

app = FastAPI(title="whisper-kokoro-thor-openai", version="0.2.0")


# ---- Schemas ----------------------------------------------------------------

class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: Optional[str] = None
    format: Optional[str] = "wav"


# ---- Health check -----------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "whisper": whisper_model_id, "kokoro": kokoro_onnx_path}


# ---- /v1/audio/transcriptions ----------------------------------------------

@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    prompt: Optional[str] = Form(None),
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Write to temp file for faster-whisper
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        # faster-whisper transcribe
        segments, info = whisper_model.transcribe(
            tmp.name,
            language=language,
            initial_prompt=prompt,
            temperature=temperature if temperature > 0 else 0.0,
            vad_filter=True,
        )
        # Collect all segment texts
        text = " ".join(segment.text.strip() for segment in segments)

    if response_format == "text":
        return StreamingResponse(
            io.BytesIO(text.encode("utf-8")), media_type="text/plain"
        )

    return JSONResponse({"text": text})


# ---- /v1/audio/speech -------------------------------------------------------

@app.post("/v1/audio/speech")
async def audio_speech(req: SpeechRequest):
    if req.model not in KOKORO_MODEL_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{req.model}'. Allowed: {sorted(KOKORO_MODEL_IDS)}",
        )

    voice = req.voice or default_voice
    text = req.input
    if not text:
        raise HTTPException(status_code=400, detail="Empty input text")

    try:
        # kokoro-onnx returns (samples, sample_rate)
        samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kokoro error: {e}")

    if samples is None or len(samples) == 0:
        raise HTTPException(status_code=500, detail="Kokoro returned no audio")

    # Write WAV to buffer
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )
