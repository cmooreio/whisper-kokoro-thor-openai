import io
import os
import tempfile
from typing import Optional, List

import numpy as np
import soundfile as sf
import torch
import whisper
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from kokoro import KPipeline

# ---- GPU / model init -------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

whisper_model_name = os.getenv("WHISPER_MODEL", "small")
print(f"[init] Loading Whisper model '{whisper_model_name}' on {device}")
whisper_model = whisper.load_model(whisper_model_name, device=device)

lang_code = os.getenv("KOKORO_LANG_CODE", "a")  # 'a' = American English
print(f"[init] Initializing Kokoro KPipeline(lang_code='{lang_code}')")
kokoro_pipeline = KPipeline(lang_code=lang_code)

default_voice = os.getenv("KOKORO_VOICE", "af_heart")
kokoro_model_ids_env = os.getenv(
    "KOKORO_MODEL_IDS", "kokoro-tts,speaches-ai/Kokoro-82M-v1.0-ONNX"
)
KOKORO_MODEL_IDS = {m.strip() for m in kokoro_model_ids_env.split(",") if m.strip()}

app = FastAPI(title="thor-audio-openai", version="0.1.0")


# ---- Schemas ----------------------------------------------------------------

class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: Optional[str] = None
    format: Optional[str] = "wav"  # 'wav' for now; easy to extend later


# ---- Health check -----------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "device": device}


# ---- /v1/audio/transcriptions ----------------------------------------------
# OpenAI-style endpoint:
#   POST multipart/form-data with:
#     - file: audio file
#     - model: string (ignored, but accepted; e.g. "whisper-1")
#     - language, response_format, temperature, prompt (optional)
# Response:
#   { "text": "transcribed text" }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    prompt: Optional[str] = Form(None),
):
    # Read uploaded audio into a temp file for whisper
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    with tempfile.NamedTemporaryFile(delete=True, suffix=file.filename) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        # Whisper uses fp16 on CUDA, fp32 on CPU
        result = whisper_model.transcribe(
            tmp.name,
            language=language,
            fp16=(device == "cuda"),
        )

    text = (result.get("text") or "").strip()

    if response_format == "text":
        return StreamingResponse(io.BytesIO(text.encode("utf-8")),
                                 media_type="text/plain")

    # You can extend this to support verbose_json / srt / vtt etc.
    return JSONResponse({"text": text})


# ---- /v1/audio/speech -------------------------------------------------------
# OpenAI-style endpoint:
#   POST application/json:
#     { "model": "...", "input": "...", "voice": "af_heart", "format": "wav" }
# Response body: raw audio bytes (WAV) in the HTTP body.


@app.post("/v1/audio/speech")
async def audio_speech(req: SpeechRequest):
    # Speaches drop-in: accept speaches-style Kokoro model ids by default
    if req.model not in KOKORO_MODEL_IDS:
        # You can relax this if you don't care about model names
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{req.model}'. "
                   f"Allowed: {sorted(KOKORO_MODEL_IDS)}",
        )

    voice = req.voice or default_voice
    text = req.input
    if not text:
        raise HTTPException(status_code=400, detail="Empty input text")

    # Run Kokoro: generator yields (graphemes, phonemes, audio_chunk)
    # audio_chunk is a 1D numpy array of float32 samples at 24kHz
    audio_chunks: List[np.ndarray] = []
    try:
        generator = kokoro_pipeline(
            text,
            voice=voice,
            speed=1.0,
            split_pattern=r"\n+",
        )
        for _, _, audio in generator:
            audio_chunks.append(audio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kokoro error: {e}")

    if not audio_chunks:
        raise HTTPException(status_code=500, detail="Kokoro returned no audio")

    audio = np.concatenate(audio_chunks)

    # Right now we only support WAV output.
    # You can add MP3/OGG later via ffmpeg/pydub if needed.
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            # Some clients like a filename
            "Content-Disposition": 'attachment; filename="speech.wav"',
        },
    )
