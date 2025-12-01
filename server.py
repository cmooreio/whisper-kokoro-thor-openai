import io
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
import onnxruntime as ort
import ctranslate2

# ---- Diagnostics ------------------------------------------------------------

print(f"[init] CTranslate2 CUDA device count: {ctranslate2.get_cuda_device_count()}")
print(f"[init] ONNX Runtime available providers: {ort.get_available_providers()}")

# ---- Model init -------------------------------------------------------------

whisper_model_id = os.getenv("WHISPER_MODEL", "Systran/faster-distil-whisper-large-v3")

# Kokoro model paths - can be explicit paths or will look in /models/kokoro/
kokoro_onnx_path = os.getenv("KOKORO_ONNX_PATH", "/models/kokoro/kokoro-v1.0.onnx")
kokoro_voices_path = os.getenv("KOKORO_VOICES_PATH", "/models/kokoro/voices-v1.0.bin")

# faster-whisper with CUDA (CTranslate2 compiled with CUDA support)
whisper_device = os.getenv("WHISPER_DEVICE", "cuda")
whisper_compute = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
print(f"[init] Loading faster-whisper model '{whisper_model_id}' on device={whisper_device}")
whisper_model = WhisperModel(whisper_model_id, device=whisper_device, compute_type=whisper_compute)

# kokoro-onnx: requires model.onnx and voices.bin paths
print(f"[init] Loading Kokoro ONNX model from '{kokoro_onnx_path}'")
kokoro = Kokoro(kokoro_onnx_path, kokoro_voices_path)
print(f"[init] Kokoro loaded successfully")

default_voice = os.getenv("KOKORO_VOICE", "af_heart")
kokoro_model_ids_env = os.getenv(
    "KOKORO_MODEL_IDS", "kokoro-tts,speaches-ai/Kokoro-82M-v1.0-ONNX"
)
KOKORO_MODEL_IDS = {m.strip() for m in kokoro_model_ids_env.split(",") if m.strip()}

app = FastAPI(title="whisper-kokoro-thor-openai", version="0.3.0")


# ---- Schemas ----------------------------------------------------------------

class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "mp3"  # OpenAI API default is mp3
    speed: Optional[float] = 1.0


# ---- Audio format conversion ------------------------------------------------

# Supported formats and their MIME types
AUDIO_FORMATS = {
    "mp3": {"mime": "audio/mpeg", "ext": "mp3", "ffmpeg_fmt": "mp3"},
    "opus": {"mime": "audio/opus", "ext": "opus", "ffmpeg_fmt": "opus"},
    "aac": {"mime": "audio/aac", "ext": "aac", "ffmpeg_fmt": "adts"},
    "flac": {"mime": "audio/flac", "ext": "flac", "ffmpeg_fmt": "flac"},
    "wav": {"mime": "audio/wav", "ext": "wav", "ffmpeg_fmt": "wav"},
    "pcm": {"mime": "audio/pcm", "ext": "pcm", "ffmpeg_fmt": "s16le"},
}


def convert_audio(samples: np.ndarray, sample_rate: int, output_format: str) -> tuple[io.BytesIO, str, str]:
    """Convert audio samples to the requested format using ffmpeg.

    Returns: (audio_buffer, mime_type, file_extension)
    """
    if output_format not in AUDIO_FORMATS:
        output_format = "mp3"  # fallback to default

    fmt = AUDIO_FORMATS[output_format]

    # For WAV, we can use soundfile directly (faster)
    if output_format == "wav":
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV")
        buf.seek(0)
        return buf, fmt["mime"], fmt["ext"]

    # For PCM, output raw samples
    if output_format == "pcm":
        # Convert to 16-bit signed integers
        pcm_data = (samples * 32767).astype(np.int16).tobytes()
        buf = io.BytesIO(pcm_data)
        return buf, fmt["mime"], fmt["ext"]

    # For other formats, use ffmpeg to convert from WAV
    # First write WAV to a temp buffer
    wav_buf = io.BytesIO()
    sf.write(wav_buf, samples, sample_rate, format="WAV")
    wav_data = wav_buf.getvalue()

    # Run ffmpeg to convert
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "wav",
        "-i", "pipe:0",
        "-f", fmt["ffmpeg_fmt"],
    ]

    # Add format-specific options
    if output_format == "mp3":
        cmd.extend(["-codec:a", "libmp3lame", "-q:a", "2"])
    elif output_format == "opus":
        cmd.extend(["-codec:a", "libopus", "-b:a", "96k"])
    elif output_format == "aac":
        cmd.extend(["-codec:a", "aac", "-b:a", "128k"])
    elif output_format == "flac":
        cmd.extend(["-codec:a", "flac"])

    cmd.append("pipe:1")

    try:
        result = subprocess.run(
            cmd,
            input=wav_data,
            capture_output=True,
            check=True,
        )
        buf = io.BytesIO(result.stdout)
        return buf, fmt["mime"], fmt["ext"]
    except subprocess.CalledProcessError as e:
        # Log error and fallback to WAV
        print(f"[audio] ffmpeg conversion failed: {e.stderr.decode()}")
        wav_buf.seek(0)
        return wav_buf, "audio/wav", "wav"


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

    # Get speed, default to 1.0
    speed = req.speed if req.speed and req.speed > 0 else 1.0

    try:
        # kokoro-onnx returns (samples, sample_rate)
        samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kokoro error: {e}")

    if samples is None or len(samples) == 0:
        raise HTTPException(status_code=500, detail="Kokoro returned no audio")

    # Convert to requested format (default: mp3)
    output_format = req.response_format or "mp3"
    buf, mime_type, ext = convert_audio(samples, sample_rate, output_format)

    return StreamingResponse(
        buf,
        media_type=mime_type,
        headers={"Content-Disposition": f'attachment; filename="speech.{ext}"'},
    )
