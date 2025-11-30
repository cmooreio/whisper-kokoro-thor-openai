# Thor-friendly base: JetPack 7 / CUDA 13 / PyTorch 2.8
FROM nvcr.io/nvidia/pytorch:25.08-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System deps:
# - ffmpeg: required by openai-whisper
# - espeak-ng: used by kokoro for some fallback G2P cases
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Python deps:
# - fastapi + uvicorn: API server
# - python-multipart: file uploads
# - openai-whisper: STT (Torch-based, CUDA-aware)
# - kokoro: Kokoro-82M inference library
# - soundfile: write WAV to BytesIO
# - misaki[en]: G2P for English (as per kokoro README)
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    openai-whisper \
    "kokoro>=0.9.4" \
    soundfile \
    "misaki[en]"

WORKDIR /app
COPY server.py /app/server.py

# Defaults – you can override with env vars / compose
ENV WHISPER_MODEL=small \
    KOKORO_LANG_CODE=a \
    KOKORO_VOICE=af_heart \
    # Accept these model IDs for /v1/audio/speech (drop-in for Speaches)
    KOKORO_MODEL_IDS="kokoro-tts,speaches-ai/Kokoro-82M-v1.0-ONNX" \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
