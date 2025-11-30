# CUDA 13.0 runtime with cuDNN - no pre-compiled PyTorch to avoid numpy conflicts
FROM nvcr.io/nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System deps:
# - python3 + pip: Python runtime
# - ffmpeg: required for audio processing
# - espeak-ng: used by kokoro for some fallback G2P cases
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Python deps:
# - fastapi + uvicorn: API server
# - python-multipart: file uploads
# - faster-whisper: CTranslate2-based Whisper with CUDA support
# - kokoro-onnx: ONNX-based Kokoro TTS
# - soundfile: write WAV to BytesIO
RUN pip install --no-cache-dir --break-system-packages \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    faster-whisper \
    "kokoro-onnx>=0.4.0" \
    soundfile \
    huggingface_hub

WORKDIR /app
COPY server.py /app/server.py

# Defaults – override with env vars
# WHISPER_MODEL: HuggingFace model ID or local path
# KOKORO_MODEL: HuggingFace model ID or local path
ENV WHISPER_MODEL="Systran/faster-distil-whisper-large-v3" \
    KOKORO_MODEL="speaches-ai/Kokoro-82M-v1.0-ONNX" \
    KOKORO_VOICE="af_heart" \
    KOKORO_MODEL_IDS="kokoro-tts,speaches-ai/Kokoro-82M-v1.0-ONNX" \
    # Offline mode: set HF_HUB_OFFLINE=1 and mount cache to /root/.cache/huggingface/hub
    HF_HUB_OFFLINE=0 \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
