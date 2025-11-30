# Stage 1: Build CTranslate2 with CUDA for ARM64 SBSA
FROM nvcr.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Build CTranslate2 with CUDA support
# Using v4.5.0 for stability - matches faster-whisper requirements
# Note: --recurse-submodules needed for spdlog, -DOPENMP_RUNTIME=NONE to skip Intel MKL OpenMP
# Thor (Blackwell sm_100) - patch CMakeLists.txt to force architecture since FindCUDA is outdated
RUN git clone --recurse-submodules --branch v4.5.0 https://github.com/OpenNMT/CTranslate2.git /tmp/ctranslate2 \
    && cd /tmp/ctranslate2 \
    && sed -i 's/cuda_select_nvcc_arch_flags(ARCH_FLAGS \${CUDA_ARCH_LIST})/set(ARCH_FLAGS "-gencode=arch=compute_90,code=sm_90;-gencode=arch=compute_100,code=compute_100")/' CMakeLists.txt \
    && mkdir build && cd build \
    && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=ON \
        -DWITH_MKL=OFF \
        -DWITH_OPENBLAS=OFF \
        -DOPENMP_RUNTIME=NONE \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
    && make -j$(nproc) \
    && make install

# Build Python bindings
RUN cd /tmp/ctranslate2/python \
    && pip install --break-system-packages pybind11 \
    && pip wheel --no-deps --wheel-dir /wheels . \
    && rm -rf /tmp/ctranslate2

# Stage 2: Runtime image
FROM nvcr.io/nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Copy CTranslate2 libraries from builder
COPY --from=builder /usr/local/lib/libctranslate2* /usr/local/lib/
COPY --from=builder /usr/local/include/ctranslate2 /usr/local/include/ctranslate2
COPY --from=builder /wheels /wheels

# Update library cache
RUN ldconfig

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

# Install CTranslate2 wheel first (with CUDA support)
RUN pip install --no-cache-dir --break-system-packages /wheels/*.whl \
    && rm -rf /wheels

# Python deps:
# - fastapi + uvicorn: API server
# - python-multipart: file uploads
# - faster-whisper: uses our CUDA-enabled CTranslate2
# - kokoro-onnx: ONNX-based Kokoro TTS (installed without onnxruntime dep)
# - soundfile: write WAV to BytesIO
# - onnxruntime-gpu: CUDA-enabled ONNX Runtime from NVIDIA (built for ARM64 SBSA)
#
# Note: Install onnxruntime-gpu from NVIDIA's wheel index for ARM64 CUDA support
# This provides CUDAExecutionProvider for Kokoro TTS GPU acceleration
RUN pip install --no-cache-dir --break-system-packages \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    faster-whisper \
    soundfile \
    huggingface_hub \
    && pip install --no-cache-dir --break-system-packages \
    --extra-index-url https://pypi.nvidia.com \
    onnxruntime-gpu \
    && pip install --no-cache-dir --break-system-packages --no-deps \
    "kokoro-onnx>=0.4.0"

WORKDIR /app
COPY server.py /app/server.py

# Defaults - override with env vars
ENV WHISPER_MODEL="Systran/faster-distil-whisper-large-v3" \
    WHISPER_DEVICE="cuda" \
    WHISPER_COMPUTE_TYPE="float16" \
    KOKORO_MODEL="speaches-ai/Kokoro-82M-v1.0-ONNX" \
    KOKORO_VOICE="af_heart" \
    KOKORO_MODEL_IDS="kokoro-tts,tts-1,speaches-ai/Kokoro-82M-v1.0-ONNX" \
    HF_HUB_OFFLINE=0 \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
