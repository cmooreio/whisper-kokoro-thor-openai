# Stage 1: Build CTranslate2 and ONNX Runtime with CUDA for ARM64 SBSA (Thor/Blackwell)
FROM nvcr.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Build dependencies for both CTranslate2 and ONNX Runtime
# Include ca-certificates for TLS downloads during ONNX Runtime FetchContent
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    ninja-build \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build CTranslate2 with CUDA support
# Using v4.5.0 for stability - matches faster-whisper requirements
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

# Build CTranslate2 Python bindings
RUN cd /tmp/ctranslate2/python \
    && pip install --break-system-packages pybind11 \
    && pip wheel --no-deps --wheel-dir /wheels . \
    && rm -rf /tmp/ctranslate2

# Build ONNX Runtime with CUDA support for ARM64 SBSA
# Using release branch for stability, with CUDA EP (Execution Provider)
# Thor Blackwell GPU: compute_90 (Hopper binary) + compute_100 (Blackwell PTX for JIT)
RUN pip install --break-system-packages packaging wheel setuptools numpy

# Pre-clone Eigen from git to avoid hash mismatch with GitLab's regenerated zip archives
# ONNX Runtime v1.20.1 requires this specific commit
RUN git clone https://gitlab.com/libeigen/eigen.git /tmp/eigen \
    && cd /tmp/eigen \
    && git checkout e7248b26a1ed53fa030c5c459f7ea095dfd276ac

RUN git clone --recursive --branch v1.20.1 https://github.com/microsoft/onnxruntime.git /tmp/onnxruntime \
    && cd /tmp/onnxruntime \
    && ./build.sh \
        --config Release \
        --build_shared_lib \
        --build_wheel \
        --use_cuda \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr \
        --parallel $(nproc) \
        --skip_tests \
        --allow_running_as_root \
        --cmake_extra_defines \
            CMAKE_CUDA_ARCHITECTURES="90;100" \
            FETCHCONTENT_SOURCE_DIR_EIGEN=/tmp/eigen \
            onnxruntime_BUILD_UNIT_TESTS=OFF \
    && cp /tmp/onnxruntime/build/Linux/Release/dist/*.whl /wheels/ \
    && rm -rf /tmp/onnxruntime /tmp/eigen

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

# Install our custom-built wheels first (CTranslate2 and ONNX Runtime with CUDA)
RUN pip install --no-cache-dir --break-system-packages /wheels/*.whl \
    && rm -rf /wheels

# Python deps:
# - fastapi + uvicorn: API server
# - python-multipart: file uploads
# - faster-whisper: uses our CUDA-enabled CTranslate2
# - kokoro-onnx: ONNX-based Kokoro TTS (--no-deps to use our CUDA ONNX Runtime)
# - soundfile: write WAV to BytesIO
RUN pip install --no-cache-dir --break-system-packages \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    faster-whisper \
    soundfile \
    huggingface_hub \
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
