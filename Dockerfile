# Stage 1: Build CTranslate2 and ONNX Runtime with CUDA for ARM64 SBSA (Jetson AGX Thor / JP 7)
# CUDA 13.0.2 NGC base (newest cudnn ubuntu24.04 tag); host JetPack 7 uses driver-forward compat.
FROM nvcr.io/nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04@sha256:5e9d0c68200eb01201617eb0d29a26d9a472104a2b8240de40dab58101ec948f AS builder

ARG CTRANSLATE2_REF=383d063daf0bf338a22b491864cb2018eb8efd15
# v1.24.4: CUDA 13 nvcc flag checks (check_nvcc_compiler_flag for -Wstrict-aliasing)
ARG ONNXRUNTIME_REF=2d924974ef147392ced8409d36bd6d2e7fcc8a74
ARG BUILD_PARALLEL=10

ENV DEBIAN_FRONTEND=noninteractive

# Build dependencies for both CTranslate2 and ONNX Runtime
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

# CTranslate2 with CUDA for Thor (sm_110 / compute_110)
RUN git clone --filter=blob:none --recurse-submodules https://github.com/OpenNMT/CTranslate2.git /tmp/ctranslate2 \
    && cd /tmp/ctranslate2 \
    && git checkout --detach "${CTRANSLATE2_REF}" \
    && git submodule update --init --recursive \
    && sed -i 's/cuda_select_nvcc_arch_flags(ARCH_FLAGS \${CUDA_ARCH_LIST})/set(ARCH_FLAGS "-gencode=arch=compute_110,code=sm_110;-gencode=arch=compute_110,code=compute_110")/' CMakeLists.txt \
    && mkdir build && cd build \
    && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=ON \
        -DWITH_MKL=OFF \
        -DWITH_OPENBLAS=OFF \
        -DOPENMP_RUNTIME=NONE \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
    && make -j"${BUILD_PARALLEL}" \
    && make install

RUN cd /tmp/ctranslate2/python \
    && pip install --break-system-packages "pybind11==3.0.2" \
    && pip wheel --no-deps --wheel-dir /wheels . \
    && rm -rf /tmp/ctranslate2

RUN pip install --break-system-packages --ignore-installed \
    "packaging==26.0" \
    "wheel==0.46.3" \
    "setuptools==82.0.1" \
    "numpy==2.4.3"

# ONNX Runtime with CUDA EP for ARM64 SBSA (Thor sm_110)
RUN git clone --filter=blob:none --recursive https://github.com/microsoft/onnxruntime.git /tmp/onnxruntime \
    && cd /tmp/onnxruntime \
    && git checkout --detach "${ONNXRUNTIME_REF}" \
    && git submodule update --init --recursive \
    && ./build.sh \
        --config Release \
        --build_shared_lib \
        --build_wheel \
        --use_cuda \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr \
        --parallel "${BUILD_PARALLEL}" \
        --skip_tests \
        --allow_running_as_root \
        --cmake_extra_defines \
            CMAKE_CUDA_ARCHITECTURES="110" \
            CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
            onnxruntime_BUILD_UNIT_TESTS=OFF \
            onnxruntime_USE_CUDA=ON \
            onnxruntime_CUDA_HOME=/usr/local/cuda \
            onnxruntime_CUDNN_HOME=/usr \
    && cp /tmp/onnxruntime/build/Linux/Release/dist/*.whl /wheels/ \
    && rm -rf /tmp/onnxruntime

# Stage 2: Runtime image
FROM nvcr.io/nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04@sha256:e5a14fe36b99bb7ef417749837aa7a5150c9b5fbd07474835aa707c82fff85ba

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

COPY --from=builder /usr/local/lib/libctranslate2* /usr/local/lib/
COPY --from=builder /usr/local/include/ctranslate2 /usr/local/include/ctranslate2
COPY --from=builder /wheels /wheels

RUN ldconfig

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --no-cache-dir --break-system-packages /wheels/*.whl \
    && rm -rf /wheels

RUN pip install --no-cache-dir --break-system-packages \
    "fastapi==0.135.1" \
    "uvicorn[standard]==0.42.0" \
    "python-multipart==0.0.22" \
    "soundfile==0.13.1" \
    "huggingface_hub==1.7.2" \
    "tokenizers==0.22.2" \
    "av==17.0.0" \
    && pip install --no-cache-dir --break-system-packages --no-deps \
    "faster-whisper==1.2.1" \
    "kokoro-onnx==0.5.0" \
    && pip install --no-cache-dir --break-system-packages \
    "colorlog==6.10.1" \
    "espeakng-loader==0.2.4" \
    "phonemizer-fork==3.3.2"

WORKDIR /app
COPY server.py /app/server.py

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
