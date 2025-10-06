FROM nvcr.io/nvidia/pytorch:24.02-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV HF_HOME=/app/cache
ENV HUGGING_FACE_HUB_CACHE=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV DIFFUSERS_CACHE=/app/cache
ENV HOME=/app
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV TORCHINDUCTOR_CACHE_DIR=/app/torchin_cache

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    pkg-config \
    libdbus-1-dev \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip uninstall -y torch torchvision torchaudio && \
    pip install --pre --no-cache-dir \
    torch==2.8.0.dev20251005+cu128 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu121

COPY requirements.txt .

RUN pip cache purge && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir runpod

COPY qwenimage/ ./qwenimage/
COPY optimization.py ./optimization.py
COPY handler.py ./handler.py
COPY app.py ./app.py
COPY README.md ./README.md

CMD ["python", "-u", "handler.py"]