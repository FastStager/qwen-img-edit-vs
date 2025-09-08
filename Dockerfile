FROM nvcr.io/nvidia/pytorch:24.05-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV HF_HOME=/app/cache
ENV HUGGING_FACE_HUB_CACHE=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV DIFFUSERS_CACHE=/app/cache
ENV HOME=/app
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir runpod

COPY qwenimage/ ./qwenimage/
COPY optimization.py ./optimization.py
COPY download.py ./download.py
COPY handler.py ./handler.py

CMD ["python", "-u", "handler.py"]
