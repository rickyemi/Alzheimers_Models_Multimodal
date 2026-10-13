# Multimodal AD inference service: CPU image (ONNX Runtime). For GPU see Dockerfile.gpu.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src AD_MM_MODEL_PATH=/app/models/ad_mm_fusion.onnx

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY deployment/docker/requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# code + inference artifacts only (no training data, no PHI)
COPY configs/ configs/
COPY src/ src/
COPY models/ad_mm_fusion.onnx models/tabular_transformer.json models/
COPY models/tokenizer/ models/tokenizer/
COPY data/interim/image_template.npy data/interim/

RUN useradd -u 10001 -m appuser && chown -R appuser /app
USER 10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8080/health')"
CMD ["uvicorn", "ad_mm.deployment.serve_api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
