"""FastAPI inference service (containerized with Docker, scaled on Kubernetes).

    uvicorn ad_mm.deployment.serve_api:app --host 0.0.0.0 --port 8080 --workers 2

Endpoints
    GET  /health          liveness/readiness probe
    GET  /model-info      model version, classes, providers
    GET  /metrics         Prometheus-format counters (requests, errors, latency, class distribution for drift)
    POST /predict         multipart: image file (.dcm/.nii/.tiff/.jpg, optional), note (str), numeric (JSON str)
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from ad_mm.deployment.onnx_inference import ADPredictor

logging.basicConfig(level=logging.INFO, format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":%(message)s}')
log = logging.getLogger("ad_mm_api")
_predictor: ADPredictor | None = None
STATS = {"requests": 0, "errors": 0, "latency_ms_sum": 0.0, "referrals": 0, "classes": Counter()}
ALLOWED = {".dcm", ".nii", ".gz", ".tif", ".tiff", ".jpg", ".jpeg"}


def predictor() -> ADPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ADPredictor(os.environ.get("AD_MM_MODEL_PATH"))
    return _predictor


@asynccontextmanager
async def lifespan(_app: FastAPI):
    predictor()          # load model once per worker at startup
    yield


app = FastAPI(title="Multimodal AD Diagnostic Model API", version="1.0.0", lifespan=lifespan,
              description="Research use only. 5-class (CN/SMC/EMCI/LMCI/AD) decision support.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.get("/model-info")
def model_info():
    p = predictor()
    return {"model_version": p.model_version, "classes": p.classes, "providers": p.providers,
            "inputs": [i.name for i in p.session.get_inputs()], "intended_use": "Research use only"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lines = [f"ad_mm_requests_total {STATS['requests']}", f"ad_mm_errors_total {STATS['errors']}",
             f"ad_mm_latency_ms_sum {STATS['latency_ms_sum']:.1f}", f"ad_mm_referrals_total {STATS['referrals']}"]
    lines += [f'ad_mm_predictions_total{{class="{c}"}} {n}' for c, n in STATS["classes"].items()]
    return "\n".join(lines) + "\n"


@app.post("/predict")
async def predict(image: UploadFile | None = File(None), note: str | None = Form(None), numeric: str | None = Form(None)):
    rid = str(uuid.uuid4())
    t0 = time.perf_counter()
    STATS["requests"] += 1
    tmp = None
    try:
        num = json.loads(numeric) if numeric else None
        if image is not None:
            suffix = "".join(Path(image.filename).suffixes[-2:]) if image.filename.endswith(".nii.gz") else Path(image.filename).suffix
            if suffix.lower() not in ALLOWED and Path(image.filename).suffix.lower() not in ALLOWED:
                raise HTTPException(415, f"Unsupported image type {image.filename}")
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(await image.read())
                tmp = f.name
        out = predictor().predict(tmp, note, num)
        out["request_id"] = rid
        STATS["classes"][out["prediction"]] += 1
        STATS["referrals"] += int(out["refer_to_clinician"])
        STATS["latency_ms_sum"] += (time.perf_counter() - t0) * 1000
        # audit log: no PHI, only request id, outputs and model version (21 CFR Part 11 style traceability)
        log.info(json.dumps({"request_id": rid, "prediction": out["prediction"], "confidence": out["confidence"],
                             "modalities": out["modalities_used"], "model_version": out["model_version"]}))
        return out
    except HTTPException:
        STATS["errors"] += 1
        raise
    except Exception as e:  # noqa: BLE001
        STATS["errors"] += 1
        raise HTTPException(400, str(e)) from e
    finally:
        if tmp:
            os.unlink(tmp)
