"""Unit + integration tests.  Run:  PYTHONPATH=src pytest -q"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from ad_mm.config import CFG, ROOT
from ad_mm.models.multimodal import MultimodalADNet
from ad_mm.preprocessing import imaging as im
from ad_mm.preprocessing.tabular import clean_tabular
from ad_mm.preprocessing.text import clean_note

RAW = ROOT / "data" / "raw" / "imaging"


@pytest.mark.parametrize("raw,exp", [("sub-0007_T1w.nii", "PT-0007"), ("pt_0007.tiff", "PT-0007"),
                                     ("PATIENT0007.jpg", "PT-0007"), ("PT-0007", "PT-0007"), ("0007", "PT-0007")])
def test_patient_id_normalization(raw, exp):
    assert im.normalize_patient_id(raw) == exp


@pytest.mark.parametrize("folder", ["dicom", "nifti", "tiff", "jpg"])
def test_every_format_loads_and_preprocesses(folder):
    f = sorted((RAW / folder).iterdir())[0]
    arr, meta = im.load_any(f)
    assert arr.ndim == 2
    out, info = im.preprocess(arr, 96)
    assert out.shape == (96, 96) and np.isfinite(out).all()
    assert 0.2 < info["mask_fraction"] < 0.9


def test_note_cleaning_expands_and_deidentifies():
    t = clean_note("72 yo pt c/o memory loss, hx HTN")
    assert "patient" in t and "complains of" in t and "hypertension" in t and "72" not in t
    assert clean_note("") == "no clinical note available"


def test_tabular_cleaning_fixes_units_and_typos():
    df = pd.DataFrame({"patient_id": ["PT-0001", "0001", "0002"], "site": "SITE-C", "age": [7.2, 7.2, 70],
                       "sex": ["Male", "Male", "f"], "education_yrs": 16, "apoe4_copies": 0, "mmse": [280, 280, 29],
                       "moca": 26, "cdr_sb": 0.5, "adas_cog13": 10, "faq": 0, "ecog_self": 1.5, "ecog_partner": 1.2,
                       "hippocampus_vol_mm3": [7.1, 7.1, 7000], "entorhinal_thick_mm": 3.5, "ventricles_vol_mm3": 30000,
                       "ptau217_pg_ml": 0.2, "ab42_40_ratio": 0.1, "nfl_pg_ml": 12, "gfap_pg_ml": 110, "dx": ["Normal", "Normal", "AD"]})
    out, log = clean_tabular(df, im.normalize_patient_id)
    assert len(out) == 2 and log["duplicates_removed"] == 1
    r = out.set_index("patient_id").loc["PT-0001"]
    assert r.age == 72 and r.mmse == 28 and r.hippocampus_vol_mm3 == 7100 and r.dx == "CN" and r.sex == 1


def test_final_dataset_contract():
    ds = pd.read_csv(ROOT / "data" / "processed" / "multimodal_dataset.csv")
    assert len(ds) == CFG["data"]["n_patients"] and ds.patient_id.is_unique
    assert np.load(ROOT / "data" / "processed" / "images.npy").shape == (500, 1, 96, 96)
    assert set(ds.split) == {"train", "val", "test"}


def test_model_forward_and_missing_modality():
    m = MultimodalADNet(500, 22, 5, CFG["model"]).eval()
    x = (torch.randn(4, 1, 96, 96), torch.randint(5, 500, (4, 96)), torch.ones(4, 96, dtype=torch.long), torch.randn(4, 22))
    logits, w = m(*x)
    assert logits.shape == (4, 5) and torch.allclose(w.sum(1), torch.ones(4))
    _, w2 = m(*x, modality_mask=torch.tensor([[1.0, 0, 1]] * 4))
    assert (w2[:, 1] < 1e-3).all()


@pytest.mark.skipif(not (ROOT / CFG["deployment"]["onnx_path"]).exists(), reason="ONNX model not exported")
def test_api_predict_end_to_end():
    from fastapi.testclient import TestClient

    from ad_mm.deployment.serve_api import app

    ds = pd.read_csv(ROOT / "data" / "processed" / "multimodal_dataset.csv")
    r = ds[(ds.split == "test") & (ds.format == "dcm")].iloc[0]
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        with open(RAW / r.file, "rb") as f:
            resp = c.post("/predict", files={"image": (Path(r.file).name, f, "application/dicom")},
                          data={"note": r.note_text, "numeric": json.dumps({"age": r.age, "mmse": r.mmse, "sex": "M"})})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["prediction"] in CFG["classes"] and abs(sum(body["probabilities"].values()) - 1) < 1e-3
        assert c.post("/predict", data={"note": "patient denies memory complaints"}).status_code == 200
        assert "ad_mm_requests_total 2" in c.get("/metrics").text


def test_k8s_manifests_parse():
    for f in (ROOT / "deployment" / "k8s").glob("*.yaml"):
        docs = [d for d in yaml.safe_load_all(f.read_text()) if d]
        assert all("kind" in d for d in docs)
