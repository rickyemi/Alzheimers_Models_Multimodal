"""Production predictor: raw inputs (any image format + free-text note + numeric dict) -> ONNX Runtime -> result.

Applies exactly the same preprocessing as training (shared code in ad_mm.preprocessing) using the
artifacts fitted on the training split (tokenizer, tabular transformer, orientation template).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from ad_mm.config import CFG, ROOT
from ad_mm.preprocessing import imaging as im
from ad_mm.preprocessing.tabular import NUMERIC, TabularTransformer
from ad_mm.preprocessing.text import clean_note

REFER_THRESHOLD = 0.60     # below this calibrated confidence -> flag for clinician review


class ADPredictor:
    def __init__(self, model_path: str | Path | None = None, providers: list[str] | None = None):
        from transformers import PreTrainedTokenizerFast

        model_path = Path(model_path or ROOT / CFG["deployment"]["onnx_path"])
        avail = ort.get_available_providers()
        providers = providers or [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider",
                                              "CoreMLExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        meta = self.session.get_modelmeta().custom_metadata_map
        self.classes = meta.get("classes", ",".join(CFG["classes"])).split(",")
        self.tok = PreTrainedTokenizerFast.from_pretrained(str(ROOT / "models" / "tokenizer"))
        tt = json.load(open(ROOT / "models" / "tabular_transformer.json"))
        self.tab = TabularTransformer()
        self.tab.columns, self.tab.medians = tt["columns"], pd.Series(tt["medians"])
        self.tab.mean, self.tab.std = pd.Series(tt["mean"]), pd.Series(tt["std"])
        self.template = np.load(ROOT / "data" / "interim" / "image_template.npy")
        self.size = CFG["preprocessing"]["image_size"]
        self.max_len = CFG["preprocessing"]["text_max_len"]
        self.providers = self.session.get_providers()
        self.model_version = model_path.stem

    def _image(self, path: str | Path | None) -> tuple[np.ndarray, float, dict]:
        if path is None:
            return np.zeros((1, 1, self.size, self.size), np.float32), 0.0, {}
        arr, meta = im.load_any(path)
        x, info = im.preprocess(arr, self.size, tuple(CFG["preprocessing"]["intensity_clip_percentiles"]), self.template)
        qc_ok = info["mask_fraction"] > 0.2 and info["snr"] > 1.5
        return x[None, None].astype(np.float32), 1.0, {"format": meta["format"], "qc_pass": bool(qc_ok), **{
            k: round(float(v), 3) for k, v in info.items() if isinstance(v, (int, float))}}

    def _text(self, note: str | None):
        enc = self.tok([clean_note(note)], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="np")
        return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64), 1.0 if note else 0.0

    def _tabular(self, values: dict | None):
        if not values:
            return np.zeros((1, len(self.tab.columns)), np.float32), 0.0
        row = {c: values.get(c, np.nan) for c in NUMERIC}
        if isinstance(row["sex"], str):
            row["sex"] = 1.0 if row["sex"].lower().startswith("m") else 0.0
        return self.tab.transform(pd.DataFrame([row])).values.astype(np.float32), 1.0

    def predict(self, image_path=None, note: str | None = None, numeric: dict | None = None) -> dict:
        t0 = time.perf_counter()
        img, m_img, img_qc = self._image(image_path)
        ids, mask, m_txt = self._text(note)
        tab, m_tab = self._tabular(numeric)
        mm = np.array([[m_img, m_txt, m_tab]], np.float32)
        if mm.sum() == 0:
            raise ValueError("At least one modality is required")
        probs, w = self.session.run(None, {"image": img, "input_ids": ids, "attention_mask": mask, "tabular": tab,
                                           "modality_mask": mm})
        p = probs[0]
        k = int(p.argmax())
        return {"prediction": self.classes[k], "confidence": round(float(p[k]), 4),
                "probabilities": {c: round(float(v), 4) for c, v in zip(self.classes, p)},
                "modality_weights": dict(zip(["image", "text", "tabular"], [round(float(v), 3) for v in w[0]])),
                "modalities_used": [m for m, v in zip(["image", "text", "tabular"], mm[0]) if v],
                "refer_to_clinician": bool(p[k] < REFER_THRESHOLD or (img_qc and not img_qc.get("qc_pass", True))),
                "image_qc": img_qc, "model_version": self.model_version, "execution_providers": self.providers,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "disclaimer": "Research use only. Not a medical device. Decision support for qualified clinicians."}


if __name__ == "__main__":
    ds = pd.read_csv(ROOT / "data" / "processed" / "multimodal_dataset.csv")
    r = ds[ds.split == "test"].iloc[0]
    pred = ADPredictor()
    out = pred.predict(ROOT / "data" / "raw" / "imaging" / r.file, r.note_text, r[NUMERIC].to_dict())
    print(json.dumps(out, indent=2), "\ntrue label:", r.label)
