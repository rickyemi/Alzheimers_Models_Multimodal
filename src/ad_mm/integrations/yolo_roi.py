"""YOLO region-of-interest detector (hippocampus, lateral ventricles) with Ultralytics.

Why: hippocampal atrophy and ventricular enlargement are the key structural AD markers. A detector lets us
(1) crop a high-resolution hippocampal ROI for a second image branch, (2) quality-check that the ROI is in
the field of view, and (3) derive ROI size biomarkers. It is also the model deployed on DeepStream/TensorRT.

    python -m ad_mm.integrations.yolo_roi export          # build YOLO dataset from processed images + ROI annotations
    python -m ad_mm.integrations.yolo_roi train           # requires `pip install ultralytics` (GPU recommended)
    python -m ad_mm.integrations.yolo_roi export-onnx     # YOLO -> ONNX for TensorRT / DeepStream
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from ad_mm.config import CFG, ROOT
from ad_mm.viz import save

YOLO_DIR = ROOT / CFG["data"]["yolo_dir"]
NAMES = {0: "hippocampus", 1: "ventricle"}


def to_uint8(z: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(z, [1, 99.5])
    return (np.clip((z - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


def export_dataset() -> dict:
    ds = pd.read_csv(ROOT / "data" / "processed" / "multimodal_dataset.csv")
    imgs = np.load(ROOT / "data" / "processed" / "images.npy")
    rois = {r["file"]: r for r in json.load(open(ROOT / "data" / "raw" / "imaging" / "roi_annotations.json"))}
    counts = {}
    for split in ("train", "val", "test"):
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    for r in ds.itertuples():
        name = r.patient_id
        Image.fromarray(to_uint8(imgs[r.row_idx, 0])).save(YOLO_DIR / "images" / r.split / f"{name}.png")
        lines = []
        for cls, key in ((0, "hippocampus"), (1, "ventricle")):
            for x0, y0, x1, y1 in rois[r.file][key]:
                x0, y0, x1, y1 = (float(np.clip(v, 0, 1)) for v in (x0, y0, x1, y1))
                lines.append(f"{cls} {(x0 + x1) / 2:.5f} {(y0 + y1) / 2:.5f} {x1 - x0:.5f} {y1 - y0:.5f}")
        (YOLO_DIR / "labels" / r.split / f"{name}.txt").write_text("\n".join(lines))
        counts[r.split] = counts.get(r.split, 0) + 1
    (YOLO_DIR / "data.yaml").write_text(
        f"path: {YOLO_DIR}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: hippocampus\n  1: ventricle\n")
    _preview(ds)
    return counts


def _preview(ds: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, cls in zip(axes, CFG["classes"]):
        r = ds[(ds.label == cls) & (ds.split == "train")].iloc[0]
        im = np.array(Image.open(YOLO_DIR / "images" / "train" / f"{r.patient_id}.png"))
        ax.imshow(im, cmap="gray")
        n = im.shape[0]
        for line in (YOLO_DIR / "labels" / "train" / f"{r.patient_id}.txt").read_text().splitlines():
            c, xc, yc, w, h = map(float, line.split())
            col = "#E8B04B" if c == 0 else "#3B6EA5"
            ax.add_patch(plt.Rectangle(((xc - w / 2) * n, (yc - h / 2) * n), w * n, h * n, fill=False, ec=col, lw=1.5))
        ax.set_title(cls)
        ax.axis("off")
    fig.suptitle("YOLO training labels: hippocampus (gold) and lateral ventricles (blue)", fontweight="bold")
    save(fig, "31_yolo_roi_labels")


def train(epochs: int = 60, imgsz: int = 96, model_cfg: str = "yolo11n.yaml", device: str = "cpu"):
    from ultralytics import YOLO

    model = YOLO(model_cfg)            # from scratch; or 'yolo11n.pt' to fine-tune COCO weights
    model.train(data=str(YOLO_DIR / "data.yaml"), epochs=epochs, imgsz=imgsz, batch=32, single_cls=False,
                project=str(ROOT / "models" / "yolo"), name="roi", fliplr=0.5, mosaic=0.0, seed=CFG["project"]["seed"],
                device=device, workers=0, plots=False, exist_ok=True, verbose=False)
    m = model.val(data=str(YOLO_DIR / "data.yaml"), split="test", imgsz=imgsz, device=device, plots=False, verbose=False)
    out = {"test_mAP50": float(m.box.map50), "test_mAP50_95": float(m.box.map),
           "per_class_mAP50_95": dict(zip(NAMES.values(), [float(x) for x in m.box.maps]))}
    (ROOT / "reports" / "metrics" / "yolo_roi_metrics.json").write_text(json.dumps(out, indent=2))
    return out


def export_onnx(weights: str = "models/yolo/roi/weights/best.pt"):
    from ultralytics import YOLO

    return YOLO(str(ROOT / weights)).export(format="onnx", opset=17, dynamic=True, simplify=False)


def crop_hippocampus(image: np.ndarray, boxes: list[tuple], pad: float = 0.1) -> list[np.ndarray]:
    """Crop detected hippocampal boxes (normalized xyxy) for a high-resolution ROI branch."""
    n = image.shape[-1]
    crops = []
    for x0, y0, x1, y1 in boxes:
        w, h = x1 - x0, y1 - y0
        xa, ya = int(max(0, (x0 - pad * w) * n)), int(max(0, (y0 - pad * h) * n))
        xb, yb = int(min(n, (x1 + pad * w) * n)), int(min(n, (y1 + pad * h) * n))
        crops.append(image[..., ya:yb, xa:xb])
    return crops


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "train", "export-onnx"])
    a = ap.parse_args()
    print({"export": export_dataset, "train": train, "export-onnx": export_onnx}[a.cmd]())
