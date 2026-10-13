"""FiftyOne dataset curation: visual QA of images, labels, ROI boxes and model predictions.

    pip install fiftyone
    python -m ad_mm.integrations.fiftyone_curation           # builds the dataset and launches the App

What it gives the team:
  * Browse every study with its diagnosis, format, scanner, QC metrics, ROI detections and prediction.
  * `compute_uniqueness` to find near-duplicate scans, `compute_mistakenness` to surface likely label errors,
    and `compute_hardness` to find ambiguous cases for adjudication.
  * Saved views (e.g. "jpg false negatives", "low SNR") that are exported to Roboflow / V7 for re-labelling.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ad_mm.config import CLASSES, ROOT
from ad_mm.integrations.yolo_roi import YOLO_DIR


def build_dataset(name: str = "ad-multimodal", predictions_csv: str | None = None):
    import fiftyone as fo
    import fiftyone.brain as fob

    ds_df = pd.read_csv(ROOT / "data" / "processed" / "multimodal_dataset.csv")
    qc = pd.read_csv(ROOT / "data" / "interim" / "imaging_qc.csv").set_index("patient_id")
    preds = pd.read_csv(predictions_csv).set_index("patient_id") if predictions_csv else None
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    dataset = fo.Dataset(name, persistent=True)
    samples = []
    for r in ds_df.itertuples():
        fp = YOLO_DIR / "images" / r.split / f"{r.patient_id}.png"
        s = fo.Sample(filepath=str(fp), tags=[r.split, r.format])
        s["ground_truth"] = fo.Classification(label=r.label)
        s["patient_id"], s["site"], s["scanner"] = r.patient_id, r.site, r.scanner
        s["snr"], s["noise_index"] = float(qc.loc[r.patient_id, "snr"]), float(qc.loc[r.patient_id, "noise_index"])
        dets = []
        for line in (YOLO_DIR / "labels" / r.split / f"{r.patient_id}.txt").read_text().splitlines():
            c, xc, yc, w, h = map(float, line.split())
            dets.append(fo.Detection(label=["hippocampus", "ventricle"][int(c)], bounding_box=[xc - w / 2, yc - h / 2, w, h]))
        s["roi"] = fo.Detections(detections=dets)
        if preds is not None and r.patient_id in preds.index:
            p = preds.loc[r.patient_id]
            probs = [float(p[f"p_{c}"]) for c in CLASSES]
            s["prediction"] = fo.Classification(label=CLASSES[int(np.argmax(probs))], confidence=max(probs),
                                                logits=np.log(np.array(probs) + 1e-9).tolist())
        samples.append(s)
    dataset.add_samples(samples)
    fob.compute_uniqueness(dataset)
    if preds is not None:
        fob.compute_mistakenness(dataset, "prediction", label_field="ground_truth")
        fob.compute_hardness(dataset, "prediction")
        F = fo.ViewField
        dataset.save_view("jpg_errors", dataset.match_tags("jpg").match(F("prediction.label") != F("ground_truth.label")))
        dataset.save_view("most_mistaken", dataset.sort_by("mistakenness", reverse=True)[:25])
    dataset.save_view("low_snr", dataset.sort_by("snr")[:25])
    return dataset


def export_for_relabeling(dataset, view_name: str, out_dir: str):
    import fiftyone as fo

    view = dataset.load_saved_view(view_name)
    view.export(export_dir=out_dir, dataset_type=fo.types.YOLOv5Dataset, label_field="roi")


if __name__ == "__main__":
    import fiftyone as fo

    pred_csv = ROOT / "reports" / "metrics" / "test_predictions.csv"
    d = build_dataset(predictions_csv=str(pred_csv) if pred_csv.exists() else None)
    print(d)
    print(json.dumps({"samples": len(d)}, indent=2))
    fo.launch_app(d).wait()
