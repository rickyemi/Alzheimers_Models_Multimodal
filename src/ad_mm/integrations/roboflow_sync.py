"""Roboflow: hosted annotation, dataset versioning and augmentation for the YOLO ROI dataset.

    pip install roboflow
    export ROBOFLOW_API_KEY=...            # never commit keys; use a secret manager in CI
    python -m ad_mm.integrations.roboflow_sync upload --workspace my-ws --project ad-roi
    python -m ad_mm.integrations.roboflow_sync download --workspace my-ws --project ad-roi --version 3

Compliance note: only upload DE-IDENTIFIED derivatives (PNG exports produced by our pipeline after
dicom_utils.deidentify). Confirm a BAA / data processing agreement covers any real patient data.
"""
from __future__ import annotations

import argparse
import os

from ad_mm.integrations.yolo_roi import YOLO_DIR


def _project(workspace: str, project: str):
    from roboflow import Roboflow

    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    return rf.workspace(workspace).project(project)


def upload(workspace: str, project: str, splits=("train", "val", "test")) -> int:
    proj = _project(workspace, project)
    n = 0
    for split in splits:
        for img in sorted((YOLO_DIR / "images" / split).glob("*.png")):
            lbl = YOLO_DIR / "labels" / split / f"{img.stem}.txt"
            proj.single_upload(image_path=str(img), annotation_path=str(lbl), annotation_labelmap={0: "hippocampus", 1: "ventricle"},
                               split={"val": "valid"}.get(split, split), batch_name="ad-mm-synthetic", tag_names=["synthetic"])
            n += 1
    return n


def create_version(workspace: str, project: str) -> int:
    proj = _project(workspace, project)
    settings = {"preprocessing": {"auto-orient": True, "resize": {"width": 96, "height": 96, "format": "Stretch to"}},
                "augmentation": {"flip": {"horizontal": True}, "brightness": {"percent": 10}, "noise": {"percent": 2}}}
    return proj.generate_version(settings)


def download(workspace: str, project: str, version: int, fmt: str = "yolov11", location: str | None = None):
    proj = _project(workspace, project)
    return proj.version(version).download(fmt, location=location or str(YOLO_DIR.parent / f"roboflow_v{version}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["upload", "version", "download"])
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--version", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "upload":
        print(upload(a.workspace, a.project))
    elif a.cmd == "version":
        print(create_version(a.workspace, a.project))
    else:
        print(download(a.workspace, a.project, a.version))
