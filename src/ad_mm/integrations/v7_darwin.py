"""V7 Darwin: medical-grade annotation (native DICOM/NIfTI viewer, consensus workflows, reviewer QA).

Use V7 for expert neuroradiologist labelling of the *source volumes* (e.g. hippocampal segmentation on
NIfTI/DICOM), then pull the annotations back and convert to YOLO boxes / masks for training.

    pip install darwin-py
    export V7_API_KEY=...
    python -m ad_mm.integrations.v7_darwin push   --team my-team --dataset ad-mri
    python -m ad_mm.integrations.v7_darwin pull   --team my-team --dataset ad-mri --release v1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ad_mm.config import ROOT


def _client():
    from darwin.client import Client

    return Client.from_api_key(os.environ["V7_API_KEY"])


def push(team: str, dataset: str) -> int:
    """Upload de-identified DICOM + NIfTI studies (V7 renders them natively as volumes)."""
    client = _client()
    ds = client.get_remote_dataset(f"{team}/{dataset}")
    files = [str(p) for p in (ROOT / "data" / "interim" / "dicom_deid").glob("*.dcm")]
    files += [str(p) for p in (ROOT / "data" / "raw" / "imaging" / "nifti").glob("*.nii")]
    ds.push(files, multi_threaded=True)
    return len(files)


def pull(team: str, dataset: str, release: str) -> Path:
    client = _client()
    ds = client.get_remote_dataset(f"{team}/{dataset}")
    rel = ds.get_release(release)
    ds.pull(release=rel)
    return Path(ds.local_path)


def darwin_json_to_yolo(darwin_dir: Path, out_dir: Path, names=("hippocampus", "ventricle")) -> int:
    """Convert Darwin JSON 2.0 bounding boxes to YOLO txt labels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for jf in darwin_dir.rglob("*.json"):
        d = json.loads(jf.read_text())
        slot = d["item"]["slots"][0]
        W, H = slot["width"], slot["height"]
        lines = []
        for ann in d.get("annotations", []):
            if "bounding_box" in ann and ann["name"] in names:
                b = ann["bounding_box"]
                lines.append(f"{names.index(ann['name'])} {(b['x'] + b['w'] / 2) / W:.5f} {(b['y'] + b['h'] / 2) / H:.5f} "
                             f"{b['w'] / W:.5f} {b['h'] / H:.5f}")
        (out_dir / f"{Path(d['item']['name']).stem}.txt").write_text("\n".join(lines))
        n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "pull"])
    ap.add_argument("--team", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--release", default="latest")
    a = ap.parse_args()
    print(push(a.team, a.dataset) if a.cmd == "push" else pull(a.team, a.dataset, a.release))
