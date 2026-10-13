"""DICOM workflow utilities: audit, de-identification, windowing, AI result encoding.

De-identification follows a subset of the DICOM PS3.15 Annex E Basic Application Confidentiality
Profile: direct identifiers are removed or replaced, UIDs are re-mapped consistently, dates are
shifted, and a pseudonymous PatientID (study code) is kept for linkage with the text/tabular data.

Run:  python -m ad_mm.dicom.dicom_utils     (de-identifies data/raw/imaging/dicom -> data/interim/dicom_deid)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from ad_mm.config import METRIC_DIR, ROOT

REMOVE_TAGS = ["PatientName", "PatientBirthDate", "PatientAddress", "PatientTelephoneNumbers", "OtherPatientIDs",
               "OtherPatientNames", "ReferringPhysicianName", "InstitutionAddress", "PerformingPhysicianName",
               "OperatorsName", "AccessionNumber", "DeviceSerialNumber"]
UID_TAGS = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"]
SALT = "ad-mm-synthetic-salt"   # in production: secret from a key vault, never committed


def _uid_map(uid: str) -> str:
    return generate_uid(entropy_srcs=[SALT, uid])


def _date_shift(pid: str) -> int:
    return int(hashlib.sha256((SALT + pid).encode()).hexdigest(), 16) % 365 + 30


def deidentify(ds: pydicom.Dataset) -> tuple[pydicom.Dataset, dict]:
    log = {"removed": [], "uids_remapped": 0}
    pid = str(ds.PatientID)
    for t in REMOVE_TAGS:
        if t in ds:
            log["removed"].append(t)
            delattr(ds, t)
    ds.PatientName = "ANON"
    ds.remove_private_tags()
    for t in UID_TAGS:
        if t in ds:
            setattr(ds, t, _uid_map(str(getattr(ds, t))))
            log["uids_remapped"] += 1
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    if "StudyDate" in ds:
        d = dt.datetime.strptime(ds.StudyDate, "%Y%m%d") - dt.timedelta(days=_date_shift(pid))
        ds.StudyDate = d.strftime("%Y%m%d")
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "PS3.15 Basic Profile; UID remap; date shift"
    return ds, log


def window(arr: np.ndarray, center: float, width: float) -> np.ndarray:
    lo, hi = center - width / 2, center + width / 2
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def audit_header(ds: pydicom.Dataset) -> dict:
    return {k: str(ds.get(k, "")) for k in ["Modality", "Manufacturer", "MagneticFieldStrength", "SeriesDescription",
                                             "Rows", "Columns", "PixelSpacing", "BitsStored", "PhotometricInterpretation"]}


def create_ai_result_sc(src: pydicom.Dataset, overlay_rgb: np.ndarray, summary: str, out: Path) -> Path:
    """Encode the AI output (e.g. Grad-CAM overlay + predicted class) as a DICOM Secondary Capture
    in the same Study, so it can be pushed back to PACS and viewed next to the source series."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(out), {}, file_meta=meta, preamble=b"\0" * 128)
    for t in ["PatientID", "PatientName", "StudyInstanceUID", "StudyDate"]:
        if t in src:
            setattr(ds, t, getattr(src, t))
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.Modality = "OT"
    ds.SeriesDescription = "AI RESULT - RESEARCH USE ONLY"
    ds.ImageComments = summary[:10240]
    ds.Rows, ds.Columns = overlay_rgb.shape[:2]
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = overlay_rgb.astype(np.uint8).tobytes()
    ds.save_as(out, enforce_file_format=True)
    return out


def main() -> None:
    src = ROOT / "data" / "raw" / "imaging" / "dicom"
    dst = ROOT / "data" / "interim" / "dicom_deid"
    dst.mkdir(parents=True, exist_ok=True)
    logs = []
    for f in sorted(src.glob("*.dcm")):
        ds = pydicom.dcmread(f)
        ds, log = deidentify(ds)
        ds.save_as(dst / f.name, enforce_file_format=True)
        logs.append(log)
    chk = pydicom.dcmread(next(dst.glob("*.dcm")))
    assert "PatientBirthDate" not in chk and str(chk.PatientName) == "ANON"
    summary = {"files": len(logs), "tags_removed_per_file": len(logs[0]["removed"]),
               "uids_remapped_per_file": logs[0]["uids_remapped"], "example_header": audit_header(chk)}
    with open(METRIC_DIR / "dicom_deid_log.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
