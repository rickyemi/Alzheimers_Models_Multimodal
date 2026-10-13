"""Synthetic multimodal Alzheimer's disease dataset generator.

Generates three *source* datasets for 500 patients across 5 diagnostic classes
(CN, SMC, EMCI, LMCI, AD):

1. Vision   : one T1-weighted MRI study per patient, deliberately stored in a mix of
              formats (.dcm, .nii, .tiff, .jpg) with format-specific ID conventions,
              resolutions, bit depths, scanner intensity scaling and QC defects.
2. Text     : free-text clinical / neuropsychology notes (with abbreviations,
              duplicates, orphan notes and empty notes).
3. Numeric  : demographics, cognitive scores, MRI volumetrics and plasma biomarkers
              (with missingness, duplicate rows, unit errors, typos, messy coding).

A shared latent disease *severity* drives all three modalities, each with
independent noise and some modality-specific signal, so that fusion is useful:
    * SMC vs CN is mostly separable by subjective complaints (text, ECog-self).
    * EMCI vs LMCI is mostly separable by cognitive scores (numeric).
    * LMCI vs AD is separable by atrophy (imaging) and functional loss (text).

All data are fully artificial. No real patient information is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ad_mm.config import CFG, CLASSES, ROOT, seed_everything

SEVERITY = {"CN": 0.0, "SMC": 0.25, "EMCI": 1.2, "LMCI": 2.2, "AD": 3.3}
SITES = ["SITE-A", "SITE-B", "SITE-C", "SITE-D", "SITE-E"]
SCANNERS = {"SIEMENS": 1.00, "GE MEDICAL SYSTEMS": 0.82, "Philips": 1.18}
FIRST = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery", "Quinn", "Drew"]
LAST = ["Smith", "Lee", "Garcia", "Patel", "Nguyen", "Brown", "Okafor", "Kim", "Silva", "Cohen"]


# ----------------------------------------------------------------------------- latent cohort
def make_cohort(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    pid = 1
    for cls, n in CFG["data"]["class_counts"].items():
        for _ in range(n):
            sev = SEVERITY[cls] + rng.normal(0, 0.35)
            complaint = int(cls in ("SMC", "EMCI", "LMCI") or (cls == "AD" and rng.random() < 0.3)
                            or (cls == "CN" and rng.random() < 0.08))
            p_amy = 1 / (1 + np.exp(-(-1.6 + 1.25 * SEVERITY[cls] + (0.8 if cls == "SMC" else 0))))
            rows.append(dict(
                pid=pid, dx=cls, severity=sev, complaint=complaint,
                amyloid=int(rng.random() < p_amy),
                img_sev=max(sev + rng.normal(0, 0.55), -0.5),
                txt_sev=sev + rng.normal(0, 0.75),
                site=rng.choice(SITES),
                scanner=rng.choice(list(SCANNERS)),
                fmt=None,
            ))
            pid += 1
    df = pd.DataFrame(rows).sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(drop=True)
    df["pid"] = np.arange(1, len(df) + 1)
    fmts = np.array(CFG["data"]["image_formats"] * (len(df) // 4 + 1))[: len(df)]
    df["fmt"] = rng.permutation(fmts)
    return df


# ----------------------------------------------------------------------------- imaging
def brain_slice(n: int, atrophy: float, rng: np.random.Generator, z: float = 0.0,
                shift=(0.0, 0.0)) -> tuple[np.ndarray, dict]:
    """Render a synthetic axial T1w slice in [0,1]; returns image + ROI boxes (normalized)."""
    a = max(atrophy, 0.0)
    yy, xx = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]
    xx = xx - shift[0]
    yy = yy - shift[1]
    zf = np.sqrt(max(1 - z ** 2, 0.05))          # sphere-like shrink away from mid slice
    img = np.zeros((n, n), np.float32)
    skull = ((xx / (0.86 * zf)) ** 2 + (yy / (0.96 * zf)) ** 2) <= 1
    brain_r = (1 - 0.035 * a)
    brain = ((xx / (0.79 * zf * brain_r)) ** 2 + (yy / (0.89 * zf * brain_r)) ** 2) <= 1
    theta = np.arctan2(yy, xx)
    rr = np.sqrt(xx ** 2 + yy ** 2)
    gyri = 1 + 0.07 * np.sin(11 * theta + rng.uniform(0, np.pi)) * (1 + 0.15 * a)     # gyral folding of GM/WM border
    wm = ((xx / (0.6 * zf * brain_r * gyri)) ** 2 + (yy / (0.7 * zf * brain_r * gyri)) ** 2) <= 1
    img[skull] = 0.30
    img[(~brain) & skull & (((xx / (0.82 * zf)) ** 2 + (yy / (0.92 * zf)) ** 2) <= 1)] = 0.05  # CSF rim
    img[brain] = 0.58
    img[wm] = 0.86
    # sulcal widening: radial dark spokes whose width grows with atrophy
    spokes = (np.abs(np.sin(14 * theta + rng.uniform(0, np.pi))) < 0.02 + 0.022 * a) & brain & ~wm & (rr > 0.5 * zf)
    img[spokes] = 0.12
    # lateral ventricles (dark, CSF) enlarge with atrophy
    vr = (0.055 + 0.021 * a) * zf
    boxes = {}
    for sx in (-1, 1):
        cx = sx * (0.11 + 0.01 * a)
        v = (((xx - cx) / vr) ** 2 + ((yy + 0.05) / (vr * 2.0)) ** 2) <= 1
        img[v & brain] = 0.08
    falx = (np.abs(xx) < 0.008) & brain & ~wm & (yy < 0)                               # interhemispheric fissure
    img[falx] = 0.15
    vx0, vx1 = -(0.11 + 0.01 * a) - vr, (0.11 + 0.01 * a) + vr
    vy0, vy1 = -0.05 - vr * 2.0, -0.05 + vr * 2.0
    boxes["ventricle"] = [(vx0, vy0, vx1, vy1)]
    # hippocampi shrink with atrophy
    hr = max(0.085 * (1 - 0.11 * a), 0.035) * zf
    boxes["hippocampus"] = []
    for sx in (-1, 1):
        cx, cy = sx * 0.33, 0.32
        h = (((xx - cx) / (hr * 1.6)) ** 2 + ((yy - cy) / hr) ** 2) <= 1
        img[h] = 0.66
        ring = ((((xx - cx) / (hr * 1.6 + 0.03)) ** 2 + ((yy - cy) / (hr + 0.03)) ** 2) <= 1) & ~h
        img[ring & brain] = 0.10 + 0.02 * rng.random()          # temporal horn CSF
        boxes["hippocampus"].append((cx - hr * 1.6 - 0.03, cy - hr - 0.03, cx + hr * 1.6 + 0.03, cy + hr + 0.03))
    # bias field + noise
    bias = 1 + 0.12 * (xx * rng.normal() + yy * rng.normal())
    img = img * bias + rng.normal(0, 0.035, img.shape)
    img = np.clip(img, 0, 1.2)
    # convert box coords from [-1,1] grid (shifted) to normalized [0,1] image coords
    norm = {}
    for k, bl in boxes.items():
        norm[k] = [((x0 + shift[0] + 1) / 2, (y0 + shift[1] + 1) / 2, (x1 + shift[0] + 1) / 2, (y1 + shift[1] + 1) / 2)
                   for x0, y0, x1, y1 in bl]
    return img.astype(np.float32), norm


def write_dicom(path: Path, img: np.ndarray, row, rng) -> None:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    scale = SCANNERS[row.scanner]
    arr = np.clip(img * 2600 * scale + 150, 0, 4095).astype(np.uint16)
    ds.PatientName = f"{rng.choice(LAST)}^{rng.choice(FIRST)}"   # synthetic PHI for de-id demo
    ds.PatientID = f"PT-{row.pid:04d}"
    ds.PatientBirthDate = f"19{rng.integers(40, 60)}0{rng.integers(1, 9)}1{rng.integers(0, 9)}"
    ds.PatientSex = "O"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = MRImageStorage
    ds.Modality = "MR"
    ds.StudyDate = f"2025{rng.integers(1, 12):02d}{rng.integers(1, 28):02d}"
    ds.InstitutionName = row.site
    ds.Manufacturer = row.scanner
    ds.MagneticFieldStrength = float(rng.choice([1.5, 3.0]))
    ds.SeriesDescription = "AX T1 MPRAGE"
    ds.Rows, ds.Columns = arr.shape
    ds.PixelSpacing = [1.8, 1.8]
    ds.SliceThickness = 1.2
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.WindowCenter = 1400
    ds.WindowWidth = 2800
    ds.PixelData = arr.tobytes()
    ds.is_little_endian = True
    ds.save_as(path, enforce_file_format=True)


def write_nifti(path: Path, row, rng, shift) -> dict:
    import nibabel as nib

    nz, n = 24, 64
    vol = np.zeros((n, n, nz), np.float32)
    boxes = None
    for k in range(nz):
        z = (k - nz / 2 + 0.5) / (nz / 2)
        sl, bx = brain_slice(n, row.img_sev, rng, z=z, shift=shift)
        vol[:, :, k] = sl.T            # store in RAS-like (x, y, z) order
        if k == nz // 2:
            boxes = bx
    vol = (vol * 1000 * SCANNERS[row.scanner]).astype(np.int16)
    affine = np.diag([3.0, 3.0, 4.0, 1.0])
    img = nib.Nifti1Image(vol, affine)
    img.header["descrip"] = f"synthetic T1w {row.site}".encode()
    nib.save(img, str(path))
    return boxes


def generate_images(df: pd.DataFrame, rng, out: Path) -> pd.DataFrame:
    import tifffile
    from PIL import Image

    rois, meta = [], []
    defect_ids = set(rng.choice(df.pid, size=24, replace=False).tolist())
    defects = dict(zip(sorted(defect_ids), rng.choice(["low_contrast", "high_noise", "flip_lr", "flip_ud"], size=24)))
    for row in df.itertuples():
        shift = (rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04))
        defect = defects.get(row.pid, "none")
        if row.fmt == "nii":
            fname = f"sub-{row.pid:04d}_T1w.nii"
            boxes = write_nifti(out / "nifti" / fname, row, rng, shift)
            shape = (64, 64, 24)
        else:
            n = {"dcm": 128, "tiff": 160, "jpg": 192}[row.fmt]
            img, boxes = brain_slice(n, row.img_sev, rng, shift=shift)
            if defect == "low_contrast":
                img = 0.45 + 0.25 * img
            elif defect == "high_noise":
                img = np.clip(img + rng.normal(0, 0.12, img.shape), 0, 1.2)
            elif defect == "flip_lr":
                img = img[:, ::-1]
            elif defect == "flip_ud":
                img = img[::-1, :]
            shape = img.shape
            if row.fmt == "dcm":
                fname = f"PT-{row.pid:04d}.dcm"
                write_dicom(out / "dicom" / fname, img, row, rng)
            elif row.fmt == "tiff":
                fname = f"pt_{row.pid:04d}.tiff"
                tifffile.imwrite(out / "tiff" / fname, (np.clip(img, 0, 1.2) / 1.2 * 65535 * 0.9).astype(np.uint16))
            else:
                fname = f"PATIENT{row.pid:04d}.jpg"
                Image.fromarray((np.clip(img, 0, 1.2) / 1.2 * 255).astype(np.uint8)).save(out / "jpg" / fname, quality=82)
        folder = {"dcm": "dicom", "nii": "nifti", "tiff": "tiff", "jpg": "jpg"}[row.fmt]
        meta.append(dict(file=f"{folder}/{fname}", format=row.fmt, shape=str(shape), site=row.site,
                         scanner=row.scanner, injected_defect=defect if row.fmt != "nii" else "none"))
        rois.append(dict(file=f"{folder}/{fname}", **{k: v for k, v in boxes.items()}))
    with open(out / "roi_annotations.json", "w") as f:
        json.dump(rois, f)
    return pd.DataFrame(meta)


# ----------------------------------------------------------------------------- text
REASONS = {
    0: ["routine cognitive screening as part of a longitudinal aging study", "annual research follow-up visit",
        "baseline screening visit; volunteer with no cognitive concerns"],
    1: ["self-referral for forgetfulness", "concern about memory after a relative's dementia diagnosis",
        "pt c/o misplacing objects and word-finding pauses"],
    2: ["memory lapses noticed by patient and spouse over the past year", "referral from PCP for mild memory decline",
        "evaluation of repeated questions and missed appointments"],
    3: ["progressive memory decline over 18-24 months", "worsening short-term memory and difficulty managing finances",
        "family reports getting lost while driving in a familiar area"],
    4: ["family concern about confusion, disorientation and safety at home", "follow-up of established dementia, increasing care needs",
        "behavioural change, repetitive speech and inability to manage medications"],
}
MEMORY = {
    0: ["Denies memory complaints.", "No subjective cognitive decline reported.", "Cognition grossly intact on interview."],
    1: ["Reports subjective memory decline but performs within normal limits on testing.",
        "Endorses worry about memory; objective testing unremarkable.", "Subjective complaint; no objective deficit on screening."],
    2: ["Mild delayed recall deficit on word list; recognition preserved.", "Subtle episodic memory weakness, otherwise intact.",
        "Delayed recall mildly below age norms."],
    3: ["Impaired delayed recall with poor benefit from cues.", "Clear amnestic deficit on story recall and word list.",
        "Episodic memory impaired; executive function borderline."],
    4: ["Severe amnestic deficit with disorientation to date.", "Marked impairment across memory, language and visuospatial domains.",
        "Unable to recall any words after delay; disoriented to time and place."],
}
FUNCTION = {
    0: ["Fully independent in IADLs and ADLs.", "Independent; works part-time and drives."],
    1: ["Independent in all daily activities.", "No functional change reported by informant."],
    2: ["Largely independent; uses reminders and calendars more often.", "Minimal IADL change."],
    3: ["Needs help with finances and complex medication schedules.", "Some difficulty with IADLs; spouse now manages bills."],
    4: ["Requires assistance with dressing and bathing.", "Dependent for IADLs and needs supervision for ADLs."],
}
IMPRESSION = {
    0: ["Impression: cognitively normal.", "No evidence of neurocognitive disorder."],
    1: ["Impression: subjective cognitive decline, monitor.", "SCD without objective impairment."],
    2: ["Impression: possible early MCI.", "Findings consistent with mild cognitive change."],
    3: ["Impression: amnestic MCI, likely due to AD.", "Single-domain amnestic MCI, progressive."],
    4: ["Impression: dementia, probable AD type.", "Major neurocognitive disorder, moderate stage."],
}
FILLER = ["Hx HTN, well controlled on lisinopril.", "Sleep 6-7 h per night.", "Mood euthymic; denies depression.",
          "Hearing aids bilaterally.", "Former smoker, quit 20 yrs ago.", "Walks 30 min daily.",
          "Type 2 DM on metformin.", "No focal neuro deficits on exam.", "Gait normal.", "Vision corrected with glasses.",
          "Retired teacher.", "Enjoys gardening and crosswords."]
LIVING = ["Lives with spouse.", "Lives alone, daughter nearby.", "Lives with adult son."]


def bucket(x: float) -> int:
    return int(np.clip(np.digitize(x, [0.6, 1.7, 2.7, 3.5]), 0, 4))


def make_note(row, age, sex, rng) -> str:
    b = bucket(row.txt_sev)
    reason_b = 1 if (row.complaint and b == 0) else b
    mem_b = 1 if (row.complaint and b == 0) else (0 if (not row.complaint and b == 1) else b)
    func_b = bucket(row.severity + rng.normal(0, 0.5))
    parts = [f"{age}-year-old {'male' if sex == 'M' else 'female'} seen for {rng.choice(REASONS[reason_b])}.",
             rng.choice(MEMORY[mem_b]), rng.choice(FUNCTION[func_b])]
    parts += list(rng.choice(FILLER, size=rng.integers(1, 4), replace=False)) + [rng.choice(LIVING)]
    if rng.random() < 0.75:
        parts.append(rng.choice(IMPRESSION[bucket(row.txt_sev + rng.normal(0, 0.4))]))
    if rng.random() < 0.5:
        parts.append("Plan: " + rng.choice(["repeat testing in 12 months.", "plasma biomarker panel ordered.",
                                              "MRI brain and neuropsych referral.", "discuss amyloid PET eligibility."]))
    text = " ".join(parts)
    if rng.random() < 0.3:
        text = text.replace("patient", "pt").replace("history", "hx").replace("complains", "c/o")
    return text


def generate_notes(df: pd.DataFrame, demo: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for row in df.itertuples():
        d = demo.loc[row.pid]
        rows.append(dict(patient_id=f"PT{row.pid:04d}" if rng.random() < 0.5 else f"pt-{row.pid:04d}",
                         visit_date=f"2025-{rng.integers(1, 12):02d}-{rng.integers(1, 28):02d}",
                         author_role=rng.choice(["neurologist", "neuropsychologist", "geriatrician", "NP"]),
                         note_text=make_note(row, int(d.age), d.sex, rng)))
    notes = pd.DataFrame(rows)
    empty = rng.choice(len(notes), 3, replace=False)
    notes.loc[empty, "note_text"] = ""
    dups = notes.sample(6, random_state=1).copy()
    dups["visit_date"] = dups["visit_date"]  # exact re-entries
    orphans = pd.DataFrame([dict(patient_id=f"PT{9000 + i:04d}", visit_date="2025-03-01", author_role="NP",
                                 note_text="Screen fail, consent withdrawn before imaging.") for i in range(8)])
    return pd.concat([notes, dups, orphans]).sample(frac=1, random_state=3).reset_index(drop=True)


# ----------------------------------------------------------------------------- numeric
def generate_tabular(df: pd.DataFrame, rng) -> pd.DataFrame:
    n = len(df)
    s = df.severity.values
    si = df.img_sev.values
    amy = df.amyloid.values
    cpl = df.complaint.values
    age = np.clip(68 + 2.3 * s + rng.normal(0, 6.5, n), 55, 92).round()
    sex = rng.choice(["M", "F"], n)
    out = pd.DataFrame({
        "patient_id": [f"{p:04d}" for p in df.pid],
        "site": df.site.values,
        "age": age,
        "sex": sex,
        "education_yrs": np.clip(16 - 0.35 * s + rng.normal(0, 2.6, n), 6, 22).round(),
        "apoe4_copies": [rng.choice([0, 1, 2], p=[0.72 - 0.12 * a - 0.05 * min(x, 3), 0.23 + 0.08 * a + 0.03 * min(x, 3),
                                                   0.05 + 0.04 * a + 0.02 * min(x, 3)]) for a, x in zip(amy, np.clip(s, 0, 3.5))],
        "mmse": np.clip(29.4 - 1.55 * s + rng.normal(0, 1.2, n), 8, 30).round(),
        "moca": np.clip(27.6 - 2.1 * s + rng.normal(0, 1.7, n), 5, 30).round(),
        "cdr_sb": np.clip(0.05 + 0.85 * np.clip(s, 0, None) ** 1.35 + rng.normal(0, 0.55, n), 0, 18).round(1),
        "adas_cog13": np.clip(8 + 6.2 * s + rng.normal(0, 4, n), 0, 85).round(1),
        "faq": np.clip(0.3 + 2.1 * np.clip(s - 1.2, 0, None) ** 1.6 + rng.normal(0, 1.4, n), 0, 30).round(),
        "ecog_self": np.clip(1.25 + 0.55 * cpl + 0.12 * s - 0.25 * (s > 3) + rng.normal(0, 0.22, n), 1, 4).round(2),
        "ecog_partner": np.clip(1.15 + 0.38 * np.clip(s, 0, None) + rng.normal(0, 0.25, n), 1, 4).round(2),
        "hippocampus_vol_mm3": np.clip(7650 - 620 * si + rng.normal(0, 430, n), 3500, 9800).round(),
        "entorhinal_thick_mm": np.clip(3.65 - 0.24 * si + rng.normal(0, 0.22, n), 1.8, 4.5).round(2),
        "ventricles_vol_mm3": np.clip(27000 + 9500 * si + rng.normal(0, 8000, n), 8000, 120000).round(),
        "ptau217_pg_ml": np.round(np.exp(np.log(0.18) + 0.95 * amy + 0.12 * s + rng.normal(0, 0.28, n)), 3),
        "ab42_40_ratio": np.round(0.104 - 0.022 * amy - 0.002 * s + rng.normal(0, 0.006, n), 4),
        "nfl_pg_ml": np.round(np.clip(11 + 0.25 * (age - 68) + 4.5 * s + rng.normal(0, 4.5, n), 3, 80), 1),
        "gfap_pg_ml": np.round(np.clip(115 + 45 * amy + 22 * s + rng.normal(0, 35, n), 30, 600), 1),
        "dx": df.dx.values,
    })
    # ---- inject real-world messiness ----
    m = len(out)
    for col, rate in [("ptau217_pg_ml", .06), ("ab42_40_ratio", .06), ("nfl_pg_ml", .10), ("gfap_pg_ml", .08),
                      ("moca", .04), ("faq", .05), ("ecog_partner", .07), ("education_yrs", .02)]:
        out.loc[rng.random(m) < rate, col] = np.nan
    site_c = out.site == "SITE-C"                                  # SITE-C reports hippocampus in cm3
    out.loc[site_c, "hippocampus_vol_mm3"] = (out.loc[site_c, "hippocampus_vol_mm3"] / 1000).round(3)
    out["sex"] = [rng.choice({"M": ["M", "Male", "m"], "F": ["F", "Female", "f"]}[x], p=[.8, .15, .05]) for x in out.sex]
    out["dx"] = [rng.choice({"CN": ["CN", "Normal"], "SMC": ["SMC", "SCD"], "EMCI": ["EMCI", "E-MCI"],
                             "LMCI": ["LMCI", "L-MCI"], "AD": ["AD", "Dementia"]}[x], p=[.9, .1]) for x in out.dx]
    typo = rng.choice(m, 6, replace=False)
    out.loc[typo[:3], "age"] = out.loc[typo[:3], "age"] / 10           # 72 -> 7.2 entry error
    out.loc[typo[3:], "mmse"] = out.loc[typo[3:], "mmse"] * 10          # 28 -> 280
    out["patient_id"] = [f"PT-{p}" if rng.random() < 0.6 else p for p in out.patient_id]
    dups = out.sample(15, random_state=7)
    return pd.concat([out, dups]).sample(frac=1, random_state=11).reset_index(drop=True)


# ----------------------------------------------------------------------------- main
def main() -> None:
    seed_everything(CFG["project"]["seed"])
    rng = np.random.default_rng(CFG["project"]["seed"])
    raw = ROOT / CFG["data"]["raw_dir"]
    cohort = make_cohort(rng)
    print(f"Cohort: {len(cohort)} patients\n{cohort.dx.value_counts().reindex(CLASSES)}")
    tab = generate_tabular(cohort, rng)
    tab.to_csv(raw / "tabular" / "clinical_biomarkers.csv", index=False)
    demo = cohort.set_index("pid")[[]].copy()
    clean_age = tab.drop_duplicates("patient_id").copy()
    clean_age["pid"] = clean_age.patient_id.astype(str).str.extract(r"(\d+)").astype(int)
    demo = clean_age.set_index("pid")[["age", "sex"]].copy()
    demo["age"] = demo.age.where(demo.age > 30, demo.age * 10).astype(int)
    demo["sex"] = demo.sex.str[0].str.upper()
    notes = generate_notes(cohort, demo, rng)
    notes.to_csv(raw / "clinical_notes" / "clinical_notes.csv", index=False)
    img_meta = generate_images(cohort, rng, raw / "imaging")
    img_meta.to_csv(raw / "imaging" / "imaging_manifest.csv", index=False)
    cohort.to_csv(ROOT / CFG["data"]["interim_dir"] / "_latent_ground_truth_DO_NOT_USE_FOR_TRAINING.csv", index=False)
    print(f"Tabular rows: {len(tab)} | Notes: {len(notes)} | Images: {len(img_meta)}")
    print(img_meta.format.value_counts())


if __name__ == "__main__":
    main()
