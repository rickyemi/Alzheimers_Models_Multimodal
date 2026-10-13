"""Format-agnostic neuroimaging loading, QC and preprocessing.

Supports DICOM (.dcm), NIfTI (.nii/.nii.gz), TIFF (.tif/.tiff) and JPEG (.jpg/.jpeg).
Every loader returns a 2D float32 axial slice (rows = anterior->posterior) plus metadata,
so downstream steps are identical regardless of the source format.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

ID_RE = re.compile(r"(\d{3,4})")


def normalize_patient_id(raw: str | int) -> str:
    """'sub-0007_T1w.nii', 'pt_0007', 'PATIENT0007', 'PT-0007', 7 -> 'PT-0007'."""
    m = ID_RE.search(str(raw))
    if not m:
        raise ValueError(f"No patient number in {raw!r}")
    return f"PT-{int(m.group(1)):04d}"


# ----------------------------------------------------------------------------- loaders
def load_dicom(path: Path) -> tuple[np.ndarray, dict]:
    import pydicom

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
    meta = dict(manufacturer=str(ds.get("Manufacturer", "")), field_strength=float(ds.get("MagneticFieldStrength", 0)),
                pixel_spacing=float(ds.PixelSpacing[0]), bits_stored=int(ds.BitsStored), modality=str(ds.Modality),
                dicom_patient_id=str(ds.PatientID))
    return arr, meta


def load_nifti(path: Path) -> tuple[np.ndarray, dict]:
    import nibabel as nib

    img = nib.load(str(path))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    k = vol.shape[2] // 2
    arr = vol[:, :, k].T                      # back to (row=y, col=x)
    zooms = img.header.get_zooms()
    return arr, dict(volume_shape=vol.shape, voxel_mm=tuple(float(z) for z in zooms), slice_index=k)


def load_tiff(path: Path) -> tuple[np.ndarray, dict]:
    import tifffile

    arr = tifffile.imread(str(path))
    return arr.astype(np.float32), dict(dtype=str(arr.dtype))


def load_jpg(path: Path) -> tuple[np.ndarray, dict]:
    from PIL import Image

    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.float32), dict(dtype="uint8", jpeg=True)


LOADERS = {".dcm": load_dicom, ".nii": load_nifti, ".gz": load_nifti, ".tif": load_tiff,
           ".tiff": load_tiff, ".jpg": load_jpg, ".jpeg": load_jpg}


def load_any(path: Path | str) -> tuple[np.ndarray, dict]:
    path = Path(path)
    loader = LOADERS[path.suffix.lower()]
    arr, meta = loader(path)
    meta.update(format=path.suffix.lower().lstrip("."), raw_shape=arr.shape, raw_min=float(arr.min()),
                raw_max=float(arr.max()))
    return arr, meta


# ----------------------------------------------------------------------------- QC
def qc_metrics(arr: np.ndarray) -> dict:
    a = (arr - arr.min()) / (np.ptp(arr) + 1e-6)
    n = a.shape[0]
    border = np.concatenate([a[:n // 10].ravel(), a[-n // 10:].ravel()])
    fg = a[a > 0.25]
    snr = float(fg.mean() / (border.std() + 1e-3)) if fg.size else 0.0
    # high-frequency noise estimate (Laplacian residual inside foreground)
    lap = ndimage.laplace(ndimage.gaussian_filter(a, 0.5))
    noise = float(np.median(np.abs(lap[a > 0.25]))) if fg.size else 0.0
    p1, p99 = np.percentile(arr, [1, 99])
    contrast = float((np.percentile(a, 99) - np.percentile(a, 50)))
    return dict(snr=snr, noise_index=noise, contrast=contrast, p1=float(p1), p99=float(p99))


def orientation_score(img: np.ndarray, template: np.ndarray) -> dict:
    """Correlate against a population template in the 4 flip states; returns best flip."""
    cands = {"none": img, "flip_ud": img[::-1, :], "flip_lr": img[:, ::-1], "flip_both": img[::-1, ::-1]}
    scores = {k: float(np.corrcoef(v.ravel(), template.ravel())[0, 1]) for k, v in cands.items()}
    return scores


# ----------------------------------------------------------------------------- preprocessing
def resize(arr: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)


def robust_rescale(arr: np.ndarray, pct=(1, 99)) -> np.ndarray:
    lo, hi = np.percentile(arr, pct)
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def brain_mask(img01: np.ndarray, thr: float = 0.2) -> np.ndarray:
    """Skull stripping by connected components.

    Foreground (smoothed intensity > thr) splits into the skull ring and the brain, separated by the dark
    CSF rim. The skull is ring-shaped (hole-filled area >> pixel area); the brain is the largest
    non-ring component. Holes (ventricles, sulci) are then filled.
    """
    fg = ndimage.gaussian_filter(img01, 0.8) > thr
    lab, n = ndimage.label(fg)
    best, best_area = None, 0
    for k in range(1, n + 1):
        comp = lab == k
        area = comp.sum()
        filled = ndimage.binary_fill_holes(comp).sum()
        if filled / max(area, 1) < 2.5 and area > best_area:
            best, best_area = comp, area
    if best is None:
        return np.ones_like(fg)
    m = ndimage.binary_closing(best, iterations=2)
    return ndimage.binary_fill_holes(m)


def denoise(img01: np.ndarray, noise_index: float, thr: float = 0.105) -> np.ndarray:
    if noise_index > thr:
        u8 = (img01 * 255).astype(np.uint8)
        return cv2.fastNlMeansDenoising(u8, None, h=18, templateWindowSize=5, searchWindowSize=11).astype(np.float32) / 255
    return img01


def zscore_in_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = img[mask]
    out = (img - vals.mean()) / (vals.std() + 1e-6)
    out[~mask] = out[mask].min()
    return out.astype(np.float32)


def preprocess(arr: np.ndarray, size: int, pct=(1, 99), template: np.ndarray | None = None,
               return_steps: bool = False) -> tuple[np.ndarray, dict]:
    steps = {"0_raw": arr}
    img = resize(arr, size)
    steps["1_resized"] = img
    img = robust_rescale(img, pct)
    steps["2_rescaled"] = img
    q = qc_metrics(img)
    img = denoise(img, q["noise_index"])
    steps["3_denoised"] = img
    info = dict(q)
    info["flip_applied"] = "none"
    if template is not None:
        sc = orientation_score(img, template)
        best = max(sc, key=sc.get)
        if best in ("flip_ud", "flip_both") and sc[best] - sc["none"] > 0.02:   # LR is anatomically symmetric
            img = img[::-1, :].copy()
            info["flip_applied"] = "flip_ud"
    steps["4_reoriented"] = img
    mask = brain_mask(img)
    steps["5_brain_mask"] = mask.astype(np.float32)
    out = zscore_in_mask(img, mask)
    steps["6_skullstripped_zscore"] = out
    info["mask_fraction"] = float(mask.mean())
    return (out, info, steps) if return_steps else (out, info)
