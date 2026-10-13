"""End-to-end preprocessing: harmonize the three source datasets and merge into the
final 500-patient multimodal dataset used for model development.

    vision (500 studies, 4 formats) --+
    text   (514 notes)              ---+--> patient_id join --> 500 x {image, note, 22 numeric} --> train/val/test
    numeric(515 rows)               --+

Run:  python -m ad_mm.preprocessing.merge
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ad_mm.config import CFG, CLASS_TO_IDX, METRIC_DIR, MODEL_DIR, ROOT, seed_everything
from ad_mm.preprocessing import imaging as im
from ad_mm.preprocessing.tabular import TabularTransformer, clean_tabular
from ad_mm.preprocessing.text import clean_notes_table, train_wordpiece_tokenizer

RAW = ROOT / CFG["data"]["raw_dir"]
INTERIM = ROOT / CFG["data"]["interim_dir"]
PROC = ROOT / CFG["data"]["processed_dir"]


def process_images() -> tuple[pd.DataFrame, np.ndarray, dict]:
    size = CFG["preprocessing"]["image_size"]
    pct = tuple(CFG["preprocessing"]["intensity_clip_percentiles"])
    manifest = pd.read_csv(RAW / "imaging" / "imaging_manifest.csv")
    raw_cache = {}
    for f in manifest.file:
        arr, meta = im.load_any(RAW / "imaging" / f)
        raw_cache[f] = (arr, meta)
    # population template (median of rescaled images) used for orientation QC
    stack = np.stack([im.robust_rescale(im.resize(a, size), pct) for a, _ in raw_cache.values()])
    template = np.median(stack, axis=0)
    rows, arrays = [], []
    for f in manifest.file:
        arr, meta = raw_cache[f]
        out, info = im.preprocess(arr, size, pct, template=template)
        pid = im.normalize_patient_id(f)
        rows.append(dict(patient_id=pid, file=f, **{k: v for k, v in meta.items() if not isinstance(v, (tuple, list))},
                         raw_shape=str(meta["raw_shape"]), **info))
        arrays.append(out)
    qc = pd.DataFrame(rows).merge(manifest[["file", "injected_defect", "scanner"]], on="file")
    qc["img_idx"] = np.arange(len(qc))
    qc.to_csv(INTERIM / "imaging_qc.csv", index=False)
    return qc, np.stack(arrays)[:, None].astype(np.float32), {"template": template}


def main() -> None:
    seed_everything(CFG["project"]["seed"])
    log = {}
    # 1) vision
    qc, images, extra = process_images()
    np.save(INTERIM / "image_template.npy", extra["template"])
    log["imaging"] = {"studies": len(qc), "by_format": qc.format.value_counts().to_dict(),
                      "flip_corrected": int((qc.flip_applied != "none").sum()),
                      "denoised": int((qc.noise_index > 0.105).sum())}
    # 2) text
    notes, log["text"] = clean_notes_table(pd.read_csv(RAW / "clinical_notes" / "clinical_notes.csv"), im.normalize_patient_id)
    notes.to_csv(INTERIM / "notes_clean.csv", index=False)
    # 3) numeric
    tab, log["tabular"] = clean_tabular(pd.read_csv(RAW / "tabular" / "clinical_biomarkers.csv"), im.normalize_patient_id)
    tab.to_csv(INTERIM / "tabular_clean.csv", index=False)
    # 4) merge (inner join on harmonized patient_id)
    ids_img, ids_txt, ids_tab = set(qc.patient_id), set(notes.patient_id), set(tab.patient_id)
    log["merge"] = {"image_ids": len(ids_img), "note_ids": len(ids_txt), "tabular_ids": len(ids_tab),
                    "notes_without_image_or_tabular(orphans)": len(ids_txt - ids_img - ids_tab),
                    "all_three": len(ids_img & ids_txt & ids_tab)}
    df = (qc[["patient_id", "img_idx", "file", "format", "scanner"]]
          .merge(notes[["patient_id", "note_clean", "note_text"]], on="patient_id", how="inner")
          .merge(tab, on="patient_id", how="inner"))
    assert len(df) == CFG["data"]["n_patients"], len(df)
    df = df.rename(columns={"dx": "label"})
    df["label_idx"] = df.label.map(CLASS_TO_IDX)
    # 5) stratified split (patient level, no leakage)
    s = CFG["split"]
    tr, te = train_test_split(df.index, test_size=s["test_size"], stratify=df.label, random_state=CFG["project"]["seed"])
    tr, va = train_test_split(tr, test_size=s["val_size"] / (1 - s["test_size"]), stratify=df.loc[tr, "label"],
                              random_state=CFG["project"]["seed"])
    df["split"] = "train"
    df.loc[va, "split"] = "val"
    df.loc[te, "split"] = "test"
    # 6) fit transformers on TRAIN only
    tt = TabularTransformer().fit(df[df.split == "train"])
    X = tt.transform(df)
    with open(MODEL_DIR / "tabular_transformer.json", "w") as f:
        json.dump(tt.to_dict(), f, indent=1)
    tok = train_wordpiece_tokenizer(df.loc[df.split == "train", "note_clean"].tolist(),
                                    CFG["preprocessing"]["tokenizer_vocab_size"], MODEL_DIR / "tokenizer")
    enc = tok(df.note_clean.tolist(), padding="max_length", truncation=True,
              max_length=CFG["preprocessing"]["text_max_len"], return_tensors="np")
    log["text"]["truncated_notes"] = int((np.array([len(tok(t)["input_ids"]) for t in df.note_clean])
                                          > CFG["preprocessing"]["text_max_len"]).sum())
    # 7) persist final dataset
    imgs = images[df.img_idx.values]
    np.save(PROC / "images.npy", imgs)
    np.save(PROC / "tabular_features.npy", X.values)
    np.save(PROC / "input_ids.npy", enc["input_ids"].astype(np.int64))
    np.save(PROC / "attention_mask.npy", enc["attention_mask"].astype(np.int64))
    df["row_idx"] = np.arange(len(df))
    df.to_csv(PROC / "multimodal_dataset.csv", index=False)
    pd.DataFrame(X.values, columns=tt.columns).assign(patient_id=df.patient_id.values).to_csv(
        PROC / "tabular_features_scaled.csv", index=False)
    log["final"] = {"rows": len(df), "image_tensor": list(imgs.shape), "text_tensor": list(enc["input_ids"].shape),
                    "tabular_tensor": list(X.shape), "split": df.split.value_counts().to_dict(),
                    "class_counts": df.label.value_counts().to_dict(), "vocab_size": len(tok)}
    with open(METRIC_DIR / "preprocessing_log.json", "w") as f:
        json.dump(log, f, indent=2, default=int)
    print(json.dumps(log, indent=2, default=int))


if __name__ == "__main__":
    main()
