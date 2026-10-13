"""Exploratory data analysis for the three source datasets and the merged dataset.

Run:  python -m ad_mm.eda.run_eda      (after data generation and preprocessing)
All figures are written to reports/figures/*.png
"""
from __future__ import annotations

import json
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from ad_mm.config import CFG, CLASSES, METRIC_DIR, ROOT
from ad_mm.preprocessing import imaging as im
from ad_mm.preprocessing.tabular import DX_MAP, NUMERIC
from ad_mm.viz import CLASS_COLORS, FORMAT_COLORS, save

RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROC = ROOT / "data" / "processed"


def _labels() -> pd.Series:
    t = pd.read_csv(INTERIM / "tabular_clean.csv")
    return t.set_index("patient_id").dx


# =============================================================================== imaging
def imaging_eda() -> list[str]:
    out = []
    qc = pd.read_csv(INTERIM / "imaging_qc.csv")
    lab = _labels()
    qc["dx"] = qc.patient_id.map(lab)
    man = pd.read_csv(RAW / "imaging" / "imaging_manifest.csv")

    # 01 inventory
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    cnt = qc.format.value_counts().reindex(FORMAT_COLORS)
    ax[0].bar(cnt.index, cnt.values, color=[FORMAT_COLORS[f] for f in cnt.index])
    ax[0].set_title("Studies by file format")
    for i, v in enumerate(cnt.values):
        ax[0].text(i, v + 2, str(v), ha="center")
    pd.crosstab(qc.format, qc.dx).reindex(columns=CLASSES).plot.bar(stacked=True, ax=ax[1],
                                                                     color=[CLASS_COLORS[c] for c in CLASSES], rot=0)
    ax[1].set_title("Diagnosis mix within each format")
    ax[1].legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    tbl = qc.groupby("format").agg(shape=("raw_shape", "first"), min=("raw_min", "median"),
                                   max=("raw_max", "median")).reindex(FORMAT_COLORS)
    tbl["dtype"] = ["uint16 (12-bit)", "int16 volume", "uint16", "uint8 (lossy)"]
    ax[2].axis("off")
    t = ax[2].table(cellText=tbl[["shape", "dtype", "min", "max"]].round(0).astype(str).values, rowLabels=tbl.index,
                    colLabels=["raw shape", "dtype", "median min", "median max"], loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.6)
    ax[2].set_title("Heterogeneous raw characteristics")
    out.append(save(fig, "01_img_format_inventory"))

    # 02 raw sample grid
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    for r, fmt in enumerate(FORMAT_COLORS):
        for c, cls in enumerate(CLASSES):
            f = qc[(qc.format == fmt) & (qc.dx == cls) & (qc.injected_defect == "none")].file.iloc[0]
            arr, _ = im.load_any(RAW / "imaging" / f)
            axes[r, c].imshow(arr, cmap="gray")
            axes[r, c].set_xticks([]), axes[r, c].set_yticks([])
            axes[r, c].grid(False)
            if r == 0:
                axes[r, c].set_title(cls, color=CLASS_COLORS[cls])
            if c == 0:
                axes[r, c].set_ylabel(f".{fmt}\n{arr.shape}", fontsize=10)
    fig.suptitle("Raw source images by format (rows) and diagnosis (columns): note ventricle and hippocampal change", fontweight="bold")
    out.append(save(fig, "02_img_raw_samples_grid"))

    # 03 raw intensity heterogeneity
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    sns.boxplot(data=qc, x="format", y="raw_max", hue="format", palette=FORMAT_COLORS, ax=ax[0], legend=False)
    ax[0].set_yscale("log")
    ax[0].set_title("Raw maximum intensity by format (log scale)")
    for scn, g in man[man.format == "dcm"].groupby("scanner"):
        vals = np.concatenate([im.load_any(RAW / "imaging" / f)[0].ravel()[::7] for f in g.file.head(15)])
        sns.kdeplot(vals[vals > 300], ax=ax[1], label=scn, bw_adjust=0.6)
    ax[1].set_title("DICOM raw intensity by scanner vendor (scanner effect)")
    ax[1].set_xlabel("signal intensity (a.u.)")
    ax[1].legend()
    out.append(save(fig, "03_img_raw_intensity_heterogeneity"))

    # 04 DICOM header summary
    d = qc[qc.format == "dcm"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    d.manufacturer.value_counts().plot.barh(ax=ax[0], color="#3B6EA5")
    ax[0].set_title("Manufacturer (0008,0070)")
    d.field_strength.value_counts().sort_index().plot.bar(ax=ax[1], color="#7A5BA6", rot=0)
    ax[1].set_title("Field strength T (0018,0087)")
    d.bits_stored.value_counts().plot.bar(ax=ax[2], color="#3F9C8F", rot=0)
    ax[2].set_title("Bits stored (0028,0101)")
    fig.suptitle("DICOM metadata audit (n=125)", fontweight="bold")
    out.append(save(fig, "04_img_dicom_header_audit"))

    # 05 NIfTI ortho views
    import nibabel as nib
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for r, cls in enumerate(["CN", "AD"]):
        f = qc[(qc.format == "nii") & (qc.dx == cls)].file.iloc[0]
        vol = np.asarray(nib.load(RAW / "imaging" / f).dataobj)
        x, y, z = (s // 2 for s in vol.shape)
        for c, (sl, nm) in enumerate([(vol[:, :, z].T, "axial"), (vol[:, y, :].T, "coronal"), (vol[x, :, :].T, "sagittal")]):
            axes[r, c].imshow(sl, cmap="gray", origin="lower" if c else "upper", aspect="auto")
            axes[r, c].set_title(f"{cls}: {nm}")
            axes[r, c].axis("off")
    fig.suptitle("NIfTI volumes (64x64x24, 3x3x4 mm): orthogonal views; axial mid-slice is used downstream", fontweight="bold")
    out.append(save(fig, "05_img_nifti_orthoviews"))

    # 06 QC scatter
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    pal = {"none": "#B0B0B0", "low_contrast": "#3B6EA5", "high_noise": "#A23B3B", "flip_lr": "#7A5BA6", "flip_ud": "#E8B04B"}
    sns.scatterplot(data=qc, x="noise_index", y="contrast", hue="injected_defect", palette=pal, ax=ax[0], s=22)
    ax[0].axvline(0.105, ls="--", c="k", lw=1)
    ax[0].text(0.107, ax[0].get_ylim()[1] * 0.95, "denoise threshold", fontsize=8)
    ax[0].set_title("Automated image QC: noise vs contrast")
    rate = qc.groupby("injected_defect").apply(lambda g: pd.Series({
        "n": len(g), "denoised": (g.noise_index > 0.105).mean(), "flip_fixed": (g.flip_applied != "none").mean()}),
        include_groups=False)
    rate[["denoised", "flip_fixed"]].plot.bar(ax=ax[1], rot=0, color=["#A23B3B", "#E8B04B"])
    ax[1].set_title("Remediation rate by injected defect")
    ax[1].set_ylabel("fraction of studies")
    out.append(save(fig, "06_img_qc_defects"))

    # 07 preprocessing steps
    template = np.load(INTERIM / "image_template.npy")
    examples = [qc[(qc.format == "dcm") & (qc.injected_defect == "none")].file.iloc[0],
                qc[qc.injected_defect == "high_noise"].file.iloc[0],
                qc[(qc.injected_defect == "flip_ud") & (qc.flip_applied != "none")].file.iloc[0]
                if ((qc.injected_defect == "flip_ud") & (qc.flip_applied != "none")).any() else qc.file.iloc[3],
                qc[qc.format == "nii"].file.iloc[0]]
    fig, axes = plt.subplots(len(examples), 7, figsize=(15, 9))
    for r, f in enumerate(examples):
        arr, _ = im.load_any(RAW / "imaging" / f)
        _, info, steps = im.preprocess(arr, CFG["preprocessing"]["image_size"], template=template, return_steps=True)
        for c, (nm, s) in enumerate(steps.items()):
            axes[r, c].imshow(s, cmap="gray")
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(nm.split("_", 1)[1].replace("_", " "), fontsize=9)
        axes[r, 0].text(-0.1, 0.5, f.split("/")[-1], transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    fig.suptitle("Imaging preprocessing pipeline: load -> resize 96x96 -> percentile rescale -> NLM denoise (if noisy) -> "
                 "orientation QC -> skull strip -> z-score", fontweight="bold", fontsize=10)
    out.append(save(fig, "07_img_preprocessing_steps"))

    # 08 class mean maps
    ds = pd.read_csv(PROC / "multimodal_dataset.csv")
    imgs = np.load(PROC / "images.npy")[:, 0]
    fig, axes = plt.subplots(1, 6, figsize=(16, 3.2))
    means = {c: imgs[ds.label.values == c].mean(0) for c in CLASSES}
    for i, c in enumerate(CLASSES):
        axes[i].imshow(means[c], cmap="gray")
        axes[i].set_title(f"{c} mean (n={int((ds.label == c).sum())})", color=CLASS_COLORS[c])
        axes[i].axis("off")
    diff = means["AD"] - means["CN"]
    lim = np.abs(diff).max()
    h = axes[5].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[5].set_title("AD minus CN")
    axes[5].axis("off")
    fig.colorbar(h, ax=axes[5], fraction=0.046)
    out.append(save(fig, "08_img_class_mean_maps"))

    # 09 harmonization before/after
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for fmt in FORMAT_COLORS:
        sub = qc[qc.format == fmt].file.head(30)
        raw = np.concatenate([im.resize(im.load_any(RAW / "imaging" / f)[0], 96).ravel() for f in sub])
        sns.kdeplot(raw / np.percentile(raw, 99.5), ax=ax[0], label=fmt, color=FORMAT_COLORS[fmt], bw_adjust=0.5)
        post = imgs[ds.format.values == fmt][:30].ravel()
        sns.kdeplot(post, ax=ax[1], label=fmt, color=FORMAT_COLORS[fmt], bw_adjust=0.5)
    ax[0].set_title("Before: raw intensity (scaled by global p99.5)")
    ax[1].set_title("After: brain-masked z-score (harmonized)")
    ax[0].legend(), ax[1].legend()
    out.append(save(fig, "09_img_intensity_harmonization"))
    return out


# =============================================================================== text
def text_eda() -> list[str]:
    out = []
    raw = pd.read_csv(RAW / "clinical_notes" / "clinical_notes.csv")
    log = json.load(open(METRIC_DIR / "preprocessing_log.json"))
    ds = pd.read_csv(PROC / "multimodal_dataset.csv")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    steps = {"raw notes": len(raw), "exact duplicates": -log["text"]["exact_duplicates_removed"],
             "orphans (no image/tabular)": -log["merge"]["notes_without_image_or_tabular(orphans)"],
             "final notes": len(ds)}
    cols = ["#3B6EA5", "#A23B3B", "#A23B3B", "#2E7D6B"]
    ax.barh(list(steps)[::-1], [abs(v) for v in steps.values()][::-1], color=cols[::-1])
    for i, v in enumerate(list(steps.values())[::-1]):
        ax.text(abs(v) + 4, i, str(v), va="center")
    ax.set_title(f"Clinical notes quality audit ({log['text']['empty_notes']} empty notes filled with placeholder)")
    out.append(save(fig, "10_txt_quality_audit"))

    ds["n_words"] = ds.note_text.fillna("").str.split().str.len()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    sns.violinplot(data=ds, x="label", y="n_words", order=CLASSES, hue="label", palette=CLASS_COLORS, ax=ax[0], legend=False, cut=0)
    ax[0].set_title("Note length (words) by diagnosis")
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(str(ROOT / "models" / "tokenizer"))
    lens = [len(tok(t)["input_ids"]) for t in ds.note_clean]
    ax[1].hist(lens, bins=30, color="#C9733A")
    ax[1].axvline(CFG["preprocessing"]["text_max_len"], c="k", ls="--")
    ax[1].set_title(f"WordPiece token length (vocab={len(tok)}); dashed = max_len")
    out.append(save(fig, "11_txt_length_distributions"))

    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, stop_words="english")
    X = tf.fit_transform(ds.note_clean)
    terms = np.array(tf.get_feature_names_out())
    m = pd.DataFrame({c: np.asarray(X[ds.label.values == c].mean(0)).ravel() for c in CLASSES}, index=terms)
    spec = m.div(m.mean(1) + 1e-6, axis=0)
    top = pd.unique(np.concatenate([spec[c].sort_values(ascending=False).head(5).index for c in CLASSES]))
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(m.loc[top], cmap="YlOrBr", ax=ax, cbar_kws={"label": "mean TF-IDF"})
    ax.set_title("Most class-specific terms (TF-IDF)")
    out.append(save(fig, "12_txt_tfidf_terms_by_class"))

    kws = ["denies", "subjective", "worry", "mild", "impaired", "amnestic", "dementia", "independent", "assistance", "disoriented"]
    prev = pd.DataFrame({k: ds.groupby("label").note_clean.apply(lambda s: s.str.contains(k).mean()) for k in kws}).reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    sns.heatmap(prev, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar=False)
    ax.set_title("Keyword prevalence by diagnosis (fraction of notes)")
    out.append(save(fig, "13_txt_keyword_prevalence"))
    return out


# =============================================================================== tabular
def tabular_eda() -> list[str]:
    out = []
    raw = pd.read_csv(RAW / "tabular" / "clinical_biomarkers.csv")
    clean = pd.read_csv(INTERIM / "tabular_clean.csv")
    log = json.load(open(METRIC_DIR / "preprocessing_log.json"))

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1, 1.6]})
    miss = raw[NUMERIC].drop(columns=["sex"]).isna().mean().sort_values()
    ax[0].barh(miss.index, miss.values * 100, color="#A23B3B")
    ax[0].set_xlabel("% missing")
    ax[0].set_title("Missingness per feature (raw)")
    sns.heatmap(raw[miss.index[::-1]].isna().T, cbar=False, cmap=["#F2F2F2", "#A23B3B"], ax=ax[1], xticklabels=False)
    ax[1].set_title("Missingness matrix (rows = records)")
    out.append(save(fig, "14_tab_missingness"))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    sns.stripplot(data=raw, x="site", y="hippocampus_vol_mm3", ax=ax[0], s=3, order=sorted(raw.site.unique()))
    ax[0].set_yscale("log")
    ax[0].set_title("Before: SITE-C reports hippocampus in cm³")
    sns.stripplot(data=clean, x="site", y="hippocampus_vol_mm3", ax=ax[1], s=3, order=sorted(clean.site.unique()))
    ax[1].set_title(f"After unit harmonization ({log['tabular']['hippocampus_cm3_to_mm3']} records fixed)")
    issues = {"duplicate rows": log["tabular"]["duplicates_removed"], "age decimal typo": log["tabular"]["age_decimal_fix"],
              "MMSE x10 typo": log["tabular"]["mmse_decimal_fix"], "unit errors": log["tabular"]["hippocampus_cm3_to_mm3"],
              "dx label variants": int((~raw.dx.isin(["CN", "SMC", "EMCI", "LMCI", "AD"])).sum()),
              "sex coding variants": int((~raw.sex.isin(["M", "F"])).sum())}
    ax[2].barh(list(issues), list(issues.values()), color="#D9773B")
    ax[2].set_title("Data quality issues detected and fixed")
    for i, v in enumerate(issues.values()):
        ax[2].text(v + 1, i, str(v), va="center")
    out.append(save(fig, "15_tab_data_quality_fixes"))

    feats = ["age", "mmse", "moca", "cdr_sb", "adas_cog13", "faq", "ecog_self", "hippocampus_vol_mm3",
             "entorhinal_thick_mm", "ventricles_vol_mm3", "ptau217_pg_ml", "ab42_40_ratio", "nfl_pg_ml", "gfap_pg_ml", "ecog_partner"]
    fig, axes = plt.subplots(3, 5, figsize=(17, 9))
    for a, f in zip(axes.ravel(), feats):
        sns.boxplot(data=clean, x="dx", y=f, order=CLASSES, hue="dx", palette=CLASS_COLORS, ax=a, legend=False, fliersize=1.5)
        a.set_title(f)
        a.set_xlabel(""), a.set_ylabel("")
    fig.suptitle("Numeric feature distributions by diagnosis (cleaned)", fontweight="bold")
    out.append(save(fig, "16_tab_features_by_class"))

    fig, ax = plt.subplots(figsize=(10, 8.5))
    corr = clean[NUMERIC].corr(method="spearman")
    sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax, square=True, cbar_kws={"shrink": .7})
    ax.set_title("Spearman correlation of numeric features")
    out.append(save(fig, "17_tab_correlation"))

    X = np.load(PROC / "tabular_features.npy")
    ds = pd.read_csv(PROC / "multimodal_dataset.csv")
    pc = PCA(2).fit(X)
    Z = pc.transform(X)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for c in CLASSES:
        k = ds.label.values == c
        ax.scatter(Z[k, 0], Z[k, 1], s=14, label=c, color=CLASS_COLORS[c], alpha=.8)
    ax.set_xlabel(f"PC1 ({pc.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pc.explained_variance_ratio_[1]:.0%})")
    ax.set_title("PCA of scaled tabular features: CN and SMC overlap")
    ax.legend()
    out.append(save(fig, "18_tab_pca"))
    return out


# =============================================================================== merge
def merge_eda() -> list[str]:
    out = []
    log = json.load(open(METRIC_DIR / "preprocessing_log.json"))
    ds = pd.read_csv(PROC / "multimodal_dataset.csv")
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 12), ax.set_ylim(0, 5)
    src = [("Vision: 500 studies\n125 .dcm | 125 .nii | 125 .tiff | 125 .jpg", 4.1, "#3B6EA5"),
           (f"Text: {log['text']['rows_raw']} notes\n-{log['text']['exact_duplicates_removed']} dup, "
            f"-{log['merge']['notes_without_image_or_tabular(orphans)']} orphans", 2.5, "#C9733A"),
           (f"Numeric: {log['tabular']['rows_raw']} rows x 21 cols\n-{log['tabular']['duplicates_removed']} dup, "
            f"unit/typo fixes", 0.9, "#3F9C8F")]
    for txt, y, c in src:
        ax.add_patch(plt.Rectangle((0.2, y - 0.5), 3.6, 1.0, color=c, alpha=.9))
        ax.text(2.0, y, txt, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        ax.annotate("", xy=(5.0, 2.5), xytext=(3.85, y), arrowprops=dict(arrowstyle="->", lw=1.6, color="#555"))
    ax.add_patch(plt.Rectangle((5.0, 1.7), 2.4, 1.6, color="#22223B"))
    ax.text(6.2, 2.5, "patient_id\nharmonization\n+ inner join\n(PT-####)", ha="center", va="center", color="white", fontsize=9)
    ax.annotate("", xy=(8.2, 2.5), xytext=(7.45, 2.5), arrowprops=dict(arrowstyle="->", lw=1.6))
    f = log["final"]
    ax.add_patch(plt.Rectangle((8.2, 0.6), 3.6, 3.8, color="#2E7D6B"))
    ax.text(10.0, 2.5, f"FINAL DATASET\n{f['rows']} patients\n\nimage {tuple(f['image_tensor'])}\ntext ids {tuple(f['text_tensor'])}\n"
            f"numeric {tuple(f['tabular_tensor'])}\n\ntrain {f['split']['train']} | val {f['split']['val']} | test {f['split']['test']}",
            ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.set_title("Merging 500 imaging studies x 500 notes x 500 numeric records into one multimodal dataset", fontweight="bold")
    out.append(save(fig, "19_merge_flow"))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ds.label.value_counts().reindex(CLASSES).plot.bar(ax=ax[0], color=[CLASS_COLORS[c] for c in CLASSES], rot=0)
    ax[0].set_title("Final class balance (n=500)")
    pd.crosstab(ds.label, ds.split).reindex(CLASSES)[["train", "val", "test"]].plot.bar(ax=ax[1], rot=0,
                                                                                         color=["#3B6EA5", "#E8B04B", "#A23B3B"])
    ax[1].set_title("Stratified patient-level split (70/15/15)")
    out.append(save(fig, "20_final_class_balance_split"))
    return out


def main() -> None:
    figs = imaging_eda() + text_eda() + tabular_eda() + merge_eda()
    print("\n".join(figs))


if __name__ == "__main__":
    main()
