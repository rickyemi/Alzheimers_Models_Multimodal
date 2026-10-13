"""Held-out test evaluation: discrimination, calibration, ablation, subgroup fairness and explainability.

Run:  python -m ad_mm.evaluation.evaluate
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, auc, balanced_accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

from ad_mm.config import CFG, CLASSES, METRIC_DIR, MODEL_DIR
from ad_mm.models.multimodal import MultimodalADNet
from ad_mm.training.train import load_arrays
from ad_mm.viz import CLASS_COLORS, MODALITY_COLORS, save

TAGS = ["image", "text", "tabular", "fusion"]
K = len(CLASSES)


def load_model(tag: str) -> MultimodalADNet:
    """tag: 'fusion', 'image', ... (deployed best-val seed) or 'fusion_s7' for a specific seed."""
    ck = torch.load(MODEL_DIR / f"ad_mm_{tag}.pt", weights_only=False)
    m = MultimodalADNet(ck["vocab_size"], ck["n_tabular"], len(ck["classes"]), ck["model_cfg"], ck["modalities"], ck["max_len"])
    m.load_state_dict(ck["state_dict"])
    return m.eval()


@torch.no_grad()
def predict(model, a, idx, modality_mask=None, tab_override=None):
    tab = torch.from_numpy(a["tabular"][idx] if tab_override is None else tab_override)
    mm = None if modality_mask is None else torch.tensor(modality_mask, dtype=torch.float32).repeat(len(idx), 1)
    logits, w = model(torch.from_numpy(a["images"][idx]), torch.from_numpy(a["input_ids"][idx]),
                      torch.from_numpy(a["attention_mask"][idx]), tab, mm)
    return logits.numpy(), w.numpy()


def softmax(z, T=1.0):
    z = z / T
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def ece(probs, y, n_bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = (conf > lo) & (conf <= hi)
        if k.any():
            e += k.mean() * abs((pred[k] == y[k]).mean() - conf[k].mean())
    return float(e)


def metrics(y, probs) -> dict:
    p = probs.argmax(1)
    Y = label_binarize(y, classes=range(K))
    return dict(accuracy=accuracy_score(y, p), balanced_accuracy=balanced_accuracy_score(y, p),
                macro_f1=f1_score(y, p, average="macro"), weighted_f1=f1_score(y, p, average="weighted"),
                macro_auc_ovr=roc_auc_score(y, probs, multi_class="ovr", average="macro"),
                quadratic_kappa=cohen_kappa_score(y, p, weights="quadratic"),
                brier=float(((probs - Y) ** 2).sum(1).mean()), ece=ece(probs, y))


def bootstrap_ci(y, probs, n=1000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ["accuracy", "macro_f1", "macro_auc_ovr", "balanced_accuracy"]}
    for _ in range(n):
        b = rng.integers(0, len(y), len(y))
        if len(np.unique(y[b])) < K:
            continue
        m = metrics(y[b], probs[b])
        for k in out:
            out[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def fit_temperature(logits, y) -> float:
    Ts = np.linspace(0.5, 3.0, 51)
    nll = [-np.log(softmax(logits, T)[np.arange(len(y)), y] + 1e-12).mean() for T in Ts]
    return float(Ts[int(np.argmin(nll))])


def gradcam(model, img: torch.Tensor, a_ids, a_mask, tab, target: int) -> np.ndarray:
    enc = model.encoders["image"]
    acts = {}
    layer = enc.features[-1]
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: acts.__setitem__("g", go[0]))
    model.zero_grad()
    logits, _ = model(img, a_ids, a_mask, tab)
    logits[0, target].backward()
    h1.remove(), h2.remove()
    w = acts["g"].mean((2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=img.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    return cam / (cam.max() + 1e-8)


def main() -> None:
    df, a = load_arrays()
    te = np.where(df.split == "test")[0]
    va = np.where(df.split == "val")[0]
    y = a["labels"][te]
    results, probs_all, logits_all = {}, {}, {}
    models = {t: load_model(t) for t in TAGS}
    for t, m in models.items():
        lg, _ = predict(m, a, te)
        logits_all[t] = lg
        probs_all[t] = softmax(lg)
        results[t] = metrics(y, probs_all[t])
    # ---- seed robustness: every seed of every model on the same test set
    seeds = CFG["training"].get("seeds", [])
    seed_rows = []
    for t in TAGS:
        for sd in seeds:
            if (MODEL_DIR / f"ad_mm_{t}_s{sd}.pt").exists():
                pr = softmax(predict(load_model(f"{t}_s{sd}"), a, te)[0])
                seed_rows.append(dict(model=t, seed=sd, **metrics(y, pr)))
    seed_df = pd.DataFrame(seed_rows)
    seed_summary = seed_df.groupby("model")[["accuracy", "balanced_accuracy", "macro_f1", "macro_auc_ovr", "quadratic_kappa"]].agg(
        ["mean", "std"]).reindex(TAGS) if len(seed_df) else pd.DataFrame()
    # temperature scaling for fusion (fit on validation)
    fus = models["fusion"]
    lv, _ = predict(fus, a, va)
    T = fit_temperature(lv, a["labels"][va])
    p_cal = softmax(logits_all["fusion"], T)
    results["fusion_calibrated"] = metrics(y, p_cal)
    ci = bootstrap_ci(y, p_cal)
    report = classification_report(y, p_cal.argmax(1), target_names=CLASSES, output_dict=True)

    # ---- 21 learning curves
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6), sharey=True)
    for ax, t in zip(axes, TAGS):
        h = pd.DataFrame(json.load(open(METRIC_DIR / f"history_{t}.json"))["history"])
        ax.plot(h.epoch, h.train_f1, label="train macro-F1", color=MODALITY_COLORS[t], ls="--")
        ax.plot(h.epoch, h.val_f1, label="val macro-F1", color=MODALITY_COLORS[t])
        ax.plot(h.epoch, h.val_loss / h.val_loss.max(), label="val loss (scaled)", color="#999")
        ax.set_title(f"{t}")
        ax.set_xlabel("epoch")
    axes[0].legend(fontsize=8)
    fig.suptitle("Learning curves (early stopping on validation macro-F1)", fontweight="bold")
    save(fig, "21_learning_curves")

    # ---- 22 model comparison (mean +/- SD across seeds)
    show = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc_ovr", "quadratic_kappa"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(show))
    for i, t in enumerate(TAGS):
        if len(seed_summary):
            mu = [seed_summary.loc[t, (k, "mean")] for k in show]
            sd = [seed_summary.loc[t, (k, "std")] for k in show]
        else:
            mu, sd = [results[t][k] for k in show], [0] * len(show)
        ax.bar(x + (i - 1.5) * 0.2, mu, 0.2, yerr=sd, capsize=2, label=t, color=MODALITY_COLORS[t])
    ax.set_xticks(x, [k.replace("_", " ") for k in show])
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=4, loc="upper left")
    ax.set_title(f"Held-out test set (n={len(te)}): unimodal baselines vs fusion, mean +/- SD over {len(seeds)} training seeds")
    save(fig, "22_model_comparison")

    # ---- 23 confusion matrix
    cm = confusion_matrix(y, p_cal.argmax(1))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax[0], cbar=False)
    sns.heatmap(cm / cm.sum(1, keepdims=True), annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES,
                yticklabels=CLASSES, ax=ax[1], cbar=False)
    for a_ in ax:
        a_.set_xlabel("predicted"), a_.set_ylabel("true")
    ax[0].set_title("Fusion confusion matrix (counts)")
    ax[1].set_title("Row-normalized (recall)")
    save(fig, "23_confusion_matrix_fusion")

    # ---- 24 ROC + PR
    Y = label_binarize(y, classes=range(K))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for k, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(Y[:, k], p_cal[:, k])
        ax[0].plot(fpr, tpr, color=CLASS_COLORS[c], label=f"{c} AUC={auc(fpr, tpr):.2f}")
        pr, rc, _ = precision_recall_curve(Y[:, k], p_cal[:, k])
        ax[1].plot(rc, pr, color=CLASS_COLORS[c], label=f"{c} AP={auc(rc, pr):.2f}")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set_xlabel("1 - specificity"), ax[0].set_ylabel("sensitivity")
    ax[1].set_xlabel("recall"), ax[1].set_ylabel("precision")
    ax[0].set_title("One-vs-rest ROC (fusion)")
    ax[1].set_title("Precision-recall (fusion)")
    ax[0].legend(), ax[1].legend()
    save(fig, "24_roc_pr_fusion")

    # ---- 25 calibration
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for probs, nm, col in [(probs_all["fusion"], f"uncalibrated ECE={results['fusion']['ece']:.3f}", "#999"),
                           (p_cal, f"temperature T={T:.2f} ECE={results['fusion_calibrated']['ece']:.3f}", "#22223B")]:
        conf, corr = probs.max(1), probs.argmax(1) == y
        bins = np.linspace(0, 1, 8)
        mid, acc = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            k = (conf > lo) & (conf <= hi)
            if k.sum() >= 2:
                mid.append(conf[k].mean()), acc.append(corr[k].mean())
        ax[0].plot(mid, acc, "o-", label=nm, color=col)
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set_xlabel("confidence"), ax[0].set_ylabel("accuracy")
    ax[0].set_title("Reliability diagram (test)")
    ax[0].legend()
    ax[1].hist(p_cal.max(1)[p_cal.argmax(1) == y], bins=15, alpha=.7, label="correct", color="#2E7D6B")
    ax[1].hist(p_cal.max(1)[p_cal.argmax(1) != y], bins=15, alpha=.7, label="incorrect", color="#A23B3B")
    ax[1].axvline(0.6, ls="--", c="k")
    ax[1].text(0.61, ax[1].get_ylim()[1] * .9, "refer-to-clinician\nthreshold", fontsize=8)
    ax[1].set_title("Confidence of correct vs incorrect predictions")
    ax[1].legend()
    save(fig, "25_calibration")

    # ---- 26 modality attention by class
    _, w = predict(fus, a, te)
    wdf = pd.DataFrame(w, columns=["image", "text", "tabular"]).assign(label=[CLASSES[i] for i in y]).melt("label")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=wdf, x="label", y="value", hue="variable", order=CLASSES, palette=MODALITY_COLORS, ax=ax, errorbar="sd")
    ax.set_ylabel("mean attention weight")
    ax.set_title("Learned modality attention by true class (per-patient, test set)")
    save(fig, "26_modality_attention")

    # ---- 27 modality ablation (missing modality robustness)
    abl = {}
    for nm, mask in {"all": [1, 1, 1], "no image": [0, 1, 1], "no text": [1, 0, 1], "no tabular": [1, 1, 0],
                     "image only": [1, 0, 0], "text only": [0, 1, 0], "tabular only": [0, 0, 1]}.items():
        lg, _ = predict(fus, a, te, modality_mask=mask)
        abl[nm] = f1_score(y, lg.argmax(1), average="macro")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(list(abl), list(abl.values()), color=["#22223B"] + ["#8D99AE"] * 3 + ["#BFC6D1"] * 3)
    for i, v in enumerate(abl.values()):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
    ax.set_ylim(0, 1)
    ax.set_ylabel("test macro-F1")
    ax.set_title("Missing-modality robustness of the fusion model (modality masked at inference)")
    save(fig, "27_modality_ablation")

    # ---- 28 subgroup performance
    tdf = df.iloc[te].copy()
    tdf["pred"], tdf["y"] = p_cal.argmax(1), y
    tdf["age_band"] = pd.cut(tdf.age, [0, 70, 78, 120], labels=["<70", "70-77", ">=78"])
    tdf["sex_lbl"] = tdf.sex.map({1: "male", 0: "female"})
    tdf["apoe4"] = np.where(tdf.apoe4_copies > 0, "APOE4+", "APOE4-")
    rows = []
    for g in ["sex_lbl", "age_band", "apoe4", "format", "site"]:
        for lvl, s in tdf.groupby(g, observed=True):
            rows.append(dict(group=g.replace("_lbl", ""), level=str(lvl), n=len(s), accuracy=(s.pred == s.y).mean()))
    sub = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(13, 4))
    colors = dict(zip(sub.group.unique(), ["#3B6EA5", "#7A5BA6", "#3F9C8F", "#C9733A", "#A23B3B"]))
    ax.bar(range(len(sub)), sub.accuracy, color=[colors[g] for g in sub.group])
    ax.set_xticks(range(len(sub)), [f"{l}\n(n={n})" for l, n in zip(sub.level, sub.n)], fontsize=8)
    ax.axhline(results["fusion_calibrated"]["accuracy"], ls="--", c="k", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_title("Subgroup accuracy (dashed = overall): sex, age band, APOE4, image format, site")
    for i, g in enumerate(sub.group):
        if i == 0 or sub.group.iloc[i - 1] != g:
            ax.text(i, 1.0, g, fontweight="bold", fontsize=9)
    save(fig, "28_subgroup_performance")

    # ---- 29 Grad-CAM
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for c_i, c in enumerate(CLASSES):
        j = te[np.where((y == c_i) & (p_cal.argmax(1) == c_i))[0][0]] if ((y == c_i) & (p_cal.argmax(1) == c_i)).any() else te[np.where(y == c_i)[0][0]]
        img = torch.from_numpy(a["images"][j:j + 1]).requires_grad_(True)
        cam = gradcam(fus, img, torch.from_numpy(a["input_ids"][j:j + 1]), torch.from_numpy(a["attention_mask"][j:j + 1]),
                      torch.from_numpy(a["tabular"][j:j + 1]), c_i)
        axes[0, c_i].imshow(a["images"][j, 0], cmap="gray")
        axes[0, c_i].set_title(c, color=CLASS_COLORS[c])
        axes[1, c_i].imshow(a["images"][j, 0], cmap="gray")
        axes[1, c_i].imshow(cam, cmap="jet", alpha=0.45)
        for r in (0, 1):
            axes[r, c_i].axis("off")
    fig.suptitle("Grad-CAM on the image encoder of the fusion model (true-class target)", fontweight="bold")
    save(fig, "29_gradcam")

    # ---- 30 permutation importance of tabular features
    cols = json.load(open(MODEL_DIR / "tabular_transformer.json"))["columns"]
    base = f1_score(y, predict(fus, a, te)[0].argmax(1), average="macro")
    rng = np.random.default_rng(0)
    imp = {}
    for k, c in enumerate(cols):
        drops = []
        for _ in range(10):
            X = a["tabular"][te].copy()
            X[:, k] = rng.permutation(X[:, k])
            drops.append(base - f1_score(y, predict(fus, a, te, tab_override=X)[0].argmax(1), average="macro"))
        imp[c] = float(np.mean(drops))
    imp_s = pd.Series(imp).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(imp_s.index, imp_s.values, color="#3F9C8F")
    ax.set_xlabel("drop in test macro-F1 when permuted")
    ax.set_title("Numeric feature importance within the fusion model")
    save(fig, "30_tabular_permutation_importance")

    pd.DataFrame(p_cal, columns=[f"p_{c}" for c in CLASSES]).assign(
        patient_id=df.patient_id.values[te], true=[CLASSES[i] for i in y], pred=[CLASSES[i] for i in p_cal.argmax(1)]
    ).to_csv(METRIC_DIR / "test_predictions.csv", index=False)
    seed_df.to_csv(METRIC_DIR / "test_metrics_by_seed.csv", index=False)
    out = dict(test_n=int(len(te)), temperature=T, results=results,
               seed_summary={f"{m}|{k}|{st}": float(v) for (k, st), col in seed_summary.items() for m, v in col.items()}, fusion_calibrated_95ci=ci,
               per_class=report, modality_ablation_macro_f1=abl, subgroups=sub.to_dict("records"),
               tabular_permutation_importance=imp)
    with open(METRIC_DIR / "test_metrics.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(pd.DataFrame(results).T.round(3).to_string())
    print(seed_summary.round(3).to_string())
    print("fusion calibrated 95% CI:", {k: [round(x, 3) for x in v] for k, v in ci.items()})
    print("ablation:", {k: round(v, 3) for k, v in abl.items()})


if __name__ == "__main__":
    main()
