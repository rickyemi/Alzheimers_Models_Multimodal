"""Train the multimodal fusion model and unimodal baselines (PyTorch).

Run:  python -m ad_mm.training.train                     # fusion + 3 unimodal baselines
      python -m ad_mm.training.train --modalities image text tabular
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from ad_mm.config import CFG, CLASSES, METRIC_DIR, MODEL_DIR, ROOT, seed_everything
from ad_mm.models.multimodal import MODALITIES, MultimodalADNet

PROC = ROOT / CFG["data"]["processed_dir"]


class MultimodalDataset(Dataset):
    def __init__(self, idx: np.ndarray, arrays: dict, augment: bool = False):
        self.idx, self.a, self.augment = idx, arrays, augment

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        img = torch.from_numpy(self.a["images"][j].copy())
        tab = torch.from_numpy(self.a["tabular"][j].copy())
        ids = torch.from_numpy(self.a["input_ids"][j].copy())
        if self.augment:
            if torch.rand(1) < 0.5:
                img = torch.flip(img, dims=[2])                       # left-right flip (anatomically symmetric phantom)
            dy, dx = np.random.randint(-4, 5, 2)
            img = torch.roll(img, shifts=(int(dy), int(dx)), dims=(1, 2))
            img = img * (1 + 0.1 * torch.randn(1)) + 0.05 * torch.randn_like(img)
            tab = tab + 0.05 * torch.randn_like(tab)
            m = (torch.rand_like(ids, dtype=torch.float) < 0.08) & (ids > 4)  # token dropout -> [UNK]
            ids = torch.where(m, torch.ones_like(ids), ids)
        return dict(image=img, input_ids=ids, attention_mask=torch.from_numpy(self.a["attention_mask"][j]),
                    tabular=tab, label=int(self.a["labels"][j]))


def load_arrays() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(PROC / "multimodal_dataset.csv")
    arrays = dict(images=np.load(PROC / "images.npy"), tabular=np.load(PROC / "tabular_features.npy").astype(np.float32),
                  input_ids=np.load(PROC / "input_ids.npy"), attention_mask=np.load(PROC / "attention_mask.npy"),
                  labels=df.label_idx.values)
    return df, arrays


def build_model(modalities, vocab_size: int, n_tab: int) -> MultimodalADNet:
    return MultimodalADNet(vocab_size, n_tab, len(CLASSES), CFG["model"], modalities, CFG["preprocessing"]["text_max_len"])


def run_epoch(model, loader, optim=None, loss_fn=None, aux_w: float = 0.0):
    train = optim is not None
    model.train(train)
    losses, ys, ps = [], [], []
    with torch.set_grad_enabled(train):
        for b in loader:
            logits, _ = model(b["image"], b["input_ids"], b["attention_mask"], b["tabular"])
            loss = loss_fn(logits, b["label"])
            if train and aux_w > 0 and getattr(model, "aux_logits", None):
                loss = loss + aux_w * sum(loss_fn(a_, b["label"]) for a_ in model.aux_logits) / len(model.aux_logits)
            if train:
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optim.step()
            losses.append(loss.item() * len(b["label"]))
            ys.append(b["label"].numpy())
            ps.append(logits.argmax(1).numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return sum(losses) / len(y), (y == p).mean(), f1_score(y, p, average="macro")


def train_model(modalities=MODALITIES, tag: str | None = None, verbose: bool = True, seed: int | None = None):
    seed = CFG["project"]["seed"] if seed is None else seed
    seed_everything(seed)
    torch.set_num_threads(2)
    tcfg = CFG["training"]
    df, a = load_arrays()
    tr, va = np.where(df.split == "train")[0], np.where(df.split == "val")[0]
    from transformers import PreTrainedTokenizerFast
    vocab = len(PreTrainedTokenizerFast.from_pretrained(str(MODEL_DIR / "tokenizer")))
    model = build_model(modalities, vocab, a["tabular"].shape[1])
    counts = np.bincount(a["labels"][tr], minlength=len(CLASSES))
    w = torch.tensor(len(tr) / (len(CLASSES) * counts), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    optim = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=tcfg["lr"], total_steps=tcfg["epochs"] *
                                                int(np.ceil(len(tr) / tcfg["batch_size"])), pct_start=0.15)

    class _Opt:  # step scheduler with optimizer
        def zero_grad(self): optim.zero_grad()
        def step(self): optim.step(); sched.step()

    g = torch.Generator().manual_seed(seed)
    tl = DataLoader(MultimodalDataset(tr, a, augment=True), batch_size=tcfg["batch_size"], shuffle=True, generator=g)
    vl = DataLoader(MultimodalDataset(va, a), batch_size=64)
    tag = tag or "_".join(modalities)
    hist, best, best_state, wait = [], -1, None, 0
    t0 = time.time()
    for ep in range(1, tcfg["epochs"] + 1):
        trl, tra, trf = run_epoch(model, tl, _Opt(), loss_fn, CFG["model"].get("aux_loss_weight", 0.0))
        vll, vla, vlf = run_epoch(model, vl, None, loss_fn)
        hist.append(dict(epoch=ep, train_loss=trl, train_acc=tra, train_f1=trf, val_loss=vll, val_acc=vla, val_f1=vlf))
        if vlf > best + 1e-4:
            best, wait = vlf, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"[{tag}] ep {ep:3d} | train loss {trl:.3f} f1 {trf:.3f} | val loss {vll:.3f} acc {vla:.3f} f1 {vlf:.3f}")
        if wait >= tcfg["early_stopping_patience"]:
            print(f"[{tag}] early stop at epoch {ep}")
            break
    model.load_state_dict(best_state)
    ckpt = MODEL_DIR / f"ad_mm_{tag}_s{seed}.pt"
    torch.save({"state_dict": best_state, "modalities": list(modalities), "vocab_size": vocab,
                "n_tabular": a["tabular"].shape[1], "classes": CLASSES, "model_cfg": CFG["model"],
                "max_len": CFG["preprocessing"]["text_max_len"]}, ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    meta = dict(tag=tag, seed=seed, modalities=list(modalities), best_val_macro_f1=best, epochs_run=len(hist),
                train_seconds=round(time.time() - t0, 1), n_params=n_params)
    with open(METRIC_DIR / f"history_{tag}_s{seed}.json", "w") as f:
        json.dump({"meta": meta, "history": hist}, f, indent=1)
    print(f"[{tag}] best val macro-F1 {best:.3f} | params {n_params:,} | {meta['train_seconds']}s -> {ckpt.name}")
    return model, hist, meta


def select_best_seed(tag: str, seeds: list[int]) -> dict:
    """Keep every seed for reporting; promote the best-validation seed as the deployable model (no test peeking)."""
    import shutil

    metas = [json.load(open(METRIC_DIR / f"history_{tag}_s{s}.json")) for s in seeds]
    best = max(metas, key=lambda m: m["meta"]["best_val_macro_f1"])
    s = best["meta"]["seed"]
    shutil.copy(MODEL_DIR / f"ad_mm_{tag}_s{s}.pt", MODEL_DIR / f"ad_mm_{tag}.pt")
    shutil.copy(METRIC_DIR / f"history_{tag}_s{s}.json", METRIC_DIR / f"history_{tag}.json")
    return {"tag": tag, "selected_seed": s, "val_macro_f1_by_seed": {m["meta"]["seed"]: m["meta"]["best_val_macro_f1"] for m in metas}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modalities", nargs="+", default=None)
    args = ap.parse_args()
    seeds = CFG["training"].get("seeds", [CFG["project"]["seed"]])
    configs = [tuple(args.modalities)] if args.modalities else [("image",), ("text",), ("tabular",), MODALITIES]
    summary = []
    for mods in configs:
        tag = "fusion" if len(mods) == 3 else "_".join(mods)
        for sd in seeds:
            train_model(mods, tag=tag, seed=sd)
        summary.append(select_best_seed(tag, seeds))
    with open(METRIC_DIR / "seed_selection.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
