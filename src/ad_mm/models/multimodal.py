"""Multimodal AD classifier: CNN (vision) + Hugging Face BERT (text) + MLP (numeric) with gated attention fusion.

                 image (1x96x96) --> ResNet-style CNN ------> 128-d --+
    input_ids / attention_mask --> BertModel (HF) [CLS] ----> 128-d --+--> modality attention --> fused 128-d --> 5 logits
             numeric (22 features) --> MLP -----------------> 128-d --+

* Modality attention produces per-patient weights over modalities (interpretable, logged at inference).
* Modality dropout during training makes the model robust to a missing modality at deployment time.
* Auxiliary per-modality heads (training only) add a deep-supervision loss on each modality embedding,
  which counters "modality laziness" (the fused head otherwise over-relies on the easiest modality).
* `modalities` lets the same class train unimodal baselines for ablation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel

MODALITIES = ("image", "text", "tabular")


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.skip = nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.max_pool2d(F.relu(h + self.skip(x)), 2)


class ImageEncoder(nn.Module):
    """Lightweight residual CNN. Swap for a pretrained backbone (timm / torchvision / MONAI) on real data."""

    def __init__(self, channels=(16, 32, 64, 128), out_dim=128, dropout=0.3):
        super().__init__()
        layers, cin = [], 1
        for c in channels:
            layers.append(ConvBlock(cin, c))
            cin = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, out_dim), nn.GELU())

    def forward(self, x):
        f = self.features(x)
        return self.head(F.adaptive_avg_pool2d(f, 1))


class TextEncoder(nn.Module):
    """Hugging Face `BertModel`. Built from a config here (offline, synthetic vocab);
    in production load clinical weights: BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')."""

    def __init__(self, vocab_size: int, hidden=96, layers=2, heads=4, max_len=96, out_dim=128, dropout=0.3):
        super().__init__()
        cfg = BertConfig(vocab_size=vocab_size, hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
                         intermediate_size=hidden * 4, max_position_embeddings=max_len + 2, hidden_dropout_prob=0.1,
                         attention_probs_dropout_prob=0.1, pad_token_id=0)
        self.bert = BertModel(cfg, add_pooling_layer=False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, out_dim), nn.GELU())

    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)          # masked mean pooling
        return self.head(pooled)


class TabularEncoder(nn.Module):
    def __init__(self, n_features: int, hidden=(64, 128), out_dim=128, dropout=0.3):
        super().__init__()
        dims, layers = [n_features, *hidden], []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-1], out_dim), nn.GELU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultimodalADNet(nn.Module):
    def __init__(self, vocab_size: int, n_tabular: int, n_classes: int = 5, cfg: dict | None = None,
                 modalities=MODALITIES, max_len: int = 96):
        super().__init__()
        cfg = cfg or {}
        d = cfg.get("embed_dim", 128)
        drop = cfg.get("dropout", 0.3)
        self.modalities = tuple(modalities)
        self.modality_dropout = cfg.get("modality_dropout", 0.15)
        self.encoders = nn.ModuleDict()
        if "image" in self.modalities:
            self.encoders["image"] = ImageEncoder(tuple(cfg.get("image_channels", (16, 32, 64, 128))), d, drop)
        if "text" in self.modalities:
            self.encoders["text"] = TextEncoder(vocab_size, cfg.get("text_hidden", 96), cfg.get("text_layers", 2),
                                                cfg.get("text_heads", 4), max_len, d, drop)
        if "tabular" in self.modalities:
            self.encoders["tabular"] = TabularEncoder(n_tabular, tuple(cfg.get("tabular_hidden", (64, 128))), d, drop)
        self.norm = nn.LayerNorm(d)
        self.gate = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))
        self.classifier = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, n_classes))
        self.aux_heads = nn.ModuleList([nn.Linear(d, n_classes) for _ in self.modalities]) if len(self.modalities) > 1 else None
        self.aux_logits: list[torch.Tensor] = []

    def embed(self, image, input_ids, attention_mask, tabular) -> torch.Tensor:
        embs = []
        for m in self.modalities:
            if m == "image":
                embs.append(self.encoders["image"](image))
            elif m == "text":
                embs.append(self.encoders["text"](input_ids, attention_mask))
            else:
                embs.append(self.encoders["tabular"](tabular))
        return self.norm(torch.stack(embs, 1))                 # (B, M, d)

    def forward(self, image, input_ids, attention_mask, tabular, modality_mask: torch.Tensor | None = None):
        """modality_mask: (B, M) with 1 = available, 0 = missing. Returns (logits, attention weights)."""
        e = self.embed(image, input_ids, attention_mask, tabular)
        B, M, _ = e.shape
        if modality_mask is None:
            modality_mask = torch.ones(B, M, device=e.device)
        if self.training and M > 1 and self.modality_dropout > 0:
            drop = (torch.rand(B, M, device=e.device) < self.modality_dropout).float()
            keep_one = F.one_hot(torch.randint(0, M, (B,), device=e.device), M).float()
            modality_mask = modality_mask * torch.clamp(1 - drop + keep_one, max=1.0)
        scores = self.gate(e).squeeze(-1).masked_fill(modality_mask == 0, -1e4)
        w = torch.softmax(scores, dim=1)
        fused = (w.unsqueeze(-1) * e).sum(1)
        if self.training and self.aux_heads is not None:
            self.aux_logits = [h(e[:, i]) for i, h in enumerate(self.aux_heads)]
        return self.classifier(fused), w
