"""Shared plotting style; every figure is saved as PNG under reports/figures/."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ad_mm.config import CLASSES, FIG_DIR  # noqa: E402

CLASS_COLORS = dict(zip(CLASSES, ["#2E7D6B", "#7FB77E", "#E8B04B", "#D9773B", "#A23B3B"]))
FORMAT_COLORS = {"dcm": "#3B6EA5", "nii": "#7A5BA6", "tiff": "#3F9C8F", "jpg": "#C9733A"}
MODALITY_COLORS = {"image": "#3B6EA5", "text": "#C9733A", "tabular": "#3F9C8F", "fusion": "#22223B"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})


def save(fig, name: str) -> str:
    p = FIG_DIR / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(p)
