"""Export the fusion model to ONNX, verify numerical parity with ONNX Runtime, quantize to INT8, and benchmark.

The exported graph includes temperature scaling + softmax, so it outputs calibrated probabilities and
per-patient modality attention weights directly.

Run:  python -m ad_mm.deployment.export_onnx
"""
from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic

from ad_mm.config import CFG, METRIC_DIR, MODEL_DIR, ROOT
from ad_mm.evaluation.evaluate import load_model
from ad_mm.training.train import load_arrays
from ad_mm.viz import save

ONNX_PATH = ROOT / CFG["deployment"]["onnx_path"]
INT8_PATH = ONNX_PATH.with_name("ad_mm_fusion_int8.onnx")
INPUTS = ["image", "input_ids", "attention_mask", "tabular", "modality_mask"]


class DeployWrapper(nn.Module):
    def __init__(self, model: nn.Module, temperature: float):
        super().__init__()
        self.model = model
        self.register_buffer("T", torch.tensor(float(temperature)))

    def forward(self, image, input_ids, attention_mask, tabular, modality_mask):
        logits, w = self.model(image, input_ids, attention_mask, tabular, modality_mask)
        return torch.softmax(logits / self.T, dim=-1), w


def feeds(a: dict, idx: np.ndarray) -> dict:
    return {"image": a["images"][idx], "input_ids": a["input_ids"][idx], "attention_mask": a["attention_mask"][idx],
            "tabular": a["tabular"][idx], "modality_mask": np.ones((len(idx), 3), np.float32)}


def export() -> dict:
    T = json.load(open(METRIC_DIR / "test_metrics.json"))["temperature"]
    model = DeployWrapper(load_model("fusion"), T).eval()
    df, a = load_arrays()
    te = np.where(df.split == "test")[0]
    ex = tuple(torch.from_numpy(v) for v in feeds(a, te[:2]).values())
    dyn = {k: {0: "batch"} for k in INPUTS} | {"probabilities": {0: "batch"}, "modality_weights": {0: "batch"}}
    torch.onnx.export(model, ex, str(ONNX_PATH), input_names=INPUTS, output_names=["probabilities", "modality_weights"],
                      dynamic_axes=dyn, opset_version=CFG["deployment"]["onnx_opset"], dynamo=False)
    m = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(m)
    m.metadata_props.add(key="classes", value=",".join(CFG["classes"]))
    m.metadata_props.add(key="temperature", value=str(T))
    m.metadata_props.add(key="intended_use", value="Research use only. Not for clinical diagnosis.")
    onnx.save(m, str(ONNX_PATH))
    quantize_dynamic(str(ONNX_PATH), str(INT8_PATH), weight_type=QuantType.QInt8)

    # ---- parity
    f = feeds(a, te)
    with torch.no_grad():
        pt = model(*[torch.from_numpy(v) for v in f.values()])[0].numpy()
    res = {}
    sessions = {}
    for nm, p in [("onnx_fp32", ONNX_PATH), ("onnx_int8", INT8_PATH)]:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 2
        s = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
        sessions[nm] = s
        out = s.run(["probabilities"], f)[0]
        res[nm] = {"max_abs_diff_vs_pytorch": float(np.abs(out - pt).max()),
                   "prediction_agreement": float((out.argmax(1) == pt.argmax(1)).mean()),
                   "test_accuracy": float((out.argmax(1) == a["labels"][te]).mean()),
                   "size_mb": round(p.stat().st_size / 1e6, 2)}

    # ---- latency benchmark
    torch.set_num_threads(2)
    bench = {"pytorch": {}, "onnx_fp32": {}, "onnx_int8": {}}
    for bs in (1, 8, 32):
        fb = feeds(a, te[:bs])
        tb = [torch.from_numpy(v) for v in fb.values()]
        for nm in bench:
            fn = (lambda: model(*tb)) if nm == "pytorch" else (lambda s=sessions[nm]: s.run(None, fb))
            with torch.no_grad():
                for _ in range(5):
                    fn()
                ts = []
                for _ in range(30):
                    t0 = time.perf_counter()
                    fn()
                    ts.append((time.perf_counter() - t0) * 1000)
            bench[nm][bs] = {"p50_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)),
                             "throughput_per_s": float(bs / (np.median(ts) / 1000))}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    cols = {"pytorch": "#8D99AE", "onnx_fp32": "#3B6EA5", "onnx_int8": "#2E7D6B"}
    x = np.arange(3)
    for i, nm in enumerate(bench):
        ax[0].bar(x + (i - 1) * 0.27, [bench[nm][b]["p50_ms"] for b in (1, 8, 32)], 0.27, label=nm, color=cols[nm])
        ax[1].bar(x + (i - 1) * 0.27, [bench[nm][b]["throughput_per_s"] for b in (1, 8, 32)], 0.27, label=nm, color=cols[nm])
    for k, t in enumerate(["p50 latency (ms), lower is better", "throughput (patients/s), higher is better"]):
        ax[k].set_xticks(x, ["batch 1", "batch 8", "batch 32"])
        ax[k].set_title(t)
        ax[k].legend()
    fig.suptitle(f"CPU inference benchmark (2 threads). INT8 size {res['onnx_int8']['size_mb']} MB vs FP32 "
                 f"{res['onnx_fp32']['size_mb']} MB; INT8 agreement {res['onnx_int8']['prediction_agreement']:.0%}",
                 fontweight="bold")
    save(fig, "32_deployment_benchmark")
    out = {"parity": res, "latency": bench, "onnx_path": str(ONNX_PATH.relative_to(ROOT)),
           "int8_path": str(INT8_PATH.relative_to(ROOT)), "opset": CFG["deployment"]["onnx_opset"]}
    with open(METRIC_DIR / "deployment_benchmark.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(res, indent=2))
    print({k: {b: round(v["p50_ms"], 2) for b, v in d.items()} for k, d in bench.items()})
    return out


if __name__ == "__main__":
    export()
