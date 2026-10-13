"""Convert the fusion model to Apple Core ML (.mlpackage) for on-device inference (iPad/Mac at point of care).

    pip install coremltools          # macOS recommended for running predictions
    python -m ad_mm.deployment.coreml_export

The TorchScript-traced deploy wrapper (calibrated softmax + modality weights) is converted with
fixed batch 1 and FP16 compute precision; Core ML dispatches to the Neural Engine / GPU / CPU.
ONNX Runtime can alternatively use the CoreMLExecutionProvider on Apple silicon with the same .onnx.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from ad_mm.config import CFG, METRIC_DIR, MODEL_DIR
from ad_mm.deployment.export_onnx import DeployWrapper
from ad_mm.evaluation.evaluate import load_model


def export() -> str:
    import coremltools as ct

    T = json.load(open(METRIC_DIR / "test_metrics.json"))["temperature"]
    model = DeployWrapper(load_model("fusion"), T).eval()
    L, n_tab = CFG["preprocessing"]["text_max_len"], len(json.load(open(MODEL_DIR / "tabular_transformer.json"))["columns"])
    ex = (torch.zeros(1, 1, 96, 96), torch.ones(1, L, dtype=torch.int32), torch.ones(1, L, dtype=torch.int32),
          torch.zeros(1, n_tab), torch.ones(1, 3))
    traced = torch.jit.trace(model, ex, strict=False)
    mlmodel = ct.convert(
        traced, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS16,
        inputs=[ct.TensorType("image", shape=(1, 1, 96, 96)), ct.TensorType("input_ids", shape=(1, L), dtype=np.int32),
                ct.TensorType("attention_mask", shape=(1, L), dtype=np.int32), ct.TensorType("tabular", shape=(1, n_tab)),
                ct.TensorType("modality_mask", shape=(1, 3))],
        outputs=[ct.TensorType("probabilities"), ct.TensorType("modality_weights")])
    mlmodel.short_description = "Multimodal AD 5-class classifier (research use only)"
    mlmodel.user_defined_metadata["classes"] = ",".join(CFG["classes"])
    out = MODEL_DIR / "ad_mm_fusion.mlpackage"
    mlmodel.save(str(out))
    return str(out)


if __name__ == "__main__":
    print(export())
