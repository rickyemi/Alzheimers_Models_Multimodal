"""Build NVIDIA TensorRT engines from the ONNX models (run on a GPU host / inside nvcr.io/nvidia/tensorrt).

    python -m ad_mm.deployment.tensorrt_build --onnx models/ad_mm_fusion.onnx --fp16
    python -m ad_mm.deployment.tensorrt_build --onnx models/yolo/roi/weights/best.onnx --fp16 --yolo

Equivalent CLI:
    trtexec --onnx=models/ad_mm_fusion.onnx --saveEngine=models/ad_mm_fusion_fp16.plan --fp16 \
            --minShapes=image:1x1x96x96,input_ids:1x96,attention_mask:1x96,tabular:1x22,modality_mask:1x3 \
            --optShapes=image:8x1x96x96,input_ids:8x96,attention_mask:8x96,tabular:8x22,modality_mask:8x3 \
            --maxShapes=image:32x1x96x96,input_ids:32x96,attention_mask:32x96,tabular:32x22,modality_mask:32x3

Alternatively, ONNX Runtime can use TensorRT transparently via the TensorrtExecutionProvider
(ADPredictor prefers it automatically when available), and Triton serves the .plan via the tensorrt backend.
"""
from __future__ import annotations

import argparse
from pathlib import Path

FUSION_SHAPES = {"image": (1, 96, 96), "input_ids": (96,), "attention_mask": (96,), "tabular": (22,), "modality_mask": (3,)}


def build_engine(onnx_path: str, fp16: bool = True, yolo: bool = False, max_batch: int = 32, workspace_gb: int = 2) -> Path:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    shapes = {"images": (3, 96, 96)} if yolo else FUSION_SHAPES
    for name, s in shapes.items():
        profile.set_shape(name, (1, *s), (min(8, max_batch), *s), (max_batch, *s))
    config.add_optimization_profile(profile)
    engine = builder.build_serialized_network(network, config)
    out = Path(onnx_path).with_suffix(".fp16.plan" if fp16 else ".plan")
    out.write_bytes(engine)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--yolo", action="store_true")
    ap.add_argument("--max-batch", type=int, default=32)
    a = ap.parse_args()
    print(build_engine(a.onnx, a.fp16, a.yolo, a.max_batch))
