#!/usr/bin/env bash
# End-to-end pipeline: data -> EDA -> preprocessing -> training -> evaluation -> DICOM/PACS -> YOLO -> ONNX -> tests
#   bash scripts/run_pipeline.sh            # full run (~25 min on a laptop CPU)
#   SKIP_YOLO=1 bash scripts/run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

echo "== 1. Generate synthetic vision/text/numeric source datasets"
python -m ad_mm.data.generate_synthetic
echo "== 2. Preprocess each modality and merge into the final 500-patient dataset"
python -m ad_mm.preprocessing.merge
echo "== 3. EDA figures"
python -m ad_mm.eda.run_eda
echo "== 4. DICOM de-identification + PACS (C-ECHO/C-STORE) demo"
python -m ad_mm.dicom.dicom_utils
python -m ad_mm.dicom.pacs_client --demo
echo "== 5. Train unimodal baselines + multimodal fusion"
python -m ad_mm.training.train
echo "== 6. Evaluate on held-out test set"
python -m ad_mm.evaluation.evaluate
echo "== 7. YOLO ROI dataset (+ training if ultralytics is installed)"
python -m ad_mm.integrations.yolo_roi export
if [[ -z "${SKIP_YOLO:-}" ]] && python -c "import ultralytics" 2>/dev/null; then
  python -m ad_mm.integrations.yolo_roi train
  python -m ad_mm.integrations.yolo_roi export-onnx
fi
echo "== 8. ONNX export, parity check, INT8 quantization, benchmark"
python -m ad_mm.deployment.export_onnx
echo "== 9. Tests"
python -m pytest -q tests
echo "Done. Figures: reports/figures  Metrics: reports/metrics  Models: models/"
