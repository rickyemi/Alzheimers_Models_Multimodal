export PYTHONPATH := src
.PHONY: install data preprocess eda train evaluate yolo onnx serve docker k8s test all

install:     ; pip install -r requirements.txt && pip install -e .
data:        ; python -m ad_mm.data.generate_synthetic
preprocess:  ; python -m ad_mm.preprocessing.merge
eda:         ; python -m ad_mm.eda.run_eda
train:       ; python -m ad_mm.training.train
evaluate:    ; python -m ad_mm.evaluation.evaluate
yolo:        ; python -m ad_mm.integrations.yolo_roi export && python -m ad_mm.integrations.yolo_roi train
onnx:        ; python -m ad_mm.deployment.export_onnx
serve:       ; uvicorn ad_mm.deployment.serve_api:app --port 8080
docker:      ; docker build -f deployment/docker/Dockerfile -t ad-mm-api:1.0.0 .
k8s:         ; kubectl apply -f deployment/k8s/ad-mm-deployment.yaml
test:        ; pytest -q
all:         ; bash scripts/run_pipeline.sh
