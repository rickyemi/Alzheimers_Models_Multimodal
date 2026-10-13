"""Build (and optionally execute) the project notebooks from the source modules.

    python scripts/build_notebooks.py --execute
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
SETUP = """import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from IPython.display import Image, display, Markdown
FIG = ROOT / 'reports' / 'figures'
def show(name, width=900): display(Image(filename=str(FIG / f'{name}.png'), width=width))"""


def md(s): return nbf.v4.new_markdown_cell(s.strip())
def code(s): return nbf.v4.new_code_cell(s.strip())


NOTEBOOKS = {
"01_synthetic_data_generation.ipynb": [
    md("""# 01 · Synthetic multimodal data generation
Three **source datasets** are generated for 500 artificial patients across five classes
(CN, SMC, EMCI, LMCI, AD):

| Modality | Source format | Deliberate real-world issues |
|---|---|---|
| Vision | 125 `.dcm`, 125 `.nii`, 125 `.tiff`, 125 `.jpg` | 4 ID conventions, 4 resolutions, 3 bit depths, scanner-vendor intensity scaling, noise / low-contrast / flipped images |
| Text | clinical notes CSV | abbreviations, 6 duplicate notes, 8 orphan notes, 3 empty notes |
| Numeric | biomarker CSV (21 columns) | 15 duplicate rows, MCAR missingness, unit error at one site, decimal typos, label and sex coding variants |

A shared latent disease severity drives all modalities with independent noise, so each modality is
only partly informative and **fusion is needed** (SMC vs CN lives mostly in text, EMCI vs LMCI in
cognitive scores, LMCI vs AD in atrophy and function). All data are artificial."""),
    code(SETUP),
    code("""from ad_mm.data import generate_synthetic as g
from ad_mm.config import CFG
print(CFG['data'])
# g.main()   # uncomment to regenerate (scripts/run_pipeline.sh already ran it)"""),
    code("""raw = ROOT / 'data' / 'raw'
man = pd.read_csv(raw / 'imaging' / 'imaging_manifest.csv')
notes = pd.read_csv(raw / 'clinical_notes' / 'clinical_notes.csv')
tab = pd.read_csv(raw / 'tabular' / 'clinical_biomarkers.csv')
print('imaging studies:', len(man), '| notes:', len(notes), '| tabular rows:', len(tab))
display(man.head()); display(notes.head(3)); display(tab.head())"""),
    code("""for folder in ['dicom', 'nifti', 'tiff', 'jpg']:
    files = sorted((raw / 'imaging' / folder).iterdir())
    print(f'{folder:6s} {len(files):4d} files  e.g. {files[0].name}')"""),
],
"02_eda_imaging_and_dicom.ipynb": [
    md("""# 02 · Imaging EDA and DICOM workflow
Format inventory, raw heterogeneity, DICOM header audit, NIfTI volumes, automated QC, and
DICOM de-identification + PACS connectivity."""),
    code(SETUP),
    code("""from ad_mm.preprocessing import imaging as im
for f in ['dicom/PT-0001.dcm', 'nifti/sub-0002_T1w.nii', 'tiff/pt_0003.tiff', 'jpg/PATIENT0004.jpg']:
    p = ROOT / 'data/raw/imaging' / f
    if p.exists():
        arr, meta = im.load_any(p)
        print(f, {k: meta[k] for k in ('format', 'raw_shape', 'raw_min', 'raw_max')})"""),
    code("for n in ['01_img_format_inventory', '02_img_raw_samples_grid', '03_img_raw_intensity_heterogeneity', '04_img_dicom_header_audit', '05_img_nifti_orthoviews']: show(n)"),
    md("## Automated QC and remediation"),
    code("""qc = pd.read_csv(ROOT / 'data/interim/imaging_qc.csv')
display(qc.groupby(['format', 'injected_defect'])[['snr', 'noise_index', 'contrast', 'mask_fraction']].median().round(3))
show('06_img_qc_defects')"""),
    md("## DICOM de-identification (PS3.15 Basic Profile subset) and PACS C-ECHO / C-STORE"),
    code("""import pydicom
from ad_mm.dicom.dicom_utils import deidentify
ds = pydicom.dcmread(ROOT / 'data/raw/imaging/dicom' / sorted((ROOT / 'data/raw/imaging/dicom').iterdir())[0].name)
print('before:', ds.PatientName, ds.get('PatientBirthDate'), ds.StudyDate)
ds2, log = deidentify(ds)
print('after :', ds2.PatientName, ds2.get('PatientBirthDate'), ds2.StudyDate, '| removed', log['removed'])
print(json.load(open(ROOT / 'reports/metrics/pacs_demo.json')))"""),
],
"03_eda_text.ipynb": [
    md("# 03 · Clinical notes EDA\nQuality audit, length distributions, class-specific vocabulary (TF-IDF) and keyword prevalence."),
    code(SETUP),
    code("""notes = pd.read_csv(ROOT / 'data/raw/clinical_notes/clinical_notes.csv')
print('raw notes', len(notes), '| exact duplicates', notes.duplicated(['patient_id', 'visit_date', 'note_text']).sum(),
      '| empty', notes.note_text.isna().sum())
for t in notes.note_text.dropna().sample(3, random_state=0): print('-', t, '\\n')"""),
    code("""from ad_mm.preprocessing.text import clean_note
print(clean_note(notes.note_text.dropna().iloc[0]))"""),
    code("for n in ['10_txt_quality_audit', '11_txt_length_distributions', '12_txt_tfidf_terms_by_class', '13_txt_keyword_prevalence']: show(n)"),
],
"04_eda_tabular.ipynb": [
    md("# 04 · Numeric (clinical + biomarker) EDA\nMissingness, data-quality issues, distributions by class, correlations and PCA."),
    code(SETUP),
    code("""raw = pd.read_csv(ROOT / 'data/raw/tabular/clinical_biomarkers.csv')
display(raw.describe().T.round(2))
print(raw.dx.value_counts(), raw.sex.value_counts(), sep='\\n')"""),
    code("for n in ['14_tab_missingness', '15_tab_data_quality_fixes', '16_tab_features_by_class', '17_tab_correlation', '18_tab_pca']: show(n)"),
],
"05_preprocessing_and_merge.ipynb": [
    md("""# 05 · Preprocessing and merge into the final dataset
**Vision**: format-specific loader → 2D axial slice → resize 96×96 → percentile rescale → NLM denoise (if noisy) →
orientation QC vs population template → skull strip → brain-masked z-score.
**Text**: abbreviation expansion, number de-identification, de-duplication, latest note per patient, WordPiece tokenizer (fit on train).
**Numeric**: ID harmonization, de-duplication, unit/typo fixes, coding maps, missing indicators, median impute, log transform, z-score (fit on train).
**Merge**: inner join on harmonized `PT-####` → 500 patients; stratified 70/15/15 patient-level split."""),
    code(SETUP),
    code("for n in ['07_img_preprocessing_steps', '08_img_class_mean_maps', '09_img_intensity_harmonization']: show(n)"),
    code("""log = json.load(open(ROOT / 'reports/metrics/preprocessing_log.json'))
print(json.dumps(log, indent=1))"""),
    code("""ds = pd.read_csv(ROOT / 'data/processed/multimodal_dataset.csv')
print(ds.shape); display(ds[['patient_id', 'label', 'split', 'format', 'note_clean', 'age', 'mmse', 'ptau217_pg_ml']].head())
for f in ['images', 'input_ids', 'attention_mask', 'tabular_features']:
    print(f, np.load(ROOT / f'data/processed/{f}.npy').shape)"""),
    code("show('19_merge_flow'); show('20_final_class_balance_split')"),
],
"06_model_training.ipynb": [
    md("""# 06 · Model development (PyTorch + Hugging Face)
* **Vision**: residual CNN (swap for a pretrained backbone on real data)
* **Text**: Hugging Face `BertModel` (2 layers, 96 hidden) with a WordPiece tokenizer trained by `tokenizers`; production would load `Bio_ClinicalBERT`
* **Numeric**: MLP with batch norm
* **Fusion**: per-patient gated attention over modality embeddings + modality dropout (robust to a missing modality)
* Class-weighted cross-entropy, label smoothing, AdamW + OneCycle, early stopping on validation macro-F1.
Unimodal baselines use the same code with a subset of modalities."""),
    code(SETUP),
    code("""import torch
from ad_mm.models.multimodal import MultimodalADNet
from ad_mm.config import CFG
m = MultimodalADNet(843, 22, 5, CFG['model'])
print(m.encoders['text'].bert.config)
print({k: sum(p.numel() for p in v.parameters()) for k, v in m.encoders.items()})"""),
    code("""RETRAIN = False
if RETRAIN:
    from ad_mm.training.train import main; main()
for t in ['image', 'text', 'tabular', 'fusion']:
    print(json.load(open(ROOT / f'reports/metrics/history_{t}.json'))['meta'])
show('21_learning_curves', 1100)"""),
],
"07_evaluation_and_explainability.ipynb": [
    md("# 07 · Performance evaluation\nHeld-out test set (n=75): discrimination, calibration, missing-modality robustness, subgroup fairness, Grad-CAM, feature importance."),
    code(SETUP),
    code("""tm = json.load(open(ROOT / 'reports/metrics/test_metrics.json'))
display(pd.DataFrame(tm['results']).T.round(3))
print('95% CI (fusion, calibrated):', tm['fusion_calibrated_95ci'])
display(pd.DataFrame(tm['per_class']).T.round(3))"""),
    code("for n in ['22_model_comparison', '23_confusion_matrix_fusion', '24_roc_pr_fusion', '25_calibration']: show(n)"),
    code("for n in ['26_modality_attention', '27_modality_ablation', '28_subgroup_performance', '29_gradcam', '30_tabular_permutation_importance']: show(n)"),
],
"08_yolo_fiftyone_roboflow_v7.ipynb": [
    md("""# 08 · ROI detection and data-centric tooling
* **YOLO (Ultralytics)**: hippocampus + ventricle detector trained from scratch (YOLO11n), exported to ONNX for TensorRT / DeepStream
* **FiftyOne**: dataset browser, uniqueness / mistakenness / hardness to find label errors
* **Roboflow**: hosted annotation, versioning and augmentation of the YOLO dataset
* **V7 Darwin**: expert annotation on native DICOM / NIfTI volumes, converted back to YOLO labels
FiftyOne, Roboflow and V7 need their packages and API keys; their code paths are in `src/ad_mm/integrations/`."""),
    code(SETUP),
    code("""print((ROOT / 'data/yolo/data.yaml').read_text())
print(json.load(open(ROOT / 'reports/metrics/yolo_roi_metrics.json')) if (ROOT / 'reports/metrics/yolo_roi_metrics.json').exists() else 'train with: python -m ad_mm.integrations.yolo_roi train')
show('31_yolo_roi_labels', 1000)
if (FIG / '33_yolo_roi_predictions.png').exists(): show('33_yolo_roi_predictions', 1000)"""),
    code("""import importlib.util
for pkg in ['ultralytics', 'fiftyone', 'roboflow', 'darwin']:
    print(f'{pkg:12s}', 'installed' if importlib.util.find_spec(pkg) else 'not installed (optional)')
# FiftyOne:  from ad_mm.integrations.fiftyone_curation import build_dataset; d = build_dataset(predictions_csv=str(ROOT/'reports/metrics/test_predictions.csv'))
# Roboflow:  python -m ad_mm.integrations.roboflow_sync upload --workspace <ws> --project ad-roi
# V7:        python -m ad_mm.integrations.v7_darwin push --team <team> --dataset ad-mri"""),
],
"09_deployment_onnx_api.ipynb": [
    md("""# 09 · Model deployment
PyTorch → **ONNX** (calibrated softmax baked in) → **ONNX Runtime** (CPU / CUDA / TensorRT / CoreML execution providers) →
**FastAPI** service → **Docker** image → **Kubernetes** (HPA, PDB, NetworkPolicy) with optional **Triton** GPU serving
(**TensorRT** FP16) and **DeepStream** for streaming ROI detection. **Core ML** export targets on-device Apple inference."""),
    code(SETUP),
    code("""bm = json.load(open(ROOT / 'reports/metrics/deployment_benchmark.json'))
print(json.dumps(bm['parity'], indent=1)); show('32_deployment_benchmark', 1000)"""),
    code("""from ad_mm.deployment.onnx_inference import ADPredictor
from ad_mm.preprocessing.tabular import NUMERIC
ds = pd.read_csv(ROOT / 'data/processed/multimodal_dataset.csv'); r = ds[ds.split == 'test'].iloc[3]
pred = ADPredictor()
out = pred.predict(ROOT / 'data/raw/imaging' / r.file, r.note_text, r[NUMERIC].to_dict())
print('true:', r.label); print(json.dumps(out, indent=1))"""),
    code("""# Same patient with a missing modality (no imaging): the model re-weights the remaining modalities
print(json.dumps({k: v for k, v in pred.predict(None, r.note_text, r[NUMERIC].to_dict()).items() if k in ('prediction', 'confidence', 'modality_weights')}, indent=1))"""),
    code("""from fastapi.testclient import TestClient
from ad_mm.deployment.serve_api import app
with TestClient(app) as c:
    with open(ROOT / 'data/raw/imaging' / r.file, 'rb') as f:
        resp = c.post('/predict', files={'image': (Path(r.file).name, f)}, data={'note': r.note_text, 'numeric': json.dumps({'age': float(r.age), 'mmse': float(r.mmse)})})
    print(resp.status_code, resp.json()['prediction'], resp.json()['probabilities']); print(c.get('/metrics').text)"""),
    code("""for f in ['deployment/docker/Dockerfile', 'deployment/k8s/ad-mm-deployment.yaml', 'deployment/triton/model_repository/ad_mm/config.pbtxt']:
    print('=' * 20, f); print((ROOT / f).read_text()[:900])"""),
],
}


def main(execute: bool):
    NB.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        if execute:
            from nbclient import NotebookClient
            NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NB)}}).execute()
        nbf.write(nb, NB / name)
        print("wrote", name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    main(ap.parse_args().execute)
