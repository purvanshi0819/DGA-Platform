# Transformer DGA Fault Diagnosis Platform

Academic research prototype (BTech final-year project). A React dashboard and FastAPI backend that unify three
notebook-based research implementations on dissolved gas analysis (DGA):

| Notebook | Implementation | Models |
|---|---|---|
| A `DGA_RandomForest_Replication` | RF comparative ML, SMOTE + grid search, 15 engineered features | RF, KNN, SVM-RBF, ANN |
| B `DGA_transformer_fault_colab_clean` | Duval % / Doernenburg ratios / Rogers ratios | RF, BPNN, KNN, SVM, DT, GNB (x3 feature sets) |
| C `PCA_ANN_paper3_colab` | 15 features, PCA, ANN (20 tanh, L-BFGS) | PCA + ANN, plus baselines |

Dataset: `DuVal.csv` (ahmedtariq71/Dataset1), fetched once into `backend/data/raw/` and never modified.
**The notebooks had no executed outputs and saved no models; every metric and model here was produced by `scripts/train_all.py`.**

## Quick start

```bash
# backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_all.py          # trains + exports 23 models and 3 experiments (~8 min on 1 CPU)
uvicorn app.main:app --port 8000     # API docs: http://localhost:8000/docs

# frontend (new terminal)
cd frontend
cp .env.example .env                 # VITE_API_BASE_URL, default http://localhost:8000/api
npm install
npm run dev                          # http://localhost:5173
```

Tests: `cd backend && pytest` (35 tests, including execution of the original notebook cells for parity)
and `cd frontend && npm test`.

Backend settings (environment variables): `DGA_DATASET_PATH`, `DGA_ARTIFACT_DIR`, `DGA_DB_PATH`,
`DGA_MAX_UPLOAD_BYTES`, `DGA_MAX_UPLOAD_ROWS`, `DGA_CORS_ORIGINS`, `DGA_SEED`.

## Architecture

```
React/Vite UI --REST--> FastAPI --> inference_service --> joblib pipelines + metadata JSON
                           |                                   ^
                           +--> SQLite (experiment index,      | scripts/train_all.py
                                prediction activity)           | (app/ml: dataset, feature_engineering,
                                                               |  rf_comparison, dga_methods, pca_ann)
                                              data/raw/DuVal.csv (immutable, SHA-256 recorded)
```
Training and inference are separate; the API never retrains and loads each model once (cached).
Feature functions in `app/ml/feature_engineering.py` are shared by training and inference, and
`tests/test_notebook_parity.py` executes the notebook cells to prove they match.

## API reference (prefix `/api`)

| Method & path | Purpose |
|---|---|
| GET `/health` | status, counts |
| GET `/models`, `/feature-sets` | trained models + provenance; feature definitions |
| POST `/diagnosis` | single reading, one or more `model_ids` |
| POST `/features` | all feature sets for one reading (zero denominators reported) |
| GET `/batch/template` | CSV template |
| POST `/batch/validate`, `/batch/predict` | multipart CSV (+ `model_ids`) |
| GET `/datasets/summary`, `/datasets/quality-report` | data-quality report |
| GET `/experiments`, `/experiments/{id}`, `/{id}/metrics`, `/{id}/confusion?model=`, `/{id}/feature-importance`, `/{id}/export` | results |
| GET `/pca/summary` | PCA experiment (variance, loadings, sweep, comparison) |
| GET `/overview` | dashboard statistics and recent activity |

## Design decisions on research integrity

- Notebook algorithms, parameters and seeds are ported verbatim; corrected variants are separate, labelled experiments (e.g. PCA+ANN with k re-selected per split).
- No imputation: missing/negative/non-numeric inputs are rejected; feature sets that use no epsilon in the notebook (Duval %, Doernenburg, Rogers) report zero denominators as "not computable". CO is accepted but unused by every model.
- Classifier scores are shown only where the model provides them and are labelled uncalibrated.
- Metrics from different protocols are never merged into one ranking; the comparison page is scoped to one experiment and protocol and shows macro vs. weighted averaging.
- Not implemented (shown as such in the UI): training from the browser, re-running PCA for an arbitrary k, per-prediction explanations, rule-based Duval/IEC/Rogers diagnosis.

## Known limitations

- Single public dataset (452 de-duplicated records; Arcing has 4-5); test sets are ~90 records with 1-2 Arcing samples, so differences of a few points are noise.
- Units and provenance of the data are undocumented. No external validation.
- Notebook C's paper rule selects k with test accuracy (optimistic); notebooks B and C keep conflicting-label rows.
- Component UI library is hand-built Tailwind components (shadcn/ui not used).

## Two-minute demo

1. **Overview** - show real counts and the per-experiment accuracy chart; point out the protocol caveat.
2. **Diagnosis** - "Load demo values", select 3 models, Run; show agreement, feature values, uncalibrated-score note, disclaimer.
3. **Batch** - upload the template CSV, run, search/filter results, download.
4. **Model comparison** - switch experiment/protocol/metric; read the generated research summary and the non-comparability warning.
5. **PCA + ANN lab** - scree plot, k-sweep (train vs CV vs test), loadings, baselines incl. the corrected comparison.
6. **History & method** - registry with seed, dataset hash and captured warnings.
