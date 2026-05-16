# User & Operational Manual

## 1. Installation

```bash
git clone <repo>
cd edu-predictive-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data acquisition

Either download the real OULAD release:

```bash
python scripts/download_oulad.py --target data/raw
```

…or generate a synthetic stand-in for development:

```bash
python scripts/generate_synthetic.py --target data/raw --students 5000
```

## 3. Running the pipeline

End-to-end:

```bash
python scripts/run_pipeline.py --config configs/pipeline.yaml
```

Selective stages:

```bash
python scripts/run_pipeline.py --stages ingest clean
python scripts/run_pipeline.py --stages features train
```

## 4. Serving

```bash
uvicorn src.serving.app:app --reload
# OpenAPI docs at http://localhost:8000/docs
```

## 5. Backup and recovery

* `data/bronze`, `data/silver`, `data/gold` are recoverable from `data/raw`.
* `models/artifacts/` should be backed up offsite — promotions cannot be
  recomputed without the same training snapshot.
* Snapshots are time-stamped (`run_YYYYMMDD_HHMMSS`).

## 6. Access controls

* The FastAPI service is intended to sit behind an authenticated reverse
  proxy (e.g. NGINX with OAuth2 proxy). No authentication is enabled in the
  reference build.
* Pipeline credentials are loaded from environment variables; do **not**
  commit secrets to the repo.

## 7. Monitoring

* Logs are written to `logs/pipeline_YYYYMMDD.log`, rotated at 10 MB.
* Drift reports are produced after each training run as
  `models/artifacts/<run_id>/drift_report.csv`.

## 8. Re-training cadence

* Recommended weekly retraining triggered by Prefect / Airflow.
* Manual trigger: `python scripts/run_pipeline.py --stages train`.
