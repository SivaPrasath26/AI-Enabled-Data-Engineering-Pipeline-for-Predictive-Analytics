# AI-Enabled Data Engineering Pipeline for Predictive Analytics in Educational Settings

End-to-end reference implementation of a medallion-architecture data pipeline that ingests Learning Management System (LMS) data, engineers behavioural features at scale on Apache Spark, trains gradient-boosting and deep-learning models in TensorFlow / XGBoost, generates SHAP-based explanations, and serves student risk predictions through a FastAPI inference endpoint.

The implementation accompanies the project report titled *Design and Implementation of an AI-Enabled Data Engineering Pipeline for Predictive Analytics in Educational Settings*, submitted in partial fulfilment of the M.Sc. (Data Science) degree.

---

## Repository layout

```
.
├── configs/                # YAML configuration files for each pipeline stage
├── data/
│   ├── raw/                # raw OULAD CSV drops (gitignored)
│   ├── bronze/             # ingested parquet, fidelity preserved
│   ├── silver/             # cleaned, schema-validated, deduplicated
│   └── gold/               # feature-engineered, model-ready datasets
├── docs/                   # extra documentation, ADRs, diagrams
├── notebooks/              # exploratory data analysis notebooks
├── scripts/                # operational entry-points (run_pipeline.sh etc.)
├── src/
│   ├── ingestion/          # CSV / Kafka / API source adapters
│   ├── preprocessing/      # cleaning, schema, deduplication
│   ├── features/           # PySpark window-aggregation feature factory
│   ├── models/             # XGBoost, LightGBM, LSTM trainers
│   ├── evaluation/         # metrics, cross-validation, drift detection
│   ├── explainability/     # SHAP integration
│   ├── serving/            # FastAPI inference service + schemas
│   ├── spark_jobs/         # Spark application drivers
│   └── utils/              # logging, IO, config loaders
├── tests/                  # unit + integration tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Stack

| Layer            | Technology                            |
|------------------|---------------------------------------|
| Language         | Python 3.11                           |
| Distributed ETL  | Apache Spark 3.5 (PySpark)            |
| Storage          | Parquet on local FS (S3-ready)        |
| Tabular models   | XGBoost 2.0, LightGBM 4.x             |
| Deep learning    | TensorFlow 2.16, Keras                |
| Explainability   | SHAP 0.45                             |
| Serving          | FastAPI + Uvicorn                     |
| Validation       | Great Expectations, Pydantic          |
| Orchestration    | Prefect 2 (optional)                  |
| Container        | Docker, docker-compose                |
| CI / quality     | pytest, ruff, mypy                    |

## Quick start

```bash
# 1. create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. download the OULAD dataset (see scripts/download_oulad.py)
python scripts/download_oulad.py --target data/raw

# 4. run the full pipeline (ingest -> bronze -> silver -> gold -> train -> evaluate)
python scripts/run_pipeline.py --config configs/pipeline.yaml

# 5. launch the inference API
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

To run the pipeline inside Docker:

```bash
docker compose up --build
```

## Modules

### 1. Ingestion (`src/ingestion`)

CSV-based batch ingestion adapter for the OULAD release, plus a Kafka adapter stub for live LMS event streams. Each adapter writes parquet partitions to the Bronze tier with ingest timestamps and source provenance columns.

### 2. Preprocessing (`src/preprocessing`)

Schema enforcement, type casting, null-handling policies, deduplication, and outlier filtering. Promotes Bronze data to the Silver tier. Schema definitions live in `configs/schemas/`.

### 3. Feature engineering (`src/features`)

Spark-based feature factory that computes student-week aggregates over the VLE click-stream and assessment tables: cumulative GPA, login frequency, days-since-last-activity, late-submission ratios, session-duration variance, and resource-type access counts. Output: Gold-tier feature store partitioned by `code_module` and `code_presentation`.

### 4. Models (`src/models`)

Three trainers with a shared `BaseTrainer` interface:

* `xgboost_trainer.py` — tabular ensemble (primary baseline)
* `lightgbm_trainer.py` — faster gradient boosting alternative
* `lstm_trainer.py` — sequential model on weekly VLE click counts

All trainers checkpoint to `models/artifacts/` with a manifest JSON containing hyper-parameters, metrics, and feature schema.

### 5. Evaluation (`src/evaluation`)

Stratified k-fold CV, ROC / PR / calibration metrics, fairness slicing by demographic attributes, and PSI-based drift detection between training and serving distributions.

### 6. Explainability (`src/explainability`)

Global SHAP summary plots and per-prediction local attributions exposed through the serving API so educators see *why* a student was flagged.

### 7. Serving (`src/serving`)

FastAPI application loading the latest model artifact at startup. Endpoints:

* `GET  /healthz`               — liveness check
* `POST /predict`               — single student risk score
* `POST /predict/batch`         — bulk inference (≤1 000 rows)
* `POST /explain`               — SHAP attribution for one record

OpenAPI schema served at `/docs`.

## Testing

```bash
pytest tests/ -v --cov=src
```

`tests/` contains unit tests for cleaning rules, feature transformations, schema contracts, and a smoke test for the inference API.

## Reproducibility

* Random seeds fixed in `configs/pipeline.yaml`
* All artifacts written to a versioned `models/artifacts/run_<timestamp>/` directory
* `manifest.json` captures git SHA, dependency versions, dataset hash, and metrics

## Author

**Siva Prasath M** — University ID O24MSD110221, M.Sc. (Data Science), Batch July 2024.

## Licence

MIT — see `LICENSE` for full text.
