"""End-to-end orchestrator — ingest -> clean -> features -> train -> evaluate."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import classification_metrics, fairness_metrics
from src.features.encoders import CATEGORICAL_COLS, apply_one_hot, fit_one_hot
from src.features.factory import FeatureFactory
from src.ingestion.csv_adapter import CSVIngestionAdapter
from src.models import XGBoostTrainer, LightGBMTrainer
from src.preprocessing import Cleaner, deduplicate, load_schema, SchemaValidator
from src.preprocessing.cleaner import encode_final_result
from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger("orchestrator", log_dir="logs")


SCHEMAS = {
    "studentInfo": "configs/schemas/student_info.yaml",
    "studentVle":  "configs/schemas/student_vle.yaml",
    "assessments": "configs/schemas/assessments.yaml",
}


def _ingest(cfg) -> dict[str, Path]:
    adapter = CSVIngestionAdapter(cfg.paths.raw, cfg.paths.bronze)
    return adapter.ingest(cfg.ingestion["files"])


def _validate_and_clean(cfg) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    cleaner = Cleaner(critical_cols=["id_student", "code_module", "code_presentation"])
    for fname in cfg.ingestion["files"]:
        name = Path(fname).stem
        df = pd.read_parquet(Path(cfg.paths.bronze) / f"{name}.parquet")
        if name in SCHEMAS:
            schema = load_schema(SCHEMAS[name])
            df = SchemaValidator(schema).validate(df)
        df = cleaner.apply(df)
        keys = cfg.preprocessing.get("dedup_keys", {}).get(name)
        if keys:
            df = deduplicate(df, keys)
        Path(cfg.paths.silver).mkdir(parents=True, exist_ok=True)
        df.to_parquet(Path(cfg.paths.silver) / f"{name}.parquet", index=False)
        out[name] = df
    return out


def _build_features(cfg, silver: dict[str, pd.DataFrame]) -> pd.DataFrame:
    factory = FeatureFactory(target_horizon_weeks=cfg.features["target_horizon_weeks"])
    si = encode_final_result(silver["studentInfo"])
    features = factory.build(
        student_info=si,
        registration=silver["studentRegistration"],
        assessments=silver["assessments"],
        student_assessment=silver["studentAssessment"],
        vle=silver["vle"],
        student_vle=silver["studentVle"],
    )
    Path(cfg.paths.gold).mkdir(parents=True, exist_ok=True)
    features.to_parquet(Path(cfg.paths.gold) / "features.parquet", index=False)
    return features


def _train(cfg, features: pd.DataFrame) -> Path:
    y = features["final_result_binary"].astype(int)
    drop_cols = ["final_result_binary", "id_student"]
    X = features.drop(columns=[c for c in drop_cols if c in features.columns])
    cat = [c for c in CATEGORICAL_COLS if c in X.columns]
    enc = fit_one_hot(X, cat)
    X = apply_one_hot(X, enc, cat)
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.training.test_size,
        random_state=cfg.seed,
        stratify=y,
    )

    trainer = XGBoostTrainer(cfg.models["xgboost"], seed=cfg.seed)
    trainer.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    scores = trainer.predict_proba(X_test)
    metrics = classification_metrics(y_test.to_numpy(), scores)
    log.info(f"  test metrics: {metrics}")

    run_dir = Path(cfg.paths.artifacts) / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = trainer.save(run_dir)
    manifest = artifact.manifest()
    manifest["metrics"] = metrics
    manifest["threshold"] = 0.5
    manifest["dataset_hash"] = hashlib.sha1(pd.util.hash_pandas_object(X_train).values.tobytes()).hexdigest()[:12]
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # latest symlink (copy on Windows)
    latest = Path(cfg.paths.artifacts) / "latest"
    if latest.exists():
        shutil.rmtree(latest, ignore_errors=True)
    shutil.copytree(run_dir, latest)
    log.info(f"  promoted -> {latest}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["ingest", "clean", "features", "train"],
        choices=["ingest", "clean", "features", "train"],
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    np.random.seed(cfg.seed)

    if "ingest" in args.stages:
        log.info("=== STAGE 1: INGEST ===")
        _ingest(cfg)
    silver: dict[str, pd.DataFrame] = {}
    if "clean" in args.stages:
        log.info("=== STAGE 2: VALIDATE + CLEAN ===")
        silver = _validate_and_clean(cfg)
    elif "features" in args.stages:
        for fname in cfg.ingestion["files"]:
            name = Path(fname).stem
            silver[name] = pd.read_parquet(Path(cfg.paths.silver) / f"{name}.parquet")
    features = None
    if "features" in args.stages:
        log.info("=== STAGE 3: FEATURES ===")
        features = _build_features(cfg, silver)
    elif "train" in args.stages:
        features = pd.read_parquet(Path(cfg.paths.gold) / "features.parquet")
    if "train" in args.stages and features is not None:
        log.info("=== STAGE 4: TRAIN ===")
        _train(cfg, features)
    log.info("pipeline complete")


if __name__ == "__main__":
    main()
