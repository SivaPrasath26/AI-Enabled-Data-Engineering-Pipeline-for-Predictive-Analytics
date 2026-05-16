"""FastAPI inference application."""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from ..utils.logging import get_logger
from .schemas import (
    BatchPredictRequest,
    ExplainResponse,
    FeatureAttribution,
    PredictRequest,
    PredictResponse,
    StudentRecord,
)

log = get_logger("serving.app")
app = FastAPI(
    title="Educational Risk Prediction API",
    description="Predict at-risk students and explain predictions with SHAP.",
    version="1.0.0",
)


class ModelBundle:
    """In-memory wrapper around the loaded model and SHAP explainer."""

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        with (self.artifact_dir / "manifest.json").open("r", encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        with (self.artifact_dir / "model.pkl").open("rb") as fh:
            self.model = pickle.load(fh)
        self.feature_columns: list[str] = self.manifest["feature_columns"]
        self.threshold: float = float(self.manifest.get("threshold", 0.5))
        self.version: str = self.manifest.get("run_id", "unknown")
        self._explainer: Any = None
        log.info(f"loaded model={self.manifest['model_name']} version={self.version}")

    def predict_proba(self, df: pd.DataFrame):
        try:
            import xgboost as xgb
            if hasattr(self.model, "predict") and isinstance(self.model, xgb.Booster):
                return self.model.predict(xgb.DMatrix(df[self.feature_columns]))
        except ImportError:
            pass
        return self.model.predict(df[self.feature_columns])

    def explainer(self):
        if self._explainer is None:
            from ..explainability import ShapExplainer
            self._explainer = ShapExplainer(self.model, self.feature_columns)
        return self._explainer


def _record_to_frame(record: StudentRecord, columns: list[str]) -> pd.DataFrame:
    row = record.model_dump()
    df = pd.DataFrame([row])
    for col in columns:
        if col not in df.columns:
            df[col] = 0.0
    return df[columns]


@app.on_event("startup")
def _load_model() -> None:
    artifact_dir = Path(os.environ.get("MODEL_DIR", "models/artifacts/latest"))
    if not artifact_dir.exists():
        log.warning(f"model dir {artifact_dir} not present — API will fail until trained")
        app.state.bundle = None
        return
    app.state.bundle = ModelBundle(artifact_dir)


@app.get("/healthz")
def health() -> dict[str, str]:
    bundle = getattr(app.state, "bundle", None)
    return {"status": "ok", "model_loaded": str(bundle is not None)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    bundle = _ensure_bundle()
    df = _record_to_frame(req.record, bundle.feature_columns)
    score = float(bundle.predict_proba(df)[0])
    label = "at_risk" if score >= bundle.threshold else "on_track"
    return PredictResponse(
        id_student=req.record.id_student,
        risk_score=score,
        risk_label=label,
        threshold=bundle.threshold,
        model_version=bundle.version,
    )


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(req: BatchPredictRequest) -> list[PredictResponse]:
    bundle = _ensure_bundle()
    rows = [r.model_dump() for r in req.records]
    df = pd.DataFrame(rows)
    for col in bundle.feature_columns:
        if col not in df.columns:
            df[col] = 0.0
    df = df[bundle.feature_columns]
    scores = bundle.predict_proba(df)
    out = []
    for r, s in zip(req.records, scores):
        out.append(
            PredictResponse(
                id_student=r.id_student,
                risk_score=float(s),
                risk_label="at_risk" if s >= bundle.threshold else "on_track",
                threshold=bundle.threshold,
                model_version=bundle.version,
            )
        )
    return out


@app.post("/explain", response_model=ExplainResponse)
def explain(req: PredictRequest) -> ExplainResponse:
    bundle = _ensure_bundle()
    df = _record_to_frame(req.record, bundle.feature_columns)
    score = float(bundle.predict_proba(df)[0])
    expl = bundle.explainer().explain_one(df, prediction=score)
    return ExplainResponse(
        id_student=req.record.id_student,
        risk_score=score,
        base_value=expl.base_value,
        top_contributions=[
            FeatureAttribution(feature=f, contribution=v) for f, v in expl.top(10)
        ],
    )


def _ensure_bundle() -> ModelBundle:
    bundle = getattr(app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="model artifact not loaded")
    return bundle
