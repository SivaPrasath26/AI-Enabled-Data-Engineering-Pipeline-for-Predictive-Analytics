# Architecture Decision Record

## ADR-001 Medallion architecture

We adopt a three-tier (Bronze / Silver / Gold) lake layout. Bronze captures
raw source records with provenance metadata; Silver enforces schemas and
quality rules; Gold exposes the model-ready feature matrix. This separation
lets us reprocess any stage without losing source fidelity.

## ADR-002 Spark for feature engineering

The pandas implementation is sufficient up to the OULAD-scale (~10 M VLE
events) but the production pipeline must scale to multi-institutional data.
We mirror every transformation in `src/spark_jobs/feature_pipeline.py` to
keep production parity with the pandas reference.

## ADR-003 XGBoost as primary tabular model

Gradient-boosted decision trees consistently lead the OULAD leaderboard;
they are interpretable via SHAP and fast to train. XGBoost is the default;
LightGBM is offered as an alternative.

## ADR-004 LSTM for sequential signals

To capture week-over-week engagement dynamics that tabular aggregates lose,
we train an LSTM on weekly click sequences. The two heads are combined in a
late-fusion stacker for the final risk score.

## ADR-005 SHAP for explainability

SHAP TreeExplainer produces deterministic, per-prediction attributions
suitable for surfacing in the educator dashboard.
