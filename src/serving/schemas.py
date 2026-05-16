"""Pydantic request / response schemas for the inference API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class StudentRecord(BaseModel):
    id_student: int
    code_module: str
    code_presentation: str
    gender: str
    region: str
    highest_education: str
    imd_band: str | None = None
    age_band: str
    num_of_prev_attempts: int = 0
    studied_credits: int = 0
    disability: str = "N"
    total_clicks: float = 0.0
    active_days: float = 0.0
    mean_clicks_per_day: float = 0.0
    max_clicks_in_day: float = 0.0
    n_assessments_submitted: float = 0.0
    n_assessments_missed: float = 0.0
    mean_assessment_score: float = 0.0
    std_assessment_score: float = 0.0
    median_days_to_submission: float = 0.0
    late_submission_ratio: float = 0.0
    clicks_week_1: float = 0.0
    clicks_week_2: float = 0.0
    clicks_week_3: float = 0.0
    clicks_week_4: float = 0.0
    click_growth_rate: float = 0.0
    days_since_last_activity: float = 0.0
    session_duration_variance: float = 0.0


class PredictRequest(BaseModel):
    record: StudentRecord


class PredictResponse(BaseModel):
    id_student: int
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_label: str
    threshold: float
    model_version: str


class BatchPredictRequest(BaseModel):
    records: list[StudentRecord] = Field(..., max_length=1000)


class FeatureAttribution(BaseModel):
    feature: str
    contribution: float


class ExplainResponse(BaseModel):
    id_student: int
    risk_score: float
    base_value: float
    top_contributions: list[FeatureAttribution]
