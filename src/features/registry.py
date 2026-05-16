"""Registry of feature groups derivable from the cleaned OULAD tables."""
from __future__ import annotations

FEATURE_GROUPS: dict[str, list[str]] = {
    "demographic": [
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability",
    ],
    "registration": [
        "num_of_prev_attempts",
        "studied_credits",
        "days_to_registration",
        "days_to_unregistration",
    ],
    "assessment_history": [
        "n_assessments_submitted",
        "n_assessments_missed",
        "mean_assessment_score",
        "std_assessment_score",
        "median_days_to_submission",
        "late_submission_ratio",
    ],
    "vle_activity": [
        "total_clicks",
        "active_days",
        "mean_clicks_per_day",
        "max_clicks_in_day",
        "n_resource_types",
    ],
    "temporal_dynamics": [
        "clicks_week_1",
        "clicks_week_2",
        "clicks_week_3",
        "clicks_week_4",
        "click_growth_rate",
        "days_since_last_activity",
        "session_duration_variance",
    ],
}
