"""Feature factory — Spark + pandas implementations.

The factory exposes one `build` method that, given the Silver-tier tables,
produces a Gold-tier feature matrix keyed by (id_student, code_module,
code_presentation). The implementation favours pandas for portability;
the Spark equivalents in ``src/spark_jobs/feature_pipeline.py`` are used
for production-scale runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .registry import FEATURE_GROUPS

log = get_logger("features.factory")


@dataclass
class FeatureFactory:
    target_horizon_weeks: int = 4

    def build(
        self,
        student_info: pd.DataFrame,
        registration: pd.DataFrame,
        assessments: pd.DataFrame,
        student_assessment: pd.DataFrame,
        vle: pd.DataFrame,
        student_vle: pd.DataFrame,
    ) -> pd.DataFrame:
        log.info("Building Gold-tier feature matrix")

        base = self._base(student_info)
        reg = self._registration_features(registration)
        asmt = self._assessment_features(assessments, student_assessment)
        click = self._vle_features(student_vle, vle)
        temporal = self._temporal_features(student_vle)

        keys = ["id_student", "code_module", "code_presentation"]
        out = base.merge(reg, on=keys, how="left")
        out = out.merge(asmt, on=keys, how="left")
        out = out.merge(click, on=keys, how="left")
        out = out.merge(temporal, on=keys, how="left")

        out = self._fill_zero_for_counts(out)
        log.info(f"  feature matrix shape={out.shape}")
        return out

    # ------------------------------------------------------------- base
    def _base(self, student_info: pd.DataFrame) -> pd.DataFrame:
        cols = (
            ["id_student", "code_module", "code_presentation"]
            + FEATURE_GROUPS["demographic"]
            + ["num_of_prev_attempts", "studied_credits"]
        )
        existing = [c for c in cols if c in student_info.columns]
        out = student_info[existing].copy()
        if "final_result_binary" in student_info.columns:
            out["final_result_binary"] = student_info["final_result_binary"]
        return out

    # ----------------------------------------------------- registration
    def _registration_features(self, reg: pd.DataFrame) -> pd.DataFrame:
        cols = ["id_student", "code_module", "code_presentation"]
        out = reg[cols].drop_duplicates().copy()
        if "date_registration" in reg.columns:
            out["days_to_registration"] = reg["date_registration"]
        if "date_unregistration" in reg.columns:
            out["days_to_unregistration"] = reg["date_unregistration"]
        return out

    # ------------------------------------------------------ assessments
    def _assessment_features(
        self, asmt: pd.DataFrame, sasmt: pd.DataFrame
    ) -> pd.DataFrame:
        joined = sasmt.merge(asmt, on="id_assessment", how="left")
        joined["late_days"] = joined["date_submitted"] - joined["date"]
        joined["is_late"] = (joined["late_days"] > 0).astype(int)
        agg = (
            joined.groupby(["id_student", "code_module", "code_presentation"])
            .agg(
                n_assessments_submitted=("id_assessment", "count"),
                mean_assessment_score=("score", "mean"),
                std_assessment_score=("score", "std"),
                median_days_to_submission=("late_days", "median"),
                late_submission_ratio=("is_late", "mean"),
            )
            .reset_index()
        )
        # Missed assessments require knowledge of expected counts.
        expected = (
            asmt.groupby(["code_module", "code_presentation"])
            .agg(expected=("id_assessment", "count"))
            .reset_index()
        )
        agg = agg.merge(expected, on=["code_module", "code_presentation"], how="left")
        agg["n_assessments_missed"] = (
            agg["expected"] - agg["n_assessments_submitted"]
        ).clip(lower=0)
        return agg.drop(columns=["expected"])

    # -------------------------------------------------------- vle stats
    def _vle_features(self, svle: pd.DataFrame, vle: pd.DataFrame) -> pd.DataFrame:
        merged = svle.merge(
            vle[["id_site", "activity_type"]] if "activity_type" in vle.columns else vle,
            on="id_site",
            how="left",
        )
        agg = (
            merged.groupby(["id_student", "code_module", "code_presentation"])
            .agg(
                total_clicks=("sum_click", "sum"),
                active_days=("date", "nunique"),
                mean_clicks_per_day=("sum_click", "mean"),
                max_clicks_in_day=("sum_click", "max"),
                n_resource_types=(
                    "activity_type" if "activity_type" in merged.columns else "id_site",
                    "nunique",
                ),
            )
            .reset_index()
        )
        return agg

    # ----------------------------------------------------- temporal dyn
    def _temporal_features(self, svle: pd.DataFrame) -> pd.DataFrame:
        df = svle.copy()
        df = df[df["date"] >= 0]  # ignore pre-start exploration days
        df["week"] = (df["date"] // 7).astype(int) + 1
        weekly = (
            df.groupby(
                ["id_student", "code_module", "code_presentation", "week"]
            )["sum_click"]
            .sum()
            .reset_index()
        )
        pivot = weekly.pivot_table(
            index=["id_student", "code_module", "code_presentation"],
            columns="week",
            values="sum_click",
            fill_value=0,
        ).reset_index()
        pivot.columns.name = None
        for w in range(1, self.target_horizon_weeks + 1):
            pivot[f"clicks_week_{w}"] = pivot.get(w, 0)
        # growth = (w_t - w_t-1) / w_t-1, averaged across the horizon
        deltas = []
        for w in range(2, self.target_horizon_weeks + 1):
            prev = pivot[f"clicks_week_{w-1}"].replace(0, np.nan)
            deltas.append((pivot[f"clicks_week_{w}"] - prev) / prev)
        pivot["click_growth_rate"] = pd.concat(deltas, axis=1).mean(axis=1)

        # days since last activity (relative to the analysis cut-off = day 28)
        last_day = (
            df.groupby(["id_student", "code_module", "code_presentation"])["date"]
            .max()
            .reset_index()
            .rename(columns={"date": "_last_day"})
        )
        pivot = pivot.merge(
            last_day,
            on=["id_student", "code_module", "code_presentation"],
            how="left",
        )
        pivot["days_since_last_activity"] = (
            self.target_horizon_weeks * 7 - pivot["_last_day"]
        ).clip(lower=0)
        pivot = pivot.drop(columns=["_last_day"])

        # session variance proxy: per-day click std
        var = (
            df.groupby(["id_student", "code_module", "code_presentation"])["sum_click"]
            .std()
            .reset_index()
            .rename(columns={"sum_click": "session_duration_variance"})
        )
        pivot = pivot.merge(
            var,
            on=["id_student", "code_module", "code_presentation"],
            how="left",
        )

        keep_cols = [
            "id_student",
            "code_module",
            "code_presentation",
            *[f"clicks_week_{w}" for w in range(1, self.target_horizon_weeks + 1)],
            "click_growth_rate",
            "days_since_last_activity",
            "session_duration_variance",
        ]
        return pivot[keep_cols]

    # ------------------------------------------------------------ misc
    @staticmethod
    def _fill_zero_for_counts(df: pd.DataFrame) -> pd.DataFrame:
        count_cols = [c for c in df.columns if c.startswith("n_") or c.startswith("clicks_") or c.startswith("total_")]
        for c in count_cols:
            df[c] = df[c].fillna(0)
        return df
