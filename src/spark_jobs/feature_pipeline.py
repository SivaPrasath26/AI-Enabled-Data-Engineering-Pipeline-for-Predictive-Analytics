"""Distributed feature pipeline — Spark equivalent of features.factory.

Reads Silver-tier parquet, computes the same feature set as the pandas
implementation using PySpark window functions, and writes Gold-tier output.
"""
from __future__ import annotations

from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.window import Window

from ..utils.config import load_config
from ..utils.logging import get_logger
from ..utils.spark import build_spark

log = get_logger("spark.feature_pipeline")


def build_features(spark: SparkSession, silver_dir: Path) -> DataFrame:
    student_info = spark.read.parquet(str(silver_dir / "studentInfo.parquet"))
    reg = spark.read.parquet(str(silver_dir / "studentRegistration.parquet"))
    sasmt = spark.read.parquet(str(silver_dir / "studentAssessment.parquet"))
    asmt = spark.read.parquet(str(silver_dir / "assessments.parquet"))
    svle = spark.read.parquet(str(silver_dir / "studentVle.parquet"))
    vle = spark.read.parquet(str(silver_dir / "vle.parquet"))

    keys = ["id_student", "code_module", "code_presentation"]

    # ---------------- VLE aggregates
    vle_join = svle.join(vle.select("id_site", "activity_type"), on="id_site", how="left")
    vle_agg = vle_join.groupBy(*keys).agg(
        F.sum("sum_click").alias("total_clicks"),
        F.countDistinct("date").alias("active_days"),
        F.avg("sum_click").alias("mean_clicks_per_day"),
        F.max("sum_click").alias("max_clicks_in_day"),
        F.countDistinct("activity_type").alias("n_resource_types"),
    )

    # ---------------- Weekly window for temporal dynamics
    svle_w = svle.filter(F.col("date") >= 0).withColumn(
        "week", (F.col("date") / F.lit(7)).cast("int") + F.lit(1)
    )
    weekly = svle_w.groupBy(*keys, "week").agg(F.sum("sum_click").alias("week_clicks"))
    pivot = weekly.groupBy(*keys).pivot("week", [1, 2, 3, 4]).agg(F.first("week_clicks"))
    pivot = pivot.na.fill(0)
    pivot = pivot.toDF(*keys, "clicks_week_1", "clicks_week_2", "clicks_week_3", "clicks_week_4")

    # ---------------- Assessment aggregates
    join_a = sasmt.join(asmt.select("id_assessment", "date", "code_module", "code_presentation"), on="id_assessment", how="left")
    join_a = join_a.withColumn("late_days", F.col("date_submitted") - F.col("date"))
    join_a = join_a.withColumn("is_late", (F.col("late_days") > 0).cast("int"))
    a_agg = join_a.groupBy(*keys).agg(
        F.count("id_assessment").alias("n_assessments_submitted"),
        F.avg("score").alias("mean_assessment_score"),
        F.stddev("score").alias("std_assessment_score"),
        F.expr("percentile_approx(late_days, 0.5)").alias("median_days_to_submission"),
        F.avg("is_late").alias("late_submission_ratio"),
    )

    out = (
        student_info
        .join(reg, on=keys, how="left")
        .join(a_agg, on=keys, how="left")
        .join(vle_agg, on=keys, how="left")
        .join(pivot, on=keys, how="left")
        .na.fill(0)
    )
    return out


def main(config_path: str = "configs/pipeline.yaml") -> None:
    cfg = load_config(config_path)
    spark = build_spark(cfg.spark)
    silver = Path(cfg.paths.silver)
    gold = Path(cfg.paths.gold)
    gold.mkdir(parents=True, exist_ok=True)
    df = build_features(spark, silver)
    df.write.mode("overwrite").parquet(str(gold / "features.parquet"))
    log.info(f"wrote gold features -> {gold/'features.parquet'}")
    spark.stop()


if __name__ == "__main__":
    main()
