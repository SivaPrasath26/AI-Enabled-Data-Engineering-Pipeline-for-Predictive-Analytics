"""SparkSession factory."""
from __future__ import annotations

from pyspark.sql import SparkSession

from .config import SparkConfig


def build_spark(cfg: SparkConfig) -> SparkSession:
    builder = (
        SparkSession.builder
        .appName(cfg.app_name)
        .master(cfg.master)
        .config("spark.driver.memory", cfg.driver_memory)
        .config("spark.executor.memory", cfg.executor_memory)
        .config("spark.sql.shuffle.partitions", cfg.shuffle_partitions)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )
    return builder.getOrCreate()
