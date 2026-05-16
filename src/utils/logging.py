"""Structured logging via loguru with sensible defaults."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_INITIALISED = False


def get_logger(name: str = "edu-pipeline", log_dir: str | Path | None = None):
    global _INITIALISED
    if not _INITIALISED:
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
                   "| <level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            level="INFO",
        )
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            logger.add(
                Path(log_dir) / "pipeline_{time:YYYYMMDD}.log",
                rotation="10 MB",
                retention="14 days",
                level="DEBUG",
            )
        _INITIALISED = True
    return logger.bind(component=name)
