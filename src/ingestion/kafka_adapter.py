"""Kafka ingestion adapter — stub illustrating the streaming path.

In production the adapter consumes LMS events (login, page-view, submission)
from a Kafka topic and lands them in Bronze as micro-batched parquet files.
The stub avoids a hard dependency on a running broker for tests and demos.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd

from ..utils.logging import get_logger
from .base import BaseIngestionAdapter

log = get_logger("ingestion.kafka")


class KafkaIngestionAdapter(BaseIngestionAdapter):
    source_name = "kafka_lms_events"

    def __init__(
        self,
        target_dir: str | Path,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "lms.events",
        batch_seconds: int = 30,
    ) -> None:
        super().__init__(target_dir)
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.batch_seconds = batch_seconds

    def read(self, payload: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(payload)

    def consume(self) -> Iterator[pd.DataFrame]:
        """Yield micro-batches of events. Replace `_stub_messages` with a real
        kafka-python KafkaConsumer in production."""
        log.info(
            f"connecting to brokers={self.bootstrap_servers} topic={self.topic}"
        )
        while True:
            buffer: list[dict] = []
            t_end = time.time() + self.batch_seconds
            for msg in self._stub_messages():
                buffer.append(msg)
                if time.time() >= t_end:
                    break
            if buffer:
                yield self.stamp_provenance(self.read(buffer), source_file=self.topic)
            else:
                return

    def _stub_messages(self) -> Iterator[dict]:
        sample = {
            "id_student": 11391,
            "event_type": "page_view",
            "code_module": "AAA",
            "code_presentation": "2014J",
            "id_site": 546610,
            "ts": datetime.utcnow().isoformat(),
            "payload": json.dumps({"duration": 47}),
        }
        for _ in range(5):
            yield sample
            time.sleep(0.01)
