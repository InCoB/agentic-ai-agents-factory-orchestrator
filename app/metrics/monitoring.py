# app/metrics/monitoring.py
import time
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self) -> None:
        self.metrics: Dict[str, float] = {}

    def record(self, name: str, value: float) -> None:
        self.metrics[name] = value
        logger.info(f"Metric recorded: {name} = {value}")

    def get_metrics(self) -> Dict[str, float]:
        return self.metrics

metrics_collector = MetricsCollector()
