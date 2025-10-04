"""Analytic communication estimator (routing/all-reduce/all-to-all)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from core.data import HardwareSpec, LayerExecution
from core.ops import (
    communication_all_reduce,
    communication_all_to_all,
    dominant_latency_ms,
    interconnect_time_ms,
    memory_time_ms,
)

DEFAULT_PAYLOAD_MB = 1.0


@dataclass
class CommunicationConfig:
    pattern: str = "all_to_all"
    payload_mb: float = DEFAULT_PAYLOAD_MB

    @classmethod
    def from_dict(cls, data: Dict) -> "CommunicationConfig":
        return cls(
            pattern=str(data.get("pattern", "all_to_all")),
            payload_mb=float(data.get("payload_mb", DEFAULT_PAYLOAD_MB)),
        )


class Communication:
    def __init__(self, communication_config: Dict, hardware_config: Dict | None = None):
        self.config = CommunicationConfig.from_dict(communication_config or {})
        self.hardware_cfg = hardware_config or {}

    def analytic_flops(self, batch: int, seq: int) -> int:
        # Communication is assumed to be bandwidth bound; no meaningful FLOPs.
        return 0

    def estimate_execution_time(self, batch: int, seq: int, hardware: HardwareSpec) -> LayerExecution:
        payload_bytes = self.config.payload_mb * 1e6
        if self.config.pattern == "all_reduce":
            metric = communication_all_reduce(payload_bytes)
        else:
            metric = communication_all_to_all(payload_bytes)

        interconnect_ms = interconnect_time_ms(metric.bytes_accessed, hardware)
        memory_ms = memory_time_ms(metric.bytes_accessed, hardware)
        latency_ms = dominant_latency_ms(interconnect_ms, memory_ms, hardware)

        breakdown = {metric.name: {"flops": metric.flops, "bytes": metric.bytes_accessed}}
        features = {
            "pattern": 1.0 if self.config.pattern == "all_reduce" else 0.0,
            "payload_mb": float(self.config.payload_mb),
            "batch": float(batch),
            "seq": float(seq),
        }

        return LayerExecution(
            layer_name="communication",
            layer_type="communication",
            flops=0.0,
            bytes_read=metric.bytes_accessed,
            bytes_written=metric.bytes_accessed,
            compute_time_ms=interconnect_ms,
            memory_time_ms=memory_ms,
            dominant_latency_ms=latency_ms,
            estimated_execution_time_ms=latency_ms,
            features=features,
            breakdown=breakdown,
        )

    def mix(self, messages, mask=None):
        raise NotImplementedError("Numeric communication path not implemented for analytic simulator")
