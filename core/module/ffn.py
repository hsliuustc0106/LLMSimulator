"""Analytic FFN estimator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from core.data import HardwareSpec, LayerExecution
from core.ops import (
    FusionMetrics,
    compute_time_ms,
    dominant_latency_ms,
    ffn_activation,
    memory_time_ms,
    tensor_bytes,
)

DEFAULT_DTYPE_BITS = 16


@dataclass
class FFNConfig:
    d_model: int = 768
    d_ff: int = 3072
    dtype_bits: int = DEFAULT_DTYPE_BITS

    @classmethod
    def from_dict(cls, data: Dict) -> "FFNConfig":
        return cls(
            d_model=int(data.get("d_model", 768)),
            d_ff=int(data.get("d_ff", data.get("intermediate_size", 3072))),
            dtype_bits=int(data.get("dtype_bits", DEFAULT_DTYPE_BITS)),
        )


class FFN:
    def __init__(self, ffn_config: Dict, hardware_config: Dict | None = None):
        self.config = FFNConfig.from_dict(ffn_config or {})
        self.hardware_cfg = hardware_config or {}

    def _metrics(self, batch: int, seq: int) -> FusionMetrics:
        cfg = self.config
        return ffn_activation(batch, seq, cfg.d_model, cfg.d_ff, dtype_bits=cfg.dtype_bits)

    def analytic_flops(self, batch: int, seq: int) -> float:
        return self._metrics(batch, seq).flops

    def estimate_execution_time(self, batch: int, seq: int, hardware: HardwareSpec) -> LayerExecution:
        metric = self._metrics(batch, seq)
        total_flops = metric.flops
        bytes_accessed = metric.bytes_accessed
        output_bytes = tensor_bytes((batch, seq, self.config.d_model), self.config.dtype_bits)

        compute_ms = compute_time_ms(total_flops, hardware)
        memory_ms = memory_time_ms(bytes_accessed + output_bytes, hardware)
        latency_ms = dominant_latency_ms(compute_ms, memory_ms, hardware)

        features = {
            "d_model": float(self.config.d_model),
            "d_ff": float(self.config.d_ff),
            "batch": float(batch),
            "seq": float(seq),
            "dtype_bits": float(self.config.dtype_bits),
        }

        breakdown = {metric.name: {"flops": metric.flops, "bytes": metric.bytes_accessed}}

        return LayerExecution(
            layer_name="ffn",
            layer_type="ffn",
            flops=total_flops,
            bytes_read=bytes_accessed,
            bytes_written=output_bytes,
            compute_time_ms=compute_ms,
            memory_time_ms=memory_ms,
            dominant_latency_ms=latency_ms,
            estimated_execution_time_ms=latency_ms,
            features=features,
            breakdown=breakdown,
        )

    def forward(self, x):
        raise NotImplementedError("Numeric forward not implemented for analytic simulator")
