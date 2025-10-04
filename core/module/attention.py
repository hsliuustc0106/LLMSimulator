"""Analytic attention estimator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from core.data import HardwareSpec, LayerExecution
from core.ops import (
    FusionMetrics,
    attention_output_projection,
    attention_qkv_projections,
    attention_scores,
    attention_weighted_sum,
    compute_time_ms,
    dominant_latency_ms,
    memory_time_ms,
    tensor_bytes,
)

DEFAULT_DTYPE_BITS = 16


@dataclass
class AttentionConfig:
    d_model: int = 768
    num_heads: int = 8
    head_dim: int | None = None
    dtype_bits: int = DEFAULT_DTYPE_BITS

    @classmethod
    def from_dict(cls, data: Dict) -> "AttentionConfig":
        return cls(
            d_model=int(data.get("d_model", 768)),
            num_heads=int(data.get("num_attention_heads", data.get("num_heads", 8))),
            head_dim=data.get("head_dim"),
            dtype_bits=int(data.get("dtype_bits", DEFAULT_DTYPE_BITS)),
        )

    @property
    def resolved_head_dim(self) -> int:
        if self.head_dim is not None:
            return int(self.head_dim)
        return self.d_model // max(self.num_heads, 1)

    @property
    def qkv_dim(self) -> int:
        return self.num_heads * self.resolved_head_dim


class Attention:
    def __init__(self, attn_config: Dict, hardware_config: Dict | None = None):
        self.config = AttentionConfig.from_dict(attn_config or {})
        self.hardware_cfg = hardware_config or {}

    def _metrics(self, batch: int, seq: int) -> Tuple[FusionMetrics, ...]:
        cfg = self.config
        qkv = attention_qkv_projections(batch, seq, cfg.d_model, cfg.qkv_dim, dtype_bits=cfg.dtype_bits)
        scores = attention_scores(batch, seq, cfg.num_heads, cfg.resolved_head_dim, dtype_bits=cfg.dtype_bits)
        weighted = attention_weighted_sum(batch, seq, cfg.num_heads, cfg.resolved_head_dim, dtype_bits=cfg.dtype_bits)
        out_proj = attention_output_projection(batch, seq, cfg.d_model, cfg.qkv_dim, dtype_bits=cfg.dtype_bits)
        return qkv, scores, weighted, out_proj

    def analytic_flops(self, batch: int, seq: int) -> float:
        return sum(metric.flops for metric in self._metrics(batch, seq))

    def estimate_execution_time(self, batch: int, seq: int, hardware: HardwareSpec) -> LayerExecution:
        metrics = self._metrics(batch, seq)
        total_flops = sum(m.flops for m in metrics)
        total_bytes = sum(m.bytes_accessed for m in metrics)
        output_bytes = tensor_bytes((batch, seq, self.config.d_model), self.config.dtype_bits)

        compute_ms = compute_time_ms(total_flops, hardware)
        memory_ms = memory_time_ms(total_bytes + output_bytes, hardware)
        latency_ms = dominant_latency_ms(compute_ms, memory_ms, hardware)

        breakdown = {m.name: {"flops": m.flops, "bytes": m.bytes_accessed} for m in metrics}
        features = {
            "d_model": float(self.config.d_model),
            "num_heads": float(self.config.num_heads),
            "batch": float(batch),
            "seq": float(seq),
            "dtype_bits": float(self.config.dtype_bits),
        }

        return LayerExecution(
            layer_name="attention",
            layer_type="attention",
            flops=total_flops,
            bytes_read=total_bytes,
            bytes_written=output_bytes,
            compute_time_ms=compute_ms,
            memory_time_ms=memory_ms,
            dominant_latency_ms=latency_ms,
            estimated_execution_time_ms=latency_ms,
            features=features,
            breakdown=breakdown,
        )

    def forward(self, q, k, v, mask=None):
        raise NotImplementedError("Numeric forward not implemented for analytic simulator")
