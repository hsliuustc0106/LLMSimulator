"""Analytic MoE estimator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from core.data import HardwareSpec, LayerExecution
from core.ops import (
    FusionMetrics,
    communication_all_to_all,
    compute_time_ms,
    dominant_latency_ms,
    memory_time_ms,
    moe_expert_forward,
    moe_routing,
    tensor_bytes,
)

DEFAULT_DTYPE_BITS = 16


@dataclass
class MoEConfig:
    d_model: int = 768
    expert_hidden: int = 3072
    num_experts: int = 1
    top_k: int = 1
    avg_experts_per_token: float = 1.0
    num_groups: int = 1
    dtype_bits: int = DEFAULT_DTYPE_BITS

    @classmethod
    def from_dict(cls, data: Dict) -> "MoEConfig":
        d_model = int(data.get("d_model", data.get("model_dim", 768)))
        hidden = int(data.get("moe_intermediate_size", data.get("d_ff", 3072)))
        num_experts = int(data.get("n_routed_experts", data.get("num_experts", 1)))
        top_k = int(data.get("topk_group", data.get("top_k", data.get("num_experts_per_tok", 1))))
        avg_experts = float(data.get("num_experts_per_tok", top_k))
        num_groups = int(data.get("n_group", data.get("num_groups", 1)))
        dtype_bits = int(data.get("dtype_bits", DEFAULT_DTYPE_BITS))
        return cls(
            d_model=d_model,
            expert_hidden=hidden,
            num_experts=max(num_experts, 1),
            top_k=max(top_k, 1),
            avg_experts_per_token=max(avg_experts, 1.0),
            num_groups=max(num_groups, 1),
            dtype_bits=dtype_bits,
        )


class MoE:
    def __init__(self, moe_config: Dict, hardware_config: Dict | None = None):
        self.config = MoEConfig.from_dict(moe_config or {})
        self.hardware_cfg = hardware_config or {}

    def _metrics(self, batch: int, seq: int) -> Tuple[FusionMetrics, ...]:
        cfg = self.config
        tokens = batch * seq
        routing = moe_routing(batch, seq, cfg.num_experts, cfg.top_k, dtype_bits=cfg.dtype_bits)
        active_tokens = int(tokens * cfg.avg_experts_per_token)
        expert = moe_expert_forward(active_tokens, cfg.d_model, cfg.expert_hidden, dtype_bits=cfg.dtype_bits)
        bytes_per_device = tensor_bytes((active_tokens, cfg.d_model), cfg.dtype_bits) / cfg.num_groups
        comm = communication_all_to_all(bytes_per_device)
        return routing, expert, comm

    def analytic_flops(self, batch: int, seq: int) -> float:
        return sum(metric.flops for metric in self._metrics(batch, seq))

    def estimate_execution_time(self, batch: int, seq: int, hardware: HardwareSpec) -> LayerExecution:
        metrics = self._metrics(batch, seq)
        total_flops = sum(m.flops for m in metrics)
        bytes_accessed = sum(m.bytes_accessed for m in metrics)
        output_bytes = tensor_bytes((batch, seq, self.config.d_model), self.config.dtype_bits)

        compute_ms = compute_time_ms(total_flops, hardware)
        memory_ms = memory_time_ms(bytes_accessed + output_bytes, hardware)
        latency_ms = dominant_latency_ms(compute_ms, memory_ms, hardware)

        breakdown = {m.name: {"flops": m.flops, "bytes": m.bytes_accessed} for m in metrics}
        features = {
            "d_model": float(self.config.d_model),
            "expert_hidden": float(self.config.expert_hidden),
            "num_experts": float(self.config.num_experts),
            "top_k": float(self.config.top_k),
            "avg_experts_per_token": float(self.config.avg_experts_per_token),
            "batch": float(batch),
            "seq": float(seq),
        }

        return LayerExecution(
            layer_name="moe",
            layer_type="moe",
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
