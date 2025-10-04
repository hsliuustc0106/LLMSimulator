"""Shared dataclasses describing hardware, runtime, and per-layer execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class HardwareSpec:
    """Hardware capabilities used to convert analytic work into timing."""

    name: str
    peak_tflops: float
    memory_bandwidth_gbps: float
    hbm_gb: float
    interconnect_gbps: float
    max_concurrency: int = 1
    overlap_efficiency: float = 1.0

    def compute_throughput_tflops(self) -> float:
        return max(self.peak_tflops, 0.0)

    def memory_bandwidth_bytes(self) -> float:
        return max(self.memory_bandwidth_gbps, 0.0) * 1e9

    def interconnect_bandwidth_bytes(self) -> float:
        return max(self.interconnect_gbps, 0.0) * 1e9

    def effective_overlap(self) -> float:
        return max(self.overlap_efficiency, 1e-3)


@dataclass(frozen=True)
class RuntimeSpec:
    """Runtime overrides supplied through the CLI."""

    batch_size: int
    seq_len: int
    micro_batch: Optional[int] = None
    tokens_per_expert: Optional[float] = None


@dataclass
class BaseLayerConfig:
    """Common metadata shared by all layer configs."""

    layer_type: str
    name: str
    layer_id: int
    attn_config: Dict[str, Any] = field(default_factory=dict)
    fused_ops: List[str] = field(default_factory=list)


@dataclass
class FFNLayerConfig(BaseLayerConfig):
    ffn_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MoELayerConfig(BaseLayerConfig):
    moe_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommunicationLayerConfig(BaseLayerConfig):
    comm_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerExecution:
    """Structured result emitted by analytic/ML estimators."""

    layer_name: str
    layer_type: str
    flops: float
    bytes_read: float
    bytes_written: float
    compute_time_ms: float
    memory_time_ms: float
    dominant_latency_ms: float
    estimated_execution_time_ms: float
    features: Dict[str, float] = field(default_factory=dict)
    breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "layer_type": self.layer_type,
            "flops": self.flops,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "compute_time_ms": self.compute_time_ms,
            "memory_time_ms": self.memory_time_ms,
            "dominant_latency_ms": self.dominant_latency_ms,
            "estimated_execution_time_ms": self.estimated_execution_time_ms,
            "features": dict(self.features),
            "breakdown": dict(self.breakdown),
        }


@dataclass
class SimulationResult:
    """Aggregate output for an end-to-end simulation run."""

    layers: List[LayerExecution]
    total_flops: float
    total_latency_ms: float
    peak_memory_bytes: float
    bottleneck_layer: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layers": [layer.to_dict() for layer in self.layers],
            "total_flops": self.total_flops,
            "total_latency_ms": self.total_latency_ms,
            "peak_memory_bytes": self.peak_memory_bytes,
            "bottleneck_layer": self.bottleneck_layer,
        }
