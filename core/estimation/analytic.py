"""Analytic estimator backend that wraps layer modules."""
from __future__ import annotations

from typing import List

from core.data import (
    BaseLayerConfig,
    CommunicationLayerConfig,
    FFNLayerConfig,
    HardwareSpec,
    LayerExecution,
    MoELayerConfig,
    RuntimeSpec,
)
from core.module import Attention, Communication, FFN, MoE


class AnalyticEstimator:
    """Estimate per-layer execution using deterministic fused-op formulas."""

    def __init__(self, hardware: HardwareSpec, runtime: RuntimeSpec):
        self.hardware = hardware
        self.runtime = runtime

    def estimate_layer(self, layer_config: BaseLayerConfig) -> LayerExecution:
        batch = self.runtime.batch_size
        seq = self.runtime.seq_len

        if isinstance(layer_config, FFNLayerConfig):
            module = FFN(layer_config.ffn_config)
            execution = module.estimate_execution_time(batch, seq, self.hardware)
        elif isinstance(layer_config, MoELayerConfig):
            module = MoE(layer_config.moe_config)
            execution = module.estimate_execution_time(batch, seq, self.hardware)
        elif isinstance(layer_config, CommunicationLayerConfig):
            module = Communication(layer_config.comm_config)
            execution = module.estimate_execution_time(batch, seq, self.hardware)
        else:
            module = Attention(layer_config.attn_config)
            execution = module.estimate_execution_time(batch, seq, self.hardware)

        execution.layer_name = layer_config.name
        execution.layer_type = layer_config.layer_type
        execution.features.setdefault("layer_id", float(layer_config.layer_id))
        execution.features.setdefault("layer_type", 0.0)
        return execution

    def estimate_layers(self, layer_configs: List[BaseLayerConfig]) -> List[LayerExecution]:
        return [self.estimate_layer(config) for config in layer_configs]
