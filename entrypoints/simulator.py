
"""Scenario loading, simulation orchestration, and reporting helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union

from core.data import (
    BaseLayerConfig,
    HardwareSpec,
    LayerExecution,
    RuntimeSpec,
    SimulationResult,
)
from core.estimation import AnalyticEstimator

from .utils import (
    hardware_from_dict,
    layer_config_from_dict,
    maybe_load_reference,
    read_yaml,
)


@dataclass
class Scenario:
    name: str
    hardware: HardwareSpec
    layers: List[BaseLayerConfig]


def load_scenario(path: Union[str, Path]) -> Scenario:
    scenario_path = Path(path).resolve()
    data = read_yaml(scenario_path)
    base_dir = scenario_path.parent

    hardware_ref = data.get("hardware")
    if hardware_ref is None:
        raise ValueError("Scenario must specify a 'hardware' block or reference")
    hardware_dict = maybe_load_reference(base_dir, hardware_ref)
    hardware = hardware_from_dict(hardware_dict)

    layers_block: Iterable[Dict] = data.get("layers", [])
    if not layers_block:
        raise ValueError("Scenario must include at least one layer entry")

    layer_configs: List[BaseLayerConfig] = []
    for idx, layer_entry in enumerate(layers_block):
        if not isinstance(layer_entry, dict):
            raise ValueError(f"Layer entry #{idx} must be a mapping")
        layer_type = layer_entry.get("type")
        if "config" in layer_entry:
            config_dict = maybe_load_reference(base_dir, layer_entry["config"])
        else:
            config_dict = layer_entry
        merged = dict(config_dict)
        overrides = layer_entry.get("overrides", {})
        if overrides:
            merged.update(overrides)
        merged.setdefault("name", layer_entry.get("name"))
        layer_config = layer_config_from_dict(idx, layer_type, merged)
        layer_configs.append(layer_config)

    scenario_name = data.get("name", scenario_path.stem)
    return Scenario(name=scenario_name, hardware=hardware, layers=layer_configs)


def run_simulation(
    scenario: Scenario,
    runtime: RuntimeSpec,
    *,
    estimator: AnalyticEstimator | None = None,
) -> SimulationResult:
    active_estimator = estimator or AnalyticEstimator(scenario.hardware, runtime)
    executions: List[LayerExecution] = active_estimator.estimate_layers(scenario.layers)

    total_flops = sum(exec.flops for exec in executions)
    total_latency = sum(exec.dominant_latency_ms for exec in executions)
    peak_memory = max((exec.bytes_read + exec.bytes_written) for exec in executions)
    bottleneck_layer = (
        max(executions, key=lambda e: e.dominant_latency_ms).layer_name if executions else None
    )

    return SimulationResult(
        layers=executions,
        total_flops=total_flops,
        total_latency_ms=total_latency,
        peak_memory_bytes=peak_memory,
        bottleneck_layer=bottleneck_layer,
    )


def layer_table(result: SimulationResult) -> List[dict]:
    table: List[dict] = []
    for layer in result.layers:
        table.append(
            {
                "layer": layer.layer_name,
                "type": layer.layer_type,
                "gflops": layer.flops / 1e9,
                "compute_ms": layer.compute_time_ms,
                "memory_ms": layer.memory_time_ms,
                "latency_ms": layer.dominant_latency_ms,
                "bytes_gb": (layer.bytes_read + layer.bytes_written) / 1e9,
            }
        )
    return table


def summary_row(result: SimulationResult) -> dict:
    return {
        "total_latency_ms": result.total_latency_ms,
        "total_flops_g": result.total_flops / 1e9,
        "peak_memory_gb": result.peak_memory_bytes / 1e9,
        "bottleneck_layer": result.bottleneck_layer,
    }
