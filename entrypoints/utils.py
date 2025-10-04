"""Shared utilities for entrypoint modules."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import yaml

from core.data import (
    BaseLayerConfig,
    CommunicationLayerConfig,
    FFNLayerConfig,
    HardwareSpec,
    MoELayerConfig,
)

SUPPORTED_LAYER_TYPES = {
    "attention": "attention",
    "attention_layer": "attention",
    "ffn_layer": "ffn",
    "ffn": "ffn",
    "moe_layer": "moe",
    "moe": "moe",
    "communication": "communication",
}


def read_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must be a mapping")
    return data


def maybe_load_reference(base_dir: Path, value: Union[str, Dict]) -> Dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ValueError(f"Unsupported reference value: {value!r}")
    ref_path = (base_dir / value).resolve()
    return read_yaml(ref_path)


def hardware_from_dict(data: Dict) -> HardwareSpec:
    required = [
        "name",
        "peak_tflops",
        "memory_bandwidth_gbps",
        "hbm_gb",
        "interconnect_gbps",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Hardware config missing keys: {missing}")
    return HardwareSpec(
        name=str(data["name"]),
        peak_tflops=float(data["peak_tflops"]),
        memory_bandwidth_gbps=float(data["memory_bandwidth_gbps"]),
        hbm_gb=float(data["hbm_gb"]),
        interconnect_gbps=float(data["interconnect_gbps"]),
        max_concurrency=int(data.get("max_concurrency", 1)),
        overlap_efficiency=float(data.get("overlap_efficiency", 1.0)),
    )


def layer_config_from_dict(idx: int, layer_type: str, data: Dict) -> BaseLayerConfig:
    canonical_type = SUPPORTED_LAYER_TYPES.get(layer_type)
    if canonical_type is None:
        raise ValueError(f"Unsupported layer type: {layer_type}")

    name = data.get("name") or f"{canonical_type}_{idx}"
    attn_config = data.get("attn_config", {})

    if canonical_type == "ffn":
        ffn_config = data.get("ffn_config", {})
        return FFNLayerConfig(
            layer_type="ffn",
            name=name,
            layer_id=idx,
            attn_config=attn_config,
            ffn_config=ffn_config,
        )
    if canonical_type == "moe":
        moe_config = data.get("moe_config", {})
        return MoELayerConfig(
            layer_type="moe",
            name=name,
            layer_id=idx,
            attn_config=attn_config,
            moe_config=moe_config,
        )
    if canonical_type == "communication":
        comm_config = data.get("comm_config", data)
        return CommunicationLayerConfig(
            layer_type="communication",
            name=name,
            layer_id=idx,
            attn_config=attn_config,
            comm_config=comm_config,
        )
    return BaseLayerConfig(
        layer_type="attention",
        name=name,
        layer_id=idx,
        attn_config=attn_config or data,
    )


def format_ms(value: float) -> str:
    return f"{value:8.3f}"


def format_gflops(value: float) -> str:
    return f"{value / 1e9:8.3f}"


def format_gb(value: float) -> str:
    return f"{value / 1e9:8.3f}"


__all__ = [
    "SUPPORTED_LAYER_TYPES",
    "read_yaml",
    "maybe_load_reference",
    "hardware_from_dict",
    "layer_config_from_dict",
    "format_ms",
    "format_gflops",
    "format_gb",
]
