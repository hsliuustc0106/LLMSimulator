"""Reusable metric helpers for FLOPs, tensor sizing, and timing."""
from __future__ import annotations

from typing import Iterable, Sequence

from core.data import HardwareSpec

BYTES_PER_GB = 1e9


def matmul_flops(m: int, n: int, k: int) -> float:
    """Return FLOPs for a dense matmul (2*m*n*k)."""
    return float(2 * m * n * k)


def tensor_elements(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= max(int(dim), 0)
    return total


def tensor_bytes(shape: Sequence[int], dtype_bits: int = 16) -> float:
    return tensor_elements(shape) * dtype_bits / 8.0


def sum_tensor_bytes(shapes: Iterable[Sequence[int]], dtype_bits: int = 16) -> float:
    return sum(tensor_bytes(shape, dtype_bits=dtype_bits) for shape in shapes)


def compute_time_ms(flops: float, hardware: HardwareSpec) -> float:
    throughput = hardware.compute_throughput_tflops()
    if throughput <= 0:
        return float("inf")
    seconds = flops / (throughput * 1e12)
    return seconds * 1e3


def memory_time_ms(bytes_moved: float, hardware: HardwareSpec) -> float:
    bandwidth = hardware.memory_bandwidth_bytes()
    if bandwidth <= 0:
        return float("inf")
    seconds = bytes_moved / bandwidth
    return seconds * 1e3


def interconnect_time_ms(bytes_moved: float, hardware: HardwareSpec) -> float:
    bandwidth = hardware.interconnect_bandwidth_bytes()
    if bandwidth <= 0:
        return 0.0
    seconds = bytes_moved / bandwidth
    return seconds * 1e3


def dominant_latency_ms(compute_ms: float, memory_ms: float, hardware: HardwareSpec) -> float:
    overlap = hardware.effective_overlap()
    adjusted_memory = memory_ms / overlap
    return max(compute_ms, adjusted_memory)
