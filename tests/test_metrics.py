from core.data import HardwareSpec
from core.ops.metrics import (
    compute_time_ms,
    dominant_latency_ms,
    matmul_flops,
    memory_time_ms,
    tensor_bytes,
)


def test_matmul_and_tensor_bytes():
    assert matmul_flops(2, 3, 4) == 48
    assert tensor_bytes((2, 2), dtype_bits=32) == 16


def test_timing_helpers():
    hardware = HardwareSpec(
        name="TestGPU",
        peak_tflops=100,
        memory_bandwidth_gbps=1000,
        hbm_gb=80,
        interconnect_gbps=600,
    )
    compute_ms = compute_time_ms(2e12, hardware)  # 2 TFLOPs
    memory_ms = memory_time_ms(2e12 / 8, hardware)  # bytes
    latency_ms = dominant_latency_ms(compute_ms, memory_ms, hardware)

    assert compute_ms > 0
    assert memory_ms > 0
    assert latency_ms >= max(compute_ms, memory_ms / hardware.effective_overlap())
