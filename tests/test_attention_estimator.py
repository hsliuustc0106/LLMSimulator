from core.data import HardwareSpec
from core.module.attention import Attention


def test_attention_estimate_execution():
    hardware = HardwareSpec(
        name="TestGPU",
        peak_tflops=150,
        memory_bandwidth_gbps=1555,
        hbm_gb=80,
        interconnect_gbps=600,
    )
    attn = Attention({"d_model": 128, "num_attention_heads": 8, "head_dim": 16})
    execution = attn.estimate_execution_time(batch=4, seq=128, hardware=hardware)

    assert execution.layer_name == "attention"
    assert execution.flops > 0
    assert execution.compute_time_ms > 0
    assert execution.dominant_latency_ms >= execution.compute_time_ms
