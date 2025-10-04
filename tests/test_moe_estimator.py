from core.data import HardwareSpec
from core.module.moe import MoE


def test_moe_estimate_execution():
    hardware = HardwareSpec(
        name="TestGPU",
        peak_tflops=200,
        memory_bandwidth_gbps=2000,
        hbm_gb=80,
        interconnect_gbps=900,
    )
    moe_config = {
        "d_model": 256,
        "moe_intermediate_size": 512,
        "n_routed_experts": 16,
        "topk_group": 2,
        "num_experts_per_tok": 2,
    }
    moe = MoE(moe_config)
    execution = moe.estimate_execution_time(batch=2, seq=64, hardware=hardware)

    assert execution.layer_name == "moe"
    assert execution.flops > 0
    assert execution.dominant_latency_ms >= execution.compute_time_ms
