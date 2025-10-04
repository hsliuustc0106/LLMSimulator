from pathlib import Path

from core.data import RuntimeSpec
from core.estimation import AnalyticEstimator
from entrypoints.simulator import load_scenario, run_simulation


def test_scenario_loading_and_simulation(tmp_path: Path):
    hardware_yaml = tmp_path / "hardware.yaml"
    hardware_yaml.write_text(
        """
name: TestGPU
peak_tflops: 120
memory_bandwidth_gbps: 1500
hbm_gb: 80
interconnect_gbps: 600
max_concurrency: 2
overlap_efficiency: 1.0
"""
    )

    ffn_yaml = tmp_path / "ffn.yaml"
    ffn_yaml.write_text(
        """
attn_config:
  d_model: 256
  num_attention_heads: 8
ffn_config:
  d_model: 256
  d_ff: 1024
"""
    )

    scenario_yaml = tmp_path / "scenario.yaml"
    scenario_yaml.write_text(
        f"""
name: test_scenario
hardware: {hardware_yaml.name}
layers:
  - type: "ffn_layer"
    config: {ffn_yaml.name}
"""
    )

    scenario = load_scenario(scenario_yaml)
    assert scenario.hardware.name == "TestGPU"
    assert scenario.layers[0].layer_type == "ffn"

    runtime = RuntimeSpec(batch_size=2, seq_len=32)
    estimator = AnalyticEstimator(scenario.hardware, runtime)
    result = run_simulation(scenario, runtime, estimator=estimator)

    assert result.total_latency_ms > 0
    assert result.total_flops > 0
    assert result.bottleneck_layer == scenario.layers[0].name
