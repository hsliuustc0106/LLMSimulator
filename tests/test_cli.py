from pathlib import Path

from entrypoints import cli


def _write_basic_files(tmp_path: Path) -> Path:
    hardware_yaml = tmp_path / "hardware.yaml"
    hardware_yaml.write_text(
        """
name: TestGPU
peak_tflops: 120
memory_bandwidth_gbps: 1500
hbm_gb: 80
interconnect_gbps: 600
"""
    )

    ffn_yaml = tmp_path / "ffn.yaml"
    ffn_yaml.write_text(
        """
attn_config:
  d_model: 128
  num_attention_heads: 8
ffn_config:
  d_model: 128
  d_ff: 512
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
    return scenario_yaml


def test_unified_cli_afd(monkeypatch, tmp_path, capsys):
    scenario = _write_basic_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    cli.main([
        "afd",
        "simulate",
        str(scenario.name),
        "--batch",
        "2",
        "--seq",
        "16",
    ])
    out = capsys.readouterr().out
    assert "Scenario:" in out
    assert "Total latency" in out


def test_unified_cli_large_ep(monkeypatch, tmp_path, capsys):
    scenario = _write_basic_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    cli.main([
        "large-ep",
        "evaluate",
        str(scenario.name),
        "--batch",
        "2",
        "--seq",
        "16",
    ])
    out = capsys.readouterr().out
    assert "Scenario:" in out
    assert "Total latency" in out
