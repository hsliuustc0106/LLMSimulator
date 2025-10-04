# LLMSimulator

LLMSimulator is a planning sandbox for giant LLM inference. It keeps FLOP estimates deterministic and hardware-agnostic, then turns those numbers into latency projections (in **milliseconds**) using per-device bandwidth/throughput specs. The framework is layer-centric: attention, FFN/MoE, and communication components are evaluated independently and rolled up into an end-to-end serving profile.

## Highlights
- **Deterministic analytics**: fused-op formulas capture QKV projections, expert matmuls, and routing paths so FLOPs are reproducible across runs.
- **Hardware-aware timing**: peak TFLOPs, HBM bandwidth, and interconnect constraints map analytic workloads to compute/memory times (reported in ms).
- **Scenario-ready configs**: hardware definitions live beside model-layer templates (DeepSeek-style YAML), while runtime knobs (batch/sequence) come from the CLI.
- **Dual estimators**: analytic backend ships by default; an ML-backed predictor can plug in later to refine latency with offline profiling data.
- **Serving playbooks**: `afd.py` targets attention–FFN disaggregation, and `large_ep.py` covers large expert-parallel layouts.

## Architecture Overview
```
configs/        hardware + layer templates (YAML)
core/
  ops/          fused-op primitives & hardware helpers
  module/       layer estimators returning LayerExecution records
entrypoints/
  cli.py        unified CLI entry
  simulator.py  scenario loading, simulation, and reporting helpers
```

1. **Config ingestion**: load hardware YAML (bandwidth/throughput) and layer templates (attention/ffn/moe blocks) via `entrypoints.simulator.load_scenario`. Runtime (`batch`, `seq`) supplied through CLI args.
2. **Fused-op accounting (`core/ops/`)**: reusable kernels (e.g., `attention_qkv_proj`, `moe_expert_mm`) expose FLOP and byte formulas.
3. **Layer estimators (`core/module/`)**: compose fused-ops, returning a `LayerExecution` dataclass with
   - analytic FLOPs
   - bytes read/written
   - compute time (ms)
   - memory time (ms)
   - dominant latency (ms, via `max(compute, memory)`)
   - `estimated_execution_time_ms` alias for downstream tooling
4. **Simulation**: `entrypoints.simulator` walks ordered layers, aggregates totals (latency, FLOPs, peak memory) and surfaces the bottleneck stage.
5. **Reporting**: CLI renders human tables or JSON/CSV for dashboards; ML pipeline can dump features for training.

## Core Data Structures
- `HardwareSpec`: `{name, peak_tflops, memory_bandwidth_gbps, hbm_gb, interconnect_gbps, max_concurrency, overlap_efficiency}`
- `RuntimeSpec`: `{batch_size, seq_len, optional micro_batch/tokens_per_expert}` *(from CLI)*
- `LayerConfig` variants:
  - `FFNLayerConfig`: inherits shared `attn_config`, adds `ffn_config` (e.g., `d_model`, `d_ff`, `hidden_act`)
  - `MoELayerConfig`: shared `attn_config`, plus `moe_config` (`moe_intermediate_size`, `num_experts_per_tok`, `topk_group`, ...)
  - `CommLayerConfig`: `comm_config` (`payload_mb`, `pattern`)
- `FusionOp`: named kernel with `flops_fn`, `bytes_fn`, optional notes.
- `LayerExecution`: `{layer_name, layer_type, flops, bytes_read, bytes_written, compute_time_ms, memory_time_ms, dominant_latency_ms, estimated_execution_time_ms, features, breakdown}`
- `SimulationResult`: ordered list of `LayerExecution` + totals (`total_latency_ms`, `total_flops`, `peak_memory_bytes`, `bottleneck_layer`).

## Preparing Configs
1. **Hardware** (`configs/hardware/NV-A100.yaml`):
   ```yaml
   name: "A100-80GB"
   peak_tflops: 312
   memory_bandwidth_gbps: 2030
   hbm_gb: 80
   interconnect_gbps: 600
   max_concurrency: 2
   overlap_efficiency: 1.0
   ```
2. **Layer templates** (`configs/models/deepseek-v3/ffn_layer_config.yaml`):
   ```yaml
   attn_config:
     type: "MLA"
     d_model: 7168
     num_attention_heads: 128
     head_dim: 56
     num_key_value_heads: 128
     v_head_dim: 128
   ffn_config:
     d_model: 7168
     d_ff: 18432
     hidden_act: silu
   ```
3. **Scenario YAML** (example `configs/scenarios/deepseek_v3_a100.yaml`):
   ```yaml
   hardware: configs/hardware/NV-A100.yaml
   layers:
     - type: "ffn_layer"
       config: configs/models/deepseek-v3/ffn_layer_config.yaml
     - type: "moe_layer"
       config: configs/models/deepseek-v3/moe_layer_config.yaml
   ```
   Runtime overrides come from the CLI (`--batch 8 --seq 4096`).

## CLI
Unified entry point (`python -m entrypoints.cli`):
```bash
python -m entrypoints.cli afd simulate \
    configs/scenarios/deepseek_v3_a100.yaml \
    --batch 8 --seq 4096
```
Expert-parallel workflow:
```bash
python -m entrypoints.cli large-ep evaluate \
    configs/scenarios/deepseek_v3_a100.yaml \
    --batch 4 --seq 8192
```
Planned ML estimator support will reuse the same CLI (e.g., `--estimator ml --model models/afd_latency.pkl`).
Outputs can be rendered as tables or exported to JSON/CSV for dashboards.

## Use Cases
1. **Layer Budgeting** – identify compute vs memory bottlenecks for attention/FFN across shapes.
2. **Expert Parallel Planning** – evaluate MoE routing + interconnect overhead with different expert counts.
3. **Batch/Sequence Sweeps** – sweep runtime knobs to find throughput sweet spots that respect HBM limits.
4. **ML Estimator Validation** – compare analytic predictions vs profiled latencies, refine with joblib models.
5. **Fused-Op Inspection** – inspect individual kernels’ FLOPs/bytes to debug formula assumptions.

## Offline Profiling & ML Integration
- Use analytic simulator to emit feature CSVs (`LayerExecution.features`).
- Collect measured latencies from real runs, align with features, train an ML regressor (scikit-learn/joblib).
- Drop the trained model into `models/` and enable `--estimator ml` for tighter latency forecasts.
- Fall back to analytic numbers if the ML model is unavailable or outside its training envelope.


## Contributing
Issues and PRs are welcome—especially around new fused ops, profiling data, or estimator improvements. Please keep contributions ASCII-friendly and accompany complex logic with concise comments.