# AFDSimulator Agent Guide

This project is actively co-developed by a fleet of automation agents. The notes below describe expectations, guardrails, and hand-off conventions so future agents can contribute without accidental regressions.

## Mission
- Deliver accurate execution-time estimates (milliseconds) for giant LLM inference components.
- Keep analytic FLOP/memory accounting deterministic; allow an ML estimator to refine latency when profiling data is available.
- Treat attention–FFN disaggregation (AFD) and large expert parallelism (Large-EP) as first-class scenarios with dedicated CLIs.

## Ground Rules
- **Never overwrite user edits unintentionally.** Inspect diffs before saving; preserve non-agent changes.
- **ASCII-first.** Stay in ASCII unless context requires otherwise.
- **Concise comments.** Only add comments where the intent isn’t obvious; avoid boilerplate explanations.
- **Milliseconds everywhere.** Any time metric exposed to users must be in ms.
- **Read-only vs write mode.** Verify write permissions before editing; request escalation when the harness demands it.

## Repository Conventions
- `core/ops/`: fused-op primitives (FLOPs + bytes). Implement reusable helpers here before touching module estimators.
- `core/module/`: layer-level estimators that assemble fused ops and output `LayerExecution` structures with analytic and derived timing.
- `entrypoints/`: simulators + unified CLI (`cli.py`) that load configs, parse runtime args, and orchestrate per-layer execution.
- `configs/`: hardware specs, model-layer templates, and scenario definitions. Runtime knobs (batch/seq) are CLI-only; do not bake them into YAML.
- `tests/`: each new fused op or estimator needs unit coverage; integration tests must load at least one scenario YAML.

## LayerExecution Contract
Every layer estimator must return a populated `LayerExecution` with:
- `layer_name`, `layer_type`
- Analytic `flops`, `bytes_read`, `bytes_written`
- `compute_time_ms`, `memory_time_ms`, `dominant_latency_ms`
- `estimated_execution_time_ms` (alias to dominant latency)
- Optional `breakdown` (per fused op) and `features` (for ML estimator training)

## Estimation Backends
- **AnalyticEstimator** (default): use fused-op formulas + `HardwareSpec` to compute times.
- **MLEstimator** (optional): load joblib/sklearn model; consume `LayerExecution.features`. Always fall back to analytic path if the model is missing or throws.

## Development Workflow
1. Sync with main and `pip install -r requirements.txt` (or use the `afdsim` conda env via `conda activate afdsim`).
2. If adding new fused ops, implement in `core/ops/fused_ops.py`, then write estimator logic.
3. Update tests and run `pytest` before submitting.
4. Update `readme.md` and scenario examples when you expose new knobs.
5. Document CLI affordances clearly; all runtime defaults should be discoverable via `--help`.

## Communication & Handoff
- Record major design decisions in this file to prevent divergent implementations.
- When starting larger refactors, append a short plan here so the next agent can track context.
- Mention pending todos or known issues explicitly (e.g., missing ML model file, inaccurate bandwidth assumptions).

## Known Todos
- Implement fused-op library and metrics helpers under `core/ops/`.
- Stand up shared dataclasses (`core/data/`) plus config loader plumbing.
- Deliver analytic layer estimators emitting `LayerExecution` outputs in `core/module/`.
- Build simulation orchestrators + CLIs (`afd.py`, `large_ep.py`) with runtime arg parsing.
- Add reporting utilities (tabular + JSON/CSV) and scenario sweep helpers.
- Wire up ML estimator hooks alongside profiling/training scripts.
- Expand regression tests across ops, estimators, and ingestion paths.

Keep iterations tight, verify with tests, and coordinate through this guide.
