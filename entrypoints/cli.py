"""Unified CLI for AFDSimulator workflows."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from core.data import LayerExecution, RuntimeSpec
from core.estimation import AnalyticEstimator

from .simulator import Scenario, load_scenario, run_simulation
from .utils import format_gb, format_gflops, format_ms


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


_DEF_PAD = 12


def _print_afd_table(layers: Sequence[LayerExecution]) -> None:
    headers = ["layer", "type", "gflops", "compute_ms", "memory_ms", "latency_ms"]
    row_fmt = "{layer:>12} {layer_type:>10} {gflops:>10} {compute_ms:>12} {memory_ms:>12} {latency_ms:>12}"
    print(" ".join(h.rjust(_DEF_PAD) for h in headers))
    for layer in layers:
        print(
            row_fmt.format(
                layer=layer.layer_name,
                layer_type=layer.layer_type,
                gflops=format_gflops(layer.flops),
                compute_ms=format_ms(layer.compute_time_ms),
                memory_ms=format_ms(layer.memory_time_ms),
                latency_ms=format_ms(layer.dominant_latency_ms),
            )
        )


def _print_large_ep_table(layers: Sequence[LayerExecution]) -> None:
    headers = ["layer", "type", "gflops", "bytes_gb", "latency_ms"]
    row_fmt = "{layer:>12} {layer_type:>12} {gflops:>10} {bytes_gb:>10} {latency_ms:>12}"
    print(" ".join(h.rjust(_DEF_PAD) for h in headers))
    for layer in layers:
        bytes_total = layer.bytes_read + layer.bytes_written
        print(
            row_fmt.format(
                layer=layer.layer_name,
                layer_type=layer.layer_type,
                gflops=format_gflops(layer.flops),
                bytes_gb=format_gb(bytes_total),
                latency_ms=format_ms(layer.dominant_latency_ms),
            )
        )


_TABLE_RENDERERS: Dict[str, Callable[[Sequence[LayerExecution]], None]] = {
    "afd": _print_afd_table,
    "large_ep": _print_large_ep_table,
}


# ---------------------------------------------------------------------------
# Workflow registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkflowDefinition:
    name: str
    help_text: str
    command: str
    command_help: str
    table_renderer: Callable[[Sequence[LayerExecution]], None]


WORKFLOWS: Dict[str, WorkflowDefinition] = {
    "afd": WorkflowDefinition(
        name="afd",
        help_text="Attention–FFN disaggregation workflows",
        command="simulate",
        command_help="Run analytic AFD simulation",
        table_renderer=_TABLE_RENDERERS["afd"],
    ),
    "large-ep": WorkflowDefinition(
        name="large-ep",
        help_text="Large expert-parallel workflows",
        command="evaluate",
        command_help="Run expert-parallel analytic simulation",
        table_renderer=_TABLE_RENDERERS["large_ep"],
    ),
}


# ---------------------------------------------------------------------------
# Core execution
# ---------------------------------------------------------------------------


def _attach_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("scenario", type=str, help="Path to scenario YAML")
    parser.add_argument("--batch", type=int, required=True, help="Batch size")
    parser.add_argument("--seq", type=int, required=True, help="Sequence length")
    parser.add_argument("--output", type=str, help="Optional path to dump raw result dict")


def _run_simulation(args: argparse.Namespace) -> Scenario:
    scenario: Scenario = load_scenario(args.scenario)
    runtime = RuntimeSpec(batch_size=args.batch, seq_len=args.seq)
    estimator = AnalyticEstimator(scenario.hardware, runtime)
    result = run_simulation(scenario, runtime, estimator=estimator)

    print(f"Scenario: {scenario.name}")
    print(f"Hardware: {scenario.hardware.name}")

    workflow_key = args.workflow
    definition = WORKFLOWS.get(workflow_key)
    if definition is None:
        raise ValueError(f"Unknown workflow '{workflow_key}'")
    definition.table_renderer(result.layers)

    print("\nTotals:")
    print(f"  Total latency (ms): {result.total_latency_ms:.3f}")
    print(f"  Total FLOPs (GFLOPs): {result.total_flops / 1e9:.3f}")
    print(f"  Peak memory (GB): {result.peak_memory_bytes / 1e9:.3f}")
    print(f"  Bottleneck layer: {result.bottleneck_layer}")

    if args.output:
        path = Path(args.output)
        path.write_text(str(result.to_dict()))
        print(f"\nSaved raw result to {path}")

    return scenario


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AFDSimulator unified CLI")
    workflow_subparsers = parser.add_subparsers(dest="workflow", required=True)

    for key, definition in WORKFLOWS.items():
        workflow_parser = workflow_subparsers.add_parser(key, help=definition.help_text)
        workflow_parser.set_defaults(workflow=key)
        command_subparsers = workflow_parser.add_subparsers(dest="command", required=True)

        command_parser = command_subparsers.add_parser(definition.command, help=definition.command_help)
        _attach_shared_arguments(command_parser)
        command_parser.set_defaults(handler=_run_simulation)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("No command handler registered")
    handler(args)


if __name__ == "__main__":
    main()
