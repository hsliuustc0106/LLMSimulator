"""Catalog of fused operations used by analytic estimators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .metrics import matmul_flops, tensor_bytes

DEFAULT_DTYPE_BITS = 16


@dataclass(frozen=True)
class FusionMetrics:
    name: str
    flops: float
    bytes_accessed: float

    def as_dict(self) -> Dict[str, float]:
        return {"flops": self.flops, "bytes_accessed": self.bytes_accessed}


def attention_qkv_projections(batch: int, seq: int, d_model: int, qkv_dim: int, *, dtype_bits: int = DEFAULT_DTYPE_BITS) -> FusionMetrics:
    tokens = batch * seq
    flops_per = matmul_flops(tokens, qkv_dim, d_model)
    flops = 3.0 * flops_per
    input_bytes = tensor_bytes((batch, seq, d_model), dtype_bits)
    weight_bytes = tensor_bytes((d_model, qkv_dim), dtype_bits) * 3
    output_bytes = tensor_bytes((batch, seq, qkv_dim), dtype_bits) * 3
    total_bytes = input_bytes + weight_bytes + output_bytes
    return FusionMetrics("attention_qkv_proj", flops, total_bytes)


def attention_scores(batch: int, seq: int, num_heads: int, head_dim: int, *, dtype_bits: int = DEFAULT_DTYPE_BITS) -> FusionMetrics:
    # Q x K^T matmul plus softmax
    flops_per_head = matmul_flops(seq, seq, head_dim)
    flops = flops_per_head * batch * num_heads
    softmax_flops = batch * num_heads * seq * seq
    total_flops = flops + softmax_flops
    q_bytes = tensor_bytes((batch, num_heads, seq, head_dim), dtype_bits)
    k_bytes = tensor_bytes((batch, num_heads, seq, head_dim), dtype_bits)
    attn_bytes = tensor_bytes((batch, num_heads, seq, seq), dtype_bits)
    total_bytes = q_bytes + k_bytes + attn_bytes
    return FusionMetrics("attention_scores", total_flops, total_bytes)


def attention_weighted_sum(batch: int, seq: int, num_heads: int, head_dim: int, *, dtype_bits: int = DEFAULT_DTYPE_BITS) -> FusionMetrics:
    flops_per_head = matmul_flops(seq, head_dim, seq)
    flops = flops_per_head * batch * num_heads
    attn_bytes = tensor_bytes((batch, num_heads, seq, seq), dtype_bits)
    v_bytes = tensor_bytes((batch, num_heads, seq, head_dim), dtype_bits)
    output_bytes = tensor_bytes((batch, seq, num_heads * head_dim), dtype_bits)
    total_bytes = attn_bytes + v_bytes + output_bytes
    return FusionMetrics("attention_weighted_sum", flops, total_bytes)


def attention_output_projection(batch: int, seq: int, d_model: int, qkv_dim: int, *, dtype_bits: int = DEFAULT_DTYPE_BITS) -> FusionMetrics:
    tokens = batch * seq
    flops = matmul_flops(tokens, d_model, qkv_dim)
    input_bytes = tensor_bytes((batch, seq, qkv_dim), dtype_bits)
    weight_bytes = tensor_bytes((qkv_dim, d_model), dtype_bits)
    output_bytes = tensor_bytes((batch, seq, d_model), dtype_bits)
    total_bytes = input_bytes + weight_bytes + output_bytes
    return FusionMetrics("attention_output_proj", flops, total_bytes)


def ffn_activation(batch: int, seq: int, d_model: int, hidden_dim: int, *, dtype_bits: int = DEFAULT_DTYPE_BITS) -> FusionMetrics:
    tokens = batch * seq
    flops_first = matmul_flops(tokens, hidden_dim, d_model)
    flops_second = matmul_flops(tokens, d_model, hidden_dim)
    activation_flops = tokens * hidden_dim  # simple approximation for SwiGLU/SiLU
    total_flops = flops_first + flops_second + activation_flops
    input_bytes = tensor_bytes((batch, seq, d_model), dtype_bits)
    hidden_bytes = tensor_bytes((batch, seq, hidden_dim), dtype_bits) * 2
    weight_bytes = tensor_bytes((d_model, hidden_dim), dtype_bits) + tensor_bytes((hidden_dim, d_model), dtype_bits)
    total_bytes = input_bytes + hidden_bytes + weight_bytes
    return FusionMetrics("ffn", total_flops, total_bytes)


def moe_expert_forward(active_tokens: int, d_model: int, expert_hidden: int, *, dtype_bits: int = DEFAULT_DTYPE_BITS) -> FusionMetrics:
    flops_first = matmul_flops(active_tokens, expert_hidden, d_model)
    flops_second = matmul_flops(active_tokens, d_model, expert_hidden)
    activation_flops = active_tokens * expert_hidden
    total_flops = flops_first + flops_second + activation_flops
    act_bytes = tensor_bytes((active_tokens, d_model), dtype_bits)
    hidden_bytes = tensor_bytes((active_tokens, expert_hidden), dtype_bits)
    weight_bytes = tensor_bytes((d_model, expert_hidden), dtype_bits) + tensor_bytes((expert_hidden, d_model), dtype_bits)
    total_bytes = act_bytes + hidden_bytes + weight_bytes
    return FusionMetrics("moe_expert", total_flops, total_bytes)


def moe_routing(batch: int, seq: int, num_experts: int, top_k: int, *, dtype_bits: int = DEFAULT_DTYPE_BITS) -> FusionMetrics:
    tokens = batch * seq
    flops = tokens * num_experts  # gating scores
    select_flops = tokens * top_k
    total_flops = flops + select_flops
    gate_bytes = tensor_bytes((tokens, num_experts), dtype_bits)
    return FusionMetrics("moe_routing", total_flops, gate_bytes)


def communication_all_to_all(bytes_per_device: float) -> FusionMetrics:
    return FusionMetrics("all_to_all", 0.0, bytes_per_device)


def communication_all_reduce(bytes_per_device: float) -> FusionMetrics:
    return FusionMetrics("all_reduce", 0.0, bytes_per_device)
