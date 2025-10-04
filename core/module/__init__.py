# Public  APIs
from .attention import Attention
from .communication import Communication
from .moe import MoE  # main new class
from .ffn import FFN  # wrapper that aliases FFN -> MoE

__all__ = ["Attention", "FFN", "Communication", "MoE"]