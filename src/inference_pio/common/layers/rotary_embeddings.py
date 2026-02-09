from ...core.engine.layers import Module
from ...core.engine.backend import precompute_freqs_cis, Tensor

class GenericRotaryEmbedding(Module):
    def __init__(self, dim, max_pos=2048, base=10000.0, device="cpu"):
        super().__init__()
        self.cos, self.sin = precompute_freqs_cis(dim, max_pos, base, device)
    def forward(self, x, seq_len=None):
        return self.cos, self.sin

__all__ = ["GenericRotaryEmbedding"]
