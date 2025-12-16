"""
Adaptive computation optimization components for Qwen3-VL.
Implements mechanisms for dynamic compute allocation, such as early exiting
and conditional layer execution based on token confidence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any

class AdaptiveAttention(nn.Module):
    """
    Adaptive Attention module that can dynamically adjust its computational cost.
    It includes a 'gate' that determines if the full attention mechanism is needed
    for a given token or sequence.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Standard attention projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

        # Router/Gate for adaptivity
        # Outputs a score [0, 1]. If < threshold, we might skip or use approximation.
        self.gate = nn.Linear(self.hidden_size, 1)
        self.threshold = getattr(config, 'adaptive_threshold', 0.5)

        # Fallback linear approximation (cheaper than Attention)
        self.fallback_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:

        bsz, q_len, _ = hidden_states.size()

        # Compute gating score
        # Using sigmoid to get probability/confidence
        gate_scores = torch.sigmoid(self.gate(hidden_states)).mean(dim=1) # Average over sequence for decision

        # Decisions for the batch (simplification: entire batch takes same path)
        # In more complex implementations, this could be per-token
        should_compute_full = gate_scores.mean() > self.threshold

        if not should_compute_full and self.training is False:
            # Fast path: Linear approximation
            # This skips the O(N^2) attention matrix calc
            output = self.fallback_proj(hidden_states)
            return output, None, None

        # Full Attention Path (Standard)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
             attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, None


class AdaptiveMLP(nn.Module):
    """
    Adaptive MLP (Mixture of Experts style or Early Exit).
    This implementation uses a simple skipping mechanism:
    If confidence is high, skip the MLP block (identity), or use a smaller MLP.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False)
        self.act_fn = nn.SiLU()

        # Importance gate
        self.importance_gate = nn.Linear(self.hidden_dim, 1)
        self.importance_threshold = getattr(config, 'mlp_importance_threshold', 0.1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Calculate importance/residual requirement
        importance = torch.sigmoid(self.importance_gate(hidden_states))

        # Mask for tokens that need update
        # We only compute MLP for tokens with high importance score
        # For simplicity in batch operations, we'll use soft masking here
        # (compute all, but scale by importance)
        # An optimized version would use index_select to only compute for specific tokens.

        down_proj = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

        if not self.training:
            # In inference, we can hard-skip updates for low importance
            mask = (importance > self.importance_threshold).float()
            return down_proj * mask

        # During training, use soft gating to learn the gate
        return down_proj * importance

import math
