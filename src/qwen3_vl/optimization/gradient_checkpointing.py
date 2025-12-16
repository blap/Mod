"""
Gradient checkpointing optimization components for Qwen3-VL.
Implements memory-efficient variants of core layers using gradient checkpointing and optimal activation recomputation.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple


class MemoryEfficientAttention(nn.Module):
    """
    Attention module that supports gradient checkpointing to save memory.
    Wraps standard attention logic with checkpoint calls.
    """
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # In a real scenario, this would initialize the underlying attention mechanism
        # For this consolidation, we assume it wraps an external implementation or acts as a mixin.
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Basic projections to make it functional
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        def custom_forward(hidden_states, attention_mask, position_ids):
            # Simplified self-attention logic for demonstration/functional completeness
            batch_size, seq_len, _ = hidden_states.shape

            q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Scaled Dot Product Attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, is_causal=True if position_ids is not None else False
            )

            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
            output = self.o_proj(attn_output)
            return output

        if self.training and getattr(self.config, 'gradient_checkpointing', False):
            # Use checkpointing
            output = checkpoint(custom_forward, hidden_states, attention_mask, position_ids, use_reentrant=False)
        else:
            output = custom_forward(hidden_states, attention_mask, position_ids)

        return output, None, past_key_value


class MemoryEfficientMLP(nn.Module):
    """
    MLP module that supports gradient checkpointing.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        def custom_forward(x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        if self.training and getattr(self.config, 'gradient_checkpointing', False):
             return checkpoint(custom_forward, x, use_reentrant=False)
        return custom_forward(x)
