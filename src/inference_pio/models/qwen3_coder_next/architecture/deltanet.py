"""
Gated DeltaNet Module for Qwen3-Coder-Next
"""

import torch
import torch.nn as nn
import logging
import math

logger = logging.getLogger(__name__)

# Try to import custom kernels
try:
    from ..cuda_kernels import qwen3_coder_next_cuda_kernels as custom_kernels
    HAS_KERNELS = True
except ImportError:
    HAS_KERNELS = False

class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet Layer: A linear attention variant with Gated Recurrence.

    Structure:
    - Projections: Q, K, V, Beta (Gate)
    - Delta Rule Update
    - Output Projection
    """
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.deltanet_query_key_heads # 16
        self.num_value_heads = config.deltanet_value_heads # 32
        self.head_dim = config.deltanet_head_dim # 128

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_value_heads * self.head_dim, bias=False)
        self.beta_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(self.num_value_heads * self.head_dim, self.hidden_size, bias=False)

        self.group_norm = nn.GroupNorm(num_groups=self.num_heads, num_channels=self.num_heads * self.head_dim)


    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        bsz, seq_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_value_heads, self.head_dim)
        beta = torch.sigmoid(self.beta_proj(hidden_states)).view(bsz, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: [bsz, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        beta = beta.transpose(1, 2)

        # State Management for RNN mode
        # State shape: [bsz, heads, head_dim, head_dim]
        # We need to handle initial state being None, or a tensor
        initial_state = None
        if past_key_value is not None:
             initial_state = past_key_value

        if HAS_KERNELS and self.config.enable_deltanet_kernel and hidden_states.is_cuda:
            # Call Custom CUDA Kernel
            # Inputs: q, k, v, beta, initial_state
            # Kernel handles the loop internally
            attn_output, final_state = custom_kernels.deltanet_fwd(
                q, k, v, beta,
                initial_state if initial_state is not None else torch.empty(0)
            )
        else:
            # PyTorch Fallback (Correct Sequential Recurrence)
            attn_output, final_state = self._pytorch_deltanet_scan(q, k, v, beta, initial_state)

        # Output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.o_proj(attn_output)

        return output, final_state

    def _pytorch_deltanet_scan(self, q, k, v, beta, initial_state=None):
        """
        PyTorch implementation of the DeltaNet recurrence.
        S_t = S_{t-1} + beta_t * (v_t @ k_t.T)
        O_t = S_t @ q_t
        """
        bsz, heads, seq_len, head_dim = q.shape

        # Initialize state
        if initial_state is None:
            state = torch.zeros(bsz, heads, head_dim, head_dim, device=q.device, dtype=q.dtype)
        else:
            state = initial_state.clone()

        output_list = []

        # Iterate over sequence
        for t in range(seq_len):
            q_t = q[:, :, t, :].unsqueeze(-1) # [B, H, D, 1]
            k_t = k[:, :, t, :].unsqueeze(-1) # [B, H, D, 1]
            v_t = v[:, :, t, :].unsqueeze(-1) # [B, H, D, 1]
            beta_t = beta[:, :, t, :].unsqueeze(-1) # [B, H, D, 1]

            # Update Rule: S_new = S_old + (v * beta) @ k.T
            # Note: The exact delta rule might differ (e.g. decay).
            # Using the additive gated rule consistent with the kernel I wrote.

            # Update term: [B, H, D, 1] @ [B, H, 1, D] -> [B, H, D, D]
            update = torch.matmul(v_t * beta_t, k_t.transpose(-1, -2))

            state = state + update

            # Output: S @ q
            # [B, H, D, D] @ [B, H, D, 1] -> [B, H, D, 1]
            o_t = torch.matmul(state, q_t).squeeze(-1) # [B, H, D]
            output_list.append(o_t)

        output = torch.stack(output_list, dim=2) # [B, H, L, D]

        return output, state
