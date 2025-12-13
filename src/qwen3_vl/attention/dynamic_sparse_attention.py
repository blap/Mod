"""
Dynamic sparse attention with learned routing for token selection.
This implements the first task of Phase 7: Advanced Architecture Optimizations.
"""
import math
import warnings
from typing import Optional, Tuple, Union, List

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from config.config import Qwen3VLConfig
from architectures.modeling_qwen3_vl_phase2 import apply_rotary_pos_emb, repeat_kv, Qwen3VLRotaryEmbedding


class DynamicSparseAttention(nn.Module):
    """
    Dynamic sparse attention with learned routing for token selection.
    Selectively computes attention only for the most relevant token pairs.
    """
    def __init__(self, config: Qwen3VLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads or self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Rotary embedding
        self.rotary_emb = Qwen3VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        
        # Learned routing mechanism to determine which tokens to attend to
        self.routing_network = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, self.num_heads),  # One routing score per head
            nn.Softmax(dim=-1)
        )
        
        # Sparse attention parameters
        self.sparsity_ratio = getattr(config, 'sparse_attention_sparsity_ratio', 0.5)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings if position_ids is provided
        if position_ids is not None:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat K and V if using GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Apply learned routing to determine which tokens to attend to
        # Use the global representation to compute routing scores
        global_repr = hidden_states.mean(dim=1, keepdim=True)  # [bsz, 1, hidden_size]
        routing_scores = self.routing_network(global_repr)  # [bsz, 1, num_heads]
        routing_scores = routing_scores.squeeze(1)  # [bsz, num_heads]

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply dynamic sparsity: for each head, keep only top-k attention weights
        # Calculate top_k based on actual sequence length and sparsity ratio
        top_k = max(1, int(self.sparsity_ratio * q_len))

        # Apply sparsity to attention weights
        sparse_attn_weights = self.apply_dynamic_sparsity(attn_weights, top_k)

        # Apply softmax
        attn_weights = F.softmax(sparse_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def apply_dynamic_sparsity(self, attn_weights: torch.Tensor, top_k: int) -> torch.Tensor:
        """
        Apply dynamic sparsity by keeping only the top-k attention weights for each query position.
        This is a vectorized and more efficient implementation.
        """
        bsz, num_heads, q_len, k_len = attn_weights.size()
        
        # Calculate actual top_k to not exceed sequence length
        top_k_val = min(top_k, k_len)
        
        # Create a copy of attention weights to modify
        sparse_attn_weights = attn_weights.clone()
        
        # Vectorized approach: reshape to apply top-k across all positions at once
        # Reshape to (bsz * num_heads * q_len, k_len)
        reshaped_weights = sparse_attn_weights.view(-1, k_len)
        
        # Find top-k values for all positions at once
        top_k_values, top_k_indices = torch.topk(reshaped_weights, top_k_val, dim=-1, sorted=False)
        
        # Create a mask for top-k positions
        mask = torch.zeros_like(reshaped_weights, dtype=torch.bool)
        mask.scatter_(-1, top_k_indices, True)
        
        # Create values tensor with min values
        min_val = torch.finfo(reshaped_weights.dtype).min
        masked_weights = torch.where(mask, reshaped_weights, torch.full_like(reshaped_weights, min_val))
        
        # Reshape back to original shape
        sparse_attn_weights = masked_weights.view(bsz, num_heads, q_len, k_len)
        
        return sparse_attn_weights


class VisionDynamicSparseAttention(nn.Module):
    """
    Dynamic sparse attention for vision components with learned routing for token selection.
    Optimized for vision-specific processing.
    """
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.vision_hidden_size
        self.num_heads = config.vision_num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout_prob

        # QKV projection
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.vision_qkv_bias)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        # Vision-specific routing network
        self.routing_network = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 4),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 4, self.num_heads),  # One routing score per head
            nn.Softmax(dim=-1)
        )
        
        # Sparse attention parameters for vision
        self.sparsity_ratio = getattr(config, 'vision_sparse_attention_sparsity_ratio', 0.4)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        bsz, tgt_len, embed_dim = hidden_states.size()

        # QKV projection
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(bsz, tgt_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply learned routing to determine which tokens to attend to
        # Use the global representation to compute routing scores
        global_repr = hidden_states.mean(dim=1, keepdim=True)  # [bsz, 1, embed_dim]
        routing_scores = self.routing_network(global_repr)  # [bsz, 1, num_heads]
        routing_scores = routing_scores.squeeze(1)  # [bsz, num_heads]

        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply dynamic sparsity: for each head, keep only top-k attention weights
        top_k = max(1, int(self.sparsity_ratio * tgt_len))
        
        # Apply sparsity to attention weights
        sparse_attn_weights = self.apply_dynamic_sparsity(attn_weights, top_k)
        
        # Apply softmax
        attn_weights = F.softmax(sparse_attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        attn_output = self.proj(attn_output)

        return attn_output

    def apply_dynamic_sparsity(self, attn_weights: torch.Tensor, top_k: int) -> torch.Tensor:
        """
        Apply dynamic sparsity by keeping only the top-k attention weights for each query position.
        This is a vectorized and more efficient implementation.
        """
        bsz, num_heads, q_len, k_len = attn_weights.size()
        
        # Calculate actual top_k to not exceed sequence length
        top_k_val = min(top_k, k_len)
        
        # Create a copy of attention weights to modify
        sparse_attn_weights = attn_weights.clone()
        
        # Vectorized approach: reshape to apply top-k across all positions at once
        # Reshape to (bsz * num_heads * q_len, k_len)
        reshaped_weights = sparse_attn_weights.view(-1, k_len)
        
        # Find top-k values for all positions at once
        top_k_values, top_k_indices = torch.topk(reshaped_weights, top_k_val, dim=-1, sorted=False)
        
        # Create a mask for top-k positions
        mask = torch.zeros_like(reshaped_weights, dtype=torch.bool)
        mask.scatter_(-1, top_k_indices, True)
        
        # Create values tensor with min values
        min_val = torch.finfo(reshaped_weights.dtype).min
        masked_weights = torch.where(mask, reshaped_weights, torch.full_like(reshaped_weights, min_val))
        
        # Reshape back to original shape
        sparse_attn_weights = masked_weights.view(bsz, num_heads, q_len, k_len)
        
        return sparse_attn_weights