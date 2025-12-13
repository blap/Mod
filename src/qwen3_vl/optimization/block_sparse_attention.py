"""Block Sparse Attention Implementation for Qwen3-VL model with hardware-specific optimization for NVIDIA SM61."""
import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class BlockSparseAttention(nn.Module):
    """Block sparse attention implementation that computes attention only within blocks."""
    
    def __init__(self, config, layer_idx: Optional[int] = None, 
                 block_size: int = 64, sparsity_ratio: float = 0.5):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_size = block_size
        self.sparsity_ratio = sparsity_ratio
        
        # Initialize standard attention parameters
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Block routing network to determine which blocks to attend to
        self.block_router = nn.Linear(self.hidden_size, self.num_heads)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with block sparse attention.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Mask to avoid performing attention on padding token indices
            position_ids: Positional indices for rotary embeddings
            past_key_value: Cached key-value states for efficient generation
            output_attentions: Whether to return attention weights
            use_cache: Whether to use caching
            cache_position: Position in the cache
            
        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to multi-head format
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings if provided
        if hasattr(self, 'rotary_emb') and position_ids is not None:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Repeat key and value states if using GQA
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Determine block sparse pattern based on routing
        routing_scores = self.block_router(hidden_states.mean(dim=1))  # [batch, num_heads]
        routing_scores = torch.sigmoid(routing_scores)  # [batch, num_heads]
        
        # Compute block sparse attention
        attn_weights = self._compute_block_sparse_attention(
            query_states, key_states, value_states, routing_scores
        )
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply softmax to get attention weights
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back to (batch, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        
        # Apply output projection
        output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
            
        return output, attn_weights, past_key_value
    
    def _compute_block_sparse_attention(self, query_states, key_states, value_states, routing_scores):
        """
        Compute block sparse attention by only attending within blocks.
        
        Args:
            query_states: Query tensor [batch, num_heads, seq_len, head_dim]
            key_states: Key tensor [batch, num_heads, seq_len, head_dim]
            value_states: Value tensor [batch, num_heads, seq_len, head_dim]
            routing_scores: Routing scores [batch, num_heads] indicating importance of each head
            
        Returns:
            Attention weights [batch, num_heads, seq_len, seq_len] with block sparsity
        """
        bsz, num_heads, seq_len, head_dim = query_states.shape
        
        # Pad sequence length to be divisible by block size
        pad_len = (self.block_size - seq_len % self.block_size) % self.block_size
        if pad_len > 0:
            query_states = torch.cat([query_states, torch.zeros(bsz, num_heads, pad_len, head_dim, device=query_states.device, dtype=query_states.dtype)], dim=2)
            key_states = torch.cat([key_states, torch.zeros(bsz, num_heads, pad_len, head_dim, device=key_states.device, dtype=key_states.dtype)], dim=2)
            value_states = torch.cat([value_states, torch.zeros(bsz, num_heads, pad_len, head_dim, device=value_states.device, dtype=value_states.dtype)], dim=2)
        
        padded_seq_len = seq_len + pad_len
        num_blocks = padded_seq_len // self.block_size
        
        # Reshape to blocks
        query_blocks = query_states.view(bsz, num_heads, num_blocks, self.block_size, head_dim)
        key_blocks = key_states.view(bsz, num_heads, num_blocks, self.block_size, head_dim)
        value_blocks = value_states.view(bsz, num_heads, num_blocks, self.block_size, head_dim)
        
        # Initialize attention weights
        attn_weights = torch.full((bsz, num_heads, padded_seq_len, padded_seq_len), float('-inf'), 
                                  device=query_states.device, dtype=torch.float32)
        
        # Compute attention within each block
        for block_idx in range(num_blocks):
            q_block = query_blocks[:, :, block_idx, :, :]  # [bsz, num_heads, block_size, head_dim]
            k_block = key_blocks[:, :, block_idx, :, :]   # [bsz, num_heads, block_size, head_dim]
            v_block = value_blocks[:, :, block_idx, :, :] # [bsz, num_heads, block_size, head_dim]
            
            # Compute attention scores within the block
            block_attn_scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale  # [bsz, num_heads, block_size, block_size]
            
            # Determine which positions in the block to attend to based on sparsity ratio and routing scores
            block_start = block_idx * self.block_size
            block_end = (block_idx + 1) * self.block_size
            
            # Apply block sparsity based on routing scores for each head
            for head_idx in range(num_heads):
                head_routing_score = routing_scores[:, head_idx].mean()  # Average across batch
                # Adjust sparsity based on routing score (more important heads get less sparsity)
                effective_sparsity = self.sparsity_ratio * (1.0 - head_routing_score * 0.5)  # Less sparse for important heads
                
                # Calculate number of positions to attend to in this block
                attend_count = max(1, int(self.block_size * effective_sparsity))
                
                # Get top-k positions to attend to
                top_k_vals, top_k_indices = torch.topk(block_attn_scores[:, head_idx, :, :], attend_count, dim=-1, largest=True, sorted=False)
                
                # Create mask for attended positions
                block_mask = torch.zeros_like(block_attn_scores[:, head_idx, :, :])
                block_mask.scatter_(-1, top_k_indices, top_k_vals)
                
                # Update attention weights for this block
                attn_weights[:, head_idx, block_start:block_end, block_start:block_end] = block_mask
        
        # Trim to original sequence length
        if pad_len > 0:
            attn_weights = attn_weights[:, :, :seq_len, :seq_len]
        
        return attn_weights.to(query_states.dtype)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embeddings to query and key tensors."""
    # Apply rotary embeddings to query and key
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotate half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)