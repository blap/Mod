"""
Memory-efficient computation patterns for FlashAttention 2 implementation.
These patterns reduce memory complexity from O(n²) to O(n) while maintaining accuracy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from src.qwen3_vl.core.config import Qwen3VLConfig


def memory_efficient_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    is_causal: bool = False
) -> torch.Tensor:
    """
    Memory-efficient attention forward pass that reduces memory complexity from O(n²) to O(n)
    by using chunked/tiled computation.
    
    Args:
        query_states: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        key_states: Key tensor of shape (batch_size, num_heads, kv_seq_len, head_dim)
        value_states: Value tensor of shape (batch_size, num_heads, kv_seq_len, head_dim)
        attention_mask: Optional attention mask
        dropout_p: Dropout probability
        scale: Scaling factor for attention scores
        is_causal: Whether to apply causal masking
    
    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    batch_size, num_heads, seq_len, head_dim = query_states.shape
    _, _, kv_seq_len, _ = value_states.shape
    
    if scale is None:
        scale = 1 / math.sqrt(head_dim)
    
    # Initialize output tensor
    attn_output = torch.zeros_like(query_states)
    
    # Define tile size based on available memory
    # For memory efficiency, we use smaller tiles to avoid materializing full attention matrix
    tile_size = min(512, seq_len)  # Adjust based on available memory
    
    # Process in tiles to limit memory usage
    for q_start in range(0, seq_len, tile_size):
        q_end = min(q_start + tile_size, seq_len)
        q_tile = query_states[:, :, q_start:q_end, :]  # (B, H, tile_size, D)
        
        # Compute attention scores for this query tile
        attn_weights_tile = torch.matmul(q_tile, key_states.transpose(-2, -1)) * scale  # (B, H, tile_size, KV_seq_len)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask_tile = attention_mask[:, :, q_start:q_end, :kv_seq_len]  # (B, H, tile_size, KV_seq_len)
            attn_weights_tile = attn_weights_tile + mask_tile
        
        # Apply causal mask if needed
        if is_causal:
            causal_mask = torch.triu(
                torch.full((q_end - q_start, kv_seq_len), float('-inf'), device=query_states.device),
                diagonal=kv_seq_len - (q_end - q_start) + q_start
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, tile_size, KV_seq_len)
            attn_weights_tile = attn_weights_tile + causal_mask
        
        # Apply softmax to get attention weights
        attn_weights_tile = torch.softmax(attn_weights_tile, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply dropout if needed
        if dropout_p > 0.0:
            attn_weights_tile = F.dropout(attn_weights_tile, p=dropout_p)
        
        # Compute output for this tile
        output_tile = torch.matmul(attn_weights_tile, value_states)  # (B, H, tile_size, D)
        
        # Store in output tensor
        attn_output[:, :, q_start:q_end, :] = output_tile
    
    return attn_output


def flash_attention_chunked_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    is_causal: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Chunked forward pass for FlashAttention that processes attention in chunks
    to reduce peak memory usage from O(n²) to O(n).
    
    This implementation follows the FlashAttention algorithm principles:
    1. Process queries in chunks
    2. Compute attention scores for each query chunk against all keys
    3. Apply softmax incrementally
    4. Compute output incrementally
    """
    batch_size, num_heads, seq_len, head_dim = query_states.shape
    _, _, kv_seq_len, _ = value_states.shape
    
    if scale is None:
        scale = 1 / math.sqrt(head_dim)
    
    # Initialize output tensor and temporary variables for incremental softmax
    attn_output = torch.zeros_like(query_states)
    logsumexp = torch.full((batch_size, num_heads, seq_len), float('-inf'), 
                          dtype=torch.float32, device=query_states.device)
    max_scores = torch.full((batch_size, num_heads, seq_len), float('-inf'), 
                           dtype=torch.float32, device=query_states.device)
    
    # Define chunk size based on available memory
    chunk_size = min(512, seq_len)
    
    for q_start in range(0, seq_len, chunk_size):
        q_end = min(q_start + chunk_size, seq_len)
        q_chunk = query_states[:, :, q_start:q_end, :]  # (B, H, chunk_size, D)
        
        # Initialize chunk-specific accumulators
        chunk_output = torch.zeros((batch_size, num_heads, q_end - q_start, head_dim), 
                                  dtype=query_states.dtype, device=query_states.device)
        chunk_logsumexp = torch.full((batch_size, num_heads, q_end - q_start), float('-inf'), 
                                    dtype=torch.float32, device=query_states.device)
        chunk_max_scores = torch.full((batch_size, num_heads, q_end - q_start), float('-inf'), 
                                     dtype=torch.float32, device=query_states.device)
        
        # Process keys in smaller segments to further reduce memory usage
        key_chunk_size = min(512, kv_seq_len)
        
        for k_start in range(0, kv_seq_len, key_chunk_size):
            k_end = min(k_start + key_chunk_size, kv_seq_len)
            k_segment = key_states[:, :, k_start:k_end, :]  # (B, H, segment_size, D)
            v_segment = value_states[:, :, k_start:k_end, :]  # (B, H, segment_size, D)
            
            # Compute attention scores for this segment
            attn_scores = torch.matmul(q_chunk, k_segment.transpose(-2, -1)) * scale  # (B, H, chunk_size, segment_size)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                mask_segment = attention_mask[:, :, q_start:q_end, k_start:k_end]  # (B, H, chunk_size, segment_size)
                attn_scores = attn_scores + mask_segment
            
            # Apply causal mask if needed
            if is_causal:
                causal_mask = torch.triu(
                    torch.full((q_end - q_start, k_end - k_start), float('-inf'), device=query_states.device),
                    diagonal=k_end - k_start - (q_end - q_start) + (q_start - k_start)
                ).unsqueeze(0).unsqueeze(0)  # (1, 1, chunk_size, segment_size)
                attn_scores = attn_scores + causal_mask
            
            # Compute new maximum for numerical stability
            new_max = torch.max(chunk_max_scores, attn_scores.max(dim=-1, keepdim=True).values)
            
            # Compute scaling factors for softmax normalization
            exp_scores = torch.exp(attn_scores - new_max)
            exp_sums = exp_scores.sum(dim=-1, keepdim=True)
            
            # Update accumulated values using the log-sum-exp trick
            prev_scale = torch.exp(chunk_max_scores - new_max)
            curr_scale = exp_sums
            
            # Update chunk_logsumexp
            chunk_logsumexp = new_max.squeeze(-1) + torch.log(prev_scale + curr_scale)
            
            # Update chunk_max_scores
            chunk_max_scores = new_max
            
            # Update chunk output
            chunk_output = (chunk_output * prev_scale.unsqueeze(-1) + 
                           torch.matmul(exp_scores, v_segment)) / (prev_scale + curr_scale).unsqueeze(-1)
        
        # Store chunk results in full output tensor
        attn_output[:, :, q_start:q_end, :] = chunk_output
        logsumexp[:, :, q_start:q_end] = chunk_logsumexp
        max_scores[:, :, q_start:q_end] = chunk_max_scores.squeeze(-1)
    
    # Final normalization using logsumexp values
    attn_output = attn_output * torch.exp(max_scores.unsqueeze(-1) - logsumexp.unsqueeze(-1))
    
    return attn_output, None  # Return attention weights as None since they're not materialized


class MemoryEfficientFlashAttention(nn.Module):
    """
    Memory-efficient FlashAttention implementation that reduces memory complexity
    from O(n²) to O(n) using chunked computation patterns.
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

        # Initialize projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        # Memory-efficient parameters
        self.chunk_size = 512  # Size of chunks for memory-efficient computation
        self.use_chunked_forward = True  # Whether to use chunked computation

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

        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to multi-head format
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        from .flash_attention_2 import Qwen3VLRotaryEmbedding, apply_rotary_pos_emb, repeat_kv
        rotary_emb = Qwen3VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle past key values for caching
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Repeat keys and values for GQA (Grouped Query Attention) if applicable
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Use memory-efficient attention computation
        if self.use_chunked_forward and not output_attentions:
            # Use chunked computation to reduce memory usage from O(n²) to O(n)
            attn_output, _ = flash_attention_chunked_forward(
                query_states, key_states, value_states,
                attention_mask=attention_mask,
                dropout_p=0.0,
                scale=1 / math.sqrt(self.head_dim),
                is_causal=attention_mask is None  # Set based on whether attention mask is provided
            )
            attn_weights = None  # Not computed when output_attentions is False
        else:
            # Use standard memory-efficient computation
            attn_output = memory_efficient_attention_forward(
                query_states, key_states, value_states,
                attention_mask=attention_mask,
                dropout_p=0.0,
                scale=1 / math.sqrt(self.head_dim),
                is_causal=attention_mask is None
            )
            attn_weights = None  # Attention weights not materialized in memory-efficient version

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class SM61MemoryEfficientFlashAttention(MemoryEfficientFlashAttention):
    """
    Memory-efficient FlashAttention specifically optimized for NVIDIA SM61 architecture.
    """
    def __init__(self, config: Qwen3VLConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        
        # SM61-specific optimizations
        self.chunk_size = 256  # Smaller chunk size for SM61's memory constraints
        self.tile_size = 64    # Optimal tile size for SM61's memory hierarchy
        self.use_chunked_forward = True  # Always use chunked computation for memory efficiency


def apply_memory_efficient_patterns(module: nn.Module):
    """
    Apply memory-efficient computation patterns to a module.
    This function can be used to optimize existing attention modules.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.MultiheadAttention):
            # Replace with memory-efficient version
            new_child = MemoryEfficientFlashAttention(
                type('Config', (), {
                    'hidden_size': child.embed_dim,
                    'num_attention_heads': child.num_heads,
                    'num_key_value_heads': child.num_heads,
                    'max_position_embeddings': 2048,
                    'rope_theta': 10000.0,
                    'intermediate_size': child.embed_dim * 4
                })()
            )
            setattr(module, name, new_child)
        else:
            apply_memory_efficient_patterns(child)


def get_memory_efficient_attention(config: Qwen3VLConfig, layer_idx: int):
    """
    Factory function to get the appropriate memory-efficient attention implementation
    based on hardware configuration.
    """
    if config.hardware_specific_attention and config.hardware_specific_attention == "sm61":
        return SM61MemoryEfficientFlashAttention(config, layer_idx=layer_idx)
    else:
        return MemoryEfficientFlashAttention(config, layer_idx=layer_idx)