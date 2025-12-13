"""
KV Cache Optimization with Multiple Strategies for Qwen3-VL model.
Implements adaptive strategy selection per layer/context with multiple optimization approaches.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union
import math


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class AdaptiveKVCacheStrategySelector(nn.Module):
    """
    Adaptive KV cache strategy selector that chooses the best strategy
    based on layer characteristics and context.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Strategy selection network
        self.strategy_selector = nn.Sequential(
            nn.Linear(self.hidden_size + 2, self.hidden_size // 4),  # +2 for layer_idx and seq_len
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, 4),  # [dense, low_rank, sliding_window, hybrid]
            nn.Softmax(dim=-1)
        )
        
        # Context analyzer for strategy selection
        self.context_analyzer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, 3),  # [compressibility, access_pattern, reuse_frequency]
            nn.Sigmoid()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        seq_len: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Select the best KV cache strategy for the given context.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            layer_idx: Current layer index
            seq_len: Current sequence length
            
        Returns:
            Tuple of (strategy_weights, strategy_info)
        """
        batch_size, _, _ = hidden_states.shape
        
        # Analyze context characteristics
        context_features = self.context_analyzer(hidden_states.mean(dim=1))  # [batch_size, 3]
        
        # Prepare features for strategy selection
        layer_info = torch.full((batch_size, 1), layer_idx, dtype=torch.float, device=hidden_states.device)
        seq_info = torch.full((batch_size, 1), seq_len, dtype=torch.float, device=hidden_states.device)
        
        combined_features = torch.cat([
            hidden_states.mean(dim=1),  # [batch_size, hidden_size]
            layer_info,
            seq_info
        ], dim=-1)  # [batch_size, hidden_size + 2]
        
        # Get strategy weights
        strategy_weights = self.strategy_selector(combined_features)  # [batch_size, 4]
        
        strategy_info = {
            'context_features': context_features,
            'layer_idx': layer_idx,
            'seq_len': seq_len,
            'strategy_weights': strategy_weights
        }
        
        return strategy_weights, strategy_info


class DenseKVCache(nn.Module):
    """
    Standard dense KV cache for comparison and fallback.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads or self.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        # Initialize KV cache
        self.k_cache = None
        self.v_cache = None
        self.current_seq_len = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_position: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the dense KV cache with new states.
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        if self.k_cache is None:
            # Initialize cache
            max_seq_len = self.max_position_embeddings
            self.k_cache = torch.zeros(
                batch_size, self.num_key_value_heads, max_seq_len, head_dim,
                dtype=key_states.dtype, device=key_states.device
            )
            self.v_cache = torch.zeros(
                batch_size, self.num_key_value_heads, max_seq_len, head_dim,
                dtype=value_states.dtype, device=value_states.device
            )
            self.current_seq_len = 0
        
        # Determine where to store new values
        if cache_position is not None:
            # Use provided cache positions
            if cache_position.numel() > 0:  # Check that cache_position is not empty
                self.k_cache[:, :, cache_position, :] = key_states
                self.v_cache[:, :, cache_position, :] = value_states
                self.current_seq_len = max(self.current_seq_len, cache_position.max().item() + 1)
        else:
            # Append to the end
            start_pos = self.current_seq_len
            end_pos = start_pos + seq_len
            self.k_cache[:, :, start_pos:end_pos, :] = key_states
            self.v_cache[:, :, start_pos:end_pos, :] = value_states
            self.current_seq_len = end_pos
        
        # Return the full cache up to current position
        return (self.k_cache[:, :, :self.current_seq_len, :], 
                self.v_cache[:, :, :self.current_seq_len, :])

    def get_seq_length(self) -> int:
        """Get the current sequence length in cache."""
        return self.current_seq_len

    def reset(self):
        """Reset the cache."""
        self.k_cache = None
        self.v_cache = None
        self.current_seq_len = 0


class LowRankKVCache(nn.Module):
    """
    Low-rank approximation KV cache for memory efficiency.
    """
    def __init__(self, config, layer_idx: Optional[int] = None, rank: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads or self.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rank = rank or min(64, self.head_dim)  # Default rank
        
        # Initialize low-rank decomposed caches
        self.k_left = None
        self.k_right = None
        self.v_left = None
        self.v_right = None
        self.current_seq_len = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_position: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the low-rank KV cache with new states.
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        if self.k_left is None:
            # Initialize low-rank caches
            max_seq_len = self.max_position_embeddings
            self.k_left = torch.zeros(
                batch_size, self.num_key_value_heads, max_seq_len, self.rank,
                dtype=key_states.dtype, device=key_states.device
            )
            self.k_right = torch.zeros(
                batch_size, self.num_key_value_heads, self.rank, head_dim,
                dtype=key_states.dtype, device=key_states.device
            )
            self.v_left = torch.zeros(
                batch_size, self.num_key_value_heads, max_seq_len, self.rank,
                dtype=value_states.dtype, device=value_states.device
            )
            self.v_right = torch.zeros(
                batch_size, self.num_key_value_heads, self.rank, head_dim,
                dtype=value_states.dtype, device=value_states.device
            )
            self.current_seq_len = 0
        
        # Determine where to store new values
        if cache_position is not None:
            # Use provided cache positions
            start_pos = cache_position[0].item() if cache_position.numel() > 0 else self.current_seq_len
            end_pos = start_pos + seq_len
        else:
            # Append to the end
            start_pos = self.current_seq_len
            end_pos = start_pos + seq_len
        
        # Update low-rank components (simplified SVD approximation)
        for head_idx in range(self.num_key_value_heads):
            k_head = key_states[:, head_idx, :, :]  # [batch, seq_len, head_dim]
            v_head = value_states[:, head_idx, :, :]  # [batch, seq_len, head_dim]
            
            # Compute low-rank approximation using a simple approach
            # In practice, you might use actual SVD or other decomposition methods
            k_left_update = k_head[:, :, :self.rank] if k_head.size(-1) >= self.rank else F.pad(k_head, (0, self.rank - k_head.size(-1)))
            v_left_update = v_head[:, :, :self.rank] if v_head.size(-1) >= self.rank else F.pad(v_head, (0, self.rank - v_head.size(-1)))
            
            self.k_left[:, head_idx, start_pos:end_pos, :] = k_left_update
            self.v_left[:, head_idx, start_pos:end_pos, :] = v_left_update
            
            # For right matrices, store original dimensions
            self.k_right[:, head_idx, :, :] = torch.eye(self.rank, head_dim, device=k_head.device, dtype=k_head.dtype)
            self.v_right[:, head_idx, :, :] = torch.eye(self.rank, head_dim, device=v_head.device, dtype=v_head.dtype)
        
        self.current_seq_len = max(self.current_seq_len, end_pos)
        
        # Reconstruct full KV tensors from low-rank approximation
        k_full = torch.matmul(self.k_left[:, :, :self.current_seq_len, :], self.k_right)
        v_full = torch.matmul(self.v_left[:, :, :self.current_seq_len, :], self.v_right)
        
        return k_full, v_full

    def get_seq_length(self) -> int:
        """Get the current sequence length in cache."""
        return self.current_seq_len

    def reset(self):
        """Reset the cache."""
        self.k_left = None
        self.k_right = None
        self.v_left = None
        self.v_right = None
        self.current_seq_len = 0


class SlidingWindowKVCache(nn.Module):
    """
    Sliding window KV cache to limit memory usage.
    """
    def __init__(self, config, layer_idx: Optional[int] = None, window_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads or self.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.window_size = window_size or getattr(config, 'kv_cache_window_size', 1024)
        
        # Initialize sliding window cache
        self.k_cache = None
        self.v_cache = None
        self.window_start = 0
        self.total_seq_len = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_position: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the sliding window KV cache with new states.
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        if self.k_cache is None:
            # Initialize sliding window cache
            self.k_cache = torch.zeros(
                batch_size, self.num_key_value_heads, self.window_size, head_dim,
                dtype=key_states.dtype, device=key_states.device
            )
            self.v_cache = torch.zeros(
                batch_size, self.num_key_value_heads, self.window_size, head_dim,
                dtype=value_states.dtype, device=value_states.device
            )
            self.window_start = 0
            self.total_seq_len = 0
        
        # Determine where to store new values in the sliding window
        if cache_position is not None:
            # For sliding window, we'll place values based on relative positions
            relative_positions = cache_position - self.window_start
            valid_mask = (relative_positions >= 0) & (relative_positions < self.window_size)
            
            if valid_mask.any():
                valid_pos = relative_positions[valid_mask]
                self.k_cache[:, :, valid_pos, :] = key_states[:, :, valid_mask, :]
                self.v_cache[:, :, valid_pos, :] = value_states[:, :, valid_mask, :]
        else:
            # Add new values to sliding window
            for i in range(seq_len):
                pos_in_window = (self.window_start + self.total_seq_len + i) % self.window_size
                self.k_cache[:, :, pos_in_window, :] = key_states[:, :, i, :]
                self.v_cache[:, :, pos_in_window, :] = value_states[:, :, i, :]
        
        # Update tracking variables
        self.total_seq_len += seq_len
        if self.total_seq_len > self.window_size:
            # Move window forward
            self.window_start = (self.window_start + seq_len) % self.window_size
        
        # Return the current window content
        start_idx = self.window_start
        end_idx = (self.window_start + min(self.total_seq_len, self.window_size)) % self.window_size

        if end_idx >= start_idx:
            k_window = self.k_cache[:, :, start_idx:end_idx, :]
            v_window = self.v_cache[:, :, start_idx:end_idx, :]
        else:
            # Handle wraparound case
            k_window = torch.cat([self.k_cache[:, :, start_idx:, :], self.k_cache[:, :, :end_idx, :]], dim=-2)
            v_window = torch.cat([self.v_cache[:, :, start_idx:, :], self.v_cache[:, :, :end_idx, :]], dim=-2)

        return k_window, v_window

    def get_seq_length(self) -> int:
        """Get the current effective sequence length in cache."""
        return min(self.total_seq_len, self.window_size)

    def reset(self):
        """Reset the cache."""
        self.k_cache = None
        self.v_cache = None
        self.window_start = 0
        self.total_seq_len = 0


class HybridKVCache(nn.Module):
    """
    Hybrid KV cache combining low-rank and sliding window approaches.
    """
    def __init__(self, config, layer_idx: Optional[int] = None, 
                 rank: Optional[int] = None, window_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.rank = rank or min(64, config.hidden_size // config.num_attention_heads)
        self.window_size = window_size or getattr(config, 'kv_cache_window_size', 1024)
        
        # Combine low-rank and sliding window approaches
        self.low_rank_cache = LowRankKVCache(config, layer_idx, rank=self.rank)
        self.sliding_window_cache = SlidingWindowKVCache(config, layer_idx, window_size=self.window_size)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_position: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the hybrid KV cache with new states.
        First apply sliding window, then low-rank approximation.
        """
        # First apply sliding window to limit sequence length
        k_windowed, v_windowed = self.sliding_window_cache.update(key_states, value_states, cache_position)
        
        # Then apply low-rank approximation
        k_low_rank, v_low_rank = self.low_rank_cache.update(k_windowed, v_windowed, cache_position)
        
        return k_low_rank, v_low_rank

    def get_seq_length(self) -> int:
        """Get the current sequence length in cache."""
        return self.low_rank_cache.get_seq_length()

    def reset(self):
        """Reset the cache."""
        self.low_rank_cache.reset()
        self.sliding_window_cache.reset()


class MultiStrategyKVCache(nn.Module):
    """
    Multi-strategy KV cache that adaptively selects the best strategy.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Initialize all strategy caches
        self.dense_cache = DenseKVCache(config, layer_idx)
        self.low_rank_cache = LowRankKVCache(config, layer_idx)
        self.sliding_window_cache = SlidingWindowKVCache(config, layer_idx)
        self.hybrid_cache = HybridKVCache(config, layer_idx)
        
        # Strategy selector
        self.strategy_selector = AdaptiveKVCacheStrategySelector(config)
        
        # Memory efficiency tracker
        self.memory_efficiency = 1.0

    def forward(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_position: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with adaptive strategy selection.
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        # Select the best strategy
        strategy_weights, strategy_info = self.strategy_selector(
            hidden_states, 
            self.layer_idx or 0, 
            seq_len
        )
        
        # Choose strategy based on weights
        selected_strategy = torch.argmax(strategy_weights, dim=-1)[0].item()  # Use first batch item
        
        # Apply selected strategy
        if selected_strategy == 0:  # Dense
            k_out, v_out = self.dense_cache.update(key_states, value_states, cache_position)
            strategy_name = 'dense'
        elif selected_strategy == 1:  # Low-rank
            k_out, v_out = self.low_rank_cache.update(key_states, value_states, cache_position)
            strategy_name = 'low_rank'
        elif selected_strategy == 2:  # Sliding window
            k_out, v_out = self.sliding_window_cache.update(key_states, value_states, cache_position)
            strategy_name = 'sliding_window'
        else:  # Hybrid
            k_out, v_out = self.hybrid_cache.update(key_states, value_states, cache_position)
            strategy_name = 'hybrid'
        
        # Calculate memory efficiency (simplified)
        original_memory = batch_size * num_heads * seq_len * head_dim
        if strategy_name == 'low_rank':
            compressed_memory = batch_size * num_heads * seq_len * self.low_rank_cache.rank + \
                               batch_size * num_heads * self.low_rank_cache.rank * head_dim * 2
            self.memory_efficiency = compressed_memory / original_memory
        elif strategy_name == 'sliding_window':
            self.memory_efficiency = min(seq_len, self.sliding_window_cache.window_size) / seq_len
        elif strategy_name == 'hybrid':
            # Combination of low-rank and sliding window efficiency
            self.memory_efficiency = min(
                (self.low_rank_cache.rank * 2) / head_dim,  # Low-rank efficiency
                self.sliding_window_cache.window_size / seq_len  # Window efficiency
            )
        else:
            self.memory_efficiency = 1.0  # Dense has no compression
        
        cache_info = {
            'strategy': strategy_name,
            'strategy_weights': strategy_weights,
            'selected_strategy': selected_strategy,
            'memory_efficiency': self.memory_efficiency,
            'seq_length': k_out.shape[-2]  # Length of output sequence
        }
        
        return k_out, v_out, cache_info


class OptimizedAttentionWithKVCache(nn.Module):
    """
    Attention mechanism with optimized KV caching strategies.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
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

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        self.rotary_emb = Qwen3VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        
        # Multi-strategy KV cache
        self.kv_cache = MultiStrategyKVCache(config, layer_idx)

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
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV caching with optimized strategies
        if use_cache:
            key_states, value_states, cache_info = self.kv_cache(
                key_states, value_states, hidden_states, cache_position
            )
            
            # For compatibility, return cache info as past_key_value
            past_key_value = (key_states, value_states, cache_info)
        elif past_key_value is not None:
            # Use existing cache
            if len(past_key_value) == 3:  # Includes cache_info
                key_states = torch.cat([past_key_value[0], key_states], dim=-2)
                value_states = torch.cat([past_key_value[1], value_states], dim=-2)
            else:
                key_states = torch.cat([past_key_value[0], key_states], dim=-2)
                value_states = torch.cat([past_key_value[1], value_states], dim=-2)

        # Repeat keys and values for GQA (Grouped Query Attention) if applicable
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)




class Qwen3VLRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)