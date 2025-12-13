"""
Consolidated Rotary Embeddings Module for Qwen3-VL Model
Combines rotary_embeddings.py, rotary_embedding_approximations.py, and related files
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
import numpy as np


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors."""
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    else:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3VLRotaryEmbedding(nn.Module):
    """
    Rotary Embedding implementation for Qwen3-VL model.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


class OptimizedRotaryEmbedding(nn.Module):
    """
    Hardware-optimized rotary embedding implementation.
    Includes approximations and caching for improved performance on specific hardware.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Pre-compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Create lookup table for common position IDs to accelerate computation
        self.use_lookup_table = True
        self.lookup_table_size = min(max_position_embeddings, 4096)
        self._create_lookup_table()

    def _create_lookup_table(self):
        """Create a lookup table for common position embeddings."""
        if self.use_lookup_table:
            position_ids = torch.arange(self.lookup_table_size, dtype=torch.float).unsqueeze(0)
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(1, -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
            
            # Compute frequencies
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            
            self.register_buffer("lookup_cos", cos.squeeze(0).to(torch.float16), persistent=False)
            self.register_buffer("lookup_sin", sin.squeeze(0).to(torch.float16), persistent=False)
        else:
            self.lookup_cos = None
            self.lookup_sin = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        
        # Use lookup table if available and position IDs are within range
        if (self.lookup_cos is not None and 
            position_ids.max() < self.lookup_table_size and 
            position_ids.min() >= 0):
            # Use precomputed embeddings for common positions
            cos = self.lookup_cos[position_ids].unsqueeze(1).expand(-1, x.shape[1], -1, -1)
            sin = self.lookup_sin[position_ids].unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        else:
            # Compute embeddings on-the-fly
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


class ApproximatedRotaryEmbedding(nn.Module):
    """
    Approximated rotary embedding using Taylor series or other approximation methods
    for faster computation on resource-constrained devices.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000,
                 approximation_method: str = "taylor"):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.approximation_method = approximation_method
        
        # Pre-compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Determine the best approximation based on hardware
        self.taylor_order = 3  # Order of Taylor series approximation

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Compute frequencies
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # Apply approximation based on method
            if self.approximation_method == "taylor":
                cos = self._taylor_cos(emb)
                sin = self._taylor_sin(emb)
            elif self.approximation_method == "piecewise_linear":
                cos = self._piecewise_linear_cos(emb)
                sin = self._piecewise_linear_sin(emb)
            else:  # Default to standard computation
                cos = emb.cos()
                sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _taylor_cos(self, x: torch.Tensor, order: int = 3) -> torch.Tensor:
        """Taylor series approximation of cosine function."""
        # cos(x) ≈ 1 - x²/2! + x⁴/4! - x⁶/6! + ...
        x_squared = x * x
        result = torch.ones_like(x)
        sign = -1
        
        factorial = 2  # Start with 2! = 2
        power = x_squared  # Start with x^2
        
        for i in range(1, order + 1):
            term = power / factorial
            result += sign * term
            sign *= -1
            power *= x_squared  # Next power is x^(2i+2)
            factorial *= (2*i + 1) * (2*i + 2)  # Next factorial is (2i+2)!
        
        return result

    def _taylor_sin(self, x: torch.Tensor, order: int = 3) -> torch.Tensor:
        """Taylor series approximation of sine function."""
        # sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7! + ...
        x_squared = x * x
        x_orig = x
        result = x_orig.clone()
        sign = -1
        
        factorial = 6  # Start with 3! = 6
        power = x_orig * x_squared  # Start with x^3
        
        for i in range(1, order + 1):
            term = power / factorial
            result += sign * term
            sign *= -1
            power *= x_squared  # Next power is x^(2i+3)
            factorial *= (2*i + 2) * (2*i + 3)  # Next factorial is (2i+3)!
        
        return result

    def _piecewise_linear_cos(self, x: torch.Tensor) -> torch.Tensor:
        """Piecewise linear approximation of cosine function."""
        # Normalize x to [-π, π] range
        x_norm = x % (2 * math.pi)
        x_norm = torch.where(x_norm > math.pi, x_norm - 2 * math.pi, x_norm)
        x_norm = torch.where(x_norm < -math.pi, x_norm + 2 * math.pi, x_norm)
        
        # Piecewise linear approximation
        abs_x = torch.abs(x_norm)
        cos_val = torch.where(
            abs_x <= math.pi / 2,
            1 - (2 / math.pi) * abs_x,  # Linear approximation near 0
            -(1 - (2 / math.pi) * (math.pi - abs_x))  # Linear approximation near π
        )
        
        return cos_val

    def _piecewise_linear_sin(self, x: torch.Tensor) -> torch.Tensor:
        """Piecewise linear approximation of sine function."""
        # Normalize x to [-π, π] range
        x_norm = x % (2 * math.pi)
        x_norm = torch.where(x_norm > math.pi, x_norm - 2 * math.pi, x_norm)
        x_norm = torch.where(x_norm < -math.pi, x_norm + 2 * math.pi, x_norm)
        
        # Piecewise linear approximation
        sin_val = torch.where(
            torch.abs(x_norm) <= math.pi / 2,
            (2 / math.pi) * x_norm,  # Linear approximation near 0
            torch.sign(x_norm) * (2 - (2 / math.pi) * torch.abs(x_norm))  # Near ±π/2
        )
        
        return sin_val


class CachedRotaryEmbedding(nn.Module):
    """
    Cached rotary embedding implementation that stores computed embeddings
    to avoid recomputation for common position IDs.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000,
                 cache_size: int = 4096):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.cache_size = cache_size
        
        # Pre-compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Initialize cache
        self.register_buffer(
            "cached_cos", 
            torch.zeros((cache_size, self.dim), dtype=torch.float16),
            persistent=False
        )
        self.register_buffer(
            "cached_sin", 
            torch.zeros((cache_size, self.dim), dtype=torch.float16),
            persistent=False
        )
        
        self.max_cached_position = -1
        self.cache_dtype = torch.float16

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        
        # Check if we need to expand the cache
        max_pos_id = position_ids.max().item()
        if max_pos_id >= self.cache_size:
            # For positions beyond cache, compute on-demand
            return self._compute_on_demand(x, position_ids)
        
        if max_pos_id > self.max_cached_position:
            # Compute embeddings for new positions and cache them
            self._update_cache(max_pos_id)
        
        # Retrieve from cache
        cos = self.cached_cos[position_ids].unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        sin = self.cached_sin[position_ids].unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        
        # Expand to match x's sequence length dimension
        seq_len = x.shape[2]
        if cos.shape[2] != seq_len or sin.shape[2] != seq_len:
            # If sequence length doesn't match cached length, we need to compute again
            return self._compute_on_demand(x, position_ids)
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _update_cache(self, max_pos: int):
        """Update the cache up to max_pos."""
        if max_pos <= self.max_cached_position:
            return  # Nothing to update

        # Compute embeddings for positions [max_cached_position+1, max_pos]
        new_positions = torch.arange(self.max_cached_position + 1, max_pos + 1, 
                                     dtype=torch.long, device=self.inv_freq.device)
        
        # Compute frequencies for new positions
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(len(new_positions), -1, 1)
        position_ids_expanded = new_positions[:, None, :].float()
        
        with torch.autocast(device_type="cpu", enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        
        # Store in cache
        self.cached_cos[self.max_cached_position + 1:max_pos + 1] = cos.squeeze(1).to(self.cache_dtype)
        self.cached_sin[self.max_cached_position + 1:max_pos + 1] = sin.squeeze(1).to(self.cache_dtype)
        
        self.max_cached_position = max_pos

    def _compute_on_demand(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute embeddings on demand when cache is insufficient."""
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class InterpolatedRotaryEmbedding(nn.Module):
    """
    Interpolated rotary embedding for handling sequences longer than the trained context.
    Useful for extending the effective context length beyond the model's original training.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000,
                 scaling_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Pre-compute inverse frequencies with scaling
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        
        # Apply scaling factor to position IDs to extend context
        scaled_position_ids = position_ids.float() / self.scaling_factor
        
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(scaled_position_ids.shape[0], -1, 1)
        position_ids_expanded = scaled_position_ids[:, None, :].float()
        
        # Force float32 since bfloat16 loses precision on long contexts
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class RotaryEmbeddingOptimizer:
    """
    Factory and optimizer for different types of rotary embeddings based on hardware and requirements.
    """
    @staticmethod
    def create_embedding(embedding_type: str, dim: int, max_position_embeddings: int = 2048, 
                         base: float = 10000, **kwargs) -> nn.Module:
        """
        Create the appropriate rotary embedding based on the requested type.
        
        Args:
            embedding_type: Type of embedding ('standard', 'optimized', 'approximated', 'cached', 'interpolated')
            dim: Dimension of the embedding
            max_position_embeddings: Maximum position embeddings
            base: Base value for RoPE calculation
            **kwargs: Additional arguments specific to embedding type
            
        Returns:
            Rotary embedding module
        """
        if embedding_type == 'standard':
            return Qwen3VLRotaryEmbedding(dim, max_position_embeddings, base)
        elif embedding_type == 'optimized':
            return OptimizedRotaryEmbedding(dim, max_position_embeddings, base)
        elif embedding_type == 'approximated':
            approximation_method = kwargs.get('approximation_method', 'taylor')
            return ApproximatedRotaryEmbedding(dim, max_position_embeddings, base, approximation_method)
        elif embedding_type == 'cached':
            cache_size = kwargs.get('cache_size', 4096)
            return CachedRotaryEmbedding(dim, max_position_embeddings, base, cache_size)
        elif embedding_type == 'interpolated':
            scaling_factor = kwargs.get('scaling_factor', 1.0)
            return InterpolatedRotaryEmbedding(dim, max_position_embeddings, base, scaling_factor)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

    @staticmethod
    def optimize_for_hardware(hardware_config: dict, dim: int, max_position_embeddings: int = 2048, 
                             base: float = 10000) -> nn.Module:
        """
        Create the optimal rotary embedding based on hardware configuration.
        
        Args:
            hardware_config: Dictionary containing hardware specifications
            dim: Dimension of the embedding
            max_position_embeddings: Maximum position embeddings
            base: Base value for RoPE calculation
            
        Returns:
            Optimized rotary embedding module
        """
        cpu_model = hardware_config.get('cpu_model', '').lower()
        memory_size = hardware_config.get('memory_size', 8 * 1024 * 1024 * 1024)  # 8GB default
        
        # For Intel i5-10210U with limited memory, use cached embeddings for efficiency
        if 'i5-10210u' in cpu_model:
            if memory_size < 16 * 1024 * 1024 * 1024:  # Less than 16GB RAM
                # Use cached embeddings to reduce computation
                return CachedRotaryEmbedding(dim, max_position_embeddings, base, cache_size=2048)
            else:
                # For systems with more memory, use optimized embeddings
                return OptimizedRotaryEmbedding(dim, max_position_embeddings, base)
        elif 'sm61' in hardware_config.get('gpu_model', '').lower():
            # For SM61 GPUs, use optimized embeddings that balance memory and compute
            return OptimizedRotaryEmbedding(dim, max_position_embeddings, base)
        else:
            # Default to standard implementation
            return Qwen3VLRotaryEmbedding(dim, max_position_embeddings, base)


class Qwen3VLAttentionWithRotary(nn.Module):
    """
    Attention module that integrates the rotary embeddings.
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

        # Initialize projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        # Initialize rotary embedding based on config
        rotary_embedding_type = getattr(config, 'rotary_embedding_type', 'standard')
        self.rotary_emb = RotaryEmbeddingOptimizer.create_embedding(
            rotary_embedding_type,
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            **{
                k.replace('rotary_embedding_', ''): v 
                for k, v in config.__dict__.items() 
                if k.startswith('rotary_embedding_')
            }
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Handle past key values for caching
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Repeat keys and values for GQA (Grouped Query Attention) if applicable
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def apply_rotary_pos_emb_for_vision(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, 
                                  sin: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding specifically optimized for vision transformers.
    Vision tokens often have different positional patterns than text.
    """
    # For vision, we may want to use a different approach to position IDs
    # since spatial positions are 2D rather than 1D sequential
    
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        # For vision, if no position_ids provided, we assume sequential ordering
        # In a full implementation, we would handle 2D spatial positions differently
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    """
    Rotary embedding specifically adapted for vision components of Qwen3-VL.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        # For vision, we may need to handle position IDs differently
        # The current implementation is similar to text, but could be extended
        # to handle 2D spatial positions
        
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


# Export the main classes and functions
__all__ = [
    'rotate_half',
    'apply_rotary_pos_emb', 
    'Qwen3VLRotaryEmbedding',
    'OptimizedRotaryEmbedding',
    'ApproximatedRotaryEmbedding', 
    'CachedRotaryEmbedding',
    'InterpolatedRotaryEmbedding',
    'RotaryEmbeddingOptimizer',
    'Qwen3VLAttentionWithRotary',
    'apply_rotary_pos_emb_for_vision',
    'Qwen3VLVisionRotaryEmbedding'
]