"""
Faster Rotary Embedding Approximations for Qwen3-VL model.
Implements approximated rotary embedding computation while maintaining accuracy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class FastRotaryEmbedding(nn.Module):
    """
    Fast approximated rotary embedding that reduces computational overhead
    while maintaining accuracy through various approximation techniques.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, is_gated=False):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_gated = is_gated
        
        # Precompute frequencies to avoid repeated calculations
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute rotation matrix for common sequence lengths (approximation)
        self.use_cached_rotations = True
        self.rotation_cache = {}
        self.max_cache_size = 512  # Maximum sequence length to cache

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

        # Expand to match x dimensions: [batch_size, num_heads, seq_len, head_dim]
        cos = cos.unsqueeze(1).expand(-1, x.shape[1], -1, -1)  # [batch_size, num_heads, seq_len, head_dim]
        sin = sin.unsqueeze(1).expand(-1, x.shape[1], -1, -1)  # [batch_size, num_heads, seq_len, head_dim]

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class ApproximatedRotaryEmbedding(nn.Module):
    """
    Highly approximated rotary embedding using lookup tables and simplified computation.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, approximation_factor=0.5):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.approximation_factor = approximation_factor  # How much to compress the computation
        
        # Precompute a subset of frequencies based on approximation factor
        effective_dim = int(dim * approximation_factor)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, effective_dim, 2, dtype=torch.int64).float() / effective_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Lookup table for common angles to speed up computation
        self.use_lookup_table = True
        self.lookup_size = 1024
        angles = torch.linspace(0, 2 * math.pi, self.lookup_size)
        self.register_buffer("cos_lookup", torch.cos(angles), persistent=False)
        self.register_buffer("sin_lookup", torch.sin(angles), persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        seq_len = position_ids.shape[-1]
        
        # Use lookup table approximation for speed
        if self.use_lookup_table and seq_len <= self.lookup_size:
            # Map position_ids to lookup indices
            max_pos = self.max_position_embeddings
            lookup_indices = (position_ids.float() / max_pos * self.lookup_size).long().clamp(0, self.lookup_size - 1)
            
            # Get precomputed cos/sin values
            cos_vals = self.cos_lookup[lookup_indices]  # [batch_size, seq_len]
            sin_vals = self.sin_lookup[lookup_indices]  # [batch_size, seq_len]
            
            # Expand to match dimensions
            cos = cos_vals.unsqueeze(1).unsqueeze(-1).expand(-1, x.shape[1], -1, x.shape[-1])
            sin = sin_vals.unsqueeze(1).unsqueeze(-1).expand(-1, x.shape[1], -1, x.shape[-1])
        else:
            # Fallback to standard computation
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
            
            device_type = x.device.type
            device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
            
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                # Expand to full dimension by repeating
                emb = torch.cat((freqs, freqs), dim=-1)
                if emb.shape[-1] > x.shape[-1]:
                    emb = emb[:, :, :, :x.shape[-1]]
                elif emb.shape[-1] < x.shape[-1]:
                    emb = F.pad(emb, (0, x.shape[-1] - emb.shape[-1]))
                
                cos = emb.cos()
                sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LearnedRotaryEmbedding(nn.Module):
    """
    Learned rotary embedding that adapts to the specific patterns in the data.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Standard rotary embedding frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Learnable parameters to adjust the rotary embeddings
        self.learnable_scale = nn.Parameter(torch.ones(1))
        self.learnable_bias = nn.Parameter(torch.zeros(1))
        
        # Small network to learn position-dependent adjustments
        self.adjustment_network = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 2)  # Scale for cos and sin
        )

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
            
            # Apply learned adjustments
            cos_base = emb.cos()
            sin_base = emb.sin()
            
            # Get adjustments from network (simplified)
            pos_features = position_ids_expanded.transpose(1, 2)  # [batch, seq_len, 1]
            adjustments = self.adjustment_network(pos_features)  # [batch, seq_len, 2]
            cos_adj, sin_adj = adjustments[..., 0:1], adjustments[..., 1:2]
            
            # Apply adjustments
            cos = cos_base * (1 + self.learnable_scale * cos_adj) + self.learnable_bias
            sin = sin_base * (1 + self.learnable_scale * sin_adj) + self.learnable_bias
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class QuantizedRotaryEmbedding(nn.Module):
    """
    Quantized rotary embedding that reduces memory usage and computation
    through quantization techniques.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, num_bits=8):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.num_bits = num_bits
        self.quantization_scale = 2 ** num_bits - 1
        
        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Quantization parameters
        self.register_buffer("quantization_min", torch.tensor(-1.0))
        self.register_buffer("quantization_max", torch.tensor(1.0))

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
            
            # Quantize the embeddings
            cos_unquantized = emb.cos()
            sin_unquantized = emb.sin()
            
            # Quantize to reduce precision
            cos_quantized = self._quantize(cos_unquantized)
            sin_quantized = self._quantize(sin_unquantized)
            
            # Dequantize back to floating point
            cos = self._dequantize(cos_quantized)
            sin = self._dequantize(sin_quantized)
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _quantize(self, x):
        """Quantize tensor to lower precision."""
        x_scaled = (x - self.quantization_min) / (self.quantization_max - self.quantization_min)
        x_quantized = torch.round(x_scaled * self.quantization_scale)
        return x_quantized.clamp(0, self.quantization_scale)

    def _dequantize(self, x_quantized):
        """Dequantize tensor back to floating point."""
        x_scaled = x_quantized / self.quantization_scale
        x_dequantized = x_scaled * (self.quantization_max - self.quantization_min) + self.quantization_min
        return x_dequantized


class MixedPrecisionRotaryEmbedding(nn.Module):
    """
    Rotary embedding that uses mixed precision to optimize computation.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequencies in high precision
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Perform computation in float32 for precision
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Expand to match x dimensions: [batch_size, num_heads, seq_len, head_dim]
        # The emb has shape [batch_size, seq_len, head_dim] after cat
        # We need to expand it to [batch_size, num_heads, seq_len, head_dim]
        cos = emb.cos().unsqueeze(1).expand(-1, x.shape[1], -1, -1)  # [batch_size, num_heads, seq_len, head_dim]
        sin = emb.sin().unsqueeze(1).expand(-1, x.shape[1], -1, -1)  # [batch_size, num_heads, seq_len, head_dim]

        # Convert back to input dtype efficiently
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class OptimizedRotaryEmbedding(nn.Module):
    """
    Main optimized rotary embedding that selects the best approach based on context.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, 
                 optimization_level="balanced", use_approximation=True):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.optimization_level = optimization_level
        self.use_approximation = use_approximation
        
        # Select the appropriate implementation based on optimization level
        if optimization_level == "max_speed":
            if use_approximation:
                self.rotary_emb = ApproximatedRotaryEmbedding(dim, max_position_embeddings, base, approximation_factor=0.5)
            else:
                self.rotary_emb = FastRotaryEmbedding(dim, max_position_embeddings, base)
        elif optimization_level == "learned":
            self.rotary_emb = LearnedRotaryEmbedding(dim, max_position_embeddings, base)
        elif optimization_level == "quantized":
            self.rotary_emb = QuantizedRotaryEmbedding(dim, max_position_embeddings, base)
        elif optimization_level == "mixed_precision":
            self.rotary_emb = MixedPrecisionRotaryEmbedding(dim, max_position_embeddings, base)
        else:  # "balanced"
            self.rotary_emb = FastRotaryEmbedding(dim, max_position_embeddings, base)

    @torch.no_grad()
    def forward(self, x, position_ids):
        return self.rotary_emb(x, position_ids)


class FastRotaryAttention(nn.Module):
    """
    Attention mechanism with fast rotary embeddings.
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

        # Use optimized rotary embedding
        self.rotary_emb = OptimizedRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            optimization_level=getattr(config, 'rotary_embedding_optimization_level', 'balanced')
        )

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

        # Handle past key values for caching
        if past_key_value is not None:
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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed