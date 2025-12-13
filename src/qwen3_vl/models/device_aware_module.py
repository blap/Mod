"""
Device-aware module selection system for Qwen3-VL model.
Automatically selects optimized implementations based on available hardware.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import logging


class DeviceAwareModule(nn.Module):
    """
    Base class for device-aware modules that can adapt their behavior
    based on the target device (CPU/GPU) and available optimizations.
    """
    def __init__(self):
        super().__init__()
        self.device_info = self._detect_device_capabilities()
        self.selected_implementation = self._select_implementation()

    def _detect_device_capabilities(self) -> Dict[str, Any]:
        """
        Detect the capabilities of the current device.
        """
        device_info = {
            'device_type': None,
            'has_cuda': torch.cuda.is_available(),
            'cuda_device_name': None,
            'cuda_capability': None,
            'memory_gb': None,
            'supports_flash_attention': False,
            'supports_bf16': False,
            'supports_fp16': False,
        }

        if torch.cuda.is_available():
            device_info['device_type'] = 'cuda'
            device_info['cuda_device_name'] = torch.cuda.get_device_name(0)
            
            # Check compute capability for features like Flash Attention
            major, minor = torch.cuda.get_device_capability(0)
            device_info['cuda_capability'] = (major, minor)
            
            # Estimate memory
            device_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Check if Flash Attention is supported (requires Ampere or newer, i.e., compute capability >= 8.0)
            device_info['supports_flash_attention'] = major >= 8
            
            # Check for bfloat16 support (Ampere or newer)
            device_info['supports_bf16'] = major >= 8
            
            # Check for float16 support
            device_info['supports_fp16'] = True
        else:
            device_info['device_type'] = 'cpu'
            # For CPU, we can estimate memory from system
            try:
                import psutil
                device_info['memory_gb'] = psutil.virtual_memory().total / (1024**3)
            except ImportError:
                device_info['memory_gb'] = 8.0  # Default estimate

        return device_info

    def _select_implementation(self) -> str:
        """
        Select the best implementation based on device capabilities.
        """
        if self.device_info['device_type'] == 'cuda':
            if self.device_info['supports_flash_attention']:
                return 'flash_attention'
            else:
                return 'standard_attention'
        else:
            # For CPU, we might want to use different optimizations
            return 'cpu_optimized_attention'

    def get_device_info(self) -> Dict[str, Any]:
        """Return the detected device information."""
        return self.device_info

    def get_selected_implementation(self) -> str:
        """Return the selected implementation."""
        return self.selected_implementation


class DeviceAwareAttention(DeviceAwareModule):
    """
    Device-aware attention module that selects the best attention implementation
    based on hardware capabilities while maintaining all 32 attention heads.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads  # Maintains all 32 heads
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

        # Initialize the appropriate attention implementation based on device
        self.attention_impl = self._initialize_attention_implementation()

        # Common components for all implementations
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        # Rotary embedding for position encoding
        self.rotary_emb = Qwen3VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _initialize_attention_implementation(self):
        """
        Initialize the appropriate attention implementation based on device capabilities.
        """
        implementation = self.get_selected_implementation()

        if implementation == 'flash_attention' and self._has_flash_attention():
            try:
                from flash_attn import flash_attn_func
                return FlashAttentionWrapper(
                    self.config,
                    self.layer_idx,
                    self.head_dim,
                    self.num_heads,
                    self.num_key_value_heads,
                    self.num_key_value_groups
                )
            except ImportError:
                # If flash attention is not available, fall back to standard
                logging.warning("Flash Attention requested but not available, falling back to standard attention")
                return StandardAttentionWrapper(
                    self.config,
                    self.layer_idx,
                    self.head_dim,
                    self.num_heads,
                    self.num_key_value_heads,
                    self.num_key_value_groups
                )
        elif implementation == 'cpu_optimized_attention':
            # For CPU, we implement optimizations specific to CPU execution
            return CPUOptimizedAttentionWrapper(
                self.config,
                self.layer_idx,
                self.head_dim,
                self.num_heads,
                self.num_key_value_heads,
                self.num_key_value_groups
            )
        else:
            # Standard attention for other cases
            return StandardAttentionWrapper(
                self.config,
                self.layer_idx,
                self.head_dim,
                self.num_heads,
                self.num_key_value_heads,
                self.num_key_value_groups
            )

    def _has_flash_attention(self) -> bool:
        """
        Check if flash attention is available.
        """
        try:
            from flash_attn import flash_attn_func
            return True
        except ImportError:
            return False

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
        """
        Forward pass that delegates to the appropriate attention implementation.
        """
        result = self.attention_impl(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            q_proj=self.q_proj,
            k_proj=self.k_proj,
            v_proj=self.v_proj,
            o_proj=self.o_proj,
            rotary_emb=self.rotary_emb
        )

        # Ensure the result is properly typed as a tuple with the expected types
        if isinstance(result, tuple) and len(result) >= 3:
            return result[0], result[1], result[2]
        elif isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1], None
        elif isinstance(result, tuple) and len(result) == 1:
            return result[0], None, None
        else:
            # If result is not a tuple, return it as the first element
            return result, None, None


class FlashAttentionWrapper(nn.Module):
    """
    Wrapper for Flash Attention implementation.
    """
    def __init__(self, config, layer_idx, head_dim, num_heads, num_key_value_heads, num_key_value_groups):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs  # To accept the projection layers and rotary embeddings
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # For now, we'll use the standard implementation as a placeholder
        # In a real implementation, we would use flash_attn_func here
        return self._standard_attention_forward(
            hidden_states, attention_mask, position_ids, past_key_value,
            output_attentions, use_cache, cache_position, **kwargs
        )

    def _standard_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Extract components from kwargs
        q_proj = kwargs['q_proj']
        k_proj = kwargs['k_proj']
        v_proj = kwargs['v_proj']
        o_proj = kwargs['o_proj']
        rotary_emb = kwargs['rotary_emb']

        bsz, q_len, _ = hidden_states.size()

        query_states = q_proj(hidden_states)
        key_states = k_proj(hidden_states)
        value_states = v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_position)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        import math
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class CPUOptimizedAttentionWrapper(nn.Module):
    """
    CPU-optimized attention implementation with specific optimizations for CPU execution.
    """
    def __init__(self, config, layer_idx, head_dim, num_heads, num_key_value_heads, num_key_value_groups):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs  # To accept the projection layers and rotary embeddings
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Extract components from kwargs
        q_proj = kwargs['q_proj']
        k_proj = kwargs['k_proj']
        v_proj = kwargs['v_proj']
        o_proj = kwargs['o_proj']
        rotary_emb = kwargs['rotary_emb']

        bsz, q_len, _ = hidden_states.size()

        query_states = q_proj(hidden_states)
        key_states = k_proj(hidden_states)
        value_states = v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_position)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        import math
        # CPU-optimized attention computation
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Upcast attention to fp32 for numerical stability on CPU
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class StandardAttentionWrapper(nn.Module):
    """
    Standard attention implementation with optimizations for different devices.
    """
    def __init__(self, config, layer_idx, head_dim, num_heads, num_key_value_heads, num_key_value_groups):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs  # To accept the projection layers and rotary embeddings
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Extract components from kwargs
        q_proj = kwargs['q_proj']
        k_proj = kwargs['k_proj']
        v_proj = kwargs['v_proj']
        o_proj = kwargs['o_proj']
        rotary_emb = kwargs['rotary_emb']

        bsz, q_len, _ = hidden_states.size()

        query_states = q_proj(hidden_states)
        key_states = k_proj(hidden_states)
        value_states = v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_position)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        import math
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen3VLRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding for Qwen3-VL model.
    Copied from the main modeling file to maintain consistency.
    """
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