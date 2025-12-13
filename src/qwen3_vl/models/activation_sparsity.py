"""
Activation sparsity and early exit mechanisms for Qwen3-VL model.
Implements Top-K activation sparsity, confidence-gated early exit mechanisms,
and input-adaptive routing for computational efficiency.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, Dict, Any
import math


class TopKSparsify(nn.Module):
    """
    Implements Top-K activation sparsity to reduce memory usage during inference.
    """
    def __init__(self, sparsity_ratio: float = 0.5, dim: int = -1):
        super().__init__()
        self.sparsity_ratio = sparsity_ratio
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Top-K sparsification to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Tensor with Top-K sparsification applied
        """
        # Calculate the number of elements to keep
        k = max(1, int(x.size(self.dim) * (1 - self.sparsity_ratio)))

        # Get the top-k values and indices
        top_k_values, top_k_indices = torch.topk(x.abs(), k, dim=self.dim, sorted=False)

        # Create a mask to zero out non-top-k elements
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(self.dim, top_k_indices, torch.ones_like(top_k_indices, dtype=torch.bool))

        # Apply the mask to the original tensor
        return x * mask.float()


class InputAdaptiveRouter(nn.Module):
    """
    Implements input-adaptive routing to determine if a layer should be processed
    based on input complexity. For simple inputs, layers can be skipped to improve efficiency.
    """
    def __init__(self, hidden_size: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size

        # Complexity estimator - learns to identify simple vs complex inputs
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

        # Threshold for determining if input is simple (lower = more aggressive skipping)
        self.register_buffer("skip_threshold", torch.tensor(0.3))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Determine if the current layer should be skipped based on input complexity.

        Args:
            hidden_states: Input hidden states to the layer

        Returns:
            Tuple of (hidden_states, should_skip_layer)
        """
        # Estimate input complexity
        # Use mean across sequence length to get a single representation per batch
        batch_complexity = hidden_states.mean(dim=1)  # Shape: (batch_size, hidden_size)
        complexity_score = self.complexity_estimator(batch_complexity).mean()  # Average across batch

        # Determine if layer should be skipped
        should_skip = complexity_score < self.skip_threshold

        return hidden_states, should_skip


class ConfidenceGatedEarlyExit(nn.Module):
    """
    Implements confidence-gated early exit mechanism at intermediate layers.
    """
    def __init__(self, hidden_size: int, num_layers: int, exit_threshold: float = 0.8):
        super().__init__()
        self.num_layers = num_layers
        self.exit_threshold = exit_threshold

        # Layer-specific exit classifiers
        self.exit_classifiers = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_layers)
        ])

    def forward(self, hidden_states: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, bool]:
        """
        Determine if the model should exit early based on confidence.

        Args:
            hidden_states: Hidden states from the current layer
            layer_idx: Current layer index

        Returns:
            Tuple of (hidden_states, should_exit)
        """
        if layer_idx >= self.num_layers - 1:
            # Always exit at the last layer
            return hidden_states, True

        # Compute confidence score for the current layer
        # Average over sequence and batch dimensions to get a single confidence value
        exit_logits = self.exit_classifiers[layer_idx](hidden_states.mean(dim=1).mean(dim=0, keepdim=True))  # Average over sequence and batch
        confidence = torch.sigmoid(exit_logits).squeeze(-1).item()

        # Check if confidence is above threshold
        should_exit = confidence >= self.exit_threshold

        return hidden_states, should_exit


class SparseMLP(nn.Module):
    """
    MLP with integrated sparsity mechanism.
    """
    def __init__(self, config, sparsity_ratio: float = 0.5):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.sparsity_ratio = sparsity_ratio

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

        # Sparsification layer
        self.sparsify = TopKSparsify(sparsity_ratio=sparsity_ratio, dim=-1)

        # Whether to use gradient checkpointing for memory efficiency
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

    def _compute_mlp(self, x: torch.Tensor) -> torch.Tensor:
        """Internal function to compute MLP, can be checkpointed."""
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        act = self.act_fn(gate)
        intermediate = act * up

        # Apply sparsification to intermediate values
        sparse_intermediate = self.sparsify(intermediate)

        output = self.down_proj(sparse_intermediate)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing during training to save memory
            return checkpoint(self._compute_mlp, x, use_reentrant=False)
        else:
            return self._compute_mlp(x)


class SparseAttention(nn.Module):
    """
    Attention mechanism with integrated sparsity.
    """
    def __init__(self, config, layer_idx: Optional[int] = None, sparsity_ratio: float = 0.3):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.sparsity_ratio = sparsity_ratio

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

        # Sparsification for attention weights
        self.sparsify = TopKSparsify(sparsity_ratio=sparsity_ratio, dim=-1)

        # Whether to use gradient checkpointing for memory efficiency
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

    def _compute_attention(self, query_states, key_states, value_states, attention_mask, cos, sin):
        """Internal function to compute attention, can be checkpointed."""
        import math

        # Apply rotary embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Apply sparsity to attention weights
        attn_weights = self.sparsify(attn_weights)

        # Upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output

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

        # Handle past key values for caching
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_position)

        # Repeat keys and values for GQA (Grouped Query Attention) if applicable
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Apply gradient checkpointing to attention computation if enabled
        if self.use_gradient_checkpointing and self.training:
            attn_output = checkpoint(
                self._compute_attention,
                query_states,
                key_states,
                value_states,
                attention_mask,
                cos,
                sin,
                use_reentrant=False
            )
        else:
            # Apply rotary embeddings and compute attention normally
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            import math
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # Apply sparsity to attention weights
            attn_weights = self.sparsify(attn_weights)

            # Upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class AdaptiveComputationLayer(nn.Module):
    """
    Transformer layer with adaptive computation, early exit mechanisms, and input-adaptive routing.
    """
    def __init__(self, config, layer_idx: int, sparsity_ratio: float = 0.5, exit_threshold: float = 0.8):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Use sparse attention and MLP
        self.self_attn = SparseAttention(config, layer_idx=layer_idx, sparsity_ratio=sparsity_ratio)
        self.mlp = SparseMLP(config, sparsity_ratio=sparsity_ratio)

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Early exit mechanism
        self.early_exit = ConfidenceGatedEarlyExit(config.hidden_size, config.num_hidden_layers, exit_threshold)

        # Input-adaptive routing
        self.router = InputAdaptiveRouter(config.hidden_size, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]], bool, bool]:
        # Check if this layer should be skipped based on input complexity
        hidden_states, should_skip = self.router(hidden_states)

        if should_skip:
            # If the layer should be skipped, return the input as output
            outputs = (hidden_states,)
            if output_attentions:
                outputs += (None,)  # No attention weights computed
            if use_cache:
                outputs += (past_key_value,)
            # Add both early exit flag (False - we're not exiting) and skip flag (True)
            outputs += (False, True)  # should_exit=False, was_skipped=True
            return outputs

        # Process normally if not skipped
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Check for early exit
        hidden_states, should_exit = self.early_exit(hidden_states, self.layer_idx)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # Add early exit flag and skip flag to outputs
        outputs += (should_exit, False)  # should_exit, was_skipped=False (because we processed the layer)

        return outputs


class EfficientTransformerBlock(nn.Module):
    """
    A complete transformer block that integrates sparsity, early exit, and input-adaptive routing
    with gradient checkpointing for memory efficiency.
    """
    def __init__(self, config, layer_idx: int, sparsity_ratio: float = 0.5, exit_threshold: float = 0.8):
        super().__init__()
        self.layer = AdaptiveComputationLayer(
            config=config,
            layer_idx=layer_idx,
            sparsity_ratio=sparsity_ratio,
            exit_threshold=exit_threshold
        )

        # Whether to use gradient checkpointing for this block
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]], bool, bool]:
        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing during training to save memory
            def create_custom_forward(module):
                def custom_forward(*inputs, **kwdargs):
                    return module(*inputs, **kwdargs)
                return custom_forward

            outputs = checkpoint(
                create_custom_forward(self.layer),
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                use_reentrant=False
            )
        else:
            outputs = self.layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        return outputs


# Helper functions for rotary embeddings and tensor operations
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