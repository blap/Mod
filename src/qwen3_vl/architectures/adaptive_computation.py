"""
Adaptive computation pathways for Qwen3-VL model.
Implements dynamic routing of computations based on input characteristics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class AdaptiveComputationGate(nn.Module):
    """
    Adaptive gate that determines how much computation to apply based on input features.
    Uses a learned gating mechanism to route inputs through different computation paths.
    """
    def __init__(self, hidden_size: int, path_count: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.path_count = path_count

        # Gating network to determine which computation path to use
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, path_count),
            nn.Softmax(dim=-1)
        )

        # Confidence threshold for routing decisions
        self.confidence_threshold = 0.6

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute gating weights and determine computation path.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Tuple of (gating_weights, selected_paths, low_confidence_mask)
        """
        # Compute global features for routing decision
        global_features = torch.mean(hidden_states, dim=1)  # Average across sequence length

        # Compute gating weights
        gating_weights = self.gate_network(global_features)  # Shape: (batch_size, path_count)

        # Determine the most confident path for each sample in the batch
        max_weights, selected_paths = torch.max(gating_weights, dim=-1)

        # For low-confidence cases, use all paths with weighted combination
        low_confidence_mask = max_weights < self.confidence_threshold
        return gating_weights, selected_paths, low_confidence_mask


class AdaptiveAttention(nn.Module):
    """
    Adaptive attention mechanism that selects different attention strategies
    based on input characteristics while maintaining all 32 attention heads.
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

        # Initialize different attention computation paths
        self.paths = nn.ModuleDict({
            'standard': StandardAttentionPath(config, layer_idx),
            'sparse': SparseAttentionPath(config, layer_idx),
            'linear': LinearAttentionPath(config, layer_idx)
        })

        # Adaptive gate to select computation path
        self.adaptive_gate = AdaptiveComputationGate(self.hidden_size, path_count=3)

        # Final output projection
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

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
        # Get gating decisions
        gating_weights, selected_paths, low_confidence_mask = self.adaptive_gate(hidden_states)

        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Initialize output tensor
        final_attn_output = torch.zeros_like(hidden_states)

        # Process each path separately based on gating decisions
        for path_idx, path_name in enumerate(self.paths.keys()):
            path_mask = (selected_paths == path_idx) & ~low_confidence_mask

            if path_mask.any():
                # Get the subset of hidden states for this path
                path_hidden_states = hidden_states[path_mask]
                path_attention_mask = attention_mask[path_mask] if attention_mask is not None else None
                path_position_ids = position_ids[path_mask] if position_ids is not None else None

                # Process through the selected path
                path_output, path_attn_weights, path_past_key_value = self.paths[path_name](
                    path_hidden_states,
                    attention_mask=path_attention_mask,
                    position_ids=path_position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position
                )

                # Apply path weight and place back in final output
                path_gating_weights = gating_weights[path_mask, path_idx].unsqueeze(-1).unsqueeze(-1)
                weighted_output = path_output * path_gating_weights
                final_attn_output[path_mask] = weighted_output

        # For low-confidence inputs, use weighted combination of all paths
        if low_confidence_mask.any():
            low_conf_hidden_states = hidden_states[low_confidence_mask]
            low_conf_attention_mask = attention_mask[low_confidence_mask] if attention_mask is not None else None
            low_conf_position_ids = position_ids[low_confidence_mask] if position_ids is not None else None

            # Compute outputs from all paths for low-confidence inputs
            all_path_outputs = []
            for path_idx, path_name in enumerate(self.paths.keys()):
                path_output, _, _ = self.paths[path_name](
                    low_conf_hidden_states,
                    attention_mask=low_conf_attention_mask,
                    position_ids=low_conf_position_ids,
                    past_key_value=past_key_value,
                    output_attentions=False,  # Don't output attentions for combined paths
                    use_cache=use_cache,
                    cache_position=cache_position
                )

                # Apply path-specific gating weight
                path_gating_weights = gating_weights[low_confidence_mask, path_idx].unsqueeze(-1).unsqueeze(-1)
                weighted_path_output = path_output * path_gating_weights
                all_path_outputs.append(weighted_path_output)

            # Sum the weighted outputs
            combined_output = sum(all_path_outputs)
            final_attn_output[low_confidence_mask] = combined_output

        # Apply final output projection
        final_output = self.o_proj(final_attn_output)

        if not output_attentions:
            attn_weights = None
        else:
            # For now, return None for attentions in adaptive case
            attn_weights = None

        return final_output, attn_weights, past_key_value


class StandardAttentionPath(nn.Module):
    """
    Standard attention implementation path.
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

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)

        self.rotary_emb = Qwen3VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
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

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(value_states, position_ids)
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
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class SparseAttentionPath(nn.Module):
    """
    Sparse attention implementation path for efficiency.
    Uses sparse attention patterns to reduce computation.
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

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)

        self.rotary_emb = Qwen3VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        
        # Sparse attention parameters
        self.sparse_factor = 0.25  # Use 25% of full attention

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

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_position)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        import math
        # Compute sparse attention - only attend to a subset of keys
        sparse_k_len = max(1, int(key_states.size(2) * self.sparse_factor))
        sparse_key_states = key_states[:, :, :sparse_k_len, :]
        sparse_value_states = value_states[:, :, :sparse_k_len, :]
        
        attn_weights = torch.matmul(query_states, sparse_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : sparse_k_len]
            attn_weights = attn_weights + causal_mask

        # Upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, sparse_value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LinearAttentionPath(nn.Module):
    """
    Linear attention implementation path using kernel approximation.
    Implements Performer-style linear attention.
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

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)

        self.rotary_emb = Qwen3VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        
        # Parameters for the kernel approximation
        self.feature_dim = self.head_dim  # Feature dimension for kernel approximation
        
        # Random matrix for feature mapping (orthogonal random features)
        self.register_buffer(
            "unstructured_random_matrix", 
            torch.randn(self.feature_dim // 2, self.head_dim)
        )

    def _get_random_features(self, query_prime):
        """
        Compute random features for the query using orthogonal random features.
        """
        # Apply the random matrix to the query
        rand_proj = torch.einsum("...d,ed->...e", query_prime, self.unstructured_random_matrix)
        
        # Apply sine and cosine to create orthogonal features
        # This approximates the softmax kernel
        return torch.cat([torch.sin(rand_proj), torch.cos(rand_proj)], dim=-1)

    def _softmax_kernel(self, x, is_query=False, normalize_data=True):
        """
        Compute softmax kernel approximation using random features.
        """
        # Normalize the input if needed
        if normalize_data:
            x = x / torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        
        # Get random features
        if is_query:
            # For queries, apply normalization factor
            return self._get_random_features(x) / math.sqrt(self.feature_dim)
        else:
            # For keys, apply normalization factor
            return self._get_random_features(x) / math.sqrt(self.feature_dim)

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

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_position)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Apply linear attention using kernel approximation
        # Instead of computing full attention matrix (O(n^2)), we use random feature approximation
        # This gives us O(n) complexity while maintaining the ability to use all 32 heads
        
        # Normalize query and key states for stable computation
        query_states = query_states / math.sqrt(math.sqrt(self.head_dim))
        key_states = key_states / math.sqrt(math.sqrt(self.head_dim))
        
        # Apply softmax kernel approximation to queries and keys
        query_prime = self._softmax_kernel(query_states, is_query=True)
        key_prime = self._softmax_kernel(key_states, is_query=False)
        
        # Compute linear attention: (Q' @ K') @ V instead of Q @ K^T @ V
        # This is the key to achieving linear complexity
        key_value = torch.einsum("bhld,bhlv->bhdv", key_prime, value_states)
        attn_output = torch.einsum("bhld,bhdv->bhlv", query_prime, key_value)
        
        # Reshape back to original format
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class AdaptiveMLP(nn.Module):
    """
    Adaptive MLP that selects different computation paths based on input characteristics.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Different computation paths
        self.paths = nn.ModuleDict({
            'full': nn.Sequential(
                nn.Linear(self.hidden_size, self.intermediate_size),
                nn.SiLU(),
                nn.Linear(self.intermediate_size, self.hidden_size)
            ),
            'compressed': nn.Sequential(
                nn.Linear(self.hidden_size, self.intermediate_size // 2),
                nn.SiLU(),
                nn.Linear(self.intermediate_size // 2, self.hidden_size)
            ),
            'sparse': nn.Sequential(
                nn.Linear(self.hidden_size, self.intermediate_size),
                nn.SiLU(),
                nn.Dropout(0.1),  # Add some sparsity
                nn.Linear(self.intermediate_size, self.hidden_size)
            )
        })
        
        # Adaptive gate to select computation path
        self.adaptive_gate = AdaptiveComputationGate(self.hidden_size, path_count=3)
        
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Get gating decisions
        gating_weights, selected_paths, low_confidence_mask = self.adaptive_gate(hidden_states)

        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Initialize output tensor
        final_output = torch.zeros_like(hidden_states)

        # Process each path separately based on gating decisions
        for path_idx, path_name in enumerate(self.paths.keys()):
            path_mask = (selected_paths == path_idx) & ~low_confidence_mask

            if path_mask.any():
                # Get the subset of hidden states for this path
                path_hidden_states = hidden_states[path_mask]

                # Process through the selected path
                path_output = self.paths[path_name](path_hidden_states)

                # Apply path weight and place back in final output
                path_gating_weights = gating_weights[path_mask, path_idx].unsqueeze(-1).unsqueeze(-1)
                weighted_output = path_output * path_gating_weights
                final_output[path_mask] = weighted_output

        # For low-confidence inputs, use weighted combination of all paths
        if low_confidence_mask.any():
            low_conf_hidden_states = hidden_states[low_confidence_mask]

            # Compute outputs from all paths for low-confidence inputs
            all_path_outputs = []
            for path_idx, path_name in enumerate(self.paths.keys()):
                path_output = self.paths[path_name](low_conf_hidden_states)

                # Apply path-specific gating weight
                path_gating_weights = gating_weights[low_confidence_mask, path_idx].unsqueeze(-1).unsqueeze(-1)
                weighted_path_output = path_output * path_gating_weights
                all_path_outputs.append(weighted_path_output)

            # Sum the weighted outputs
            combined_output = sum(all_path_outputs)
            final_output[low_confidence_mask] = combined_output

        return final_output


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