"""
Utility functions for tensor operations in the Qwen3-VL model.

This module contains commonly used tensor utility functions that were previously
duplicated across multiple files. Centralizing these functions improves code
maintainability, reduces redundancy, and ensures consistent behavior across the
codebase.

The functions here are focused on operations commonly used in transformer models,
particularly attention mechanisms and rotary embeddings.
"""

import torch
from typing import Optional, Tuple, Union
from torch import nn
import torch.nn.functional as F


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value tensors to match the number of query heads in multi-head attention.

    This function is used in Grouped-Query Attention (GQA) where the number of key-value
    heads is smaller than the number of query heads. It expands the key-value tensors
    to match the query tensor dimensions by repeating along the head dimension.

    The operation transforms the hidden states from:
    (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)

    Where num_attention_heads = num_key_value_heads * n_rep.

    Args:
        hidden_states: Input tensor of shape (batch, num_key_value_heads, seqlen, head_dim)
        n_rep: Number of times to repeat each key-value head to match query heads

    Returns:
        Expanded tensor of shape (batch, num_attention_heads, seqlen, head_dim)

    Raises:
        ValueError: If n_rep is not a positive integer
        TypeError: If hidden_states is not a torch.Tensor

    Example:
        >>> query_heads = 8
        >>> kv_heads = 2
        >>> repeat_factor = query_heads // kv_heads  # 4
        >>> kv_tensor = torch.randn(1, 2, 100, 64)  # (batch, kv_heads, seq_len, head_dim)
        >>> expanded = repeat_kv(kv_tensor, repeat_factor)
        >>> print(expanded.shape)  # torch.Size([1, 8, 100, 64])
    """
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError(f"hidden_states must be a torch.Tensor, got {type(hidden_states)}")
    if not isinstance(n_rep, int) or n_rep <= 0:
        raise ValueError(f"n_rep must be a positive integer, got {n_rep}")

    # Handle the case where we don't need to repeat (n_rep = 1)
    if n_rep == 1:
        return hidden_states

    batch, num_key_value_heads, slen, head_dim = hidden_states.shape

    # Expand and reshape to repeat the key-value heads
    # First expand to insert repetition dimension
    expanded = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # Then reshape to merge the heads and repetition dimensions
    return expanded.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dimensions of the input tensor.

    This function is used in Rotary Position Embeddings (RoPE) to implement
    rotation-based positional encoding. It rotates the vector by swapping
    and negating elements in the two halves of the hidden dimensions.

    The rotation is implemented as:
    - x1 = x[... : x.shape[-1] // 2]
    - x2 = x[..., x.shape[-1] // 2 :]
    - output = torch.cat((-x2, x1), dim=-1)

    Args:
        x: Input tensor of shape (*, hidden_dim) where the last dimension will be rotated

    Returns:
        Tensor with the same shape as input but with rotated last dimension

    Raises:
        TypeError: If x is not a torch.Tensor

    Example:
        >>> x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        >>> rotated = rotate_half(x)
        >>> print(rotated)  # tensor([-3., -4.,  1.,  2.])
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)}")

    # Split the last dimension into two halves
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    # Rotate by swapping and negating: [-x2, x1]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embedding (RoPE) to query and key tensors.

    This function implements the RoPE mechanism which applies rotation-based
    positional encoding to query and key tensors in transformer attention.

    The RoPE formula is:
    - q_rot = q * cos + rotate_half(q) * sin
    - k_rot = k * cos + rotate_half(k) * sin

    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim) or similar
        k: Key tensor of shape (batch, heads, seq_len, head_dim) or similar
        cos: Cosine values from positional embedding of shape compatible with q/k
        sin: Sine values from positional embedding of shape compatible with q/k
        unsqueeze_dim: Dimension to unsqueeze cos/sin tensors (default: 1)

    Returns:
        Tuple of (rotated_query, rotated_key) tensors with RoPE applied

    Raises:
        TypeError: If inputs are not torch.Tensor
        ValueError: If tensor shapes are incompatible

    Example:
        >>> batch_size, heads, seq_len, head_dim = 1, 8, 100, 64
        >>> q = torch.randn(batch_size, heads, seq_len, head_dim)
        >>> k = torch.randn(batch_size, heads, seq_len, head_dim)
        >>> cos = torch.randn(1, 1, seq_len, head_dim)  # from RoPE
        >>> sin = torch.randn(1, 1, seq_len, head_dim)  # from RoPE
        >>> q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        >>> print(q_rot.shape, k_rot.shape)  # torch.Size([1, 8, 100, 64]) torch.Size([1, 8, 100, 64])
    """
    if not all(isinstance(t, torch.Tensor) for t in [q, k, cos, sin]):
        raise TypeError("All inputs must be torch.Tensor objects")
    if not isinstance(unsqueeze_dim, int):
        raise TypeError(f"unsqueeze_dim must be an int, got {type(unsqueeze_dim)}")

    # Ensure cos and sin have the right shape by unsqueezing the specified dimension
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Apply RoPE: q * cos + rotate_half(q) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def apply_rotary_pos_emb_with_position_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embedding with position IDs indexing for flexible RoPE application.

    This enhanced version of RoPE application allows for flexible positional encoding
    by using position_ids to index into the cos/sin tensors, enabling efficient processing
    of packed sequences or sequences with complex positional patterns.

    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim)
        k: Key tensor of shape (batch, heads, seq_len, head_dim)
        cos: Cosine embedding tensor of shape (batch, head_dim/2, max_seq_len)
        sin: Sine embedding tensor of shape (batch, head_dim/2, max_seq_len)
        position_ids: Position indices of shape (batch, seq_len) to index embeddings
        unsqueeze_dim: Dimension to unsqueeze cos/sin tensors (default: 1)

    Returns:
        Tuple of (rotated_query, rotated_key) tensors with RoPE applied

    Raises:
        TypeError: If inputs are not torch.Tensor (except position_ids which can be None)
        ValueError: If tensor shapes are incompatible
    """
    if not all(isinstance(t, torch.Tensor) for t in [q, k, cos, sin] if t is not None):
        raise TypeError("q, k, cos, sin must be torch.Tensor objects")
    if position_ids is not None and not isinstance(position_ids, torch.Tensor):
        raise TypeError(f"position_ids must be a torch.Tensor or None, got {type(position_ids)}")
    if not isinstance(unsqueeze_dim, int):
        raise TypeError(f"unsqueeze_dim must be an int, got {type(unsqueeze_dim)}")

    # If position_ids is provided, index the cos/sin tensors
    if position_ids is not None:
        # Expand position_ids to match cos/sin dimensions for indexing
        inv_freq_expanded = cos  # Already precomputed
        position_ids_expanded = position_ids.unsqueeze(-1).expand(-1, -1, cos.size(-1))

        # Index into the cos/sin tensors using position_ids
        cos = torch.gather(cos, -1, position_ids_expanded)
        sin = torch.gather(sin, -1, position_ids_expanded)

    # Apply RoPE using the standard function
    return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim)


def compute_attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: Optional[float] = None,
    use_flash_attention: bool = False
) -> torch.Tensor:
    """
    Compute attention scores between query and key tensors with optional scaling.

    This function provides a unified interface for computing attention scores,
    with optional support for flash attention for memory efficiency.

    Args:
        query: Query tensor of shape (batch, heads, seq_len_q, head_dim)
        key: Key tensor of shape (batch, heads, seq_len_k, head_dim)
        scale: Optional scaling factor (default: 1/sqrt(head_dim))
        use_flash_attention: Whether to use flash attention (not implemented here,
                           but included for interface compatibility)

    Returns:
        Attention scores tensor of shape (batch, heads, seq_len_q, seq_len_k)

    Raises:
        TypeError: If query or key are not torch.Tensor
        ValueError: If query and key have incompatible shapes
    """
    if not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor):
        raise TypeError("query and key must be torch.Tensor objects")

    # Validate shapes are compatible
    if query.dim() != 4 or key.dim() != 4:
        raise ValueError(f"Expected 4D tensors, got query: {query.dim()}D, key: {key.dim()}D")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(f"Last dimension mismatch: query {query.shape[-1]} vs key {key.shape[-1]}")

    # Set default scale if not provided
    if scale is None:
        scale = 1.0 / (query.shape[-1] ** 0.5)  # Standard transformer scaling

    # Compute attention scores: Q @ K.T
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    return attn_scores


def mask_attention_scores(
    attn_scores: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    causal_mask: bool = False
) -> torch.Tensor:
    """
    Apply attention masks to attention scores.

    This function applies various types of attention masks to the computed
    attention scores, including user-provided masks and causal masking.

    Args:
        attn_scores: Attention scores tensor of shape (batch, heads, seq_len_q, seq_len_k)
        attention_mask: Optional user-provided attention mask of shape
                       (batch, heads, seq_len_q, seq_len_k) or broadcastable
        causal_mask: Whether to apply causal (upper triangular) masking

    Returns:
        Masked attention scores tensor of same shape as input

    Raises:
        TypeError: If attn_scores is not a torch.Tensor
        ValueError: If attention_mask shape is incompatible
    """
    if not isinstance(attn_scores, torch.Tensor):
        raise TypeError("attn_scores must be a torch.Tensor")

    masked_scores = attn_scores.float()  # Use float for numerical stability

    # Apply user-provided attention mask if provided
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("attention_mask must be a torch.Tensor or None")
        # Ensure mask is broadcastable
        try:
            masked_scores = masked_scores + attention_mask
        except RuntimeError as e:
            raise ValueError(f"Attention mask shape {attention_mask.shape} is not "
                           f"broadcastable with scores shape {attn_scores.shape}: {e}")

    # Apply causal (upper triangular) masking if requested
    if causal_mask:
        seq_len_q, seq_len_k = attn_scores.shape[-2], attn_scores.shape[-1]
        causal_mask_tensor = torch.triu(
            torch.full((seq_len_q, seq_len_k), float('-inf'), device=attn_scores.device),
            diagonal=1
        )
        masked_scores = masked_scores + causal_mask_tensor

    return masked_scores


def softmax_with_dtype(
    attn_scores: torch.Tensor,
    dim: int = -1,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Apply softmax to attention scores with optional dtype casting for numerical stability.

    This function applies softmax with proper dtype handling to ensure numerical
    stability, especially important for large attention score matrices.

    Args:
        attn_scores: Attention scores tensor
        dim: Dimension along which to apply softmax (default: -1)
        dtype: Optional dtype to cast to before softmax for numerical stability

    Returns:
        Softmax-transformed attention weights tensor

    Raises:
        TypeError: If attn_scores is not a torch.Tensor
    """
    if not isinstance(attn_scores, torch.Tensor):
        raise TypeError("attn_scores must be a torch.Tensor")

    # Apply softmax with dtype for numerical stability
    attn_weights = nn.functional.softmax(attn_scores, dim=dim, dtype=dtype)
    return attn_weights.type(attn_scores.dtype)  # Cast back to original dtype


def apply_attention_weights(
    attn_weights: torch.Tensor,
    value: torch.Tensor
) -> torch.Tensor:
    """
    Apply attention weights to value tensor to compute the attention output.

    This function performs the final step of attention computation: multiplying
    attention weights with value tensors.

    Args:
        attn_weights: Attention weights tensor of shape (batch, heads, seq_len_q, seq_len_k)
        value: Value tensor of shape (batch, heads, seq_len_k, head_dim)

    Returns:
        Attention output tensor of shape (batch, heads, seq_len_q, head_dim)

    Raises:
        TypeError: If inputs are not torch.Tensor
        ValueError: If shapes are incompatible
    """
    if not isinstance(attn_weights, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise TypeError("attn_weights and value must be torch.Tensor objects")

    # Validate shapes are compatible
    if attn_weights.shape[-1] != value.shape[-2]:
        raise ValueError(f"Last dimension of weights {attn_weights.shape[-1]} must match "
                        f"second-to-last dimension of value {value.shape[-2]}")

    # Apply attention: weights @ value
    attn_output = torch.matmul(attn_weights, value)
    return attn_output


def reshape_for_output(
    attn_output: torch.Tensor,
    batch_size: int,
    q_len: int,
    hidden_size: int
) -> torch.Tensor:
    """
    Reshape attention output to the expected output format.

    This function reshapes the multi-head attention output from
    (batch, heads, seq_len, head_dim) to (batch, seq_len, hidden_size).

    Args:
        attn_output: Multi-head attention output tensor of shape (batch, heads, seq_len, head_dim)
        batch_size: Batch size dimension
        q_len: Query sequence length
        hidden_size: Model hidden size

    Returns:
        Reshaped tensor of shape (batch, seq_len, hidden_size)

    Raises:
        TypeError: If attn_output is not a torch.Tensor
        ValueError: If shapes are incompatible
    """
    if not isinstance(attn_output, torch.Tensor):
        raise TypeError("attn_output must be a torch.Tensor")

    # Reshape from (batch, heads, seq_len, head_dim) to (batch, seq_len, hidden_size)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, q_len, hidden_size)
    return attn_output


__all__ = [
    "repeat_kv",
    "rotate_half", 
    "apply_rotary_pos_emb",
    "apply_rotary_pos_emb_with_position_ids",
    "compute_attention_scores",
    "mask_attention_scores",
    "softmax_with_dtype",
    "apply_attention_weights",
    "reshape_for_output"
]