"""
Consolidated Tensor Utilities for Inference-PIO System

This module provides utility functions for tensor operations in the Inference-PIO system.
It consolidates duplicate implementations to ensure consistent behavior across the codebase.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key tensors.

    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        cos: Cosine values of shape (seq_len, head_dim)
        sin: Sine values of shape (seq_len, head_dim)

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # Apply rotary embeddings to query
    q_embed = (q * cos) + (rotate_half(q) * sin)

    # Apply rotary embeddings to key
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def apply_rotary_pos_emb_with_position_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key tensors with specific position IDs.

    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        cos: Cosine values of shape (max_seq_len, head_dim)
        sin: Sine values of shape (max_seq_len, head_dim)
        position_ids: Position IDs of shape (batch_size, seq_len)

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # Gather cosine and sine values for the specific positions
    cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, head_dim]
    sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, head_dim]

    # Apply rotary embeddings to query
    q_embed = (q * cos) + (rotate_half(q) * sin)

    # Apply rotary embeddings to key
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dimensions of the input.

    Args:
        x: Input tensor of shape (..., head_dim)

    Returns:
        Rotated tensor of the same shape
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key and value tensors for grouped-query attention.

    Args:
        hidden_states: Input tensor of shape (batch_size, num_key_value_heads, seq_len, head_dim)
        n_rep: Number of times to repeat each key-value head

    Returns:
        Repeated tensor of shape (batch_size, num_key_value_heads * n_rep, seq_len, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def apply_chunking_to_forward(
    forward_fn, chunk_size: int, chunk_dim: int, *input_tensors
) -> torch.Tensor:
    """
    Apply a forward function to input tensors in chunks to save memory.

    Args:
        forward_fn: Forward function to apply
        chunk_size: Size of each chunk
        chunk_dim: Dimension along which to chunk
        *input_tensors: Input tensors to chunk

    Returns:
        Output tensor
    """
    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tensors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # Chunk along the chosen dimension
        input_tensors_chunks = [
            input_tensor.chunk(num_chunks, dim=chunk_dim)
            for input_tensor in input_tensors
        ]
        # Apply the forward function to each chunk
        output_chunks = [
            forward_fn(*input_tensors_chunk)
            for input_tensors_chunk in zip(*input_tensors_chunks)
        ]
        # Concatenate the chunks back
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """
    Gaussian Error Linear Unit implementation.

    Args:
        x: Input tensor

    Returns:
        GELU applied tensor
    """
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
        )
    )


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    Sigmoid Linear Unit (SiLU) activation function.

    Args:
        x: Input tensor

    Returns:
        SiLU applied tensor
    """
    return x * torch.sigmoid(x)


def swish(x: torch.Tensor) -> torch.Tensor:
    """
    Swish activation function (also known as SiLU).

    Args:
        x: Input tensor

    Returns:
        Swish applied tensor
    """
    return silu(x)


def softmax_with_temperature(
    x: torch.Tensor, dim: int = -1, temperature: float = 1.0
) -> torch.Tensor:
    """
    Apply softmax with temperature scaling.

    Args:
        x: Input tensor
        dim: Dimension along which to apply softmax
        temperature: Temperature for scaling

    Returns:
        Softmax applied tensor with temperature scaling
    """
    if temperature == 1.0:
        return torch.softmax(x, dim=dim)
    else:
        return torch.softmax(x / temperature, dim=dim)


def masked_fill_with_broadcast(
    x: torch.Tensor, mask: torch.Tensor, value: float
) -> torch.Tensor:
    """
    Fill tensor with value where mask is True, with broadcasting support.

    Args:
        x: Input tensor
        mask: Boolean mask
        value: Value to fill

    Returns:
        Filled tensor
    """
    return x.masked_fill(mask, value)


def normalize_with_l2(
    x: torch.Tensor, dim: int = -1, eps: float = 1e-12
) -> torch.Tensor:
    """
    Normalize tensor with L2 norm.

    Args:
        x: Input tensor
        dim: Dimension along which to normalize
        eps: Small value to avoid division by zero

    Returns:
        L2 normalized tensor
    """
    return x / torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min=eps)


def pad_sequence_to_length(
    sequence: torch.Tensor, target_length: int, pad_value: float = 0.0, dim: int = -1
) -> torch.Tensor:
    """
    Pad sequence to target length.

    Args:
        sequence: Input sequence tensor
        target_length: Target length to pad to
        pad_value: Value to use for padding
        dim: Dimension along which to pad

    Returns:
        Padded tensor
    """
    if sequence.size(dim) >= target_length:
        return sequence

    pad_shape = list(sequence.shape)
    pad_shape[dim] = target_length - sequence.size(dim)
    padding = torch.full(
        pad_shape, pad_value, dtype=sequence.dtype, device=sequence.device
    )

    return torch.cat([sequence, padding], dim=dim)


def truncate_sequence_to_length(
    sequence: torch.Tensor, target_length: int, dim: int = -1
) -> torch.Tensor:
    """
    Truncate sequence to target length.

    Args:
        sequence: Input sequence tensor
        target_length: Target length to truncate to
        dim: Dimension along which to truncate

    Returns:
        Truncated tensor
    """
    if sequence.size(dim) <= target_length:
        return sequence

    return sequence.narrow(dim, 0, target_length)


def safe_tensor_operation(
    operation_name: str, operation_fn, *args, **kwargs
) -> torch.Tensor:
    """
    Safely perform a tensor operation with error handling.

    Args:
        operation_name: Name of the operation for logging
        operation_fn: Function to perform
        *args: Arguments to the function
        **kwargs: Keyword arguments to the function

    Returns:
        Result of the operation
    """
    try:
        return operation_fn(*args, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Error in {operation_name}: {str(e)}")


def validate_tensor_shape(
    tensor: torch.Tensor,
    expected_shape: Tuple[Optional[int], ...],
    tensor_name: str = "tensor",
) -> bool:
    """
    Validate that a tensor has the expected shape.

    Args:
        tensor: Input tensor
        expected_shape: Expected shape (use None for any size in that dimension)
        tensor_name: Name of the tensor for error messages

    Returns:
        True if shape is valid, False otherwise
    """
    if len(tensor.shape) != len(expected_shape):
        raise ValueError(
            f"{tensor_name} has incorrect number of dimensions: {len(tensor.shape)} expected {len(expected_shape)}"
        )

    for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
        if expected is not None and actual != expected:
            raise ValueError(
                f"{tensor_name} has incorrect size at dimension {i}: {actual} expected {expected}"
            )

    return True


__all__ = [
    "apply_rotary_pos_emb",
    "apply_rotary_pos_emb_with_position_ids",
    "rotate_half",
    "repeat_kv",
    "apply_chunking_to_forward",
    "gelu_new",
    "silu",
    "swish",
    "softmax_with_temperature",
    "masked_fill_with_broadcast",
    "normalize_with_l2",
    "pad_sequence_to_length",
    "truncate_sequence_to_length",
    "safe_tensor_operation",
    "validate_tensor_shape",
]
