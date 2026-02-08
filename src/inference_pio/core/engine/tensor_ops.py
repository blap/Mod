"""
Core Tensor Operations for Numpy-based Inference Engine
Provides vectorized implementations of standard neural network functions.
"""

import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax implementation."""
    max_x = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def silu(x: np.ndarray) -> np.ndarray:
    """Sigmoid Linear Unit (SiLU) activation function."""
    return x / (1 + np.exp(-x))

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Root Mean Square Normalization."""
    variance = np.mean(x**2, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(variance + eps)) * weight

def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiplication."""
    return np.matmul(a, b)

def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray = None) -> np.ndarray:
    """Linear layer operation: x @ w.T + b."""
    # Assuming weight is [out_features, in_features] like torch.nn.Linear
    output = np.matmul(x, weight.T)
    if bias is not None:
        output += bias
    return output

def apply_rotary_emb(q: np.ndarray, k: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> tuple:
    """Apply Rotary Position Embeddings (RoPE)."""
    # Helper to split tensor into real/imag parts conceptually
    def rotate_half(x):
        # x is [..., dim]
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return np.concatenate((-x2, x1), axis=-1)

    # Ensure cos/sin match dimensions (broadcasting)
    # cos, sin are usually [seq_len, dim] or [1, seq_len, dim]

    # RoPE formula: (x * cos) + (rotate_half(x) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray = None, scale: float = None) -> np.ndarray:
    """Scaled Dot-Product Attention."""
    if scale is None:
        scale = 1.0 / np.sqrt(q.shape[-1])

    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale

    if mask is not None:
        # Assuming mask is additive (0 for keep, -inf for mask)
        scores += mask

    attn_weights = softmax(scores, axis=-1)
    return np.matmul(attn_weights, v)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> tuple:
    """Precompute cosine and sine frequencies for RoPE."""
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = np.arange(end)
    freqs = np.outer(t, freqs)
    # Convert to full dim (cos, sin)
    emb = np.concatenate((freqs, freqs), axis=-1)
    return np.cos(emb), np.sin(emb)

def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (GELU) approximation."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
