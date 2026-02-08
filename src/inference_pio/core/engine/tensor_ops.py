"""
Tensor Operations Interface (C Backend)
"""
from .backend import Tensor, arange
import math

def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return x.softmax()

def silu(x: Tensor) -> Tensor:
    return x.silu()

def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    return x.rms_norm(weight, eps)

def matmul(a: Tensor, b: Tensor) -> Tensor:
    return a.matmul(b)

def linear(x: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
    return x.linear(weight, bias)

def apply_rotary_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple:
    return q.rope(k, cos, sin)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> tuple:
    # Logic is implemented in backend factories usually, but for complex precompute:
    # 1D implementation in Python to create Tensors, passed to C later
    freqs = [1.0 / (theta ** ((i * 2) / dim)) for i in range(dim // 2)]
    t = [float(i) for i in range(end)]

    cos_data = []
    sin_data = []
    for ti in t:
        for f in freqs:
            val = ti * f
            cos_data.append(math.cos(val))
            sin_data.append(math.sin(val))
            # Append twice for real/imag equivalent in simplified RoPE
            cos_data.append(math.cos(val))
            sin_data.append(math.sin(val))

    cos = Tensor([end, dim], cos_data)
    sin = Tensor([end, dim], sin_data)
    return cos, sin
