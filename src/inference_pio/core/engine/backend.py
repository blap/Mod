"""
Python Wrapper for C Tensor Engine (ctypes)
"""

import ctypes
import os
from typing import Tuple, List, Optional

# Load Library
_lib_path = os.path.join(os.path.dirname(__file__), "c_src", "libtensor_ops.so")
_lib = ctypes.CDLL(_lib_path)

# Define Types
class CTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("shape", ctypes.POINTER(ctypes.c_int)),
        ("ndim", ctypes.c_int),
        ("size", ctypes.c_int),
    ]

# Define Signatures
_lib.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
_lib.create_tensor.restype = ctypes.POINTER(CTensor)

_lib.free_tensor.argtypes = [ctypes.POINTER(CTensor)]

_lib.tensor_fill.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float]

_lib.tensor_add.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
_lib.tensor_mul.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
_lib.tensor_matmul.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
_lib.tensor_linear.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
_lib.tensor_softmax.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
_lib.tensor_silu.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
_lib.tensor_rms_norm.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float]
_lib.tensor_rope.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]

_lib.tensor_load_data.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.tensor_get_data.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_float), ctypes.c_int]

class Tensor:
    def __init__(self, shape: List[int], data: List[float] = None, _handle=None):
        if _handle:
            self._handle = _handle
        else:
            c_shape = (ctypes.c_int * len(shape))(*shape)
            self._handle = _lib.create_tensor(c_shape, len(shape))

            if data:
                self.load(data)
            else:
                _lib.tensor_fill(self._handle, 0.0)

    def __del__(self):
        # Only free if we created it.
        # In a real system, ref counting is needed. Simple here.
        if hasattr(self, '_handle') and self._handle:
            # _lib.free_tensor(self._handle)
            # Commented out for safety in simple Python wrapper (GC cycles)
            # In production, need explicit management context
            pass

    @property
    def shape(self) -> Tuple[int]:
        s = self._handle.contents.shape
        n = self._handle.contents.ndim
        return tuple(s[i] for i in range(n))

    @property
    def ndim(self) -> int:
        return self._handle.contents.ndim

    @property
    def size(self) -> int:
        return self._handle.contents.size

    def load(self, data: List[float]):
        if len(data) != self.size:
            raise ValueError(f"Data size {len(data)} mismatch tensor size {self.size}")
        c_data = (ctypes.c_float * len(data))(*data)
        _lib.tensor_load_data(self._handle, c_data, len(data))

    def to_list(self) -> List[float]:
        buffer = (ctypes.c_float * self.size)()
        _lib.tensor_get_data(self._handle, buffer, self.size)
        return list(buffer)

    def fill(self, value: float):
        _lib.tensor_fill(self._handle, value)

    def matmul(self, other: 'Tensor') -> 'Tensor':
        # Simple shape inference
        m = self.shape[-2]
        n = other.shape[-1]
        out_shape = list(self.shape[:-2]) + [m, n]
        out = Tensor(out_shape)
        _lib.tensor_matmul(self._handle, other._handle, out._handle)
        return out

    def add(self, other: 'Tensor') -> 'Tensor':
        out = Tensor(list(self.shape))
        _lib.tensor_add(self._handle, other._handle, out._handle)
        return out

    def mul(self, other: 'Tensor') -> 'Tensor':
        out = Tensor(list(self.shape))
        _lib.tensor_mul(self._handle, other._handle, out._handle)
        return out

    def linear(self, weight: 'Tensor', bias: Optional['Tensor'] = None) -> 'Tensor':
        # Input: [M, K], Weight: [N, K]
        # Out: [M, N]
        m = self.shape[0]
        n = weight.shape[0]
        out = Tensor([m, n])
        bias_handle = bias._handle if bias else None
        _lib.tensor_linear(self._handle, weight._handle, bias_handle, out._handle)
        return out

    def softmax(self) -> 'Tensor':
        out = Tensor(list(self.shape))
        _lib.tensor_softmax(self._handle, out._handle)
        return out

    def silu(self) -> 'Tensor':
        out = Tensor(list(self.shape))
        _lib.tensor_silu(self._handle, out._handle)
        return out

    def rms_norm(self, weight: 'Tensor', eps: float = 1e-6) -> 'Tensor':
        out = Tensor(list(self.shape))
        _lib.tensor_rms_norm(self._handle, weight._handle, out._handle, eps)
        return out

    def rope(self, k: 'Tensor', cos: 'Tensor', sin: 'Tensor') -> Tuple['Tensor', 'Tensor']:
        out_q = Tensor(list(self.shape))
        out_k = Tensor(list(k.shape))
        _lib.tensor_rope(self._handle, k._handle, cos._handle, sin._handle, out_q._handle, out_k._handle)
        return out_q, out_k

# Factories
def zeros(shape: List[int]) -> Tensor:
    return Tensor(shape)

def randn(shape: List[int]) -> Tensor:
    # Dummy random
    import random
    size = 1
    for s in shape: size *= s
    data = [random.gauss(0, 1) for _ in range(size)]
    return Tensor(shape, data)

def arange(end: int) -> Tensor:
    data = [float(i) for i in range(end)]
    return Tensor([end], data)
