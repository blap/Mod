"""
Python Wrapper for C Tensor Engine (ctypes) - Updated with Loader
"""

import ctypes
import os
from typing import Tuple, List, Optional, Dict

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

# Operations Signatures
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

# Loader Signatures
_lib.open_safetensors.argtypes = [ctypes.c_char_p]
_lib.open_safetensors.restype = ctypes.c_int
_lib.load_tensor_data.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.load_tensor_data.restype = ctypes.c_int
_lib.close_safetensors.argtypes = []

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

    # ... (Previous methods remain the same) ...
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
    import random
    size = 1
    for s in shape: size *= s
    data = [random.gauss(0, 1) for _ in range(size)]
    return Tensor(shape, data)

def arange(end: int) -> Tensor:
    data = [float(i) for i in range(end)]
    return Tensor([end], data)

# Loader Interface
def load_safetensors(filepath: str, model_layers: Dict[str, Tensor]):
    """
    Load weights from a SafeTensors file into existing C Tensors.
    """
    if not os.path.exists(filepath):
        return False

    res = _lib.open_safetensors(filepath.encode('utf-8'))
    if res <= 0:
        return False

    for name, tensor in model_layers.items():
        # Pass the tensor's data pointer to C loader
        # C loader finds 'name' in file and copies bytes
        _lib.load_tensor_data(name.encode('utf-8'), tensor._handle.contents.data, tensor.size)

    _lib.close_safetensors()
    return True
