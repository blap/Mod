"""
Python Wrapper for C Tensor Engine (ctypes) - Updated with New Ops
"""

import ctypes
import os
import sys
from typing import Tuple, List, Optional, Dict

# ... (Library loading logic same as before) ...
_plugin_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "plugins", "cpu", "c_src")
if os.name == 'nt': _lib_name = "libtensor_ops.dll"
else: _lib_name = "libtensor_ops.so"
_lib_path = os.path.join(_plugin_dir, _lib_name)
if not os.path.exists(_lib_path): _lib_path = os.path.join(os.path.dirname(__file__), "c_src", _lib_name)

try:
    _lib = ctypes.CDLL(_lib_path)
except OSError:
    _lib = None

# Define Types
class CTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("shape", ctypes.POINTER(ctypes.c_int)),
        ("ndim", ctypes.c_int),
        ("size", ctypes.c_int),
    ]

if _lib:
    # Basic Ops
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

    # New Ops
    _lib.tensor_argmax.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    _lib.tensor_embed.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    _lib.tensor_cat.argtypes = [ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.c_int, ctypes.POINTER(CTensor)]

    # Loader
    _lib.open_safetensors.argtypes = [ctypes.c_char_p]
    _lib.open_safetensors.restype = ctypes.c_int
    _lib.load_tensor_data.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    _lib.load_tensor_data.restype = ctypes.c_int
    _lib.close_safetensors.argtypes = []

    # Image Ops
    _lib.image_resize_bilinear.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
    _lib.image_normalize.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    _lib.image_rescale.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float]

class Tensor:
    def __init__(self, shape: List[int], data: List[float] = None, _handle=None):
        if not _lib: raise RuntimeError("C Engine library not loaded.")
        if _handle:
            self._handle = _handle
        else:
            c_shape = (ctypes.c_int * len(shape))(*shape)
            self._handle = _lib.create_tensor(c_shape, len(shape))
            if data: self.load(data)
            else: _lib.tensor_fill(self._handle, 0.0)

    @property
    def shape(self) -> Tuple[int]:
        if not self._handle: return ()
        s = self._handle.contents.shape
        n = self._handle.contents.ndim
        return tuple(s[i] for i in range(n))

    @property
    def ndim(self) -> int:
        if not self._handle: return 0
        return self._handle.contents.ndim

    @property
    def size(self) -> int:
        if not self._handle: return 0
        return self._handle.contents.size

    def load(self, data: List[float]):
        if len(data) != self.size: raise ValueError(f"Size mismatch {len(data)} vs {self.size}")
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
        b_handle = bias._handle if bias else None
        _lib.tensor_linear(self._handle, weight._handle, b_handle, out._handle)
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

    def argmax(self) -> 'Tensor':
        # Reduces last dim
        out_shape = list(self.shape[:-1])
        out = Tensor(out_shape)
        _lib.tensor_argmax(self._handle, out._handle)
        return out

    def embed(self, indices: 'Tensor') -> 'Tensor':
        # Self is weights [Vocab, Dim]
        # Indices [Batch, Seq]
        # Out [Batch, Seq, Dim]
        out_shape = list(indices.shape) + [self.shape[1]]
        out = Tensor(out_shape)
        _lib.tensor_embed(self._handle, indices._handle, out._handle)
        return out

    # Image Ops
    def resize_image(self, target_h: int, target_w: int) -> 'Tensor':
        if self.ndim != 3: raise ValueError("Image resize expects 3D tensor")
        c, h, w = self.shape
        out = Tensor([c, target_h, target_w])
        _lib.image_resize_bilinear(self._handle.contents.data, c, h, w, out._handle.contents.data, target_h, target_w)
        return out

    def normalize_image(self, mean: List[float], std: List[float]) -> 'Tensor':
        c, h, w = self.shape
        out = Tensor(list(self.shape))
        _lib.tensor_add(self._handle, zeros(list(self.shape))._handle, out._handle) # Copy
        c_mean = (ctypes.c_float * len(mean))(*mean)
        c_std = (ctypes.c_float * len(std))(*std)
        _lib.image_normalize(out._handle.contents.data, c, h, w, c_mean, c_std)
        return out

def cat(tensors: List[Tensor], axis: int = 0) -> Tensor:
    if not tensors: return None
    # Assuming same shape except axis
    shape = list(tensors[0].shape)
    total_dim = 0
    for t in tensors: total_dim += t.shape[axis]
    shape[axis] = total_dim

    out = Tensor(shape)

    # Create pointer array
    c_tensors = (ctypes.POINTER(CTensor) * len(tensors))()
    for i, t in enumerate(tensors):
        c_tensors[i] = t._handle

    _lib.tensor_cat(c_tensors, len(tensors), axis, out._handle)
    return out

# ... (Factories and Loader Interface same as before) ...
def zeros(shape: List[int]) -> Tensor: return Tensor(shape)
def randn(shape: List[int]) -> Tensor:
    import random
    size = 1
    for s in shape: size *= s
    data = [random.gauss(0, 1) for _ in range(size)]
    return Tensor(shape, data)
def arange(end: int) -> Tensor:
    data = [float(i) for i in range(end)]
    return Tensor([end], data)
def load_safetensors(filepath: str, model_layers: Dict[str, Tensor]):
    if not _lib or not os.path.exists(filepath): return False
    if _lib.open_safetensors(filepath.encode('utf-8')) <= 0: return False
    for name, tensor in model_layers.items():
        _lib.load_tensor_data(name.encode('utf-8'), tensor._handle.contents.data, tensor.size)
    _lib.close_safetensors()
    return True
