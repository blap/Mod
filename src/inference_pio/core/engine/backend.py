# ... (Previous parts of backend.py) ...

"""
Unified Backend for C Tensor Engine (CPU/CUDA)
Removes torch dependency entirely.
"""

import ctypes
import os
import sys
from typing import Tuple, List, Optional, Dict, Union, Any

def _load_library(name: str) -> Optional[ctypes.CDLL]:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    paths = []
    if "cuda" in name:
        paths.append(os.path.join(base_dir, "plugins", "cuda", "c_src", name))
    else:
        paths.append(os.path.join(base_dir, "plugins", "cpu", "c_src", name))
    paths.append(os.path.join(os.path.dirname(__file__), "c_src", name))
    paths.append(name)
    for path in paths:
        if os.path.exists(path):
            try: return ctypes.CDLL(path)
            except OSError as e: print(f"Warning: Failed to load {path}: {e}")
    return None

if os.name == 'nt':
    _lib_cpu = _load_library("libtensor_ops.dll")
    _lib_cuda = _load_library("libtensor_ops_cuda.dll")
else:
    _lib_cpu = _load_library("libtensor_ops.so")
    _lib_cuda = _load_library("libtensor_ops_cuda.so")

HAS_CPU = _lib_cpu is not None
HAS_CUDA = _lib_cuda is not None

class CTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("shape", ctypes.POINTER(ctypes.c_int)),
        ("ndim", ctypes.c_int),
        ("size", ctypes.c_int),
        ("device_id", ctypes.c_int),
    ]

def _setup_sigs(lib):
    if not lib: return
    lib.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    lib.create_tensor.restype = ctypes.POINTER(CTensor)
    lib.free_tensor.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_fill.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float]
    lib.tensor_add.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    lib.tensor_mul.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]

    if hasattr(lib, 'tensor_matmul'): lib.tensor_matmul.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_matmul_transposed'): lib.tensor_matmul_transposed.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_linear'): lib.tensor_linear.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_softmax'): lib.tensor_softmax.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_silu'): lib.tensor_silu.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_gelu'): lib.tensor_gelu.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_rms_norm'): lib.tensor_rms_norm.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float]
    if hasattr(lib, 'tensor_rope'): lib.tensor_rope.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]

    lib.tensor_load_data.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.tensor_get_data.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_float), ctypes.c_int]

    if hasattr(lib, 'tensor_argmax'): lib.tensor_argmax.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_embed'): lib.tensor_embed.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_cat'): lib.tensor_cat.argtypes = [ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.c_int, ctypes.POINTER(CTensor)]

    # New Ops
    if hasattr(lib, 'tensor_slice'):
        lib.tensor_slice.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    if hasattr(lib, 'tensor_precompute_freqs_cis'):
        lib.tensor_precompute_freqs_cis.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]

    if hasattr(lib, 'open_safetensors'):
        lib.open_safetensors.argtypes = [ctypes.c_char_p]
        lib.open_safetensors.restype = ctypes.c_int
        lib.load_tensor_data.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.load_tensor_data.restype = ctypes.c_int
        lib.close_safetensors.argtypes = []
    if hasattr(lib, 'image_resize_bilinear'):
        lib.image_resize_bilinear.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]

_setup_sigs(_lib_cpu)
_setup_sigs(_lib_cuda)

# ... (Tensor Class) ...

class Tensor:
    def __init__(self, shape: List[int], data: List[float] = None, device: str = "cpu", _handle=None):
        if not HAS_CPU: raise RuntimeError("CPU Engine library not loaded. Build failed?")
        self.device = device
        self._lib = _lib_cpu
        dev_id = -1
        if "cuda" in device:
            if not HAS_CUDA:
                print("Warning: CUDA requested but not available. Falling back to CPU.")
                self.device = "cpu"
            else:
                self._lib = _lib_cuda
                try: dev_id = int(device.split(":")[-1])
                except: dev_id = 0

        if _handle: self._handle = _handle
        else:
            c_shape = (ctypes.c_int * len(shape))(*shape)
            self._handle = self._lib.create_tensor(c_shape, len(shape), dev_id)
            if data: self.load(data)
            else: self._lib.tensor_fill(self._handle, 0.0)

    def __del__(self):
        if hasattr(self, '_handle') and self._handle: self._lib.free_tensor(self._handle)

    @property
    def shape(self) -> Tuple[int]:
        if not self._handle: return ()
        s = self._handle.contents.shape
        n = self._handle.contents.ndim
        return tuple(s[i] for i in range(n))

    @property
    def ndim(self) -> int: return self._handle.contents.ndim
    @property
    def size(self) -> int: return self._handle.contents.size
    @property
    def dtype(self): return "float32"

    def to(self, device: str) -> 'Tensor':
        if device == self.device: return self
        data = self.to_list()
        return Tensor(list(self.shape), data, device=device)

    def load(self, data: List[float]):
        c_data = (ctypes.c_float * len(data))(*data)
        self._lib.tensor_load_data(self._handle, c_data, len(data))

    def to_list(self) -> List[float]:
        buffer = (ctypes.c_float * self.size)()
        self._lib.tensor_get_data(self._handle, buffer, self.size)
        return list(buffer)

    # --- Operations ---

    def fill(self, value: float): self._lib.tensor_fill(self._handle, value)

    def matmul(self, other: 'Tensor', transpose_b: bool = False) -> 'Tensor':
        if self.device != other.device: raise ValueError("Device mismatch")
        m = self.shape[-2]
        k = self.shape[-1]

        if transpose_b:
            n = other.shape[-2]
            if other.shape[-1] != k: raise ValueError(f"Matmul Transpose shape mismatch {self.shape} vs {other.shape}")
        else:
            n = other.shape[-1]
            if other.shape[-2] != k: raise ValueError(f"Matmul shape mismatch {self.shape} vs {other.shape}")

        out_shape = list(self.shape[:-2]) + [m, n]
        out = Tensor(out_shape, device=self.device)

        if transpose_b and hasattr(self._lib, 'tensor_matmul_transposed'):
            self._lib.tensor_matmul_transposed(self._handle, other._handle, out._handle)
        else:
            self._lib.tensor_matmul(self._handle, other._handle, out._handle)
        return out

    def add(self, other: 'Tensor') -> 'Tensor':
        out = Tensor(list(self.shape), device=self.device)
        self._lib.tensor_add(self._handle, other._handle, out._handle)
        return out
    def __add__(self, other): return self.add(other)

    def mul(self, other: 'Tensor') -> 'Tensor':
        out = Tensor(list(self.shape), device=self.device)
        self._lib.tensor_mul(self._handle, other._handle, out._handle)
        return out
    def __mul__(self, other): return self.mul(other)

    def linear(self, weight: 'Tensor', bias: Optional['Tensor'] = None) -> 'Tensor':
        in_features = weight.shape[1] if weight.ndim >= 2 else weight.shape[0]
        out_features = weight.shape[0] if weight.ndim >= 2 else 1
        if self.shape[-1] != in_features: raise ValueError(f"Linear shape mismatch: Input {self.shape} vs Weight {weight.shape}")
        out_shape = list(self.shape[:-1]) + [out_features]
        out = Tensor(out_shape, device=self.device)
        b_handle = bias._handle if bias else None
        self._lib.tensor_linear(self._handle, weight._handle, b_handle, out._handle)
        return out

    def softmax(self, dim: int = -1) -> 'Tensor':
        out = Tensor(list(self.shape), device=self.device)
        self._lib.tensor_softmax(self._handle, out._handle)
        return out

    def silu(self) -> 'Tensor':
        out = Tensor(list(self.shape), device=self.device)
        self._lib.tensor_silu(self._handle, out._handle)
        return out
    def gelu(self) -> 'Tensor':
        out = Tensor(list(self.shape), device=self.device)
        if hasattr(self._lib, 'tensor_gelu'): self._lib.tensor_gelu(self._handle, out._handle)
        else: self._lib.tensor_silu(self._handle, out._handle)
        return out

    def rms_norm(self, weight: 'Tensor', eps: float = 1e-6) -> 'Tensor':
        out = Tensor(list(self.shape), device=self.device)
        self._lib.tensor_rms_norm(self._handle, weight._handle, out._handle, eps)
        return out

    def rope(self, k: 'Tensor', cos: 'Tensor', sin: 'Tensor') -> Tuple['Tensor', 'Tensor']:
        out_q = Tensor(list(self.shape), device=self.device)
        out_k = Tensor(list(k.shape), device=self.device)
        self._lib.tensor_rope(self._handle, k._handle, cos._handle, sin._handle, out_q._handle, out_k._handle)
        return out_q, out_k

    def argmax(self, dim: int = -1) -> 'Tensor':
        out_shape = list(self.shape[:-1])
        out = Tensor(out_shape, device=self.device)
        if hasattr(self._lib, 'tensor_argmax'): self._lib.tensor_argmax(self._handle, out._handle)
        return out

    def embed(self, indices: 'Tensor') -> 'Tensor':
        out_shape = list(indices.shape) + [self.shape[1]]
        out = Tensor(out_shape, device=self.device)
        if hasattr(self._lib, 'tensor_embed'): self._lib.tensor_embed(self._handle, indices._handle, out._handle)
        return out

    def slice(self, start_indices: List[int], slice_shapes: List[int]) -> 'Tensor':
        # out = input[start:start+slice]
        if len(start_indices) != self.ndim or len(slice_shapes) != self.ndim:
            raise ValueError("Slice args dim mismatch")

        out = Tensor(slice_shapes, device=self.device)
        if hasattr(self._lib, 'tensor_slice'):
            c_start = (ctypes.c_int * self.ndim)(*start_indices)
            c_shape = (ctypes.c_int * self.ndim)(*slice_shapes)
            self._lib.tensor_slice(self._handle, out._handle, c_start, c_shape)
        return out

    # Image Ops
    def resize_image(self, target_h: int, target_w: int) -> 'Tensor':
        if self.ndim != 3: raise ValueError("Image resize expects 3D tensor")
        if self.device != "cpu": raise ValueError("Image ops only on CPU currently")
        c, h, w = self.shape
        out = Tensor([c, target_h, target_w])
        if hasattr(self._lib, 'image_resize_bilinear'):
            self._lib.image_resize_bilinear(self._handle.contents.data, c, h, w, out._handle.contents.data, target_h, target_w)
        return out

# ... (Module, Linear, Embedding, etc classes same as before) ...
class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = False
    def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)
    def forward(self, *args, **kwargs): raise NotImplementedError
    def register_parameter(self, name: str, param: Optional[Tensor]): self._parameters[name] = param
    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True): self._parameters[name] = tensor
    def to(self, device: str):
        for name, param in self._parameters.items():
            if param: self._parameters[name] = param.to(device)
        for module in self._modules.values(): module.to(device)
        return self
    def parameters(self):
        for p in self._parameters.values():
            if p: yield p
        for m in self._modules.values(): yield from m.parameters()

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor([out_features, in_features])
        self.bias = Tensor([out_features]) if bias else None
        self.register_parameter("weight", self.weight)
        if bias: self.register_parameter("bias", self.bias)
    def forward(self, input: Tensor) -> Tensor: return input.linear(self.weight, self.bias)

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = Tensor([num_embeddings, embedding_dim])
        self.register_parameter("weight", self.weight)
    def forward(self, input: Tensor) -> Tensor: return self.weight.embed(input)

class RMSNorm(Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = Tensor([normalized_shape])
        self.register_parameter("weight", self.weight)
    def forward(self, input: Tensor) -> Tensor: return input.rms_norm(self.weight, self.eps)

class SiLU(Module):
    def forward(self, input: Tensor) -> Tensor: return input.silu()

class GELU(Module):
    def forward(self, input: Tensor) -> Tensor: return input.gelu()

def cat(tensors: List[Tensor], axis: int = 0) -> Tensor:
    if not tensors: return None
    dev = tensors[0].device
    if hasattr(_lib_cpu, 'tensor_cat'):
        c_tensors = (ctypes.POINTER(CTensor) * len(tensors))()
        total_dim = 0
        for i, t in enumerate(tensors):
            c_tensors[i] = t._handle
            total_dim += t.shape[axis]
        shape = list(tensors[0].shape)
        shape[axis] = total_dim
        out = Tensor(shape, device=dev)
        lib = _lib_cuda if "cuda" in dev and HAS_CUDA else _lib_cpu
        lib.tensor_cat(c_tensors, len(tensors), axis, out._handle)
        return out
    return None

def zeros(shape: List[int], device="cpu") -> Tensor:
    t = Tensor(shape, device=device)
    t.fill(0.0)
    return t

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device="cpu") -> Tuple[Tensor, Tensor]:
    # Returns [end, dim/2] tensors for cos/sin
    half_dim = dim // 2
    cos = Tensor([end, half_dim], device=device)
    sin = Tensor([end, half_dim], device=device)

    lib = _lib_cuda if "cuda" in device and HAS_CUDA else _lib_cpu
    if hasattr(lib, 'tensor_precompute_freqs_cis'):
        lib.tensor_precompute_freqs_cis(dim, end, ctypes.c_float(theta), cos._handle, sin._handle)
    return cos, sin

def load_safetensors(filepath: str, model_layers: Dict[str, Tensor]):
    if not _lib_cpu or not os.path.exists(filepath): return False
    if _lib_cpu.open_safetensors(filepath.encode('utf-8')) <= 0: return False
    for name, tensor in model_layers.items():
        if tensor.device != "cpu":
            size = tensor.size
            host_buffer = (ctypes.c_float * size)()
            if _lib_cpu.load_tensor_data(name.encode('utf-8'), host_buffer, size) == 0:
                tensor.load(list(host_buffer))
        else:
            _lib_cpu.load_tensor_data(name.encode('utf-8'), tensor._handle.contents.data, tensor.size)
    _lib_cpu.close_safetensors()
    return True
