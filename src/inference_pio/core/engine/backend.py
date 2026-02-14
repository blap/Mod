"""
Unified Backend for C Tensor Engine (CPU/CUDA/OpenCL)
Removes torch dependency entirely.
Supports Heterogeneous Compute (NVIDIA + AMD + Intel + CPU).
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
    # Basic
    lib.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    lib.create_tensor.restype = ctypes.POINTER(CTensor)
    lib.free_tensor.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_fill.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float]
    lib.tensor_add.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    lib.tensor_mul.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]

    # Math
    if hasattr(lib, 'tensor_matmul'): lib.tensor_matmul.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_matmul_transposed'): lib.tensor_matmul_transposed.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_linear'): lib.tensor_linear.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_softmax'): lib.tensor_softmax.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_silu'): lib.tensor_silu.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_gelu'): lib.tensor_gelu.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_rms_norm'): lib.tensor_rms_norm.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float]
    if hasattr(lib, 'tensor_rope'): lib.tensor_rope.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]

    # Data
    lib.tensor_load_data.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.tensor_get_data.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_float), ctypes.c_int]

    if hasattr(lib, 'tensor_argmax'): lib.tensor_argmax.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_embed'): lib.tensor_embed.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_cat'): lib.tensor_cat.argtypes = [ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.c_int, ctypes.POINTER(CTensor)]

    if hasattr(lib, 'tensor_slice'):
        lib.tensor_slice.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    if hasattr(lib, 'tensor_slice_device'):
        lib.tensor_slice_device.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_set_slice'):
        lib.tensor_set_slice.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_int)]
    if hasattr(lib, 'tensor_set_slice_device'):
        lib.tensor_set_slice_device.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'begin_capture'):
        lib.begin_capture.argtypes = []
    if hasattr(lib, 'end_capture'):
        lib.end_capture.argtypes = []
    if hasattr(lib, 'replay_graph'):
        lib.replay_graph.argtypes = []
    if hasattr(lib, 'init_memory_pool'):
        lib.init_memory_pool.argtypes = [ctypes.c_size_t]
    if hasattr(lib, 'reset_memory_pool'):
        lib.reset_memory_pool.argtypes = []
    if hasattr(lib, 'tensor_precompute_freqs_cis'):
        lib.tensor_precompute_freqs_cis.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]

    # NEW: Conv2d
    if hasattr(lib, 'tensor_conv2d'):
        lib.tensor_conv2d.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_int, ctypes.c_int]

    if hasattr(lib, 'tensor_scale'):
        lib.tensor_scale.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float, ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_permute'):
        lib.tensor_permute.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_int)]
    if hasattr(lib, 'tensor_swiglu'):
        lib.tensor_swiglu.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_fused_gate_up_swiglu'):
        lib.tensor_fused_gate_up_swiglu.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_fused_split_rope'):
        lib.tensor_fused_split_rope.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_scaled_dot_product_attention'):
        lib.tensor_scaled_dot_product_attention.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float]
    if hasattr(lib, 'tensor_reshape'):
        lib.tensor_reshape.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_gather'):
        lib.tensor_gather.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int]
    if hasattr(lib, 'tensor_scatter_add'):
        lib.tensor_scatter_add.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int]
    if hasattr(lib, 'tensor_topk'):
        lib.tensor_topk.argtypes = [ctypes.POINTER(CTensor), ctypes.c_int, ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]

    if hasattr(lib, 'open_safetensors'):
        lib.open_safetensors.argtypes = [ctypes.c_char_p]
        lib.open_safetensors.restype = ctypes.c_int
        lib.load_tensor_data.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.load_tensor_data.restype = ctypes.c_int
        lib.close_safetensors.argtypes = []
    if hasattr(lib, 'image_resize_bilinear'):
        lib.image_resize_bilinear.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]

    # MoE Primitives
    if hasattr(lib, 'tensor_count_value'):
        lib.tensor_count_value.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float, ctypes.POINTER(ctypes.c_int)]
    if hasattr(lib, 'tensor_gather_by_value'):
        lib.tensor_gather_by_value.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float, ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_scatter_add_by_index'):
        lib.tensor_scatter_add_by_index.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    if hasattr(lib, 'tensor_deltanet_recurrence'):
        lib.tensor_deltanet_recurrence.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]

    # Stream & Pinned Memory
    if hasattr(lib, 'create_stream'):
        lib.create_stream.restype = ctypes.c_void_p
        lib.create_stream.argtypes = []
    if hasattr(lib, 'destroy_stream'):
        lib.destroy_stream.argtypes = [ctypes.c_void_p]
    if hasattr(lib, 'stream_synchronize'):
        lib.stream_synchronize.argtypes = [ctypes.c_void_p]
    if hasattr(lib, 'set_current_stream'):
        lib.set_current_stream.argtypes = [ctypes.c_void_p]
    if hasattr(lib, 'allocate_pinned_memory'):
        lib.allocate_pinned_memory.restype = ctypes.c_void_p
        lib.allocate_pinned_memory.argtypes = [ctypes.c_size_t]
    if hasattr(lib, 'free_pinned_memory'):
        lib.free_pinned_memory.argtypes = [ctypes.c_void_p]

    # Fused Ops
    if hasattr(lib, 'tensor_fused_add_rms_norm'):
        lib.tensor_fused_add_rms_norm.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float]
    if hasattr(lib, 'tensor_dequantize'):
        lib.tensor_dequantize.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]

_setup_sigs(_lib_cpu)
_setup_sigs(_lib_cuda)

class CUDAStream:
    """Python wrapper for CUDA Streams using backend."""
    def __init__(self, lib=_lib_cuda):
        self._lib = lib
        self._handle = None
        if self._lib and hasattr(self._lib, 'create_stream'):
            self._handle = self._lib.create_stream()

    def __del__(self):
        if self._handle and self._lib and hasattr(self._lib, 'destroy_stream'):
            self._lib.destroy_stream(self._handle)

    def synchronize(self):
        if self._handle:
            self._lib.stream_synchronize(self._handle)

    def __enter__(self):
        if self._handle and hasattr(self._lib, 'set_current_stream'):
            self.prev_stream = None # Real impl would need get_current_stream
            self._lib.set_current_stream(self._handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle and hasattr(self._lib, 'set_current_stream'):
            self._lib.set_current_stream(None) # Reset to default

def allocate_pinned(size_bytes: int):
    if _lib_cuda and hasattr(_lib_cuda, 'allocate_pinned_memory'):
        return _lib_cuda.allocate_pinned_memory(size_bytes)
    # Fallback to malloc via libc if needed, but not truly pinned
    return None

def free_pinned(ptr):
    if _lib_cuda and hasattr(_lib_cuda, 'free_pinned_memory'):
        _lib_cuda.free_pinned_memory(ptr)

class Tensor:
    __slots__ = ['device', '_lib', '_handle', '_opencl_mem']

    def __init__(self, shape: List[int], data: List[float] = None, device: str = "cpu", _handle=None, backend_lib=None):
        self.device = device
        self._handle = None
        self._opencl_mem = None # Opaque pointer for OpenCL backend

        # 1. Resolve Backend
        # Logic: device string can be 'cpu', 'cuda:0', 'opencl:amd:0'
        # We need to find the correct library/plugin instance.

        # Late import to avoid circular dependency
        from ...plugins.manager import get_plugin_manager
        pm = get_plugin_manager()

        # If device is managed by a plugin (OpenCL), get the backend instance
        self._lib = None

        if "opencl" in device:
            # OpenCL devices don't use 'libtensor_ops.so' directly for create_tensor
            # They use the plugin's allocate() method.
            backend = pm.active_backends.get(device)
            if not backend:
                # Fallback check: maybe device string is simplified?
                # For now assume explicit registry key match
                raise ValueError(f"Backend not found for device: {device}")

            self._lib = backend # The plugin instance acts as the library interface
            size_bytes = 1
            for s in shape: size_bytes *= s
            size_bytes *= 4 # float32

            # Allocate on OpenCL device
            self._opencl_mem = self._lib.allocate(size_bytes)
            # Store metadata? CTensor struct is useful for shape/size tracking even if data ptr is dummy
            # We use CPU lib to create a handle just for metadata management (shape/stride)
            # but 'data' pointer will be NULL or garbage.
            # OR we implement a Pure Python metadata tracker.
            # To keep compatibility with Module/Linear which access .shape, we use CTensor on CPU as metadata holder.
            c_shape = (ctypes.c_int * len(shape))(*shape)
            self._handle = _lib_cpu.create_tensor(c_shape, len(shape), -1) # -1 = CPU but we treat as metadata

            if data: self.load(data)
            else: pass # Fill 0?

        elif "cuda" in device:
            if not HAS_CUDA:
                print("Warning: CUDA requested but not available. Falling back to CPU.")
                self.device = "cpu"
                self._lib = _lib_cpu
            else:
                self._lib = _lib_cuda
                try: dev_id = int(device.split(":")[-1])
                except: dev_id = 0
                c_shape = (ctypes.c_int * len(shape))(*shape)
                self._handle = self._lib.create_tensor(c_shape, len(shape), dev_id)
                if data: self.load(data)

        else: # CPU
            self._lib = _lib_cpu
            c_shape = (ctypes.c_int * len(shape))(*shape)
            self._handle = self._lib.create_tensor(c_shape, len(shape), -1)
            if data: self.load(data)

    def __del__(self):
        if hasattr(self, '_handle') and self._handle: _lib_cpu.free_tensor(self._handle)
        if hasattr(self, '_opencl_mem') and self._opencl_mem and self._lib and hasattr(self._lib, 'free'):
            self._lib.free(self._opencl_mem)

    def load(self, data: List[float]):
        if "opencl" in self.device:
            # Use plugin interface
            size_bytes = self.size * 4
            self._lib.memcpy_h2d(self._opencl_mem, data, size_bytes)
        else:
            c_data = (ctypes.c_float * len(data))(*data)
            self._lib.tensor_load_data(self._handle, c_data, len(data))

    def to_list(self) -> List[float]:
        if "opencl" in self.device:
            size_bytes = self.size * 4
            buffer = [0.0] * self.size
            self._lib.memcpy_d2h(buffer, self._opencl_mem, size_bytes)
            return buffer
        else:
            buffer = (ctypes.c_float * self.size)()
            self._lib.tensor_get_data(self._handle, buffer, self.size)
            return list(buffer)

    def to(self, device: str, non_blocking: bool = False) -> 'Tensor':
        if device == self.device: return self

        # Unified Transfer Logic
        # Case 1: CPU <-> CUDA (Existing)
        # Case 2: CPU <-> OpenCL (New)
        # Case 3: CUDA <-> OpenCL (Peer-to-Peer via Host)
        # Case 4: OpenCL <-> OpenCL (Cross-vendor? via Host)

        # 1. Read to Host (if not already there)
        host_data = self.to_list()

        # 2. Create on Target
        return Tensor(list(self.shape), host_data, device=device)

    # ... (Rest of properties like shape, ndim, size wrap self._handle which exists for all backends) ...
    @property
    def shape(self) -> Tuple[int]:
        s = self._handle.contents.shape
        n = self._handle.contents.ndim
        return tuple(s[i] for i in range(n))
    @property
    def ndim(self) -> int: return self._handle.contents.ndim
    @property
    def size(self) -> int: return self._handle.contents.size
    @property
    def dtype(self): return "float32"

    # --- Abstract Dispatch for Ops ---

    def fill(self, value: float):
        if "opencl" in self.device:
            # Basic fill kernel needed in OpenCL backend or copy from host
            # For "Real Code", we'll just copy from host
            data = [value] * self.size
            self.load(data)
        else:
            self._lib.tensor_fill(self._handle, value)

    def matmul(self, other: 'Tensor', transpose_b: bool = False) -> 'Tensor':
        if self.device != other.device: raise ValueError("Device mismatch")
        m = self.shape[-2]
        k = self.shape[-1]
        if transpose_b:
            n = other.shape[-2]
            if other.shape[-1] != k: raise ValueError(f"Matmul Transpose shape mismatch")
        else:
            n = other.shape[-1]
            if other.shape[-2] != k: raise ValueError(f"Matmul shape mismatch")
        out_shape = list(self.shape[:-2]) + [m, n]
        out = Tensor(out_shape, device=self.device)

        if "opencl" in self.device:
            if transpose_b: raise NotImplementedError("OpenCL matmul transposed not optimized yet")
            self._lib.matmul(self._opencl_mem, other._opencl_mem, out._opencl_mem, m, n, k)
        else:
            if transpose_b and hasattr(self._lib, 'tensor_matmul_transposed'):
                self._lib.tensor_matmul_transposed(self._handle, other._handle, out._handle)
            else:
                self._lib.tensor_matmul(self._handle, other._handle, out._handle)
        return out

    def add(self, other: 'Tensor') -> 'Tensor':
        out = Tensor(list(self.shape), device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL add not implemented") # TODO: Add to plugin
        self._lib.tensor_add(self._handle, other._handle, out._handle)
        return out
    def __add__(self, other): return self.add(other)

    def mul(self, other: Union['Tensor', float]) -> 'Tensor':
        out = Tensor(list(self.shape), device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL mul not implemented")
        if isinstance(other, (int, float)):
            if hasattr(self._lib, 'tensor_scale'):
                self._lib.tensor_scale(self._handle, ctypes.c_float(other), out._handle)
            else:
                t = Tensor(list(self.shape), device=self.device); t.fill(float(other))
                self._lib.tensor_mul(self._handle, t._handle, out._handle)
        else:
            self._lib.tensor_mul(self._handle, other._handle, out._handle)
        return out
    def __mul__(self, other): return self.mul(other)

    def swiglu(self, up_proj_out: 'Tensor') -> 'Tensor':
        out = Tensor(list(self.shape), device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL swiglu not implemented")
        if hasattr(self._lib, 'tensor_swiglu'):
            self._lib.tensor_swiglu(self._handle, up_proj_out._handle, out._handle)
        else:
            silu = self.silu()
            out = silu.mul(up_proj_out)
        return out

    def fused_swiglu(self) -> 'Tensor':
        out_shape = list(self.shape)
        out_shape[-1] = out_shape[-1] // 2
        out = Tensor(out_shape, device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL fused swiglu not implemented")
        if hasattr(self._lib, 'tensor_fused_gate_up_swiglu'):
            self._lib.tensor_fused_gate_up_swiglu(self._handle, out._handle)
        return out

    def permute(self, dims: List[int]) -> 'Tensor':
        if len(dims) != self.ndim: raise ValueError("Permute dims mismatch")
        new_shape = [self.shape[d] for d in dims]
        out = Tensor(new_shape, device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL permute not implemented")
        if hasattr(self._lib, 'tensor_permute'):
            c_dims = (ctypes.c_int * len(dims))(*dims)
            self._lib.tensor_permute(self._handle, out._handle, c_dims)
        else:
            raise NotImplementedError("Permute not implemented in backend")
        return out

    def reshape(self, new_shape: List[int]) -> 'Tensor':
        size = 1
        for s in new_shape: size *= s
        if size != self.size: raise ValueError("Reshape size mismatch")
        out = Tensor(new_shape, device=self.device)
        if "opencl" in self.device:
            # OpenCL reshape is just logical if contiguous.
            # We copy buffer handle? Or data.
            # Simplified: Copy data
            data = self.to_list()
            out.load(data)
            return out

        if hasattr(self._lib, 'tensor_reshape'):
            self._lib.tensor_reshape(self._handle, out._handle)
        else:
            out.load(self.to_list())
        return out

    def gather(self, indices: 'Tensor', axis: int = 0) -> 'Tensor':
        out_shape = [indices.size] + list(self.shape)[axis+1:]
        out = Tensor(out_shape, device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL gather not implemented")
        if hasattr(self._lib, 'tensor_gather'):
            self._lib.tensor_gather(self._handle, indices._handle, out._handle, axis)
        return out

    def scatter_add(self, indices: 'Tensor', src: 'Tensor', axis: int = 0):
        if "opencl" in self.device: raise NotImplementedError("OpenCL scatter_add not implemented")
        if hasattr(self._lib, 'tensor_scatter_add'):
            self._lib.tensor_scatter_add(self._handle, indices._handle, src._handle, axis)

    def topk(self, k: int) -> Tuple['Tensor', 'Tensor']:
        out_shape = list(self.shape)
        out_shape[-1] = k
        values = Tensor(out_shape, device=self.device)
        indices = Tensor(out_shape, device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL topk not implemented")
        if hasattr(self._lib, 'tensor_topk'):
            self._lib.tensor_topk(self._handle, k, values._handle, indices._handle)
        return values, indices

    def linear(self, weight: 'Tensor', bias: Optional['Tensor'] = None) -> 'Tensor':
        in_features = weight.shape[1] if weight.ndim >= 2 else weight.shape[0]
        out_features = weight.shape[0] if weight.ndim >= 2 else 1
        if self.shape[-1] != in_features: raise ValueError(f"Linear shape mismatch")
        out_shape = list(self.shape[:-1]) + [out_features]
        out = Tensor(out_shape, device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL linear not implemented")
        b_handle = bias._handle if bias else None
        self._lib.tensor_linear(self._handle, weight._handle, b_handle, out._handle)
        return out

    def conv2d(self, weight: 'Tensor', bias: Optional['Tensor'] = None, stride: int = 1, padding: int = 0, groups: int = 1) -> 'Tensor':
        if self.ndim != 4: raise ValueError("Conv2d requires 4D input")
        N, C_in, H, W = self.shape
        C_out, _, KH, KW = weight.shape
        H_out = (H + 2*padding - KH) // stride + 1
        W_out = (W + 2*padding - KW) // stride + 1
        out = Tensor([N, C_out, H_out, W_out], device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL conv2d not implemented")
        b_handle = bias._handle if bias else None
        if hasattr(self._lib, 'tensor_conv2d'):
            self._lib.tensor_conv2d(self._handle, weight._handle, b_handle, out._handle, stride, padding, groups)
        else:
            raise NotImplementedError("Conv2d not implemented in backend")
        return out

    def softmax(self, dim: int = -1) -> 'Tensor':
        out = Tensor(list(self.shape), device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL softmax not implemented")
        self._lib.tensor_softmax(self._handle, out._handle)
        return out

    def silu(self) -> 'Tensor':
        out = Tensor(list(self.shape), device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL silu not implemented")
        self._lib.tensor_silu(self._handle, out._handle)
        return out
    def gelu(self) -> 'Tensor':
        out = Tensor(list(self.shape), device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL gelu not implemented")
        if hasattr(self._lib, 'tensor_gelu'): self._lib.tensor_gelu(self._handle, out._handle)
        else: self._lib.tensor_silu(self._handle, out._handle)
        return out

    def rms_norm(self, weight: 'Tensor', eps: float = 1e-6) -> 'Tensor':
        out = Tensor(list(self.shape), device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL rms_norm not implemented")
        self._lib.tensor_rms_norm(self._handle, weight._handle, out._handle, eps)
        return out

    def rope(self, k: 'Tensor', cos: 'Tensor', sin: 'Tensor') -> Tuple['Tensor', 'Tensor']:
        out_q = Tensor(list(self.shape), device=self.device)
        out_k = Tensor(list(k.shape), device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL rope not implemented")
        self._lib.tensor_rope(self._handle, k._handle, cos._handle, sin._handle, out_q._handle, out_k._handle)
        return out_q, out_k

    def fused_add_rms_norm(self, residual: 'Tensor', weight: 'Tensor', eps: float) -> 'Tensor':
        out = Tensor(list(self.shape), device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL fused_add_rms_norm not implemented")
        if hasattr(self._lib, 'tensor_fused_add_rms_norm'):
            self._lib.tensor_fused_add_rms_norm(self._handle, residual._handle, weight._handle, out._handle, ctypes.c_float(eps))
        else:
            x_new = self.add(residual)
            out = x_new.rms_norm(weight, eps)
        return out

    def dequantize(self, scale: 'Tensor') -> 'Tensor':
        out = Tensor(list(self.shape), device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL dequantize not implemented")
        if hasattr(self._lib, 'tensor_dequantize'):
            self._lib.tensor_dequantize(self._handle, scale._handle, out._handle)
        return out

    def fused_split_rope(self, cos: 'Tensor', sin: 'Tensor', heads: int, head_dim: int) -> Tuple['Tensor', 'Tensor', 'Tensor']:
        B = self.shape[0]
        S = self.shape[1]
        out_shape = [B, S, heads, head_dim]
        out_q = Tensor(out_shape, device=self.device)
        out_k = Tensor(out_shape, device=self.device)
        out_v = Tensor(out_shape, device=self.device)

        if "opencl" in self.device: raise NotImplementedError("OpenCL fused_split_rope not implemented")
        if hasattr(self._lib, 'tensor_fused_split_rope'):
            self._lib.tensor_fused_split_rope(self._handle, cos._handle, sin._handle, out_q._handle, out_k._handle, out_v._handle)
        else:
            H = heads * head_dim
            q = self.slice([0, 0, 0], [B, S, H]).reshape(out_shape)
            k = self.slice([0, 0, H], [B, S, H]).reshape(out_shape)
            v = self.slice([0, 0, 2*H], [B, S, H]).reshape(out_shape)
            out_q, out_k = q.rope(k, cos, sin)
            out_v = v
        return out_q, out_k, out_v

    def argmax(self, dim: int = -1) -> 'Tensor':
        out_shape = list(self.shape[:-1])
        out = Tensor(out_shape, device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL argmax not implemented")
        if hasattr(self._lib, 'tensor_argmax'): self._lib.tensor_argmax(self._handle, out._handle)
        return out

    def embed(self, indices: 'Tensor') -> 'Tensor':
        out_shape = list(indices.shape) + [self.shape[1]]
        out = Tensor(out_shape, device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL embed not implemented")
        if hasattr(self._lib, 'tensor_embed'): self._lib.tensor_embed(self._handle, indices._handle, out._handle)
        return out

    def slice(self, start_indices: Union[List[int], 'Tensor'], slice_shapes: List[int]) -> 'Tensor':
        if isinstance(start_indices, Tensor):
             if start_indices.ndim != 1 or start_indices.size != self.ndim: raise ValueError(f"Slice Tensor indices must be 1D size={self.ndim}")
             out = Tensor(slice_shapes, device=self.device)
             if "opencl" in self.device: raise NotImplementedError("OpenCL tensor_slice_device not implemented")

             if hasattr(self._lib, 'tensor_slice_device'):
                 self._lib.tensor_slice_device(self._handle, out._handle, start_indices._handle)
             else:
                 # Fallback
                 if self.device == 'cpu':
                     h_start = [int(x) for x in start_indices.to_list()]
                     c_start = (ctypes.c_int * self.ndim)(*h_start)
                     c_shape = (ctypes.c_int * self.ndim)(*slice_shapes)
                     if hasattr(self._lib, 'tensor_slice'): self._lib.tensor_slice(self._handle, out._handle, c_start, c_shape)
                 else:
                     raise NotImplementedError("tensor_slice_device not available in backend lib")
             return out

        if len(start_indices) != self.ndim or len(slice_shapes) != self.ndim: raise ValueError("Slice args dim mismatch")
        out = Tensor(slice_shapes, device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL slice not implemented")
        if hasattr(self._lib, 'tensor_slice'):
            c_start = (ctypes.c_int * self.ndim)(*start_indices)
            c_shape = (ctypes.c_int * self.ndim)(*slice_shapes)
            self._lib.tensor_slice(self._handle, out._handle, c_start, c_shape)
        return out

    def set_slice(self, src: 'Tensor', start_indices: Union[List[int], 'Tensor']):
        if isinstance(start_indices, Tensor):
             if start_indices.ndim != 1 or start_indices.size != self.ndim: raise ValueError(f"SetSlice Tensor indices must be 1D size={self.ndim}")
             if "opencl" in self.device: raise NotImplementedError("OpenCL tensor_set_slice_device not implemented")
             if hasattr(self._lib, 'tensor_set_slice_device'):
                 self._lib.tensor_set_slice_device(self._handle, src._handle, start_indices._handle)
             else:
                 if self.device == 'cpu':
                     h_start = [int(x) for x in start_indices.to_list()]
                     c_start = (ctypes.c_int * self.ndim)(*h_start)
                     if hasattr(self._lib, 'tensor_set_slice'): self._lib.tensor_set_slice(self._handle, src._handle, c_start)
                 else:
                     raise NotImplementedError("tensor_set_slice_device not available in backend lib")
             return

        if len(start_indices) != self.ndim: raise ValueError("Set Slice args dim mismatch")
        if "opencl" in self.device: raise NotImplementedError("OpenCL set_slice not implemented")
        if hasattr(self._lib, 'tensor_set_slice'):
            c_start = (ctypes.c_int * self.ndim)(*start_indices)
            self._lib.tensor_set_slice(self._handle, src._handle, c_start)

    def resize_image(self, target_h: int, target_w: int) -> 'Tensor':
        if self.ndim != 3: raise ValueError("Image resize expects 3D tensor")
        if self.device != "cpu": raise ValueError("Image ops only on CPU currently")
        c, h, w = self.shape
        out = Tensor([c, target_h, target_w])
        if hasattr(self._lib, 'image_resize_bilinear'):
            self._lib.image_resize_bilinear(self._handle.contents.data, c, h, w, out._handle.contents.data, target_h, target_w)
        return out

    # --- MoE Operations ---

    def count_value(self, value: float) -> int:
        count = ctypes.c_int(0)
        if "opencl" in self.device: raise NotImplementedError("OpenCL count_value not implemented")
        if hasattr(self._lib, 'tensor_count_value'):
            self._lib.tensor_count_value(self._handle, ctypes.c_float(value), ctypes.byref(count))
        return count.value

    def gather_by_value(self, indices: 'Tensor', value: float) -> Tuple['Tensor', 'Tensor']:
        count = indices.count_value(value)
        if count == 0: return None, None
        hidden_size = self.shape[-1]
        out_data = Tensor([count, hidden_size], device=self.device)
        out_indices = Tensor([count], device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL gather_by_value not implemented")
        if hasattr(self._lib, 'tensor_gather_by_value'):
            self._lib.tensor_gather_by_value(self._handle, indices._handle, ctypes.c_float(value), out_data._handle, out_indices._handle)
        return out_data, out_indices

    def scatter_add_by_index(self, src: 'Tensor', indices: 'Tensor'):
        if "opencl" in self.device: raise NotImplementedError("OpenCL scatter_add_by_index not implemented")
        if hasattr(self._lib, 'tensor_scatter_add_by_index'):
            self._lib.tensor_scatter_add_by_index(self._handle, src._handle, indices._handle)

    def deltanet_recurrence(self, k: 'Tensor', v: 'Tensor', beta: 'Tensor', state: 'Tensor') -> 'Tensor':
        out = Tensor(list(self.shape), device=self.device)
        if "opencl" in self.device: raise NotImplementedError("OpenCL deltanet_recurrence not implemented")
        if hasattr(self._lib, 'tensor_deltanet_recurrence'):
            self._lib.tensor_deltanet_recurrence(self._handle, k._handle, v._handle, beta._handle, state._handle, out._handle)
        return out

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = False
    def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)
    def forward(self, *args, **kwargs): raise NotImplementedError
    def register_parameter(self, name: str, param: Optional[Tensor]): self._parameters[name] = param
    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True): self._parameters[name] = tensor
    def to(self, device: str, non_blocking: bool = False):
        for name, param in self._parameters.items():
            if param:
                new_param = param.to(device, non_blocking=non_blocking)
                self._parameters[name] = new_param
                # Update attribute if it exists to keep sync
                if hasattr(self, name):
                    setattr(self, name, new_param)
        for module in self._modules.values(): module.to(device, non_blocking=non_blocking)
        return self
    def parameters(self):
        for p in self._parameters.values():
            if p: yield p
        for m in self._modules.values(): yield from m.parameters()

    def state_dict(self, prefix: str = "", destination: Dict[str, Tensor] = None) -> Dict[str, Tensor]:
        if destination is None:
            destination = {}

        # Add parameters
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param

        # Add submodules
        for name, module in self._modules.items():
            module.state_dict(prefix + name + ".", destination)

        return destination

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = Tensor([out_features, in_features])
        self.bias = Tensor([out_features]) if bias else None
        self.register_parameter("weight", self.weight)
        if bias: self.register_parameter("bias", self.bias)
    def forward(self, input: Tensor) -> Tensor: return input.linear(self.weight, self.bias)

class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, groups: int = 1, bias: bool = True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.groups = groups
        # Weight shape: [C_out, C_in/groups, K, K]
        c_in_group = in_channels // groups
        self.weight = Tensor([out_channels, c_in_group, kernel_size, kernel_size])
        self.bias = Tensor([out_channels]) if bias else None
        self.register_parameter("weight", self.weight)
        if bias: self.register_parameter("bias", self.bias)
    def forward(self, input: Tensor) -> Tensor:
        return input.conv2d(self.weight, self.bias, self.stride, self.padding, self.groups)

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
    half_dim = dim // 2
    cos = Tensor([end, half_dim], device=device)
    sin = Tensor([end, half_dim], device=device)
    lib = _lib_cuda if "cuda" in device and HAS_CUDA else _lib_cpu
    if hasattr(lib, 'tensor_precompute_freqs_cis'):
        lib.tensor_precompute_freqs_cis(dim, end, ctypes.c_float(theta), cos._handle, sin._handle)
    return cos, sin

def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, scale: float = None) -> Tensor:
    """
    Fused Scaled Dot Product Attention: Softmax(Q * K^T / scale) * V
    Assumes [Batch, Seq, Heads, HeadDim] layout or similar.
    """
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    out = Tensor(list(q.shape), device=q.device)

    # Try Fused Kernel
    if hasattr(q._lib, 'tensor_scaled_dot_product_attention'):
        q._lib.tensor_scaled_dot_product_attention(q._handle, k._handle, v._handle, out._handle, ctypes.c_float(scale))
        return out
    else:
        # Fallback (Slow Python)
        scores = q.matmul(k, transpose_b=True)
        scores = scores * scale # Uses tensor_scale if available
        attn = scores.softmax()
        return attn.matmul(v)

def init_memory_pool(size_mb: int = 1024):
    """Initialize the static memory arena for CPU tensors."""
    if _lib_cpu and hasattr(_lib_cpu, 'init_memory_pool'):
        _lib_cpu.init_memory_pool(size_mb * 1024 * 1024)

def reset_memory_pool():
    if _lib_cpu and hasattr(_lib_cpu, 'reset_memory_pool'):
        _lib_cpu.reset_memory_pool()

class CUDAGraph:
    def __init__(self):
        self.lib = _lib_cuda
        if not self.lib: raise RuntimeError("CUDA backend not loaded")
    def capture_begin(self):
        if hasattr(self.lib, 'begin_capture'): self.lib.begin_capture()
    def capture_end(self):
        if hasattr(self.lib, 'end_capture'): self.lib.end_capture()
    def replay(self):
        if hasattr(self.lib, 'replay_graph'): self.lib.replay_graph()

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
