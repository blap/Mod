"""
Neural Network Layers for Numpy-based Inference Engine
Provides Drop-in replacements for standard NN layers using pure Numpy.
"""

import numpy as np
from typing import Optional, Tuple, Union

class Module:
    """Base class for all neural network modules."""
    def __init__(self):
        self._parameters = {}
        self._buffers = {}
        self._modules = {}
        self.training = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def register_parameter(self, name: str, param: np.ndarray):
        if param is None:
            self._parameters[name] = None
        else:
            self._parameters[name] = param

    def register_buffer(self, name: str, buffer: np.ndarray):
        self._buffers[name] = buffer

    def add_module(self, name: str, module: 'Module'):
        self._modules[name] = module

    def to(self, *args, **kwargs):
        """Move logic (noop for numpy on CPU)."""
        return self

    def eval(self):
        self.training = False
        for module in self._modules.values():
            module.eval()

    def train(self):
        self.training = True
        for module in self._modules.values():
            module.train()

    def state_dict(self):
        # Simplified state dict
        sd = {}
        for k, v in self._parameters.items():
            if v is not None: sd[k] = v
        for k, v in self._buffers.items():
            if v is not None: sd[k] = v
        for name, module in self._modules.items():
            for k, v in module.state_dict().items():
                sd[f"{name}.{k}"] = v
        return sd

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        # Simplified loading logic
        missing_keys = []
        unexpected_keys = list(state_dict.keys())

        # Load parameters
        for name, param in self._parameters.items():
            if name in state_dict:
                if param is not None and param.shape != state_dict[name].shape:
                     # Check for transpose match (Linear weights often transposed)
                     if param.shape == state_dict[name].T.shape:
                         self._parameters[name] = state_dict[name].T
                     else:
                         print(f"Shape mismatch for {name}: expected {param.shape}, got {state_dict[name].shape}")
                else:
                    self._parameters[name] = state_dict[name]
                unexpected_keys.remove(name)
            else:
                missing_keys.append(name)

        # Load buffers
        for name, buffer in self._buffers.items():
            if name in state_dict:
                 self._buffers[name] = state_dict[name]
                 unexpected_keys.remove(name)

        # Recursively load submodules
        # Note: keys in state_dict are flattened (e.g. "layer.0.weight")
        # We need to route appropriate keys to submodules
        for module_name, module in self._modules.items():
            prefix = f"{module_name}."
            sub_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            if sub_dict:
                module.load_state_dict(sub_dict, strict=strict)
                # Remove keys consumed by submodule from unexpected list
                for k in sub_dict:
                    if f"{prefix}{k}" in unexpected_keys:
                        unexpected_keys.remove(f"{prefix}{k}")

        if strict and (missing_keys or unexpected_keys):
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights (random or zeros, usually overwritten by load_state_dict)
        # Weight shape matches PyTorch convention: [out_features, in_features]
        self.weight = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
        if bias:
            self.bias = np.zeros(out_features, dtype=np.float32)
        else:
            self.bias = None

        self.register_parameter("weight", self.weight)
        if bias:
            self.register_parameter("bias", self.bias)

    def forward(self, input: np.ndarray) -> np.ndarray:
        # input: [..., in_features]
        # output: [..., out_features]
        return np.matmul(input, self.weight.T) + (self.bias if self.bias is not None else 0)

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.weight = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
        if padding_idx is not None:
            self.weight[padding_idx] = 0

        self.register_parameter("weight", self.weight)

    def forward(self, input: np.ndarray) -> np.ndarray:
        # input: [batch, seq_len] indices
        # output: [batch, seq_len, embed_dim]
        return self.weight[input]

class RMSNorm(Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = np.ones(hidden_size, dtype=np.float32)
        self.eps = eps
        self.register_parameter("weight", self.weight)

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        variance = np.mean(hidden_states**2, axis=-1, keepdims=True)
        hidden_states = hidden_states * (1.0 / np.sqrt(variance + self.eps))
        return self.weight * hidden_states

class LayerNorm(Module):
    def __init__(self, normalized_shape: Union[int, Tuple[int]], eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = np.ones(normalized_shape, dtype=np.float32)
            self.bias = np.zeros(normalized_shape, dtype=np.float32)
            self.register_parameter("weight", self.weight)
            self.register_parameter("bias", self.bias)
        else:
            self.weight = None
            self.bias = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        mean = np.mean(input, axis=-1, keepdims=True)
        var = np.var(input, axis=-1, keepdims=True)
        x = (input - mean) / np.sqrt(var + self.eps)
        if self.elementwise_affine:
            return x * self.weight + self.bias
        return x

class ModuleList(Module):
    def __init__(self, modules: Optional[list] = None):
        super().__init__()
        self._modules_list = []
        if modules:
            for i, module in enumerate(modules):
                self.append(module)

    def append(self, module: Module):
        self._modules_list.append(module)
        self.add_module(str(len(self._modules_list) - 1), module)

    def __iter__(self):
        return iter(self._modules_list)

    def __getitem__(self, idx):
        return self._modules_list[idx]

    def __len__(self):
        return len(self._modules_list)

class Conv2d(Module):
    """Simplified Conv2d for Patch Embedding (stride=kernel_size)."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]] = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # [out_channels, in_channels, kH, kW]
        self.weight = np.random.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]).astype(np.float32) * 0.02
        if bias:
            self.bias = np.zeros(out_channels, dtype=np.float32)
        else:
            self.bias = None

        self.register_parameter("weight", self.weight)
        if bias:
            self.register_parameter("bias", self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: [B, C, H, W]
        # Only implementing for stride=kernel_size (Patch Embedding use case) for efficiency
        # Otherwise need im2col
        B, C, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride

        # Output dimensions
        oH = (H + 2 * self.padding - kH) // sH + 1
        oW = (W + 2 * self.padding - kW) // sW + 1

        # Naive implementation for stride == kernel size
        if sH == kH and sW == kW and self.padding == 0:
            # Reshape input to isolate patches
            # [B, C, oH, sH, oW, sW] -> [B, oH, oW, C, sH, sW]
            patches = x.reshape(B, C, oH, sH, oW, sW).transpose(0, 2, 4, 1, 3, 5)
            # Flatten patches: [B, oH, oW, C*sH*sW]
            patches_flat = patches.reshape(B, oH, oW, -1)

            # Reshape weights: [out_channels, C, kH, kW] -> [out_channels, C*kH*kW]
            weights_flat = self.weight.reshape(self.out_channels, -1)

            # Matmul: [B, oH, oW, InFeatures] @ [OutFeatures, InFeatures].T -> [B, oH, oW, OutFeatures]
            out = np.matmul(patches_flat, weights_flat.T)

            if self.bias is not None:
                out += self.bias

            # Permute back to [B, OutFeatures, oH, oW]
            return out.transpose(0, 3, 1, 2)
        else:
            raise NotImplementedError("Only stride=kernel_size supported for efficient numpy Conv2d currently.")
