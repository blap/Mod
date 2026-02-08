"""
Neural Network Layers (C Backend)
"""
from typing import Optional, Tuple, Union
from .backend import Tensor, randn, zeros

class Module:
    def __init__(self):
        self._parameters = {}
        self._buffers = {}
        self._modules = {}
        self.training = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def register_parameter(self, name: str, param: Tensor):
        self._parameters[name] = param

    def register_buffer(self, name: str, buffer: Tensor):
        self._buffers[name] = buffer

    def add_module(self, name: str, module: 'Module'):
        self._modules[name] = module

    def state_dict(self):
        sd = {}
        for k, v in self._parameters.items():
            if v is not None: sd[k] = v
        for k, v in self._buffers.items():
            if v is not None: sd[k] = v
        for name, module in self._modules.items():
            for k, v in module.state_dict().items():
                sd[f"{name}.{k}"] = v
        return sd

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = randn([out_features, in_features])
        self.bias = zeros([out_features]) if bias else None
        self.register_parameter("weight", self.weight)
        if bias: self.register_parameter("bias", self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return input.linear(self.weight, self.bias)

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.weight = randn([num_embeddings, embedding_dim])
        self.register_parameter("weight", self.weight)

    def forward(self, input: Tensor) -> Tensor:
        # NOTE: Gather is now implicitly handled by C backend if indices passed
        # OR we assume dense input for this prototype.
        # Ideally: input.gather(self.weight)
        # For prototype: return full weights linearly transformed (incorrect but functional for flow)
        # TODO: Implement proper C-gather
        return self.weight # Placeholder

class RMSNorm(Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = zeros([hidden_size])
        self.weight.fill(1.0)
        self.eps = eps
        self.register_parameter("weight", self.weight)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states.rms_norm(self.weight, self.eps)

class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        self._modules_list = []
        if modules:
            for m in modules: self.append(m)

    def append(self, module):
        self._modules_list.append(module)
        self.add_module(str(len(self._modules_list)-1), module)

    def __getitem__(self, idx):
        return self._modules_list[idx]

    def __iter__(self):
        return iter(self._modules_list)

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = randn([out_channels, in_channels]) # Simplified
        self.bias = zeros([out_channels]) if bias else None

    def forward(self, x):
        return x
