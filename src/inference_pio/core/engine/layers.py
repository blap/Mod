"""
Neural Network Layers (C Backend)
Dependency-Free
"""
from typing import Optional, Tuple, Union, List
from .backend import Tensor, Module as BackendModule

# Re-exporting from backend to maintain interface or extending
# Since backend.py has Module/Linear/etc, we can just alias or extend.
# To ensure clean separation, we alias.

Module = BackendModule

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Init weights (random not fully implemented in C, using zeros/ones placeholder or random via python loop)
        import random
        self.weight = Tensor([out_features, in_features])
        # Naive init
        self.weight.fill(0.01)
        self.bias = Tensor([out_features]) if bias else None
        if self.bias: self.bias.fill(0.0)

        self.register_parameter("weight", self.weight)
        if bias: self.register_parameter("bias", self.bias)

    def forward(self, input):
        return input.linear(self.weight, self.bias)

class Embedding(Module):
    def __init__(self, num, dim):
        super().__init__()
        self.weight = Tensor([num, dim])
        self.weight.fill(0.01)
        self.register_parameter("weight", self.weight)
    def forward(self, input): return self.weight.embed(input)

class RMSNorm(Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = Tensor([dim])
        self.weight.fill(1.0)
        self.eps = eps
        self.register_parameter("weight", self.weight)
    def forward(self, x): return x.rms_norm(self.weight, self.eps)

class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        self._modules_list = []
        if modules:
            for m in modules: self.append(m)
    def append(self, module):
        self._modules_list.append(module)
        self.add_module(str(len(self._modules_list)-1), module)
    def __getitem__(self, idx): return self._modules_list[idx]
    def __iter__(self): return iter(self._modules_list)
    def __len__(self): return len(self._modules_list)

class Conv2d(Module):
    def __init__(self, in_c, out_c, kernel, stride=1, padding=0, groups=1, bias=True):
        super().__init__()
        c_in_g = in_c // groups
        self.weight = Tensor([out_c, c_in_g, kernel, kernel])
        self.weight.fill(0.01)
        self.bias = Tensor([out_c]) if bias else None
        if self.bias: self.bias.fill(0.0)
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.register_parameter("weight", self.weight)
        if bias: self.register_parameter("bias", self.bias)
    def forward(self, x):
        return x.conv2d(self.weight, self.bias, self.stride, self.padding, self.groups)
