"""
Buddy Allocator System for Qwen3-VL

Este pacote fornece uma implementação otimizada do algoritmo Buddy Allocator
para gerenciamento eficiente de memória em modelos de linguagem de grande porte
com capacidades de visão (Qwen3-VL).

O sistema inclui otimizações específicas para hardware Intel i5-10210U + 
NVIDIA SM61 + NVMe SSD e suporte integrado para tensores PyTorch.
"""

from .buddy_allocator import (
    BuddyAllocator,
    PyTorchBuddyAllocator,
    OptimizedBuddyAllocator,
    create_default_allocator
)

__version__ = "1.0.0"
__author__ = "Qwen3-VL Team"
__all__ = [
    "BuddyAllocator",
    "PyTorchBuddyAllocator", 
    "OptimizedBuddyAllocator",
    "create_default_allocator"
]