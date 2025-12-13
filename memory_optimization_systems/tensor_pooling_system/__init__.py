"""
Tensor Pooling System - Pacote para gerenciamento eficiente de memória de tensores.

Este módulo fornece um sistema de pooling de tensores otimizado para reduzir alocações 
e desalocações frequentes de memória, especialmente útil em aplicações de deep learning
com grandes volumes de operações tensoriais.
"""

from .memory_pool import MemoryPool, DataType, allocate_tensor, deallocate_tensor, get_global_memory_pool

__all__ = [
    'MemoryPool',
    'DataType',
    'allocate_tensor',
    'deallocate_tensor',
    'get_global_memory_pool'
]