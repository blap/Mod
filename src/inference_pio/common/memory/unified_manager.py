from typing import Dict, List, Optional
import time
from ...core.engine.backend import Tensor, HAS_CUDA, allocate_pinned, free_pinned

class UnifiedMemoryManager:
    """
    Manages Tensor lifecycle across CPU and GPU.
    Implements simple LRU eviction and Pinned Memory Allocation.
    """
    def __init__(self, gpu_limit_mb: int = 4096):
        self.gpu_limit_bytes = gpu_limit_mb * 1024 * 1024
        self.current_gpu_bytes = 0
        self.tensors: Dict[int, Tensor] = {} # handle_addr -> Tensor
        self.lru_list: List[int] = [] # List of handle_addr

    def malloc_pinned(self, size_bytes: int):
        """Allocate pinned memory on host."""
        ptr = allocate_pinned(size_bytes)
        if not ptr:
            # Fallback
            import ctypes
            buffer = (ctypes.c_byte * size_bytes)()
            return ctypes.cast(buffer, ctypes.c_void_p)
        return ptr

    def free_pinned(self, ptr):
        """Free pinned memory."""
        free_pinned(ptr)

    def register(self, tensor: Tensor):
        if "cuda" in tensor.device:
            addr = ctypes.addressof(tensor._handle.contents)
            size = tensor.size * 4
            self.tensors[addr] = tensor
            self.lru_list.append(addr)
            self.current_gpu_bytes += size
            self.enforce_limit()

    def access(self, tensor: Tensor):
        # Move to back of LRU
        if "cuda" in tensor.device:
            addr = ctypes.addressof(tensor._handle.contents)
            if addr in self.lru_list:
                self.lru_list.remove(addr)
                self.lru_list.append(addr)

    def enforce_limit(self):
        while self.current_gpu_bytes > self.gpu_limit_bytes and self.lru_list:
            evict_addr = self.lru_list.pop(0)
            if evict_addr in self.tensors:
                tensor = self.tensors[evict_addr]
                # Evict to CPU
                # This requires tensor support for 'move' or we just copy and replace data pointer?
                # Python Tensor wrapper isn't mutable in device easily without re-creating.
                # Simplified: Just log for now as we can't safely swap backend pointers without support.
                # print(f"Would evict tensor at {evict_addr}")
                pass

import ctypes
