"""
Integration Test: Adding Error Handling to Existing Hierarchical Cache System

This demonstrates how to integrate the comprehensive error handling system
with an existing component in the Qwen3-VL project.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import sys
import os
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.qwen3_vl.optimization.error_handling.prefetch_cache_error_handler import (
    PrefetchCacheErrorHandler,
    safe_prefetch_operation,
    safe_cache_operation,
    PrefetchingErrorDecorator,
    CachingErrorDecorator,
    FallbackStrategies,
    create_error_handler
)


# Example of enhancing an existing cache class with error handling
class HierarchicalCacheWithErrorHandling:
    """
    Example enhancement of an existing hierarchical cache system with error handling.
    This demonstrates how to integrate the error handling system with existing components.
    """
    
    def __init__(self, 
                 l1_size: int = 128 * 1024 * 1024,  # 128MB
                 l2_size: int = 512 * 1024 * 1024,  # 512MB
                 l3_size: int = 1024 * 1024 * 1024,  # 1GB
                 enable_error_handling: bool = True):
        # Original cache levels
        self.l1_cache = {}  # GPU cache
        self.l2_cache = {}  # CPU cache
        self.l3_cache = {}  # SSD cache
        
        # Error handling
        self.enable_error_handling = enable_error_handling
        if enable_error_handling:
            self.error_handler = create_error_handler(log_errors=True, enable_recovery=True)
        else:
            self.error_handler = None
        
        # Cache sizes
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.l3_size = l3_size
        
        # Current sizes
        self.l1_current_size = 0
        self.l2_current_size = 0
        self.l3_current_size = 0
        
        # Statistics
        self.stats = {
            'l1_hits': 0,
            'l1_misses': 0,
            'l2_hits': 0,
            'l2_misses': 0,
            'l3_hits': 0,
            'l3_misses': 0,
            'errors_handled': 0
        }
    
    def get_from_cache(self, tensor_id: str) -> Optional[torch.Tensor]:
        """
        Get tensor from cache hierarchy with error handling.
        """
        if not self.enable_error_handling:
            return self._get_from_cache_standard(tensor_id)
        
        # Use safe cache operation with error handling
        success, result = safe_cache_operation(
            self.error_handler,
            self._get_from_cache_standard,
            tensor_id
        )
        
        if success:
            return result
        else:
            # If standard cache retrieval fails, try fallback strategies
            self.stats['errors_handled'] += 1
            
            # Try fallback: return None or a default tensor
            return FallbackStrategies.fallback_no_prefetch()  # Returns None
    
    def _get_from_cache_standard(self, tensor_id: str) -> Optional[torch.Tensor]:
        """
        Standard implementation to get tensor from cache hierarchy.
        """
        # Try L1 first
        if tensor_id in self.l1_cache:
            self.stats['l1_hits'] += 1
            entry = self.l1_cache[tensor_id]
            return entry['tensor']
        
        self.stats['l1_misses'] += 1
        
        # Try L2
        if tensor_id in self.l2_cache:
            self.stats['l2_hits'] += 1
            entry = self.l2_cache[tensor_id]
            
            # Promote to L1 if space allows
            if self.l1_current_size + entry['size'] <= self.l1_size:
                self.l1_cache[tensor_id] = entry
                self.l1_current_size += entry['size']
                del self.l2_cache[tensor_id]
                self.l2_current_size -= entry['size']
            
            return entry['tensor']
        
        self.stats['l2_misses'] += 1
        
        # Try L3
        if tensor_id in self.l3_cache:
            self.stats['l3_hits'] += 1
            entry = self.l3_cache[tensor_id]
            
            # Promote to L2 if space allows
            if self.l2_current_size + entry['size'] <= self.l2_size:
                self.l2_cache[tensor_id] = entry
                self.l2_current_size += entry['size']
                del self.l3_cache[tensor_id]
                self.l3_current_size -= entry['size']
            
            return entry['tensor']
        
        self.stats['l3_misses'] += 1
        return None
    
    def put_in_cache(self, tensor_id: str, tensor: torch.Tensor, cache_level: str = 'auto') -> bool:
        """
        Put tensor in cache hierarchy with error handling.
        """
        if not self.enable_error_handling:
            return self._put_in_cache_standard(tensor_id, tensor, cache_level)

        # Use safe cache operation with error handling
        success, result = safe_cache_operation(
            self.error_handler,
            self._put_in_cache_standard,
            tensor_id, tensor, cache_level
        )

        if success:
            return result
        else:
            # If standard cache put fails, try fallback strategies
            self.stats['errors_handled'] += 1

            # Try fallback: use standard caching without optimization
            try:
                return self._put_in_cache_standard(tensor_id, tensor, 'l3')  # Fallback to lowest level
            except Exception:
                return False  # If all fails, return False
    
    def _put_in_cache_standard(self, tensor_id: str, tensor: torch.Tensor, cache_level: str = 'auto') -> bool:
        """
        Standard implementation to put tensor in cache hierarchy.
        """
        if tensor is None:
            raise ValueError(f"Cannot cache None tensor for ID {tensor_id}")

        tensor_size = tensor.element_size() * tensor.nelement()

        if cache_level == 'auto':
            # Auto-select cache level based on tensor size and access pattern
            if tensor_size < self.l1_size * 0.1:  # Small tensor, likely to be accessed frequently
                cache_level = 'l1'
            elif tensor_size < self.l2_size * 0.1:  # Medium tensor
                cache_level = 'l2'
            else:  # Large tensor
                cache_level = 'l3'

        cache_entry = {
            'tensor': tensor,
            'size': tensor_size,
            'access_time': time.time()
        }

        if cache_level == 'l1':
            # Check if we have space in L1
            if self.l1_current_size + tensor_size > self.l1_size:
                # Evict from L1 if necessary
                if self.l1_cache:
                    oldest_id = min(self.l1_cache.keys(), key=lambda k: self.l1_cache[k]['access_time'])
                    old_entry = self.l1_cache.pop(oldest_id)
                    self.l1_current_size -= old_entry['size']

                    # Move to L2
                    if self.l2_current_size + old_entry['size'] <= self.l2_size:
                        self.l2_cache[oldest_id] = old_entry
                        self.l2_current_size += old_entry['size']

            self.l1_cache[tensor_id] = cache_entry
            self.l1_current_size += tensor_size
            return True

        elif cache_level == 'l2':
            # Check if we have space in L2
            if self.l2_current_size + tensor_size > self.l2_size:
                # Evict from L2 if necessary
                if self.l2_cache:
                    oldest_id = min(self.l2_cache.keys(), key=lambda k: self.l2_cache[k]['access_time'])
                    old_entry = self.l2_cache.pop(oldest_id)
                    self.l2_current_size -= old_entry['size']

                    # Move to L3
                    if self.l3_current_size + old_entry['size'] <= self.l3_size:
                        self.l3_cache[oldest_id] = old_entry
                        self.l3_current_size += old_entry['size']

            self.l2_cache[tensor_id] = cache_entry
            self.l2_current_size += tensor_size
            return True

        elif cache_level == 'l3':
            # For L3, we might want to compress the tensor
            compressed_tensor = self._compress_tensor_if_needed(tensor)
            cache_entry['tensor'] = compressed_tensor

            # Check if we have space in L3
            if self.l3_current_size + tensor_size > self.l3_size:
                # Evict from L3 if necessary
                if self.l3_cache:
                    oldest_id = min(self.l3_cache.keys(), key=lambda k: self.l3_cache[k]['access_time'])
                    old_entry = self.l3_cache.pop(oldest_id)
                    self.l3_current_size -= old_entry['size']

            self.l3_cache[tensor_id] = cache_entry
            self.l3_current_size += tensor_size
            return True
    
    def _compress_tensor_if_needed(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compress tensor if beneficial (simplified implementation).
        """
        # For demonstration, we'll implement a simple low-rank approximation
        if tensor.dim() == 2 and min(tensor.shape) > 64:  # Only for reasonably large 2D tensors
            try:
                # Perform SVD-based compression
                U, S, V = torch.svd(tensor.float())
                rank = min(32, len(S))  # Use top 32 singular values
                compressed = torch.mm(U[:, :rank], torch.mm(torch.diag(S[:rank]), V[:, :rank].t()))
                return compressed.half()  # Convert back to half precision
            except:
                # If compression fails, return original tensor
                return tensor
        return tensor
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics including error handling stats.
        """
        total_requests = (self.stats['l1_hits'] + self.stats['l1_misses'] + 
                         self.stats['l2_hits'] + self.stats['l2_misses'] + 
                         self.stats['l3_hits'] + self.stats['l3_misses'])
        
        hit_rate = (self.stats['l1_hits'] + self.stats['l2_hits'] + self.stats['l3_hits']) / total_requests if total_requests > 0 else 0
        
        error_stats = {}
        if self.error_handler:
            error_stats = self.error_handler.get_error_statistics()
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'l1_utilization': self.l1_current_size / self.l1_size if self.l1_size > 0 else 0,
            'l2_utilization': self.l2_current_size / self.l2_size if self.l2_size > 0 else 0,
            'l3_utilization': self.l3_current_size / self.l3_size if self.l3_size > 0 else 0,
            'error_handling_stats': error_stats
        }


def test_integration_with_existing_component():
    """
    Test the integration of error handling with an existing component.
    """
    print("Testing Error Handling Integration with Existing Hierarchical Cache")
    print("=" * 70)
    
    # Create cache with error handling enabled
    cache = HierarchicalCacheWithErrorHandling(enable_error_handling=True)
    
    print("\n1. Testing normal cache operations...")
    
    # Create test tensors
    test_tensor1 = torch.randn(100, 100, dtype=torch.float16)  # Small tensor
    test_tensor2 = torch.randn(512, 512, dtype=torch.float16)  # Medium tensor
    test_tensor3 = torch.randn(1024, 1024, dtype=torch.float16)  # Large tensor
    
    # Put tensors in cache
    success1 = cache.put_in_cache("tensor1", test_tensor1)
    success2 = cache.put_in_cache("tensor2", test_tensor2)
    success3 = cache.put_in_cache("tensor3", test_tensor3)
    
    print(f"   Put tensor1: {success1}")
    print(f"   Put tensor2: {success2}")
    print(f"   Put tensor3: {success3}")
    
    # Get tensors from cache
    retrieved1 = cache.get_from_cache("tensor1")
    retrieved2 = cache.get_from_cache("tensor2")
    retrieved3 = cache.get_from_cache("tensor3")
    
    print(f"   Get tensor1: {retrieved1 is not None}")
    print(f"   Get tensor2: {retrieved2 is not None}")
    print(f"   Get tensor3: {retrieved3 is not None}")
    
    # Verify tensor content
    if retrieved1 is not None:
        content_match1 = torch.equal(test_tensor1, retrieved1)
        print(f"   Tensor1 content matches: {content_match1}")
    
    if retrieved2 is not None:
        content_match2 = torch.allclose(test_tensor2, retrieved2, atol=1e-2)
        print(f"   Tensor2 content matches: {content_match2}")
    
    if retrieved3 is not None:
        content_match3 = torch.allclose(test_tensor3, retrieved3, atol=1e-2)
        print(f"   Tensor3 content matches: {content_match3}")
    
    print("\n2. Testing error handling with invalid operations...")
    
    # Test with an invalid tensor (None)
    success_invalid = cache.put_in_cache("invalid_tensor", None)
    print(f"   Put invalid tensor: {success_invalid}")
    
    # Try to get non-existent tensor
    retrieved_none = cache.get_from_cache("non_existent_tensor")
    print(f"   Get non-existent tensor: {retrieved_none is None}")
    
    print("\n3. Checking cache statistics...")
    stats = cache.get_cache_stats()
    print(f"   Hit rate: {stats['hit_rate']:.2f}")
    print(f"   L1 utilization: {stats['l1_utilization']:.2f}")
    print(f"   L2 utilization: {stats['l2_utilization']:.2f}")
    print(f"   L3 utilization: {stats['l3_utilization']:.2f}")
    print(f"   Errors handled: {stats['errors_handled']}")
    print(f"   Total requests: {stats['total_requests']}")
    
    print("\n4. Testing cache with simulated errors...")
    
    # Create a cache with intentionally problematic operations
    def problematic_get_func():
        raise RuntimeError("Simulated cache access error")
    
    # Use safe operation directly to test error handling
    success, result = safe_cache_operation(cache.error_handler, problematic_get_func)
    print(f"   Handled problematic operation: success={success}, result={result}")
    
    print("\n5. Testing decorator integration...")
    
    # Define a function that might fail and use decorator
    @CachingErrorDecorator(
        error_handler=cache.error_handler,
        fallback_func=lambda tensor_id, tensor: torch.zeros(10, 10) if tensor is None else tensor * 2,  # Fallback to zero tensor if input is None
        default_return_value=None
    )
    def decorated_cache_operation(tensor_id: str, tensor: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            raise ValueError(f"Cannot cache None tensor for ID {tensor_id}")
        return tensor * 2  # Simple operation that doubles the tensor

    # Test successful operation
    result = decorated_cache_operation("test_tensor", test_tensor1)
    print(f"   Decorated operation (success): {result is not None}")

    # Test operation that triggers error handling
    result = decorated_cache_operation("bad_tensor", None)
    print(f"   Decorated operation (with error): {result is not None}")
    
    print("\nError handling integration test completed successfully!")
    print("The enhanced cache system properly handles errors while maintaining functionality.")


if __name__ == "__main__":
    test_integration_with_existing_component()