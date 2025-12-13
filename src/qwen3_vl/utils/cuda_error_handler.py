"""
CUDA Error Handler Module for Qwen3-VL-2B-Instruct Project

This module provides comprehensive error handling for CUDA operations,
including memory allocation failures, kernel launch errors, and proper
cleanup mechanisms.
"""
import torch
import logging
from typing import Optional, Union, Tuple, Any
from contextlib import contextmanager
import sys
import traceback
import warnings
from functools import wraps


class CUDAErrorHandler:
    """
    Comprehensive error handler for CUDA operations with logging and fallback mechanisms.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cuda_available = torch.cuda.is_available()
        
        # Check if CUDA is available and log device info
        if self.cuda_available:
            self.logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
            for i in range(torch.cuda.device_count()):
                self.logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            self.logger.warning("CUDA is not available on this system")
    
    def check_cuda_status(self) -> bool:
        """
        Check if CUDA is available and properly initialized.
        
        Returns:
            bool: True if CUDA is available, False otherwise
        """
        if not self.cuda_available:
            return False
            
        try:
            # Try a simple CUDA operation to verify functionality
            torch.cuda.synchronize()
            return True
        except RuntimeError as e:
            self.logger.error(f"CUDA runtime error: {e}")
            return False
    
    def handle_cuda_error(self, operation_name: str, exception: Exception) -> bool:
        """
        Handle CUDA errors with logging and return whether to fallback to CPU.
        
        Args:
            operation_name: Name of the operation that failed
            exception: The exception that occurred
            
        Returns:
            bool: True if should fallback to CPU, False if should raise
        """
        error_msg = f"CUDA operation '{operation_name}' failed: {str(exception)}"
        
        if isinstance(exception, torch.cuda.OutOfMemoryError):
            self.logger.error(f"Out of memory error in {operation_name}: {exception}")
            # Always fallback for OOM errors
            return True
        elif isinstance(exception, RuntimeError) and "CUDA" in str(exception):
            self.logger.error(f"CUDA runtime error in {operation_name}: {exception}")
            # Check if it's a recoverable error
            if "out of memory" in str(exception).lower():
                return True
            else:
                # For other CUDA runtime errors, log and potentially fallback
                return True
        else:
            self.logger.error(error_msg)
            # For non-CUDA specific errors, don't fallback
            return False
    
    def check_memory_usage(self) -> Tuple[float, float, float]:
        """
        Check current GPU memory usage.
        
        Returns:
            Tuple of (allocated_memory, reserved_memory, max_memory) in MB
        """
        if not self.cuda_available:
            return 0.0, 0.0, 0.0
            
        try:
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # Convert to MB
            max_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # Convert to MB
            
            return allocated, reserved, max_memory
        except Exception as e:
            self.logger.warning(f"Could not get memory stats: {e}")
            return 0.0, 0.0, 0.0
    
    @contextmanager
    def cuda_memory_guard(self, operation_name: str, max_memory_ratio: float = 0.9):
        """
        Context manager to guard against memory allocation failures.
        
        Args:
            operation_name: Name of the operation
            max_memory_ratio: Maximum ratio of memory to use (default 0.9 = 90%)
        """
        initial_allocated, initial_reserved, max_memory = self.check_memory_usage()
        
        if max_memory > 0:
            current_usage_ratio = initial_allocated / max_memory
            if current_usage_ratio > max_memory_ratio:
                warning_msg = (
                    f"GPU memory usage is high ({current_usage_ratio:.2%}) "
                    f"for operation '{operation_name}'. "
                    f"Consider clearing cache or reducing batch size."
                )
                self.logger.warning(warning_msg)
        
        try:
            yield
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"Out of memory during {operation_name}: {e}")
            
            # Try to clear cache
            torch.cuda.empty_cache()
            
            # Log memory stats after clearing cache
            post_clear_allocated, post_clear_reserved, _ = self.check_memory_usage()
            self.logger.info(
                f"Memory after cache clear: allocated={post_clear_allocated:.2f}MB, "
                f"reserved={post_clear_reserved:.2f}MB"
            )
            
            # Re-raise the exception
            raise
        except Exception as e:
            self.logger.error(f"Error during {operation_name}: {e}")
            raise
    
    def safe_cuda_call(self, func, *args, fallback_func=None, **kwargs) -> Any:
        """
        Safely execute a CUDA function with fallback.

        Args:
            func: The CUDA function to call
            *args: Arguments to pass to the function
            fallback_func: Optional fallback function to call if CUDA fails
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of the function call
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            should_fallback = self.handle_cuda_error(func.__name__ if hasattr(func, '__name__') else 'unknown', e)

            if should_fallback and fallback_func is not None:
                self.logger.info(f"Falling back to CPU implementation for {func.__name__ if hasattr(func, '__name__') else 'unknown'}")
                return fallback_func(*args, **kwargs)
            else:
                raise e
    
    def check_kernel_launch_status(self) -> bool:
        """
        Check if the last kernel launch was successful.
        
        Returns:
            bool: True if no kernel errors, False otherwise
        """
        try:
            torch.cuda.synchronize()
            return True
        except RuntimeError as e:
            self.logger.error(f"Kernel launch error detected: {e}")
            return False
    
    def safe_tensor_operation(self, operation_name: str, tensor_func, *args, **kwargs) -> Any:
        """
        Perform a tensor operation safely with error handling.

        Args:
            operation_name: Name of the operation for logging
            tensor_func: The tensor operation function
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of the tensor operation
        """
        try:
            # Check if tensors are on CUDA
            cuda_tensors = [arg for arg in args if torch.is_tensor(arg) and arg.is_cuda]
            if not cuda_tensors and 'input_tensor' in kwargs:
                cuda_tensors = [kwargs['input_tensor']] if kwargs['input_tensor'].is_cuda else []
            
            if cuda_tensors:
                # Check memory before operation
                allocated, reserved, max_memory = self.check_memory_usage()
                if max_memory > 0:
                    usage_ratio = reserved / max_memory
                    if usage_ratio > 0.85:  # 85% threshold
                        self.logger.warning(
                            f"High GPU memory usage ({usage_ratio:.2%}) before {operation_name}"
                        )
            
            result = tensor_func(*args, **kwargs)
            
            # Check for kernel errors after operation
            if not self.check_kernel_launch_status():
                raise RuntimeError(f"Kernel error detected after {operation_name}")
                
            return result
            
        except Exception as e:
            should_fallback = self.handle_cuda_error(operation_name, e)
            
            if should_fallback:
                # Try to move tensors to CPU and retry
                cpu_args = []
                for arg in args:
                    if torch.is_tensor(arg) and arg.is_cuda:
                        cpu_args.append(arg.cpu())
                    else:
                        cpu_args.append(arg)
                
                cpu_kwargs = {}
                for k, v in kwargs.items():
                    if torch.is_tensor(v) and v.is_cuda:
                        cpu_kwargs[k] = v.cpu()
                    else:
                        cpu_kwargs[k] = v
                
                # Retry operation on CPU
                try:
                    self.logger.info(f"Falling back to CPU for {operation_name}")
                    return tensor_func(*cpu_args, **cpu_kwargs)
                except Exception as cpu_e:
                    self.logger.error(f"CPU fallback also failed for {operation_name}: {cpu_e}")
                    raise e  # Re-raise original CUDA error
            else:
                raise e


class MemoryPoolManager:
    """
    Enhanced memory pool manager with error handling and fallback mechanisms.
    """
    
    def __init__(self, initial_pool_size: int = 64 * 1024 * 1024):  # 64MB default
        self.initial_pool_size = initial_pool_size
        self.error_handler = CUDAErrorHandler()
        self.logger = logging.getLogger(__name__)
        
        # Try to initialize memory pool, with fallback to PyTorch allocation
        self._memory_pool = None
        self._use_fallback = True
        
        if self.error_handler.check_cuda_status():
            try:
                # Attempt to import CUDA extension for memory pool
                import sys
                import os
                cuda_kernels_dir = os.path.join(os.path.dirname(__file__), '..', 'cuda_kernels')
                sys.path.insert(0, cuda_kernels_dir)
                
                import sm61_cuda_kernels
                self._memory_pool = sm61_cuda_kernels.SM61MemoryPool(initial_pool_size)
                self._use_fallback = False
                self.logger.info("Successfully initialized SM61 memory pool")
            except ImportError as e:
                self.logger.warning(f"Could not import SM61 memory pool: {e}, using PyTorch fallback")
            except Exception as e:
                self.logger.error(f"Error initializing SM61 memory pool: {e}")
    
    def allocate_tensor(self, sizes: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Allocate a tensor with error handling and fallback.
        
        Args:
            sizes: Size tuple for the tensor
            dtype: Data type of the tensor
            
        Returns:
            Allocated tensor
        """
        if self._use_fallback or self._memory_pool is None:
            # Fallback to PyTorch allocation
            try:
                with self.error_handler.cuda_memory_guard(f"tensor_allocation_{sizes}"):
                    return torch.empty(sizes, dtype=dtype, device='cuda' if self.error_handler.cuda_available else 'cpu')
            except torch.cuda.OutOfMemoryError:
                self.logger.warning(f"Out of memory during tensor allocation of size {sizes}, clearing cache...")
                torch.cuda.empty_cache()
                
                # Try again after clearing cache
                try:
                    return torch.empty(sizes, dtype=dtype, device='cuda' if self.error_handler.cuda_available else 'cpu')
                except torch.cuda.OutOfMemoryError:
                    self.logger.error(f"Still out of memory after cache clear, falling back to CPU for tensor {sizes}")
                    return torch.empty(sizes, dtype=dtype, device='cpu')
        else:
            # Try to use memory pool
            try:
                return self.error_handler.safe_cuda_call(
                    lambda: self._memory_pool.allocate_tensor(list(sizes), dtype),
                    fallback_func=lambda: torch.empty(sizes, dtype=dtype, device='cuda')
                )
            except Exception as e:
                self.logger.warning(f"Memory pool allocation failed: {e}, falling back to PyTorch allocation")
                try:
                    with self.error_handler.cuda_memory_guard(f"tensor_allocation_{sizes}"):
                        return torch.empty(sizes, dtype=dtype, device='cuda' if self.error_handler.cuda_available else 'cpu')
                except torch.cuda.OutOfMemoryError:
                    return torch.empty(sizes, dtype=dtype, device='cpu')
    
    def get_stats(self) -> dict:
        """
        Get memory pool statistics with error handling.
        
        Returns:
            Dictionary with memory statistics
        """
        if self._use_fallback or self._memory_pool is None:
            # Return PyTorch memory stats
            try:
                allocated, reserved, max_memory = self.error_handler.check_memory_usage()
                return {
                    "total_size": max_memory if max_memory > 0 else self.initial_pool_size / (1024**2),
                    "allocated": allocated,
                    "reserved": reserved,
                    "free": max_memory - allocated if max_memory > 0 else self.initial_pool_size / (1024**2) - allocated,
                    "fragmentation": 0.0,  # Not applicable for PyTorch
                    "num_free_blocks": 0    # Not applicable for PyTorch
                }
            except Exception as e:
                self.logger.error(f"Error getting memory stats: {e}")
                return {
                    "total_size": 0.0,
                    "allocated": 0.0,
                    "reserved": 0.0,
                    "free": 0.0,
                    "fragmentation": 0.0,
                    "num_free_blocks": 0
                }
        else:
            try:
                return self._memory_pool.get_stats()
            except Exception as e:
                self.logger.warning(f"Could not get memory pool stats: {e}, using PyTorch stats")
                try:
                    allocated, reserved, max_memory = self.error_handler.check_memory_usage()
                    return {
                        "total_size": max_memory if max_memory > 0 else self.initial_pool_size / (1024**2),
                        "allocated": allocated,
                        "reserved": reserved,
                        "free": max_memory - allocated if max_memory > 0 else self.initial_pool_size / (1024**2) - allocated,
                        "fragmentation": 0.0,
                        "num_free_blocks": 0
                    }
                except Exception:
                    return {
                        "total_size": 0.0,
                        "allocated": 0.0,
                        "reserved": 0.0,
                        "free": 0.0,
                        "fragmentation": 0.0,
                        "num_free_blocks": 0
                    }
    
    def clear_cache(self):
        """
        Clear PyTorch CUDA cache to free up memory.
        """
        if self.error_handler.cuda_available:
            torch.cuda.empty_cache()
            self.logger.info("Cleared PyTorch CUDA cache")


# Global instance for use across the application
cuda_error_handler = CUDAErrorHandler()
memory_pool_manager = MemoryPoolManager()


def cuda_error_handler_decorator(func):
    """
    Decorator to add comprehensive error handling to CUDA operations.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        handler = CUDAErrorHandler()
        operation_name = f"{func.__module__}.{func.__name__}"
        
        try:
            # Log memory usage before operation
            allocated, reserved, max_memory = handler.check_memory_usage()
            if max_memory > 0:
                handler.logger.debug(
                    f"Before {operation_name}: allocated={allocated:.2f}MB, "
                    f"reserved={reserved:.2f}MB ({reserved/max_memory:.2%})"
                )
            
            # Execute the function within memory guard
            with handler.cuda_memory_guard(operation_name):
                result = func(*args, **kwargs)
            
            # Check for kernel errors after execution
            if not handler.check_kernel_launch_status():
                raise RuntimeError(f"Kernel error detected after {operation_name}")
            
            # Log memory usage after operation
            post_allocated, post_reserved, _ = handler.check_memory_usage()
            handler.logger.debug(
                f"After {operation_name}: allocated={post_allocated:.2f}MB, "
                f"reserved={post_reserved:.2f}MB"
            )
            
            return result
            
        except Exception as e:
            # Handle the error appropriately
            should_fallback = handler.handle_cuda_error(operation_name, e)
            
            if should_fallback:
                handler.logger.info(f"Attempting fallback for {operation_name}")
                # In a real implementation, we would have specific fallback logic
                # For now, we'll just log and potentially restructure the call
                raise e
            else:
                raise e
    
    return wrapper


def safe_cuda_execution(func, *args, fallback_func=None, **kwargs) -> Any:
    """
    Execute a CUDA function safely with automatic fallback.

    Args:
        func: The CUDA function to execute
        *args: Arguments to pass to the function
        fallback_func: Optional fallback function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Result of the function execution
    """
    handler = CUDAErrorHandler()
    operation_name = getattr(func, '__name__', 'unknown')
    
    try:
        with handler.cuda_memory_guard(operation_name):
            result = func(*args, **kwargs)
            
            # Verify kernel execution
            if not handler.check_kernel_launch_status():
                raise RuntimeError(f"Kernel execution error in {operation_name}")
                
            return result
    except Exception as e:
        should_fallback = handler.handle_cuda_error(operation_name, e)
        
        if should_fallback and fallback_func is not None:
            handler.logger.info(f"Falling back for {operation_name}")
            return fallback_func(*args, **kwargs)
        else:
            raise e


# Export the main components
__all__ = [
    'CUDAErrorHandler',
    'MemoryPoolManager', 
    'cuda_error_handler',
    'memory_pool_manager',
    'cuda_error_handler_decorator',
    'safe_cuda_execution'
]


def apply_hardware_specific_optimizations(model, config):
    """
    Apply hardware-specific optimizations to the model based on the configuration.

    Args:
        model: The PyTorch model to optimize
        config: Configuration object containing optimization settings

    Returns:
        Optimized model
    """
    import torch.nn as nn

    # Apply optimizations based on hardware configuration
    if hasattr(config, 'hardware_compute_capability'):
        compute_capability = config.hardware_compute_capability
        if compute_capability[0] >= 6:  # SM61 and above
            # For SM61 architecture, set appropriate memory fraction
            if torch.cuda.is_available():
                try:
                    # Limit memory usage to prevent out of memory errors
                    torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of available memory
                except Exception as e:
                    pass  # If setting memory fraction fails, continue

    # Apply mixed precision if supported
    if hasattr(config, 'use_mixed_precision') and config.use_mixed_precision:
        # Apply any mixed precision optimizations here
        pass

    # Apply memory-efficient techniques if specified in config
    if hasattr(config, 'use_gradient_checkpointing') and config.use_gradient_checkpointing:
        # Enable gradient checkpointing for memory efficiency
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                try:
                    module.gradient_checkpointing_enable()
                except:
                    pass

    return model


def get_hardware_optimization_recommendations(config):
    """
    Get hardware-specific optimization recommendations based on the configuration.

    Args:
        config: Configuration object containing hardware settings

    Returns:
        Dictionary with optimization recommendations
    """
    recommendations = {
        'memory_optimization': True,
        'gradient_checkpointing': True,
        'mixed_precision': True,
        'use_memory_efficient_attention': True,
        'use_memory_pooling': True
    }

    if hasattr(config, 'hardware_compute_capability'):
        compute_capability = config.hardware_compute_capability
        if compute_capability[0] >= 6:  # SM61 and above
            recommendations.update({
                'use_flash_attention': True,
                'memory_fraction_limit': 0.9,
                'enable_tensor_cores': True
            })
        elif compute_capability[0] < 6:
            recommendations.update({
                'avoid_tensor_cores': True,
                'conservative_memory_fraction': 0.8
            })

    if hasattr(config, 'memory_size_gb'):
        memory_gb = config.memory_size_gb
        if memory_gb < 8:
            recommendations.update({
                'aggressive_memory_optimization': True,
                'enable_gradient_checkpointing': True,
                'use_sparsity': True
            })
        elif memory_gb >= 16:
            recommendations.update({
                'performance_over_memory': True,
                'disable_unnecessary_optimizations': True
            })

    return recommendations