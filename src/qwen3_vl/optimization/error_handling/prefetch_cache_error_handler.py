"""
Comprehensive Error Handling for Prefetching and Caching Operations in Qwen3-VL

This module implements robust error handling for prefetching and caching systems,
including exception handling, error recovery mechanisms, fallback strategies,
logging, monitoring, and performance impact minimization.
"""

import torch
import numpy as np
from typing import Any, Dict, Optional, Callable, Tuple, Union
import logging
import time
import traceback
import threading
from enum import Enum
from dataclasses import dataclass
import warnings
from functools import wraps
import sys
import gc
import os
from pathlib import Path


class ErrorSeverity(Enum):
    """Enumeration for error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorHandlerType(Enum):
    """Types of error handlers"""
    PREFETCH_ERROR_HANDLER = "prefetch_error_handler"
    CACHE_ERROR_HANDLER = "cache_error_handler"
    MEMORY_ERROR_HANDLER = "memory_error_handler"
    SYSTEM_ERROR_HANDLER = "system_error_handler"


@dataclass
class ErrorInfo:
    """Information about an error that occurred"""
    error_type: str
    error_message: str
    severity: ErrorSeverity
    timestamp: float
    operation: str
    details: Dict[str, Any]


class PrefetchCacheErrorHandler:
    """
    Comprehensive error handler for prefetching and caching operations.
    Provides centralized error handling, recovery, and fallback mechanisms.
    """

    def __init__(self, log_errors: bool = True, enable_recovery: bool = True):
        self.log_errors = log_errors
        self.enable_recovery = enable_recovery
        self.errors = []
        self.recovery_attempts = 0
        self.fallback_strategies = {}
        self.lock = threading.Lock()

        # Set up logging
        self.logger = logging.getLogger("PrefetchCacheErrorHandler")
        if log_errors and not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)


    def log_error(self, error_info: ErrorInfo):
        """Log an error with appropriate level based on severity."""
        if not self.log_errors:
            return

        message = f"Operation: {error_info.operation}, Error: {error_info.error_message}"
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(message)
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(message)
        else:
            self.logger.info(message)


    def handle_error(self,
                     error: Exception,
                     operation: str,
                     fallback_func: Optional[Callable] = None,
                     *args,
                     **kwargs) -> Tuple[bool, Any]:
        """
        Handle an error during prefetching/caching operations.

        Args:
            error: The exception that occurred
            operation: Name of the operation that failed
            fallback_func: Optional fallback function to execute
            *args, **kwargs: Arguments for fallback function

        Returns:
            Tuple of (success flag, result from fallback or None)
        """
        error_info = ErrorInfo(
            error_type=type(error).__name__,
            error_message=str(error),
            severity=self._determine_severity(error),
            timestamp=time.time(),
            operation=operation,
            details={
                'traceback': traceback.format_exc(),
                'args': str(args)[:200],  # Limit length for safety
                'kwargs_keys': list(kwargs.keys())
            }
        )

        with self.lock:
            self.errors.append(error_info)

        self.log_error(error_info)

        # Attempt recovery if enabled and a fallback is provided
        if self.enable_recovery and fallback_func:
            try:
                result = fallback_func(*args, **kwargs)
                self.logger.info(f"Fallback succeeded for operation: {operation}")
                return True, result
            except Exception as fallback_error:
                self.logger.error(f"Fallback also failed for operation {operation}: {fallback_error}")
                # Add fallback error to error log
                fallback_error_info = ErrorInfo(
                    error_type=type(fallback_error).__name__,
                    error_message=f"Fallback failed: {str(fallback_error)}",
                    severity=ErrorSeverity.HIGH,
                    timestamp=time.time(),
                    operation=f"{operation}_fallback",
                    details={'original_error': str(error)}
                )

                with self.lock:
                    self.errors.append(fallback_error_info)

                return False, None

        return False, None

    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type."""
        error_type = type(error).__name__

        critical_errors = [
            'OutOfMemoryError', 'MemoryError', 'CUDAOutOfMemoryError',
            'KeyboardInterrupt', 'SystemExit'
        ]

        high_errors = [
            'RuntimeError', 'ValueError', 'TypeError',
            'AttributeError', 'ImportError'
        ]

        medium_errors = [
            'KeyError', 'IndexError', 'AssertionError',
            'NotImplementedError'
        ]

        if error_type in critical_errors:
            return ErrorSeverity.CRITICAL
        elif error_type in high_errors:
            return ErrorSeverity.HIGH
        elif error_type in medium_errors:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about handled errors."""
        with self.lock:
            if not self.errors:
                return {
                    'total_errors': 0,
                    'severity_breakdown': {},
                    'recent_errors': []
                }

            total_errors = len(self.errors)
            severity_counts = {}
            for error in self.errors:
                severity = error.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            recent_errors = [err for err in self.errors[-10:]]

            return {
                'total_errors': total_errors,
                'severity_breakdown': severity_counts,
                'recovery_attempts': self.recovery_attempts,
                'recent_errors': [
                    {
                        'operation': err.operation,
                        'error_type': err.error_type,
                        'severity': err.severity.value,
                        'timestamp': err.timestamp
                    } for err in recent_errors
                ]
            }

    def clear_error_log(self):
        """Clear the error log."""
        with self.lock:
            self.errors.clear()


class PrefetchingErrorDecorator:
    """
    Decorator for adding error handling to prefetching operations.
    """

    def __init__(self,
                 error_handler: PrefetchCacheErrorHandler,
                 fallback_func: Optional[Callable] = None,
                 default_return_value: Any = None):
        self.error_handler = error_handler
        self.fallback_func = fallback_func
        self.default_return_value = default_return_value

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                operation_name = f"{func.__module__}.{func.__name__}"
                success, result = self.error_handler.handle_error(
                    e, operation_name, self.fallback_func, *args, **kwargs
                )
                if success:
                    return result
                else:
                    # Log the error and return default value
                    self.error_handler.logger.warning(
                        f"Operation {operation_name} failed, returning default value: {type(e).__name__}: {str(e)}"
                    )
                    return self.default_return_value
        return wrapper


class CachingErrorDecorator:
    """
    Decorator for adding error handling to caching operations.
    """

    def __init__(self,
                 error_handler: PrefetchCacheErrorHandler,
                 fallback_func: Optional[Callable] = None,
                 default_return_value: Any = None):
        self.error_handler = error_handler
        self.fallback_func = fallback_func
        self.default_return_value = default_return_value

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                operation_name = f"{func.__module__}.{func.__name__}"
                success, result = self.error_handler.handle_error(
                    e, operation_name, self.fallback_func, *args, **kwargs
                )
                if success:
                    return result
                else:
                    # Log the error and return default value
                    self.error_handler.logger.warning(
                        f"Caching operation {operation_name} failed, returning default value: {type(e).__name__}: {str(e)}"
                    )
                    return self.default_return_value
        return wrapper


class FallbackStrategies:
    """
    Collection of fallback strategies for prefetching and caching operations.
    """

    @staticmethod
    def fallback_no_prefetch(*args, **kwargs):
        """Fallback strategy: return tensors without prefetching."""
        if len(args) >= 1:
            return args[0]  # Return the original tensor/data
        return None

    @staticmethod
    def fallback_standard_cache(*args, **kwargs):
        """Fallback strategy: use standard caching without optimization."""
        if len(args) >= 2:
            # Return the data and indicate it's not cached
            return args[0], False
        return None, False

    @staticmethod
    def fallback_cpu_cache(*args, **kwargs):
        """Fallback strategy: move to CPU if GPU fails."""
        if len(args) >= 1 and isinstance(args[0], torch.Tensor):
            try:
                return args[0].cpu()
            except Exception:
                return args[0]  # Return original if even CPU move fails
        return None

    @staticmethod
    def fallback_empty_cache_and_retry(operation_func, *args, **kwargs):
        """Fallback strategy: clear cache and retry operation."""
        try:
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Clear Python garbage collector
            gc.collect()

            # Retry the original operation
            return operation_func(*args, **kwargs)
        except Exception:
            # If retry also fails, return None
            return None

    @staticmethod
    def fallback_reduce_tensor_size(tensor: torch.Tensor, target_size: int = None) -> torch.Tensor:
        """Fallback strategy: reduce tensor size if memory constrained."""
        if not isinstance(tensor, torch.Tensor):
            return tensor

        if target_size is None:
            # Reduce by half
            new_shape = list(tensor.shape)
            if len(new_shape) > 0 and new_shape[0] > 1:
                new_shape[0] = max(1, new_shape[0] // 2)
            else:
                # Try to reduce the last dimension if possible
                if len(new_shape) > 0 and new_shape[-1] > 1:
                    new_shape[-1] = max(1, new_shape[-1] // 2)
            return tensor[tuple(slice(None, size) for size in new_shape)]

        # Reduce to target size
        original_size = tensor.numel()
        if original_size <= target_size:
            return tensor

        # Calculate reduction factor
        reduction_factor = target_size / original_size
        new_size = int(original_size * reduction_factor)

        # Flatten tensor temporarily to get first 'new_size' elements
        flat_tensor = tensor.flatten()
        reduced_flat = flat_tensor[:new_size]

        # Reshape to approximate original shape
        return reduced_flat.reshape((-1,) + tensor.shape[1:])


def create_error_handler(log_errors: bool = True, enable_recovery: bool = True) -> PrefetchCacheErrorHandler:
    """
    Factory function to create a prefetching and caching error handler.

    Args:
        log_errors: Whether to log errors
        enable_recovery: Whether to enable recovery mechanisms

    Returns:
        PrefetchCacheErrorHandler instance
    """
    return PrefetchCacheErrorHandler(log_errors=log_errors, enable_recovery=enable_recovery)


def safe_prefetch_operation(error_handler: PrefetchCacheErrorHandler,
                          operation_func: Callable,
                          *args,
                          **kwargs) -> Tuple[bool, Any]:
    """
    Safely execute a prefetch operation with error handling.

    Args:
        error_handler: The error handler instance
        operation_func: The prefetch operation to execute
        *args, **kwargs: Arguments for the operation

    Returns:
        Tuple of (success flag, result from operation or fallback)
    """
    try:
        result = operation_func(*args, **kwargs)
        return True, result
    except Exception as e:
        operation_name = f"{operation_func.__module__}.{operation_func.__name__}"
        success, result = error_handler.handle_error(
            e, operation_name, FallbackStrategies.fallback_no_prefetch,
            *args, **kwargs
        )
        return success, result


def safe_cache_operation(error_handler: PrefetchCacheErrorHandler,
                        operation_func: Callable,
                        *args,
                        **kwargs) -> Tuple[bool, Any]:
    """
    Safely execute a cache operation with error handling.

    Args:
        error_handler: The error handler instance
        operation_func: The cache operation to execute
        *args, **kwargs: Arguments for the operation

    Returns:
        Tuple of (success flag, result from operation or fallback)
    """
    try:
        result = operation_func(*args, **kwargs)
        return True, result
    except Exception as e:
        operation_name = f"{operation_func.__module__}.{operation_func.__name__}"
        success, result = error_handler.handle_error(
            e, operation_name, FallbackStrategies.fallback_standard_cache,
            *args, **kwargs
        )
        return success, result


class PrefetchMonitor:
    """
    Monitor for prefetching operations to track performance and errors.
    """

    def __init__(self, error_handler: PrefetchCacheErrorHandler):
        self.error_handler = error_handler
        self.prefetch_stats = {
            'total_prefetches': 0,
            'successful_prefetches': 0,
            'failed_prefetches': 0,
            'avg_prefetch_time': 0.0,
            'prefetch_times': []
        }
        self.lock = threading.Lock()

    def record_prefetch_attempt(self, success: bool, time_taken: float = 0.0):
        """Record a prefetch attempt."""
        with self.lock:
            self.prefetch_stats['total_prefetches'] += 1
            if success:
                self.prefetch_stats['successful_prefetches'] += 1
            else:
                self.prefetch_stats['failed_prefetches'] += 1

            if time_taken > 0:
                self.prefetch_stats['prefetch_times'].append(time_taken)
                # Calculate moving average
                self.prefetch_stats['avg_prefetch_time'] = sum(self.prefetch_stats['prefetch_times']) / len(self.prefetch_stats['prefetch_times'])

    def get_prefetch_performance(self) -> Dict[str, Any]:
        """Get prefetch performance metrics."""
        with self.lock:
            total = self.prefetch_stats['total_prefetches']
            success_rate = self.prefetch_stats['successful_prefetches'] / total if total > 0 else 0
            return {
                'success_rate': success_rate,
                'failure_rate': 1 - success_rate,
                'total_attempts': total,
                'successful_attempts': self.prefetch_stats['successful_prefetches'],
                'failed_attempts': self.prefetch_stats['failed_prefetches'],
                'average_time': self.prefetch_stats['avg_prefetch_time'],
                'total_time': sum(self.prefetch_stats['prefetch_times'])
            }


class CacheMonitor:
    """
    Monitor for cache operations to track performance and errors.
    """

    def __init__(self, error_handler: PrefetchCacheErrorHandler):
        self.error_handler = error_handler
        self.cache_stats = {
            'total_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_evictions': 0,
            'avg_access_time': 0.0,
            'access_times': [],
            'current_cache_size': 0
        }
        self.lock = threading.Lock()

    def record_cache_operation(self, operation_type: str, time_taken: float = 0.0, size_change: int = 0):
        """Record a cache operation."""
        with self.lock:
            self.cache_stats['total_operations'] += 1

            if operation_type == 'hit':
                self.cache_stats['cache_hits'] += 1
            elif operation_type == 'miss':
                self.cache_stats['cache_misses'] += 1
            elif operation_type == 'eviction':
                self.cache_stats['cache_evictions'] += 1

            if time_taken > 0:
                self.cache_stats['access_times'].append(time_taken)
                # Calculate moving average
                self.cache_stats['avg_access_time'] = sum(self.cache_stats['access_times']) / len(self.cache_stats['access_times'])

            self.cache_stats['current_cache_size'] += size_change

    def get_cache_performance(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        with self.lock:
            total_ops = self.cache_stats['total_operations']
            hit_rate = self.cache_stats['cache_hits'] / total_ops if total_ops > 0 else 0
            miss_rate = self.cache_stats['cache_misses'] / total_ops if total_ops > 0 else 0

            return {
                'hit_rate': hit_rate,
                'miss_rate': miss_rate,
                'eviction_count': self.cache_stats['cache_evictions'],
                'total_operations': total_ops,
                'current_size': self.cache_stats['current_cache_size'],
                'average_access_time': self.cache_stats['avg_access_time'],
                'total_access_time': sum(self.cache_stats['access_times'])
            }


# Example usage and testing
def test_error_handling():
    """Test the error handling system."""
    print("Testing Prefetch and Cache Error Handling System")
    print("=" * 60)

    # Create error handler
    error_handler = create_error_handler(log_errors=True, enable_recovery=True)

    # Create monitors
    prefetch_monitor = PrefetchMonitor(error_handler)
    cache_monitor = CacheMonitor(error_handler)

    print("\n1. Testing basic error handling...")

    # Test successful operation
    def successful_operation():
        return "success_result"

    success, result = safe_prefetch_operation(error_handler, successful_operation)
    print(f"   Successful operation: {success}, result: {result}")
    prefetch_monitor.record_prefetch_attempt(success, 0.001)

    # Test operation that raises an exception
    def failing_operation():
        raise ValueError("This operation intentionally fails")

    success, result = safe_prefetch_operation(error_handler, failing_operation)
    print(f"   Failing operation: {success}, result: {result}")
    prefetch_monitor.record_prefetch_attempt(success, 0.002)

    # Test operation with fallback
    def operation_with_fallback():
        raise RuntimeError("Memory allocation failed")

    success, result = safe_cache_operation(
        error_handler,
        operation_with_fallback
    )
    print(f"   Operation with fallback: {success}, result: {result}")
    cache_monitor.record_cache_operation('miss', 0.003, 0)

    print("\n2. Testing fallback strategies...")

    # Test fallback strategies directly
    tensor = torch.randn(10, 10)
    reduced_tensor = FallbackStrategies.fallback_reduce_tensor_size(tensor, target_size=50)
    print(f"   Original tensor shape: {tensor.shape}")
    print(f"   Reduced tensor shape: {reduced_tensor.shape}")

    if torch.cuda.is_available():
        cpu_tensor = FallbackStrategies.fallback_cpu_cache(tensor.cuda())
        print(f"   CPU fallback tensor device: {cpu_tensor.device}")

    print("\n3. Getting performance metrics...")

    prefetch_perf = prefetch_monitor.get_prefetch_performance()
    cache_perf = cache_monitor.get_cache_performance()

    print(f"   Prefetch Performance: {prefetch_perf}")
    print(f"   Cache Performance: {cache_perf}")

    print("\n4. Getting error statistics...")
    error_stats = error_handler.get_error_statistics()
    print(f"   Error Statistics: {error_stats}")

    print("\n5. Testing decorators...")

    # Test prefetch decorator
    @PrefetchingErrorDecorator(error_handler)
    def decorated_prefetch_op():
        return "prefetch_result"

    result = decorated_prefetch_op()
    print(f"   Decorated prefetch result: {result}")

    # Test caching decorator
    @CachingErrorDecorator(error_handler)
    def decorated_cache_op():
        raise MemoryError("Simulated memory error")
        return "cache_result"

    result = decorated_cache_op()
    print(f"   Decorated cache result (after error): {result}")

    print("\nError handling system test completed successfully!")


if __name__ == "__main__":
    test_error_handling()