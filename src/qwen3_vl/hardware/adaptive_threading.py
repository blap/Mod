"""
Adaptive Threading Model for Qwen3-VL
Implements threading models that adapt to different core/thread counts
"""
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import time
import logging
from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass
import os

from src.qwen3_vl.hardware.cpu_detector import CPUDetector, CPUModel
from src.qwen3_vl.hardware.cpu_profiles import get_optimization_profile, AdaptiveOptimizationManager


logger = logging.getLogger(__name__)


@dataclass
class ThreadingConfig:
    """Configuration for adaptive threading model"""
    num_workers: int
    max_concurrent_threads: int
    thread_affinity_enabled: bool
    use_hyperthreading: bool
    io_bound_optimization: bool
    cpu_bound_optimization: bool
    work_stealing_enabled: bool
    thread_priority: int  # 0 (normal), 1 (above normal), -1 (below normal)


class AdaptiveThreadingModel:
    """
    Adaptive threading model that optimizes thread usage based on detected CPU characteristics
    """
    
    def __init__(self, optimization_manager: AdaptiveOptimizationManager):
        self.optimization_manager = optimization_manager
        self.config = self._create_threading_config()
        
        # Initialize thread pools based on CPU characteristics
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._io_thread_pool: Optional[ThreadPoolExecutor] = None
        
        self._initialize_thread_pools()
        
        logger.info(f"Adaptive threading model initialized for {optimization_manager.config.cpu_model.value} "
                   f"with {self.config.num_workers} workers and {self.config.max_concurrent_threads} max threads")
    
    def _create_threading_config(self) -> ThreadingConfig:
        """Create threading configuration based on CPU-specific optimizations"""
        profile = self.optimization_manager.config
        
        # Determine if this is an I/O bound or CPU-bound task based on CPU characteristics
        is_io_bound = profile.num_cores <= 4  # Smaller CPUs may benefit more from I/O optimization
        is_cpu_bound = not is_io_bound
        
        return ThreadingConfig(
            num_workers=profile.num_workers,
            max_concurrent_threads=profile.max_concurrent_threads,
            thread_affinity_enabled=profile.thread_affinity_enabled,
            use_hyperthreading=profile.use_hyperthreading,
            io_bound_optimization=is_io_bound,
            cpu_bound_optimization=is_cpu_bound,
            work_stealing_enabled=getattr(profile, 'work_stealing_enabled', True),
            thread_priority=0  # Normal priority
        )
    
    def _initialize_thread_pools(self):
        """Initialize thread pools based on CPU characteristics"""
        # Main thread pool for CPU-bound tasks
        max_workers = min(self.config.max_concurrent_threads, os.cpu_count() or 1)
        self._thread_pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="qwen3vl_cpu"
        )
        
        # Process pool for heavy computational tasks
        if self.config.use_hyperthreading:
            # Use number of physical cores for process pool to avoid oversubscription
            num_processes = max(1, psutil.cpu_count(logical=False))
        else:
            num_processes = max(1, min(self.config.num_workers // 2, psutil.cpu_count(logical=False)))
        
        self._process_pool = ProcessPoolExecutor(
            max_workers=num_processes,
            mp_context=multiprocessing.get_context('spawn')  # Use spawn to avoid issues with CUDA
        )
        
        # I/O thread pool for I/O-bound tasks
        io_workers = min(4, self.config.num_workers)  # Limit I/O workers
        self._io_thread_pool = ThreadPoolExecutor(
            max_workers=io_workers,
            thread_name_prefix="qwen3vl_io"
        )
        
        logger.info(f"Thread pools initialized: {max_workers} CPU threads, "
                   f"{num_processes} processes, {io_workers} I/O threads")
    
    def submit_cpu_task(self, func: Callable, *args, **kwargs):
        """
        Submit a CPU-bound task to the appropriate thread pool
        
        Args:
            func: Function to execute
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function
            
        Returns:
            Future object for the submitted task
        """
        if self._thread_pool is None:
            raise RuntimeError("Threading model not properly initialized")
        
        return self._thread_pool.submit(func, *args, **kwargs)
    
    def submit_io_task(self, func: Callable, *args, **kwargs):
        """
        Submit an I/O-bound task to the I/O thread pool
        
        Args:
            func: Function to execute
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function
            
        Returns:
            Future object for the submitted task
        """
        if self._io_thread_pool is None:
            raise RuntimeError("I/O thread pool not properly initialized")
        
        return self._io_thread_pool.submit(func, *args, **kwargs)
    
    def submit_heavy_task(self, func: Callable, *args, **kwargs):
        """
        Submit a heavy computational task to the process pool
        
        Args:
            func: Function to execute
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function
            
        Returns:
            Future object for the submitted task
        """
        if self._process_pool is None:
            raise RuntimeError("Process pool not properly initialized")
        
        return self._process_pool.submit(func, *args, **kwargs)
    
    def execute_with_optimal_threading(self, task_type: str, func: Callable, *args, **kwargs):
        """
        Execute a function using the most appropriate threading model based on task type
        
        Args:
            task_type: Type of task ('cpu', 'io', 'heavy')
            func: Function to execute
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function
            
        Returns:
            Result of the function execution
        """
        if task_type == 'cpu':
            future = self.submit_cpu_task(func, *args, **kwargs)
        elif task_type == 'io':
            future = self.submit_io_task(func, *args, **kwargs)
        elif task_type == 'heavy':
            future = self.submit_heavy_task(func, *args, **kwargs)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return future.result()
    
    def get_thread_pool_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the thread pools
        
        Returns:
            Dictionary containing thread pool statistics
        """
        # Note: ThreadPoolExecutor doesn't expose many statistics directly
        # This is a simplified version - in practice you might want to implement custom tracking
        return {
            'cpu_thread_pool_size': self.config.max_concurrent_threads,
            'io_thread_pool_size': 4 if self._io_thread_pool else 0,
            'process_pool_size': psutil.cpu_count(logical=False) if self._process_pool else 0,
            'cpu_model': self.optimization_manager.config.cpu_model.value
        }
    
    def set_thread_affinity(self):
        """
        Set thread affinity based on CPU model for better performance
        Note: This is a simplified implementation - full implementation would require platform-specific code
        """
        if not self.config.thread_affinity_enabled:
            return
        
        # This is a simplified version - a full implementation would set actual CPU affinity
        # which requires platform-specific code and elevated privileges in some cases
        logger.info(f"Thread affinity optimization enabled for {self.optimization_manager.config.cpu_model.value}")
        
        # On Linux, you could use: os.sched_setaffinity(0, cpu_list)
        # On Windows, you could use: psutil.Process().cpu_affinity(cpu_list)
        
        # For now, we'll just log that this optimization is enabled
        pass
    
    def get_optimal_batch_size(self, base_batch_size: int = 1) -> int:
        """
        Get the optimal batch size based on CPU characteristics
        
        Args:
            base_batch_size: Base batch size to multiply
            
        Returns:
            Optimal batch size for the current CPU
        """
        multiplier = self.optimization_manager.get_batch_size_multiplier()
        optimal_size = int(base_batch_size * multiplier)
        
        # Ensure batch size is at least 1 and reasonable
        return max(1, min(optimal_size, 32))  # Cap at 32 to avoid excessive memory usage
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown all thread pools gracefully
        
        Args:
            wait: Whether to wait for tasks to complete before shutting down
        """
        if self._thread_pool:
            self._thread_pool.shutdown(wait=wait)
        if self._io_thread_pool:
            self._io_thread_pool.shutdown(wait=wait)
        if self._process_pool:
            self._process_pool.shutdown(wait=wait)
        
        logger.info("Adaptive threading model shut down")


class ThreadingModelFactory:
    """Factory for creating adaptive threading models based on detected CPU"""
    
    @staticmethod
    def create_for_detected_cpu() -> AdaptiveThreadingModel:
        """
        Create an adaptive threading model for the currently detected CPU
        
        Returns:
            AdaptiveThreadingModel configured for the detected CPU
        """
        # Use the CPU detector to identify the CPU
        detector = CPUDetector()
        features = detector.get_cpu_features()
        
        # Get the appropriate optimization profile
        optimization_manager = get_optimization_profile(features.model)
        
        # Create and return the adaptive threading model
        return AdaptiveThreadingModel(optimization_manager)
    
    @staticmethod
    def create_for_cpu_model(cpu_model: CPUModel) -> AdaptiveThreadingModel:
        """
        Create an adaptive threading model for a specific CPU model
        
        Args:
            cpu_model: The CPU model to create threading model for
            
        Returns:
            AdaptiveThreadingModel configured for the specified CPU model
        """
        optimization_manager = get_optimization_profile(cpu_model)
        return AdaptiveThreadingModel(optimization_manager)


class ThreadingOptimizer:
    """High-level interface for threading optimization"""
    
    def __init__(self):
        self.threading_model = ThreadingModelFactory.create_for_detected_cpu()
        self._active_tasks = []
    
    def preprocess_with_optimal_threading(self, preprocess_func: Callable, 
                                        data: List[Any], 
                                        batch_size: int = 1) -> List[Any]:
        """
        Preprocess data using optimal threading based on CPU characteristics
        
        Args:
            preprocess_func: Function to preprocess individual items
            data: List of data items to preprocess
            batch_size: Base batch size (will be adjusted based on CPU)
            
        Returns:
            List of preprocessed results
        """
        # Adjust batch size based on CPU characteristics
        optimal_batch_size = self.threading_model.get_optimal_batch_size(batch_size)
        
        # Split data into batches
        batches = [data[i:i + optimal_batch_size] for i in range(0, len(data), optimal_batch_size)]
        
        results = []
        for batch in batches:
            # Submit batch processing as a single task
            batch_result = self.threading_model.execute_with_optimal_threading(
                'cpu', 
                self._process_batch, 
                preprocess_func, 
                batch
            )
            results.extend(batch_result)
        
        return results
    
    def _process_batch(self, preprocess_func: Callable, batch: List[Any]) -> List[Any]:
        """Process a batch of items"""
        return [preprocess_func(item) for item in batch]
    
    def run_inference_with_optimal_threading(self, model_func: Callable, 
                                           inputs: List[Any]) -> List[Any]:
        """
        Run inference with optimal threading based on CPU characteristics
        
        Args:
            model_func: Function to run inference
            inputs: List of inputs to process
            
        Returns:
            List of inference results
        """
        # For inference, we might want to use different threading strategies
        # based on the input size and model complexity
        
        if len(inputs) <= 4:
            # For small inputs, process sequentially to avoid overhead
            return [model_func(inp) for inp in inputs]
        else:
            # For larger inputs, use optimal threading
            optimal_batch_size = self.threading_model.get_optimal_batch_size(2)
            batches = [inputs[i:i + optimal_batch_size] for i in range(0, len(inputs), optimal_batch_size)]
            
            results = []
            for batch in batches:
                batch_result = self.threading_model.execute_with_optimal_threading(
                    'cpu', 
                    self._run_inference_batch, 
                    model_func, 
                    batch
                )
                results.extend(batch_result)
            
            return results
    
    def _run_inference_batch(self, model_func: Callable, batch: List[Any]) -> List[Any]:
        """Run inference on a batch of inputs"""
        return [model_func(inp) for inp in batch]
    
    def optimize_data_loading(self, load_func: Callable, 
                            paths: List[str]) -> List[Any]:
        """
        Optimize data loading with appropriate I/O threading
        
        Args:
            load_func: Function to load individual items
            paths: List of paths to load
            
        Returns:
            List of loaded items
        """
        # Use I/O optimized threading for data loading
        results = []
        for path in paths:
            result = self.threading_model.execute_with_optimal_threading(
                'io',
                load_func,
                path
            )
            results.append(result)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the threading model
        
        Returns:
            Dictionary containing performance statistics
        """
        return {
            **self.threading_model.get_thread_pool_stats(),
            'optimal_batch_size_factor': self.threading_model.optimization_manager.get_batch_size_multiplier(),
            'recommended_workers': self.threading_model.config.num_workers
        }
    
    def shutdown(self):
        """Shutdown the threading optimizer"""
        self.threading_model.shutdown()


def get_threading_optimizer() -> ThreadingOptimizer:
    """
    Get a threading optimizer configured for the detected CPU
    
    Returns:
        ThreadingOptimizer instance
    """
    return ThreadingOptimizer()


if __name__ == "__main__":
    print("Adaptive Threading Model for Qwen3-VL")
    print("=" * 45)
    
    # Test threading model creation for different CPUs
    i5_model = ThreadingModelFactory.create_for_cpu_model(CPUModel.INTEL_I5_10210U)
    i7_model = ThreadingModelFactory.create_for_cpu_model(CPUModel.INTEL_I7_8700)
    
    print(f"Intel i5-10210U Threading Model:")
    stats_i5 = i5_model.get_thread_pool_stats()
    print(f"  CPU Thread Pool Size: {stats_i5['cpu_thread_pool_size']}")
    print(f"  Optimal Batch Size (base=1): {i5_model.get_optimal_batch_size(1)}")
    print(f"  Optimal Batch Size (base=4): {i5_model.get_optimal_batch_size(4)}")
    print()
    
    print(f"Intel i7-8700 Threading Model:")
    stats_i7 = i7_model.get_thread_pool_stats()
    print(f"  CPU Thread Pool Size: {stats_i7['cpu_thread_pool_size']}")
    print(f"  Optimal Batch Size (base=1): {i7_model.get_optimal_batch_size(1)}")
    print(f"  Optimal Batch Size (base=4): {i7_model.get_optimal_batch_size(4)}")
    
    # Test the high-level optimizer
    print(f"\nTesting High-Level Threading Optimizer:")
    optimizer = get_threading_optimizer()
    stats = optimizer.get_performance_stats()
    print(f"  Detected CPU: {stats['cpu_model']}")
    print(f"  CPU Threads: {stats['cpu_thread_pool_size']}")
    print(f"  Recommended Workers: {stats['recommended_workers']}")
    print(f"  Batch Size Factor: {stats['optimal_batch_size_factor']}")
    
    # Shutdown
    optimizer.shutdown()
    
    print("\nAdaptive threading model implementation completed!")