"""
Implementation of Comprehensive Inference Pipeline Optimizations for Qwen3-VL-2B-Instruct
Optimized for Intel i5-10210U + NVIDIA SM61 (Compute Capability 6.1) Hardware
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import logging
import numpy as np
import math
from collections import defaultdict, deque
import gc
import psutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InferencePipelineConfig:
    """Configuration for the optimized inference pipeline."""
    # Hardware-specific parameters
    target_hardware: str = "nvidia_sm61_intel_i5_10210u"
    max_batch_size: int = 8  # Reduced for constrained hardware
    variable_batch_size: bool = True  # Enable variable batch sizes for different input types
    
    # Caching parameters
    enable_tensor_caching: bool = True
    enable_model_caching: bool = True
    cache_size: int = 512  # Max number of cached tensors
    
    # I/O optimization parameters
    enable_async_io: bool = True
    enable_pinned_memory: bool = True
    enable_async_transfers: bool = True
    num_workers: int = 2  # Limited for i5-10210U
    
    # Pipeline parameters
    pipeline_depth: int = 3  # Number of pipeline stages
    pipeline_buffer_size: int = 4  # Buffer size between pipeline stages
    
    # Memory optimization parameters
    memory_efficient: bool = True
    memory_threshold: float = 0.8  # Threshold for memory-efficient operations


class VariableBatchProcessor:
    """
    Variable batch processor that handles different input sizes efficiently.
    Groups inputs by size to maximize utilization while minimizing padding waste.
    """
    def __init__(self, config: InferencePipelineConfig):
        self.config = config
        self.max_batch_size = config.max_batch_size
        
        # Size-based batching groups
        self.size_groups = defaultdict(list)
        self.group_thresholds = [64, 128, 256, 512, 1024]  # Different sequence length thresholds
        
        # Track batch statistics
        self.batch_stats = {
            'total_batches': 0,
            'avg_batch_size': 0,
            'padding_saved': 0
        }

    def group_inputs_by_size(self, inputs: List[Tuple[torch.Tensor, str]]) -> List[Dict[str, Any]]:
        """
        Group inputs by size to optimize batch processing.

        Args:
            inputs: List of (input_tensor, input_type) tuples

        Returns:
            List of batch dictionaries
        """
        # Clear previous groups
        self.size_groups.clear()
        
        # Group inputs by sequence length
        for tensor, input_type in inputs:
            seq_len = tensor.shape[1] if len(tensor.shape) > 1 else tensor.shape[0]
            
            # Find appropriate size group
            group_key = self._get_size_group(seq_len)
            self.size_groups[group_key].append((tensor, input_type))

        # Create optimized batches for each group
        batches = []
        for group_key, group_inputs in self.size_groups.items():
            # Process group inputs into batches
            for i in range(0, len(group_inputs), self.max_batch_size):
                batch_group = group_inputs[i:i + self.max_batch_size]
                
                # Stack tensors in the batch
                batch_tensors = [item[0] for item in batch_group]
                batch_types = [item[1] for item in batch_group]
                
                # Pad tensors to same length if needed
                padded_batch = self._pad_batch(batch_tensors)
                
                batch_dict = {
                    'inputs': padded_batch,
                    'input_types': batch_types,
                    'batch_size': len(batch_group),
                    'size_group': group_key
                }
                batches.append(batch_dict)

        # Update statistics
        self.batch_stats['total_batches'] += len(batches)
        avg_batch_size = sum(batch['batch_size'] for batch in batches) / len(batches) if batches else 0
        self.batch_stats['avg_batch_size'] = avg_batch_size
        
        return batches

    def _get_size_group(self, seq_len: int) -> str:
        """Get size group key based on sequence length."""
        for threshold in self.group_thresholds:
            if seq_len <= threshold:
                return f"size_{threshold}"
        return f"size_large"

    def _pad_batch(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Pad tensors in batch to same length."""
        if not tensors:
            return torch.empty(0)
        
        # Find max sequence length
        max_len = max(tensor.shape[1] if len(tensor.shape) > 1 else tensor.shape[0] for tensor in tensors)
        
        # Pad each tensor to max length
        padded_tensors = []
        for tensor in tensors:
            if len(tensor.shape) > 1:
                # 2D tensor: (batch, seq_len, ...)
                if tensor.shape[1] < max_len:
                    pad_size = max_len - tensor.shape[1]
                    padded = F.pad(tensor, (0, 0, 0, pad_size), value=0)
                else:
                    padded = tensor
            else:
                # 1D tensor: (seq_len, ...)
                if tensor.shape[0] < max_len:
                    pad_size = max_len - tensor.shape[0]
                    padded = F.pad(tensor, (0, pad_size), value=0)
                else:
                    padded = tensor
            padded_tensors.append(padded)
        
        # Stack tensors
        if len(tensors[0].shape) > 1:
            return torch.stack(padded_tensors, dim=0)
        else:
            return torch.stack(padded_tensors, dim=0)

    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return self.batch_stats.copy()


class TensorCache:
    """
    Efficient tensor caching mechanism with LRU eviction policy.
    """
    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self.cache = {}
        self.access_order = deque()
        self.cache_lock = threading.Lock()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get tensor from cache."""
        with self.cache_lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                self.stats['hits'] += 1
                return self.cache[key]
            else:
                self.stats['misses'] += 1
                return None

    def put(self, key: str, tensor: torch.Tensor):
        """Put tensor in cache."""
        with self.cache_lock:
            if key in self.cache:
                # Update existing
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # Evict least recently used
                lru_key = self.access_order.popleft()
                del self.cache[lru_key]
                self.stats['evictions'] += 1
                
            self.cache[key] = tensor
            self.access_order.append(key)

    def clear(self):
        """Clear cache."""
        with self.cache_lock:
            self.cache.clear()
            self.access_order.clear()
            self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.cache_lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'cache_size': len(self.cache),
                'max_size': self.max_size
            }


class CachingMechanism:
    """
    Comprehensive caching mechanism for model components and intermediate results.
    """
    def __init__(self, config: InferencePipelineConfig):
        self.config = config
        self.tensor_cache = TensorCache(max_size=config.cache_size) if config.enable_tensor_caching else None
        self.model_cache = {}
        self.cache_lock = threading.Lock()

    def cache_tensor(self, key: str, tensor: torch.Tensor) -> bool:
        """Cache a tensor."""
        if self.tensor_cache:
            try:
                # Clone tensor to avoid reference issues
                self.tensor_cache.put(key, tensor.clone().detach())
                return True
            except Exception as e:
                logger.warning(f"Failed to cache tensor {key}: {e}")
                return False
        return False

    def get_cached_tensor(self, key: str) -> Optional[torch.Tensor]:
        """Get cached tensor."""
        if self.tensor_cache:
            return self.tensor_cache.get(key)
        return None

    def cache_model_component(self, key: str, component: nn.Module) -> bool:
        """Cache a model component."""
        if self.config.enable_model_caching:
            try:
                # Store state dict for model components
                self.model_cache[key] = component.state_dict()
                return True
            except Exception as e:
                logger.warning(f"Failed to cache model component {key}: {e}")
                return False
        return False

    def get_cached_model_component(self, key: str, component: nn.Module) -> bool:
        """Load cached model component."""
        if key in self.model_cache:
            try:
                component.load_state_dict(self.model_cache[key])
                return True
            except Exception as e:
                logger.warning(f"Failed to load cached model component {key}: {e}")
                return False
        return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {}
        if self.tensor_cache:
            stats['tensor_cache'] = self.tensor_cache.get_stats()
        stats['model_cache_size'] = len(self.model_cache)
        return stats


class OptimizedIOMechanism:
    """
    Optimized I/O operations for faster data transfer.
    """
    def __init__(self, config: InferencePipelineConfig):
        self.config = config
        self.enable_pinned_memory = config.enable_pinned_memory
        self.enable_async_transfers = config.enable_async_transfers
        
        # CUDA streams for overlapping operations
        self.compute_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.transfer_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        # Statistics
        self.stats = {
            'transfers_completed': 0,
            'async_transfers': 0,
            'pinned_memory_used': 0
        }

    def transfer_to_device(self, tensor: torch.Tensor, device: torch.device,
                          async_transfer: bool = True) -> torch.Tensor:
        """Transfer tensor to device with optimizations."""
        start_time = time.time()
        
        if self.enable_async_transfers and async_transfer and self.transfer_stream:
            with torch.cuda.stream(self.transfer_stream):
                result = tensor.to(device, non_blocking=True)
        else:
            result = tensor.to(device, non_blocking=False)
        
        transfer_time = time.time() - start_time
        self.stats['transfers_completed'] += 1
        if async_transfer and self.enable_async_transfers:
            self.stats['async_transfers'] += 1
        
        return result

    def create_optimized_dataloader(self, dataset, **kwargs) -> DataLoader:
        """Create an optimized DataLoader with pinned memory."""
        # Set default values based on config
        kwargs.setdefault('pin_memory', self.config.enable_pinned_memory)
        kwargs.setdefault('num_workers', self.config.num_workers)
        kwargs.setdefault('batch_size', self.config.max_batch_size)
        
        return DataLoader(dataset, **kwargs)

    def get_io_stats(self) -> Dict[str, Any]:
        """Get I/O statistics."""
        return self.stats.copy()


class PipelineStage:
    """
    Represents a single stage in the inference pipeline.
    """
    def __init__(self, stage_id: int, stage_func, config: InferencePipelineConfig):
        self.stage_id = stage_id
        self.stage_func = stage_func
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Performance tracking
        self.stage_times = deque(maxlen=100)

    def execute(self, data: Any) -> Any:
        """Execute the pipeline stage."""
        start_time = time.time()
        result = self.stage_func(data)
        end_time = time.time()
        
        self.stage_times.append(end_time - start_time)
        return result

    def get_avg_time(self) -> float:
        """Get average execution time for this stage."""
        return sum(self.stage_times) / len(self.stage_times) if self.stage_times else 0


class EfficientPipeline:
    """
    Efficient end-to-end inference pipeline with all optimizations.
    """
    def __init__(self, model: nn.Module, config: InferencePipelineConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize optimization components
        self.variable_batch_processor = VariableBatchProcessor(config)
        self.caching_mechanism = CachingMechanism(config)
        self.io_mechanism = OptimizedIOMechanism(config)
        
        # Pipeline stages
        self.pipeline_stages = [
            PipelineStage(0, self._preprocessing_stage, config),
            PipelineStage(1, self._memory_transfer_stage, config),
            PipelineStage(2, self._inference_stage, config),
            PipelineStage(3, self._postprocessing_stage, config)
        ]
        
        # Pipeline buffers between stages
        self.pipeline_buffers = [
            queue.Queue(maxsize=config.pipeline_buffer_size),
            queue.Queue(maxsize=config.pipeline_buffer_size),
            queue.Queue(maxsize=config.pipeline_buffer_size)
        ]
        
        # Pipeline execution tracking
        self.pipeline_stats = {
            'total_executions': 0,
            'avg_pipeline_time': 0,
            'stage_times': {i: 0 for i in range(len(self.pipeline_stages))}
        }
        
        # Memory management
        self.memory_threshold = config.memory_threshold

    def _preprocessing_stage(self, data: Any) -> Any:
        """Stage 1: Preprocessing and variable batching."""
        # Apply variable batch processing
        if self.config.variable_batch_size and isinstance(data, list):
            # Group inputs by size for optimal batching
            inputs_with_types = [(item, "unknown") for item in data]
            batches = self.variable_batch_processor.group_inputs_by_size(inputs_with_types)
            return batches
        else:
            # Standard processing
            return data

    def _memory_transfer_stage(self, data: Any) -> Any:
        """Stage 2: Memory transfer and caching."""
        if isinstance(data, list):
            # Process batched data
            processed_batches = []
            for batch in data:
                if isinstance(batch, dict) and 'inputs' in batch:
                    # Transfer to device with optimizations
                    inputs = batch['inputs']
                    if isinstance(inputs, torch.Tensor):
                        inputs = self.io_mechanism.transfer_to_device(
                            inputs, self.device,
                            async_transfer=self.config.enable_async_transfers
                        )
                        batch['inputs'] = inputs
                    processed_batches.append(batch)
            return processed_batches
        else:
            # Single tensor transfer
            if isinstance(data, torch.Tensor):
                return self.io_mechanism.transfer_to_device(
                    data, self.device,
                    async_transfer=self.config.enable_async_transfers
                )
            return data

    def _inference_stage(self, data: Any) -> Any:
        """Stage 3: Model inference."""
        with torch.no_grad():
            if isinstance(data, list):
                # Process batched data
                outputs = []
                for batch in data:
                    if isinstance(batch, dict) and 'inputs' in batch:
                        output = self.model(batch['inputs'])
                        outputs.append({
                            'output': output,
                            'metadata': batch
                        })
                return outputs
            else:
                # Single inference
                return self.model(data)

    def _postprocessing_stage(self, data: Any) -> Any:
        """Stage 4: Post-processing and output formatting."""
        if isinstance(data, list):
            # Process batched outputs
            results = []
            for item in data:
                if isinstance(item, dict) and 'output' in item:
                    # Apply post-processing
                    output = item['output']
                    if isinstance(output, torch.Tensor):
                        # Move to CPU if needed
                        if output.device != torch.device('cpu'):
                            output = output.cpu()
                    results.append(output)
                else:
                    # Direct output
                    if isinstance(item, torch.Tensor) and item.device != torch.device('cpu'):
                        item = item.cpu()
                    results.append(item)
            return results
        else:
            # Single output
            if isinstance(data, torch.Tensor) and data.device != torch.device('cpu'):
                data = data.cpu()
            return data

    def run_inference(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Run inference through the optimized pipeline."""
        start_time = time.time()
        
        # Stage 1: Preprocessing
        preprocessed_data = self.pipeline_stages[0].execute(inputs)
        
        # Stage 2: Memory transfer
        transferred_data = self.pipeline_stages[1].execute(preprocessed_data)
        
        # Stage 3: Inference
        inference_output = self.pipeline_stages[2].execute(transferred_data)
        
        # Stage 4: Post-processing
        final_output = self.pipeline_stages[3].execute(inference_output)
        
        # Update pipeline statistics
        pipeline_time = time.time() - start_time
        self.pipeline_stats['total_executions'] += 1
        self.pipeline_stats['avg_pipeline_time'] = (
            (self.pipeline_stats['avg_pipeline_time'] * (self.pipeline_stats['total_executions'] - 1) + pipeline_time) /
            self.pipeline_stats['total_executions']
        )
        
        return final_output

    def run_batch_inference(self, data_loader: DataLoader) -> List[torch.Tensor]:
        """Run batch inference using the optimized pipeline."""
        all_outputs = []
        
        for batch in data_loader:
            if isinstance(batch, torch.Tensor):
                inputs = [batch]
            elif isinstance(batch, (list, tuple)):
                inputs = list(batch)
            else:
                # Assume it's a dictionary with tensor values
                inputs = [v for v in batch.values() if isinstance(v, torch.Tensor)]
            
            outputs = self.run_inference(inputs)
            all_outputs.extend(outputs)
        
        return all_outputs

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'pipeline_stats': self.pipeline_stats.copy(),
            'batch_stats': self.variable_batch_processor.get_batch_stats(),
            'cache_stats': self.caching_mechanism.get_cache_stats(),
            'io_stats': self.io_mechanism.get_io_stats(),
            'memory_usage': psutil.virtual_memory()._asdict() if psutil else {}
        }
        
        # Add GPU memory stats if available
        if torch.cuda.is_available():
            stats['gpu_memory'] = {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated(),
                'max_reserved': torch.cuda.max_memory_reserved()
            }
        
        return stats

    def clear_caches(self):
        """Clear all caches to free memory."""
        if self.caching_mechanism.tensor_cache:
            self.caching_mechanism.tensor_cache.clear()
        self.caching_mechanism.model_cache.clear()


class OptimizedInferencePipeline:
    """
    Main optimized inference pipeline that combines all optimizations.
    """
    def __init__(self, model: nn.Module, config: Optional[InferencePipelineConfig] = None):
        self.config = config or InferencePipelineConfig()
        self.efficient_pipeline = EfficientPipeline(model, self.config)
        
        # Performance tracking
        self.performance_metrics = {
            'avg_latency': 0,
            'throughput': 0,
            'memory_efficiency': 0
        }

    def generate_response(self, input_ids: torch.Tensor,
                         pixel_values: Optional[torch.Tensor] = None,
                         attention_mask: Optional[torch.Tensor] = None,
                         **kwargs) -> torch.Tensor:
        """
        Generate response using the optimized pipeline.

        Args:
            input_ids: Input token IDs
            pixel_values: Optional pixel values for vision input
            attention_mask: Attention mask
            **kwargs: Additional generation arguments

        Returns:
            Generated output tensor
        """
        # Prepare inputs
        inputs = [input_ids]
        if pixel_values is not None:
            inputs.append(pixel_values)
        if attention_mask is not None:
            inputs.append(attention_mask)
        
        # Run inference
        results = self.efficient_pipeline.run_inference(inputs)
        
        # Return the first result (corresponding to input_ids)
        return results[0] if results else torch.empty(0)

    def run_batch_inference(self, data_loader: DataLoader) -> List[torch.Tensor]:
        """Run batch inference using the optimized pipeline."""
        return self.efficient_pipeline.run_batch_inference(data_loader)

    def benchmark_performance(self, test_inputs: List[torch.Tensor], num_runs: int = 10) -> Dict[str, Any]:
        """Benchmark the performance of the optimized pipeline."""
        times = []
        
        # Warm up
        for _ in range(3):
            _ = self.efficient_pipeline.run_inference(test_inputs)
        
        # Benchmark
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.efficient_pipeline.run_inference(test_inputs)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        
        # Calculate throughput
        total_inputs = sum(inp.shape[0] if inp.dim() > 0 else 1 for inp in test_inputs)
        throughput = total_inputs / avg_time if avg_time > 0 else 0
        
        return {
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'throughput_samples_per_sec': throughput,
            'num_runs': num_runs,
            'total_inputs_processed': total_inputs
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the pipeline."""
        return {
            **self.performance_metrics,
            **self.efficient_pipeline.get_performance_stats()
        }

    def cleanup(self):
        """Clean up resources."""
        self.efficient_pipeline.clear_caches()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def create_optimized_inference_pipeline(model: nn.Module,
                                      config: Optional[InferencePipelineConfig] = None) -> OptimizedInferencePipeline:
    """
    Factory function to create an optimized inference pipeline.

    Args:
        model: The model to optimize
        config: Configuration for the pipeline

    Returns:
        Optimized inference pipeline
    """
    return OptimizedInferencePipeline(model, config)


# Example usage and testing
def test_optimized_pipeline():
    """Test the optimized inference pipeline."""
    print("Testing Optimized Inference Pipeline...")
    
    # Create a dummy model for testing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(512, 512)
            self.norm = nn.LayerNorm(512)
        
        def forward(self, x):
            if isinstance(x, list):
                x = x[0]  # Take first tensor if list
            if len(x.shape) == 2:
                x = x.unsqueeze(0)  # Add batch dimension if needed
            x = self.linear(x)
            x = self.norm(x)
            return x

    model = DummyModel()
    
    # Create pipeline configuration
    config = InferencePipelineConfig(
        max_batch_size=4,
        variable_batch_size=True,
        enable_tensor_caching=True,
        enable_model_caching=True,
        enable_async_io=True,
        enable_pinned_memory=True,
        enable_async_transfers=True,
        memory_efficient=True
    )
    
    # Create optimized pipeline
    pipeline = create_optimized_inference_pipeline(model, config)
    
    # Create test inputs
    test_inputs = [
        torch.randn(2, 10, 512),  # Batch of 2, seq_len=10, hidden_size=512
        torch.randn(1, 20, 512),  # Different sequence length
        torch.randn(3, 5, 512)    # Another batch
    ]
    
    print("Running inference with optimized pipeline...")
    results = pipeline.efficient_pipeline.run_inference(test_inputs)
    print(f"Got {len(results)} results")
    
    for i, result in enumerate(results):
        print(f"  Result {i}: shape {result.shape}")
    
    # Run benchmark
    print("\nRunning performance benchmark...")
    benchmark_results = pipeline.benchmark_performance(test_inputs, num_runs=5)
    print(f"  Average inference time: {benchmark_results['avg_inference_time']:.4f}s")
    print(f"  Throughput: {benchmark_results['throughput_samples_per_sec']:.2f} samples/sec")
    
    # Get performance metrics
    metrics = pipeline.get_performance_metrics()
    print(f"\nPipeline statistics:")
    print(f"  Total executions: {metrics['pipeline_stats']['total_executions']}")
    print(f"  Avg pipeline time: {metrics['pipeline_stats']['avg_pipeline_time']:.4f}s")
    print(f"  Cache hit rate: {metrics['cache_stats'].get('tensor_cache', {}).get('hit_rate', 0):.2f}")
    
    # Clean up
    pipeline.cleanup()
    print("\nOptimized inference pipeline test completed successfully!")


if __name__ == "__main__":
    test_optimized_pipeline()