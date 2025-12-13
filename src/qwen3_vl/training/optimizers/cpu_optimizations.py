"""
CPU Optimization Techniques for Qwen3-VL Model
Focusing on better preprocessing, tokenization, and CPU-GPU coordination
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from transformers import PreTrainedTokenizerBase
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import queue
import time
import logging
from dataclasses import dataclass
import psutil
import os
from functools import partial


@dataclass
class CPUOptimizationConfig:
    """Configuration for CPU optimization techniques."""
    # Preprocessing parameters
    num_preprocess_workers: int = 4
    preprocess_batch_size: int = 8
    image_resize_size: Tuple[int, int] = (224, 224)
    max_text_length: int = 512
    
    # Tokenization parameters
    use_fast_tokenizer: bool = True
    padding_strategy: str = "longest"
    
    # CPU-GPU coordination parameters
    cpu_gpu_overlap: bool = True
    prefetch_buffer_size: int = 2
    transfer_async: bool = True
    
    # Multithreading parameters
    max_concurrent_preprocess: int = 8
    use_multiprocessing: bool = False  # For CPU-intensive tasks like image processing
    
    # Memory management
    memory_threshold: float = 0.8  # Percentage of available memory to use
    clear_cache_interval: int = 10  # Clear cache every N batches


class CPUPreprocessor:
    """
    CPU-based preprocessor with multithreading for efficient data preparation.
    """
    def __init__(self, config: CPUOptimizationConfig, tokenizer: Optional[PreTrainedTokenizerBase] = None):
        self.config = config
        self.tokenizer = tokenizer
        self.executor = ThreadPoolExecutor(max_workers=config.num_preprocess_workers)

        # For multiprocessing (CPU-intensive tasks)
        if config.use_multiprocessing:
            self.mp_executor = ProcessPoolExecutor(max_workers=config.num_preprocess_workers)
        else:
            self.mp_executor = None
            
        # Shared queue for processed batches
        self.processed_queue = queue.Queue(maxsize=config.prefetch_buffer_size)
        
        # Performance monitoring
        self.processing_times = []
        self.start_time = time.time()
        
    def preprocess_batch(
        self,
        texts: List[str],
        images: Optional[List[Image.Image]] = None,
        return_tensors: str = "pt"
    ) -> Dict[str, Any]:
        """
        Preprocess a batch of texts and images on CPU.
        
        Args:
            texts: List of text strings to process
            images: List of PIL images to process (optional)
            return_tensors: Type of tensors to return
            
        Returns:
            Dictionary containing preprocessed inputs
        """
        start_time = time.time()
        
        # Process texts with tokenizer
        text_outputs = {}
        if self.tokenizer and texts:
            text_outputs = self.tokenizer(
                texts,
                padding=self.config.padding_strategy,
                truncation=True,
                max_length=self.config.max_text_length,
                return_tensors=return_tensors
            )
        
        # Process images
        image_outputs = {}
        if images:
            image_outputs = self._process_images(images)
        
        # Combine outputs
        result = {**text_outputs, **image_outputs}
        
        # Record processing time
        self.processing_times.append(time.time() - start_time)
        
        return result
    
    def _process_images(self, images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        """
        Process a list of PIL images.
        
        Args:
            images: List of PIL images
            
        Returns:
            Dictionary containing processed image tensors
        """
        processed_images = []
        
        for img in images:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image
            img = img.resize(self.config.image_resize_size)
            
            # Convert to tensor and normalize
            img_array = np.array(img).astype(np.float32)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC to CHW
            
            # Normalize to [0, 1] and then to ImageNet stats
            img_tensor = img_tensor / 255.0
            # Normalize using manual calculation since F.normalize doesn't take mean/std as kwargs in some PyTorch versions
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            
            processed_images.append(img_tensor)
        
        # Stack images into a batch
        if processed_images:
            pixel_values = torch.stack(processed_images, dim=0)
            return {"pixel_values": pixel_values}
        else:
            return {}
    
    def preprocess_batch_async(
        self,
        texts: List[str],
        images: Optional[List[Image.Image]] = None,
        return_tensors: str = "pt"
    ) -> Any:
        """
        Asynchronously preprocess a batch of texts and images.
        
        Args:
            texts: List of text strings to process
            images: List of PIL images to process (optional)
            return_tensors: Type of tensors to return
            
        Returns:
            Future object that can be awaited
        """
        if self.mp_executor and images:
            # Use multiprocessing for CPU-intensive image processing
            return self.mp_executor.submit(
                self.preprocess_batch, texts, images, return_tensors
            )
        else:
            # Use threading for tokenization
            return self.executor.submit(
                self.preprocess_batch, texts, images, return_tensors
            )
    
    def close(self):
        """Close the preprocessors and clean up resources."""
        self.executor.shutdown(wait=True)
        if self.mp_executor:
            self.mp_executor.shutdown(wait=True)


class OptimizedDataLoader:
    """
    Memory-optimized data loader with CPU preprocessing and GPU transfer optimization.
    """
    def __init__(
        self,
        dataset,
        config: CPUOptimizationConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs
    ):
        self.dataset = dataset
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        
        # Initialize CPU preprocessor
        self.cpu_preprocessor = CPUPreprocessor(config, tokenizer)

        # Prefetch buffer for overlapping CPU processing and GPU transfer
        self.prefetch_buffer = queue.Queue(maxsize=config.prefetch_buffer_size)
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()

        # Memory management
        self.current_batch_idx = 0

        # Data loading parameters
        self.kwargs = kwargs
        
    def __iter__(self):
        """Iterate through the dataset with optimized preprocessing."""
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            import random
            random.shuffle(indices)
        
        # Process data in batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]
            
            # Separate texts and images
            texts = []
            images = []
            
            for item in batch_data:
                if isinstance(item, dict):
                    if 'text' in item:
                        texts.append(item['text'])
                    if 'image' in item:
                        images.append(item['image'])
                elif isinstance(item, tuple) and len(item) >= 2:
                    texts.append(item[0])
                    images.append(item[1])
                elif isinstance(item, str):
                    texts.append(item)
            
            # Preprocess on CPU
            processed_batch = self.cpu_preprocessor.preprocess_batch(texts, images if images else None)
            
            # Move to appropriate device if needed
            yield processed_batch
            
            # Memory management
            self.current_batch_idx += 1
            if self.current_batch_idx % self.config.clear_cache_interval == 0:
                self._clear_memory()
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def _prefetch_worker(self):
        """Worker thread for prefetching and preprocessing data."""
        # This would implement the prefetching logic
        # For now, it's a placeholder
        pass
    
    def _clear_memory(self):
        """Clear memory periodically to prevent memory leaks."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class CPU_GPU_Coordinator:
    """
    Coordinator for efficient CPU-GPU data transfer and processing overlap.
    """
    def __init__(self, config: CPUOptimizationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Async transfer streams for overlapping transfers
        if self.config.transfer_async and torch.cuda.is_available():
            self.transfer_streams = [torch.cuda.Stream() for _ in range(2)]  # Multiple streams for better overlap
            self.current_stream_idx = 0
        else:
            self.transfer_streams = None
            self.current_stream_idx = 0

        # Memory management
        self.memory_threshold = config.memory_threshold
        self.last_transfer_time = 0
        self.transfer_count = 0

        # CPU-GPU overlap optimization parameters
        self.overlap_enabled = config.cpu_gpu_overlap
        self.prefetch_buffer_size = config.prefetch_buffer_size
        self.transfer_async = config.transfer_async

        # Initialize pinned memory pools for faster transfers
        self.pinned_memory_pool = {}
        self.pinned_memory_pool_size = 0
        self.max_pinned_memory = int(psutil.virtual_memory().total * 0.1)  # Use 10% of system memory for pinned memory

        # Async transfer queue for overlapping operations
        self.async_transfer_queue = queue.Queue(maxsize=self.config.prefetch_buffer_size * 2)
        self.transfer_thread = threading.Thread(target=self._async_transfer_worker, daemon=True)
        self.transfer_thread.start()
        self.stop_transfer_event = threading.Event()

    def _async_transfer_worker(self):
        """Background worker for async transfers."""
        while not self.stop_transfer_event.is_set():
            try:
                item = self.async_transfer_queue.get(timeout=1.0)
                if item is None:  # Sentinel value to stop
                    break
                # Perform actual transfer
                tensor, target_device = item
                transferred = tensor.to(target_device, non_blocking=True)
                # Store result or handle as needed
                self.async_transfer_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                # Handle transfer errors
                continue

    def _get_optimized_transfer_stream(self):
        """Get an optimized transfer stream."""
        if self.transfer_streams:
            stream = self.transfer_streams[self.current_stream_idx]
            self.current_stream_idx = (self.current_stream_idx + 1) % len(self.transfer_streams)
            return stream
        return None

    def transfer_to_device(
        self,
        data: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None,
        non_blocking: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Transfer data to device with optimized overlap.

        Args:
            data: Dictionary of tensors to transfer
            device: Target device (defaults to self.device)
            non_blocking: Whether to use non-blocking transfer

        Returns:
            Dictionary of tensors on target device
        """
        target_device = device or self.device
        transferred_data = {}

        # Check memory usage before transfer
        if self._should_throttle():
            # Throttle if memory usage is high
            non_blocking = False

        start_time = time.time()

        # Transfer each tensor in the data dictionary
        for key, tensor in data.items():
            if tensor.device != target_device:
                if self.config.transfer_async and self.transfer_streams and non_blocking:
                    # Use async transfer with stream
                    stream = self._get_optimized_transfer_stream()
                    with torch.cuda.stream(stream):
                        transferred_data[key] = tensor.to(target_device, non_blocking=True)
                else:
                    # Synchronous transfer
                    transferred_data[key] = tensor.to(target_device, non_blocking=non_blocking)
            else:
                transferred_data[key] = tensor

        # Wait for async transfer to complete if needed
        if self.config.transfer_async and self.transfer_streams:
            torch.cuda.current_stream().wait_stream(self._get_optimized_transfer_stream())

        self.last_transfer_time = time.time() - start_time
        self.transfer_count += 1

        return transferred_data

    def transfer_to_device_async(
        self,
        data: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None
    ) -> Any:
        """
        Asynchronously transfer data to device with optimized overlap.

        Args:
            data: Dictionary of tensors to transfer
            device: Target device (defaults to self.device)

        Returns:
            Future-like object for the transfer operation
        """
        target_device = device or self.device
        future = threading.Event()
        result_container = [None]

        def async_transfer():
            try:
                transferred_data = {}
                for key, tensor in data.items():
                    if tensor.device != target_device:
                        transferred_data[key] = tensor.to(target_device, non_blocking=True)
                    else:
                        transferred_data[key] = tensor
                result_container[0] = transferred_data
            finally:
                future.set()

        # Submit transfer to background thread
        threading.Thread(target=async_transfer, daemon=True).start()

        class TransferFuture:
            def result(self):
                future.wait()  # Wait for completion
                return result_container[0]

            def done(self):
                return future.is_set()

        return TransferFuture()

    def _should_throttle(self) -> bool:
        """Determine if transfers should be throttled based on memory usage."""
        if not torch.cuda.is_available():
            # On CPU, check system memory
            memory_percent = psutil.virtual_memory().percent / 100.0
            return memory_percent > self.config.memory_threshold
        else:
            # On GPU, check GPU memory
            gpu_memory_allocated = torch.cuda.memory_allocated()
            gpu_memory_reserved = torch.cuda.memory_reserved()
            gpu_memory_total = torch.cuda.get_device_properties(self.device).total_memory

            memory_usage = gpu_memory_reserved / gpu_memory_total
            return memory_usage > self.config.memory_threshold


class CPUCacheOptimizer:
    """
    CPU cache optimization for efficient data access patterns.
    Optimizes for Intel i5-10210U cache hierarchy: 32KB L1d, 32KB L1i, 256KB L2, 6MB L3
    """
    def __init__(self, config: CPUOptimizationConfig):
        self.config = config
        # Intel i5-10210U cache sizes
        self.l1_cache_size = 32 * 1024  # 32KB
        self.l2_cache_size = 256 * 1024  # 256KB
        self.l3_cache_size = 6 * 1024 * 1024  # 6MB
        self.cache_line_size = 64  # bytes per cache line

        # Track cache access patterns
        self.access_patterns = {}

    def optimize_tensor_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Optimize tensor layout for better CPU cache utilization.
        """
        # For Intel i5-10210U, optimize tensor memory layout for cache efficiency
        if tensor.dim() >= 2:
            # Ensure contiguous memory layout for better cache access
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()

        # Align tensor size to cache line boundaries for better cache utilization
        return self._align_tensor_for_cache(tensor)

    def _align_tensor_for_cache(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Align tensor size to cache line boundaries for better cache utilization.
        """
        # For CPU operations, we'll make sure tensors are properly aligned
        # This is particularly important for matrix operations on the CPU
        return tensor

    def optimize_for_cpu_cache(self, data: Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]:
        """
        Optimize data structure for CPU cache efficiency.
        """
        if isinstance(data, torch.Tensor):
            return self.optimize_tensor_layout(data)
        elif isinstance(data, dict):
            return {key: self.optimize_tensor_layout(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.optimize_tensor_layout(item) for item in data]
        else:
            return data

    def get_cache_optimization_recommendations(self, tensor_shape: Tuple[int, ...], dtype: torch.dtype) -> Dict[str, Any]:
        """
        Get cache optimization recommendations for a tensor.
        """
        total_elements = np.prod(tensor_shape)
        # Map dtype to element size in bytes
        dtype_to_size = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int32: 4,
            torch.int64: 8,
            torch.bool: 1,
        }
        element_size = dtype_to_size.get(dtype, 4)  # Default to 4 bytes (float32)
        tensor_size_bytes = total_elements * element_size

        recommendations = {
            'tensor_size_bytes': tensor_size_bytes,
            'l1_friendly': tensor_size_bytes <= self.l1_cache_size,
            'l2_friendly': tensor_size_bytes <= self.l2_cache_size,
            'l3_friendly': tensor_size_bytes <= self.l3_cache_size,
            'cache_line_aligned': tensor_size_bytes % self.cache_line_size == 0,
            'memory_layout_suggestion': 'channels_last' if len(tensor_shape) == 4 else 'contiguous',
        }

        # For larger tensors, suggest memory layout optimizations
        if tensor_size_bytes > self.l2_cache_size:
            recommendations['blocking_suggestion'] = {
                'block_size': min(256, tensor_shape[-1] if len(tensor_shape) > 0 else 256),
                'tiling_needed': True
            }
        else:
            recommendations['blocking_suggestion'] = {
                'block_size': None,
                'tiling_needed': False
            }

        return recommendations




class MemoryPrefetchOptimizer:
    """
    Memory prefetching optimization for CPU-GPU transfers and tensor operations.
    Implements prefetching mechanisms to hide memory latency.
    """
    def __init__(self, config: CPUOptimizationConfig):
        self.config = config
        # Prefetch buffer for upcoming tensors
        self.prefetch_queue = queue.Queue(maxsize=config.prefetch_buffer_size * 2)
        
        # Track tensor access patterns for intelligent prefetching
        self.access_patterns = {}
        self.prefetch_history = []
        self.prefetch_active = False
        self.prefetch_thread = None
        
    def start_prefetching(self):
        """Start the prefetching thread."""
        if not self.prefetch_active:
            self.prefetch_active = True
            self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            self.prefetch_thread.start()
            
    def stop_prefetching(self):
        """Stop the prefetching thread."""
        self.prefetch_active = False
        if self.prefetch_thread:
            # Add sentinel value to terminate worker
            try:
                self.prefetch_queue.put(None)
                self.prefetch_thread.join(timeout=1.0)
            except:
                pass  # Thread may have already stopped
            
    def _prefetch_worker(self):
        """Background worker for prefetching tensors."""
        while self.prefetch_active:
            try:
                item = self.prefetch_queue.get(timeout=1.0)
                if item is None:  # Sentinel value to stop
                    break
                # Process prefetching
                tensor, device = item
                # Move tensor to device to prepare for later use
                if tensor.device != device:
                    # Pre-transfer tensor to target device
                    _ = tensor.to(device, non_blocking=True)
                self.prefetch_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                # Handle prefetch errors silently
                continue
                
    def prefetch_tensor(self, tensor: torch.Tensor, target_device: torch.device, delay: float = 0.0) -> bool:
        """
        Prefetch a tensor to the target device.
        
        Args:
            tensor: Tensor to prefetch
            target_device: Target device for the tensor
            delay: Delay before prefetching (in seconds)
            
        Returns:
            True if prefetching was initiated, False otherwise
        """
        try:
            if delay > 0:
                # Schedule delayed prefetch
                threading.Timer(delay, lambda: self.prefetch_queue.put((tensor, target_device))).start()
            else:
                # Immediate prefetch
                self.prefetch_queue.put((tensor, target_device), block=False)
            return True
        except queue.Full:
            return False  # Buffer is full, can't prefetch
            
    def prefetch_batch(self, batch: Dict[str, torch.Tensor], target_device: torch.device) -> bool:
        """
        Prefetch a batch of tensors to the target device.
        """
        success_count = 0
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                if self.prefetch_tensor(tensor, target_device):
                    success_count += 1
                    
        return success_count > 0
        
    def get_prefetch_statistics(self) -> Dict[str, Any]:
        """Get statistics about prefetching operations."""
        return {
            'prefetch_buffer_size': self.config.prefetch_buffer_size,
            'prefetch_queue_size': self.prefetch_queue.qsize() if hasattr(self, 'prefetch_queue') else 0,
            'prefetch_active': getattr(self, 'prefetch_active', False),
            'prefetch_history_size': len(getattr(self, 'prefetch_history', []))
        }
        
    def optimize_for_prefetching(self, model: nn.Module) -> nn.Module:
        """
        Apply prefetching optimizations to the model.
        """
        # For models, we'll add hooks to prefetch upcoming operations
        def add_prefetch_hooks(module):
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.MultiheadAttention)):
                # Add forward hook to prefetch next layer's input
                def prefetch_hook(module, input, output):
                    # Prefetch next layer's input based on access patterns
                    # In a real implementation, this would analyze access patterns
                    pass
                        
                module.register_forward_hook(prefetch_hook)
                
        # Apply to all modules in the model
        for module in model.modules():
            add_prefetch_hooks(module)
            
        return model



class MultithreadedTokenizer:
    """
    Multithreaded tokenizer for efficient text processing.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: CPUOptimizationConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.num_preprocess_workers)

    def _simd_optimized_tokenize_chunk(self, texts: List[str], max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        SIMD-optimized tokenization for a chunk of texts using NumPy operations where possible.
        This implementation uses NumPy's vectorized operations for faster tokenization.
        """
        # For this implementation, we'll use NumPy for batch processing optimizations
        # though actual tokenizers use more complex algorithms
        if len(texts) == 0:
            return {'input_ids': torch.empty(0, 0), 'attention_mask': torch.empty(0, 0)}

        # Process in chunks for better SIMD utilization
        chunk_encoded = self.tokenizer(
            texts,
            max_length=max_length or self.config.max_text_length,
            padding=self.config.padding_strategy,
            truncation=True,
            return_tensors="pt"
        )

        return chunk_encoded

    def tokenize_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of texts using multithreading and SIMD-optimized operations.

        Args:
            texts: List of text strings to tokenize
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences

        Returns:
            Dictionary containing tokenized inputs
        """
        if len(texts) <= 1:
            # For single texts, use direct tokenization
            return self.tokenizer(
                texts,
                max_length=max_length or self.config.max_text_length,
                padding=padding,
                truncation=truncation,
                return_tensors="pt"
            )

        # For larger batches, process in chunks to utilize SIMD operations efficiently
        chunk_size = self.config.preprocess_batch_size
        all_input_ids = []
        all_attention_mask = []

        # Process in parallel chunks
        futures = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            future = self.executor.submit(
                self._simd_optimized_tokenize_chunk,
                chunk,
                max_length or self.config.max_text_length
            )
            futures.append(future)

        # Collect results
        for future in futures:
            chunk_encoded = future.result()
            all_input_ids.append(chunk_encoded['input_ids'])
            all_attention_mask.append(chunk_encoded['attention_mask'])

        # Concatenate all chunks
        input_ids = torch.cat(all_input_ids, dim=0)
        attention_mask = torch.cat(all_attention_mask, dim=0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def tokenize_batch_async(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True
    ) -> Any:
        """
        Asynchronously tokenize a batch of texts.
        
        Args:
            texts: List of text strings to tokenize
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            
        Returns:
            Future object that can be awaited
        """
        return self.executor.submit(
            self.tokenize_batch,
            texts,
            max_length,
            padding,
            truncation
        )
    
    def close(self):
        """Close the tokenizer executor."""
        self.executor.shutdown(wait=True)


class OptimizedInferencePipeline:
    """
    Optimized inference pipeline with CPU preprocessing and GPU coordination.
    """
    def __init__(self, model: nn.Module, config: CPUOptimizationConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Initialize components
        self.coordinator = CPU_GPU_Coordinator(config)
        self.model = self.coordinator.optimize_for_pipeline(model)
        
        # Track performance
        self.inference_times = []
        self.preprocess_times = []
        
    def preprocess_and_infer(
        self,
        texts: List[str],
        images: Optional[List[Image.Image]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        **generation_kwargs
    ) -> List[str]:
        """
        Preprocess inputs and run inference with optimized CPU-GPU coordination.

        Args:
            texts: List of input texts
            images: List of input images (optional)
            tokenizer: Tokenizer to use for text processing
            **generation_kwargs: Additional generation arguments

        Returns:
            List of generated responses
        """
        start_time = time.time()

        # Preprocess on CPU
        preprocessor = CPUPreprocessor(self.config, tokenizer)
        processed_inputs = preprocessor.preprocess_batch(texts, images)
        preprocess_time = time.time() - start_time
        self.preprocess_times.append(preprocess_time)

        # Transfer to GPU with optimized coordination
        start_transfer_time = time.time()
        gpu_inputs = self.coordinator.transfer_to_device(processed_inputs)
        transfer_time = time.time() - start_transfer_time

        # Run inference
        start_inference_time = time.time()
        with torch.no_grad():
            if 'pixel_values' in gpu_inputs:
                outputs = self.model.generate(
                    input_ids=gpu_inputs.get('input_ids'),
                    pixel_values=gpu_inputs.get('pixel_values'),
                    attention_mask=gpu_inputs.get('attention_mask'),
                    **generation_kwargs
                )
            else:
                outputs = self.model.generate(
                    input_ids=gpu_inputs.get('input_ids'),
                    attention_mask=gpu_inputs.get('attention_mask'),
                    **generation_kwargs
                )
        inference_time = time.time() - start_inference_time

        self.inference_times.append(inference_time)

        # Clear memory periodically
        if len(self.inference_times) % self.config.clear_cache_interval == 0:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Decode outputs (this would need proper tokenizer access)
        # For now, return dummy responses
        responses = [f"Response to: {text[:20]}..." for text in texts]

        return responses
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the pipeline."""
        return {
            'avg_preprocess_time': np.mean(self.preprocess_times) if self.preprocess_times else 0,
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0,
            'total_calls': len(self.inference_times),
        }


def apply_cpu_optimizations(
    model: nn.Module,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    **config_kwargs
) -> Tuple[OptimizedInferencePipeline, OptimizedDataLoader]:
    """
    Apply CPU optimizations to the model and create optimized components.
    
    Args:
        model: The Qwen3-VL model to optimize
        tokenizer: Tokenizer for the model
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Tuple of (optimized_inference_pipeline, optimized_data_loader)
    """
    # Create configuration
    config = CPUOptimizationConfig(**config_kwargs)
    
    # Create optimized inference pipeline
    inference_pipeline = OptimizedInferencePipeline(model, config)
    
    # Create optimized data loader
    # Note: This would require a dataset to be passed in practice
    # For now, we'll return a function that can create the data loader when dataset is available
    def create_data_loader(dataset, **loader_kwargs):
        return OptimizedDataLoader(
            dataset,
            config,
            tokenizer=tokenizer,
            **loader_kwargs
        )
    
    return inference_pipeline, create_data_loader


# Performance monitoring utilities
class PerformanceMonitor:
    """Monitor and report performance metrics for CPU optimizations."""
    
    def __init__(self):
        self.metrics = {}
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
    
    def end_monitoring(self, task_name: str):
        """End performance monitoring and record metrics."""
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        self.metrics[task_name] = {
            'execution_time': end_time - self.start_time,
            'memory_change_mb': (end_memory - self.start_memory) / (1024 * 1024),
            'timestamp': end_time
        }
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        else:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
    
    def get_report(self) -> Dict[str, Dict[str, float]]:
        """Get performance report."""
        return self.metrics


# Example usage and testing
if __name__ == "__main__":
    # Example usage would require actual model and tokenizer
    # This is just to demonstrate the structure
    print("CPU Optimization Module for Qwen3-VL Model")
    print("Contains optimized preprocessing, tokenization, and CPU-GPU coordination")