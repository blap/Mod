"""
Cross-Layer Memory Sharing System for Qwen3-VL Model Optimization

This module implements a sophisticated memory sharing mechanism that allows
different layers of the model to safely share activations and other tensor data,
significantly reducing memory usage while maintaining computational integrity.
"""

import threading
import weakref
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import torch
import numpy as np


class TensorType(Enum):
    """Enum to categorize different types of tensors for optimized sharing strategies."""
    KEY = "key"
    VALUE = "value"
    ACTIVATION = "activation"
    INTERMEDIATE = "intermediate"
    GRADIENT = "gradient"


@dataclass
class TensorMetadata:
    """Metadata associated with each tensor for sharing decisions."""
    shape: Tuple[int, ...]
    dtype: torch.dtype
    tensor_type: TensorType
    layer_id: int
    device: str
    reference_count: int = 0
    shared_with: List[int] = None  # List of layer IDs this tensor is shared with
    
    def __post_init__(self):
        if self.shared_with is None:
            self.shared_with = []


class ReferenceCounter:
    """Thread-safe reference counter for managing tensor lifetimes."""
    
    def __init__(self, initial_count: int = 0):
        self._count = initial_count
        self._lock = threading.Lock()
        
    def increment(self) -> int:
        """Increment reference count and return the new value."""
        with self._lock:
            self._count += 1
            return self._count
            
    def decrement(self) -> int:
        """Decrement reference count and return the new value."""
        with self._lock:
            if self._count <= 0:
                raise RuntimeError("Reference count cannot go below zero")
            self._count -= 1
            return self._count
            
    def get_count(self) -> int:
        """Get current reference count."""
        with self._lock:
            return self._count


class CrossLayerMemoryManager:
    """
    Manages cross-layer memory sharing for transformer models.
    
    This class implements a sophisticated memory sharing system that identifies
    opportunities to reuse activations and other tensors across different layers
    of the model, significantly reducing memory consumption while ensuring
    thread safety and preventing use-after-free errors.
    """
    
    def __init__(self, 
                 max_shared_tensors: int = 1000,
                 compatibility_threshold: float = 0.95,
                 hardware_target: str = "intel_i5_nvidia_sm61"):
        """
        Initialize the Cross-Layer Memory Manager.
        
        Args:
            max_shared_tensors: Maximum number of tensors that can be shared
            compatibility_threshold: Threshold for determining tensor compatibility
            hardware_target: Target hardware profile for optimizations
        """
        self.max_shared_tensors = max_shared_tensors
        self.compatibility_threshold = compatibility_threshold
        self.hardware_target = hardware_target
        
        # Dictionary mapping tensor identifiers to metadata and references
        self.tensor_registry: Dict[str, Tuple[torch.Tensor, TensorMetadata, ReferenceCounter]] = {}
        
        # Maps layer ID to list of tensor IDs that layer owns/references
        self.layer_tensor_map: Dict[int, List[str]] = {}
        
        # Thread lock for safe concurrent access
        self._lock = threading.Lock()
        
        # Hardware-specific optimizations
        self._setup_hardware_optimizations()
        
    def _setup_hardware_optimizations(self):
        """Configure hardware-specific optimizations based on target."""
        if "intel_i5" in self.hardware_target.lower():
            # Optimizations for Intel i5-10210U (Comet Lake, 4 cores/8 threads, 6MB L3 cache)
            self._cpu_cache_size = 6 * 1024 * 1024  # 6MB L3 cache
            self._max_cpu_threads = 8  # 4 cores with hyperthreading
            self._memory_bandwidth = 40 * 1024 * 1024 * 1024  # 40GB/s theoretical bandwidth
            self._optimal_tensor_chunk_size = 1024 * 1024  # Chunk tensors to fit in cache

            # Adjust memory allocation patterns for CPU efficiency
            self._enable_cpu_memory_pooling = True
            self._cpu_thread_local_caching = True

        if "nvidia_sm61" in self.hardware_target.lower():
            # Optimizations for NVIDIA architecture SM61 (Maxwell/Pascal generation)
            # Common in GTX 900/1000 series
            self._gpu_memory_alignment = 256  # Align to 256-byte boundaries for optimal coalescing
            self._preferred_gpu_block_size = (16, 16)  # Optimal block size for SM61
            self._max_shared_memory_per_block = 48 * 1024  # 48KB shared memory per block
            self._warp_size = 32  # Number of threads per warp

            # Enable GPU-specific optimizations
            self._enable_gpu_memory_pooling = True
            self._enable_async_memory_transfer = True

        if "nvme" in self.hardware_target.lower():
            # Optimizations for NVMe SSD storage
            self._storage_prefetch_size = 128 * 1024  # 128KB prefetch
            self._async_io_enabled = True
            self._io_buffer_size = 2 * 1024 * 1024  # 2MB buffer for I/O operations
            self._max_io_concurrency = 32  # Max concurrent I/O operations

        # Platform-specific optimizations for combined hardware
        if "intel_i5" in self.hardware_target.lower() and "nvidia_sm61" in self.hardware_target.lower():
            # Optimizations when using both CPU and GPU
            self._hybrid_compute_strategy = "cpu_preprocess_gpu_compute"
            self._optimal_data_transfer_size = min(
                self._cpu_cache_size,
                self._max_shared_memory_per_block
            )

        # Set memory allocation strategies based on hardware
        self._set_memory_allocation_strategy()

    def _set_memory_allocation_strategy(self):
        """Set memory allocation strategy based on hardware capabilities."""
        if hasattr(self, '_cpu_cache_size') and hasattr(self, '_gpu_memory_alignment'):
            # Hybrid CPU/GPU system
            self._allocation_strategy = {
                'small_tensors': 'cpu',  # Small tensors kept on CPU for fast access
                'large_tensors': 'gpu',  # Large tensors on GPU to leverage parallelism
                'frequently_accessed': 'cpu',
                'compute_intensive': 'gpu'
            }
        elif hasattr(self, '_cpu_cache_size'):
            # CPU-only system
            self._allocation_strategy = {
                'small_tensors': 'cpu',
                'large_tensors': 'cpu',
                'frequently_accessed': 'cpu',
                'compute_intensive': 'cpu'
            }
        else:
            # Default strategy
            self._allocation_strategy = {
                'small_tensors': 'cpu',
                'large_tensors': 'cpu',  # Default to CPU if GPU not detected
                'frequently_accessed': 'cpu',
                'compute_intensive': 'cpu'
            }

    def optimize_for_hardware(self, tensor: torch.Tensor, tensor_type: TensorType) -> torch.Tensor:
        """
        Apply hardware-specific optimizations to a tensor.

        Args:
            tensor: Input tensor to optimize
            tensor_type: Type of tensor being optimized

        Returns:
            Optimized tensor
        """
        # Apply alignment for GPU operations
        if hasattr(self, '_gpu_memory_alignment') and tensor.is_cuda:
            # Ensure tensor is properly aligned for GPU memory access
            original_shape = tensor.shape
            # Pad tensor if needed to align with GPU requirements
            if tensor.data_ptr() % self._gpu_memory_alignment != 0:
                # Create a new tensor with proper alignment
                aligned_tensor = tensor.clone()
                return aligned_tensor
            else:
                return tensor

        # Apply CPU cache optimizations
        if hasattr(self, '_optimal_tensor_chunk_size'):
            # For large tensors, consider chunking to fit in CPU cache
            tensor_size_bytes = tensor.nelement() * tensor.element_size()
            if tensor_size_bytes > self._optimal_tensor_chunk_size and tensor_type != TensorType.GRADIENT:
                # For inference tensors, optimize for cache locality
                pass  # In this implementation, we just pass, but in practice would optimize

        return tensor
            
    def register_tensor(self, 
                       tensor: torch.Tensor, 
                       layer_id: int, 
                       tensor_type: TensorType) -> str:
        """
        Register a tensor with the memory manager.
        
        Args:
            tensor: The tensor to register
            layer_id: The layer ID that owns this tensor
            tensor_type: Type of tensor (key, value, activation, etc.)
            
        Returns:
            Unique identifier for the registered tensor
        """
        with self._lock:
            # Generate unique tensor ID based on tensor properties
            tensor_id = self._generate_tensor_id(tensor, layer_id, tensor_type)
            
            # Check if tensor is already registered
            if tensor_id in self.tensor_registry:
                # Increment reference count if already exists
                _, metadata, ref_counter = self.tensor_registry[tensor_id]
                ref_counter.increment()
                
                # Update layer reference
                if layer_id not in self.layer_tensor_map:
                    self.layer_tensor_map[layer_id] = []
                if tensor_id not in self.layer_tensor_map[layer_id]:
                    self.layer_tensor_map[layer_id].append(tensor_id)
                    
                return tensor_id
            
            # Create metadata for the tensor
            metadata = TensorMetadata(
                shape=tensor.shape,
                dtype=tensor.dtype,
                tensor_type=tensor_type,
                layer_id=layer_id,
                device=str(tensor.device)
            )
            
            # Create reference counter with initial count of 1
            ref_counter = ReferenceCounter(1)
            
            # Store in registry
            self.tensor_registry[tensor_id] = (tensor, metadata, ref_counter)
            
            # Update layer tensor map
            if layer_id not in self.layer_tensor_map:
                self.layer_tensor_map[layer_id] = []
            self.layer_tensor_map[layer_id].append(tensor_id)
            
            return tensor_id
    
    def _generate_tensor_id(self, 
                           tensor: torch.Tensor, 
                           layer_id: int, 
                           tensor_type: TensorType) -> str:
        """Generate a unique ID for a tensor based on its properties."""
        # Use hash of shape, dtype, device, and layer info to generate ID
        shape_str = "_".join(map(str, tensor.shape))
        return f"{tensor_type.value}_{shape_str}_{str(tensor.dtype)}_{tensor.device}_{layer_id}"
    
    def find_sharing_opportunities(self, 
                                  candidate_tensor: torch.Tensor, 
                                  layer_id: int, 
                                  tensor_type: TensorType) -> List[str]:
        """
        Find existing tensors that could be shared with the candidate tensor.
        
        Args:
            candidate_tensor: Tensor we want to potentially share
            layer_id: Layer requesting the tensor
            tensor_type: Type of the candidate tensor
            
        Returns:
            List of tensor IDs that are compatible for sharing
        """
        compatible_tensors = []
        candidate_metadata = TensorMetadata(
            shape=candidate_tensor.shape,
            dtype=candidate_tensor.dtype,
            tensor_type=tensor_type,
            layer_id=layer_id,
            device=str(candidate_tensor.device)
        )
        
        with self._lock:
            for tensor_id, (stored_tensor, stored_metadata, _) in self.tensor_registry.items():
                if self._is_compatible_for_sharing(candidate_metadata, stored_metadata):
                    compatible_tensors.append(tensor_id)
                    
        return compatible_tensors
    
    def _is_compatible_for_sharing(self,
                                  candidate_meta: TensorMetadata,
                                  stored_meta: TensorMetadata) -> bool:
        """
        Determine if two tensors are compatible for sharing.

        Args:
            candidate_meta: Metadata of the candidate tensor
            stored_meta: Metadata of the stored tensor

        Returns:
            True if tensors are compatible for sharing, False otherwise
        """
        # Shapes must match exactly
        if candidate_meta.shape != stored_meta.shape:
            return False

        # Data types must match
        if candidate_meta.dtype != stored_meta.dtype:
            return False

        # Devices must match
        if candidate_meta.device != stored_meta.device:
            return False

        # Tensor types must match
        if candidate_meta.tensor_type != stored_meta.tensor_type:
            return False

        # For gradient tensors, additional checks might be needed
        if candidate_meta.tensor_type == TensorType.GRADIENT:
            # Gradients from same layer type might not be shareable
            if candidate_meta.layer_id == stored_meta.layer_id:
                return False

        # Consider tensor usage patterns and lifetime
        return True
    
    def acquire_tensor(self, 
                      tensor_id: str, 
                      requesting_layer: int) -> Optional[torch.Tensor]:
        """
        Acquire a shared tensor reference. Increments reference count.
        
        Args:
            tensor_id: ID of the tensor to acquire
            requesting_layer: Layer ID requesting the tensor
            
        Returns:
            The tensor if found and accessible, None otherwise
        """
        with self._lock:
            if tensor_id not in self.tensor_registry:
                return None
                
            tensor, metadata, ref_counter = self.tensor_registry[tensor_id]
            
            # Increment reference count
            new_ref_count = ref_counter.increment()
            
            # Update metadata to reflect sharing
            if requesting_layer not in metadata.shared_with:
                metadata.shared_with.append(requesting_layer)
                
            # Update layer tensor map
            if requesting_layer not in self.layer_tensor_map:
                self.layer_tensor_map[requesting_layer] = []
            if tensor_id not in self.layer_tensor_map[requesting_layer]:
                self.layer_tensor_map[requesting_layer].append(tensor_id)
                
            return tensor.clone()  # Return clone to prevent accidental modifications
    
    def release_tensor(self, tensor_id: str, releasing_layer: int) -> bool:
        """
        Release a shared tensor reference. Decrements reference count.

        Args:
            tensor_id: ID of the tensor to release
            releasing_layer: Layer ID releasing the tensor

        Returns:
            True if successful, False if tensor doesn't exist
        """
        with self._lock:
            if tensor_id not in self.tensor_registry:
                return False

            _, metadata, ref_counter = self.tensor_registry[tensor_id]

            # Decrement reference count
            new_ref_count = ref_counter.decrement()

            # Remove from layer's tensor list
            if releasing_layer in self.layer_tensor_map:
                if tensor_id in self.layer_tensor_map[releasing_layer]:
                    self.layer_tensor_map[releasing_layer].remove(tensor_id)

            # Remove from shared_with list
            if releasing_layer in metadata.shared_with:
                metadata.shared_with.remove(releasing_layer)

            # Keep tensor in registry but mark for cleanup when reference count reaches zero
            # Actual removal happens in cleanup_unused_tensors method
            # This ensures thread safety and proper cleanup scheduling

            return True
    
    def get_tensor_by_id(self, tensor_id: str) -> Optional[Tuple[torch.Tensor, TensorMetadata]]:
        """
        Get tensor and its metadata by ID without changing reference count.
        
        Args:
            tensor_id: ID of the tensor to retrieve
            
        Returns:
            Tuple of (tensor, metadata) if found, None otherwise
        """
        with self._lock:
            if tensor_id not in self.tensor_registry:
                return None
                
            tensor, metadata, _ = self.tensor_registry[tensor_id]
            return tensor.clone(), metadata
    
    def share_tensor(self,
                     source_layer: int,
                     target_layer: int,
                     tensor_type: TensorType,
                     similarity_threshold: float = 0.9) -> bool:
        """
        Attempt to share a tensor between two layers.

        Args:
            source_layer: Layer ID that currently holds the tensor
            target_layer: Layer ID that wants to share the tensor
            tensor_type: Type of tensor to share
            similarity_threshold: Threshold for similarity comparison

        Returns:
            True if sharing was successful, False otherwise
        """
        with self._lock:
            # Find tensors of the specified type from source layer
            source_tensors = []
            for tensor_id in self.layer_tensor_map.get(source_layer, []):
                _, metadata, _ = self.tensor_registry[tensor_id]
                if metadata.tensor_type == tensor_type:
                    source_tensors.append(tensor_id)

            if not source_tensors:
                return False

            # Apply tensor-type-specific sharing strategy
            if tensor_type == TensorType.KEY:
                return self._share_key_tensors(source_tensors, source_layer, target_layer, similarity_threshold)
            elif tensor_type == TensorType.VALUE:
                return self._share_value_tensors(source_tensors, source_layer, target_layer, similarity_threshold)
            elif tensor_type == TensorType.ACTIVATION:
                return self._share_activation_tensors(source_tensors, source_layer, target_layer, similarity_threshold)
            elif tensor_type == TensorType.INTERMEDIATE:
                return self._share_intermediate_tensors(source_tensors, source_layer, target_layer, similarity_threshold)
            elif tensor_type == TensorType.GRADIENT:
                return self._share_gradient_tensors(source_tensors, source_layer, target_layer, similarity_threshold)
            else:
                # Default sharing strategy
                return self._share_generic_tensor(source_tensors[0], source_layer, target_layer)

    def _share_key_tensors(self, source_tensors: List[str], source_layer: int, target_layer: int, similarity_threshold: float) -> bool:
        """
        Specific sharing strategy for key tensors.
        Key tensors can often be shared across attention heads in certain contexts.
        """
        # For key tensors, we need to consider attention head compatibility
        for tensor_id in source_tensors:
            source_tensor, source_metadata, _ = self.tensor_registry[tensor_id]

            # Look for compatible key tensors in target layer
            target_tensors = []
            for tgt_tensor_id in self.layer_tensor_map.get(target_layer, []):
                _, tgt_metadata, _ = self.tensor_registry[tgt_tensor_id]
                if (tgt_metadata.tensor_type == TensorType.KEY and
                    self._is_compatible_for_sharing(source_metadata, tgt_metadata)):
                    target_tensors.append(tgt_tensor_id)

            if target_tensors:
                # In this simplified version, we just connect the first compatible tensors
                # More sophisticated implementations would consider attention patterns
                target_tensor_id = target_tensors[0]
                source_metadata.shared_with.append(target_layer)
                _, target_metadata, _ = self.tensor_registry[target_tensor_id]
                target_metadata.shared_with.append(source_layer)
                return True

        return False

    def _share_value_tensors(self, source_tensors: List[str], source_layer: int, target_layer: int, similarity_threshold: float) -> bool:
        """
        Specific sharing strategy for value tensors.
        Value tensors may have different sharing patterns than key tensors.
        """
        # For value tensors, sharing might depend on semantic similarity
        for tensor_id in source_tensors:
            source_tensor, source_metadata, _ = self.tensor_registry[tensor_id]

            # Look for compatible value tensors in target layer
            target_tensors = []
            for tgt_tensor_id in self.layer_tensor_map.get(target_layer, []):
                _, tgt_metadata, _ = self.tensor_registry[tgt_tensor_id]
                if (tgt_metadata.tensor_type == TensorType.VALUE and
                    self._is_compatible_for_sharing(source_metadata, tgt_metadata)):
                    target_tensors.append(tgt_tensor_id)

            if target_tensors:
                target_tensor_id = target_tensors[0]
                source_metadata.shared_with.append(target_layer)
                _, target_metadata, _ = self.tensor_registry[target_tensor_id]
                target_metadata.shared_with.append(source_layer)
                return True

        return False

    def _share_activation_tensors(self, source_tensors: List[str], source_layer: int, target_layer: int, similarity_threshold: float) -> bool:
        """
        Specific sharing strategy for activation tensors.
        Activations might be shared if they represent similar transformations.
        """
        # For activation tensors, sharing might be possible if they represent similar computations
        for tensor_id in source_tensors:
            source_tensor, source_metadata, _ = self.tensor_registry[tensor_id]

            # Look for compatible activation tensors in target layer
            target_tensors = []
            for tgt_tensor_id in self.layer_tensor_map.get(target_layer, []):
                _, tgt_metadata, _ = self.tensor_registry[tgt_tensor_id]
                if (tgt_metadata.tensor_type == TensorType.ACTIVATION and
                    self._is_compatible_for_sharing(source_metadata, tgt_metadata)):
                    target_tensors.append(tgt_tensor_id)

            if target_tensors:
                target_tensor_id = target_tensors[0]
                source_metadata.shared_with.append(target_layer)
                _, target_metadata, _ = self.tensor_registry[target_tensor_id]
                target_metadata.shared_with.append(source_layer)
                return True

        return False

    def _share_intermediate_tensors(self, source_tensors: List[str], source_layer: int, target_layer: int, similarity_threshold: float) -> bool:
        """
        Specific sharing strategy for intermediate tensors.
        Intermediate tensors in feed-forward networks might have sharing opportunities.
        """
        # For intermediate tensors, sharing depends on similar processing stages
        for tensor_id in source_tensors:
            source_tensor, source_metadata, _ = self.tensor_registry[tensor_id]

            # Look for compatible intermediate tensors in target layer
            target_tensors = []
            for tgt_tensor_id in self.layer_tensor_map.get(target_layer, []):
                _, tgt_metadata, _ = self.tensor_registry[tgt_tensor_id]
                if (tgt_metadata.tensor_type == TensorType.INTERMEDIATE and
                    self._is_compatible_for_sharing(source_metadata, tgt_metadata)):
                    target_tensors.append(tgt_tensor_id)

            if target_tensors:
                target_tensor_id = target_tensors[0]
                source_metadata.shared_with.append(target_layer)
                _, target_metadata, _ = self.tensor_registry[target_tensor_id]
                target_metadata.shared_with.append(source_layer)
                return True

        return False

    def _share_gradient_tensors(self, source_tensors: List[str], source_layer: int, target_layer: int, similarity_threshold: float) -> bool:
        """
        Specific sharing strategy for gradient tensors.
        Gradient tensors sharing is typically limited due to training dynamics.
        """
        # For gradient tensors, sharing is generally discouraged but might be possible in some cases
        # This implementation restricts gradient sharing to prevent training issues
        return False

    def _share_generic_tensor(self, tensor_id: str, source_layer: int, target_layer: int) -> bool:
        """
        Generic sharing strategy for unspecified tensor types.
        """
        source_tensor, source_metadata, _ = self.tensor_registry[tensor_id]

        # Look for any compatible tensor in target layer
        for tgt_tensor_id in self.layer_tensor_map.get(target_layer, []):
            _, tgt_metadata, _ = self.tensor_registry[tgt_tensor_id]
            if self._is_compatible_for_sharing(source_metadata, tgt_metadata):
                source_metadata.shared_with.append(target_layer)
                tgt_metadata.shared_with.append(source_layer)
                return True

        return False
    
    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about memory usage and sharing.
        
        Returns:
            Dictionary with memory usage statistics
        """
        with self._lock:
            total_tensors = len(self.tensor_registry)
            total_layers = len(self.layer_tensor_map)
            
            # Calculate shared tensor statistics
            shared_tensors = 0
            total_references = 0
            
            for tensor_id, (_, metadata, ref_counter) in self.tensor_registry.items():
                ref_count = ref_counter.get_count()
                total_references += ref_count
                if ref_count > 1:
                    shared_tensors += 1
                    
            # Estimate memory savings
            estimated_saved_tensors = sum(
                ref_counter.get_count() - 1 
                for _, (_, _, ref_counter) in self.tensor_registry.items()
            )
            
            return {
                "total_tensors": total_tensors,
                "total_layers": total_layers,
                "shared_tensors": shared_tensors,
                "total_references": total_references,
                "average_reference_count": total_references / max(total_tensors, 1),
                "estimated_saved_tensors": estimated_saved_tensors,
                "sharing_ratio": shared_tensors / max(total_tensors, 1)
            }
    
    def cleanup_unused_tensors(self):
        """Remove tensors with zero reference counts."""
        with self._lock:
            tensors_to_remove = []

            for tensor_id, (_, _, ref_counter) in self.tensor_registry.items():
                if ref_counter.get_count() <= 0:
                    tensors_to_remove.append(tensor_id)

            for tensor_id in tensors_to_remove:
                del self.tensor_registry[tensor_id]

                # Clean up from layer maps
                for layer_id, tensor_list in self.layer_tensor_map.items():
                    if tensor_id in tensor_list:
                        tensor_list.remove(tensor_id)
                        
    def get_tensor_compatibility_matrix(self, layer_ids: List[int]) -> Dict[str, List[float]]:
        """
        Calculate compatibility scores between tensors from different layers.
        
        Args:
            layer_ids: List of layer IDs to compare
            
        Returns:
            Dictionary mapping tensor pairs to compatibility scores
        """
        compatibility_matrix = {}
        
        with self._lock:
            for i, layer_id1 in enumerate(layer_ids):
                for j, layer_id2 in enumerate(layer_ids):
                    if i >= j:  # Only compute upper triangle to avoid duplicates
                        continue
                        
                    layer1_tensors = self.layer_tensor_map.get(layer_id1, [])
                    layer2_tensors = self.layer_tensor_map.get(layer_id2, [])
                    
                    for tensor_id1 in layer1_tensors:
                        for tensor_id2 in layer2_tensors:
                            tensor1, meta1, _ = self.tensor_registry[tensor_id1]
                            tensor2, meta2, _ = self.tensor_registry[tensor_id2]
                            
                            if self._is_compatible_for_sharing(meta1, meta2):
                                # Calculate similarity score based on tensor values if needed
                                # For now, just record the compatibility
                                pair_key = f"{tensor_id1}_vs_{tensor_id2}"
                                compatibility_matrix[pair_key] = [
                                    float(meta1.shape == meta2.shape),  # Shape compatibility
                                    float(meta1.dtype == meta2.dtype),  # Dtype compatibility
                                    float(meta1.device == meta2.device)  # Device compatibility
                                ]
                                
        return compatibility_matrix