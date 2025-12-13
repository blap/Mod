"""
Integration Module for Advanced Memory Tiering System

This module integrates the advanced memory tiering system with existing
cache, compression, and swapping systems in the Qwen3-VL project.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import threading
import logging

# Import our newly created modules
from advanced_memory_tiering_system import AdvancedMemoryTieringSystem, MemoryTier, TensorType
from ml_pattern_prediction_system import LightweightMLPredictor, PredictionAlgorithm
from hardware_specific_optimizations import HardwareSpecificOptimizer, create_hardware_optimizer
from memory_compression_system import MemoryCompressionManager
from advanced_memory_swapping_system import AdvancedMemorySwapper, SwapAlgorithm
from src.qwen3_vl.optimization.hierarchical_cache_manager import HierarchicalCacheManager, CacheConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedMemoryManager:
    """
    Integrated memory manager that combines tiering, caching, compression, and swapping.
    This is the main interface for memory management in Qwen3-VL.
    """
    
    def __init__(self,
                 tiering_system: AdvancedMemoryTieringSystem,
                 compression_manager: MemoryCompressionManager,
                 swapping_system: AdvancedMemorySwapper,
                 hierarchical_cache: HierarchicalCacheManager,
                 ml_predictor: LightweightMLPredictor,
                 hardware_optimizer: HardwareSpecificOptimizer):
        """
        Initialize the integrated memory manager.
        
        Args:
            tiering_system: Advanced memory tiering system
            compression_manager: Memory compression system
            swapping_system: Memory swapping system
            hierarchical_cache: Hierarchical cache system
            ml_predictor: ML-based prediction system
            hardware_optimizer: Hardware-specific optimizations
        """
        self.tiering_system = tiering_system
        self.compression_manager = compression_manager
        self.swapping_system = swapping_system
        self.hierarchical_cache = hierarchical_cache
        self.ml_predictor = ml_predictor
        self.hardware_optimizer = hardware_optimizer
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Stats tracking
        self.stats = {
            'tensor_allocations': 0,
            'tensor_accesses': 0,
            'compressions_performed': 0,
            'swaps_performed': 0,
            'tier_migrations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Start background optimization threads
        self._start_background_optimizations()
        
        logger.info("Integrated Memory Manager initialized successfully")

    def _start_background_optimizations(self):
        """Start background threads for ongoing optimizations."""
        
        def run_tiering_optimizations():
            """Run tiering optimizations in background."""
            while True:
                try:
                    # Perform predictive migrations every 10 seconds
                    self.tiering_system._perform_predictive_migrations()
                    time.sleep(10)
                except Exception as e:
                    logger.error(f"Error in tiering optimization thread: {e}")
                    time.sleep(5)

        def run_swapping_optimizations():
            """Run swapping optimizations in background."""
            while True:
                try:
                    # Check if swapping is needed every 5 seconds
                    swapped_count = self.swapping_system.perform_swapping()
                    if swapped_count > 0:
                        self.stats['swaps_performed'] += swapped_count
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"Error in swapping optimization thread: {e}")
                    time.sleep(5)

        # Start background threads
        tiering_thread = threading.Thread(target=run_tiering_optimizations, daemon=True)
        swapping_thread = threading.Thread(target=run_swapping_optimizations, daemon=True)
        
        tiering_thread.start()
        swapping_thread.start()

    def allocate_tensor(self,
                       shape: Tuple[int, ...],
                       dtype: torch.dtype = torch.float16,
                       tensor_type: TensorType = TensorType.GENERAL,
                       pinned: bool = False,
                       compress: bool = True,
                       preferred_tier: Optional[MemoryTier] = None) -> Tuple[torch.Tensor, str]:
        """
        Allocate a tensor with integrated memory management.
        
        Args:
            shape: Shape of the tensor
            dtype: Data type of the tensor
            tensor_type: Type of tensor for optimization
            pinned: Whether tensor should be pinned (not eligible for migration)
            compress: Whether to compress the tensor if possible
            preferred_tier: Preferred tier (None for automatic selection)
            
        Returns:
            Tuple of (tensor, tensor_id)
        """
        with self._lock:
            self.stats['tensor_allocations'] += 1
            
            # Create the tensor
            tensor = torch.zeros(shape, dtype=dtype)
            
            # Apply compression if requested and beneficial
            if compress:
                original_size = tensor.element_size() * tensor.nelement()
                compressed_data = self.compression_manager.compress_tensor(tensor, method='auto')
                compressed_size = compressed_data.get('memory_saved_bytes', 0)
                
                if compressed_size > original_size * 0.1:  # Only compress if saving >10%
                    tensor = self.compression_manager.decompress_tensor(compressed_data)
                    self.stats['compressions_performed'] += 1
            
            # Generate tensor ID first to ensure consistency
            tensor_id = self.tiering_system._generate_tensor_id(shape, dtype, tensor_type)

            # Store in the tiering system
            success, actual_tensor_id = self.tiering_system.put_tensor(
                tensor,
                tensor_type=tensor_type,
                preferred_tier=preferred_tier,
                pinned=pinned
            )

            if not success:
                # Fallback to direct allocation if tiering fails
                logger.warning("Tiering allocation failed, using direct allocation")
                tensor_id = f"direct_{id(tensor)}_{int(time.time())}"
                return tensor, tensor_id

            tensor_id = actual_tensor_id
            
            # Register with swapping system
            # Use the actual tensor ID from the tiering system
            tensor_size = tensor.element_size() * tensor.nelement()
            self.swapping_system.register_memory_block(
                tensor_id,
                tensor_size,
                self._convert_tensor_type_for_swapping(tensor_type),
                pinned
            )
            
            # Record access pattern for prediction
            self.ml_predictor.record_tensor_access(
                tensor_id,
                access_type='write',
                tensor_size=tensor_size,
                tensor_type=tensor_type.value,
                context={'allocation_time': time.time()}
            )
            
            return tensor, tensor_id

    def access_tensor(self,
                     tensor_id: str,
                     target_device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        """
        Access a tensor, triggering all integrated optimizations.

        Args:
            tensor_id: ID of the tensor to access
            target_device: Target device for the tensor

        Returns:
            Tensor if found, None otherwise
        """
        with self._lock:
            self.stats['tensor_accesses'] += 1

            # Get from tiering system
            tensor = self.tiering_system.get_tensor(tensor_id, target_device)

            if tensor is not None:
                # Update access in predictor
                self.ml_predictor.record_tensor_access(
                    tensor_id,
                    access_type='read',
                    tensor_size=tensor.element_size() * tensor.nelement(),
                    tensor_type='tiered',
                    context={'access_time': time.time()}
                )

                # Update access in swapping system
                self.swapping_system.access_memory_block(tensor_id)

                # Put in hierarchical cache for faster future access
                try:
                    self.hierarchical_cache.put_tensor(tensor, tensor_id)
                except Exception as e:
                    logger.debug(f"Could not put tensor in hierarchical cache: {e}")

                return tensor

            # If not in tiering system, check if it's swapped out
            try:
                swapped_block = self.swapping_system.access_memory_block(tensor_id)
                if swapped_block is not None:
                    # This would involve swapping the tensor back in
                    # For now, we'll return None as the actual tensor data isn't stored in the swapping system
                    logger.warning(f"Tensor {tensor_id} is registered in swapping system but data not available")
            except Exception as e:
                logger.debug(f"Error accessing swapping system: {e}")

            return None

    def _convert_tensor_type_for_swapping(self, tensor_type: TensorType):
        """Convert our TensorType to the swapping system's MemoryRegionType."""
        from advanced_memory_swapping_system import MemoryRegionType
        
        conversion_map = {
            TensorType.GENERAL: MemoryRegionType.TENSOR_DATA,
            TensorType.KV_CACHE: MemoryRegionType.KV_CACHE,
            TensorType.IMAGE_FEATURES: MemoryRegionType.TENSOR_DATA,
            TensorType.TEXT_EMBEDDINGS: MemoryRegionType.TENSOR_DATA,
            TensorType.TEMPORARY: MemoryRegionType.TEMPORARY
        }
        
        return conversion_map.get(tensor_type, MemoryRegionType.TENSOR_DATA)

    def migrate_tensor_proactively(self, tensor_id: str):
        """
        Proactively migrate a tensor based on predictions.
        
        Args:
            tensor_id: ID of the tensor to migrate proactively
        """
        with self._lock:
            # Get prediction for the tensor
            predicted_time, confidence = self.ml_predictor.predict_tensor_access(tensor_id)
            
            if predicted_time is None or confidence < 0.5:
                return  # Not enough confidence to migrate
            
            # Determine if migration is needed based on prediction
            time_until_access = max(0, predicted_time - time.time())
            priority = self.ml_predictor.get_tensor_priority(tensor_id)
            
            if priority > 0.7:  # High priority tensor
                # Check current location and predicted needs
                current_tier = self._get_tensor_tier(tensor_id)
                
                if current_tier == MemoryTier.NVME_SSD and time_until_access < 10.0:
                    # Tensor predicted to be accessed soon but is on slow storage
                    # Move to faster tier
                    self._migrate_tensor_to_optimal_tier(tensor_id, predicted_time, priority)
                elif current_tier == MemoryTier.CPU_RAM and time_until_access < 2.0 and priority > 0.9:
                    # High priority tensor that will be accessed very soon, move to GPU
                    self._migrate_tensor_to_optimal_tier(tensor_id, predicted_time, priority)

    def _get_tensor_tier(self, tensor_id: str) -> Optional[MemoryTier]:
        """Get the current tier of a tensor."""
        if tensor_id in self.tiering_system.tensor_locations:
            return self.tiering_system.tensor_locations[tensor_id]
        return None

    def _migrate_tensor_to_optimal_tier(self, tensor_id: str, predicted_time: float, priority: float):
        """Migrate a tensor to its optimal tier based on prediction."""
        # This is a simplified version - in a real implementation, we would
        # determine the optimal tier and trigger migration
        access_frequency = self.ml_predictor.get_access_frequency(tensor_id)
        time_until_access = max(0, predicted_time - time.time())
        
        optimal_tier_str = self.hardware_optimizer.get_optimal_tier_assignment(
            tensor_id,  # In a real implementation, we'd need to get the actual size
            "general",  # Placeholder tensor type
            access_frequency,
            time_until_access
        )
        
        # Convert string to enum
        if optimal_tier_str == 'gpu_hbm':
            optimal_tier = MemoryTier.GPU_HBM
        elif optimal_tier_str == 'cpu_ram':
            optimal_tier = MemoryTier.CPU_RAM
        else:
            optimal_tier = MemoryTier.NVME_SSD
        
        # In a real implementation, we would perform the migration here
        # For now, we'll just log the recommendation
        current_tier = self._get_tensor_tier(tensor_id)
        if current_tier and current_tier != optimal_tier:
            logger.info(f"Recommended migration for {tensor_id}: {current_tier.value} -> {optimal_tier.value}")

    def get_tensor_compression_ratio(self, tensor: torch.Tensor) -> float:
        """
        Get the compression ratio that would be achieved for a tensor.
        
        Args:
            tensor: Tensor to evaluate for compression
            
        Returns:
            Compression ratio (1.0 = no compression, <1.0 = compressed)
        """
        original_size = tensor.element_size() * tensor.nelement()
        compressed_data = self.compression_manager.compress_tensor(tensor, method='auto')
        compressed_size = compressed_data.get('memory_saved_bytes', 0)
        
        if original_size > 0:
            return (original_size - compressed_size) / original_size
        return 0.0

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics for the integrated system.
        
        Returns:
            Dictionary with system statistics
        """
        tiering_stats = self.tiering_system.get_stats()
        compression_stats = self.compression_manager.get_compression_stats()
        swapping_stats = self.swapping_system.get_status()
        cache_stats = self.hierarchical_cache.get_stats()
        
        # Combine all stats
        combined_stats = {
            'integrated_stats': self.stats,
            'tiering_stats': tiering_stats,
            'compression_stats': {
                'compression_ratio': compression_stats.compression_ratio,
                'compression_time': compression_stats.compression_time,
                'memory_saved_bytes': compression_stats.memory_saved_bytes,
                'total_tensors_compressed': compression_stats.total_tensors_compressed
            },
            'swapping_stats': swapping_stats,
            'cache_stats': cache_stats,
            'overall_cache_hit_rate': self._calculate_overall_cache_hit_rate()
        }
        
        return combined_stats

    def _calculate_overall_cache_hit_rate(self) -> float:
        """Calculate the overall cache hit rate across all cache levels."""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        if total_requests > 0:
            return self.stats['cache_hits'] / total_requests
        return 0.0

    def clear_memory(self):
        """Clear all memory systems."""
        with self._lock:
            self.tiering_system.clear_all()
            self.compression_manager.clear_cache()
            self.swapping_system.cleanup()  # Assuming this method exists
            self.hierarchical_cache.clear_cache()


def create_integrated_memory_manager(hardware_config: Optional[Dict[str, Any]] = None) -> IntegratedMemoryManager:
    """
    Factory function to create an integrated memory manager with all systems.
    
    Args:
        hardware_config: Hardware configuration for optimizations
        
    Returns:
        IntegratedMemoryManager instance
    """
    # Create hardware optimizer
    hardware_optimizer = create_hardware_optimizer()
    
    # Create memory tiering system with hardware-aware sizing
    tiering_system = AdvancedMemoryTieringSystem(
        gpu_hbm_size=hardware_optimizer.optimal_tier_sizes['gpu_hbm'],
        cpu_ram_size=hardware_optimizer.optimal_tier_sizes['cpu_ram'],
        nvme_ssd_size=hardware_optimizer.optimal_tier_sizes['nvme_ssd']
    )
    
    # Create compression manager
    compression_manager = MemoryCompressionManager()
    
    # Create swapping system
    swapping_system = AdvancedMemorySwapper()
    
    # Create hierarchical cache
    cache_config = CacheConfig()
    hierarchical_cache = HierarchicalCacheManager(cache_config)
    
    # Create ML predictor
    ml_predictor = LightweightMLPredictor(algorithm=PredictionAlgorithm.ENSEMBLE)
    
    # Create the integrated manager
    integrated_manager = IntegratedMemoryManager(
        tiering_system=tiering_system,
        compression_manager=compression_manager,
        swapping_system=swapping_system,
        hierarchical_cache=hierarchical_cache,
        ml_predictor=ml_predictor,
        hardware_optimizer=hardware_optimizer
    )
    
    logger.info("Integrated Memory Manager created with all systems")
    return integrated_manager


def integrate_with_qwen3_vl_model(integrated_manager: IntegratedMemoryManager):
    """
    Example of how to integrate the memory manager with a Qwen3-VL model.
    
    Args:
        integrated_manager: Instance of IntegratedMemoryManager
    """
    
    # Example tensor allocation for different model components
    def allocate_kv_cache(shape: Tuple[int, ...], dtype: torch.dtype = torch.float16) -> Tuple[torch.Tensor, str]:
        """Allocate KV cache tensor with appropriate optimizations."""
        return integrated_manager.allocate_tensor(
            shape=shape,
            dtype=dtype,
            tensor_type=TensorType.KV_CACHE,
            compress=True,  # KV cache can often be compressed
            pinned=False
        )
    
    def allocate_image_features(shape: Tuple[int, ...], dtype: torch.dtype = torch.float16) -> Tuple[torch.Tensor, str]:
        """Allocate image feature tensor with appropriate optimizations."""
        return integrated_manager.allocate_tensor(
            shape=shape,
            dtype=dtype,
            tensor_type=TensorType.IMAGE_FEATURES,
            compress=True,  # Image features can often be compressed
            pinned=False
        )
    
    def allocate_model_weights(shape: Tuple[int, ...], dtype: torch.dtype = torch.float16) -> Tuple[torch.Tensor, str]:
        """Allocate model weight tensor with appropriate optimizations."""
        return integrated_manager.allocate_tensor(
            shape=shape,
            dtype=dtype,
            tensor_type=TensorType.GENERAL,
            compress=False,  # Weights typically need full precision
            pinned=True  # Weights shouldn't be migrated
        )
    
    return {
        'allocate_kv_cache': allocate_kv_cache,
        'allocate_image_features': allocate_image_features,
        'allocate_model_weights': allocate_model_weights,
        'access_tensor': integrated_manager.access_tensor,
        'get_stats': integrated_manager.get_system_stats
    }


if __name__ == "__main__":
    print("Integration Module for Advanced Memory Tiering System")
    print("=" * 60)
    
    # Create the integrated memory manager
    integrated_manager = create_integrated_memory_manager()
    
    print(f"\n1. Created integrated memory manager with:")
    print(f"   - Memory Tiering System")
    print(f"   - Compression Manager")
    print(f"   - Swapping System")
    print(f"   - Hierarchical Cache")
    print(f"   - ML Prediction System")
    print(f"   - Hardware Optimizations")
    
    # Test tensor allocation and access
    print(f"\n2. Testing tensor allocation and access...")
    
    # Allocate a tensor
    tensor, tensor_id = integrated_manager.allocate_tensor(
        shape=(100, 100),
        dtype=torch.float16,
        tensor_type=TensorType.GENERAL,
        compress=True
    )
    print(f"   Allocated tensor with ID: {tensor_id}")
    
    # Access the tensor
    retrieved_tensor = integrated_manager.access_tensor(tensor_id)
    print(f"   Retrieved tensor: {retrieved_tensor is not None}")
    
    # Show system stats
    print(f"\n3. System statistics:")
    stats = integrated_manager.get_system_stats()
    
    print(f"   Tensor allocations: {stats['integrated_stats']['tensor_allocations']}")
    print(f"   Tensor accesses: {stats['integrated_stats']['tensor_accesses']}")
    print(f"   Compressions performed: {stats['integrated_stats']['compressions_performed']}")
    print(f"   Tier migrations: {stats['integrated_stats']['tier_migrations']}")
    print(f"   Overall cache hit rate: {stats['overall_cache_hit_rate']:.2%}")
    
    print(f"\n4. Tiering system stats:")
    tier_stats = stats['tiering_stats']
    print(f"   Global hit rate: {tier_stats['global_stats']['global_hit_rate']:.2%}")
    print(f"   Total migrations: {tier_stats['global_stats']['total_migrations']}")
    
    print(f"\nIntegration module initialized successfully!")