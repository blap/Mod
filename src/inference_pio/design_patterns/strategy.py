"""
Strategy Pattern Implementation for Inference-PIO

This module implements the Strategy pattern for selecting different optimization approaches.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from ..common.memory_manager import MemoryManager
from ..common.tensor_compression import AdaptiveTensorCompressor
from ..common.disk_offloading import DiskOffloader
from ..common.kernel_fusion import get_kernel_fusion_manager


logger = logging.getLogger(__name__)


class OptimizationStrategy(ABC):
    """
    Abstract base class for optimization strategies.
    """
    
    @abstractmethod
    def optimize(self, model: nn.Module, **kwargs) -> nn.Module:
        """
        Apply the optimization strategy to the model.
        
        Args:
            model: Model to optimize
            **kwargs: Additional optimization parameters
            
        Returns:
            Optimized model
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Get the name of the strategy.
        
        Returns:
            Strategy name
        """
        pass


class MemoryOptimizationStrategy(OptimizationStrategy):
    """
    Strategy for optimizing memory usage.
    """
    
    def __init__(self):
        self.memory_manager: Optional[MemoryManager] = None
        self.compressor: Optional[AdaptiveTensorCompressor] = None
        self.disk_offloader: Optional[DiskOffloader] = None
    
    def get_strategy_name(self) -> str:
        """
        Get the name of the strategy.
        
        Returns:
            Strategy name
        """
        return "Memory Optimization Strategy"
    
    def optimize(self, model: nn.Module, **kwargs) -> nn.Module:
        """
        Apply memory optimization strategy to the model.
        
        Args:
            model: Model to optimize
            **kwargs: Additional optimization parameters
            
        Returns:
            Optimized model
        """
        logger.info("Applying memory optimization strategy...")
        
        # Get parameters
        max_memory_ratio = kwargs.get('max_memory_ratio', 0.8)
        compression_method = kwargs.get('compression_method', 'incremental_pca')
        compression_ratio = kwargs.get('compression_ratio', 0.5)
        offload_directory = kwargs.get('offload_directory', './offload')
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(
            max_memory_ratio=max_memory_ratio,
            swap_directory=kwargs.get('swap_directory', None),
            page_size_mb=kwargs.get('page_size_mb', 16),
            eviction_policy=kwargs.get('eviction_policy', 'lru')
        )
        
        # Initialize tensor compressor
        self.compressor = AdaptiveTensorCompressor(
            compression_method=compression_method,
            base_compression_ratio=compression_ratio,
            max_components=kwargs.get('max_components', 256),
            device=kwargs.get('device', 'cpu'),
            memory_threshold_high=kwargs.get('memory_threshold_high', 0.8),
            memory_threshold_critical=kwargs.get('memory_threshold_critical', 0.9)
        )
        
        # Initialize disk offloader
        self.disk_offloader = DiskOffloader(
            max_memory_ratio=max_memory_ratio,
            offload_directory=offload_directory,
            page_size_mb=kwargs.get('page_size_mb', 16),
            eviction_policy=kwargs.get('eviction_policy', 'predictive')
        )
        
        # Apply memory optimizations
        model = self._apply_tensor_compression(model, **kwargs)
        model = self._apply_disk_offloading(model, **kwargs)
        
        logger.info("Memory optimization strategy applied successfully")
        return model
    
    def _apply_tensor_compression(self, model: nn.Module, **kwargs) -> nn.Module:
        """
        Apply tensor compression to the model.

        Args:
            model: Model to compress
            **kwargs: Additional parameters

        Returns:
            Compressed model
        """
        if not self.compressor:
            logger.warning("Tensor compressor not initialized")
            return model

        compression_ratio = kwargs.get('compression_ratio', 0.5)
        self.compressor.compression_ratio = compression_ratio

        # Compress model weights
        for name, param in model.named_parameters():
            if param.requires_grad or len(param.shape) > 1:  # Only compress trainable or multi-dimensional params
                compressed_param, metadata = self.compressor.compress_tensor(param, name)

                # Check if compression returned a tensor or a dict
                if isinstance(compressed_param, dict):
                    # If compression returns a dict, we need to decompress it back to tensor for now
                    # In a real implementation, we would need to handle compressed tensors differently
                    logger.warning(f"Compression returned dict for {name}, skipping compression for now")
                    continue
                else:
                    # Replace original parameter with compressed version
                    param.data = compressed_param

                    logger.debug(f"Compressed parameter {name}: {param.shape} -> {compressed_param.shape if hasattr(compressed_param, 'shape') else 'dict'}")

        logger.info("Tensor compression applied successfully")
        return model
    
    def _apply_disk_offloading(self, model: nn.Module, **kwargs) -> nn.Module:
        """
        Apply disk offloading to the model.
        
        Args:
            model: Model to offload
            **kwargs: Additional parameters
            
        Returns:
            Model with offloading applied
        """
        if not self.disk_offloader:
            logger.warning("Disk offloader not initialized")
            return model
        
        # Offload embedding layers
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'embedding'):
            embedding = model.transformer.embedding
            for name, param in embedding.named_parameters():
                tensor_id = f"embedding_{name}"
                self.disk_offloader.page_tensor(param.data, tensor_id)
        elif hasattr(model, 'embeddings'):
            embeddings = model.embeddings
            for name, param in embeddings.named_parameters():
                tensor_id = f"embeddings_{name}"
                self.disk_offloader.page_tensor(param.data, tensor_id)
        
        logger.info("Disk offloading applied successfully")
        return model


class ComputeOptimizationStrategy(OptimizationStrategy):
    """
    Strategy for optimizing compute performance.
    """
    
    def __init__(self):
        self.fusion_manager = None
    
    def get_strategy_name(self) -> str:
        """
        Get the name of the strategy.
        
        Returns:
            Strategy name
        """
        return "Compute Optimization Strategy"
    
    def optimize(self, model: nn.Module, **kwargs) -> nn.Module:
        """
        Apply compute optimization strategy to the model.
        
        Args:
            model: Model to optimize
            **kwargs: Additional optimization parameters
            
        Returns:
            Optimized model
        """
        logger.info("Applying compute optimization strategy...")
        
        # Get parameters
        use_custom_kernels = kwargs.get('use_custom_kernels', True)
        fallback_enabled = kwargs.get('custom_kernel_fallback_enabled', True)
        mode = kwargs.get('torch_compile_mode', 'reduce-overhead')
        fullgraph = kwargs.get('torch_compile_fullgraph', False)
        dynamic = kwargs.get('torch_compile_dynamic', True)
        
        # Apply kernel fusion optimizations
        model = self._apply_kernel_fusion(model, use_custom_kernels, fallback_enabled)
        
        # Apply torch.compile optimizations
        model = self._apply_torch_compile(model, mode, fullgraph, dynamic)
        
        logger.info("Compute optimization strategy applied successfully")
        return model
    
    def _apply_kernel_fusion(self, model: nn.Module, use_custom_kernels: bool, fallback_enabled: bool) -> nn.Module:
        """
        Apply kernel fusion optimizations to the model.
        
        Args:
            model: Model to optimize
            use_custom_kernels: Whether to use custom CUDA kernels
            fallback_enabled: Whether to enable fallback
            
        Returns:
            Optimized model
        """
        try:
            self.fusion_manager = get_kernel_fusion_manager()
            self.fusion_manager.enable_fusion()
            
            # Apply kernel fusion optimizations
            if use_custom_kernels:
                if fallback_enabled:
                    # Apply custom kernels with fallback
                    model = self.fusion_manager.apply_custom_kernels(model)
                else:
                    # Apply custom kernels without fallback
                    model = self.fusion_manager.apply_custom_kernels(model)
            
            # Apply graph fusion
            model = self.fusion_manager.fuse_model(model)
            
            logger.info("Kernel fusion optimizations applied successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to apply kernel fusion: {e}")
            return model
    
    def _apply_torch_compile(self, model: nn.Module, mode: str, fullgraph: bool, dynamic: bool) -> nn.Module:
        """
        Apply torch.compile optimizations to the model.
        
        Args:
            model: Model to optimize
            mode: Compilation mode
            fullgraph: Whether to compile the entire forward pass as a single graph
            dynamic: Whether to enable dynamic shape compilation
            
        Returns:
            Optimized model
        """
        try:
            # Enable cuDNN benchmarking for better performance
            torch.backends.cudnn.benchmark = True
            
            # Compile the model with specified optimizations
            compiled_model = torch.compile(
                model,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic
            )
            
            logger.info(f"Model optimized with torch.compile using mode: {mode}")
            return compiled_model
        except Exception as e:
            logger.error(f"Failed to apply torch.compile optimization: {e}")
            return model


class AdaptiveOptimizationStrategy(OptimizationStrategy):
    """
    Strategy for adaptively selecting optimizations based on runtime conditions.
    """
    
    def __init__(self):
        self.strategies: List[OptimizationStrategy] = [
            MemoryOptimizationStrategy(),
            ComputeOptimizationStrategy()
        ]
    
    def get_strategy_name(self) -> str:
        """
        Get the name of the strategy.
        
        Returns:
            Strategy name
        """
        return "Adaptive Optimization Strategy"
    
    def optimize(self, model: nn.Module, **kwargs) -> nn.Module:
        """
        Apply adaptive optimization strategy to the model.
        
        Args:
            model: Model to optimize
            **kwargs: Additional optimization parameters
            
        Returns:
            Optimized model
        """
        logger.info("Applying adaptive optimization strategy...")
        
        # Determine available memory
        available_memory_gb = self._get_available_memory()
        required_memory_gb = kwargs.get('required_memory_gb', 8.0)
        
        # Select strategy based on memory availability
        if available_memory_gb < required_memory_gb * 0.5:  # Less than half of required memory
            logger.info("Low memory detected, applying memory optimization strategy")
            strategy = MemoryOptimizationStrategy()
        elif available_memory_gb > required_memory_gb * 2:  # More than twice the required memory
            logger.info("High memory available, applying compute optimization strategy")
            strategy = ComputeOptimizationStrategy()
        else:
            logger.info("Moderate memory available, applying balanced optimization strategy")
            # For moderate memory, we can combine both strategies
            model = MemoryOptimizationStrategy().optimize(model, **kwargs)
            model = ComputeOptimizationStrategy().optimize(model, **kwargs)
            return model
        
        # Apply selected strategy
        optimized_model = strategy.optimize(model, **kwargs)
        
        logger.info("Adaptive optimization strategy applied successfully")
        return optimized_model
    
    def _get_available_memory(self) -> float:
        """
        Get available system memory in GB.
        
        Returns:
            Available memory in GB
        """
        import psutil
        memory = psutil.virtual_memory()
        return memory.available / (1024**3)


class OptimizationSelector:
    """
    Selector class that chooses the appropriate optimization strategy based on criteria.
    """
    
    def __init__(self):
        self.strategies: Dict[str, OptimizationStrategy] = {
            'memory': MemoryOptimizationStrategy(),
            'compute': ComputeOptimizationStrategy(),
            'adaptive': AdaptiveOptimizationStrategy()
        }
    
    def select_strategy(self, model: nn.Module, criteria: Dict[str, Any]) -> OptimizationStrategy:
        """
        Select the most appropriate optimization strategy based on criteria.
        
        Args:
            model: Model to optimize
            criteria: Criteria for strategy selection
            
        Returns:
            Selected optimization strategy
        """
        # Check if adaptive strategy is requested
        if criteria.get('adaptive', False):
            return self.strategies['adaptive']
        
        # Check memory constraints
        available_memory_gb = criteria.get('available_memory_gb', self._get_available_memory())
        required_memory_gb = criteria.get('required_memory_gb', 8.0)
        
        if available_memory_gb < required_memory_gb:
            logger.info("Selecting memory optimization strategy due to memory constraints")
            return self.strategies['memory']
        
        # Check performance requirements
        if criteria.get('performance_priority', 'balanced') == 'compute':
            logger.info("Selecting compute optimization strategy due to performance priority")
            return self.strategies['compute']
        
        # Default to adaptive strategy
        logger.info("Selecting adaptive optimization strategy")
        return self.strategies['adaptive']
    
    def optimize_with_criteria(self, model: nn.Module, criteria: Dict[str, Any]) -> nn.Module:
        """
        Optimize the model using the strategy selected based on criteria.
        
        Args:
            model: Model to optimize
            criteria: Criteria for strategy selection
            
        Returns:
            Optimized model
        """
        strategy = self.select_strategy(model, criteria)
        logger.info(f"Using strategy: {strategy.get_strategy_name()}")
        return strategy.optimize(model, **criteria)
    
    def _get_available_memory(self) -> float:
        """
        Get available system memory in GB.
        
        Returns:
            Available memory in GB
        """
        import psutil
        memory = psutil.virtual_memory()
        return memory.available / (1024**3)


__all__ = [
    'OptimizationStrategy',
    'MemoryOptimizationStrategy',
    'ComputeOptimizationStrategy',
    'AdaptiveOptimizationStrategy',
    'OptimizationSelector'
]