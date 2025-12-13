"""
Performance optimization that scales based on model size.

This module provides performance optimization that can scale based on model size.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import time
from enum import Enum


class PerformanceLevel(Enum):
    """Performance levels based on model size and hardware."""
    LOW = "low"  # Small models or limited hardware
    MEDIUM = "medium"  # Medium models or standard hardware
    HIGH = "high"  # Large models or high-end hardware
    MAXIMUM = "maximum"  # XLarge models or specialized hardware


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    performance_level: PerformanceLevel
    batch_size: int
    num_workers: int
    pin_memory: bool
    use_amp: bool  # Automatic Mixed Precision
    use_jit: bool  # Just-In-Time compilation
    use_cache: bool
    max_length: int
    gradient_accumulation_steps: int = 1
    use_gradient_checkpointing: bool = False


class PerformanceOptimizer:
    """
    System for optimizing performance based on model size and hardware capabilities.
    """
    
    def __init__(self):
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._performance_configs: Dict[str, PerformanceConfig] = {}
    
    def register_performance_config(self, model_name: str, config: PerformanceConfig) -> bool:
        """
        Register a performance configuration for a model.
        
        Args:
            model_name: Name of the model
            config: PerformanceConfig instance
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            self._performance_configs[model_name] = config
            self._logger.info(f"Performance config registered for {model_name}")
            return True
        except Exception as e:
            self._logger.error(f"Error registering performance config for {model_name}: {e}")
            return False
    
    def get_performance_config(self, model_name: str) -> Optional[PerformanceConfig]:
        """
        Get performance configuration for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            PerformanceConfig if found, None otherwise
        """
        return self._performance_configs.get(model_name)
    
    def calculate_performance_config(
        self,
        model_size_params: int,
        available_memory_gb: float,
        num_gpus: int = 1
    ) -> PerformanceConfig:
        """
        Calculate optimal performance configuration based on model size and hardware.
        
        Args:
            model_size_params: Number of parameters in the model
            available_memory_gb: Available memory in GB
            num_gpus: Number of available GPUs
            
        Returns:
            PerformanceConfig instance
        """
        # Determine performance level based on model size and memory
        if model_size_params < 1e9:  # Less than 1 billion parameters
            level = PerformanceLevel.LOW
            base_batch_size = 16
            max_length = 512
        elif model_size_params < 5e9:  # Less than 5 billion parameters
            level = PerformanceLevel.MEDIUM
            base_batch_size = 8
            max_length = 1024
        elif model_size_params < 15e9:  # Less than 15 billion parameters
            level = PerformanceLevel.HIGH
            base_batch_size = 4
            max_length = 2048
        else:  # 15 billion parameters or more
            level = PerformanceLevel.MAXIMUM
            base_batch_size = 1
            max_length = 4096
        
        # Adjust batch size based on available memory
        memory_factor = available_memory_gb / 8.0  # Normalize to 8GB baseline
        adjusted_batch_size = int(base_batch_size * memory_factor)
        adjusted_batch_size = max(1, min(adjusted_batch_size, base_batch_size * 4))  # Cap at 4x
        
        # Determine other parameters based on level and hardware
        use_amp = available_memory_gb >= 6.0
        use_jit = level in [PerformanceLevel.LOW, PerformanceLevel.MEDIUM] and available_memory_gb >= 8.0
        pin_memory = num_gpus > 0
        use_cache = level != PerformanceLevel.MAXIMUM  # Disable cache for largest models to save memory
        gradient_accumulation_steps = 1
        use_gradient_checkpointing = level in [PerformanceLevel.HIGH, PerformanceLevel.MAXIMUM]
        
        # Adjust for multi-GPU
        if num_gpus > 1:
            adjusted_batch_size = adjusted_batch_size * num_gpus
            gradient_accumulation_steps = max(1, gradient_accumulation_steps // num_gpus)
        
        config = PerformanceConfig(
            performance_level=level,
            batch_size=adjusted_batch_size,
            num_workers=min(8, adjusted_batch_size),  # Cap workers to avoid overhead
            pin_memory=pin_memory,
            use_amp=use_amp,
            use_jit=use_jit,
            use_cache=use_cache,
            max_length=max_length,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        return config
    
    def apply_performance_config(
        self,
        model: nn.Module,
        config: PerformanceConfig
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Apply performance configuration to a model.
        
        Args:
            model: Model to optimize
            config: PerformanceConfig to apply
            
        Returns:
            Tuple of (optimized model, performance settings dict)
        """
        # Apply gradient checkpointing if enabled
        if config.use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            self._logger.info("Gradient checkpointing enabled")
        
        # Prepare performance settings
        settings = {
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
            "use_amp": config.use_amp,
            "use_jit": config.use_jit,
            "use_cache": config.use_cache,
            "max_length": config.max_length,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "use_gradient_checkpointing": config.use_gradient_checkpointing
        }
        
        # Apply JIT compilation if enabled
        if config.use_jit:
            try:
                # Try to compile the forward method
                model.forward = torch.jit.script(model.forward)
                self._logger.info("JIT compilation applied")
            except Exception as e:
                self._logger.warning(f"JIT compilation failed: {e}")
                # Fallback: don't apply JIT if it fails
                settings["use_jit"] = False
        
        # Apply cache settings if applicable
        if hasattr(model, 'config'):
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = config.use_cache
        
        return model, settings
    
    def benchmark_model(
        self,
        model: nn.Module,
        input_data: Any,
        num_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark a model's performance.
        
        Args:
            model: Model to benchmark
            input_data: Input data for the model
            num_iterations: Number of iterations to run
            
        Returns:
            Dictionary with performance metrics
        """
        model.eval()
        
        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = model(**input_data) if isinstance(input_data, dict) else model(input_data)
        
        # Benchmark
        start_time = time.time()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        for i in range(num_iterations):
            with torch.no_grad():
                _ = model(**input_data) if isinstance(input_data, dict) else model(input_data)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = num_iterations / total_time
        
        metrics = {
            "total_time_seconds": total_time,
            "average_time_per_iteration": avg_time,
            "throughput_iterations_per_second": throughput,
            "num_iterations": num_iterations
        }
        
        self._logger.info(f"Benchmark results: {metrics}")
        return metrics
    
    def optimize_for_inference(
        self,
        model: nn.Module,
        config: Optional[PerformanceConfig] = None
    ) -> nn.Module:
        """
        Optimize model specifically for inference.
        
        Args:
            model: Model to optimize
            config: PerformanceConfig to use (optional)
            
        Returns:
            Optimized model
        """
        model.eval()
        
        if config is None:
            # Create a default inference config
            config = PerformanceConfig(
                performance_level=PerformanceLevel.MEDIUM,
                batch_size=1,
                num_workers=1,
                pin_memory=torch.cuda.is_available(),
                use_amp=True,
                use_jit=True,
                use_cache=True,
                max_length=2048
            )
        
        # Apply inference-specific optimizations
        optimized_model = model
        
        # Disable gradients for inference
        for param in optimized_model.parameters():
            param.requires_grad = False
        
        # Apply JIT if requested and possible
        if config.use_jit:
            try:
                optimized_model = torch.jit.trace(optimized_model, 
                                                torch.randn(1, config.max_length))
                self._logger.info("Model traced for inference optimization")
            except Exception as e:
                self._logger.warning(f"Model tracing failed: {e}")
        
        return optimized_model
    
    def get_scaled_performance_params(
        self,
        model_size_params: int,
        target_latency: Optional[float] = None,
        target_throughput: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get performance parameters scaled to model size and targets.
        
        Args:
            model_size_params: Number of parameters in the model
            target_latency: Target latency in seconds (optional)
            target_throughput: Target throughput in requests/second (optional)
            
        Returns:
            Dictionary with scaled performance parameters
        """
        # Base parameters based on model size
        if model_size_params < 1e9:
            base_batch_size = 32
            base_workers = 4
        elif model_size_params < 5e9:
            base_batch_size = 16
            base_workers = 4
        elif model_size_params < 15e9:
            base_batch_size = 8
            base_workers = 2
        else:
            base_batch_size = 1
            base_workers = 1
        
        # Adjust based on targets
        if target_latency is not None and target_latency < 0.1:  # Very low latency required
            base_batch_size = max(1, base_batch_size // 4)
            base_workers = max(1, base_workers // 2)
        elif target_throughput is not None and target_throughput > 100:  # High throughput required
            base_batch_size = min(64, base_batch_size * 2)
            base_workers = min(8, base_workers * 2)
        
        return {
            "batch_size": base_batch_size,
            "num_workers": base_workers,
            "max_concurrent_requests": base_batch_size * 2,  # Allow some buffering
            "prefetch_factor": 2 if base_workers > 1 else None
        }


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()


def get_performance_optimizer() -> PerformanceOptimizer:
    """
    Get the global performance optimizer instance.
    
    Returns:
        PerformanceOptimizer instance
    """
    return performance_optimizer


# Register default performance configurations for known models
def _register_default_performance_configs():
    """Register default performance configurations for known models."""
    
    # Qwen3-VL performance config
    qwen3_vl_config = PerformanceConfig(
        performance_level=PerformanceLevel.HIGH,
        batch_size=4,
        num_workers=2,
        pin_memory=True,
        use_amp=True,
        use_jit=True,
        use_cache=True,
        max_length=32768,
        gradient_accumulation_steps=1,
        use_gradient_checkpointing=True
    )
    
    performance_optimizer.register_performance_config("Qwen3-VL", qwen3_vl_config)
    
    # Qwen3-4B performance config
    qwen3_4b_config = PerformanceConfig(
        performance_level=PerformanceLevel.MEDIUM,
        batch_size=8,
        num_workers=2,
        pin_memory=True,
        use_amp=True,
        use_jit=True,
        use_cache=True,
        max_length=32768,
        gradient_accumulation_steps=1,
        use_gradient_checkpointing=True
    )
    
    performance_optimizer.register_performance_config("Qwen3-4B-Instruct-2507", qwen3_4b_config)


_register_default_performance_configs()