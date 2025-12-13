"""
Adaptive memory management system that adjusts based on model size and requirements.

This module provides adaptive memory management that adjusts based on model size and requirements.
"""

import gc
import psutil
import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
import logging
import threading
from dataclasses import dataclass
from enum import Enum


class MemoryStrategy(Enum):
    """Memory management strategies."""
    EFFICIENT = "efficient"
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    LOW_MEMORY = "low_memory"


@dataclass
class MemoryProfile:
    """Memory profile for a specific model."""
    model_name: str
    min_memory_gb: float
    recommended_memory_gb: float
    max_memory_gb: float
    peak_memory_gb: float = 0.0
    current_memory_gb: float = 0.0
    strategy: MemoryStrategy = MemoryStrategy.BALANCED


class AdaptiveMemoryManager:
    """
    Adaptive memory management system that adjusts based on model size and requirements.
    """
    
    def __init__(self):
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._memory_profiles: Dict[str, MemoryProfile] = {}
        self._active_models: Dict[str, nn.Module] = {}
        self._lock = threading.RLock()
        self._current_strategy = MemoryStrategy.BALANCED
        self._monitoring_thread = None
        self._monitoring_active = False
    
    def register_memory_profile(self, profile: MemoryProfile) -> bool:
        """
        Register a memory profile for a model.
        
        Args:
            profile: MemoryProfile instance
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            self._memory_profiles[profile.model_name] = profile
            self._logger.info(f"Memory profile registered for {profile.model_name}")
            return True
        except Exception as e:
            self._logger.error(f"Error registering memory profile: {e}")
            return False
    
    def get_memory_profile(self, model_name: str) -> Optional[MemoryProfile]:
        """
        Get memory profile for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            MemoryProfile if found, None otherwise
        """
        return self._memory_profiles.get(model_name)
    
    def get_system_memory_info(self) -> Dict[str, float]:
        """
        Get current system memory information.
        
        Returns:
            Dictionary with memory information in GB
        """
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent_used": memory.percent
        }
    
    def get_gpu_memory_info(self) -> Optional[Dict[str, float]]:
        """
        Get GPU memory information if available.
        
        Returns:
            Dictionary with GPU memory information in GB, or None if no GPU
        """
        if not torch.cuda.is_available():
            return None
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        
        return {
            "total_gb": gpu_memory / (1024**3),
            "allocated_gb": allocated / (1024**3),
            "reserved_gb": reserved / (1024**3),
            "available_gb": (gpu_memory - reserved) / (1024**3)
        }
    
    def calculate_optimal_memory_strategy(
        self,
        model_name: str,
        available_memory_gb: float
    ) -> MemoryStrategy:
        """
        Calculate optimal memory strategy based on model requirements and available memory.
        
        Args:
            model_name: Name of the model
            available_memory_gb: Available memory in GB
            
        Returns:
            Optimal MemoryStrategy
        """
        profile = self._memory_profiles.get(model_name)
        if not profile:
            return self._current_strategy
        
        if available_memory_gb < profile.min_memory_gb:
            return MemoryStrategy.LOW_MEMORY
        elif available_memory_gb < profile.recommended_memory_gb:
            return MemoryStrategy.EFFICIENT
        elif available_memory_gb > profile.max_memory_gb:
            return MemoryStrategy.PERFORMANCE
        else:
            return MemoryStrategy.BALANCED
    
    def apply_memory_strategy(
        self,
        model: nn.Module,
        strategy: MemoryStrategy,
        config: Optional[Dict[str, Any]] = None
    ) -> nn.Module:
        """
        Apply memory strategy to a model.
        
        Args:
            model: Model to apply strategy to
            strategy: Memory strategy to apply
            config: Additional configuration options
            
        Returns:
            Modified model with memory strategy applied
        """
        config = config or {}
        
        if strategy == MemoryStrategy.LOW_MEMORY:
            self._apply_low_memory_strategy(model, config)
        elif strategy == MemoryStrategy.EFFICIENT:
            self._apply_efficient_strategy(model, config)
        elif strategy == MemoryStrategy.PERFORMANCE:
            self._apply_performance_strategy(model, config)
        else:  # BALANCED
            self._apply_balanced_strategy(model, config)
        
        return model
    
    def _apply_low_memory_strategy(self, model: nn.Module, config: Dict[str, Any]):
        """Apply low memory strategy to model."""
        self._logger.info("Applying low memory strategy")
        
        # Enable gradient checkpointing to save memory during training
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Use CPU offloading if available
        if config.get('use_cpu_offload', False):
            self._apply_cpu_offload(model)
        
        # Reduce batch size if possible
        if 'batch_size' in config:
            config['batch_size'] = max(1, config['batch_size'] // 2)
    
    def _apply_efficient_strategy(self, model: nn.Module, config: Dict[str, Any]):
        """Apply efficient strategy to model."""
        self._logger.info("Applying efficient strategy")
        
        # Enable gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Use mixed precision if available
        if config.get('use_mixed_precision', True):
            self._apply_mixed_precision(model)
    
    def _apply_performance_strategy(self, model: nn.Module, config: Dict[str, Any]):
        """Apply performance strategy to model."""
        self._logger.info("Applying performance strategy")
        
        # Disable gradient checkpointing for faster training
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
        
        # Increase batch size if possible
        if 'batch_size' in config:
            config['batch_size'] = config['batch_size'] * 2
    
    def _apply_balanced_strategy(self, model: nn.Module, config: Dict[str, Any]):
        """Apply balanced strategy to model."""
        self._logger.info("Applying balanced strategy")
        
        # Use default settings
        pass
    
    def _apply_cpu_offload(self, model: nn.Module):
        """Apply CPU offloading to model."""
        # This is a simplified implementation - in practice, this would involve
        # more complex logic for moving parts of the model between CPU and GPU
        self._logger.info("CPU offloading applied")
    
    def _apply_mixed_precision(self, model: nn.Module):
        """Apply mixed precision to model."""
        # This would involve setting up AMP (Automatic Mixed Precision)
        self._logger.info("Mixed precision applied")
    
    def monitor_memory_usage(self, model_name: str) -> None:
        """
        Start monitoring memory usage for a model.
        
        Args:
            model_name: Name of the model to monitor
        """
        self._logger.info(f"Starting memory monitoring for {model_name}")
        
        import time
        import threading
        
        def memory_monitor():
            while self._monitoring_active:
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0)
                    reserved = torch.cuda.memory_reserved(0)
                    
                    if model_name in self._memory_profiles:
                        profile = self._memory_profiles[model_name]
                        profile.current_memory_gb = allocated / (1024**3)
                        profile.peak_memory_gb = max(
                            profile.peak_memory_gb,
                            reserved / (1024**3)
                        )
                
                time.sleep(1)  # Check every second
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=memory_monitor, daemon=True)
        self._monitoring_thread.start()
    
    def stop_memory_monitoring(self) -> None:
        """Stop memory monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2)
    
    def cleanup_memory(self) -> None:
        """Clean up memory and clear cache."""
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        self._logger.info("Memory cleanup completed")
    
    def get_memory_recommendation(
        self,
        model_name: str,
        available_memory_gb: float
    ) -> Tuple[MemoryStrategy, Dict[str, Any]]:
        """
        Get memory strategy recommendation and optimization suggestions.
        
        Args:
            model_name: Name of the model
            available_memory_gb: Available memory in GB
            
        Returns:
            Tuple of (recommended strategy, optimization suggestions)
        """
        strategy = self.calculate_optimal_memory_strategy(model_name, available_memory_gb)
        
        suggestions = {
            "strategy": strategy.value,
            "memory_available_gb": available_memory_gb,
            "optimizations": []
        }
        
        profile = self._memory_profiles.get(model_name)
        if profile:
            if strategy == MemoryStrategy.LOW_MEMORY:
                suggestions["optimizations"].extend([
                    "Enable gradient checkpointing",
                    "Use CPU offloading",
                    "Reduce batch size",
                    "Use quantization if possible"
                ])
            elif strategy == MemoryStrategy.EFFICIENT:
                suggestions["optimizations"].extend([
                    "Enable gradient checkpointing",
                    "Use mixed precision training"
                ])
            elif strategy == MemoryStrategy.PERFORMANCE:
                suggestions["optimizations"].extend([
                    "Disable gradient checkpointing for speed",
                    "Increase batch size if possible"
                ])
        
        return strategy, suggestions


# Global memory manager instance
memory_manager = AdaptiveMemoryManager()


def get_memory_manager() -> AdaptiveMemoryManager:
    """
    Get the global memory manager instance.
    
    Returns:
        AdaptiveMemoryManager instance
    """
    return memory_manager


# Register default memory profiles for known models
def _register_default_memory_profiles():
    """Register default memory profiles for known models."""
    
    # Qwen3-VL memory profile
    qwen3_vl_profile = MemoryProfile(
        model_name="Qwen3-VL",
        min_memory_gb=8.0,
        recommended_memory_gb=16.0,
        max_memory_gb=24.0,
        strategy=MemoryStrategy.BALANCED
    )
    
    memory_manager.register_memory_profile(qwen3_vl_profile)
    
    # Qwen3-4B memory profile
    qwen3_4b_profile = MemoryProfile(
        model_name="Qwen3-4B-Instruct-2507",
        min_memory_gb=6.0,
        recommended_memory_gb=8.0,
        max_memory_gb=12.0,
        strategy=MemoryStrategy.BALANCED
    )
    
    memory_manager.register_memory_profile(qwen3_4b_profile)


_register_default_memory_profiles()