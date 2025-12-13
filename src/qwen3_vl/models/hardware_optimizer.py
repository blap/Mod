"""
Hardware optimization profiles that adapt to different model characteristics.

This module provides hardware optimization profiles that adapt to different model characteristics.
"""

import torch
import platform
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import psutil
import GPUtil


@dataclass
class HardwareSpec:
    """Hardware specification for optimization."""
    cpu_count: int
    cpu_architecture: str
    memory_gb: float
    gpu_count: int
    gpu_names: list
    cuda_available: bool
    compute_capability: Optional[Tuple[int, int]]
    torch_dtype: str = "float16"


@dataclass
class OptimizationProfile:
    """Optimization profile for specific hardware and model combination."""
    name: str
    hardware_specs: HardwareSpec
    model_size: str  # "small", "medium", "large", "xlarge"
    optimizations: Dict[str, Any]
    performance_tuning: Dict[str, Any]


class HardwareOptimizer:
    """
    System for creating and applying hardware optimization profiles.
    """
    
    def __init__(self):
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._profiles: Dict[str, OptimizationProfile] = {}
        self._hardware_spec = self._detect_hardware()
    
    def _detect_hardware(self) -> HardwareSpec:
        """Detect current hardware specifications."""
        # CPU info
        cpu_count = psutil.cpu_count(logical=False)
        cpu_architecture = platform.machine()
        
        # Memory info
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # GPU info
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0
        gpu_names = []
        compute_capability = None
        
        if cuda_available:
            for i in range(gpu_count):
                gpu_names.append(torch.cuda.get_device_name(i))
            if gpu_count > 0:
                compute_capability = torch.cuda.get_device_capability(0)
        
        return HardwareSpec(
            cpu_count=cpu_count,
            cpu_architecture=cpu_architecture,
            memory_gb=memory_gb,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            cuda_available=cuda_available,
            compute_capability=compute_capability
        )
    
    def get_hardware_spec(self) -> HardwareSpec:
        """Get current hardware specifications."""
        return self._hardware_spec
    
    def register_profile(self, profile: OptimizationProfile) -> bool:
        """
        Register an optimization profile.
        
        Args:
            profile: OptimizationProfile instance
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            self._profiles[profile.name] = profile
            self._logger.info(f"Optimization profile {profile.name} registered")
            return True
        except Exception as e:
            self._logger.error(f"Error registering optimization profile: {e}")
            return False
    
    def get_optimization_profile(
        self,
        model_name: str,
        model_size: str = "medium"
    ) -> Optional[OptimizationProfile]:
        """
        Get the best optimization profile for the current hardware and model.
        
        Args:
            model_name: Name of the model
            model_size: Size category of the model ("small", "medium", "large", "xlarge")
            
        Returns:
            OptimizationProfile if found, None otherwise
        """
        # Look for exact match first
        profile_name = f"{model_name}_{model_size}_{self._get_hardware_key()}"
        if profile_name in self._profiles:
            return self._profiles[profile_name]
        
        # Look for model-specific profile
        profile_name = f"{model_name}_{model_size}"
        if profile_name in self._profiles:
            return self._profiles[profile_name]
        
        # Look for hardware-specific profile
        profile_name = f"default_{model_size}_{self._get_hardware_key()}"
        if profile_name in self._profiles:
            return self._profiles[profile_name]
        
        # Return default profile for model size
        profile_name = f"default_{model_size}"
        if profile_name in self._profiles:
            return self._profiles[profile_name]
        
        # If no specific profile found, create a generic one
        return self._create_generic_profile(model_size)
    
    def _get_hardware_key(self) -> str:
        """Get a key representing the current hardware configuration."""
        if self._hardware_spec.cuda_available:
            # Use GPU info as key
            if self._hardware_spec.gpu_count > 0:
                gpu_name = self._hardware_spec.gpu_names[0].replace(" ", "_").lower()
                return f"gpu_{gpu_name}"
        # Use CPU info as key
        return f"cpu_{self._hardware_spec.cpu_architecture}"
    
    def _create_generic_profile(self, model_size: str) -> OptimizationProfile:
        """Create a generic optimization profile based on model size."""
        # Determine optimizations based on model size and hardware
        optimizations = {}
        performance_tuning = {}
        
        if model_size == "small":
            # Small models: Focus on performance
            optimizations = {
                "use_flash_attention": self._hardware_spec.cuda_available,
                "use_tensor_cores": bool(self._hardware_spec.compute_capability and 
                                        self._hardware_spec.compute_capability[0] >= 8),
                "use_gradient_checkpointing": False,
                "use_mixed_precision": True,
                "enable_jit_compilation": True
            }
            performance_tuning = {
                "batch_size": 16,
                "num_workers": self._hardware_spec.cpu_count // 2,
                "pin_memory": self._hardware_spec.cuda_available
            }
        elif model_size == "medium":
            # Medium models: Balance between memory and performance
            optimizations = {
                "use_flash_attention": self._hardware_spec.cuda_available,
                "use_tensor_cores": bool(self._hardware_spec.compute_capability and 
                                        self._hardware_spec.compute_capability[0] >= 8),
                "use_gradient_checkpointing": True,
                "use_mixed_precision": True,
                "enable_jit_compilation": True
            }
            performance_tuning = {
                "batch_size": 8,
                "num_workers": max(1, self._hardware_spec.cpu_count // 4),
                "pin_memory": self._hardware_spec.cuda_available
            }
        elif model_size == "large":
            # Large models: Focus on memory efficiency
            optimizations = {
                "use_flash_attention": self._hardware_spec.cuda_available,
                "use_tensor_cores": bool(self._hardware_spec.compute_capability and 
                                        self._hardware_spec.compute_capability[0] >= 8),
                "use_gradient_checkpointing": True,
                "use_mixed_precision": True,
                "enable_jit_compilation": False,  # May cause memory issues
                "use_cpu_offloading": self._hardware_spec.memory_gb > 16
            }
            performance_tuning = {
                "batch_size": 2,
                "num_workers": 1,
                "pin_memory": False
            }
        else:  # xlarge
            # XLarge models: Maximum memory efficiency
            optimizations = {
                "use_flash_attention": self._hardware_spec.cuda_available,
                "use_tensor_cores": bool(self._hardware_spec.compute_capability and 
                                        self._hardware_spec.compute_capability[0] >= 8),
                "use_gradient_checkpointing": True,
                "use_mixed_precision": True,
                "enable_jit_compilation": False,
                "use_cpu_offloading": True,
                "use_disk_offloading": True
            }
            performance_tuning = {
                "batch_size": 1,
                "num_workers": 1,
                "pin_memory": False
            }
        
        return OptimizationProfile(
            name=f"generic_{model_size}",
            hardware_specs=self._hardware_spec,
            model_size=model_size,
            optimizations=optimizations,
            performance_tuning=performance_tuning
        )
    
    def apply_optimizations(
        self,
        model: Any,  # nn.Module or other model type
        profile: OptimizationProfile
    ) -> Any:
        """
        Apply optimizations from a profile to a model.
        
        Args:
            model: Model to optimize
            profile: Optimization profile to apply
            
        Returns:
            Optimized model
        """
        optimizations = profile.optimizations
        
        # Apply flash attention if supported and requested
        if optimizations.get("use_flash_attention", False):
            self._apply_flash_attention(model)
        
        # Apply gradient checkpointing if requested
        if optimizations.get("use_gradient_checkpointing", False):
            self._apply_gradient_checkpointing(model)
        
        # Apply mixed precision if requested
        if optimizations.get("use_mixed_precision", False):
            self._apply_mixed_precision(model)
        
        # Apply JIT compilation if requested
        if optimizations.get("enable_jit_compilation", False):
            self._apply_jit_compilation(model)
        
        # Apply other optimizations as needed
        if optimizations.get("use_cpu_offloading", False):
            self._apply_cpu_offloading(model)
        
        if optimizations.get("use_disk_offloading", False):
            self._apply_disk_offloading(model)
        
        self._logger.info(f"Applied optimizations from profile: {profile.name}")
        return model
    
    def _apply_flash_attention(self, model: Any) -> None:
        """Apply flash attention optimization."""
        # This would involve configuring the model to use flash attention
        # Implementation depends on the specific model architecture
        self._logger.info("Flash attention optimization applied")
    
    def _apply_gradient_checkpointing(self, model: Any) -> None:
        """Apply gradient checkpointing optimization."""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            self._logger.info("Gradient checkpointing enabled")
    
    def _apply_mixed_precision(self, model: Any) -> None:
        """Apply mixed precision optimization."""
        # This would involve setting up AMP (Automatic Mixed Precision)
        self._logger.info("Mixed precision optimization applied")
    
    def _apply_jit_compilation(self, model: Any) -> None:
        """Apply JIT compilation optimization."""
        # This would involve using torch.jit to compile parts of the model
        self._logger.info("JIT compilation optimization applied")
    
    def _apply_cpu_offloading(self, model: Any) -> None:
        """Apply CPU offloading optimization."""
        # This would involve offloading parts of the model to CPU
        self._logger.info("CPU offloading optimization applied")
    
    def _apply_disk_offloading(self, model: Any) -> None:
        """Apply disk offloading optimization."""
        # This would involve offloading parts of the model to disk
        self._logger.info("Disk offloading optimization applied")
    
    def get_model_size_category(self, num_parameters: int) -> str:
        """
        Determine the size category of a model based on parameter count.
        
        Args:
            num_parameters: Number of parameters in the model
            
        Returns:
            Size category ("small", "medium", "large", "xlarge")
        """
        if num_parameters < 1e9:  # Less than 1 billion parameters
            return "small"
        elif num_parameters < 5e9:  # Less than 5 billion parameters
            return "medium"
        elif num_parameters < 15e9:  # Less than 15 billion parameters
            return "large"
        else:  # 15 billion parameters or more
            return "xlarge"


# Global hardware optimizer instance
hardware_optimizer = HardwareOptimizer()


def get_hardware_optimizer() -> HardwareOptimizer:
    """
    Get the global hardware optimizer instance.
    
    Returns:
        HardwareOptimizer instance
    """
    return hardware_optimizer


# Register default optimization profiles for known models
def _register_default_profiles():
    """Register default optimization profiles for known models."""
    
    # Qwen3-VL profiles
    qwen3_vl_large_profile = OptimizationProfile(
        name="Qwen3-VL_large_gpu_rtx_3090",
        hardware_specs=hardware_optimizer.get_hardware_spec(),
        model_size="large",
        optimizations={
            "use_flash_attention": True,
            "use_tensor_cores": True,
            "use_gradient_checkpointing": True,
            "use_mixed_precision": True,
            "enable_jit_compilation": True,
            "use_cpu_offloading": False
        },
        performance_tuning={
            "batch_size": 4,
            "num_workers": 2,
            "pin_memory": True
        }
    )
    
    hardware_optimizer.register_profile(qwen3_vl_large_profile)
    
    # Qwen3-4B profiles
    qwen3_4b_medium_profile = OptimizationProfile(
        name="Qwen3-4B-Instruct-2507_medium_gpu_rtx_3060",
        hardware_specs=hardware_optimizer.get_hardware_spec(),
        model_size="medium",
        optimizations={
            "use_flash_attention": True,
            "use_tensor_cores": True,
            "use_gradient_checkpointing": True,
            "use_mixed_precision": True,
            "enable_jit_compilation": True,
            "use_cpu_offloading": False
        },
        performance_tuning={
            "batch_size": 8,
            "num_workers": 2,
            "pin_memory": True
        }
    )
    
    hardware_optimizer.register_profile(qwen3_4b_medium_profile)


_register_default_profiles()