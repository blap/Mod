"""
Optimization Configuration System for Inference-PIO

This module provides configuration classes and utilities for managing
optimization settings across different models.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
from .optimization_manager import OptimizationConfig, OptimizationType


class ModelFamily(Enum):
    """Enumeration of supported model families."""
    GLM = "glm"
    QWEN = "qwen"
    LLAMA = "llama"
    MISTRAL = "mistral"
    OTHER = "other"


@dataclass
class ModelOptimizationConfig:
    """Configuration for optimizations for a specific model."""
    model_family: ModelFamily
    optimizations: List[OptimizationConfig]
    default_enabled: bool = True
    priority_order: List[str] = None  # Order in which optimizations should be applied
    
    def __post_init__(self):
        if self.priority_order is None:
            self.priority_order = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_family": self.model_family.value,
            "default_enabled": self.default_enabled,
            "priority_order": self.priority_order,
            "optimizations": [asdict(opt) for opt in self.optimizations]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelOptimizationConfig':
        """Create from dictionary representation."""
        optimizations = [
            OptimizationConfig(**opt_data) 
            for opt_data in data["optimizations"]
        ]
        
        return cls(
            model_family=ModelFamily(data["model_family"]),
            optimizations=optimizations,
            default_enabled=data.get("default_enabled", True),
            priority_order=data.get("priority_order", [])
        )


@dataclass
class GlobalOptimizationProfile:
    """Global profile for optimization settings across all models."""
    name: str
    description: str
    default_settings: Dict[str, bool]  # optimization_name -> enabled
    performance_targets: Dict[str, float]  # metric -> target_value
    resource_constraints: Dict[str, Any]  # memory, compute, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "default_settings": self.default_settings,
            "performance_targets": self.performance_targets,
            "resource_constraints": self.resource_constraints
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GlobalOptimizationProfile':
        """Create from dictionary representation."""
        return cls(
            name=data["name"],
            description=data["description"],
            default_settings=data["default_settings"],
            performance_targets=data["performance_targets"],
            resource_constraints=data["resource_constraints"]
        )


class OptimizationConfigManager:
    """Manager for optimization configurations."""
    
    def __init__(self):
        self.model_configs: Dict[ModelFamily, ModelOptimizationConfig] = {}
        self.global_profiles: Dict[str, GlobalOptimizationProfile] = {}
        self.active_profile: Optional[str] = None
    
    def register_model_config(self, config: ModelOptimizationConfig):
        """Register a model-specific optimization configuration."""
        self.model_configs[config.model_family] = config
    
    def get_model_config(self, model_family: ModelFamily) -> Optional[ModelOptimizationConfig]:
        """Get configuration for a specific model family."""
        return self.model_configs.get(model_family)
    
    def register_global_profile(self, profile: GlobalOptimizationProfile):
        """Register a global optimization profile."""
        self.global_profiles[profile.name] = profile
    
    def get_global_profile(self, profile_name: str) -> Optional[GlobalOptimizationProfile]:
        """Get a global optimization profile."""
        return self.global_profiles.get(profile_name)
    
    def set_active_profile(self, profile_name: str) -> bool:
        """Set the active global profile."""
        if profile_name in self.global_profiles:
            self.active_profile = profile_name
            return True
        return False
    
    def get_active_profile(self) -> Optional[GlobalOptimizationProfile]:
        """Get the active global profile."""
        if self.active_profile:
            return self.global_profiles.get(self.active_profile)
        return None
    
    def apply_profile_to_model_config(self, model_family: ModelFamily) -> ModelOptimizationConfig:
        """Apply the active profile settings to a model configuration."""
        model_config = self.get_model_config(model_family)
        if not model_config:
            return None
        
        active_profile = self.get_active_profile()
        if not active_profile:
            return model_config
        
        # Update optimization settings based on profile
        for opt in model_config.optimizations:
            if opt.name in active_profile.default_settings:
                opt.enabled = active_profile.default_settings[opt.name]
        
        return model_config
    
    def save_model_config(self, model_family: ModelFamily, filepath: str, format: str = "json"):
        """Save model configuration to file."""
        config = self.get_model_config(model_family)
        if not config:
            raise ValueError(f"No configuration found for model family: {model_family}")
        
        data = config.to_dict()
        
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format.lower() == "yaml":
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_model_config(self, filepath: str, format: str = "json") -> ModelOptimizationConfig:
        """Load model configuration from file."""
        if format.lower() == "json":
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif format.lower() == "yaml":
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        config = ModelOptimizationConfig.from_dict(data)
        self.register_model_config(config)
        return config
    
    def save_global_profile(self, profile_name: str, filepath: str, format: str = "json"):
        """Save global profile to file."""
        profile = self.get_global_profile(profile_name)
        if not profile:
            raise ValueError(f"No profile found with name: {profile_name}")
        
        data = profile.to_dict()
        
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format.lower() == "yaml":
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_global_profile(self, filepath: str, format: str = "json") -> GlobalOptimizationProfile:
        """Load global profile from file."""
        if format.lower() == "json":
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif format.lower() == "yaml":
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        profile = GlobalOptimizationProfile.from_dict(data)
        self.register_global_profile(profile)
        return profile


# Predefined optimization profiles
def create_balanced_profile() -> GlobalOptimizationProfile:
    """Create a balanced optimization profile."""
    return GlobalOptimizationProfile(
        name="balanced",
        description="Balanced optimization profile focusing on both performance and memory efficiency",
        default_settings={
            "flash_attention": True,
            "sparse_attention": True,
            "adaptive_sparse_attention": True,
            "disk_offloading": True,
            "activation_offloading": True,
            "tensor_compression": True,
            "structured_pruning": False,  # Conservative default
            "tensor_decomposition": False,  # Conservative default
            "snn": False,  # Experimental
            "kernel_fusion": True,
            "distributed_simulation": False,  # Only when needed
            "adaptive_batching": True,
        },
        performance_targets={
            "latency_reduction": 0.2,  # 20% reduction target
            "memory_efficiency": 0.3,  # 30% memory reduction target
            "throughput_improvement": 0.15  # 15% throughput improvement target
        },
        resource_constraints={
            "max_memory_usage_ratio": 0.8,
            "min_compute_capability": 6.0,
            "prefer_gpu_over_cpu": True
        }
    )


def create_performance_profile() -> GlobalOptimizationProfile:
    """Create a performance-focused optimization profile."""
    return GlobalOptimizationProfile(
        name="performance",
        description="Performance-focused optimization profile prioritizing speed over memory usage",
        default_settings={
            "flash_attention": True,
            "sparse_attention": True,
            "adaptive_sparse_attention": True,
            "disk_offloading": False,  # Minimize disk I/O
            "activation_offloading": False,  # Keep activations in memory
            "tensor_compression": True,
            "structured_pruning": True,
            "tensor_decomposition": True,
            "snn": False,  # Experimental
            "kernel_fusion": True,
            "distributed_simulation": True,  # Use all available resources
            "adaptive_batching": True,
        },
        performance_targets={
            "latency_reduction": 0.35,  # 35% reduction target
            "memory_efficiency": 0.1,  # Lower memory target for performance
            "throughput_improvement": 0.25  # 25% throughput improvement target
        },
        resource_constraints={
            "max_memory_usage_ratio": 0.9,
            "min_compute_capability": 7.0,
            "prefer_gpu_over_cpu": True
        }
    )


def create_memory_efficient_profile() -> GlobalOptimizationProfile:
    """Create a memory-efficient optimization profile."""
    return GlobalOptimizationProfile(
        name="memory_efficient",
        description="Memory-efficient optimization profile prioritizing memory usage over speed",
        default_settings={
            "flash_attention": True,
            "sparse_attention": True,
            "adaptive_sparse_attention": True,
            "disk_offloading": True,
            "activation_offloading": True,
            "tensor_compression": True,
            "structured_pruning": True,
            "tensor_decomposition": True,
            "snn": True,  # For energy efficiency
            "kernel_fusion": True,
            "distributed_simulation": False,  # Conserve memory
            "adaptive_batching": True,
        },
        performance_targets={
            "latency_reduction": 0.1,  # Lower priority
            "memory_efficiency": 0.5,  # 50% memory reduction target
            "throughput_improvement": 0.05  # Lower priority
        },
        resource_constraints={
            "max_memory_usage_ratio": 0.6,
            "min_compute_capability": 5.0,
            "prefer_gpu_over_cpu": False  # Allow CPU fallback
        }
    )


def create_experimental_profile() -> GlobalOptimizationProfile:
    """Create an experimental optimization profile."""
    return GlobalOptimizationProfile(
        name="experimental",
        description="Experimental optimization profile with cutting-edge techniques",
        default_settings={
            "flash_attention": True,
            "sparse_attention": True,
            "adaptive_sparse_attention": True,
            "disk_offloading": True,
            "activation_offloading": True,
            "tensor_compression": True,
            "structured_pruning": True,
            "tensor_decomposition": True,
            "snn": True,  # Enable experimental SNN
            "kernel_fusion": True,
            "distributed_simulation": True,
            "adaptive_batching": True,
        },
        performance_targets={
            "latency_reduction": 0.3,
            "memory_efficiency": 0.4,
            "throughput_improvement": 0.3,
            "energy_efficiency": 0.35  # Focus on energy efficiency
        },
        resource_constraints={
            "max_memory_usage_ratio": 0.75,
            "min_compute_capability": 7.0,
            "prefer_gpu_over_cpu": True
        }
    )


# Global configuration manager instance
config_manager = OptimizationConfigManager()


def get_config_manager() -> OptimizationConfigManager:
    """Get the global configuration manager instance."""
    return config_manager


# Register default profiles
def register_default_profiles():
    """Register default optimization profiles."""
    manager = get_config_manager()
    
    profiles = [
        create_balanced_profile(),
        create_performance_profile(),
        create_memory_efficient_profile(),
        create_experimental_profile(),
    ]
    
    for profile in profiles:
        manager.register_global_profile(profile)


register_default_profiles()


# Predefined model configurations
def create_glm_optimization_config() -> ModelOptimizationConfig:
    """Create optimization configuration for GLM models."""
    return ModelOptimizationConfig(
        model_family=ModelFamily.GLM,
        optimizations=[
            OptimizationConfig(
                name="flash_attention",
                enabled=True,
                optimization_type=OptimizationType.ATTENTION,
                priority=10,
                parameters={"use_triton": True}
            ),
            OptimizationConfig(
                name="sparse_attention",
                enabled=True,
                optimization_type=OptimizationType.ATTENTION,
                priority=15,
                parameters={"sparsity_ratio": 0.25}
            ),
            OptimizationConfig(
                name="adaptive_sparse_attention",
                enabled=True,
                optimization_type=OptimizationType.ATTENTION,
                priority=20,
                parameters={"adaptive_strategy": "input_dependent"}
            ),
            OptimizationConfig(
                name="disk_offloading",
                enabled=True,
                optimization_type=OptimizationType.MEMORY,
                priority=30,
                parameters={"max_memory_ratio": 0.8}
            ),
            OptimizationConfig(
                name="activation_offloading",
                enabled=True,
                optimization_type=OptimizationType.MEMORY,
                priority=35,
                parameters={"max_memory_ratio": 0.7}
            ),
            OptimizationConfig(
                name="tensor_compression",
                enabled=True,
                optimization_type=OptimizationType.COMPUTE,
                priority=40,
                parameters={"compression_ratio": 0.5}
            ),
            OptimizationConfig(
                name="structured_pruning",
                enabled=False,
                optimization_type=OptimizationType.MODEL_STRUCTURE,
                priority=50,
                parameters={"pruning_ratio": 0.2}
            ),
            OptimizationConfig(
                name="tensor_decomposition",
                enabled=False,
                optimization_type=OptimizationType.COMPUTE,
                priority=60,
                parameters={"rank_ratio": 0.5}
            ),
            OptimizationConfig(
                name="kernel_fusion",
                enabled=True,
                optimization_type=OptimizationType.COMPUTE,
                priority=5,
                parameters={"fusion_passes": ["linear_relu", "add_norm"]}
            ),
            OptimizationConfig(
                name="adaptive_batching",
                enabled=True,
                optimization_type=OptimizationType.COMPUTE,
                priority=25,
                parameters={"max_batch_size": 32}
            )
        ],
        priority_order=[
            "kernel_fusion",  # Apply first
            "flash_attention",
            "sparse_attention", 
            "adaptive_sparse_attention",
            "adaptive_batching",
            "disk_offloading",
            "activation_offloading",
            "tensor_compression",
            "structured_pruning",
            "tensor_decomposition"
        ]
    )


def create_qwen_optimization_config() -> ModelOptimizationConfig:
    """Create optimization configuration for Qwen models."""
    return ModelOptimizationConfig(
        model_family=ModelFamily.QWEN,
        optimizations=[
            OptimizationConfig(
                name="flash_attention",
                enabled=True,
                optimization_type=OptimizationType.ATTENTION,
                priority=10,
                parameters={"use_triton": True}
            ),
            OptimizationConfig(
                name="sparse_attention",
                enabled=True,
                optimization_type=OptimizationType.ATTENTION,
                priority=15,
                parameters={"sparsity_ratio": 0.3}
            ),
            OptimizationConfig(
                name="adaptive_sparse_attention",
                enabled=True,
                optimization_type=OptimizationType.ATTENTION,
                priority=20,
                parameters={"adaptive_strategy": "dynamic"}
            ),
            OptimizationConfig(
                name="disk_offloading",
                enabled=True,
                optimization_type=OptimizationType.MEMORY,
                priority=30,
                parameters={"max_memory_ratio": 0.85}
            ),
            OptimizationConfig(
                name="activation_offloading",
                enabled=True,
                optimization_type=OptimizationType.MEMORY,
                priority=35,
                parameters={"max_memory_ratio": 0.75}
            ),
            OptimizationConfig(
                name="tensor_compression",
                enabled=True,
                optimization_type=OptimizationType.COMPUTE,
                priority=40,
                parameters={"compression_ratio": 0.6}
            ),
            OptimizationConfig(
                name="structured_pruning",
                enabled=True,
                optimization_type=OptimizationType.MODEL_STRUCTURE,
                priority=50,
                parameters={"pruning_ratio": 0.15}
            ),
            OptimizationConfig(
                name="tensor_decomposition",
                enabled=True,
                optimization_type=OptimizationType.COMPUTE,
                priority=60,
                parameters={"rank_ratio": 0.4}
            ),
            OptimizationConfig(
                name="snn",
                enabled=False,
                optimization_type=OptimizationType.COMPUTE,
                priority=70,
                parameters={"neuron_type": "LIF", "threshold": 1.0}
            ),
            OptimizationConfig(
                name="kernel_fusion",
                enabled=True,
                optimization_type=OptimizationType.COMPUTE,
                priority=5,
                parameters={"fusion_passes": ["linear_relu", "add_norm", "matmul_add"]}
            ),
            OptimizationConfig(
                name="adaptive_batching",
                enabled=True,
                optimization_type=OptimizationType.COMPUTE,
                priority=25,
                parameters={"max_batch_size": 64}
            )
        ],
        priority_order=[
            "kernel_fusion",
            "flash_attention",
            "sparse_attention",
            "adaptive_sparse_attention",
            "adaptive_batching",
            "disk_offloading",
            "activation_offloading",
            "tensor_compression",
            "structured_pruning",
            "tensor_decomposition",
            "snn"
        ]
    )


# Register default model configurations
def register_default_model_configs():
    """Register default model configurations."""
    manager = get_config_manager()
    
    configs = [
        create_glm_optimization_config(),
        create_qwen_optimization_config(),
    ]
    
    for config in configs:
        manager.register_model_config(config)


register_default_model_configs()


__all__ = [
    "ModelFamily",
    "OptimizationConfig",
    "ModelOptimizationConfig",
    "GlobalOptimizationProfile",
    "OptimizationConfigManager",
    "get_config_manager",
    "config_manager",
    # Profile creation functions
    "create_balanced_profile",
    "create_performance_profile",
    "create_memory_efficient_profile",
    "create_experimental_profile",
    "register_default_profiles",
    # Model config creation functions
    "create_glm_optimization_config",
    "create_qwen_optimization_config",
    "register_default_model_configs",
]