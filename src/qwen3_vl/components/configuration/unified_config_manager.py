"""Production-ready unified configuration system for Qwen3-VL model.

This module provides a comprehensive configuration system that handles all configuration needs
across the project with support for multiple sources, validation, type checking, and runtime updates.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import yaml
import logging
from copy import deepcopy
import warnings
import torch


logger = logging.getLogger(__name__)


@dataclass
class BaseConfig:
    """Base configuration class with validation and serialization capabilities."""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Default implementation - subclasses can override for specific validation
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if hasattr(value, 'to_dict'):
                result[f.name] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                result[f.name] = [v.to_dict() if hasattr(v, 'to_dict') else v for v in value]
            elif isinstance(value, dict):
                result[f.name] = {k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in value.items()}
            else:
                result[f.name] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create configuration from dictionary."""
        # Get all field names for this class
        field_names = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)
    
    def save_to_file(self, file_path: Union[str, Path]):
        """Save configuration to JSON or YAML file."""
        file_path = Path(file_path)
        config_dict = self.to_dict()
        
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]):
        """Load configuration from JSON or YAML file."""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


@dataclass
class MemoryConfig(BaseConfig):
    """Memory optimization configuration."""
    # Memory pool settings
    memory_pool_size: int = 2 * 1024 * 1024 * 1024  # 2GB default
    memory_pool_dtype: str = "float16"
    memory_pool_device: Optional[str] = None

    # Memory tiering settings
    enable_memory_tiering: bool = True
    gpu_memory_size: int = 6 * 1024 * 1024 * 1024  # 6GB for GPU HBM
    cpu_memory_size: int = 8 * 1024 * 1024 * 1024  # 8GB for CPU RAM
    ssd_memory_size: int = 50 * 1024 * 1024 * 1024  # 50GB for NVMe SSD

    # Memory compression settings
    enable_memory_compression: bool = True
    compression_level: str = "medium"  # "low", "medium", "high"
    compression_threshold: float = 0.5  # Threshold for compression

    # Memory swapping settings
    enable_memory_swapping: bool = True
    swap_threshold: float = 0.8  # Percentage of memory usage that triggers swapping
    swap_algorithm: str = "lru"  # "lru", "fifo", "priority"

    # Memory defragmentation settings
    enable_memory_defragmentation: bool = True
    defragmentation_interval: int = 1000  # Steps between defragmentation attempts
    defragmentation_threshold: float = 0.7  # Memory fragmentation percentage that triggers defragmentation

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MemoryConfig':
        """Create MemoryConfig from dictionary."""
        # Get all field names for this class
        field_names = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)


@dataclass
class CPUConfig(BaseConfig):
    """CPU optimization configuration."""
    # Threading settings
    num_threads: int = 4
    num_workers: int = 4
    max_concurrent_threads: int = 8

    # Cache optimization settings
    l1_cache_size: int = 32 * 1024  # 32KB
    l2_cache_size: int = 256 * 1024  # 256KB
    l3_cache_size: int = 6 * 1024 * 1024  # 6MB
    cache_line_size: int = 64  # 64 bytes

    # CPU-specific optimizations
    enable_cpu_optimizations: bool = True
    use_hyperthreading: bool = True
    enable_simd_optimizations: bool = True
    simd_instruction_set: str = "avx2"  # "avx2", "sse", "scalar"

    # Preprocessing settings
    num_preprocess_workers: int = 4
    preprocess_batch_size: int = 8
    memory_threshold: float = 0.8
    transfer_async: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CPUConfig':
        """Create CPUConfig from dictionary."""
        # Get all field names for this class
        field_names = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)


@dataclass
class GPUConfig(BaseConfig):
    """GPU optimization configuration."""
    # Hardware-specific settings
    gpu_compute_capability: tuple = (6, 1)  # SM61 default
    max_threads_per_block: int = 1024
    shared_memory_per_block: int = 48 * 1024  # 48KB
    memory_bandwidth_gbps: float = 320.0  # Example bandwidth for GTX 1080 Ti
    gpu_memory_size: int = 6 * 1024 * 1024 * 1024  # 6GB for GPU HBM - Add this missing attribute

    # GPU optimization settings
    enable_gpu_optimizations: bool = True
    use_tensor_cores: bool = True
    use_mixed_precision: bool = True
    enable_cuda_graphs: bool = True

    # Attention optimization settings
    attention_implementation: str = "flash_attention_2"  # "standard", "flash_attention_2", "optimized"
    use_memory_efficient_attention: bool = True
    kv_cache_strategy: str = "hybrid"  # "standard", "low_rank", "sliding_window", "hybrid"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GPUConfig':
        """Create GPUConfig from dictionary."""
        # Get all field names for this class
        field_names = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)


@dataclass
class PowerManagementConfig(BaseConfig):
    """Power management configuration."""
    # Power optimization settings
    enable_power_optimization: bool = True
    power_constraint: float = 0.8  # Target power usage percentage
    thermal_constraint: float = 75.0  # Max temperature in Celsius
    performance_target: float = 0.9  # Target performance level

    # Adaptive settings
    adaptation_frequency: float = 1.0  # Frequency of power/thermal adaptation in seconds
    enable_dynamic_power_scaling: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PowerManagementConfig':
        """Create PowerManagementConfig from dictionary."""
        # Get all field names for this class
        field_names = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)


@dataclass
class OptimizationConfig(BaseConfig):
    """Configuration for all optimization techniques."""
    # Memory optimization settings
    use_memory_pooling: bool = True
    use_hierarchical_memory_compression: bool = True
    use_memory_efficient_attention: bool = True
    use_kv_cache_optimization: bool = True
    use_cross_layer_parameter_sharing: bool = True

    # Computation optimization settings
    use_sparsity: bool = True
    sparsity_ratio: float = 0.5
    use_dynamic_sparse_attention: bool = True
    use_adaptive_precision: bool = True
    use_moe: bool = True
    moe_num_experts: int = 4
    moe_top_k: int = 2
    use_flash_attention_2: bool = True
    use_adaptive_depth: bool = True
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True  # Add this missing attribute

    # Architecture optimization settings
    use_context_adaptive_positional_encoding: bool = True
    use_conditional_feature_extraction: bool = True
    use_cross_modal_compression: bool = True
    use_cross_layer_memory_sharing: bool = True
    use_hierarchical_vision: bool = True
    use_learned_activation_routing: bool = True
    use_adaptive_batch_processing: bool = True
    use_adaptive_sequence_packing: bool = True
    use_memory_efficient_grad_accumulation: bool = True
    use_faster_rotary_embeddings: bool = True
    use_distributed_pipeline_parallelism: bool = False
    use_hardware_specific_kernels: bool = True

    # Performance thresholds
    performance_improvement_threshold: float = 0.05  # 5% improvement required
    accuracy_preservation_threshold: float = 0.95  # 95% of original accuracy maintained

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OptimizationConfig':
        """Create OptimizationConfig from dictionary."""
        # Get all field names for this class
        field_names = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)


@dataclass
class UnifiedConfig(BaseConfig):
    """Unified configuration for the entire Qwen3-VL system."""
    # Core model configuration (maintains full capacity)
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    hidden_size: int = 4096
    intermediate_size: int = 11008
    vocab_size: int = 152064
    max_position_embeddings: int = 32768
    vision_num_hidden_layers: int = 24
    vision_num_attention_heads: int = 16
    vision_hidden_size: int = 1152
    vision_image_size: int = 448
    vision_patch_size: int = 14
    
    # Standard configuration parameters
    hidden_act: str = "silu"
    hidden_dropout_prob: float = 0.0
    attention_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    pad_token_id: int = 0
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = False
    
    # Optimization configurations
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    cpu_config: CPUConfig = field(default_factory=CPUConfig)
    gpu_config: GPUConfig = field(default_factory=GPUConfig)
    power_config: PowerManagementConfig = field(default_factory=PowerManagementConfig)
    optimization_config: OptimizationConfig = field(default_factory=OptimizationConfig)
    
    # Hardware-specific settings
    hardware_target: str = "intel_i5_10210u_nvidia_sm61_nvme"  # Current hardware configuration
    target_hardware: str = "nvidia_sm61"  # Specific GPU target
    compute_units: int = 4  # CPU cores
    memory_gb: float = 8.0  # Available memory in GB
    
    # Performance and resource management
    output_attentions: bool = False
    output_hidden_states: bool = False
    use_cache: bool = True
    torch_dtype: str = "float16"
    optimization_level: str = "balanced"  # "minimal", "balanced", "aggressive", "maximum"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate that hidden_size is divisible by num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})")

        if self.vision_hidden_size % self.vision_num_attention_heads != 0:
            raise ValueError(f"vision_hidden_size ({self.vision_hidden_size}) must be divisible by vision_num_attention_heads ({self.vision_num_attention_heads})")

        # Validate sparsity ratio
        if not (0.0 <= self.optimization_config.sparsity_ratio <= 1.0):
            raise ValueError(f"sparsity_ratio must be between 0.0 and 1.0, got {self.optimization_config.sparsity_ratio}")

        # Validate exit threshold
        if not (0.0 <= getattr(self.optimization_config, 'exit_threshold', 0.8) <= 1.0):
            raise ValueError(f"exit_threshold must be between 0.0 and 1.0, got {getattr(self.optimization_config, 'exit_threshold', 0.8)}")

        # Validate MoE settings
        if self.optimization_config.use_moe:
            if self.optimization_config.moe_num_experts < 2:
                raise ValueError(f"moe_num_experts must be at least 2, got {self.optimization_config.moe_num_experts}")
            if self.optimization_config.moe_top_k < 1 or self.optimization_config.moe_top_k > self.optimization_config.moe_num_experts:
                raise ValueError(f"moe_top_k must be between 1 and moe_num_experts ({self.optimization_config.moe_num_experts}), got {self.optimization_config.moe_top_k}")

        # Validate compression level
        if self.memory_config.compression_level not in ["low", "medium", "high"]:
            raise ValueError(f"compression_level must be 'low', 'medium', or 'high', got {self.memory_config.compression_level}")

        # Validate swap algorithm
        if self.memory_config.swap_algorithm not in ["lru", "fifo", "priority"]:
            raise ValueError(f"swap_algorithm must be 'lru', 'fifo', or 'priority', got {self.memory_config.swap_algorithm}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UnifiedConfig':
        """Create UnifiedConfig from dictionary with proper nested config handling."""
        # Get all field names for this class
        field_names = {f.name for f in fields(cls)}
        filtered_dict = {}

        for k, v in config_dict.items():
            if k in field_names:
                # Check if the field is a configuration object and needs to be reconstructed
                if k == 'memory_config' and v is not None:
                    filtered_dict[k] = MemoryConfig.from_dict(v) if isinstance(v, dict) else v
                elif k == 'cpu_config' and v is not None:
                    filtered_dict[k] = CPUConfig.from_dict(v) if isinstance(v, dict) else v
                elif k == 'gpu_config' and v is not None:
                    filtered_dict[k] = GPUConfig.from_dict(v) if isinstance(v, dict) else v
                elif k == 'power_config' and v is not None:
                    filtered_dict[k] = PowerManagementConfig.from_dict(v) if isinstance(v, dict) else v
                elif k == 'optimization_config' and v is not None:
                    filtered_dict[k] = OptimizationConfig.from_dict(v) if isinstance(v, dict) else v
                else:
                    filtered_dict[k] = v

        return cls(**filtered_dict)


class ConfigValidator:
    """Validates configuration parameters and their compatibility."""
    
    @staticmethod
    def validate_config(config: UnifiedConfig) -> List[str]:
        """Validate the configuration and return list of validation errors."""
        errors = []

        # Validate core model parameters
        if config.num_hidden_layers <= 0:
            errors.append(f"num_hidden_layers must be positive, got {config.num_hidden_layers}")

        if config.num_attention_heads <= 0:
            errors.append(f"num_attention_heads must be positive, got {config.num_attention_heads}")

        if config.hidden_size <= 0:
            errors.append(f"hidden_size must be positive, got {config.hidden_size}")

        if config.intermediate_size <= 0:
            errors.append(f"intermediate_size must be positive, got {config.intermediate_size}")

        if config.vocab_size <= 0:
            errors.append(f"vocab_size must be positive, got {config.vocab_size}")

        if config.max_position_embeddings <= 0:
            errors.append(f"max_position_embeddings must be positive, got {config.max_position_embeddings}")

        # Validate memory configuration
        if config.memory_config.memory_pool_size <= 0:
            errors.append(f"memory_pool_size must be positive, got {config.memory_config.memory_pool_size}")

        if config.memory_config.compression_threshold < 0 or config.memory_config.compression_threshold > 1:
            errors.append(f"compression_threshold must be between 0 and 1, got {config.memory_config.compression_threshold}")

        if config.memory_config.swap_threshold < 0 or config.memory_config.swap_threshold > 1:
            errors.append(f"swap_threshold must be between 0 and 1, got {config.memory_config.swap_threshold}")

        # Validate CPU configuration
        if config.cpu_config.num_threads <= 0:
            errors.append(f"num_threads must be positive, got {config.cpu_config.num_threads}")

        if config.cpu_config.l1_cache_size <= 0:
            errors.append(f"l1_cache_size must be positive, got {config.cpu_config.l1_cache_size}")

        # Validate GPU configuration
        if config.gpu_config.max_threads_per_block <= 0:
            errors.append(f"max_threads_per_block must be positive, got {config.gpu_config.max_threads_per_block}")

        if config.gpu_config.memory_bandwidth_gbps <= 0:
            errors.append(f"memory_bandwidth_gbps must be positive, got {config.gpu_config.memory_bandwidth_gbps}")

        # Validate optimization configuration
        if config.optimization_config.sparsity_ratio < 0 or config.optimization_config.sparsity_ratio > 1:
            errors.append(f"sparsity_ratio must be between 0 and 1, got {config.optimization_config.sparsity_ratio}")

        if config.optimization_config.moe_num_experts < 2:
            errors.append(f"moe_num_experts must be at least 2, got {config.optimization_config.moe_num_experts}")

        if config.optimization_config.moe_top_k < 1 or config.optimization_config.moe_top_k > config.optimization_config.moe_num_experts:
            errors.append(f"moe_top_k must be between 1 and moe_num_experts ({config.optimization_config.moe_num_experts}), got {config.optimization_config.moe_top_k}")

        return errors
    
    @staticmethod
    def validate_optimization_compatibility(config: OptimizationConfig) -> List[str]:
        """Validate that optimization settings are compatible with each other."""
        errors = []

        # Some optimizations might conflict with each other
        if config.use_gradient_checkpointing and config.use_memory_efficient_attention:
            warnings.warn("Using gradient checkpointing with memory efficient attention may lead to suboptimal performance.")

        if config.use_moe and config.use_sparsity:
            # These can coexist but may need special handling
            pass

        return errors


class ConfigSourceManager:
    """Manages configuration from multiple sources with proper precedence."""
    
    def __init__(self):
        self.sources: Dict[str, Any] = {}
    
    def add_source(self, name: str, source: Union[Dict[str, Any], str, Path, UnifiedConfig]):
        """Add a configuration source."""
        self.sources[name] = source
    
    def load_from_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file."""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    def load_from_env(self, prefix: str = "QWEN3_") -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                
                # Try to convert to appropriate type
                if value.lower() in ['true', 'false']:
                    env_config[config_key] = value.lower() == 'true'
                elif value.isdigit():
                    env_config[config_key] = int(value)
                elif '.' in value and all(part.isdigit() for part in value.split('.', 1)):
                    env_config[config_key] = float(value)
                else:
                    # Check if it looks like a list/tuple
                    if value.startswith('[') and value.endswith(']'):
                        # Handle list values
                        try:
                            env_config[config_key] = eval(value)
                        except:
                            env_config[config_key] = value
                    elif value.startswith('(') and value.endswith(')'):
                        # Handle tuple values
                        try:
                            env_config[config_key] = eval(value)
                        except:
                            env_config[config_key] = value
                    else:
                        env_config[config_key] = value
        
        return env_config
    
    def merge_configs(self, base_config: Union[Dict[str, Any], UnifiedConfig], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge override configuration into base configuration."""
        if isinstance(base_config, UnifiedConfig):
            base_dict = base_config.to_dict()
        else:
            base_dict = base_config
        
        # Deep merge the dictionaries
        merged_dict = self._deep_merge(base_dict, overrides)
        
        return merged_dict
    
    def _deep_merge(self, base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = deepcopy(base)

        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result
    
    def resolve_config(self, precedence: Optional[List[str]] = None) -> UnifiedConfig:
        """Resolve configuration from all sources based on precedence."""
        if precedence is None:
            precedence = list(self.sources.keys())

        resolved_config = UnifiedConfig()

        for source_name in precedence:
            if source_name in self.sources:
                source = self.sources[source_name]

                if isinstance(source, UnifiedConfig):
                    # If source is already a config object, merge its dict representation
                    source_dict = source.to_dict()
                elif isinstance(source, (str, Path)):
                    # If source is a file path, load from file
                    source_dict = self.load_from_file(source)
                elif isinstance(source, dict):
                    # If source is a dict, use it directly
                    source_dict = source
                else:
                    raise ValueError(f"Invalid source type: {type(source)}")

                # Merge this source into the resolved config
                resolved_dict = self.merge_configs(resolved_config, source_dict)
                resolved_config = UnifiedConfig.from_dict(resolved_dict)

        return resolved_config


class UnifiedConfigManager:
    """Centralized configuration manager for all optimization techniques."""
    
    def __init__(self, base_config: Optional[UnifiedConfig] = None):
        self.validator: ConfigValidator = ConfigValidator()
        self.source_manager: ConfigSourceManager = ConfigSourceManager()

        # Initialize with base config or default
        self.base_config: UnifiedConfig = base_config or UnifiedConfig()

        # Register default configurations
        self._register_default_configs()

    def _register_default_configs(self) -> None:
        """Register default configurations for different optimization levels."""
        # Minimal optimization config
        minimal_config = UnifiedConfig(
            optimization_config=OptimizationConfig(
                use_memory_pooling=False,
                use_hierarchical_memory_compression=False,
                use_memory_efficient_attention=False,
                use_kv_cache_optimization=False,
                use_cross_layer_parameter_sharing=False,
                use_sparsity=False,
                use_dynamic_sparse_attention=False,
                use_adaptive_precision=False,
                use_moe=False,
                use_flash_attention_2=False,
                use_adaptive_depth=False,
                use_gradient_checkpointing=False,
                use_context_adaptive_positional_encoding=False,
                use_conditional_feature_extraction=False,
                use_cross_modal_compression=False,
                use_cross_layer_memory_sharing=False,
                use_hierarchical_vision=False,
                use_learned_activation_routing=False,
                use_adaptive_batch_processing=False,
                use_adaptive_sequence_packing=False,
                use_memory_efficient_grad_accumulation=False,
                use_faster_rotary_embeddings=False,
                use_distributed_pipeline_parallelism=False,
                use_hardware_specific_kernels=False,
                performance_improvement_threshold=0.0,
                accuracy_preservation_threshold=0.9
            ),
            optimization_level="minimal"
        )

        # Balanced optimization config
        balanced_config = UnifiedConfig(
            optimization_config=OptimizationConfig(
                use_memory_pooling=True,
                use_hierarchical_memory_compression=True,
                use_memory_efficient_attention=True,
                use_kv_cache_optimization=True,
                use_cross_layer_parameter_sharing=True,
                use_sparsity=True,
                sparsity_ratio=0.3,
                use_dynamic_sparse_attention=True,
                use_adaptive_precision=True,
                use_moe=True,
                moe_num_experts=2,
                moe_top_k=1,
                use_flash_attention_2=True,
                use_adaptive_depth=True,
                use_gradient_checkpointing=True,
                use_context_adaptive_positional_encoding=True,
                use_conditional_feature_extraction=True,
                use_cross_modal_compression=True,
                use_cross_layer_memory_sharing=True,
                use_hierarchical_vision=True,
                use_learned_activation_routing=True,
                use_adaptive_batch_processing=True,
                use_adaptive_sequence_packing=True,
                use_memory_efficient_grad_accumulation=True,
                use_faster_rotary_embeddings=True,
                use_distributed_pipeline_parallelism=False,
                use_hardware_specific_kernels=True,
                performance_improvement_threshold=0.05,
                accuracy_preservation_threshold=0.95
            ),
            optimization_level="balanced"
        )

        # Aggressive optimization config
        aggressive_config = UnifiedConfig(
            optimization_config=OptimizationConfig(
                use_memory_pooling=True,
                use_hierarchical_memory_compression=True,
                use_memory_efficient_attention=True,
                use_kv_cache_optimization=True,
                use_cross_layer_parameter_sharing=True,
                use_sparsity=True,
                sparsity_ratio=0.6,
                use_dynamic_sparse_attention=True,
                use_adaptive_precision=True,
                use_moe=True,
                moe_num_experts=4,
                moe_top_k=2,
                use_flash_attention_2=True,
                use_adaptive_depth=True,
                use_gradient_checkpointing=True,
                use_context_adaptive_positional_encoding=True,
                use_conditional_feature_extraction=True,
                use_cross_modal_compression=True,
                use_cross_layer_memory_sharing=True,
                use_hierarchical_vision=True,
                use_learned_activation_routing=True,
                use_adaptive_batch_processing=True,
                use_adaptive_sequence_packing=True,
                use_memory_efficient_grad_accumulation=True,
                use_faster_rotary_embeddings=True,
                use_distributed_pipeline_parallelism=False,
                use_hardware_specific_kernels=True,
                performance_improvement_threshold=0.1,
                accuracy_preservation_threshold=0.9
            ),
            optimization_level="aggressive"
        )

        # Register configurations
        self.source_manager.add_source("minimal", minimal_config)
        self.source_manager.add_source("balanced", balanced_config)
        self.source_manager.add_source("aggressive", aggressive_config)

    def get_config(self, optimization_level: str = "balanced") -> UnifiedConfig:
        """Get a configuration with the specified optimization level."""
        if optimization_level in self.source_manager.sources:
            config = self.source_manager.sources[optimization_level]
            if isinstance(config, dict):
                return UnifiedConfig.from_dict(config)
            else:
                return config
        else:
            # Create a config based on the level
            if optimization_level == "minimal":
                return UnifiedConfig(
                    optimization_config=OptimizationConfig(
                        use_memory_pooling=False,
                        use_hierarchical_memory_compression=False,
                        use_memory_efficient_attention=False,
                        use_kv_cache_optimization=False,
                        use_cross_layer_parameter_sharing=False,
                        use_sparsity=False,
                        use_dynamic_sparse_attention=False,
                        use_adaptive_precision=False,
                        use_moe=False,
                        use_flash_attention_2=False,
                        use_adaptive_depth=False,
                        use_gradient_checkpointing=False,
                        use_context_adaptive_positional_encoding=False,
                        use_conditional_feature_extraction=False,
                        use_cross_modal_compression=False,
                        use_cross_layer_memory_sharing=False,
                        use_hierarchical_vision=False,
                        use_learned_activation_routing=False,
                        use_adaptive_batch_processing=False,
                        use_adaptive_sequence_packing=False,
                        use_memory_efficient_grad_accumulation=False,
                        use_faster_rotary_embeddings=False,
                        use_distributed_pipeline_parallelism=False,
                        use_hardware_specific_kernels=False,
                        performance_improvement_threshold=0.0,
                        accuracy_preservation_threshold=0.9
                    ),
                    optimization_level="minimal"
                )
            elif optimization_level == "aggressive":
                return UnifiedConfig(
                    optimization_config=OptimizationConfig(
                        use_memory_pooling=True,
                        use_hierarchical_memory_compression=True,
                        use_memory_efficient_attention=True,
                        use_kv_cache_optimization=True,
                        use_cross_layer_parameter_sharing=True,
                        use_sparsity=True,
                        sparsity_ratio=0.6,
                        use_dynamic_sparse_attention=True,
                        use_adaptive_precision=True,
                        use_moe=True,
                        moe_num_experts=4,
                        moe_top_k=2,
                        use_flash_attention_2=True,
                        use_adaptive_depth=True,
                        use_gradient_checkpointing=True,
                        use_context_adaptive_positional_encoding=True,
                        use_conditional_feature_extraction=True,
                        use_cross_modal_compression=True,
                        use_cross_layer_memory_sharing=True,
                        use_hierarchical_vision=True,
                        use_learned_activation_routing=True,
                        use_adaptive_batch_processing=True,
                        use_adaptive_sequence_packing=True,
                        use_memory_efficient_grad_accumulation=True,
                        use_faster_rotary_embeddings=True,
                        use_distributed_pipeline_parallelism=False,
                        use_hardware_specific_kernels=True,
                        performance_improvement_threshold=0.1,
                        accuracy_preservation_threshold=0.9
                    ),
                    optimization_level="aggressive"
                )
            else:  # Default to balanced
                return UnifiedConfig(
                    optimization_config=OptimizationConfig(
                        use_memory_pooling=True,
                        use_hierarchical_memory_compression=True,
                        use_memory_efficient_attention=True,
                        use_kv_cache_optimization=True,
                        use_cross_layer_parameter_sharing=True,
                        use_sparsity=True,
                        sparsity_ratio=0.3,
                        use_dynamic_sparse_attention=True,
                        use_adaptive_precision=True,
                        use_moe=True,
                        moe_num_experts=2,
                        moe_top_k=1,
                        use_flash_attention_2=True,
                        use_adaptive_depth=True,
                        use_gradient_checkpointing=True,
                        use_context_adaptive_positional_encoding=True,
                        use_conditional_feature_extraction=True,
                        use_cross_modal_compression=True,
                        use_cross_layer_memory_sharing=True,
                        use_hierarchical_vision=True,
                        use_learned_activation_routing=True,
                        use_adaptive_batch_processing=True,
                        use_adaptive_sequence_packing=True,
                        use_memory_efficient_grad_accumulation=True,
                        use_faster_rotary_embeddings=True,
                        use_distributed_pipeline_parallelism=False,
                        use_hardware_specific_kernels=True,
                        performance_improvement_threshold=0.05,
                        accuracy_preservation_threshold=0.95
                    ),
                    optimization_level="balanced"
                )
        # Default return if no condition is met
        return UnifiedConfig(optimization_level="default")
    
    def load_config_from_file(self, file_path: Union[str, Path], name: str = "file_config") -> bool:
        """Load configuration from file."""
        try:
            config_dict = self.source_manager.load_from_file(file_path)
            config = UnifiedConfig.from_dict(config_dict)
            self.source_manager.add_source(name, config)
            logger.info(f"Loaded configuration from {file_path} as {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load configuration from {file_path}: {e}")
            return False
    
    def load_config_from_env(self, name: str = "env_config", prefix: str = "QWEN3_") -> bool:
        """Load configuration from environment variables."""
        try:
            env_config = self.source_manager.load_from_env(prefix)
            self.source_manager.add_source(name, env_config)
            logger.info(f"Loaded configuration from environment variables with prefix {prefix} as {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load configuration from environment: {e}")
            return False
    
    def get_config(self, optimization_level: str = "balanced") -> UnifiedConfig:
        """Get a configuration with the specified optimization level."""
        if optimization_level in self.source_manager.sources:
            config = self.source_manager.sources[optimization_level]
            if isinstance(config, dict):
                return UnifiedConfig.from_dict(config)
            else:
                return config
        else:
            # Create a config based on the level
            if optimization_level == "minimal":
                return UnifiedConfig(
                    optimization_config=OptimizationConfig(
                        use_memory_pooling=False,
                        use_hierarchical_memory_compression=False,
                        use_memory_efficient_attention=False,
                        use_kv_cache_optimization=False,
                        use_cross_layer_parameter_sharing=False,
                        use_sparsity=False,
                        use_dynamic_sparse_attention=False,
                        use_adaptive_precision=False,
                        use_moe=False,
                        use_flash_attention_2=False,
                        use_adaptive_depth=False,
                        use_gradient_checkpointing=False,
                        use_context_adaptive_positional_encoding=False,
                        use_conditional_feature_extraction=False,
                        use_cross_modal_compression=False,
                        use_cross_layer_memory_sharing=False,
                        use_hierarchical_vision=False,
                        use_learned_activation_routing=False,
                        use_adaptive_batch_processing=False,
                        use_adaptive_sequence_packing=False,
                        use_memory_efficient_grad_accumulation=False,
                        use_faster_rotary_embeddings=False,
                        use_distributed_pipeline_parallelism=False,
                        use_hardware_specific_kernels=False,
                        performance_improvement_threshold=0.0,
                        accuracy_preservation_threshold=0.9
                    ),
                    optimization_level="minimal"
                )
            elif optimization_level == "aggressive":
                return UnifiedConfig(
                    optimization_config=OptimizationConfig(
                        use_memory_pooling=True,
                        use_hierarchical_memory_compression=True,
                        use_memory_efficient_attention=True,
                        use_kv_cache_optimization=True,
                        use_cross_layer_parameter_sharing=True,
                        use_sparsity=True,
                        sparsity_ratio=0.6,
                        use_dynamic_sparse_attention=True,
                        use_adaptive_precision=True,
                        use_moe=True,
                        moe_num_experts=4,
                        moe_top_k=2,
                        use_flash_attention_2=True,
                        use_adaptive_depth=True,
                        use_gradient_checkpointing=True,
                        use_context_adaptive_positional_encoding=True,
                        use_conditional_feature_extraction=True,
                        use_cross_modal_compression=True,
                        use_cross_layer_memory_sharing=True,
                        use_hierarchical_vision=True,
                        use_learned_activation_routing=True,
                        use_adaptive_batch_processing=True,
                        use_adaptive_sequence_packing=True,
                        use_memory_efficient_grad_accumulation=True,
                        use_faster_rotary_embeddings=True,
                        use_distributed_pipeline_parallelism=False,
                        use_hardware_specific_kernels=True,
                        performance_improvement_threshold=0.1,
                        accuracy_preservation_threshold=0.9
                    ),
                    optimization_level="aggressive"
                )
            else:  # Default to balanced
                return UnifiedConfig(
                    optimization_config=OptimizationConfig(
                        use_memory_pooling=True,
                        use_hierarchical_memory_compression=True,
                        use_memory_efficient_attention=True,
                        use_kv_cache_optimization=True,
                        use_cross_layer_parameter_sharing=True,
                        use_sparsity=True,
                        sparsity_ratio=0.3,
                        use_dynamic_sparse_attention=True,
                        use_adaptive_precision=True,
                        use_moe=True,
                        moe_num_experts=2,
                        moe_top_k=1,
                        use_flash_attention_2=True,
                        use_adaptive_depth=True,
                        use_gradient_checkpointing=True,
                        use_context_adaptive_positional_encoding=True,
                        use_conditional_feature_extraction=True,
                        use_cross_modal_compression=True,
                        use_cross_layer_memory_sharing=True,
                        use_hierarchical_vision=True,
                        use_learned_activation_routing=True,
                        use_adaptive_batch_processing=True,
                        use_adaptive_sequence_packing=True,
                        use_memory_efficient_grad_accumulation=True,
                        use_faster_rotary_embeddings=True,
                        use_distributed_pipeline_parallelism=False,
                        use_hardware_specific_kernels=True,
                        performance_improvement_threshold=0.05,
                        accuracy_preservation_threshold=0.95
                    ),
                    optimization_level="balanced"
                )
        # Default return if no condition is met
        return UnifiedConfig(optimization_level="default")

    def validate_config(self, config: UnifiedConfig) -> bool:
        """Validate configuration and return True if valid."""
        errors = self.validator.validate_config(config)
        optimization_errors = self.validator.validate_optimization_compatibility(config.optimization_config)
        all_errors = errors + optimization_errors
        
        if all_errors:
            logger.error(f"Configuration validation failed with {len(all_errors)} errors:")
            for error in all_errors:
                logger.error(f"  - {error}")
            return False
        return True
    
    def update_config(self, config: UnifiedConfig, updates: Dict[str, Any]) -> UnifiedConfig:
        """Update configuration with new values."""
        try:
            # Create updated config
            updated_dict = config.to_dict()
            
            # Apply updates
            for key, value in updates.items():
                if key in updated_dict:
                    updated_dict[key] = value
                else:
                    logger.warning(f"Key '{key}' not found in configuration")
            
            # Create new config from updated dict
            new_config = UnifiedConfig.from_dict(updated_dict)
            
            # Validate the new config
            if self.validate_config(new_config):
                logger.info("Configuration updated successfully")
                return new_config
            else:
                logger.error("Updated configuration failed validation")
                return config  # Return original config if validation fails
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return config
    
    def get_hardware_optimized_config(self, hardware_specs: Dict[str, Any]) -> UnifiedConfig:
        """Get configuration optimized for specific hardware."""
        # Start with the base balanced config
        config = self.get_config("balanced")
        
        # Update based on hardware specs
        if "gpu_memory" in hardware_specs:
            gpu_memory_gb = hardware_specs["gpu_memory"] / (1024**3)
            config.gpu_config.gpu_memory_size = hardware_specs["gpu_memory"]
            
            # Adjust precision based on available GPU memory
            if gpu_memory_gb < 4:
                config.optimization_config.use_mixed_precision = True
                config.torch_dtype = "float16"
            elif gpu_memory_gb < 8:
                config.optimization_config.use_mixed_precision = True
                config.torch_dtype = "bfloat16"
        
        if "cpu_cores" in hardware_specs:
            config.cpu_config.num_threads = hardware_specs["cpu_cores"]
            config.cpu_config.num_workers = hardware_specs["cpu_cores"]
        
        if "memory_gb" in hardware_specs:
            memory_gb = hardware_specs["memory_gb"]
            config.memory_config.memory_pool_size = int(memory_gb * 0.5 * 1024**3)  # 50% of RAM
            
            # Adjust optimization aggressiveness based on memory
            if memory_gb < 8:
                config.optimization_config.sparsity_ratio = 0.5
                config.optimization_config.moe_num_experts = 2
            elif memory_gb < 16:
                config.optimization_config.sparsity_ratio = 0.3
                config.optimization_config.moe_num_experts = 3
            else:
                config.optimization_config.sparsity_ratio = 0.2
                config.optimization_config.moe_num_experts = 4
        
        if "storage_type" in hardware_specs:
            storage_type = hardware_specs["storage_type"].lower()
            if "nvme" in storage_type:
                config.memory_config.enable_memory_swapping = True
                config.memory_config.ssd_memory_size = 100 * 1024**3  # 100GB for NVMe
            elif "ssd" in storage_type:
                config.memory_config.enable_memory_swapping = True
                config.memory_config.ssd_memory_size = 50 * 1024**3  # 50GB for SSD
            else:
                config.memory_config.enable_memory_swapping = False
                config.memory_config.ssd_memory_size = 20 * 1024**3  # 20GB for HDD
        
        # Validate the optimized config
        self.validate_config(config)
        
        return config


def get_default_config() -> UnifiedConfig:
    """Get the default unified configuration."""
    config_manager = UnifiedConfigManager()
    config = config_manager.get_config()
    if config is None:
        # Return a default configuration if the manager returns None
        return UnifiedConfig(
            optimization_config=OptimizationConfig(),
            optimization_level="balanced"
        )
    return config


def create_unified_config_manager() -> UnifiedConfigManager:
    """Create and return a unified configuration manager instance."""
    return UnifiedConfigManager()


# Backward compatibility functions
def get_legacy_config() -> Dict[str, Any]:
    """Return a legacy-style configuration dictionary for backward compatibility."""
    config = get_default_config()
    if config is not None:
        return config.to_dict()
    else:
        # Return a default configuration as dictionary if get_default_config returns None
        default_manager = UnifiedConfigManager()
        default_config = default_manager.get_config()
        return default_config.to_dict() if default_config is not None else {}


def update_legacy_config(config_dict: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update legacy configuration dictionary for backward compatibility."""
    # Convert to unified config, update, then convert back
    unified_config = UnifiedConfig.from_dict(config_dict)
    updated_config = UnifiedConfigManager().update_config(unified_config, updates)
    if updated_config is not None:
        return updated_config.to_dict()
    else:
        # Return original config_dict if update fails
        return config_dict


# Export classes and functions
__all__ = [
    "BaseConfig", "MemoryConfig", "CPUConfig", "GPUConfig", "PowerManagementConfig",
    "OptimizationConfig", "UnifiedConfig", "ConfigValidator", "ConfigSourceManager",
    "UnifiedConfigManager", "get_default_config", "create_unified_config_manager",
    "get_legacy_config", "update_legacy_config"
]


if __name__ == "__main__":
    # Example usage
    print("Creating unified configuration manager...")
    config_manager = create_unified_config_manager()
    
    print("\nGetting balanced configuration...")
    config = config_manager.get_config("balanced")
    print(f"  - Num hidden layers: {config.num_hidden_layers}")
    print(f"  - Num attention heads: {config.num_attention_heads}")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Sparsity ratio: {config.optimization_config.sparsity_ratio}")
    print(f"  - MoE enabled: {config.optimization_config.use_moe}")
    
    print("\nValidating configuration...")
    is_valid = config_manager.validate_config(config)
    print(f"  - Valid: {is_valid}")
    
    print("\nCreating hardware-optimized config for Intel i5-10210U + NVIDIA SM61...")
    hardware_specs = {
        "gpu_memory": 6 * 1024 * 1024 * 1024,  # 6GB
        "cpu_cores": 4,
        "memory_gb": 8,
        "storage_type": "nvme"
    }
    hw_config = config_manager.get_hardware_optimized_config(hardware_specs)
    print(f"  - GPU memory: {hw_config.gpu_config.gpu_memory_size / (1024**3):.1f}GB")
    print(f"  - CPU threads: {hw_config.cpu_config.num_threads}")
    print(f"  - Memory pool size: {hw_config.memory_config.memory_pool_size / (1024**3):.1f}GB")
    
    print("\nSaving configuration to file...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    config.save_to_file(temp_file)
    print(f"  - Saved to {temp_file}")
    
    print("\nLoading configuration from file...")
    loaded_config = UnifiedConfig.from_file(temp_file)
    print(f"  - Loaded config with {loaded_config.num_hidden_layers} layers")
    
    print("\nBackward compatibility test...")
    legacy_dict = get_legacy_config()
    print(f"  - Legacy config keys: {len(legacy_dict)}")
    
    print("\nUnified configuration system initialized successfully!")
    
    # Clean up
    os.unlink(temp_file)