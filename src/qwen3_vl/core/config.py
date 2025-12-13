"""Configuration management system for Qwen3-VL model with support for multiple optimization techniques."""

import json
import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import yaml
    from transformers import LogitsProcessorList, StoppingCriteriaList
    from transformers.generation import Constraint
else:
    try:
        import yaml
    except ImportError:
        yaml = None
    LogitsProcessorList = None
    StoppingCriteriaList = None
    Constraint = None
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
    
    def save_to_file(self, file_path: Union[str, Path]):
        """Save configuration to JSON or YAML file."""
        file_path = Path(file_path)
        config_dict = self.to_dict()

        if file_path.suffix.lower() in ['.yaml', '.yml']:
            if yaml is None:
                raise ImportError("PyYAML is required to save/load YAML configuration files. Install it with 'pip install PyYAML'.")
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
            if yaml is None:
                raise ImportError("PyYAML is required to save/load YAML configuration files. Install it with 'pip install PyYAML'.")
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


@dataclass
class GPUConfig(BaseConfig):
    """GPU optimization configuration."""
    # Hardware-specific settings
    gpu_compute_capability: tuple = (6, 1)  # SM61 default
    max_threads_per_block: int = 1024
    shared_memory_per_block: int = 48 * 1024  # 48KB
    memory_bandwidth_gbps: float = 320.0  # Example bandwidth for GTX 1080 Ti
    gpu_memory_size: int = 6 * 1024 * 1024 * 1024  # 6GB for GPU HBM - Add the missing attribute

    # GPU optimization settings
    enable_gpu_optimizations: bool = True
    use_tensor_cores: bool = True
    use_mixed_precision: bool = True
    enable_cuda_graphs: bool = True
    use_flash_attention_2: bool = True  # Add the missing attribute

    # Attention optimization settings
    attention_implementation: str = "flash_attention_2"  # "standard", "flash_attention_2", "optimized"
    use_memory_efficient_attention: bool = True
    kv_cache_strategy: str = "hybrid"  # "standard", "low_rank", "sliding_window", "hybrid"


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


@dataclass
class Qwen3VLConfig(BaseConfig):
    """Configuration for the Qwen3-VL model."""
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

    # Optimization configurations (using default factory functions)
    memory_config: Optional[MemoryConfig] = None
    cpu_config: Optional[CPUConfig] = None
    gpu_config: Optional[GPUConfig] = None
    power_config: Optional[PowerManagementConfig] = None
    optimization_config: Optional[OptimizationConfig] = None

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

    # KV cache optimization attributes
    use_kv_cache_optimization: bool = False
    kv_cache_window_size: int = 1024
    kv_low_rank_dimension: int = 64
    use_low_rank_kv_cache: bool = False

    # Gradient checkpointing
    use_gradient_checkpointing: bool = True

    # Adaptive depth attributes
    use_adaptive_depth: bool = False
    min_depth_ratio: float = 0.2
    max_depth_ratio: float = 1.0
    vision_min_depth_ratio: float = 0.3
    vision_max_depth_ratio: float = 1.0
    depth_temperature: float = 1.0

    # Additional attributes from various config modules
    num_key_value_heads: Optional[int] = None  # For grouped query attention
    use_rotary_embedding: bool = True  # Use rotary embedding
    use_dynamic_sparse_attention: bool = False  # Use dynamic sparse attention
    use_context_adaptive_positional_encoding: bool = False  # Use context adaptive positional encoding
    use_conditional_feature_extraction: bool = False  # Use conditional feature extraction
    use_hierarchical_vision: bool = False  # Use hierarchical vision processing
    use_learned_activation_routing: bool = False  # Use learned activation routing
    use_adaptive_batch_processing: bool = False  # Use adaptive batch processing
    use_adaptive_sequence_packing: bool = False  # Use adaptive sequence packing
    use_memory_efficient_grad_accumulation: bool = False  # Use memory efficient gradient accumulation
    use_faster_rotary_embeddings: bool = False  # Use faster rotary embeddings
    use_distributed_pipeline_parallelism: bool = False  # Use distributed pipeline parallelism
    use_hardware_specific_kernels: bool = False  # Use hardware specific kernels
    use_cross_layer_parameter_sharing: bool = False  # Use cross layer parameter sharing
    vision_num_channels: int = 3  # Number of channels in vision model
    vision_qkv_bias: bool = True  # Bias for QKV projection in vision attention
    vision_window_size: int = 14  # Window size for vision model
    vision_projection_dim: int = 2048  # Projection dimension for vision model
    language_projection_dim: int = 2048  # Projection dimension for language model
    num_query_tokens: int = 64  # Number of query tokens for vision-language fusion
    vision_model_type: str = "clip_vision_model"  # Type of vision model
    vision_intermediate_size: int = 4304  # Intermediate size for vision model
    vision_hidden_act: str = "gelu"  # Activation function for vision model
    pretraining_tp: int = 1  # Parameter for pretraining tensor parallelism
    use_vision_adaptive_depth: bool = False  # Use adaptive depth for vision transformer
    use_multimodal_adaptive_depth: bool = False  # Use adaptive depth for multimodal fusion
    use_early_exit: bool = False  # Use early exit mechanisms
    exit_threshold: float = 0.8  # Confidence threshold for early exit
    use_adapters: bool = False  # Use parameter-efficient adaptation
    vision_compute_capability: tuple = (6, 1)  # Vision compute capability
    eos_token_id: Optional[int] = None  # End of sequence token id
    bos_token_id: Optional[int] = None  # Beginning of sequence token id
    num_beams: int = 1  # Number of beams for beam search
    do_sample: bool = True  # Do sampling during generation
    temperature: float = 1.0  # Sampling temperature
    top_k: int = 50  # Top-k sampling
    top_p: float = 1.0  # Top-p sampling
    repetition_penalty: float = 1.0  # Repetition penalty
    length_penalty: float = 1.0  # Length penalty
    no_repeat_ngram_size: int = 0  # No repeat ngram size
    early_stopping: bool = False  # Early stopping
    max_new_tokens: int = 512  # Maximum new tokens to generate
    min_new_tokens: Optional[int] = None  # Minimum new tokens to generate
    decoder_start_token_id: Optional[int] = None  # Decoder start token id
    forced_eos_token_id: Optional[int] = None  # Forced EOS token id
    suppress_tokens: Optional[List[int]] = None  # Suppressed tokens
    begin_suppress_tokens: Optional[List[int]] = None  # Begin suppressed tokens
    forced_decoder_ids: Optional[List[List[int]]] = None  # Forced decoder ids
    num_beam_groups: int = 1  # Number of beam groups
    diversity_penalty: float = 0.0  # Diversity penalty
    remove_invalid_values: bool = False  # Remove invalid values
    exponential_decay_length_penalty: Optional[Tuple[Union[int, float], int]] = None  # Exponential decay length penalty
    generation_kwargs: Optional[Dict[str, Any]] = None  # Generation kwargs
    chunk_size_feed_forward: int = 0  # Chunk size for feed forward
    is_decoder: bool = True  # Is decoder
    is_encoder_decoder: bool = False  # Is encoder decoder
    add_cross_attention: bool = False  # Add cross attention
    tie_encoder_decoder: bool = False  # Tie encoder decoder
    max_length: int = 20  # Maximum length
    min_length: int = 0  # Minimum length
    num_return_sequences: int = 1  # Number of return sequences
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None  # Prefix allowed tokens function
    logits_processor: Optional['LogitsProcessorList'] = None  # Logits processor
    stopping_criteria: Optional['StoppingCriteriaList'] = None  # Stopping criteria
    constraints: Optional[List['Constraint']] = None  # Constraints
    output_scores: bool = False  # Output scores
    return_dict_in_generate: bool = False  # Return dict in generate
    forced_bos_token_id: Optional[int] = None  # Forced BOS token id
    forced_eos_token_id: Optional[int] = None  # Forced EOS token id
    renormalize_logits: bool = False  # Renormalize logits
    torchscript: bool = False  # Torchscript
    use_bfloat16: bool = False  # Use bfloat16
    prune_heads: Optional[Dict[int, List[int]]] = None  # Prune heads
    tie_word_embeddings: bool = True  # Tie word embeddings
    cross_attention_hidden_size: Optional[int] = None  # Cross attention hidden size
    typical_p: float = 1.0  # Typical p
    encoder_no_repeat_ngram_size: int = 0  # Encoder no repeat ngram size
    bad_words_ids: Optional[List[List[int]]] = None  # Bad words ids
    param_version: str = "1.0"  # Parameter version
    architectures: Optional[List[str]] = None  # Architectures
    model_type: str = "qwen3_vl"  # Model type
    transformers_version: str = "4.21.0"  # Transformers version
    finetuning_task: Optional[str] = None  # Finetuning task
    id2label: Optional[Dict[int, str]] = None  # Id to label mapping
    label2id: Optional[Dict[str, int]] = None  # Label to id mapping
    tokenizer_class: Optional[str] = None  # Tokenizer class
    prefix: Optional[str] = None  # Prefix
    sep_token_id: Optional[int] = None  # Sep token id
    cls_token_id: Optional[int] = None  # Cls token id
    unk_token_id: Optional[int] = None  # Unk token id
    mask_token_id: Optional[int] = None  # Mask token id
    additional_special_tokens: Optional[List[str]] = None  # Additional special tokens
    problem_type: Optional[str] = None  # Problem type
    classifier_dropout: Optional[float] = None  # Classifier dropout
    forced_deconv_weights: Optional[List[int]] = None  # Forced deconv weights

    # GPUConfig specific attributes that were causing errors
    use_flash_attention_2: bool = True  # Add the missing attribute
    attention_implementation: str = "flash_attention_2"  # Add the missing attribute
    kv_cache_strategy: str = "hybrid"  # Add the missing attribute
    # Note: gpu_memory_size is in the gpu_config sub-object, not directly in Qwen3VLConfig
    use_mixed_precision: bool = True  # Add the missing attribute

    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.memory_config is None:
            self.memory_config = MemoryConfig()
        if self.cpu_config is None:
            self.cpu_config = CPUConfig()
        if self.gpu_config is None:
            self.gpu_config = GPUConfig()
        if self.power_config is None:
            self.power_config = PowerManagementConfig()
        if self.optimization_config is None:
            self.optimization_config = OptimizationConfig()

        """Validate configuration after initialization."""
        # Validate that hidden_size is divisible by num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})")

        if self.vision_hidden_size % self.vision_num_attention_heads != 0:
            raise ValueError(f"vision_hidden_size ({self.vision_hidden_size}) must be divisible by vision_num_attention_heads ({self.vision_num_attention_heads})")

        # Validate sparsity ratio
        if not (0.0 <= self.optimization_config.sparsity_ratio <= 1.0):
            raise ValueError(f"sparsity_ratio must be between 0.0 and 1.0, got {self.optimization_config.sparsity_ratio}")

        # Validate exit threshold (if it exists in the config)
        if hasattr(self.optimization_config, 'exit_threshold'):
            exit_threshold = self.optimization_config.exit_threshold
            if not (0.0 <= exit_threshold <= 1.0):
                raise ValueError(f"exit_threshold must be between 0.0 and 1.0, got {exit_threshold}")

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

    # Properties to access attributes from optimization_config for backward compatibility
    @property
    def use_gradient_checkpointing(self):
        """Access use_gradient_checkpointing from optimization_config."""
        if self.optimization_config is not None and hasattr(self.optimization_config, 'use_gradient_checkpointing'):
            return self.optimization_config.use_gradient_checkpointing
        return True  # Default value

    @use_gradient_checkpointing.setter
    def use_gradient_checkpointing(self, value):
        """Set use_gradient_checkpointing in optimization_config."""
        if self.optimization_config is None:
            self.optimization_config = OptimizationConfig()
        if hasattr(self.optimization_config, 'use_gradient_checkpointing'):
            self.optimization_config.use_gradient_checkpointing = value

    @property
    def attention_implementation(self):
        """Access attention_implementation from gpu_config."""
        if self.gpu_config is not None and hasattr(self.gpu_config, 'attention_implementation'):
            return self.gpu_config.attention_implementation
        return "flash_attention_2"  # Default value

    @attention_implementation.setter
    def attention_implementation(self, value):
        """Set attention_implementation in gpu_config."""
        if self.gpu_config is None:
            self.gpu_config = GPUConfig()
        if hasattr(self.gpu_config, 'attention_implementation'):
            self.gpu_config.attention_implementation = value

    @property
    def use_flash_attention_2(self):
        """Access use_flash_attention_2 from gpu_config."""
        if self.gpu_config is not None and hasattr(self.gpu_config, 'use_flash_attention_2'):
            return self.gpu_config.use_flash_attention_2
        return True  # Default value

    @use_flash_attention_2.setter
    def use_flash_attention_2(self, value):
        """Set use_flash_attention_2 in gpu_config."""
        if self.gpu_config is None:
            self.gpu_config = GPUConfig()
        if hasattr(self.gpu_config, 'use_flash_attention_2'):
            self.gpu_config.use_flash_attention_2 = value

    @property
    def use_memory_efficient_attention(self):
        """Access use_memory_efficient_attention from gpu_config."""
        if self.gpu_config is not None and hasattr(self.gpu_config, 'use_memory_efficient_attention'):
            return self.gpu_config.use_memory_efficient_attention
        return True  # Default value

    @use_memory_efficient_attention.setter
    def use_memory_efficient_attention(self, value):
        """Set use_memory_efficient_attention in gpu_config."""
        if self.gpu_config is None:
            self.gpu_config = GPUConfig()
        if hasattr(self.gpu_config, 'use_memory_efficient_attention'):
            self.gpu_config.use_memory_efficient_attention = value

    @property
    def kv_cache_strategy(self):
        """Access kv_cache_strategy from gpu_config."""
        if self.gpu_config is not None and hasattr(self.gpu_config, 'kv_cache_strategy'):
            return self.gpu_config.kv_cache_strategy
        return "hybrid"  # Default value

    @kv_cache_strategy.setter
    def kv_cache_strategy(self, value):
        """Set kv_cache_strategy in gpu_config."""
        if self.gpu_config is None:
            self.gpu_config = GPUConfig()
        if hasattr(self.gpu_config, 'kv_cache_strategy'):
            self.gpu_config.kv_cache_strategy = value

    @property
    def use_dynamic_sparse_attention(self):
        """Access use_dynamic_sparse_attention from optimization_config."""
        if self.optimization_config is not None and hasattr(self.optimization_config, 'use_dynamic_sparse_attention'):
            return self.optimization_config.use_dynamic_sparse_attention
        return True  # Default value

    @use_dynamic_sparse_attention.setter
    def use_dynamic_sparse_attention(self, value):
        """Set use_dynamic_sparse_attention in optimization_config."""
        if self.optimization_config is None:
            self.optimization_config = OptimizationConfig()
        if hasattr(self.optimization_config, 'use_dynamic_sparse_attention'):
            self.optimization_config.use_dynamic_sparse_attention = value

    @property
    def use_activation_sparsity(self):
        """Access use_sparsity from optimization_config."""
        if self.optimization_config is not None and hasattr(self.optimization_config, 'use_sparsity'):
            return self.optimization_config.use_sparsity
        return True  # Default value

    @use_activation_sparsity.setter
    def use_activation_sparsity(self, value):
        """Set use_sparsity in optimization_config."""
        if self.optimization_config is None:
            self.optimization_config = OptimizationConfig()
        if hasattr(self.optimization_config, 'use_sparsity'):
            self.optimization_config.use_sparsity = value

    @property
    def sparsity_ratio(self):
        """Access sparsity_ratio from optimization_config."""
        if self.optimization_config is not None and hasattr(self.optimization_config, 'sparsity_ratio'):
            return self.optimization_config.sparsity_ratio
        return 0.5  # Default value

    @sparsity_ratio.setter
    def sparsity_ratio(self, value):
        """Set sparsity_ratio in optimization_config."""
        if self.optimization_config is None:
            self.optimization_config = OptimizationConfig()
        if hasattr(self.optimization_config, 'sparsity_ratio'):
            self.optimization_config.sparsity_ratio = value

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path], **kwargs):
        """Load configuration from a pretrained model path or name."""
        path = Path(pretrained_model_name_or_path)
        if path.is_file():
            # If it's a file path, load directly
            return cls.from_file(path)
        elif path.is_dir():
            # If it's a directory, look for config.json inside
            config_path = path / "config.json"
            if config_path.exists():
                return cls.from_file(config_path)

        # If it's a model name, try to download from hub
        try:
            from huggingface_hub import hf_hub_download
            repo_id_str = str(pretrained_model_name_or_path)
            config_path_str = hf_hub_download(
                repo_id=repo_id_str,
                filename="config.json"
            )
            return cls.from_file(config_path_str)
        except:
            # If not found on hub, return default config
            return cls(**kwargs)


# Export classes
__all__ = ["Qwen3VLConfig", "MemoryConfig", "CPUConfig", "GPUConfig", "PowerManagementConfig", "OptimizationConfig"]