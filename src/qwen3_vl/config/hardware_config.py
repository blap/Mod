"""
Hardware configuration classes for Qwen3-VL model components.

This module contains configuration classes specifically for hardware-related
optimizations and abstractions with clear separation of concerns.
"""
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class HardwareConfig:
    """
    Configuration class for hardware-related optimizations and abstractions.
    """
    # Hardware detection and abstraction
    hardware_detection_timeout: int = 30  # Timeout for hardware detection in seconds
    hardware_fallback_enabled: bool = True  # Enable fallback to CPU when GPU unavailable
    hardware_monitoring_enabled: bool = True  # Enable hardware monitoring

    # GPU-specific configuration
    gpu_device_ids: Optional[List[int]] = None  # Specific GPU device IDs to use (None for auto-detection)
    gpu_memory_fraction: float = 0.9  # Fraction of GPU memory to use (for TensorFlow-like control)
    gpu_allow_growth: bool = True  # Allow GPU memory growth
    gpu_precision: str = "mixed"  # Precision for GPU operations: "fp16", "fp32", "mixed", "bf16"

    # CPU-specific configuration
    cpu_threads: Optional[int] = None  # Number of CPU threads to use (None for auto-detection)
    cpu_affinity_enabled: bool = False  # Enable CPU affinity for better performance
    cpu_inter_op_parallelism: Optional[int] = None  # Inter-op parallelism threads
    cpu_intra_op_parallelism: Optional[int] = None  # Intra-op parallelism threads

    # Memory transfer optimization
    use_pinned_memory: bool = True  # Use pinned (page-locked) memory for transfers
    use_cuda_streams: bool = True  # Use CUDA streams for async transfers
    memory_transfer_overlap: bool = True  # Overlap computation and memory transfers

    # Hardware-specific optimizations
    hardware_target: str = "auto"  # Target hardware: "auto", "intel_i5_10210u", "nvidia_sm61", "generic"
    enable_intel_optimizations: bool = True  # Enable Intel-specific optimizations
    enable_nvidia_optimizations: bool = True  # Enable NVIDIA-specific optimizations
    enable_avx_instructions: bool = True  # Enable AVX instruction set optimizations
    enable_tensor_cores: bool = True  # Enable tensor core usage on supported GPUs

    # Storage configuration (NVMe SSD caching)
    nvme_cache_enabled: bool = True  # Enable NVMe SSD caching for model components
    nvme_cache_path: Optional[str] = None  # Path for NVMe cache (None for auto)
    nvme_cache_size: int = 1024 * 1024 * 1024 * 10  # 10GB default cache size
    nvme_cache_policy: str = "lru"  # Cache eviction policy: "lru", "fifo", "lfu"
    nvme_prefetch_enabled: bool = True  # Enable prefetching for cached components

    # Power and thermal management
    power_management_enabled: bool = True  # Enable power management features
    thermal_throttling_enabled: bool = True  # Enable thermal throttling protection
    power_limit_watts: Optional[float] = None  # Power limit in watts (None for no limit)
    thermal_limit_celsius: Optional[float] = 85.0  # Thermal limit in Celsius (None for no limit)

    # SIMD and vectorization
    simd_optimization_enabled: bool = True  # Enable SIMD optimizations
    vector_instruction_set: str = "auto"  # Instruction set: "auto", "avx2", "avx", "sse", "none"
    use_jit_compilation: bool = True  # Use JIT compilation for optimized kernels

    # Distributed computing (Phase 9)
    distributed_enabled: bool = False  # Enable distributed computing features
    pipeline_parallelism_degree: int = 1  # Degree of pipeline parallelism
    tensor_parallelism_degree: int = 1  # Degree of tensor parallelism
    use_deepspeed: bool = False  # Use DeepSpeed for distributed training
    use_fairscale: bool = False  # Use FairScale for distributed training

    # Additional attributes to fix errors from temp_errors.txt
    use_tensor_cores: bool = True  # Enable tensor core usage on supported GPUs
    shared_memory_per_block: int = 48 * 1024  # Shared memory per block in bytes
    max_threads_per_block: int = 1024  # Maximum threads per block
    memory_bandwidth_gbps: float = 320.0  # Memory bandwidth in GB/s
    enable_gpu_optimizations: bool = True  # Enable GPU optimizations
    use_memory_pooling: bool = True  # Enable memory pooling
    use_hierarchical_memory_compression: bool = True  # Enable hierarchical memory compression
    enable_cuda_graphs: bool = True  # Enable CUDA graphs
    use_context_adaptive_positional_encoding: bool = False  # Enable context adaptive positional encoding
    use_conditional_feature_extraction: bool = False  # Enable conditional feature extraction
    use_hierarchical_vision: bool = False  # Enable hierarchical vision processing
    use_learned_activation_routing: bool = False  # Enable learned activation routing
    use_adaptive_batch_processing: bool = False  # Enable adaptive batch processing
    use_adaptive_sequence_packing: bool = False  # Enable adaptive sequence packing
    use_memory_efficient_grad_accumulation: bool = False  # Enable memory efficient gradient accumulation
    use_faster_rotary_embeddings: bool = False  # Enable faster rotary embeddings
    use_distributed_pipeline_parallelism: bool = False  # Enable distributed pipeline parallelism
    use_hardware_specific_kernels: bool = False  # Enable hardware specific kernels
    performance_improvement_threshold: float = 0.05  # Performance improvement threshold
    accuracy_preservation_threshold: float = 0.95  # Accuracy preservation threshold
    use_cross_layer_parameter_sharing: bool = False  # Enable cross layer parameter sharing
    use_dynamic_sparse_attention: bool = False  # Enable dynamic sparse attention
    use_cross_modal_compression: bool = False  # Enable cross modal compression
    use_cross_layer_memory_sharing: bool = False  # Enable cross layer memory sharing
    l1_cache_size: int = 32 * 1024  # L1 cache size in bytes
    l2_cache_size: int = 256 * 1024  # L2 cache size in bytes
    l3_cache_size: int = 6 * 1024 * 1024  # L3 cache size in bytes
    cache_line_size: int = 64  # Cache line size in bytes
    enable_cpu_optimizations: bool = True  # Enable CPU optimizations
    use_hyperthreading: bool = True  # Use hyperthreading
    enable_simd_optimizations: bool = True  # Enable SIMD optimizations
    simd_instruction_set: str = "avx2"  # SIMD instruction set
    num_preprocess_workers: int = 4  # Number of preprocessing workers
    preprocess_batch_size: int = 8  # Preprocessing batch size
    memory_threshold: float = 0.8  # Memory threshold
    transfer_async: bool = True  # Enable asynchronous transfer
    enable_power_optimization: bool = True  # Enable power optimization
    power_constraint: float = 0.8  # Power constraint
    thermal_constraint: float = 75.0  # Thermal constraint in Celsius
    performance_target: float = 0.9  # Performance target
    adaptation_frequency: float = 1.0  # Adaptation frequency in seconds
    enable_dynamic_power_scaling: bool = True  # Enable dynamic power scaling
    memory_pool_size: int = 2 * 1024 * 1024 * 1024  # Memory pool size in bytes
    memory_pool_dtype: str = "float16"  # Memory pool data type
    memory_pool_device: Optional[str] = None  # Memory pool device
    enable_memory_tiering: bool = True  # Enable memory tiering
    cpu_memory_size: int = 8 * 1024 * 1024 * 1024  # CPU memory size in bytes
    ssd_memory_size: int = 50 * 1024 * 1024 * 1024  # SSD memory size in bytes
    enable_memory_compression: bool = True  # Enable memory compression
    compression_level: str = "medium"  # Compression level
    compression_threshold: float = 0.5  # Compression threshold
    enable_memory_swapping: bool = True  # Enable memory swapping
    swap_threshold: float = 0.8  # Swap threshold
    swap_algorithm: str = "lru"  # Swap algorithm
    enable_memory_defragmentation: bool = True  # Enable memory defragmentation
    defragmentation_interval: int = 1000  # Defragmentation interval
    defragmentation_threshold: float = 0.7  # Defragmentation threshold
    num_threads: int = 4  # Number of threads
    num_workers: int = 4  # Number of workers
    max_concurrent_threads: int = 8  # Maximum concurrent threads
    use_activation_sparsity: bool = False  # Enable activation sparsity
    use_rotary_embedding: bool = True  # Enable rotary embedding
    use_memory_efficient_attention: bool = False  # Enable memory efficient attention
    attention_dropout_prob: float = 0.0  # Attention dropout probability
    use_rotary_embedding: bool = True  # Use rotary embedding
    use_dynamic_sparse_attention: bool = False  # Use dynamic sparse attention
    use_low_rank_kv_cache: bool = True  # Use low rank KV cache
    kv_cache_strategy: str = "hybrid"  # KV cache strategy

    def __post_init__(self):
        """Validate hardware configuration after initialization."""
        if self.gpu_memory_fraction <= 0 or self.gpu_memory_fraction > 1.0:
            raise ValueError(f"gpu_memory_fraction must be between 0 and 1.0, got {self.gpu_memory_fraction}")

        if self.hardware_detection_timeout <= 0:
            raise ValueError(f"hardware_detection_timeout must be positive, got {self.hardware_detection_timeout}")

        valid_gpu_precision = ["fp16", "fp32", "mixed", "bf16"]
        if self.gpu_precision not in valid_gpu_precision:
            raise ValueError(f"gpu_precision must be one of {valid_gpu_precision}, got {self.gpu_precision}")

        if self.cpu_threads is not None and self.cpu_threads <= 0:
            raise ValueError(f"cpu_threads must be positive or None, got {self.cpu_threads}")

        valid_cache_policies = ["lru", "fifo", "lfu"]
        if self.nvme_cache_policy not in valid_cache_policies:
            raise ValueError(f"nvme_cache_policy must be one of {valid_cache_policies}, got {self.nvme_cache_policy}")

        if self.power_limit_watts is not None and self.power_limit_watts <= 0:
            raise ValueError(f"power_limit_watts must be positive or None, got {self.power_limit_watts}")

        if self.thermal_limit_celsius is not None and self.thermal_limit_celsius <= 0:
            raise ValueError(f"thermal_limit_celsius must be positive or None, got {self.thermal_limit_celsius}")

        valid_instruction_sets = ["auto", "avx2", "avx", "sse", "none"]
        if self.vector_instruction_set not in valid_instruction_sets:
            raise ValueError(f"vector_instruction_set must be one of {valid_instruction_sets}, got {self.vector_instruction_set}")

        if self.pipeline_parallelism_degree < 1:
            raise ValueError(f"pipeline_parallelism_degree must be at least 1, got {self.pipeline_parallelism_degree}")

        if self.tensor_parallelism_degree < 1:
            raise ValueError(f"tensor_parallelism_degree must be at least 1, got {self.tensor_parallelism_degree}")

        if self.gpu_device_ids is None:
            self.gpu_device_ids = []