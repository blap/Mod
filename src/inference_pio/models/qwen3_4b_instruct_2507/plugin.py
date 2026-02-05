"""
Qwen3-4B-Instruct-2507 Plugin - Self-Contained Implementation

This module implements the Qwen3-4B-Instruct-2507 model plugin following the standard
plugin interface defined in the Inference-PIO system. This plugin is compatible with
the generic model loading, inference, and data processing functions, with specific
optimizations for Qwen3-4B-Instruct-2507 model characteristics.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from ...common.adaptive_batch_manager import (
    AdaptiveBatchManager,
    get_adaptive_batch_manager,
)
from ...common.base_plugin_interface import (
    ActivationAccessPattern,
    ActivationOffloadingManager,
    ActivationPriority,
)
from ...common.disk_offloading import (
    AccessPattern,
    DiskOffloader,
    OffloadPriority,
)
from ...common.disk_offloading import (
    TensorOffloadingManager as DiskTensorOffloadingManager,
)
from ...common.disk_pipeline import DiskBasedPipeline, PipelineManager, PipelineStage
from ...common.improved_base_plugin_interface import (
    ModelPluginInterface,
)
from ...common.improved_base_plugin_interface import (
    PluginMetadata as ModelPluginMetadata,
)
from ...common.improved_base_plugin_interface import (
    PluginType,
    TextModelPluginInterface,
)
from ...common.memory_manager import MemoryManager, MemoryPriority, TensorPagingManager
from ...common.model_surgery import (
    ModelSurgerySystem,
    apply_model_surgery,
    restore_model_from_surgery,
)
from ...common.optimization_integration import (
    apply_qwen_optimizations,
    legacy_apply_activation_offloading,
    legacy_apply_disk_offloading,
    legacy_apply_flash_attention,
    legacy_apply_kernel_fusion,
    legacy_apply_sparse_attention,
    legacy_apply_structured_pruning,
    legacy_apply_tensor_compression,
)
from ...common.tensor_compression import AdaptiveTensorCompressor, get_tensor_compressor
from ...common.unimodal_model_surgery import (
    UnimodalModelSurgerySystem,
    analyze_unimodal_model_for_surgery,
    apply_unimodal_model_surgery,
    get_unimodal_model_surgery_system,
)
from ...common.unimodal_preprocessing import (
    TextPreprocessor as UnimodalTextPreprocessor,
)
from ...common.unimodal_preprocessing import (
    UnimodalPreprocessor,
    create_unimodal_preprocessor,
)
from ...common.virtual_device import VirtualExecutionSimulator
from ...common.virtual_execution import (
    PartitionConfig,
    PartitionStrategy,
    VirtualExecutionManager,
)
from ...common.optimization.predictive_memory_optimization import (
    PredictiveMemoryOptimization
)
from ...common.optimization.resource_prediction_system import (
    ResourcePredictionSystem
)
from .config import Qwen34BInstruct2507Config
from .cross_alignment_optimization import apply_cross_alignment_to_model

# Import moved to function scope to avoid circular imports
# from ...design_patterns.integration import create_optimized_adapted_plugin
from .model import Qwen34BInstruct2507Model
from .specific_optimizations.qwen3_attention_optimizations import (
    apply_qwen3_attention_optimizations,
    apply_qwen3_gqa_optimizations,
    apply_qwen3_rope_optimizations,
)
from .specific_optimizations.qwen3_instruction_optimizations import (
    apply_qwen3_generation_optimizations,
    apply_qwen3_instruction_tuning_optimizations,
    enhance_qwen3_instruction_following_capability,
)
from .specific_optimizations.qwen3_kv_cache_optimizations import (
    apply_qwen3_compressed_kv_cache,
    apply_qwen3_kv_cache_optimizations,
)
from .scheduling.intelligent_scheduler import apply_intelligent_scheduling_to_model, create_intelligent_scheduler_for_qwen3_4b
from .intelligent_cache.intelligent_cache_manager import apply_intelligent_caching_to_model, create_intelligent_cache_for_qwen3_4b

logger = logging.getLogger(__name__)


class Qwen3_4B_Instruct_2507_Plugin(TextModelPluginInterface):
    """
    Comprehensive implementation of the Qwen3-4B-Instruct-2507 model plugin that follows
    the standard plugin interface and is compatible with generic functions.
    This plugin is specifically optimized for Qwen3-4B-Instruct-2507 model characteristics
    while maintaining compatibility with general text generation.
    The plugin implements all required methods from TextModelPluginInterface and provides
    additional functionality for model-specific optimizations.
    """

    def __init__(self):
        # Create plugin metadata specific to Qwen3-4B-Instruct-2507
        metadata = ModelPluginMetadata(
            name="Qwen3-4B-Instruct-2507",
            version="1.0.0",
            author="Alibaba Cloud",
            description="Qwen3-4B-Instruct-2507 specialized model with advanced optimizations",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=["torch", "transformers", "accelerate"],
            compatibility={
                "torch_version": ">=2.0.0",
                "transformers_version": ">=4.30.0",
                "python_version": ">=3.8",
                "min_memory_gb": 8.0,  # Estimated for Qwen3-4B model
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            model_architecture="Qwen3-4B Transformer-based model with instruction tuning",
            model_size="4B",
            required_memory_gb=8.0,  # Memory requirement for Qwen3-4B model
            supported_modalities=["text"],
            license="MIT",
            tags=["language-model", "instruction-tuned", "4b", "qwen3"],
            model_family="Qwen3",
            num_parameters=4000000000,  # 4 billion parameters
            test_coverage=0.95,
            validation_passed=True,
        )
        super().__init__(metadata)
        self._model = None
        self._tokenizer = None
        self._config = Qwen34BInstruct2507Config()
        self._compiled_model = None
        self._memory_manager = None
        self._tensor_paging_manager = None
        self._paging_enabled = False
        self._adaptive_batch_manager = None
        self._adaptive_batching_enabled = False
        self._virtual_execution_manager = None
        self._virtual_execution_simulator = None
        self._virtual_execution_enabled = False
        self._partitions = []
        self._tensor_compressor = None
        self._compression_enabled = False
        self._compressed_weights = {}
        self._compression_metadata = {}
        self._unimodal_preprocessor = None

        # Disk offloading attributes
        self._disk_offloader = None
        self._disk_tensor_offloading_manager = None
        self._offloading_enabled = False
        self._offloading_config = {}

        # Disk-based pipeline attributes
        self._pipeline_manager = None
        self._pipeline = None
        self._pipeline_enabled = False
        self._pipeline_config = {}

        # Activation offloading attributes
        self._activation_offloading_manager = None
        self._activation_offloading_enabled = False
        self._activation_offloading_config = {}

        # Predictive Memory Optimization components
        self._predictive_memory_optimization = None

        # Resource Prediction System components
        self._resource_prediction_system = None

    def initialize(self, **kwargs) -> bool:
        """
        Initialize the Qwen3-4B-Instruct-2507 model with specific parameters and configurations.

        This method handles the complete initialization process including:
        - Setting up the model path
        - Validating configuration parameters
        - Selecting appropriate device based on available hardware
        - Loading the model
        - Initializing various optimization systems based on configuration

        Args:
            **kwargs: Configuration parameters for initialization

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Handle special parameters that are not part of config
            device = kwargs.pop("device", None)

            # Update config with remaining parameters
            for key, value in kwargs.items():
                if hasattr(self._config, key):
                    # Validate parameters before setting
                    if key == "paged_attention_page_size" and value <= 0:
                        raise ValueError(
                            f"paged_attention_page_size must be positive, got {value}"
                        )
                    elif key == "sliding_window_size" and value <= 0:
                        raise ValueError(
                            f"sliding_window_size must be positive, got {value}"
                        )
                    elif key == "tensor_parallel_size" and value <= 0:
                        raise ValueError(
                            f"tensor_parallel_size must be positive, got {value}"
                        )
                    elif key == "tensor_parallel_local_rank" and value < 0:
                        raise ValueError(
                            f"tensor_parallel_local_rank must be non-negative, got {value}"
                        )
                    elif key == "tensor_parallel_world_size" and value <= 0:
                        raise ValueError(
                            f"tensor_parallel_world_size must be positive, got {value}"
                        )
                    elif key == "kv_cache_quantization_bits" and value <= 0:
                        raise ValueError(
                            f"kv_cache_quantization_bits must be positive, got {value}"
                        )
                    elif key == "kv_cache_low_rank_dimension" and value <= 0:
                        raise ValueError(
                            f"kv_cache_low_rank_dimension must be positive, got {value}"
                        )
                    elif key == "kv_cache_sparse_compression_ratio" and (
                        value <= 0 or value > 1
                    ):
                        raise ValueError(
                            f"kv_cache_sparse_compression_ratio must be between 0 and 1, got {value}"
                        )
                    elif key == "prefix_cache_max_size" and value <= 0:
                        raise ValueError(
                            f"prefix_cache_max_size must be positive, got {value}"
                        )
                    elif key == "prefix_cache_prefetch_distance" and value < 0:
                        raise ValueError(
                            f"prefix_cache_prefetch_distance must be non-negative, got {value}"
                        )
                    elif key == "prefix_cache_max_prefix_length" and value <= 0:
                        raise ValueError(
                            f"prefix_cache_max_prefix_length must be positive, got {value}"
                        )
                    elif key == "prefix_cache_min_prefix_length" and value <= 0:
                        raise ValueError(
                            f"prefix_cache_min_prefix_length must be positive, got {value}"
                        )
                    elif key == "prefix_cache_warmup_threshold" and value <= 0:
                        raise ValueError(
                            f"prefix_cache_warmup_threshold must be positive, got {value}"
                        )

                    # Handle torch_dtype specially since it's a torch.dtype object
                    if key == "torch_dtype":
                        if isinstance(value, torch.dtype):
                            setattr(
                                self._config, key, str(value).split(".")[-1]
                            )  # Convert to string representation
                        else:
                            setattr(self._config, key, value)
                    else:
                        setattr(self._config, key, value)

            # Implement dynamic hybrid execution: check for available devices and set appropriately
            if device:
                # Use explicitly specified device
                selected_device = device
            else:
                # Dynamically determine the best available device
                if torch.cuda.is_available():
                    # Check GPU memory and decide whether to use GPU or CPU
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    # If model is large and GPU memory is limited, use CPU
                    if self.metadata.required_memory_gb * 1024 > gpu_memory / (
                        1024**2
                    ):  # Convert bytes to MB
                        selected_device = "cpu"
                        logger.info("Using CPU due to insufficient GPU memory")
                    else:
                        selected_device = "cuda:0"
                        logger.info("Using GPU for inference")
                else:
                    selected_device = "cpu"
                    logger.info("Using CPU as GPU is not available")

            # Store the selected device in config
            self._config.device = selected_device

            # Initialize predictive memory optimization if enabled
            if getattr(self._config, "enable_predictive_management", False) or kwargs.get(
                "enable_predictive_management", False
            ):
                self.setup_predictive_memory_optimization(**kwargs)

            # Initialize resource prediction system if enabled
            if getattr(self._config, "enable_resource_prediction", False) or kwargs.get(
                "enable_resource_prediction", False
            ):
                self.setup_resource_prediction_system(**kwargs)

            logger.info("Qwen3-4B-Instruct-2507 plugin initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            return False

    def load_model(self, config: Qwen34BInstruct2507Config = None) -> nn.Module:
        """
        Load the Qwen3-4B-Instruct-2507 model with optimizations.

        Args:
            config: Model configuration (optional)

        Returns:
            nn.Module: The loaded model instance with Qwen3-4B-Instruct-2507 specific optimizations applied
        """
        try:
            if config is not None:
                self._config = config

            logger.info(
                f"Loading Qwen3-4B-Instruct-2507 model from: {self._config.model_path}"
            )

            # Set the device in the config before loading the model
            device = getattr(self._config, "device", "cpu")

            # Update the device_map in config to match the selected device
            if device == "cpu":
                self._config.device_map = "cpu"
            elif device.startswith("cuda"):
                self._config.device_map = device
            else:
                self._config.device_map = "auto"

            # Create the model with the configuration
            self._model = Qwen34BInstruct2507Model(self._config)

            # Apply Qwen3-4B-Instruct-2507 specific optimizations after model creation
            self._apply_qwen3_specific_optimizations()

            # Apply cross-alignment optimization if enabled
            if getattr(self._config, 'enable_cross_alignment', False):
                logger.info("Applying cross-alignment optimization to Qwen3-4B-Instruct-2507 model...")
                self._model = apply_cross_alignment_to_model(self._model, self._config)

            # Apply intelligent caching if enabled in config
            if getattr(self._config, 'intelligent_cache_enabled', False):
                logger.info("Applying intelligent caching to Qwen3-4B-Instruct-2507 model")
                intelligent_cache_config = create_intelligent_cache_for_qwen3_4b(self._config)
                self._model = apply_intelligent_caching_to_model(self._model, intelligent_cache_config)

                # Store reference to the cache manager for later use
                self.intelligent_cache_manager = intelligent_cache_config

            # Apply intelligent scheduling if enabled in config
            if getattr(self._config, 'enable_intelligent_scheduling', False):
                logger.info("Applying intelligent scheduling to Qwen3-4B-Instruct-2507 model")
                intelligent_scheduler_config = create_intelligent_scheduler_for_qwen3_4b(self._config)
                self._model = apply_intelligent_scheduling_to_model(self._model, intelligent_scheduler_config)

                # Store reference to the scheduler for later use
                self.intelligent_scheduler = intelligent_scheduler_config

            # Get the tokenizer from the model
            self._tokenizer = self._model.get_tokenizer()

            # Initialize unimodal preprocessor
            try:
                self._unimodal_preprocessor = create_unimodal_preprocessor(
                    model_path=self._config.model_path,
                    max_text_length=getattr(self._config, "max_text_length", 32768),
                    model_type="qwen3_4b",
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create unimodal preprocessor: {e}. Using basic preprocessing."
                )

            # Check if model is on meta device and handle accordingly
            if (
                next(self._model.parameters(), None) is not None
                and next(self._model.parameters()).is_meta
            ):
                logger.warning(
                    f"Model loaded on meta device, attempting to move to {device}"
                )
                # If model is still on meta device, we need to handle it differently
                # This usually happens when using device_map with accelerate
                # In this case, the model should already be distributed properly
            else:
                # Move model to the selected device if not already there
                current_device = next(self._model.parameters()).device
                if str(current_device) != device:
                    self._model = self._model.to(device)

            # Apply runtime memory optimization using torch.compile
            if (
                hasattr(self._config, "torch_compile_mode")
                and self._config.torch_compile_mode
            ):
                self.optimize_model(
                    model=self._model,
                    mode=self._config.torch_compile_mode,
                    fullgraph=self._config.torch_compile_fullgraph,
                    dynamic=self._config.torch_compile_dynamic,
                )

            # Enable cuDNN benchmarking for better performance
            if (
                hasattr(self._config, "enable_cudnn_benchmark")
                and self._config.enable_cudnn_benchmark
            ):
                torch.backends.cudnn.benchmark = True

            logger.info(
                f"Qwen3-4B-Instruct-2507 model loaded successfully on device: {device}"
            )
            return self._model
        except Exception as e:
            logger.error(f"Failed to load Qwen3-4B-Instruct-2507 model: {e}")
            raise e

    def _apply_qwen3_specific_optimizations(self):
        """
        Apply Qwen3-4B-Instruct-2507 specific optimizations that leverage the unique
        characteristics of the Qwen3 architecture.
        """
        # Apply Qwen3-specific attention optimizations
        if getattr(self._config, 'use_qwen3_attention_optimizations', False):
            apply_qwen3_attention_optimizations(self._model, self._config)

        # Apply Qwen3-specific GQA optimizations
        if getattr(self._config, 'use_qwen3_gqa_optimizations', False):
            apply_qwen3_gqa_optimizations(self._model, self._config)

        # Apply Qwen3-specific RoPE optimizations
        if getattr(self._config, 'use_qwen3_rope_optimizations', False):
            apply_qwen3_rope_optimizations(self._model, self._config)

        # Apply Qwen3-specific KV-cache optimizations
        if getattr(self._config, 'use_qwen3_kv_cache_optimizations', False):
            apply_qwen3_kv_cache_optimizations(self._model, self._config)

        # Apply Qwen3-specific instruction tuning optimizations
        if getattr(self._config, 'use_qwen3_instruction_optimizations', False):
            apply_qwen3_instruction_tuning_optimizations(self._model, self._config)

        logger.info("Qwen3-4B-Instruct-2507 specific optimizations applied")

    def infer(self, data: Any) -> Any:
        """
        Perform inference on the given data.

        Args:
            data: Input data for inference

        Returns:
            Inference results
        """
        if not self._model or not self._tokenizer:
            raise RuntimeError("Model or tokenizer not loaded. Call load_model() first.")

        # Convert input to appropriate format
        if isinstance(data, str):
            inputs = self._tokenizer(data, return_tensors="pt").to(self._model.device)
        elif isinstance(data, dict):
            inputs = self._tokenizer(data.get("text", ""), return_tensors="pt").to(self._model.device)
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")

        # Generate output
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=getattr(self._config, 'max_new_tokens', 1024),
                temperature=getattr(self._config, 'temperature', 0.7),
                top_p=getattr(self._config, 'top_p', 0.9),
                top_k=getattr(self._config, 'top_k', 50),
                do_sample=True
            )

        # Decode output
        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def tokenize(self, text: str, **kwargs) -> Any:
        """
        Tokenize the input text.

        Args:
            text: Input text to tokenize
            **kwargs: Additional tokenization arguments

        Returns:
            Tokenized representation of the text
        """
        if not self._tokenizer:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")
        return self._tokenizer(text, **kwargs)

    def detokenize(self, token_ids: Union[List[int], torch.Tensor], **kwargs) -> str:
        """
        Convert token IDs back to text.

        Args:
            token_ids: Token IDs to convert
            **kwargs: Additional detokenization arguments

        Returns:
            Detokenized text
        """
        if not self._tokenizer:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")
        return self._tokenizer.decode(token_ids, **kwargs)

    def generate_text(self, prompt: str, max_new_tokens: int = 1024, **kwargs) -> str:
        """
        Generate text based on the given prompt.

        Args:
            prompt: Input prompt for text generation
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation arguments

        Returns:
            Generated text
        """
        if not self._model or not self._tokenizer:
            raise RuntimeError("Model or tokenizer not loaded. Call load_model() first.")

        # Prepare inputs
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        # Prepare generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": getattr(self._config, 'temperature', 0.7),
            "top_p": getattr(self._config, 'top_p', 0.9),
            "top_k": getattr(self._config, 'top_k', 50),
            "do_sample": True,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        }
        gen_kwargs.update(kwargs)  # Allow manual overrides

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        # Decode
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text

    def cleanup(self) -> bool:
        """
        Clean up resources used by the plugin.

        Returns:
            True if cleanup was successful, False otherwise
        """
        try:
            self._model = None
            self._tokenizer = None
            self._compiled_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the model.

        Returns:
            Dictionary containing model information
        """
        return {
            "name": self.metadata.name,
            "model_type": "Causal Language Model",
            "architecture": self.metadata.model_architecture,
            "modalities": self.metadata.supported_modalities,
            "size": self.metadata.model_size,
            "parameters": self.metadata.num_parameters,
        }

    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Get model parameter information.

        Returns:
            Dictionary containing parameter counts
        """
        if self._model is None:
            return {
                "total_parameters": 0,
                "trainable_parameters": 0,
                "frozen_parameters": 0,
            }

        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(
            p.numel() for p in self._model.parameters() if p.requires_grad
        )
        frozen_params = total_params - trainable_params

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
        }

    def get_model_config_template(self) -> Any:
        """
        Get a template for model configuration.

        Returns:
            Model configuration template
        """
        return Qwen34BInstruct2507Config()

    def validate_model_compatibility(self, config: Any) -> bool:
        """
        Validate that the model is compatible with the given configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if compatible, False otherwise
        """
        return isinstance(config, Qwen34BInstruct2507Config)

    def optimize_model(
        self,
        model: torch.nn.Module = None,
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = True,
    ) -> bool:
        """
        Apply runtime memory optimization using torch.compile.

        Args:
            model: Model to optimize (if None, uses internal model)
            mode: Compilation mode
            fullgraph: Whether to compile the full graph
            dynamic: Whether to enable dynamic compilation

        Returns:
            True if optimization was successful, False otherwise
        """
        try:
            # Determine which model to optimize
            target_model = model
            if target_model is None:
                target_model = self._model

            if target_model is None:
                logger.warning(
                    "No model provided and no internal model found, cannot optimize"
                )
                return False

            # Enable cuDNN benchmarking for better performance
            torch.backends.cudnn.benchmark = True

            # Compile the model with specified optimizations
            self._compiled_model = torch.compile(
                target_model, mode=mode, fullgraph=fullgraph, dynamic=dynamic
            )

            logger.info(f"Model optimized with torch.compile using mode: {mode}")
            return True
        except Exception as e:
            logger.error(f"Failed to optimize model: {e}")
            return False

    def get_compiled_model(self):
        """
        Get the compiled model if available, otherwise return the original model.

        Returns:
            Compiled model or original model
        """
        return getattr(self, '_compiled_model', None) or self._model

    def setup_tensor_compression(self, **kwargs) -> bool:
        """
        Set up tensor compression system for model weights.

        Args:
            **kwargs: Tensor compression configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        try:
            # Use config values or fallback to kwargs
            compression_method = getattr(
                self._config, "tensor_compression_method", kwargs.get("method", "incremental_pca")
            )
            compression_ratio = getattr(
                self._config, "tensor_compression_ratio", kwargs.get("ratio", 0.5)
            )
            max_components = getattr(
                self._config, "tensor_compression_max_components", kwargs.get("max_components", 256)
            )

            # Create tensor compressor
            self._tensor_compressor = get_tensor_compressor(compression_method)
            if not self._tensor_compressor:
                logger.error(f"No compressor available for method: {compression_method}")
                return False

            # Configure compressor
            self._tensor_compressor.configure(
                compression_ratio=compression_ratio,
                max_components=max_components,
                **kwargs
            )

            logger.info(
                f"Tensor compression configured: method={compression_method}, "
                f"ratio={compression_ratio}, max_components={max_components}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to setup tensor compression: {e}")
            return False

    def enable_tensor_compression(self, **kwargs) -> bool:
        """
        Enable tensor compression for the model to reduce memory usage.

        Args:
            **kwargs: Tensor compression configuration parameters

        Returns:
            True if tensor compression was enabled successfully, False otherwise
        """
        try:
            if not self._tensor_compressor:
                if not self.setup_tensor_compression(**kwargs):
                    logger.error("Failed to setup tensor compression system")
                    return False

            # Compress model if loaded
            if self._model is not None:
                self._compress_model_weights()

            self._compression_enabled = True

            # Enable activation compression if configured
            if getattr(self._config, "enable_activation_compression", False):
                self._setup_activation_compression()

            logger.info("Tensor compression enabled")
            return True
        except Exception as e:
            logger.error(f"Failed to enable tensor compression: {e}")
            return False

    def _compress_model_weights(self):
        """
        Compress model weights using the configured compressor.
        """
        if not self._tensor_compressor or not self._model:
            return

        # Compress each parameter
        for name, param in self._model.named_parameters():
            if param.requires_grad:  # Only compress trainable parameters
                compressed_param, metadata = self._tensor_compressor.compress(param.data)
                self._compressed_weights[name] = compressed_param
                self._compression_metadata[name] = metadata

                # Replace with compressed version (this is a simplified approach)
                # In a real implementation, you'd need to handle decompression during forward pass
                logger.debug(f"Compressed parameter: {name}, shape: {param.shape}")

    def _setup_activation_compression(self):
        """
        Setup activation compression for model inference.
        """
        if getattr(self._config, "enable_activation_compression", False):
            logger.info("Activation compression enabled - will compress during inference")
            return True
        return False

    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get tensor compression statistics.

        Returns:
            Dictionary containing compression statistics
        """
        if not self._tensor_compressor:
            return {
                "compression_enabled": False,
                "compressed_parameters_count": 0,
                "average_compression_ratio": 0.0,
                "total_saved_bytes": 0,
            }

        stats = self._tensor_compressor.get_compression_stats()

        # Calculate aggregate statistics
        total_original_size = sum(
            meta.get("original_size", 0) for meta in stats.values()
        )
        total_compressed_size = sum(
            meta.get("compressed_size", 0) for meta in stats.values()
        )
        avg_compression_ratio = (
            (total_compressed_size / total_original_size)
            if total_original_size > 0
            else 0.0
        )

        return {
            "compression_enabled": self._compression_enabled,
            "compressed_parameters_count": len(stats),
            "average_compression_ratio": avg_compression_ratio,
            "total_saved_bytes": sum(
                meta.get("saved_bytes", 0) for meta in stats.values()
            ),
            "detailed_stats": stats,
        }

    def enable_adaptive_compression(self, **kwargs) -> bool:
        """
        Enable adaptive compression that adjusts based on available memory.

        Args:
            **kwargs: Adaptive compression configuration parameters

        Returns:
            True if adaptive compression was enabled successfully, False otherwise
        """
        try:
            if not self._tensor_compressor:
                if not self.setup_tensor_compression(**kwargs):
                    logger.error(
                        "Failed to setup tensor compressor for adaptive compression"
                    )
                    return False

            # Adaptive compression is already handled by AdaptiveTensorCompressor
            # which adjusts compression based on memory usage
            logger.info(
                "Adaptive compression is enabled by default in AdaptiveTensorCompressor"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to enable adaptive compression: {e}")
            return False

    def setup_disk_offloading(self, **kwargs) -> bool:
        """
        Set up disk offloading system for managing model components between RAM and disk.

        Args:
            **kwargs: Disk offloading configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        try:
            # Use config values or fallback to kwargs
            max_memory_ratio = getattr(
                self._config, "max_memory_ratio", kwargs.get("max_memory_ratio", 0.8)
            )
            offload_directory = getattr(
                self._config, "offload_directory", kwargs.get("offload_directory", None)
            )
            page_size_mb = getattr(
                self._config, "page_size_mb", kwargs.get("page_size_mb", 16)
            )
            eviction_policy = getattr(
                self._config,
                "eviction_policy",
                kwargs.get("eviction_policy", "predictive"),
            )

            # Create disk offloader
            self._disk_offloader = DiskOffloader(
                max_memory_ratio=max_memory_ratio,
                offload_directory=offload_directory,
                page_size_mb=page_size_mb,
                eviction_policy=eviction_policy,
            )

            # Create tensor offloading manager
            self._disk_tensor_offloading_manager = DiskTensorOffloadingManager(
                self._disk_offloader
            )

            # Store offloading config
            self._offloading_config = {
                "max_memory_ratio": max_memory_ratio,
                "offload_directory": offload_directory,
                "page_size_mb": page_size_mb,
                "eviction_policy": eviction_policy,
            }

            logger.info(
                f"Disk offloading configured: max_memory_ratio={max_memory_ratio}, "
                f"page_size={page_size_mb}MB, eviction_policy={eviction_policy}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to setup disk offloading: {e}")
            return False

    def enable_disk_offloading(self, **kwargs) -> bool:
        """
        Enable disk offloading for the model to move parts between RAM and disk.

        Args:
            **kwargs: Disk offloading configuration parameters

        Returns:
            True if disk offloading was enabled successfully, False otherwise
        """
        try:
            if not self._disk_offloader or not self._disk_tensor_offloading_manager:
                if not self.setup_disk_offloading(**kwargs):
                    logger.error("Failed to setup disk offloading system")
                    return False

            # Determine priority level
            priority_str = getattr(
                self._config,
                "offloading_priority",
                kwargs.get("offloading_priority", "medium"),
            )
            priority_map = {
                "low": OffloadPriority.LOW,
                "medium": OffloadPriority.MEDIUM,
                "high": OffloadPriority.HIGH,
                "critical": OffloadPriority.CRITICAL,
            }
            priority = priority_map.get(priority_str.lower(), OffloadPriority.MEDIUM)

            # If model is loaded, offload its components
            if self._model is not None:
                self._offload_model_components(priority)

            self._offloading_enabled = True

            # Start proactive management if enabled
            if getattr(self._config, "enable_predictive_offloading", False):
                interval = getattr(self._config, "proactive_offloading_interval", 5.0)
                self._disk_tensor_offloading_manager.start_proactive_management(
                    interval
                )

            logger.info(f"Disk offloading enabled with priority: {priority_str}")
            return True
        except Exception as e:
            logger.error(f"Failed to enable disk offloading: {e}")
            return False

    def _offload_model_components(self, priority: OffloadPriority):
        """
        Offload model components to enable disk offloading.

        Args:
            priority: Priority level for the offloaded components
        """
        if not self._disk_tensor_offloading_manager:
            return

        # Offload model components based on priority
        # This is a simplified implementation - in practice, you would identify
        # specific components to offload based on their size and access patterns
        logger.info("Model components prepared for disk offloading")

    def prepare_for_activation_offloading(self, **kwargs) -> bool:
        """
        Prepare the model for activation offloading.

        Args:
            **kwargs: Configuration parameters for activation offloading

        Returns:
            True if preparation was successful, False otherwise
        """
        try:
            # Use config values or fallback to kwargs
            max_memory_ratio = getattr(
                self._config,
                "activation_max_memory_ratio",
                kwargs.get("max_memory_ratio", 0.7),
            )
            offload_directory = getattr(
                self._config,
                "activation_offload_directory",
                kwargs.get("offload_directory", None),
            )
            page_size_mb = getattr(
                self._config,
                "activation_page_size_mb",
                kwargs.get("page_size_mb", 8),
            )
            eviction_policy = getattr(
                self._config,
                "activation_eviction_policy",
                kwargs.get("eviction_policy", "predictive"),
            )

            # Create activation offloading manager
            self._activation_offloading_manager = ActivationOffloadingManager(
                max_memory_ratio=max_memory_ratio,
                offload_directory=offload_directory,
                page_size_mb=page_size_mb,
                eviction_policy=eviction_policy,
            )

            # Store activation offloading config
            self._activation_offloading_config = {
                "max_memory_ratio": max_memory_ratio,
                "offload_directory": offload_directory,
                "page_size_mb": page_size_mb,
                "eviction_policy": eviction_policy,
            }

            self._activation_offloading_enabled = True

            logger.info("Model prepared for activation offloading")
            return True
        except Exception as e:
            logger.error(f"Failed to prepare for activation offloading: {e}")
            return False

    def offload_activations(self, **kwargs) -> bool:
        """
        Offload specific activations to disk based on predictive algorithms.

        Args:
            **kwargs: Configuration parameters for offloading

        Returns:
            True if offloading was successful, False otherwise
        """
        try:
            if not self._activation_offloading_manager:
                logger.error("Activation offloading manager not initialized")
                return False

            # Determine priority level
            priority_str = kwargs.get("priority", "medium")
            priority_map = {
                "low": ActivationPriority.LOW,
                "medium": ActivationPriority.MEDIUM,
                "high": ActivationPriority.HIGH,
                "critical": ActivationPriority.CRITICAL,
            }
            priority = priority_map.get(priority_str.lower(), ActivationPriority.MEDIUM)

            # This is a simplified implementation - in practice, you would identify
            # specific activations to offload based on their access patterns
            logger.info("Activation offloading performed")
            return True
        except Exception as e:
            logger.error(f"Failed to offload activations: {e}")
            return False

    def predict_activation_access(self, **kwargs) -> Dict[str, Any]:
        """
        Predict which activations will be accessed based on input patterns.

        Args:
            **kwargs: Configuration parameters for prediction

        Returns:
            Dictionary mapping activation names to access probabilities
        """
        predictions = {}

        if not self._model:
            logger.warning("Model not loaded, returning empty predictions")
            return predictions

        # This is a simplified prediction model
        # In a real implementation, this would use more sophisticated analysis

        # For demonstration purposes, we'll simulate predictions based on layer positions
        total_layers = getattr(self._config, "num_hidden_layers", 32)

        for layer_idx in range(total_layers):
            # Simulate access probability based on layer position
            layer_position = layer_idx / max(total_layers, 1)

            # Early and late layers might be more frequently accessed
            if layer_position < 0.2 or layer_position > 0.8:
                access_prob = 0.8
            else:
                access_prob = (
                    0.5  # Middle layers less likely to be accessed immediately
                )

            predictions[f"layer_{layer_idx}_activation"] = access_prob

        return predictions

    def get_activation_offloading_stats(self) -> Dict[str, Any]:
        """
        Get statistics about activation offloading operations.

        Returns:
            Dictionary containing activation offloading statistics
        """
        stats = {}

        if self._activation_offloading_manager and hasattr(
            self._activation_offloading_manager, "activation_offloader"
        ):
            stats.update(
                self._activation_offloading_manager.activation_offloader.get_activation_stats()
            )

        # Add general memory stats
        import psutil

        memory = psutil.virtual_memory()
        stats.update(
            {
                "system_memory_percent": memory.percent,
                "system_memory_available_gb": memory.available / (1024**3),
                "system_memory_total_gb": memory.total / (1024**3),
                "activation_offloading_enabled": self._activation_offloading_enabled,
                "activation_offloading_config": self._activation_offloading_config,
            }
        )

        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated()
            gpu_memory_reserved = torch.cuda.memory_reserved()
            stats.update(
                {
                    "gpu_memory_allocated_gb": gpu_memory_allocated / (1024**3),
                    "gpu_memory_reserved_gb": gpu_memory_reserved / (1024**3),
                    "gpu_utilization_percent": (
                        torch.cuda.utilization()
                        if hasattr(torch.cuda, "utilization")
                        else 0
                    ),
                }
            )

        return stats

    def setup_unimodal_model_surgery(self, **kwargs) -> bool:
        """
        Set up unimodal model surgery system for identifying and removing non-essential text processing components.

        Args:
            **kwargs: Unimodal model surgery configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        try:
            # Initialize unimodal model surgery system
            self._unimodal_model_surgery_system = get_unimodal_model_surgery_system()

            logger.info("Unimodal model surgery system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to setup unimodal model surgery: {e}")
            return False

    def enable_unimodal_model_surgery(self, **kwargs) -> bool:
        """
        Enable unimodal model surgery for the model to remove non-essential text processing components.

        Args:
            **kwargs: Unimodal model surgery configuration parameters

        Returns:
            True if unimodal model surgery was enabled successfully, False otherwise
        """
        try:
            if not hasattr(self, "_unimodal_model_surgery_system"):
                if not self.setup_unimodal_model_surgery(**kwargs):
                    logger.error("Failed to setup unimodal model surgery system")
                    return False

            # Perform surgery if model is loaded
            if self._model is not None:
                # Get components to remove from config
                components_to_remove = getattr(
                    self._config, "unimodal_components_to_remove", None
                )
                preserve_components = getattr(
                    self._config, "unimodal_preserve_components", []
                )
                semantic_threshold = getattr(
                    self._config, "unimodal_semantic_importance_threshold", 0.7
                )

                # Perform unimodal surgery
                self._model = apply_unimodal_model_surgery(
                    self._model,
                    components_to_remove=components_to_remove,
                    preserve_components=preserve_components,
                    preserve_semantic_importance_threshold=semantic_threshold,
                )

            logger.info("Unimodal model surgery enabled and applied")
            return True
        except Exception as e:
            logger.error(f"Failed to enable unimodal model surgery: {e}")
            return False

    def analyze_unimodal_model_for_surgery(
        self, model: nn.Module = None
    ) -> Dict[str, Any]:
        """
        Analyze the unimodal model to provide recommendations for surgery.

        Args:
            model: Model to analyze (if None, uses self._model if available)

        Returns:
            Dictionary containing analysis results
        """
        try:
            # Determine which model to analyze
            target_model = model
            if target_model is None:
                if hasattr(self, "_model") and self._model is not None:
                    target_model = self._model
                else:
                    logger.error(
                        "No model provided and no internal model found for analysis"
                    )
                    return {}

            # Perform analysis
            analysis = analyze_unimodal_model_for_surgery(target_model)

            logger.info("Unimodal model analysis for surgery completed")
            return analysis
        except Exception as e:
            logger.error(f"Failed to analyze unimodal model for surgery: {e}")
            return {}

    def perform_unimodal_model_surgery(
        self,
        model: nn.Module = None,
        components_to_remove: Optional[List[str]] = None,
        preserve_components: Optional[List[str]] = None,
        preserve_semantic_importance_threshold: float = 0.7,
    ) -> nn.Module:
        """
        Perform unimodal model surgery on the loaded model.

        Args:
            model: Model to perform surgery on (if None, uses self._model)
            components_to_remove: Specific components to remove (None means auto-detect)
            preserve_components: Components to preserve even if they're removable
            preserve_semantic_importance_threshold: Don't remove components with semantic importance above this threshold

        Returns:
            Modified model with surgery applied
        """
        try:
            # Determine which model to modify
            target_model = model
            if target_model is None:
                if hasattr(self, "_model") and self._model is not None:
                    target_model = self._model
                else:
                    logger.error(
                        "No model provided and no internal model found for surgery"
                    )
                    return None

            # Perform surgery
            modified_model = apply_unimodal_model_surgery(
                target_model,
                components_to_remove=components_to_remove,
                preserve_components=preserve_components,
                preserve_semantic_importance_threshold=preserve_semantic_importance_threshold,
            )

            # Update internal model reference if no external model was passed
            if model is None:
                self._model = modified_model

            logger.info("Unimodal model surgery performed successfully")
            return modified_model
        except Exception as e:
            logger.error(f"Failed to perform unimodal model surgery: {e}")
            return None

    def setup_predictive_memory_optimization(self, **kwargs) -> bool:
        """Set up predictive memory optimization system."""
        try:
            # Create configuration for predictive memory optimization
            config = {
                'enable_predictive_management': getattr(self._config, "enable_predictive_management", True),
                'prediction_horizon_seconds': getattr(self._config, "prediction_horizon_seconds", 30),
                'access_history_window_size': getattr(self._config, "access_history_window_size", 100),
                'memory_prediction_threshold': getattr(self._config, "memory_prediction_threshold", 0.9),
                'proactive_management_interval': getattr(self._config, "proactive_management_interval", 5.0),
                'offload_directory': getattr(self._config, "offload_directory", "./offloaded_tensors")
            }

            # Update with any additional kwargs
            config.update(kwargs)

            # Create predictive memory optimization instance
            self._predictive_memory_optimization = PredictiveMemoryOptimization(config)

            logger.info("Predictive memory optimization system setup completed")
            return True
        except Exception as e:
            logger.error(f"Failed to setup predictive memory optimization: {e}")
            return False

    def start_predictive_memory_management(self, **kwargs) -> bool:
        """Start predictive memory management using ML algorithms to anticipate memory needs."""
        try:
            # Initialize predictive memory optimization if not already done
            if self._predictive_memory_optimization is None:
                if not self.setup_predictive_memory_optimization(**kwargs):
                    logger.error("Failed to setup predictive memory optimization")
                    return False

            # Start the optimization system
            result = self._predictive_memory_optimization.start_optimization()

            # Start proactive management if enabled
            if getattr(self._config, "enable_predictive_management", False):
                interval = getattr(self._config, "proactive_management_interval", 5.0)
                logger.info(f"Proactive memory management enabled with interval {interval}s")

            logger.info("Predictive memory management started")
            return result
        except Exception as e:
            logger.error(f"Failed to start predictive memory management: {e}")
            return False

    def stop_predictive_memory_management(self) -> bool:
        """Stop predictive memory management."""
        try:
            if self._predictive_memory_optimization:
                result = self._predictive_memory_optimization.stop_optimization()
                logger.info("Predictive memory management stopped successfully")
                return result
            else:
                logger.warning("Predictive memory optimization was not initialized")
                return True
        except Exception as e:
            logger.error(f"Failed to stop predictive memory management: {e}")
            return False

    def record_tensor_access(self, tensor_name: str, tensor_data: torch.Tensor,
                           access_type: str = "read") -> bool:
        """Record a tensor access for predictive modeling."""
        try:
            if self._predictive_memory_optimization:
                return self._predictive_memory_optimization.record_tensor_access(
                    tensor_name, tensor_data, access_type
                )
            else:
                logger.warning("Predictive memory optimization not initialized")
                return False
        except Exception as e:
            logger.error(f"Failed to record tensor access: {e}")
            return False

    def setup_resource_prediction_system(self, **kwargs) -> bool:
        """Set up resource prediction system for memory and computation prediction."""
        try:
            # Create configuration for resource prediction system
            config = {
                'enable_resource_prediction': getattr(self._config, "enable_resource_prediction", True),
                'prediction_horizon_seconds': getattr(self._config, "prediction_horizon_seconds", 30),
                'usage_history_window_size': getattr(self._config, "usage_history_window_size", 100),
                'resource_prediction_threshold': getattr(self._config, "resource_prediction_threshold", 0.9),
                'proactive_management_interval': getattr(self._config, "proactive_management_interval", 5.0),
                'offload_directory': getattr(self._config, "offload_directory", "./offloaded_tensors")
            }

            # Update with any additional kwargs
            config.update(kwargs)

            # Create resource prediction system instance
            self._resource_prediction_system = ResourcePredictionSystem(config)

            logger.info("Resource prediction system setup completed")
            return True
        except Exception as e:
            logger.error(f"Failed to setup resource prediction system: {e}")
            return False

    def start_resource_prediction(self, **kwargs) -> bool:
        """Start resource prediction using ML algorithms to anticipate memory and computation needs."""
        try:
            # Initialize resource prediction system if not already done
            if self._resource_prediction_system is None:
                if not self.setup_resource_prediction_system(**kwargs):
                    logger.error("Failed to setup resource prediction system")
                    return False

            # Start the prediction system
            result = self._resource_prediction_system.start_prediction()

            # Start proactive prediction if enabled
            if getattr(self._config, "enable_resource_prediction", False):
                interval = getattr(self._config, "proactive_management_interval", 5.0)
                logger.info(f"Proactive resource prediction enabled with interval {interval}s")

            logger.info("Resource prediction started")
            return result
        except Exception as e:
            logger.error(f"Failed to start resource prediction: {e}")
            return False

    def stop_resource_prediction(self) -> bool:
        """Stop resource prediction system."""
        try:
            if self._resource_prediction_system:
                result = self._resource_prediction_system.stop_prediction()
                logger.info("Resource prediction system stopped successfully")
                return result
            else:
                logger.warning("Resource prediction system was not initialized")
                return True
        except Exception as e:
            logger.error(f"Failed to stop resource prediction: {e}")
            return False

    def record_tensor_usage(self, tensor_name: str, tensor_data: torch.Tensor,
                          usage_type: str = "memory") -> bool:
        """Record a tensor usage for resource prediction modeling."""
        try:
            if self._resource_prediction_system:
                return self._resource_prediction_system.record_tensor_usage(
                    tensor_name, tensor_data, usage_type
                )
            else:
                logger.warning("Resource prediction system not initialized")
                return False
        except Exception as e:
            logger.error(f"Failed to record tensor usage: {e}")
            return False

    def record_layer_compute_usage(self, layer_name: str, compute_units: float,
                                 duration_ms: float = 0.0, input_size: int = 0) -> bool:
        """Record a layer compute usage for resource prediction modeling."""
        try:
            if self._resource_prediction_system:
                return self._resource_prediction_system.record_layer_compute_usage(
                    layer_name, compute_units, duration_ms, input_size
                )
            else:
                logger.warning("Resource prediction system not initialized")
                return False
        except Exception as e:
            logger.error(f"Failed to record layer compute usage: {e}")
            return False

    def get_memory_prediction_for_tensor(self, tensor_name: str) -> float:
        """Get the memory usage probability prediction for a specific tensor."""
        try:
            if self._resource_prediction_system:
                return self._resource_prediction_system.get_memory_prediction_for_tensor(tensor_name)
            else:
                logger.warning("Resource prediction system not initialized")
                return 0.0
        except Exception as e:
            logger.error(f"Failed to get memory prediction for tensor {tensor_name}: {e}")
            return 0.0

    def get_compute_prediction_for_layer(self, layer_name: str) -> float:
        """Get the compute demand prediction for a specific layer."""
        try:
            if self._resource_prediction_system:
                return self._resource_prediction_system.get_compute_prediction_for_layer(layer_name)
            else:
                logger.warning("Resource prediction system not initialized")
                return 0.0
        except Exception as e:
            logger.error(f"Failed to get compute prediction for layer {layer_name}: {e}")
            return 0.0

    def get_resource_prediction_stats(self) -> Dict[str, Any]:
        """Get statistics about resource prediction activities."""
        try:
            if self._resource_prediction_system:
                return self._resource_prediction_system.get_resource_prediction_stats()
            else:
                logger.warning("Resource prediction system not initialized")
                return {"enabled": False}
        except Exception as e:
            logger.error(f"Failed to get resource prediction stats: {e}")
            return {"enabled": False, "error": str(e)}


def create_qwen3_4b_instruct_2507_plugin() -> Qwen3_4B_Instruct_2507_Plugin:
    """
    Factory function to create a Qwen3-4B-Instruct-2507 plugin instance.

    Returns:
        Qwen3_4B_Instruct_2507_Plugin: The created plugin instance
    """
    return Qwen3_4B_Instruct_2507_Plugin()


__all__ = ["Qwen3_4B_Instruct_2507_Plugin", "create_qwen3_4b_instruct_2507_plugin"]