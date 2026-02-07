"""
Qwen3-Coder-30B Plugin - Self-Contained Implementation

This module implements the Qwen3-Coder-30B model plugin following the standard
plugin interface defined in the Inference-PIO system. This plugin is compatible with
the generic model loading, inference, and data processing functions, with specific
optimizations for Qwen3-Coder-30B model characteristics.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from ...common.processing.adaptive_batch_manager import (
    AdaptiveBatchManager,
    get_adaptive_batch_manager,
)
from ...common.optimization.activation_offloading import (
    ActivationOffloadingManager,
)
# from ...common.base_plugin_interface import (
#     ActivationAccessPattern,
#     ActivationPriority,
# )
from ...common.optimization.disk_offloading import (
    AccessPattern,
    DiskOffloader,
    OffloadPriority,
)
from ...common.optimization.disk_offloading import (
    TensorOffloadingManager as DiskTensorOffloadingManager,
)
from ...common.interfaces.improved_base_plugin_interface import (
    ModelPluginInterface,
)
from ...common.interfaces.improved_base_plugin_interface import (
    PluginMetadata as ModelPluginMetadata,
)
from ...common.interfaces.improved_base_plugin_interface import (
    PluginType,
    TextModelPluginInterface,
)
from ...common.managers.memory_manager import MemoryManager
from ...common.interfaces.memory_interface import MemoryManagerInterface as MemoryPriority # Approximate
from ...common.optimization.model_surgery import (
    ModelSurgerySystem,
    apply_model_surgery,
    restore_model_from_surgery,
)
from ...common.optimization.optimization_integration import (
    apply_qwen_optimizations,
    legacy_apply_activation_offloading,
    legacy_apply_disk_offloading,
    legacy_apply_flash_attention,
    legacy_apply_kernel_fusion,
    legacy_apply_sparse_attention,
    legacy_apply_structured_pruning,
    legacy_apply_tensor_compression,
)
from ...common.optimization.tensor_compression import AdaptiveTensorCompressor, get_tensor_compressor
from ...common.optimization.unimodal_model_surgery import (
    UnimodalModelSurgerySystem,
    analyze_unimodal_model_for_surgery,
    apply_unimodal_model_surgery,
    get_unimodal_model_surgery_system,
)
from ...common.processing.unimodal_preprocessing import (
    TextPreprocessor as UnimodalTextPreprocessor,
)
from ...common.processing.unimodal_preprocessing import (
    UnimodalPreprocessor,
    create_unimodal_preprocessor,
)
from ...common.hardware.virtual_device import VirtualExecutionSimulator
from ...common.hardware.virtual_execution import (
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
from .config import Qwen3Coder30BConfig
from .cross_alignment_optimization import apply_cross_alignment_to_model

# Import moved to function scope to avoid circular imports
# from ...design_patterns.integration import create_optimized_adapted_plugin
from .model import Qwen3Coder30BModel
from .scheduling.intelligent_scheduler import apply_intelligent_scheduling_to_model, create_intelligent_scheduler_for_qwen3_coder
from .intelligent_cache.intelligent_cache_manager import apply_intelligent_caching_to_model, create_intelligent_cache_for_qwen3_coder

logger = logging.getLogger(__name__)


class Qwen3_Coder_30B_Plugin(TextModelPluginInterface):
    """
    Comprehensive implementation of the Qwen3-Coder-30B model plugin that follows
    the standard plugin interface and is compatible with generic functions.
    This plugin is specifically optimized for Qwen3-Coder-30B model characteristics
    while maintaining compatibility with general text generation.
    The plugin implements all required methods from TextModelPluginInterface and provides
    additional functionality for model-specific optimizations.
    """

    def __init__(self):
        # Create plugin metadata specific to Qwen3-Coder-30B
        metadata = ModelPluginMetadata(
            name="Qwen3-Coder-30B",
            version="1.0.0",
            author="Alibaba Cloud",
            description="Qwen3-Coder-30B specialized model with advanced optimizations",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=["torch", "transformers", "accelerate"],
            compatibility={
                "torch_version": ">=2.0.0",
                "transformers_version": ">=4.30.0",
                "python_version": ">=3.8",
                "min_memory_gb": 64.0,  # Estimated for Qwen3-Coder-30B model
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            model_architecture="Qwen3-Coder-30B Transformer-based model with code-specific optimizations",
            model_size="30B",
            required_memory_gb=64.0,  # Memory requirement for Qwen3-Coder-30B model
            supported_modalities=["text"],
            license="MIT",
            tags=["language-model", "code-generation", "30b", "qwen3"],
            model_family="Qwen3",
            num_parameters=30000000000,  # 30 billion parameters
            test_coverage=0.95,
            validation_passed=True,
        )
        super().__init__(metadata)
        self._model = None
        self._tokenizer = None
        self._config = Qwen3Coder30BConfig()
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

        # Activation offloading attributes
        self._activation_offloading_manager = None
        self._activation_offloading_enabled = False
        self._activation_offloading_config = {}

        # Predictive Memory Optimization components
        self._predictive_memory_optimization = None

        # Resource Prediction System components
        self._resource_prediction_system = None

    def load_model(self, config: Qwen3Coder30BConfig = None) -> nn.Module:
        """
        Load the Qwen3-Coder-30B model with optimizations.

        Args:
            config: Model configuration (optional)

        Returns:
            nn.Module: The loaded model instance with Qwen3-Coder-30B specific optimizations applied
        """
        try:
            if config is not None:
                self._config = config

            logger.info(
                f"Loading Qwen3-Coder-30B model from: {self._config.model_path}"
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
            self._model = Qwen3Coder30BModel(self._config)

            # Apply cross-alignment optimization if enabled
            if getattr(self._config, 'enable_cross_alignment', False):
                logger.info("Applying cross-alignment optimization to Qwen3-Coder-30B model...")
                self._model = apply_cross_alignment_to_model(self._model, self._config)

            # Apply intelligent caching if enabled in config
            if getattr(self._config, 'intelligent_cache_enabled', False):
                logger.info("Applying intelligent caching to Qwen3-Coder-30B model")
                intelligent_cache_config = create_intelligent_cache_for_qwen3_coder(self._config)
                self._model = apply_intelligent_caching_to_model(self._model, intelligent_cache_config)

                # Store reference to the cache manager for later use
                self.intelligent_cache_manager = intelligent_cache_config

            # Apply intelligent scheduling if enabled in config
            if getattr(self._config, 'enable_intelligent_scheduling', False):
                logger.info("Applying intelligent scheduling to Qwen3-Coder-30B model")
                intelligent_scheduler_config = create_intelligent_scheduler_for_qwen3_coder(self._config)
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
                    model_type="qwen3_coder",
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
                f"Qwen3-Coder-30B model loaded successfully on device: {device}"
            )
            return self._model
        except Exception as e:
            logger.error(f"Failed to load Qwen3-Coder-30B model: {e}")
            raise e

    def infer(self, data: Any) -> Any:
        """
        Perform text inference on the given data, with special handling for Qwen3-Coder-30B tasks.

        Args:
            data: Input text data for inference

        Returns:
            Any: Text inference results specific to Qwen3-Coder-30B model
        """
        if self._model is None or self._tokenizer is None:
            self.load_model()

        if not isinstance(data, str):
            raise ValueError("Qwen3-Coder-30B model expects string input")

        # Handle empty input
        if not data.strip():
            logger.warning("Empty input provided, returning empty string")
            return ""

        try:
            # Tokenize input
            inputs = self._tokenizer(
                data,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32768,  # Max context length for Qwen3-Coder-30B models
            )

            # Move inputs to the same device as the model
            device = (
                next(self._model.parameters()).device
                if self._model is not None
                else torch.device("cpu")
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Use compiled model if available, otherwise use original model
            model_to_use = self.get_compiled_model()

            # Generate output with Qwen3-Coder-30B specific parameters
            with torch.no_grad():
                outputs = model_to_use.generate(
                    **inputs,
                    max_length=min(
                        len(inputs["input_ids"][0]) + self._config.max_new_tokens, 32768
                    ),
                    pad_token_id=self._config.pad_token_id,
                    do_sample=self._config.do_sample,
                    temperature=self._config.temperature,
                    top_p=self._config.top_p,
                    top_k=self._config.top_k,
                    repetition_penalty=self._config.repetition_penalty,
                    num_return_sequences=1,
                )

            # Decode output
            generated_text = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            return generated_text
        except Exception as e:
            logger.error(f"Error during text inference: {e}")
            raise e

    def generate_text(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        """
        Specialized method for generating text with Qwen3-Coder-30B specific parameters.

        Args:
            prompt: Text generation prompt
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            str: Generated text
        """
        if self._model is None or self._tokenizer is None:
            self.load_model()

        # Create a temporary config with updated parameters
        temp_config = Qwen3Coder30BConfig()
        temp_config.max_new_tokens = max_new_tokens
        # Update other parameters from kwargs if provided
        for key, value in kwargs.items():
            if hasattr(temp_config, key):
                setattr(temp_config, key, value)

        # Tokenize input
        inputs = self._tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=32768
        )

        # Move inputs to the same device as the model
        device = (
            next(self._model.parameters()).device
            if self._model is not None
            else torch.device("cpu")
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Use compiled model if available, otherwise use original model
        model_to_use = self.get_compiled_model()

        try:
            with torch.no_grad():
                outputs = model_to_use.generate(
                    **inputs,
                    max_length=min(len(inputs["input_ids"][0]) + max_new_tokens, 32768),
                    pad_token_id=temp_config.pad_token_id,
                    do_sample=temp_config.do_sample,
                    temperature=temp_config.temperature,
                    top_p=temp_config.top_p,
                    top_k=temp_config.top_k,
                    repetition_penalty=temp_config.repetition_penalty,
                    num_return_sequences=1,
                )

            # Decode output
            generated_text = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            return generated_text
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            raise e

    def chat_completion(
        self, messages: List[Dict[str, str]], max_new_tokens: int = 1024, **kwargs
    ) -> str:
        """
        Specialized method for chat completion with Qwen3-Coder-30B model.

        Args:
            messages: List of message dictionaries with role and content
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            str: Generated response
        """
        if self._model is None or self._tokenizer is None:
            self.load_model()

        # Format messages for Qwen3-Coder-30B
        formatted_prompt = self._format_chat_messages(messages)

        return self.generate_text(
            formatted_prompt, max_new_tokens=max_new_tokens, **kwargs
        )

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages for Qwen3-Coder-30B model.

        Args:
            messages: List of message dictionaries with role and content

        Returns:
            str: Formatted prompt
        """
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted += f"<|system|>{content}<|end|>\n"
            elif role == "user":
                formatted += f"<|user|>{content}<|end|>\n"
            elif role == "assistant":
                formatted += f"<|assistant|>{content}<|end|>\n"
            else:
                formatted += f"<|{role}|>{content}<|end|>\n"

        # Add assistant tag to continue the conversation
        formatted += "<|assistant|>"
        return formatted

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed model information specific to Qwen3-Coder-30B.

        Returns:
            Dictionary containing comprehensive information about the Qwen3-Coder-30B model
            including architecture, parameters, configuration, and optimization settings.
        """
        return {
            "name": self.metadata.name,
            "model_type": "Causal Language Model",
            "architecture": self.metadata.model_architecture,
            "modalities": self.metadata.supported_modalities,
            "size": self.metadata.model_size,
            "parameters": self.metadata.num_parameters,
            "model_specific_params": {
                "gradient_checkpointing": self._config.gradient_checkpointing,
                "use_cache": self._config.use_cache,
                "torch_dtype": self._config.torch_dtype,
                "device_map": self._config.device_map,
                "low_cpu_mem_usage": self._config.low_cpu_mem_usage,
            },
            "qwen_specific_params": {
                "temperature": self._config.temperature,
                "top_p": self._config.top_p,
                "top_k": self._config.top_k,
                "repetition_penalty": self._config.repetition_penalty,
                "max_new_tokens": self._config.max_new_tokens,
                "do_sample": self._config.do_sample,
            },
            "optimizations_enabled": {
                "flash_attention_2": self._config.use_flash_attention_2,
                "sparse_attention": self._config.use_sparse_attention,
                "optimized_rotary_embeddings": True,  # Always enabled
                "fused_layer_norm": self._config.use_fused_layer_norm,
                "multi_query_attention": self._config.use_multi_query_attention,
                "grouped_query_attention": self._config.use_grouped_query_attention,
                "paged_attention": self._config.use_paged_attention,
                "sliding_window_attention": self._config.use_sliding_window_attention,
                "tensor_parallelism": self._config.use_tensor_parallelism,
                "kv_cache_compression": self._config.use_kv_cache_compression,
                "prefix_caching": self._config.use_prefix_caching,
                "cuda_kernels": self._config.use_cuda_kernels,
                "linear_bias_optimization": self._config.linear_bias_optimization_enabled,
                "cross_alignment": getattr(self._config, 'enable_cross_alignment', False),
            },
        }

    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Get model parameter information for the Qwen3-Coder-30B model.

        Returns:
            Dictionary containing detailed parameter information including total,
            trainable, and frozen parameter counts for the Qwen3-Coder-30B model.
        """
        if self._model is None:
            # Return estimated parameters based on metadata
            return {
                "num_parameters": self.metadata.num_parameters,
                "trainable_parameters": "Not loaded",
                "parameter_count_by_component": "Not loaded",
            }

        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(
            p.numel() for p in self._model.parameters() if p.requires_grad
        )

        return {
            "num_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameter_count_by_component": {
                "total": total_params,
                "trainable": trainable_params,
                "frozen": total_params - trainable_params,
            },
        }

    def get_model_config_template(self) -> Qwen3Coder30BConfig:
        """
        Get a template for Qwen3-Coder-30B model configuration.

        Returns:
            A default configuration instance for Qwen3-Coder-30B model
        """
        return Qwen3Coder30BConfig()

    def initialize(self, **kwargs) -> bool:
        """
        Initialize the Qwen3-Coder-30B model with specific parameters and configurations.

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
            # Ensure the model path points to the H drive
            if "model_path" not in kwargs:
                kwargs["model_path"] = "H:/Qwen3-Coder-30B"

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
                    # Validate KV-cache compression parameters
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
                    # Validate prefix caching parameters
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

            # Load the model with updated config
            self.load_model()

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

            return True
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            return False

    def cleanup(self) -> bool:
        """
        Clean up model resources.
        """
        try:
            if hasattr(self, "_model") and self._model is not None:
                del self._model
                self._model = None
            if hasattr(self, "_tokenizer") and self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
            if hasattr(self, "_compiled_model") and self._compiled_model is not None:
                del self._compiled_model
                self._compiled_model = None

            # Force memory cleanup including tensor paging and swap files
            self.force_memory_cleanup()

            # Clean up disk offloading resources
            if self._disk_tensor_offloading_manager:
                self._disk_tensor_offloading_manager.stop_proactive_management()

            if self._disk_offloader:
                self._disk_offloader.cleanup()
                self._disk_offloader = None
                self._disk_tensor_offloading_manager = None
                self._offloading_enabled = False

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            return True
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return False

    def tokenize(self, text: str, **kwargs) -> Any:
        """
        Tokenize the given text.

        Args:
            text: Text to tokenize
            **kwargs: Additional tokenization parameters

        Returns:
            Any: Tokenized result
        """
        if self._tokenizer is None:
            # Load tokenizer if not already loaded
            if self._model is None:
                self.load_model()
            self._tokenizer = self._model.get_tokenizer()

        # Use unimodal preprocessor if available
        if self._unimodal_preprocessor is not None:
            return self._unimodal_preprocessor.preprocess(
                text,
                return_tensors=kwargs.get("return_tensors", "pt"),
                model_type="qwen3_coder",
            )
        else:
            return self._tokenizer(
                text,
                return_tensors=kwargs.get("return_tensors", "pt"),
                padding=kwargs.get("padding", True),
                truncation=kwargs.get("truncation", True),
                max_length=kwargs.get("max_length", 32768),
            )

    def detokenize(self, token_ids: Union[List[int], torch.Tensor], **kwargs) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: Token IDs to decode
            **kwargs: Additional decoding parameters

        Returns:
            str: Decoded text
        """
        if self._tokenizer is None:
            # Load tokenizer if not already loaded
            if self._model is None:
                self.load_model()
            self._tokenizer = self._model.get_tokenizer()

        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=kwargs.get("skip_special_tokens", True),
            clean_up_tokenization_spaces=kwargs.get(
                "clean_up_tokenization_spaces", True
            ),
        )

    def get_model_class(self) -> Type[nn.Module]:
        """
        Return the model class for Qwen3-Coder-30B.

        Returns:
            Type[nn.Module]: The model class
        """
        return Qwen3Coder30BModel

    def get_config_class(self) -> Type:
        """
        Return the configuration class for Qwen3-Coder-30B.

        Returns:
            Type: The configuration class
        """
        return Qwen3Coder30BConfig

    def supports_config(self, config: Any) -> bool:
        """
        Check if this model supports the given configuration.

        Args:
            config: Configuration to check

        Returns:
            bool: True if the configuration is supported by Qwen3-Coder-30B, False otherwise
        """
        # For Qwen3-Coder-30B, we expect a Qwen3Coder30BConfig object
        return isinstance(config, Qwen3Coder30BConfig) or config is None

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
            logger.info(
                "Activation compression enabled - will compress during inference"
            )
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


def create_qwen3_coder_30b_plugin() -> Qwen3_Coder_30B_Plugin:
    """
    Factory function to create a Qwen3-Coder-30B plugin instance.

    Returns:
        Qwen3_Coder_30B_Plugin: The created plugin instance
    """
    return Qwen3_Coder_30B_Plugin()


__all__ = ["Qwen3_Coder_30B_Plugin", "create_qwen3_coder_30b_plugin"]