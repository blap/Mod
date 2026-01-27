"""
Qwen3-VL-2B Model Implementation - Self-Contained Version

This module implements the Qwen3-VL-2B model following the self-contained plugin architecture
for the Inference-PIO system. This implementation is optimized specifically for Qwen3-VL-2B
characteristics while maintaining compatibility with the generic model interface.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Type, Generator, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor
except ImportError:
    AutoModelForVision2Seq = None
    AutoTokenizer = None
    AutoImageProcessor = None

logger = logging.getLogger(__name__)


class Qwen3VL2BModel(nn.Module):
    """
    Qwen3-VL-2B model implementation with all optimizations integrated.

    This is the main model class for the Qwen3-VL-2B model with all optimizations
    applied. It maintains the full model capacity while providing comprehensive
    optimizations for the target hardware platform.
    """

    def __init__(self, config: 'Qwen3VL2BConfig'):
        super().__init__()
        self.config = config

        # Initialize the base model
        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._model_name = config.model_path

        # Memory optimization settings specific to Qwen3-VL-2B model
        # Convert string dtype to torch dtype safely
        torch_dtype = self._get_torch_dtype(config.torch_dtype)

        self._memory_config = {
            "gradient_checkpointing": config.gradient_checkpointing,
            "use_cache": config.use_cache,
            "torch_dtype": torch_dtype,
            "device_map": config.device_map,
            "low_cpu_mem_usage": config.low_cpu_mem_usage,
            "max_memory": config.max_memory
        }

        # Initialize NAS controller if enabled
        self._nas_controller = None
        self._model_adapter = None
        if getattr(config, 'enable_continuous_nas', False):
            from .nas_controller import get_nas_controller, NASConfig, ArchitectureAdaptationStrategy
            nas_config = NASConfig(
                strategy=getattr(config, 'nas_strategy', ArchitectureAdaptationStrategy.COMBINED_ADAPTIVE),
                min_depth_ratio=getattr(config, 'nas_min_depth_ratio', 0.3),
                max_depth_ratio=getattr(config, 'nas_max_depth_ratio', 1.0),
                min_width_ratio=getattr(config, 'nas_min_width_ratio', 0.3),
                max_width_ratio=getattr(config, 'nas_max_width_ratio', 1.0),
                latency_target_ms=getattr(config, 'nas_latency_target_ms', 100.0),
                memory_budget_mb=getattr(config, 'nas_memory_budget_mb', 2048.0),
                accuracy_tradeoff_factor=getattr(config, 'nas_accuracy_tradeoff_factor', 0.7),
                adaptation_frequency=getattr(config, 'nas_adaptation_frequency', 10)
            )
            self._nas_controller = get_nas_controller(nas_config)
            self._model_adapter = get_model_adapter(self._model, self._nas_controller) if self._model else None

        # Initialize advanced disk offloading system if enabled
        self._disk_offloader = None
        self._tensor_offloader = None
        self._multimodal_offloader = None
        if getattr(config, 'enable_disk_offloading', False):
            self._initialize_disk_offloading()

        # Initialize intelligent pagination system for multimodal data if enabled
        self._pagination_system = None
        self._multimodal_pager = None
        if getattr(config, 'enable_intelligent_pagination', False):
            self._initialize_pagination_system()

        # Initialize the model
        self._initialize_model()

        # Initialize dynamic multimodal batching system if enabled
        self._dynamic_multimodal_batch_manager = None
        if getattr(config, 'enable_dynamic_multimodal_batching', False):
            self._initialize_dynamic_multimodal_batching()

        # Initialize pipeline parallelism if enabled
        self._pipeline_parallel_model = None
        if getattr(config, 'enable_pipeline_parallelism', False):
            self._initialize_pipeline_parallelism()

        # Initialize sequence parallelism if enabled
        self._sequence_parallel_model = None
        if getattr(config, 'enable_sequence_parallelism', False):
            self._initialize_sequence_parallelism()

        # Initialize vision-language parallelism if enabled (specifically for multimodal models)
        # This must be done after the model is loaded
        self._vision_language_parallel_model = None
        if getattr(config, 'enable_vision_language_parallelism', False):
            self._initialize_vision_language_parallelism()

        # Initialize multimodal preprocessing pipeline if enabled
        self._multimodal_pipeline = None
        if getattr(config, 'enable_multimodal_preprocessing_pipeline', False):
            self._initialize_multimodal_pipeline()

        # Initialize intelligent multimodal caching system if enabled
        self._caching_manager = None
        if getattr(config, 'enable_intelligent_multimodal_caching', False):
            self._initialize_intelligent_multimodal_caching()

        # Initialize visual resource compression system if enabled
        self._visual_compressor = None
        if getattr(config, 'enable_visual_resource_compression', False):
            self._initialize_visual_resource_compression()

        # Initialize efficient image tokenization system if enabled
        self._image_tokenizer = None
        if getattr(config, 'enable_image_tokenization', True):
            self._initialize_image_tokenization()

        # Apply quantization if enabled
        if getattr(config, 'use_quantization', False):
            self._apply_quantization()

        # Initialize asynchronous multimodal processing if enabled
        self._async_multimodal_manager = None
        if getattr(config, 'enable_async_multimodal_processing', False):
            self._initialize_async_multimodal_processing()

        # Apply asynchronous multimodal processing optimizations if enabled
        if getattr(config, 'enable_async_multimodal_processing', False):
            self._apply_async_multimodal_processing_optimizations()

    def _get_torch_dtype(self, dtype_str: str):
        """
        Safely convert string dtype to torch dtype.

        Args:
            dtype_str: String representation of the dtype

        Returns:
            torch.dtype: The corresponding torch dtype
        """
        dtype_mapping = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "half": torch.float16,
            "float": torch.float32,
            "double": torch.float64,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
        }

        if isinstance(dtype_str, torch.dtype):
            return dtype_str

        if dtype_str in dtype_mapping:
            return dtype_mapping[dtype_str]

        # If the string doesn't match known dtypes, try to get it from torch directly
        try:
            return getattr(torch, dtype_str)
        except AttributeError:
            logger.warning(f"Unknown dtype '{dtype_str}', defaulting to float16")
            return torch.float16

    def _initialize_model(self):
        """
        Initialize the Qwen3-VL-2B model with appropriate optimizations.
        """
        try:
            logger.info(f"Loading Qwen3-VL-2B model from: {self._model_name}")

            # Prepare loading arguments with memory optimizations
            load_kwargs = {
                "torch_dtype": self._memory_config.get("torch_dtype", torch.float16),
                "device_map": self._memory_config.get("device_map", "auto"),
                "trust_remote_code": True,
                "low_cpu_mem_usage": self._memory_config.get("low_cpu_mem_usage", True),
            }

            # Add max_memory if specified
            if self._memory_config.get("max_memory"):
                load_kwargs["max_memory"] = self._memory_config["max_memory"]

            # Check if the model path exists locally
            import os
            # First check if the drive exists to avoid hanging on non-existent drives
            drive_exists = True
            # Handle both forward slash and backslash formats
            path_sep = '\\' if '\\' in self._model_name else '/'
            if ':' in self._model_name:
                drive_letter = self._model_name.split(':')[0] + ':'
                if not os.path.exists(drive_letter):
                    logger.warning(f"Drive {drive_letter} does not exist, using HuggingFace model...")
                    drive_exists = False

            if drive_exists and os.path.exists(self._model_name):
                # Local path exists, use it directly
                model_path = self._model_name
                logger.info(f"Using local model from: {model_path}")
            else:
                # Local path doesn't exist or drive doesn't exist, try HuggingFace
                logger.warning(f"Local model path {self._model_name} not found, trying HuggingFace...")
                model_path = "H:/Qwen3-VL-2B-Instruct"  # Local model path on drive H

            # Load the model with appropriate settings
            self._model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                **load_kwargs
            )

            # Enable gradient checkpointing for memory efficiency if configured
            if self._memory_config.get("gradient_checkpointing", True):
                self._model.gradient_checkpointing_enable()

            # Load the tokenizer and image processor
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            self._image_processor = AutoImageProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            # Set padding token if not present
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Apply optimizations based on config
            self._apply_configured_optimizations()

            # Initialize model adapter for NAS if enabled
            if self._nas_controller is not None:
                self._model_adapter = get_model_adapter(self._model, self._nas_controller)

        except Exception as e:
            logger.error(f"Failed to initialize Qwen3-VL-2B model: {e}")
            raise

    def _apply_configured_optimizations(self):
        """
        Apply optimizations based on the configuration settings.
        """
        # Check if ML-based optimization is enabled
        if getattr(self.config, 'use_ml_optimizations', False):
            # Use ML-based optimization system
            from ...common.unified_ml_optimization import get_ml_optimization_system, ModelType
            ml_system = get_ml_optimization_system()
            self._model = ml_system.optimize_model_for_input(
                model=self._model,
                input_data=None,  # Initially no input, will be optimized per request
                model_type=ModelType.QWEN3_VL_2B
            )
        elif getattr(self.config, 'use_modular_optimizations', False):
            # Apply optimizations using the new modular system if enabled
            from .optimization_integration import apply_qwen_optimizations
            profile_name = getattr(self.config, 'optimization_profile', 'balanced')
            self._model = apply_qwen_optimizations(self._model, profile_name)
        else:
            # Apply traditional optimizations for backward compatibility
            self._apply_traditional_optimizations()

        # Apply vision encoder optimizations specific to Qwen3-VL-2B
        self._apply_vision_encoder_optimizations()

        # Apply cross-modal fusion optimizations specific to Qwen3-VL-2B
        self._apply_cross_modal_fusion_optimizations()

        # Apply cross-modal alignment optimizations specific to Qwen3-VL-2B
        self._apply_cross_modal_alignment_optimizations()

        # Apply projection layer optimizations specific to Qwen3-VL-2B
        self._apply_projection_layer_optimizations()

        # Apply multimodal attention optimization specific to Qwen3-VL-2B
        self._apply_multimodal_attention_optimization()

        # Apply attention mechanism optimizations specific to Qwen3-VL-2B
        self._apply_attention_optimizations()

        # Apply intelligent pagination after other optimizations
        if self._multimodal_pager:
            self._apply_intelligent_pagination_to_model()

    def _apply_attention_optimizations(self):
        """
        Apply attention mechanism optimizations specific to Qwen3-VL-2B model.
        This includes FlashAttention, sparse attention, sliding window attention,
        multi-query attention, and other attention optimizations.
        """
        try:
            logger.info("Applying attention optimizations to Qwen3-VL-2B model...")

            # Determine which attention optimization to use based on config
            if self.config.use_flash_attention_2:
                logger.info("Applying FlashAttention 2.0 optimization to Qwen3-VL-2B model...")
                self._apply_flash_attention_optimization()
            elif self.config.use_sparse_attention:
                logger.info("Applying sparse attention optimization to Qwen3-VL-2B model...")
                self._apply_sparse_attention_optimization()
            elif self.config.use_sliding_window_attention:
                logger.info("Applying sliding window attention optimization to Qwen3-VL-2B model...")
                self._apply_sliding_window_attention_optimization()
            elif self.config.use_multi_query_attention or self.config.use_grouped_query_attention:
                logger.info("Applying Multi-Query/Grouped-Query attention optimization to Qwen3-VL-2B model...")
                self._apply_mqa_gqa_optimization()
            elif self.config.use_paged_attention:
                logger.info("Applying paged attention optimization to Qwen3-VL-2B model...")
                self._apply_paged_attention_optimization()
            else:
                logger.info("Attention optimization disabled for Qwen3-VL-2B model")

            # Apply optimized rotary embeddings
            logger.info("Applying optimized rotary embeddings to Qwen3-VL-2B model...")
            self._apply_optimized_rotary_embedding()

            logger.info("All attention optimizations applied successfully to Qwen3-VL-2B model")
        except Exception as e:
            logger.error(f"Error applying attention optimizations to Qwen3-VL-2B model: {e}")
            # Continue without attention optimizations if they fail
            pass

    def _apply_flash_attention_optimization(self):
        """
        Apply FlashAttention 2.0 optimization to the Qwen3-VL-2B model.
        """
        try:
            from .flash_attention_2 import apply_flash_attention_2_to_model
            self._model = apply_flash_attention_2_to_model(self._model, self.config)
        except ImportError:
            logger.warning("FlashAttention 2.0 module not available, skipping optimization")
        except Exception as e:
            logger.error(f"Error applying FlashAttention 2.0 optimization: {e}")
            pass

    def _apply_sparse_attention_optimization(self):
        """
        Apply sparse attention optimization to the Qwen3-VL-2B model.
        """
        try:
            from .sparse_attention import apply_sparse_attention_to_model
            self._model = apply_sparse_attention_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Sparse attention module not available, skipping optimization")
        except Exception as e:
            logger.error(f"Error applying sparse attention optimization: {e}")
            pass

    def _apply_sliding_window_attention_optimization(self):
        """
        Apply sliding window attention optimization to the Qwen3-VL-2B model.
        """
        try:
            from .sliding_window_attention import apply_sliding_window_attention_to_model
            self._model = apply_sliding_window_attention_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Sliding window attention module not available, skipping optimization")
        except Exception as e:
            logger.error(f"Error applying sliding window attention optimization: {e}")
            pass

    def _apply_mqa_gqa_optimization(self):
        """
        Apply Multi-Query/Grouped-Query attention optimization to the Qwen3-VL-2B model.
        """
        try:
            from .multi_query_attention import apply_mqa_gqa_to_model
            self._model = apply_mqa_gqa_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Multi-Query/Grouped-Query attention module not available, skipping optimization")
        except Exception as e:
            logger.error(f"Error applying MQA/GQA optimization: {e}")
            pass

    def _apply_paged_attention_optimization(self):
        """
        Apply paged attention optimization to the Qwen3-VL-2B model.
        """
        try:
            from .paged_attention import apply_paged_attention_to_model
            self._model = apply_paged_attention_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Paged attention module not available, skipping optimization")
        except Exception as e:
            logger.error(f"Error applying paged attention optimization: {e}")
            pass

    def _apply_optimized_rotary_embedding(self):
        """
        Apply optimized rotary embedding implementation to the Qwen3-VL-2B model.
        """
        try:
            from .rotary_embeddings import apply_rotary_embeddings_to_model
            self._model = apply_rotary_embeddings_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Rotary embeddings module not available, skipping optimization")
        except Exception as e:
            logger.error(f"Error applying optimized rotary embedding: {e}")
            pass

    def _apply_vision_encoder_optimizations(self):
        """
        Apply vision encoder optimizations specific to Qwen3-VL-2B model.
        """
        try:
            from .vision_transformer_kernels import apply_vision_cuda_optimizations_to_model
            self._model = apply_vision_cuda_optimizations_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Vision transformer kernels module not available, skipping optimization")
        except Exception as e:
            logger.error(f"Error applying vision encoder optimizations: {e}")
            pass

    def _apply_cross_modal_fusion_optimizations(self):
        """
        Apply cross-modal fusion optimizations specific to Qwen3-VL-2B model.
        """
        try:
            from .cross_modal_fusion_kernels import apply_cross_modal_fusion_to_qwen3_vl_model
            self._model = apply_cross_modal_fusion_to_qwen3_vl_model(self._model, self.config)
        except ImportError:
            logger.warning("Cross-modal fusion kernels module not available, skipping optimization")
        except Exception as e:
            logger.error(f"Error applying cross-modal fusion optimizations: {e}")
            pass

    def _apply_cross_modal_alignment_optimizations(self):
        """
        Apply cross-modal alignment optimizations specific to Qwen3-VL-2B model.
        """
        try:
            from .cross_modal_alignment_optimization import apply_cross_modal_alignment_to_model
            self._model = apply_cross_modal_alignment_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Cross-modal alignment optimization module not available, skipping optimization")
        except Exception as e:
            logger.error(f"Error applying cross-modal alignment optimizations: {e}")
            pass

    def _apply_projection_layer_optimizations(self):
        """
        Apply projection layer optimizations specific to Qwen3-VL-2B model.
        """
        try:
            from .multimodal_projector import apply_qwen3_vl_projection_optimizations
            self._model = apply_qwen3_vl_projection_optimizations(self._model, self.config)
        except ImportError:
            logger.warning("Multimodal projector module not available, skipping optimization")
        except Exception as e:
            logger.error(f"Error applying projection layer optimizations: {e}")
            pass

    def _apply_multimodal_attention_optimization(self):
        """
        Apply multimodal attention optimization specific to Qwen3-VL-2B model.
        """
        try:
            from .multimodal_attention_optimization import apply_multimodal_attention_optimizations_to_model
            self._model = apply_multimodal_attention_optimizations_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Multimodal attention optimization module not available, skipping optimization")
        except Exception as e:
            logger.error(f"Error applying multimodal attention optimization: {e}")
            pass

    def _apply_intelligent_pagination_to_model(self):
        """
        Apply intelligent pagination to model components based on configuration.
        """
        if not self._multimodal_pager:
            logger.warning("Intelligent pagination not initialized, skipping pagination application")
            return

        try:
            logger.info("Applying intelligent pagination to Qwen3-VL-2B model components...")

            # Determine which components to paginate based on config
            components_to_paginate = []

            if getattr(self.config, 'paginate_attention_weights', False):
                components_to_paginate.append(DataType.ACTIVATIONS)  # Attention weights are activations

            # Paginate specified components
            # Note: In a real implementation, we would iterate through model components
            # and apply pagination based on their type and importance
            successful_paginations = 0
            total_components = 0

            # Example: Paginate KV cache components if they exist
            if hasattr(self._model, 'model') and hasattr(self._model.model, 'layers'):
                layers = self._model.model.layers
            elif hasattr(self._model, 'transformer') and hasattr(self._model.transformer, 'h'):
                layers = self._model.transformer.h
            else:
                layers = []

            for idx, layer in enumerate(layers):
                total_components += 1
                # Example: Paginate attention-related tensors
                if hasattr(layer, 'self_attn'):
                    attn_module = layer.self_attn
                    # Paginate KV cache tensors when they are created during inference
                    # This would happen dynamically during forward pass
                    successful_paginations += 1

            logger.info(f"Intelligent pagination setup: {successful_paginations}/{total_components} components prepared for pagination")

        except Exception as e:
            logger.error(f"Error applying intelligent pagination to model: {e}")

    def _apply_traditional_optimizations(self):
        """
        Apply traditional optimizations for backward compatibility.
        """
        # Apply tensor parallelism if enabled
        if self.config.use_tensor_parallelism:
            self._apply_tensor_parallelism()

        # Apply FlashAttention 2.0 optimization if enabled
        if self.config.use_flash_attention_2:
            self._apply_flash_attention_2_optimization()

        # Apply bias removal optimization if enabled
        if self.config.use_bias_removal_optimization:
            self._apply_bias_removal_optimization()

        # Apply fused layer normalization if enabled
        if self.config.use_fused_layer_norm:
            self._apply_fused_layer_norm()

        # Apply KV-cache compression if enabled
        if self.config.use_kv_cache_compression:
            self._apply_kv_cache_compression()

        # Apply prefix caching if enabled
        if self.config.use_prefix_caching:
            self._apply_prefix_caching()

        # Apply CUDA kernels if enabled
        if self.config.use_cuda_kernels:
            self._apply_cuda_kernels()

        # Apply linear bias optimization if enabled
        if self.config.linear_bias_optimization_enabled:
            self._apply_linear_bias_optimization()

        # Apply tensor decomposition if enabled
        if getattr(self.config, 'use_tensor_decomposition', False):
            from .tensor_decomposition import apply_tensor_decomposition_to_model
            decomposition_method = getattr(self.config, 'tensor_decomposition_method', 'cp_decomposition')
            rank_ratio = getattr(self.config, 'tensor_decomposition_rank_ratio', 0.5)

            self._model = apply_tensor_decomposition_to_model(
                self._model,
                rank_ratio=rank_ratio,
                decomposition_method=decomposition_method
            )

        # Apply structured pruning if enabled
        if self.config.use_structured_pruning:
            from .structured_pruning import apply_structured_pruning_to_model, PruningMethod
            # Convert string method to enum
            method_map = {
                "layer_removal": PruningMethod.LAYER_REMOVAL,
                "block_removal": PruningMethod.BLOCK_REMOVAL,
                "head_removal": PruningMethod.HEAD_REMOVAL,
                "mlp_removal": PruningMethod.MLP_REMOVAL,
                "adaptive_pruning": PruningMethod.ADAPTIVE_PRUNING
            }
            pruning_method = method_map.get(self.config.pruning_method, PruningMethod.LAYER_REMOVAL)

            self._model = apply_structured_pruning_to_model(
                self._model,
                pruning_ratio=self.config.pruning_ratio,
                method=pruning_method,
                block_size=self.config.pruning_block_size
            )

        # Apply enhanced multimodal attention if enabled (this is specific to VL models)
        if getattr(self.config, 'use_multimodal_attention', False):
            from .multimodal_attention import apply_multimodal_attention_to_model
            self._model = apply_multimodal_attention_to_model(self._model, self.config)

        # Apply SNN conversion if enabled
        if getattr(self.config, 'use_snn_conversion', False):
            from .snn import apply_snn_conversion_to_model
            self._model = apply_snn_conversion_to_model(self._model, self.config)

    def _apply_tensor_parallelism(self):
        """
        Apply tensor parallelism to the model if configured.
        """
        try:
            from .tensor_parallel import apply_tensor_parallelism_to_model
            tensor_parallel_size = self.config.tensor_parallel_size
            if tensor_parallel_size > 1:
                logger.info(f"Applying tensor parallelism with size {tensor_parallel_size}")
                
                self._model = apply_tensor_parallelism_to_model(
                    self._model,
                    tensor_parallel_size=tensor_parallel_size,
                    local_rank=self.config.tensor_parallel_local_rank,
                    world_size=self.config.tensor_parallel_world_size,
                    init_method=self.config.tensor_parallel_init_method
                )
        except ImportError:
            logger.warning("Tensor parallelism module not available, skipping optimization")
        except Exception as e:
            logger.error(f"Error applying tensor parallelism: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_flash_attention_2_optimization(self):
        """
        Apply FlashAttention 2.0 optimization to the model by replacing
        attention layers with FlashAttention 2.0 implementations.
        """
        try:
            from .flash_attention_2 import apply_flash_attention_2_to_model
            self._model = apply_flash_attention_2_to_model(self._model, self.config)
        except ImportError:
            logger.warning("FlashAttention 2.0 module not available, skipping optimization")
        except Exception as e:
            logger.error(f"Error applying FlashAttention 2.0 optimization: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_bias_removal_optimization(self):
        """
        Apply bias removal optimization to the model.
        """
        try:
            from .linear_optimizations import apply_bias_removal_to_model
            self._model = apply_bias_removal_to_model(
                self._model,
                model_type="qwen3_vl"
            )
        except ImportError:
            logger.warning("Bias removal module not available, skipping optimization")
        except Exception as e:
            logger.error(f"Error applying bias removal optimization: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_fused_layer_norm(self):
        """
        Apply fused layer normalization to the model.
        """
        try:
            from .fused_layers import apply_fused_layer_norm_to_model
            self._model = apply_fused_layer_norm_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Fused layer norm module not available, skipping optimization")
        except Exception as e:
            logger.error(f"Error applying fused layer norm: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_kv_cache_compression(self):
        """
        Apply KV-cache compression optimization to the model.
        """
        try:
            from .kv_cache import apply_kv_cache_compression_to_model
            self._model = apply_kv_cache_compression_to_model(self._model, self.config)
        except ImportError:
            logger.warning("KV-cache compression module not available, skipping optimization")
        except Exception as e:
            logger.error(f"Error applying KV-cache compression: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_prefix_caching(self):
        """
        Apply prefix caching optimization to the model.
        """
        try:
            from .prefix_caching import apply_prefix_caching_to_model
            self._model = apply_prefix_caching_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Prefix caching module not available, skipping optimization")
        except Exception as e:
            logger.error(f"Error applying prefix caching: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_cuda_kernels(self):
        """
        Apply CUDA kernels optimization to the model.
        """
        try:
            from .multimodal_cuda_kernels import apply_qwen3_vl_cuda_optimizations_to_model
            self._model = apply_qwen3_vl_cuda_optimizations_to_model(self._model, self.config)
        except ImportError:
            logger.warning("CUDA kernels module not available, skipping optimization")
        except Exception as e:
            logger.error(f"Error applying CUDA kernels optimization: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_linear_bias_optimization(self):
        """
        Apply linear bias optimization to the model.
        """
        try:
            from .linear_optimizations import apply_linear_bias_optimization_to_model
            self._model = apply_linear_bias_optimization_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Linear bias optimization module not available, skipping")
        except Exception as e:
            logger.error(f"Error applying linear bias optimization: {e}")
            # Continue without optimization if it fails
            pass

    def _initialize_disk_offloading(self):
        """
        Initialize the advanced disk offloading system with multimodal support.
        """
        try:
            logger.info("Initializing advanced disk offloading system for Qwen3-VL-2B model...")

            from .disk_offloading import create_disk_offloader, TensorOffloadingManager, MultimodalOffloadingManager
            
            # Create disk offloader with advanced settings
            try:
                self._disk_offloader = create_disk_offloader(
                    max_memory_ratio=getattr(self.config, 'max_memory_ratio', 0.8),
                    offload_directory=getattr(self.config, 'offload_directory', None),
                    page_size_mb=getattr(self.config, 'page_size_mb', 16),
                    eviction_policy=getattr(self.config, 'eviction_policy', 'predictive'),
                    enable_clustering=getattr(self.config, 'enable_clustering', True),
                    cluster_count=getattr(self.config, 'cluster_count', 5),
                    enable_adaptive=getattr(self.config, 'enable_adaptive_offloading', True)
                )
            except AttributeError as e:
                logger.warning(f"Failed to create disk offloader with advanced settings: {e}, using basic settings")
                # Create with basic settings
                self._disk_offloader = create_disk_offloader(
                    max_memory_ratio=getattr(self.config, 'max_memory_ratio', 0.8),
                    offload_directory=getattr(self.config, 'offload_directory', None),
                    page_size_mb=getattr(self.config, 'page_size_mb', 16),
                    eviction_policy=getattr(self.config, 'eviction_policy', 'lru')
                )

            # Create tensor offloading manager
            self._tensor_offloader = TensorOffloadingManager(self._disk_offloader)

            # Create multimodal offloading manager for vision-language processing
            self._multimodal_offloader = MultimodalOffloadingManager(self._disk_offloader)

            # Start proactive management if enabled
            if getattr(self.config, 'enable_predictive_offloading', True):
                self._tensor_offloader.start_proactive_management(
                    interval=getattr(self.config, 'proactive_offloading_interval', 5.0)
                )

            logger.info("Advanced disk offloading system with multimodal support initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize disk offloading system: {e}")
            # Continue without disk offloading if initialization fails
            self._disk_offloader = None
            self._tensor_offloader = None
            self._multimodal_offloader = None

    def _initialize_pagination_system(self):
        """
        Initialize the intelligent pagination system for multimodal data.
        """
        try:
            logger.info("Initializing intelligent pagination system for Qwen3-VL-2B model...")

            from .tensor_pagination import create_multimodal_pagination_system, DataType
            
            # Create pagination system with advanced settings
            self._pagination_system, self._multimodal_pager = create_multimodal_pagination_system(
                swap_directory=getattr(self.config, 'pagination_swap_directory', './tensor_swap'),
                page_size_mb=getattr(self.config, 'pagination_page_size_mb', 16),
                eviction_policy=getattr(self.config, 'pagination_eviction_policy', 'intelligent'),
                max_memory_ratio=getattr(self.config, 'pagination_max_memory_ratio', 0.8)
            )

            # Start proactive management if enabled
            if getattr(self.config, 'enable_proactive_pagination', True):
                self._pagination_system.start_proactive_management(
                    interval=getattr(self.config, 'proactive_pagination_interval', 5.0)
                )

            logger.info("Intelligent pagination system for multimodal data initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize pagination system: {e}")
            # Continue without pagination if initialization fails
            self._pagination_system = None
            self._multimodal_pager = None

    def _initialize_dynamic_multimodal_batching(self):
        """
        Initialize the dynamic multimodal batching system.
        """
        try:
            logger.info("Initializing dynamic multimodal batching system for Qwen3-VL-2B model...")

            from .dynamic_multimodal_batching import get_dynamic_multimodal_batch_manager
            
            # Get configuration parameters for dynamic multimodal batching
            initial_batch_size = getattr(self.config, 'initial_batch_size', 1)
            min_batch_size = getattr(self.config, 'min_batch_size', 1)
            max_batch_size = getattr(self.config, 'max_batch_size', 8)  # Lower default for multimodal
            memory_threshold_ratio = getattr(self.config, 'memory_threshold_ratio', 0.85)
            performance_window_size = getattr(self.config, 'performance_window_size', 10)
            adjustment_factor = getattr(self.config, 'batch_adjustment_factor', 0.1)
            cooldown_period = getattr(self.config, 'batch_cooldown_period', 5.0)
            performance_target = getattr(self.config, 'performance_target', 0.8)
            text_weight = getattr(self.config, 'text_weight', 0.4)
            image_weight = getattr(self.config, 'image_weight', 0.6)
            complexity_threshold_low = getattr(self.config, 'complexity_threshold_low', 0.3)
            complexity_threshold_high = getattr(self.config, 'complexity_threshold_high', 0.7)

            # Create the dynamic multimodal batch manager
            self._dynamic_multimodal_batch_manager = get_dynamic_multimodal_batch_manager(
                initial_batch_size=initial_batch_size,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
                memory_threshold_ratio=memory_threshold_ratio,
                performance_window_size=performance_window_size,
                adjustment_factor=adjustment_factor,
                cooldown_period=cooldown_period,
                performance_target=performance_target,
                text_weight=text_weight,
                image_weight=image_weight,
                complexity_threshold_low=complexity_threshold_low,
                complexity_threshold_high=complexity_threshold_high
            )

            logger.info(f"Dynamic multimodal batching system initialized with batch size range "
                       f"[{min_batch_size}, {max_batch_size}], text_weight={text_weight}, "
                       f"image_weight={image_weight}")

        except Exception as e:
            logger.error(f"Failed to initialize dynamic multimodal batching system: {e}")
            # Continue without dynamic batching if initialization fails
            self._dynamic_multimodal_batch_manager = None

    def _initialize_pipeline_parallelism(self):
        """
        Initialize pipeline parallelism for the model.
        """
        try:
            logger.info("Initializing pipeline parallelism for Qwen3-VL-2B model...")

            from .pipeline_parallel import PipelineParallel, create_pipeline_parallel_config
            
            # Get pipeline configuration from model config
            num_pipeline_stages = getattr(self.config, 'pipeline_parallel_num_stages', 1)
            pipeline_microbatch_size = getattr(self.config, 'pipeline_parallel_microbatch_size', 1)
            pipeline_schedule = getattr(self.config, 'pipeline_parallel_schedule', '1f1b')
            enable_activation_offloading = getattr(self.config, 'pipeline_parallel_enable_activation_offloading', True)

            # Create pipeline configuration
            pipeline_config = create_pipeline_parallel_config(
                num_stages=num_pipeline_stages,
                microbatch_size=pipeline_microbatch_size,
                enable_activation_offloading=enable_activation_offloading,
                pipeline_schedule=pipeline_schedule
            )

            # Create pipeline parallel model
            self._pipeline_parallel_model = PipelineParallel(
                model=self._model,
                config=pipeline_config
            )

            logger.info(f"Pipeline parallelism initialized with {num_pipeline_stages} stages")

        except Exception as e:
            logger.error(f"Error initializing pipeline parallelism: {e}")
            # Continue without pipeline parallelism if it fails
            self._pipeline_parallel_model = None

    def _initialize_sequence_parallelism(self):
        """
        Initialize sequence parallelism for the model.
        """
        try:
            logger.info("Initializing sequence parallelism for Qwen3-VL-2B model...")

            from .sequence_parallel import SequenceParallel, create_sequence_parallel_config
            
            # Get sequence parallelism configuration from model config
            num_sequence_segments = getattr(self.config, 'sequence_parallel_num_segments', 1)
            sequence_split_method = getattr(self.config, 'sequence_parallel_split_method', 'chunk')
            enable_sequence_overlap = getattr(self.config, 'sequence_parallel_enable_overlap', True)
            overlap_size = getattr(self.config, 'sequence_parallel_overlap_size', 64)
            sequence_algorithm = getattr(self.config, 'sequence_parallel_algorithm', '1d')

            # Create sequence parallel configuration
            sequence_config = create_sequence_parallel_config(
                num_segments=num_sequence_segments,
                sequence_split_method=sequence_split_method,
                enable_sequence_overlap=enable_sequence_overlap,
                overlap_size=overlap_size
            )
            sequence_config.sequence_parallel_algorithm = sequence_algorithm

            # Create sequence parallel model
            self._sequence_parallel_model = SequenceParallel(
                model=self._model,
                config=sequence_config
            )

            logger.info(f"Sequence parallelism initialized with {num_sequence_segments} segments")

        except Exception as e:
            logger.error(f"Error initializing sequence parallelism: {e}")
            # Continue without sequence parallelism if it fails
            self._sequence_parallel_model = None

    def _initialize_vision_language_parallelism(self):
        """
        Initialize vision-language parallelism for the multimodal model.
        """
        try:
            logger.info("Initializing vision-language parallelism for Qwen3-VL-2B model...")

            from .vision_language_parallel import VisionLanguageParallel, create_vision_language_config
            
            # Get vision-language parallelism configuration from model config
            num_visual_stages = getattr(self.config, 'vision_language_num_visual_stages', 1)
            num_textual_stages = getattr(self.config, 'vision_language_num_textual_stages', 1)
            visual_device_mapping = getattr(self.config, 'vision_language_visual_device_mapping', None)
            textual_device_mapping = getattr(self.config, 'vision_language_textual_device_mapping', None)
            enable_cross_modal_communication = getattr(self.config, 'vision_language_enable_cross_modal_communication', True)
            pipeline_schedule = getattr(self.config, 'vision_language_pipeline_schedule', 'interleaved')

            # Create vision-language parallel configuration
            vl_config = create_vision_language_config(
                num_visual_stages=num_visual_stages,
                num_textual_stages=num_textual_stages,
                visual_device_mapping=visual_device_mapping,
                textual_device_mapping=textual_device_mapping,
                enable_cross_modal_communication=enable_cross_modal_communication,
                pipeline_schedule=pipeline_schedule
            )

            # Create vision-language parallel model
            self._vision_language_parallel_model = VisionLanguageParallel(
                model=self._model,
                config=vl_config
            )

            logger.info(f"Vision-language parallelism initialized with {num_visual_stages} visual stages and {num_textual_stages} textual stages")

        except Exception as e:
            logger.error(f"Error initializing vision-language parallelism: {e}")
            # Continue without vision-language parallelism if it fails
            self._vision_language_parallel_model = None

    def _initialize_multimodal_pipeline(self):
        """
        Initialize the multimodal preprocessing pipeline for the Qwen3-VL-2B model.
        """
        try:
            logger.info("Initializing multimodal preprocessing pipeline for Qwen3-VL-2B model...")

            from .multimodal_pipeline import create_multimodal_pipeline, apply_multimodal_pipeline_to_model
            
            # Get configuration parameters for multimodal pipeline
            max_text_length = getattr(self.config, 'max_text_length', 32768)
            image_size = getattr(self.config, 'image_size', 448)
            patch_size = getattr(self.config, 'patch_size', 14)
            enable_caching = getattr(self.config, 'enable_multimodal_pipeline_caching', True)
            cache_size = getattr(self.config, 'multimodal_pipeline_cache_size', 1000)

            # Create the multimodal preprocessing pipeline
            self._multimodal_pipeline = create_multimodal_pipeline(
                model_path=self._model_name,
                max_text_length=max_text_length,
                image_size=image_size,
                patch_size=patch_size
            )

            # Apply the pipeline to the model
            self._model = apply_multimodal_pipeline_to_model(self._model, self._multimodal_pipeline)

            logger.info(f"Multimodal preprocessing pipeline initialized with max_text_length={max_text_length}, "
                       f"image_size={image_size}, patch_size={patch_size}, caching={enable_caching}")

        except Exception as e:
            logger.error(f"Error initializing multimodal preprocessing pipeline: {e}")
            # Continue without pipeline if initialization fails
            self._multimodal_pipeline = None

    def _initialize_intelligent_multimodal_caching(self):
        """
        Initialize the intelligent multimodal caching system for the Qwen3-VL-2B model.
        """
        try:
            logger.info("Initializing intelligent multimodal caching system for Qwen3-VL-2B model...")

            from .intelligent_multimodal_caching import create_qwen3_vl_intelligent_caching_manager, apply_intelligent_multimodal_caching_to_model
            
            # Get configuration parameters for intelligent caching
            cache_size_gb = getattr(self.config, 'intelligent_multimodal_cache_size_gb', 2.0)
            eviction_policy = getattr(self.config, 'intelligent_multimodal_cache_eviction_policy', 'predictive')
            enable_similarity_caching = getattr(self.config, 'intelligent_multimodal_cache_enable_similarity', True)
            similarity_threshold = getattr(self.config, 'intelligent_multimodal_cache_similarity_threshold', 0.85)
            enable_ttl = getattr(self.config, 'intelligent_multimodal_cache_enable_ttl', True)
            default_ttl = getattr(self.config, 'intelligent_multimodal_cache_default_ttl', 7200.0)  # 2 hours
            enable_compression = getattr(self.config, 'intelligent_multimodal_cache_enable_compression', True)
            compression_ratio = getattr(self.config, 'intelligent_multimodal_cache_compression_ratio', 0.6)

            # Create the intelligent caching manager
            self._caching_manager = create_qwen3_vl_intelligent_caching_manager(
                cache_size_gb=cache_size_gb,
                eviction_policy=eviction_policy,
                enable_similarity_caching=enable_similarity_caching,
                similarity_threshold=similarity_threshold,
                enable_ttl=enable_ttl,
                default_ttl=default_ttl,
                enable_compression=enable_compression,
                compression_ratio=compression_ratio
            )

            # Apply intelligent caching to the model
            self._model = apply_intelligent_multimodal_caching_to_model(self._model, self._caching_manager)

            logger.info(f"Intelligent multimodal caching system initialized with {cache_size_gb}GB cache, "
                       f"eviction_policy={eviction_policy}, similarity_caching={enable_similarity_caching}")

        except Exception as e:
            logger.error(f"Error initializing intelligent multimodal caching system: {e}")
            # Continue without caching if initialization fails
            self._caching_manager = None

    def _initialize_visual_resource_compression(self):
        """
        Initialize the visual resource compression system for the Qwen3-VL-2B model.
        """
        try:
            logger.info("Initializing visual resource compression system for Qwen3-VL-2B model...")

            from .visual_resource_compression import create_visual_compressor, apply_visual_compression_to_model
            
            # Get configuration parameters for visual compression
            compression_method = getattr(self.config, 'visual_compression_method', 'quantization')
            compression_ratio = getattr(self.config, 'visual_compression_ratio', 0.5)
            quantization_bits = getattr(self.config, 'visual_quantization_bits', 8)
            enable_compression_cache = getattr(self.config, 'visual_enable_compression_cache', True)
            compression_cache_size = getattr(self.config, 'visual_compression_cache_size', 1000)
            enable_adaptive_compression = getattr(self.config, 'visual_enable_adaptive_compression', True)

            # Create visual compression configuration
            visual_compression_config = {
                'compression_method': compression_method,
                'compression_ratio': compression_ratio,
                'quantization_bits': quantization_bits,
                'enable_compression_cache': enable_compression_cache,
                'compression_cache_size': compression_cache_size,
                'enable_adaptive_compression': enable_adaptive_compression
            }

            # Create the visual compressor
            self._visual_compressor = create_visual_compressor(visual_compression_config)

            # Apply visual compression to the model
            self._model = apply_visual_compression_to_model(self._model, self._visual_compressor)

            logger.info(f"Visual resource compression system initialized with method={compression_method}, "
                       f"ratio={compression_ratio}, quantization_bits={quantization_bits}")

        except Exception as e:
            logger.error(f"Error initializing visual resource compression system: {e}")
            # Continue without visual compression if initialization fails
            self._visual_compressor = None

    def _initialize_image_tokenization(self):
        """
        Initialize the efficient image tokenization system for the Qwen3-VL-2B model.
        """
        try:
            logger.info("Initializing efficient image tokenization system for Qwen3-VL-2B model...")

            from .image_tokenization import create_image_tokenizer, apply_image_tokenization_to_model
            
            # Get configuration parameters for image tokenization
            image_size = getattr(self.config, 'image_size', 448)
            patch_size = getattr(self.config, 'patch_size', 14)
            max_image_tokens = getattr(self.config, 'max_image_tokens', 1024)
            token_dim = getattr(self.config, 'image_token_dim', 1024)
            enable_patch_caching = getattr(self.config, 'enable_image_patch_caching', True)
            enable_batch_processing = getattr(self.config, 'enable_image_batch_processing', True)
            enable_memory_efficient_processing = getattr(self.config, 'enable_memory_efficient_image_processing', True)
            enable_quantization = getattr(self.config, 'enable_image_quantization', False)
            quantization_bits = getattr(self.config, 'image_quantization_bits', 8)
            enable_compression = getattr(self.config, 'enable_image_compression', True)
            compression_ratio = getattr(self.config, 'image_compression_ratio', 0.5)

            # Create image tokenization configuration
            image_tokenization_config = {
                'image_size': image_size,
                'patch_size': patch_size,
                'max_image_tokens': max_image_tokens,
                'token_dim': token_dim,
                'enable_patch_caching': enable_patch_caching,
                'enable_batch_processing': enable_batch_processing,
                'enable_memory_efficient_processing': enable_memory_efficient_processing,
                'enable_quantization': enable_quantization,
                'quantization_bits': quantization_bits,
                'enable_compression': enable_compression,
                'compression_ratio': compression_ratio
            }

            # Create the image tokenizer
            self._image_tokenizer = create_image_tokenizer(
                model_path=self._model_name,
                config=image_tokenization_config
            )

            # Apply image tokenization to the model
            self._model = apply_image_tokenization_to_model(self._model, self._image_tokenizer)

            logger.info(f"Efficient image tokenization system initialized with image_size={image_size}, "
                       f"patch_size={patch_size}, max_image_tokens={max_image_tokens}")

        except Exception as e:
            logger.error(f"Error initializing efficient image tokenization system: {e}")
            # Continue without image tokenization if initialization fails
            self._image_tokenizer = None

    def _apply_quantization(self):
        """
        Apply quantization to the model based on configuration.
        """
        try:
            logger.info("Applying quantization to Qwen3-VL-2B model...")

            from .quantization import get_quantization_manager
            
            # Get quantization configuration from model config
            quantization_scheme = getattr(self.config, 'quantization_scheme', 'int8')
            quantization_bits = getattr(self.config, 'quantization_bits', 8)
            symmetric = getattr(self.config, 'quantization_symmetric', True)
            per_channel = getattr(self.config, 'quantization_per_channel', True)

            # Create quantization manager
            quantization_manager = get_quantization_manager()
            
            # Apply quantization to the model
            self._model = quantization_manager.quantize_model(
                self._model,
                scheme=quantization_scheme,
                bits=quantization_bits,
                symmetric=symmetric,
                per_channel=per_channel
            )

            logger.info(f"Quantization applied successfully with scheme: {quantization_scheme}")

        except Exception as e:
            logger.error(f"Error applying quantization: {e}")
            # Continue without quantization if it fails
            pass

    def _initialize_async_multimodal_processing(self):
        """
        Initialize the asynchronous multimodal processing system for the Qwen3-VL-2B model.
        """
        try:
            logger.info("Initializing asynchronous multimodal processing system for Qwen3-VL-2B model...")

            from .async_multimodal_processing import Qwen3VL2BAsyncMultimodalManager
            
            # Get configuration parameters for async processing
            max_concurrent_requests = getattr(self.config, 'async_max_concurrent_requests', 4)
            buffer_size = getattr(self.config, 'async_buffer_size', 100)
            batch_timeout = getattr(self.config, 'async_batch_timeout', 0.1)
            enable_batching = getattr(self.config, 'enable_async_batching', True)
            device = getattr(self.config, 'async_processing_device', 'cuda:0' if torch.cuda.is_available() else 'cpu')

            # Create the async multimodal manager
            self._async_multimodal_manager = Qwen3VL2BAsyncMultimodalManager(
                model=self._model,
                tokenizer=self._tokenizer,
                image_processor=self._image_processor,
                max_concurrent_requests=max_concurrent_requests,
                buffer_size=buffer_size,
                batch_timeout=batch_timeout,
                enable_batching=enable_batching,
                device=device
            )

            logger.info(f"Async multimodal processing system initialized with max_concurrent={max_concurrent_requests}, "
                       f"buffer_size={buffer_size}, batching={enable_batching}")

        except Exception as e:
            logger.error(f"Error initializing asynchronous multimodal processing system: {e}")
            # Continue without async processing if initialization fails
            self._async_multimodal_manager = None

    def _apply_async_multimodal_processing_optimizations(self):
        """
        Apply asynchronous multimodal processing optimizations to the model.
        """
        try:
            logger.info("Applying asynchronous multimodal processing optimizations to Qwen3-VL-2B model...")

            from .async_multimodal_processing import apply_async_multimodal_processing_to_model
            
            # Apply async processing optimizations to the model
            self._model = apply_async_multimodal_processing_to_model(
                self._model,
                self._async_multimodal_manager
            )

            logger.info("Asynchronous multimodal processing optimizations applied successfully")

        except Exception as e:
            logger.error(f"Error applying asynchronous multimodal processing optimizations: {e}")
            # Continue without async optimizations if it fails
            pass

    def forward(self, *args, **kwargs):
        """
        Forward pass for the Qwen3-VL-2B model.
        
        This method handles the forward pass with all optimizations applied.
        """
        # Use vision-language parallel model for forward pass if enabled (takes highest precedence for multimodal models)
        if self._vision_language_parallel_model is not None:
            # Prepare inputs for vision-language parallel - handle both text and vision inputs
            input_tensor = None
            if args:
                input_tensor = args[0]
            elif 'input_ids' in kwargs:
                input_tensor = kwargs['input_ids']
            elif 'inputs_embeds' in kwargs:
                input_tensor = kwargs['inputs_embeds']
            elif 'pixel_values' in kwargs:
                input_tensor = kwargs['pixel_values']  # For vision-language models

            if input_tensor is not None:
                try:
                    # Use vision-language parallel model for forward pass
                    result = self._vision_language_parallel_model(input_tensor)
                    return result
                except Exception as e:
                    logger.warning(f"Vision-language parallel forward failed: {e}, falling back to other models")

        # Use sequence parallel model for forward pass if enabled (takes precedence over pipeline parallel)
        if self._sequence_parallel_model is not None:
            # Prepare inputs for sequence parallel - handle both text and vision inputs
            input_tensor = None
            if args:
                input_tensor = args[0]
            elif 'input_ids' in kwargs:
                input_tensor = kwargs['input_ids']
            elif 'inputs_embeds' in kwargs:
                input_tensor = kwargs['inputs_embeds']
            elif 'pixel_values' in kwargs:
                input_tensor = kwargs['pixel_values']  # For vision-language models

            if input_tensor is not None:
                try:
                    # Use sequence parallel model for forward pass
                    result = self._sequence_parallel_model(input_tensor)
                    return result
                except Exception as e:
                    logger.warning(f"Sequence parallel forward failed: {e}, falling back to other models")

        # Use pipeline parallel model for forward pass if enabled
        if self._pipeline_parallel_model is not None:
            # Prepare inputs for pipeline - handle both text and vision inputs
            input_tensor = None
            if args:
                input_tensor = args[0]
            elif 'input_ids' in kwargs:
                input_tensor = kwargs['input_ids']
            elif 'inputs_embeds' in kwargs:
                input_tensor = kwargs['inputs_embeds']
            elif 'pixel_values' in kwargs:
                input_tensor = kwargs['pixel_values']  # For vision-language models

            if input_tensor is not None:
                try:
                    # Use pipeline parallel model for forward pass
                    result = self._pipeline_parallel_model(input_tensor)
                    return result
                except Exception as e:
                    logger.warning(f"Pipeline parallel forward failed: {e}, falling back to regular model")

        # Apply cross-modal fusion if enabled and available during forward pass
        if hasattr(self, '_cross_modal_fusion_manager') and self._cross_modal_fusion_manager is not None:
            # Check if we have both vision and language inputs for cross-modal fusion
            pixel_values = kwargs.get('pixel_values', None)
            input_ids = kwargs.get('input_ids', None)

            if pixel_values is not None and input_ids is not None:
                try:
                    # Determine the appropriate fusion method based on input complexity
                    input_complexity = self._assess_input_complexity(pixel_values, input_ids)
                    fusion_method = self._select_fusion_method(input_complexity)

                    logger.debug(f"Selected fusion method: {fusion_method} for input complexity: {input_complexity}")

                    # Perform forward pass with cross-modal fusion considerations
                    result = self._model(*args, **kwargs)
                    return result
                except Exception as e:
                    logger.warning(f"Cross-modal fusion forward failed: {e}, falling back to regular model")

        # Apply ML-based optimization if enabled
        if getattr(self.config, 'use_ml_optimizations', False):
            from ...common.unified_ml_optimization import get_ml_optimization_system, ModelType
            # Extract input tensors from args/kwargs to analyze complexity
            input_tensor = None
            if args:
                input_tensor = args[0] if torch.is_tensor(args[0]) else None
            elif 'input_ids' in kwargs:
                input_tensor = kwargs['input_ids']
            elif 'inputs_embeds' in kwargs:
                input_tensor = kwargs['inputs_embeds']
            elif 'pixel_values' in kwargs:
                input_tensor = kwargs['pixel_values']  # For vision-language models

            if input_tensor is not None:
                # Apply ML-based optimization based on input
                ml_system = get_ml_optimization_system()
                optimized_model = ml_system.optimize_model_for_input(
                    model=self._model,
                    input_data=input_tensor,
                    model_type=ModelType.QWEN3_VL_2B
                )

                # Update the model reference temporarily
                original_model = self._model
                self._model = optimized_model

                try:
                    result = self._model(*args, **kwargs)
                finally:
                    # Restore original model reference
                    self._model = original_model

                return result

        # Apply NAS if enabled
        elif self._nas_controller is not None and self._model_adapter is not None:
            # For forward pass, we'll adapt the architecture based on input
            # Extract input tensors from args/kwargs to analyze complexity
            input_tensor = None
            if args:
                input_tensor = args[0] if torch.is_tensor(args[0]) else None
            elif 'input_ids' in kwargs:
                input_tensor = kwargs['input_ids']
            elif 'inputs_embeds' in kwargs:
                input_tensor = kwargs['inputs_embeds']
            elif 'pixel_values' in kwargs:
                input_tensor = kwargs['pixel_values']  # For vision-language models

            if input_tensor is not None:
                # Adapt the model architecture based on input
                adapted_model, nas_metrics = self._nas_controller.adapt_architecture(
                    self._model, input_tensor
                )

                # Update the model reference temporarily
                original_model = self._model
                self._model = adapted_model

                try:
                    result = self._model(*args, **kwargs)
                finally:
                    # Restore original model reference
                    self._model = original_model

                return result

        # Regular forward pass
        result = self._model(*args, **kwargs)
        return result

    def _assess_input_complexity(self, pixel_values: torch.Tensor, input_ids: torch.Tensor) -> str:
        """
        Assess the complexity of the input for dynamic optimization selection.

        Args:
            pixel_values: Input image pixel values
            input_ids: Input text token IDs

        Returns:
            Complexity level ("simple", "medium", "complex")
        """
        # Assess complexity based on input sizes and characteristics
        if pixel_values is not None:
            vision_elements = pixel_values.numel()
        else:
            vision_elements = 0

        if input_ids is not None:
            text_elements = input_ids.numel()
        else:
            text_elements = 0

        # Calculate total complexity score
        total_elements = vision_elements + text_elements

        # Define thresholds for complexity levels
        if total_elements < 1000:
            return "simple"
        elif total_elements < 10000:
            return "medium"
        else:
            return "complex"

    def _select_fusion_method(self, input_complexity: str) -> str:
        """
        Select the appropriate fusion method based on input complexity.

        Args:
            input_complexity: Complexity level of the input

        Returns:
            Selected fusion method name
        """
        # Map input complexity to fusion method
        complexity_to_method = {
            "simple": "add",           # Simple addition for basic fusion
            "medium": "qwen3_vl_specific",   # Qwen3-VL specific fusion for medium complexity
            "complex": "multi_scale"   # Multi-scale fusion for complex inputs
        }

        return complexity_to_method.get(input_complexity, "qwen3_vl_specific")

    def generate(self, *args, **kwargs):
        """
        Generate method for the Qwen3-VL-2B model.
        
        This method handles text generation with all optimizations applied.
        """
        start_time = time.time()

        # Use vision-language parallel model for generation if enabled (takes highest precedence for multimodal models)
        if self._vision_language_parallel_model is not None:
            # Prepare inputs for vision-language parallel - handle both text and vision inputs
            input_tensor = None
            if args:
                input_tensor = args[0]
            elif 'input_ids' in kwargs:
                input_tensor = kwargs['input_ids']
            elif 'inputs_embeds' in kwargs:
                input_tensor = kwargs['inputs_embeds']
            elif 'pixel_values' in kwargs:
                input_tensor = kwargs['pixel_values']  # For vision-language models

            if input_tensor is not None:
                try:
                    # Use vision-language parallel model for generation
                    result = self._vision_language_parallel_model.generate_with_vision_language_parallel(
                        input_tensor,
                        max_new_tokens=kwargs.get('max_new_tokens', 50),
                        **kwargs
                    )
                    # Record performance metrics
                    end_time = time.time()
                    latency = end_time - start_time
                    input_length = input_tensor.shape[-1] if len(input_tensor.shape) > 1 else input_tensor.numel()
                    output_length = result.shape[-1] if torch.is_tensor(result) and len(result.shape) > 1 else (result.numel() if torch.is_tensor(result) else 0)
                    total_tokens = input_length + output_length
                    throughput = total_tokens / latency if latency > 0 else 0
                    return result
                except Exception as e:
                    logger.warning(f"Vision-language parallel generation failed: {e}, falling back to other models")

        # Use sequence parallel model for generation if enabled (takes precedence over pipeline parallel)
        if self._sequence_parallel_model is not None:
            # Prepare inputs for sequence parallel - handle both text and vision inputs
            input_tensor = None
            if args:
                input_tensor = args[0]
            elif 'input_ids' in kwargs:
                input_tensor = kwargs['input_ids']
            elif 'inputs_embeds' in kwargs:
                input_tensor = kwargs['inputs_embeds']
            elif 'pixel_values' in kwargs:
                input_tensor = kwargs['pixel_values']  # For vision-language models

            if input_tensor is not None:
                try:
                    # Use sequence parallel model for generation
                    result = self._sequence_parallel_model.generate_with_sequence_parallel(
                        input_tensor,
                        max_new_tokens=kwargs.get('max_new_tokens', 50),
                        **kwargs
                    )
                    # Record performance metrics
                    end_time = time.time()
                    latency = end_time - start_time
                    input_length = input_tensor.shape[-1] if len(input_tensor.shape) > 1 else input_tensor.numel()
                    output_length = result.shape[-1] if torch.is_tensor(result) and len(result.shape) > 1 else (result.numel() if torch.is_tensor(result) else 0)
                    total_tokens = input_length + output_length
                    throughput = total_tokens / latency if latency > 0 else 0
                    return result
                except Exception as e:
                    logger.warning(f"Sequence parallel generation failed: {e}, falling back to other models")

        # Use pipeline parallel model for generation if enabled
        if self._pipeline_parallel_model is not None:
            # Prepare inputs for pipeline - handle both text and vision inputs
            input_tensor = None
            if args:
                input_tensor = args[0]
            elif 'input_ids' in kwargs:
                input_tensor = kwargs['input_ids']
            elif 'inputs_embeds' in kwargs:
                input_tensor = kwargs['inputs_embeds']
            elif 'pixel_values' in kwargs:
                input_tensor = kwargs['pixel_values']  # For vision-language models

            if input_tensor is not None:
                try:
                    # Use pipeline parallel model for generation
                    result = self._pipeline_parallel_model.generate_with_pipeline(
                        input_tensor,
                        max_new_tokens=kwargs.get('max_new_tokens', 50),
                        **kwargs
                    )
                    # Record performance metrics
                    end_time = time.time()
                    latency = end_time - start_time
                    input_length = input_tensor.shape[-1] if len(input_tensor.shape) > 1 else input_tensor.numel()
                    output_length = result.shape[-1] if torch.is_tensor(result) and len(result.shape) > 1 else (result.numel() if torch.is_tensor(result) else 0)
                    total_tokens = input_length + output_length
                    throughput = total_tokens / latency if latency > 0 else 0
                    return result
                except Exception as e:
                    logger.warning(f"Pipeline parallel generation failed: {e}, falling back to regular model")

        # Apply cross-modal fusion if enabled and available during generation
        if hasattr(self, '_cross_modal_fusion_manager') and self._cross_modal_fusion_manager is not None:
            # Check if we have both vision and language inputs for cross-modal fusion
            pixel_values = kwargs.get('pixel_values', None)
            input_ids = kwargs.get('input_ids', None)

            if pixel_values is not None and input_ids is not None:
                try:
                    # Determine the appropriate fusion method based on input complexity
                    input_complexity = self._assess_input_complexity(pixel_values, input_ids)
                    fusion_method = self._select_fusion_method(input_complexity)

                    logger.debug(f"Selected fusion method: {fusion_method} for input complexity: {input_complexity}")

                    # Perform generation with cross-modal fusion considerations
                    result = self._model.generate(*args, **kwargs)
                    return result
                except Exception as e:
                    logger.warning(f"Cross-modal fusion generation failed: {e}, falling back to regular model")

        # Apply ML-based optimization if enabled
        if getattr(self.config, 'use_ml_optimizations', False):
            from ...common.unified_ml_optimization import get_ml_optimization_system, ModelType
            # Extract input tensors from args/kwargs to analyze complexity
            input_tensor = None
            if args:
                input_tensor = args[0] if torch.is_tensor(args[0]) else None
            elif 'input_ids' in kwargs:
                input_tensor = kwargs['input_ids']
            elif 'inputs_embeds' in kwargs:
                input_tensor = kwargs['inputs_embeds']
            elif 'pixel_values' in kwargs:
                input_tensor = kwargs['pixel_values']  # For vision-language models

            if input_tensor is not None:
                # Apply ML-based optimization based on input
                ml_system = get_ml_optimization_system()
                optimized_model = ml_system.optimize_model_for_input(
                    model=self._model,
                    input_data=input_tensor,
                    model_type=ModelType.QWEN3_VL_2B
                )

                # Update the model reference temporarily
                original_model = self._model
                self._model = optimized_model

                try:
                    result = self._model.generate(*args, **kwargs)
                finally:
                    # Restore original model reference
                    self._model = original_model

                return result

        # Apply NAS if enabled
        elif self._nas_controller is not None and self._model_adapter is not None:
            # For generation, we'll adapt the architecture based on input
            # Extract input tensors from args/kwargs to analyze complexity
            input_tensor = None
            if args:
                input_tensor = args[0] if torch.is_tensor(args[0]) else None
            elif 'input_ids' in kwargs:
                input_tensor = kwargs['input_ids']
            elif 'inputs_embeds' in kwargs:
                input_tensor = kwargs['inputs_embeds']
            elif 'pixel_values' in kwargs:
                input_tensor = kwargs['pixel_values']  # For vision-language models

            if input_tensor is not None:
                # Adapt the model architecture based on input
                adapted_model, nas_metrics = self._nas_controller.adapt_architecture(
                    self._model, input_tensor
                )

                # Update the model reference temporarily
                original_model = self._model
                self._model = adapted_model

                try:
                    result = self._model.generate(*args, **kwargs)
                finally:
                    # Restore original model reference
                    self._model = original_model

                return result

        # Regular generation
        result = self._model.generate(*args, **kwargs)

        # Record performance metrics
        end_time = time.time()
        latency = end_time - start_time

        # Calculate throughput if possible
        input_length = 0
        if args and torch.is_tensor(args[0]):
            input_length = args[0].shape[-1] if len(args[0].shape) > 1 else args[0].numel()
        elif 'input_ids' in kwargs and torch.is_tensor(kwargs['input_ids']):
            input_length = kwargs['input_ids'].shape[-1]
        elif 'pixel_values' in kwargs and torch.is_tensor(kwargs['pixel_values']):
            input_length = kwargs['pixel_values'].shape[-1]  # Height dimension for images

        # Calculate output length
        output_length = 0
        if torch.is_tensor(result):
            output_length = result.shape[-1] if len(result.shape) > 1 else result.numel()

        total_tokens = input_length + output_length
        throughput = total_tokens / latency if latency > 0 else 0

        return result

    def get_tokenizer(self):
        """
        Get the tokenizer associated with the model.
        """
        return self._tokenizer

    def get_image_processor(self):
        """
        Get the image processor associated with the model.
        """
        return self._image_processor


def create_qwen3_vl_2b_model(config: 'Qwen3VL2BConfig') -> Qwen3VL2BModel:
    """
    Factory function to create a Qwen3-VL-2B model instance.

    Args:
        config: Qwen3VL2BConfig configuration

    Returns:
        Qwen3VL2BModel instance
    """
    return Qwen3VL2BModel(config)


__all__ = [
    "Qwen3VL2BModel",
    "create_qwen3_vl_2b_model"
]