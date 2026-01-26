"""
GLM-4.7-Flash Safe Model Implementation - Ensuring Unimodal (Text-Only) Operation

This module implements the GLM-4.7-Flash model with explicit safeguards to ensure
it remains unimodal (text-only) and does not incorporate any multimodal optimizations.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Type, Generator

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import GLM47Config
from .plugin_modules.glm47_attention import create_glm47_flash_attention_2
from .plugin_modules.glm47_sparse_attention import create_glm47_sparse_attention
from .plugin_modules.glm47_sliding_window_attention import create_glm47_sliding_window_attention
from .plugin_modules.glm47_multi_query_attention import create_mqa_gqa_attention
from .plugin_modules.glm47_paged_attention import create_glm47_paged_attention
from ...common.adaptive_sparse_attention import create_adaptive_sparse_attention
from ...common.adaptive_batch_manager import get_adaptive_batch_manager
from ...common.input_complexity_analyzer import get_complexity_analyzer
from ...common.streaming_computation import (
    StreamRequest,
    StreamResult,
    StreamingComputationEngine,
    create_streaming_engine
)
from ...common.optimization_integration import (
    apply_glm_optimizations,
    legacy_apply_flash_attention,
    legacy_apply_sparse_attention,
    legacy_apply_disk_offloading,
    legacy_apply_activation_offloading,
    legacy_apply_tensor_compression,
    legacy_apply_structured_pruning,
    legacy_apply_kernel_fusion
)
from ...common.unified_ml_optimization import get_ml_optimization_system, ModelType
from typing import Callable
from concurrent.futures import Future
from .plugin_modules.glm47_rotary_embeddings import GLM47RotaryEmbedding
from .plugin_modules.glm47_fused_layers import replace_layer_norm_in_model
from .plugin_modules.glm47_bias_removal import apply_bias_removal_to_model
from .plugin_modules.glm47_tensor_parallel import (
    TensorParallelConfig,
    safe_convert_to_tensor_parallel
)
from .plugin_modules.glm47_kv_cache import (
    apply_compressed_kv_cache_to_model
)
from .plugin_modules.glm47_prefix_cache import (
    apply_prefix_cache_to_model
)
from .plugin_modules.glm47_cuda_kernels import (
    apply_glm47_optimizations_to_model
)

from ...common.structured_pruning import (
    apply_structured_pruning,
    PruningMethod
)
from ...common.nas_controller import (
    get_nas_controller,
    NASConfig,
    ArchitectureAdaptationStrategy
)
from ...common.model_adapter import get_model_adapter
from ...common.tensor_decomposition import (
    get_tensor_decomposer,
    decompose_model_weights,
    recompose_model_weights
)
from ...common.snn import (
    convert_dense_to_snn,
    apply_snn_optimizations
)

logger = logging.getLogger(__name__)


class GLM47SafeModel(nn.Module):
    """
    GLM-4.7-Flash model implementation with explicit safeguards to ensure unimodal (text-only) operation.
    
    This implementation ensures that the model remains text-only and does not incorporate
    any multimodal optimizations that would be inappropriate for this unimodal model.
    """

    def __init__(self, config: GLM47Config):
        super().__init__()
        
        # Validate that the model is configured as unimodal
        if hasattr(config, 'supported_modalities'):
            if 'vision' in config.supported_modalities or 'image' in config.supported_modalities:
                raise ValueError("GLM-4.7 is a unimodal text model and should not support vision/image modalities")
        
        self.config = config

        # Initialize the base model
        self._model = None
        self._tokenizer = None
        self._model_name = config.model_path

        # Memory optimization settings specific to GLM-4.7-Flash model
        self._memory_config = {
            "gradient_checkpointing": config.gradient_checkpointing,
            "use_cache": config.use_cache,
            "torch_dtype": getattr(torch, config.torch_dtype),
            "device_map": config.device_map,
            "low_cpu_mem_usage": config.low_cpu_mem_usage,
            "max_memory": config.max_memory
        }

        # Initialize NAS controller if enabled
        self._nas_controller = None
        self._model_adapter = None
        if getattr(config, 'enable_continuous_nas', False):
            # Get the strategy as string and convert to enum
            strategy_str = getattr(config, 'nas_strategy', 'combined_adaptive')
            if isinstance(strategy_str, str):
                # Map string to enum
                strategy_map = {
                    'depth_adaptive': ArchitectureAdaptationStrategy.DEPTH_ADAPTIVE,
                    'width_adaptive': ArchitectureAdaptationStrategy.WIDTH_ADAPTIVE,
                    'combined_adaptive': ArchitectureAdaptationStrategy.COMBINED_ADAPTIVE,
                    'latency_based': ArchitectureAdaptationStrategy.LATENCY_BASED,
                    'memory_based': ArchitectureAdaptationStrategy.MEMORY_BASED
                }
                strategy_enum = strategy_map.get(strategy_str.lower(), ArchitectureAdaptationStrategy.COMBINED_ADAPTIVE)
            else:
                strategy_enum = strategy_str  # Already an enum

            nas_config = NASConfig(
                strategy=strategy_enum,
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

        # Initialize the model
        self._initialize_model()

    def _validate_unimodal_operation(self, inputs: Any) -> bool:
        """
        Validate that inputs are appropriate for unimodal (text-only) operation.
        
        Args:
            inputs: Input data to validate
            
        Returns:
            True if inputs are valid for unimodal operation, False otherwise
        """
        # Check if inputs contain any vision/image data
        if isinstance(inputs, dict):
            # Check for common vision-related keys
            vision_keys = ['pixel_values', 'images', 'image_features', 'vision_input', 'visual_features']
            for key in vision_keys:
                if key in inputs:
                    logger.error(f"Invalid multimodal input detected: '{key}' key found in inputs")
                    return False
        
        # Check if inputs are tensors with unexpected dimensions for text
        if torch.is_tensor(inputs):
            # Text inputs typically have 2-3 dimensions: [batch, seq] or [batch, seq, features]
            # Vision inputs often have 4 dimensions: [batch, channels, height, width]
            if len(inputs.shape) == 4:
                logger.warning("Input tensor has 4 dimensions, which may indicate vision data in text-only model")
                return False
        
        return True

    def _initialize_model(self):
        """
        Initialize the GLM-4.7-Flash model with appropriate optimizations.
        """
        try:
            logger.info(f"Loading GLM-4.7-Flash model from: {self._model_name}")

            # Prepare loading arguments with memory optimizations
            load_kwargs = {
                "torch_dtype": self._memory_config.get("torch_dtype", torch.float16),
                "trust_remote_code": True,
                "low_cpu_mem_usage": self._memory_config.get("low_cpu_mem_usage", True),
            }

            # Add device_map if specified (avoid disk offloading issues)
            device_map = self._memory_config.get("device_map", "auto")
            if device_map and device_map not in ["disk", "disk_0"]:  # Avoid invalid "disk" device
                load_kwargs["device_map"] = device_map

            # Add max_memory if specified (but avoid disk-related issues)
            max_memory = self._memory_config.get("max_memory")
            if max_memory:
                # Only add max_memory if it doesn't cause disk offloading issues
                load_kwargs["max_memory"] = max_memory

            # Check if the drive exists to avoid hanging on non-existent drives
            import os
            drive_exists = True
            # Handle both forward slash and backslash formats
            path_sep = '\\' if '\\' in self._model_name else '/'
            if ':' in self._model_name:
                drive_letter = self._model_name.split(':')[0] + ':'
                if not os.path.exists(drive_letter):
                    logger.warning(f"Drive {drive_letter} does not exist, using HuggingFace model...")
                    drive_exists = False

            # Try to load the model from the local path
            try:
                if drive_exists and os.path.exists(self._model_name):
                    logger.info(f"Using local model from: {self._model_name}")

                    # Load the real GLM-4.7-Flash model with its specific architecture
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    # Load tokenizer first
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self._model_name,
                        trust_remote_code=True,
                        padding_side="left"
                    )

                    # Load the model with specific GLM-4.7-Flash parameters
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self._model_name,
                        **load_kwargs
                    )

                    # Set pad token if not set
                    if self._tokenizer.pad_token is None:
                        self._tokenizer.pad_token = self._tokenizer.eos_token
                else:
                    # Drive doesn't exist or path doesn't exist, use fallback
                    raise OSError(f"Local model path {self._model_name} not found")
            except OSError:
                # If the local path doesn't exist, try to load from HuggingFace
                logger.warning(f"Local model path {self._model_name} not found, trying HuggingFace...")
                self._model = AutoModelForCausalLM.from_pretrained(
                    "THUDM/glm-4-9b",  # Fallback to HF name (closest available)
                    **load_kwargs
                )
                self._model_name = "THUDM/glm-4-9b"

            # Enable gradient checkpointing for memory efficiency if configured
            if self._memory_config.get("gradient_checkpointing", True):
                self._model.gradient_checkpointing_enable()

            # Load the tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
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
            logger.error(f"Failed to initialize GLM-4.7 model: {e}")
            raise

    def _apply_configured_optimizations(self):
        """
        Apply optimizations based on the configuration settings.
        """
        # Check if ML-based optimization is enabled
        if getattr(self.config, 'use_ml_optimizations', False):
            # Use ML-based optimization system
            ml_system = get_ml_optimization_system()
            self._model = ml_system.optimize_model_for_input(
                model=self._model,
                input_data=None,  # Initially no input, will be optimized per request
                model_type=ModelType.GLM_4_7
            )
        elif getattr(self.config, 'use_modular_optimizations', False):
            # Apply optimizations using the new modular system if enabled
            profile_name = getattr(self.config, 'optimization_profile', 'balanced')
            self._model = apply_glm_optimizations(self._model, profile_name)
        else:
            # Apply traditional optimizations for backward compatibility
            self._apply_traditional_optimizations()

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
            decomposition_method = getattr(self.config, 'tensor_decomposition_method', 'cp_decomposition')
            rank_ratio = getattr(self.config, 'tensor_decomposition_rank_ratio', 0.5)

            self._apply_tensor_decomposition(
                rank_ratio=rank_ratio,
                decomposition_method=decomposition_method
            )

        # Apply structured pruning if enabled
        if self.config.use_structured_pruning:
            # Convert string method to enum
            method_map = {
                "layer_removal": PruningMethod.LAYER_REMOVAL,
                "block_removal": PruningMethod.BLOCK_REMOVAL,
                "head_removal": PruningMethod.HEAD_REMOVAL,
                "mlp_removal": PruningMethod.MLP_REMOVAL,
                "adaptive_pruning": PruningMethod.ADAPTIVE_PRUNING
            }
            pruning_method = method_map.get(self.config.pruning_method, PruningMethod.LAYER_REMOVAL)

            self._apply_structured_pruning(
                pruning_ratio=self.config.pruning_ratio,
                method=pruning_method,
                block_size=self.config.pruning_block_size
            )

        # Apply SNN conversion if enabled
        if getattr(self.config, 'use_snn_conversion', False):
            self._apply_snn_conversion()

    def _apply_tensor_parallelism(self):
        """
        Apply tensor parallelism to the model if configured.
        """
        try:
            tensor_parallel_size = self.config.tensor_parallel_size
            if tensor_parallel_size > 1:
                logger.info(f"Applying tensor parallelism with size {tensor_parallel_size}")

                # Initialize tensor parallelism
                tp_config = TensorParallelConfig(
                    tensor_parallel_size=tensor_parallel_size,
                    local_rank=self.config.tensor_parallel_local_rank,
                    world_size=self.config.tensor_parallel_world_size,
                    init_method=self.config.tensor_parallel_init_method
                )

                # Use safe conversion with error handling
                self._model, success, error_msg = safe_convert_to_tensor_parallel(self._model, tp_config)
                if not success:
                    logger.warning(f"Failed to convert model to tensor parallel: {error_msg}")
                    logger.info("Disabling tensor parallelism and proceeding with regular model")
                    self.config.use_tensor_parallelism = False
                else:
                    logger.info(f"Model converted to tensor parallel with size {tensor_parallel_size}")
        except Exception as e:
            logger.error(f"Error applying tensor parallelism: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_flash_attention_2_optimization(self):
        """
        Apply FlashAttention 2.0, sparse attention, sliding window attention, paged attention, or MQA/GQA optimization to the model by replacing
        attention layers with optimized implementations based on configuration.
        """
        try:
            # Determine which attention optimization to use based on config
            use_sparse_attention = self.config.use_sparse_attention
            use_sliding_window_attention = self.config.use_sliding_window_attention
            use_flash_attention = self.config.use_flash_attention_2
            use_multi_query_attention = self.config.use_multi_query_attention
            use_grouped_query_attention = self.config.use_grouped_query_attention
            use_paged_attention = self.config.use_paged_attention

            # Priority order: Paged > Sliding Window > MQA/GQA > Sparse > FlashAttention
            if use_paged_attention:
                logger.info("Applying paged attention optimization to GLM-4.7 model...")
                self._apply_paged_attention_optimization()
            elif use_sliding_window_attention:
                logger.info("Applying sliding window attention optimization to GLM-4.7 model...")
                self._apply_sliding_window_attention_optimization()
            elif use_multi_query_attention or use_grouped_query_attention:
                logger.info("Applying Multi-Query/Grouped-Query attention optimization to GLM-4.7 model...")
                self._apply_mqa_gqa_optimization()
            elif use_sparse_attention:
                logger.info("Applying sparse attention optimization to GLM-4.7 model...")
                self._apply_sparse_attention_optimization()
            elif use_flash_attention:
                logger.info("Applying FlashAttention 2.0 optimization to GLM-4.7 model...")
                self._apply_flash_attention_optimization()
            else:
                logger.info("Attention optimization disabled for GLM-4.7 model")

            # Always apply optimized rotary embeddings regardless of attention optimization choice
            logger.info("Applying optimized rotary embeddings to GLM-4.7 model...")
            self._apply_optimized_rotary_embedding()

        except ImportError as e:
            logger.warning(f"Attention optimization not available, skipping: {e}")
        except Exception as e:
            logger.error(f"Error applying attention optimization: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_sparse_attention_optimization(self):
        """
        Apply sparse attention optimization to the model by replacing
        attention layers with sparse attention implementations.
        """
        try:
            # Replace attention layers in each transformer layer
            replaced_count = 0
            total_layers = 0

            # Access the model's transformer layers
            if hasattr(self._model, 'transformer') and hasattr(self._model.transformer, 'layers'):
                # GLM-style architecture
                layers = self._model.transformer.layers
            elif hasattr(self._model, 'model') and hasattr(self._model.model, 'layers'):
                # Another common architecture pattern
                layers = self._model.model.layers
            elif hasattr(self._model, 'transformer') and hasattr(self._model.transformer, 'h'):
                # GPT-style architecture
                layers = self._model.transformer.h
            else:
                # Default to empty list
                layers = []

            # Iterate through layers and replace attention mechanisms
            for idx, layer in enumerate(layers):
                total_layers += 1

                # Check if the layer has self-attention mechanism
                if hasattr(layer, 'self_attn') or hasattr(layer, 'attn'):
                    try:
                        # Get the attention submodule
                        if hasattr(layer, 'self_attn'):
                            orig_attn = layer.self_attn
                            # Create sparse attention implementation - use adaptive version if enabled
                            if getattr(self.config, 'use_adaptive_sparse_attention', False):
                                sparse_attn = create_adaptive_sparse_attention(
                                    self.config,
                                    layer_idx=idx,
                                    adaptive=True,
                                    adaptive_strategy=getattr(self.config, 'adaptive_strategy', 'input_dependent'),
                                    sparsity_ratio=getattr(self.config, 'sparse_attention_sparsity_ratio', 0.25)
                                )
                            else:
                                sparse_attn = create_glm47_sparse_attention(self.config, layer_idx=idx)

                            # Replace the attention mechanism
                            layer.self_attn = sparse_attn
                        elif hasattr(layer, 'attn'):
                            orig_attn = layer.attn
                            # Create sparse attention implementation - use adaptive version if enabled
                            if getattr(self.config, 'use_adaptive_sparse_attention', False):
                                sparse_attn = create_adaptive_sparse_attention(
                                    self.config,
                                    layer_idx=idx,
                                    adaptive=True,
                                    adaptive_strategy=getattr(self.config, 'adaptive_strategy', 'input_dependent'),
                                    sparsity_ratio=getattr(self.config, 'sparse_attention_sparsity_ratio', 0.25)
                                )
                            else:
                                sparse_attn = create_glm47_sparse_attention(self.config, layer_idx=idx)

                            # Replace the attention mechanism
                            layer.attn = sparse_attn

                        replaced_count += 1
                        logger.debug(f"Replaced attention with sparse attention in layer {idx}")
                    except Exception as layer_e:
                        logger.warning(f"Could not replace attention in layer {idx}: {layer_e}")
                        continue

            logger.info(f"Sparse attention optimization applied: {replaced_count}/{total_layers} attention layers replaced")

        except ImportError as e:
            logger.warning(f"Sparse attention not available, skipping optimization: {e}")
        except Exception as e:
            logger.error(f"Error applying sparse attention optimization: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_flash_attention_optimization(self):
        """
        Apply FlashAttention 2.0 optimization to the model by replacing
        attention layers with FlashAttention 2.0 implementations.
        """
        try:
            # Replace attention layers in each transformer layer
            replaced_count = 0
            total_layers = 0

            # Access the model's transformer layers
            if hasattr(self._model, 'transformer') and hasattr(self._model.transformer, 'layers'):
                # GLM-style architecture
                layers = self._model.transformer.layers
            elif hasattr(self._model, 'model') and hasattr(self._model.model, 'layers'):
                # Another common architecture pattern
                layers = self._model.model.layers
            elif hasattr(self._model, 'transformer') and hasattr(self._model.transformer, 'h'):
                # GPT-style architecture
                layers = self._model.transformer.h
            else:
                # Default to empty list
                layers = []

            # Iterate through layers and replace attention mechanisms
            for idx, layer in enumerate(layers):
                total_layers += 1

                # Check if the layer has self-attention mechanism
                if hasattr(layer, 'self_attn') or hasattr(layer, 'attn'):
                    try:
                        # Get the attention submodule
                        if hasattr(layer, 'self_attn'):
                            orig_attn = layer.self_attn
                            # Create FlashAttention 2.0 implementation
                            flash_attn = create_glm47_flash_attention_2(self.config, layer_idx=idx)

                            # Replace the attention mechanism
                            layer.self_attn = flash_attn
                        elif hasattr(layer, 'attn'):
                            orig_attn = layer.attn
                            # Create FlashAttention 2.0 implementation
                            flash_attn = create_glm47_flash_attention_2(self.config, layer_idx=idx)

                            # Replace the attention mechanism
                            layer.attn = flash_attn

                        replaced_count += 1
                        logger.debug(f"Replaced attention in layer {idx}")
                    except Exception as layer_e:
                        logger.warning(f"Could not replace attention in layer {idx}: {layer_e}")
                        continue

            logger.info(f"FlashAttention 2.0 optimization applied: {replaced_count}/{total_layers} attention layers replaced")

        except ImportError as e:
            logger.warning(f"FlashAttention 2.0 not available, skipping optimization: {e}")
        except Exception as e:
            logger.error(f"Error applying FlashAttention 2.0 optimization: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_sliding_window_attention_optimization(self):
        """
        Apply sliding window attention optimization to the model by replacing
        attention layers with sliding window attention implementations.
        """
        try:
            # Replace attention layers in each transformer layer
            replaced_count = 0
            total_layers = 0

            # Access the model's transformer layers
            if hasattr(self._model, 'transformer') and hasattr(self._model.transformer, 'layers'):
                # GLM-style architecture
                layers = self._model.transformer.layers
            elif hasattr(self._model, 'model') and hasattr(self._model.model, 'layers'):
                # Another common architecture pattern
                layers = self._model.model.layers
            elif hasattr(self._model, 'transformer') and hasattr(self._model.transformer, 'h'):
                # GPT-style architecture
                layers = self._model.transformer.h
            else:
                # Default to empty list
                layers = []

            # Iterate through layers and replace attention mechanisms
            for idx, layer in enumerate(layers):
                total_layers += 1

                # Check if the layer has self-attention mechanism
                if hasattr(layer, 'self_attn') or hasattr(layer, 'attn'):
                    try:
                        # Get the attention submodule
                        if hasattr(layer, 'self_attn'):
                            orig_attn = layer.self_attn
                            # Create sliding window attention implementation
                            sliding_window_attn = create_glm47_sliding_window_attention(
                                self.config,
                                layer_idx=idx
                            )

                            # Replace the attention mechanism
                            layer.self_attn = sliding_window_attn
                        elif hasattr(layer, 'attn'):
                            orig_attn = layer.attn
                            # Create sliding window attention implementation
                            sliding_window_attn = create_glm47_sliding_window_attention(
                                self.config,
                                layer_idx=idx
                            )

                            # Replace the attention mechanism
                            layer.attn = sliding_window_attn

                        replaced_count += 1
                        logger.debug(f"Replaced attention with sliding window attention in layer {idx}")
                    except Exception as layer_e:
                        logger.warning(f"Could not replace attention in layer {idx}: {layer_e}")
                        continue

            logger.info(f"Sliding window attention optimization applied: {replaced_count}/{total_layers} attention layers replaced")

        except ImportError as e:
            logger.warning(f"Sliding window attention not available, skipping optimization: {e}")
        except Exception as e:
            logger.error(f"Error applying sliding window attention optimization: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_mqa_gqa_optimization(self):
        """
        Apply Multi-Query Attention (MQA) or Grouped-Query Attention (GQA) optimization
        to the model by replacing attention layers with MQA/GQA implementations.
        """
        try:
            # Replace attention layers in each transformer layer
            replaced_count = 0
            total_layers = 0

            # Access the model's transformer layers
            if hasattr(self._model, 'transformer') and hasattr(self._model.transformer, 'layers'):
                # GLM-style architecture
                layers = self._model.transformer.layers
            elif hasattr(self._model, 'model') and hasattr(self._model.model, 'layers'):
                # Another common architecture pattern
                layers = self._model.model.layers
            elif hasattr(self._model, 'transformer') and hasattr(self._model.transformer, 'h'):
                # GPT-style architecture
                layers = self._model.transformer.h
            else:
                # Default to empty list
                layers = []

            # Iterate through layers and replace attention mechanisms
            for idx, layer in enumerate(layers):
                total_layers += 1

                # Check if the layer has self-attention mechanism
                if hasattr(layer, 'self_attn') or hasattr(layer, 'attn'):
                    try:
                        # Get the attention submodule
                        if hasattr(layer, 'self_attn'):
                            orig_attn = layer.self_attn
                            # Create MQA/GQA attention implementation
                            mqa_gqa_attn = create_mqa_gqa_attention(self.config, layer_idx=idx)

                            # Replace the attention mechanism
                            layer.self_attn = mqa_gqa_attn
                        elif hasattr(layer, 'attn'):
                            orig_attn = layer.attn
                            # Create MQA/GQA attention implementation
                            mqa_gqa_attn = create_mqa_gqa_attention(self.config, layer_idx=idx)

                            # Replace the attention mechanism
                            layer.attn = mqa_gqa_attn

                        replaced_count += 1
                        logger.debug(f"Replaced attention with MQA/GQA in layer {idx}")
                    except Exception as layer_e:
                        logger.warning(f"Could not replace attention in layer {idx}: {layer_e}")
                        continue

            logger.info(f"MQA/GQA optimization applied: {replaced_count}/{total_layers} attention layers replaced")

        except ImportError as e:
            logger.warning(f"MQA/GQA attention not available, skipping optimization: {e}")
        except Exception as e:
            logger.error(f"Error applying MQA/GQA optimization: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_paged_attention_optimization(self):
        """
        Apply paged attention optimization to the model by replacing
        attention layers with paged attention implementations.
        """
        try:
            # Replace attention layers in each transformer layer
            replaced_count = 0
            total_layers = 0

            # Access the model's transformer layers
            if hasattr(self._model, 'transformer') and hasattr(self._model.transformer, 'layers'):
                # GLM-style architecture
                layers = self._model.transformer.layers
            elif hasattr(self._model, 'model') and hasattr(self._model.model, 'layers'):
                # Another common architecture pattern
                layers = self._model.model.layers
            elif hasattr(self._model, 'transformer') and hasattr(self._model.transformer, 'h'):
                # GPT-style architecture
                layers = self._model.transformer.h
            else:
                # Default to empty list
                layers = []

            # Iterate through layers and replace attention mechanisms
            for idx, layer in enumerate(layers):
                total_layers += 1

                # Check if the layer has self-attention mechanism
                if hasattr(layer, 'self_attn') or hasattr(layer, 'attn'):
                    try:
                        # Get the attention submodule
                        if hasattr(layer, 'self_attn'):
                            orig_attn = layer.self_attn
                            # Create paged attention implementation
                            paged_attn = create_glm47_paged_attention(
                                self.config,
                                layer_idx=idx,
                                page_size=self.config.paged_attention_page_size,
                                use_sliding_window=self.config.use_sliding_window_attention,
                                sliding_window_size=self.config.sliding_window_size
                            )

                            # Replace the attention mechanism
                            layer.self_attn = paged_attn
                        elif hasattr(layer, 'attn'):
                            orig_attn = layer.attn
                            # Create paged attention implementation
                            paged_attn = create_glm47_paged_attention(
                                self.config,
                                layer_idx=idx,
                                page_size=self.config.paged_attention_page_size,
                                use_sliding_window=self.config.use_sliding_window_attention,
                                sliding_window_size=self.config.sliding_window_size
                            )

                            # Replace the attention mechanism
                            layer.attn = paged_attn

                        replaced_count += 1
                        logger.debug(f"Replaced attention with paged attention in layer {idx}")
                    except Exception as layer_e:
                        logger.warning(f"Could not replace attention in layer {idx}: {layer_e}")
                        continue

            logger.info(f"Paged attention optimization applied: {replaced_count}/{total_layers} attention layers replaced")

        except ImportError as e:
            logger.warning(f"Paged attention not available, skipping optimization: {e}")
        except Exception as e:
            logger.error(f"Error applying paged attention optimization: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_optimized_rotary_embedding(self):
        """
        Apply optimized rotary embedding implementation to the model by replacing
        rotary embedding layers with optimized implementations.
        """
        try:
            # Replace rotary embedding layers in each transformer layer
            replaced_count = 0
            total_layers = 0

            # Access the model's transformer layers
            if hasattr(self._model, 'transformer') and hasattr(self._model.transformer, 'layers'):
                # GLM-style architecture
                layers = self._model.transformer.layers
            elif hasattr(self._model, 'model') and hasattr(self._model.model, 'layers'):
                # Another common architecture pattern
                layers = self._model.model.layers
            elif hasattr(self._model, 'transformer') and hasattr(self._model.transformer, 'h'):
                # GPT-style architecture
                layers = self._model.transformer.h
            else:
                # Default to empty list
                layers = []

            # Iterate through layers and replace rotary embedding mechanisms
            for idx, layer in enumerate(layers):
                total_layers += 1

                # Check if the layer has rotary embedding mechanism
                if hasattr(layer, 'self_attn') or hasattr(layer, 'attn'):
                    try:
                        # Get the attention submodule
                        if hasattr(layer, 'self_attn'):
                            attn_module = layer.self_attn
                        elif hasattr(layer, 'attn'):
                            attn_module = layer.attn

                        # Check if the attention module has rotary embedding
                        if hasattr(attn_module, 'rotary_emb'):
                            # Replace with optimized rotary embedding
                            original_emb = attn_module.rotary_emb
                            optimized_emb = GLM47RotaryEmbedding(
                                dim=original_emb.dim if hasattr(original_emb, 'dim') else self.config.hidden_size // self.config.num_attention_heads,
                                max_position_embeddings=self.config.max_position_embeddings,
                                base=self.config.rope_theta,
                                precision=torch.float16 if self._memory_config.get("torch_dtype") == torch.float16 else torch.float32
                            )

                            attn_module.rotary_emb = optimized_emb
                            replaced_count += 1
                            logger.debug(f"Replaced rotary embedding in layer {idx}")
                        else:
                            # If no rotary embedding exists, try to add one if needed
                            head_dim = self.config.hidden_size // self.config.num_attention_heads
                            optimized_emb = GLM47RotaryEmbedding(
                                dim=head_dim,
                                max_position_embeddings=self.config.max_position_embeddings,
                                base=self.config.rope_theta,
                                precision=torch.float16 if self._memory_config.get("torch_dtype") == torch.float16 else torch.float32
                            )

                            # Add rotary embedding to the attention module
                            attn_module.rotary_emb = optimized_emb
                            replaced_count += 1
                            logger.debug(f"Added optimized rotary embedding in layer {idx}")

                    except Exception as layer_e:
                        logger.warning(f"Could not replace/add rotary embedding in layer {idx}: {layer_e}")
                        continue

            logger.info(f"Optimized rotary embedding applied: {replaced_count}/{total_layers} attention layers updated")

        except ImportError as e:
            logger.warning(f"Optimized rotary embedding not available, skipping optimization: {e}")
        except Exception as e:
            logger.error(f"Error applying optimized rotary embedding: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_fused_layer_norm(self):
        """
        Apply fused layer normalization to the model by replacing
        standard LayerNorm modules with fused implementations for improved performance.
        """
        try:
            logger.info("Applying fused layer normalization optimization to GLM-4.7 model...")

            # Replace standard LayerNorm modules with fused implementations
            original_norm_count = 0
            for name, module in self._model.named_modules():
                if isinstance(module, torch.nn.LayerNorm):
                    original_norm_count += 1

            if original_norm_count == 0:
                logger.warning("No LayerNorm modules found in the model to fuse")
                return

            # Apply fused layer norm replacement
            self._model = replace_layer_norm_in_model(self._model, self.config)

            logger.info(f"Fused layer normalization applied: {original_norm_count} LayerNorm modules replaced")

        except Exception as e:
            logger.error(f"Error applying fused layer normalization: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_bias_removal_optimization(self):
        """
        Apply bias removal optimization to the model.
        """
        try:
            logger.info("Applying bias removal optimization to GLM-4.7 model...")
            self._model, bias_report = apply_bias_removal_to_model(
                self._model,
                model_type="glm47"
            )
            logger.info(f"Bias removal optimization completed: {bias_report}")
        except Exception as e:
            logger.error(f"Error applying bias removal optimization: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_kv_cache_compression(self):
        """
        Apply KV-cache compression optimization to the model.
        """
        try:
            logger.info("Applying KV-cache compression to GLM-4.7 model...")
            from .plugin.glm47_kv_cache import CompressedKVCacheConfig
            kv_config = CompressedKVCacheConfig(
                compression_method=self.config.kv_cache_compression_method,
                quantization_bits=self.config.kv_cache_quantization_bits,
                low_rank_dimension=self.config.kv_cache_low_rank_dimension,
                adaptive_precision_threshold=self.config.kv_cache_adaptive_precision_threshold,
                sparse_compression_ratio=self.config.kv_cache_sparse_compression_ratio,
                enable_dynamic_compression=self.config.kv_cache_enable_dynamic_compression
            )

            # Apply compression to the model
            self._model = apply_compressed_kv_cache_to_model(self._model, kv_config)

            logger.info("KV-cache compression optimization applied successfully")
        except ImportError as e:
            logger.warning(f"KV-cache compression not available, skipping optimization: {e}")
        except Exception as e:
            logger.error(f"Error applying KV-cache compression: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_prefix_caching(self):
        """
        Apply prefix caching optimization to the model.
        """
        try:
            logger.info("Applying prefix caching to GLM-4.7 model...")

            # Create prefix cache configuration
            from .plugin.glm47_prefix_cache import PrefixCacheConfig
            precision_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16
            }
            cache_precision = precision_map.get(self.config.prefix_cache_precision, torch.float16)

            prefix_config = PrefixCacheConfig(
                max_cache_size=self.config.prefix_cache_max_size,
                cache_precision=cache_precision,
                compression_enabled=self.config.prefix_cache_compression_enabled,
                eviction_policy=self.config.prefix_cache_eviction_policy,
                enable_prefetching=self.config.prefix_cache_enable_prefetching,
                prefetch_distance=self.config.prefix_cache_prefetch_distance,
                max_prefix_length=self.config.prefix_cache_max_prefix_length,
                min_prefix_length=self.config.prefix_cache_min_prefix_length,
                cache_warmup_threshold=self.config.prefix_cache_warmup_threshold,
            )

            # Apply prefix caching to the model
            self._model = apply_prefix_cache_to_model(self._model, prefix_config)

            logger.info("Prefix caching optimization applied successfully")
        except ImportError as e:
            logger.warning(f"Prefix caching not available, skipping optimization: {e}")
        except Exception as e:
            logger.error(f"Error applying prefix caching: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_cuda_kernels(self):
        """
        Apply CUDA kernels optimization to the model.
        """
        try:
            logger.info("Applying CUDA kernels optimization to GLM-4.7 model...")

            # Check if CUDA is available
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, skipping CUDA kernels optimization")
                return

            # Apply CUDA kernel optimizations to the model
            self._model = apply_glm47_optimizations_to_model(self._model, self.config)

            logger.info("CUDA kernels optimization applied successfully")
        except ImportError as e:
            logger.warning(f"CUDA kernels not available, skipping optimization: {e}")
        except Exception as e:
            logger.error(f"Error applying CUDA kernels optimization: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_linear_bias_optimization(self):
        """
        Apply linear bias optimization to the model.
        """
        try:
            logger.info("Applying linear bias optimization to GLM-4.7 model...")

            # Apply bias removal optimization to the model
            self._model, report = apply_bias_removal_to_model(
                self._model,
                model_type="glm47"
            )

            logger.info(f"Linear bias optimization applied: {report}")
        except ImportError as e:
            logger.warning(f"Linear bias optimization not available, skipping: {e}")
        except Exception as e:
            logger.error(f"Error applying linear bias optimization: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_tensor_decomposition(self, rank_ratio: float = 0.5,
                                   decomposition_method: str = "cp_decomposition"):
        """
        Apply tensor decomposition to the model to compress weights while maintaining accuracy.

        Args:
            rank_ratio: Ratio of ranks to keep (0.0 to 1.0, where 0.5 means 50% of original rank)
            decomposition_method: Method to use for decomposition
                                 ("cp_decomposition", "tucker_decomposition", "tensor_train", "matrix_svd")
        """
        try:
            logger.info(f"Applying tensor decomposition to GLM-4.7 model with rank_ratio {rank_ratio}, "
                       f"method {decomposition_method}...")

            # Apply tensor decomposition to the model
            decomposed_model, decomposition_metadata = decompose_model_weights(
                self._model,
                rank_ratio=rank_ratio,
                decomposition_method=decomposition_method,
                device=self._model.device
            )

            # In a real implementation, we would need to wrap the model to handle decomposed weights
            # For now, we'll just log the metadata
            total_params_before = sum(p.numel() for p in self._model.parameters())
            total_params_after = total_params_before  # Placeholder - actual implementation would vary

            logger.info(f"Tensor decomposition applied: rank_ratio={rank_ratio}, "
                       f"method={decomposition_method}")
            logger.info(f"Note: Actual weight replacement requires model-specific wrapper implementation")

            return decomposition_metadata

        except ImportError as e:
            logger.warning(f"Tensor decomposition not available, skipping: {e}")
        except Exception as e:
            logger.error(f"Error applying tensor decomposition: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_structured_pruning(self, pruning_ratio: float = 0.2,
                                  method: PruningMethod = PruningMethod.LAYER_REMOVAL,
                                  block_size: int = 1):
        """
        Apply structured pruning to the model to reduce complexity while preserving accuracy.

        Args:
            pruning_ratio: Ratio of blocks/layers to remove (0.0 to 1.0)
            method: Pruning method to use
            block_size: Size of blocks to remove (for block pruning)
        """
        try:
            logger.info(f"Applying structured pruning to GLM-4.7 model with ratio {pruning_ratio}, "
                       f"method {method.value}, block_size {block_size}...")

            # Apply structured pruning to the model
            pruning_result = apply_structured_pruning(
                self._model,
                pruning_ratio=pruning_ratio,
                method=method,
                block_size=block_size
            )

            self._model = pruning_result.pruned_model

            logger.info(f"Structured pruning completed: {pruning_result.compression_ratio:.2%} "
                       f"compression achieved, accuracy preserved: {pruning_result.accuracy_preserved}")
            logger.info(f"Removed {len(pruning_result.removed_layers)} layers/components")

            return pruning_result

        except ImportError as e:
            logger.warning(f"Structured pruning not available, skipping: {e}")
        except Exception as e:
            logger.error(f"Error applying structured pruning: {e}")
            # Continue without optimization if it fails
            pass

    def _apply_snn_conversion(self):
        """
        Apply Spiking Neural Network conversion to the model for energy efficiency.
        """
        try:
            logger.info("Applying SNN conversion to GLM-4.7 model for energy efficiency...")

            # Create SNN configuration
            snn_config = {
                'neuron_type': getattr(self.config, 'snn_neuron_type', 'LIF'),
                'threshold': getattr(self.config, 'snn_threshold', 1.0),
                'decay': getattr(self.config, 'snn_decay', 0.9),
                'dropout_rate': getattr(self.config, 'snn_dropout_rate', 0.0),
                'temporal_encoding': getattr(self.config, 'snn_temporal_encoding', False)
            }

            # Convert the model to SNN
            self._model = convert_dense_to_snn(self._model, snn_config)

            # Apply additional SNN optimizations
            if getattr(self.config, 'enable_snn_optimizations', False):
                optimization_config = {
                    'pruning_ratio': getattr(self.config, 'snn_pruning_ratio', 0.2),
                    'quantization_bits': getattr(self.config, 'snn_quantization_bits', 8),
                    'temporal_sparsity': getattr(self.config, 'snn_temporal_sparsity', True),
                    'neural_efficiency': getattr(self.config, 'snn_neural_efficiency', True)
                }

                self._model = apply_snn_optimizations(self._model, optimization_config)

            logger.info("SNN conversion applied successfully")

        except ImportError as e:
            logger.warning(f"SNN conversion not available, skipping: {e}")
        except Exception as e:
            logger.error(f"Error applying SNN conversion: {e}")
            # Continue without optimization if it fails
            pass

    def forward(self, *args, **kwargs):
        """
        Forward pass through the model with validation for unimodal operation.
        """
        # Validate inputs are appropriate for unimodal (text-only) operation
        if args:
            if not self._validate_unimodal_operation(args[0]):
                raise ValueError("Invalid multimodal input detected for unimodal GLM-4.7 model")
        
        if 'input_ids' in kwargs:
            if not self._validate_unimodal_operation(kwargs['input_ids']):
                raise ValueError("Invalid multimodal input detected for unimodal GLM-4.7 model")
        
        start_time = time.time()

        # Apply ML-based optimization if enabled
        if getattr(self.config, 'use_ml_optimizations', False):
            # Extract input tensors from args/kwargs to analyze complexity
            input_tensor = None
            if args:
                input_tensor = args[0] if torch.is_tensor(args[0]) else None
            elif 'input_ids' in kwargs:
                input_tensor = kwargs['input_ids']
            elif 'inputs_embeds' in kwargs:
                input_tensor = kwargs['inputs_embeds']

            if input_tensor is not None:
                # Apply ML-based optimization based on input
                ml_system = get_ml_optimization_system()
                optimized_model = ml_system.optimize_model_for_input(
                    model=self._model,
                    input_data=input_tensor,
                    model_type=ModelType.GLM_4_7
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

        result = self._model(*args, **kwargs)

        # Record performance metrics
        end_time = time.time()
        latency = end_time - start_time

        # Calculate throughput if possible
        input_length = 0
        if args and torch.is_tensor(args[0]):
            input_length = args[0].shape[-1] if len(args[0].shape) > 1 else args[0].numel()
        elif 'input_ids' in kwargs and torch.is_tensor(kwargs['input_ids']):
            input_length = kwargs['input_ids'].shape[-1]

        throughput = input_length / latency if latency > 0 and input_length > 0 else 0

        return result

    def generate(self, *args, **kwargs):
        """
        Generate text using the model with validation for unimodal operation.
        """
        # Validate inputs are appropriate for unimodal (text-only) operation
        if args:
            if not self._validate_unimodal_operation(args[0]):
                raise ValueError("Invalid multimodal input detected for unimodal GLM-4.7 model")
        
        if 'input_ids' in kwargs:
            if not self._validate_unimodal_operation(kwargs['input_ids']):
                raise ValueError("Invalid multimodal input detected for unimodal GLM-4.7 model")

        start_time = time.time()

        # Apply ML-based optimization if enabled
        if getattr(self.config, 'use_ml_optimizations', False):
            # Extract input tensors from args/kwargs to analyze complexity
            input_tensor = None
            if args:
                input_tensor = args[0] if torch.is_tensor(args[0]) else None
            elif 'input_ids' in kwargs:
                input_tensor = kwargs['input_ids']
            elif 'inputs_embeds' in kwargs:
                input_tensor = kwargs['inputs_embeds']

            if input_tensor is not None:
                # Apply ML-based optimization based on input
                ml_system = get_ml_optimization_system()
                optimized_model = ml_system.optimize_model_for_input(
                    model=self._model,
                    input_data=input_tensor,
                    model_type=ModelType.GLM_4_7
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


__all__ = [
    "GLM47SafeModel"
]