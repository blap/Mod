"""
Qwen3-VL-2B Optimization Logic
"""
import logging
import torch
from ....common.optimization.unified_ml_optimization import (
    ModelType,
    get_ml_optimization_system,
)
from ....common.optimization.structured_pruning import PruningMethod

logger = logging.getLogger(__name__)

class Qwen3VL2BOptimizer:
    def __init__(self, model_instance):
        self.model_instance = model_instance
        self.config = model_instance.config
        self._model = model_instance._model

    def apply_configured_optimizations(self):
        """
        Apply optimizations based on the configuration settings.
        """
        # Check if ML-based optimization is enabled
        if getattr(self.config, "use_ml_optimizations", False):
            # Use ML-based optimization system
            ml_system = get_ml_optimization_system()
            self.model_instance._model = ml_system.optimize_model_for_input(
                model=self._model,
                input_data=None,  # Initially no input, will be optimized per request
                model_type=ModelType.QWEN3_VL_2B,
            )
            # Update local ref
            self._model = self.model_instance._model
        elif getattr(self.config, "use_modular_optimizations", False):
            # Apply optimizations using the new modular system if enabled
            from ..specific_optimizations.optimization_integration import apply_qwen_optimizations

            profile_name = getattr(self.config, "optimization_profile", "balanced")
            self.model_instance._model = apply_qwen_optimizations(self._model, profile_name)
            self._model = self.model_instance._model
        else:
            # Apply traditional optimizations for backward compatibility
            self._apply_traditional_optimizations()

        # Apply specific optimizations
        self._apply_vision_encoder_optimizations()
        self._apply_cross_modal_fusion_optimizations()
        self._apply_cross_modal_alignment_optimizations()
        self._apply_projection_layer_optimizations()
        self._apply_multimodal_attention_optimization()
        self._apply_attention_optimizations()

        # Update model instance with final optimized model
        self.model_instance._model = self._model

    def _apply_attention_optimizations(self):
        try:
            logger.info("Applying attention optimizations to Qwen3-VL-2B model...")

            if self.config.use_flash_attention_2:
                logger.info("Applying FlashAttention 2.0 optimization...")
                self._apply_flash_attention_optimization()
            elif self.config.use_sparse_attention:
                logger.info("Applying sparse attention optimization...")
                self._apply_sparse_attention_optimization()
            elif self.config.use_sliding_window_attention:
                logger.info("Applying sliding window attention optimization...")
                self._apply_sliding_window_attention_optimization()
            elif (
                self.config.use_multi_query_attention
                or self.config.use_grouped_query_attention
            ):
                logger.info("Applying Multi-Query/Grouped-Query attention optimization...")
                self._apply_mqa_gqa_optimization()
            elif self.config.use_paged_attention:
                logger.info("Applying paged attention optimization...")
                self._apply_paged_attention_optimization()
            else:
                logger.info("Attention optimization disabled")

            logger.info("Applying optimized rotary embeddings...")
            self._apply_optimized_rotary_embedding()

        except Exception as e:
            logger.error(f"Error applying attention optimizations: {e}")
            pass

    def _apply_flash_attention_optimization(self):
        try:
            from ...common.attention.flash_attention_2 import apply_flash_attention_2_to_model
            self._model = apply_flash_attention_2_to_model(self._model, self.config)
        except ImportError:
            logger.warning("FlashAttention 2.0 module not available")
        except Exception as e:
            logger.error(f"Error applying FlashAttention 2.0: {e}")

    def _apply_sparse_attention_optimization(self):
        try:
            from ...common.attention.sparse_attention import apply_sparse_attention_to_model
            self._model = apply_sparse_attention_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Sparse attention module not available")
        except Exception as e:
            logger.error(f"Error applying sparse attention: {e}")

    def _apply_sliding_window_attention_optimization(self):
        try:
            from ...common.attention.sliding_window_attention import apply_sliding_window_attention_to_model
            self._model = apply_sliding_window_attention_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Sliding window attention module not available")
        except Exception as e:
            logger.error(f"Error applying sliding window attention: {e}")

    def _apply_mqa_gqa_optimization(self):
        try:
            from ...common.attention.multi_query_attention import apply_mqa_gqa_to_model
            self._model = apply_mqa_gqa_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Multi-Query/Grouped-Query attention module not available")
        except Exception as e:
            logger.error(f"Error applying MQA/GQA: {e}")

    def _apply_paged_attention_optimization(self):
        try:
            from ...common.attention.paged_attention import apply_paged_attention_to_model
            self._model = apply_paged_attention_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Paged attention module not available")
        except Exception as e:
            logger.error(f"Error applying paged attention: {e}")

    def _apply_optimized_rotary_embedding(self):
        try:
            from ...common.layers.rotary_embeddings import apply_rotary_embeddings_to_model
            self._model = apply_rotary_embeddings_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Rotary embeddings module not available")
        except Exception as e:
            logger.error(f"Error applying optimized rotary embedding: {e}")

    def _apply_vision_encoder_optimizations(self):
        try:
            from ..vision_transformer_kernels import apply_vision_cuda_optimizations_to_model
            self._model = apply_vision_cuda_optimizations_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Vision transformer kernels module not available")
        except Exception as e:
            logger.error(f"Error applying vision encoder optimizations: {e}")

    def _apply_cross_modal_fusion_optimizations(self):
        try:
            from ..cross_modal_fusion_kernels import apply_cross_modal_fusion_to_qwen3_vl_model
            self._model = apply_cross_modal_fusion_to_qwen3_vl_model(self._model, self.config)
        except ImportError:
            logger.warning("Cross-modal fusion kernels module not available")
        except Exception as e:
            logger.error(f"Error applying cross-modal fusion optimizations: {e}")

    def _apply_cross_modal_alignment_optimizations(self):
        try:
            from ..cross_modal_alignment_optimization import apply_cross_modal_alignment_to_model
            self._model = apply_cross_modal_alignment_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Cross-modal alignment optimization module not available")
        except Exception as e:
            logger.error(f"Error applying cross-modal alignment optimizations: {e}")

    def _apply_projection_layer_optimizations(self):
        try:
            from ..multimodal_projector import apply_qwen3_vl_projection_optimizations
            self._model = apply_qwen3_vl_projection_optimizations(self._model, self.config)
        except ImportError:
            logger.warning("Multimodal projector module not available")
        except Exception as e:
            logger.error(f"Error applying projection layer optimizations: {e}")

    def _apply_multimodal_attention_optimization(self):
        try:
            from ...common.attention.multimodal_attention import apply_multimodal_attention_optimizations_to_model
            self._model = apply_multimodal_attention_optimizations_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Multimodal attention optimization module not available")
        except Exception as e:
            logger.error(f"Error applying multimodal attention optimization: {e}")

    def _apply_traditional_optimizations(self):
        if self.config.use_tensor_parallelism:
            self._apply_tensor_parallelism()
        if self.config.use_flash_attention_2:
            self._apply_flash_attention_2_optimization()
        if self.config.use_bias_removal_optimization:
            self._apply_bias_removal_optimization()
        if self.config.use_fused_layer_norm:
            self._apply_fused_layer_norm()
        if self.config.use_kv_cache_compression:
            self._apply_kv_cache_compression()
        if self.config.use_prefix_caching:
            self._apply_prefix_caching()
        if self.config.use_cuda_kernels:
            self._apply_cuda_kernels()
        if self.config.linear_bias_optimization_enabled:
            self._apply_linear_bias_optimization()

        if getattr(self.config, "use_tensor_decomposition", False):
            from ...common.optimization.tensor_decomposition import apply_tensor_decomposition_to_model
            decomposition_method = getattr(
                self.config, "tensor_decomposition_method", "cp_decomposition"
            )
            rank_ratio = getattr(self.config, "tensor_decomposition_rank_ratio", 0.5)
            self._model = apply_tensor_decomposition_to_model(
                self._model,
                rank_ratio=rank_ratio,
                decomposition_method=decomposition_method,
            )

        if self.config.use_structured_pruning:
            from ...common.optimization.structured_pruning import (
                PruningMethod,
                apply_structured_pruning_to_model,
            )
            method_map = {
                "layer_removal": PruningMethod.LAYER_REMOVAL,
                "block_removal": PruningMethod.BLOCK_REMOVAL,
                "head_removal": PruningMethod.HEAD_REMOVAL,
                "mlp_removal": PruningMethod.MLP_REMOVAL,
                "adaptive_pruning": PruningMethod.ADAPTIVE_PRUNING,
            }
            pruning_method = method_map.get(
                self.config.pruning_method, PruningMethod.LAYER_REMOVAL
            )
            self._model = apply_structured_pruning_to_model(
                self._model,
                pruning_ratio=self.config.pruning_ratio,
                method=pruning_method,
                block_size=self.config.pruning_block_size,
            )

        if getattr(self.config, "use_multimodal_attention", False):
            from ...common.attention.multimodal_attention import apply_multimodal_attention_to_model
            self._model = apply_multimodal_attention_to_model(self._model, self.config)

        if getattr(self.config, "use_snn_conversion", False):
            from ...common.layers.snn import apply_snn_conversion_to_model
            self._model = apply_snn_conversion_to_model(self._model, self.config)

    def _apply_tensor_parallelism(self):
        try:
            from ..tensor_parallel import apply_tensor_parallelism_to_model
            tensor_parallel_size = self.config.tensor_parallel_size
            if tensor_parallel_size > 1:
                logger.info(f"Applying tensor parallelism with size {tensor_parallel_size}")
                self._model = apply_tensor_parallelism_to_model(
                    self._model,
                    tensor_parallel_size=tensor_parallel_size,
                    local_rank=self.config.tensor_parallel_local_rank,
                    world_size=self.config.tensor_parallel_world_size,
                    init_method=self.config.tensor_parallel_init_method,
                )
        except ImportError:
            logger.warning("Tensor parallelism module not available")
        except Exception as e:
            logger.error(f"Error applying tensor parallelism: {e}")

    def _apply_flash_attention_2_optimization(self):
        try:
            from ...common.attention.flash_attention_2 import apply_flash_attention_2_to_model
            self._model = apply_flash_attention_2_to_model(self._model, self.config)
        except ImportError:
            logger.warning("FlashAttention 2.0 module not available")
        except Exception as e:
            logger.error(f"Error applying FlashAttention 2.0: {e}")

    def _apply_bias_removal_optimization(self):
        try:
            from ...common.optimization.optimization_integration import apply_bias_removal_to_model
            self._model = apply_bias_removal_to_model(self._model, model_type="qwen3_vl")
        except ImportError:
            logger.warning("Bias removal module not available")
        except Exception as e:
            logger.error(f"Error applying bias removal optimization: {e}")

    def _apply_fused_layer_norm(self):
        try:
            from ..fused_layers import apply_fused_layer_norm_to_model
            self._model = apply_fused_layer_norm_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Fused layer norm module not available")
        except Exception as e:
            logger.error(f"Error applying fused layer norm: {e}")

    def _apply_kv_cache_compression(self):
        try:
            from ..kv_cache import apply_kv_cache_compression_to_model
            self._model = apply_kv_cache_compression_to_model(self._model, self.config)
        except ImportError:
            logger.warning("KV-cache compression module not available")
        except Exception as e:
            logger.error(f"Error applying KV-cache compression: {e}")

    def _apply_prefix_caching(self):
        try:
            from ..prefix_caching import apply_prefix_caching_to_model
            self._model = apply_prefix_caching_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Prefix caching module not available")
        except Exception as e:
            logger.error(f"Error applying prefix caching: {e}")

    def _apply_cuda_kernels(self):
        try:
            from ...common.layers.qwen3_vl_cuda_kernels import apply_cuda_optimizations_to_model as apply_qwen3_vl_cuda_optimizations_to_model
            self._model = apply_qwen3_vl_cuda_optimizations_to_model(self._model, self.config)
        except ImportError:
            logger.warning("CUDA kernels module not available")
        except Exception as e:
            logger.error(f"Error applying CUDA kernels optimization: {e}")

    def _apply_linear_bias_optimization(self):
        try:
            from ...common.optimization.optimization_integration import apply_linear_bias_optimization_to_model
            self._model = apply_linear_bias_optimization_to_model(self._model, self.config)
        except ImportError:
            logger.warning("Linear bias optimization module not available")
        except Exception as e:
            logger.error(f"Error applying linear bias optimization: {e}")
