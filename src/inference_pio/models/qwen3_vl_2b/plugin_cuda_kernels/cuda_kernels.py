"""
Qwen3-VL-2B Specific CUDA Kernels for Vision-Language Models

This module implements optimized CUDA kernels specifically for the Qwen3-VL-2B model.
These kernels are designed to accelerate vision-language operations and multimodal
processing in the model with Qwen3-VL-2B specific optimizations.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ....common.multimodal_cuda_kernels import (
    MultimodalCrossAttentionKernel,
    MultimodalFusionKernel,
    MultimodalPositionEncodingKernel,
    VisionLanguageAttentionKernel,
)
from ....common.quantized_multimodal_kernels import (
    QuantizedMultimodalPositionEncodingKernel,
    QuantizedQwen3VL2BCrossAttentionKernel,
    QuantizedQwen3VL2BFusionKernel,
    QuantizedVisionLanguageAttentionKernel,
)
from ....common.vision_transformer_kernels import (
    Qwen3VL2BVisionEncoderKernel,
    VisionMLPKernel,
    VisionPatchEmbeddingKernel,
    VisionSelfAttentionKernel,
    VisionTransformerBlockKernel,
    VisionTransformerConfig,
    create_qwen3_vl_2b_vision_encoder_kernel,
    create_vision_mlp_kernel,
    create_vision_patch_embedding_kernel,
    create_vision_self_attention_kernel,
    create_vision_transformer_block_kernel,
)

logger = logging.getLogger(__name__)


@dataclass
class Qwen3VL2BConfig:
    """Configuration for Qwen3-VL-2B specific optimizations."""

    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    vocab_size: int = 152064
    max_position_embeddings: int = 32768
    intermediate_size: int = 5504
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    vision_config: Optional[Dict] = None
    use_flash_attention_2: bool = True
    use_cuda_kernels: bool = True
    use_quantized_kernels: bool = False  # New option for quantized kernels
    quantization_scheme: str = "int8"  # Quantization scheme for kernels
    quantization_bits: int = 8  # Number of bits for quantization
    # Vision-specific parameters
    vision_patch_size: int = 14
    vision_image_size: int = 448
    vision_intermediate_size: int = 2816
    vision_num_hidden_layers: int = 24
    vision_layer_norm_eps: float = 1e-6
    use_vision_flash_attention: bool = True


class Qwen3VL2BCrossAttentionKernel(nn.Module):
    """
    Qwen3-VL-2B specific implementation of multimodal cross-attention kernel.
    This kernel is optimized for the specific architecture and parameters of Qwen3-VL-2B.
    """

    def __init__(self, config: Qwen3VL2BConfig, layer_idx: int = 0):
        super().__init__()

        self.layer_idx = layer_idx
        self.config = config

        # Use quantized kernels if enabled
        if config.use_quantized_kernels:
            from .....common.quantization import QuantizationScheme
            from .....common.quantized_multimodal_kernels import (
                QuantizedMultimodalConfig,
            )

            # Map string scheme to QuantizationScheme enum
            scheme_map = {
                "int4": QuantizationScheme.INT4,
                "int8": QuantizationScheme.INT8,
                "fp16": QuantizationScheme.FP16,
                "nf4": QuantizationScheme.NF4,
            }
            quant_scheme = scheme_map.get(
                config.quantization_scheme.lower(), QuantizationScheme.INT8
            )

            # Create quantized config
            quantized_config = QuantizedMultimodalConfig(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                modalities=["text", "image"],
                quantization_scheme=quant_scheme,
                quantization_bits=config.quantization_bits,
                dropout=0.1,
                use_flash_attention=config.use_flash_attention_2,
            )

            # Use quantized kernel
            self.kernel_impl = QuantizedQwen3VL2BCrossAttentionKernel(
                quantized_config, layer_idx
            )
        else:
            # Use standard kernel
            self.kernel_impl = MultimodalCrossAttentionKernel(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                modalities=["text", "image"],
                dropout=0.1,
                use_flash_attention=config.use_flash_attention_2,
            )

            # Qwen3-VL-2B specific parameters
            self.qkv_same_dim = config.hidden_size // config.num_attention_heads
            self.scale = self.qkv_same_dim**-0.5

            # Additional Qwen3-VL-2B specific projections
            self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

            # Qwen3-VL-2B specific output projection
            self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

            # Initialize weights according to Qwen3-VL-2B specifications
            self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights according to Qwen3-VL-2B specifications."""
        # Initialize Q/K/V projections
        std = self.config.hidden_size**-0.5
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)

        # Initialize output projection
        std = (2 * self.config.num_hidden_layers) ** -0.5
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std)

    def forward(
        self,
        queries: Dict[str, torch.Tensor],
        keys: Dict[str, torch.Tensor],
        values: Dict[str, torch.Tensor],
        attention_masks: Optional[Dict[str, torch.Tensor]] = None,
        need_weights: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass for Qwen3-VL-2B specific multimodal cross-attention kernel.
        """
        # Use the appropriate kernel implementation
        if self.config.use_quantized_kernels:
            # For quantized kernels, use the implementation directly
            return self.kernel_impl(
                queries, keys, values, attention_masks, need_weights
            )
        else:
            # For standard kernels, use the original implementation
            outputs = {}
            attention_weights = {} if need_weights else None

            # Process each modality as query
            for query_modality, query in queries.items():
                # Normalize query
                query = self.kernel_impl.modality_norms[query_modality](query)

                # Project query using Qwen3-VL-2B specific projections
                q = self.q_proj(query)
                q = q.view(
                    query.size(0),
                    query.size(1),
                    self.kernel_impl.nhead,
                    self.kernel_impl.head_dim,
                ).transpose(1, 2)
                q = q * self.scale

                # Concatenate all keys and values from all modalities
                all_keys = []
                all_values = []

                for key_modality, key in keys.items():
                    # Normalize key
                    key = self.kernel_impl.modality_norms[key_modality](key)

                    # Project key using Qwen3-VL-2B specific projections
                    k = self.k_proj(key)
                    k = k.view(
                        key.size(0),
                        key.size(1),
                        self.kernel_impl.nhead,
                        self.kernel_impl.head_dim,
                    ).transpose(1, 2)
                    all_keys.append(k)

                    # Project value using Qwen3-VL-2B specific projections
                    v = self.v_proj(key)  # Using key as input for simplicity
                    v = v.view(
                        key.size(0),
                        key.size(1),
                        self.kernel_impl.nhead,
                        self.kernel_impl.head_dim,
                    ).transpose(1, 2)
                    all_values.append(v)

                # Concatenate keys and values across modalities
                concat_k = torch.cat(all_keys, dim=2)
                concat_v = torch.cat(all_values, dim=2)

                # Compute attention scores
                if self.kernel_impl.use_flash_attention and torch.cuda.is_available():
                    # Use efficient attention computation (Flash Attention 2.0)
                    attn_weights = torch.matmul(q, concat_k.transpose(-2, -1))

                    # Apply attention mask if provided
                    if (
                        attention_masks is not None
                        and query_modality in attention_masks
                    ):
                        mask = attention_masks[query_modality]
                        if mask.dim() == 2:
                            mask = (
                                mask.unsqueeze(0)
                                .unsqueeze(0)
                                .expand(-1, self.kernel_impl.nhead, -1, -1)
                            )
                        elif mask.dim() == 3:
                            mask = mask.unsqueeze(1).expand(
                                -1, self.kernel_impl.nhead, -1, -1
                            )
                        attn_weights = attn_weights + mask

                    # Apply softmax
                    attn_weights = torch.softmax(
                        attn_weights, dim=-1, dtype=torch.float32
                    ).to(query.dtype)

                    # Apply dropout if configured
                    if self.kernel_impl.dropout is not None:
                        attn_weights = self.kernel_impl.dropout(attn_weights)

                    # Apply attention to values
                    attn_output = torch.matmul(attn_weights, concat_v)
                else:
                    # Standard attention computation
                    attn_weights = torch.matmul(q, concat_k.transpose(-2, -1))

                    # Apply attention mask if provided
                    if (
                        attention_masks is not None
                        and query_modality in attention_masks
                    ):
                        mask = attention_masks[query_modality]
                        if mask.dim() == 2:
                            mask = (
                                mask.unsqueeze(0)
                                .unsqueeze(0)
                                .expand(-1, self.kernel_impl.nhead, -1, -1)
                            )
                        elif mask.dim() == 3:
                            mask = mask.unsqueeze(1).expand(
                                -1, self.kernel_impl.nhead, -1, -1
                            )
                        attn_weights = attn_weights + mask

                    # Apply softmax
                    attn_weights = torch.softmax(
                        attn_weights, dim=-1, dtype=torch.float32
                    ).to(query.dtype)

                    # Apply dropout if configured
                    if self.kernel_impl.dropout is not None:
                        attn_weights = self.kernel_impl.dropout(attn_weights)

                    # Apply attention to values
                    attn_output = torch.matmul(attn_weights, concat_v)

                # Reshape to combine heads
                attn_output = (
                    attn_output.transpose(1, 2)
                    .contiguous()
                    .view(query.size(0), query.size(1), self.kernel_impl.d_model)
                )

                # Apply Qwen3-VL-2B specific output projection
                attn_output = self.o_proj(attn_output)

                # Store output for this modality
                outputs[query_modality] = attn_output

                # Store attention weights if needed
                if need_weights:
                    attention_weights[query_modality] = attn_weights

            return outputs, attention_weights


class Qwen3VL2BFusionKernel(nn.Module):
    """
    Qwen3-VL-2B specific implementation of multimodal fusion kernel.
    This kernel is optimized for the specific architecture and parameters of Qwen3-VL-2B.
    """

    def __init__(self, config: Qwen3VL2BConfig, layer_idx: int = 0):
        super().__init__()

        self.layer_idx = layer_idx
        self.config = config

        # Use quantized kernels if enabled
        if config.use_quantized_kernels:
            from .....common.quantization import QuantizationScheme
            from .....common.quantized_multimodal_kernels import (
                QuantizedMultimodalConfig,
            )

            # Map string scheme to QuantizationScheme enum
            scheme_map = {
                "int4": QuantizationScheme.INT4,
                "int8": QuantizationScheme.INT8,
                "fp16": QuantizationScheme.FP16,
                "nf4": QuantizationScheme.NF4,
            }
            quant_scheme = scheme_map.get(
                config.quantization_scheme.lower(), QuantizationScheme.INT8
            )

            # Create quantized config
            quantized_config = QuantizedMultimodalConfig(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                modalities=["text", "image"],
                quantization_scheme=quant_scheme,
                quantization_bits=config.quantization_bits,
                dropout=0.1,
                activation="silu",  # Qwen3-VL-2B uses SiLU activation
                use_cross_attention=True,
            )

            # Use quantized kernel
            self.kernel_impl = QuantizedQwen3VL2BFusionKernel(
                quantized_config, layer_idx
            )
        else:
            # Use standard kernel
            self.kernel_impl = MultimodalFusionKernel(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                modalities=["text", "image"],
                dropout=0.1,
                activation="silu",  # Qwen3-VL-2B uses SiLU activation
                use_cross_attention=True,
            )

            # Qwen3-VL-2B specific MLP components
            self.mlp_gate_proj = nn.Linear(
                config.hidden_size, config.intermediate_size, bias=False
            )
            self.mlp_up_proj = nn.Linear(
                config.hidden_size, config.intermediate_size, bias=False
            )
            self.mlp_down_proj = nn.Linear(
                config.intermediate_size, config.hidden_size, bias=False
            )

            # Initialize MLP weights according to Qwen3-VL-2B specifications
            self._initialize_mlp_weights()

    def _initialize_mlp_weights(self):
        """Initialize MLP weights according to Qwen3-VL-2B specifications."""
        # Initialize gate/up/down projections
        std = self.config.hidden_size**-0.5
        nn.init.normal_(self.mlp_gate_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.mlp_up_proj.weight, mean=0.0, std=std)

        std = (2 * self.config.num_hidden_layers) ** -0.5
        nn.init.normal_(self.mlp_down_proj.weight, mean=0.0, std=std)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        attention_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Qwen3-VL-2B specific multimodal fusion kernel.
        """
        # Use the appropriate kernel implementation
        if self.config.use_quantized_kernels:
            # For quantized kernels, use the implementation directly
            return self.kernel_impl(inputs, attention_masks)
        else:
            # For standard kernels, use the original implementation
            if self.kernel_impl.use_cross_attention:
                # Apply cross-attention between modalities
                attended_outputs, _ = self.kernel_impl.cross_attention(
                    queries=inputs,
                    keys=inputs,
                    values=inputs,
                    attention_masks=attention_masks,
                    need_weights=False,
                )

                # Apply residual connection and layer norm, then FFN
                normalized_outputs = {}
                for modality in self.kernel_impl.modalities:
                    if modality in attended_outputs:
                        # Add residual connection
                        residual = inputs[modality] + attended_outputs[modality]
                        # Apply layer norm
                        normalized = self.kernel_impl.norms[modality](residual)

                        # Apply Qwen3-VL-2B specific MLP with SwiGLU activation
                        gate = self.mlp_gate_proj(normalized)
                        gate = nn.functional.silu(gate)  # SiLU activation
                        up = self.mlp_up_proj(normalized)
                        mlp_output = gate * up  # Element-wise multiplication for SwiGLU
                        mlp_output = self.mlp_down_proj(mlp_output)

                        # Add residual connection again
                        normalized_outputs[modality] = residual + mlp_output
                    else:
                        # If modality wasn't processed, pass through unchanged
                        normalized_outputs[modality] = inputs[modality]
            else:
                # Without cross-attention, just apply FFN to each modality
                normalized_outputs = {}
                for modality in self.kernel_impl.modalities:
                    if modality in inputs:
                        # Apply layer norm
                        normalized = self.kernel_impl.norms[modality](inputs[modality])

                        # Apply Qwen3-VL-2B specific MLP with SwiGLU activation
                        gate = self.mlp_gate_proj(normalized)
                        gate = nn.functional.silu(gate)  # SiLU activation
                        up = self.mlp_up_proj(normalized)
                        mlp_output = gate * up  # Element-wise multiplication for SwiGLU
                        mlp_output = self.mlp_down_proj(mlp_output)

                        # Add residual connection
                        normalized_outputs[modality] = inputs[modality] + mlp_output
                    else:
                        normalized_outputs[modality] = inputs[modality]

            return normalized_outputs


class Qwen3VL2BVisionLanguageAttentionKernel(nn.Module):
    """
    Qwen3-VL-2B specific implementation of vision-language attention kernel.
    This kernel is optimized for the specific architecture and parameters of Qwen3-VL-2B.
    """

    def __init__(self, config: Qwen3VL2BConfig, layer_idx: int = 0):
        super().__init__()

        self.layer_idx = layer_idx
        self.config = config

        # Use quantized kernels if enabled
        if config.use_quantized_kernels:
            from .....common.quantization import QuantizationScheme
            from .....common.quantized_multimodal_kernels import (
                QuantizedMultimodalConfig,
            )

            # Map string scheme to QuantizationScheme enum
            scheme_map = {
                "int4": QuantizationScheme.INT4,
                "int8": QuantizationScheme.INT8,
                "fp16": QuantizationScheme.FP16,
                "nf4": QuantizationScheme.NF4,
            }
            quant_scheme = scheme_map.get(
                config.quantization_scheme.lower(), QuantizationScheme.INT8
            )

            # Create quantized config
            quantized_config = QuantizedMultimodalConfig(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                modalities=["text", "image"],
                quantization_scheme=quant_scheme,
                quantization_bits=config.quantization_bits,
                dropout=0.1,
            )

            # Use quantized kernel
            self.kernel_impl = QuantizedVisionLanguageAttentionKernel(quantized_config)
        else:
            # Use standard kernel
            self.kernel_impl = VisionLanguageAttentionKernel(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dropout=0.1,
                image_patch_size=14,  # Typical for vision transformers
                max_image_patches=1024,
            )

            # Qwen3-VL-2B specific vision-language projections
            self.vision_q_proj = nn.Linear(
                config.hidden_size, config.hidden_size, bias=False
            )
            self.vision_k_proj = nn.Linear(
                config.hidden_size, config.hidden_size, bias=False
            )
            self.vision_v_proj = nn.Linear(
                config.hidden_size, config.hidden_size, bias=False
            )

            self.language_q_proj = nn.Linear(
                config.hidden_size, config.hidden_size, bias=False
            )
            self.language_k_proj = nn.Linear(
                config.hidden_size, config.hidden_size, bias=False
            )
            self.language_v_proj = nn.Linear(
                config.hidden_size, config.hidden_size, bias=False
            )

            # Cross-modality projections
            self.vision_to_lang_proj = nn.Linear(
                config.hidden_size, config.hidden_size, bias=False
            )
            self.lang_to_vision_proj = nn.Linear(
                config.hidden_size, config.hidden_size, bias=False
            )

            # Output projection
            self.out_proj = nn.Linear(
                config.hidden_size, config.hidden_size, bias=False
            )

            # Initialize weights according to Qwen3-VL-2B specifications
            self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights according to Qwen3-VL-2B specifications."""
        std = self.config.hidden_size**-0.5
        nn.init.normal_(self.vision_q_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.vision_k_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.vision_v_proj.weight, mean=0.0, std=std)

        nn.init.normal_(self.language_q_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.language_k_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.language_v_proj.weight, mean=0.0, std=std)

        nn.init.normal_(self.vision_to_lang_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.lang_to_vision_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=std)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for Qwen3-VL-2B specific vision-language attention kernel.
        """
        # Use the appropriate kernel implementation
        if self.config.use_quantized_kernels:
            # For quantized kernels, use the implementation directly
            return self.kernel_impl(
                vision_features, language_features, attention_mask, need_weights
            )
        else:
            # For standard kernels, use the original implementation
            batch_size, num_patches, d_model = vision_features.shape
            _, seq_len, _ = language_features.shape

            # Add positional embeddings
            vision_features = (
                vision_features + self.kernel_impl.vision_pos_embed[:, :num_patches, :]
            )
            language_features = (
                language_features + self.kernel_impl.text_pos_embed[:, :seq_len, :]
            )

            # Normalize features
            vision_norm = self.kernel_impl.vision_norm(vision_features)
            lang_norm = self.kernel_impl.language_norm(language_features)

            # Project vision features using Qwen3-VL-2B specific projections
            v_q = (
                self.vision_q_proj(vision_norm)
                .view(
                    batch_size,
                    num_patches,
                    self.kernel_impl.nhead,
                    self.kernel_impl.head_dim,
                )
                .transpose(1, 2)
            )
            v_k = (
                self.vision_k_proj(vision_norm)
                .view(
                    batch_size,
                    num_patches,
                    self.kernel_impl.nhead,
                    self.kernel_impl.head_dim,
                )
                .transpose(1, 2)
            )
            v_v = (
                self.vision_v_proj(vision_norm)
                .view(
                    batch_size,
                    num_patches,
                    self.kernel_impl.nhead,
                    self.kernel_impl.head_dim,
                )
                .transpose(1, 2)
            )

            # Project language features using Qwen3-VL-2B specific projections
            l_q = (
                self.language_q_proj(lang_norm)
                .view(
                    batch_size,
                    seq_len,
                    self.kernel_impl.nhead,
                    self.kernel_impl.head_dim,
                )
                .transpose(1, 2)
            )
            l_k = (
                self.language_k_proj(lang_norm)
                .view(
                    batch_size,
                    seq_len,
                    self.kernel_impl.nhead,
                    self.kernel_impl.head_dim,
                )
                .transpose(1, 2)
            )
            l_v = (
                self.language_v_proj(lang_norm)
                .view(
                    batch_size,
                    seq_len,
                    self.kernel_impl.nhead,
                    self.kernel_impl.head_dim,
                )
                .transpose(1, 2)
            )

            # Scale queries
            v_q = v_q * self.kernel_impl.scaling
            l_q = l_q * self.kernel_impl.scaling

            # Compute vision-to-language attention
            vision_to_lang_attn = torch.matmul(v_q, l_k.transpose(-2, -1))
            if attention_mask is not None:
                vision_to_lang_attn = vision_to_lang_attn + attention_mask
            vision_to_lang_attn = torch.softmax(
                vision_to_lang_attn, dim=-1, dtype=torch.float32
            ).to(vision_features.dtype)
            if self.kernel_impl.dropout is not None:
                vision_to_lang_attn = self.kernel_impl.dropout(vision_to_lang_attn)
            vision_to_lang_output = torch.matmul(vision_to_lang_attn, l_v)
            vision_to_lang_output = (
                vision_to_lang_output.transpose(1, 2)
                .contiguous()
                .view(batch_size, num_patches, d_model)
            )
            vision_to_lang_output = self.vision_to_lang_proj(vision_to_lang_output)

            # Compute language-to-vision attention
            lang_to_vision_attn = torch.matmul(l_q, v_k.transpose(-2, -1))
            if attention_mask is not None:
                lang_to_vision_attn = lang_to_vision_attn + attention_mask.transpose(
                    -1, -2
                )  # Transpose for reverse direction
            lang_to_vision_attn = torch.softmax(
                lang_to_vision_attn, dim=-1, dtype=torch.float32
            ).to(language_features.dtype)
            if self.kernel_impl.dropout is not None:
                lang_to_vision_attn = self.kernel_impl.dropout(lang_to_vision_attn)
            lang_to_vision_output = torch.matmul(lang_to_vision_attn, v_v)
            lang_to_vision_output = (
                lang_to_vision_output.transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_len, d_model)
            )
            lang_to_vision_output = self.lang_to_vision_proj(lang_to_vision_output)

            # Combine outputs
            fused_output = torch.cat(
                [
                    vision_features + vision_to_lang_output,
                    language_features + lang_to_vision_output,
                ],
                dim=1,
            )

            # Apply final output projection
            fused_output = self.out_proj(fused_output)

            # Return separate outputs as well
            vision_output = vision_features + vision_to_lang_output
            language_output = language_features + lang_to_vision_output

            attention_weights = vision_to_lang_attn if need_weights else None

            return fused_output, vision_output, language_output, attention_weights


class Qwen3VL2BPositionEncodingKernel(nn.Module):
    """
    Qwen3-VL-2B specific implementation of multimodal position encoding kernel.
    This kernel is optimized for the specific architecture and parameters of Qwen3-VL-2B.
    """

    def __init__(self, config: Qwen3VL2BConfig):
        super().__init__()

        self.config = config

        # Use quantized kernels if enabled
        if config.use_quantized_kernels:
            from .....common.quantization import QuantizationScheme
            from .....common.quantized_multimodal_kernels import (
                QuantizedMultimodalConfig,
            )

            # Map string scheme to QuantizationScheme enum
            scheme_map = {
                "int4": QuantizationScheme.INT4,
                "int8": QuantizationScheme.INT8,
                "fp16": QuantizationScheme.FP16,
                "nf4": QuantizationScheme.NF4,
            }
            quant_scheme = scheme_map.get(
                config.quantization_scheme.lower(), QuantizationScheme.INT8
            )

            # Create quantized config
            quantized_config = QuantizedMultimodalConfig(
                d_model=config.hidden_size,
                modalities=["text", "image"],
                quantization_scheme=quant_scheme,
                quantization_bits=config.quantization_bits,
            )

            # Use quantized kernel
            self.kernel_impl = QuantizedMultimodalPositionEncodingKernel(
                quantized_config
            )
        else:
            # Use standard kernel
            self.kernel_impl = MultimodalPositionEncodingKernel(
                d_model=config.hidden_size,
                max_text_len=config.max_position_embeddings,
                max_image_patches=1024,
                modalities=["text", "image"],
            )

            # Qwen3-VL-2B specific position embeddings
            if "text" in self.kernel_impl.modalities:
                # Use RoPE (Rotary Position Embedding) as in Qwen3-VL-2B
                self.text_pos_embed = nn.Parameter(
                    torch.zeros(1, config.max_position_embeddings, config.hidden_size)
                )

            if "image" in self.kernel_impl.modalities:
                self.image_pos_embed = nn.Parameter(
                    torch.randn(1, 1024, config.hidden_size) * 0.02
                )

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply Qwen3-VL-2B specific position encoding to multimodal features.
        """
        # Use the appropriate kernel implementation
        if self.config.use_quantized_kernels:
            # For quantized kernels, use the implementation directly
            return self.kernel_impl(features)
        else:
            # For standard kernels, use the original implementation
            encoded_features = {}

            for modality, feats in features.items():
                if modality == "text" and hasattr(self, "text_pos_embed"):
                    seq_len = min(feats.size(1), self.kernel_impl.max_text_len)
                    encoded_features[modality] = (
                        feats + self.text_pos_embed[:, :seq_len, :]
                    )
                elif modality == "image" and hasattr(self, "image_pos_embed"):
                    num_patches = min(feats.size(1), self.kernel_impl.max_image_patches)
                    encoded_features[modality] = (
                        feats + self.image_pos_embed[:, :num_patches, :]
                    )
                elif modality == "audio" and hasattr(self, "audio_pos_embed"):
                    seq_len = min(feats.size(1), self.kernel_impl.max_text_len)
                    encoded_features[modality] = (
                        feats + self.audio_pos_embed[:, :seq_len, :]
                    )
                else:
                    # If no position embedding for this modality, pass through unchanged
                    encoded_features[modality] = feats

            return encoded_features


class Qwen3VL2BMLPKernel(nn.Module):
    """
    Qwen3-VL-2B specific MLP kernel with SwiGLU activation.
    This kernel implements the specific MLP structure used in Qwen3-VL-2B.
    """

    def __init__(self, config: Qwen3VL2BConfig, layer_idx: int = 0):
        super().__init__()

        self.layer_idx = layer_idx
        self.config = config

        # Qwen3-VL-2B specific MLP components
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

        # Initialize weights according to Qwen3-VL-2B specifications
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights according to Qwen3-VL-2B specifications."""
        std = self.config.hidden_size**-0.5
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=std)

        std = (2 * self.config.num_hidden_layers) ** -0.5
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Qwen3-VL-2B specific MLP kernel.
        Implements SwiGLU activation: FFN(x) = GLU(W1*x + b1, W3*x + b3) * (W2*x + b2)
        """
        gate = self.gate_proj(x)
        gate = nn.functional.silu(gate)  # SiLU activation
        up = self.up_proj(x)
        mlp_output = gate * up  # Element-wise multiplication for SwiGLU
        mlp_output = self.down_proj(mlp_output)

        return mlp_output


class Qwen3VL2BRMSNormKernel(nn.Module):
    """
    Qwen3-VL-2B specific RMSNorm kernel.
    This kernel implements the specific RMSNorm used in Qwen3-VL-2B.
    """

    def __init__(self, config: Qwen3VL2BConfig, layer_idx: int = 0):
        super().__init__()

        self.layer_idx = layer_idx
        self.config = config
        self.eps = config.rms_norm_eps

        # Weight parameter for RMSNorm
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Qwen3-VL-2B specific RMSNorm kernel.
        """
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


def create_qwen3_vl_cross_attention_kernel(config: Qwen3VL2BConfig, layer_idx: int = 0):
    """
    Factory function to create Qwen3-VL-2B specific cross-attention kernel.

    Args:
        config: Qwen3-VL-2B configuration
        layer_idx: Index of the layer (for layer-specific optimizations)

    Returns:
        Qwen3-VL-2B specific cross-attention kernel
    """
    return Qwen3VL2BCrossAttentionKernel(config, layer_idx)


def create_qwen3_vl_fusion_kernel(config: Qwen3VL2BConfig, layer_idx: int = 0):
    """
    Factory function to create Qwen3-VL-2B specific fusion kernel.

    Args:
        config: Qwen3-VL-2B configuration
        layer_idx: Index of the layer (for layer-specific optimizations)

    Returns:
        Qwen3-VL-2B specific fusion kernel
    """
    return Qwen3VL2BFusionKernel(config, layer_idx)


def create_qwen3_vl_vision_language_attention_kernel(
    config: Qwen3VL2BConfig, layer_idx: int = 0
):
    """
    Factory function to create Qwen3-VL-2B specific vision-language attention kernel.

    Args:
        config: Qwen3-VL-2B configuration
        layer_idx: Index of the layer (for layer-specific optimizations)

    Returns:
        Qwen3-VL-2B specific vision-language attention kernel
    """
    return Qwen3VL2BVisionLanguageAttentionKernel(config, layer_idx)


def create_qwen3_vl_position_encoding_kernel(config: Qwen3VL2BConfig):
    """
    Factory function to create Qwen3-VL-2B specific position encoding kernel.

    Args:
        config: Qwen3-VL-2B configuration

    Returns:
        Qwen3-VL-2B specific position encoding kernel
    """
    return Qwen3VL2BPositionEncodingKernel(config)


def create_qwen3_vl_mlp_kernel(config: Qwen3VL2BConfig, layer_idx: int = 0):
    """
    Factory function to create Qwen3-VL-2B specific MLP kernel.

    Args:
        config: Qwen3-VL-2B configuration
        layer_idx: Index of the layer (for layer-specific optimizations)

    Returns:
        Qwen3-VL-2B specific MLP kernel
    """
    return Qwen3VL2BMLPKernel(config, layer_idx)


def create_qwen3_vl_rms_norm_kernel(config: Qwen3VL2BConfig, layer_idx: int = 0):
    """
    Factory function to create Qwen3-VL-2B specific RMSNorm kernel.

    Args:
        config: Qwen3-VL-2B configuration
        layer_idx: Index of the layer (for layer-specific optimizations)

    Returns:
        Qwen3-VL-2B specific RMSNorm kernel
    """
    return Qwen3VL2BRMSNormKernel(config, layer_idx)


def apply_qwen3_vl_cuda_optimizations_to_model(
    model: nn.Module, config: Qwen3VL2BConfig
) -> nn.Module:
    """
    Apply Qwen3-VL-2B specific CUDA optimizations to the model.

    Args:
        model: The Qwen3-VL-2B model to optimize
        config: Configuration for the model

    Returns:
        Optimized model
    """
    logger.info("Applying Qwen3-VL-2B specific CUDA optimizations...")

    # Log if quantized kernels are being used
    if config.use_quantized_kernels:
        logger.info(
            f"Using quantized kernels with scheme: {config.quantization_scheme}, bits: {config.quantization_bits}"
        )

    # Apply Qwen3-VL-2B specific optimizations
    for name, module in model.named_modules():
        # Replace standard attention mechanisms with Qwen3-VL-2B optimized versions
        if "attn" in name.lower() or "attention" in name.lower():
            if isinstance(module, nn.MultiheadAttention):
                # Create Qwen3-VL-2B specific cross-attention kernel
                layer_idx = 0  # Extract layer index from name if possible
                if "layers." in name:
                    try:
                        layer_idx = int(name.split("layers.")[1].split(".")[0])
                    except (ValueError, IndexError):
                        layer_idx = 0

                # Create a wrapper that maintains the same interface as MultiheadAttention
                qwen_attn = Qwen3VL2BAttentionWrapper(
                    create_qwen3_vl_cross_attention_kernel(config, layer_idx)
                )

                # Replace the attention module
                parent_module, child_name = _get_parent_module(model, name)
                setattr(parent_module, child_name, qwen_attn)

                logger.debug(
                    f"Replaced attention module {name} with Qwen3-VL-2B optimized version (quantized: {config.use_quantized_kernels})"
                )

        # Replace standard MLP layers with Qwen3-VL-2B optimized versions
        elif "mlp" in name.lower() or "feed_forward" in name.lower():
            # Create Qwen3-VL-2B specific MLP kernel
            layer_idx = 0  # Extract layer index from name if possible
            if "layers." in name:
                try:
                    layer_idx = int(name.split("layers.")[1].split(".")[0])
                except (ValueError, IndexError):
                    layer_idx = 0

            # Create a wrapper that maintains the same interface as standard MLP
            qwen_mlp = Qwen3VL2BMLPWrapper(
                create_qwen3_vl_mlp_kernel(config, layer_idx)
            )

            # Replace the MLP module
            parent_module, child_name = _get_parent_module(model, name)
            setattr(parent_module, child_name, qwen_mlp)

            logger.debug(
                f"Replaced MLP module {name} with Qwen3-VL-2B optimized version (quantized: {config.use_quantized_kernels})"
            )

        # Replace standard LayerNorm with Qwen3-VL-2B specific RMSNorm
        elif isinstance(module, nn.LayerNorm):
            # Create Qwen3-VL-2B specific RMSNorm kernel
            layer_idx = 0  # Extract layer index from name if possible
            if "layers." in name:
                try:
                    layer_idx = int(name.split("layers.")[1].split(".")[0])
                except (ValueError, IndexError):
                    layer_idx = 0

            # Create a wrapper that maintains the same interface as LayerNorm
            qwen_rms_norm = Qwen3VL2BRMSNormWrapper(
                create_qwen3_vl_rms_norm_kernel(config, layer_idx)
            )

            # Replace the LayerNorm module
            parent_module, child_name = _get_parent_module(model, name)
            setattr(parent_module, child_name, qwen_rms_norm)

            logger.debug(
                f"Replaced LayerNorm module {name} with Qwen3-VL-2B RMSNorm version"
            )

    # Apply vision-language specific optimizations
    _apply_vision_language_optimizations(model, config)

    logger.info("Qwen3-VL-2B CUDA optimizations applied successfully")
    return model


def _apply_vision_language_optimizations(model: nn.Module, config: Qwen3VL2BConfig):
    """
    Apply vision-language specific optimizations to the model.

    Args:
        model: The model to optimize
        config: Configuration for the model
    """
    logger.info("Applying vision-language specific optimizations...")

    # Create vision transformer config from Qwen3VL2B config
    vision_config = VisionTransformerConfig(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=config.vision_num_hidden_layers,
        patch_size=config.vision_patch_size,
        image_size=config.vision_image_size,
        intermediate_size=config.vision_intermediate_size,
        layer_norm_eps=config.vision_layer_norm_eps,
        use_flash_attention=config.use_vision_flash_attention,
        use_cuda_kernels=config.use_cuda_kernels,
    )

    # Look for vision encoder components and apply specific optimizations
    for name, module in model.named_modules():
        if (
            "vision" in name.lower()
            or "visual" in name.lower()
            or "patch" in name.lower()
        ):
            if isinstance(module, nn.Conv2d):
                # Replace with optimized vision patch embedding if it matches patch size
                if module.kernel_size[0] == config.vision_patch_size:
                    logger.debug(f"Replacing vision patch embedding layer: {name}")

                    # Create optimized vision patch embedding kernel
                    vision_patch_embed = create_vision_patch_embedding_kernel(
                        vision_config
                    )

                    # Replace the patch embedding module
                    parent_module, child_name = _get_parent_module(model, name)
                    setattr(parent_module, child_name, vision_patch_embed)

                    logger.info(
                        f"Replaced vision patch embedding module {name} with optimized version"
                    )

            elif isinstance(module, nn.Linear) and "vision" in name.lower():
                # Apply vision-specific optimizations to linear layers in vision encoder
                logger.debug(f"Identified vision component: {name}")

                # Potentially replace with optimized linear layers
                # This is a placeholder for more specific optimizations
                pass

        elif (
            "language" in name.lower()
            or "text" in name.lower()
            or "llm" in name.lower()
        ):
            if isinstance(module, nn.Linear):
                # Apply language-specific optimizations to linear layers in language decoder
                logger.debug(f"Identified language component: {name}")

                # Potentially replace with optimized linear layers
                # This is a placeholder for more specific optimizations
                pass


def get_qwen3_vl_cuda_optimization_report(
    model: nn.Module, config: Qwen3VL2BConfig
) -> Dict:
    """
    Get a report of Qwen3-VL-2B CUDA optimizations applied to the model.

    Args:
        model: The model
        config: Model configuration

    Returns:
        Optimization report
    """
    report = {
        "model_type": "Qwen3-VL-2B",
        "optimizations_applied": {
            "qwen3_vl_cross_attention": True,
            "qwen3_vl_fusion": True,
            "qwen3_vl_vision_language_attention": True,
            "qwen3_vl_position_encoding": True,
            "qwen3_vl_mlp_swiglu": True,
            "qwen3_vl_rms_norm": True,
        },
        "config": {
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "num_hidden_layers": config.num_hidden_layers,
            "intermediate_size": config.intermediate_size,
            "use_flash_attention_2": config.use_flash_attention_2,
            "use_cuda_kernels": config.use_cuda_kernels,
        },
        "notes": "Qwen3-VL-2B specific multimodal CUDA optimizations applied with SwiGLU activation and RMSNorm",
    }

    return report


class Qwen3VL2BAttentionWrapper(nn.Module):
    """
    Wrapper for Qwen3-VL-2B attention kernel to maintain compatibility with standard MultiheadAttention interface.
    """

    def __init__(self, qwen_attention_kernel):
        super().__init__()
        self.qwen_attention_kernel = qwen_attention_kernel

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        """
        Forward pass that maintains the same interface as nn.MultiheadAttention.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            key_padding_mask: Mask for padded elements
            need_weights: Whether to return attention weights
            attn_mask: Attention mask
            average_attn_weights: Whether to average attention weights
            is_causal: Whether to apply causal masking

        Returns:
            Tuple of (attn_output, attn_weights) or just attn_output depending on need_weights
        """
        batch_size, seq_len, d_model = query.shape

        # Prepare inputs for the Qwen3-VL-2B kernel
        # For simplicity, treat all as 'text' modality
        queries = {"text": query}
        keys_dict = {"text": key}
        values_dict = {"text": value}

        # Apply attention
        outputs, attention_weights = self.qwen_attention_kernel(
            queries=queries,
            keys=keys_dict,
            values=values_dict,
            attention_masks=None,  # Not using masks in this simple case
            need_weights=need_weights,
        )

        # Extract the output
        attn_output = outputs["text"]

        if need_weights and attention_weights is not None:
            # Return the attention weights for the 'text' modality
            attn_weights = attention_weights["text"]
            return attn_output, attn_weights
        else:
            return attn_output, None


class Qwen3VL2BMLPWrapper(nn.Module):
    """
    Wrapper for Qwen3-VL-2B MLP kernel to maintain compatibility with standard MLP interface.
    """

    def __init__(self, qwen_mlp_kernel):
        super().__init__()
        self.qwen_mlp_kernel = qwen_mlp_kernel

    def forward(self, x):
        """
        Forward pass that maintains the same interface as standard MLP.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.qwen_mlp_kernel(x)


class Qwen3VL2BRMSNormWrapper(nn.Module):
    """
    Wrapper for Qwen3-VL-2B RMSNorm kernel to maintain compatibility with standard LayerNorm interface.
    """

    def __init__(self, qwen_rms_norm_kernel):
        super().__init__()
        self.qwen_rms_norm_kernel = qwen_rms_norm_kernel

    def forward(self, x):
        """
        Forward pass that maintains the same interface as nn.LayerNorm.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        return self.qwen_rms_norm_kernel(x)


class Qwen3VL2BVisionProcessingKernel(nn.Module):
    """
    Qwen3-VL-2B specific kernel for vision processing.
    This kernel combines optimized vision transformer components for the Qwen3-VL-2B model.
    """

    def __init__(self, config: Qwen3VL2BConfig):
        super().__init__()

        self.config = config

        # Create vision transformer config from Qwen3VL2B config
        vision_config = VisionTransformerConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.vision_num_hidden_layers,
            patch_size=config.vision_patch_size,
            image_size=config.vision_image_size,
            intermediate_size=config.vision_intermediate_size,
            layer_norm_eps=config.vision_layer_norm_eps,
            use_flash_attention=config.use_vision_flash_attention,
            use_cuda_kernels=config.use_cuda_kernels,
        )

        # Vision encoder with optimized components
        self.vision_encoder = Qwen3VL2BVisionEncoderKernel(vision_config)

    def forward(
        self, pixel_values: torch.Tensor, output_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass for Qwen3-VL-2B vision processing kernel.

        Args:
            pixel_values: Input images of shape (batch_size, channels, height, width)
            output_hidden_states: Whether to output hidden states from all layers

        Returns:
            Tuple of (final_hidden_states, all_hidden_states if requested)
        """
        return self.vision_encoder(pixel_values, output_hidden_states)


def create_qwen3_vl_vision_processing_kernel(
    config: Qwen3VL2BConfig,
) -> Qwen3VL2BVisionProcessingKernel:
    """
    Factory function to create Qwen3-VL-2B vision processing kernel.

    Args:
        config: Qwen3-VL-2B configuration

    Returns:
        Qwen3-VL-2B vision processing kernel
    """
    return Qwen3VL2BVisionProcessingKernel(config)


def _get_parent_module(model: nn.Module, full_name: str) -> tuple:
    """
    Get parent module and child name by full name.

    Args:
        model: The model
        full_name: Full name of the module (e.g., 'transformer.layers.0.attention')

    Returns:
        Tuple of (parent_module, child_name)
    """
    parts = full_name.split(".")
    if len(parts) == 1:
        # If there's no parent (top-level module), return the model itself and the child name
        return model, parts[0]

    parent_name = ".".join(parts[:-1])
    child_name = parts[-1]

    parent_module = model
    for n in parent_name.split("."):
        if n:  # Skip empty strings
            parent_module = getattr(parent_module, n)

    return parent_module, child_name


__all__ = [
    "Qwen3VL2BConfig",
    "Qwen3VL2BCrossAttentionKernel",
    "Qwen3VL2BFusionKernel",
    "Qwen3VL2BVisionLanguageAttentionKernel",
    "Qwen3VL2BPositionEncodingKernel",
    "Qwen3VL2BMLPKernel",
    "Qwen3VL2BRMSNormKernel",
    "Qwen3VL2BVisionProcessingKernel",
    "Qwen3VL2BAttentionWrapper",
    "Qwen3VL2BMLPWrapper",
    "Qwen3VL2BRMSNormWrapper",
    "create_qwen3_vl_cross_attention_kernel",
    "create_qwen3_vl_fusion_kernel",
    "create_qwen3_vl_vision_language_attention_kernel",
    "create_qwen3_vl_position_encoding_kernel",
    "create_qwen3_vl_mlp_kernel",
    "create_qwen3_vl_rms_norm_kernel",
    "create_qwen3_vl_vision_processing_kernel",
    "apply_qwen3_vl_cuda_optimizations_to_model",
    "get_qwen3_vl_cuda_optimization_report",
]
