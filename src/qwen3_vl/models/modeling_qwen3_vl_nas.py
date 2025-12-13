"""
Qwen3-VL Model implementation with integrated Neural Architecture Search (NAS) system
for layer-specific configuration optimization.
"""
import math
import warnings
from typing import Optional, Tuple, Union, List

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from src.qwen3_vl.config.config import Qwen3VLConfig
from src.qwen3_vl.models.linear_attention import PerformerAttention
from src.qwen3_vl.models.device_aware_module import DeviceAwareAttention
from src.qwen3_vl.models.gradient_checkpointing import MemoryEfficientAttention, MemoryEfficientMLP
from src.qwen3_vl.models.adaptive_computation import AdaptiveAttention, AdaptiveMLP
from src.qwen3_vl.models.memory_management import OptimizedQwen3VLAttention
from src.qwen3_vl.models.adapter_layers import AdapterLayer
from src.qwen3_vl.nas_system import Qwen3VLNeuralArchitectureSearch, LayerConfig, VisionLayerConfig, LanguageLayerConfig


class Qwen3VLPreTrainedModel(nn.Module):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
    """
    config_class = Qwen3VLConfig
    base_model_prefix = "qwen3_vl"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3VLDecoderLayer", "Qwen3VLVisionLayer"]

    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def post_init(self):
        """Initialize weights and apply final processing."""
        self.apply(self._init_weights)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Qwen3VLDecoder, Qwen3VLVisionTransformer)):
            module.gradient_checkpointing = value


class Qwen3VLAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper with Qwen3-specific modifications.
    Integrates efficiency improvements from Phase 2 while maintaining all 32 attention heads.
    """
    def __init__(self, config: Qwen3VLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            warnings.warn(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if model `use_cache=True`."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads or self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Use the most appropriate attention implementation based on configuration
        # Prioritize FlashAttention-2 if enabled
        if getattr(config, 'use_dynamic_sparse_attention', False):
            # Use dynamic sparse attention with learned routing for token selection
            from src.qwen3_vl.models.dynamic_sparse_attention import DynamicSparseAttention
            self.attention_impl = DynamicSparseAttention(config, layer_idx=layer_idx)
        elif config.use_flash_attention_2:
            # Use FlashAttention-2 for memory efficiency
            from src.qwen3_vl.models.moe_flash_attention import FlashAttention
            self.attention_impl = FlashAttention(config, layer_idx=layer_idx)
        elif config.attention_implementation == "performer":
            # Use Performer-style linear attention
            self.attention_impl = PerformerAttention(config, layer_idx)
        elif config.attention_implementation == "device_aware":
            # Use device-aware attention
            self.attention_impl = DeviceAwareAttention(config, layer_idx)
        elif config.attention_implementation == "adaptive":
            # Use adaptive attention
            self.attention_impl = AdaptiveAttention(config, layer_idx)
        elif config.attention_implementation == "memory_efficient":
            # Use memory-efficient attention with gradient checkpointing
            self.attention_impl = MemoryEfficientAttention(config, layer_idx)
        elif config.attention_implementation == "kv_cache_optimized":
            # Use KV cache optimized attention with low-rank and sliding window techniques
            from src.qwen3_vl.models.kv_cache_optimization import OptimizedKVCachingAttention
            self.attention_impl = OptimizedKVCachingAttention(
                config,
                layer_idx,
                cache_strategy=getattr(config, 'kv_cache_strategy', 'hybrid'),
                use_low_rank=getattr(config, 'use_low_rank_kv_cache', True),
                window_size=getattr(config, 'kv_cache_window_size', 1024),
                low_rank_rank=getattr(config, 'kv_low_rank_dimension', 64)
            )
        else:
            # Use optimized attention with memory management
            self.attention_impl = OptimizedQwen3VLAttention(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        return self.attention_impl(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen3VLRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3VLMLP(nn.Module):
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Use the most appropriate MLP implementation based on configuration
        if config.attention_implementation == "adaptive":
            # Use adaptive MLP
            from src.qwen3_vl.models.adaptive_computation import AdaptiveMLP
            self.mlp_impl = AdaptiveMLP(config)
        elif config.attention_implementation == "memory_efficient":
            # Use memory-efficient MLP with gradient checkpointing
            from src.qwen3_vl.models.gradient_checkpointing import MemoryEfficientMLP
            self.mlp_impl = MemoryEfficientMLP(config)
        else:
            # Use standard implementation
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
            self.act_fn = nn.SiLU()

    def forward(self, x):
        if hasattr(self, 'mlp_impl'):
            return self.mlp_impl(x)
        else:
            # For standard implementation
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3VLConfig, layer_idx: int, layer_config: Optional[LayerConfig] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Use provided layer configuration if available (from NAS system)
        if layer_config:
            # Use NAS-optimized configuration
            self.hidden_size = layer_config.hidden_size
            self.num_heads = layer_config.num_attention_heads
            self.intermediate_size = layer_config.intermediate_size
        else:
            # Use default configuration
            self.hidden_size = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.intermediate_size = config.intermediate_size

        # Update config with NAS-optimized parameters
        updated_config = config
        updated_config.hidden_size = self.hidden_size
        updated_config.num_attention_heads = self.num_heads
        updated_config.intermediate_size = self.intermediate_size

        # Check if sparsity and early exit mechanisms should be used
        if hasattr(config, 'use_sparsity') and config.use_sparsity:
            from src.qwen3_vl.models.activation_sparsity import EfficientTransformerBlock
            # Use the enhanced transformer block with sparsity, early exit, and routing
            self.layer_block = EfficientTransformerBlock(
                config=updated_config,
                layer_idx=layer_idx,
                sparsity_ratio=getattr(config, 'sparsity_ratio', 0.5),
                exit_threshold=getattr(config, 'exit_threshold', 0.8)
            )
        else:
            # Use standard attention and MLP, with potential MoE and parameter sharing
            self.self_attn = Qwen3VLAttention(config=config, layer_idx=layer_idx)

            # Use MoE if configured, otherwise standard MLP
            if hasattr(config, 'use_moe') and config.use_moe:
                from src.qwen3_vl.models.moe_flash_attention import MoeLayer
                self.mlp = MoeLayer(
                    updated_config,
                    num_experts=config.moe_num_experts,
                    top_k=config.moe_top_k
                )
            else:
                self.mlp = Qwen3VLMLP(updated_config)

            self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
            self.layer_block = None

        # Add optional adapter layers for parameter-efficient adaptation
        if hasattr(config, 'use_adapters') and config.use_adapters:
            from src.qwen3_vl.models.adapter_layers import AdapterConfig
            adapter_config = getattr(config, 'adapter_config', AdapterConfig())
            self.attn_adapter = AdapterLayer(adapter_config, self.hidden_size)
            self.mlp_adapter = AdapterLayer(adapter_config, self.hidden_size)
        else:
            self.attn_adapter = None
            self.mlp_adapter = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if self.layer_block is not None:
            # Use the enhanced transformer block with sparsity, early exit, and routing
            outputs = self.layer_block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            # Extract the hidden states (first element) and handle the additional flags
            hidden_states = outputs[0]

            # If we have early exit or skip information, we can use it for optimization
            # but for now we just return the standard format
            result = (hidden_states,)

            if output_attentions and len(outputs) > 1:
                result += (outputs[1],)  # attention weights if available
            if use_cache and len(outputs) > 2:
                result += (outputs[2],)  # cache if available

            return result
        else:
            # Use standard transformer layer computation
            residual = hidden_states

            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = residual + hidden_states

            # Apply attention adapter if enabled
            if self.attn_adapter is not None:
                hidden_states = self.attn_adapter(hidden_states)

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            # Apply MLP adapter if enabled
            if self.mlp_adapter is not None:
                hidden_states = self.mlp_adapter(hidden_states)

            outputs = (hidden_states,)

            if output_attentions:
                outputs += (self_attn_weights,)

            if use_cache:
                outputs += (present_key_value,)

            return outputs


class Qwen3VLVisionAttention(nn.Module):
    """Vision attention mechanism for the vision encoder."""
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.vision_hidden_size
        self.num_heads = config.vision_num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout_prob

        # Use dynamic sparse attention for vision if configured
        if getattr(config, 'use_dynamic_sparse_attention', False):
            from src.qwen3_vl.models.dynamic_sparse_attention import VisionDynamicSparseAttention
            self.attention_impl = VisionDynamicSparseAttention(config)
        else:
            self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.vision_qkv_bias)
            self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self, 'attention_impl'):
            # Use dynamic sparse attention
            return self.attention_impl(hidden_states)
        else:
            # Use standard attention
            bsz, tgt_len, embed_dim = hidden_states.size()

            # QKV projection
            qkv = self.qkv(hidden_states)
            qkv = qkv.reshape(bsz, tgt_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Attention computation
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
            attn_output = self.proj(attn_output)

            return attn_output


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(config.vision_hidden_size, config.vision_intermediate_size)
        self.fc2 = nn.Linear(config.vision_intermediate_size, config.vision_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Qwen3VLVisionLayer(nn.Module):
    def __init__(self, config: Qwen3VLConfig, layer_idx: int, layer_config: Optional[VisionLayerConfig] = None):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Use provided layer configuration if available (from NAS system)
        if layer_config:
            self.embed_dim = layer_config.hidden_size
            self.num_heads = layer_config.num_attention_heads
            self.intermediate_size = layer_config.intermediate_size
        else:
            self.embed_dim = config.vision_hidden_size
            self.num_heads = config.vision_num_attention_heads
            self.intermediate_size = config.vision_intermediate_size

        self.norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.attn = Qwen3VLVisionAttention(config)
        self.norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Qwen3VLVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3VLVisionTransformer(nn.Module):
    def __init__(self, config: Qwen3VLConfig, vision_layer_configs: Optional[List[VisionLayerConfig]] = None):
        super().__init__()
        self.config = config
        self.embed_dim = config.vision_hidden_size

        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels=config.vision_num_channels,
            out_channels=self.embed_dim,
            kernel_size=config.vision_patch_size,
            stride=config.vision_patch_size,
            bias=False
        )

        # Calculate the number of patches based on the expected image size
        self.num_patches_per_dim = config.vision_image_size // config.vision_patch_size
        self.num_patches = self.num_patches_per_dim ** 2

        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False
        )

        self.pre_layrnorm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        
        # Create vision layers with NAS-optimized configurations if provided
        if vision_layer_configs:
            # Use NAS-optimized configurations
            self.layers = nn.ModuleList([
                Qwen3VLVisionLayer(config, layer_idx, layer_config=vision_layer_configs[layer_idx])
                for layer_idx in range(config.vision_num_hidden_layers)
            ])
        else:
            # Use default configurations
            self.layers = nn.ModuleList([
                Qwen3VLVisionLayer(config, layer_idx)
                for layer_idx in range(config.vision_num_hidden_layers)
            ])
        
        self.post_layernorm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
    ) -> torch.FloatTensor:
        batch_size = pixel_values.shape[0]
        image_height, image_width = pixel_values.shape[-2], pixel_values.shape[-1]

        # Calculate how many patches we'll have based on input image size
        patch_height = image_height // self.config.vision_patch_size
        patch_width = image_width // self.config.vision_patch_size
        num_patches = patch_height * patch_width

        # Patch embedding
        patch_embeds = self.patch_embedding(pixel_values)  # shape (batch_size, embed_dim, patch_height, patch_width)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # shape (batch_size, num_patches, embed_dim)

        # Handle position embeddings - use only the needed portion or interpolate if needed
        if num_patches <= self.num_patches:
            # Use the first num_patches position embeddings
            position_embeddings = self.position_embedding.weight[:num_patches].unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # If more patches than expected, we need to handle this case
            # For now, we'll raise an error as this would require interpolation
            raise ValueError(f"Input image has {num_patches} patches, but model expects max {self.num_patches} patches. "
                             f"Input size: ({image_height}, {image_width}), expected: ({self.config.vision_image_size}, {self.config.vision_image_size})")

        # Add position embeddings
        embeddings = patch_embeds + position_embeddings

        # Pre-layer norm
        hidden_states = self.pre_layrnorm(embeddings)

        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Post-layer norm
        hidden_states = self.post_layernorm(hidden_states)

        return hidden_states


class Qwen3VLDecoder(nn.Module):
    def __init__(self, config: Qwen3VLConfig, language_layer_configs: Optional[List[LanguageLayerConfig]] = None):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Create layers with NAS-optimized configurations if provided
        if language_layer_configs:
            # Use NAS-optimized configurations
            self.layers = nn.ModuleList([
                Qwen3VLDecoderLayer(config, layer_idx, layer_config=language_layer_configs[layer_idx])
                for layer_idx in range(config.num_hidden_layers)
            ])
        else:
            # Create layers with sparsity and early exit enabled if configured
            self.layers = nn.ModuleList(
                [Qwen3VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )

        self._use_gradient_checkpointing = config.use_gradient_checkpointing
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        # Embed tokens
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids cannot be None if inputs_embeds is None")
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # Create attention mask if not provided
        if attention_mask is None:
            if input_ids is not None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                # If input_ids is None, use inputs_embeds shape to create attention mask
                attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)

        # Apply causal mask
        causal_mask = self._prepare_decoder_attention_mask(
            attention_mask, hidden_states.shape[:2], hidden_states.dtype, past_key_values
        )

        # Apply transformer layers
        for decoder_layer in self.layers:
            if self._use_gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return hidden_states

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, dtype, past_key_values):
        """
        Create causal attention mask for decoder.
        """
        # Create causal mask
        batch_size, tgt_len = input_shape
        causal_mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=attention_mask.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)

        # Expand attention mask
        expanded_attn_mask = attention_mask[:, None, None, :].expand(batch_size, 1, tgt_len, tgt_len).to(dtype)
        expanded_attn_mask.masked_fill_(expanded_attn_mask == 0, torch.finfo(dtype).min)

        # Combine masks
        combined_mask = causal_mask.unsqueeze(0) + expanded_attn_mask
        return combined_mask


class Qwen3VLMultimodalProjector(nn.Module):
    """Projector to align vision and language features."""
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.linear_1 = nn.Linear(config.vision_hidden_size, config.vision_projection_dim, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(config.vision_projection_dim, config.hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Qwen3VLForConditionalGeneration(Qwen3VLPreTrainedModel):
    """ 
    Qwen3-VL model for multimodal conditional generation tasks with integrated NAS system.
    Maintains full capacity with 32 transformer layers and 32 attention heads.
    """
    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)
        self.config = config

        # Initialize NAS system for layer-specific optimization
        self.nas_system = Qwen3VLNeuralArchitectureSearch(
            num_layers=config.num_hidden_layers,
            base_hidden_size=config.hidden_size,
            base_num_heads=config.num_attention_heads,
            base_intermediate_size=getattr(config, 'intermediate_size', config.hidden_size * 4)
        )

        # Vision encoder - will be updated with NAS-optimized configs when available
        self.vision_tower = Qwen3VLVisionTransformer(config)

        # Multimodal projector
        self.multi_modal_projector = Qwen3VLMultimodalProjector(config)

        # Language model - will be updated with NAS-optimized configs when available
        self.language_model = Qwen3VLDecoder(config)

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.language_model.embed_tokens

    def set_output_embeddings(self, new_embeddings):
        self.language_model.embed_tokens = new_embeddings

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def optimize_architecture_for_input(self, input_data, input_type="text"):
        """
        Optimize the model architecture for the given input type using NAS.
        
        Args:
            input_data: Input data to optimize for (text tokens, image tensors, etc.)
            input_type: Type of input ("text", "vision", or "multimodal")
            
        Returns:
            Optimized layer configurations
        """
        optimized_configs = self.nas_system.search_optimal_architecture(
            input_data=input_data,
            input_type=input_type,
            num_search_steps=5,  # Adjust based on requirements
            num_candidates_per_step=3
        )
        
        # Update the model with optimized configurations
        if input_type == "vision":
            # Update vision tower with optimized configs
            self.vision_tower = Qwen3VLVisionTransformer(
                self.config, 
                vision_layer_configs=optimized_configs
            )
        elif input_type == "text":
            # Update language model with optimized configs
            self.language_model = Qwen3VLDecoder(
                self.config, 
                language_layer_configs=optimized_configs
            )
        else:  # multimodal
            # For multimodal, we might want to optimize both but for now just optimize language
            self.language_model = Qwen3VLDecoder(
                self.config, 
                language_layer_configs=optimized_configs
            )
        
        return optimized_configs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        # Process image if provided
        if pixel_values is not None:
            # Extract visual features
            image_features = self.vision_tower(pixel_values)

            # Project visual features to language model dimension
            image_features = self.multi_modal_projector(image_features)

            # Process text with language model
            if input_ids is not None and inputs_embeds is None:
                inputs_embeds = self.language_model.embed_tokens(input_ids)

            # Combine visual and text features if both are available
            if inputs_embeds is not None:
                # For simplicity, we'll concatenate them (in practice, this would be more complex)
                combined_embeds = torch.cat([image_features, inputs_embeds], dim=1)

                # Create appropriate attention mask for combined embeddings
                if attention_mask is not None:
                    # Expand attention mask to account for image features
                    batch_size, seq_len = attention_mask.shape
                    image_seq_len = image_features.size(1)
                    image_attn_mask = torch.ones(batch_size, image_seq_len, dtype=attention_mask.dtype, device=attention_mask.device)
                    combined_attention_mask = torch.cat([image_attn_mask, attention_mask], dim=1)
                else:
                    combined_attention_mask = None

                # Forward through language model
                outputs = self.language_model(
                    input_ids=None,  # We're using combined_embeds
                    attention_mask=combined_attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=combined_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            else:
                # Only image features, no text input
                # Create dummy input_ids to satisfy language model requirements
                batch_size, image_seq_len, embed_dim = image_features.shape
                dummy_input_ids = torch.zeros(batch_size, image_seq_len, dtype=torch.long, device=image_features.device)

                # Create attention mask for image features
                attention_mask = torch.ones(batch_size, image_seq_len, dtype=torch.bool, device=image_features.device)

                outputs = self.language_model(
                    input_ids=dummy_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=image_features,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        else:
            # Process text only
            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        return outputs

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        max_new_tokens: int = 50,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        optimize_for_input: bool = False,
        **kwargs
    ):
        """
        Generate tokens using the Qwen3-VL model with optional NAS optimization.
        """
        # Optionally optimize architecture for input before generation
        if optimize_for_input:
            if pixel_values is not None and input_ids is not None:
                # Multimodal input
                self.optimize_architecture_for_input(
                    input_data=(input_ids, pixel_values), 
                    input_type="multimodal"
                )
            elif pixel_values is not None:
                # Vision input
                self.optimize_architecture_for_input(
                    input_data=pixel_values, 
                    input_type="vision"
                )
            elif input_ids is not None:
                # Text input
                self.optimize_architecture_for_input(
                    input_data=input_ids, 
                    input_type="text"
                )

        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id

        # Get the initial input embeddings
        if pixel_values is not None:
            # Process visual features
            image_features = self.vision_tower(pixel_values)
            image_features = self.multi_modal_projector(image_features)

            if input_ids is not None:
                # Combine image and text features
                text_embeds = self.language_model.embed_tokens(input_ids)
                combined_embeds = torch.cat([image_features, text_embeds], dim=1)

                # Create attention mask for combined embeddings
                batch_size, text_seq_len = input_ids.shape
                image_seq_len = image_features.size(1)
                text_attn_mask = torch.ones(batch_size, text_seq_len, dtype=torch.bool, device=input_ids.device)
                image_attn_mask = torch.ones(batch_size, image_seq_len, dtype=torch.bool, device=input_ids.device)
                attention_mask = torch.cat([image_attn_mask, text_attn_mask], dim=1)
            else:
                # Only image features
                combined_embeds = image_features
                batch_size, image_seq_len = image_features.shape[:2]
                attention_mask = torch.ones(batch_size, image_seq_len, dtype=torch.bool, device=image_features.device)

            # Generate tokens using the language model
            generated_ids = input_ids if input_ids is not None else torch.zeros((batch_size, 0), dtype=torch.long, device=image_features.device)
            current_embeds = combined_embeds if input_ids is not None else image_features

            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.language_model(
                    input_ids=None,
                    inputs_embeds=current_embeds,
                    attention_mask=attention_mask
                )

                # Get the last token's logits
                next_token_logits = outputs[:, -1, :]

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k and top-p filtering
                if do_sample:
                    # Filter top-k
                    if top_k > 0:
                        indices_to_remove = next_token_logits > torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')

                    # Filter top-p
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Keep at least one option
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        for i in range(next_token_logits.size(0)):
                            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                            next_token_logits[i][indices_to_remove] = float('-inf')

                    # Sample next token
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Stop if EOS token is generated
                if eos_token_id is not None and next_tokens[0, 0].item() == eos_token_id:
                    break

                # Append generated token
                generated_ids = torch.cat([generated_ids, next_tokens], dim=1)

                # Get embeddings for the new token
                new_token_embeds = self.language_model.embed_tokens(next_tokens)
                current_embeds = new_token_embeds  # Only the new token embeddings for the next step

                # Extend attention mask
                new_attn_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, new_attn_mask], dim=1)

        else:
            # Text-only generation
            generated_ids = input_ids.clone()
            for _ in range(max_new_tokens):
                outputs = self.language_model(
                    input_ids=generated_ids,
                    attention_mask=torch.ones_like(generated_ids, dtype=torch.bool)
                )

                # Get the last token's logits
                next_token_logits = outputs[:, -1, :]

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k and top-p filtering
                if do_sample:
                    # Filter top-k
                    if top_k > 0:
                        indices_to_remove = next_token_logits > torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')

                    # Filter top-p
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Keep at least one option
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        for i in range(next_token_logits.size(0)):
                            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                            next_token_logits[i][indices_to_remove] = float('-inf')

                    # Sample next token
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Stop if EOS token is generated
                if eos_token_id is not None and next_tokens[0, 0].item() == eos_token_id:
                    break

                # Append generated token
                generated_ids = torch.cat([generated_ids, next_tokens], dim=1)

        return generated_ids

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        **kwargs
    ):
        """Prepare inputs for generation."""
        # Handle past_key_values
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # Process pixel values if provided
        if pixel_values is not None and past_key_values is None:
            # Extract visual features only once at the beginning
            image_features = self.vision_tower(pixel_values)
            image_features = self.multi_modal_projector(image_features)
            kwargs["image_features"] = image_features

        # Prepare model inputs
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "pixel_values": pixel_values if past_key_values is None else None,  # Only pass pixel_values on first call
        }

        if inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds

        return model_inputs