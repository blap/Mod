"""
Qwen3-VL-2B Attention Implementation - Self-Contained Version

This module implements optimized attention mechanisms specifically for the Qwen3-VL-2B model.
It includes multimodal attention, vision-language attention, and other Qwen3-VL-2B specific
attention optimizations.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Qwen3VL2BMultimodalAttention(nn.Module):
    """
    Qwen3-VL-2B specific multimodal attention mechanism.
    This attention mechanism is optimized for vision-language tasks with Qwen3-VL-2B specific parameters.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(
            config, "num_key_value_heads", self.num_heads
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got hidden_size: {self.hidden_size}, num_heads: {self.num_heads})"
            )

        # Qwen3-VL-2B specific attention projections
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Initialize weights according to Qwen3-VL-2B specifications
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights according to Qwen3-VL-2B specifications."""
        # Initialize projections with model-specific std
        std = self.hidden_size**-0.5
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)

        # Initialize output projection with different std based on layer
        std = (2 * self.config.num_hidden_layers) ** -0.5
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for Qwen3-VL-2B multimodal attention.

        Args:
            hidden_states: Input hidden states of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position IDs for rotary embeddings
            past_key_value: Past key-value states for caching
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache

        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        bsz, q_len, _ = hidden_states.size()

        # Apply projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Repeat key and value states for GQA (Grouped-Query Attention)
        key_states = torch.repeat_interleave(
            key_states, dim=2, repeats=self.num_key_value_groups
        )
        value_states = torch.repeat_interleave(
            value_states, dim=2, repeats=self.num_key_value_groups
        )

        # Apply rotary embeddings if position_ids are provided
        if position_ids is not None:
            from .rotary_embeddings import apply_rotary_pos_emb

            cos, sin = self.rotary_emb(
                value_states,
                seq_len=max(position_ids.max().item() + 1, key_states.shape[-2]),
            )
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

        # Handle past key values for caching
        if past_key_value is not None:
            # Reuse k,v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape for output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Apply output projection
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen3VL2BModalitySpecificAttention(nn.Module):
    """
    Qwen3-VL-2B specific attention mechanism for modality-specific processing.
    This attention focuses on processing specific modalities (text or image) with specialized parameters.
    """

    def __init__(self, config, modality: str = "text"):
        super().__init__()

        self.config = config
        self.modality = modality
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got hidden_size: {self.hidden_size}, num_heads: {self.num_heads})"
            )

        # Modality-specific projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Modality-specific normalization
        self.norm = nn.LayerNorm(self.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights according to Qwen3-VL-2B specifications
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights according to Qwen3-VL-2B specifications."""
        std = self.hidden_size**-0.5
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)

        std = (2 * self.config.num_hidden_layers) ** -0.5
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for Qwen3-VL-2B modality-specific attention.

        Args:
            hidden_states: Input hidden states of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position IDs for rotary embeddings
            past_key_value: Past key-value states for caching
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache

        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        bsz, q_len, _ = hidden_states.size()

        # Apply layer norm
        hidden_states = self.norm(hidden_states)

        # Apply projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Apply rotary embeddings if position_ids are provided
        if position_ids is not None:
            from .rotary_embeddings import apply_rotary_pos_emb

            cos, sin = self.rotary_emb(
                value_states,
                seq_len=max(position_ids.max().item() + 1, key_states.shape[-2]),
            )
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

        # Handle past key values for caching
        if past_key_value is not None:
            # Reuse k,v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape for output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Apply output projection
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen3VL2BMultimodalFusionLayer(nn.Module):
    """
    Qwen3-VL-2B specific multimodal fusion layer that combines vision and language representations.
    This layer implements the specific fusion mechanisms used in Qwen3-VL-2B.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size

        # Qwen3-VL-2B specific fusion components
        # SwiGLU-based fusion (similar to the one used in the model)
        self.gate_proj = nn.Linear(
            self.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(self.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            config.intermediate_size, self.hidden_size, bias=False
        )

        # Layer norm for fusion
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )

        # Initialize weights according to Qwen3-VL-2B specifications
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights according to Qwen3-VL-2B specifications."""
        # Initialize gate/up projections
        std = self.hidden_size**-0.5
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=std)

        # Initialize down projection with different std
        std = (2 * self.config.num_hidden_layers) ** -0.5
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=std)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for Qwen3-VL-2B multimodal fusion layer.

        Args:
            hidden_states: Input hidden states of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position IDs for rotary embeddings
            past_key_value: Past key-value states for caching
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache

        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        # Apply input layer norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Apply SwiGLU-based fusion
        gate = self.gate_proj(hidden_states)
        gate = F.silu(gate)  # SiLU activation
        up = self.up_proj(hidden_states)
        fused_intermediate = gate * up  # Element-wise multiplication for SwiGLU
        fused_output = self.down_proj(fused_intermediate)

        # Add residual connection
        hidden_states = residual + fused_output

        # Apply post-attention layer norm
        hidden_states = self.post_attention_layernorm(hidden_states)

        return hidden_states, None, past_key_value


class Qwen3VL2BAdaptiveMultimodalAttention(nn.Module):
    """
    Qwen3-VL-2B specific adaptive multimodal attention that adjusts based on input complexity.
    This attention mechanism dynamically adjusts its behavior based on the complexity of input modalities.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Multiple attention heads for different complexity levels
        self.simple_attention = Qwen3VL2BMultimodalAttention(config)
        self.complex_attention = Qwen3VL2BMultimodalAttention(config)

        # Complexity assessment layer
        self.complexity_assessment = nn.Linear(self.hidden_size, 1, bias=False)

        # Gating mechanism for selecting attention type
        self.gate = nn.Linear(
            self.hidden_size, 2, bias=False
        )  # For simple/complex selection

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for adaptive attention."""
        # Initialize complexity assessment
        std = self.hidden_size**-0.5
        nn.init.normal_(self.complexity_assessment.weight, mean=0.0, std=std)
        nn.init.normal_(self.gate.weight, mean=0.0, std=std)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for Qwen3-VL-2B adaptive multimodal attention.

        Args:
            hidden_states: Input hidden states of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position IDs for rotary embeddings
            past_key_value: Past key-value states for caching
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache

        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        bsz, seq_len, _ = hidden_states.shape

        # Assess input complexity
        complexity_score = torch.sigmoid(
            self.complexity_assessment(hidden_states.mean(dim=1, keepdim=True))
        ).squeeze(-1)

        # Get gate values for attention selection
        gate_values = torch.softmax(
            self.gate(hidden_states.mean(dim=1, keepdim=True)), dim=-1
        ).squeeze(1)

        # Use different attention mechanisms based on complexity
        if complexity_score.mean() < 0.5:  # Simple input
            output, attn_weights, pkv = self.simple_attention(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        else:  # Complex input
            output, attn_weights, pkv = self.complex_attention(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )

        # Apply gating to blend outputs if needed
        # For now, we'll just return the selected attention output
        # In a more complex implementation, we could blend outputs based on gate values

        return output, attn_weights, pkv


def create_qwen3_vl_multimodal_attention(config):
    """
    Factory function to create Qwen3-VL-2B multimodal attention.

    Args:
        config: Model configuration

    Returns:
        Qwen3VL2BMultimodalAttention instance
    """
    return Qwen3VL2BMultimodalAttention(config)


def create_qwen3_vl_modality_specific_attention(config, modality: str = "text"):
    """
    Factory function to create Qwen3-VL-2B modality-specific attention.

    Args:
        config: Model configuration
        modality: Modality for the attention ("text", "image", etc.)

    Returns:
        Qwen3VL2BModalitySpecificAttention instance
    """
    return Qwen3VL2BModalitySpecificAttention(config, modality)


def create_qwen3_vl_multimodal_fusion_layer(config):
    """
    Factory function to create Qwen3-VL-2B multimodal fusion layer.

    Args:
        config: Model configuration

    Returns:
        Qwen3VL2BMultimodalFusionLayer instance
    """
    return Qwen3VL2BMultimodalFusionLayer(config)


def create_qwen3_vl_adaptive_multimodal_attention(config):
    """
    Factory function to create Qwen3-VL-2B adaptive multimodal attention.

    Args:
        config: Model configuration

    Returns:
        Qwen3VL2BAdaptiveMultimodalAttention instance
    """
    return Qwen3VL2BAdaptiveMultimodalAttention(config)


__all__ = [
    "Qwen3VL2BMultimodalAttention",
    "Qwen3VL2BModalitySpecificAttention",
    "Qwen3VL2BMultimodalFusionLayer",
    "Qwen3VL2BAdaptiveMultimodalAttention",
    "create_qwen3_vl_multimodal_attention",
    "create_qwen3_vl_modality_specific_attention",
    "create_qwen3_vl_multimodal_fusion_layer",
    "create_qwen3_vl_adaptive_multimodal_attention",
]
