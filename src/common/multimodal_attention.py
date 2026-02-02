"""
Multimodal Attention Implementation for Inference-PIO System

This module provides implementations of multimodal attention mechanisms for the Inference-PIO system.
These attention mechanisms are designed to efficiently process inputs from multiple modalities
(text, image, audio) with optimized memory usage and performance.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientMultimodalCrossAttention(nn.Module):
    """
    Efficient cross-modal attention mechanism that allows interaction between different modalities.
    Supports attention between text, image, and audio modalities with computational optimizations.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        modalities: List[str] = ["text", "image", "audio"],
        dropout: float = 0.0,
        bias: bool = True,
        is_causal: bool = True,
        use_flash_attention: bool = True,
        use_sparse_attention: bool = False,
        sparse_topk: int = 32,
    ):
        """
        Initialize the efficient multimodal cross-attention module.

        Args:
            d_model: Total model dimension
            nhead: Number of attention heads
            modalities: List of modalities to support
            dropout: Dropout rate
            bias: Whether to use bias in projections
            is_causal: Whether to apply causal masking
            use_flash_attention: Whether to use flash attention for efficiency
            use_sparse_attention: Whether to use sparse attention for efficiency
            sparse_topk: Top-k elements to keep in sparse attention
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.modalities = modalities
        self.head_dim = d_model // nhead
        self.use_flash_attention = use_flash_attention
        self.use_sparse_attention = use_sparse_attention
        self.sparse_topk = sparse_topk

        if self.head_dim * nhead != d_model:
            raise ValueError(
                f"d_model must be divisible by nhead (got d_model: {d_model}, nhead: {nhead})"
            )

        self.scaling = self.head_dim**-0.5
        self.dropout = dropout
        self.is_causal = is_causal

        # Create projections for each modality
        self.modality_projections = nn.ModuleDict()
        for modality in modalities:
            self.modality_projections[modality] = nn.ModuleDict(
                {
                    "q": nn.Linear(d_model, d_model, bias=bias),
                    "k": nn.Linear(d_model, d_model, bias=bias),
                    "v": nn.Linear(d_model, d_model, bias=bias),
                }
            )

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout_module = nn.Dropout(dropout) if dropout > 0.0 else None

        # Modality-specific layer norms
        self.modality_norms = nn.ModuleDict(
            {modality: nn.LayerNorm(d_model) for modality in modalities}
        )

    def forward(
        self,
        queries: Dict[str, torch.Tensor],
        keys: Dict[str, torch.Tensor],
        values: Dict[str, torch.Tensor],
        attention_masks: Optional[Dict[str, torch.Tensor]] = None,
        need_weights: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass for efficient multimodal cross-attention.

        Args:
            queries: Dictionary mapping modality names to query tensors
            keys: Dictionary mapping modality names to key tensors
            values: Dictionary mapping modality names to value tensors
            attention_masks: Optional dictionary of attention masks for each modality
            need_weights: Whether to return attention weights

        Returns:
            Tuple of (outputs, attention_weights)
        """
        outputs = {}
        attention_weights = {} if need_weights else None

        # Process each modality as query
        for query_modality, query in queries.items():
            # Normalize query
            query = self.modality_norms[query_modality](query)

            # Project query
            q = self.modality_projections[query_modality]["q"](query)
            q = q.view(
                query.size(0), query.size(1), self.nhead, self.head_dim
            ).transpose(
                1, 2
            )  # (bsz, nhead, seq_len, head_dim)
            q = q * self.scaling

            # Concatenate all keys and values from all modalities
            all_keys = []
            all_values = []

            for key_modality, key in keys.items():
                # Normalize key
                key = self.modality_norms[key_modality](key)

                # Project key
                k = self.modality_projections[key_modality]["k"](key)
                k = k.view(
                    key.size(0), key.size(1), self.nhead, self.head_dim
                ).transpose(
                    1, 2
                )  # (bsz, nhead, seq_len, head_dim)
                all_keys.append(k)

                # Project value
                v = self.modality_projections[key_modality]["v"](key)
                v = v.view(
                    key.size(0), key.size(1), self.nhead, self.head_dim
                ).transpose(
                    1, 2
                )  # (bsz, nhead, seq_len, head_dim)
                all_values.append(v)

            # Concatenate keys and values across modalities
            concat_k = torch.cat(
                all_keys, dim=2
            )  # (bsz, nhead, total_seq_len, head_dim)
            concat_v = torch.cat(
                all_values, dim=2
            )  # (bsz, nhead, total_seq_len, head_dim)

            # Compute attention scores
            if self.use_flash_attention and torch.cuda.is_available():
                # Use efficient attention computation
                attn_weights = torch.matmul(
                    q, concat_k.transpose(-1, -2)
                )  # (bsz, nhead, query_seq_len, total_key_seq_len)

                # Apply attention mask if provided
                if attention_masks is not None and query_modality in attention_masks:
                    mask = attention_masks[query_modality]
                    # Expand mask to match attention weights shape
                    if (
                        mask.dim() == 2
                    ):  # (seq_len, seq_len) -> (bsz, nhead, seq_len, total_seq_len)
                        mask = (
                            mask.unsqueeze(0)
                            .unsqueeze(0)
                            .expand(-1, self.nhead, -1, -1)
                        )
                    elif (
                        mask.dim() == 3
                    ):  # (bsz, seq_len, seq_len) -> (bsz, nhead, seq_len, total_seq_len)
                        mask = mask.unsqueeze(1).expand(-1, self.nhead, -1, -1)

                    attn_weights = attn_weights + mask

                # Apply softmax
                attn_weights = torch.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(query.dtype)

                # Apply sparse attention if enabled
                if self.use_sparse_attention:
                    attn_weights = self._apply_sparse_attention(attn_weights)

                # Apply dropout if configured
                if self.dropout_module is not None:
                    attn_weights = self.dropout_module(attn_weights)

                # Apply attention to values
                attn_output = torch.matmul(
                    attn_weights, concat_v
                )  # (bsz, nhead, query_seq_len, head_dim)
            else:
                attn_weights = torch.matmul(
                    q, concat_k.transpose(-1, -2)
                )  # (bsz, nhead, query_seq_len, total_key_seq_len)

                # Apply attention mask if provided
                if attention_masks is not None and query_modality in attention_masks:
                    mask = attention_masks[query_modality]
                    # Expand mask to match attention weights shape
                    if (
                        mask.dim() == 2
                    ):  # (seq_len, seq_len) -> (bsz, nhead, seq_len, total_seq_len)
                        mask = (
                            mask.unsqueeze(0)
                            .unsqueeze(0)
                            .expand(-1, self.nhead, -1, -1)
                        )
                    elif (
                        mask.dim() == 3
                    ):  # (bsz, seq_len, seq_len) -> (bsz, nhead, seq_len, total_seq_len)
                        mask = mask.unsqueeze(1).expand(-1, self.nhead, -1, -1)

                    attn_weights = attn_weights + mask

                # Apply causal mask if needed
                if self.is_causal:
                    causal_mask = torch.triu(
                        torch.ones(
                            query.size(1),
                            concat_k.size(2),
                            dtype=torch.bool,
                            device=query.device,
                        ),
                        diagonal=1,
                    )
                    attn_weights.masked_fill_(causal_mask, float("-inf"))

                # Apply softmax
                attn_weights = torch.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(query.dtype)

                # Apply sparse attention if enabled
                if self.use_sparse_attention:
                    attn_weights = self._apply_sparse_attention(attn_weights)

                # Apply dropout if configured
                if self.dropout_module is not None:
                    attn_weights = self.dropout_module(attn_weights)

                # Apply attention to values
                attn_output = torch.matmul(
                    attn_weights, concat_v
                )  # (bsz, nhead, query_seq_len, head_dim)

            # Reshape to combine heads
            attn_output = (
                attn_output.transpose(1, 2)
                .contiguous()
                .view(query.size(0), query.size(1), self.d_model)
            )

            # Apply output projection
            attn_output = self.out_proj(attn_output)

            # Store output for this modality
            outputs[query_modality] = attn_output

            # Store attention weights if needed
            if need_weights:
                attention_weights[query_modality] = attn_weights

        return outputs, attention_weights

    def _apply_sparse_attention(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Apply sparse attention by keeping only top-k values.
        """
        # Get top-k values along the last dimension
        topk_values, topk_indices = torch.topk(
            attn_weights, k=min(self.sparse_topk, attn_weights.size(-1)), dim=-1
        )

        # Create a sparse attention matrix with only top-k values
        sparse_attn = torch.zeros_like(attn_weights)
        sparse_attn.scatter_(-1, topk_indices, topk_values)

        # Renormalize the sparse attention weights
        sparse_attn = torch.softmax(sparse_attn, dim=-1, dtype=torch.float32).to(
            attn_weights.dtype
        )

        return sparse_attn


class ModalitySpecificAttention(nn.Module):
    """
    Attention mechanism tailored for specific modalities (text, image, audio).
    Each modality has its own specialized attention computation.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        modality: str,
        dropout: float = 0.1,
        bias: bool = True,
        is_causal: bool = True,
    ):
        """
        Initialize modality-specific attention.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            modality: Modality type ('text', 'image', or 'audio')
            dropout: Dropout rate
            bias: Whether to use bias in projections
            is_causal: Whether to apply causal masking
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.modality = modality
        self.head_dim = d_model // nhead
        self.is_causal = is_causal

        if self.head_dim * nhead != d_model:
            raise ValueError(
                f"d_model must be divisible by nhead (got d_model: {d_model}, nhead: {nhead})"
            )

        self.scaling = self.head_dim**-0.5
        self.dropout = dropout

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout_module = nn.Dropout(dropout) if dropout > 0.0 else None

        # Modality-specific processing
        if modality == "image":
            # For images, we might want to add spatial awareness
            self.spatial_conv = nn.Conv2d(
                in_channels=nhead,
                out_channels=nhead,
                kernel_size=3,
                padding=1,
                groups=nhead,
            )
        elif modality == "audio":
            # For audio, we might want to add temporal awareness
            self.temporal_conv = nn.Conv1d(
                in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1
            )

        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for modality-specific attention.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attention_mask: Optional attention mask
            need_weights: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
        """
        # Apply layer norm
        query = self.layer_norm(query)
        key = self.layer_norm(key)
        value = self.layer_norm(value)

        # Apply projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)

        q = q.view(bsz, tgt_len, self.nhead, self.head_dim).transpose(
            1, 2
        )  # (bsz, nhead, tgt_len, head_dim)
        k = k.view(bsz, src_len, self.nhead, self.head_dim).transpose(
            1, 2
        )  # (bsz, nhead, src_len, head_dim)
        v = v.view(bsz, src_len, self.nhead, self.head_dim).transpose(
            1, 2
        )  # (bsz, nhead, src_len, head_dim)

        # Scale query
        q = q * self.scaling

        # Compute attention scores
        attn_weights = torch.matmul(
            q, k.transpose(-2, -1)
        )  # (bsz, nhead, tgt_len, src_len)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply causal mask if needed
        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(tgt_len, src_len, dtype=torch.bool, device=query.device),
                diagonal=1,
            )
            attn_weights.masked_fill_(causal_mask, float("-inf"))

        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )

        # Apply modality-specific processing
        if self.modality == "image":
            # For images, we might want to add spatial awareness
            # This is a simplified approach - in practice, you'd need to reshape appropriately
            pass
        elif self.modality == "audio":
            # For audio, we'll skip the convolution for now to avoid dimensional issues
            # The temporal_conv layer expects different channel dimensions than what we have
            # This is a placeholder for future implementation
            pass

        # Apply dropout if configured
        if self.dropout_module is not None:
            attn_weights = self.dropout_module(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (bsz, nhead, tgt_len, head_dim)

        # Reshape to combine heads
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        )

        # Apply output projection
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights if need_weights else None


class MultimodalFusionLayer(nn.Module):
    """
    Fuses information from multiple modalities using cross-attention mechanisms.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        modalities: List[str] = ["text", "image", "audio"],
        dropout: float = 0.1,
        activation: str = "relu",
        use_alignment: bool = True,
        alignment_method: str = "learned_projection",
    ):
        """
        Initialize the multimodal fusion layer.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            modalities: List of modalities to support
            dropout: Dropout rate
            activation: Activation function to use
            use_alignment: Whether to use alignment module
            alignment_method: Method for aligning modalities
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.modalities = modalities

        # Alignment module
        self.use_alignment = use_alignment
        if use_alignment:
            self.alignment_module = MultimodalAlignmentModule(
                d_model=d_model,
                modalities=modalities,
                alignment_method=alignment_method,
            )

        # Cross-attention module
        self.cross_attention = EfficientMultimodalCrossAttention(
            d_model=d_model, nhead=nhead, modalities=modalities, dropout=dropout
        )

        # Feed-forward networks for each modality
        self.ffn = nn.ModuleDict()
        for modality in modalities:
            self.ffn[modality] = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                self._get_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )

        # Layer norms
        self.norms = nn.ModuleDict(
            {modality: nn.LayerNorm(d_model) for modality in modalities}
        )

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        return activations.get(activation, nn.ReLU())

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        attention_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multimodal fusion.

        Args:
            inputs: Dictionary mapping modality names to input tensors
            attention_masks: Optional dictionary of attention masks for each modality

        Returns:
            Dictionary mapping modality names to output tensors
        """
        # Align modalities if enabled
        if self.use_alignment:
            aligned_inputs = self.alignment_module(inputs)
        else:
            aligned_inputs = inputs

        # Apply cross-attention
        attended_outputs, _ = self.cross_attention(
            queries=aligned_inputs,
            keys=aligned_inputs,
            values=aligned_inputs,
            attention_masks=attention_masks,
            need_weights=False,
        )

        # Apply residual connection and layer norm
        normalized_outputs = {}
        for modality in self.modalities:
            if modality in attended_outputs:
                # Add residual connection
                residual = inputs[modality] + attended_outputs[modality]
                # Apply layer norm
                normalized = self.norms[modality](residual)
                # Apply feed-forward network
                output = self.ffn[modality](normalized)
                # Add residual connection again
                normalized_outputs[modality] = residual + output
            else:
                # If modality wasn't processed, pass through unchanged
                normalized_outputs[modality] = inputs[modality]

        return normalized_outputs


class AdaptiveMultimodalAttention(nn.Module):
    """
    Adaptive attention mechanism that adjusts its behavior based on input characteristics.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        modalities: List[str] = ["text", "image", "audio"],
        dropout: float = 0.1,
        bias: bool = True,
        is_causal: bool = True,
        use_efficient_attention: bool = True,
        adaptive_strategy: str = "input_dependent",
    ):
        """
        Initialize adaptive multimodal attention.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            modalities: List of modalities to support
            dropout: Dropout rate
            bias: Whether to use bias in projections
            is_causal: Whether to apply causal masking
            use_efficient_attention: Whether to use efficient attention mechanisms
            adaptive_strategy: Strategy for adapting attention ('input_dependent', 'dynamic', 'static')
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.modalities = modalities
        self.adaptive_strategy = adaptive_strategy
        self.is_causal = is_causal

        # Base attention mechanism
        self.base_attention = EfficientMultimodalCrossAttention(
            d_model=d_model,
            nhead=nhead,
            modalities=modalities,
            dropout=dropout,
            bias=bias,
            is_causal=is_causal,
            use_flash_attention=use_efficient_attention,
            use_sparse_attention=use_efficient_attention,
        )

        # Adaptive parameters
        if adaptive_strategy == "input_dependent":
            # Create a small network to predict attention parameters based on input
            self.adaptation_network = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.ReLU(),
                nn.Linear(
                    d_model // 4, 4
                ),  # Predict 4 parameters: sparsity, temperature, dropout, window_size
                nn.Softplus(),  # Ensure positive values
            )

    def forward(
        self,
        queries: Dict[str, torch.Tensor],
        keys: Dict[str, torch.Tensor],
        values: Dict[str, torch.Tensor],
        attention_masks: Optional[Dict[str, torch.Tensor]] = None,
        need_weights: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass with adaptive attention.

        Args:
            queries: Dictionary mapping modality names to query tensors
            keys: Dictionary mapping modality names to key tensors
            values: Dictionary mapping modality names to value tensors
            attention_masks: Optional dictionary of attention masks for each modality
            need_weights: Whether to return attention weights

        Returns:
            Tuple of (outputs, attention_weights)
        """
        # Get base attention outputs
        outputs, attention_weights = self.base_attention(
            queries=queries,
            keys=keys,
            values=values,
            attention_masks=attention_masks,
            need_weights=need_weights,
        )

        # Apply adaptive modifications if needed
        if self.adaptive_strategy == "input_dependent":
            # Use the first modality as a representative for adaptive parameters
            representative_input = next(iter(queries.values()))

            # Predict adaptive parameters
            adaptive_params = self.adaptation_network(
                torch.mean(
                    representative_input, dim=1
                )  # Average across sequence dimension
            )  # Shape: (batch_size, 4)

            # Apply adaptive modifications to attention weights
            if attention_weights is not None:
                for modality, weights in attention_weights.items():
                    # Apply temperature scaling based on predicted parameter
                    temp = (
                        adaptive_params[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                        + 1e-8
                    )  # Prevent division by zero
                    attention_weights[modality] = weights / temp

        return outputs, attention_weights


class MultimodalAlignmentModule(nn.Module):
    """
    Module for aligning representations from different modalities.
    Uses learned transformations to map different modalities to a shared space.
    """

    def __init__(
        self,
        d_model: int,
        modalities: List[str] = ["text", "image", "audio"],
        alignment_method: str = "learned_projection",
    ):
        """
        Initialize the multimodal alignment module.

        Args:
            d_model: Model dimension
            modalities: List of modalities to align
            alignment_method: Method for alignment ('learned_projection', 'cross_attention', 'contrastive')
        """
        super().__init__()

        self.d_model = d_model
        self.modalities = modalities
        self.alignment_method = alignment_method

        if alignment_method == "learned_projection":
            # Create learned projection layers for each modality to a shared space
            self.projection_layers = nn.ModuleDict()
            for modality in modalities:
                self.projection_layers[modality] = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, d_model),
                    nn.LayerNorm(d_model),
                )
        elif alignment_method == "cross_attention":
            # Use cross-attention for alignment
            self.alignment_attention = EfficientMultimodalCrossAttention(
                d_model=d_model,
                nhead=8,  # Fixed number of heads for alignment
                modalities=modalities,
                dropout=0.1,
            )
        elif alignment_method == "contrastive":
            # Use contrastive learning approach
            self.temperature = nn.Parameter(torch.tensor(0.07))  # Learnable temperature

    def forward(
        self, modalities_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Align modalities to a shared representation space.

        Args:
            modalities_dict: Dictionary mapping modality names to their representations

        Returns:
            Dictionary mapping modality names to aligned representations
        """
        aligned_outputs = {}

        if self.alignment_method == "learned_projection":
            # Apply learned projections to align modalities
            for modality, representation in modalities_dict.items():
                aligned_outputs[modality] = self.projection_layers[modality](
                    representation
                )

        elif self.alignment_method == "cross_attention":
            # Use cross-attention for alignment
            aligned_outputs, _ = self.alignment_attention(
                queries=modalities_dict,
                keys=modalities_dict,
                values=modalities_dict,
                need_weights=False,
            )

        elif self.alignment_method == "contrastive":
            # For contrastive alignment, we return the same representations
            # but the loss function would encourage alignment
            aligned_outputs = modalities_dict.copy()

        return aligned_outputs


def create_multimodal_attention(
    config: Any, layer_idx: Optional[int] = None
) -> EfficientMultimodalCrossAttention:
    """
    Factory function to create multimodal attention implementation.

    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer (optional)

    Returns:
        EfficientMultimodalCrossAttention: The multimodal attention implementation
    """
    return EfficientMultimodalCrossAttention(
        d_model=config.hidden_size,
        nhead=config.num_attention_heads,
        modalities=getattr(config, "modalities", ["text", "image"]),
        dropout=getattr(config, "attention_dropout_prob", 0.0),
        bias=not getattr(config, "remove_bias_in_attention", False),
        is_causal=getattr(config, "is_causal", True),
        use_flash_attention=getattr(config, "use_flash_attention_2", True),
        use_sparse_attention=getattr(config, "use_sparse_attention", False),
        sparse_topk=getattr(config, "sparse_attention_topk", 32),
    )


__all__ = [
    "EfficientMultimodalCrossAttention",
    "ModalitySpecificAttention",
    "MultimodalFusionLayer",
    "AdaptiveMultimodalAttention",
    "MultimodalAlignmentModule",
    "create_multimodal_attention",
]
