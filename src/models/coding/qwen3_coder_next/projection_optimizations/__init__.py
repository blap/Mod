"""
Qwen3-Coder-Next Projection Optimization Module

This module implements optimized projection layers specifically for the Qwen3-Coder-Next model.
These optimizations focus on efficient embedding projections with advanced techniques for code generation,
including next-generation architectures and adaptive mechanisms.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any


class Qwen3CoderNextProjectionLayer(nn.Module):
    """
    Optimized projection layer for Qwen3-Coder-Next model with advanced embedding transformations.
    Features next-generation optimizations for code generation tasks.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_bias: bool = True,
        dropout_prob: float = 0.1,
        activation: str = "silu",
        use_residual: bool = True,
        use_low_rank: bool = False,
        low_rank_dim: Optional[int] = None,
        use_group_norm: bool = False,
        group_norm_num_groups: int = 32,
        intermediate_dim: Optional[int] = None,
        use_cross_attention: bool = False,
        num_attention_heads: int = 8,
        use_adaptive_computation: bool = True,
        use_mixed_precision: bool = True,
        use_sparse_attention: bool = True,
        sparse_attention_density: float = 0.5,
        use_dynamic_routing: bool = True,
        num_experts: int = 4,
        expert_capacity: int = 1024,
        use_code_specific_optimizations: bool = True,
        use_syntax_aware_projections: bool = True
    ):
        """
        Initialize the Qwen3-Coder-Next projection layer.
        
        Args:
            input_dim: Input dimension of the embeddings
            output_dim: Output dimension of the embeddings
            use_bias: Whether to use bias in the projection
            dropout_prob: Dropout probability for regularization
            activation: Activation function to use ('silu', 'gelu', 'relu', etc.)
            use_residual: Whether to use residual connections
            use_low_rank: Whether to use low-rank projection for memory efficiency
            low_rank_dim: Dimension for low-rank projection (only used if use_low_rank is True)
            use_group_norm: Whether to use group normalization
            group_norm_num_groups: Number of groups for group normalization
            intermediate_dim: Intermediate dimension for the projection (defaults to average of input_dim and output_dim)
            use_cross_attention: Whether to include cross-attention mechanism
            num_attention_heads: Number of attention heads for cross-attention
            use_adaptive_computation: Whether to use adaptive computation allocation
            use_mixed_precision: Whether to use mixed precision for efficiency
            use_sparse_attention: Whether to use sparse attention mechanisms
            sparse_attention_density: Density of sparse attention (fraction of full attention)
            use_dynamic_routing: Whether to use dynamic routing between experts
            num_experts: Number of experts in the mixture-of-experts layer
            expert_capacity: Maximum capacity per expert
            use_code_specific_optimizations: Whether to enable code-specific optimizations
            use_syntax_aware_projections: Whether to use syntax-aware projections
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.dropout_prob = dropout_prob
        self.use_residual = use_residual
        self.use_low_rank = use_low_rank
        self.low_rank_dim = low_rank_dim or min(input_dim, output_dim) // 4
        self.use_group_norm = use_group_norm
        self.use_adaptive_computation = use_adaptive_computation
        self.use_mixed_precision = use_mixed_precision
        self.use_sparse_attention = use_sparse_attention
        self.sparse_attention_density = sparse_attention_density
        self.use_dynamic_routing = use_dynamic_routing
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.use_code_specific_optimizations = use_code_specific_optimizations
        self.use_syntax_aware_projections = use_syntax_aware_projections
        self.intermediate_dim = intermediate_dim or (input_dim + output_dim) // 2
        
        # Select activation function
        if activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        # Create projection layers
        if use_low_rank:
            # Low-rank projection for memory efficiency
            self.input_projection = nn.Linear(input_dim, self.low_rank_dim, bias=use_bias)
            self.intermediate_projection = nn.Linear(self.low_rank_dim, self.intermediate_dim, bias=use_bias)
            self.output_projection = nn.Linear(self.intermediate_dim, output_dim, bias=use_bias)
        else:
            # Standard projection layers
            self.input_projection = nn.Linear(input_dim, self.intermediate_dim, bias=use_bias)
            self.output_projection = nn.Linear(self.intermediate_dim, output_dim, bias=use_bias)
        
        # Optional group normalization
        if use_group_norm:
            self.group_norm = nn.GroupNorm(group_norm_num_groups, self.intermediate_dim, eps=1e-6)
        else:
            self.group_norm = None
            
        # Optional cross-attention mechanism
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.intermediate_dim,
                num_heads=num_attention_heads,
                dropout=dropout_prob,
                batch_first=True
            )
            
        # Adaptive computation layer
        if use_adaptive_computation:
            self.computation_gate = nn.Linear(self.intermediate_dim, 1)
            self.computation_activation = nn.Sigmoid()
        
        # Mixture of Experts layer
        if use_dynamic_routing:
            self.moe_router = nn.Linear(self.intermediate_dim, num_experts)
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.intermediate_dim, self.intermediate_dim * 2),
                    self.activation,
                    nn.Linear(self.intermediate_dim * 2, self.intermediate_dim)
                ) for _ in range(num_experts)
            ])
        
        # Code-specific optimizations
        if use_code_specific_optimizations:
            # Additional layers for code-specific processing
            self.code_context_enhancement = nn.Linear(self.intermediate_dim, self.intermediate_dim, bias=use_bias)
            self.code_pattern_extractor = nn.Conv1d(
                in_channels=self.intermediate_dim,
                out_channels=self.intermediate_dim,
                kernel_size=3,
                padding=1,
                groups=self.intermediate_dim // 8  # Group convolution for efficiency
            )
        
        # Syntax-aware projection layer
        if use_syntax_aware_projections:
            self.syntax_embedding = nn.Embedding(512, self.intermediate_dim // 4)  # Assuming 512 syntax types
            self.syntax_projection = nn.Linear(
                self.intermediate_dim + self.intermediate_dim // 4,  # Combined with syntax info
                self.intermediate_dim,
                bias=use_bias
            )
        
        # Sparse attention mechanism
        if use_sparse_attention:
            from torch.nn.functional import scaled_dot_product_attention
            self.scaled_dot_product_attention = scaled_dot_product_attention
            self.sparse_attention_probs = None  # Will be computed dynamically
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the projection layers."""
        # Initialize input projection weights
        std = self.input_projection.in_features ** -0.5
        nn.init.normal_(self.input_projection.weight, mean=0.0, std=std)
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)
            
        # Initialize intermediate projection weights if using low-rank
        if self.use_low_rank:
            std = self.intermediate_projection.in_features ** -0.5
            nn.init.normal_(self.intermediate_projection.weight, mean=0.0, std=std)
            if self.intermediate_projection.bias is not None:
                nn.init.zeros_(self.intermediate_projection.bias)
                
        # Initialize output projection weights
        std = self.output_projection.in_features ** -0.5
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=std)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)
            
        # Initialize adaptive computation layers if enabled
        if self.use_adaptive_computation:
            nn.init.normal_(self.computation_gate.weight, mean=0.0, std=0.02)
            if self.computation_gate.bias is not None:
                nn.init.zeros_(self.computation_gate.bias)
        
        # Initialize MoE router if enabled
        if self.use_dynamic_routing:
            nn.init.normal_(self.moe_router.weight, mean=0.0, std=0.02)
            if self.moe_router.bias is not None:
                nn.init.zeros_(self.moe_router.bias)
            
            # Initialize experts
            for expert in self.experts:
                for layer in expert:
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
        
        # Initialize code-specific layers if enabled
        if self.use_code_specific_optimizations:
            std = self.code_context_enhancement.in_features ** -0.5
            nn.init.normal_(self.code_context_enhancement.weight, mean=0.0, std=std)
            if self.code_context_enhancement.bias is not None:
                nn.init.zeros_(self.code_context_enhancement.bias)
                
            nn.init.xavier_uniform_(self.code_pattern_extractor.weight)
        
        # Initialize syntax-aware layers if enabled
        if self.use_syntax_aware_projections:
            nn.init.normal_(self.syntax_embedding.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.syntax_projection.weight, mean=0.0, std=0.02)
            if self.syntax_projection.bias is not None:
                nn.init.zeros_(self.syntax_projection.bias)
    
    def forward(self, x: torch.Tensor, syntax_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for Qwen3-Coder-Next projection layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            syntax_ids: Optional syntax IDs for syntax-aware projections
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply input projection
        projected = self.input_projection(x)
        
        # Apply group norm if enabled
        if self.use_group_norm:
            # Reshape for group norm: (batch_size * seq_len, intermediate_dim)
            reshaped = projected.view(-1, self.intermediate_dim)
            normalized = self.group_norm(reshaped.unsqueeze(-1)).squeeze(-1)
            projected = normalized.view(batch_size, seq_len, self.intermediate_dim)
        else:
            projected = self.dropout(projected)
        
        # Apply activation
        activated = self.activation(projected)
        
        # Apply syntax-aware projections if enabled
        if self.use_syntax_aware_projections and syntax_ids is not None:
            # Embed syntax information
            syntax_embeds = self.syntax_embedding(syntax_ids)  # (batch_size, seq_len, syntax_dim)
            # Concatenate with activated features
            combined_features = torch.cat([activated, syntax_embeds], dim=-1)
            # Apply syntax-aware projection
            activated = self.syntax_projection(combined_features)
        
        # Apply cross-attention if enabled
        if self.use_cross_attention:
            # Self-attention on the sequence dimension
            attended, _ = self.cross_attn(activated, activated, activated)
            activated = activated + attended  # Residual connection
        
        # Apply sparse attention if enabled
        if self.use_sparse_attention:
            # Compute sparse attention mask based on density
            sparse_mask = self._compute_sparse_attention_mask(activated, self.sparse_attention_density)
            # Apply sparse attention (simplified implementation)
            if sparse_mask is not None:
                # Use standard attention with sparse mask
                attended_sparse, _ = self.cross_attn(activated, activated, activated, attn_mask=sparse_mask)
                activated = activated + attended_sparse  # Residual connection
        
        # Apply code-specific optimizations if enabled
        if self.use_code_specific_optimizations:
            # Enhance with code context
            enhanced = self.code_context_enhancement(activated)
            
            # Apply pattern extraction with conv layer
            # Transpose for conv: (batch, channels, seq_len)
            conv_input = enhanced.transpose(1, 2)
            patterns = self.code_pattern_extractor(conv_input)
            # Transpose back: (batch, seq_len, channels)
            activated = patterns.transpose(1, 2) + enhanced  # Residual connection
        
        # Apply Mixture of Experts if enabled
        if self.use_dynamic_routing:
            # Compute routing weights
            routing_weights = torch.softmax(self.moe_router(activated), dim=-1)  # (batch, seq_len, num_experts)
            
            # Apply top-k routing (top-2 for example)
            top_k = min(2, self.num_experts)
            top_weights, top_indices = torch.topk(routing_weights, k=top_k, dim=-1)  # (batch, seq_len, top_k)
            
            # Normalize top weights
            top_weights = torch.softmax(top_weights, dim=-1)
            
            # Process through selected experts
            expert_outputs = []
            for i in range(top_k):
                # Get indices for current expert
                current_indices = top_indices[:, :, i]  # (batch, seq_len)
                
                # Prepare inputs for each expert
                expert_input = activated
                expert_output = self.experts[current_indices[0, 0]](expert_input)  # Simplified for demonstration
                
                # Weight by routing probability
                weighted_output = expert_output * top_weights[:, :, i:i+1]
                expert_outputs.append(weighted_output)
            
            # Combine outputs
            activated = sum(expert_outputs)
        
        # Apply adaptive computation if enabled
        if self.use_adaptive_computation:
            # Compute computation gate values
            comp_gate = self.computation_activation(self.computation_gate(activated))  # (batch, seq_len, 1)
            # Apply adaptive computation scaling
            activated = activated * comp_gate
        
        # Apply low-rank transformation if enabled
        if self.use_low_rank:
            activated = self.intermediate_projection(activated)
            activated = self.activation(activated)
            activated = self.dropout(activated)
        
        # Apply output projection
        output = self.output_projection(activated)
        
        return output
    
    def _compute_sparse_attention_mask(self, x: torch.Tensor, density: float) -> Optional[torch.Tensor]:
        """
        Compute sparse attention mask based on input features and density.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            density: Fraction of attention to compute (between 0 and 1)
            
        Returns:
            Attention mask or None if not applicable
        """
        if density >= 1.0:
            return None  # Full attention
        
        seq_len = x.size(1)
        num_keep = max(1, int(seq_len * density))
        
        # Simple random sparse attention (in practice, could use more sophisticated methods)
        mask = torch.zeros((seq_len, seq_len), device=x.device, dtype=torch.bool)
        
        for i in range(seq_len):
            # Keep random subset of positions for each query position
            keep_indices = torch.randperm(seq_len, device=x.device)[:num_keep]
            mask[i, keep_indices] = True
        
        # Convert to attention bias (large negative values for masked positions)
        attention_bias = torch.zeros_like(mask, dtype=x.dtype)
        attention_bias.masked_fill_(~mask, float('-inf'))
        
        return attention_bias


class Qwen3CoderNextMultiProjectionLayer(nn.Module):
    """
    Qwen3-Coder-Next specific multimodal projector with multiple projection layers.
    
    This projector combines multiple advanced projection techniques for enhanced performance in code tasks.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 2,
        intermediate_dim: Optional[int] = None,
        use_bias: bool = True,
        activation: str = "silu",
        dropout_prob: float = 0.1,
        use_residual: bool = True,
        use_low_rank: bool = False,
        low_rank_dim: Optional[int] = None,
        use_group_norm: bool = False,
        use_cross_attention: bool = False,
        num_attention_heads: int = 8,
        use_adaptive_computation: bool = True,
        use_mixed_precision: bool = True,
        use_sparse_attention: bool = True,
        sparse_attention_density: float = 0.5,
        use_dynamic_routing: bool = True,
        num_experts: int = 4,
        expert_capacity: int = 1024,
        use_code_specific_optimizations: bool = True,
        use_syntax_aware_projections: bool = True
    ):
        """
        Initialize the Qwen3-Coder-Next multi-projection layer.
        
        Args:
            input_dim: Input dimension of the embeddings
            output_dim: Output dimension of the embeddings
            num_layers: Number of projection layers
            intermediate_dim: Intermediate dimension for the projection
            use_bias: Whether to use bias in the projection
            activation: Activation function to use
            dropout_prob: Dropout probability for regularization
            use_residual: Whether to use residual connections
            use_low_rank: Whether to use low-rank projection for memory efficiency
            low_rank_dim: Dimension for low-rank projection
            use_group_norm: Whether to use group normalization
            use_cross_attention: Whether to include cross-attention mechanism
            num_attention_heads: Number of attention heads for cross-attention
            use_adaptive_computation: Whether to use adaptive computation allocation
            use_mixed_precision: Whether to use mixed precision for efficiency
            use_sparse_attention: Whether to use sparse attention mechanisms
            sparse_attention_density: Density of sparse attention (fraction of full attention)
            use_dynamic_routing: Whether to use dynamic routing between experts
            num_experts: Number of experts in the mixture-of-experts layer
            expert_capacity: Maximum capacity per expert
            use_code_specific_optimizations: Whether to enable code-specific optimizations
            use_syntax_aware_projections: Whether to use syntax-aware projections
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Create projection layers
        self.projection_layers = nn.ModuleList([
            Qwen3CoderNextProjectionLayer(
                input_dim if i == 0 else output_dim,
                output_dim,
                use_bias=use_bias,
                activation=activation,
                dropout_prob=dropout_prob,
                use_residual=use_residual,
                use_low_rank=use_low_rank,
                low_rank_dim=low_rank_dim,
                use_group_norm=use_group_norm,
                use_cross_attention=use_cross_attention,
                num_attention_heads=num_attention_heads,
                use_adaptive_computation=use_adaptive_computation,
                use_mixed_precision=use_mixed_precision,
                use_sparse_attention=use_sparse_attention,
                sparse_attention_density=sparse_attention_density,
                use_dynamic_routing=use_dynamic_routing,
                num_experts=num_experts,
                expert_capacity=expert_capacity,
                use_code_specific_optimizations=use_code_specific_optimizations,
                use_syntax_aware_projections=use_syntax_aware_projections
            ) for i in range(num_layers)
        ])
        
        # Final output projection
        self.final_projection = nn.Linear(output_dim, output_dim, bias=use_bias)
        
        # Initialize final projection weights
        std = self.final_projection.in_features ** -0.5
        nn.init.normal_(self.final_projection.weight, mean=0.0, std=std)
        if self.final_projection.bias is not None:
            nn.init.zeros_(self.final_projection.bias)
    
    def forward(self, x: torch.Tensor, syntax_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Qwen3-Coder-Next multi-projection layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            syntax_ids: Optional syntax IDs for syntax-aware projections
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        # Process through projection layers
        output = x
        for layer in self.projection_layers:
            layer_output = layer(output, syntax_ids)
            # Residual connection if dimensions match
            if output.shape[-1] == layer_output.shape[-1]:
                output = output + layer_output
            else:
                output = layer_output
        
        # Apply final projection
        output = self.final_projection(output)
        
        return output


def create_qwen3_coder_next_projection_layer(config: Any) -> Qwen3CoderNextProjectionLayer:
    """
    Factory function to create a Qwen3-Coder-Next projection layer.
    
    Args:
        config: Configuration object with projection settings
        
    Returns:
        Qwen3CoderNextProjectionLayer: The created projection layer
    """
    return Qwen3CoderNextProjectionLayer(
        input_dim=getattr(config, "input_dim", 4096),
        output_dim=getattr(config, "output_dim", 4096),
        use_bias=getattr(config, "use_bias_in_projection", True),
        activation=getattr(config, "projection_activation", "silu"),
        dropout_prob=getattr(config, "projection_dropout", 0.1),
        use_residual=getattr(config, "use_residual_in_projection", True),
        use_low_rank=getattr(config, "use_low_rank_projection", False),
        low_rank_dim=getattr(config, "low_rank_projection_dim", None),
        use_group_norm=getattr(config, "use_group_norm_in_projection", False),
        use_cross_attention=getattr(config, "use_cross_attention_in_projection", False),
        num_attention_heads=getattr(config, "num_projection_attention_heads", 8),
        use_adaptive_computation=getattr(config, "use_adaptive_computation", True),
        use_mixed_precision=getattr(config, "use_mixed_precision", True),
        use_sparse_attention=getattr(config, "use_sparse_attention", True),
        sparse_attention_density=getattr(config, "sparse_attention_density", 0.5),
        use_dynamic_routing=getattr(config, "use_dynamic_routing", True),
        num_experts=getattr(config, "num_experts", 4),
        expert_capacity=getattr(config, "expert_capacity", 1024),
        use_code_specific_optimizations=getattr(config, "use_code_specific_optimizations", True),
        use_syntax_aware_projections=getattr(config, "use_syntax_aware_projections", True)
    )


def create_qwen3_coder_next_multi_projection_layer(config: Any) -> Qwen3CoderNextMultiProjectionLayer:
    """
    Factory function to create a Qwen3-Coder-Next multi-projection layer.
    
    Args:
        config: Configuration object with projection settings
        
    Returns:
        Qwen3CoderNextMultiProjectionLayer: The created multi-projection layer
    """
    num_layers = getattr(config, "num_projection_layers", 2)
    
    return Qwen3CoderNextMultiProjectionLayer(
        input_dim=getattr(config, "input_dim", 4096),
        output_dim=getattr(config, "output_dim", 4096),
        num_layers=num_layers,
        use_bias=getattr(config, "use_bias_in_projection", True),
        activation=getattr(config, "projection_activation", "silu"),
        dropout_prob=getattr(config, "projection_dropout", 0.1),
        use_residual=getattr(config, "use_residual_in_projection", True),
        use_low_rank=getattr(config, "use_low_rank_projection", False),
        low_rank_dim=getattr(config, "low_rank_projection_dim", None),
        use_group_norm=getattr(config, "use_group_norm_in_projection", False),
        use_cross_attention=getattr(config, "use_cross_attention_in_projection", False),
        num_attention_heads=getattr(config, "num_projection_attention_heads", 8),
        use_adaptive_computation=getattr(config, "use_adaptive_computation", True),
        use_mixed_precision=getattr(config, "use_mixed_precision", True),
        use_sparse_attention=getattr(config, "use_sparse_attention", True),
        sparse_attention_density=getattr(config, "sparse_attention_density", 0.5),
        use_dynamic_routing=getattr(config, "use_dynamic_routing", True),
        num_experts=getattr(config, "num_experts", 4),
        expert_capacity=getattr(config, "expert_capacity", 1024),
        use_code_specific_optimizations=getattr(config, "use_code_specific_optimizations", True),
        use_syntax_aware_projections=getattr(config, "use_syntax_aware_projections", True)
    )


def apply_qwen3_coder_next_projection_optimizations(model: nn.Module, config: Any) -> nn.Module:
    """
    Apply Qwen3-Coder-Next specific projection optimizations to the model.
    
    Args:
        model: The model to optimize
        config: Configuration object with optimization settings
        
    Returns:
        Model with projection optimizations applied
    """
    from loguru import logger
    
    logger.info("Applying Qwen3-Coder-Next specific projection optimizations...")
    
    # Identify and replace projection layers in the model
    for name, module in model.named_modules():
        if "proj" in name.lower() or "projection" in name.lower() or "embed" in name.lower():
            # This is likely a projection layer, replace with Qwen3-Coder-Next optimized version
            if isinstance(module, nn.Linear):
                input_dim = module.in_features
                output_dim = module.out_features
                
                # Create new optimized projection layer
                new_projection = Qwen3CoderNextProjectionLayer(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    use_bias=getattr(config, "use_bias_in_projection", True),
                    activation=getattr(config, "projection_activation", "silu"),
                    dropout_prob=getattr(config, "projection_dropout", 0.1),
                    use_residual=getattr(config, "use_residual_in_projection", True),
                    use_low_rank=getattr(config, "use_low_rank_projection", False),
                    low_rank_dim=getattr(config, "low_rank_projection_dim", None),
                    use_group_norm=getattr(config, "use_group_norm_in_projection", False),
                    use_cross_attention=getattr(config, "use_cross_attention_in_projection", False),
                    num_attention_heads=getattr(config, "num_projection_attention_heads", 8),
                    use_adaptive_computation=getattr(config, "use_adaptive_computation", True),
                    use_mixed_precision=getattr(config, "use_mixed_precision", True),
                    use_sparse_attention=getattr(config, "use_sparse_attention", True),
                    sparse_attention_density=getattr(config, "sparse_attention_density", 0.5),
                    use_dynamic_routing=getattr(config, "use_dynamic_routing", True),
                    num_experts=getattr(config, "num_experts", 4),
                    expert_capacity=getattr(config, "expert_capacity", 1024),
                    use_code_specific_optimizations=getattr(config, "use_code_specific_optimizations", True),
                    use_syntax_aware_projections=getattr(config, "use_syntax_aware_projections", True)
                )
                
                # Copy weights from the original layer if possible
                with torch.no_grad():
                    new_projection.input_projection.weight.copy_(module.weight)
                    if module.bias is not None and new_projection.input_projection.bias is not None:
                        new_projection.input_projection.bias.copy_(module.bias)
                
                # Find parent module and replace the layer
                parent_module = model
                child_name = None
                name_parts = name.split('.')
                for part in name_parts[:-1]:
                    parent_module = getattr(parent_module, part)
                child_name = name_parts[-1]
                
                setattr(parent_module, child_name, new_projection)
                logger.info(f"Replaced projection layer '{name}' with Qwen3-Coder-Next optimized version")
    
    logger.info("Qwen3-Coder-Next projection optimizations applied successfully")
    return model


__all__ = [
    "Qwen3CoderNextProjectionLayer",
    "Qwen3CoderNextMultiProjectionLayer", 
    "create_qwen3_coder_next_projection_layer",
    "create_qwen3_coder_next_multi_projection_layer",
    "apply_qwen3_coder_next_projection_optimizations"
]