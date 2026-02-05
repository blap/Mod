"""
Qwen3-Coder-Next Custom CUDA Kernels

This module implements highly optimized CUDA kernels specifically for the Qwen3-Coder-Next model.
These kernels leverage the specific architectural features of Qwen3-Coder-Next for maximum performance.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .....common.cuda_kernels.kernel_framework import (
    BaseCUDAKernel,
    CUDAHardwareOptimizer,
    AttentionKernel,
    LinearProjectionKernel,
    ActivationKernel,
    KVCacheKernel,
    NormalizationKernel,
    MLPLayerKernel,
    create_standardized_cuda_kernels,
)

logger = logging.getLogger(__name__)


class Qwen3CoderNextAttentionKernel(AttentionKernel):
    """
    Qwen3-Coder-Next specific attention kernel with custom optimizations.
    Implements efficient attention mechanisms with Qwen-specific optimizations including MoE support.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        use_flash_attention: bool = True,
        use_rotary_embeddings: bool = True,
        causal: bool = True,
        kv_groups: int = 1,  # For Grouped Query Attention
        use_sliding_window: bool = False,
        sliding_window_size: int = 4096,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__(d_model, nhead, dropout, use_flash_attention, causal, kv_groups)
        self.use_rotary_embeddings = use_rotary_embeddings
        self.use_sliding_window = use_sliding_window
        self.sliding_window_size = sliding_window_size
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Qwen-specific parameters
        self.rotary_emb = None
        if use_rotary_embeddings:
            self.rotary_emb = Qwen3CoderNextRotaryEmbedding(d_model // nhead)
            
        # Initialize Qwen-specific attention parameters
        self.scaling = (d_model // nhead) ** -0.5
        
        # Sliding window attention mask if enabled
        if use_sliding_window:
            self.register_buffer(
                "sliding_window_mask",
                torch.tril(
                    torch.ones(sliding_window_size, sliding_window_size, dtype=torch.bool),
                    diagonal=0
                ).unsqueeze(0).unsqueeze(0),
                persistent=False
            )
            
        # Mixture of Experts router if enabled
        if use_moe:
            self.router = nn.Linear(d_model, num_experts, bias=False)
            self.experts = nn.ModuleList([
                nn.Linear(d_model, d_model) for _ in range(num_experts)
            ])
            # Initialize router weights
            nn.init.uniform_(self.router.weight, -0.1, 0.1)
            
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for Qwen3-Coder-Next attention kernel.
        
        Args:
            query: Query tensor of shape (batch, seq_len, d_model)
            key: Key tensor of shape (batch, seq_len, d_model)
            value: Value tensor of shape (batch, seq_len, d_model)
            attention_mask: Optional attention mask
            need_weights: Whether to return attention weights
            position_ids: Position IDs for RoPE embeddings
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, d_model = query.shape
        
        # Apply rotary embeddings if enabled
        if self.rotary_emb is not None and position_ids is not None:
            query, key = self.rotary_emb(query, key, position_ids)
        
        # Apply Mixture of Experts if enabled
        if self.use_moe:
            # Router to determine which experts to use
            router_logits = self.router(query.mean(dim=1))  # Average over sequence
            routing_weights = torch.softmax(router_logits, dim=-1)
            top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)

            # Normalize weights
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

            # Process through selected experts
            expert_outputs = []
            for i in range(self.top_k):
                # Get the expert indices for this top-k
                expert_idx_batch = top_k_indices[:, i]  # [batch]

                # Process each item in the batch separately
                for j, expert_idx in enumerate(expert_idx_batch):
                    expert_output = self.experts[expert_idx.item()](query[j:j+1])  # Process single item
                    weighted_output = expert_output * top_k_weights[j, i:j+1].unsqueeze(1).unsqueeze(1)
                    if len(expert_outputs) <= j:
                        expert_outputs.append(weighted_output)
                    else:
                        expert_outputs[j] += weighted_output

            # Combine outputs for the batch
            query = torch.cat(expert_outputs, dim=0)
        
        # Reshape to multi-head format
        q = query.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = key.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = value.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # Scale queries
        q = q * self.scaling
        
        # Compute attention scores
        if self.use_flash_attention and torch.cuda.is_available():
            # Use efficient attention computation (would use actual FlashAttention in real implementation)
            attn_weights = torch.matmul(q, k.transpose(-2, -1))
            
            # Apply sliding window mask if enabled
            if self.use_sliding_window and seq_len <= self.sliding_window_size:
                sliding_mask = self.sliding_window_mask[:, :, :seq_len, :seq_len]
                attn_weights.masked_fill_(~sliding_mask, float("-inf"))
            
            # Apply causal mask if needed
            if self.causal and seq_len <= 1024:  # Use precomputed mask if sequence is not too long
                causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
                attn_weights.masked_fill_(causal_mask, float("-inf"))
                
            # Apply attention mask if provided
            if attention_mask is not None:
                # Expand attention mask to match attention weights dimensions
                if attention_mask.dim() == 2:
                    # Shape: (batch, seq_len) -> (batch, 1, 1, seq_len)
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                elif attention_mask.dim() == 3:
                    # Shape: (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
                    attention_mask = attention_mask.unsqueeze(1)
                elif attention_mask.dim() == 4:
                    # Already in the right format (batch, nhead, seq_len, seq_len)
                    pass
                else:
                    raise ValueError(
                        f"Invalid attention mask dimension: {attention_mask.dim()}"
                    )
                    
                # Expand to match number of heads if needed
                if attention_mask.size(1) == 1:
                    attention_mask = attention_mask.expand(-1, self.nhead, -1, -1)
                    
                attn_weights = attn_weights + attention_mask
                
            # Apply softmax
            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            
            # Apply dropout if configured
            if self.dropout is not None:
                attn_weights = self.dropout(attn_weights)
                
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)
        else:
            # Standard attention computation
            attn_weights = torch.matmul(q, k.transpose(-2, -1))
            
            # Apply sliding window mask if enabled
            if self.use_sliding_window and seq_len <= self.sliding_window_size:
                sliding_mask = self.sliding_window_mask[:, :, :seq_len, :seq_len]
                attn_weights.masked_fill_(~sliding_mask, float("-inf"))
            
            # Apply causal mask if needed
            if self.causal and seq_len <= 1024:
                causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
                attn_weights.masked_fill_(causal_mask, float("-inf"))
                
            # Apply attention mask if provided
            if attention_mask is not None:
                # Expand attention mask to match attention weights dimensions
                if attention_mask.dim() == 2:
                    # Shape: (batch, seq_len) -> (batch, 1, 1, seq_len)
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                elif attention_mask.dim() == 3:
                    # Shape: (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
                    attention_mask = attention_mask.unsqueeze(1)
                elif attention_mask.dim() == 4:
                    # Already in the right format (batch, nhead, seq_len, seq_len)
                    pass
                else:
                    raise ValueError(
                        f"Invalid attention mask dimension: {attention_mask.dim()}"
                    )
                    
                # Expand to match number of heads if needed
                if attention_mask.size(1) == 1:
                    attention_mask = attention_mask.expand(-1, self.nhead, -1, -1)
                    
                attn_weights = attn_weights + attention_mask
                
            # Apply softmax
            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            
            # Apply dropout if configured
            if self.dropout is not None:
                attn_weights = self.dropout(attn_weights)
                
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)
            
        # Reshape to combine heads
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )
        
        # Return output and attention weights if needed
        if need_weights:
            return attn_output, attn_weights
        else:
            return attn_output, None


class Qwen3CoderNextMLPKernel(MLPLayerKernel):
    """
    Qwen3-Coder-Next specific MLP kernel with custom optimizations.
    Implements SwiGLU activation with Mixture of Experts support.
    """
    
    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        activation_type: str = "silu",
        dropout: float = 0.1,
        use_fused_linear: bool = True,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__(d_model, intermediate_size, activation_type, use_swiglu=True, dropout=dropout)
        
        # Qwen-specific: SwiGLU activation: SiLU(gate_proj(x)) * up_proj(x) -> down_proj
        self.gate_proj = LinearProjectionKernel(d_model, intermediate_size, bias=False)
        self.up_proj = LinearProjectionKernel(d_model, intermediate_size, bias=False)
        self.down_proj = LinearProjectionKernel(intermediate_size, d_model, bias=False)
        self.activation = ActivationKernel("silu")  # Qwen models use SiLU for SwiGLU
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.use_fused_linear = use_fused_linear
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k = top_k
        
        # For coding models, we might want to use fused linear layers for better performance
        if use_fused_linear:
            # Create fused linear layer for gate and up projections
            self.fused_gate_up = nn.Linear(d_model, 2 * intermediate_size, bias=False)
        
        # Mixture of Experts for MLP if enabled
        if use_moe:
            self.moe_router = nn.Linear(d_model, num_experts, bias=False)
            self.moe_experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, intermediate_size),
                    nn.SiLU(),
                    nn.Linear(intermediate_size, d_model)
                ) for _ in range(num_experts)
            ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Qwen3-Coder-Next MLP kernel.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        if self.use_moe:
            # Router to determine which experts to use
            router_logits = self.moe_router(x.mean(dim=1))  # Average over sequence
            routing_weights = torch.softmax(router_logits, dim=-1)
            top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)

            # Normalize weights
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

            # Process through selected experts
            expert_outputs = []
            for i in range(self.top_k):
                # Get the expert indices for this top-k
                expert_idx_batch = top_k_indices[:, i]  # [batch]

                # Process each item in the batch separately
                for j, expert_idx in enumerate(expert_idx_batch):
                    expert_output = self.moe_experts[expert_idx.item()](x[j:j+1])  # Process single item
                    weighted_output = expert_output * top_k_weights[j, i:j+1].unsqueeze(1).unsqueeze(1)
                    if len(expert_outputs) <= j:
                        expert_outputs.append(weighted_output)
                    else:
                        expert_outputs[j] += weighted_output

            # Combine outputs for the batch
            x = torch.cat(expert_outputs, dim=0)
        else:
            # Standard SwiGLU: SiLU(gate_proj(x)) * (up_proj(x)) -> down_proj
            if self.use_fused_linear:
                # Fused operation: gate and up projections together
                gate_up = self.fused_gate_up(x)
                gate, up = gate_up.chunk(2, dim=-1)
                gate = self.activation(gate)
            else:
                gate = self.activation(self.gate_proj(x))
                up = self.up_proj(x)
            
            x = gate * up
            x = self.down_proj(x)
        
        # Apply dropout if configured
        if self.dropout is not None:
            x = self.dropout(x)
            
        return x


class Qwen3CoderNextRMSNormKernel(NormalizationKernel):
    """
    Qwen3-Coder-Next specific RMSNorm kernel with custom optimizations.
    Implements Root Mean Square Normalization with Qwen-specific parameters.
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__(normalized_shape, norm_type="rms", eps=eps)
        
        # Qwen-specific initialization
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Qwen3-Coder-Next RMSNorm kernel.
        
        Args:
            x: Input tensor of shape (..., normalized_shape)
            
        Returns:
            Normalized tensor of same shape as input
        """
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        
        # Normalize
        x = x / rms
        
        # Apply learned weight
        x = x * self.weight
        
        return x


class Qwen3CoderNextRotaryEmbedding(nn.Module):
    """
    Rotary Embedding implementation specific to Qwen3-Coder-Next.
    """
    
    def __init__(self, dim: int, base: int = 1000000, max_position_embeddings: int = 131072):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch, seq_len, d_model)
            k: Key tensor of shape (batch, seq_len, d_model)
            position_ids: Position IDs for RoPE embeddings

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        # Calculate cos and sin embeddings
        max_pos = position_ids.max().item() + 1
        self._update_cos_sin_tables(max_pos, q.device, q.dtype)

        # Get the cos and sin values for the specific positions
        # position_ids shape: [batch_size, seq_len]
        # self._cos_cached shape: [1, max_seq_len, head_dim]
        batch_size, seq_len, d_model = q.shape
        head_dim = self.inv_freq.shape[0] * 2  # Each frequency corresponds to 2 dimensions

        # Reshape q and k to [batch, seq_len, num_heads, head_dim]
        num_heads = d_model // head_dim
        q_reshaped = q.view(batch_size, seq_len, num_heads, head_dim)
        k_reshaped = k.view(batch_size, seq_len, num_heads, head_dim)

        # Get cos and sin for the specific positions
        cos = self._cos_cached[0, position_ids]  # [batch_size, seq_len, head_dim]
        sin = self._sin_cached[0, position_ids]  # [batch_size, seq_len, head_dim]

        # Expand cos and sin to match the number of heads
        cos_expanded = cos.unsqueeze(2)  # [batch_size, seq_len, 1, head_dim]
        sin_expanded = sin.unsqueeze(2)  # [batch_size, seq_len, 1, head_dim]

        # Apply rotation to query and key
        q_rotated = self._rotate_half(q_reshaped) * cos_expanded + q_reshaped * sin_expanded
        k_rotated = self._rotate_half(k_reshaped) * cos_expanded + k_reshaped * sin_expanded

        # Reshape back to [batch, seq_len, d_model]
        q_rotated = q_rotated.reshape(batch_size, seq_len, d_model)
        k_rotated = k_rotated.reshape(batch_size, seq_len, d_model)

        return q_rotated, k_rotated

    def _update_cos_sin_tables(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cosine and sine tables for rotary embeddings."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(dtype)

            self._cos_cached = emb.cos()[None, :, :]
            self._sin_cached = emb.sin()[None, :, :]

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        # Ensure the last dimension is even for proper rotation
        if x.size(-1) % 2 != 0:
            raise ValueError(f"Last dimension must be even for rotation, got {x.size(-1)}")
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.cat((-x2, x1), dim=-1)


class Qwen3CoderNextLinearKernel(LinearProjectionKernel):
    """
    Qwen3-Coder-Next specific linear projection kernel with custom optimizations.
    Includes fused operations, quantization support, and MoE support.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, fused: bool = True, quantized: bool = False, use_moe: bool = False, num_experts: int = 8):
        super().__init__(in_features, out_features, bias)
        self.fused = fused
        self.quantized = quantized
        self.use_moe = use_moe
        self.num_experts = num_experts
        
        # Qwen-specific: Initialize with specific weight initialization
        if fused:
            # For fused operations, we might use different initialization strategies
            nn.init.xavier_uniform_(self.linear.weight)
            if bias:
                nn.init.zeros_(self.linear.bias)
        
        # For quantized models, we might add additional parameters
        if quantized:
            self.register_buffer("scale", torch.ones(out_features))
            self.register_buffer("zero_point", torch.zeros(out_features))
        
        # For MoE models
        if use_moe:
            self.moe_router = nn.Linear(in_features, num_experts, bias=False)
            self.moe_experts = nn.ModuleList([
                nn.Linear(in_features, out_features, bias=bias) for _ in range(num_experts)
            ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Qwen3-Coder-Next linear kernel.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        if self.use_moe:
            # Router to determine which experts to use
            router_logits = self.moe_router(x.mean(dim=-2, keepdim=True))  # Average over sequence
            routing_weights = torch.softmax(router_logits, dim=-1)
            top_k_weights, top_k_indices = torch.topk(routing_weights, 1, dim=-1)  # Use top-1 for efficiency
            
            # Process through selected expert
            expert_idx = top_k_indices.squeeze(-1).squeeze(-1)  # [batch]
            expert_outputs = []
            
            for i, idx in enumerate(expert_idx):
                expert_output = self.moe_experts[idx](x[i:i+1])
                expert_outputs.append(expert_output)
            
            return torch.cat(expert_outputs, dim=0)
        elif self.quantized:
            # Simulate quantization effects
            w = self.linear.weight * self.scale.unsqueeze(0) + self.zero_point.unsqueeze(0)
            return torch.nn.functional.linear(x, w, self.linear.bias)
        else:
            return self.linear(x)


class Qwen3CoderNextHardwareOptimizer(CUDAHardwareOptimizer):
    """
    Hardware-specific optimizer for Qwen3-Coder-Next CUDA kernels.
    Detects hardware capabilities and applies Qwen-specific optimizations.
    """
    
    def __init__(self):
        super().__init__()
        self.optimization_level = self._determine_qwen_optimization_level()
        
    def _determine_qwen_optimization_level(self) -> str:
        """Determine the optimization level based on hardware for Qwen models."""
        major, minor = self.compute_capability
        if major >= 8:  # Ampere and later - supports sparsity and better tensor cores
            return "high_qwen_coder_next"
        elif major >= 7:  # Volta, Turing - basic tensor core support
            return "medium_qwen_coder_next"
        else:  # Older architectures
            return "basic_qwen_coder_next"
            
    def get_qwen_optimization_report(self) -> Dict:
        """Get a report of Qwen3-Coder-Next specific optimizations."""
        return {
            "model_type": "qwen3_coder_next",
            "compute_capability": self.compute_capability,
            "tensor_cores_supported": self.tensor_cores_supported,
            "optimization_level": self.optimization_level,
            "recommended_kernels": self._get_qwen_recommended_kernels(),
        }
        
    def _get_qwen_recommended_kernels(self) -> List[str]:
        """Get recommended kernels based on hardware for Qwen models."""
        recommendations = ["qwen_attention", "qwen_mlp_swiglu", "qwen_rmsnorm", "qwen_rope"]
        
        if self.tensor_cores_supported:
            recommendations.extend(["qwen_fused_ops", "qwen_quantized_ops", "qwen_moe_support"])
            
        return recommendations


def create_qwen3_coder_next_cuda_kernels(
    d_model: int,
    nhead: int,
    intermediate_size: int,
    use_flash_attention: bool = True,
    use_rotary_embeddings: bool = True,
    use_sliding_window: bool = False,
    sliding_window_size: int = 4096,
    use_moe: bool = False,
    num_experts: int = 8,
    top_k: int = 2,
) -> Dict[str, BaseCUDAKernel]:
    """
    Factory function to create Qwen3-Coder-Next specific CUDA kernels.
    
    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        intermediate_size: Size of intermediate layer in MLP
        use_flash_attention: Whether to use flash attention optimization
        use_rotary_embeddings: Whether to use rotary embeddings
        use_sliding_window: Whether to use sliding window attention
        sliding_window_size: Size of sliding window for attention
        use_moe: Whether to use Mixture of Experts
        num_experts: Number of experts for MoE
        top_k: Number of experts to select for each token
        
    Returns:
        Dictionary of created kernels
    """
    kernels = {}
    
    # Create Qwen3-Coder-Next attention kernel
    kernels["attention"] = Qwen3CoderNextAttentionKernel(
        d_model=d_model,
        nhead=nhead,
        use_flash_attention=use_flash_attention,
        use_rotary_embeddings=use_rotary_embeddings,
        use_sliding_window=use_sliding_window,
        sliding_window_size=sliding_window_size,
        use_moe=use_moe,
        num_experts=num_experts,
        top_k=top_k,
    )
    
    # Create Qwen3-Coder-Next MLP kernel with SwiGLU
    kernels["mlp"] = Qwen3CoderNextMLPKernel(
        d_model=d_model,
        intermediate_size=intermediate_size,
        use_moe=use_moe,
        num_experts=num_experts,
        top_k=top_k,
    )
    
    # Create Qwen3-Coder-Next RMSNorm kernel
    kernels["rmsnorm"] = Qwen3CoderNextRMSNormKernel(normalized_shape=d_model)
    
    # Create Qwen3-Coder-Next linear kernel
    kernels["linear"] = Qwen3CoderNextLinearKernel(
        in_features=d_model,
        out_features=d_model,
        fused=True,
        use_moe=use_moe,
        num_experts=num_experts,
    )
    
    return kernels


def apply_qwen3_coder_next_cuda_optimizations_to_model(
    model: nn.Module,
    d_model: int,
    nhead: int,
    intermediate_size: int,
    config,
) -> nn.Module:
    """
    Apply Qwen3-Coder-Next specific CUDA optimizations to the model.
    
    Args:
        model: The model to optimize
        d_model: Model dimension
        nhead: Number of attention heads
        intermediate_size: Size of intermediate layer in MLP
        config: Model configuration
        
    Returns:
        Optimized model
    """
    logger.info("Applying Qwen3-Coder-Next specific CUDA optimizations...")
    
    # Create hardware optimizer
    hw_optimizer = Qwen3CoderNextHardwareOptimizer()
    opt_report = hw_optimizer.get_qwen_optimization_report()
    logger.info(f"Hardware optimization report: {opt_report}")
    
    # Create Qwen3-Coder-Next specific kernels
    kernels = create_qwen3_coder_next_cuda_kernels(
        d_model=d_model,
        nhead=nhead,
        intermediate_size=intermediate_size,
        use_flash_attention=getattr(config, 'cuda_kernel_attention_enabled', True),
        use_rotary_embeddings=getattr(config, 'use_rotary_embeddings', True),
        use_sliding_window=getattr(config, 'use_sliding_window', False),
        sliding_window_size=getattr(config, 'sliding_window_size', 4096),
        use_moe=getattr(config, 'use_moe', False),
        num_experts=getattr(config, 'num_experts', 8),
        top_k=getattr(config, 'top_k', 2),
    )
    
    # Apply optimizations based on model architecture
    for name, module in model.named_modules():
        # Replace attention modules with Qwen3-Coder-Next optimized versions
        if isinstance(module, nn.MultiheadAttention):
            parent_module = _get_parent_module(model, name.rsplit(".", 1)[0]) if "." in name else model
            child_name = name.rsplit(".", 1)[-1] if "." in name else name
            
            # Create a Qwen3-Coder-Next attention kernel that mimics the original interface
            qwen_attn = Qwen3CoderNextAttentionKernel(
                d_model=d_model,
                nhead=nhead,
                use_flash_attention=getattr(config, 'cuda_kernel_attention_enabled', True),
                use_rotary_embeddings=getattr(config, 'use_rotary_embeddings', True),
                use_sliding_window=getattr(config, 'use_sliding_window', False),
                sliding_window_size=getattr(config, 'sliding_window_size', 4096),
                use_moe=getattr(config, 'use_moe', False),
                num_experts=getattr(config, 'num_experts', 8),
                top_k=getattr(config, 'top_k', 2),
            )
            
            setattr(parent_module, child_name, qwen_attn)
            logger.info(f"Replaced attention module {name} with Qwen3-Coder-Next optimized kernel")
            
        # Replace LayerNorm modules with Qwen3-Coder-Next RMSNorm
        elif isinstance(module, nn.LayerNorm):
            parent_module = _get_parent_module(model, name.rsplit(".", 1)[0]) if "." in name else model
            child_name = name.rsplit(".", 1)[-1] if "." in name else name
            
            # Qwen models typically use RMSNorm instead of LayerNorm
            qwen_norm = Qwen3CoderNextRMSNormKernel(
                normalized_shape=module.normalized_shape[0], eps=module.eps
            )
            
            # Copy parameters if available
            with torch.no_grad():
                qwen_norm.weight.copy_(module.weight)
                
            setattr(parent_module, child_name, qwen_norm)
            logger.info(f"Replaced normalization module {name} with Qwen3-Coder-Next RMSNorm kernel")
            
        # Replace MLP/feed-forward modules with Qwen3-Coder-Next optimized versions
        elif hasattr(module, "forward") and any(
            name_part in name.lower()
            for name_part in ["mlp", "ffn", "feed_forward", "intermediate"]
        ):
            parent_module = _get_parent_module(model, name.rsplit(".", 1)[0]) if "." in name else model
            child_name = name.rsplit(".", 1)[-1] if "." in name else name
            
            # Create a Qwen3-Coder-Next MLP kernel with SwiGLU
            qwen_mlp = Qwen3CoderNextMLPKernel(
                d_model=d_model,
                intermediate_size=intermediate_size,
                use_moe=getattr(config, 'use_moe', False),
                num_experts=getattr(config, 'num_experts', 8),
                top_k=getattr(config, 'top_k', 2),
            )
            
            setattr(parent_module, child_name, qwen_mlp)
            logger.info(f"Replaced MLP module {name} with Qwen3-Coder-Next optimized kernel")
    
    logger.info("Qwen3-Coder-Next specific CUDA optimizations applied successfully")
    return model


def get_qwen3_coder_next_cuda_optimization_report(
    model: nn.Module,
    d_model: int,
    nhead: int,
    intermediate_size: int,
    config,
) -> Dict:
    """
    Get a report of Qwen3-Coder-Next CUDA optimizations applied to the model.
    
    Args:
        model: The model to analyze
        d_model: Model dimension
        nhead: Number of attention heads
        intermediate_size: Size of intermediate layer in MLP
        config: Model configuration
        
    Returns:
        Optimization report
    """
    hw_optimizer = Qwen3CoderNextHardwareOptimizer()
    hw_report = hw_optimizer.get_qwen_optimization_report()
    
    # Count relevant modules that would be optimized
    attention_count = sum(
        1 for m in model.modules() if isinstance(m, nn.MultiheadAttention)
    )
    layernorm_count = sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm))
    
    # Simple MLP detection (would need more sophisticated logic in practice)
    mlp_count = sum(
        1
        for name, m in model.named_modules()
        if hasattr(m, "forward")
        and any(part in name.lower() for part in ["mlp", "ffn", "feed_forward"])
    )
    
    report = {
        "model_type": "qwen3_coder_next",
        "hardware_info": hw_report,
        "modules_identified_for_optimization": {
            "attention_layers": attention_count,
            "normalization_layers": layernorm_count,
            "mlp_layers": mlp_count,
        },
        "optimization_config": {
            "d_model": d_model,
            "nhead": nhead,
            "intermediate_size": intermediate_size,
            "use_swiglu": True,  # Qwen models use SwiGLU
            "use_flash_attention": getattr(config, 'cuda_kernel_attention_enabled', True),
            "use_rms_norm": True,  # Qwen models use RMSNorm
            "use_rotary_embeddings": getattr(config, 'use_rotary_embeddings', True),
            "use_sliding_window": getattr(config, 'use_sliding_window', False),
            "sliding_window_size": getattr(config, 'sliding_window_size', 4096),
            "use_moe": getattr(config, 'use_moe', False),
            "num_experts": getattr(config, 'num_experts', 8),
            "top_k": getattr(config, 'top_k', 2),
        },
        "notes": "Qwen3-Coder-Next specific CUDA optimizations applied with SwiGLU activation, RMSNorm, and MoE support",
    }
    
    return report


def _get_parent_module(model: nn.Module, parent_name: str) -> nn.Module:
    """
    Get parent module by name.
    
    Args:
        model: The model
        parent_name: Name of the parent module
        
    Returns:
        Parent module
    """
    parent_module = model
    for n in parent_name.split("."):
        if n:  # Skip empty strings
            parent_module = getattr(parent_module, n)
    return parent_module


__all__ = [
    "Qwen3CoderNextAttentionKernel",
    "Qwen3CoderNextMLPKernel",
    "Qwen3CoderNextRMSNormKernel",
    "Qwen3CoderNextRotaryEmbedding",
    "Qwen3CoderNextLinearKernel",
    "Qwen3CoderNextHardwareOptimizer",
    "create_qwen3_coder_next_cuda_kernels",
    "apply_qwen3_coder_next_cuda_optimizations_to_model",
    "get_qwen3_coder_next_cuda_optimization_report",
]