"""
Mixture of Experts (MoE) layer for Qwen3-VL model.

This module implements configurable Mixture of Experts functionality
with proper configuration support.
"""
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.qwen3_vl.config.routing_config import RoutingConfig


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with configurable routing mechanisms.
    """
    
    def __init__(self, config: RoutingConfig):
        """
        Initialize the MoE layer with the provided configuration.
        
        Args:
            config: RoutingConfig instance containing routing settings
        """
        super().__init__()
        self.config = config
        
        # Set up MoE parameters
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.use_residual = config.moe_use_residual
        self.jitter_noise = config.moe_jitter_noise
        self.normalize_gate = config.moe_normalize_gate
        self.capacity_factor = config.moe_capacity_factor
        self.drop_tokens = config.moe_drop_tokens
        self.use_tutel = config.moe_use_tutel
        
        # Initialize experts (these would typically be passed in)
        self.experts = nn.ModuleList()
        self.gate = None  # Will be initialized later with proper input/output dimensions
        
        # Initialize routing-specific parameters
        self.router_zloss_coef = config.moe_router_zloss_coef
        self.router_aux_loss_coef = config.moe_router_aux_loss_coef
        self.label_smoothing = config.moe_label_smoothing
        
        # For now, we'll just store the config - actual initialization happens when we know input/output dimensions
    
    def setup_experts_and_gate(self, input_dim: int, output_dim: int, intermediate_dim: int):
        """
        Set up experts and gate with specific dimensions.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            intermediate_dim: Intermediate dimension for FFN experts
        """
        # Create expert networks (simplified as FFN blocks)
        for _ in range(self.num_experts):
            expert = nn.Sequential(
                nn.Linear(input_dim, intermediate_dim),
                nn.GELU(),
                nn.Linear(intermediate_dim, output_dim)
            )
            self.experts.append(expert)
        
        # Create router/gate network
        self.gate = nn.Linear(input_dim, self.num_experts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, input_dim = x.size()
        
        # Flatten batch and sequence dimensions for processing
        flat_x = x.view(-1, input_dim)  # (batch_size * seq_len, input_dim)
        
        # Add jitter noise for load balancing if enabled
        if self.jitter_noise > 0 and self.training:
            jitter = torch.empty_like(flat_x).uniform_(-self.jitter_noise, self.jitter_noise)
            gate_input = flat_x + jitter
        else:
            gate_input = flat_x
        
        # Compute gate logits
        gate_logits = self.gate(gate_input)  # (batch_size * seq_len, num_experts)
        
        # Apply softmax to get routing weights
        raw_weights = F.softmax(gate_logits, dim=-1)  # (batch_size * seq_len, num_experts)
        
        # Select top-k experts for each token
        top_k_weights, top_k_indices = torch.topk(raw_weights, self.top_k, dim=-1)  # (..., top_k)
        
        # Create one-hot representation for selected experts
        flat_zero_one_hot = F.one_hot(top_k_indices, num_classes=self.num_experts).float()  # (..., top_k, num_experts)
        gate_output = (flat_zero_one_hot * top_k_weights.unsqueeze(-1)).sum(dim=-2)  # (..., num_experts)
        
        # Compute auxiliary loss for load balancing
        if self.training:
            # Compute auxiliary loss for MoE routing
            # This is a simplified version - in practice, you'd implement the full auxiliary loss
            aux_loss = self._compute_auxiliary_loss(gate_logits, gate_output)
        else:
            aux_loss = None
        
        # Process tokens through selected experts
        expert_outputs = []
        for expert_idx, expert in enumerate(self.experts):
            # Create a mask for tokens assigned to this expert
            expert_mask = gate_output[:, expert_idx]  # (batch_size * seq_len,)
            
            # Process tokens through this expert
            expert_input = flat_x  # For all tokens
            expert_out = expert(expert_input)  # (batch_size * seq_len, output_dim)
            
            # Apply the mask to only include contributions from assigned tokens
            masked_expert_out = expert_out * expert_mask.unsqueeze(-1)  # (batch_size * seq_len, output_dim)
            expert_outputs.append(masked_expert_out)
        
        # Combine outputs from all experts
        combined_output = torch.stack(expert_outputs, dim=0).sum(dim=0)  # (batch_size * seq_len, output_dim)
        
        # Reshape back to original dimensions
        output = combined_output.view(batch_size, seq_len, -1)
        
        return output
    
    def _compute_auxiliary_loss(self, gate_logits: torch.Tensor, gate_output: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary loss for load balancing in MoE routing.
        
        Args:
            gate_logits: Raw logits from the gate network
            gate_output: Final gate output after top-k selection
            
        Returns:
            Auxiliary loss value
        """
        # Compute auxiliary loss based on the difference between average probability per expert
        # and average probability per example
        num_samples = gate_logits.size(0)
        
        # Average probability per expert (over all tokens)
        avg_prob_per_expert = gate_output.mean(dim=0)  # (num_experts,)
        
        # Average probability per token (over all experts)
        avg_prob_per_token = gate_output.mean(dim=1)  # (num_samples,)
        
        # Compute auxiliary loss as the product of these averages
        aux_loss = torch.mean(avg_prob_per_expert * avg_prob_per_token) * self.num_experts
        
        return aux_loss * self.router_aux_loss_coef