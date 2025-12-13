"""
Routing configuration classes for Qwen3-VL model components.

This module contains configuration classes specifically for routing mechanisms
such as Mixture of Experts (MoE) with clear separation of concerns.
"""
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class RoutingConfig:
    """
    Configuration class for routing mechanisms (e.g., Mixture of Experts).
    """
    # Mixture of Experts configuration
    use_moe: bool = False  # Enable Mixture of Experts
    moe_num_experts: int = 4  # Number of experts in MoE (2-4 as specified)
    moe_top_k: int = 2  # Top-k routing for MoE (top-2 as specified)
    moe_use_residual: bool = True  # Use residual connection in MoE
    moe_jitter_noise: float = 0.01  # Jitter noise for load balancing
    moe_normalize_gate: bool = True  # Normalize gate probabilities
    moe_capacity_factor: float = 1.0  # Factor to determine expert capacity
    moe_drop_tokens: bool = True  # Whether to drop tokens when exceeding capacity
    moe_use_tutel: bool = False  # Use Tutel for MoE optimization

    # Expert routing configuration
    moe_router_zloss_coef: float = 1e-4  # Coefficient for z-loss in MoE router
    moe_router_aux_loss_coef: float = 1e-2  # Coefficient for auxiliary loss in MoE router
    moe_label_smoothing: float = 0.0  # Label smoothing for MoE routing
    moe_router_dtype: str = "float32"  # Data type for router computations

    # Token-level routing configuration (Phase 9)
    use_token_level_routing: bool = False  # Enable token-level routing optimization
    token_routing_temperature: float = 1.0  # Temperature for token routing softmax
    token_routing_confidence_threshold: float = 0.1  # Threshold for token routing confidence

    # Adaptive routing configuration (Phase 7)
    use_adaptive_routing: bool = False  # Enable adaptive routing based on input complexity
    adaptive_routing_complexity_metric: str = "entropy"  # Metric for complexity assessment
    adaptive_routing_temperature: float = 1.0  # Temperature for adaptive routing

    # Cross-layer parameter recycling configuration (Phase 9)
    use_cross_layer_recycling: bool = False  # Enable cross-layer parameter recycling
    cross_layer_recycling_ratio: float = 0.5  # Ratio of parameters to recycle
    cross_layer_adapter_dim: int = 64  # Dimension of adapters for recycled parameters

    # Routing optimization configuration
    use_load_balancing: bool = True  # Enable load balancing in routing
    load_balancing_frequency: int = 10  # Frequency of load balancing updates
    use_frequency_regularization: bool = True  # Use frequency regularization for load balancing

    def __post_init__(self):
        """Validate routing configuration after initialization."""
        if self.moe_num_experts < 1:
            raise ValueError(f"moe_num_experts must be at least 1, got {self.moe_num_experts}")

        if self.moe_top_k < 1 or self.moe_top_k > self.moe_num_experts:
            raise ValueError(f"moe_top_k must be between 1 and moe_num_experts ({self.moe_num_experts}), got {self.moe_top_k}")

        if self.moe_jitter_noise < 0:
            raise ValueError(f"moe_jitter_noise must be non-negative, got {self.moe_jitter_noise}")

        if self.moe_capacity_factor <= 0:
            raise ValueError(f"moe_capacity_factor must be positive, got {self.moe_capacity_factor}")

        if self.token_routing_temperature <= 0:
            raise ValueError(f"token_routing_temperature must be positive, got {self.token_routing_temperature}")

        if not 0.0 <= self.token_routing_confidence_threshold <= 1.0:
            raise ValueError(f"token_routing_confidence_threshold must be between 0.0 and 1.0, got {self.token_routing_confidence_threshold}")