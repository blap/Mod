"""Learned Activation Routing Implementation for Qwen3-VL model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class LearnedActivationRouter(nn.Module):
    """Learned activation router that determines which tokens to attend to based on input characteristics."""
    
    def __init__(self, config, temperature: float = 1.0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.temperature = temperature
        self.num_attention_heads = config.num_attention_heads
        
        # Router network to learn which tokens to attend to
        self.router_network = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, self.num_attention_heads),  # One score per head
            nn.Softmax(dim=-1)
        )
        
        # Additional routing components for more sophisticated selection
        self.token_importance_predictor = nn.Linear(self.hidden_size, 1)
        self.routing_temperature = nn.Parameter(torch.tensor(temperature))
        
        # Initialize router weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize router weights."""
        for module in self.router_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor, layer_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route activations based on learned importance.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            layer_idx: Current layer index for layer-specific routing
            
        Returns:
            Tuple of (routed_hidden_states, routing_weights)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute routing scores for each head
        routing_scores = self.router_network(hidden_states.mean(dim=1, keepdim=True))  # [batch, 1, num_heads]
        routing_scores = routing_scores.squeeze(1)  # [batch, num_heads]
        
        # Compute token importance scores
        token_importance = self.token_importance_predictor(hidden_states).squeeze(-1)  # [batch, seq_len]
        
        # Combine routing scores with token importance
        # Add layer-specific bias to routing
        if layer_idx is not None:
            layer_bias = torch.tensor(layer_idx % self.num_attention_heads, dtype=torch.float, device=hidden_states.device)
            routing_scores = routing_scores + layer_bias / self.num_attention_heads
        
        # Normalize routing scores
        routing_weights = F.softmax(routing_scores / self.routing_temperature, dim=-1)  # [batch, num_heads]
        
        # Apply learned routing to determine which tokens to process
        # Use token importance to select top-k tokens per head
        num_select = max(1, int(seq_len * 0.5))  # Select top 50% of tokens by default
        _, top_k_indices = torch.topk(token_importance, num_select, dim=-1, sorted=False)  # [batch, num_select]
        
        # Create mask for selected tokens
        token_mask = torch.zeros_like(token_importance, dtype=torch.bool)  # [batch, seq_len]
        batch_indices = torch.arange(batch_size, device=hidden_states.device).unsqueeze(1).expand(-1, num_select)
        token_mask[batch_indices, top_k_indices] = True
        
        # Apply routing mask to hidden states
        masked_hidden_states = hidden_states.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        
        # Return both the routed states and the routing weights
        return masked_hidden_states, routing_weights


class AdaptiveActivationRouter(nn.Module):
    """Adaptive activation router that adjusts routing based on input complexity and layer position."""
    
    def __init__(self, config, temperature: float = 1.0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.temperature = temperature
        self.num_attention_heads = config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        
        # Complexity assessment network
        self.complexity_assessor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Layer-specific routing parameters
        self.layer_routing_params = nn.Parameter(torch.randn(self.num_hidden_layers, self.num_attention_heads))
        
        # Adaptive routing network
        self.adaptive_router = nn.Sequential(
            nn.Linear(self.hidden_size + 2, self.hidden_size // 4),  # +2 for layer_idx and complexity
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, self.num_attention_heads),
            nn.Softmax(dim=-1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize router weights."""
        for module in self.adaptive_router:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize layer-specific parameters
        nn.init.normal_(self.layer_routing_params, std=0.02)
    
    def forward(self, hidden_states: torch.Tensor, layer_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive routing based on input complexity and layer position.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            layer_idx: Current layer index for layer-specific routing
            
        Returns:
            Tuple of (routed_hidden_states, routing_weights)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Assess input complexity
        complexity_score = self.complexity_assessor(hidden_states.mean(dim=1))  # [batch, 1]
        
        # Create layer index tensor (normalized)
        if layer_idx is not None:
            layer_tensor = torch.tensor(layer_idx / self.num_hidden_layers, 
                                      dtype=torch.float, device=hidden_states.device)
            layer_tensor = layer_tensor.expand(batch_size, 1)
        else:
            layer_tensor = torch.zeros(batch_size, 1, device=hidden_states.device)
        
        # Combine hidden states with complexity and layer info
        combined_features = torch.cat([
            hidden_states.mean(dim=1),  # [batch, hidden_size]
            complexity_score,          # [batch, 1]
            layer_tensor               # [batch, 1]
        ], dim=-1)  # [batch, hidden_size + 2]
        
        # Compute adaptive routing scores
        routing_scores = self.adaptive_router(combined_features)  # [batch, num_heads]
        
        # Apply layer-specific adjustments
        if layer_idx is not None and layer_idx < self.num_hidden_layers:
            layer_adjustment = self.layer_routing_params[layer_idx]  # [num_heads]
            routing_scores = routing_scores + layer_adjustment.unsqueeze(0)
        
        # Apply temperature scaling
        routing_weights = F.softmax(routing_scores / self.temperature, dim=-1)  # [batch, num_heads]
        
        # Determine how many tokens to process based on complexity
        # More complex inputs need more tokens processed
        tokens_to_process = max(1, int(seq_len * (0.3 + 0.7 * complexity_score.mean())))
        
        # Select top-k tokens based on importance
        token_importance = self._compute_token_importance(hidden_states)
        _, top_k_indices = torch.topk(token_importance, tokens_to_process, dim=-1, sorted=False)
        
        # Create mask for selected tokens
        token_mask = torch.zeros_like(token_importance, dtype=torch.bool)
        batch_indices = torch.arange(batch_size, device=hidden_states.device).unsqueeze(1).expand(-1, tokens_to_process)
        token_mask[batch_indices, top_k_indices] = True
        
        # Apply routing mask to hidden states
        routed_hidden_states = hidden_states.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        
        return routed_hidden_states, routing_weights
    
    def _compute_token_importance(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute importance scores for each token."""
        # Use variance as a measure of importance
        token_variance = torch.var(hidden_states, dim=-1)  # [batch, seq_len]
        
        # Use gradient magnitude proxy (L2 norm of hidden states) as importance
        token_magnitude = torch.norm(hidden_states, p=2, dim=-1)  # [batch, seq_len]
        
        # Combine variance and magnitude
        token_importance = token_variance + token_magnitude
        
        return token_importance


class ContextAwareActivationRouter(nn.Module):
    """Context-aware activation router that considers both spatial and temporal context."""
    
    def __init__(self, config, temperature: float = 1.0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.temperature = temperature
        self.num_attention_heads = config.num_attention_heads
        
        # Context-aware routing network
        self.context_router = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.num_attention_heads),
            nn.Softmax(dim=-1)
        )
        
        # Spatial context processor for vision tasks
        self.spatial_context_processor = nn.Conv2d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size // 4,
            kernel_size=3,
            padding=1,
            groups=self.hidden_size // 4  # Depthwise convolution for efficiency
        )
        
        # Temporal context processor for language tasks
        self.temporal_context_processor = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size // 4,
            kernel_size=3,
            padding=1
        )
        
        # Context aggregation layer
        self.context_aggregator = nn.Linear(self.hidden_size + self.hidden_size // 4, self.hidden_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize router weights."""
        for module in self.context_router:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize context processors
        nn.init.kaiming_normal_(self.spatial_context_processor.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.temporal_context_processor.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                layer_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Context-aware routing based on spatial and temporal context.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            layer_idx: Current layer index
            
        Returns:
            Tuple of (routed_hidden_states, routing_weights)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Process context
        if seq_len > 100:  # Likely a language sequence
            # Apply temporal context processing
            temporal_context = hidden_states.transpose(1, 2)  # [batch, hidden_size, seq_len]
            temporal_processed = self.temporal_context_processor(temporal_context)  # [batch, hidden_size//4, seq_len]
            temporal_processed = temporal_processed.transpose(1, 2)  # [batch, seq_len, hidden_size//4]
        else:  # Likely vision patches
            # Reshape to 2D for spatial processing (assuming square patches)
            spatial_dim = int(math.sqrt(seq_len))
            if spatial_dim * spatial_dim == seq_len:
                spatial_context = hidden_states.view(batch_size, spatial_dim, spatial_dim, hidden_size)
                spatial_context = spatial_context.permute(0, 3, 1, 2)  # [batch, hidden_size, spatial_dim, spatial_dim]
                spatial_processed = self.spatial_context_processor(spatial_context)  # [batch, hidden_size//4, spatial_dim, spatial_dim]
                spatial_processed = spatial_processed.permute(0, 2, 3, 1).contiguous().view(batch_size, seq_len, -1)  # [batch, seq_len, hidden_size//4]
            else:
                # If not square, use temporal processing as fallback
                temporal_context = hidden_states.transpose(1, 2)
                temporal_processed = self.temporal_context_processor(temporal_context)
                temporal_processed = temporal_processed.transpose(1, 2)
        
        # Aggregate context with original hidden states
        context_enhanced = torch.cat([hidden_states, temporal_processed], dim=-1)  # [batch, seq_len, hidden_size + hidden_size//4]
        context_enhanced = self.context_aggregator(context_enhanced)  # [batch, seq_len, hidden_size]
        
        # Compute routing scores using context-enhanced representations
        routing_input = context_enhanced.mean(dim=1)  # [batch, hidden_size]
        routing_scores = self.context_router(routing_input)  # [batch, num_heads]
        
        # Apply temperature scaling
        routing_weights = F.softmax(routing_scores / self.temperature, dim=-1)  # [batch, num_heads]
        
        # Determine tokens to route based on attention mask and context
        token_importance = self._compute_context_aware_importance(context_enhanced, attention_mask)
        
        # Select top-k tokens based on importance
        tokens_to_process = max(1, int(seq_len * 0.5))  # Default to 50%
        _, top_k_indices = torch.topk(token_importance, tokens_to_process, dim=-1, sorted=False)
        
        # Create mask for selected tokens
        token_mask = torch.zeros_like(token_importance, dtype=torch.bool)
        batch_indices = torch.arange(batch_size, device=hidden_states.device).unsqueeze(1).expand(-1, tokens_to_process)
        token_mask[batch_indices, top_k_indices] = True
        
        # Apply attention mask if provided
        if attention_mask is not None:
            token_mask = token_mask & attention_mask.bool()
        
        # Apply routing mask to hidden states
        routed_hidden_states = hidden_states.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        
        return routed_hidden_states, routing_weights
    
    def _compute_context_aware_importance(self, hidden_states: torch.Tensor, 
                                         attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute importance scores considering context."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Use variance across sequence as a measure of importance
        token_variance = torch.var(hidden_states, dim=1)  # [batch, hidden_size]
        
        # Use attention mask to prioritize non-padded tokens
        if attention_mask is not None:
            mask_importance = attention_mask.float()  # [batch, seq_len]
        else:
            mask_importance = torch.ones(batch_size, seq_len, device=hidden_states.device)
        
        # Combine context-aware measures
        token_importance = torch.norm(hidden_states, p=2, dim=-1)  # [batch, seq_len]
        token_importance = token_importance * mask_importance  # Prioritize non-masked tokens
        
        return token_importance


class ActivationRoutingFusion(nn.Module):
    """Fusion module that combines multiple activation routing strategies."""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Multiple routing strategies
        self.routers = nn.ModuleDict({
            'learned': LearnedActivationRouter(config, temperature=1.0),
            'adaptive': AdaptiveActivationRouter(config, temperature=1.0),
            'context_aware': ContextAwareActivationRouter(config, temperature=1.0)
        })
        
        # Router selector network to determine which router to use
        self.router_selector = nn.Linear(self.hidden_size, 3)  # 3 routers
        
        # Final output layer
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize fusion weights."""
        nn.init.xavier_uniform_(self.router_selector.weight)
        if self.router_selector.bias is not None:
            nn.init.zeros_(self.router_selector.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                layer_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Fuse multiple activation routing strategies.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            layer_idx: Current layer index
            
        Returns:
            Tuple of (fused_hidden_states, routing_weights, router_usage_stats)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get routing decisions from all routers
        learned_out, learned_weights = self.routers['learned'](hidden_states, layer_idx)
        adaptive_out, adaptive_weights = self.routers['adaptive'](hidden_states, layer_idx)
        context_out, context_weights = self.routers['context_aware'](hidden_states, attention_mask, layer_idx)
        
        # Select the best router based on input characteristics
        router_scores = self.router_selector(hidden_states.mean(dim=1))  # [batch, 3]
        router_probs = F.softmax(router_scores, dim=-1)  # [batch, 3]
        
        # Weighted combination of router outputs
        combined_hidden = (
            router_probs[:, 0:1].unsqueeze(-1) * learned_out +
            router_probs[:, 1:2].unsqueeze(-1) * adaptive_out +
            router_probs[:, 2:3].unsqueeze(-1) * context_out
        )
        
        # Average routing weights
        combined_weights = (
            router_probs[:, 0:1] * learned_weights +
            router_probs[:, 1:2] * adaptive_weights +
            router_probs[:, 2:3] * context_weights
        )
        
        # Apply output projection
        output = self.output_proj(combined_hidden)
        
        # Router usage statistics
        router_usage = {
            'learned_router_used': router_probs[:, 0].mean().item(),
            'adaptive_router_used': router_probs[:, 1].mean().item(),
            'context_aware_router_used': router_probs[:, 2].mean().item()
        }
        
        return output, combined_weights, router_usage