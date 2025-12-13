"""
Cross-layer memory sharing for intermediate representation reuse
in the Qwen3-VL architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
from src.qwen3_vl.config import Qwen3VLConfig


class LayerMemoryBank(nn.Module):
    """
    Memory bank for storing and reusing intermediate representations across layers.
    """
    def __init__(self, config: Qwen3VLConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.max_memory_entries = getattr(config, 'max_memory_entries', 64)
        self.memory_reuse_threshold = getattr(config, 'memory_reuse_threshold', 0.8)

        # Memory storage for intermediate representations
        self.memory_keys = nn.Parameter(
            torch.empty(self.max_memory_entries, self.hidden_size),
            requires_grad=True
        )
        self.memory_values = nn.Parameter(
            torch.empty(self.max_memory_entries, self.hidden_size),
            requires_grad=True
        )

        # Initialize parameters
        nn.init.normal_(self.memory_keys, mean=0.0, std=0.02)
        nn.init.normal_(self.memory_values, mean=0.0, std=0.02)

        # Projections for memory operations
        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # Gate mechanism to control memory reuse
        self.gate_network = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # Layer normalization
        self.norm = nn.LayerNorm(self.hidden_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Track memory usage statistics
        self.register_buffer('memory_usage_count', torch.zeros(self.max_memory_entries, dtype=torch.long))
        self.register_buffer('memory_timestamp', torch.zeros(self.max_memory_entries, dtype=torch.long))

    def forward(
        self,
        current_hidden_states: torch.Tensor,
        layer_input: torch.Tensor,
        reuse_gate: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Store current hidden states and retrieve relevant memories.

        Args:
            current_hidden_states: Current layer's output of shape (batch_size, seq_len, hidden_size)
            layer_input: Input to the current layer of shape (batch_size, seq_len, hidden_size)
            reuse_gate: Optional gate tensor to control memory reuse

        Returns:
            Tuple of (updated_hidden_states, reuse_gate)
        """
        batch_size, seq_len, hidden_size = current_hidden_states.shape

        # Project inputs for memory operations
        queries = self.query_proj(layer_input)
        keys = self.key_proj(current_hidden_states)
        values = self.value_proj(current_hidden_states)

        # Compute similarity with stored memory keys
        memory_similarity = torch.matmul(queries, self.memory_keys.t())  # [batch, seq, max_entries]
        memory_attention = F.softmax(memory_similarity, dim=-1)

        # Retrieve from memory
        retrieved_memory = torch.matmul(memory_attention, self.memory_values)  # [batch, seq, hidden_size]

        # Compute reuse gate if not provided
        if reuse_gate is None:
            combined_repr = torch.cat([layer_input, retrieved_memory], dim=-1)
            reuse_gate = self.gate_network(combined_repr)  # [batch, seq, 1]

        # Apply memory reuse with gating
        updated_hidden_states = (1 - reuse_gate) * current_hidden_states + reuse_gate * retrieved_memory

        # Update memory bank with current representations
        self._update_memory(keys.mean(dim=1), values.mean(dim=1))  # Average over sequence dimension

        # Apply normalization and dropout
        updated_hidden_states = self.norm(updated_hidden_states)
        updated_hidden_states = self.dropout(updated_hidden_states)

        return updated_hidden_states, reuse_gate

    def _update_memory(self, key_batch: torch.Tensor, value_batch: torch.Tensor):
        """
        Update the memory bank with new key-value pairs.

        Args:
            key_batch: Batch of keys of shape (batch_size, hidden_size)
            value_batch: Batch of values of shape (batch_size, hidden_size)
        """
        batch_size = key_batch.size(0)

        for i in range(batch_size):
            # Find the least recently used memory slot
            least_used_idx = torch.argmin(self.memory_usage_count + self.memory_timestamp * 0.1)

            # Update the memory slot
            with torch.no_grad():
                self.memory_keys[least_used_idx] = key_batch[i]
                self.memory_values[least_used_idx] = value_batch[i]
                self.memory_usage_count[least_used_idx] += 1
                self.memory_timestamp[least_used_idx] = self.memory_timestamp.max() + 1


class CrossLayerMemorySharing(nn.Module):
    """
    Manages memory sharing across transformer layers.
    """
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size

        # Create memory banks for each layer
        self.memory_banks = nn.ModuleList([
            LayerMemoryBank(config, layer_idx) for layer_idx in range(self.num_hidden_layers)
        ])

        # Cross-layer attention for memory sharing
        self.cross_layer_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=max(1, self.hidden_size // 64),
            batch_first=True
        )

        # Memory routing network to decide which layers to share with
        self.memory_router = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.num_hidden_layers),
            nn.Softmax(dim=-1)
        )

        # Layer normalization
        self.norm = nn.LayerNorm(self.hidden_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        layer_input: Optional[torch.Tensor] = None,
        previous_memory: Optional[Dict[int, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Apply cross-layer memory sharing to the hidden states.

        Args:
            hidden_states: Current layer's hidden states of shape (batch_size, seq_len, hidden_size)
            layer_idx: Index of the current layer
            layer_input: Input to the current layer (optional)
            previous_memory: Dictionary of memories from previous layers (optional)

        Returns:
            Tuple of (updated_hidden_states, updated_memory_dict)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        if layer_input is None:
            layer_input = hidden_states

        # Use the current layer's memory bank
        memory_bank = self.memory_banks[layer_idx]
        updated_states, reuse_gate = memory_bank(hidden_states, layer_input)

        # If we have previous memories, apply cross-layer attention
        if previous_memory is not None and len(previous_memory) > 0:
            # Collect representations from previous layers
            prev_reprs = []
            prev_layer_indices = []
            for prev_idx, prev_repr in previous_memory.items():
                if prev_idx < layer_idx:  # Only use previous layers
                    prev_reprs.append(prev_repr)
                    prev_layer_indices.append(prev_idx)

            if prev_reprs:
                # Stack previous representations
                stacked_prev = torch.stack(prev_reprs, dim=1)  # [batch, num_prev_layers, seq, hidden]

                # Apply cross-layer attention
                # Reshape for attention: [batch*seq, num_prev_layers, hidden]
                stacked_prev_2d = stacked_prev.transpose(1, 2).reshape(-1, len(prev_reprs), hidden_size)
                current_2d = updated_states.reshape(-1, hidden_size).unsqueeze(1)

                # Cross-attention
                attended_memory, _ = self.cross_layer_attention(
                    query=current_2d,
                    key=stacked_prev_2d,
                    value=stacked_prev_2d
                )

                # Reshape back
                attended_memory = attended_memory.squeeze(1).reshape(batch_size, seq_len, hidden_size)

                # Apply memory routing to decide how much to incorporate
                memory_routing_weights = self.memory_router(updated_states.mean(dim=1))  # [batch, num_layers]
                current_layer_weight = memory_routing_weights[:, layer_idx]

                # Blend current representation with attended memory
                alpha = current_layer_weight.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]
                updated_states = alpha * updated_states + (1 - alpha) * attended_memory

        # Update normalization
        updated_states = self.norm(updated_states)
        updated_states = self.dropout(updated_states)

        # Update memory dictionary
        if previous_memory is None:
            previous_memory = {}
        previous_memory[layer_idx] = updated_states

        return updated_states, previous_memory


class AdaptiveMemorySharingController(nn.Module):
    """
    Controller that adaptively decides when and how to share memory across layers.
    """
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers

        # Network to assess when memory sharing is beneficial
        self.sharing_decision_network = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, 2),  # [share_prob, reuse_intensity]
            nn.Softmax(dim=-1)
        )

        # Layer-specific sharing parameters
        self.layer_sharing_params = nn.Parameter(
            torch.randn(self.num_hidden_layers, 2) * 0.1  # [share_enabled, reuse_intensity]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int
    ) -> Tuple[bool, float]:
        """
        Decide whether to enable memory sharing and at what intensity.

        Args:
            hidden_states: Current hidden states of shape (batch_size, seq_len, hidden_size)
            layer_idx: Index of the current layer

        Returns:
            Tuple of (should_share_memory, reuse_intensity)
        """
        # Assess current state for sharing decision
        state_summary = hidden_states.mean(dim=1)  # Average across sequence
        sharing_decision = self.sharing_decision_network(state_summary)  # [batch, 2]

        # Average across batch to get single decision
        avg_decision = sharing_decision.mean(dim=0)  # [2]
        share_prob = avg_decision[0]
        reuse_intensity = avg_decision[1]

        # Use layer-specific parameters to modulate decision
        layer_params = torch.sigmoid(self.layer_sharing_params[layer_idx])  # [2]
        share_enabled = layer_params[0] > 0.5 and share_prob > 0.5

        # Modulate reuse intensity with layer-specific parameters
        final_reuse_intensity = reuse_intensity * layer_params[1]

        return share_enabled, final_reuse_intensity.item()


class CrossLayerMemoryManager(nn.Module):
    """
    Main module that manages cross-layer memory sharing with adaptive control.
    """
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config

        # Initialize components
        self.memory_sharing = CrossLayerMemorySharing(config)
        self.adaptive_controller = AdaptiveMemorySharingController(config)

        # Memory retention across the forward pass
        self.memory_cache = {}

        # Memory sharing enabled flag
        self.sharing_enabled = getattr(config, 'enable_cross_layer_memory_sharing', True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        layer_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-layer memory sharing to the hidden states.

        Args:
            hidden_states: Current layer's hidden states of shape (batch_size, seq_len, hidden_size)
            layer_idx: Index of the current layer
            layer_input: Input to the current layer (optional)

        Returns:
            Updated hidden states with memory sharing applied
        """
        if not self.sharing_enabled:
            return hidden_states

        # Check if memory sharing is advisable for this layer and input
        should_share, reuse_intensity = self.adaptive_controller(hidden_states, layer_idx)

        if should_share:
            # Apply cross-layer memory sharing
            updated_states, self.memory_cache = self.memory_sharing(
                hidden_states,
                layer_idx,
                layer_input,
                self.memory_cache
            )

            # Apply reuse intensity scaling
            updated_states = reuse_intensity * updated_states + (1 - reuse_intensity) * hidden_states
        else:
            # No memory sharing, just store current representation
            self.memory_cache[layer_idx] = hidden_states
            updated_states = hidden_states

        return updated_states

    def reset_memory(self):
        """Reset the memory cache."""
        self.memory_cache = {}

    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage."""
        stats = {
            'num_cached_layers': len(self.memory_cache),
            'sharing_enabled': self.sharing_enabled
        }
        return stats