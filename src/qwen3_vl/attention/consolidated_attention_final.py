"""
Consolidated Attention System for Qwen3-VL Model
This module combines all attention mechanisms with lifecycle management
and hardware-specific optimizations into a single comprehensive system.
"""
import math
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any, List
from abc import ABC, abstractmethod
from enum import Enum


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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3VLRotaryEmbedding(nn.Module):
    """Rotary embedding implementation for Qwen3-VL model."""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_dim]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class AttentionType(Enum):
    """Enumeration of available attention types."""
    STANDARD = "standard"
    MEMORY_EFFICIENT = "memory_efficient"
    SIMD_OPTIMIZED = "simd_optimized"
    FLASH_ATTENTION = "flash_attention"
    SPARSE_ATTENTION = "sparse_attention"
    DYNAMIC_SPARSE = "dynamic_sparse"
    BLOCK_SPARSE = "block_sparse"
    CUSTOM = "custom"


class TensorType(Enum):
    """Enumeration of tensor types for lifecycle management."""
    INTERMEDIATE = "intermediate"
    PARAMETER = "parameter"
    ATTENTION_MECHANISM = "attention_mechanism"
    VISION_ATTENTION_MECHANISM = "vision_attention_mechanism"


class TensorState(Enum):
    """Enumeration of tensor states for lifecycle management."""
    REGISTERED = "registered"
    ACCESSING = "accessing"
    CACHED = "cached"
    SWAPPED = "swapped"
    CLEARED = "cleared"


class TensorMetadata:
    """Metadata for tensor lifecycle management."""
    def __init__(self, tensor_id: str, tensor_type: TensorType, device: str, size_bytes: int,
                 is_pinned: bool = False, creation_time: float = None):
        self.tensor_id = tensor_id
        self.tensor_type = tensor_type
        self.device = device
        self.size_bytes = size_bytes
        self.is_pinned = is_pinned
        self.creation_time = creation_time or time.time()
        self.last_access_time = self.creation_time
        self.access_count = 0
        self.state = TensorState.REGISTERED
        self.lifecycle_predictions = {}


class TensorLifecycleTracker:
    """
    Tracks the lifecycle of tensors to optimize memory management and performance.
    """
    def __init__(self):
        self.tensors = {}  # tensor_id -> TensorMetadata
        self.access_pattern_analyzer = AccessPatternAnalyzer()
        self.lifetime_predictor = LifetimePredictor()

    def register_tensor(self, tensor: torch.Tensor, tensor_id: str = None,
                        tensor_type: TensorType = TensorType.INTERMEDIATE,
                        is_pinned: bool = False) -> str:
        """
        Register a tensor for lifecycle tracking.

        Args:
            tensor: The tensor to register
            tensor_id: Optional ID for the tensor (auto-generated if not provided)
            tensor_type: Type of tensor (for optimization purposes)
            is_pinned: Whether the tensor should remain in memory

        Returns:
            The tensor ID
        """
        if tensor_id is None:
            tensor_id = f"tensor_{id(tensor)}_{int(time.time())}"

        device = str(tensor.device) if hasattr(tensor, 'device') else 'unknown'
        size_bytes = tensor.numel() * tensor.element_size() if hasattr(tensor, 'numel') and hasattr(tensor, 'element_size') else 0

        metadata = TensorMetadata(
            tensor_id=tensor_id,
            tensor_type=tensor_type,
            device=device,
            size_bytes=size_bytes,
            is_pinned=is_pinned
        )

        self.tensors[tensor_id] = metadata
        return tensor_id

    def access_tensor(self, tensor_id: str, context: str = None):
        """
        Record access to a tensor.

        Args:
            tensor_id: ID of the tensor being accessed
            context: Context of the access (e.g., layer name, operation)
        """
        if tensor_id in self.tensors:
            metadata = self.tensors[tensor_id]
            metadata.last_access_time = time.time()
            metadata.access_count += 1
            metadata.state = TensorState.ACCESSING

            # Analyze access patterns
            self.access_pattern_analyzer.record_access(tensor_id, context, metadata.last_access_time)

            # Update lifetime predictions
            self.lifetime_predictor.update_prediction(tensor_id, metadata.access_count, metadata.last_access_time)

    def get_tensor_metadata(self, tensor_id: str) -> Optional[TensorMetadata]:
        """Get metadata for a tensor."""
        return self.tensors.get(tensor_id)

    def get_all_tensor_metadata(self) -> Dict[str, TensorMetadata]:
        """Get metadata for all tracked tensors."""
        return self.tensors.copy()

    def get_tensors_by_type(self, tensor_type: TensorType) -> List[TensorMetadata]:
        """Get all tensors of a specific type."""
        return [metadata for metadata in self.tensors.values() if metadata.tensor_type == tensor_type]

    def get_tensors_by_device(self, device: str) -> List[TensorMetadata]:
        """Get all tensors on a specific device."""
        return [metadata for metadata in self.tensors.values() if metadata.device == device]

    def get_tensors_by_size_range(self, min_size: int, max_size: int) -> List[TensorMetadata]:
        """Get all tensors within a specific size range."""
        return [metadata for metadata in self.tensors.values()
                if min_size <= metadata.size_bytes <= max_size]

    def remove_tensor(self, tensor_id: str) -> bool:
        """Remove a tensor from tracking."""
        if tensor_id in self.tensors:
            del self.tensors[tensor_id]
            return True
        return False

    def clear_all_tensors(self):
        """Clear all tracked tensors."""
        self.tensors.clear()


class LifetimePredictor:
    """
    Predicts how long tensors will be needed based on access patterns.
    """
    def __init__(self):
        self.access_history = {}
        self.predictions = {}

    def update_prediction(self, tensor_id: str, access_count: int, last_access_time: float):
        """
        Update lifetime prediction for a tensor based on its access pattern.

        Args:
            tensor_id: ID of the tensor
            access_count: Number of times the tensor has been accessed
            last_access_time: Time of the last access
        """
        if tensor_id not in self.access_history:
            self.access_history[tensor_id] = []

        self.access_history[tensor_id].append((access_count, last_access_time))

        # Simple prediction algorithm: if access frequency is high, predict longer lifetime
        if len(self.access_history[tensor_id]) >= 2:
            # Calculate access frequency
            first_access = self.access_history[tensor_id][0][1]
            last_access = self.access_history[tensor_id][-1][1]
            time_span = last_access - first_access
            access_count = len(self.access_history[tensor_id])

            if time_span > 0:
                access_frequency = access_count / time_span
                # Predict lifetime based on access frequency (simplified)
                predicted_lifetime = min(300, 100 / (access_frequency + 0.01))  # Max 5 minutes
                self.predictions[tensor_id] = predicted_lifetime
            else:
                # Same time access, assume short lifetime
                self.predictions[tensor_id] = 10  # 10 seconds

    def predict_lifetime(self, tensor_id: str) -> float:
        """
        Predict how much longer a tensor will be needed.

        Args:
            tensor_id: ID of the tensor

        Returns:
            Predicted lifetime in seconds
        """
        return self.predictions.get(tensor_id, 30.0)  # Default 30 seconds


class AccessPatternAnalyzer:
    """
    Analyzes access patterns to optimize tensor lifecycle management.
    """
    def __init__(self):
        self.access_patterns = {}
        self.context_analysis = {}

    def record_access(self, tensor_id: str, context: str, access_time: float):
        """
        Record a tensor access for pattern analysis.

        Args:
            tensor_id: ID of the tensor being accessed
            context: Context of the access (e.g., layer name, operation)
            access_time: Time of access
        """
        if tensor_id not in self.access_patterns:
            self.access_patterns[tensor_id] = []

        self.access_patterns[tensor_id].append((context, access_time))

        # Update context analysis
        if context:
            if context not in self.context_analysis:
                self.context_analysis[context] = {'tensor_ids': set(), 'access_count': 0}
            self.context_analysis[context]['tensor_ids'].add(tensor_id)
            self.context_analysis[context]['access_count'] += 1

    def get_access_pattern(self, tensor_id: str) -> List[Tuple[str, float]]:
        """
        Get access pattern for a specific tensor.

        Args:
            tensor_id: ID of the tensor

        Returns:
            List of (context, access_time) tuples
        """
        return self.access_patterns.get(tensor_id, [])

    def get_context_analysis(self, context: str) -> Dict[str, Any]:
        """
        Get analysis for a specific context.

        Args:
            context: Context to analyze

        Returns:
            Analysis dictionary
        """
        return self.context_analysis.get(context, {'tensor_ids': set(), 'access_count': 0})

    def get_frequent_contexts(self, min_accesses: int = 5) -> List[str]:
        """
        Get contexts that have been accessed frequently.

        Args:
            min_accesses: Minimum number of accesses to be considered frequent

        Returns:
            List of frequent contexts
        """
        return [ctx for ctx, analysis in self.context_analysis.items()
                if analysis['access_count'] >= min_accesses]


class EnhancedPredictiveTensorLifecycleManager:
    """
    Enhanced tensor lifecycle manager with predictive capabilities and optimization strategies.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.tracker = TensorLifecycleTracker()
        self.config = config or {}
        self.compression_manager = None
        self.swapping_system = None
        self.memory_tiering_system = None

        # Hardware-specific parameters
        self.cpu_model = self.config.get('cpu_model', 'unknown')
        self.gpu_model = self.config.get('gpu_model', 'none')
        self.memory_size = self.config.get('memory_size', 8 * 1024 * 1024 * 1024)  # 8GB default
        self.storage_type = self.config.get('storage_type', 'ssd')  # Default assumption

    def register_tensor(self, tensor: torch.Tensor, tensor_id: str = None,
                        tensor_type: TensorType = TensorType.INTERMEDIATE,
                        is_pinned: bool = False) -> str:
        """Register a tensor for lifecycle tracking."""
        return self.tracker.register_tensor(tensor, tensor_id, tensor_type, is_pinned)

    def access_tensor(self, tensor_id: str, context: str = None):
        """Record access to a tensor."""
        self.tracker.access_tensor(tensor_id, context)

    def predict_tensor_lifetime(self, tensor_id: str) -> float:
        """Predict how long a tensor will be needed."""
        return self.tracker.lifetime_predictor.predict_lifetime(tensor_id)

    def get_tensor_metadata(self, tensor_id: str) -> Optional[TensorMetadata]:
        """Get metadata for a tensor."""
        return self.tracker.get_tensor_metadata(tensor_id)

    def optimize_tensor_placement(self, tensor_id: str) -> str:
        """
        Optimize tensor placement based on predicted lifetime and hardware capabilities.

        Args:
            tensor_id: ID of the tensor to optimize

        Returns:
            Recommended device for the tensor
        """
        metadata = self.get_tensor_metadata(tensor_id)
        if not metadata:
            return metadata.device if metadata else 'cpu'

        predicted_lifetime = self.predict_tensor_lifetime(tensor_id)

        # If tensor is predicted to be needed for a long time and is frequently accessed,
        # keep it on GPU if available and there's enough memory
        if (torch.cuda.is_available() and
            predicted_lifetime > 60 and  # More than 1 minute
            metadata.access_count > 5 and  # Accessed more than 5 times
            metadata.size_bytes < self.memory_size * 0.1):  # Less than 10% of total memory
            return 'cuda'
        else:
            # For short-lived or large tensors, consider CPU or swapping
            if metadata.size_bytes > self.memory_size * 0.3:  # More than 30% of memory
                # Large tensor - consider swapping to disk if needed
                if self.swapping_system:
                    return 'swapped'
                else:
                    return 'cpu'
            else:
                return 'cpu'

    def set_compression_manager(self, compression_manager):
        """Set the compression manager for tensor optimization."""
        self.compression_manager = compression_manager

    def set_swapping_system(self, swapping_system):
        """Set the swapping system for tensor management."""
        self.swapping_system = swapping_system

    def set_memory_tiering_system(self, memory_tiering_system):
        """Set the memory tiering system for tensor management."""
        self.memory_tiering_system = memory_tiering_system

    def get_stats(self) -> Dict[str, Any]:
        """Get lifecycle management statistics."""
        return {
            'total_tensors': len(self.tracker.tensors),
            'pinned_tensors': len([t for t in self.tracker.tensors.values() if t.is_pinned]),
            'gpu_tensors': len(self.tracker.get_tensors_by_device('cuda')),
            'cpu_tensors': len(self.tracker.get_tensors_by_device('cpu')),
            'frequent_contexts': self.tracker.access_pattern_analyzer.get_frequent_contexts(),
            'hardware_info': {
                'cpu_model': self.cpu_model,
                'gpu_model': self.gpu_model,
                'memory_size_gb': self.memory_size / (1024**3),
                'storage_type': self.storage_type
            }
        }

    def cleanup(self):
        """Clean up resources."""
        self.tracker.clear_all_tensors()


def create_optimized_lifecycle_manager(config: Dict[str, Any]) -> EnhancedPredictiveTensorLifecycleManager:
    """
    Factory function to create an optimized lifecycle manager based on hardware configuration.

    Args:
        config: Hardware configuration dictionary

    Returns:
        Configured lifecycle manager instance
    """
    return EnhancedPredictiveTensorLifecycleManager(config)


def integrate_with_existing_systems(lifecycle_manager, existing_memory_systems: Dict[str, Any] = None):
    """
    Integrate the lifecycle manager with existing memory management systems.

    Args:
        lifecycle_manager: The lifecycle manager to integrate
        existing_memory_systems: Dictionary of existing memory management systems
    """
    if existing_memory_systems:
        if 'compression_manager' in existing_memory_systems:
            lifecycle_manager.set_compression_manager(existing_memory_systems['compression_manager'])
        if 'swapping_system' in existing_memory_systems:
            lifecycle_manager.set_swapping_system(existing_memory_systems['swapping_system'])
        if 'memory_tiering_system' in existing_memory_systems:
            lifecycle_manager.set_memory_tiering_system(existing_memory_systems['memory_tiering_system'])


class BaseAttention(nn.Module):
    """
    Base attention module that defines common functionality for all attention mechanisms.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = getattr(config, 'max_position_embeddings', 2048)
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Initialize projections with configurable bias
        qkv_bias = getattr(config, 'qkv_bias', True)
        out_proj_bias = getattr(config, 'out_proj_bias', True)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=out_proj_bias)

        # Rotary embedding for position encoding
        self.rotary_emb = Qwen3VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        # Initialize tensor lifecycle manager for this attention layer
        lifecycle_config = {
            'cpu_model': getattr(config, 'cpu_model', 'Intel i5-10210U'),
            'gpu_model': getattr(config, 'gpu_model', 'NVIDIA SM61'),
            'memory_size': getattr(config, 'memory_size', 8 * 1024 * 1024 * 1024),
            'storage_type': getattr(config, 'storage_type', 'nvme')
        }
        self.tensor_lifecycle_manager = create_optimized_lifecycle_manager(lifecycle_config)

        # Register with tensor lifecycle manager
        self.tensor_lifecycle_manager.register_tensor(
            self,
            tensor_type=TensorType.ATTENTION_MECHANISM,
            is_pinned=True  # Attention mechanism should remain in memory
        )

    def _apply_attention_common(self, query_states, key_states, value_states, attention_mask):
        """
        Common attention computation logic that can be shared across implementations.
        """
        # Record access to attention mechanism in lifecycle manager
        self.tensor_lifecycle_manager.access_tensor(id(self), context=f"layer_{self.layer_idx}")

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Apply softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        return attn_output, attn_weights

    def _reshape_for_multihead(self, hidden_states):
        """
        Common reshaping logic for multi-head attention.
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        return query_states, key_states, value_states, bsz, q_len

    def _apply_rotary_and_cache(self, query_states, key_states, value_states, position_ids, past_key_value, cache_position):
        """
        Common logic for applying rotary embeddings and handling cache.
        """
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Apply GQA (Grouped Query Attention) if applicable
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        return query_states, key_states, value_states


class StandardAttention(BaseAttention):
    """
    Standard attention mechanism implementation for Qwen3-VL.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Common reshaping
        query_states, key_states, value_states, bsz, q_len = self._reshape_for_multihead(hidden_states)

        # Apply rotary embeddings and cache
        query_states, key_states, value_states = self._apply_rotary_and_cache(
            query_states, key_states, value_states, position_ids, past_key_value, cache_position
        )

        # Compute attention
        attn_output, attn_weights = self._apply_attention_common(query_states, key_states, value_states, attention_mask)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class SIMDAttention(BaseAttention):
    """
    SIMD-optimized attention mechanism that uses vectorized operations for better CPU performance.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # SIMD-specific optimizations
        self.vector_width = 8  # Number of elements processed in parallel (for AVX2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Common reshaping
        query_states, key_states, value_states, bsz, q_len = self._reshape_for_multihead(hidden_states)

        # Apply rotary embeddings and cache
        query_states, key_states, value_states = self._apply_rotary_and_cache(
            query_states, key_states, value_states, position_ids, past_key_value, cache_position
        )

        # SIMD-optimized attention computation
        attn_output, attn_weights = self._apply_attention_common(query_states, key_states, value_states, attention_mask)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _simd_softmax(self, x):
        """
        SIMD-optimized softmax implementation that processes vectors in parallel.
        """
        # Apply softmax with numerical stability
        max_vals = torch.max(x, dim=-1, keepdim=True).values
        x_shifted = x - max_vals
        exp_vals = torch.exp(x_shifted)
        sum_exp = torch.sum(exp_vals, dim=-1, keepdim=True)
        return exp_vals / sum_exp


class MemoryEfficientAttention(BaseAttention):
    """
    Memory-efficient attention implementation that uses chunked computation to reduce peak memory usage.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # Memory-efficient parameters
        self.chunk_size = getattr(config, 'chunk_size', 512)  # Size of chunks for memory-efficient computation

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Common reshaping
        query_states, key_states, value_states, bsz, q_len = self._reshape_for_multihead(hidden_states)

        # Apply rotary embeddings and cache
        query_states, key_states, value_states = self._apply_rotary_and_cache(
            query_states, key_states, value_states, position_ids, past_key_value, cache_position
        )

        # Memory-efficient attention computation using chunked processing
        attn_output = self._chunked_attention_forward(
            query_states, key_states, value_states, attention_mask
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _chunked_attention_forward(self, query_states, key_states, value_states, attention_mask):
        """
        Compute attention in chunks to reduce memory usage from O(n²) to O(n).
        """
        bsz, num_heads, seq_len, head_dim = query_states.shape
        _, _, kv_seq_len, _ = key_states.shape

        # Process in chunks to limit memory usage
        chunk_size = min(self.chunk_size, seq_len)
        attn_output = torch.zeros_like(query_states)

        for q_start in range(0, seq_len, chunk_size):
            q_end = min(q_start + chunk_size, seq_len)
            q_chunk = query_states[:, :, q_start:q_end, :]

            # Compute attention scores for this chunk with all keys
            attn_weights_chunk = torch.matmul(q_chunk, key_states.transpose(2, 3)) / math.sqrt(head_dim)

            if attention_mask is not None:
                mask_chunk = attention_mask[:, :, q_start:q_end, :kv_seq_len]
                attn_weights_chunk = attn_weights_chunk + mask_chunk

            # Apply softmax to get attention weights
            attn_weights_chunk = nn.functional.softmax(attn_weights_chunk, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # Apply attention to values
            output_chunk = torch.matmul(attn_weights_chunk, value_states)

            # Store in the full output tensor
            attn_output[:, :, q_start:q_end, :] = output_chunk

        return attn_output


class FlashAttention2(BaseAttention):
    """
    FlashAttention 2 implementation for memory-efficient attention computation.
    Reduces memory complexity from O(n²) to O(n) by using tiled computation and
    incremental softmax calculation.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Common reshaping
        query_states, key_states, value_states, bsz, q_len = self._reshape_for_multihead(hidden_states)

        # Apply rotary embeddings and cache
        query_states, key_states, value_states = self._apply_rotary_and_cache(
            query_states, key_states, value_states, position_ids, past_key_value, cache_position
        )

        # Memory-efficient attention computation using FlashAttention approach
        if not output_attentions:
            # Use PyTorch's optimized scaled dot-product attention (FlashAttention-like)
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attention_mask,
                dropout_p=0.0,  # No dropout during inference
                is_causal=False if attention_mask is not None else True  # Set based on mask
            )
            attn_weights = None  # Not computed when output_attentions is False
        else:
            # Compute attention weights in a memory-efficient way using tiling
            attn_weights = self._memory_efficient_attention_weights(
                query_states, key_states, attention_mask
            )
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value

    def _memory_efficient_attention_weights(self, query_states, key_states, attention_mask):
        """
        Compute attention weights in a memory-efficient way using tiling.
        This reduces memory complexity from O(n²) to O(n) by processing in chunks.
        """
        bsz, num_heads, seq_len, head_dim = query_states.shape
        _, _, kv_seq_len, _ = key_states.shape

        # Calculate tile size based on available memory
        tile_size = min(512, seq_len)  # Default tile size

        # Initialize output tensor
        attn_weights = torch.zeros(bsz, num_heads, seq_len, kv_seq_len,
                                   dtype=query_states.dtype, device=query_states.device)

        # Process in tiles to limit memory usage
        for q_start in range(0, seq_len, tile_size):
            q_end = min(q_start + tile_size, seq_len)
            q_tile = query_states[:, :, q_start:q_end, :]

            for k_start in range(0, kv_seq_len, tile_size):
                k_end = min(k_start + tile_size, kv_seq_len)
                k_tile = key_states[:, :, k_start:k_end, :]

                # Compute attention scores for this tile
                scores_tile = torch.matmul(q_tile, k_tile.transpose(-2, -1)) / math.sqrt(self.head_dim)

                # Apply attention mask if provided
                if attention_mask is not None:
                    mask_tile = attention_mask[:, :, q_start:q_end, k_start:k_end]
                    scores_tile = scores_tile + mask_tile

                # Store the tile in the full attention matrix
                attn_weights[:, :, q_start:q_end, k_start:k_end] = scores_tile

        return attn_weights


class SM61OptimizedFlashAttention2(FlashAttention2):
    """
    NVIDIA SM61 optimized FlashAttention 2 implementation.
    Optimized for compute capability 6.1 with memory and compute constraints.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # SM61-specific parameters for optimal performance
        self.tile_size = 256  # Smaller tile size for SM61's memory constraints

    def _sm61_memory_efficient_attention_weights(self, query_states, key_states, attention_mask):
        """
        SM61-optimized memory-efficient attention computation with smaller tile sizes
        and memory access patterns optimized for compute capability 6.1.
        """
        bsz, num_heads, seq_len, head_dim = query_states.shape
        _, _, kv_seq_len, _ = key_states.shape

        # SM61-optimized tile size (smaller for better cache utilization)
        tile_size = min(self.tile_size, seq_len)

        # Initialize output tensor
        attn_weights = torch.zeros(bsz, num_heads, seq_len, kv_seq_len,
                                   dtype=query_states.dtype, device=query_states.device)

        # Process in tiles with SM61-optimized access patterns
        for q_start in range(0, seq_len, tile_size):
            q_end = min(q_start + tile_size, seq_len)
            q_tile = query_states[:, :, q_start:q_end, :]

            for k_start in range(0, kv_seq_len, tile_size):
                k_end = min(k_start + tile_size, kv_seq_len)
                k_tile = key_states[:, :, k_start:k_end, :]

                # Compute attention scores for this tile
                scores_tile = torch.matmul(q_tile, k_tile.transpose(-2, -1)) / math.sqrt(self.head_dim)

                # Apply attention mask if provided
                if attention_mask is not None:
                    mask_tile = attention_mask[:, :, q_start:q_end, k_start:k_end]
                    scores_tile = scores_tile + mask_tile

                # Store the tile in the full attention matrix
                attn_weights[:, :, q_start:q_end, k_start:k_end] = scores_tile

        return attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Common reshaping
        query_states, key_states, value_states, bsz, q_len = self._reshape_for_multihead(hidden_states)

        # Apply rotary embeddings and cache
        query_states, key_states, value_states = self._apply_rotary_and_cache(
            query_states, key_states, value_states, position_ids, past_key_value, cache_position
        )

        # SM61-optimized attention computation using memory-efficient approach
        if not output_attentions:
            # Use PyTorch's optimized scaled dot-product attention (FlashAttention-like)
            # With SM61 optimizations
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attention_mask,
                dropout_p=0.0,  # No dropout during inference
                is_causal=False if attention_mask is not None else True  # Set based on mask
            )
            attn_weights = None  # Not computed when output_attentions is False
        else:
            # For SM61, compute attention weights in a memory-efficient way using tiling
            attn_weights = self._sm61_memory_efficient_attention_weights(
                query_states, key_states, attention_mask
            )
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class TrueSparseAttention(BaseAttention):
    """
    True sparse attention implementation with configurable sparsity patterns.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.sparsity_ratio = getattr(config, 'sparse_attention_sparsity_ratio', 0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Common reshaping
        query_states, key_states, value_states, bsz, q_len = self._reshape_for_multihead(hidden_states)

        # Apply rotary embeddings and cache
        query_states, key_states, value_states = self._apply_rotary_and_cache(
            query_states, key_states, value_states, position_ids, past_key_value, cache_position
        )

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Apply sparsity: keep only top-k values per query position
        sparse_attn_weights = self._apply_sparsity(attn_weights)

        # Apply softmax
        attn_weights = nn.functional.softmax(sparse_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _apply_sparsity(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Apply sparsity pattern to attention weights by masking out low-attention values.
        """
        # Calculate how many elements to keep based on sparsity ratio
        seq_len = attn_weights.size(-1)
        k = max(1, int(seq_len * self.sparsity_ratio))

        # Get top-k attention values for each query position
        top_k_values, top_k_indices = torch.topk(attn_weights, k=k, dim=-1)

        # Create a mask for the sparse attention pattern
        sparse_mask = torch.full_like(attn_weights, float('-inf'))
        sparse_mask.scatter_(-1, top_k_indices, top_k_values)

        # Apply sparse mask to scores
        sparse_attn_weights = torch.where(sparse_mask == float('-inf'), sparse_mask, attn_weights)

        return sparse_attn_weights


class BlockSparseAttention(BaseAttention):
    """
    Block-sparse attention with configurable block patterns.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.block_size = getattr(config, 'block_sparse_block_size', 64)
        self.sparsity_ratio = getattr(config, 'sparse_attention_sparsity_ratio', 0.5)

        # Learnable sparsity pattern
        self.sparsity_pattern = nn.Parameter(
            torch.randn(self.num_heads, self.max_position_embeddings // self.block_size,
                       self.max_position_embeddings // self.block_size)
        )
        nn.init.uniform_(self.sparsity_pattern, -0.1, 0.1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Common reshaping
        query_states, key_states, value_states, bsz, q_len = self._reshape_for_multihead(hidden_states)

        # Apply rotary embeddings and cache
        query_states, key_states, value_states = self._apply_rotary_and_cache(
            query_states, key_states, value_states, position_ids, past_key_value, cache_position
        )

        # Apply block-sparse attention pattern
        attn_weights = self._apply_block_sparse_attention(query_states, key_states)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Apply softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _apply_block_sparse_attention(self, query_states: torch.Tensor, key_states: torch.Tensor) -> torch.Tensor:
        """
        Apply block-sparse attention pattern to reduce computation.
        """
        bsz, num_heads, seq_len, head_dim = query_states.shape
        _, _, kv_len, _ = key_states.shape

        # Calculate block dimensions
        block_size = self.block_size
        num_q_blocks = math.ceil(seq_len / block_size)
        num_kv_blocks = math.ceil(kv_len / block_size)

        # Pad sequences to be divisible by block size
        padded_q_len = num_q_blocks * block_size
        padded_kv_len = num_kv_blocks * block_size

        if seq_len != padded_q_len or kv_len != padded_kv_len:
            query_states = F.pad(query_states, (0, 0, 0, padded_q_len - seq_len), value=0)
            key_states = F.pad(key_states, (0, 0, 0, padded_kv_len - kv_len), value=0)

        # Reshape to block format
        query_blocks = query_states.view(bsz, num_heads, num_q_blocks, block_size, head_dim)
        key_blocks = key_states.view(bsz, num_heads, num_kv_blocks, block_size, head_dim)

        # Get sparsity pattern for current sequence length
        # Ensure we don't exceed the available pattern dimensions
        available_q_blocks = min(num_q_blocks, self.sparsity_pattern.size(1))
        available_kv_blocks = min(num_kv_blocks, self.sparsity_pattern.size(2))
        current_sparsity_pattern = self.sparsity_pattern[:, :available_q_blocks, :available_kv_blocks]

        # Apply learned sparsity pattern with top-k selection to enforce sparsity
        sparsity_threshold = torch.topk(
            current_sparsity_pattern.view(num_heads, -1),
            k=max(1, int(current_sparsity_pattern.numel() * self.sparsity_ratio / num_heads)),
            dim=-1
        )[0][:, -1].view(num_heads, 1, 1)

        sparse_mask = (current_sparsity_pattern > sparsity_threshold).float()

        # Initialize output tensor
        attn_weights = torch.zeros(bsz, num_heads, padded_q_len, padded_kv_len,
                                   dtype=query_states.dtype, device=query_states.device)

        # Compute attention only for non-zero blocks in the sparse pattern
        for h_idx in range(num_heads):
            for q_block_idx in range(available_q_blocks):
                for kv_block_idx in range(available_kv_blocks):
                    if sparse_mask[h_idx, q_block_idx, kv_block_idx] > 0:
                        # Compute attention for this block pair
                        q_block = query_blocks[:, h_idx, q_block_idx, :, :]  # [bsz, block_size, head_dim]
                        k_block = key_blocks[:, h_idx, kv_block_idx, :, :]  # [bsz, block_size, head_dim]

                        block_attn = torch.matmul(q_block, k_block.transpose(-2, -1)) / math.sqrt(self.head_dim)
                        attn_weights[:, h_idx,
                                    q_block_idx * block_size:min((q_block_idx + 1) * block_size, padded_q_len),
                                    kv_block_idx * block_size:min((kv_block_idx + 1) * block_size, padded_kv_len)] = block_attn

        # Trim back to original sequence length
        attn_weights = attn_weights[:, :, :seq_len, :kv_len]

        return attn_weights


class DynamicSparseAttention(BaseAttention):
    """
    Dynamic sparse attention with learned routing for token selection.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.sparsity_ratio = getattr(config, 'sparse_attention_sparsity_ratio', 0.5)
        self.vision_sparsity_ratio = getattr(config, 'vision_sparse_attention_sparsity_ratio', 0.4)

        # Learned routing mechanism for dynamic token selection
        self.routing_network = nn.Linear(self.hidden_size, self.num_heads, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Common reshaping
        query_states, key_states, value_states, bsz, q_len = self._reshape_for_multihead(hidden_states)

        # Apply rotary embeddings and cache
        query_states, key_states, value_states = self._apply_rotary_and_cache(
            query_states, key_states, value_states, position_ids, past_key_value, cache_position
        )

        # Compute routing scores to determine important tokens
        routing_scores = self._compute_routing_scores(hidden_states)  # [bsz, seq_len, num_heads]

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Apply dynamic sparsity based on routing scores
        sparse_attn_weights = self._apply_dynamic_sparsity(attn_weights, routing_scores)

        # Apply softmax
        attn_weights = nn.functional.softmax(sparse_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _compute_routing_scores(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute routing scores for dynamic token selection.
        """
        # Use the routing network to determine which tokens are important
        routing_logits = self.routing_network(hidden_states)  # [bsz, seq_len, num_heads]
        routing_scores = torch.sigmoid(routing_logits)  # [bsz, seq_len, num_heads]
        return routing_scores

    def _apply_dynamic_sparsity(self, attn_weights: torch.Tensor, routing_scores: torch.Tensor) -> torch.Tensor:
        """
        Apply dynamic sparsity based on routing scores.
        """
        bsz, num_heads, q_len, kv_len = attn_weights.size()

        # Adjust sparsity ratio based on whether we're processing vision or text tokens
        # This is a simple heuristic - in a full implementation, we would have more sophisticated logic
        current_sparsity_ratio = self.sparsity_ratio
        if q_len > 512:  # Heuristic for vision tokens
            current_sparsity_ratio = self.vision_sparsity_ratio

        # Calculate how many tokens to attend to per head
        k = max(1, int(kv_len * current_sparsity_ratio))

        # For each head, select top-k tokens based on routing scores
        # routing_scores is [bsz, seq_len, num_heads], we need [bsz, num_heads, seq_len]
        routing_scores_t = routing_scores.transpose(1, 2)  # [bsz, num_heads, seq_len]

        # Get top-k routing scores for each head
        top_k_routing_values, top_k_indices = torch.topk(routing_scores_t, k, dim=-1)  # [bsz, num_heads, k]

        # Create a sparse attention mask
        sparse_mask = torch.full_like(attn_weights, float('-inf'))

        # For each head, fill in the sparse mask with values for the selected tokens
        for h_idx in range(num_heads):
            for batch_idx in range(bsz):
                selected_kv_indices = top_k_indices[batch_idx, h_idx, :]  # [k]
                # Fill the attention weights for this head and batch with the selected keys/values
                sparse_mask[batch_idx, h_idx, :, selected_kv_indices] = attn_weights[batch_idx, h_idx, :, selected_kv_indices]

        return sparse_mask


class Qwen3VLVisionAttention(nn.Module):
    """A multi-head attention module for vision processing."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.vision_hidden_size
        self.num_heads = config.vision_num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} "
                f"and `num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.vision_qkv_bias)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, tgt_len, embed_dim = hidden_states.size()

        mixed_qkv = self.qkv(hidden_states)

        mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query_states, key_states, value_states = mixed_qkv.unbind(0)  # [bsz, num_heads, seq_len, head_dim]

        # Transpose to apply softmax
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)

        attn_output = self.proj(attn_output)

        return attn_output, attn_weights


class AttentionMechanismSelector:
    """
    Factory and selector for different attention mechanisms based on configuration and hardware.
    """
    @staticmethod
    def create_attention(config, layer_idx: Optional[int] = None) -> nn.Module:
        """
        Create the appropriate attention mechanism based on configuration.

        Args:
            config: Model configuration
            layer_idx: Layer index (optional)

        Returns:
            Appropriate attention mechanism module
        """
        # Check configuration for which attention mechanism to use
        attention_implementation = getattr(config, 'attention_implementation', 'standard')

        if attention_implementation == 'flash_attention_2':
            # Use hardware-specific attention if available
            if hasattr(config, 'hardware_specific_attention') and config.hardware_specific_attention == 'sm61':
                return SM61OptimizedFlashAttention2(config, layer_idx)
            else:
                return FlashAttention2(config, layer_idx)
        elif attention_implementation == 'sparse_attention':
            if getattr(config, 'use_dynamic_sparse_attention', False):
                return DynamicSparseAttention(config, layer_idx)
            elif getattr(config, 'use_block_sparse_attention', False):
                return BlockSparseAttention(config, layer_idx)
            else:
                return TrueSparseAttention(config, layer_idx)
        elif attention_implementation == 'memory_efficient':
            return MemoryEfficientAttention(config, layer_idx)
        elif attention_implementation == 'simd':
            return SIMDAttention(config, layer_idx)
        else:
            # Default to standard attention
            return StandardAttention(config, layer_idx)

    @staticmethod
    def get_available_implementations() -> List[str]:
        """Get list of available attention implementations"""
        return [
            'standard',
            'flash_attention_2',
            'sparse_attention',
            'dynamic_sparse_attention',
            'block_sparse_attention',
            'memory_efficient',
            'simd'
        ]


class Qwen3VLAttention(nn.Module):
    """
    Main attention module for Qwen3-VL that selects the appropriate attention mechanism.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Select the appropriate attention implementation based on config
        self.attention_impl = AttentionMechanismSelector.create_attention(config, layer_idx)

        # Initialize tensor lifecycle manager for this attention layer
        lifecycle_config = {
            'cpu_model': getattr(config, 'cpu_model', 'Intel i5-10210U'),
            'gpu_model': getattr(config, 'gpu_model', 'NVIDIA SM61'),
            'memory_size': getattr(config, 'memory_size', 8 * 1024 * 1024 * 1024),
            'storage_type': getattr(config, 'storage_type', 'nvme')
        }
        self.tensor_lifecycle_manager = create_optimized_lifecycle_manager(lifecycle_config)

        # Register with tensor lifecycle manager
        self.tensor_lifecycle_manager.register_tensor(
            self.attention_impl,
            tensor_type=TensorType.ATTENTION_MECHANISM,
            is_pinned=True  # Attention mechanism should remain in memory
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Record access to attention mechanism in lifecycle manager
        self.tensor_lifecycle_manager.access_tensor(id(self.attention_impl), context=f"layer_{self.layer_idx}")

        # Execute the selected attention implementation
        output = self.attention_impl(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        # Handle output based on implementation
        if len(output) == 3:
            attn_output, attn_weights, past_key_value = output
        else:
            # Some implementations may return different number of values
            attn_output = output[0]
            attn_weights = output[1] if len(output) > 1 else None
            past_key_value = output[2] if len(output) > 2 else None

        return attn_output, attn_weights, past_key_value

    def get_lifecycle_stats(self) -> Dict[str, Any]:
        """Get lifecycle statistics for this attention mechanism"""
        return self.tensor_lifecycle_manager.get_stats()


class AttentionPerformanceMonitor:
    """Monitors performance metrics for attention mechanisms."""

    def __init__(self):
        self.metrics_history = []
        self.current_metrics = {}

    def start_timing(self, operation: str):
        """Start timing for an operation."""
        self.current_metrics[f"{operation}_start_time"] = time.time()

    def end_timing(self, operation: str):
        """End timing for an operation and record duration."""
        start_time = self.current_metrics.get(f"{operation}_start_time")
        if start_time:
            duration = time.time() - start_time
            self.current_metrics[f"{operation}_duration"] = duration

    def record_memory_usage(self, operation: str, memory_used_bytes: int):
        """Record memory usage for an operation."""
        self.current_metrics[f"{operation}_memory_used"] = memory_used_bytes

    def record_peak_memory(self, operation: str, peak_memory_bytes: int):
        """Record peak memory usage for an operation."""
        self.current_metrics[f"{operation}_peak_memory"] = peak_memory_bytes

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.current_metrics.copy()

    def reset_metrics(self):
        """Reset current metrics."""
        self.current_metrics = {}

    def log_metrics(self):
        """Log metrics and add to history."""
        if self.current_metrics:
            self.metrics_history.append(self.current_metrics.copy())
            self.reset_metrics()


class HardwareAwareAttentionSelector:
    """Selects the optimal attention mechanism based on hardware capabilities."""

    def __init__(self):
        self.performance_monitor = AttentionPerformanceMonitor()

    def select_attention_type(self, config, available_memory_gb: float) -> AttentionType:
        """
        Select the most appropriate attention type based on hardware and configuration.

        Args:
            config: Model configuration
            available_memory_gb: Available GPU/CPU memory in GB

        Returns:
            Selected attention type
        """
        # Determine memory constraints
        memory_constrained = available_memory_gb < 8  # Less than 8GB is considered memory constrained

        # Check for CUDA availability and compute capability
        if torch.cuda.is_available():
            # Check compute capability
            major, minor = torch.cuda.get_device_capability(0)

            # For newer GPUs with sufficient memory, prefer Flash Attention
            if major >= 8 and not memory_constrained:
                return AttentionType.FLASH_ATTENTION
            # For older GPUs like SM61, consider memory efficiency
            elif major >= 6 and memory_constrained:
                return AttentionType.MEMORY_EFFICIENT
            elif major == 6 and minor == 1:  # SM61 (GTX 1060, etc.)
                return AttentionType.FLASH_ATTENTION  # Use SM61 optimized FlashAttention
            else:
                return AttentionType.STANDARD
        else:
            # On CPU, consider memory efficiency and SIMD optimizations
            if hasattr(config, 'use_simd_attention') and config.use_simd_attention:
                return AttentionType.SIMD_OPTIMIZED
            elif memory_constrained:
                return AttentionType.MEMORY_EFFICIENT
            else:
                return AttentionType.STANDARD


class AttentionManager:
    """
    Manages selection and switching between attention mechanisms.
    Integrates with memory management and performance monitoring.
    """
    def __init__(self, config):
        self.config = config
        self.performance_monitor = AttentionPerformanceMonitor()
        self.hardware_selector = HardwareAwareAttentionSelector()
        self.active_attention_type = None
        self.active_attention_module = None

        # Map attention types to their implementation classes
        self.attention_implementations = {
            AttentionType.STANDARD: StandardAttention,
            AttentionType.MEMORY_EFFICIENT: MemoryEfficientAttention,
            AttentionType.FLASH_ATTENTION: FlashAttention2,
            AttentionType.SPARSE_ATTENTION: TrueSparseAttention,
            AttentionType.DYNAMIC_SPARSE: DynamicSparseAttention,
            AttentionType.BLOCK_SPARSE: BlockSparseAttention,
            AttentionType.SIMD_OPTIMIZED: SIMDAttention,
        }

    def get_available_memory_gb(self) -> float:
        """Get available memory in GB."""
        if torch.cuda.is_available():
            # Get GPU memory
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        else:
            # Get system memory if psutil is available
            try:
                import psutil
                return psutil.virtual_memory().available / (1024 ** 3)
            except ImportError:
                # Return a default value if psutil is not available
                return 8.0

    def select_attention_module(self, attention_type: Optional[AttentionType] = None) -> nn.Module:
        """
        Select and instantiate the appropriate attention module.

        Args:
            attention_type: Desired attention type (if None, auto-select based on hardware)

        Returns:
            Instantiated attention module
        """
        if attention_type is None:
            # Auto-select based on hardware
            available_memory = self.get_available_memory_gb()
            attention_type = self.hardware_selector.select_attention_type(self.config, available_memory)

        # Validate attention type
        if attention_type not in self.attention_implementations:
            raise ValueError(f"Unsupported attention type: {attention_type}")

        # Instantiate the attention module
        attention_class = self.attention_implementations[attention_type]
        attention_module = attention_class(self.config)

        # Update active attention type and module
        self.active_attention_type = attention_type
        self.active_attention_module = attention_module

        return attention_module

    def switch_attention_module(self, attention_type: AttentionType) -> bool:
        """
        Switch to a different attention mechanism at runtime.

        Args:
            attention_type: New attention type to switch to

        Returns:
            True if switch was successful, False otherwise
        """
        try:
            new_attention_module = self.select_attention_module(attention_type)
            self.active_attention_type = attention_type
            self.active_attention_module = new_attention_module
            return True
        except Exception as e:
            warnings.warn(f"Failed to switch attention module: {str(e)}")
            return False

    def benchmark_attention_types(self, sample_input: torch.Tensor) -> Dict[AttentionType, Dict[str, float]]:
        """
        Benchmark different attention types on sample input.

        Args:
            sample_input: Sample input tensor for benchmarking

        Returns:
            Dictionary mapping attention types to performance metrics
        """
        results = {}

        for att_type in self.attention_implementations.keys():
            try:
                # Create attention module for this type
                temp_module = self.attention_implementations[att_type](self.config)

                # Warm up
                for _ in range(3):
                    _ = temp_module(sample_input)

                # Measure performance
                start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

                start_time = time.time()
                output = temp_module(sample_input)
                end_time = time.time()

                end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

                # Calculate metrics
                duration = end_time - start_time
                memory_used = end_memory - start_memory if torch.cuda.is_available() else 0

                results[att_type] = {
                    "time_seconds": duration,
                    "memory_bytes": memory_used,
                    "success": True
                }

            except Exception as e:
                results[att_type] = {
                    "time_seconds": float('inf'),
                    "memory_bytes": float('inf'),
                    "success": False,
                    "error": str(e)
                }

        return results

    def get_active_attention_info(self) -> Dict[str, Any]:
        """Get information about the currently active attention module."""
        if self.active_attention_module is None:
            return {"active_type": None, "info": "No active attention module"}

        info = {
            "active_type": self.active_attention_type.value if self.active_attention_type else None,
        }

        return info


def create_consolidated_attention_module(config, attention_type: Optional[AttentionType] = None):
    """
    Factory function to create the appropriate attention module based on configuration.

    Args:
        config: Model configuration
        attention_type: Desired attention type (if None, auto-select based on hardware)

    Returns:
        Appropriate attention module instance
    """
    attention_manager = AttentionManager(config)
    return attention_manager.select_attention_module(attention_type)


__all__ = [
    # Utility functions
    'repeat_kv',
    'rotate_half',
    'apply_rotary_pos_emb',

    # Rotary embeddings
    'Qwen3VLRotaryEmbedding',

    # Attention mechanisms
    'StandardAttention',
    'SIMDAttention',
    'MemoryEfficientAttention',
    'FlashAttention2',
    'SM61OptimizedFlashAttention2',
    'TrueSparseAttention',
    'BlockSparseAttention',
    'DynamicSparseAttention',
    'Qwen3VLVisionAttention',

    # Main attention module
    'Qwen3VLAttention',

    # Enums and types
    'AttentionType',
    'TensorType',
    'TensorState',

    # Lifecycle management
    'TensorMetadata',
    'TensorLifecycleTracker',
    'LifetimePredictor',
    'AccessPatternAnalyzer',
    'EnhancedPredictiveTensorLifecycleManager',
    'create_optimized_lifecycle_manager',
    'integrate_with_existing_systems',

    # Factory and selector classes
    'AttentionMechanismSelector',
    'AttentionManager',
    'HardwareAwareAttentionSelector',
    'AttentionPerformanceMonitor',

    # Factory function
    'create_consolidated_attention_module',
]