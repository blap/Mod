"""
KV Cache optimization strategies for Qwen3-VL model.
Implements low-rank approximation, sliding window attention, and vision-language task optimizations.
NO INT8 quantization as per updated plan.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict
import math


class LowRankKVCache:
    """
    Low-rank approximation for KV cache compression.
    """
    def __init__(self, num_layers: int, num_heads: int, head_dim: int, max_seq_len: int, rank: int, device: torch.device):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.rank = rank
        self.device = device

        # Initialize low-rank decomposed KV caches
        # K: (num_layers, num_heads, max_seq_len, rank) * (num_layers, num_heads, rank, head_dim)
        self.k_left = torch.zeros((num_layers, num_heads, max_seq_len, rank), device=device, dtype=torch.float16)
        self.k_right = torch.zeros((num_layers, num_heads, rank, head_dim), device=device, dtype=torch.float16)

        # V: (num_layers, num_heads, max_seq_len, rank) * (num_layers, num_heads, rank, head_dim)
        self.v_left = torch.zeros((num_layers, num_heads, max_seq_len, rank), device=device, dtype=torch.float16)
        self.v_right = torch.zeros((num_layers, num_heads, rank, head_dim), device=device, dtype=torch.float16)

        # Track current sequence length for each layer
        self.current_seq_len = [0] * num_layers

        # Use half precision to save memory
        self.dtype = torch.float16

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int,
               cache_position: Optional[torch.LongTensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the low-rank KV cache with new key and value states.
        Expected key_states and value_states format: [batch, num_heads, seq_len, head_dim]
        """
        # key_states and value_states are in format [batch, num_heads, seq_len, head_dim]
        # For now, assume batch_size = 1 and take the first batch
        if key_states.size(0) != 1 or value_states.size(0) != 1:
            raise ValueError(f"LowRankKVCache expects batch_size=1, got {key_states.size(0)} for keys and {value_states.size(0)} for values")

        k_to_cache = key_states.squeeze(0)  # [num_heads, seq_len, head_dim]
        v_to_cache = value_states.squeeze(0)  # [num_heads, seq_len, head_dim]

        if cache_position is not None and cache_position.numel() > 0:
            # Update specific positions in cache
            # cache_position is [seq_positions] - indices to update in the sequence dimension
            seq_len = k_to_cache.size(1)  # [num_heads, seq_len, head_dim] after squeeze, so seq_len is dim 1

            if seq_len != cache_position.size(0):
                raise ValueError(f"Number of key/value states ({seq_len}) must match number of cache positions ({cache_position.size(0)})")

            # For position-specific updates, decompose each sequence element separately
            for head_idx in range(self.num_heads):
                k_head = k_to_cache[head_idx]  # [seq_len, head_dim]
                v_head = v_to_cache[head_idx]  # [seq_len, head_dim]

                for i, pos in enumerate(cache_position):
                    pos_idx = pos.item()

                    # Ensure position is within bounds
                    if pos_idx >= self.max_seq_len:
                        raise ValueError(f"Cache position {pos_idx} exceeds maximum sequence length {self.max_seq_len}")

                    # Update this specific position with the corresponding sequence element
                    k_single = k_head[i:i+1, :]  # [1, head_dim] - single sequence element for this head
                    v_single = v_head[i:i+1, :]  # [1, head_dim] - single sequence element for this head

                    # For a single vector, create a simple low-rank approximation
                    # Use outer product approximation: for vector v, find u, s, vh such that v ≈ u @ s @ vh
                    # For a single vector, we can think of it as rank-1: v ≈ (v) @ (1) @ (unit vector)
                    # Or we can pad to match the rank dimension
                    if self.rank == 1:
                        # If rank is 1, store the vector directly in left and use 1 in right
                        k_left_val = k_single  # [1, head_dim] - but we need [1, rank] so take first element or pad
                        if k_single.size(1) >= 1:
                            k_left_val = k_single[:, :1]  # [1, 1]
                        else:
                            k_left_val = torch.cat([k_single, torch.zeros(1, 1 - k_single.size(1),
                                                 device=k_single.device, dtype=k_single.dtype)], dim=1)

                        k_right_val = torch.zeros(1, self.head_dim, device=k_single.device, dtype=k_single.dtype)
                        if self.head_dim >= 1:
                            k_right_val[0, :k_single.size(1)] = k_single[0, :]
                    else:
                        # For higher ranks, we can store the vector in the first position and pad with zeros
                        # A simple approach: store the vector in the first row of left and identity in right
                        k_left_val = torch.zeros(1, self.rank, device=k_single.device, dtype=k_single.dtype)
                        k_left_val[0, 0] = torch.norm(k_single, p=2)  # Store the norm in first position

                        k_right_val = torch.zeros(self.rank, self.head_dim, device=k_single.device, dtype=k_single.dtype)
                        if k_single.size(1) > 0:
                            # Store normalized vector in the first row of right
                            k_normalized = k_single / (torch.norm(k_single, p=2, keepdim=True) + 1e-12)
                            k_right_val[0, :k_single.size(1)] = k_normalized[0, :]

                    # Store in cache using half precision to save memory
                    self.k_left[layer_idx, head_idx, pos_idx:pos_idx+1, :] = k_left_val.to(self.dtype)
                    # For position-specific updates, we should only update the right matrix if it improves reconstruction
                    # For now, we'll update it for all heads, but in practice might want to be more selective
                    self.k_right[layer_idx, head_idx, :, :] = k_right_val.to(self.dtype)

                    # Similarly for value states
                    if self.rank == 1:
                        # If rank is 1, store the vector directly in left and use 1 in right
                        v_left_val = v_single  # [1, head_dim] - but we need [1, rank] so take first element or pad
                        if v_single.size(1) >= 1:
                            v_left_val = v_single[:, :1]  # [1, 1]
                        else:
                            v_left_val = torch.cat([v_single, torch.zeros(1, 1 - v_single.size(1),
                                                 device=v_single.device, dtype=v_single.dtype)], dim=1)

                        v_right_val = torch.zeros(1, self.head_dim, device=v_single.device, dtype=v_single.dtype)
                        if self.head_dim >= 1:
                            v_right_val[0, :v_single.size(1)] = v_single[0, :]
                    else:
                        # For higher ranks, we can store the vector in the first position and pad with zeros
                        v_left_val = torch.zeros(1, self.rank, device=v_single.device, dtype=v_single.dtype)
                        v_left_val[0, 0] = torch.norm(v_single, p=2)  # Store the norm in first position

                        v_right_val = torch.zeros(self.rank, self.head_dim, device=v_single.device, dtype=v_single.dtype)
                        if v_single.size(1) > 0:
                            # Store normalized vector in the first row of right
                            v_normalized = v_single / (torch.norm(v_single, p=2, keepdim=True) + 1e-12)
                            v_right_val[0, :v_single.size(1)] = v_normalized[0, :]

                    self.v_left[layer_idx, head_idx, pos_idx:pos_idx+1, :] = v_left_val.to(self.dtype)
                    self.v_right[layer_idx, head_idx, :, :] = v_right_val.to(self.dtype)

            # Update the current sequence length to accommodate the highest position
            max_pos = cache_position.max().item()
            self.current_seq_len[layer_idx] = max(self.current_seq_len[layer_idx], max_pos + 1)
        else:
            # Append to the end of cache (empty cache_position is treated as None)
            seq_len = k_to_cache.size(1)  # [num_heads, seq_len, head_dim] after squeeze, so seq_len is dim 1
            start_pos = self.current_seq_len[layer_idx]
            end_pos = start_pos + seq_len

            # Ensure we don't exceed max sequence length
            if end_pos > self.max_seq_len:
                raise ValueError(f"Sequence length {end_pos} exceeds maximum {self.max_seq_len}")

            # For a proper low-rank implementation, we would decompose the tensors
            # For this implementation, we'll simulate low-rank decomposition by using SVD for each head
            for head_idx in range(self.num_heads):
                k_head = k_to_cache[head_idx]  # [seq_len, head_dim]
                v_head = v_to_cache[head_idx]  # [seq_len, head_dim]

                # Perform SVD to get low-rank approximation
                # K: U_k * S_k * V_k^T, but we only keep top rank components
                try:
                    # Use torch.linalg.svd which is more numerically stable than torch.svd
                    U_k, S_k, Vh_k = torch.linalg.svd(k_head, full_matrices=False)
                    U_v, S_v, Vh_v = torch.linalg.svd(v_head, full_matrices=False)

                    # Truncate to rank
                    rank_k = min(self.rank, len(S_k))
                    rank_v = min(self.rank, len(S_v))

                    U_k = U_k[:, :rank_k]
                    S_k = S_k[:rank_k]
                    Vh_k = Vh_k[:rank_k, :]

                    U_v = U_v[:, :rank_v]
                    S_v = S_v[:rank_v]
                    Vh_v = Vh_v[:rank_v, :]

                    # Store in low-rank form: left = U*sqrt(S), right = sqrt(S)*V^T
                    k_left_comp = U_k @ torch.diag(torch.sqrt(S_k))
                    k_right_comp = torch.diag(torch.sqrt(S_k)) @ Vh_k

                    v_left_comp = U_v @ torch.diag(torch.sqrt(S_v))
                    v_right_comp = torch.diag(torch.sqrt(S_v)) @ Vh_v

                    # Store in cache using half precision to save memory
                    self.k_left[layer_idx, head_idx, start_pos:end_pos, :k_left_comp.shape[1]] = k_left_comp.to(self.dtype)
                    self.k_right[layer_idx, head_idx, :k_right_comp.shape[0], :] = k_right_comp.to(self.dtype)

                    self.v_left[layer_idx, head_idx, start_pos:end_pos, :v_left_comp.shape[1]] = v_left_comp.to(self.dtype)
                    self.v_right[layer_idx, head_idx, :v_right_comp.shape[0], :] = v_right_comp.to(self.dtype)

                except RuntimeError:
                    # If SVD fails (e.g., matrix is too small), just store zeros
                    # In a real implementation, we would handle this more gracefully
                    # For now, we'll just continue with zeros
                    pass

            self.current_seq_len[layer_idx] = end_pos

        # Reconstruct full KV tensors from low-rank approximation
        k_reconstructed_list = []
        v_reconstructed_list = []

        for head_idx in range(self.num_heads):
            k_left_part = self.k_left[layer_idx, head_idx, :self.current_seq_len[layer_idx], :].to(torch.float32)
            k_right_part = self.k_right[layer_idx, head_idx, :, :].to(torch.float32)
            k_reconstructed = k_left_part @ k_right_part  # [current_seq_len, head_dim]

            v_left_part = self.v_left[layer_idx, head_idx, :self.current_seq_len[layer_idx], :].to(torch.float32)
            v_right_part = self.v_right[layer_idx, head_idx, :, :].to(torch.float32)
            v_reconstructed = v_left_part @ v_right_part  # [current_seq_len, head_dim]

            k_reconstructed_list.append(k_reconstructed.unsqueeze(0))  # [1, current_seq_len, head_dim]
            v_reconstructed_list.append(v_reconstructed.unsqueeze(0))  # [1, current_seq_len, head_dim]

        k_full = torch.cat(k_reconstructed_list, dim=0)  # [num_heads, current_seq_len, head_dim]
        v_full = torch.cat(v_reconstructed_list, dim=0)  # [num_heads, current_seq_len, head_dim]

        k_full = k_full.unsqueeze(0)  # [1, num_heads, current_seq_len, head_dim]
        v_full = v_full.unsqueeze(0)  # [1, num_heads, current_seq_len, head_dim]

        return k_full, v_full

    def get_seq_length(self, layer_idx: int) -> int:
        """Get the current sequence length for a given layer."""
        return self.current_seq_len[layer_idx]

    def reset(self, layer_idx: Optional[int] = None):
        """Reset the cache for a specific layer or all layers."""
        if layer_idx is not None:
            self.current_seq_len[layer_idx] = 0
        else:
            self.current_seq_len = [0] * self.num_layers

    def compress_with_svd(self, layer_idx: int):
        """
        Compress the stored tensors using SVD to maintain low-rank structure.
        """
        # Get current full tensors
        k_left_full = self.k_left[layer_idx, :, :self.current_seq_len[layer_idx], :]
        k_right_full = self.k_right[layer_idx, :, :, :]
        v_left_full = self.v_left[layer_idx, :, :self.current_seq_len[layer_idx], :]
        v_right_full = self.v_right[layer_idx, :, :, :]

        # Reconstruct the full tensors first
        k_full_reconstructed = torch.zeros((self.num_heads, self.current_seq_len[layer_idx], self.head_dim),
                                          device=self.device, dtype=torch.float32)
        v_full_reconstructed = torch.zeros((self.num_heads, self.current_seq_len[layer_idx], self.head_dim),
                                          device=self.device, dtype=torch.float32)

        for head_idx in range(self.num_heads):
            k_left_part = k_left_full[head_idx, :self.current_seq_len[layer_idx], :].to(torch.float32)
            k_right_part = k_right_full[head_idx, :, :].to(torch.float32)
            k_full_reconstructed[head_idx] = k_left_part @ k_right_part

            v_left_part = v_left_full[head_idx, :self.current_seq_len[layer_idx], :].to(torch.float32)
            v_right_part = v_right_full[head_idx, :, :].to(torch.float32)
            v_full_reconstructed[head_idx] = v_left_part @ v_right_part

        # Now decompose again to ensure low-rank structure
        for head_idx in range(self.num_heads):
            k_head = k_full_reconstructed[head_idx]  # [seq_len, head_dim]
            v_head = v_full_reconstructed[head_idx]  # [seq_len, head_dim]

            try:
                # Use torch.linalg.svd which is more numerically stable than torch.svd
                U_k, S_k, Vh_k = torch.linalg.svd(k_head, full_matrices=False)
                U_v, S_v, Vh_v = torch.linalg.svd(v_head, full_matrices=False)

                # Truncate to rank
                rank_k = min(self.rank, len(S_k))
                rank_v = min(self.rank, len(S_v))

                U_k = U_k[:, :rank_k]
                S_k = S_k[:rank_k]
                Vh_k = Vh_k[:rank_k, :]

                U_v = U_v[:, :rank_v]
                S_v = S_v[:rank_v]
                Vh_v = Vh_v[:rank_v, :]

                # Store in low-rank form: left = U*sqrt(S), right = sqrt(S)*V^T
                k_left_comp = U_k @ torch.diag(torch.sqrt(S_k))
                k_right_comp = torch.diag(torch.sqrt(S_k)) @ Vh_k

                v_left_comp = U_v @ torch.diag(torch.sqrt(S_v))
                v_right_comp = torch.diag(torch.sqrt(S_v)) @ Vh_v

                # Store in cache using half precision to save memory
                self.k_left[layer_idx, head_idx, :self.current_seq_len[layer_idx], :k_left_comp.shape[1]] = k_left_comp.to(self.dtype)
                self.k_right[layer_idx, head_idx, :k_right_comp.shape[0], :] = k_right_comp.to(self.dtype)

                self.v_left[layer_idx, head_idx, :self.current_seq_len[layer_idx], :v_left_comp.shape[1]] = v_left_comp.to(self.dtype)
                self.v_right[layer_idx, head_idx, :v_right_comp.shape[0], :] = v_right_comp.to(self.dtype)

            except RuntimeError:
                # If SVD fails (e.g., matrix is too small), just continue with existing values
                pass


class SlidingWindowKVCache:
    """
    KV cache with sliding window mechanism to limit cache size.
    """
    def __init__(self, num_layers: int, num_heads: int, head_dim: int, max_seq_len: int, window_size: int, device: torch.device):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.device = device

        # Initialize KV caches with sliding window size
        self.k_cache = torch.zeros((num_layers, num_heads, window_size, head_dim), device=device, dtype=torch.float16)
        self.v_cache = torch.zeros((num_layers, num_heads, window_size, head_dim), device=device, dtype=torch.float16)

        # Track current position in the sliding window for each layer
        self.current_pos = [0] * num_layers
        # Track the actual sequence length for each layer
        self.actual_seq_len = [0] * num_layers

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int,
               cache_position: Optional[torch.LongTensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the sliding window KV cache with new key and value states.
        Expected key_states and value_states format: [batch, num_heads, seq_len, head_dim]
        """
        if key_states.size(0) != 1 or value_states.size(0) != 1:
            raise ValueError(f"SlidingWindowKVCache expects batch_size=1, got {key_states.size(0)} for keys and {value_states.size(0)} for values")

        k_to_cache = key_states.squeeze(0)  # [num_heads, seq_len, head_dim]
        v_to_cache = value_states.squeeze(0)  # [num_heads, seq_len, head_dim]

        seq_len = k_to_cache.size(1)  # The sequence length dimension

        # Update the sliding window cache
        for head_idx in range(self.num_heads):
            k_head = k_to_cache[head_idx]  # [seq_len, head_dim]
            v_head = v_to_cache[head_idx]  # [seq_len, head_dim]

            # Calculate where to store the new values in the sliding window
            start_pos = self.current_pos[layer_idx]
            end_pos = start_pos + seq_len

            if end_pos <= self.window_size:
                # No wraparound needed
                self.k_cache[layer_idx, head_idx, start_pos:end_pos, :] = k_head.to(torch.float16)
                self.v_cache[layer_idx, head_idx, start_pos:end_pos, :] = v_head.to(torch.float16)
            else:
                # Wraparound needed - store in two parts
                first_part_size = self.window_size - start_pos
                self.k_cache[layer_idx, head_idx, start_pos:, :] = k_head[:first_part_size, :].to(torch.float16)
                self.v_cache[layer_idx, head_idx, start_pos:, :] = v_head[:first_part_size, :].to(torch.float16)

                if first_part_size < seq_len:
                    # Store remaining values at the beginning of the window
                    remaining = seq_len - first_part_size
                    self.k_cache[layer_idx, head_idx, :remaining, :] = k_head[first_part_size:, :].to(torch.float16)
                    self.v_cache[layer_idx, head_idx, :remaining, :] = v_head[first_part_size:, :].to(torch.float16)

        # Update positions
        self.current_pos[layer_idx] = (self.current_pos[layer_idx] + seq_len) % self.window_size
        self.actual_seq_len[layer_idx] += seq_len

        # Return the full cache (not just the sliding window) for attention computation
        # For sliding window, we return only the most recent tokens up to window_size
        effective_len = min(self.actual_seq_len[layer_idx], self.window_size)
        start_idx = max(0, self.actual_seq_len[layer_idx] - self.window_size)

        k_full = self.k_cache[layer_idx, :, :effective_len, :].to(torch.float32).unsqueeze(0)  # [1, num_heads, effective_len, head_dim]
        v_full = self.v_cache[layer_idx, :, :effective_len, :].to(torch.float32).unsqueeze(0)  # [1, num_heads, effective_len, head_dim]

        return k_full, v_full

    def get_seq_length(self, layer_idx: int) -> int:
        """Get the current sequence length for a given layer."""
        return min(self.actual_seq_len[layer_idx], self.window_size)

    def reset(self, layer_idx: Optional[int] = None):
        """Reset the cache for a specific layer or all layers."""
        if layer_idx is not None:
            self.current_pos[layer_idx] = 0
            self.actual_seq_len[layer_idx] = 0
        else:
            self.current_pos = [0] * self.num_layers
            self.actual_seq_len = [0] * self.num_layers


class HybridKVCache:
    """
    Hybrid KV cache that combines low-rank approximation with sliding window attention.
    """
    def __init__(self, num_layers: int, num_heads: int, head_dim: int, max_seq_len: int,
                 low_rank_rank: int, window_size: int, device: torch.device):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.low_rank_rank = low_rank_rank
        self.window_size = window_size
        self.device = device

        # Use both low-rank and sliding window mechanisms
        self.low_rank_cache = LowRankKVCache(num_layers, num_heads, head_dim, max_seq_len, low_rank_rank, device)
        self.sliding_window_cache = SlidingWindowKVCache(num_layers, num_heads, head_dim, max_seq_len, window_size, device)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int,
               cache_position: Optional[torch.LongTensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the hybrid KV cache with new key and value states.
        """
        # First apply sliding window to limit sequence length
        k_windowed, v_windowed = self.sliding_window_cache.update(key_states, value_states, layer_idx, cache_position)

        # Then apply low-rank approximation
        k_low_rank, v_low_rank = self.low_rank_cache.update(k_windowed, v_windowed, layer_idx, cache_position)

        return k_low_rank, v_low_rank

    def get_seq_length(self, layer_idx: int) -> int:
        """Get the current sequence length for a given layer."""
        return self.low_rank_cache.get_seq_length(layer_idx)

    def reset(self, layer_idx: Optional[int] = None):
        """Reset the cache for a specific layer or all layers."""
        self.low_rank_cache.reset(layer_idx)
        self.sliding_window_cache.reset(layer_idx)


class VisionLanguageKVCache(nn.Module):
    """
    Optimized KV cache for vision-language tasks with specialized handling for multimodal inputs.
    """
    def __init__(self, config, layer_idx: Optional[int] = None, use_low_rank: bool = True, window_size: int = 1024):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

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

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        self.rotary_emb = Qwen3VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        # For vision-language tasks, use specialized cache
        self.use_low_rank = use_low_rank
        self.window_size = window_size
        self.kv_cache = None

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
        bsz, q_len, _ = hidden_states.size()

        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to multi-head format
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle past key values for caching
        if use_cache:
            # Initialize cache if not already done
            if self.kv_cache is None:
                device = hidden_states.device
                if self.use_low_rank:
                    # Use hybrid cache for vision-language tasks
                    self.kv_cache = HybridKVCache(
                        num_layers=1,
                        num_heads=self.num_key_value_heads,
                        head_dim=self.head_dim,
                        max_seq_len=self.config.max_position_embeddings,
                        low_rank_rank=min(64, self.head_dim),
                        window_size=self.window_size,
                        device=device
                    )
                else:
                    # Use sliding window only
                    self.kv_cache = SlidingWindowKVCache(
                        num_layers=1,
                        num_heads=self.num_key_value_heads,
                        head_dim=self.head_dim,
                        max_seq_len=self.config.max_position_embeddings,
                        window_size=self.window_size,
                        device=device
                    )

            key_states, value_states = self.kv_cache.update(
                key_states, value_states, 0, cache_position
            )
            # For compatibility with existing caching system, return the cache object as past_key_value
            past_key_value = self.kv_cache
        elif past_key_value is not None:
            # If past_key_value is a custom cache object, use it
            if hasattr(past_key_value, 'update'):
                key_states, value_states = past_key_value.update(
                    key_states, value_states, 0, cache_position
                )
            else:
                # Standard caching mechanism
                if past_key_value is not None:
                    key_states = torch.cat([past_key_value[0], key_states], dim=-2)
                    value_states = torch.cat([past_key_value[1], value_states], dim=-2)

        # Repeat keys and values for GQA (Grouped Query Attention) if applicable
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        import math
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class OptimizedKVCachingAttention(nn.Module):
    """
    Attention mechanism with optimized KV caching strategies (low-rank and sliding window).
    """
    def __init__(self, config, layer_idx: Optional[int] = None,
                 use_low_rank: bool = True,
                 window_size: Optional[int] = 1024,
                 low_rank_rank: int = 64,
                 cache_strategy: str = "hybrid",  # Options: "low_rank", "sliding_window", "hybrid"
                 device: Optional[torch.device] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # Don't force a specific device, let it match the model's device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        self.rotary_emb = Qwen3VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        # Initialize optimized KV cache based on strategy
        self.use_low_rank = use_low_rank
        self.window_size = window_size
        self.low_rank_rank = low_rank_rank
        self.cache_strategy = cache_strategy
        self.kv_cache = None

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
        bsz, q_len, _ = hidden_states.size()

        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to multi-head format
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle past key values with optimized caching
        if use_cache:
            # Initialize cache if not already done
            if self.kv_cache is None:
                device = hidden_states.device
                if self.cache_strategy == "low_rank":
                    self.kv_cache = LowRankKVCache(
                        num_layers=1,
                        num_heads=self.num_key_value_heads,
                        head_dim=self.head_dim,
                        max_seq_len=self.config.max_position_embeddings,
                        rank=min(self.low_rank_rank, self.head_dim),
                        device=device
                    )
                elif self.cache_strategy == "sliding_window":
                    self.kv_cache = SlidingWindowKVCache(
                        num_layers=1,
                        num_heads=self.num_key_value_heads,
                        head_dim=self.head_dim,
                        max_seq_len=self.config.max_position_embeddings,
                        window_size=self.window_size,
                        device=device
                    )
                elif self.cache_strategy == "hybrid":
                    self.kv_cache = HybridKVCache(
                        num_layers=1,
                        num_heads=self.num_key_value_heads,
                        head_dim=self.head_dim,
                        max_seq_len=self.config.max_position_embeddings,
                        low_rank_rank=min(self.low_rank_rank, self.head_dim),
                        window_size=self.window_size,
                        device=device
                    )

            key_states, value_states = self.kv_cache.update(
                key_states, value_states, 0, cache_position
            )
            # For compatibility with existing caching system, return the cache object as past_key_value
            past_key_value = self.kv_cache
        elif past_key_value is not None:
            # If past_key_value is a custom cache object, use it
            if hasattr(past_key_value, 'update'):
                key_states, value_states = past_key_value.update(
                    key_states, value_states, 0, cache_position
                )
            else:
                # Standard caching mechanism
                if past_key_value is not None:
                    key_states = torch.cat([past_key_value[0], key_states], dim=-2)
                    value_states = torch.cat([past_key_value[1], value_states], dim=-2)

        # Repeat keys and values for GQA (Grouped Query Attention) if applicable
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        import math
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Helper functions for rotary embeddings and tensor operations
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
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)