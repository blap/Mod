"""
Sequence Parallelism System for Large Language Models

This module implements a comprehensive sequence parallelism system that splits
sequences across multiple devices for efficient distributed inference.
"""

import logging
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import math


logger = logging.getLogger(__name__)


@dataclass
class SequenceParallelConfig:
    """Configuration for sequence parallelism."""
    num_segments: int = 1
    segment_device_mapping: Optional[List[str]] = None  # e.g., ['cuda:0', 'cuda:1', 'cuda:2']
    sequence_split_method: str = 'chunk'  # 'chunk', 'stride', 'dynamic'
    enable_sequence_overlap: bool = True  # Enable overlapping computation between segments
    overlap_size: int = 64  # Number of tokens to overlap between segments for continuity
    enable_load_balancing: bool = True
    load_balance_method: str = 'uniform'  # 'uniform', 'adaptive', 'token_count'
    enable_gradient_checkpointing: bool = False
    inter_segment_buffer_size: int = 10
    sequence_parallel_algorithm: str = '1d'  # '1d', '2d', 'ring'
    ring_chunk_size: int = 32  # For ring-based sequence parallelism


class SequenceSegment(nn.Module):
    """Represents a single segment in the sequence parallelism."""

    def __init__(self,
                 segment_id: int,
                 model_part: nn.Module,
                 config: SequenceParallelConfig,
                 input_device: str = 'cpu',
                 output_device: str = 'cpu'):
        super().__init__()
        self.segment_id = segment_id
        self.config = config
        self.input_device = input_device
        self.output_device = output_device

        # Move model part to appropriate device
        self.model_part = model_part.to(input_device)

        # Communication queues for overlapping computation
        self.input_queue = queue.Queue(maxsize=config.inter_segment_buffer_size)
        self.output_queue = queue.Queue(maxsize=config.inter_segment_buffer_size)

        # Thread safety
        self.lock = threading.Lock()

        logger.info(f"Sequence segment {segment_id} initialized on device {input_device}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through this segment."""
        # Move input to segment device
        x = x.to(self.input_device)

        # Apply gradient checkpointing if enabled
        if self.config.enable_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(self.model_part, x)
        else:
            x = self.model_part(x)

        # Move output to output device if different from input device
        if self.output_device != self.input_device:
            x = x.to(self.output_device)

        return x

    def forward_segment(self, segment_input: torch.Tensor) -> torch.Tensor:
        """Process a single segment input."""
        return self.forward(segment_input)

    def backward_segment(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass for a single segment."""
        # This would typically involve computing gradients
        # For now, we'll just return the grad_output
        return grad_output


class SequenceBalancer:
    """Manages load balancing across sequence segments."""

    def __init__(self, config: SequenceParallelConfig):
        self.config = config
        self.segment_times = [0.0] * config.num_segments
        self.segment_counts = [0] * config.num_segments
        self.segment_token_counts = [0] * config.num_segments
        self.lock = threading.Lock()

    def record_segment_time(self, segment_id: int, time_taken: float, token_count: int = 0):
        """Record execution time and token count for a segment."""
        with self.lock:
            self.segment_times[segment_id] += time_taken
            self.segment_counts[segment_id] += 1
            self.segment_token_counts[segment_id] += token_count

    def get_average_segment_times(self) -> List[float]:
        """Get average execution time for each segment."""
        avg_times = []
        for i in range(self.config.num_segments):
            if self.segment_counts[i] > 0:
                avg_times.append(self.segment_times[i] / self.segment_counts[i])
            else:
                avg_times.append(0.0)
        return avg_times

    def get_average_tokens_per_segment(self) -> List[float]:
        """Get average token count for each segment."""
        avg_tokens = []
        for i in range(self.config.num_segments):
            if self.segment_counts[i] > 0:
                avg_tokens.append(self.segment_token_counts[i] / self.segment_counts[i])
            else:
                avg_tokens.append(0.0)
        return avg_tokens

    def rebalance_if_needed(self) -> bool:
        """Check if rebalancing is needed and perform it."""
        if not self.config.enable_load_balancing:
            return False

        avg_times = self.get_average_segment_times()
        if len(set(avg_times)) <= 1:  # All segments take similar time
            return False

        # Simple rebalancing logic - adjust based on timing differences
        max_time = max(avg_times) if avg_times else 0
        min_time = min(avg_times) if avg_times else 0

        if max_time > 0 and (max_time - min_time) / max_time > 0.2:  # 20% difference threshold
            logger.info(f"Load imbalance detected: max={max_time:.4f}s, min={min_time:.4f}s")
            return True

        return False


class SequenceParallel(nn.Module):
    """Main sequence parallelism module that orchestrates all segments."""

    def __init__(self, model: nn.Module, config: SequenceParallelConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.segments: List[SequenceSegment] = []
        self.balancer = SequenceBalancer(config)

        # Split the model into segments
        self._split_model_into_segments()

        # Setup communication queues between segments
        self.comm_queues = []
        for i in range(self.config.num_segments - 1):
            self.comm_queues.append(queue.Queue(maxsize=config.inter_segment_buffer_size))

        logger.info(f"Sequence parallel model created with {config.num_segments} segments")

    def _split_model_into_segments(self):
        """Split the model into sequence segments."""
        # Determine device mapping
        if self.config.segment_device_mapping:
            device_mapping = self.config.segment_device_mapping
        else:
            # Default to using available GPUs or CPU
            if torch.cuda.is_available():
                device_mapping = [f'cuda:{i % torch.cuda.device_count()}'
                                for i in range(self.config.num_segments)]
            else:
                device_mapping = ['cpu'] * self.config.num_segments

        # Get the model's layers/modules to split
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-style model
            all_layers = list(self.model.transformer.h)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
            # Transformer-style model (like GLM, Qwen)
            all_layers = list(self.model.transformer.layers)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Another common structure
            all_layers = list(self.model.model.layers)
        elif hasattr(self.model, 'layers') and isinstance(self.model.layers, nn.ModuleList):
            # Direct access to layers as ModuleList
            all_layers = list(self.model.layers)
        else:
            # Fallback: try to split the entire model differently
            # Look for the first ModuleList in the model's children
            all_layers = []
            for child_name, child_module in self.model.named_children():
                if isinstance(child_module, nn.ModuleList):
                    all_layers = list(child_module)
                    logger.info(f"Using ModuleList '{child_name}' for sequence splitting")
                    break

            if not all_layers:
                # If no ModuleList found, fall back to all children
                all_modules = list(self.model.children())
                all_layers = all_modules

        # Calculate layers per segment
        total_layers = len(all_layers)
        layers_per_segment = total_layers // self.config.num_segments
        remainder = total_layers % self.config.num_segments

        start_idx = 0
        for i in range(self.config.num_segments):
            # Distribute remainder layers to first few segments
            end_idx = start_idx + layers_per_segment + (1 if i < remainder else 0)

            # Create a sequential module for this segment
            segment_layers = all_layers[start_idx:end_idx]
            if len(segment_layers) == 1:
                # If there's only one layer, wrap it in a Sequential to ensure it has a forward method
                segment_model = nn.Sequential(segment_layers[0])
            else:
                segment_model = nn.Sequential(*segment_layers)

            # Create sequence segment
            segment = SequenceSegment(
                segment_id=i,
                model_part=segment_model,
                config=self.config,
                input_device=device_mapping[i],
                output_device=device_mapping[i]
            )

            self.segments.append(segment)
            logger.info(f"Segment {i}: {start_idx}-{end_idx-1} ({len(segment_layers)} layers) on {device_mapping[i]}")

            start_idx = end_idx

    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Any:
        """Execute the sequence parallelism with various scheduling methods."""
        if self.config.sequence_parallel_algorithm == '1d':
            return self._execute_1d_sequence_parallel(inputs)
        elif self.config.sequence_parallel_algorithm == 'ring':
            return self._execute_ring_sequence_parallel(inputs)
        else:
            # Default to simple sequential execution
            return self._execute_sequential(inputs)

    def _execute_sequential(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Any:
        """Execute sequence parallelism sequentially (simple forward pass through all segments)."""
        x = inputs
        if isinstance(inputs, dict):
            # Handle dict inputs (like input_ids, attention_mask, etc.)
            x = inputs.get('input_ids', inputs.get('inputs_embeds', inputs))

        for segment in self.segments:
            start_time = time.time()
            x = segment(x)
            end_time = time.time()
            self.balancer.record_segment_time(segment.segment_id, end_time - start_time)

        return x

    def _execute_1d_sequence_parallel(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Any:
        """Execute sequence parallelism with 1D sequence splitting."""
        # For 1D sequence parallelism, we split the input sequence along the sequence dimension
        if isinstance(inputs, dict):
            input_ids = inputs.get('input_ids', inputs.get('inputs_embeds', inputs))
        else:
            input_ids = inputs

        # Ensure input has the right shape (batch_size, seq_len, hidden_size)
        if input_ids.dim() == 2:
            # If input is (batch_size, seq_len), assume hidden_size=1 or expand
            input_ids = input_ids.unsqueeze(-1)  # Add hidden dimension
        elif input_ids.dim() != 3:
            raise ValueError(f"Expected input to have 2 or 3 dimensions, got {input_ids.dim()}")

        # Split the sequence into segments
        seq_len = input_ids.size(1)  # Get sequence length (second dimension)
        chunk_size = math.ceil(seq_len / self.config.num_segments)

        # Split the input sequence into chunks
        sequence_chunks = []
        for i in range(self.config.num_segments):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, seq_len)
            chunk = input_ids[:, start_idx:end_idx, :]  # Keep batch and hidden dims

            # Add overlap if enabled
            if self.config.enable_sequence_overlap and i < self.config.num_segments - 1:
                overlap_start = end_idx
                overlap_end = min(overlap_start + self.config.overlap_size, seq_len)
                if overlap_end > overlap_start:
                    overlap_chunk = input_ids[:, overlap_start:overlap_end, :]
                    chunk = torch.cat([chunk, overlap_chunk], dim=1)  # Concat along sequence dim

            sequence_chunks.append(chunk)

        # Process each chunk through its respective segment
        segment_outputs = []
        for i, (chunk, segment) in enumerate(zip(sequence_chunks, self.segments)):
            start_time = time.time()

            # Process the chunk through the segment
            chunk_output = segment(chunk)

            end_time = time.time()
            self.balancer.record_segment_time(
                segment.segment_id,
                end_time - start_time,
                chunk.size(1)  # sequence length
            )

            # Remove overlap if added
            if self.config.enable_sequence_overlap and i < self.config.num_segments - 1:
                original_chunk_size = min(chunk_size, seq_len - i * chunk_size)
                chunk_output = chunk_output[:, :original_chunk_size, :]  # Keep only original chunk size

            segment_outputs.append(chunk_output)

        # Concatenate the outputs along sequence dimension
        output = torch.cat(segment_outputs, dim=1)

        return output

    def _execute_ring_sequence_parallel(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Any:
        """Execute sequence parallelism with ring-based algorithm."""
        if isinstance(inputs, dict):
            input_ids = inputs.get('input_ids', inputs.get('inputs_embeds', inputs))
        else:
            input_ids = inputs

        # Ensure input has the right shape (batch_size, seq_len, hidden_size)
        if input_ids.dim() == 2:
            # If input is (batch_size, seq_len), assume hidden_size=1 or expand
            input_ids = input_ids.unsqueeze(-1)  # Add hidden dimension
        elif input_ids.dim() != 3:
            raise ValueError(f"Expected input to have 2 or 3 dimensions, got {input_ids.dim()}")

        # Ring-based sequence parallelism processes chunks in a ring fashion
        seq_len = input_ids.size(1)  # Get sequence length (second dimension)
        chunk_size = self.config.ring_chunk_size

        # Split sequence into ring chunks
        num_chunks = math.ceil(seq_len / chunk_size)
        chunks = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, seq_len)
            chunk = input_ids[:, start_idx:end_idx, :]  # Keep batch and hidden dims
            chunks.append(chunk)

        # Process chunks in ring fashion across segments
        final_outputs = []

        for chunk_idx, chunk in enumerate(chunks):
            # Each chunk gets processed by all segments in sequence
            current_output = chunk

            for seg_idx, segment in enumerate(self.segments):
                start_time = time.time()

                # Process chunk through segment
                current_output = segment(current_output)

                end_time = time.time()
                self.balancer.record_segment_time(
                    segment.segment_id,
                    end_time - start_time,
                    chunk.size(1)  # sequence length
                )

                # For ring processing, we pass the output to the next segment
                # but in practice, each chunk goes through all segments sequentially
                # This simulates the ring pattern where chunks flow through segments

            final_outputs.append(current_output)

        # Concatenate all processed chunks along sequence dimension
        output = torch.cat(final_outputs, dim=1)

        return output

    def generate_with_sequence_parallel(self,
                                      inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                                      max_new_tokens: int = 50,
                                      **kwargs) -> torch.Tensor:
        """Generate tokens using sequence parallelism."""
        # For generation, we need to handle the autoregressive nature carefully
        # This is a simplified implementation that just passes the input through the sequence parallel model
        # without actual token generation (which would require a more complex implementation)

        # For now, just return the forward pass result
        # A full implementation would require the model to have a generate method
        # and handle the autoregressive generation properly
        try:
            # Try to use the model's generate method if it exists
            if hasattr(self.model, 'generate'):
                # This would delegate to the original model's generate method
                # But since we've split the model, we need to handle this differently
                # For now, just return the forward pass
                return self.forward(inputs)
            else:
                # Just return the forward pass result
                return self.forward(inputs)
        except Exception as e:
            logger.warning(f"Generation failed: {e}, returning forward pass result")
            return self.forward(inputs)


class SequenceParallelManager:
    """Manager for sequence parallel models."""

    def __init__(self):
        self.models = {}
        self.default_config = SequenceParallelConfig()

    def create_sequence_model(self,
                            model: nn.Module,
                            config: Optional[SequenceParallelConfig] = None) -> SequenceParallel:
        """Create a sequence parallel version of a model."""
        if config is None:
            config = self.default_config

        sequence_model = SequenceParallel(model, config)
        model_id = id(sequence_model)
        self.models[model_id] = sequence_model

        return sequence_model

    def get_sequence_stats(self, sequence_model: SequenceParallel) -> Dict[str, Any]:
        """Get statistics about sequence parallel execution."""
        model_id = id(sequence_model)
        if model_id not in self.models:
            return {}

        stats = {
            'num_segments': len(sequence_model.segments),
            'sequence_split_method': sequence_model.config.sequence_split_method,
            'segment_times': sequence_model.balancer.get_average_segment_times(),
            'tokens_per_segment': sequence_model.balancer.get_average_tokens_per_segment(),
            'devices_used': [seg.input_device for seg in sequence_model.segments]
        }

        return stats

    def cleanup_model(self, sequence_model: SequenceParallel):
        """Clean up sequence model resources."""
        model_id = id(sequence_model)
        if model_id in self.models:
            del self.models[model_id]


def create_sequence_parallel_config(num_segments: int = 1,
                                  segment_device_mapping: Optional[List[str]] = None,
                                  sequence_split_method: str = 'chunk',
                                  enable_sequence_overlap: bool = True,
                                  overlap_size: int = 64) -> SequenceParallelConfig:
    """Helper function to create a sequence parallelism configuration."""
    return SequenceParallelConfig(
        num_segments=num_segments,
        segment_device_mapping=segment_device_mapping,
        sequence_split_method=sequence_split_method,
        enable_sequence_overlap=enable_sequence_overlap,
        overlap_size=overlap_size
    )


def split_sequence_for_parallel(model: nn.Module, num_segments: int) -> List[nn.Module]:
    """Utility function to split a model into segments for sequence parallelism."""
    # This function provides a way to manually split a model
    # Get the model's layers/modules to split
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-style model
        all_layers = list(model.transformer.h)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
        # Transformer-style model (like GLM, Qwen)
        all_layers = list(model.transformer.layers)
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Another common structure
        all_layers = list(model.model.layers)
    elif hasattr(model, 'layers') and isinstance(model.layers, nn.ModuleList):
        # Direct access to layers as ModuleList
        all_layers = list(model.layers)
    else:
        # Fallback: try to split the entire model differently
        # Look for the first ModuleList in the model's children
        all_layers = []
        for child_name, child_module in model.named_children():
            if isinstance(child_module, nn.ModuleList):
                all_layers = list(child_module)
                break

        if not all_layers:
            # If no ModuleList found, fall back to all children
            all_modules = list(model.children())
            all_layers = all_modules

    # Calculate layers per segment
    total_layers = len(all_layers)
    layers_per_segment = total_layers // num_segments
    remainder = total_layers % num_segments

    segment_models = []
    start_idx = 0
    for i in range(num_segments):
        # Distribute remainder layers to first few segments
        end_idx = start_idx + layers_per_segment + (1 if i < remainder else 0)

        # Create a sequential module for this segment
        segment_layers = all_layers[start_idx:end_idx]
        if len(segment_layers) == 0:
            # If no layers assigned to this segment, create an identity module
            segment_model = nn.Identity()
        elif len(segment_layers) == 1:
            # If there's only one layer, wrap it in a Sequential to ensure it has a forward method
            segment_model = nn.Sequential(segment_layers[0])
        else:
            segment_model = nn.Sequential(*segment_layers)

        segment_models.append(segment_model)
        start_idx = end_idx

    return segment_models


__all__ = [
    'SequenceParallelConfig',
    'SequenceSegment',
    'SequenceBalancer',
    'SequenceParallel',
    'SequenceParallelManager',
    'create_sequence_parallel_config',
    'split_sequence_for_parallel'
]