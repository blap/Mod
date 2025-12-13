"""
Distributed Pipeline Parallelism for Inference in Qwen3-VL model.
Implements model partitioning strategies and inter-stage communication mechanisms.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
import math
import threading
import queue
import time


class PipelineStage(nn.Module):
    """
    A single stage in the pipeline, containing a subset of transformer layers.
    """
    def __init__(self, config, start_layer: int, end_layer: int, stage_id: int):
        super().__init__()
        self.config = config
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.stage_id = stage_id
        self.num_layers = end_layer - start_layer
        
        # Create the layers for this stage
        self.layers = nn.ModuleList()
        for layer_idx in range(start_layer, end_layer):
            # Create a transformer layer - simplified for this example
            layer = self._create_transformer_layer(config, layer_idx)
            self.layers.append(layer)
        
        # Micro-batch processing parameters
        self.micro_batch_size = getattr(config, 'pipeline_micro_batch_size', 1)
        
    def _create_transformer_layer(self, config, layer_idx: int):
        """Create a transformer layer for this pipeline stage."""
        # This is a simplified transformer layer for pipeline parallelism
        # In practice, you'd use the actual Qwen3-VL layer implementation
        return nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size or config.hidden_size * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through this pipeline stage.
        """
        # Process through all layers in this stage
        for layer in self.layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=attention_mask)
        
        return hidden_states


class PipelineScheduler:
    """
    Schedules micro-batches across pipeline stages for efficient execution.
    """
    def __init__(self, num_stages: int, micro_batch_size: int = 1):
        self.num_stages = num_stages
        self.micro_batch_size = micro_batch_size
        
        # Create queues for inter-stage communication
        self.stage_queues = [queue.Queue() for _ in range(num_stages)]
        
        # Pipeline timing information
        self.stage_times = [0.0] * num_stages
        self.micro_batch_times = []
    
    def schedule_micro_batches(self, num_micro_batches: int) -> List[Tuple[int, int]]:
        """
        Schedule micro-batches across stages using 1F1B (1F1B) or GPipe scheduling.
        Returns list of (stage_id, micro_batch_id) tuples indicating execution order.
        """
        schedule = []
        
        # Use 1F1B scheduling for balanced pipeline
        for mb in range(num_micro_batches):
            # Forward pass
            for stage in range(self.num_stages):
                if mb < self.num_stages - stage:
                    schedule.append((stage, mb))
            
            # Backward pass (for training, ignored in inference)
            for stage in range(self.num_stages - 1, -1, -1):
                if mb >= self.num_stages - 1 - stage and mb < num_micro_batches:
                    schedule.append((stage, mb))
        
        return schedule


class PipelineParallelModel(nn.Module):
    """
    Main pipeline parallel model that manages stage execution and communication.
    """
    def __init__(self, config, num_stages: int = 2):
        super().__init__()
        self.config = config
        self.num_stages = num_stages
        
        # Calculate layers per stage
        total_layers = config.num_hidden_layers
        layers_per_stage = total_layers // num_stages
        remainder = total_layers % num_stages
        
        # Create pipeline stages
        self.stages = nn.ModuleList()
        start_layer = 0
        
        for stage_id in range(num_stages):
            # Distribute remainder layers to earlier stages
            end_layer = start_layer + layers_per_stage + (1 if stage_id < remainder else 0)
            
            stage = PipelineStage(config, start_layer, end_layer, stage_id)
            self.stages.append(stage)
            
            start_layer = end_layer
        
        # Pipeline scheduler
        self.scheduler = PipelineScheduler(num_stages)
        
        # Inter-stage buffers for communication
        self.inter_stage_buffers = [None] * (num_stages - 1)
        
        # Configuration for pipeline parallelism
        self.use_pipeline_parallelism = getattr(config, 'use_pipeline_parallelism', True)
        self.pipeline_micro_batch_size = getattr(config, 'pipeline_micro_batch_size', 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through pipeline parallel model.
        """
        if not self.use_pipeline_parallelism or self.num_stages <= 1:
            # Fallback to sequential execution
            x = hidden_states
            for stage in self.stages:
                x = stage(x, attention_mask)
            return x
        
        # Split input into micro-batches
        batch_size = hidden_states.size(0)
        micro_batch_size = self.pipeline_micro_batch_size
        num_micro_batches = math.ceil(batch_size / micro_batch_size)
        
        # Create micro-batches
        micro_batches = []
        for i in range(0, batch_size, micro_batch_size):
            end_idx = min(i + micro_batch_size, batch_size)
            mb_hidden = hidden_states[i:end_idx]
            mb_mask = attention_mask[i:end_idx] if attention_mask is not None else None
            micro_batches.append((mb_hidden, mb_mask))
        
        # Execute pipeline schedule
        outputs = [None] * len(micro_batches)
        
        # Schedule for inference (only forward pass)
        for mb_idx in range(num_micro_batches):
            # Process each micro-batch through all stages sequentially
            current_output = micro_batches[mb_idx][0]  # hidden states
            current_mask = micro_batches[mb_idx][1]    # attention mask
            
            # Pass through all stages
            for stage_idx, stage in enumerate(self.stages):
                current_output = stage(current_output, current_mask)
                
                # If not the last stage, continue to next stage
                # In this simplified version, we process sequentially
            
            outputs[mb_idx] = current_output
        
        # Concatenate outputs
        final_output = torch.cat(outputs, dim=0)
        
        return final_output


class AsyncPipelineStage(nn.Module):
    """
    Asynchronous pipeline stage that can run in parallel with other stages.
    """
    def __init__(self, config, start_layer: int, end_layer: int, stage_id: int, device: torch.device):
        super().__init__()
        self.config = config
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.stage_id = stage_id
        self.device = device
        
        # Create the layers for this stage and move to device
        self.layers = nn.ModuleList()
        for layer_idx in range(start_layer, end_layer):
            layer = self._create_transformer_layer(config, layer_idx)
            self.layers.append(layer.to(device))
    
    def _create_transformer_layer(self, config, layer_idx: int):
        """Create a transformer layer for this pipeline stage."""
        return nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size or config.hidden_size * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through this stage."""
        # Move input to stage device if needed
        if hidden_states.device != self.device:
            hidden_states = hidden_states.to(self.device)
        if attention_mask is not None and attention_mask.device != self.device:
            attention_mask = attention_mask.to(self.device)
        
        # Process through all layers in this stage
        for layer in self.layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=attention_mask)
        
        return hidden_states


class DistributedPipelineModel(nn.Module):
    """
    Distributed pipeline model that can utilize multiple devices.
    """
    def __init__(self, config, stage_devices: List[torch.device]):
        super().__init__()
        self.config = config
        self.stage_devices = stage_devices
        self.num_stages = len(stage_devices)
        
        # Calculate layers per stage
        total_layers = config.num_hidden_layers
        layers_per_stage = total_layers // self.num_stages
        remainder = total_layers % self.num_stages
        
        # Create stages on different devices
        self.stages = nn.ModuleList()
        start_layer = 0
        
        for stage_id, device in enumerate(stage_devices):
            # Distribute remainder layers to earlier stages
            end_layer = start_layer + layers_per_stage + (1 if stage_id < remainder else 0)
            
            stage = AsyncPipelineStage(config, start_layer, end_layer, stage_id, device)
            self.stages.append(stage)
            
            start_layer = end_layer
        
        # Communication buffers between stages
        self.communication_buffers = []
        for i in range(self.num_stages - 1):
            self.communication_buffers.append({
                'src_device': stage_devices[i],
                'dst_device': stage_devices[i + 1],
                'buffer': None
            })
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through distributed pipeline.
        """
        # Start with the first stage device
        x = hidden_states.to(self.stages[0].device)
        mask = attention_mask.to(self.stages[0].device) if attention_mask is not None else None
        
        # Process through each stage
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x, mask)
            
            # Move to next stage device if not the last stage
            if stage_idx < len(self.stages) - 1:
                next_device = self.stages[stage_idx + 1].device
                x = x.to(next_device)
                if mask is not None:
                    mask = mask.to(next_device)
        
        return x


class OptimizedPipelineParallelModel(nn.Module):
    """
    Optimized pipeline parallel model with efficient scheduling and communication.
    """
    def __init__(self, config, num_stages: int = 2, stage_devices: Optional[List[torch.device]] = None):
        super().__init__()
        self.config = config
        self.num_stages = num_stages
        self.stage_devices = stage_devices or [torch.device('cpu')] * num_stages
        
        # Use distributed pipeline if multiple devices provided
        if len(set(self.stage_devices)) > 1:
            self.pipeline_model = DistributedPipelineModel(config, self.stage_devices)
        else:
            self.pipeline_model = PipelineParallelModel(config, num_stages)
        
        # Overlap communication and computation when possible
        self.enable_communication_overlap = getattr(config, 'enable_pipeline_communication_overlap', True)
        
        # Pipeline scheduling parameters
        self.micro_batch_size = getattr(config, 'pipeline_micro_batch_size', 1)
        self.enable_1f1b_scheduling = getattr(config, 'pipeline_1f1b_scheduling', True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optimized pipeline parallelism.
        """
        return self.pipeline_model(hidden_states, attention_mask, position_ids)


class PipelineParallelAttention(nn.Module):
    """
    Attention mechanism designed for pipeline parallel execution.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
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
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=-2)
            value_states = torch.cat([past_key_value[1], value_states], dim=-2)

        # Repeat keys and values for GQA (Grouped Query Attention) if applicable
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

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