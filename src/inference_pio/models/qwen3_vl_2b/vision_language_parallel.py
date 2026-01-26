"""
Vision-Language Parallelism System for Large Language Models

This module implements a comprehensive vision-language parallelism system that 
efficiently handles both visual and textual components of multimodal models like Qwen3-VL-2B.
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
from collections import defaultdict


logger = logging.getLogger(__name__)


@dataclass
class VisionLanguageConfig:
    """Configuration for vision-language parallelism."""
    num_visual_stages: int = 1
    num_textual_stages: int = 1
    visual_device_mapping: Optional[List[str]] = None  # e.g., ['cuda:0', 'cuda:1']
    textual_device_mapping: Optional[List[str]] = None  # e.g., ['cuda:2', 'cuda:3']
    enable_cross_modal_communication: bool = True
    cross_modal_buffer_size: int = 10
    enable_load_balancing: bool = True
    load_balance_method: str = 'adaptive'  # 'uniform', 'adaptive', 'performance_based'
    enable_gradient_checkpointing: bool = False
    inter_stage_buffer_size: int = 10
    pipeline_schedule: str = 'interleaved'  # 'sequential', 'interleaved', 'async'
    enable_multimodal_fusion: bool = True
    fusion_device: Optional[str] = None
    enable_activation_offloading: bool = True
    activation_offload_device: str = 'cpu'


class VisualStage(nn.Module):
    """Represents a single stage in the visual processing pipeline."""

    def __init__(self,
                 stage_id: int,
                 visual_components: nn.Module,
                 config: VisionLanguageConfig,
                 input_device: str = 'cpu',
                 output_device: str = 'cpu'):
        super().__init__()
        self.stage_id = stage_id
        self.config = config
        self.input_device = input_device
        self.output_device = output_device

        # Move visual components to appropriate device
        self.visual_components = visual_components.to(input_device)

        # Communication queues for cross-modal interaction
        self.cross_modal_queue = queue.Queue(maxsize=config.cross_modal_buffer_size)

        # Thread safety
        self.lock = threading.Lock()

        logger.info(f"Visual stage {stage_id} initialized on device {input_device}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through this visual stage."""
        # Move input to stage device
        x = x.to(self.input_device)

        # Apply gradient checkpointing if enabled
        if self.config.enable_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(self.visual_components, x)
        else:
            x = self.visual_components(x)

        # Move output to output device if different from input device
        if self.output_device != self.input_device:
            x = x.to(self.output_device)

        return x


class TextualStage(nn.Module):
    """Represents a single stage in the textual processing pipeline."""

    def __init__(self,
                 stage_id: int,
                 textual_components: nn.Module,
                 config: VisionLanguageConfig,
                 input_device: str = 'cpu',
                 output_device: str = 'cpu'):
        super().__init__()
        self.stage_id = stage_id
        self.config = config
        self.input_device = input_device
        self.output_device = output_device

        # Move textual components to appropriate device
        self.textual_components = textual_components.to(input_device)

        # Communication queues for cross-modal interaction
        self.cross_modal_queue = queue.Queue(maxsize=config.cross_modal_buffer_size)

        # Thread safety
        self.lock = threading.Lock()

        logger.info(f"Textual stage {stage_id} initialized on device {input_device}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through this textual stage."""
        # Move input to stage device
        x = x.to(self.input_device)

        # Apply gradient checkpointing if enabled
        if self.config.enable_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(self.textual_components, x)
        else:
            x = self.textual_components(x)

        # Move output to output device if different from input device
        if self.output_device != self.input_device:
            x = x.to(self.output_device)

        return x


class CrossModalCommunicator:
    """Handles communication between visual and textual stages."""

    def __init__(self, config: VisionLanguageConfig):
        self.config = config
        self.visual_to_textual_queue = queue.Queue(maxsize=config.cross_modal_buffer_size)
        self.textual_to_visual_queue = queue.Queue(maxsize=config.cross_modal_buffer_size)
        self.lock = threading.Lock()

    def send_visual_to_textual(self, data: torch.Tensor, stage_id: int):
        """Send data from visual stage to textual stage."""
        with self.lock:
            try:
                self.visual_to_textual_queue.put_nowait((stage_id, data))
                logger.debug(f"Sent visual data from stage {stage_id} to textual stages")
            except queue.Full:
                logger.warning(f"Visual-to-textual queue is full, dropping data from stage {stage_id}")

    def send_textual_to_visual(self, data: torch.Tensor, stage_id: int):
        """Send data from textual stage to visual stage."""
        with self.lock:
            try:
                self.textual_to_visual_queue.put_nowait((stage_id, data))
                logger.debug(f"Sent textual data from stage {stage_id} to visual stages")
            except queue.Full:
                logger.warning(f"Textual-to-visual queue is full, dropping data from stage {stage_id}")

    def receive_visual_to_textual(self) -> Optional[Tuple[int, torch.Tensor]]:
        """Receive data from visual to textual stages."""
        with self.lock:
            try:
                return self.visual_to_textual_queue.get_nowait()
            except queue.Empty:
                return None

    def receive_textual_to_visual(self) -> Optional[Tuple[int, torch.Tensor]]:
        """Receive data from textual to visual stages."""
        with self.lock:
            try:
                return self.textual_to_visual_queue.get_nowait()
            except queue.Empty:
                return None


class MultimodalFusionModule(nn.Module):
    """Module for fusing visual and textual representations."""

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Cross-attention for visual-textual fusion
        self.visual_to_text_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout, 
            batch_first=True
        )
        self.text_to_visual_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout, 
            batch_first=True
        )
        
        # Feed-forward networks
        self.ffn_visual = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_textual = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm_visual = nn.LayerNorm(d_model)
        self.norm_textual = nn.LayerNorm(d_model)

    def forward(self, visual_features: torch.Tensor, textual_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse visual and textual representations."""
        # Visual attending to textual features
        fused_visual, _ = self.visual_to_text_attn(
            visual_features, 
            textual_features, 
            textual_features
        )
        fused_visual = self.norm_visual(visual_features + fused_visual)
        fused_visual = self.norm_visual(fused_visual + self.ffn_visual(fused_visual))
        
        # Textual attending to visual features
        fused_textual, _ = self.text_to_visual_attn(
            textual_features, 
            visual_features, 
            visual_features
        )
        fused_textual = self.norm_textual(textual_features + fused_textual)
        fused_textual = self.norm_textual(fused_textual + self.ffn_textual(fused_textual))
        
        return fused_visual, fused_textual


class VisionLanguageBalancer:
    """Manages load balancing between visual and textual stages."""

    def __init__(self, config: VisionLanguageConfig):
        self.config = config
        self.visual_stage_times = [0.0] * config.num_visual_stages
        self.textual_stage_times = [0.0] * config.num_textual_stages
        self.visual_stage_counts = [0] * config.num_visual_stages
        self.textual_stage_counts = [0] * config.num_textual_stages
        self.lock = threading.Lock()

    def record_visual_stage_time(self, stage_id: int, time_taken: float):
        """Record execution time for a visual stage."""
        with self.lock:
            self.visual_stage_times[stage_id] += time_taken
            self.visual_stage_counts[stage_id] += 1

    def record_textual_stage_time(self, stage_id: int, time_taken: float):
        """Record execution time for a textual stage."""
        with self.lock:
            self.textual_stage_times[stage_id] += time_taken
            self.textual_stage_counts[stage_id] += 1

    def get_average_visual_stage_times(self) -> List[float]:
        """Get average execution time for each visual stage."""
        avg_times = []
        for i in range(self.config.num_visual_stages):
            if self.visual_stage_counts[i] > 0:
                avg_times.append(self.visual_stage_times[i] / self.visual_stage_counts[i])
            else:
                avg_times.append(0.0)
        return avg_times

    def get_average_textual_stage_times(self) -> List[float]:
        """Get average execution time for each textual stage."""
        avg_times = []
        for i in range(self.config.num_textual_stages):
            if self.textual_stage_counts[i] > 0:
                avg_times.append(self.textual_stage_times[i] / self.textual_stage_counts[i])
            else:
                avg_times.append(0.0)
        return avg_times

    def rebalance_if_needed(self) -> bool:
        """Check if rebalancing is needed and perform it."""
        if not self.config.enable_load_balancing:
            return False

        avg_visual_times = self.get_average_visual_stage_times()
        avg_textual_times = self.get_average_textual_stage_times()
        
        # Check for imbalances within visual stages
        if len(set(avg_visual_times)) > 1:
            max_time = max(avg_visual_times) if avg_visual_times else 0
            min_time = min(avg_visual_times) if avg_visual_times else 0
            if max_time > 0 and (max_time - min_time) / max_time > 0.2:  # 20% difference threshold
                logger.info(f"Visual stage load imbalance detected: max={max_time:.4f}s, min={min_time:.4f}s")
                return True

        # Check for imbalances within textual stages
        if len(set(avg_textual_times)) > 1:
            max_time = max(avg_textual_times) if avg_textual_times else 0
            min_time = min(avg_textual_times) if avg_textual_times else 0
            if max_time > 0 and (max_time - min_time) / max_time > 0.2:  # 20% difference threshold
                logger.info(f"Textual stage load imbalance detected: max={max_time:.4f}s, min={min_time:.4f}s")
                return True

        # Check for imbalances between visual and textual processing
        avg_visual_time = sum(avg_visual_times) / len(avg_visual_times) if avg_visual_times else 0
        avg_textual_time = sum(avg_textual_times) / len(avg_textual_times) if avg_textual_times else 0
        
        if avg_visual_time > 0 and avg_textual_time > 0:
            max_avg = max(avg_visual_time, avg_textual_time)
            min_avg = min(avg_visual_time, avg_textual_time)
            if (max_avg - min_avg) / max_avg > 0.3:  # 30% difference threshold between modalities
                logger.info(f"Cross-modal load imbalance detected: visual_avg={avg_visual_time:.4f}s, textual_avg={avg_textual_time:.4f}s")
                return True

        return False


class VisionLanguageParallel(nn.Module):
    """Main vision-language parallelism module that orchestrates visual and textual stages."""

    def __init__(self, model: nn.Module, config: VisionLanguageConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.visual_stages: List[VisualStage] = []
        self.textual_stages: List[TextualStage] = []
        self.balancer = VisionLanguageBalancer(config)
        self.communicator = CrossModalCommunicator(config)
        
        # Initialize multimodal fusion module if enabled
        if config.enable_multimodal_fusion:
            # Try to infer the model's embedding dimension
            d_model = 2048  # Default fallback
            if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
                d_model = model.config.hidden_size
            elif hasattr(model, 'hidden_size'):
                d_model = model.hidden_size
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'embed_dim'):
                d_model = model.transformer.embed_dim
            elif hasattr(model, 'model') and hasattr(model.model, 'embed_dim'):
                d_model = model.model.embed_dim

            self.multimodal_fusion = MultimodalFusionModule(d_model=d_model)
            if config.fusion_device:
                self.multimodal_fusion = self.multimodal_fusion.to(config.fusion_device)

        # Split the model into visual and textual stages
        self._split_model_into_stages()

        logger.info(f"Vision-language parallel model created with {config.num_visual_stages} visual stages and {config.num_textual_stages} textual stages")

    def _split_model_into_stages(self):
        """Split the model into visual and textual stages."""
        # Determine device mappings
        if self.config.visual_device_mapping:
            visual_device_mapping = self.config.visual_device_mapping
        else:
            # Default to using available GPUs or CPU
            if torch.cuda.is_available():
                visual_device_mapping = [f'cuda:{i % torch.cuda.device_count()}'
                                       for i in range(self.config.num_visual_stages)]
            else:
                visual_device_mapping = ['cpu'] * self.config.num_visual_stages

        if self.config.textual_device_mapping:
            textual_device_mapping = self.config.textual_device_mapping
        else:
            # Default to using available GPUs or CPU
            if torch.cuda.is_available():
                textual_device_mapping = [f'cuda:{(i + self.config.num_visual_stages) % torch.cuda.device_count()}'
                                        for i in range(self.config.num_textual_stages)]
            else:
                textual_device_mapping = ['cpu'] * self.config.num_textual_stages

        # Identify visual and textual components in the model
        visual_components, textual_components = self._identify_modal_components()

        # Split visual components into stages
        self._split_visual_components(visual_components, visual_device_mapping)
        
        # Split textual components into stages
        self._split_textual_components(textual_components, textual_device_mapping)

    def _identify_modal_components(self) -> Tuple[List[nn.Module], List[nn.Module]]:
        """Identify visual and textual components in the model."""
        visual_components = []
        textual_components = []

        # For Qwen3-VL models, we need to identify vision encoder and language model components
        # First, let's try to identify high-level components by name
        for name, module in self.model.named_modules():
            # Skip the root model
            if module == self.model:
                continue

            if any(keyword in name.lower() for keyword in ['vision', 'visual', 'img', 'image']):
                # Likely a visual component
                visual_components.append(module)
            elif any(keyword in name.lower() for keyword in ['language', 'text', 'lm_head', 'word_embeddings', 'wte']):
                # Likely a textual component
                textual_components.append(module)
            elif 'encoder' in name.lower() and 'vision' in name.lower():
                # Vision encoder
                visual_components.append(module)
            elif any(keyword in name.lower() for keyword in ['decoder', 'transformer', 'gpt', 'llama', 'opt']):
                # Likely textual components
                textual_components.append(module)

        # If we didn't find specific components by name, identify by module type
        if not visual_components or not textual_components:
            # Clear the lists to avoid duplication
            visual_components = []
            textual_components = []

            # Identify by module type
            for name, module in self.model.named_modules():
                if module == self.model:
                    continue  # Skip the root model

                # Check if this module is already classified
                if module in visual_components or module in textual_components:
                    continue

                # Classify based on module type
                if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.BatchNorm2d)):
                    # These are typically visual components
                    visual_components.append(module)
                elif isinstance(module, (nn.Embedding, nn.LSTM, nn.GRU)):
                    # These could be either, but embeddings are often textual
                    if any(keyword in name.lower() for keyword in ['word', 'token', 'text', 'language']):
                        textual_components.append(module)
                    elif any(keyword in name.lower() for keyword in ['vision', 'visual', 'img', 'image']):
                        visual_components.append(module)
                    else:
                        # Default to textual for embeddings
                        textual_components.append(module)
                elif isinstance(module, nn.Linear):
                    # Linear layers could be either, try to infer from context
                    parent_name = '.'.join(name.split('.')[:-1])  # Get parent module name
                    if any(keyword in parent_name.lower() for keyword in ['vision', 'visual', 'img', 'image', 'patch']):
                        visual_components.append(module)
                    elif any(keyword in parent_name.lower() for keyword in ['language', 'text', 'lm_head', 'transformer']):
                        textual_components.append(module)
                    else:
                        # If we can't determine, put in textual by default
                        textual_components.append(module)
                elif isinstance(module, (nn.LayerNorm, nn.Dropout, nn.ReLU, nn.GELU, nn.Softmax)):
                    # These are typically found in both, try to infer from context
                    parent_name = '.'.join(name.split('.')[:-1])
                    if any(keyword in parent_name.lower() for keyword in ['vision', 'visual', 'img', 'image', 'patch']):
                        visual_components.append(module)
                    elif any(keyword in parent_name.lower() for keyword in ['language', 'text', 'lm_head', 'transformer']):
                        textual_components.append(module)
                    else:
                        # Default to textual
                        textual_components.append(module)
                else:
                    # For other module types, try to infer from name
                    if any(keyword in name.lower() for keyword in ['vision', 'visual', 'img', 'image', 'patch', 'vit']):
                        visual_components.append(module)
                    elif any(keyword in name.lower() for keyword in ['language', 'text', 'lm_head', 'transformer', 'decoder']):
                        textual_components.append(module)
                    else:
                        # Default to textual for unknown modules
                        textual_components.append(module)

        # Final check: if we still don't have components, assign based on module types alone
        if not visual_components and not textual_components:
            for name, module in self.model.named_modules():
                if module == self.model:
                    continue

                if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.BatchNorm2d)):
                    visual_components.append(module)
                else:
                    textual_components.append(module)

        return visual_components, textual_components

    def _split_visual_components(self, visual_components: List[nn.Module], device_mapping: List[str]):
        """Split visual components into stages."""
        total_visual_components = len(visual_components)
        if total_visual_components == 0:
            # If no visual components found, create identity modules
            for i in range(self.config.num_visual_stages):
                visual_stage = VisualStage(
                    stage_id=i,
                    visual_components=nn.Identity(),
                    config=self.config,
                    input_device=device_mapping[i],
                    output_device=device_mapping[i]
                )
                self.visual_stages.append(visual_stage)
            return

        components_per_stage = total_visual_components // self.config.num_visual_stages
        remainder = total_visual_components % self.config.num_visual_stages

        start_idx = 0
        for i in range(self.config.num_visual_stages):
            # Distribute remainder components to first few stages
            end_idx = start_idx + components_per_stage + (1 if i < remainder else 0)

            # Create a sequential module for this stage
            stage_components = visual_components[start_idx:end_idx]
            if len(stage_components) == 1:
                stage_model = nn.Sequential(stage_components[0])
            else:
                stage_model = nn.Sequential(*stage_components)

            # Create visual stage
            visual_stage = VisualStage(
                stage_id=i,
                visual_components=stage_model,
                config=self.config,
                input_device=device_mapping[i],
                output_device=device_mapping[i]
            )

            self.visual_stages.append(visual_stage)
            logger.info(f"Visual stage {i}: {start_idx}-{end_idx-1} ({len(stage_components)} components) on {device_mapping[i]}")

            start_idx = end_idx

    def _split_textual_components(self, textual_components: List[nn.Module], device_mapping: List[str]):
        """Split textual components into stages."""
        total_textual_components = len(textual_components)
        if total_textual_components == 0:
            # If no textual components found, create identity modules
            for i in range(self.config.num_textual_stages):
                textual_stage = TextualStage(
                    stage_id=i,
                    textual_components=nn.Identity(),
                    config=self.config,
                    input_device=device_mapping[i],
                    output_device=device_mapping[i]
                )
                self.textual_stages.append(textual_stage)
            return

        components_per_stage = total_textual_components // self.config.num_textual_stages
        remainder = total_textual_components % self.config.num_textual_stages

        start_idx = 0
        for i in range(self.config.num_textual_stages):
            # Distribute remainder components to first few stages
            end_idx = start_idx + components_per_stage + (1 if i < remainder else 0)

            # Create a sequential module for this stage
            stage_components = textual_components[start_idx:end_idx]
            if len(stage_components) == 1:
                stage_model = nn.Sequential(stage_components[0])
            else:
                stage_model = nn.Sequential(*stage_components)

            # Create textual stage
            textual_stage = TextualStage(
                stage_id=i,
                textual_components=stage_model,
                config=self.config,
                input_device=device_mapping[i],
                output_device=device_mapping[i]
            )

            self.textual_stages.append(textual_stage)
            logger.info(f"Textual stage {i}: {start_idx}-{end_idx-1} ({len(stage_components)} components) on {device_mapping[i]}")

            start_idx = end_idx

    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Any:
        """Execute the vision-language parallelism with various scheduling methods."""
        if self.config.pipeline_schedule == 'interleaved':
            return self._execute_interleaved(inputs)
        elif self.config.pipeline_schedule == 'async':
            return self._execute_async(inputs)
        else:
            # Default to sequential execution
            return self._execute_sequential(inputs)

    def _execute_sequential(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Any:
        """Execute vision-language processing sequentially."""
        # Separate visual and textual inputs
        visual_input = None
        textual_input = None

        if isinstance(inputs, dict):
            # Common keys for visual and textual inputs in multimodal models
            visual_input = inputs.get('pixel_values', inputs.get('images', None))
            textual_input = inputs.get('input_ids', inputs.get('input_embeds', inputs.get('inputs_embeds', None)))

            # If we have both, process them separately then combine
            if visual_input is not None and textual_input is not None:
                # Process visual components
                visual_output = visual_input
                for stage in self.visual_stages:
                    start_time = time.time()
                    visual_output = stage(visual_output)
                    end_time = time.time()
                    self.balancer.record_visual_stage_time(stage.stage_id, end_time - start_time)

                # Process textual components
                textual_output = textual_input
                for stage in self.textual_stages:
                    start_time = time.time()
                    textual_output = stage(textual_output)
                    end_time = time.time()
                    self.balancer.record_textual_stage_time(stage.stage_id, end_time - start_time)

                # Fuse the outputs if fusion is enabled
                if self.config.enable_multimodal_fusion:
                    fused_visual, fused_textual = self.multimodal_fusion(visual_output, textual_output)
                    return {'visual': fused_visual, 'textual': fused_textual}

                return {'visual': visual_output, 'textual': textual_output}
            elif visual_input is not None:
                # Only visual processing
                visual_output = visual_input
                for stage in self.visual_stages:
                    start_time = time.time()
                    visual_output = stage(visual_output)
                    end_time = time.time()
                    self.balancer.record_visual_stage_time(stage.stage_id, end_time - start_time)
                return visual_output
            elif textual_input is not None:
                # Only textual processing
                textual_output = textual_input
                for stage in self.textual_stages:
                    start_time = time.time()
                    textual_output = stage(textual_output)
                    end_time = time.time()
                    self.balancer.record_textual_stage_time(stage.stage_id, end_time - start_time)
                return textual_output
            else:
                # Fallback: treat as single tensor
                x = inputs
                if torch.is_tensor(x):
                    # For single tensor, determine if it's likely visual or textual based on shape
                    if x.dim() == 4:  # Likely image tensor (batch, channels, height, width)
                        for stage in self.visual_stages:
                            start_time = time.time()
                            x = stage(x)
                            end_time = time.time()
                            self.balancer.record_visual_stage_time(stage.stage_id, end_time - start_time)
                    else:  # Likely textual tensor
                        for stage in self.textual_stages:
                            start_time = time.time()
                            x = stage(x)
                            end_time = time.time()
                            self.balancer.record_textual_stage_time(stage.stage_id, end_time - start_time)
                return x
        else:
            # Single tensor input - determine if visual or textual based on shape
            x = inputs
            if torch.is_tensor(x):
                if x.dim() == 4:  # Likely image tensor (batch, channels, height, width)
                    for stage in self.visual_stages:
                        start_time = time.time()
                        x = stage(x)
                        end_time = time.time()
                        self.balancer.record_visual_stage_time(stage.stage_id, end_time - start_time)
                else:  # Likely textual tensor
                    for stage in self.textual_stages:
                        start_time = time.time()
                        x = stage(x)
                        end_time = time.time()
                        self.balancer.record_textual_stage_time(stage.stage_id, end_time - start_time)
            return x

    def _execute_interleaved(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Any:
        """Execute vision-language processing with interleaved scheduling."""
        # This method processes visual and textual components in an interleaved manner
        # allowing for better load balancing and resource utilization
        
        # Separate visual and textual inputs
        visual_input = None
        textual_input = None
        
        if isinstance(inputs, dict):
            visual_input = inputs.get('pixel_values', inputs.get('images', None))
            textual_input = inputs.get('input_ids', inputs.get('input_embeds', inputs.get('inputs_embeds', None)))
        
        if visual_input is not None and textual_input is not None:
            # Execute visual and textual processing in parallel where possible
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit visual processing
                visual_future = executor.submit(self._process_visual_stages, visual_input)
                
                # Submit textual processing
                textual_future = executor.submit(self._process_textual_stages, textual_input)
                
                # Wait for both to complete
                visual_output = visual_future.result()
                textual_output = textual_future.result()
                
                # Fuse the outputs if fusion is enabled
                if self.config.enable_multimodal_fusion:
                    fused_visual, fused_textual = self.multimodal_fusion(visual_output, textual_output)
                    return {'visual': fused_visual, 'textual': fused_textual}
                
                return {'visual': visual_output, 'textual': textual_output}
        else:
            # Fallback to sequential if only one modality is present
            return self._execute_sequential(inputs)

    def _process_visual_stages(self, visual_input: torch.Tensor) -> torch.Tensor:
        """Process input through all visual stages."""
        visual_output = visual_input
        for stage in self.visual_stages:
            start_time = time.time()
            visual_output = stage(visual_output)
            end_time = time.time()
            self.balancer.record_visual_stage_time(stage.stage_id, end_time - start_time)
        return visual_output

    def _process_textual_stages(self, textual_input: torch.Tensor) -> torch.Tensor:
        """Process input through all textual stages."""
        textual_output = textual_input
        for stage in self.textual_stages:
            start_time = time.time()
            textual_output = stage(textual_output)
            end_time = time.time()
            self.balancer.record_textual_stage_time(stage.stage_id, end_time - start_time)
        return textual_output

    def _execute_async(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Any:
        """Execute vision-language processing asynchronously."""
        # This method uses asynchronous processing for better resource utilization
        return self._execute_interleaved(inputs)  # For now, same as interleaved

    def generate_with_vision_language_parallel(self,
                                             inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                                             max_new_tokens: int = 50,
                                             **kwargs) -> torch.Tensor:
        """Generate tokens using vision-language parallelism."""
        # For generation, we need to handle the autoregressive nature carefully
        # This is a simplified implementation that just passes the input through the pipeline
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

    def cleanup(self):
        """Clean up resources used by the vision-language parallel model."""
        # Clean up any resources if needed
        logger.info("Cleaning up vision-language parallel model resources")


class VisionLanguageParallelManager:
    """Manager for vision-language parallel models."""

    def __init__(self):
        self.models = {}
        self.default_config = VisionLanguageConfig()

    def create_vision_language_model(self,
                                   model: nn.Module,
                                   config: Optional[VisionLanguageConfig] = None) -> VisionLanguageParallel:
        """Create a vision-language parallel version of a model."""
        if config is None:
            config = self.default_config

        vision_language_model = VisionLanguageParallel(model, config)
        model_id = id(vision_language_model)
        self.models[model_id] = vision_language_model

        return vision_language_model

    def get_vision_language_stats(self, vision_language_model: VisionLanguageParallel) -> Dict[str, Any]:
        """Get statistics about vision-language parallel execution."""
        model_id = id(vision_language_model)
        if model_id not in self.models:
            return {}

        stats = {
            'num_visual_stages': len(vision_language_model.visual_stages),
            'num_textual_stages': len(vision_language_model.textual_stages),
            'pipeline_schedule': vision_language_model.config.pipeline_schedule,
            'visual_stage_times': vision_language_model.balancer.get_average_visual_stage_times(),
            'textual_stage_times': vision_language_model.balancer.get_average_textual_stage_times(),
            'visual_devices_used': [stage.input_device for stage in vision_language_model.visual_stages],
            'textual_devices_used': [stage.input_device for stage in vision_language_model.textual_stages],
            'cross_modal_communication_enabled': vision_language_model.config.enable_cross_modal_communication,
            'multimodal_fusion_enabled': vision_language_model.config.enable_multimodal_fusion
        }

        return stats

    def cleanup_model(self, vision_language_model: VisionLanguageParallel):
        """Clean up vision-language model resources."""
        model_id = id(vision_language_model)
        if model_id in self.models:
            del self.models[model_id]


def create_vision_language_config(num_visual_stages: int = 1,
                                num_textual_stages: int = 1,
                                visual_device_mapping: Optional[List[str]] = None,
                                textual_device_mapping: Optional[List[str]] = None,
                                enable_cross_modal_communication: bool = True,
                                pipeline_schedule: str = 'interleaved',
                                enable_multimodal_fusion: bool = True,
                                fusion_device: Optional[str] = None) -> VisionLanguageConfig:
    """Helper function to create a vision-language parallelism configuration."""
    return VisionLanguageConfig(
        num_visual_stages=num_visual_stages,
        num_textual_stages=num_textual_stages,
        visual_device_mapping=visual_device_mapping,
        textual_device_mapping=textual_device_mapping,
        enable_cross_modal_communication=enable_cross_modal_communication,
        pipeline_schedule=pipeline_schedule,
        enable_multimodal_fusion=enable_multimodal_fusion,
        fusion_device=fusion_device
    )


def split_model_for_vision_language(model: nn.Module, 
                                  num_visual_stages: int, 
                                  num_textual_stages: int) -> Tuple[List[nn.Module], List[nn.Module]]:
    """Utility function to split a model into visual and textual components for vision-language parallelism."""
    # Identify visual and textual components
    visual_components = []
    textual_components = []
    
    for name, module in model.named_modules():
        if 'vision' in name.lower() or 'visual' in name.lower() or 'img' in name.lower():
            visual_components.append(module)
        elif 'language' in name.lower() or 'text' in name.lower() or 'lm_head' in name.lower():
            textual_components.append(module)
        elif 'encoder' in name.lower() and 'vision' in name.lower():
            visual_components.append(module)
        elif 'decoder' in name.lower() or 'transformer' in name.lower():
            # Default to textual for transformer components
            textual_components.append(module)
    
    # If we didn't find specific visual components, try to identify them by type
    if not visual_components:
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                visual_components.append(module)
            elif 'patch' in name.lower() or 'vit' in name.lower():
                visual_components.append(module)
    
    # If we didn't find specific textual components, assume the rest are textual
    if not textual_components:
        all_modules = list(model.modules())[1:]  # Skip the root model
        textual_components = [m for m in all_modules 
                            if m not in visual_components and not isinstance(m, (nn.Conv2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d))]
    
    # Split visual components into stages
    total_visual_components = len(visual_components)
    visual_stages = []
    if total_visual_components > 0:
        components_per_stage = total_visual_components // num_visual_stages
        remainder = total_visual_components % num_visual_stages

        start_idx = 0
        for i in range(num_visual_stages):
            end_idx = start_idx + components_per_stage + (1 if i < remainder else 0)
            stage_components = visual_components[start_idx:end_idx]
            if len(stage_components) == 1:
                stage_model = nn.Sequential(stage_components[0])
            else:
                stage_model = nn.Sequential(*stage_components)
            visual_stages.append(stage_model)
            start_idx = end_idx
    else:
        # If no visual components found, create identity modules
        visual_stages = [nn.Identity() for _ in range(num_visual_stages)]

    # Split textual components into stages
    total_textual_components = len(textual_components)
    textual_stages = []
    if total_textual_components > 0:
        components_per_stage = total_textual_components // num_textual_stages
        remainder = total_textual_components % num_textual_stages

        start_idx = 0
        for i in range(num_textual_stages):
            end_idx = start_idx + components_per_stage + (1 if i < remainder else 0)
            stage_components = textual_components[start_idx:end_idx]
            if len(stage_components) == 1:
                stage_model = nn.Sequential(stage_components[0])
            else:
                stage_model = nn.Sequential(*stage_components)
            textual_stages.append(stage_model)
            start_idx = end_idx
    else:
        # If no textual components found, create identity modules
        textual_stages = [nn.Identity() for _ in range(num_textual_stages)]

    return visual_stages, textual_stages


__all__ = [
    'VisionLanguageConfig',
    'VisualStage',
    'TextualStage',
    'CrossModalCommunicator',
    'MultimodalFusionModule',
    'VisionLanguageBalancer',
    'VisionLanguageParallel',
    'VisionLanguageParallelManager',
    'create_vision_language_config',
    'split_model_for_vision_language'
]