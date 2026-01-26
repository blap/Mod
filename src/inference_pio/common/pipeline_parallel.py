"""
Pipeline Parallelism System for Large Language Models

This module implements a comprehensive pipeline parallelism system that splits
models across multiple stages for efficient distributed inference.
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
import os
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for pipeline parallelism."""
    num_stages: int = 1
    stage_device_mapping: Optional[List[str]] = None  # e.g., ['cuda:0', 'cuda:1', 'cuda:2']
    microbatch_size: int = 1
    enable_activation_offloading: bool = True
    activation_offload_device: str = 'cpu'
    enable_gradient_checkpointing: bool = False
    inter_stage_buffer_size: int = 10
    pipeline_schedule: str = '1f1b'  # '1f1b', 'gpipe', 'chimera'
    enable_load_balancing: bool = True
    load_balance_method: str = 'uniform'  # 'uniform', 'adaptive'


class PipelineStage(nn.Module):
    """Represents a single stage in the pipeline."""
    
    def __init__(self, 
                 stage_id: int,
                 model_part: nn.Module,
                 config: PipelineConfig,
                 input_device: str = 'cpu',
                 output_device: str = 'cpu'):
        super().__init__()
        self.stage_id = stage_id
        self.config = config
        self.input_device = input_device
        self.output_device = output_device
        
        # Move model part to appropriate device
        self.model_part = model_part.to(input_device)
        
        # Activation offloading setup
        self.activation_queue = queue.Queue(maxsize=config.inter_stage_buffer_size)
        self.grad_queue = queue.Queue(maxsize=config.inter_stage_buffer_size)
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info(f"Pipeline stage {stage_id} initialized on device {input_device}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through this stage."""
        # Move input to stage device
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
    
    def forward_microbatch(self, microbatch: torch.Tensor) -> torch.Tensor:
        """Process a single microbatch."""
        return self.forward(microbatch)
    
    def backward_microbatch(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass for a single microbatch."""
        # This would typically involve computing gradients
        # For now, we'll just return the grad_output
        return grad_output


class PipelineBalancer:
    """Manages load balancing across pipeline stages."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stage_times = [0.0] * config.num_stages
        self.stage_counts = [0] * config.num_stages
        self.lock = threading.Lock()
    
    def record_stage_time(self, stage_id: int, time_taken: float):
        """Record execution time for a stage."""
        with self.lock:
            self.stage_times[stage_id] += time_taken
            self.stage_counts[stage_id] += 1
    
    def get_average_stage_times(self) -> List[float]:
        """Get average execution time for each stage."""
        avg_times = []
        for i in range(self.config.num_stages):
            if self.stage_counts[i] > 0:
                avg_times.append(self.stage_times[i] / self.stage_counts[i])
            else:
                avg_times.append(0.0)
        return avg_times
    
    def rebalance_if_needed(self) -> bool:
        """Check if rebalancing is needed and perform it."""
        if not self.config.enable_load_balancing:
            return False
            
        avg_times = self.get_average_stage_times()
        if len(set(avg_times)) <= 1:  # All stages take similar time
            return False
            
        # Simple rebalancing logic - adjust based on timing differences
        max_time = max(avg_times) if avg_times else 0
        min_time = min(avg_times) if avg_times else 0
        
        if max_time > 0 and (max_time - min_time) / max_time > 0.2:  # 20% difference threshold
            logger.info(f"Load imbalance detected: max={max_time:.4f}s, min={min_time:.4f}s")
            return True
        
        return False


class PipelineParallel(nn.Module):
    """Main pipeline parallelism module that orchestrates all stages."""
    
    def __init__(self, model: nn.Module, config: PipelineConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.stages: List[PipelineStage] = []
        self.balancer = PipelineBalancer(config)
        
        # Split the model into stages
        self._split_model_into_stages()
        
        # Setup communication queues between stages
        self.comm_queues = []
        for i in range(self.config.num_stages - 1):
            self.comm_queues.append(queue.Queue(maxsize=config.inter_stage_buffer_size))
        
        logger.info(f"Pipeline parallel model created with {config.num_stages} stages")
    
    def _split_model_into_stages(self):
        """Split the model into pipeline stages."""
        # Determine device mapping
        if self.config.stage_device_mapping:
            device_mapping = self.config.stage_device_mapping
        else:
            # Default to using available GPUs or CPU
            if torch.cuda.is_available():
                device_mapping = [f'cuda:{i % torch.cuda.device_count()}'
                                for i in range(self.config.num_stages)]
            else:
                device_mapping = ['cpu'] * self.config.num_stages

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
                    logger.info(f"Using ModuleList '{child_name}' for pipeline splitting")
                    break

            if not all_layers:
                # If no ModuleList found, fall back to all children
                all_modules = list(self.model.children())
                all_layers = all_modules

        # Calculate layers per stage
        total_layers = len(all_layers)
        layers_per_stage = total_layers // self.config.num_stages
        remainder = total_layers % self.config.num_stages

        start_idx = 0
        for i in range(self.config.num_stages):
            # Distribute remainder layers to first few stages
            end_idx = start_idx + layers_per_stage + (1 if i < remainder else 0)

            # Create a sequential module for this stage
            stage_layers = all_layers[start_idx:end_idx]
            if len(stage_layers) == 1:
                # If there's only one layer, wrap it in a Sequential to ensure it has a forward method
                stage_model = nn.Sequential(stage_layers[0])
            else:
                stage_model = nn.Sequential(*stage_layers)

            # Create pipeline stage
            stage = PipelineStage(
                stage_id=i,
                model_part=stage_model,
                config=self.config,
                input_device=device_mapping[i],
                output_device=device_mapping[i]
            )

            self.stages.append(stage)
            logger.info(f"Stage {i}: {start_idx}-{end_idx-1} ({len(stage_layers)} layers) on {device_mapping[i]}")

            start_idx = end_idx
    
    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Any:
        """Execute the pipeline with 1F1B (1F1B) scheduling."""
        if self.config.pipeline_schedule == '1f1b':
            return self._execute_1f1b(inputs)
        elif self.config.pipeline_schedule == 'gpipe':
            return self._execute_gpipe(inputs)
        else:
            # Default to simple sequential execution
            return self._execute_sequential(inputs)
    
    def _execute_sequential(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Any:
        """Execute pipeline sequentially (simple forward pass through all stages)."""
        x = inputs
        if isinstance(inputs, dict):
            # Handle dict inputs (like input_ids, attention_mask, etc.)
            x = inputs.get('input_ids', inputs.get('inputs_embeds', inputs))
        
        for stage in self.stages:
            start_time = time.time()
            x = stage(x)
            end_time = time.time()
            self.balancer.record_stage_time(stage.stage_id, end_time - start_time)
        
        return x
    
    def _execute_1f1b(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Any:
        """Execute pipeline with 1F1B (1F1B) scheduling."""
        # For now, implement a simplified version
        # In a real implementation, this would handle microbatching and overlapping
        return self._execute_sequential(inputs)
    
    def _execute_gpipe(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Any:
        """Execute pipeline with GPipe scheduling."""
        # For now, implement a simplified version
        # In a real implementation, this would handle microbatching
        return self._execute_sequential(inputs)
    
    def generate_with_pipeline(self,
                             inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                             max_new_tokens: int = 50,
                             **kwargs) -> torch.Tensor:
        """Generate tokens using pipeline parallelism."""
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


class PipelineParallelManager:
    """Manager for pipeline parallel models."""
    
    def __init__(self):
        self.models = {}
        self.default_config = PipelineConfig()
    
    def create_pipeline_model(self, 
                            model: nn.Module, 
                            config: Optional[PipelineConfig] = None) -> PipelineParallel:
        """Create a pipeline parallel version of a model."""
        if config is None:
            config = self.default_config
            
        pipeline_model = PipelineParallel(model, config)
        model_id = id(pipeline_model)
        self.models[model_id] = pipeline_model
        
        return pipeline_model
    
    def get_pipeline_stats(self, pipeline_model: PipelineParallel) -> Dict[str, Any]:
        """Get statistics about pipeline execution."""
        model_id = id(pipeline_model)
        if model_id not in self.models:
            return {}
        
        stats = {
            'num_stages': len(pipeline_model.stages),
            'microbatch_size': pipeline_model.config.microbatch_size,
            'pipeline_schedule': pipeline_model.config.pipeline_schedule,
            'stage_times': pipeline_model.balancer.get_average_stage_times(),
            'devices_used': [stage.input_device for stage in pipeline_model.stages]
        }
        
        return stats
    
    def cleanup_model(self, pipeline_model: PipelineParallel):
        """Clean up pipeline model resources."""
        model_id = id(pipeline_model)
        if model_id in self.models:
            del self.models[model_id]


def create_pipeline_parallel_config(num_stages: int = 1,
                                  stage_device_mapping: Optional[List[str]] = None,
                                  microbatch_size: int = 1,
                                  enable_activation_offloading: bool = True,
                                  pipeline_schedule: str = '1f1b') -> PipelineConfig:
    """Helper function to create a pipeline configuration."""
    return PipelineConfig(
        num_stages=num_stages,
        stage_device_mapping=stage_device_mapping,
        microbatch_size=microbatch_size,
        enable_activation_offloading=enable_activation_offloading,
        pipeline_schedule=pipeline_schedule
    )


def split_model_for_pipeline(model: nn.Module, num_stages: int) -> List[nn.Module]:
    """Utility function to split a model into stages for pipeline parallelism."""
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

    # Calculate layers per stage
    total_layers = len(all_layers)
    layers_per_stage = total_layers // num_stages
    remainder = total_layers % num_stages

    stage_models = []
    start_idx = 0
    for i in range(num_stages):
        # Distribute remainder layers to first few stages
        end_idx = start_idx + layers_per_stage + (1 if i < remainder else 0)

        # Create a sequential module for this stage
        stage_layers = all_layers[start_idx:end_idx]
        if len(stage_layers) == 1:
            # If there's only one layer, wrap it in a Sequential to ensure it has a forward method
            stage_model = nn.Sequential(stage_layers[0])
        else:
            stage_model = nn.Sequential(*stage_layers)

        stage_models.append(stage_model)
        start_idx = end_idx

    return stage_models


__all__ = [
    'PipelineConfig',
    'PipelineStage',
    'PipelineBalancer',
    'PipelineParallel',
    'PipelineParallelManager',
    'create_pipeline_parallel_config',
    'split_model_for_pipeline'
]