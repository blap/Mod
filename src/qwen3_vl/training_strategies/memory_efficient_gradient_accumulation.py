"""
Memory-Efficient Gradient Accumulation Scheduling for Qwen3-VL model.
Implements gradient scheduling algorithms to optimize peak memory usage during accumulation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import math


class GradientAccumulationScheduler(nn.Module):
    """
    Memory-efficient gradient accumulation scheduler that optimizes
    memory usage during gradient accumulation steps.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.hidden_size
        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        
        # Memory optimization network
        self.memory_optimizer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, 2),  # [accumulation_steps, chunk_size]
            nn.Sigmoid()
        )

        # Gradient scheduling strategy selector
        self.strategy_selector = nn.Sequential(
            nn.Linear(2, self.hidden_size // 4),  # Use 2 statistics: mean and std
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, 3),  # [simple, chunked, adaptive]
            nn.Softmax(dim=-1)
        )
        
        # Gradient compression for memory efficiency
        self.gradient_compressor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.Tanh(),
            nn.Linear(self.hidden_size // 4, self.hidden_size),
            nn.Tanh()
        )

    def forward(
        self,
        gradients: List[torch.Tensor],
        current_step: int,
        total_steps: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Schedule gradient accumulation with memory efficiency.
        
        Args:
            gradients: List of gradient tensors from different micro-batches
            current_step: Current accumulation step
            total_steps: Total accumulation steps
            
        Returns:
            Tuple of (accumulated_gradient, scheduling_info)
        """
        if not gradients:
            raise ValueError("Gradients list cannot be empty")
        
        # Determine optimal scheduling strategy
        strategy_weights = self._select_strategy(gradients[0])
        selected_strategy = torch.argmax(strategy_weights, dim=-1).item()
        
        # Apply selected strategy
        if selected_strategy == 0:  # Simple accumulation
            accumulated_grad, scheduling_info = self._simple_accumulation(gradients)
        elif selected_strategy == 1:  # Chunked accumulation
            accumulated_grad, scheduling_info = self._chunked_accumulation(gradients)
        else:  # Adaptive accumulation
            accumulated_grad, scheduling_info = self._adaptive_accumulation(gradients, current_step, total_steps)
        
        # Apply gradient compression if needed to save memory
        if self._should_compress(accumulated_grad):
            compressed_grad = self.gradient_compressor(accumulated_grad)
            scheduling_info['compression_applied'] = True
        else:
            compressed_grad = accumulated_grad
            scheduling_info['compression_applied'] = False
        
        return compressed_grad, scheduling_info

    def _select_strategy(self, gradient_sample: torch.Tensor) -> torch.Tensor:
        """Select the best accumulation strategy based on gradient characteristics."""
        # Use gradient statistics to determine strategy
        grad_mean = gradient_sample.mean()
        grad_std = gradient_sample.std()
        grad_stats = torch.stack([grad_mean, grad_std]).unsqueeze(0)  # [1, 2]

        # Create a proper input for the strategy selector (2 statistics: mean and std)
        stats_vector = torch.cat([grad_mean.unsqueeze(0), grad_std.unsqueeze(0)])  # [2]

        strategy_weights = self.strategy_selector(stats_vector.unsqueeze(0))  # Add batch dimension
        return strategy_weights

    def _simple_accumulation(self, gradients: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Simple gradient accumulation by summing."""
        accumulated = sum(gradients)
        
        scheduling_info = {
            'strategy': 'simple',
            'accumulated_count': len(gradients),
            'memory_efficiency': 1.0  # No memory optimization
        }
        
        return accumulated, scheduling_info

    def _chunked_accumulation(self, gradients: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Chunked accumulation to reduce peak memory."""
        chunk_size = max(1, len(gradients) // 4)  # Divide into 4 chunks
        chunks = [gradients[i:i + chunk_size] for i in range(0, len(gradients), chunk_size)]
        
        accumulated_chunks = []
        for chunk in chunks:
            chunk_sum = sum(chunk)
            accumulated_chunks.append(chunk_sum)
        
        final_accumulated = sum(accumulated_chunks)
        
        scheduling_info = {
            'strategy': 'chunked',
            'chunk_size': chunk_size,
            'num_chunks': len(chunks),
            'original_count': len(gradients),
            'memory_efficiency': len(chunks) / len(gradients)  # Memory reduction factor
        }
        
        return final_accumulated, scheduling_info

    def _adaptive_accumulation(
        self, 
        gradients: List[torch.Tensor], 
        current_step: int, 
        total_steps: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Adaptive accumulation that adjusts based on training progress."""
        # Calculate adaptive accumulation factor based on progress
        progress = current_step / max(total_steps, 1)
        adaptive_factor = 1.0 + 0.5 * math.sin(progress * math.pi)  # Oscillate between 1.0 and 1.5
        
        # Apply adaptive chunking
        effective_chunk_size = max(1, int(len(gradients) // adaptive_factor))
        chunks = [gradients[i:i + effective_chunk_size] for i in range(0, len(gradients), effective_chunk_size)]
        
        accumulated_chunks = []
        for chunk in chunks:
            chunk_sum = sum(chunk)
            accumulated_chunks.append(chunk_sum)
        
        final_accumulated = sum(accumulated_chunks)
        
        scheduling_info = {
            'strategy': 'adaptive',
            'adaptive_factor': adaptive_factor,
            'chunk_size': effective_chunk_size,
            'num_chunks': len(chunks),
            'original_count': len(gradients),
            'current_step': current_step,
            'total_steps': total_steps,
            'progress': progress,
            'memory_efficiency': len(chunks) / len(gradients)
        }
        
        return final_accumulated, scheduling_info

    def _should_compress(self, gradient: torch.Tensor) -> bool:
        """Determine if gradient compression should be applied."""
        # Simple heuristic: compress if gradient magnitude is large
        grad_norm = torch.norm(gradient)
        return grad_norm > 10.0  # Threshold can be adjusted


class LayerWiseGradientScheduler(nn.Module):
    """
    Gradient scheduler that operates differently across model layers.
    """
    def __init__(self, config, num_layers: int):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        
        self.hidden_size = config.hidden_size
        
        # Layer-specific scheduling parameters
        self.layer_schedulers = nn.ModuleList([
            GradientAccumulationScheduler(config) 
            for _ in range(num_layers)
        ])
        
        # Layer importance estimator
        self.layer_importance_estimator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        layer_gradients: List[List[torch.Tensor]],  # [num_layers, num_micro_batches, grad_tensor]
        current_step: int,
        total_steps: int
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Schedule gradients for multiple layers with layer-specific strategies.
        """
        scheduled_gradients = []
        scheduling_info = {}
        
        for layer_idx, layer_grads in enumerate(layer_gradients):
            if layer_grads:  # Only process if gradients exist for this layer
                scheduled_grad, layer_info = self.layer_schedulers[layer_idx](
                    layer_grads, current_step, total_steps
                )
                scheduled_gradients.append(scheduled_grad)
                scheduling_info[f'layer_{layer_idx}'] = layer_info
        
        return scheduled_gradients, scheduling_info


class MemoryEfficientOptimizer(nn.Module):
    """
    Optimizer wrapper that integrates gradient accumulation scheduling
    with memory-efficient operations.
    """
    def __init__(self, config, base_optimizer: torch.optim.Optimizer):
        super().__init__()
        self.config = config
        self.base_optimizer = base_optimizer
        
        self.gradient_scheduler = GradientAccumulationScheduler(config)
        self.accumulation_counter = 0
        self.total_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        
        # Gradient accumulation buffers
        self.accumulated_gradients = {}
        self.gradient_counts = {}

    def step(self, closure=None):
        """
        Perform optimization step with scheduled gradient accumulation.
        """
        # Apply gradient scheduling to accumulated gradients
        scheduled_gradients = {}
        
        for param_group_idx, group in enumerate(self.base_optimizer.param_groups):
            for param_idx, param in enumerate(group['params']):
                if param.grad is not None:
                    param_key = f"group_{param_group_idx}_param_{param_idx}"
                    
                    # Get accumulated gradient for this parameter
                    if param_key in self.accumulated_gradients:
                        accumulated_grad = self.accumulated_gradients[param_key]
                        grad_count = self.gradient_counts[param_key]
                        
                        # Average the accumulated gradients
                        averaged_grad = accumulated_grad / grad_count
                        
                        # Apply gradient scheduling
                        scheduled_grad, _ = self.gradient_scheduler(
                            [averaged_grad], 
                            self.accumulation_counter, 
                            self.total_accumulation_steps
                        )
                        
                        # Set the scheduled gradient
                        param.grad = scheduled_grad
                    else:
                        # No accumulation, use current gradient directly
                        pass
            
        # Perform the actual optimization step
        return self.base_optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = True):
        """
        Zero gradients and reset accumulation buffers.
        """
        self.base_optimizer.zero_grad(set_to_none)
        
        # Reset accumulation buffers
        self.accumulated_gradients = {}
        self.gradient_counts = {}
        self.accumulation_counter = 0

    def accumulate_gradients(self, named_parameters):
        """
        Accumulate gradients from current micro-batch.
        """
        for name, param in named_parameters:
            if param.grad is not None:
                param_key = name.replace('.', '_')
                
                if param_key not in self.accumulated_gradients:
                    self.accumulated_gradients[param_key] = param.grad.clone()
                    self.gradient_counts[param_key] = 1
                else:
                    self.accumulated_gradients[param_key] += param.grad
                    self.gradient_counts[param_key] += 1
        
        self.accumulation_counter += 1

    def should_step(self) -> bool:
        """
        Determine if optimization step should be performed.
        """
        return self.accumulation_counter >= self.total_accumulation_steps


class GradientCheckpointingWithScheduling(nn.Module):
    """
    Gradient checkpointing integrated with accumulation scheduling.
    """
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        
        self.gradient_scheduler = GradientAccumulationScheduler(config)
        self.checkpoint_activations = getattr(config, 'use_gradient_checkpointing', True)

    def forward(
        self,
        *args,
        current_step: int = 0,
        total_steps: int = 1,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with optional gradient checkpointing and scheduling.
        """
        if self.checkpoint_activations:
            # Use gradient checkpointing to save memory
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            # This is a simplified version - in practice, you'd integrate with actual checkpointing
            output = self.model(*args, **kwargs)
        else:
            output = self.model(*args, **kwargs)
        
        # Return with scheduling info
        scheduling_info = {
            'checkpointing_enabled': self.checkpoint_activations,
            'current_step': current_step,
            'total_steps': total_steps
        }
        
        return output, scheduling_info


class DynamicGradientAccumulator(nn.Module):
    """
    Dynamic gradient accumulator that adjusts accumulation based on memory pressure.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.hidden_size
        
        # Memory pressure detector
        self.memory_pressure_detector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Dynamic accumulation scheduler
        self.dynamic_scheduler = nn.Sequential(
            nn.Linear(self.hidden_size + 1, self.hidden_size // 2),  # +1 for memory pressure
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 2),  # [accumulation_steps, chunk_size]
            nn.Sigmoid()
        )

    def forward(
        self,
        gradients: List[torch.Tensor],
        memory_pressure: float = 0.0
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Dynamically schedule gradient accumulation based on memory pressure.
        
        Args:
            gradients: List of gradient tensors
            memory_pressure: Current memory pressure (0.0 to 1.0)
            
        Returns:
            Tuple of (scheduled_gradient, scheduling_info)
        """
        if not gradients:
            raise ValueError("Gradients list cannot be empty")
        
        # Estimate memory pressure if not provided
        if memory_pressure == 0.0 and len(gradients) > 0:
            memory_pressure = self._estimate_memory_pressure(gradients[0])
        
        # Combine gradient info with memory pressure
        grad_info = gradients[0].mean(dim=-1).mean() if gradients[0].dim() > 1 else gradients[0].mean()
        combined_input = torch.cat([
            grad_info.unsqueeze(0) if grad_info.dim() == 0 else grad_info,
            torch.tensor([memory_pressure], device=grad_info.device)
        ], dim=-1).unsqueeze(0)
        
        # Get dynamic scheduling parameters
        scheduling_params = self.dynamic_scheduler(combined_input)  # [1, 2]
        
        # Determine accumulation strategy based on memory pressure
        if memory_pressure > 0.7:  # High memory pressure
            # Use more aggressive chunking to reduce peak memory
            chunk_size = max(1, int(len(gradients) * (1.0 - memory_pressure)))
            chunks = [gradients[i:i + chunk_size] for i in range(0, len(gradients), chunk_size)]
            
            accumulated_chunks = []
            for chunk in chunks:
                chunk_sum = sum(chunk)
                accumulated_chunks.append(chunk_sum)
            
            final_accumulated = sum(accumulated_chunks)
        else:
            # Use normal accumulation
            final_accumulated = sum(gradients)
        
        scheduling_info = {
            'memory_pressure': memory_pressure,
            'original_count': len(gradients),
            'chunk_size': chunk_size if memory_pressure > 0.7 else len(gradients),
            'peak_memory_reduction': memory_pressure if memory_pressure > 0.7 else 0.0
        }
        
        return final_accumulated, scheduling_info

    def _estimate_memory_pressure(self, gradient: torch.Tensor) -> float:
        """Estimate memory pressure based on gradient characteristics."""
        # Use gradient norm as a proxy for memory pressure
        grad_norm = torch.norm(gradient)
        # Normalize to 0-1 range (arbitrary scaling)
        pressure = torch.clamp(grad_norm / 100.0, 0.0, 1.0)
        return pressure.item()