"""
Parameter-efficient adapter layers for Qwen3-VL model.
These adapters allow efficient fine-tuning without changing the core architecture.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class AdapterConfig:
    """
    Configuration for adapter layers.
    """
    # Bottleneck adapter configuration
    adapter_dim: int = 64  # Dimension of the bottleneck layer
    adapter_scalar: float = 1.0  # Scaling factor for adapter output
    adapter_dropout: float = 0.1  # Dropout rate in adapter layers
    
    # LoRA (Low-Rank Adaptation) configuration
    lora_r: int = 8  # Rank for LoRA adaptation
    lora_alpha: int = 16  # Scaling factor for LoRA
    lora_dropout: float = 0.05  # Dropout rate for LoRA
    
    # Hardware-specific optimization
    device_specific: bool = False  # Whether to enable device-specific optimizations
    hardware_config: Optional[Dict[str, Any]] = None  # Hardware-specific parameters
    
    # Task-specific configuration
    task_name: Optional[str] = None  # Name of the task this adapter is for
    is_trainable: bool = True  # Whether the adapter parameters are trainable


class BottleneckAdapter(nn.Module):
    """
    Bottleneck adapter layer that can be inserted into transformer layers.
    This follows the standard adapter architecture: down-project -> non-linearity -> up-project.
    """
    def __init__(self, config: AdapterConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.adapter_dim = config.adapter_dim
        self.scaling = config.adapter_scalar
        
        # Down projection: input_dim -> adapter_dim
        self.down_proj = nn.Linear(input_dim, self.adapter_dim, bias=False)
        
        # Non-linearity
        self.activation = nn.ReLU()
        
        # Up projection: adapter_dim -> input_dim
        self.up_proj = nn.Linear(self.adapter_dim, input_dim, bias=False)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.adapter_dropout)
        
        # Initialize adapter weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize adapter weights."""
        nn.init.kaiming_uniform_(self.down_proj.weight, a=0, mode='fan_in', nonlinearity='relu')
        # Initialize up_proj with small random values instead of zeros to ensure the adapter does something
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the bottleneck adapter.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of same shape as input with adapter transformation added
        """
        # Down projection
        down_projected = self.down_proj(hidden_states)
        
        # Apply activation and dropout
        activated = self.activation(down_projected)
        activated = self.dropout(activated)
        
        # Up projection
        up_projected = self.up_proj(activated)
        
        # Scale and add to original
        output = hidden_states + (up_projected * self.scaling)
        
        return output


class LoraLinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer that replaces a linear transformation.
    This creates a low-rank decomposition of the weight update: A * B where A and B are trainable.
    """
    def __init__(self, base_layer: nn.Linear, config: AdapterConfig):
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        
        # Store original dimensions
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = config.lora_r
        self.lora_alpha = config.lora_alpha
        self.scaling = self.lora_alpha / self.r
        
        # Create low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros((self.r, self.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, self.r)))
        
        # Dropout for LoRA
        self.lora_dropout = nn.Dropout(config.lora_dropout)
        
        # Initialize LoRA weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LoRA weights."""
        nn.init.kaiming_uniform_(self.lora_A, a=0, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.
        
        Args:
            x: Input tensor of shape (batch_size, ..., in_features)
            
        Returns:
            Output tensor of shape (batch_size, ..., out_features)
        """
        # Apply base layer
        base_output = self.base_layer(x)
        
        # Apply LoRA transformation
        lora_input = self.lora_dropout(x)
        lora_output = torch.matmul(lora_input, self.lora_A.T)
        lora_output = torch.matmul(lora_output, self.lora_B.T)
        
        # Scale and add to base output
        lora_output = lora_output * self.scaling
        
        return base_output + lora_output


class HardwareAwareAdapter(nn.Module):
    """
    Hardware-aware adapter that optimizes for specific hardware configurations.
    This can include different optimizations based on GPU/CPU capabilities.
    """
    def __init__(self, config: AdapterConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        # Determine hardware-specific configuration
        self.hardware_config = config.hardware_config or {}
        
        # Use different adapter implementations based on hardware
        if config.device_specific and self.hardware_config.get('use_optimized_adapter', False):
            # For optimized hardware, use a more efficient implementation
            self.adapter = self._create_optimized_adapter()
        else:
            # Use standard bottleneck adapter
            self.adapter = BottleneckAdapter(config, input_dim)
    
    def _create_optimized_adapter(self):
        """Create an optimized adapter based on hardware configuration."""
        # This could implement fused operations, specialized kernels, etc.
        # For now, we'll use a simplified version with hardware-specific parameters
        return BottleneckAdapter(self.config, self.input_dim)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through hardware-aware adapter."""
        return self.adapter(hidden_states)


class TaskSpecificAdapter(nn.Module):
    """
    Task-specific adapter that can be enabled/disabled based on the current task.
    This allows for multi-task adaptation without interference.
    """
    def __init__(self, config: AdapterConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.task_name = config.task_name or "default"
        
        # Create the underlying adapter
        if config.device_specific:
            self.adapter = HardwareAwareAdapter(config, input_dim)
        else:
            self.adapter = BottleneckAdapter(config, input_dim)
        
        # Whether this adapter is currently active
        self.is_active = True
    
    def set_active(self, active: bool):
        """Set whether this adapter is active."""
        self.is_active = active
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with task-specific activation control."""
        if self.is_active and self.config.is_trainable:
            return self.adapter(hidden_states)
        else:
            return hidden_states


class AdapterLayer(nn.Module):
    """
    A flexible adapter layer that can contain multiple types of adapters
    and route inputs based on various conditions.
    """
    def __init__(self, config: AdapterConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        # Create the primary adapter
        if config.device_specific:
            self.primary_adapter = HardwareAwareAdapter(config, input_dim)
        else:
            self.primary_adapter = BottleneckAdapter(config, input_dim)
        
        # Dictionary to hold task-specific adapters
        self.task_adapters = nn.ModuleDict()
        
        # Whether to use task-specific adapters
        self.use_task_adapters = config.task_name is not None
    
    def add_task_adapter(self, task_name: str, task_config: AdapterConfig):
        """Add a task-specific adapter."""
        if task_name not in self.task_adapters:
            task_config.task_name = task_name
            self.task_adapters[task_name] = TaskSpecificAdapter(task_config, self.input_dim)
    
    def set_active_task(self, task_name: str):
        """Set the active task, enabling its adapter and disabling others."""
        for name, adapter in self.task_adapters.items():
            adapter.set_active(name == task_name)
    
    def forward(self, hidden_states: torch.Tensor, task_name: Optional[str] = None) -> torch.Tensor:
        """
        Forward pass through the adapter layer.
        
        Args:
            hidden_states: Input tensor
            task_name: Optional task name to use task-specific adapter
            
        Returns:
            Output tensor with adapter transformation applied
        """
        output = hidden_states
        
        # Apply primary adapter if enabled
        if self.config.is_trainable:
            output = self.primary_adapter(output)
        
        # Apply task-specific adapter if provided
        if task_name and task_name in self.task_adapters:
            output = self.task_adapters[task_name](output)
        elif self.use_task_adapters and self.config.task_name in self.task_adapters:
            output = self.task_adapters[self.config.task_name](output)
        
        return output


class ParallelAdapter(nn.Module):
    """
    Parallel adapter that combines multiple adapters in parallel.
    This allows for combining different types of adaptations simultaneously.
    """
    def __init__(self, configs: List[AdapterConfig], input_dim: int):
        super().__init__()
        self.adapters = nn.ModuleList([
            AdapterLayer(config, input_dim) for config in configs
        ])
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass applying all adapters in parallel and summing outputs."""
        outputs = []
        for adapter in self.adapters:
            outputs.append(adapter(hidden_states))
        
        # Sum all adapter outputs (with original if needed)
        combined_output = hidden_states
        for output in outputs:
            combined_output = combined_output + (output - hidden_states)  # Add only the delta
        
        return combined_output


class ResidualAdapter(nn.Module):
    """
    Adapter with residual connection that can be conditionally applied.
    This allows for more flexible integration with existing models.
    """
    def __init__(self, config: AdapterConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        # Create adapter with gate for conditional application
        if config.device_specific:
            self.adapter = HardwareAwareAdapter(config, input_dim)
        else:
            self.adapter = BottleneckAdapter(config, input_dim)
        
        # Gate to control adapter application
        self.gate = nn.Parameter(torch.zeros(1))
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with gated adapter application."""
        adapter_output = self.adapter(hidden_states)
        
        # Apply gating
        gated_output = hidden_states + torch.tanh(self.gate) * (adapter_output - hidden_states)
        
        return gated_output