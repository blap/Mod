"""
GLM-4.7 Tensor Parallel Implementation

This module implements tensor parallelism for the GLM-4.7 model.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from ..config import GLM47Config


@dataclass
class TensorParallelConfig:
    """
    Configuration for tensor parallelism.
    """
    tensor_parallel_size: int = 1
    local_rank: int = 0
    world_size: int = 1
    init_method: str = "tcp://localhost:29500"


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism for GLM-4.7 model.
    
    Splits the weight matrix along the output dimension, so each GPU computes
    a portion of the output.
    """
    def __init__(self, input_size: int, output_size: int, 
                 tensor_parallel_size: int = 1, bias: bool = True):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.tensor_parallel_size = tensor_parallel_size
        
        # Calculate partitioned output size
        self.partitioned_output_size = output_size // tensor_parallel_size
        
        # Initialize the partitioned weight
        self.weight = nn.Parameter(
            torch.empty(self.partitioned_output_size, input_size)
        )
        
        # Initialize bias if needed (only on first partition)
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.partitioned_output_size)
            )
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize the parameters."""
        # Xavier initialization for weights
        nn.init.xavier_uniform_(self.weight)
        
        # Initialize bias to zeros if it exists
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with column parallelism.
        
        Args:
            input: Input tensor of shape (batch_size, ..., input_size)
            
        Returns:
            Output tensor of shape (batch_size, ..., partitioned_output_size)
        """
        # Perform local matrix multiplication
        output = torch.matmul(input, self.weight.t())
        
        # Add bias if it exists
        if self.bias is not None:
            output = output + self.bias
        
        return output


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism for GLM-4.7 model.
    
    Splits the weight matrix along the input dimension, so each GPU processes
    a portion of the input features.
    """
    def __init__(self, input_size: int, output_size: int,
                 tensor_parallel_size: int = 1, bias: bool = True):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.tensor_parallel_size = tensor_parallel_size
        
        # Calculate partitioned input size
        self.partitioned_input_size = input_size // tensor_parallel_size
        
        # Initialize the partitioned weight
        self.weight = nn.Parameter(
            torch.empty(output_size, self.partitioned_input_size)
        )
        
        # Initialize bias if needed (only on last rank for row parallel)
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize the parameters."""
        # Xavier initialization for weights
        nn.init.xavier_uniform_(self.weight)
        
        # Initialize bias to zeros if it exists
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with row parallelism.
        
        Args:
            input: Input tensor of shape (batch_size, ..., partitioned_input_size)
            
        Returns:
            Output tensor of shape (batch_size, ..., output_size)
        """
        # Perform local matrix multiplication
        output = torch.matmul(input, self.weight.t())
        
        # Add bias if it exists
        if self.bias is not None:
            output = output + self.bias
        
        return output


def convert_linear_to_tensor_parallel(
    model: nn.Module, 
    config: TensorParallelConfig
) -> nn.Module:
    """
    Convert linear layers in the model to tensor parallel versions.
    
    Args:
        model: The model to convert
        config: Tensor parallel configuration
        
    Returns:
        Converted model with tensor parallel layers
    """
    if config.tensor_parallel_size <= 1:
        return model
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Determine if this should be column or row parallel
            # Column parallel: when output dimension is split (e.g., QKV projections, FFN up-projection)
            # Row parallel: when input dimension is split (e.g., O projection, FFN down-projection)
            
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = _get_parent_module(model, parent_name)
            
            # Heuristic: if the layer name suggests it's an output projection, use row parallel
            # Otherwise, use column parallel for input/output expansions
            is_output_projection = any(proj in name for proj in ['o_proj', 'down_proj', 'lm_head'])
            
            if is_output_projection:
                # Use row parallel for output projections
                new_module = RowParallelLinear(
                    input_size=module.in_features,
                    output_size=module.out_features,
                    tensor_parallel_size=config.tensor_parallel_size,
                    bias=module.bias is not None
                )
            else:
                # Use column parallel for input expansions and intermediate computations
                new_module = ColumnParallelLinear(
                    input_size=module.in_features,
                    output_size=module.out_features,
                    tensor_parallel_size=config.tensor_parallel_size,
                    bias=module.bias is not None
                )
            
            # Copy weights to the new module
            with torch.no_grad():
                if is_output_projection:
                    # For row parallel, we partition the input dimension
                    partitioned_size = module.in_features // config.tensor_parallel_size
                    start_idx = config.local_rank * partitioned_size
                    end_idx = (config.local_rank + 1) * partitioned_size
                    
                    new_module.weight.copy_(module.weight[:, start_idx:end_idx])
                    
                    if module.bias is not None and new_module.bias is not None:
                        new_module.bias.copy_(module.bias)
                else:
                    # For column parallel, we partition the output dimension
                    partitioned_size = module.out_features // config.tensor_parallel_size
                    start_idx = config.local_rank * partitioned_size
                    end_idx = (config.local_rank + 1) * partitioned_size
                    
                    new_module.weight.copy_(module.weight[start_idx:end_idx, :])
                    
                    if module.bias is not None and new_module.bias is not None:
                        new_module.bias.copy_(module.bias[start_idx:end_idx])
            
            # Replace the module
            setattr(parent_module, child_name, new_module)
    
    return model


def validate_tensor_parallelism_compatibility(
    model: nn.Module, 
    tensor_parallel_size: int
) -> Tuple[bool, str]:
    """
    Validate if the model is compatible with tensor parallelism.
    
    Args:
        model: The model to validate
        tensor_parallel_size: Size of tensor parallelism
        
    Returns:
        Tuple of (is_compatible, reason)
    """
    # Check if all linear layer dimensions are divisible by tensor_parallel_size
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if both input and output dimensions are divisible by tensor_parallel_size
            if module.in_features % tensor_parallel_size != 0 and module.out_features % tensor_parallel_size != 0:
                return False, f"Linear layer '{name}' has dimensions ({module.in_features}, {module.out_features}) not divisible by tensor_parallel_size {tensor_parallel_size}"
            elif module.out_features % tensor_parallel_size != 0:
                # For column parallel, output must be divisible
                if not any(proj in name for proj in ['o_proj', 'down_proj', 'lm_head']):  # Not an output projection
                    return False, f"Column-parallel linear layer '{name}' output dimension {module.out_features} not divisible by tensor_parallel_size {tensor_parallel_size}"
            elif module.in_features % tensor_parallel_size != 0:
                # For row parallel, input must be divisible
                if any(proj in name for proj in ['o_proj', 'down_proj', 'lm_head']):  # Is an output projection
                    return False, f"Row-parallel linear layer '{name}' input dimension {module.in_features} not divisible by tensor_parallel_size {tensor_parallel_size}"
    
    # Check if tensor_parallel_size is valid
    if tensor_parallel_size <= 0:
        return False, f"tensor_parallel_size must be positive, got {tensor_parallel_size}"
    
    if tensor_parallel_size > torch.cuda.device_count():
        return False, f"tensor_parallel_size {tensor_parallel_size} exceeds available GPU count {torch.cuda.device_count()}"
    
    return True, "Model is compatible with tensor parallelism"


def initialize_tensor_parallelism(config: TensorParallelConfig) -> bool:
    """
    Initialize tensor parallelism environment.
    
    Args:
        config: Tensor parallel configuration
        
    Returns:
        True if initialization succeeded, False otherwise
    """
    try:
        # Check if we have enough GPUs
        if config.tensor_parallel_size > torch.cuda.device_count():
            print(f"Warning: Requested tensor parallel size {config.tensor_parallel_size} "
                  f"exceeds available GPU count {torch.cuda.device_count()}")
            return False
        
        # Set the device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(config.local_rank)
        
        return True
    except Exception as e:
        print(f"Error initializing tensor parallelism: {e}")
        return False


def safe_convert_to_tensor_parallel(
    model: nn.Module, 
    config: TensorParallelConfig
) -> Tuple[nn.Module, bool, Optional[str]]:
    """
    Safely convert a model to tensor parallel version with error handling.
    
    Args:
        model: The model to convert
        config: Tensor parallel configuration
        
    Returns:
        Tuple of (converted_model, success, error_message)
    """
    try:
        # Validate compatibility first
        is_compatible, reason = validate_tensor_parallelism_compatibility(
            model, config.tensor_parallel_size
        )
        
        if not is_compatible:
            return model, False, f"Model not compatible with tensor parallelism: {reason}"
        
        # Initialize tensor parallelism environment
        if not initialize_tensor_parallelism(config):
            return model, False, "Failed to initialize tensor parallelism environment"
        
        # Convert the model
        converted_model = convert_linear_to_tensor_parallel(model, config)
        
        return converted_model, True, None
        
    except Exception as e:
        return model, False, f"Error during tensor parallel conversion: {str(e)}"


def _get_parent_module(model: nn.Module, parent_name: str) -> nn.Module:
    """
    Get parent module by name.
    
    Args:
        model: The model
        parent_name: Name of the parent module
        
    Returns:
        Parent module
    """
    parent_module = model
    for n in parent_name.split('.'):
        if n:  # Skip empty strings
            parent_module = getattr(parent_module, n)
    return parent_module


__all__ = [
    "TensorParallelConfig",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "convert_linear_to_tensor_parallel",
    "validate_tensor_parallelism_compatibility",
    "initialize_tensor_parallelism",
    "safe_convert_to_tensor_parallel"
]