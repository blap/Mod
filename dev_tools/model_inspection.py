"""
Model Inspection Utilities for Qwen3-VL Model

This module provides comprehensive tools to understand and inspect the architecture 
changes in the Qwen3-VL model, including layer analysis, parameter inspection, 
and architecture visualization.
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dataclasses import dataclass
from functools import partial
import copy


@dataclass
class LayerInfo:
    """Information about a model layer"""
    name: str
    type: str
    parameters: int
    trainable: bool
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    memory_usage: Optional[float] = None  # in MB
    computation_flops: Optional[float] = None  # in FLOPs


@dataclass
class ModelSummary:
    """Summary of the entire model"""
    total_parameters: int
    trainable_parameters: int
    total_memory: float  # in MB
    total_flops: float  # in FLOPs
    layer_count: int
    layer_types: Dict[str, int]
    architecture: str


class ModelInspector:
    """Main model inspection class"""
    
    def __init__(self):
        self.layer_info = []
        self.model_summary = None
        self.activation_maps = {}
        self.gradient_maps = {}
        self.hooks = []
    
    def inspect_model(self, model: nn.Module, input_shape: tuple = None) -> ModelSummary:
        """Comprehensive model inspection"""
        self.layer_info = []
        
        # Count parameters and get basic info
        total_params = 0
        trainable_params = 0
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                param_count = sum(p.numel() for p in module.parameters(recurse=False))
                trainable = any(p.requires_grad for p in module.parameters(recurse=False))
                
                layer_info = LayerInfo(
                    name=name,
                    type=type(module).__name__,
                    parameters=param_count,
                    trainable=trainable
                )
                
                self.layer_info.append(layer_info)
                
                if trainable:
                    trainable_params += param_count
                total_params += param_count
        
        # Calculate summary statistics
        layer_types = defaultdict(int)
        for layer in self.layer_info:
            layer_types[layer.type] += 1
        
        self.model_summary = ModelSummary(
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            total_memory=self._estimate_memory(model),
            total_flops=self._estimate_flops(model, input_shape) if input_shape else 0,
            layer_count=len(self.layer_info),
            layer_types=dict(layer_types),
            architecture=self._get_architecture_string(model)
        )
        
        return self.model_summary
    
    def _estimate_memory(self, model: nn.Module) -> float:
        """Estimate model memory usage in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        return total_size / 1024 / 1024  # Convert to MB
    
    def _estimate_flops(self, model: nn.Module, input_shape: tuple) -> float:
        """Estimate FLOPs for the model with given input shape"""
        # This is a simplified FLOP estimation - in practice, you'd use a more sophisticated method
        try:
            dummy_input = torch.randn(1, *input_shape[1:])  # Assume batch size of 1
            flops = 0
            
            def count_conv2d(m, x, y):
                nonlocal flops
                x = x[0]
                kernel_ops = m.weight.size()[2] * m.weight.size()[3] * m.in_channels
                bias_ops = 1 if m.bias is not None else 0
                flops += y.nelement() * (kernel_ops + bias_ops)
            
            def count_linear(m, x, y):
                nonlocal flops
                total_ops = m.in_features * m.out_features
                if m.bias is not None:
                    total_ops += m.out_features
                flops += total_ops
            
            def count_matmul(m, x, y):
                nonlocal flops
                # For matrix multiplication, FLOPs = 2 * m * n * k (for A*B where A is m×k, B is k×n)
                if len(x[0].shape) == 3 and len(y.shape) == 3:  # batch matrix multiplication
                    batch, m, k = x[0].shape
                    _, k2, n = y.shape
                    flops += 2 * batch * m * k * n
                elif len(x[0].shape) == 2 and len(y.shape) == 2:
                    m, k = x[0].shape
                    k2, n = y.shape
                    flops += 2 * m * k * n
            
            # Register hooks for different layer types
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    hooks.append(module.register_forward_hook(count_conv2d))
                elif isinstance(module, nn.Linear):
                    hooks.append(module.register_forward_hook(count_linear))
                elif hasattr(module, 'matmul'):  # This is a simplification
                    hooks.append(module.register_forward_hook(count_matmul))
            
            # Run a forward pass to trigger hooks
            try:
                with torch.no_grad():
                    _ = model(dummy_input)
            except:
                # If forward pass fails, return 0 FLOPs
                flops = 0
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            return flops
        except:
            return 0  # Return 0 if estimation fails
    
    def _get_architecture_string(self, model: nn.Module) -> str:
        """Get a string representation of the model architecture"""
        arch_parts = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                arch_parts.append(f"{name}: {type(module).__name__}")
        return " -> ".join(arch_parts)
    
    def get_layer_analysis(self) -> List[LayerInfo]:
        """Get detailed layer analysis"""
        return self.layer_info
    
    def get_parameter_distribution(self) -> Dict[str, Any]:
        """Get parameter distribution by layer type"""
        distribution = defaultdict(int)
        for layer in self.layer_info:
            distribution[layer.type] += layer.parameters
        
        return dict(distribution)
    
    def visualize_parameter_distribution(self, figsize: tuple = (10, 6)):
        """Visualize parameter distribution by layer type"""
        if not self.layer_info:
            print("No layer info available. Run inspect_model first.")
            return
        
        param_dist = self.get_parameter_distribution()
        
        plt.figure(figsize=figsize)
        layers = list(param_dist.keys())
        params = list(param_dist.values())
        
        # Sort by parameter count
        sorted_pairs = sorted(zip(layers, params), key=lambda x: x[1], reverse=True)
        sorted_layers, sorted_params = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        plt.bar(range(len(sorted_layers)), sorted_params)
        plt.title('Parameter Distribution by Layer Type')
        plt.xlabel('Layer Type')
        plt.ylabel('Number of Parameters')
        plt.xticks(range(len(sorted_layers)), sorted_layers, rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def visualize_model_architecture(self, figsize: tuple = (12, 8)):
        """Visualize model architecture as a flowchart"""
        if not self.layer_info:
            print("No layer info available. Run inspect_model first.")
            return
        
        plt.figure(figsize=figsize)
        
        # Create a simple visualization of layer types
        layer_types = [layer.type for layer in self.layer_info]
        layer_names = [layer.name.split('.')[-1] for layer in self.layer_info]  # Just the layer name, not full path
        
        # Create a heatmap-like visualization
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a matrix where each row is a layer and we mark different properties
        matrix = np.zeros((len(layer_types), 3))  # 3 properties: trainable, param count category, layer type category
        
        for i, layer in enumerate(self.layer_info):
            matrix[i, 0] = 1 if layer.trainable else 0  # Trainable
            matrix[i, 1] = min(layer.parameters / 1000, 10)  # Parameter count (normalized)
            # Layer type category (simplified)
            if 'Linear' in layer.type:
                matrix[i, 2] = 1
            elif 'Conv' in layer.type:
                matrix[i, 2] = 2
            elif 'Norm' in layer.type:
                matrix[i, 2] = 3
            elif 'Activation' in layer.type:
                matrix[i, 2] = 4
            else:
                matrix[i, 2] = 5
        
        im = ax.imshow(matrix.T, aspect='auto', cmap='viridis')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Trainable', 'Param Count', 'Layer Type'])
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.set_title('Model Architecture Visualization')
        
        plt.colorbar(im)
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, model1: nn.Module, model2: nn.Module, input_shape: tuple = None) -> Dict[str, Any]:
        """Compare two models"""
        summary1 = self.inspect_model(model1, input_shape)
        
        # Store first summary
        summary1_data = {
            'model1': summary1,
            'layer_info1': self.layer_info.copy()
        }
        
        # Inspect second model
        summary2 = self.inspect_model(model2, input_shape)
        
        comparison = {
            'model1': {
                'total_parameters': summary1.total_parameters,
                'trainable_parameters': summary1.trainable_parameters,
                'total_memory': summary1.total_memory,
                'total_flops': summary1.total_flops,
                'layer_count': summary1.layer_count
            },
            'model2': {
                'total_parameters': summary2.total_parameters,
                'trainable_parameters': summary2.trainable_parameters,
                'total_memory': summary2.total_memory,
                'total_flops': summary2.total_flops,
                'layer_count': summary2.layer_count
            },
            'differences': {
                'parameter_diff': summary2.total_parameters - summary1.total_parameters,
                'memory_diff': summary2.total_memory - summary1.total_memory,
                'flops_diff': summary2.total_flops - summary1.total_flops,
                'layer_count_diff': summary2.layer_count - summary1.layer_count
            }
        }
        
        return comparison
    
    def analyze_tensor_shapes(self, model: nn.Module, sample_input: torch.Tensor):
        """Analyze tensor shapes through the model"""
        shapes = {}
        
        def hook_fn(module, input, output):
            module_name = str(module.__class__.__name__)
            if module_name not in shapes:
                shapes[module_name] = {'input_shapes': [], 'output_shapes': []}
            
            # Record input shapes
            if isinstance(input, tuple):
                for inp in input:
                    if isinstance(inp, torch.Tensor):
                        shapes[module_name]['input_shapes'].append(tuple(inp.shape))
            elif isinstance(input, torch.Tensor):
                shapes[module_name]['input_shapes'].append(tuple(input.shape))
            
            # Record output shapes
            if isinstance(output, tuple):
                for out in output:
                    if isinstance(out, torch.Tensor):
                        shapes[module_name]['output_shapes'].append(tuple(out.shape))
            elif isinstance(output, torch.Tensor):
                shapes[module_name]['output_shapes'].append(tuple(output.shape))
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Run forward pass
        with torch.no_grad():
            _ = model(sample_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return shapes
    
    def detect_architecture_changes(self, old_model: nn.Module, new_model: nn.Module) -> Dict[str, Any]:
        """Detect changes between two model architectures"""
        old_summary = self.inspect_model(old_model)
        old_info = self.layer_info.copy()
        
        new_summary = self.inspect_model(new_model)
        new_info = self.layer_info.copy()
        
        # Compare layer counts by type
        old_types = {k: v for k, v in old_summary.layer_types.items()}
        new_types = {k: v for k, v in new_summary.layer_types.items()}
        
        added_types = {k: v for k, v in new_types.items() if k not in old_types or v > old_types.get(k, 0)}
        removed_types = {k: v for k, v in old_types.items() if k not in new_types or v > new_types.get(k, 0)}
        
        changes = {
            'parameter_change': new_summary.total_parameters - old_summary.total_parameters,
            'memory_change': new_summary.total_memory - old_summary.total_memory,
            'layer_count_change': new_summary.layer_count - old_summary.layer_count,
            'added_layer_types': added_types,
            'removed_layer_types': removed_types,
            'new_vs_old_ratio': new_summary.total_parameters / (old_summary.total_parameters or 1)
        }
        
        return changes
    
    def export_inspection_report(self, path: str):
        """Export inspection results to a JSON file"""
        if not self.model_summary:
            raise ValueError("No model inspection data available. Run inspect_model first.")
        
        report = {
            'model_summary': {
                'total_parameters': self.model_summary.total_parameters,
                'trainable_parameters': self.model_summary.trainable_parameters,
                'total_memory_mb': self.model_summary.total_memory,
                'total_flops': self.model_summary.total_flops,
                'layer_count': self.model_summary.layer_count,
                'layer_types': self.model_summary.layer_types,
                'architecture': self.model_summary.architecture
            },
            'layer_analysis': [
                {
                    'name': layer.name,
                    'type': layer.type,
                    'parameters': layer.parameters,
                    'trainable': layer.trainable,
                    'input_shape': layer.input_shape,
                    'output_shape': layer.output_shape,
                    'memory_usage_mb': layer.memory_usage,
                    'computation_flops': layer.computation_flops
                }
                for layer in self.layer_info
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def print_model_summary(self):
        """Print a formatted model summary"""
        if not self.model_summary:
            print("No model summary available. Run inspect_model first.")
            return
        
        print("=== Model Architecture Summary ===")
        print(f"Total Parameters: {self.model_summary.total_parameters:,}")
        print(f"Trainable Parameters: {self.model_summary.trainable_parameters:,}")
        print(f"Total Memory: {self.model_summary.total_memory:.2f} MB")
        print(f"Estimated FLOPs: {self.model_summary.total_flops:.2e}")
        print(f"Layer Count: {self.model_summary.layer_count}")
        print("\nLayer Types Distribution:")
        for layer_type, count in self.model_summary.layer_types.items():
            print(f"  {layer_type}: {count}")
        print(f"\nArchitecture: {self.model_summary.architecture}")


class ParameterAnalyzer:
    """Advanced parameter analysis tools"""
    
    def __init__(self):
        self.parameter_stats = {}
    
    def analyze_parameters(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model parameters in detail"""
        stats = {}
        
        for name, param in model.named_parameters():
            stats[name] = {
                'shape': list(param.shape),
                'numel': param.numel(),
                'requires_grad': param.requires_grad,
                'mean': float(param.mean().item()),
                'std': float(param.std().item()),
                'min': float(param.min().item()),
                'max': float(param.max().item()),
                'has_nan': bool(torch.isnan(param).any().item()),
                'has_inf': bool(torch.isinf(param).any().item()),
                'sparsity': float((param == 0).sum().item()) / param.numel()
            }
        
        self.parameter_stats = stats
        return stats
    
    def find_large_parameters(self, threshold: float = 1.0) -> List[str]:
        """Find layers with large parameters (potential outliers)"""
        large_params = []
        for name, stats in self.parameter_stats.items():
            if abs(stats['max']) > threshold or abs(stats['min']) > threshold:
                large_params.append(name)
        return large_params
    
    def find_sparse_layers(self, threshold: float = 0.5) -> List[str]:
        """Find potentially sparse layers"""
        sparse_layers = []
        for name, stats in self.parameter_stats.items():
            if stats['sparsity'] > threshold:
                sparse_layers.append(name)
        return sparse_layers
    
    def analyze_gradient_flow(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze gradient flow through the model"""
        grad_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_stats[name] = {
                    'mean': float(param.grad.mean().item()),
                    'std': float(param.grad.std().item()),
                    'min': float(param.grad.min().item()),
                    'max': float(param.grad.max().item()),
                    'norm': float(param.grad.norm().item()),
                    'has_nan': bool(torch.isnan(param.grad).any().item()),
                    'has_inf': bool(torch.isinf(param.grad).any().item())
                }
            else:
                grad_stats[name] = {'grad_available': False}
        
        return grad_stats


class ArchitectureVisualizer:
    """Advanced visualization tools for model architecture"""
    
    def __init__(self):
        self.plt = plt
        self.sns = sns
    
    def plot_parameter_heatmap(self, model: nn.Module, figsize: tuple = (12, 8)):
        """Plot a heatmap of parameter distribution across layers"""
        # Get parameter counts for each layer
        layer_names = []
        param_counts = []
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                param_count = sum(p.numel() for p in module.parameters(recurse=False))
                if param_count > 0:  # Only include layers with parameters
                    layer_names.append(name.split('.')[-1])  # Just the layer name
                    param_counts.append(param_count)
        
        # Create a 2D representation for heatmap
        # For simplicity, we'll create a dummy 2D array
        if len(param_counts) > 1:
            # Create a square matrix by padding or reshaping
            size = int(np.ceil(np.sqrt(len(param_counts))))
            matrix = np.zeros((size, size))
            
            for i, count in enumerate(param_counts):
                if i < size * size:
                    matrix[i // size, i % size] = count
            
            plt.figure(figsize=figsize)
            sns.heatmap(matrix, annot=True, fmt='.0f', cmap='viridis')
            plt.title('Parameter Distribution Heatmap')
            plt.show()
    
    def plot_layer_complexity(self, model: nn.Module, input_shape: tuple):
        """Plot layer complexity based on parameter count and FLOPs"""
        inspector = ModelInspector()
        inspector.inspect_model(model, input_shape)
        
        layer_names = [layer.name.split('.')[-1] for layer in inspector.get_layer_analysis()]
        param_counts = [layer.parameters for layer in inspector.get_layer_analysis()]
        
        # For FLOPs, we'll use a simple approximation
        flops_approx = [p * 2 for p in param_counts]  # Rough approximation
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Parameters', color=color)
        bars = ax1.bar(range(len(layer_names)), param_counts, color=color, alpha=0.6, label='Parameters')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_xticks(range(len(layer_names)))
        ax1.set_xticklabels(layer_names)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Approx FLOPs', color=color)
        line = ax2.plot(range(len(layer_names)), flops_approx, color=color, marker='o', label='Approx FLOPs')
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.title('Layer Complexity: Parameters vs FLOPs')
        plt.show()


def create_model_diff_report(old_model: nn.Module, new_model: nn.Module, input_shape: tuple = None) -> str:
    """Create a text report showing differences between two models"""
    inspector = ModelInspector()
    changes = inspector.detect_architecture_changes(old_model, new_model)
    
    report = []
    report.append("=== Model Architecture Changes Report ===")
    report.append(f"Parameter Change: {changes['parameter_change']:+,}")
    report.append(f"Memory Change: {changes['memory_change']:+.2f} MB")
    report.append(f"Layer Count Change: {changes['layer_count_change']:+}")
    report.append(f"Size Ratio (New/Old): {changes['new_vs_old_ratio']:.2f}")
    
    if changes['added_layer_types']:
        report.append("\nAdded Layer Types:")
        for layer_type, count in changes['added_layer_types'].items():
            report.append(f"  {layer_type}: +{count}")
    
    if changes['removed_layer_types']:
        report.append("\nRemoved Layer Types:")
        for layer_type, count in changes['removed_layer_types'].items():
            report.append(f"  {layer_type}: -{count}")
    
    return "\n".join(report)


def analyze_model_capacity(model: nn.Module) -> Dict[str, Any]:
    """Analyze model capacity and complexity"""
    inspector = ModelInspector()
    summary = inspector.inspect_model(model)
    
    # Calculate capacity metrics
    capacity_metrics = {
        'total_parameters': summary.total_parameters,
        'trainable_ratio': summary.trainable_parameters / max(summary.total_parameters, 1),
        'memory_efficiency': summary.total_parameters / max(summary.total_memory, 1),  # params per MB
        'parameters_per_layer': summary.total_parameters / max(summary.layer_count, 1),
        'estimated_max_sequence_length': _estimate_max_sequence_length(summary)
    }
    
    return capacity_metrics


def _estimate_max_sequence_length(model_summary: ModelSummary) -> int:
    """Estimate maximum sequence length based on memory"""
    # This is a rough estimate - in practice, this would depend on the specific architecture
    # For transformer models, memory typically scales with sequence_length^2 due to attention
    if model_summary.total_memory > 0:
        # Assume attention dominates memory usage and scales quadratically
        # This is a very rough estimate
        estimated_length = int(np.sqrt(1e9 / max(model_summary.total_parameters, 1)))  # Simplified
        return min(estimated_length, 32768)  # Cap at reasonable value
    return 0


# Example usage functions
def example_model_inspection():
    """Example of model inspection usage"""
    print("=== Model Inspection Example ===")
    
    # Create a simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    
    # Inspect the model
    inspector = ModelInspector()
    summary = inspector.inspect_model(model, input_shape=(1, 3, 32, 32))
    
    # Print summary
    inspector.print_model_summary()
    
    # Visualize parameter distribution
    inspector.visualize_parameter_distribution()
    
    # Analyze parameters
    analyzer = ParameterAnalyzer()
    param_stats = analyzer.analyze_parameters(model)
    
    print(f"\nAnalyzed {len(param_stats)} parameter tensors")
    large_params = analyzer.find_large_parameters(threshold=0.5)
    if large_params:
        print(f"Large parameters found: {large_params}")
    
    # Visualize architecture
    visualizer = ArchitectureVisualizer()
    visualizer.plot_parameter_heatmap(model)
    visualizer.plot_layer_complexity(model, (1, 3, 32, 32))


def example_model_comparison():
    """Example of model comparison"""
    print("\n=== Model Comparison Example ===")
    
    # Create two models with different architectures
    class ModelA(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(100, 200),
                nn.ReLU(),
                nn.Linear(200, 10)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    class ModelB(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(100, 300),
                nn.ReLU(),
                nn.Linear(300, 150),
                nn.ReLU(),
                nn.Linear(150, 10)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model_a = ModelA()
    model_b = ModelB()
    
    # Compare models
    inspector = ModelInspector()
    comparison = inspector.compare_models(model_a, model_b, input_shape=(1, 100))
    
    print("Model Comparison Results:")
    print(f"Model A Parameters: {comparison['model1']['total_parameters']:,}")
    print(f"Model B Parameters: {comparison['model2']['total_parameters']:,}")
    print(f"Difference: {comparison['differences']['parameter_diff']:+,}")
    print(f"Memory Difference: {comparison['differences']['memory_diff']:+.2f} MB")


if __name__ == "__main__":
    example_model_inspection()
    example_model_comparison()