"""
Capacity validation utilities for Qwen3-VL architecture
Ensures that the hierarchical vision processing maintains full model capacity
(32 transformer layers and 32 attention heads).
"""
import torch
import torch.nn as nn
from typing import Dict, Any


def validate_model_capacity(model: nn.Module, expected_layers: int = 32, expected_heads: int = 32) -> Dict[str, Any]:
    """
    Validate that the model maintains the expected capacity.
    
    Args:
        model: The model to validate
        expected_layers: Expected number of transformer layers
        expected_heads: Expected number of attention heads per layer
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'transformer_layers': 0,
        'attention_heads': 0,
        'capacity_preserved': True,
        'details': []
    }
    
    # Count transformer layers in the language model
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
        results['transformer_layers'] = len(model.language_model.layers)
        if results['transformer_layers'] != expected_layers:
            results['capacity_preserved'] = False
            results['details'].append(
                f"Expected {expected_layers} transformer layers, found {results['transformer_layers']}"
            )
    
    # Count attention heads in the language model
    if (hasattr(model, 'language_model') and 
        hasattr(model.language_model, 'layers') and 
        len(model.language_model.layers) > 0):
        first_layer = model.language_model.layers[0]
        if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'num_heads'):
            results['attention_heads'] = first_layer.self_attn.num_heads
            if results['attention_heads'] != expected_heads:
                results['capacity_preserved'] = False
                results['details'].append(
                    f"Expected {expected_heads} attention heads, found {results['attention_heads']}"
                )
        elif hasattr(first_layer, 'layer_block') and hasattr(first_layer.layer_block, 'self_attn'):
            # For layers with enhanced transformer blocks
            results['attention_heads'] = first_layer.layer_block.self_attn.num_heads
            if results['attention_heads'] != expected_heads:
                results['capacity_preserved'] = False
                results['details'].append(
                    f"Expected {expected_heads} attention heads, found {results['attention_heads']}"
                )
    
    # Check vision transformer layers
    if hasattr(model, 'vision_tower'):
        vision_layers = 0
        if hasattr(model.vision_tower, 'layers'):
            vision_layers = len(model.vision_tower.layers)
        elif hasattr(model.vision_tower, 'hierarchical_vision_processor'):
            # For hierarchical vision processor, we need to count internal layers
            vision_processor = model.vision_tower.hierarchical_vision_processor
            if hasattr(vision_processor, 'resolution_adaptive_blocks'):
                vision_layers = len(vision_processor.resolution_adaptive_blocks)
        
        results['details'].append(f"Vision transformer has {vision_layers} layers")
    
    # Check vision attention heads
    if (hasattr(model, 'vision_tower') and 
        hasattr(model.vision_tower, 'layers') and 
        len(model.vision_tower.layers) > 0):
        first_vision_layer = model.vision_tower.layers[0]
        if hasattr(first_vision_layer, 'attn') and hasattr(first_vision_layer.attn, 'num_heads'):
            results['details'].append(
                f"Vision attention heads: {first_vision_layer.attn.num_heads}"
            )
    
    # Check hierarchical vision processor specifically
    if (hasattr(model, 'vision_tower') and 
        hasattr(model.vision_tower, 'hierarchical_vision_processor')):
        hvp = model.vision_tower.hierarchical_vision_processor
        if hasattr(hvp, 'num_hidden_layers'):
            results['details'].append(
                f"Hierarchical vision processor layers: {hvp.num_hidden_layers}"
            )
        if hasattr(hvp, 'num_attention_heads'):
            results['details'].append(
                f"Hierarchical vision processor attention heads: {hvp.num_attention_heads}"
            )
    
    if results['capacity_preserved']:
        results['details'].append("Model capacity validation passed")
    
    return results


def validate_hierarchical_vision_capacity(model: nn.Module) -> Dict[str, Any]:
    """
    Specifically validate the capacity of the hierarchical vision processor.
    
    Args:
        model: The model containing the hierarchical vision processor
    
    Returns:
        Dictionary with validation results for hierarchical vision components
    """
    results = {
        'hierarchical_vision_valid': True,
        'details': []
    }
    
    if not hasattr(model, 'vision_tower') or not hasattr(model.vision_tower, 'hierarchical_vision_processor'):
        results['hierarchical_vision_valid'] = False
        results['details'].append("Hierarchical vision processor not found in model")
        return results
    
    hvp = model.vision_tower.hierarchical_vision_processor
    
    # Check that all required components exist
    required_attrs = [
        'multi_resolution_analyzer',
        'hierarchical_feature_extractor', 
        'resolution_adaptive_blocks',
        'complexity_assessor',
        'layer_norm',
        'output_proj'
    ]
    
    missing_attrs = [attr for attr in required_attrs if not hasattr(hvp, attr)]
    if missing_attrs:
        results['hierarchical_vision_valid'] = False
        results['details'].append(f"Missing attributes in hierarchical vision processor: {missing_attrs}")
    
    # Check that resolution adaptive blocks maintain proper attention head count
    if hasattr(hvp, 'resolution_adaptive_blocks'):
        for i, block in enumerate(hvp.resolution_adaptive_blocks):
            if not hasattr(block, 'num_attention_heads'):
                results['hierarchical_vision_valid'] = False
                results['details'].append(f"ResolutionAdaptiveBlock {i} missing num_attention_heads")
            elif block.num_attention_heads != hvp.num_attention_heads:
                results['hierarchical_vision_valid'] = False
                results['details'].append(
                    f"ResolutionAdaptiveBlock {i} has {block.num_attention_heads} heads, "
                    f"expected {hvp.num_attention_heads}"
                )
    
    # Check that other components maintain proper dimensions
    if hasattr(hvp, 'multi_resolution_analyzer'):
        if hasattr(hvp.multi_resolution_analyzer, 'num_attention_heads'):
            if hvp.multi_resolution_analyzer.num_attention_heads != hvp.num_attention_heads:
                results['hierarchical_vision_valid'] = False
                results['details'].append(
                    f"MultiResolutionAnalyzer has {hvp.multi_resolution_analyzer.num_attention_heads} heads, "
                    f"expected {hvp.num_attention_heads}"
                )
    
    if hasattr(hvp, 'hierarchical_feature_extractor'):
        if hasattr(hvp.hierarchical_feature_extractor, 'num_attention_heads'):
            if hvp.hierarchical_feature_extractor.num_attention_heads != hvp.num_attention_heads:
                results['hierarchical_vision_valid'] = False
                results['details'].append(
                    f"HierarchicalFeatureExtractor has {hvp.hierarchical_feature_extractor.num_attention_heads} heads, "
                    f"expected {hvp.num_attention_heads}"
                )
    
    if results['hierarchical_vision_valid']:
        results['details'].append("Hierarchical vision processor capacity validation passed")
    
    return results


def create_capacity_preservation_hook(model: nn.Module):
    """
    Create a hook to monitor capacity preservation during model operations.
    
    Args:
        model: The model to monitor
    """
    def capacity_check_hook(module, input, output):
        # This hook would be used to validate capacity during forward passes
        # For now, it just serves as a placeholder for future implementation
        pass
    
    # Register the hook to important modules
    if hasattr(model, 'vision_tower') and hasattr(model.vision_tower, 'hierarchical_vision_processor'):
        hvp = model.vision_tower.hierarchical_vision_processor
        hvp.register_forward_hook(capacity_check_hook)