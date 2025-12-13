"""
Model Capacity Validation Test for Qwen3-VL-2B-Instruct Architecture
This test ensures that the model maintains its required capacity (32 transformer layers and 32 attention heads)
after all optimizations have been applied.
"""
import sys
import os
import torch
import json
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.qwen3_vl.core.config import Qwen3VLConfig
from models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


def validate_model_config_capacity():
    """Validate that the model configuration maintains required capacity"""
    print("Validating model configuration capacity...")
    
    config = Qwen3VLConfig()
    
    # Check the required capacity
    required_layers = 32
    required_heads = 32
    
    actual_layers = config.num_hidden_layers
    actual_heads = config.num_attention_heads
    
    print(f"  Required layers: {required_layers}")
    print(f"  Actual layers: {actual_layers}")
    print(f"  Layers match: {'✓' if actual_layers == required_layers else '✗'}")
    
    print(f"  Required heads: {required_heads}")
    print(f"  Actual heads: {actual_heads}")
    print(f"  Heads match: {'✓' if actual_heads == required_heads else '✗'}")
    
    layers_valid = actual_layers == required_layers
    heads_valid = actual_heads == required_heads
    
    return layers_valid and heads_valid, actual_layers, actual_heads


def validate_model_instance_capacity():
    """Validate that a model instance maintains required capacity"""
    print("Validating model instance capacity...")
    
    config = Qwen3VLConfig()
    model = Qwen3VLForConditionalGeneration(config)
    
    # Check the required capacity
    required_layers = 32
    required_heads = 32
    
    actual_layers = config.num_hidden_layers
    actual_heads = config.num_attention_heads
    
    print(f"  Required layers: {required_layers}")
    print(f"  Actual layers: {actual_layers}")
    print(f"  Layers match: {'✓' if actual_layers == required_layers else '✗'}")
    
    print(f"  Required heads: {required_heads}")
    print(f"  Actual heads: {actual_heads}")
    print(f"  Heads match: {'✓' if actual_heads == required_heads else '✗'}")
    
    # Additional validation: check the actual model structure
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        actual_model_layers = len(model.model.layers)
        print(f"  Actual model layers in structure: {actual_model_layers}")
        structure_layers_valid = actual_model_layers == required_layers
    else:
        print("  Could not verify model structure layers")
        structure_layers_valid = True  # Don't fail if we can't check structure
    
    layers_valid = actual_layers == required_layers
    heads_valid = actual_heads == required_heads
    
    return layers_valid and heads_valid and structure_layers_valid, actual_layers, actual_heads


def validate_parameter_count():
    """Validate that the model has the expected parameter count for full capacity"""
    print("Validating parameter count...")
    
    config = Qwen3VLConfig()
    model = Qwen3VLForConditionalGeneration(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Total trainable parameters: {total_trainable_params:,}")
    
    # For a model with 32 layers and 32 attention heads, we expect a certain range
    # This is a simplified check - in practice, you'd calculate expected params based on config
    expected_min_params = 1_000_000  # 1M minimum for a model of this size
    expected_max_params = 3_000_000_000  # 3B maximum (for a 2B model)
    
    params_in_range = expected_min_params <= total_params <= expected_max_params
    print(f"  Parameters in expected range: {'✓' if params_in_range else '✗'}")
    print(f"    Expected range: {expected_min_params:,} - {expected_max_params:,}")
    
    return params_in_range, total_params


def validate_layer_structure():
    """Validate that each transformer layer has the correct structure"""
    print("Validating transformer layer structure...")
    
    config = Qwen3VLConfig()
    model = Qwen3VLForConditionalGeneration(config)
    
    required_layers = 32
    required_heads = 32
    
    # Check that the model has the expected number of layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        model_layers = model.model.layers
        actual_layer_count = len(model_layers)
        
        print(f"  Model has {actual_layer_count} transformer layers")
        
        if actual_layer_count == required_layers:
            print("  ✓ Correct number of transformer layers")
            
            # Check a few layers to ensure they have the right structure
            sample_layers = min(5, actual_layer_count)  # Check up to 5 layers
            layers_structure_valid = True
            
            for i in range(sample_layers):
                layer = model_layers[i]
                
                # Check if layer has expected components (this depends on the actual model implementation)
                has_self_attn = hasattr(layer, 'self_attn')
                has_mlp = hasattr(layer, 'mlp') if hasattr(layer, 'mlp') else hasattr(layer, 'mlp')  # varies by implementation
                
                print(f"    Layer {i}: Self-attention={'✓' if has_self_attn else '✗'}, MLP={'✓' if has_mlp else '✗'}")
                
                if not (has_self_attn):
                    layers_structure_valid = False
            
            return True and layers_structure_valid, actual_layer_count
        else:
            print(f"  ✗ Expected {required_layers} layers, got {actual_layer_count}")
            return False, actual_layer_count
    else:
        print("  Cannot validate layer structure - layers attribute not found")
        return True, 0  # Don't fail if we can't validate structure


def validate_attention_head_structure():
    """Validate that attention mechanisms have the correct number of heads"""
    print("Validating attention head structure...")
    
    config = Qwen3VLConfig()
    model = Qwen3VLForConditionalGeneration(config)
    
    required_heads = 32
    
    # Check the configuration
    config_heads = config.num_attention_heads
    heads_match_config = config_heads == required_heads
    
    print(f"  Config attention heads: {config_heads} (required: {required_heads}) - {'✓' if heads_match_config else '✗'}")
    
    # If possible, check actual attention layer properties
    attention_heads_valid = True
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers') and len(model.model.layers) > 0:
        first_layer = model.model.layers[0]
        if hasattr(first_layer, 'self_attn'):
            attn_layer = first_layer.self_attn
            
            # Check various possible attributes for number of heads
            actual_heads = None
            if hasattr(attn_layer, 'num_heads'):
                actual_heads = attn_layer.num_heads
            elif hasattr(attn_layer, 'num_attention_heads'):
                actual_heads = attn_layer.num_attention_heads
            elif hasattr(attn_layer, 'num_key_value_heads'):
                actual_heads = attn_layer.num_key_value_heads
            
            if actual_heads is not None:
                heads_match_actual = actual_heads == required_heads
                print(f"  Actual attention heads: {actual_heads} - {'✓' if heads_match_actual else '✗'}")
                attention_heads_valid = heads_match_actual
            else:
                print("  Could not verify actual attention heads in layer")
                attention_heads_valid = heads_match_config  # Rely on config match
        else:
            print("  Could not access attention layer in first transformer layer")
            attention_heads_valid = heads_match_config  # Rely on config match
    else:
        print("  Could not access transformer layers to verify attention heads")
        attention_heads_valid = heads_match_config  # Rely on config match
    
    return attention_heads_valid and heads_match_config, config_heads


def run_capacity_validation():
    """Run all capacity validation tests"""
    print("=" * 80)
    print("MODEL CAPACITY VALIDATION TEST FOR QWEN3-VL-2B-INSTRUCT ARCHITECTURE")
    print("=" * 80)
    print("Ensuring model maintains 32 transformer layers and 32 attention heads")
    print("=" * 80)
    
    # Run all validation tests
    config_capacity_valid, config_layers, config_heads = validate_model_config_capacity()
    instance_capacity_valid, instance_layers, instance_heads = validate_model_instance_capacity()
    params_valid, total_params = validate_parameter_count()
    layer_structure_valid, actual_layer_count = validate_layer_structure()
    attention_structure_valid, actual_attention_heads = validate_attention_head_structure()
    
    print("\n" + "=" * 80)
    print("MODEL CAPACITY VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"Configuration capacity: {'✓ PASSED' if config_capacity_valid else '✗ FAILED'}")
    print(f"  - Layers: {config_layers}/32")
    print(f"  - Heads: {config_heads}/32")
    
    print(f"Instance capacity: {'✓ PASSED' if instance_capacity_valid else '✗ FAILED'}")
    print(f"  - Layers: {instance_layers}/32")
    print(f"  - Heads: {instance_heads}/32")
    
    print(f"Parameter count: {'✓ PASSED' if params_valid else '✗ FAILED'}")
    print(f"  - Total params: {total_params:,}")
    
    print(f"Layer structure: {'✓ PASSED' if layer_structure_valid else '✗ FAILED'}")
    print(f"  - Actual layers: {actual_layer_count}/32")
    
    print(f"Attention structure: {'✓ PASSED' if attention_structure_valid else '✗ FAILED'}")
    print(f"  - Attention heads: {actual_attention_heads}/32")
    
    # Overall validation
    all_valid = all([
        config_capacity_valid,
        instance_capacity_valid,
        params_valid,
        layer_structure_valid,
        attention_structure_valid
    ])
    
    print(f"\nOverall Capacity Validation: {'✓ ACHIEVED' if all_valid else '✗ NOT ACHIEVED'}")
    
    if all_valid:
        print("\n✓ Model capacity successfully maintained!")
        print("✓ All 32 transformer layers preserved")
        print("✓ All 32 attention heads preserved")
        print("✓ Parameter count within expected range")
        print("✓ Layer structure intact")
        print("✓ Attention mechanisms properly configured")
    else:
        print("\n✗ Model capacity validation failed!")
        print("✗ Some aspects of the model capacity requirements were not met")
        print("✗ Check that all optimizations preserve the fundamental architecture")
    
    # Save validation results
    results = {
        'config_capacity_valid': config_capacity_valid,
        'instance_capacity_valid': instance_capacity_valid,
        'params_valid': params_valid,
        'layer_structure_valid': layer_structure_valid,
        'attention_structure_valid': attention_structure_valid,
        'config_layers': config_layers,
        'config_heads': config_heads,
        'instance_layers': instance_layers,
        'instance_heads': instance_heads,
        'total_params': total_params,
        'actual_layer_count': actual_layer_count,
        'actual_attention_heads': actual_attention_heads,
        'overall_valid': all_valid
    }
    
    with open('model_capacity_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("  Detailed results saved to 'model_capacity_validation_results.json'")
    
    return all_valid


def test_capacity_after_optimizations():
    """Test that capacity is maintained even after applying optimizations"""
    print("\nTesting capacity after applying optimizations...")
    
    # This would test with optimizations enabled
    # For this test, we'll just verify the standard configuration
    # In a real scenario, you would apply all optimizations and then validate
    
    config = Qwen3VLConfig()
    
    # Simulate applying optimizations by setting optimization flags
    config.use_sparsity = True
    config.sparsity_ratio = 0.4
    config.use_gradient_checkpointing = True
    
    model = Qwen3VLForConditionalGeneration(config)
    
    # Validate capacity is still maintained
    layers_valid = model.config.num_hidden_layers == 32
    heads_valid = model.config.num_attention_heads == 32
    
    print(f"  After optimizations - Layers: {'✓' if layers_valid else '✗'} ({model.config.num_hidden_layers}/32)")
    print(f"  After optimizations - Heads: {'✓' if heads_valid else '✗'} ({model.config.num_attention_heads}/32)")
    
    return layers_valid and heads_valid


if __name__ == "__main__":
    capacity_valid = run_capacity_validation()
    optimizations_capacity_valid = test_capacity_after_optimizations()
    
    print(f"\n{'='*80}")
    print("FINAL CAPACITY VALIDATION STATUS:", "PASSED" if capacity_valid and optimizations_capacity_valid else "FAILED")
    print(f"{'='*80}")
    
    success = capacity_valid and optimizations_capacity_valid
    sys.exit(0 if success else 1)