"""
Compatibility verification for Qwen3-VL NAS system with existing model capacity:
- 32 transformer layers
- 32 attention heads
"""
import torch
import torch.nn as nn
from typing import List
from nas_system import Qwen3VLNeuralArchitectureSearch, LayerConfig, VisionLayerConfig, LanguageLayerConfig


def verify_model_capacity(model, expected_layers: int = 32, expected_attention_heads: int = 32):
    """
    Verify that the model maintains the expected capacity:
    - Number of transformer layers
    - Number of attention heads
    """
    results = {
        'num_layers': 0,
        'max_attention_heads': 0,
        'capacity_preserved': False,
        'details': {}
    }
    
    # Check language model layers
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
        results['num_layers'] = len(model.language_model.layers)
        results['details']['language_layers'] = len(model.language_model.layers)
    
    # Check vision model layers
    if hasattr(model, 'vision_tower') and hasattr(model.vision_tower, 'layers'):
        vision_layers = len(model.vision_tower.layers)
        results['details']['vision_layers'] = vision_layers
        # Total layers would be language + vision
        results['num_layers'] = results['details'].get('language_layers', 0) + vision_layers
    
    # Check attention heads in the first few layers as a sample
    max_heads = 0
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
        for i, layer in enumerate(model.language_model.layers):
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'num_heads'):
                max_heads = max(max_heads, layer.self_attn.num_heads)
            # Limit to first few layers to avoid excessive computation
            if i >= 5:
                break
    
    if hasattr(model, 'vision_tower') and hasattr(model.vision_tower, 'layers'):
        for i, layer in enumerate(model.vision_tower.layers):
            if hasattr(layer, 'attn') and hasattr(layer.attn, 'num_heads'):
                max_heads = max(max_heads, layer.attn.num_heads)
            # Limit to first few layers
            if i >= 5:
                break
    
    results['max_attention_heads'] = max_heads
    
    # Check if capacity is preserved
    results['capacity_preserved'] = (
        results['num_layers'] >= expected_layers and
        results['max_attention_heads'] >= expected_attention_heads
    )
    
    return results


def verify_nas_config_capacity(configs: List[LayerConfig], expected_layers: int = 32, expected_attention_heads: int = 32):
    """
    Verify that the NAS-generated configurations maintain the expected capacity.
    """
    results = {
        'num_configs': len(configs),
        'max_attention_heads': 0,
        'min_attention_heads': float('inf'),
        'max_hidden_size': 0,
        'min_hidden_size': float('inf'),
        'capacity_preserved': False,
        'details': []
    }
    
    for i, config in enumerate(configs):
        results['max_attention_heads'] = max(results['max_attention_heads'], config.num_attention_heads)
        results['min_attention_heads'] = min(results['min_attention_heads'], config.num_attention_heads)
        results['max_hidden_size'] = max(results['max_hidden_size'], config.hidden_size)
        results['min_hidden_size'] = min(results['min_hidden_size'], config.hidden_size)
        
        details = {
            'layer_idx': i,
            'hidden_size': config.hidden_size,
            'num_attention_heads': config.num_attention_heads,
            'intermediate_size': config.intermediate_size
        }
        results['details'].append(details)
    
    # Check if capacity is preserved
    results['capacity_preserved'] = (
        results['num_configs'] >= expected_layers and
        results['max_attention_heads'] >= expected_attention_heads
    )
    
    # Update min values if no configs were processed
    if results['min_attention_heads'] == float('inf'):
        results['min_attention_heads'] = 0
    if results['min_hidden_size'] == float('inf'):
        results['min_hidden_size'] = 0
    
    return results


def create_capacity_preserving_nas_system():
    """
    Create a NAS system that ensures capacity preservation.
    """
    # Initialize NAS system with parameters that ensure capacity preservation
    nas_system = Qwen3VLNeuralArchitectureSearch(
        num_layers=32,  # Explicitly set to 32 layers
        base_hidden_size=1024,  # Maintain reasonable hidden size
        base_num_heads=32,      # Maintain 32 attention heads as baseline
        base_intermediate_size=4096  # Maintain appropriate intermediate size
    )

    # Ensure the search space allows for configurations that maintain capacity
    # Since the search space is initialized in the NAS constructor, we need to update it
    nas_system.search_space.min_num_heads = 32  # Ensure minimum 32 heads
    nas_system.search_space.min_hidden_size = 1024  # Maintain minimum hidden size
    # Update max values to ensure 32 heads are possible
    nas_system.search_space.max_num_heads = max(32, nas_system.search_space.max_num_heads)

    return nas_system


def verify_capacity_preservation():
    """
    Comprehensive verification of capacity preservation in the NAS system.
    """
    print("Verifying capacity preservation in NAS system...")
    
    # Test 1: Verify NAS system initialization with proper capacity
    print("\n1. Testing NAS system initialization...")
    nas_system = create_capacity_preserving_nas_system()
    
    assert nas_system.num_layers == 32, f"Expected 32 layers, got {nas_system.num_layers}"
    assert nas_system.base_num_heads == 32, f"Expected 32 attention heads, got {nas_system.base_num_heads}"
    print("   OK - NAS system initialized with correct capacity parameters")
    
    # Test 2: Generate configurations and verify they maintain capacity
    print("\n2. Testing configuration generation...")
    
    # Create dummy input for testing
    dummy_text_input = torch.randint(0, 1000, (1, 16))
    
    # Generate text-optimized configuration
    text_configs = nas_system.search_optimal_architecture(
        input_data=dummy_text_input,
        input_type="text",
        num_search_steps=3,
        num_candidates_per_step=2
    )
    
    config_verification = verify_nas_config_capacity(
        text_configs, 
        expected_layers=32, 
        expected_attention_heads=32
    )
    
    print(f"   Generated {len(text_configs)} layer configurations")
    print(f"   Max attention heads: {config_verification['max_attention_heads']}")
    print(f"   Min attention heads: {config_verification['min_attention_heads']}")
    print(f"   Capacity preserved: {config_verification['capacity_preserved']}")
    
    assert config_verification['num_configs'] == 32, f"Expected 32 configs, got {config_verification['num_configs']}"
    assert config_verification['max_attention_heads'] >= 32, f"Max heads {config_verification['max_attention_heads']} < 32"
    assert config_verification['capacity_preserved'], "Capacity not preserved in generated configs"
    print("   OK - Configuration generation preserves capacity")
    
    # Test 3: Verify that configurations can be used to create valid models
    print("\n3. Testing model creation from configurations...")
    
    # Create a simple model structure to test configuration compatibility
    class TestTransformerLayer(nn.Module):
        def __init__(self, config: LayerConfig):
            super().__init__()
            self.hidden_size = config.hidden_size
            self.num_attention_heads = config.num_attention_heads
            self.intermediate_size = config.intermediate_size
            self.layer_idx = config.layer_idx
            
            # Verify that parameters are valid
            assert self.hidden_size % self.num_attention_heads == 0, \
                f"hidden_size {self.hidden_size} not divisible by num_heads {self.num_attention_heads}"
    
    # Test that all configurations can create valid layers
    for i, config in enumerate(text_configs):
        layer = TestTransformerLayer(config)
        assert layer.hidden_size % layer.num_attention_heads == 0, \
            f"Config {i}: hidden_size not divisible by num_heads"
    
    print("   OK - All configurations create valid transformer layers")
    
    # Test 4: Verify that configurations maintain the required capacity
    print("\n4. Testing capacity constraints...")
    
    # Check that configurations respect the minimum capacity requirements
    # For this test, we'll verify that the base configuration maintains capacity
    # but generated configs may have different values within the search space
    for i, config in enumerate(text_configs):
        # All configs should have at least the minimum required heads
        assert config.num_attention_heads >= 32, \
            f"Config {i} has {config.num_attention_heads} heads, expected >= 32"
        # The hidden size must be divisible by the number of heads
        assert config.hidden_size % config.num_attention_heads == 0, \
            f"Config {i}: hidden_size {config.hidden_size} not divisible by heads {config.num_attention_heads}"
    
    print("   OK - All configurations maintain minimum capacity requirements")
    
    # Test 5: Test with vision input as well
    print("\n5. Testing vision configuration generation...")
    
    dummy_vision_input = torch.randn(1, 3, 224, 224)
    
    vision_configs = nas_system.search_optimal_architecture(
        input_data=dummy_vision_input,
        input_type="vision",
        num_search_steps=3,
        num_candidates_per_step=2
    )
    
    vision_verification = verify_nas_config_capacity(
        vision_configs, 
        expected_layers=32, 
        expected_attention_heads=32
    )
    
    print(f"   Vision configs: {len(vision_configs)}, max heads: {vision_verification['max_attention_heads']}")
    
    assert vision_verification['num_configs'] == 32, f"Expected 32 vision configs, got {vision_verification['num_configs']}"
    assert vision_verification['capacity_preserved'], "Capacity not preserved in vision configs"
    print("   OK - Vision configuration generation preserves capacity")
    
    print("\nOK - All capacity preservation tests passed!")

    return {
        'nas_system': nas_system,
        'text_configs': text_configs,
        'vision_configs': vision_configs,
        'text_verification': config_verification,
        'vision_verification': vision_verification
    }


def run_compatibility_tests():
    """
    Run all compatibility tests to ensure the NAS system maintains model capacity.
    """
    print("Running compatibility tests for Qwen3-VL NAS system...")
    print("=" * 60)

    results = verify_capacity_preservation()

    print("\n" + "=" * 60)
    print("COMPATIBILITY TEST SUMMARY")
    print("=" * 60)
    print(f"OK - NAS system maintains 32 transformer layers")
    print(f"OK - NAS system maintains 32 attention heads minimum")
    print(f"OK - Configuration generation preserves capacity")
    print(f"OK - Vision and text configurations maintain capacity")
    print(f"OK - All configurations create valid transformer layers")

    return results


if __name__ == "__main__":
    test_results = run_compatibility_tests()
    print("\nAll compatibility tests completed successfully!")