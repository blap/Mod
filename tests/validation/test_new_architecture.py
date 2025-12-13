"""
Test module to verify the new architecture with dependency injection and proper separation of concerns.
"""

import torch
from src.qwen3_vl import Qwen3VLModel, setup_qwen3_vl_system, get_system_components
from src.qwen3_vl.config.factory import ConfigFactory


def test_new_architecture():
    """Test the new architecture with dependency injection."""
    print("Testing new architecture with dependency injection...")
    
    # Create configuration optimized for hardware
    config = ConfigFactory.create_optimized_config_for_hardware("intel_i5_10210u")
    
    # Set up the system
    container = setup_qwen3_vl_system(config=config)
    
    # Get all components
    components = get_system_components()
    
    print("\nSystem components:")
    for name, component in components.items():
        print(f"  - {name}: {type(component).__name__}")
    
    # Create model
    model = Qwen3VLModel(config)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test model forward pass
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    print(f"\nInput shape: {input_ids.shape}")
    
    with torch.no_grad():
        outputs = model(input_ids)
        print(f"Output shape: {outputs.shape}")
    
    # Test generation
    with torch.no_grad():
        generated = model.generate(input_ids, max_length=15, do_sample=False)
        print(f"Generated shape: {generated.shape}")
    
    print("\nNew architecture test completed successfully!")


def test_component_injection():
    """Test that components are properly injected."""
    print("\nTesting component injection...")
    
    # Create configuration
    config = ConfigFactory.create_default_config()
    
    # Set up the system
    container = setup_qwen3_vl_system(config=config)
    
    # Try to resolve each component
    memory_manager = container.resolve('memory_manager')
    optimizer = container.resolve('optimizer')
    preprocessor = container.resolve('preprocessor')
    pipeline = container.resolve('pipeline')
    attention = container.resolve('attention')
    mlp = container.resolve('mlp')
    layer = container.resolve('layer')
    
    print(f"Memory Manager: {type(memory_manager).__name__}")
    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"Preprocessor: {type(preprocessor).__name__}")
    print(f"Pipeline: {type(pipeline).__name__}")
    print(f"Attention: {type(attention).__name__}")
    print(f"MLP: {type(mlp).__name__}")
    print(f"Layer: {type(layer).__name__}")
    
    print("Component injection test completed successfully!")


if __name__ == "__main__":
    test_new_architecture()
    test_component_injection()
    print("\nAll tests passed! The new architecture is working correctly.")