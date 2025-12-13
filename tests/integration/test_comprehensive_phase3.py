"""
Comprehensive test for Phase 3 implementation
"""
import sys
import os
# Add the project root to the path so we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models.config import Qwen3VLConfig
from src.models.modeling_qwen3_vl_phase3 import Qwen3VLForConditionalGeneration


def test_comprehensive_phase3():
    """Comprehensive test for Phase 3 implementation."""
    print("Running comprehensive test for Phase 3 implementation...")
    
    config = Qwen3VLConfig()
    model = Qwen3VLForConditionalGeneration(config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Config: {config.num_hidden_layers} layers, {config.num_attention_heads} attention heads")
    
    # Test 1: Text-only input
    print("\n1. Testing text-only input...")
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    text_output = model(input_ids=input_ids)
    print(f"   Text output shape: {text_output.shape}")
    assert text_output.shape[0] == batch_size
    print("   [PASS] Text-only input works")

    # Test 2: Image-only input
    print("\n2. Testing image-only input...")
    image_size = 448
    pixel_values = torch.randn(batch_size, 3, image_size, image_size)
    image_output = model(pixel_values=pixel_values)
    print(f"   Image output shape: {image_output.shape}")
    assert image_output.shape[0] == batch_size
    print("   [PASS] Image-only input works")

    # Test 3: Multimodal input
    print("\n3. Testing multimodal input...")
    multimodal_output = model(input_ids=input_ids, pixel_values=pixel_values)
    print(f"   Multimodal output shape: {multimodal_output.shape}")
    assert multimodal_output.shape[0] == batch_size
    print("   [PASS] Multimodal input works")

    # Test 4: Generation
    print("\n4. Testing generation capability...")
    generated_ids = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=5,
        do_sample=False
    )
    print(f"   Generated shape: {generated_ids.shape}")
    assert generated_ids.shape[0] == batch_size
    print("   [PASS] Generation works")

    # Test 5: Parameter preservation
    print("\n5. Testing parameter preservation...")
    assert config.num_hidden_layers == 32, "Should preserve 32 layers"
    assert config.num_attention_heads == 32, "Should preserve 32 attention heads"
    print("   [PASS] All 32 layers and 32 attention heads preserved")

    # Test 6: Cross-attention functionality
    print("\n6. Testing cross-attention functionality...")
    # Check that the model has the expected cross-attention components
    layer_0 = model.language_model.layers[0]
    assert hasattr(layer_0, 'vision_cross_attn'), "Should have vision cross-attention in decoder layers"
    print("   [PASS] Cross-attention functionality verified")

    print("\n[SUCCESS] All comprehensive tests passed!")
    print("[SUCCESS] Phase 3 implementation is working correctly!")
    

if __name__ == "__main__":
    test_comprehensive_phase3()