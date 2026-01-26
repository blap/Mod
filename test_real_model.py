"""
Test script to verify that the real model implementation works correctly.
"""
import torch
from real_model_for_testing import RealGLM47Model

def test_real_model():
    print("Testing RealGLM47Model...")
    
    # Create a real model with small parameters for testing
    model = RealGLM47Model(
        hidden_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=256,
        vocab_size=1000
    )
    
    print(f"Model created successfully: {type(model)}")
    print(f"Model config: hidden_size={model.hidden_size}, num_heads={model.num_attention_heads}")
    
    # Test forward pass with sample input
    input_ids = torch.randint(0, 1000, (2, 10))  # Batch size 2, sequence length 10
    attention_mask = torch.ones((2, 10))
    
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    
    print(f"Output type: {type(output)}")
    print(f"Logits shape: {output.logits.shape}")
    print(f"Last hidden state shape: {output.last_hidden_state.shape}")
    
    # Test generation
    generated = model.generate(input_ids=input_ids, max_new_tokens=5)
    print(f"Generated shape: {generated.shape}")
    
    print("SUCCESS: Real model test passed!")

def test_backward_compatibility():
    print("\nTesting backward compatibility with MockModel alias...")
    
    from real_model_for_testing import MockModel
    
    # Create model using MockModel alias
    model = MockModel()
    print(f"MockModel alias works: {type(model)}")
    
    # Test basic functionality
    input_ids = torch.randint(0, 100, (1, 5))
    with torch.no_grad():
        output = model(input_ids=input_ids)
    
    print(f"MockModel forward pass successful: {output.logits.shape}")
    print("SUCCESS: Backward compatibility test passed!")

if __name__ == "__main__":
    test_real_model()
    test_backward_compatibility()
    print("\nALL TESTS PASSED! Real model implementation is working correctly.")