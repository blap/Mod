"""
Simple integration test for the hardware abstraction layer
"""
from src.components.hardware.hardware_abstraction import DeviceAwareAttention
from transformers import PretrainedConfig
import torch


class MockConfig(PretrainedConfig):
    pass


def test_integration():
    config = MockConfig()
    config.hidden_size = 512
    config.num_attention_heads = 8
    config.num_key_value_heads = 8
    config.max_position_embeddings = 2048
    config.rope_theta = 10000.0
    
    # This should work without errors
    attention = DeviceAwareAttention(config)
    print("DeviceAwareAttention initialized successfully")
    
    # Test a simple forward pass
    batch_size = 2
    seq_len = 16
    hidden_size = config.hidden_size
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    output, attn_weights, past_key_value = attention(
        hidden_states=hidden_states
    )
    
    print(f"Forward pass successful: output shape {output.shape}")
    print("Integration test passed!")


if __name__ == "__main__":
    test_integration()