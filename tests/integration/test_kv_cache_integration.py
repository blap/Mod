"""
Test KV cache optimization integration
"""
from src.models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from src.models.config import Qwen3VLConfig
import torch

def test_kv_cache_integration():
    print("Testing KV cache optimization integration...")
    
    # Test that the model can be initialized with KV cache optimization
    config = Qwen3VLConfig()
    # Reduce dimensions for testing
    config.hidden_size = 128
    config.num_attention_heads = 4
    config.num_hidden_layers = 2
    config.vocab_size = 1000
    
    config.attention_implementation = 'kv_cache_optimized'
    config.kv_cache_strategy = 'hybrid'

    model = Qwen3VLForConditionalGeneration(config)
    print('Model initialized successfully with KV cache optimization')

    # Test that the model can process a simple input
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    output = model(input_ids=input_ids)
    print(f'Model forward pass successful, output shape: {output.shape}')
    
    # Test with caching enabled
    output_with_cache = model(
        input_ids=input_ids,
        use_cache=True
    )
    print(f'Model forward pass with cache successful, output shape: {output_with_cache.shape}')
    
    print("All integration tests passed!")

if __name__ == "__main__":
    test_kv_cache_integration()