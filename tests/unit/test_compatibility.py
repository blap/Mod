"""
Test for CPU and GPU compatibility of Phase 2 efficiency improvements
"""
import pytest
import torch
from src.models.config import Qwen3VLConfig
from src.models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from src.models.linear_attention import PerformerAttention
from src.models.device_aware_module import DeviceAwareAttention
from src.models.gradient_checkpointing import MemoryEfficientAttention
from src.models.adaptive_computation import AdaptiveAttention
from src.models.memory_management import OptimizedQwen3VLAttention


def test_cpu_compatibility():
    """Test that the model works correctly on CPU."""
    config = Qwen3VLConfig()
    config.attention_implementation = "performer"  # Use Performer attention for testing
    
    # Create model
    model = Qwen3VLForConditionalGeneration(config)
    model = model.to('cpu')
    
    # Test with text input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    outputs = model(input_ids=input_ids)
    
    # Check output shape
    assert outputs.shape[0] == batch_size
    assert outputs.shape[1] == seq_len
    assert outputs.shape[2] == config.hidden_size
    
    # Test with gradient checkpointing enabled
    config.use_gradient_checkpointing = True
    model_with_gc = Qwen3VLForConditionalGeneration(config)
    model_with_gc = model_with_gc.to('cpu')
    
    outputs_gc = model_with_gc(input_ids=input_ids)
    assert outputs_gc.shape[0] == batch_size
    assert outputs_gc.shape[1] == seq_len
    assert outputs_gc.shape[2] == config.hidden_size


def test_different_attention_implementations_cpu():
    """Test different attention implementations on CPU."""
    config = Qwen3VLConfig()
    batch_size, seq_len = 2, 10
    hidden_size = config.hidden_size
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Test Performer attention
    config.attention_implementation = "performer"
    performer_attn = PerformerAttention(config, layer_idx=0)
    performer_attn = performer_attn.to('cpu')
    output_performer, _, _ = performer_attn(hidden_states.to('cpu'))
    assert output_performer.shape == (batch_size, seq_len, hidden_size)
    
    # Test Device-aware attention
    config.attention_implementation = "device_aware"
    device_attn = DeviceAwareAttention(config, layer_idx=0)
    device_attn = device_attn.to('cpu')
    output_device, _, _ = device_attn(hidden_states.to('cpu'))
    assert output_device.shape == (batch_size, seq_len, hidden_size)
    
    # Test Adaptive attention
    config.attention_implementation = "adaptive"
    adaptive_attn = AdaptiveAttention(config, layer_idx=0)
    adaptive_attn = adaptive_attn.to('cpu')
    output_adaptive, _, _ = adaptive_attn(hidden_states.to('cpu'))
    assert output_adaptive.shape == (batch_size, seq_len, hidden_size)
    
    # Test Memory-efficient attention
    config.attention_implementation = "memory_efficient"
    memory_attn = MemoryEfficientAttention(config, layer_idx=0)
    memory_attn = memory_attn.to('cpu')
    output_memory, _, _ = memory_attn(hidden_states.to('cpu'))
    assert output_memory.shape == (batch_size, seq_len, hidden_size)
    
    # Test Optimized attention
    optimized_attn = OptimizedQwen3VLAttention(config, layer_idx=0)
    optimized_attn = optimized_attn.to('cpu')
    output_optimized, _, _ = optimized_attn(hidden_states.to('cpu'))
    assert output_optimized.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_compatibility():
    """Test that the model works correctly on GPU."""
    device = torch.device('cuda')
    
    config = Qwen3VLConfig()
    config.attention_implementation = "performer"  # Use Performer attention for testing
    
    # Create model
    model = Qwen3VLForConditionalGeneration(config)
    model = model.to(device)
    
    # Test with text input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    
    # Forward pass
    outputs = model(input_ids=input_ids)
    
    # Check output shape
    assert outputs.shape[0] == batch_size
    assert outputs.shape[1] == seq_len
    assert outputs.shape[2] == config.hidden_size
    
    # Test with gradient checkpointing enabled
    config.use_gradient_checkpointing = True
    model_with_gc = Qwen3VLForConditionalGeneration(config)
    model_with_gc = model_with_gc.to(device)
    
    outputs_gc = model_with_gc(input_ids=input_ids)
    assert outputs_gc.shape[0] == batch_size
    assert outputs_gc.shape[1] == seq_len
    assert outputs_gc.shape[2] == config.hidden_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_different_attention_implementations_gpu():
    """Test different attention implementations on GPU."""
    device = torch.device('cuda')
    config = Qwen3VLConfig()
    batch_size, seq_len = 2, 10
    hidden_size = config.hidden_size
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(device)
    
    # Test Performer attention
    config.attention_implementation = "performer"
    performer_attn = PerformerAttention(config, layer_idx=0)
    performer_attn = performer_attn.to(device)
    output_performer, _, _ = performer_attn(hidden_states)
    assert output_performer.shape == (batch_size, seq_len, hidden_size)
    
    # Test Device-aware attention
    config.attention_implementation = "device_aware"
    device_attn = DeviceAwareAttention(config, layer_idx=0)
    device_attn = device_attn.to(device)
    output_device, _, _ = device_attn(hidden_states)
    assert output_device.shape == (batch_size, seq_len, hidden_size)
    
    # Test Adaptive attention
    config.attention_implementation = "adaptive"
    adaptive_attn = AdaptiveAttention(config, layer_idx=0)
    adaptive_attn = adaptive_attn.to(device)
    output_adaptive, _, _ = adaptive_attn(hidden_states)
    assert output_adaptive.shape == (batch_size, seq_len, hidden_size)
    
    # Test Memory-efficient attention
    config.attention_implementation = "memory_efficient"
    memory_attn = MemoryEfficientAttention(config, layer_idx=0)
    memory_attn = memory_attn.to(device)
    output_memory, _, _ = memory_attn(hidden_states)
    assert output_memory.shape == (batch_size, seq_len, hidden_size)
    
    # Test Optimized attention
    optimized_attn = OptimizedQwen3VLAttention(config, layer_idx=0)
    optimized_attn = optimized_attn.to(device)
    output_optimized, _, _ = optimized_attn(hidden_states)
    assert output_optimized.shape == (batch_size, seq_len, hidden_size)


def test_device_agnostic_config():
    """Test that the model works with device-agnostic configuration."""
    config = Qwen3VLConfig()
    
    # Test all attention implementations
    implementations = ["eager", "performer", "device_aware", "adaptive", "memory_efficient"]
    
    for impl in implementations:
        config.attention_implementation = impl
        
        # Create model
        model = Qwen3VLForConditionalGeneration(config)
        
        # Test on CPU
        model = model.to('cpu')
        batch_size, seq_len = 1, 5
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        outputs = model(input_ids=input_ids)
        
        assert outputs.shape[0] == batch_size
        assert outputs.shape[1] == seq_len
        assert outputs.shape[2] == config.hidden_size


def run_compatibility_tests():
    """Run all compatibility tests."""
    print("Running CPU compatibility tests...")
    test_cpu_compatibility()
    test_different_attention_implementations_cpu()
    test_device_agnostic_config()
    print("✓ CPU compatibility tests passed")
    
    if torch.cuda.is_available():
        print("Running GPU compatibility tests...")
        test_gpu_compatibility()
        test_different_attention_implementations_gpu()
        print("✓ GPU compatibility tests passed")
    else:
        print("⚠ GPU not available, skipping GPU tests")
    
    print("All compatibility tests completed successfully!")


if __name__ == "__main__":
    run_compatibility_tests()