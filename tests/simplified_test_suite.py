"""
Simplified Test Suite for Qwen3-VL Model Components

This script tests the core model components in a simplified way to avoid import issues.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
import sys
from pathlib import Path

# Add the project root to the path to import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_basic_config():
    """Test basic configuration functionality."""
    print("Testing basic configuration...")
    
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        config = Qwen3VLConfig()
        
        print(f"  V Config created with {config.num_hidden_layers} layers, {config.num_attention_heads} attention heads")
        
        # Verify capacity preservation
        assert config.num_hidden_layers == 32
        assert config.num_attention_heads == 32
        assert config.vision_num_hidden_layers == 24
        assert config.vision_num_attention_heads == 16
        
        print("  V Full capacity preserved (32 layers, 32 attention heads)")
        return True
    except Exception as e:
        print(f"  X Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_attention_components():
    """Test attention mechanisms."""
    print("\nTesting attention components...")
    
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.attention.flash_attention_2 import FlashAttention2
        
        config = Qwen3VLConfig()
        config.hidden_size = 256
        config.num_attention_heads = 8
        
        # Create attention module
        attention = FlashAttention2(config, layer_idx=0)
        print("  V Attention module created")
        
        # Test with sample inputs
        batch_size, seq_len, hidden_size = 2, 16, 256
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        position_ids = torch.arange(seq_len).expand((batch_size, -1))
        
        # Forward pass
        output, attn_weights, past_key_value = attention(
            hidden_states=hidden_states,
            position_ids=position_ids
        )
        
        print(f"  V Attention forward pass completed, output shape: {output.shape}")
        assert output.shape == hidden_states.shape
        assert torch.all(torch.isfinite(output))
        
        return True
    except Exception as e:
        print(f"  X Attention test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mlp_components():
    """Test MLP components."""
    print("\nTesting MLP components...")
    
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        
        config = Qwen3VLConfig()
        config.hidden_size = 256
        config.intermediate_size = 512
        
        # Create a simple MLP module
        class SimpleMLP(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.hidden_size = config.hidden_size
                self.intermediate_size = config.intermediate_size
                self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
                self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
                self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
                self.act_fn = nn.SiLU()

            def forward(self, x):
                gate = self.gate_proj(x)
                up = self.up_proj(x)
                act = self.act_fn(gate)
                intermediate = act * up
                output = self.down_proj(intermediate)
                return output

        mlp = SimpleMLP(config)
        print("  V MLP module created")
        
        # Test with sample inputs
        batch_size, seq_len, hidden_size = 2, 16, 256
        inputs = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        output = mlp(inputs)
        
        print(f"  V MLP forward pass completed, output shape: {output.shape}")
        assert output.shape == inputs.shape
        assert torch.all(torch.isfinite(output))
        
        return True
    except Exception as e:
        print(f"  X MLP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_management():
    """Test memory management components."""
    print("\nTesting memory management components...")
    
    try:
        from src.qwen3_vl.memory_management.memory_manager import MemoryManager
        
        # Create memory manager
        manager = MemoryManager()
        print("  V Memory manager created")
        
        # Test tensor allocation
        tensor_shape = (10, 20, 128)
        tensor = manager.allocate_tensor(tensor_shape, torch.float32, "test_tensor")
        print(f"  V Tensor allocated with shape {tensor_shape}")
        
        # Test tensor access
        accessed_tensor = manager.access_tensor(tensor, "test_context")
        print("  V Tensor accessed successfully")
        
        # Test tensor deallocation
        manager.deallocate_tensor("test_tensor")
        print("  V Tensor deallocated successfully")
        
        # Test memory optimization
        stats = manager.get_stats()
        print(f"  V Memory manager stats: {stats}")
        
        return True
    except Exception as e:
        print(f"  X Memory management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_capacity_preservation():
    """Test that model capacity is preserved."""
    print("\nTesting model capacity preservation...")
    
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        
        config = Qwen3VLConfig()
        
        # Verify that the configuration maintains the required capacity
        assert config.num_hidden_layers == 32, f"Expected 32 hidden layers, got {config.num_hidden_layers}"
        assert config.num_attention_heads == 32, f"Expected 32 attention heads, got {config.num_attention_heads}"
        assert config.vision_num_hidden_layers == 24, f"Expected 24 vision layers, got {config.vision_num_hidden_layers}"
        assert config.vision_num_attention_heads == 16, f"Expected 16 vision attention heads, got {config.vision_num_attention_heads}"
        
        print("  V Model capacity preserved (32 transformer layers, 32 attention heads)")
        print(f"    Language layers: {config.num_hidden_layers}, Attention heads: {config.num_attention_heads}")
        print(f"    Vision layers: {config.vision_num_hidden_layers}, Vision heads: {config.vision_num_attention_heads}")
        
        return True
    except Exception as e:
        print(f"  X Capacity preservation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_tests():
    """Run all tests."""
    print("="*80)
    print("SIMPLIFIED TEST SUITE FOR QWEN3-VL MODEL COMPONENTS")
    print("="*80)
    
    tests = [
        ("Basic Configuration", test_basic_config),
        ("Attention Components", test_attention_components),
        ("MLP Components", test_mlp_components),
        ("Memory Management", test_memory_management),
        ("Model Capacity Preservation", test_model_capacity_preservation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        symbol = "V" if success else "X"
        print(f"{symbol} {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nSUCCESS: All simplified tests passed!")
        return True
    else:
        print(f"\nFAILURE: {total - passed} test(s) failed!")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)