"""
Minimal test script to verify that cross-alignment optimization implementations work correctly.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_glm_cross_alignment():
    """Test GLM-4.7-Flash cross-alignment optimization."""
    print("Testing GLM-4.7-Flash cross-alignment optimization...")
    
    try:
        from src.models.specialized.glm_4_7_flash.cross_alignment_optimization import (
            CrossAlignmentConfig, 
            GLM47FlashCrossAlignmentOptimizer,
            apply_cross_alignment_to_model
        )
        
        # Create a dummy model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2048, 2048)
            
            def forward(self, x):
                return self.linear(x)
        
        dummy_model = DummyModel()
        
        # Create config
        config = CrossAlignmentConfig()
        config.hidden_size = 2048
        
        # Apply cross-alignment optimization
        optimized_model = apply_cross_alignment_to_model(dummy_model, config)
        
        # Test that the model has the cross-alignment attributes
        assert hasattr(optimized_model, 'cross_alignment_manager'), "Model should have cross_alignment_manager"
        assert hasattr(optimized_model, 'perform_cross_alignment'), "Model should have perform_cross_alignment method"
        
        print("PASS: GLM-4.7-Flash cross-alignment optimization test passed!")
        return True

    except Exception as e:
        print(f"FAIL: GLM-4.7-Flash cross-alignment optimization test failed: {e}")
        return False


def test_qwen3_4b_cross_alignment():
    """Test Qwen3-4B-Instruct-2507 cross-alignment optimization."""
    print("Testing Qwen3-4B-Instruct-2507 cross-alignment optimization...")
    
    try:
        from src.models.language.qwen3_4b_instruct_2507.cross_alignment_optimization import (
            CrossAlignmentConfig, 
            Qwen3InstructCrossAlignmentOptimizer,
            apply_cross_alignment_to_model
        )
        
        # Create a dummy model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2560, 2560)  # Qwen3-4B specific size
            
            def forward(self, x):
                return self.linear(x)
        
        dummy_model = DummyModel()
        
        # Create config
        config = CrossAlignmentConfig()
        config.hidden_size = 2560
        
        # Apply cross-alignment optimization
        optimized_model = apply_cross_alignment_to_model(dummy_model, config)
        
        # Test that the model has the cross-alignment attributes
        assert hasattr(optimized_model, 'cross_alignment_manager'), "Model should have cross_alignment_manager"
        assert hasattr(optimized_model, 'perform_cross_alignment'), "Model should have perform_cross_alignment method"
        
        print("PASS: Qwen3-4B-Instruct-2507 cross-alignment optimization test passed!")
        return True

    except Exception as e:
        print(f"FAIL: Qwen3-4B-Instruct-2507 cross-alignment optimization test failed: {e}")
        return False


def test_qwen3_coder_30b_cross_alignment():
    """Test Qwen3-Coder-30B cross-alignment optimization."""
    print("Testing Qwen3-Coder-30B cross-alignment optimization...")
    
    try:
        from src.models.coding.qwen3_coder_30b.cross_alignment_optimization import (
            CrossAlignmentConfig, 
            Qwen3CoderCrossAlignmentOptimizer,
            apply_cross_alignment_to_model
        )
        
        # Create a dummy model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4096, 4096)  # Qwen3-Coder-30B specific size
            
            def forward(self, x):
                return self.linear(x)
        
        dummy_model = DummyModel()
        
        # Create config
        config = CrossAlignmentConfig()
        config.hidden_size = 4096
        
        # Apply cross-alignment optimization
        optimized_model = apply_cross_alignment_to_model(dummy_model, config)
        
        # Test that the model has the cross-alignment attributes
        assert hasattr(optimized_model, 'cross_alignment_manager'), "Model should have cross_alignment_manager"
        assert hasattr(optimized_model, 'perform_cross_alignment'), "Model should have perform_cross_alignment method"
        
        print("PASS: Qwen3-Coder-30B cross-alignment optimization test passed!")
        return True

    except Exception as e:
        print(f"FAIL: Qwen3-Coder-30B cross-alignment optimization test failed: {e}")
        return False


def test_qwen3_0_6b_cross_alignment():
    """Test Qwen3-0.6B cross-alignment optimization."""
    print("Testing Qwen3-0.6B cross-alignment optimization...")
    
    try:
        from src.models.language.qwen3_0_6b.cross_alignment_optimization import (
            CrossAlignmentConfig, 
            Qwen3SmallCrossAlignmentOptimizer,
            apply_cross_alignment_to_model
        )
        
        # Create a dummy model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(896, 896)  # Qwen3-0.6B specific size
            
            def forward(self, x):
                return self.linear(x)
        
        dummy_model = DummyModel()
        
        # Create config
        config = CrossAlignmentConfig()
        config.hidden_size = 896
        
        # Apply cross-alignment optimization
        optimized_model = apply_cross_alignment_to_model(dummy_model, config)
        
        # Test that the model has the cross-alignment attributes
        assert hasattr(optimized_model, 'cross_alignment_manager'), "Model should have cross_alignment_manager"
        assert hasattr(optimized_model, 'perform_cross_alignment'), "Model should have perform_cross_alignment method"
        
        print("PASS: Qwen3-0.6B cross-alignment optimization test passed!")
        return True

    except Exception as e:
        print(f"FAIL: Qwen3-0.6B cross-alignment optimization test failed: {e}")
        return False


def test_qwen3_coder_next_cross_alignment():
    """Test Qwen3-Coder-Next cross-alignment optimization."""
    print("Testing Qwen3-Coder-Next cross-alignment optimization...")
    
    try:
        from src.models.coding.qwen3_coder_next.cross_alignment_optimization import (
            CrossAlignmentConfig, 
            Qwen3CoderNextCrossAlignmentOptimizer,
            apply_cross_alignment_to_model
        )
        
        # Create a dummy model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5120, 5120)  # Qwen3-Coder-Next specific size
            
            def forward(self, x):
                return self.linear(x)
        
        dummy_model = DummyModel()
        
        # Create config
        config = CrossAlignmentConfig()
        config.hidden_size = 5120
        
        # Apply cross-alignment optimization
        optimized_model = apply_cross_alignment_to_model(dummy_model, config)
        
        # Test that the model has the cross-alignment attributes
        assert hasattr(optimized_model, 'cross_alignment_manager'), "Model should have cross_alignment_manager"
        assert hasattr(optimized_model, 'perform_cross_alignment'), "Model should have perform_cross_alignment method"
        
        print("PASS: Qwen3-Coder-Next cross-alignment optimization test passed!")
        return True

    except Exception as e:
        print(f"FAIL: Qwen3-Coder-Next cross-alignment optimization test failed: {e}")
        return False


def test_functionality():
    """Test the actual functionality of cross-alignment."""
    print("\nTesting cross-alignment functionality...")
    
    try:
        # Test GLM-4.7-Flash specifically
        from src.models.specialized.glm_4_7_flash.cross_alignment_optimization import (
            CrossAlignmentConfig, 
            GLM47FlashCrossAlignmentOptimizer
        )
        
        config = CrossAlignmentConfig()
        config.hidden_size = 256  # Use smaller size for test
        
        optimizer = GLM47FlashCrossAlignmentOptimizer(config, layer_idx=0)
        
        # Create test tensors
        batch_size, seq_len = 2, 10
        rep1 = torch.randn(batch_size, seq_len, config.hidden_size)
        rep2 = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # Test alignment
        (aligned_rep1, aligned_rep2), alignment_loss = optimizer(rep1, rep2)
        
        # Check shapes are preserved
        assert aligned_rep1.shape == rep1.shape, f"Shape mismatch: {aligned_rep1.shape} vs {rep1.shape}"
        assert aligned_rep2.shape == rep2.shape, f"Shape mismatch: {aligned_rep2.shape} vs {rep2.shape}"
        
        # Check loss is computed
        assert alignment_loss is not None, "Loss should be computed"
        assert isinstance(alignment_loss, torch.Tensor), "Loss should be a tensor"
        
        print("PASS: Cross-alignment functionality test passed!")
        return True

    except Exception as e:
        print(f"FAIL: Cross-alignment functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Running cross-alignment optimization tests...\n")
    
    tests = [
        test_glm_cross_alignment,
        test_qwen3_4b_cross_alignment,
        test_qwen3_coder_30b_cross_alignment,
        test_qwen3_0_6b_cross_alignment,
        test_qwen3_coder_next_cross_alignment,
        test_functionality
    ]
    
    results = []
    for test_func in tests:
        results.append(test_func())
        print()  # Empty line for readability
    
    passed = sum(results)
    total = len(results)
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! Cross-alignment optimizations are working correctly.")
        return True
    else:
        print("ERROR: Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)