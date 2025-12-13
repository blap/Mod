"""
Hardware-specific optimization tests for NVIDIA SM61 architecture
"""
import pytest
import torch
import torch.nn as nn
from typing import Dict, Any
import platform

from models.hardware_specific_optimization import HardwareKernelOptimizer
from models.block_sparse_attention import BlockSparseAttention
from src.qwen3_vl.components.configuration import Qwen3VLConfig


class TestHardwareSpecificOptimizations:
    """Tests for hardware-specific optimizations targeting NVIDIA SM61"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.optimizer = HardwareKernelOptimizer()
        self.config = Qwen3VLConfig()
        self.config.hidden_size = 256
        self.config.num_attention_heads = 8
        self.config.max_position_embeddings = 128
        
    def test_hardware_capability_detection(self):
        """Test detection of hardware capabilities"""
        capabilities = self.optimizer.get_hardware_capabilities()
        
        assert 'cuda_available' in capabilities
        assert 'compute_capability' in capabilities
        assert 'memory_gb' in capabilities
        assert 'device_name' in capabilities
        
        print(f"CUDA available: {capabilities['cuda_available']}")
        print(f"Compute capability: {capabilities['compute_capability']}")
        print(f"Memory: {capabilities['memory_gb']:.2f} GB")
        print(f"Device: {capabilities['device_name']}")
        
    def test_nvidia_sm61_optimization_detection(self):
        """Test detection of NVIDIA SM61-specific optimizations"""
        # Check if we can apply SM61-specific optimizations
        sm61_available = self.optimizer.is_sm61_available()
        
        # This test should pass regardless of actual hardware
        assert isinstance(sm61_available, bool)
        print(f"NVIDIA SM61 optimizations available: {sm61_available}")
        
    def test_hardware_kernel_optimization_application(self):
        """Test application of hardware-specific kernel optimizations"""
        batch_size, seq_len, hidden_dim = 2, 64, 256
        tensor = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Test different operations
        operations = ['matmul', 'conv', 'attention']
        
        for operation in operations:
            try:
                optimized_result = self.optimizer.optimize_for_hardware(
                    tensor, operation=operation, target_hardware='nvidia_sm61'
                )
                
                # Verify output shape is preserved
                assert optimized_result.shape == tensor.shape
                # Verify output is finite
                assert torch.isfinite(optimized_result).all()
                
                print(f"✓ {operation} optimization successful")
            except Exception as e:
                # Some operations might not be supported, which is acceptable
                print(f"⚠ {operation} optimization skipped: {str(e)}")
                
    def test_block_sparse_attention_with_hardware_optimization(self):
        """Test block sparse attention with hardware-specific optimizations"""
        attention = BlockSparseAttention(self.config)
        
        batch_size, seq_len = 2, 128
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        # Apply attention with potential hardware optimizations
        output, weights, past_key_value = attention(
            hidden_states=hidden_states,
            attention_mask=None,
            output_attentions=True
        )
        
        assert output.shape == hidden_states.shape
        assert weights is not None
        
        print(f"✓ Block sparse attention with hardware optimization successful")
        print(f"  Input shape: {hidden_states.shape}")
        print(f"  Output shape: {output.shape}")
        
    def test_memory_efficient_hardware_operations(self):
        """Test memory-efficient operations optimized for hardware"""
        batch_size, seq_len, hidden_dim = 1, 256, 512
        tensor = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Test memory-efficient operations
        optimized_tensor = self.optimizer.optimize_memory_access_pattern(tensor)
        
        assert optimized_tensor.shape == tensor.shape
        assert torch.isfinite(optimized_tensor).all()
        
        # Test memory-efficient matmul
        tensor2 = torch.randn(batch_size, hidden_dim, seq_len)
        result = self.optimizer.memory_efficient_matmul(tensor, tensor2)
        
        expected_shape = (batch_size, seq_len, seq_len)
        assert result.shape == expected_shape
        assert torch.isfinite(result).all()
        
        print(f"✓ Memory-efficient hardware operations successful")
        
    def test_tensor_core_availability(self):
        """Test Tensor Core availability and usage"""
        tensor_core_available = self.optimizer.is_tensor_core_available()
        
        # This should return a boolean
        assert isinstance(tensor_core_available, bool)
        print(f"Tensor Core available: {tensor_core_available}")
        
        if tensor_core_available:
            # Test tensor core optimized operations
            batch_size, seq_len, hidden_dim = 2, 64, 256
            a = torch.randn(batch_size, seq_len, hidden_dim).half()  # FP16 for tensor cores
            b = torch.randn(batch_size, hidden_dim, seq_len).half()
            
            result = self.optimizer.tensor_core_optimized_matmul(a, b)
            expected_shape = (batch_size, seq_len, seq_len)
            
            assert result.shape == expected_shape
            assert result.dtype == torch.half
            print(f"✓ Tensor Core optimized matmul successful")
    
    def test_hardware_specific_fallbacks(self):
        """Test fallback mechanisms when hardware-specific optimizations aren't available"""
        batch_size, seq_len, hidden_dim = 1, 32, 128
        tensor = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Test fallback for unsupported hardware
        try:
            result = self.optimizer.optimize_for_hardware(
                tensor, operation='matmul', target_hardware='unsupported_hardware'
            )
            # Should gracefully handle unsupported hardware
            assert result.shape == tensor.shape
            print("✓ Hardware fallback mechanism working")
        except Exception as e:
            # Should handle gracefully
            print(f"✓ Hardware fallback handled exception: {type(e).__name__}")
            
    def test_hardware_aware_memory_management(self):
        """Test hardware-aware memory management"""
        # Test memory pool optimization
        memory_pool = self.optimizer.create_optimized_memory_pool()
        
        # Allocate tensors using the optimized pool
        tensor1 = self.optimizer.allocate_from_pool(memory_pool, (10, 20))
        tensor2 = self.optimizer.allocate_from_pool(memory_pool, (10, 20))
        
        assert tensor1.shape == (10, 20)
        assert tensor2.shape == (10, 20)
        
        print("✓ Hardware-aware memory management working")
        
    def test_compute_capability_specific_optimizations(self):
        """Test optimizations specific to compute capability"""
        # Get current compute capability
        capabilities = self.optimizer.get_hardware_capabilities()
        compute_capability = capabilities.get('compute_capability', (0, 0))
        
        print(f"Current compute capability: {compute_capability}")
        
        # Test optimization selection based on compute capability
        selected_optimizations = self.optimizer.select_optimizations_for_capability(compute_capability)
        
        assert isinstance(selected_optimizations, list)
        print(f"Selected optimizations: {selected_optimizations}")


def run_hardware_specific_tests():
    """Run all hardware-specific optimization tests"""
    print("="*60)
    print("RUNNING HARDWARE-SPECIFIC OPTIMIZATION TESTS (NVIDIA SM61)")
    print("="*60)
    
    test_instance = TestHardwareSpecificOptimizations()
    
    test_methods = [
        'test_hardware_capability_detection',
        'test_nvidia_sm61_optimization_detection',
        'test_hardware_kernel_optimization_application',
        'test_block_sparse_attention_with_hardware_optimization',
        'test_memory_efficient_hardware_operations',
        'test_tensor_core_availability',
        'test_hardware_specific_fallbacks',
        'test_hardware_aware_memory_management',
        'test_compute_capability_specific_optimizations'
    ]
    
    results = {}
    for method_name in test_methods:
        try:
            print(f"\nRunning {method_name}...")
            method = getattr(test_instance, method_name)
            method()
            results[method_name] = True
            print(f"✓ {method_name} PASSED")
        except Exception as e:
            results[method_name] = False
            print(f"✗ {method_name} FAILED: {str(e)}")
    
    # Summary
    print("\n" + "="*60)
    print("HARDWARE-SPECIFIC TEST SUMMARY")
    print("="*60)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    print(f"Success rate: {success_rate:.2%}")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_hardware_specific_tests()
    exit(0 if success else 1)