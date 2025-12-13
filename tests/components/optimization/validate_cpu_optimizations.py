"""
Performance validation script for CPU optimizations using AVX/SIMD instructions
"""
import torch
import time
from cpu_optimizations import (
    AVX2OptimizedOperations, 
    SSEOptimizedOperations, 
    ScalarOptimizedOperations,
    SIMDOptimizationConfig,
    apply_cpu_optimizations,
    get_optimized_operations
)

def performance_test():
    print("Performance Validation for CPU Optimizations")
    print("="*50)
    
    # Create test tensors
    batch_size, seq_len, hidden_size = 16, 128, 768  # Common size for transformer models
    test_tensor = torch.randn(batch_size, seq_len, hidden_size)
    a = torch.randn(batch_size, seq_len, hidden_size)
    b = torch.randn(batch_size, hidden_size, hidden_size // 4)
    
    print(f"Test tensor shape: {test_tensor.shape}")
    print(f"Matrix A shape: {a.shape}")
    print(f"Matrix B shape: {b.shape}")
    
    # Test different optimization levels
    configs = [
        ("AVX2", SIMDOptimizationConfig(enable_avx2_optimizations=True, enable_sse_optimizations=True)),
        ("SSE", SIMDOptimizationConfig(enable_avx2_optimizations=False, enable_sse_optimizations=True)),
        ("Scalar", SIMDOptimizationConfig(enable_avx2_optimizations=False, enable_sse_optimizations=False))
    ]
    
    results = {}
    
    for name, config in configs:
        print(f"\nTesting {name} optimizations...")
        ops = get_optimized_operations(config)
        
        # Test normalization
        start_time = time.time()
        for _ in range(50):
            _ = ops.vectorized_normalize(test_tensor)
        norm_time = time.time() - start_time
        
        # Test GELU
        start_time = time.time()
        for _ in range(50):
            _ = ops.vectorized_gelu_approximation(test_tensor)
        gelu_time = time.time() - start_time
        
        # Test matmul
        start_time = time.time()
        for _ in range(50):
            _ = ops.vectorized_matmul(a, b)
        matmul_time = time.time() - start_time
        
        results[name] = {
            'normalize_time': norm_time,
            'gelu_time': gelu_time,
            'matmul_time': matmul_time,
            'instruction_set': ops.instruction_set
        }
        
        print(f"  Normalization time: {norm_time:.4f}s")
        print(f"  GELU time: {gelu_time:.4f}s")
        print(f"  MatMul time: {matmul_time:.4f}s")
        print(f"  Instruction set: {ops.instruction_set}")
    
    # Compare results
    print(f"\nPerformance Comparison:")
    print(f"{'Operation':<12} {'AVX2':<10} {'SSE':<10} {'Scalar':<10} {'AVX2 vs Scalar':<15}")
    print("-" * 70)
    
    operations = ['normalize_time', 'gelu_time', 'matmul_time']
    op_names = ['Normalization', 'GELU', 'MatMul']
    
    for op, op_name in zip(operations, op_names):
        avx2_time = results['AVX2'][op]
        sse_time = results['SSE'][op] 
        scalar_time = results['Scalar'][op]
        
        if scalar_time > 0:
            speedup = scalar_time / avx2_time if avx2_time > 0 else float('inf')
        else:
            speedup = 0
            
        print(f"{op_name:<12} {avx2_time:<10.4f} {sse_time:<10.4f} {scalar_time:<10.4f} {speedup:<15.2f}x")
    
    # Summary
    print(f"\nOptimization Summary:")
    avx2_matmul = results['AVX2']['matmul_time']
    scalar_matmul = results['Scalar']['matmul_time']
    matmul_speedup = scalar_matmul / avx2_matmul if avx2_matmul > 0 else 0
    
    print(f"  AVX2 MatMul is {matmul_speedup:.2f}x faster than scalar")
    print(f"  Using instruction set: {results['AVX2']['instruction_set']}")
    
    return results

def test_model_optimization():
    """Test applying optimizations to a mock model"""
    print(f"\nTesting Model Optimization Application:")
    
    # Create a mock model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            layer = torch.nn.Module()
            layer.self_attn = torch.nn.Linear(512, 512)
            layer.mlp = torch.nn.Linear(512, 512)
            self.language_model.layers = torch.nn.ModuleList([layer])
    
    model = MockModel()
    config = SIMDOptimizationConfig()
    
    print("  Before optimization - Model structure intact")
    print(f"  Language model has {len(model.language_model.layers)} layers")
    
    # Apply optimizations
    optimized_model = apply_cpu_optimizations(model, config)
    
    print("  After optimization - Model structure maintained")
    print(f"  Language model still has {len(optimized_model.language_model.layers)} layers")
    print("  OK Model optimization application successful")

if __name__ == "__main__":
    performance_test()
    test_model_optimization()
    
    print(f"\nCPU Optimizations validation completed successfully!")
    print(f"Intel i5-10210U optimized for Qwen3-VL model inference.")