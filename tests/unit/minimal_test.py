"""
Minimal test for the end-to-end optimized inference pipeline without complex imports.
"""
import sys
import os
import torch
import torch.nn as nn

# Add the src directory to the path to enable proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test the core functionality of the end-to-end pipeline
def test_pipeline_creation():
    """Test creating the optimized inference pipeline."""
    print("Testing pipeline creation...")
    
    # Import the config
    from qwen3_vl.core.config import Qwen3VLConfig
    
    # Create a simple dummy model for testing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(512, 512)
            self.norm = nn.LayerNorm(512)
        
        def forward(self, x):
            if isinstance(x, dict):
                x = x.get('inputs', x)
            x = self.linear(x)
            x = self.norm(x)
            return x
    
    dummy_model = DummyModel()
    config = Qwen3VLConfig()
    
    # Now test our pipeline
    from qwen3_vl.optimization.end_to_end_inference_pipeline import (
        InferencePipelineConfig,
        OptimizedInferencePipeline
    )
    
    # Create pipeline configuration
    pipeline_config = InferencePipelineConfig(
        target_hardware="nvidia_sm61",
        max_batch_size=8,
        variable_batch_size=True,
        enable_async_io=True,
        enable_tensor_caching=True,
        enable_model_caching=True,
        enable_prefetching=True,
        enable_pinned_memory=True,
        enable_async_transfers=True
    )
    
    # Create the optimized pipeline
    pipeline = OptimizedInferencePipeline(dummy_model, pipeline_config)
    print("✓ Optimized inference pipeline created successfully")
    
    # Test basic functionality
    test_input = torch.randn(4, 64, 512)  # [batch, seq, features]
    result = pipeline.generate_response(test_input)
    
    assert result.shape == test_input.shape
    print("✓ Basic inference test passed")
    
    # Test with multiple inputs
    test_inputs = [
        (torch.randn(2, 32, 512), "language"),
        (torch.randn(1, 3, 224, 224), "vision"),
    ]
    
    from qwen3_vl.optimization.end_to_end_inference_pipeline import EfficientPipeline
    efficient_pipeline = EfficientPipeline(dummy_model, pipeline_config)
    results = efficient_pipeline.run_inference(test_inputs)
    
    assert len(results) > 0
    print("✓ Variable input processing test passed")
    
    # Test performance benchmarking
    benchmark_results = pipeline.benchmark_performance(test_inputs, num_runs=3)
    print(f"✓ Benchmark test passed: {benchmark_results['avg_inference_time']:.4f}s avg")
    
    print("\nAll minimal tests passed! The end-to-end optimized inference pipeline is working correctly.")


if __name__ == "__main__":
    test_pipeline_creation()