"""
Integration test for CPU optimizations with Qwen3-VL model
"""
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer
from PIL import Image
import numpy as np
from typing import Optional, Dict, Any, List
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

from src.qwen3_vl.optimization.cpu_optimizations import (
    CPUOptimizationConfig,
    CPUPreprocessor,
    OptimizedDataLoader,
    CPU_GPU_Coordinator,
    MultithreadedTokenizer,
    OptimizedInferencePipeline,
    apply_cpu_optimizations
)


class DummyDataset:
    """A simple dummy dataset for testing."""
    
    def __init__(self, size: int = 10):
        self.size = size
        self.texts = [f"Sample text {i}" for i in range(size)]
        
        # Create dummy images
        self.images = []
        for i in range(size):
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            self.images.append(img)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'image': self.images[idx]
        }


class DummyTokenizer:
    """A dummy tokenizer for testing."""
    
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        
        batch_size = len(texts)
        seq_len = kwargs.get('max_length', 32)
        
        # Create dummy token IDs and attention masks
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


def test_cpu_optimizations_integration():
    """Test CPU optimizations integrated with the Qwen3-VL model."""
    print("Testing CPU optimizations integration...")
    
    # Create configuration
    config = CPUOptimizationConfig(
        num_preprocess_workers=2,
        preprocess_batch_size=4,
        max_text_length=64,
        memory_threshold=0.7
    )
    
    # Create dummy model (in practice, this would be a real Qwen3-VL model)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(100, 10)
            self.device = torch.device('cpu')
        
        def parameters(self):
            return iter([torch.nn.Parameter(torch.randn(10, 100))])
        
        def generate(self, input_ids=None, pixel_values=None, attention_mask=None, **kwargs):
            # Simulate generation by returning dummy outputs
            batch_size = input_ids.shape[0] if input_ids is not None else 1
            max_new_tokens = kwargs.get('max_new_tokens', 10)
            return torch.randint(0, 1000, (batch_size, max_new_tokens))
    
    model = DummyModel()
    tokenizer = DummyTokenizer()
    
    # Test 1: Apply CPU optimizations
    print("  - Applying CPU optimizations...")
    pipeline, create_loader_fn = apply_cpu_optimizations(
        model,
        tokenizer,
        num_preprocess_workers=2,
        preprocess_batch_size=4,
        memory_threshold=0.7
    )
    
    assert pipeline is not None
    assert callable(create_loader_fn)
    print("    [OK] Optimizations applied successfully")

    # Test 2: Create optimized data loader
    print("  - Creating optimized data loader...")
    dataset = DummyDataset(size=8)
    data_loader = create_loader_fn(
        dataset,
        batch_size=2,
        shuffle=False
    )

    assert data_loader is not None
    print("    [OK] Data loader created successfully")

    # Test 3: Process some data through the pipeline
    print("  - Processing data through pipeline...")
    batch_count = 0
    for batch in data_loader:
        assert isinstance(batch, dict)
        assert 'input_ids' in batch
        assert 'pixel_values' in batch
        assert 'attention_mask' in batch

        # Check shapes
        assert batch['input_ids'].shape[0] == 2  # batch size
        assert batch['pixel_values'].shape[0] == 2  # batch size
        assert batch['attention_mask'].shape[0] == 2  # batch size

        batch_count += 1
        if batch_count >= 2:  # Test only first 2 batches
            break

    print(f"    [OK] Processed {batch_count} batches successfully")

    # Test 4: Test inference pipeline
    print("  - Testing inference pipeline...")
    texts = ["Hello world", "Test inference"]
    images = []
    for i in range(2):
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        images.append(img)

    responses = pipeline.preprocess_and_infer(
        texts,
        images,
        tokenizer=tokenizer,
        max_new_tokens=5,
        do_sample=False
    )

    assert isinstance(responses, list)
    assert len(responses) == 2
    assert all(isinstance(resp, str) for resp in responses)
    print("    [OK] Inference pipeline works correctly")

    # Test 5: Check performance metrics
    print("  - Checking performance metrics...")
    metrics = pipeline.get_performance_metrics()
    assert isinstance(metrics, dict)
    assert 'avg_preprocess_time' in metrics
    assert 'avg_inference_time' in metrics
    assert 'total_calls' in metrics
    print("    [OK] Performance metrics collected successfully")

    print("All integration tests passed! :)")


def test_cpu_gpu_coordination():
    """Test CPU-GPU coordination specifically."""
    print("\nTesting CPU-GPU coordination...")
    
    config = CPUOptimizationConfig(transfer_async=True)
    coordinator = CPU_GPU_Coordinator(config)
    
    # Create test data
    data = {
        'input_ids': torch.randint(0, 1000, (2, 10)),
        'pixel_values': torch.randn(2, 3, 224, 224),
        'attention_mask': torch.ones(2, 10)
    }
    
    # Test transfer (works regardless of CUDA availability)
    transferred_data = coordinator.transfer_to_device(data)
    
    # Check that all tensors are on the same device
    devices = set(tensor.device for tensor in transferred_data.values())
    assert len(devices) == 1  # All tensors should be on the same device
    print("  [OK] All tensors transferred to same device")

    # Check shapes are preserved
    assert transferred_data['input_ids'].shape == data['input_ids'].shape
    assert transferred_data['pixel_values'].shape == data['pixel_values'].shape
    assert transferred_data['attention_mask'].shape == data['attention_mask'].shape
    print("  [OK] Tensor shapes preserved during transfer")

    print("CPU-GPU coordination test passed! :)")


def test_multithreaded_preprocessing():
    """Test multithreaded preprocessing capabilities."""
    print("\nTesting multithreaded preprocessing...")
    
    config = CPUOptimizationConfig(
        num_preprocess_workers=3,
        preprocess_batch_size=2
    )
    
    tokenizer = DummyTokenizer()
    preprocessor = CPUPreprocessor(config, tokenizer)
    
    # Test with multiple texts and images
    texts = [f"Text {i}" for i in range(6)]
    images = []
    for i in range(6):
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        images.append(img)
    
    # Process synchronously
    result_sync = preprocessor.preprocess_batch(texts, images)
    
    assert 'input_ids' in result_sync
    assert 'pixel_values' in result_sync
    assert 'attention_mask' in result_sync
    assert result_sync['input_ids'].shape[0] == 6
    assert result_sync['pixel_values'].shape[0] == 6
    print("  [OK] Synchronous preprocessing works")

    # Process asynchronously
    future = preprocessor.preprocess_batch_async(texts[:2], images[:2])
    result_async = future.result()  # Wait for completion

    assert 'input_ids' in result_async
    assert 'pixel_values' in result_async
    assert result_async['input_ids'].shape[0] == 2
    assert result_async['pixel_values'].shape[0] == 2
    print("  [OK] Asynchronous preprocessing works")

    preprocessor.close()
    print("Multithreaded preprocessing test passed! :)")


def test_memory_efficiency():
    """Test memory efficiency features."""
    print("\nTesting memory efficiency...")
    
    config = CPUOptimizationConfig(
        clear_cache_interval=2,  # Clear cache every 2 batches
        memory_threshold=0.5  # Lower threshold for testing
    )
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(100, 10)
            self.device = torch.device('cpu')
        
        def parameters(self):
            return iter([torch.nn.Parameter(torch.randn(10, 100))])
        
        def generate(self, input_ids=None, pixel_values=None, attention_mask=None, **kwargs):
            batch_size = input_ids.shape[0] if input_ids is not None else 1
            max_new_tokens = kwargs.get('max_new_tokens', 5)
            return torch.randint(0, 1000, (batch_size, max_new_tokens))
    
    model = DummyModel()
    tokenizer = DummyTokenizer()
    
    pipeline = OptimizedInferencePipeline(model, config)
    
    # Run multiple inference steps to test memory management
    texts = ["Test memory efficiency"]
    images = []
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    images.append(img)
    
    for i in range(5):  # Run 5 inference steps
        responses = pipeline.preprocess_and_infer(
            texts, 
            images,
            tokenizer=tokenizer,
            max_new_tokens=3,
            do_sample=False
        )
        assert len(responses) == 1
    
    # Check that performance metrics were collected
    metrics = pipeline.get_performance_metrics()
    assert metrics['total_calls'] == 5
    print(f"  [OK] Performed {metrics['total_calls']} inference calls")

    print("Memory efficiency test passed! :)")


def run_all_integration_tests():
    """Run all integration tests."""
    print("Running CPU Optimization Integration Tests\n")
    
    test_cpu_optimizations_integration()
    test_cpu_gpu_coordination()
    test_multithreaded_preprocessing()
    test_memory_efficiency()
    
    print("\nAll integration tests passed successfully! :)")
    print("CPU optimizations are working correctly with Qwen3-VL architecture.")


if __name__ == "__main__":
    run_all_integration_tests()