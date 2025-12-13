"""
Comprehensive tests for the optimized data pipeline.
"""

import torch
import pytest
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from src.qwen3_vl.optimization.data_pipeline_optimization import (
    DataPipelineConfig,
    OptimizedDataset,
    OptimizedDataLoader,
    DataPipelineOptimizer,
    AsyncDataTransferBuffer,
    benchmark_pipeline_performance,
    create_multimodal_pipeline
)


class DummyDataset(Dataset):
    """A dummy dataset for testing."""
    def __init__(self, size=100):
        self.size = size
        self.texts = [f"Sample text {i}" for i in range(size)]
        # Create dummy images
        self.images = [Image.new('RGB', (224, 224), color='red') for _ in range(size)]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'image': self.images[idx]
        }


def dummy_tokenizer(text, **kwargs):
    """A dummy tokenizer for testing."""
    # Simulate tokenization
    return {
        'input_ids': torch.randint(0, 1000, (128,)),  # Fixed size for simplicity
        'attention_mask': torch.ones(128)
    }


def test_data_pipeline_config():
    """Test the DataPipelineConfig class."""
    config = DataPipelineConfig(
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        async_transfer=False
    )
    
    assert config.batch_size == 16
    assert config.num_workers == 4
    assert config.pin_memory is True
    assert config.async_transfer is False


def test_optimized_dataset():
    """Test the OptimizedDataset class."""
    dataset = DummyDataset(size=10)
    config = DataPipelineConfig(cache_size=5)
    
    optimized_dataset = OptimizedDataset(dataset, dummy_tokenizer, config)
    
    # Test basic functionality
    assert len(optimized_dataset) == 10
    
    # Test item retrieval
    item = optimized_dataset[0]
    assert 'text' in item
    assert 'image' in item
    assert 'input_ids' in item
    assert 'attention_mask' in item
    
    # Test caching
    stats_before = optimized_dataset.get_cache_stats()
    for i in range(3):
        _ = optimized_dataset[0]  # Access same item multiple times
    
    stats_after = optimized_dataset.get_cache_stats()
    assert stats_after['cache_hits'] > stats_before['cache_hits']
    assert stats_after['hit_rate'] >= 0.0


def test_async_data_transfer_buffer():
    """Test the AsyncDataTransferBuffer class."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buffer = AsyncDataTransferBuffer(max_buffer_size=5, device=device)
    
    # Create test tensor
    test_tensor = torch.randn(10, 20)
    
    # Test enqueuing and dequeuing
    buffer.enqueue_transfer(test_tensor)
    
    # Get the transferred tensor
    transferred_tensor = buffer.get_transferred_item(timeout=2.0)
    
    # Verify the tensor was transferred correctly
    if device.type == 'cuda':
        assert transferred_tensor.device.type == 'cuda'
    else:
        assert transferred_tensor.device.type == 'cpu'
    
    assert torch.equal(transferred_tensor.cpu(), test_tensor)
    
    # Test with dictionary
    test_dict = {
        'tensor1': torch.randn(5, 5),
        'tensor2': torch.randn(3, 3)
    }
    
    buffer.enqueue_transfer(test_dict)
    transferred_dict = buffer.get_transferred_item(timeout=2.0)
    
    if device.type == 'cuda':
        assert transferred_dict['tensor1'].device.type == 'cuda'
        assert transferred_dict['tensor2'].device.type == 'cuda'
    else:
        assert transferred_dict['tensor1'].device.type == 'cpu'
        assert transferred_dict['tensor2'].device.type == 'cpu'
    
    # Check statistics
    stats = buffer.get_stats()
    assert stats['transfers_initiated'] >= 2
    assert stats['transfers_completed'] >= 2
    
    # Stop the buffer thread
    buffer.stop()


def test_optimized_dataloader():
    """Test the OptimizedDataLoader class."""
    dataset = DummyDataset(size=20)
    config = DataPipelineConfig(batch_size=4, num_workers=0)  # Use 0 workers for testing
    
    dataloader = OptimizedDataLoader(dataset, config, dummy_tokenizer)
    
    # Test basic functionality
    assert len(dataloader) == 5  # 20 items / 4 batch_size = 5 batches
    
    # Iterate through batches
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'image' in batch
        assert batch['input_ids'].shape[0] == 4  # Batch size
        
        if batch_count >= 3:  # Test only first 3 batches
            break
    
    assert batch_count == 3
    
    # Test statistics
    stats = dataloader.get_pipeline_stats()
    assert 'avg_batch_time' in stats
    assert 'avg_transfer_time' in stats
    assert 'dataset_stats' in stats


def test_data_pipeline_optimizer():
    """Test the DataPipelineOptimizer class."""
    config = DataPipelineConfig(batch_size=2)
    optimizer = DataPipelineOptimizer(config)
    
    dataset = DummyDataset(size=10)
    dataloader = optimizer.create_optimized_dataloader(dataset, dummy_tokenizer)
    
    # Test that the dataloader was created correctly
    assert isinstance(dataloader, OptimizedDataLoader)
    
    # Test pipeline performance metrics
    perf_metrics = optimizer.get_pipeline_performance()
    assert 'total_data_loading_time' in perf_metrics
    assert 'total_preprocessing_time' in perf_metrics
    assert 'total_transfer_time' in perf_metrics
    assert 'total_pipeline_time' in perf_metrics


def test_create_multimodal_pipeline():
    """Test the create_multimodal_pipeline function."""
    dataset = DummyDataset(size=8)
    config = DataPipelineConfig(batch_size=2)
    
    dataloader = create_multimodal_pipeline(dataset, dummy_tokenizer, config)
    
    # Test that the dataloader was created correctly
    assert isinstance(dataloader, OptimizedDataLoader)
    
    # Test that it can iterate through data
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        assert 'input_ids' in batch
        assert 'image' in batch
        if batch_count >= 2:
            break
    
    assert batch_count == 2


def test_benchmark_pipeline_performance():
    """Test the benchmark_pipeline_performance function."""
    dataset = DummyDataset(size=12)
    config = DataPipelineConfig(batch_size=3, num_workers=0)
    
    optimizer = DataPipelineOptimizer(config)
    dataloader = optimizer.create_optimized_dataloader(dataset, dummy_tokenizer)
    
    # Run benchmark
    results = benchmark_pipeline_performance(dataloader, num_batches=2)
    
    # Verify results structure
    assert 'total_time' in results
    assert 'batches_processed' in results
    assert 'avg_time_per_batch' in results
    assert 'throughput_batches_per_sec' in results
    assert 'pipeline_stats' in results
    
    assert results['batches_processed'] == 2
    assert results['avg_time_per_batch'] >= 0
    assert results['throughput_batches_per_sec'] >= 0


def test_cache_clearing():
    """Test the cache clearing functionality."""
    dataset = DummyDataset(size=10)
    config = DataPipelineConfig(batch_size=2)
    
    dataloader = OptimizedDataLoader(dataset, config, dummy_tokenizer)
    
    # Process some data to populate caches
    for i, batch in enumerate(dataloader):
        if i >= 2:
            break
    
    # Clear caches
    dataloader.clear_caches()
    
    # This should not raise an exception
    assert True


def test_memory_threshold_handling():
    """Test memory threshold handling."""
    dataset = DummyDataset(size=10)
    config = DataPipelineConfig(batch_size=2, memory_threshold=0.1)  # Very low threshold
    
    dataloader = OptimizedDataLoader(dataset, config, dummy_tokenizer)
    
    # Process data - this should trigger memory checks
    for i, batch in enumerate(dataloader):
        if i >= 2:
            break
    
    # Should complete without errors
    assert True


def test_error_handling():
    """Test error handling in the pipeline."""
    # Test with invalid dataset
    with pytest.raises(Exception):
        # This would fail when trying to access the dataset
        invalid_dataset = None
        config = DataPipelineConfig(batch_size=2)
        OptimizedDataLoader(invalid_dataset, config, dummy_tokenizer)


def test_performance_under_load():
    """Test performance under load with larger dataset."""
    # Create a larger dataset
    dataset = DummyDataset(size=100)
    config = DataPipelineConfig(batch_size=8, num_workers=2)
    
    dataloader = OptimizedDataLoader(dataset, config, dummy_tokenizer)
    
    # Process multiple batches to test performance
    start_batch_count = 0
    for batch in dataloader:
        start_batch_count += 1
        if start_batch_count >= 5:  # Process 5 batches
            break
    
    # Verify we processed the expected number of batches
    assert start_batch_count == 5
    
    # Check performance statistics
    stats = dataloader.get_pipeline_stats()
    assert stats['total_batches_processed'] >= 5


def test_different_data_types():
    """Test the pipeline with different types of data."""
    # Create dataset with different structures
    class HeterogeneousDataset(Dataset):
        def __init__(self, size=10):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            if idx % 2 == 0:
                # Text-only sample
                return {'text': f'text_only_{idx}'}
            else:
                # Text + image sample
                return {
                    'text': f'text_image_{idx}',
                    'image': Image.new('RGB', (224, 224), color='blue')
                }
    
    dataset = HeterogeneousDataset(size=10)
    config = DataPipelineConfig(batch_size=2, num_workers=0)
    
    dataloader = OptimizedDataLoader(dataset, config, dummy_tokenizer)
    
    # Process batches with mixed data types
    for i, batch in enumerate(dataloader):
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        if i >= 3:
            break


if __name__ == "__main__":
    # Run all tests
    print("Running tests for optimized data pipeline...")
    
    test_data_pipeline_config()
    print("✓ DataPipelineConfig test passed")
    
    test_optimized_dataset()
    print("✓ OptimizedDataset test passed")
    
    test_async_data_transfer_buffer()
    print("✓ AsyncDataTransferBuffer test passed")
    
    test_optimized_dataloader()
    print("✓ OptimizedDataLoader test passed")
    
    test_data_pipeline_optimizer()
    print("✓ DataPipelineOptimizer test passed")
    
    test_create_multimodal_pipeline()
    print("✓ create_multimodal_pipeline test passed")
    
    test_benchmark_pipeline_performance()
    print("✓ benchmark_pipeline_performance test passed")
    
    test_cache_clearing()
    print("✓ Cache clearing test passed")
    
    test_memory_threshold_handling()
    print("✓ Memory threshold handling test passed")
    
    test_performance_under_load()
    print("✓ Performance under load test passed")
    
    test_different_data_types()
    print("✓ Different data types test passed")
    
    print("\nAll tests passed successfully!")