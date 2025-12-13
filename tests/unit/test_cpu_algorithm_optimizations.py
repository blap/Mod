"""
Test suite for CPU algorithm optimizations in Qwen3-VL model
"""
import torch
import torch.nn as nn
import numpy as np
import pytest
from unittest.mock import Mock, patch
from PIL import Image
from typing import Dict, List, Optional
from src.qwen3_vl.optimization.cpu_algorithm_optimizations import (
    AlgorithmOptimizationConfig,
    CacheOptimizedArray,
    OptimizedSortAlgorithms,
    OptimizedSearchAlgorithms,
    CacheOptimizedDict,
    OptimizedMemoizationCache,
    cpu_cache_optimized_memoize,
    OptimizedDataStructures,
    AdvancedTokenizationWithAlgorithmOptimizations,
    OptimizedPreprocessorWithAlgorithmEnhancements,
    OptimizedInferencePipelineWithAlgorithmEnhancements,
    apply_algorithm_optimizations
)


def create_dummy_model():
    """Create a dummy model for testing purposes."""
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(100, 10)

        def forward(self, **kwargs):
            # Extract input_ids from kwargs and create a dummy output
            input_ids = kwargs.get('input_ids', torch.randn(2, 10, 100))
            if input_ids.dim() == 3:
                batch_size, seq_len, hidden_size = input_ids.shape
            elif input_ids.dim() == 2:
                batch_size, seq_len = input_ids.shape
                hidden_size = 100
            else:
                batch_size = 1
                seq_len = 10
                hidden_size = 100

            # Create a dummy output based on input size
            dummy_output = torch.randn(batch_size, seq_len, 10)
            return dummy_output

        def generate(self, input_ids=None, pixel_values=None, attention_mask=None, **kwargs):
            # Simulate generation - return a tensor with shape [batch_size, generated_length]
            batch_size = input_ids.shape[0] if input_ids is not None else 1
            generated_length = kwargs.get('max_new_tokens', 10)
            return torch.randint(0, 1000, (batch_size, generated_length))

    model = DummyModel()
    return model


def create_dummy_tokenizer():
    """Create a dummy tokenizer for testing purposes."""
    def mock_tokenize(texts, **kwargs):
        batch_size = len(texts) if isinstance(texts, list) else 1
        seq_len = 10
        return {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len)
        }

    tokenizer = Mock()
    tokenizer.side_effect = mock_tokenize
    return tokenizer


def create_test_images(count: int = 2) -> List[Image.Image]:
    """Create test PIL images."""
    images = []
    for _ in range(count):
        # Create a random RGB image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        images.append(img)
    return images


def test_algorithm_optimization_config():
    """Test algorithm optimization configuration."""
    config = AlgorithmOptimizationConfig()

    assert config.l1_cache_size == 32 * 1024
    assert config.l2_cache_size == 256 * 1024
    assert config.l3_cache_size == 6 * 1024 * 1024
    assert config.cache_line_size == 64
    assert config.insertion_sort_threshold == 10
    assert config.merge_sort_threshold == 100
    assert config.quick_sort_threshold == 1000
    assert config.memoization_cache_size == 1000
    assert config.enable_memoization is True


def test_cache_optimized_array():
    """Test cache-optimized array functionality."""
    size = 100
    arr = CacheOptimizedArray(size)

    assert arr.size == size
    assert isinstance(arr.tensor, torch.Tensor)
    assert arr.tensor.shape[0] >= size  # May be padded to cache line boundary

    # Test cache line aligned view
    view = arr.get_cache_line_aligned_view(10, 20)
    assert isinstance(view, torch.Tensor)
    assert len(view) > 0

    # Test access pattern optimization
    indices = [5, 2, 8, 1, 9]
    result = arr.access_pattern_optimized(indices)
    assert isinstance(result, torch.Tensor)
    assert len(result) == len(indices)


def test_optimized_sort_algorithms():
    """Test optimized sorting algorithms."""
    sorter = OptimizedSortAlgorithms()
    config = AlgorithmOptimizationConfig()

    # Test with small array (should use insertion sort)
    small_arr = np.array([5, 2, 8, 1, 9])
    sorted_small = sorter.hybrid_sort(small_arr, config)
    expected_small = np.array([1, 2, 5, 8, 9])
    np.testing.assert_array_equal(sorted_small, expected_small)

    # Test with medium array (should use merge sort)
    medium_arr = np.random.randint(0, 100, 50)
    sorted_medium = sorter.hybrid_sort(medium_arr, config)
    expected_medium = np.sort(medium_arr)
    np.testing.assert_array_equal(sorted_medium, expected_medium)

    # Test with large array (should use quick sort)
    large_arr = np.random.randint(0, 1000, 1500)
    sorted_large = sorter.hybrid_sort(large_arr, config)
    expected_large = np.sort(large_arr)
    np.testing.assert_array_equal(sorted_large, expected_large)


def test_optimized_search_algorithms():
    """Test optimized search algorithms."""
    searcher = OptimizedSearchAlgorithms()

    # Test binary search
    sorted_arr = np.array([1, 3, 5, 7, 9, 11, 13, 15])
    index = searcher.binary_search(sorted_arr, 7)
    assert index == 3  # Value 7 is at index 3

    index = searcher.binary_search(sorted_arr, 4)
    assert index == -1  # Value 4 is not in the array

    # Test interpolation search on uniformly distributed array
    uniform_arr = np.arange(0, 100, 2)  # [0, 2, 4, ..., 98]
    index = searcher.interpolation_search(uniform_arr, 50)
    assert index == 25  # Value 50 is at index 25

    # Test optimized search (should choose best algorithm)
    index = searcher.optimized_search(sorted_arr, 11)
    assert index == 5  # Value 11 is at index 5


def test_cache_optimized_dict():
    """Test cache-optimized dictionary."""
    cache_dict = CacheOptimizedDict()

    # Test put and get
    cache_dict.put("key1", "value1")
    cache_dict.put("key2", "value2")
    
    assert cache_dict.get("key1") == "value1"
    assert cache_dict.get("key2") == "value2"
    assert cache_dict.get("nonexistent") is None
    assert cache_dict.get("nonexistent", "default") == "default"

    # Test delete
    assert cache_dict.delete("key1") is True
    assert cache_dict.get("key1") is None
    assert cache_dict.delete("nonexistent") is False

    # Test size
    assert len(cache_dict) == 1  # Only key2 remains
    cache_dict.put("key3", "value3")
    assert len(cache_dict) == 2


def test_optimized_memoization_cache():
    """Test optimized memoization cache."""
    cache = OptimizedMemoizationCache(max_size=3)

    # Test put and get
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("nonexistent") is None

    # Test LRU eviction
    cache.put("key3", "value3")
    cache.put("key4", "value4")  # This should evict key1
    
    assert cache.get("key1") is None  # Evicted
    assert cache.get("key2") == "value2"  # Still there
    assert cache.get("key3") == "value3"  # Still there
    assert cache.get("key4") == "value4"  # New entry

    # Test stats
    stats = cache.get_stats()
    assert stats['hits'] >= 0
    assert stats['misses'] >= 0
    assert 0 <= stats['hit_rate'] <= 1
    assert stats['size'] == 3


def test_cpu_cache_optimized_memoize():
    """Test CPU cache-optimized memoization decorator."""
    call_count = 0
    
    @cpu_cache_optimized_memoize(maxsize=5)
    def test_function(x):
        nonlocal call_count
        call_count += 1
        return x * x

    # Call with same argument multiple times
    result1 = test_function(5)
    result2 = test_function(5)  # Should be cached
    result3 = test_function(3)
    result4 = test_function(5)  # Should be cached

    assert result1 == 25
    assert result2 == 25
    assert result3 == 9
    assert result4 == 25
    assert call_count == 2  # Only called twice (once for 5, once for 3)

    # Check stats
    stats = test_function.cache_stats()
    assert stats['hits'] >= 1  # At least one cache hit
    assert stats['misses'] == 2  # Two cache misses (for 5 and 3)

    # Test cache clear
    test_function.cache_clear()
    stats_after_clear = test_function.cache_stats()
    assert stats_after_clear['hits'] == 0
    assert stats_after_clear['misses'] == 0


def test_optimized_data_structures():
    """Test optimized data structures."""
    # Test cache-optimized list
    cache_list = OptimizedDataStructures.create_cache_optimized_list(100)
    assert isinstance(cache_list, torch.Tensor)
    assert cache_list.shape[0] >= 100  # May be padded

    # Test spatially aware dict
    spatial_dict = OptimizedDataStructures.create_spatially_aware_dict()
    assert isinstance(spatial_dict, CacheOptimizedDict)

    # Test sorted structure
    sorted_struct = OptimizedDataStructures.create_sorted_structure()
    sorted_struct['key1'] = 'value1'
    sorted_struct['key0'] = 'value0'
    keys = list(sorted_struct.keys())
    assert keys[0] == 'key0' and keys[1] == 'key1'  # Should be sorted


def test_advanced_tokenization_with_algorithm_optimizations():
    """Test advanced tokenization with algorithm optimizations."""
    tokenizer = create_dummy_tokenizer()
    config = AlgorithmOptimizationConfig()

    advanced_tokenizer = AdvancedTokenizationWithAlgorithmOptimizations(tokenizer, config)

    # Test with repeated texts to verify memoization
    texts = ["Hello", "World", "Hello", "Test"]  # "Hello" appears twice
    result = advanced_tokenizer.tokenize_batch_optimized(texts)

    assert isinstance(result, dict)
    assert 'input_ids' in result
    assert 'attention_mask' in result
    assert result['input_ids'].shape[0] == 4  # batch size

    # Check performance stats
    stats = advanced_tokenizer.get_performance_stats()
    assert 'memoization_stats' in stats
    assert 'cache_size' in stats


def test_optimized_preprocessor_with_algorithm_enhancements():
    """Test optimized preprocessor with algorithm enhancements."""
    config = AlgorithmOptimizationConfig()
    tokenizer = create_dummy_tokenizer()
    preprocessor = OptimizedPreprocessorWithAlgorithmEnhancements(config, tokenizer)

    texts = ["Hello world", "Test text"]
    images = create_test_images(2)

    result = preprocessor.preprocess_batch_optimized(texts, images)

    assert isinstance(result, dict)
    assert 'input_ids' in result or 'pixel_values' in result  # At least one should be present

    # Test with images only
    result_images_only = preprocessor.preprocess_batch_optimized([], images)
    assert isinstance(result_images_only, dict)
    assert 'pixel_values' in result_images_only

    # Test with text only
    result_text_only = preprocessor.preprocess_batch_optimized(texts, [])
    assert isinstance(result_text_only, dict)
    assert 'input_ids' in result_text_only

    preprocessor.close()


def test_optimized_inference_pipeline_with_algorithm_enhancements():
    """Test optimized inference pipeline with algorithm enhancements."""
    config = AlgorithmOptimizationConfig()

    # Create a mock model that can handle generate calls
    mock_model = Mock()
    mock_model.generate.return_value = torch.randint(0, 1000, (2, 20))
    # Add a parameters method that returns an iterator
    param = torch.nn.Parameter(torch.randn(10, 10))
    mock_model.parameters.return_value = iter([param])

    pipeline = OptimizedInferencePipelineWithAlgorithmEnhancements(mock_model, config)

    texts = ["Hello", "World"]
    images = create_test_images(2)

    # Test preprocessing and inference
    responses = pipeline.preprocess_and_infer(
        texts, images,
        max_new_tokens=10,
        do_sample=False
    )

    assert isinstance(responses, list)
    assert len(responses) == 2
    assert all(isinstance(resp, str) for resp in responses)

    # Check performance metrics
    metrics = pipeline.get_performance_metrics()
    assert 'avg_preprocess_time' in metrics
    assert 'avg_inference_time' in metrics
    assert 'total_calls' in metrics


def test_apply_algorithm_optimizations():
    """Test the main algorithm optimization application function."""
    model = create_dummy_model()
    tokenizer = create_dummy_tokenizer()

    # Apply algorithm optimizations
    pipeline = apply_algorithm_optimizations(
        model,
        tokenizer,
        insertion_sort_threshold=5,
        merge_sort_threshold=50,
        memoization_cache_size=500
    )

    assert isinstance(pipeline, OptimizedInferencePipelineWithAlgorithmEnhancements)

    # Test that we can run inference
    texts = ["Test input"]
    images = create_test_images(1)

    responses = pipeline.preprocess_and_infer(
        texts, images,
        max_new_tokens=5,
        do_sample=False
    )

    assert isinstance(responses, list)
    assert len(responses) == 1


def test_performance_comparison():
    """Test performance comparison between algorithms."""
    config = AlgorithmOptimizationConfig()
    sorter = OptimizedSortAlgorithms()
    
    # Test different array sizes
    sizes = [5, 50, 500, 1500]
    
    for size in sizes:
        arr = np.random.randint(0, 1000, size)
        
        # Time the hybrid sort
        import time
        start_time = time.time()
        sorted_arr = sorter.hybrid_sort(arr, config)
        end_time = time.time()
        
        # Verify it's actually sorted
        assert np.array_equal(sorted_arr, np.sort(arr))
        print(f"Sorted array of size {size} in {end_time - start_time:.6f} seconds")


if __name__ == "__main__":
    # Run the tests
    test_algorithm_optimization_config()
    print("[PASS] Algorithm Optimization Config test passed")

    test_cache_optimized_array()
    print("[PASS] Cache Optimized Array test passed")

    test_optimized_sort_algorithms()
    print("[PASS] Optimized Sort Algorithms test passed")

    test_optimized_search_algorithms()
    print("[PASS] Optimized Search Algorithms test passed")

    test_cache_optimized_dict()
    print("[PASS] Cache Optimized Dict test passed")

    test_optimized_memoization_cache()
    print("[PASS] Optimized Memoization Cache test passed")

    test_cpu_cache_optimized_memoize()
    print("[PASS] CPU Cache Optimized Memoize test passed")

    test_optimized_data_structures()
    print("[PASS] Optimized Data Structures test passed")

    test_advanced_tokenization_with_algorithm_optimizations()
    print("[PASS] Advanced Tokenization with Algorithm Optimizations test passed")

    test_optimized_preprocessor_with_algorithm_enhancements()
    print("[PASS] Optimized Preprocessor with Algorithm Enhancements test passed")

    test_optimized_inference_pipeline_with_algorithm_enhancements()
    print("[PASS] Optimized Inference Pipeline with Algorithm Enhancements test passed")

    test_apply_algorithm_optimizations()
    print("[PASS] Apply Algorithm Optimizations test passed")

    test_performance_comparison()
    print("[PASS] Performance Comparison test passed")

    print("\nAll algorithm optimization tests passed! :)")