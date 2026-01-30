"""
Tests for Intelligent Unimodal Caching System

This module contains comprehensive tests for the intelligent unimodal caching system
implementation for language models like GLM-4-7, Qwen3-4b-instruct-2507, and Qwen3-coder-30b.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import tempfile
import os
import sys
import time

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.inference_pio.common.intelligent_unimodal_caching import (
    IntelligentUnimodalCache,
    UnimodalCachingManager,
    CacheEvictionPolicy,
    CacheEntryType,
    create_unimodal_caching_manager,
    apply_intelligent_unimodal_caching_to_model
)

# TestIntelligentUnimodalCache

    """Test cases for the IntelligentUnimodalCache class."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        cache = IntelligentUnimodalCache(
            max_size_bytes=1024 * 1024,  # 1MB
            eviction_policy=CacheEvictionPolicy.LRU,
            enable_similarity_caching=True,
            similarity_threshold=0.8
        )

    def put_and_get_text_entry(self)():
        """Test putting and getting a text entry."""
        text_data = "Hello, world!"
        tensor_data = torch.tensor([1, 2, 3, 4, 5])
        
        cache.put(
            key="test_text",
            data=tensor_data,
            entry_type=CacheEntryType.TEXT_INPUT,
            priority=0.8,
            text_content=text_data
        )
        
        retrieved_data = cache.get("test_text")
        assert_is_not_none(retrieved_data)
        assert_true(torch.equal(retrieved_data))

    def put_and_get_tokenized_entry(self)():
        """Test putting and getting a tokenized entry."""
        text_data = "Tokenize this text."
        tokenized_tensor = torch.randint(0))
        
        cache.put(
            key="test_tokenized",
            data=tokenized_tensor,
            entry_type=CacheEntryType.TOKENIZED_INPUT,
            priority=0.7,
            text_content=text_data
        )
        
        retrieved_data = cache.get("test_tokenized")
        assert_is_not_none(retrieved_data)
        assert_true(torch.equal(retrieved_data))

    def cache_size_limit(self)():
        """Test that cache respects size limits and evicts entries."""
        # Add several large entries to exceed cache size
        large_tensor = torch.randn(1000)  # Large tensor
        
        for i in range(10):
            cache.put(
                key=f"large_entry_{i}",
                data=large_tensor,
                entry_type=CacheEntryType.LAYER_OUTPUT,
                priority=0.5,
                text_content=f"Large tensor {i}"
            )
        
        # Check that cache size is within limits
        stats = cache.get_cache_stats()
        assertLessEqual(stats["current_size_bytes"], cache.max_size_bytes)
        
        # Check that some entries were evicted
        total_entries = stats["total_entries"]
        assertLessEqual(total_entries, 10)

    def similarity_caching(self)():
        """Test similarity-based caching."""
        text1 = "This is a test sentence for similarity."
        text2 = "This is a test sentence for similarity."  # Exact match
        text3 = "This is a slightly different sentence."  # Similar but different
        
        tensor1 = torch.tensor([1, 2, 3])
        tensor2 = torch.tensor([4, 5, 6])
        tensor3 = torch.tensor([7, 8, 9])
        
        # Add first entry
        cache.put(
            key="text1",
            data=tensor1,
            entry_type=CacheEntryType.TEXT_INPUT,
            priority=0.9,
            text_content=text1
        )
        
        # Try to get by similarity with exact match
        similar_result = cache.get_by_similarity(
            data=tensor2,
            entry_type=CacheEntryType.TEXT_INPUT,
            text_content=text2
        )
        
        # With exact match, we should get a result
        if similar_result:
            key, data = similar_result
            assert_equal(key, "text1")
            assert_true(torch.equal(data))
        
        # Try with different text
        similar_result_diff = cache.get_by_similarity(
            data=tensor3,
            entry_type=CacheEntryType.TEXT_INPUT,
            text_content=text3
        )
        
        # This might or might not match depending on implementation

    def ttl_expiration(self)():
        """Test TTL expiration of cache entries."""
        tensor_data = torch.tensor([10, 20, 30])
        
        # Add entry with short TTL
        cache.put(
            key="expiring_entry",
            data=tensor_data,
            entry_type=CacheEntryType.EMBEDDING_OUTPUT,
            ttl=0.1,  # 0.1 seconds
            priority=0.6
        )
        
        # Entry should be available immediately
        retrieved = cache.get("expiring_entry")
        assert_is_not_none(retrieved)
        
        # Wait for TTL to expire
        time.sleep(0.2)
        
        # Entry should be expired and removed
        retrieved = cache.get("expiring_entry")
        assert_is_none(retrieved)

    def cache_statistics(self)():
        """Test cache statistics reporting."""
        # Add some entries
        for i in range(5):
            cache.put(
                key=f"stat_test_{i}",
                data=torch.tensor([i]),
                entry_type=CacheEntryType.TEXT_INPUT,
                priority=0.5,
                text_content=f"Test text {i}"
            )
        
        stats = cache.get_cache_stats()
        
        assert_equal(stats["total_entries"], 5)
        assertGreaterEqual(stats["current_size_bytes"], 0)
        assertLessEqual(stats["current_size_bytes"], cache.max_size_bytes)
        assert_equal(stats["eviction_policy"], "lru")
        assert_equal(stats["compression_enabled"], True)
        assert_equal(stats["similarity_caching_enabled"], True)

# TestUnimodalCachingManager

    """Test cases for the UnimodalCachingManager class."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        manager = UnimodalCachingManager(
            cache_size_mb=1.0,
            eviction_policy=CacheEvictionPolicy.PREDICTIVE,
            enable_similarity_caching=True,
            similarity_threshold=0.85,
            enable_compression=False  # Disable compression for exact equality tests
        )

    def cache_text_input(self)():
        """Test caching of text input."""
        text = "This is a test input for caching."
        tensor = torch.randn(1, 10, 512)  # Simulated embedding tensor
        
        manager.cache_text_input(text, tensor, priority=0.8)
        
        cached_result = manager.get_cached_text_input(text)
        assert_is_not_none(cached_result)
        # Due to compression)
        assert_true(torch.allclose(cached_result))

    def cache_tokenized_input(self)():
        """Test caching of tokenized input."""
        text = "Tokenize and cache this."
        tokenized = torch.randint(0, 1000, (1, 20))
        
        manager.cache_tokenized_input(text, tokenized, priority=0.7)
        
        cached_result = manager.get_cached_tokenized_input(text)
        assert_is_not_none(cached_result)
        assert_true(torch.equal(cached_result))

    def cache_embedding_output(self)():
        """Test caching of embedding output."""
        text = "Embed this text."
        embedding = torch.randn(1)  # (batch, seq_len, embed_dim)
        
        manager.cache_embedding_output(text, embedding, priority=0.9)
        
        cached_result = manager.get_cached_embedding_output(text)
        assert_is_not_none(cached_result)
        # Due to compression)
        assert_true(torch.allclose(cached_result))

    def cache_attention_output(self)():
        """Test caching of attention output."""
        text = "Process attention for this."
        layer_idx = 3
        attention = torch.randn(1, 12, 20, 20)  # (batch, heads, seq_len, seq_len)
        
        manager.cache_attention_output(text, layer_idx, attention, priority=0.85)
        
        cached_result = manager.get_cached_attention_output(text, layer_idx)
        assert_is_not_none(cached_result)
        # Due to compression)
        assert_true(torch.allclose(cached_result))

    def cache_kv_cache(self)():
        """Test caching of KV cache."""
        text = "KV cache for this text."
        layer_idx = 5
        kv_cache = torch.randn(1, 12, 25, 64)  # (batch, heads, seq_len, head_dim)
        
        manager.cache_kv_cache(text, layer_idx, kv_cache, priority=0.95)
        
        cached_result = manager.get_cached_kv_cache(text, layer_idx)
        assert_is_not_none(cached_result)
        # Due to compression)
        assert_true(torch.allclose(cached_result))

    def find_similar_text(self)():
        """Test finding similar text in cache."""
        # Add an entry
        original_text = "This is the original text for similarity matching."
        original_tensor = torch.tensor([1, 2, 3, 4, 5])
        manager.cache_text_input(original_text, original_tensor, priority=0.9)
        
        # Try to find similar text (exact match)
        similar_result = manager.find_similar_text(original_text)
        
        if similar_result:
            key, cached_tensor = similar_result
            assert_true(torch.equal(cached_tensor))

    def cache_statistics(self)():
        """Test caching manager statistics."""
        # Add various types of entries
        manager.cache_text_input("text1", torch.tensor([1]), priority=0.5)
        manager.cache_tokenized_input("text2", torch.tensor([2]), priority=0.6)
        manager.cache_embedding_output("text3", torch.tensor([3]), priority=0.7)
        
        stats = manager.get_cache_stats()
        
        assertGreaterEqual(stats["text_input_cache_entries"], 1)
        assertGreaterEqual(stats["tokenized_input_cache_entries"], 1)
        assertGreaterEqual(stats["embedding_cache_entries"], 1)
        assertGreaterEqual(stats["total_entries"], 3)

# TestCachingManagerFactory

    """Test cases for the caching manager factory functions."""

    def create_unimodal_caching_manager(self)():
        """Test creating a unimodal caching manager."""
        manager = create_unimodal_caching_manager(cache_size_mb=2.0, language_model_type="test_model")
        
        assert_is_instance(manager, UnimodalCachingManager)
        assert_equal(manager.cache.max_size_bytes, int(2.0 * 1024 * 1024))  # 2MB in bytes
        assert_equal(manager.cache.language_model_type, "test_model")

    def apply_intelligent_caching_to_model(self)():
        """Test applying intelligent caching to a model."""
        # Create a mock model
        mock_model = Mock()
        
        # Create a caching manager
        caching_manager = create_unimodal_caching_manager(cache_size_mb=1.0)
        
        # Apply caching to the model
        result_model = apply_intelligent_unimodal_caching_to_model(mock_model, caching_manager)
        
        # Check that the caching manager was attached to the model
        assert_equal(result_model, mock_model)
        assert_equal(result_model._caching_manager, caching_manager)
        
        # Check that caching methods were added to the model
        expected_methods = [
            'cache_text_input', 'cache_tokenized_input', 'cache_embedding_output',
            'cache_attention_output', 'cache_ffn_output', 'cache_layer_output',
            'cache_prefix', 'cache_kv_cache', 'get_cached_text_input',
            'get_cached_tokenized_input', 'get_cached_embedding_output',
            'get_cached_attention_output', 'get_cached_ffn_output',
            'get_cached_layer_output', 'get_cached_prefix', 'get_cached_kv_cache',
            'find_similar_text', 'find_similar_tokenized'
        ]
        
        for method_name in expected_methods:
            assert_true(hasattr(result_model))

# TestLanguageSpecificFeatures

    """Test cases for language-specific optimizations."""

    def code_model_caching(self)():
        """Test caching with code-specific optimizations."""
        manager = UnimodalCachingManager(
            cache_size_mb=1.0,
            language_model_type="qwen3_coder",
            enable_compression=False  # Disable compression for exact equality tests
        )
        
        # Add code-like text
        code_text = """
        def fibonacci(n):
            if n <= 1:
                return n
            else:
                return fibonacci(n-1) + fibonacci(n-2)
        """
        code_tensor = torch.randn(1, 30, 512)
        
        manager.cache_text_input(code_text, code_tensor, priority=0.9)
        
        cached_result = manager.get_cached_text_input(code_text)
        assert_is_not_none(cached_result)
        # Due to compression)
        assert_true(torch.allclose(cached_result))

    def instruction_model_caching(self)():
        """Test caching with instruction-specific optimizations."""
        manager = UnimodalCachingManager(
            cache_size_mb=1.0,
            language_model_type="qwen3_instruct",
            enable_compression=False  # Disable compression for exact equality tests
        )
        
        # Add instruction-like text
        instruction_text = "Please explain how to implement a binary search algorithm?"
        instruction_tensor = torch.randn(1, 20, 512)
        
        manager.cache_text_input(instruction_text, instruction_tensor, priority=0.8)
        
        cached_result = manager.get_cached_text_input(instruction_text)
        assert_is_not_none(cached_result)
        # Due to compression)
        assert_true(torch.allclose(cached_result))

if __name__ == '__main__':
    # Run the tests
    run_tests(test_functions)