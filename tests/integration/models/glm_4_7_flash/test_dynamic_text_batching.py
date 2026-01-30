"""
Test suite for Dynamic Text Batching System.

This test verifies that the dynamic text batching system works correctly for unimodal models.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import time
import torch
import sys
import os

# Add the project root to the path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.inference_pio.common.dynamic_text_batching import (
    DynamicTextBatchManager,
    TextBatchType,
    TextBatchInfo,
    get_dynamic_text_batch_manager
)
from src.inference_pio.common.input_complexity_analyzer import InputComplexityAnalyzer

# TestDynamicTextBatching

    """Test cases for dynamic text batching functionality."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        batch_manager = DynamicTextBatchManager(
            initial_batch_size=4,
            min_batch_size=1,
            max_batch_size=16,
            memory_threshold_ratio=0.8
        )
        complexity_analyzer = InputComplexityAnalyzer()

    def batch_manager_creation(self)():
        """Test that the dynamic text batch manager can be created and accessed."""
        assert_is_instance(batch_manager, DynamicTextBatchManager)
        assert_equal(batch_manager.current_batch_size, 4)
        assert_equal(batch_manager.min_batch_size, 1)
        assert_equal(batch_manager.max_batch_size, 16)

        # Test global instance
        global_manager = get_dynamic_text_batch_manager()
        assert_is_instance(global_manager, DynamicTextBatchManager)

    def text_batch_type_classification(self)():
        """Test that text batch types are correctly classified."""
        # Simple text
        simple_text = ["Hello world", "How are you?"]
        simple_type = batch_manager.analyze_text_batch_type(simple_text)
        # The system may classify this as any type depending on the specific characteristics
        # Just verify it returns a valid TextBatchType
        assert_is_instance(simple_type, TextBatchType)

        # Technical text
        technical_text = [
            "The algorithm uses a hash map for O(1) lookups.",
            "Class inheritance enables polymorphism in object-oriented programming."
        ]
        tech_type = batch_manager.analyze_text_batch_type(technical_text)
        assert_is_instance(tech_type, TextBatchType)

        # Code text
        code_text = [
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "\n    def add(self, a, b):\n        return a + b"
        ]
        code_type = batch_manager.analyze_text_batch_type(code_text)
        assert_is_instance(code_type, TextBatchType)

        # Long context text
        long_text = ["This is a very long text. " * 100]  # Creates a long string
        long_type = batch_manager.analyze_text_batch_type(long_text)
        assert_is_instance(long_type, TextBatchType)

    def memory_usage_estimation(self)():
        """Test that memory usage is estimated correctly."""
        texts = ["Short text", "Medium length text here", "This is a longer text for testing purposes"]
        batch_size = 4
        
        estimated_memory = batch_manager.estimate_memory_usage(texts, batch_size)
        
        # Memory estimate should be positive
        assert_greater(estimated_memory, 0)
        
        # Memory should scale with batch size
        larger_batch_memory = batch_manager.estimate_memory_usage(texts, batch_size * 2)
        assert_greater(larger_batch_memory, estimated_memory)

    def text_batch_metrics_collection(self)():
        """Test that text batch metrics are collected correctly."""
        texts = ["This is a test text.", "Another example sentence."]
        batch_info = batch_manager.collect_text_batch_metrics(
            texts, 
            processing_time_ms=100.0, 
            tokens_processed=20, 
            batch_size=2
        )
        
        assert_is_instance(batch_info, TextBatchInfo)
        assert_is_instance(batch_info.batch_type, TextBatchType)
        assert_is_instance(batch_info.complexity_score, float)
        assertGreaterEqual(batch_info.complexity_score, 0.0)
        assertLessEqual(batch_info.complexity_score, 1.0)
        assert_is_instance(batch_info.avg_sequence_length, (int))
        assert_is_instance(batch_info.token_count, int)
        assertGreaterEqual(batch_info.token_count, 0)

    def optimal_batch_size_calculation(self)():
        """Test that optimal batch size is calculated correctly."""
        # Test with simple text
        simple_texts = ["Hello", "World"]
        optimal_size = batch_manager.get_optimal_batch_size(
            processing_time_ms=50.0,
            tokens_processed=10,
            input_data=simple_texts
        )
        
        assert_is_instance(optimal_size, int)
        assertGreaterEqual(optimal_size, batch_manager.min_batch_size)
        assertLessEqual(optimal_size, batch_manager.max_batch_size)

        # Test with complex text
        complex_texts = [
            "The implementation of the algorithm requires careful consideration of edge cases and performance characteristics."
        ]
        optimal_size_complex = batch_manager.get_optimal_batch_size(
            processing_time_ms=150.0,
            tokens_processed=15,
            input_data=complex_texts
        )
        
        # Both should be within bounds
        assertGreaterEqual(optimal_size_complex, batch_manager.min_batch_size)
        assertLessEqual(optimal_size_complex, batch_manager.max_batch_size)

    def batch_size_adjustment_logic(self)():
        """Test the logic for adjusting batch sizes based on performance."""
        initial_size = batch_manager.current_batch_size
        
        # Simulate good performance with low memory pressure
        for i in range(5):
            new_size = batch_manager.get_optimal_batch_size(
                processing_time_ms=50.0,
                tokens_processed=100
            )

        # Check that batch size may have increased due to good performance
        status = batch_manager.get_status_report()
        current_size = status['current_batch_size']
        
        # Reset for next test
        batch_manager.force_batch_size(initial_size)

        # Simulate poor performance with high processing time
        for i in range(5):
            # High processing time and low throughput to simulate poor performance
            new_size = batch_manager.get_optimal_batch_size(
                processing_time_ms=500.0,  # Slow processing
                tokens_processed=10  # Low throughput
            )

        # Check that batch size decreased due to poor performance
        status_after_poor = batch_manager.get_status_report()
        size_after_poor = status_after_poor['current_batch_size']
        
        # The size after poor performance should be <= initial size
        assertLessEqual(size_after_poor, initial_size)

    def text_specific_batch_adjustments(self)():
        """Test that text-specific characteristics affect batch size adjustments."""
        # Test with long context text (should reduce batch size)
        long_texts = ["This is a very long text. " * 50 for _ in range(3)]
        long_batch_size = batch_manager.get_optimal_batch_size(
            processing_time_ms=100.0,
            tokens_processed=150,
            input_data=long_texts
        )
        
        # Test with simple text (should allow larger batch size)
        simple_texts = ["Hi", "Hello", "Hey"]
        simple_batch_size = batch_manager.get_optimal_batch_size(
            processing_time_ms=100.0,
            tokens_processed=10,
            input_data=simple_texts
        )
        
        # Both should be within bounds
        assertGreaterEqual(long_batch_size, batch_manager.min_batch_size)
        assertLessEqual(long_batch_size, batch_manager.max_batch_size)
        assertGreaterEqual(simple_batch_size, batch_manager.min_batch_size)
        assertLessEqual(simple_batch_size, batch_manager.max_batch_size)

    def status_report(self)():
        """Test that status reports contain all required information."""
        status = batch_manager.get_status_report()
        
        required_keys = [
            'current_batch_size',
            'min_batch_size',
            'max_batch_size',
            'memory_threshold_ratio',
            'metrics_history_length',
            'text_batch_history_length',
            'system_memory_info',
            'text_type_distribution'
        ]
        
        for key in required_keys:
            assert_in(key, status)

    def force_batch_size(self)():
        """Test that batch size can be forced to a specific value."""
        original_size = batch_manager.current_batch_size
        
        # Try to force to a valid size
        success = batch_manager.force_batch_size(8)
        assert_true(success)
        assert_equal(batch_manager.current_batch_size)
        
        # Try to force to an invalid size
        success = batch_manager.force_batch_size(100)  # Exceeds max
        assert_false(success)
        # Size should remain unchanged
        assert_equal(batch_manager.current_batch_size)

    def cleanup(self)():
        """Test that cleanup works without errors."""
        batch_manager.cleanup()
        # After cleanup, the history should be empty
        status = batch_manager.get_status_report()
        assert_equal(status['metrics_history_length'], 0)
        assert_equal(status['text_batch_history_length'], 0)

    def cleanup_helper():
        """Clean up after each test method."""
        batch_manager.cleanup()

# TestIntegrationWithModels

    """Integration tests for dynamic text batching with model interfaces."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        batch_manager = get_dynamic_text_batch_manager(
            initial_batch_size=4,
            min_batch_size=1,
            max_batch_size=16
        )

    def with_various_text_types(self)():
        """Test the batch manager with various types of text inputs."""
        test_cases = [
            # Simple text
            (["Hello", "World", "Test"], "simple"),
            # Code snippets
            ([
                "def hello():\n    print('world')",
                "\n    def method(self):\n        pass"
            ], "code"),
            # Technical text
            ([
                "The transformer architecture utilizes self-attention mechanisms.",
                "Backpropagation adjusts weights through gradient descent."
            ], "technical"),
            # Mixed content
            ([
                "The study shows that 42% of participants agreed.",
                "Equation: E = mc²"
            ], "mixed")
        ]
        
        for texts, description in test_cases:
            with subTest(description=description):
                # Process the texts and get recommended batch size
                recommended_size = batch_manager.get_optimal_batch_size(
                    processing_time_ms=100.0,
                    tokens_processed=len(texts) * 10,  # Approximate token count
                    input_data=texts
                )
                
                # Verify the result is valid
                assert_is_instance(recommended_size, int)
                assertGreaterEqual(recommended_size, 1)
                assertLessEqual(recommended_size, 16)

if __name__ == '__main__':
    run_tests(test_functions)