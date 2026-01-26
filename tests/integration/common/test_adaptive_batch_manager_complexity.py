"""
Tests for Adaptive Batch Manager with Input Complexity Analysis

This module tests the integration of input complexity analysis with adaptive batching.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import time
from src.inference_pio.common.adaptive_batch_manager import AdaptiveBatchManager, get_adaptive_batch_manager
from src.inference_pio.common.input_complexity_analyzer import InputComplexityAnalyzer, get_complexity_analyzer

# TestAdaptiveBatchManagerWithComplexity

    """Test cases for the AdaptiveBatchManager with input complexity analysis."""

    def setup_helper():
        """Set up test fixtures."""
        batch_manager = AdaptiveBatchManager(
            initial_batch_size=4,
            min_batch_size=1,
            max_batch_size=8,
            use_input_complexity=True,
            complexity_weight=0.5
        )
        complexity_analyzer = InputComplexityAnalyzer()

    def simple_text_gets_larger_batch_size(self)():
        """Test that simple text gets a larger batch size."""
        simple_text = "Hello world."
        complexity_metrics = complexity_analyzer.analyze_input_complexity(simple_text)
        complexity_score = complexity_metrics.complexity_score
        
        recommended_batch = complexity_analyzer.get_adaptive_batch_size(
            complexity_score=complexity_score,
            base_min_batch=1,
            base_max_batch=8
        )
        
        # Simple text should get max or near-max batch size
        assertGreaterEqual(recommended_batch, 6)  # At least 6 for simple text

    def complex_text_gets_smaller_batch_size(self)():
        """Test that complex text gets a smaller batch size."""
        complex_text = ("The quick brown fox jumps over the lazy dog. "
                       "This sentence contains multiple clauses, various punctuation marks, "
                       "numbers like 123, 456.789, and special characters such as @#$%. "
                       "Additionally, it includes complex syntactic structures.")
        complexity_metrics = complexity_analyzer.analyze_input_complexity(complex_text)
        complexity_score = complexity_metrics.complexity_score
        
        recommended_batch = complexity_analyzer.get_adaptive_batch_size(
            complexity_score=complexity_score,
            base_min_batch=1,
            base_max_batch=8
        )
        
        # Complex text should get smaller batch size
        assertLessEqual(recommended_batch, 4)  # At most 4 for complex text

    def collect_metrics_with_input_complexity(self)():
        """Test collecting metrics with input complexity."""
        simple_text = "Hello world."
        
        # Collect metrics with input complexity
        metrics = batch_manager.collect_metrics(
            batch_size=4,
            processing_time_ms=100,
            tokens_processed=10,
            input_data=simple_text
        )
        
        # Verify metrics were collected
        assert_is_not_none(metrics)
        assert_equal(metrics.batch_size)

    def get_optimal_batch_size_with_complexity(self)():
        """Test getting optimal batch size considering input complexity."""
        simple_text = "Hello world."
        
        # Get optimal batch size with complexity consideration
        optimal_batch = batch_manager.get_optimal_batch_size(
            processing_time_ms=100,
            tokens_processed=10,
            input_data=simple_text
        )
        
        # Should return a reasonable batch size
        assertGreaterEqual(optimal_batch, 1)
        assertLessEqual(optimal_batch, 8)

    def get_adaptive_batch_manager_singleton(self)():
        """Test that get_adaptive_batch_manager returns a singleton instance."""
        manager1 = get_adaptive_batch_manager()
        manager2 = get_adaptive_batch_manager()
        
        assertIs(manager1, manager2)

    def get_complexity_analyzer_singleton(self)():
        """Test that get_complexity_analyzer returns a singleton instance."""
        analyzer1 = get_complexity_analyzer()
        analyzer2 = get_complexity_analyzer()
        
        assertIs(analyzer1, analyzer2)

    def batch_size_adjustment_reasons(self)():
        """Test different reasons for batch size adjustment."""
        # Initially, no adjustment should be needed
        should_adjust, reason = batch_manager.should_adjust_batch_size()
        assert_false(should_adjust)
        
        # Add some metrics to trigger adjustment logic
        batch_manager.collect_metrics(
            batch_size=4,
            processing_time_ms=100,
            tokens_processed=10,
            input_data="test input"
        )
        
        # Still shouldn't need adjustment due to cooldown period
        should_adjust, reason = batch_manager.should_adjust_batch_size()
        assert_false(should_adjust)

# TestInputComplexityAnalyzer

    """Test cases for the InputComplexityAnalyzer class."""

    def setup_helper():
        """Set up test fixtures."""
        analyzer = InputComplexityAnalyzer()

    def analyze_text_complexity(self)():
        """Test analyzing complexity of text input."""
        text = "This is a sample text for complexity analysis."
        metrics = analyzer.analyze_input_complexity(text)
        
        assert_is_not_none(metrics)
        assertGreaterEqual(metrics.sequence_length)
        assertLessEqual(metrics.complexity_score)
        assertGreaterEqual(metrics.complexity_score, 0.0)

    def analyze_list_of_texts_complexity(self)():
        """Test analyzing complexity of a list of texts."""
        texts = ["Simple text.", "More complex text with additional words and structure."]
        metrics = analyzer.analyze_input_complexity(texts)
        
        assert_is_not_none(metrics)
        assertGreaterEqual(metrics.complexity_score)
        assertLessEqual(metrics.complexity_score, 1.0)

    def analyze_tensor_complexity(self)():
        """Test analyzing complexity of tensor input."""
        tensor = torch.randn(10, 20)
        metrics = analyzer.analyze_input_complexity(tensor)
        
        assert_is_not_none(metrics)
        assertGreaterEqual(metrics.sequence_length)

    def get_adaptive_batch_size(self)():
        """Test getting adaptive batch size based on complexity."""
        base_min = 1
        base_max = 8
        
        # Very simple input should get max batch size
        simple_score = 0.1
        simple_batch = analyzer.get_adaptive_batch_size(
            simple_score, base_min, base_max
        )
        assert_equal(simple_batch, base_max)
        
        # Very complex input should get min batch size
        complex_score = 0.9
        complex_batch = analyzer.get_adaptive_batch_size(
            complex_score, base_min, base_max
        )
        assert_equal(complex_batch, base_min)
        
        # Medium complexity should get intermediate batch size
        medium_score = 0.5
        medium_batch = analyzer.get_adaptive_batch_size(
            medium_score, base_min, base_max
        )
        assertGreaterEqual(medium_batch, base_min)
        assertLessEqual(medium_batch, base_max)

if __name__ == '__main__':
    run_tests(test_functions)