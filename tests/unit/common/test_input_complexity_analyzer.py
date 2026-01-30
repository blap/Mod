"""
Tests for Input Complexity Analyzer

This module tests the input complexity analyzer functionality.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import numpy as np
from src.inference_pio.common.input_complexity_analyzer import (
    InputComplexityAnalyzer, 
    ComplexityMetrics,
    get_complexity_analyzer
)

# TestInputComplexityAnalyzer

    """Test cases for the InputComplexityAnalyzer class."""

    def setup_helper():
        """Set up test fixtures."""
        analyzer = InputComplexityAnalyzer()

    def simple_text_complexity(self)():
        """Test complexity analysis for simple text."""
        simple_text = "Hello world."
        metrics = analyzer.analyze_input_complexity(simple_text)
        
        assert_is_instance(metrics, ComplexityMetrics)
        assertGreaterEqual(metrics.sequence_length, 0)
        assertLessEqual(metrics.complexity_score, 1.0)
        assertGreaterEqual(metrics.complexity_score, 0.0)
        # Simple text should have relatively low complexity
        assert_less(metrics.complexity_score, 0.5)

    def complex_text_complexity(self)():
        """Test complexity analysis for complex text."""
        complex_text = ("The quick brown fox jumps over the lazy dog. "
                       "This sentence contains multiple clauses, various punctuation marks, "
                       "numbers like 123, 456.789, and special characters such as @#$%. "
                       "Additionally, it includes complex syntactic structures.")
        metrics = analyzer.analyze_input_complexity(complex_text)
        
        assert_is_instance(metrics, ComplexityMetrics)
        assertGreaterEqual(metrics.sequence_length, 0)
        assertLessEqual(metrics.complexity_score, 1.0)
        assertGreaterEqual(metrics.complexity_score, 0.0)
        # Complex text should have higher complexity
        assert_greater(metrics.complexity_score, 0.3)

    def list_of_texts_complexity(self)():
        """Test complexity analysis for a list of texts."""
        texts = ["Simple text.", "More complex text with additional words and structure."]
        metrics = analyzer.analyze_input_complexity(texts)
        
        assert_is_instance(metrics, ComplexityMetrics)
        # Should be an average of the individual complexities

    def tensor_complexity(self)():
        """Test complexity analysis for tensor input."""
        tensor = torch.randn(10, 20)  # Random tensor
        analysis = analyzer._analyze_tensor_complexity(tensor)
        
        assert_in('sequence_length', analysis)
        assert_in('mean', analysis)
        assert_in('std', analysis)
        assertGreaterEqual(analysis['sequence_length'], 0)

    def batch_size_determination(self)():
        """Test adaptive batch size determination based on complexity."""
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

    def get_complexity_analyzer_singleton(self)():
        """Test that get_complexity_analyzer returns a singleton instance."""
        analyzer1 = get_complexity_analyzer()
        analyzer2 = get_complexity_analyzer()
        
        assertIs(analyzer1, analyzer2)

if __name__ == '__main__':
    run_tests(test_functions)