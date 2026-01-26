"""
Input Complexity Analyzer for Adaptive Batching

This module analyzes input complexity to determine optimal batch sizes for different types of inputs.
Complex inputs (long sequences, complex patterns) receive smaller batch sizes, while simple inputs
receive larger batch sizes to maximize throughput.
"""

import logging
import re
from typing import Union, List, Dict, Any
import numpy as np
from dataclasses import dataclass
import torch


logger = logging.getLogger(__name__)


@dataclass
class ComplexityMetrics:
    """Metrics representing the complexity of input data."""
    sequence_length: int
    vocabulary_richness: float  # Ratio of unique tokens to total tokens
    syntactic_complexity: float  # Measure of structural complexity
    semantic_density: float  # Information density measure
    numerical_content_ratio: float  # Ratio of numbers to total tokens
    special_character_ratio: float  # Ratio of special chars to total tokens
    complexity_score: float  # Overall complexity score (0.0 to 1.0)


class InputComplexityAnalyzer:
    """
    Analyzes input complexity to determine optimal batch sizes.
    
    The analyzer considers multiple factors:
    - Sequence length
    - Vocabulary richness
    - Syntactic complexity
    - Semantic density
    - Numerical content
    - Special character usage
    """
    
    def __init__(self):
        self.token_pattern = re.compile(r'\b\w+\b')
        self.number_pattern = re.compile(r'\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')
        self.special_char_pattern = re.compile(r'[^\w\s]')
        
    def analyze_input_complexity(self, 
                                input_data: Union[str, List[str], torch.Tensor, List[torch.Tensor]]) -> ComplexityMetrics:
        """
        Analyze the complexity of input data.
        
        Args:
            input_data: Input data to analyze (text string, list of strings, or tensors)
            
        Returns:
            ComplexityMetrics object with calculated complexity scores
        """
        if isinstance(input_data, str):
            return self._analyze_text_complexity(input_data)
        elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            # Average complexity across multiple text inputs
            complexities = [self._analyze_text_complexity(text) for text in input_data]
            return self._aggregate_complexities(complexities)
        elif isinstance(input_data, torch.Tensor):
            return self._analyze_tensor_complexity(input_data)
        elif isinstance(input_data, list) and all(isinstance(item, torch.Tensor) for item in input_data):
            # Average complexity across multiple tensor inputs
            complexities = [self._analyze_tensor_complexity(tensor) for tensor in input_data]
            return self._aggregate_complexities([self._tensor_to_complexity_metrics(c) for c in complexities])
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _analyze_text_complexity(self, text: str) -> ComplexityMetrics:
        """Analyze complexity of a text string."""
        tokens = self.token_pattern.findall(text.lower())
        sequence_length = len(text)
        token_count = len(tokens)
        
        # Vocabulary richness: ratio of unique tokens to total tokens
        unique_tokens = set(tokens) if tokens else set()
        vocabulary_richness = len(unique_tokens) / token_count if token_count > 0 else 0.0
        
        # Syntactic complexity: based on sentence structure
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        avg_sentence_length = np.mean([len(self.token_pattern.findall(s)) for s in sentences]) if sentences else 0
        clause_separators = len(re.findall(r',|;|:', text))
        syntactic_complexity = min(1.0, (avg_sentence_length * 0.1 + clause_separators * 0.05) / 10.0)
        
        # Semantic density: based on information content
        word_chars = sum(len(token) for token in tokens)
        total_chars = len(text)
        semantic_density = word_chars / total_chars if total_chars > 0 else 0.0
        
        # Numerical content ratio
        numbers = self.number_pattern.findall(text)
        numerical_content_ratio = len(numbers) / token_count if token_count > 0 else 0.0
        
        # Special character ratio
        special_chars = len(self.special_char_pattern.findall(text))
        special_character_ratio = special_chars / len(text) if len(text) > 0 else 0.0
        
        # Calculate overall complexity score
        complexity_score = self._calculate_overall_complexity_score(
            sequence_length,
            vocabulary_richness,
            syntactic_complexity,
            semantic_density,
            numerical_content_ratio,
            special_character_ratio
        )
        
        return ComplexityMetrics(
            sequence_length=sequence_length,
            vocabulary_richness=vocabulary_richness,
            syntactic_complexity=syntactic_complexity,
            semantic_density=semantic_density,
            numerical_content_ratio=numerical_content_ratio,
            special_character_ratio=special_character_ratio,
            complexity_score=complexity_score
        )
    
    def _analyze_tensor_complexity(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Analyze complexity of a tensor input."""
        # For tensors, we'll analyze statistical properties
        tensor_flat = tensor.flatten().cpu().numpy()
        
        # Basic statistics
        sequence_length = tensor.numel()
        mean_val = float(np.mean(tensor_flat))
        std_val = float(np.std(tensor_flat))
        variance = float(np.var(tensor_flat))
        
        # Complexity based on distribution characteristics
        # Higher variance indicates more complex patterns
        complexity_from_variance = min(1.0, variance / 10.0)  # Normalize variance impact
        
        # Count of unique values vs total values
        unique_count = len(np.unique(tensor_flat))
        vocabulary_richness = unique_count / sequence_length if sequence_length > 0 else 0.0
        
        # Numerical complexity (how much variation there is)
        numerical_complexity = min(1.0, (std_val / (abs(mean_val) + 1e-8)) if mean_val != 0 else std_val)
        
        return {
            'sequence_length': sequence_length,
            'mean': mean_val,
            'std': std_val,
            'variance': variance,
            'unique_ratio': vocabulary_richness,
            'numerical_complexity': numerical_complexity,
            'complexity_from_variance': complexity_from_variance
        }
    
    def _tensor_to_complexity_metrics(self, tensor_analysis: Dict[str, Any]) -> ComplexityMetrics:
        """Convert tensor analysis to ComplexityMetrics format."""
        complexity_score = (
            0.3 * min(1.0, tensor_analysis['sequence_length'] / 10000) +  # Length factor
            0.2 * tensor_analysis['unique_ratio'] +  # Uniqueness factor
            0.2 * tensor_analysis['complexity_from_variance'] +  # Variance factor
            0.15 * tensor_analysis['numerical_complexity'] +  # Numerical complexity
            0.15 * min(1.0, tensor_analysis['std'] / 5.0)  # Standard deviation factor
        ) / 1.0  # Normalize
        
        return ComplexityMetrics(
            sequence_length=tensor_analysis['sequence_length'],
            vocabulary_richness=tensor_analysis['unique_ratio'],
            syntactic_complexity=tensor_analysis['complexity_from_variance'],
            semantic_density=tensor_analysis['numerical_complexity'],
            numerical_content_ratio=tensor_analysis['numerical_complexity'],
            special_character_ratio=min(1.0, tensor_analysis['variance'] / 10.0),
            complexity_score=min(1.0, complexity_score)
        )
    
    def _aggregate_complexities(self, complexities: List[ComplexityMetrics]) -> ComplexityMetrics:
        """Aggregate multiple complexity metrics into a single metric."""
        if not complexities:
            return ComplexityMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Calculate averages
        avg_seq_len = int(sum(c.sequence_length for c in complexities) / len(complexities))
        avg_vocab_richness = sum(c.vocabulary_richness for c in complexities) / len(complexities)
        avg_syntactic_complexity = sum(c.syntactic_complexity for c in complexities) / len(complexities)
        avg_semantic_density = sum(c.semantic_density for c in complexities) / len(complexities)
        avg_numerical_ratio = sum(c.numerical_content_ratio for c in complexities) / len(complexities)
        avg_special_ratio = sum(c.special_character_ratio for c in complexities) / len(complexities)
        avg_complexity_score = sum(c.complexity_score for c in complexities) / len(complexities)
        
        return ComplexityMetrics(
            sequence_length=avg_seq_len,
            vocabulary_richness=avg_vocab_richness,
            syntactic_complexity=avg_syntactic_complexity,
            semantic_density=avg_semantic_density,
            numerical_content_ratio=avg_numerical_ratio,
            special_character_ratio=avg_special_ratio,
            complexity_score=avg_complexity_score
        )
    
    def _calculate_overall_complexity_score(self,
                                          sequence_length: int,
                                          vocabulary_richness: float,
                                          syntactic_complexity: float,
                                          semantic_density: float,
                                          numerical_content_ratio: float,
                                          special_character_ratio: float) -> float:
        """
        Calculate an overall complexity score based on all metrics.
        
        Returns a value between 0.0 (simple) and 1.0 (complex).
        """
        # Weighted combination of all factors
        # Longer sequences are more complex
        length_factor = min(1.0, sequence_length / 10000)  # Normalize to 0-1 range
        
        # Higher vocabulary richness indicates more complex input
        vocab_factor = vocabulary_richness
        
        # Syntactic complexity contributes directly
        syntax_factor = syntactic_complexity
        
        # Semantic density (information per character) contributes
        semantic_factor = semantic_density
        
        # Numerical content can add complexity
        numerical_factor = min(1.0, numerical_content_ratio * 2)
        
        # Special characters can indicate complexity
        special_factor = min(1.0, special_character_ratio * 3)
        
        # Weighted average with emphasis on length and vocabulary richness
        complexity_score = (
            0.3 * length_factor +
            0.25 * vocab_factor +
            0.15 * syntax_factor +
            0.15 * semantic_factor +
            0.075 * numerical_factor +
            0.075 * special_factor
        )
        
        return min(1.0, complexity_score)  # Clamp to 0-1 range
    
    def get_adaptive_batch_size(self,
                               complexity_score: float,
                               base_min_batch: int = 1,
                               base_max_batch: int = 16,
                               complexity_threshold_low: float = 0.3,
                               complexity_threshold_high: float = 0.7) -> int:
        """
        Determine an appropriate batch size based on complexity score.
        
        Args:
            complexity_score: Complexity score between 0.0 and 1.0
            base_min_batch: Base minimum batch size
            base_max_batch: Base maximum batch size
            complexity_threshold_low: Below this, use max batch size
            complexity_threshold_high: Above this, use min batch size
            
        Returns:
            Appropriate batch size for the given complexity
        """
        if complexity_score <= complexity_threshold_low:
            # Simple input - use larger batch size
            return base_max_batch
        elif complexity_score >= complexity_threshold_high:
            # Complex input - use smaller batch size
            return base_min_batch
        else:
            # Moderate complexity - interpolate between min and max
            # Use inverse relationship: higher complexity = lower batch size
            complexity_range = complexity_threshold_high - complexity_threshold_low
            normalized_complexity = (complexity_score - complexity_threshold_low) / complexity_range
            batch_range = base_max_batch - base_min_batch
            
            # Calculate batch size with inverse relationship to complexity
            calculated_batch_size = int(base_max_batch - (normalized_complexity * batch_range))
            
            # Ensure it stays within bounds
            return max(base_min_batch, min(calculated_batch_size, base_max_batch))


def get_complexity_analyzer() -> InputComplexityAnalyzer:
    """
    Get a global instance of the input complexity analyzer.
    
    Returns:
        InputComplexityAnalyzer instance
    """
    if not hasattr(get_complexity_analyzer, '_instance'):
        get_complexity_analyzer._instance = InputComplexityAnalyzer()
    return get_complexity_analyzer._instance


__all__ = [
    "InputComplexityAnalyzer",
    "ComplexityMetrics",
    "get_complexity_analyzer"
]