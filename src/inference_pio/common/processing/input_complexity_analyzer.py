"""
Input Complexity Analyzer for Adaptive Batching
Dependency-Free
"""

import logging
import re
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Removed numpy and torch imports
from ...core.engine.backend import Tensor

logger = logging.getLogger(__name__)

@dataclass
class ComplexityMetrics:
    sequence_length: int
    vocabulary_richness: float
    syntactic_complexity: float
    semantic_density: float
    numerical_content_ratio: float
    special_character_ratio: float
    complexity_score: float

class InputComplexityAnalyzer:
    def __init__(self):
        self.token_pattern = re.compile(r"\b\w+\b")
        self.number_pattern = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
        self.special_char_pattern = re.compile(r"[^\w\s]")

    def analyze_input_complexity(self, input_data: Union[str, List[str], Tensor, List[Tensor]]) -> ComplexityMetrics:
        if isinstance(input_data, str):
            return self._analyze_text_complexity(input_data)
        elif isinstance(input_data, list) and all(isinstance(i, str) for i in input_data):
            return self._aggregate_complexities([self._analyze_text_complexity(t) for t in input_data])
        elif isinstance(input_data, Tensor):
            return self._tensor_to_complexity_metrics(self._analyze_tensor_complexity(input_data))
        elif isinstance(input_data, list) and all(isinstance(i, Tensor) for i in input_data):
            return self._aggregate_complexities([self._tensor_to_complexity_metrics(self._analyze_tensor_complexity(t)) for t in input_data])
        return ComplexityMetrics(0,0,0,0,0,0,0)

    def _analyze_text_complexity(self, text: str) -> ComplexityMetrics:
        tokens = self.token_pattern.findall(text.lower())
        seq_len = len(text)
        token_count = len(tokens)
        vocab_richness = len(set(tokens)) / token_count if token_count > 0 else 0.0

        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        avg_sent_len = sum(len(self.token_pattern.findall(s)) for s in sentences) / len(sentences) if sentences else 0
        syntax = min(1.0, (avg_sent_len * 0.1) / 10.0)

        nums = len(self.number_pattern.findall(text))
        num_ratio = nums / token_count if token_count > 0 else 0.0

        specials = len(self.special_char_pattern.findall(text))
        spec_ratio = specials / seq_len if seq_len > 0 else 0.0

        score = (0.3 * min(1.0, seq_len/10000) + 0.25 * vocab_richness + 0.15 * syntax + 0.15 * num_ratio + 0.15 * spec_ratio)

        return ComplexityMetrics(seq_len, vocab_richness, syntax, 0.0, num_ratio, spec_ratio, min(1.0, score))

    def _analyze_tensor_complexity(self, tensor: Tensor) -> Dict[str, Any]:
        data = tensor.to_list()
        seq_len = len(data)
        if seq_len == 0: return {"sequence_length":0, "mean":0, "std":0, "variance":0, "unique_ratio":0}

        mean = sum(data) / seq_len
        variance = sum((x - mean) ** 2 for x in data) / seq_len
        std = math.sqrt(variance)
        unique = len(set(data))

        return {
            "sequence_length": seq_len,
            "mean": mean,
            "std": std,
            "variance": variance,
            "unique_ratio": unique / seq_len,
            "numerical_complexity": min(1.0, std / (abs(mean)+1e-8) if mean!=0 else std),
            "complexity_from_variance": min(1.0, variance / 10.0)
        }

    def _tensor_to_complexity_metrics(self, analysis: Dict[str, Any]) -> ComplexityMetrics:
        score = (0.3 * min(1.0, analysis["sequence_length"]/10000) + 0.2 * analysis["unique_ratio"] + 0.5 * analysis["complexity_from_variance"])
        return ComplexityMetrics(
            analysis["sequence_length"], analysis["unique_ratio"], analysis["complexity_from_variance"],
            0.0, analysis.get("numerical_complexity", 0.0), 0.0, min(1.0, score)
        )

    def _aggregate_complexities(self, metrics: List[ComplexityMetrics]) -> ComplexityMetrics:
        if not metrics: return ComplexityMetrics(0,0,0,0,0,0,0)
        count = len(metrics)
        return ComplexityMetrics(
            int(sum(m.sequence_length for m in metrics)/count),
            sum(m.vocabulary_richness for m in metrics)/count,
            sum(m.syntactic_complexity for m in metrics)/count,
            sum(m.semantic_density for m in metrics)/count,
            sum(m.numerical_content_ratio for m in metrics)/count,
            sum(m.special_character_ratio for m in metrics)/count,
            sum(m.complexity_score for m in metrics)/count
        )

    def get_adaptive_batch_size(self, score, min_b, max_b, low, high):
        if score <= low: return max_b
        if score >= high: return min_b
        norm = (score - low) / (high - low)
        return int(max_b - norm * (max_b - min_b))

def get_complexity_analyzer():
    if not hasattr(get_complexity_analyzer, "_instance"):
        get_complexity_analyzer._instance = InputComplexityAnalyzer()
    return get_complexity_analyzer._instance

__all__ = ["InputComplexityAnalyzer", "ComplexityMetrics", "get_complexity_analyzer"]
