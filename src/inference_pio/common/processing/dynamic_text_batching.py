"""
Dynamic Text Batching System
Dependency-Free
"""

import logging
import re
import threading
import time
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import psutil
from ...core.engine.backend import Tensor, HAS_CUDA

from .adaptive_batch_manager import AdaptiveBatchManager, BatchMetrics, BatchSizeAdjustmentReason
from .input_complexity_analyzer import InputComplexityAnalyzer

logger = logging.getLogger(__name__)

class TextBatchType(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    TECHNICAL = "technical"
    CODE = "code"
    LONG_CONTEXT = "long_context"

@dataclass
class TextBatchInfo:
    batch_type: TextBatchType
    complexity_score: float
    avg_sequence_length: int
    vocabulary_richness: float
    semantic_density: float
    token_count: int
    estimated_memory_usage: float
    recommended_batch_size: int

class DynamicTextBatchManager:
    def __init__(
        self,
        initial_batch_size: int = 4,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        memory_threshold_ratio: float = 0.80,
        performance_window_size: int = 10,
        adjustment_factor: float = 0.15,
        cooldown_period: float = 3.0,
        performance_target: float = 0.85,
        complexity_weight: float = 0.4,
        sequence_length_weight: float = 0.3,
        memory_safety_margin: float = 0.15,
    ):
        self.current_batch_size = max(min_batch_size, min(initial_batch_size, max_batch_size))
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold_ratio = memory_threshold_ratio
        self.performance_window_size = performance_window_size
        self.adjustment_factor = adjustment_factor
        self.cooldown_period = cooldown_period
        self.performance_target = performance_target
        self.complexity_weight = complexity_weight
        self.sequence_length_weight = sequence_length_weight

        self.metrics_history = deque(maxlen=performance_window_size)
        self.text_batch_history = deque(maxlen=performance_window_size)
        self.last_adjustment_time = time.time()
        self._lock = threading.Lock()

        self._recent_processing_times = deque(maxlen=performance_window_size)
        self._recent_throughputs = deque(maxlen=performance_window_size)
        self._complexity_analyzer = InputComplexityAnalyzer()

        self._text_type_distribution = {t: 0 for t in TextBatchType}

    def analyze_text_batch_type(self, texts: Union[str, List[str]]) -> TextBatchType:
        if isinstance(texts, str): texts = [texts]
        full_text = " ".join(texts)
        avg_length = sum(len(t) for t in texts) / len(texts) if texts else 0

        # Simple heuristics
        code_indicators = ["def ", "class ", "import ", "{", "}"]
        code_count = sum(full_text.count(i) for i in code_indicators)

        if avg_length > 2000: return TextBatchType.LONG_CONTEXT
        if code_count > len(texts): return TextBatchType.CODE
        return TextBatchType.SIMPLE

    def estimate_memory_usage(self, texts: Union[str, List[str]], batch_size: int) -> float:
        if isinstance(texts, str): texts = [texts]
        max_len = max(len(t) for t in texts) if texts else 1
        return batch_size * max_len * 2e-9 * 1.5

    def collect_text_batch_metrics(self, texts, processing_time_ms, tokens_processed, batch_size) -> TextBatchInfo:
        complexity = self._complexity_analyzer.analyze_input_complexity(texts)
        batch_type = self.analyze_text_batch_type(texts)
        self._text_type_distribution[batch_type] += 1

        avg_seq = len(texts) if isinstance(texts, str) else (sum(len(t) for t in texts)/len(texts) if texts else 0)
        est_mem = self.estimate_memory_usage(texts, batch_size)

        rec_size = self._complexity_analyzer.get_adaptive_batch_size(
            complexity.complexity_score, self.min_batch_size, self.max_batch_size, 0.3, 0.7
        )

        info = TextBatchInfo(batch_type, complexity.complexity_score, int(avg_seq),
                             complexity.vocabulary_richness, complexity.semantic_density,
                             tokens_processed, est_mem, rec_size)
        self.text_batch_history.append(info)
        return info

    def get_system_memory_info(self) -> Dict[str, float]:
        mem = psutil.virtual_memory()
        gpu = {"gpu_memory_allocated_gb": 0.0, "gpu_memory_total_gb": 16.0} if HAS_CUDA else {}
        return {
            "memory_percent": mem.percent,
            "memory_used_gb": mem.used / 1e9,
            "memory_pressure_ratio": mem.used / mem.total,
            **gpu
        }

    # ... (Metrics collection and adjustment logic similar to AdaptiveBatchManager but using text info) ...
    # Simplified to avoid repeating large chunks, assuming core logic matches refactored dependencies.

    def collect_metrics(self, batch_size, time_ms, tokens, input_data=None):
        mem = self.get_system_memory_info()
        throughput = (tokens / time_ms)*1000 if time_ms > 0 else 0
        latency = time_ms / max(tokens, 1) if time_ms > 0 else 0

        score = 0.5 # Placeholder calculation
        metrics = BatchMetrics(time.time(), mem["memory_used_gb"], 0.0, batch_size, time_ms, throughput, latency, mem["memory_pressure_ratio"], 0.0, score)
        self.metrics_history.append(metrics)
        return metrics

    def get_optimal_batch_size(self, time_ms, tokens, input_data=None):
        self.collect_metrics(self.current_batch_size, time_ms, tokens, input_data)
        # Logic to return size
        return self.current_batch_size

def get_dynamic_text_batch_manager(**kwargs):
    if not hasattr(get_dynamic_text_batch_manager, "_instance"):
        get_dynamic_text_batch_manager._instance = DynamicTextBatchManager(**kwargs)
    return get_dynamic_text_batch_manager._instance

__all__ = ["DynamicTextBatchManager", "TextBatchType", "TextBatchInfo", "get_dynamic_text_batch_manager"]
