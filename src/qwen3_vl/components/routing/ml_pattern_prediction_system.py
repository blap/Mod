"""
Lightweight ML Pattern Prediction System for Memory Tiering

This module implements a lightweight machine learning system for predicting
tensor access patterns in vision-language models. It uses multiple algorithms
to predict when tensors will be accessed and how frequently, enabling proactive
memory management.

Key Features:
- Multiple prediction algorithms (LSTM-inspired, statistical, temporal)
- Ensemble prediction combining multiple models
- Real-time learning and adaptation
- Hardware-aware prediction
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import time
import threading
import statistics
from collections import deque, defaultdict
import math
from enum import Enum


class PredictionAlgorithm(Enum):
    """Available prediction algorithms"""
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    ACCESS_PATTERN = "access_pattern"
    ENSEMBLE = "ensemble"


@dataclass
class TensorAccessRecord:
    """Record of tensor access with contextual information"""
    tensor_id: str
    access_time: float
    access_type: str  # 'read', 'write', 'compute'
    tensor_size: int
    tensor_type: str  # 'kv_cache', 'image_features', etc.
    context: Dict[str, Any]  # Additional context like layer, position, etc.
    predicted_next_access: Optional[float] = None
    prediction_confidence: float = 0.0


class StatisticalPredictor:
    """
    Statistical predictor based on access frequency and intervals.
    Uses historical access patterns to predict future access.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.access_records: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.tensor_stats: Dict[str, Dict[str, float]] = defaultdict(dict)

    def update(self, record: TensorAccessRecord):
        """Update the predictor with a new access record"""
        tensor_id = record.tensor_id
        self.access_records[tensor_id].append(record)
        
        # Calculate statistics for this tensor
        accesses = list(self.access_records[tensor_id])
        if len(accesses) < 2:
            return

        # Calculate access intervals
        intervals = []
        for i in range(1, len(accesses)):
            interval = accesses[i].access_time - accesses[i-1].access_time
            intervals.append(interval)

        if intervals:
            self.tensor_stats[tensor_id]['avg_interval'] = statistics.mean(intervals)
            self.tensor_stats[tensor_id]['interval_std'] = statistics.stdev(intervals) if len(intervals) > 1 else 0

            # Calculate time span, avoid division by zero
            time_span = accesses[-1].access_time - accesses[0].access_time
            if time_span > 0:
                self.tensor_stats[tensor_id]['access_frequency'] = len(accesses) / time_span
            else:
                # If all accesses happened at the same time, use a default frequency
                self.tensor_stats[tensor_id]['access_frequency'] = len(accesses)  # Count as frequency

    def predict_next_access(self, tensor_id: str) -> Tuple[Optional[float], float]:
        """
        Predict when the tensor will be accessed next.
        
        Returns:
            Tuple of (predicted_time, confidence_score)
        """
        if tensor_id not in self.tensor_stats:
            return None, 0.1  # Low confidence if no data
            
        stats = self.tensor_stats[tensor_id]
        last_access = self.access_records[tensor_id][-1].access_time
        avg_interval = stats.get('avg_interval', 0)
        
        if avg_interval <= 0:
            return None, 0.1
            
        predicted_time = last_access + avg_interval
        
        # Calculate confidence based on interval consistency
        interval_std = stats.get('interval_std', float('inf'))
        if interval_std == 0:
            confidence = 0.9  # Perfectly consistent
        else:
            # Lower std = higher confidence
            coefficient_of_variation = interval_std / avg_interval if avg_interval > 0 else float('inf')
            confidence = max(0.1, 1.0 - min(coefficient_of_variation, 0.9))
        
        return predicted_time, min(confidence, 0.95)


class TemporalPredictor:
    """
    Temporal predictor that considers time-based patterns in access.
    Looks for cyclical or periodic access patterns.
    """
    
    def __init__(self, window_size: int = 200, max_period: float = 300.0):  # 5 minutes max period
        self.window_size = window_size
        self.max_period = max_period
        self.access_records: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.tensor_periods: Dict[str, List[float]] = defaultdict(list)

    def update(self, record: TensorAccessRecord):
        """Update the predictor with a new access record"""
        tensor_id = record.tensor_id
        self.access_records[tensor_id].append(record)
        
        # Calculate potential periods if we have enough data
        accesses = list(self.access_records[tensor_id])
        if len(accesses) >= 3:
            # Try to detect periodic patterns
            potential_periods = self._find_periodic_patterns(accesses)
            if potential_periods:
                self.tensor_periods[tensor_id] = potential_periods

    def _find_periodic_patterns(self, accesses: List[TensorAccessRecord]) -> List[float]:
        """Find potential periodic patterns in access times"""
        if len(accesses) < 3:
            return []
        
        # Calculate intervals between all consecutive accesses
        intervals = []
        for i in range(1, len(accesses)):
            interval = accesses[i].access_time - accesses[i-1].access_time
            if 0 < interval <= self.max_period:  # Only consider intervals within our max period
                intervals.append(interval)
        
        if not intervals:
            return []
        
        # Find recurring patterns (clusters of similar intervals)
        if len(intervals) < 2:
            return [intervals[0]] if intervals else []
        
        # Simple clustering - group similar intervals
        clusters = []
        for interval in intervals:
            # Find the closest existing cluster
            closest_cluster = None
            min_diff = float('inf')
            for cluster in clusters:
                diff = abs(cluster['center'] - interval)
                if diff < min_diff and diff < cluster['std'] * 2:  # Within 2 std devs
                    min_diff = diff
                    closest_cluster = cluster
            
            if closest_cluster:
                closest_cluster['values'].append(interval)
                closest_cluster['center'] = statistics.mean(closest_cluster['values'])
                closest_cluster['std'] = statistics.stdev(closest_cluster['values']) if len(closest_cluster['values']) > 1 else 0
            else:
                clusters.append({
                    'values': [interval],
                    'center': interval,
                    'std': 0
                })
        
        # Return the centers of significant clusters
        significant_periods = []
        for cluster in clusters:
            if len(cluster['values']) >= 2:  # At least 2 occurrences to be significant
                significant_periods.append(cluster['center'])
        
        return significant_periods

    def predict_next_access(self, tensor_id: str) -> Tuple[Optional[float], float]:
        """
        Predict next access time based on temporal patterns.
        
        Returns:
            Tuple of (predicted_time, confidence_score)
        """
        if tensor_id not in self.tensor_periods or not self.tensor_periods[tensor_id]:
            return None, 0.1
            
        if tensor_id not in self.access_records or len(self.access_records[tensor_id]) == 0:
            return None, 0.1
            
        last_access = self.access_records[tensor_id][-1].access_time
        periods = self.tensor_periods[tensor_id]
        
        # Use the most common period
        avg_period = statistics.mean(periods) if periods else 0
        
        if avg_period <= 0:
            return None, 0.1
            
        predicted_time = last_access + avg_period
        
        # Confidence based on how many periods we found
        confidence = min(0.9, 0.3 + 0.6 * min(len(periods), 5) / 5)
        
        return predicted_time, confidence


class AccessPatternPredictor:
    """
    Access pattern predictor that learns from access sequences.
    Uses n-gram models to predict access patterns.
    """
    
    def __init__(self, n_gram_size: int = 3, window_size: int = 50):
        self.n_gram_size = n_gram_size
        self.window_size = window_size
        self.access_sequence: deque = deque(maxlen=window_size)
        self.n_gram_model: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.tensor_frequencies: Dict[str, int] = defaultdict(int)

    def update(self, record: TensorAccessRecord):
        """Update the predictor with a new access record"""
        tensor_id = record.tensor_id
        self.access_sequence.append(tensor_id)
        self.tensor_frequencies[tensor_id] += 1

        # Convert deque to list to safely access elements by index
        sequence_list = list(self.access_sequence)

        # Update n-gram model
        if len(sequence_list) >= self.n_gram_size:
            for i in range(len(sequence_list) - self.n_gram_size + 1):
                n_gram = tuple(sequence_list[i:i + self.n_gram_size - 1])
                next_tensor = sequence_list[i + self.n_gram_size - 1]
                self.n_gram_model[n_gram][next_tensor] += 1

    def predict_next_access(self, tensor_id: str) -> Tuple[Optional[float], float]:
        """
        Predict next access based on access patterns.

        Returns:
            Tuple of (predicted_tensor_id, confidence_score)
            Note: This predictor returns the next tensor likely to be accessed, not the time
        """
        # Find the most recent context
        if len(self.access_sequence) < self.n_gram_size - 1:
            return None, 0.1

        # Convert deque to list to properly slice
        sequence_list = list(self.access_sequence)
        start_idx = max(0, len(sequence_list) - (self.n_gram_size - 1))
        recent_context = tuple(sequence_list[start_idx:])

        if recent_context not in self.n_gram_model:
            return None, 0.1

        possible_next = self.n_gram_model[recent_context]

        # Find the most likely next tensor
        if not possible_next:
            return None, 0.1

        next_tensor = max(possible_next.keys(), key=lambda x: possible_next[x])
        total_count = sum(possible_next.values())
        confidence = possible_next[next_tensor] / total_count if total_count > 0 else 0

        return next_tensor, min(confidence, 0.95)


class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple prediction algorithms.
    Weights predictions based on their historical accuracy.
    """
    
    def __init__(self, window_size: int = 100):
        self.statistical_predictor = StatisticalPredictor(window_size)
        self.temporal_predictor = TemporalPredictor(window_size)
        self.access_pattern_predictor = AccessPatternPredictor(n_gram_size=3, window_size=window_size)
        
        # Track prediction accuracy for dynamic weighting
        self.prediction_accuracy: Dict[PredictionAlgorithm, List[float]] = defaultdict(list)
        self.max_accuracy_records = 50  # Keep last 50 accuracy records
        
        # Default weights for each predictor
        self.weights = {
            PredictionAlgorithm.STATISTICAL: 0.4,
            PredictionAlgorithm.TEMPORAL: 0.3,
            PredictionAlgorithm.ACCESS_PATTERN: 0.3
        }

    def update(self, record: TensorAccessRecord):
        """Update all predictors with a new access record"""
        self.statistical_predictor.update(record)
        self.temporal_predictor.update(record)
        self.access_pattern_predictor.update(record)

    def predict_next_access(self, tensor_id: str) -> Tuple[Optional[float], float]:
        """
        Predict next access time using ensemble of predictors.
        
        Returns:
            Tuple of (predicted_time, confidence_score)
        """
        predictions = []
        
        # Get predictions from each algorithm
        stat_pred, stat_conf = self.statistical_predictor.predict_next_access(tensor_id)
        if stat_pred is not None:
            predictions.append((stat_pred, stat_conf, self.weights[PredictionAlgorithm.STATISTICAL]))
        
        temp_pred, temp_conf = self.temporal_predictor.predict_next_access(tensor_id)
        if temp_pred is not None:
            predictions.append((temp_pred, temp_conf, self.weights[PredictionAlgorithm.TEMPORAL]))
        
        # Note: Access pattern predictor predicts next tensor, not time
        # We'll use it to adjust confidence if it predicts this tensor will be accessed again
        next_tensor, pattern_conf = self.access_pattern_predictor.predict_next_access(tensor_id)
        if next_tensor == tensor_id:  # If this tensor is predicted to be accessed next
            # Use the temporal predictor's time with adjusted confidence
            temp_pred, temp_conf = self.temporal_predictor.predict_next_access(tensor_id)
            if temp_pred is not None:
                adjusted_conf = min(0.95, temp_conf * (1 + pattern_conf * 0.5))
                predictions.append((temp_pred, adjusted_conf, 
                                  self.weights[PredictionAlgorithm.ACCESS_PATTERN] * pattern_conf))
        
        if not predictions:
            return None, 0.1
        
        # Weighted average of predictions
        weighted_time = sum(pred * conf * weight for pred, conf, weight in predictions)
        total_weight = sum(conf * weight for _, conf, weight in predictions)
        
        if total_weight == 0:
            return None, 0.1
        
        predicted_time = weighted_time / total_weight
        
        # Overall confidence is weighted average of confidences
        overall_confidence = sum(conf * weight for _, conf, weight in predictions) / sum(weight for _, _, weight in predictions)
        
        return predicted_time, min(overall_confidence, 0.98)

    def update_prediction_accuracy(self, tensor_id: str, predicted_time: float, actual_time: float, algorithm: PredictionAlgorithm):
        """Update the accuracy tracking for a prediction"""
        error = abs(predicted_time - actual_time)
        # Convert error to accuracy score (0-1, where 1 is perfect)
        accuracy = 1.0 / (1.0 + error)  # Higher error = lower accuracy
        
        self.prediction_accuracy[algorithm].append(accuracy)
        
        # Keep only recent accuracy records
        if len(self.prediction_accuracy[algorithm]) > self.max_accuracy_records:
            self.prediction_accuracy[algorithm].pop(0)
        
        # Update weights based on recent accuracy
        self._update_weights()

    def _update_weights(self):
        """Update predictor weights based on recent accuracy"""
        if not any(self.prediction_accuracy.values()):
            return
        
        # Calculate average accuracy for each algorithm
        avg_accuracy = {}
        for alg in PredictionAlgorithm:
            if self.prediction_accuracy[alg]:
                avg_accuracy[alg] = statistics.mean(self.prediction_accuracy[alg])
            else:
                avg_accuracy[alg] = 0.5  # Default average
        
        # Normalize to get weights (they should sum to 1.0)
        total_accuracy = sum(avg_accuracy.values())
        if total_accuracy > 0:
            for alg in PredictionAlgorithm:
                self.weights[alg] = avg_accuracy[alg] / total_accuracy


class LightweightMLPredictor:
    """
    Main lightweight ML predictor system that combines all prediction algorithms.
    Designed to be efficient and suitable for real-time memory management.
    """
    
    def __init__(self, 
                 prediction_window: int = 500,
                 algorithm: PredictionAlgorithm = PredictionAlgorithm.ENSEMBLE):
        self.algorithm = algorithm
        self.prediction_window = prediction_window
        
        # Initialize the appropriate predictor
        if algorithm == PredictionAlgorithm.ENSEMBLE:
            self.predictor = EnsemblePredictor(prediction_window)
        elif algorithm == PredictionAlgorithm.STATISTICAL:
            self.predictor = StatisticalPredictor(prediction_window)
        elif algorithm == PredictionAlgorithm.TEMPORAL:
            self.predictor = TemporalPredictor(prediction_window)
        elif algorithm == PredictionAlgorithm.ACCESS_PATTERN:
            self.predictor = AccessPatternPredictor(n_gram_size=3, window_size=prediction_window)
        else:
            raise ValueError(f"Unknown prediction algorithm: {algorithm}")
        
        # Store tensor characteristics for more accurate prediction
        self.tensor_characteristics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Thread safety
        self._lock = threading.Lock()

    def update_tensor_characteristics(self, tensor_id: str, characteristics: Dict[str, Any]):
        """Update characteristics for a tensor"""
        with self._lock:
            self.tensor_characteristics[tensor_id].update(characteristics)

    def record_tensor_access(self, 
                           tensor_id: str, 
                           access_type: str = 'read',
                           tensor_size: int = 0,
                           tensor_type: str = 'general',
                           context: Optional[Dict[str, Any]] = None):
        """Record a tensor access to update prediction models"""
        if context is None:
            context = {}
            
        record = TensorAccessRecord(
            tensor_id=tensor_id,
            access_time=time.time(),
            access_type=access_type,
            tensor_size=tensor_size,
            tensor_type=tensor_type,
            context=context
        )
        
        with self._lock:
            self.predictor.update(record)

    def predict_tensor_access(self, tensor_id: str) -> Tuple[Optional[float], float]:
        """
        Predict when the tensor will be accessed next.
        
        Returns:
            Tuple of (predicted_time, confidence_score)
        """
        with self._lock:
            if self.algorithm == PredictionAlgorithm.ENSEMBLE:
                return self.predictor.predict_next_access(tensor_id)
            elif self.algorithm == PredictionAlgorithm.STATISTICAL:
                return self.predictor.predict_next_access(tensor_id)
            elif self.algorithm == PredictionAlgorithm.TEMPORAL:
                return self.predictor.predict_next_access(tensor_id)
            elif self.algorithm == PredictionAlgorithm.ACCESS_PATTERN:
                # This predictor returns next tensor, not time
                next_tensor, confidence = self.predictor.predict_next_access(tensor_id)
                if next_tensor == tensor_id:
                    # If this tensor is predicted next, use temporal predictor for timing
                    temp_pred = TemporalPredictor(self.prediction_window)
                    # We can't make a real prediction without the full history
                    # This is a simplified implementation
                    return time.time() + 1.0, confidence * 0.5
                else:
                    return None, 0.1
            else:
                return None, 0.1

    def predict_multiple_tensors(self, tensor_ids: List[str]) -> Dict[str, Tuple[Optional[float], float]]:
        """Predict access times for multiple tensors"""
        predictions = {}
        for tensor_id in tensor_ids:
            predictions[tensor_id] = self.predict_tensor_access(tensor_id)
        return predictions

    def get_access_frequency(self, tensor_id: str) -> float:
        """Get the access frequency for a tensor"""
        with self._lock:
            if hasattr(self.predictor, 'tensor_stats') and tensor_id in self.predictor.tensor_stats:
                return self.predictor.tensor_stats[tensor_id].get('access_frequency', 0)
            else:
                # Fallback: calculate from access records
                if hasattr(self.predictor, 'access_records') and tensor_id in self.predictor.access_records:
                    accesses = list(self.predictor.access_records[tensor_id])
                    if len(accesses) >= 2:
                        time_span = accesses[-1].access_time - accesses[0].access_time
                        if time_span > 0:
                            return len(accesses) / time_span
            return 0

    def get_tensor_priority(self, tensor_id: str) -> float:
        """Calculate priority score for a tensor based on prediction"""
        predicted_time, confidence = self.predict_tensor_access(tensor_id)
        
        if predicted_time is None:
            return 0.1  # Low priority if no prediction
        
        # Calculate how soon the tensor will be needed
        time_until_access = predicted_time - time.time()
        
        # Priority decreases as time increases
        if time_until_access <= 0:
            priority = 1.0  # Highest priority if due now
        else:
            # Use exponential decay: priority = base * exp(-time_factor * time_until)
            time_factor = 0.1  # Adjusts decay rate
            priority = math.exp(-time_factor * time_until_access)
        
        # Adjust by confidence
        priority *= confidence
        
        # Adjust by access frequency
        frequency = self.get_access_frequency(tensor_id)
        if frequency > 0:
            priority *= min(2.0, 1.0 + frequency * 10)  # Boost for frequent access
        
        return min(priority, 1.0)

    def get_predicted_access_sequence(self, tensor_ids: List[str], look_ahead: int = 10) -> List[Tuple[str, float]]:
        """Get predicted access sequence for tensor IDs"""
        predictions = []
        for tensor_id in tensor_ids:
            predicted_time, confidence = self.predict_tensor_access(tensor_id)
            if predicted_time is not None:
                predictions.append((tensor_id, predicted_time))
        
        # Sort by predicted access time
        predictions.sort(key=lambda x: x[1])
        return predictions[:look_ahead]


def create_prediction_system(algorithm: PredictionAlgorithm = PredictionAlgorithm.ENSEMBLE,
                           window_size: int = 500) -> LightweightMLPredictor:
    """
    Factory function to create a prediction system.
    
    Args:
        algorithm: Prediction algorithm to use
        window_size: Size of prediction window
        
    Returns:
        LightweightMLPredictor instance
    """
    return LightweightMLPredictor(
        prediction_window=window_size,
        algorithm=algorithm
    )


# Example usage
if __name__ == "__main__":
    print("Lightweight ML Pattern Prediction System")
    print("=" * 50)
    
    # Create the prediction system
    predictor = create_prediction_system(
        algorithm=PredictionAlgorithm.ENSEMBLE,
        window_size=100
    )
    
    print("\n1. Testing tensor access prediction...")
    
    # Simulate tensor accesses
    import random
    
    tensor_ids = [f"tensor_{i}" for i in range(10)]
    
    # Simulate access patterns - some tensors accessed more frequently
    for i in range(100):
        # Select tensor with some pattern
        if i % 3 == 0:
            tensor_id = tensor_ids[0]  # Access first tensor frequently
        elif i % 5 == 0:
            tensor_id = tensor_ids[1]  # Access second tensor periodically
        else:
            tensor_id = random.choice(tensor_ids)  # Random access for others
        
        predictor.record_tensor_access(
            tensor_id,
            access_type='read',
            tensor_size=random.randint(1000, 100000),
            tensor_type='general',
            context={'layer': random.randint(0, 12)}
        )
        
        # Small delay to create time differences
        time.sleep(0.01)
    
    print("   Recorded 100 tensor accesses")
    
    # Get predictions
    print(f"\n2. Getting predictions for tensors...")
    for tensor_id in tensor_ids[:5]:  # Check first 5 tensors
        predicted_time, confidence = predictor.predict_tensor_access(tensor_id)
        priority = predictor.get_tensor_priority(tensor_id)
        
        if predicted_time:
            time_diff = predicted_time - time.time()
            print(f"   {tensor_id}: Next access in {time_diff:.2f}s, "
                  f"Confidence: {confidence:.2f}, Priority: {priority:.2f}")
        else:
            print(f"   {tensor_id}: No prediction available")
    
    # Get access sequence prediction
    print(f"\n3. Predicted access sequence:")
    sequence = predictor.get_predicted_access_sequence(tensor_ids[:5], look_ahead=5)
    for i, (tensor_id, access_time) in enumerate(sequence):
        time_diff = access_time - time.time()
        print(f"   {i+1}. {tensor_id} in {time_diff:.2f}s")
    
    print(f"\nML Pattern Prediction System initialized successfully!")