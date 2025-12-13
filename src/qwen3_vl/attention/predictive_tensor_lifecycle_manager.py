"""
Advanced Predictive Tensor Lifecycle Management System for Qwen3-VL

This module implements an advanced garbage collection and memory lifecycle system
with predictive capabilities for tensor management. The system includes:
1. Predictive garbage collection based on access patterns and usage prediction
2. Tensor lifecycle policies with reference counting and usage tracking
3. Lifetime prediction algorithms for tensor lifecycle management
4. Hardware-specific optimizations for Intel i5-10210U + NVIDIA SM61 + NVMe SSD
5. Integration with existing memory tiering, cache, compression and swapping systems

Key Features:
- Predictive tensor lifecycle management based on access patterns
- Reference counting with smart deallocation
- ML-based lifetime prediction for tensors
- Hardware-aware memory optimizations
- Integration with existing memory management systems
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import time
import threading
import queue
import logging
from collections import defaultdict, deque
import weakref
from concurrent.futures import ThreadPoolExecutor
import psutil
import math
import statistics
from datetime import datetime
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.qwen3_vl.utils.debug_utils import conditional_debug
from src.qwen3_vl.utils.general_utils import is_debug_mode

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    logger.warning("psutil not available, some memory monitoring features will be limited")
    PSUTIL_AVAILABLE = False


class TensorState(Enum):
    """States in tensor lifecycle"""
    ALLOCATED = "allocated"
    IN_USE = "in_use"
    UNUSED = "unused"
    PREDICTED_FOR_DELETION = "predicted_for_deletion"
    MARKED_FOR_COLLECTION = "marked_for_collection"
    COLLECTED = "collected"


class TensorType(Enum):
    """Types of tensors for optimization"""
    GENERAL = "general"
    KV_CACHE = "kv_cache"
    IMAGE_FEATURES = "image_features"
    TEXT_EMBEDDINGS = "text_embeddings"
    GRADIENTS = "gradients"
    OPTIMIZER_STATE = "optimizer_state"
    INTERMEDIATE = "intermediate"


@dataclass
class TensorMetadata:
    """Metadata for tracked tensors"""
    tensor_id: str
    tensor_shape: Tuple[int, ...]
    tensor_dtype: torch.dtype
    tensor_type: TensorType
    size_bytes: int
    creation_time: float
    last_access_time: float
    last_predict_time: float
    access_count: int
    predicted_lifetime: float  # Predicted time until tensor is no longer needed (seconds)
    predicted_access_count: int  # Predicted number of future accesses
    reference_count: int
    tensor_state: TensorState
    access_pattern: deque  # Track access times
    is_pinned: bool = False  # If pinned, should not be collected
    predicted_for_collection: bool = False
    collection_priority: float = 0.0  # Higher means higher priority for collection
    hardware_location: str = "cpu"  # Current location: cpu, gpu, disk


class AccessPatternAnalyzer:
    """Analyzes access patterns to predict future tensor usage"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.access_history = deque(maxlen=window_size)
        self.tensor_access_intervals = defaultdict(deque)  # Track intervals between accesses
        self.tensor_access_counts = defaultdict(int)
        self.tensor_last_access = {}
        self.tensor_avg_intervals = {}
        self.tensor_residency_times = defaultdict(float)  # How long tensors stay in memory

    def record_access(self, tensor_id: str):
        """Record access to a tensor"""
        current_time = time.time()
        self.access_history.append((tensor_id, current_time))

        # Update access count
        self.tensor_access_counts[tensor_id] += 1

        # Calculate interval if tensor was accessed before
        if tensor_id in self.tensor_last_access:
            interval = current_time - self.tensor_last_access[tensor_id]
            self.tensor_access_intervals[tensor_id].append(interval)

            # Update average interval
            intervals = list(self.tensor_access_intervals[tensor_id])
            if intervals:
                self.tensor_avg_intervals[tensor_id] = statistics.mean(intervals)

        self.tensor_last_access[tensor_id] = current_time

    def predict_access(self, tensor_id: str) -> Tuple[float, Optional[float], int]:
        """
        Predict probability of tensor being accessed soon, when, and how many times.

        Returns:
            Tuple of (probability, predicted_access_time, predicted_access_count)
        """
        if tensor_id not in self.tensor_avg_intervals:
            # Low probability if no history, but assume at least 1 future access
            return 0.1, None, 1

        avg_interval = self.tensor_avg_intervals[tensor_id]
        last_access = self.tensor_last_access.get(tensor_id, 0)
        time_since_last = time.time() - last_access
        historical_access_count = self.tensor_access_counts.get(tensor_id, 0)

        # If time since last access is close to average interval, high probability
        if avg_interval > 0:
            ratio = time_since_last / avg_interval
            # Sigmoid-like function to convert to probability
            probability = 1.0 / (1.0 + math.exp(-2 * (ratio - 0.5)))
            probability = min(probability, 1.0)

            # Predict next access time
            predicted_time = last_access + avg_interval

            # Predict future access count based on historical pattern
            # For now, use a simple heuristic: if accessed frequently in past, expect more in future
            predicted_access_count = max(1, int(historical_access_count * 0.1))  # 10% of historical accesses

            return probability, predicted_time, predicted_access_count
        else:
            return 0.5, None, 1

    def get_hot_tensors(self, n: int = 10) -> List[str]:
        """Get top N most frequently accessed tensors"""
        sorted_tensors = sorted(
            self.tensor_access_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [tensor_id for tensor_id, count in sorted_tensors[:n]]

    def update_residency_time(self, tensor_id: str, duration: float):
        """Update how long a tensor stays in memory"""
        self.tensor_residency_times[tensor_id] = duration


class LifetimePredictor:
    """Predicts tensor lifetime based on various factors using advanced ML models"""

    def __init__(self):
        self.tensor_features = defaultdict(lambda: {
            'frequency': 0,
            'recency': 0,
            'interval': 0,
            'residency': 0,
            'size': 0,
            'type': 0,
            'context_changes': 0,  # How often tensor is used in different contexts
            'access_pattern_stability': 0,  # Stability of access pattern
            'memory_pressure_history': [],
        })
        self.tensor_access_times = defaultdict(deque)
        self.tensor_lifetimes = defaultdict(float)  # Observed lifetimes for learning
        self.tensor_access_patterns = defaultdict(list)  # Track access patterns over time
        self.tensor_context_changes = defaultdict(int)  # Track context switches
        self.memory_pressure_history = deque(maxlen=100)  # Track system memory pressure
        self.model_weights = self._initialize_model_weights()  # ML model weights

    def _initialize_model_weights(self) -> Dict[str, float]:
        """Initialize model weights for lifetime prediction"""
        # These weights would be learned in a real implementation
        return {
            'frequency': 0.20,
            'recency': 0.18,
            'interval': 0.15,
            'residency': 0.12,
            'size_factor': 0.10,
            'type_factor': 0.08,
            'context_stability': 0.07,
            'access_pattern_stability': 0.05,
            'memory_pressure': 0.05
        }

    def update_features(self, analyzer: AccessPatternAnalyzer):
        """Update features based on access pattern analyzer"""
        current_time = time.time()

        for tensor_id in analyzer.tensor_access_counts:
            access_count = analyzer.tensor_access_counts[tensor_id]
            last_access = analyzer.tensor_last_access.get(tensor_id, 0)
            avg_interval = analyzer.tensor_avg_intervals.get(tensor_id, float('inf'))
            residency_time = analyzer.tensor_residency_times.get(tensor_id, 0)

            # Calculate features
            frequency = access_count
            recency = 1.0 / (current_time - last_access + 1)  # Higher is more recent
            interval = 1.0 / (avg_interval + 1) if avg_interval != float('inf') else 0
            residency = residency_time

            # Calculate additional features
            context_changes = self.tensor_context_changes.get(tensor_id, 0)
            access_pattern_stability = self._calculate_access_pattern_stability(tensor_id)

            # Memory pressure factor
            recent_pressure = self._get_recent_memory_pressure()

            self.tensor_features[tensor_id] = {
                'frequency': frequency,
                'recency': recency,
                'interval': interval,
                'residency': residency,
                'size': 0,  # Will be set when tensor is registered
                'type': 0,  # Will be set based on tensor type
                'context_changes': context_changes,
                'access_pattern_stability': access_pattern_stability,
                'memory_pressure_history': recent_pressure
            }

    def _calculate_access_pattern_stability(self, tensor_id: str) -> float:
        """Calculate stability of access pattern for a tensor"""
        if tensor_id not in self.tensor_access_patterns:
            return 0.5  # Default stability

        access_pattern = self.tensor_access_patterns[tensor_id]
        if len(access_pattern) < 2:
            return 0.5

        # Calculate variance in access intervals
        intervals = [access_pattern[i+1] - access_pattern[i] for i in range(len(access_pattern)-1)]
        if len(intervals) == 0:
            return 0.5

        # More stable if intervals are consistent
        variance = statistics.variance(intervals) if len(intervals) > 1 else 0
        # Convert variance to stability score (lower variance = higher stability)
        stability = 1.0 / (1.0 + variance / 10.0)  # Normalize variance
        return min(stability, 1.0)

    def _get_recent_memory_pressure(self) -> float:
        """Get recent memory pressure as an average"""
        if not self.memory_pressure_history:
            return 0.5
        # Ensure the calculation returns a float
        total = sum(self.memory_pressure_history)
        count = len(self.memory_pressure_history)
        if count > 0:
            return float(total / count)
        else:
            return 0.5  # Fallback in case of empty history

    def record_memory_pressure(self, pressure: float):
        """Record current memory pressure for learning"""
        self.memory_pressure_history.append(pressure)

    def record_context_change(self, tensor_id: str, context: str):
        """Record context change for a tensor"""
        self.tensor_context_changes[tensor_id] += 1

    def record_tensor_access(self, tensor_id: str, access_time: float):
        """Record tensor access to track access patterns"""
        self.tensor_access_patterns[tensor_id].append(access_time)
        # Keep only recent accesses (last 100)
        if len(self.tensor_access_patterns[tensor_id]) > 100:
            self.tensor_access_patterns[tensor_id] = self.tensor_access_patterns[tensor_id][-100:]

    def predict_lifetime(self, tensor_id: str, tensor_size: int, tensor_type: TensorType) -> Tuple[float, int]:
        """
        Predict lifetime and access count for a tensor using advanced ML model.

        Args:
            tensor_id: ID of the tensor
            tensor_size: Size of the tensor in bytes
            tensor_type: Type of tensor

        Returns:
            Tuple of (predicted_lifetime_seconds, predicted_access_count)
        """
        features = self.tensor_features.get(
            tensor_id,
            {'frequency': 0, 'recency': 0, 'interval': 0, 'residency': 0,
             'size': 0, 'type': 0, 'context_changes': 0,
             'access_pattern_stability': 0, 'memory_pressure_history': 0.5}
        )

        # Update features with actual tensor properties
        features['size'] = tensor_size
        features['type'] = self._tensor_type_to_int(tensor_type)

        # Normalize features
        norm_freq = min(features['frequency'] / 10.0, 1.0)  # Normalize frequency
        norm_recency = min(features['recency'] * 100, 1.0)  # Normalize recency
        norm_interval = min(features['interval'] * 10, 1.0)  # Normalize interval
        norm_residency = min(features['residency'] / 300, 1.0)  # Normalize residency (5 min max)
        norm_size = min(tensor_size / (1024*1024*1024), 1.0)  # Normalize size (1GB max)
        norm_context_changes = min(features['context_changes'] / 10.0, 1.0)  # Normalize context changes
        norm_access_pattern_stability = features['access_pattern_stability']
        norm_memory_pressure = features['memory_pressure_history']

        # Apply model weights
        weights = self.model_weights
        lifetime_score = (
            weights['frequency'] * norm_freq +
            weights['recency'] * norm_recency +
            weights['interval'] * norm_interval +
            weights['residency'] * norm_residency +
            weights['size_factor'] * (1.0 - norm_size) +  # Larger tensors have shorter lifetimes
            weights['type_factor'] * self._get_type_lifetime_multiplier(tensor_type) +
            weights['context_stability'] * (1.0 - norm_context_changes) +  # Less context changes = longer lifetime
            weights['access_pattern_stability'] * norm_access_pattern_stability +
            weights['memory_pressure'] * (1.0 - norm_memory_pressure)  # Higher pressure = shorter lifetime
        )

        # Apply tensor type adjustment
        type_multiplier = self._get_type_lifetime_multiplier(tensor_type)
        lifetime_score = lifetime_score * type_multiplier

        # Convert score to time (in seconds)
        # Base lifetime is between 1 and 600 seconds (10 minutes)
        predicted_lifetime = max(1.0, min(600.0, lifetime_score * 300))

        # Predict access count based on frequency and stability
        access_stability_factor = (norm_access_pattern_stability + 0.5)  # Add base stability
        predicted_access_count = max(1, int(norm_freq * access_stability_factor * 5))

        return predicted_lifetime, predicted_access_count

    def _tensor_type_to_int(self, tensor_type: TensorType) -> int:
        """Convert tensor type to integer for model"""
        type_map = {
            TensorType.GENERAL: 0,
            TensorType.KV_CACHE: 1,
            TensorType.IMAGE_FEATURES: 2,
            TensorType.TEXT_EMBEDDINGS: 3,
            TensorType.GRADIENTS: 4,
            TensorType.OPTIMIZER_STATE: 5,
            TensorType.INTERMEDIATE: 6
        }
        return type_map.get(tensor_type, 0)

    def _get_type_lifetime_multiplier(self, tensor_type: TensorType) -> float:
        """Get lifetime multiplier based on tensor type"""
        multipliers = {
            TensorType.GENERAL: 1.0,
            TensorType.KV_CACHE: 1.8,  # KV cache tensors often live much longer
            TensorType.IMAGE_FEATURES: 1.5,  # Image features often reused
            TensorType.TEXT_EMBEDDINGS: 1.6,  # Text embeddings often reused
            TensorType.GRADIENTS: 0.3,  # Gradients often short-lived
            TensorType.OPTIMIZER_STATE: 2.5,  # Optimizer states live very long
            TensorType.INTERMEDIATE: 0.4  # Intermediate tensors often short-lived
        }
        return multipliers.get(tensor_type, 1.0)

    def update_lifetime_observation(self, tensor_id: str, actual_lifetime: float):
        """
        Update the model with actual lifetime observation for learning

        Args:
            tensor_id: ID of the tensor
            actual_lifetime: Actual observed lifetime in seconds
        """
        # In a real implementation, this would update model weights based on prediction error
        # For now, we'll just store the observation for future model improvements
        self.tensor_lifetimes[tensor_id] = actual_lifetime


class TensorLifecycleTracker:
    """Tracks tensor lifecycles and manages state transitions"""

    def __init__(self):
        self.tensor_metadata: Dict[str, TensorMetadata] = {}
        self.tensor_refs: Dict[str, Any] = {}  # Weak references to actual tensors
        self.tensor_lock = threading.RLock()
        self.access_analyzer = AccessPatternAnalyzer()
        self.lifetime_predictor = LifetimePredictor()
        self.executor = ThreadPoolExecutor(max_workers=2)  # For background tasks
        self.tensor_owners: Dict[str, set] = defaultdict(set)  # Track tensor owners/contexts
        self.reference_chains: Dict[str, set] = defaultdict(set)  # Track reference relationships

    def register_tensor(self, tensor: torch.Tensor, tensor_id: str,
                       tensor_type: TensorType = TensorType.GENERAL,
                       is_pinned: bool = False,
                       initial_ref_count: int = 1) -> TensorMetadata:
        """Register a tensor for lifecycle tracking"""
        with self.tensor_lock:
            size_bytes = tensor.element_size() * tensor.nelement()

            # Predict lifetime and access count
            predicted_lifetime, predicted_access_count = self.lifetime_predictor.predict_lifetime(
                tensor_id, size_bytes, tensor_type
            )

            metadata = TensorMetadata(
                tensor_id=tensor_id,
                tensor_shape=tensor.shape,
                tensor_dtype=tensor.dtype,
                tensor_type=tensor_type,
                size_bytes=size_bytes,
                creation_time=time.time(),
                last_access_time=time.time(),
                last_predict_time=time.time(),
                access_count=1,
                predicted_lifetime=predicted_lifetime,
                predicted_access_count=predicted_access_count,
                reference_count=initial_ref_count,
                tensor_state=TensorState.ALLOCATED,
                access_pattern=deque(maxlen=100),
                is_pinned=is_pinned
            )

            self.tensor_metadata[tensor_id] = metadata

            # Store weak reference to tensor
            self.tensor_refs[tensor_id] = weakref.ref(tensor,
                                                    lambda ref: self._tensor_deleted(tensor_id))

            # Record access
            self.access_analyzer.record_access(tensor_id)

            conditional_debug(logger, f"Registered tensor {tensor_id} of type {tensor_type.value}, "
                        f"predicted lifetime: {predicted_lifetime:.2f}s, "
                        f"initial ref count: {initial_ref_count}")

            return metadata

    def _tensor_deleted(self, tensor_id: str):
        """Callback when tensor is deleted externally"""
        with self.tensor_lock:
            if tensor_id in self.tensor_metadata:
                metadata = self.tensor_metadata[tensor_id]
                metadata.tensor_state = TensorState.COLLECTED
                # Clean up related tracking data
                if tensor_id in self.tensor_owners:
                    del self.tensor_owners[tensor_id]
                if tensor_id in self.reference_chains:
                    del self.reference_chains[tensor_id]
                conditional_debug(logger, f"Tensor {tensor_id} was externally deleted")

    def access_tensor(self, tensor_id: str, context: Optional[str] = None) -> Optional[TensorMetadata]:
        """Record access to a tensor and update its state"""
        with self.tensor_lock:
            if tensor_id not in self.tensor_metadata:
                return None

            metadata = self.tensor_metadata[tensor_id]

            # Update access information
            current_time = time.time()
            metadata.last_access_time = current_time
            metadata.access_count += 1
            metadata.access_pattern.append(current_time)
            metadata.tensor_state = TensorState.IN_USE

            # Track access context if provided
            if context:
                # If context is different from last access, record context change
                if hasattr(self, '_last_access_context'):
                    last_context = self._last_access_context.get(tensor_id)
                    if last_context and last_context != context:
                        self.lifetime_predictor.record_context_change(tensor_id, context)
                self._last_access_context = getattr(self, '_last_access_context', {})
                self._last_access_context[tensor_id] = context
                self.tensor_owners[tensor_id].add(context)

            # Record access in analyzer
            self.access_analyzer.record_access(tensor_id)

            # Record access pattern for stability calculation
            self.lifetime_predictor.record_tensor_access(tensor_id, current_time)

            # Update lifetime prediction periodically
            if current_time - metadata.last_predict_time > 10:  # Update every 10 seconds
                predicted_lifetime, predicted_access_count = self.lifetime_predictor.predict_lifetime(
                    tensor_id, metadata.size_bytes, metadata.tensor_type
                )
                metadata.predicted_lifetime = predicted_lifetime
                metadata.predicted_access_count = predicted_access_count
                metadata.last_predict_time = current_time

            conditional_debug(logger, f"Accessed tensor {tensor_id}, state: {metadata.tensor_state.value}, "
                        f"context: {context}")

            return metadata

    def increment_reference(self, tensor_id: str, owner_context: Optional[str] = None) -> bool:
        """Increment reference count for a tensor"""
        with self.tensor_lock:
            if tensor_id not in self.tensor_metadata:
                return False

            metadata = self.tensor_metadata[tensor_id]

            # Increment reference count
            old_count = metadata.reference_count
            metadata.reference_count += 1

            # Track the owner context if provided
            if owner_context:
                self.tensor_owners[tensor_id].add(owner_context)

            # Update state based on reference count
            if metadata.reference_count > 0:
                metadata.tensor_state = TensorState.IN_USE
                # If was predicted for collection, reconsider
                if metadata.predicted_for_collection:
                    metadata.predicted_for_collection = False
                    metadata.tensor_state = TensorState.IN_USE

            conditional_debug(logger, f"Incremented ref count for {tensor_id}: {old_count} -> {metadata.reference_count}")

            return True

    def decrement_reference(self, tensor_id: str, owner_context: Optional[str] = None) -> bool:
        """Decrement reference count for a tensor"""
        with self.tensor_lock:
            if tensor_id not in self.tensor_metadata:
                return False

            metadata = self.tensor_metadata[tensor_id]

            # Decrement reference count
            old_count = metadata.reference_count
            metadata.reference_count = max(0, metadata.reference_count - 1)

            # Remove owner context if provided
            if owner_context and owner_context in self.tensor_owners[tensor_id]:
                self.tensor_owners[tensor_id].remove(owner_context)

            # If no more owners, clear the set
            if not self.tensor_owners[tensor_id]:
                del self.tensor_owners[tensor_id]

            # Update state based on reference count
            if metadata.reference_count <= 0 and not metadata.is_pinned:
                metadata.tensor_state = TensorState.UNUSED
                # Predict for collection if not already predicted
                if not metadata.predicted_for_collection:
                    self._predict_for_collection(metadata)
            elif metadata.reference_count > 0:
                metadata.tensor_state = TensorState.IN_USE

            conditional_debug(logger, f"Decremented ref count for {tensor_id}: {old_count} -> {metadata.reference_count}, "
                        f"state: {metadata.tensor_state.value}")

            return True

    def update_reference_count(self, tensor_id: str, delta: int,
                              owner_context: Optional[str] = None) -> bool:
        """Update reference count for a tensor by a delta value"""
        with self.tensor_lock:
            if tensor_id not in self.tensor_metadata:
                return False

            metadata = self.tensor_metadata[tensor_id]
            old_count = metadata.reference_count
            new_count = max(0, metadata.reference_count + delta)

            # Update reference count
            metadata.reference_count = new_count

            # Track owner context if provided
            if owner_context:
                if delta > 0:
                    self.tensor_owners[tensor_id].add(owner_context)
                elif delta < 0 and owner_context in self.tensor_owners[tensor_id]:
                    self.tensor_owners[tensor_id].remove(owner_context)

            # If no more owners, clear the set
            if not self.tensor_owners[tensor_id]:
                del self.tensor_owners[tensor_id]

            # Update state based on reference count
            if metadata.reference_count <= 0 and not metadata.is_pinned:
                metadata.tensor_state = TensorState.UNUSED
                # Predict for collection if not already predicted
                if not metadata.predicted_for_collection:
                    self._predict_for_collection(metadata)
            elif metadata.reference_count > 0:
                metadata.tensor_state = TensorState.IN_USE

            conditional_debug(logger, f"Updated ref count for {tensor_id}: {old_count} -> {new_count}, "
                        f"delta: {delta}, state: {metadata.tensor_state.value}")

            return True

    def add_reference_chain(self, source_tensor_id: str, target_tensor_id: str) -> bool:
        """Add a reference relationship between tensors"""
        with self.tensor_lock:
            if source_tensor_id not in self.tensor_metadata or target_tensor_id not in self.tensor_metadata:
                return False

            self.reference_chains[source_tensor_id].add(target_tensor_id)
            conditional_debug(logger, f"Added reference chain: {source_tensor_id} -> {target_tensor_id}")
            return True

    def remove_reference_chain(self, source_tensor_id: str, target_tensor_id: str) -> bool:
        """Remove a reference relationship between tensors"""
        with self.tensor_lock:
            if source_tensor_id in self.reference_chains:
                if target_tensor_id in self.reference_chains[source_tensor_id]:
                    self.reference_chains[source_tensor_id].remove(target_tensor_id)
                    # Clean up empty sets
                    if not self.reference_chains[source_tensor_id]:
                        del self.reference_chains[source_tensor_id]
                    conditional_debug(logger, f"Removed reference chain: {source_tensor_id} -> {target_tensor_id}")
                    return True
            return False

    def _predict_for_collection(self, metadata: TensorMetadata):
        """Predict if tensor should be collected based on access patterns"""
        current_time = time.time()
        time_since_last_access = current_time - metadata.last_access_time

        # If time since last access exceeds predicted lifetime, mark for collection
        if time_since_last_access > metadata.predicted_lifetime * 1.2:  # 20% grace period
            metadata.predicted_for_collection = True
            metadata.collection_priority = time_since_last_access / metadata.predicted_lifetime
            metadata.tensor_state = TensorState.PREDICTED_FOR_DELETION

            conditional_debug(logger, f"Predicted tensor {metadata.tensor_id} for collection, "
                        f"priority: {metadata.collection_priority:.3f}")

    def get_tensors_for_collection(self) -> List[TensorMetadata]:
        """Get list of tensors that should be collected"""
        with self.tensor_lock:
            collectible = []
            current_time = time.time()

            for tensor_id, metadata in self.tensor_metadata.items():
                # Check if tensor should be collected
                time_since_last_access = current_time - metadata.last_access_time
                
                # Conditions for collection:
                # 1. Not pinned
                # 2. Reference count <= 0 (or will be soon)
                # 3. Predicted for collection OR time since last access exceeds predicted lifetime
                should_collect = (
                    not metadata.is_pinned and
                    metadata.reference_count <= 0 and
                    (metadata.predicted_for_collection or 
                     time_since_last_access > metadata.predicted_lifetime)
                )

                if should_collect:
                    collectible.append(metadata)

            # Sort by collection priority (highest first)
            collectible.sort(key=lambda m: m.collection_priority, reverse=True)
            return collectible

    def mark_for_collection(self, tensor_id: str) -> bool:
        """Mark a specific tensor for collection"""
        with self.tensor_lock:
            if tensor_id not in self.tensor_metadata:
                return False

            metadata = self.tensor_metadata[tensor_id]
            if not metadata.is_pinned:
                metadata.tensor_state = TensorState.MARKED_FOR_COLLECTION
                metadata.predicted_for_collection = True
                conditional_debug(logger, f"Marked tensor {tensor_id} for collection")
                return True
            return False

    def remove_tensor(self, tensor_id: str) -> bool:
        """Remove tensor from tracking (called after collection)"""
        with self.tensor_lock:
            if tensor_id in self.tensor_metadata:
                del self.tensor_metadata[tensor_id]
                if tensor_id in self.tensor_refs:
                    del self.tensor_refs[tensor_id]
                conditional_debug(logger, f"Removed tensor {tensor_id} from tracking")
                return True
            return False

    def get_tensor_stats(self) -> Dict[str, Any]:
        """Get statistics about tracked tensors"""
        with self.tensor_lock:
            stats = {
                'total_tensors': len(self.tensor_metadata),
                'pinned_tensors': 0,
                'in_use_tensors': 0,
                'unused_tensors': 0,
                'predicted_for_collection': 0,
                'total_memory_bytes': 0,
                'average_lifetime_prediction': 0.0,
                'states': defaultdict(int)
            }

            lifetimes = []
            for metadata in self.tensor_metadata.values():
                stats['total_memory_bytes'] += metadata.size_bytes
                stats['states'][metadata.tensor_state.value] += 1
                
                if metadata.is_pinned:
                    stats['pinned_tensors'] += 1
                if metadata.tensor_state == TensorState.IN_USE:
                    stats['in_use_tensors'] += 1
                elif metadata.tensor_state == TensorState.UNUSED:
                    stats['unused_tensors'] += 1
                if metadata.predicted_for_collection:
                    stats['predicted_for_collection'] += 1
                    
                lifetimes.append(metadata.predicted_lifetime)

            if lifetimes:
                stats['average_lifetime_prediction'] = statistics.mean(lifetimes)

            return stats


class PredictiveGarbageCollector:
    """Main predictive garbage collection system"""

    def __init__(self, collection_interval: float = 1.0,  # seconds
                 memory_pressure_threshold: float = 0.8,  # 80% memory usage
                 enable_background_collection: bool = True):
        """
        Initialize predictive garbage collector

        Args:
            collection_interval: How often to run collection cycles (seconds)
            memory_pressure_threshold: Memory usage threshold to trigger collection
            enable_background_collection: Whether to run collection in background
        """
        self.tracker = TensorLifecycleTracker()
        self.collection_interval = collection_interval
        self.memory_pressure_threshold = memory_pressure_threshold
        self.enable_background_collection = enable_background_collection
        self.collection_thread = None
        self.should_stop = threading.Event()
        self.collection_stats = {
            'collections_performed': 0,
            'tensors_collected': 0,
            'memory_freed_bytes': 0,
            'collection_cycles': 0
        }
        self.collection_lock = threading.Lock()

        if enable_background_collection:
            self._start_background_collection()

    def _start_background_collection(self):
        """Start background collection thread"""
        def collection_worker():
            while not self.should_stop.is_set():
                try:
                    # Check if collection is needed
                    if self._should_collect():
                        self.collect()
                    
                    # Wait for interval or stop event
                    self.should_stop.wait(timeout=self.collection_interval)
                except Exception as e:
                    logger.error(f"Error in collection worker: {e}")
                    time.sleep(self.collection_interval)

        self.collection_thread = threading.Thread(target=collection_worker, daemon=True)
        self.collection_thread.start()

    def _should_collect(self) -> bool:
        """Determine if garbage collection should be performed"""
        # Check memory pressure
        memory_pressure = 0.0
        if PSUTIL_AVAILABLE:
            memory_percent = psutil.virtual_memory().percent / 100.0
            memory_pressure = memory_percent
            # Update lifetime predictor with current memory pressure
            self.tracker.lifetime_predictor.record_memory_pressure(memory_pressure)

            if memory_percent > self.memory_pressure_threshold:
                return True

        # Check if we have tensors predicted for collection
        tensors_for_collection = self.tracker.get_tensors_for_collection()
        return len(tensors_for_collection) > 0

    def get_memory_pressure(self) -> float:
        """Get current memory pressure as a value between 0 and 1"""
        if PSUTIL_AVAILABLE:
            pressure = psutil.virtual_memory().percent / 100.0
            # Also update the lifetime predictor
            self.tracker.lifetime_predictor.record_memory_pressure(pressure)
            return pressure
        return 0.5  # Default if psutil not available

    def register_tensor(self, tensor: torch.Tensor, tensor_id: str,
                      tensor_type: TensorType = TensorType.GENERAL,
                      is_pinned: bool = False,
                      initial_ref_count: int = 1) -> TensorMetadata:
        """Register a tensor with the garbage collector"""
        return self.tracker.register_tensor(tensor, tensor_id, tensor_type, is_pinned, initial_ref_count)

    def access_tensor(self, tensor_id: str, context: Optional[str] = None) -> Optional[TensorMetadata]:
        """Record access to a tensor"""
        return self.tracker.access_tensor(tensor_id, context)

    def increment_reference(self, tensor_id: str, owner_context: Optional[str] = None) -> bool:
        """Increment reference count for a tensor"""
        return self.tracker.increment_reference(tensor_id, owner_context)

    def decrement_reference(self, tensor_id: str, owner_context: Optional[str] = None) -> bool:
        """Decrement reference count for a tensor"""
        return self.tracker.decrement_reference(tensor_id, owner_context)

    def update_reference_count(self, tensor_id: str, delta: int, owner_context: Optional[str] = None) -> bool:
        """Update reference count for a tensor by a delta value"""
        return self.tracker.update_reference_count(tensor_id, delta, owner_context)

    def add_reference_chain(self, source_tensor_id: str, target_tensor_id: str) -> bool:
        """Add a reference relationship between tensors"""
        return self.tracker.add_reference_chain(source_tensor_id, target_tensor_id)

    def remove_reference_chain(self, source_tensor_id: str, target_tensor_id: str) -> bool:
        """Remove a reference relationship between tensors"""
        return self.tracker.remove_reference_chain(source_tensor_id, target_tensor_id)

    def collect(self) -> int:
        """Perform garbage collection cycle"""
        with self.collection_lock:
            tensors_for_collection = self.tracker.get_tensors_for_collection()
            
            if not tensors_for_collection:
                conditional_debug(logger, "No tensors for collection")
                return 0

            collected_count = 0
            memory_freed = 0

            for metadata in tensors_for_collection:
                # Try to collect the tensor
                if self._collect_tensor(metadata.tensor_id):
                    collected_count += 1
                    memory_freed += metadata.size_bytes
                    conditional_debug(logger, f"Collected tensor {metadata.tensor_id}")

            # Update stats
            self.collection_stats['collections_performed'] += 1
            self.collection_stats['tensors_collected'] += collected_count
            self.collection_stats['memory_freed_bytes'] += memory_freed
            self.collection_stats['collection_cycles'] += 1

            logger.info(f"Collection cycle: {collected_count} tensors collected, "
                       f"{memory_freed / (1024**2):.2f} MB freed")

            return collected_count

    def _collect_tensor(self, tensor_id: str) -> bool:
        """Actually collect a single tensor"""
        # First, remove from tracker (this marks it as collected)
        success = self.tracker.remove_tensor(tensor_id)
        
        if success:
            # Force Python garbage collection to ensure tensor is freed
            gc.collect()
            return True
        return False

    def mark_tensor_for_collection(self, tensor_id: str) -> bool:
        """Mark a specific tensor for collection"""
        return self.tracker.mark_for_collection(tensor_id)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics"""
        stats = self.collection_stats.copy()
        tensor_stats = self.tracker.get_tensor_stats()
        stats.update(tensor_stats)
        return stats

    def get_memory_pressure(self) -> float:
        """Get current memory pressure as a value between 0 and 1"""
        if PSUTIL_AVAILABLE:
            return psutil.virtual_memory().percent / 100.0
        return 0.0

    def cleanup(self):
        """Clean up the garbage collector"""
        self.should_stop.set()
        if self.collection_thread:
            self.collection_thread.join(timeout=2.0)  # Wait up to 2 seconds

        # Force collection of all remaining tensors
        tensors_for_collection = self.tracker.get_tensors_for_collection()
        for metadata in tensors_for_collection:
            self._collect_tensor(metadata.tensor_id)


class HardwareAwareTensorManager:
    """Hardware-aware tensor manager that optimizes for specific hardware"""

    def __init__(self, hardware_config: Dict[str, Any]):
        """
        Initialize hardware-aware tensor manager

        Args:
            hardware_config: Hardware configuration with details like:
                            - cpu_model: CPU model string
                            - gpu_model: GPU model string
                            - memory_size: Total system memory in bytes
                            - storage_type: Storage type ('nvme', 'ssd', 'hdd')
        """
        self.hardware_config = hardware_config
        self.cpu_model = hardware_config.get('cpu_model', 'unknown').lower()
        self.gpu_model = hardware_config.get('gpu_model', 'unknown').lower()
        self.memory_size = hardware_config.get('memory_size', 8 * 1024 * 1024 * 1024)
        self.storage_type = hardware_config.get('storage_type', 'nvme').lower()

        # Determine optimization strategies based on hardware
        self._configure_hardware_optimizations()

    def _configure_hardware_optimizations(self):
        """Configure optimizations based on hardware capabilities"""
        # For Intel i5-10210U (4 cores, 8 threads, low power)
        if 'i5-10210u' in self.cpu_model:
            self.cpu_optimizations = {
                'max_workers': 4,  # Conservative thread usage
                'batch_size_multiplier': 0.5,  # Smaller batches
                'prefetch_distance': 2,  # Conservative prefetching
                'memory_alignment': 64,  # 64-byte alignment for optimal cache usage
                'simd_width': 256,  # AVX2 support (256-bit)
                'power_efficiency_mode': True,  # Optimize for power efficiency
                'cache_tiling_size': 32 * 1024,  # L1 cache tiling for i5-10210U
            }
        else:
            self.cpu_optimizations = {
                'max_workers': 8,
                'batch_size_multiplier': 1.0,
                'prefetch_distance': 4,
                'memory_alignment': 64,
                'simd_width': 256,
                'power_efficiency_mode': False,
                'cache_tiling_size': 32 * 1024,
            }

        # For NVIDIA SM61 (Maxwell architecture, older GPU)
        if 'sm61' in self.gpu_model.lower():
            self.gpu_optimizations = {
                'max_tensor_size_gpu': 512 * 1024 * 1024,  # 512MB max on GPU
                'use_tensor_cores': False,  # SM61 doesn't have tensor cores
                'memory_fragmentation_factor': 1.3,  # Account for fragmentation
                'warp_size': 32,  # Standard CUDA warp size
                'max_shared_memory_per_block': 48 * 1024,  # 48KB shared memory for SM61
                'max_threads_per_block': 1024,  # Max threads per block for SM61
                'memory_bandwidth_gb_s': 80,  # Estimated memory bandwidth
            }
        else:
            self.gpu_optimizations = {
                'max_tensor_size_gpu': 2 * 1024 * 1024 * 1024,  # 2GB default
                'use_tensor_cores': True,
                'memory_fragmentation_factor': 1.1,
                'warp_size': 32,
                'max_shared_memory_per_block': 48 * 1024,
                'max_threads_per_block': 1024,
                'memory_bandwidth_gb_s': 100,  # Default estimate
            }

        # For NVMe SSD storage
        if self.storage_type == 'nvme':
            self.storage_optimizations = {
                'use_compression': True,  # NVMe fast enough for compression
                'block_size': 4 * 1024 * 1024,  # 4MB blocks for NVMe
                'async_io': True,
                'queue_depth': 32,  # Optimal queue depth for NVMe
                'read_ahead_buffer': 2 * 1024 * 1024,  # 2MB read-ahead
                'iops_limit': 500000,  # Estimated IOPS for consumer NVMe
            }
        else:
            self.storage_optimizations = {
                'use_compression': True,  # Still beneficial for slower storage
                'block_size': 1 * 1024 * 1024,  # 1MB blocks for slower storage
                'async_io': False,  # May not help with slower storage
                'queue_depth': 4,  # Lower queue depth for slower storage
                'read_ahead_buffer': 512 * 1024,  # 512KB read-ahead
                'iops_limit': 100000,  # Estimated IOPS for slower storage
            }

    def optimize_tensor_placement(self, tensor: torch.Tensor,
                                 tensor_type: TensorType = TensorType.GENERAL) -> str:
        """
        Determine optimal placement for a tensor based on hardware and tensor properties

        Args:
            tensor: The tensor to place
            tensor_type: Type of tensor

        Returns:
            Optimal location ('cpu', 'gpu', or 'disk')
        """
        tensor_size = tensor.element_size() * tensor.nelement()

        # Calculate available GPU memory if CUDA is available
        gpu_available_memory = 0
        if torch.cuda.is_available():
            try:
                gpu_total_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_reserved_memory = torch.cuda.memory_reserved(0)
                gpu_available_memory = gpu_total_memory - gpu_reserved_memory
            except:
                # Fallback to configured limit
                gpu_available_memory = self.gpu_optimizations['max_tensor_size_gpu']

        # For tensors that benefit from GPU acceleration
        if tensor_type in [TensorType.KV_CACHE, TensorType.IMAGE_FEATURES, TensorType.TEXT_EMBEDDINGS]:
            # Check if GPU has enough available memory
            if gpu_available_memory > tensor_size * 1.2:  # 20% buffer
                return 'gpu'

        # For compute-intensive operations, prefer GPU if possible
        elif tensor.nelement() > 1000:  # Only for reasonably sized tensors
            if gpu_available_memory > tensor_size * 1.5:  # 50% buffer for compute overhead
                return 'gpu'

        # For smaller tensors or when GPU is not available, use CPU if it fits
        system_memory_available = self._get_available_system_memory()
        if tensor_size < system_memory_available * 0.8:  # Use 80% of available system memory
            return 'cpu'
        else:
            # For very large tensors that don't fit in memory, use disk with swapping
            return 'disk'

    def _get_available_system_memory(self) -> int:
        """Get available system memory in bytes"""
        if PSUTIL_AVAILABLE:
            available_memory = psutil.virtual_memory().available
            # Leave some memory for the OS and other processes
            return int(available_memory * 0.8)  # Use 80% of available memory
        else:
            # Fallback: use 50% of total configured memory
            return int(self.memory_size * 0.5)

    def optimize_tensor_for_hardware(self, tensor: torch.Tensor,
                                   tensor_type: TensorType = TensorType.GENERAL,
                                   operation: str = "general") -> torch.Tensor:
        """
        Apply hardware-specific optimizations to a tensor

        Args:
            tensor: Input tensor to optimize
            tensor_type: Type of tensor
            operation: Type of operation that will be performed

        Returns:
            Hardware-optimized tensor
        """
        # For Intel CPUs, optimize memory layout and alignment
        if 'intel' in self.cpu_model:
            # Ensure tensor is contiguous for better CPU performance
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()

            # For specific operations on Intel CPUs
            if operation in ["matmul", "conv"]:
                # Apply cache-friendly optimizations for Intel architecture
                tensor = self._apply_intel_cache_optimizations(tensor)

        # For GPU tensors, optimize for GPU memory access patterns
        if tensor_type in [TensorType.KV_CACHE, TensorType.GRADIENTS] and torch.cuda.is_available():
            # Pin memory for faster GPU transfers if tensor will be frequently moved
            if tensor.is_cuda:
                # Already on GPU, ensure proper memory layout
                pass
            else:
                # Prepare for GPU transfer by ensuring contiguity
                tensor = tensor.contiguous()

        # Apply size-specific optimizations
        tensor_size = tensor.element_size() * tensor.nelement()
        if tensor_size > 100 * 1024 * 1024:  # For tensors > 100MB
            # Apply optimizations for large tensors
            tensor = self._optimize_large_tensor(tensor, operation)

        return tensor

    def _apply_intel_cache_optimizations(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply Intel CPU cache optimizations to tensor"""
        # For Intel i5-10210U, optimize for L1/L2 cache usage
        if 'i5-10210u' in self.cpu_model:
            # For matrices, optimize for cache line size (64 bytes)
            if tensor.dim() == 2:
                # Ensure the inner dimension is cache-friendly
                inner_dim = tensor.size(-1)
                cache_line_elements = 64 // tensor.element_size()  # Elements per cache line
                if inner_dim % cache_line_elements != 0:
                    # This is a simplified optimization - in practice, you'd want to
                    # pad the tensor to align with cache lines, but this would change the size
                    pass

        return tensor

    def _optimize_large_tensor(self, tensor: torch.Tensor, operation: str) -> torch.Tensor:
        """Apply optimizations for large tensors"""
        # For large tensors, ensure memory is properly aligned
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # For large matrix operations, consider blocking/tiled operations
        if operation == "matmul" and tensor.dim() == 2:
            # Apply simple tiling to improve cache utilization
            # This is a simplified approach - in practice, you'd use more sophisticated blocking
            pass

        return tensor

    def get_hardware_optimization_info(self) -> Dict[str, Any]:
        """Get information about hardware optimizations"""
        return {
            'cpu_model': self.cpu_model,
            'gpu_model': self.gpu_model,
            'memory_size': self.memory_size,
            'storage_type': self.storage_type,
            'cpu_optimizations': self.cpu_optimizations,
            'gpu_optimizations': self.gpu_optimizations,
            'storage_optimizations': self.storage_optimizations
        }


class IntegratedTensorLifecycleManager:
    """Main integrated tensor lifecycle manager that combines all components"""

    def __init__(self, hardware_config: Dict[str, Any], 
                 enable_memory_tiering: bool = True,
                 enable_compression: bool = True,
                 enable_swapping: bool = True):
        """
        Initialize integrated tensor lifecycle manager

        Args:
            hardware_config: Hardware configuration
            enable_memory_tiering: Whether to enable memory tiering integration
            enable_compression: Whether to enable compression integration
            enable_swapping: Whether to enable swapping integration
        """
        self.hardware_manager = HardwareAwareTensorManager(hardware_config)
        self.garbage_collector = PredictiveGarbageCollector()
        
        # Integration flags
        self.enable_memory_tiering = enable_memory_tiering
        self.enable_compression = enable_compression
        self.enable_swapping = enable_swapping
        
        # Integration references (to be set by user)
        self.memory_tiering_system = None
        self.compression_manager = None
        self.swapping_system = None
        
        logger.info("Integrated Tensor Lifecycle Manager initialized")

    def register_tensor(self, tensor: torch.Tensor, tensor_id: Optional[str] = None,
                       tensor_type: TensorType = TensorType.GENERAL,
                       is_pinned: bool = False,
                       initial_ref_count: int = 1) -> str:
        """
        Register a tensor with the lifecycle management system

        Args:
            tensor: The tensor to register
            tensor_id: Optional tensor ID (auto-generated if None)
            tensor_type: Type of tensor
            is_pinned: Whether tensor should be pinned (not eligible for collection)
            initial_ref_count: Initial reference count for the tensor

        Returns:
            Tensor ID
        """
        if tensor_id is None:
            tensor_id = f"tensor_{id(tensor)}_{int(time.time() * 1000000)}"

        # Determine optimal placement based on hardware
        optimal_placement = self.hardware_manager.optimize_tensor_placement(
            tensor, tensor_type
        )

        # Register with garbage collector
        metadata = self.garbage_collector.register_tensor(
            tensor, tensor_id, tensor_type, is_pinned, initial_ref_count
        )

        # Integrate with other systems if enabled
        if self.enable_memory_tiering and self.memory_tiering_system:
            self._integrate_with_tiering(tensor_id, tensor, tensor_type)

        if self.enable_compression and self.compression_manager:
            self._integrate_with_compression(tensor_id, tensor, tensor_type)

        if self.enable_swapping and self.swapping_system:
            self._integrate_with_swapping(tensor_id, tensor, tensor_type)

        conditional_debug(logger, f"Registered tensor {tensor_id} of type {tensor_type.value}, "
                    f"placement: {optimal_placement}, initial ref count: {initial_ref_count}")

        return tensor_id

    def access_tensor(self, tensor_id: str, context: Optional[str] = None) -> bool:
        """Record access to a tensor"""
        metadata = self.garbage_collector.access_tensor(tensor_id, context)
        return metadata is not None

    def increment_reference(self, tensor_id: str, owner_context: Optional[str] = None) -> bool:
        """Increment reference count for a tensor"""
        return self.garbage_collector.increment_reference(tensor_id, owner_context)

    def decrement_reference(self, tensor_id: str, owner_context: Optional[str] = None) -> bool:
        """Decrement reference count for a tensor"""
        return self.garbage_collector.decrement_reference(tensor_id, owner_context)

    def update_reference_count(self, tensor_id: str, delta: int, owner_context: Optional[str] = None) -> bool:
        """Update reference count for a tensor by a delta value"""
        return self.garbage_collector.update_reference_count(tensor_id, delta, owner_context)

    def add_reference_chain(self, source_tensor_id: str, target_tensor_id: str) -> bool:
        """Add a reference relationship between tensors"""
        return self.garbage_collector.add_reference_chain(source_tensor_id, target_tensor_id)

    def remove_reference_chain(self, source_tensor_id: str, target_tensor_id: str) -> bool:
        """Remove a reference relationship between tensors"""
        return self.garbage_collector.remove_reference_chain(source_tensor_id, target_tensor_id)

    def optimize_tensor(self, tensor: torch.Tensor,
                       tensor_type: TensorType = TensorType.GENERAL,
                       operation: str = "general") -> torch.Tensor:
        """Optimize tensor for hardware and operation"""
        return self.hardware_manager.optimize_tensor_for_hardware(tensor, tensor_type, operation)

    def get_tensor_lifecycle_stats(self) -> Dict[str, Any]:
        """Get comprehensive lifecycle statistics"""
        return self.garbage_collector.get_collection_stats()

    def set_memory_tiering_system(self, tiering_system):
        """Set reference to memory tiering system for integration"""
        self.memory_tiering_system = tiering_system
        logger.info("Memory tiering system integrated")

    def set_compression_manager(self, compression_manager):
        """Set reference to compression manager for integration"""
        self.compression_manager = compression_manager
        logger.info("Compression manager integrated")

    def set_swapping_system(self, swapping_system):
        """Set reference to swapping system for integration"""
        self.swapping_system = swapping_system
        logger.info("Swapping system integrated")

    def _integrate_with_tiering(self, tensor_id: str, tensor: torch.Tensor,
                               tensor_type: TensorType) -> bool:
        """Integrate tensor with memory tiering system"""
        if not self.enable_memory_tiering or not self.memory_tiering_system:
            return False

        try:
            # Determine appropriate tier based on tensor characteristics
            tensor_size = tensor.element_size() * tensor.nelement()

            # Map tensor type to internal types used by tiering system
            tiering_tensor_type = self._map_tensor_type_for_tiering(tensor_type)

            # Put tensor in appropriate tier
            success = self.memory_tiering_system.put_tensor(
                tensor,
                tensor_type=tiering_tensor_type,
                preferred_tier=None  # Let system decide
            )

            if success:
                conditional_debug(logger, f"Successfully integrated tensor {tensor_id} with tiering system")
                return True
            else:
                logger.warning(f"Failed to integrate tensor {tensor_id} with tiering system")
                return False
        except Exception as e:
            logger.error(f"Error integrating tensor {tensor_id} with tiering system: {e}")
            return False

    def _integrate_with_compression(self, tensor_id: str, tensor: torch.Tensor,
                                   tensor_type: TensorType) -> Optional[Dict[str, Any]]:
        """Integrate tensor with compression system"""
        if not self.enable_compression or not self.compression_manager:
            return None

        try:
            # Determine if tensor should be compressed based on type and size
            should_compress = self._should_compress_tensor(tensor, tensor_type)

            if should_compress:
                # Compress tensor using the compression manager
                compressed_data = self.compression_manager.compress_tensor(
                    tensor,
                    method='auto'  # Let system choose best method
                )

                conditional_debug(logger, f"Successfully compressed tensor {tensor_id}")
                return compressed_data
            else:
                conditional_debug(logger, f"Tensor {tensor_id} not compressed (size/type not suitable)")
                return None
        except Exception as e:
            logger.error(f"Error compressing tensor {tensor_id}: {e}")
            return None

    def _integrate_with_swapping(self, tensor_id: str, tensor: torch.Tensor,
                                tensor_type: TensorType) -> bool:
        """Integrate tensor with swapping system"""
        if not self.enable_swapping or not self.swapping_system:
            return False

        try:
            # Register tensor with swapping system
            size_bytes = tensor.element_size() * tensor.nelement()
            tensor_region_type = self._map_tensor_type_for_swapping(tensor_type)

            # Register memory block with swapping system
            success = self.swapping_system.register_memory_block(
                tensor_id,
                size_bytes,
                tensor_region_type,
                pinned=False  # Allow swapping for lifecycle managed tensors
            )

            if success:
                conditional_debug(logger, f"Successfully registered tensor {tensor_id} with swapping system")
                return True
            else:
                logger.warning(f"Failed to register tensor {tensor_id} with swapping system")
                return False
        except Exception as e:
            logger.error(f"Error registering tensor {tensor_id} with swapping system: {e}")
            return False

    def _should_compress_tensor(self, tensor: torch.Tensor, tensor_type: TensorType) -> bool:
        """Determine if a tensor should be compressed"""
        # Compress large tensors or specific types that benefit from compression
        tensor_size = tensor.element_size() * tensor.nelement()

        # Always compress certain types
        if tensor_type in [TensorType.KV_CACHE, TensorType.GRADIENTS]:
            return True

        # Compress tensors larger than 10MB
        if tensor_size > 10 * 1024 * 1024:  # 10MB
            return True

        return False

    def _map_tensor_type_for_tiering(self, tensor_type: TensorType) -> Any:
        """Map our tensor type to tiering system's tensor type"""
        try:
            from memory_management.advanced_memory_tiering_system import TensorType as TieringTensorType

            mapping = {
                TensorType.GENERAL: TieringTensorType.GENERAL,
                TensorType.KV_CACHE: TieringTensorType.KV_CACHE,
                TensorType.IMAGE_FEATURES: TieringTensorType.IMAGE_FEATURES,
                TensorType.TEXT_EMBEDDINGS: TieringTensorType.TEXT_EMBEDDINGS,
                TensorType.GRADIENTS: TieringTensorType.TEMPORARY,
                TensorType.OPTIMIZER_STATE: TieringTensorType.TEMPORARY,
                TensorType.INTERMEDIATE: TieringTensorType.TEMPORARY
            }
            return mapping.get(tensor_type, TieringTensorType.GENERAL)
        except ImportError:
            # If the tiering system is not available, return a generic type
            return "general"

    def _map_tensor_type_for_swapping(self, tensor_type: TensorType) -> Any:
        """Map our tensor type to swapping system's region type"""
        try:
            from memory_management.advanced_memory_swapping_system import MemoryRegionType

            mapping = {
                TensorType.GENERAL: MemoryRegionType.TENSOR_DATA,
                TensorType.KV_CACHE: MemoryRegionType.KV_CACHE,
                TensorType.IMAGE_FEATURES: MemoryRegionType.TENSOR_DATA,
                TensorType.TEXT_EMBEDDINGS: MemoryRegionType.TENSOR_DATA,
                TensorType.GRADIENTS: MemoryRegionType.TENSOR_DATA,
                TensorType.OPTIMIZER_STATE: MemoryRegionType.TENSOR_DATA,
                TensorType.INTERMEDIATE: MemoryRegionType.TEMPORARY
            }
            return mapping.get(tensor_type, MemoryRegionType.TENSOR_DATA)
        except ImportError:
            # If the swapping system is not available, return a generic type
            return "tensor_data"

    def cleanup(self):
        """Clean up all managed resources"""
        self.garbage_collector.cleanup()
        logger.info("Tensor lifecycle manager cleaned up")


def create_optimized_lifecycle_manager(hardware_config: Optional[Dict[str, Any]] = None) -> IntegratedTensorLifecycleManager:
    """
    Factory function to create an optimized lifecycle manager for specific hardware

    Args:
        hardware_config: Hardware configuration with details like:
                        - cpu_model: CPU model string
                        - gpu_model: GPU model string
                        - memory_size: Total system memory in bytes
                        - storage_type: Storage type ('nvme', 'ssd', 'hdd')

    Returns:
        Optimized IntegratedTensorLifecycleManager instance
    """
    if hardware_config is None:
        hardware_config = {
            'cpu_model': 'Intel i5-10210U',
            'gpu_model': 'NVIDIA SM61',
            'memory_size': 8 * 1024 * 1024 * 1024,  # 8GB
            'storage_type': 'nvme'
        }

    manager = IntegratedTensorLifecycleManager(
        hardware_config=hardware_config,
        enable_memory_tiering=True,
        enable_compression=True,
        enable_swapping=True
    )

    logger.info(f"Created optimized lifecycle manager for {hardware_config.get('cpu_model', 'unknown')} "
                f"with {hardware_config.get('storage_type', 'unknown').upper()} storage")

    return manager


# Example usage and integration functions
def integrate_with_existing_systems(lifecycle_manager: IntegratedTensorLifecycleManager,
                                   memory_tiering_system=None,
                                   compression_manager=None,
                                   swapping_system=None):
    """
    Example of how to integrate the lifecycle manager with existing systems
    """
    # Set references to existing systems
    if memory_tiering_system:
        lifecycle_manager.set_memory_tiering_system(memory_tiering_system)

    if compression_manager:
        lifecycle_manager.set_compression_manager(compression_manager)

    if swapping_system:
        lifecycle_manager.set_swapping_system(swapping_system)

    # Example: Tensor allocation with lifecycle management
    def lifecycle_managed_tensor_allocation(shape, dtype=torch.float32,
                                          tensor_type=TensorType.GENERAL,
                                          pinned=False):
        # Create tensor
        tensor = torch.zeros(shape, dtype=dtype)

        # Register with lifecycle manager
        tensor_id = lifecycle_manager.register_tensor(
            tensor,
            tensor_type=tensor_type,
            is_pinned=pinned
        )

        return tensor, tensor_id

    # Example: Access tensor (records usage for prediction)
    def access_lifecycle_managed_tensor(tensor_id):
        success = lifecycle_manager.access_tensor(tensor_id)
        return success

    # Example: Get lifecycle statistics
    def get_lifecycle_stats():
        return lifecycle_manager.get_tensor_lifecycle_stats()

    return lifecycle_managed_tensor_allocation, access_lifecycle_managed_tensor, get_lifecycle_stats


if __name__ == "__main__":
    print("Advanced Predictive Tensor Lifecycle Management System for Qwen3-VL")
    print("=" * 70)

    # Create lifecycle manager optimized for our hardware
    lifecycle_manager = create_optimized_lifecycle_manager({
        'cpu_model': 'Intel i5-10210U',
        'gpu_model': 'NVIDIA SM61',
        'memory_size': 8 * 1024 * 1024 * 1024,  # 8GB
        'storage_type': 'nvme'
    })

    print("\n1. Testing tensor registration with reference counting...")
    # Create and register some tensors with different initial reference counts
    tensor_ids = []
    for i in range(3):
        tensor = torch.randn(100, 100, dtype=torch.float32)
        tensor_id = lifecycle_manager.register_tensor(
            tensor,
            tensor_type=TensorType.GENERAL,
            is_pinned=(i == 0),  # Pin the first tensor
            initial_ref_count=2 if i == 1 else 1  # Second tensor starts with ref count 2
        )
        tensor_ids.append(tensor_id)
        print(f"  Registered tensor {tensor_id} with initial ref count: {2 if i == 1 else 1}")

    # Test reference counting operations
    print("\n2. Testing reference counting operations...")
    # Increment reference for the second tensor
    success = lifecycle_manager.increment_reference(tensor_ids[1], "context_a")
    print(f"  Incremented reference for {tensor_ids[1]}: {success}")

    # Decrement reference for the second tensor
    success = lifecycle_manager.decrement_reference(tensor_ids[1], "context_a")
    print(f"  Decremented reference for {tensor_ids[1]}: {success}")

    # Update reference count by delta for the third tensor
    success = lifecycle_manager.update_reference_count(tensor_ids[2], -1, "context_b")
    print(f"  Updated reference count for {tensor_ids[2]} by -1: {success}")

    # Access some tensors to update their lifecycle state
    print("\n3. Testing tensor access recording...")
    for i, tensor_id in enumerate(tensor_ids):
        # Access the tensor multiple times to build access pattern
        for j in range(i + 2):
            lifecycle_manager.access_tensor(tensor_id, f"context_{j}")
            time.sleep(0.01)  # Small delay to create different access times
        print(f"  Accessed tensor {tensor_id} {i+2} times")

    # Test reference chains
    print("\n4. Testing reference chains...")
    success = lifecycle_manager.add_reference_chain(tensor_ids[0], tensor_ids[1])
    print(f"  Added reference chain {tensor_ids[0]} -> {tensor_ids[1]}: {success}")

    success = lifecycle_manager.add_reference_chain(tensor_ids[1], tensor_ids[2])
    print(f"  Added reference chain {tensor_ids[1]} -> {tensor_ids[2]}: {success}")

    # Show lifecycle statistics
    print("\n5. Lifecycle statistics:")
    stats = lifecycle_manager.get_tensor_lifecycle_stats()
    print(f"  Total tensors: {stats['total_tensors']}")
    print(f"  Pinned tensors: {stats['pinned_tensors']}")
    print(f"  Collections performed: {stats['collections_performed']}")
    print(f"  Tensors collected: {stats['tensors_collected']}")
    print(f"  Memory freed: {stats['memory_freed_bytes'] / (1024**2):.2f} MB")
    print(f"  Average lifetime prediction: {stats['average_lifetime_prediction']:.2f}s")
    print(f"  States: {dict(stats['states'])}")

    # Test hardware-aware optimizations
    print("\n6. Testing hardware-aware optimizations...")
    test_tensor = torch.randn(500, 500, dtype=torch.float32)
    optimal_placement = lifecycle_manager.hardware_manager.optimize_tensor_placement(
        test_tensor, TensorType.GENERAL
    )
    print(f"  Optimal placement for test tensor: {optimal_placement}")

    optimized_tensor = lifecycle_manager.optimize_tensor(test_tensor, TensorType.GENERAL, "matmul")
    print(f"  Tensor optimized for hardware: {optimized_tensor.shape}")

    # Test integration functions (these would connect to actual systems in a real implementation)
    print("\n7. Testing integration capabilities...")
    print("  Integration functions created for memory tiering, compression, and swapping systems")

    # Example of how integration would work (with None for actual systems in this example)
    alloc_func, access_func, stats_func = integrate_with_existing_systems(
        lifecycle_manager,
        memory_tiering_system=None,  # Would be actual tiering system in real use
        compression_manager=None,    # Would be actual compression manager in real use
        swapping_system=None         # Would be actual swapping system in real use
    )

    print("  Integration functions ready for use with existing systems")

    # Cleanup
    lifecycle_manager.cleanup()
    print("\nLifecycle manager cleaned up successfully!")
    print("\nAdvanced Predictive Tensor Lifecycle Management System initialized successfully!")