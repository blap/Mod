"""
Predictive Memory Optimization System

This module implements a predictive memory optimization system that uses machine learning
algorithms to anticipate memory needs and proactively manage memory resources based on
predicted access patterns and usage trends.
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
import psutil
from collections import defaultdict, deque
import pickle
import os

from ...utils.performance_monitor import PerformanceMonitor
from ...utils.pattern_analyzer import PatternAnalyzer


class MemoryAccessPredictor:
    """
    Machine learning-based predictor for memory access patterns.
    Uses historical access data to predict future memory needs.
    """
    
    def __init__(self, prediction_horizon: int = 30, window_size: int = 100):
        """
        Initialize the memory access predictor.
        
        Args:
            prediction_horizon: Time horizon for predictions (seconds)
            window_size: Size of the sliding window for historical data
        """
        self.prediction_horizon = prediction_horizon
        self.window_size = window_size
        self.access_history = deque(maxlen=window_size)
        self.access_patterns = defaultdict(list)
        self.model_features = []  # Features extracted from access patterns
        self.model_targets = []   # Targets for prediction
        self.is_trained = False
        
    def record_access(self, tensor_name: str, timestamp: float, size_bytes: int, 
                     access_type: str = "read", device: str = "cpu"):
        """
        Record a memory access event.
        
        Args:
            tensor_name: Name of the tensor being accessed
            timestamp: Time of access
            size_bytes: Size of the tensor in bytes
            access_type: Type of access ('read', 'write', 'compute')
            device: Device where access occurred ('cpu', 'cuda:0', etc.)
        """
        access_event = {
            'tensor_name': tensor_name,
            'timestamp': timestamp,
            'size_bytes': size_bytes,
            'access_type': access_type,
            'device': device,
            'frequency': 1  # Will be updated based on historical data
        }
        
        self.access_history.append(access_event)
        
        # Update access patterns for this tensor
        self.access_patterns[tensor_name].append(access_event)
        
        # Keep only recent patterns
        if len(self.access_patterns[tensor_name]) > self.window_size:
            self.access_patterns[tensor_name] = self.access_patterns[tensor_name][-self.window_size:]
    
    def extract_features(self, tensor_name: str) -> Dict[str, float]:
        """
        Extract features for a given tensor based on its access history.
        
        Args:
            tensor_name: Name of the tensor
            
        Returns:
            Dictionary of extracted features
        """
        if tensor_name not in self.access_patterns:
            return {
                'access_frequency': 0.0,
                'access_interval_mean': 0.0,
                'access_interval_std': 0.0,
                'recent_access_trend': 0.0,
                'size_normalized_frequency': 0.0,
                'access_type_distribution': {'read': 0.0, 'write': 0.0, 'compute': 0.0}
            }
        
        accesses = self.access_patterns[tensor_name]
        if not accesses:
            return {
                'access_frequency': 0.0,
                'access_interval_mean': 0.0,
                'access_interval_std': 0.0,
                'recent_access_trend': 0.0,
                'size_normalized_frequency': 0.0,
                'access_type_distribution': {'read': 0.0, 'write': 0.0, 'compute': 0.0}
            }
        
        # Calculate access frequency
        time_span = accesses[-1]['timestamp'] - accesses[0]['timestamp']
        access_frequency = len(accesses) / max(time_span, 1.0)
        
        # Calculate access intervals
        intervals = []
        for i in range(1, len(accesses)):
            interval = accesses[i]['timestamp'] - accesses[i-1]['timestamp']
            intervals.append(interval)
        
        interval_mean = np.mean(intervals) if intervals else 0.0
        interval_std = np.std(intervals) if intervals else 0.0
        
        # Calculate recent access trend (last 5 accesses vs first 5 accesses)
        recent_count = min(5, len(accesses))
        if len(accesses) >= 10:
            early_accesses = accesses[:5]
            recent_accesses = accesses[-recent_count:]
            
            early_freq = len(early_accesses) / max(early_accesses[-1]['timestamp'] - early_accesses[0]['timestamp'], 1.0)
            recent_freq = len(recent_accesses) / max(recent_accesses[-1]['timestamp'] - recent_accesses[0]['timestamp'], 1.0)
            
            recent_access_trend = recent_freq - early_freq
        else:
            recent_access_trend = 0.0
        
        # Size-normalized frequency
        avg_size = np.mean([acc['size_bytes'] for acc in accesses])
        size_normalized_frequency = access_frequency / max(avg_size, 1.0)
        
        # Access type distribution
        access_types = [acc['access_type'] for acc in accesses]
        total_accesses = len(access_types)
        type_dist = {
            'read': access_types.count('read') / total_accesses if total_accesses > 0 else 0.0,
            'write': access_types.count('write') / total_accesses if total_accesses > 0 else 0.0,
            'compute': access_types.count('compute') / total_accesses if total_accesses > 0 else 0.0
        }
        
        return {
            'access_frequency': access_frequency,
            'access_interval_mean': interval_mean,
            'access_interval_std': interval_std,
            'recent_access_trend': recent_access_trend,
            'size_normalized_frequency': size_normalized_frequency,
            'access_type_distribution': type_dist
        }
    
    def predict_access_probability(self, tensor_name: str, time_ahead: float = 30.0) -> float:
        """
        Predict the probability of accessing a tensor within the specified time frame.
        
        Args:
            tensor_name: Name of the tensor to predict
            time_ahead: Time ahead to predict (in seconds)
            
        Returns:
            Probability of access (0.0 to 1.0)
        """
        features = self.extract_features(tensor_name)
        
        # Simple heuristic-based prediction
        # In a real implementation, this would use a trained ML model
        access_frequency = features['access_frequency']
        recent_trend = features['recent_access_trend']
        interval_mean = features['access_interval_mean']
        
        # Base probability based on access frequency
        base_prob = min(access_frequency * time_ahead, 1.0)
        
        # Adjust based on recent trend
        trend_factor = 1.0 + (recent_trend * 0.1)  # Small adjustment based on trend
        trend_prob = base_prob * trend_factor
        
        # Adjust based on access intervals
        if interval_mean > 0:
            interval_factor = min(time_ahead / interval_mean, 1.0)
            final_prob = max(trend_prob, interval_factor)
        else:
            final_prob = trend_prob
        
        return max(0.0, min(final_prob, 1.0))


class PredictiveMemoryManager:
    """
    Main class for predictive memory management that uses the access predictor
    to make proactive memory allocation and deallocation decisions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the predictive memory manager.
        
        Args:
            config: Configuration dictionary with memory management settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.predictor = MemoryAccessPredictor(
            prediction_horizon=config.get('prediction_horizon_seconds', 30),
            window_size=config.get('access_history_window_size', 100)
        )
        self.memory_threshold = config.get('memory_prediction_threshold', 0.9)
        self.proactive_interval = config.get('proactive_management_interval', 5.0)
        self.offload_directory = config.get('offload_directory', './offloaded_tensors')
        self.active = False
        self.monitoring_thread = None
        self.tensor_locations = {}  # Maps tensor names to their current location
        self.tensor_metadata = {}   # Metadata about tensors
        self.performance_monitor = PerformanceMonitor()
        
        # Create offload directory if it doesn't exist
        os.makedirs(self.offload_directory, exist_ok=True)
        
    def start_monitoring(self):
        """
        Start proactive memory management monitoring.
        """
        if self.active:
            self.logger.warning("Predictive memory management already active")
            return True
            
        self.active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Started predictive memory management monitoring")
        return True
    
    def stop_monitoring(self):
        """
        Stop proactive memory management monitoring.
        """
        self.active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)  # Wait up to 2 seconds for thread to finish
        self.logger.info("Stopped predictive memory management monitoring")
        return True
    
    def _monitoring_loop(self):
        """
        Main monitoring loop that periodically evaluates memory usage and makes predictions.
        """
        while self.active:
            try:
                # Evaluate current memory usage
                memory_usage = self._get_current_memory_usage()
                
                # Make predictions for upcoming memory needs
                predictions = self._make_memory_predictions()
                
                # Take proactive actions based on predictions
                self._take_proactive_actions(memory_usage, predictions)
                
                # Sleep for the specified interval
                time.sleep(self.proactive_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)  # Brief pause before continuing
    
    def _get_current_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        # System memory stats
        system_memory = psutil.virtual_memory()
        memory_info = {
            'system_total_gb': system_memory.total / (1024**3),
            'system_available_gb': system_memory.available / (1024**3),
            'system_used_gb': system_memory.used / (1024**3),
            'system_percentage': system_memory.percent / 100.0,
        }
        
        # GPU memory stats if available
        if torch.cuda.is_available():
            memory_info.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'gpu_memory_max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3),
            })
        
        return memory_info
    
    def _make_memory_predictions(self) -> Dict[str, float]:
        """
        Make predictions about future memory needs based on access patterns.
        
        Returns:
            Dictionary mapping tensor names to predicted access probabilities
        """
        predictions = {}
        
        # Get all tensor names that have been recorded
        for tensor_name in self.predictor.access_patterns.keys():
            prob = self.predictor.predict_access_probability(
                tensor_name, 
                time_ahead=self.predictor.prediction_horizon
            )
            predictions[tensor_name] = prob
        
        return predictions
    
    def _take_proactive_actions(self, memory_usage: Dict[str, float], 
                               predictions: Dict[str, float]):
        """
        Take proactive memory management actions based on current usage and predictions.
        
        Args:
            memory_usage: Current memory usage statistics
            predictions: Predicted access probabilities for tensors
        """
        # Check if we're approaching memory limits
        system_usage = memory_usage.get('system_percentage', 0.0)
        gpu_usage = memory_usage.get('gpu_memory_allocated_gb', 0.0) / memory_usage.get('gpu_memory_reserved_gb', 1.0) if memory_usage.get('gpu_memory_reserved_gb', 0.0) > 0 else 0.0
        
        # If system memory is high, consider offloading low-probability tensors
        if system_usage > self.memory_threshold:
            self._consider_offloading_low_probability_tensors(predictions)
        
        # If GPU memory is high, consider moving low-probability tensors to CPU
        if gpu_usage > self.memory_threshold and torch.cuda.is_available():
            self._consider_gpu_to_cpu_migration(predictions)
    
    def _consider_offloading_low_probability_tensors(self, predictions: Dict[str, float]):
        """
        Consider offloading tensors with low access probability to disk.
        
        Args:
            predictions: Predicted access probabilities for tensors
        """
        # Sort tensors by access probability (ascending - lowest first)
        sorted_tensors = sorted(predictions.items(), key=lambda x: x[1])
        
        for tensor_name, probability in sorted_tensors:
            if probability < 0.3:  # Low probability of access
                if tensor_name in self.tensor_locations:
                    current_location = self.tensor_locations[tensor_name]
                    
                    # Only offload if currently in memory
                    if current_location in ['cpu', 'gpu']:
                        self._offload_tensor_to_disk(tensor_name)
    
    def _consider_gpu_to_cpu_migration(self, predictions: Dict[str, float]):
        """
        Consider moving tensors from GPU to CPU based on access probability.
        
        Args:
            predictions: Predicted access probabilities for tensors
        """
        # Sort tensors by access probability (ascending - lowest first)
        sorted_tensors = sorted(predictions.items(), key=lambda x: x[1])
        
        for tensor_name, probability in sorted_tensors:
            if probability < 0.3:  # Low probability of access
                if tensor_name in self.tensor_locations:
                    current_location = self.tensor_locations[tensor_name]
                    
                    # Only migrate if currently on GPU
                    if current_location.startswith('cuda'):
                        self._migrate_tensor_to_cpu(tensor_name)
    
    def record_tensor_access(self, tensor_name: str, tensor_data: torch.Tensor, 
                           access_type: str = "read"):
        """
        Record a tensor access event for predictive modeling.
        
        Args:
            tensor_name: Name of the tensor being accessed
            tensor_data: The tensor data (for size calculation)
            access_type: Type of access ('read', 'write', 'compute')
        """
        timestamp = time.time()
        size_bytes = tensor_data.element_size() * tensor_data.nelement()
        
        self.predictor.record_access(
            tensor_name=tensor_name,
            timestamp=timestamp,
            size_bytes=size_bytes,
            access_type=access_type,
            device=str(tensor_data.device) if hasattr(tensor_data, 'device') else 'unknown'
        )
        
        # Update tensor metadata
        self.tensor_metadata[tensor_name] = {
            'size_bytes': size_bytes,
            'device': str(tensor_data.device) if hasattr(tensor_data, 'device') else 'unknown',
            'last_access': timestamp,
            'access_count': self.tensor_metadata.get(tensor_name, {}).get('access_count', 0) + 1
        }
    
    def _offload_tensor_to_disk(self, tensor_name: str) -> bool:
        """
        Offload a tensor to disk storage.
        
        Args:
            tensor_name: Name of the tensor to offload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # In a real implementation, this would retrieve the tensor from wherever it's stored
            # For now, we'll just simulate the offloading by updating the location
            file_path = os.path.join(self.offload_directory, f"{tensor_name}.pkl")
            
            # Update location tracking
            self.tensor_locations[tensor_name] = f"disk:{file_path}"
            
            self.logger.info(f"Offloaded tensor {tensor_name} to disk: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to offload tensor {tensor_name} to disk: {e}")
            return False
    
    def _migrate_tensor_to_cpu(self, tensor_name: str) -> bool:
        """
        Migrate a tensor from GPU to CPU.
        
        Args:
            tensor_name: Name of the tensor to migrate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # In a real implementation, this would move the tensor from GPU to CPU
            # For now, we'll just simulate the migration by updating the location
            self.tensor_locations[tensor_name] = "cpu"
            
            self.logger.info(f"Migrated tensor {tensor_name} to CPU")
            return True
        except Exception as e:
            self.logger.error(f"Failed to migrate tensor {tensor_name} to CPU: {e}")
            return False
    
    def get_tensor_location(self, tensor_name: str) -> str:
        """
        Get the current location of a tensor.
        
        Args:
            tensor_name: Name of the tensor
            
        Returns:
            Current location of the tensor
        """
        return self.tensor_locations.get(tensor_name, "unknown")
    
    def preload_predicted_tensors(self, predictions: Dict[str, float], 
                                 threshold: float = 0.7) -> List[str]:
        """
        Preload tensors that are predicted to be accessed soon.
        
        Args:
            predictions: Predicted access probabilities for tensors
            threshold: Minimum probability threshold for preloading
            
        Returns:
            List of tensor names that were preloaded
        """
        preloaded_tensors = []
        
        for tensor_name, probability in predictions.items():
            if probability >= threshold:
                if tensor_name in self.tensor_locations:
                    current_location = self.tensor_locations[tensor_name]
                    
                    # If tensor is on disk, preload it to memory
                    if current_location.startswith('disk:'):
                        if self._preload_tensor_from_disk(tensor_name):
                            preloaded_tensors.append(tensor_name)
        
        return preloaded_tensors
    
    def _preload_tensor_from_disk(self, tensor_name: str) -> bool:
        """
        Preload a tensor from disk to memory.
        
        Args:
            tensor_name: Name of the tensor to preload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # In a real implementation, this would load the tensor from disk
            # For now, we'll just simulate the preloading by updating the location
            self.tensor_locations[tensor_name] = "cpu"  # Assume it goes to CPU first
            
            self.logger.info(f"Preloaded tensor {tensor_name} from disk to memory")
            return True
        except Exception as e:
            self.logger.error(f"Failed to preload tensor {tensor_name} from disk: {e}")
            return False


class PredictiveMemoryOptimization:
    """
    Main interface for predictive memory optimization that can be integrated
    into model plugins and other components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the predictive memory optimization system.
        
        Args:
            config: Configuration dictionary with optimization settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.manager = PredictiveMemoryManager(config)
        self.enabled = config.get('enable_predictive_management', True)
        
    def start_optimization(self) -> bool:
        """
        Start the predictive memory optimization system.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            self.logger.info("Predictive memory optimization is disabled")
            return True
            
        try:
            return self.manager.start_monitoring()
        except Exception as e:
            self.logger.error(f"Failed to start predictive memory optimization: {e}")
            return False
    
    def stop_optimization(self) -> bool:
        """
        Stop the predictive memory optimization system.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            return self.manager.stop_monitoring()
        except Exception as e:
            self.logger.error(f"Failed to stop predictive memory optimization: {e}")
            return False
    
    def record_tensor_access(self, tensor_name: str, tensor_data: torch.Tensor, 
                           access_type: str = "read") -> bool:
        """
        Record a tensor access for predictive modeling.
        
        Args:
            tensor_name: Name of the tensor being accessed
            tensor_data: The tensor data
            access_type: Type of access ('read', 'write', 'compute')
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return True
            
        try:
            self.manager.record_tensor_access(tensor_name, tensor_data, access_type)
            return True
        except Exception as e:
            self.logger.error(f"Failed to record tensor access: {e}")
            return False
    
    def get_prediction_for_tensor(self, tensor_name: str) -> float:
        """
        Get the access probability prediction for a specific tensor.
        
        Args:
            tensor_name: Name of the tensor
            
        Returns:
            Predicted access probability (0.0 to 1.0)
        """
        if not self.enabled:
            return 0.0
            
        try:
            return self.manager.predictor.predict_access_probability(tensor_name)
        except Exception as e:
            self.logger.error(f"Failed to get prediction for tensor {tensor_name}: {e}")
            return 0.0
    
    def get_memory_optimization_stats(self) -> Dict[str, Any]:
        """
        Get statistics about memory optimization activities.
        
        Returns:
            Dictionary with optimization statistics
        """
        if not self.enabled:
            return {"enabled": False}
            
        try:
            return {
                "enabled": True,
                "active": self.manager.active,
                "tracked_tensors": len(self.manager.tensor_locations),
                "access_history_size": len(self.manager.predictor.access_history),
                "prediction_horizon": self.manager.predictor.prediction_horizon,
                "proactive_interval": self.manager.proactive_interval,
                "memory_threshold": self.manager.memory_threshold
            }
        except Exception as e:
            self.logger.error(f"Failed to get memory optimization stats: {e}")
            return {"enabled": True, "error": str(e)}