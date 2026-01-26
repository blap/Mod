"""
Machine Learning Optimization Selector for Inference-PIO

This module implements a machine learning system that automatically selects
the most appropriate optimizations for different models based on input characteristics,
hardware constraints, and performance targets.
"""

import logging
import pickle
import os
import io
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict
import time
import json

from .optimization_manager import get_optimization_manager, OptimizationConfig
from .input_complexity_analyzer import get_complexity_analyzer, ComplexityMetrics
from .optimization_config import ModelFamily


logger = logging.getLogger(__name__)


class OptimizationOutcome(Enum):
    """Enum for possible outcomes of applying an optimization."""
    PERFORMANCE_GAIN = "performance_gain"
    MEMORY_SAVING = "memory_saving"
    LATENCY_REDUCTION = "latency_reduction"
    THROUGHPUT_IMPROVEMENT = "throughput_improvement"
    ENERGY_EFFICIENCY = "energy_efficiency"
    NO_BENEFIT = "no_benefit"
    DEGRADATION = "degradation"


@dataclass
class PerformanceMetrics:
    """Metrics collected during optimization evaluation."""
    latency_ms: float
    memory_usage_mb: float
    throughput_tokens_per_sec: float
    energy_consumption: float
    accuracy_drop: float  # For models where accuracy matters


@dataclass
class OptimizationSelectionData:
    """Data structure for optimization selection training."""
    input_features: np.ndarray  # Features describing input characteristics
    hardware_features: np.ndarray  # Features describing hardware capabilities
    model_features: np.ndarray  # Features describing model architecture
    selected_optimizations: List[str]  # List of optimizations applied
    performance_outcomes: Dict[str, PerformanceMetrics]  # Performance results for each opt
    target_metric: str  # Which metric to optimize for (latency, memory, etc.)
    timestamp: float  # When this data was collected


class MLSampler:
    """Sampler to collect performance data for ML model training."""
    
    def __init__(self):
        self.collected_data: List[OptimizationSelectionData] = []
        self.performance_cache: Dict[str, PerformanceMetrics] = {}
        
    def collect_sample(
        self,
        model: nn.Module,
        input_data: Any,
        available_optimizations: List[str],
        hardware_info: Dict[str, Any],
        target_metric: str = "latency"
    ) -> OptimizationSelectionData:
        """Collect a sample of optimization performance data."""
        
        # Analyze input complexity
        analyzer = get_complexity_analyzer()
        complexity_metrics = analyzer.analyze_input_complexity(input_data)
        
        # Extract model features
        model_features = self._extract_model_features(model)
        
        # Extract hardware features
        hw_features = self._extract_hardware_features(hardware_info)
        
        # Extract input features
        input_features = self._extract_input_features(complexity_metrics)
        
        # Test different optimization combinations
        optimization_combinations = self._generate_optimization_combinations(available_optimizations)
        
        performance_outcomes = {}
        
        for opt_combo in optimization_combinations:
            # Apply optimizations
            optimized_model = self._apply_optimizations_copy(model, opt_combo)
            
            # Benchmark performance
            perf_metrics = self._benchmark_model_performance(optimized_model, input_data)
            performance_outcomes[tuple(opt_combo)] = perf_metrics
            
            # Clean up
            del optimized_model
        
        # Store the data
        sample_data = OptimizationSelectionData(
            input_features=input_features,
            hardware_features=hw_features,
            model_features=model_features,
            selected_optimizations=available_optimizations,
            performance_outcomes=performance_outcomes,
            target_metric=target_metric,
            timestamp=time.time()
        )
        
        self.collected_data.append(sample_data)
        return sample_data
    
    def _extract_model_features(self, model: nn.Module) -> np.ndarray:
        """Extract features from the model architecture."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_layers = len(list(model.modules()))
        
        # Count different layer types
        layer_counts = defaultdict(int)
        for module in model.modules():
            layer_type = type(module).__name__.lower()
            layer_counts[layer_type] += 1
        
        # Extract some key counts
        attention_layers = layer_counts.get('attention', 0) + layer_counts.get('multiheadattention', 0)
        linear_layers = layer_counts.get('linear', 0)
        conv_layers = layer_counts.get('conv2d', 0) + layer_counts.get('conv1d', 0)
        
        return np.array([
            total_params,
            trainable_params,
            num_layers,
            attention_layers,
            linear_layers,
            conv_layers,
            float(total_params > 1e9),  # Large model indicator
            float(conv_layers > 0)     # Has conv layers indicator
        ])
    
    def _extract_hardware_features(self, hardware_info: Dict[str, Any]) -> np.ndarray:
        """Extract features from hardware specifications."""
        gpu_memory_gb = hardware_info.get('gpu_memory_gb', 0)
        cpu_cores = hardware_info.get('cpu_cores', 4)
        cuda_cores = hardware_info.get('cuda_cores', 0)
        compute_capability = hardware_info.get('compute_capability', 0.0)
        is_gpu_available = hardware_info.get('is_gpu_available', False)
        
        return np.array([
            gpu_memory_gb,
            cpu_cores,
            cuda_cores,
            compute_capability,
            float(is_gpu_available)
        ])
    
    def _extract_input_features(self, complexity_metrics) -> np.ndarray:
        """Extract features from input complexity analysis."""
        # Handle both ComplexityMetrics objects and dictionaries
        if hasattr(complexity_metrics, 'sequence_length'):
            # It's a ComplexityMetrics object
            return np.array([
                complexity_metrics.sequence_length,
                complexity_metrics.vocabulary_richness,
                complexity_metrics.syntactic_complexity,
                complexity_metrics.semantic_density,
                complexity_metrics.numerical_content_ratio,
                complexity_metrics.special_character_ratio,
                complexity_metrics.complexity_score
            ])
        else:
            # It's a dictionary-like object
            return np.array([
                getattr(complexity_metrics, 'sequence_length', 0),
                getattr(complexity_metrics, 'vocabulary_richness', 0.0),
                getattr(complexity_metrics, 'syntactic_complexity', 0.0),
                getattr(complexity_metrics, 'semantic_density', 0.0),
                getattr(complexity_metrics, 'numerical_content_ratio', 0.0),
                getattr(complexity_metrics, 'special_character_ratio', 0.0),
                getattr(complexity_metrics, 'complexity_score', 0.0)
            ])
    
    def _generate_optimization_combinations(self, optimizations: List[str]) -> List[List[str]]:
        """Generate different combinations of optimizations to test."""
        # For simplicity, we'll test individual optimizations and some common pairs
        combinations = [[opt] for opt in optimizations]
        
        # Add some common pairs
        for i, opt1 in enumerate(optimizations):
            for j, opt2 in enumerate(optimizations[i+1:], i+1):
                combinations.append([opt1, opt2])
        
        # Add all optimizations combined
        if len(optimizations) > 1:
            combinations.append(optimizations[:])  # Copy of all optimizations
        
        return combinations
    
    def _apply_optimizations_copy(self, model: nn.Module, optimization_names: List[str]) -> nn.Module:
        """Apply optimizations to a copy of the model without modifying the original."""
        # Create a deep copy of the model
        model_copy = self._deep_copy_model(model)
        
        # Apply optimizations
        opt_manager = get_optimization_manager()
        return opt_manager.apply_optimizations(model_copy, optimization_names)
    
    def _deep_copy_model(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model."""
        # Create a new instance of the same model type with same config
        # This is a simplified approach - in practice, you'd need to preserve the constructor args
        try:
            # Attempt to recreate the model with same architecture
            # This is a simplified approach - in practice you'd need to store constructor args
            model_state = model.state_dict()
            new_model = type(model)(**{k: v for k, v in model.__dict__.items()
                                      if not k.startswith('_') and not callable(v)})
            new_model.load_state_dict(model_state)
        except:
            # If recreation fails, just return the original model for this test
            # In a real implementation, we'd need a more robust copying mechanism
            logger.warning("Could not create model copy, using original for testing")
            return model
        return new_model
    
    def _benchmark_model_performance(self, model: nn.Module, input_data: Any) -> PerformanceMetrics:
        """Benchmark the performance of a model with given input."""
        # Warm up
        for _ in range(3):
            with torch.no_grad():
                try:
                    if isinstance(input_data, torch.Tensor):
                        _ = model(input_data)
                    elif isinstance(input_data, dict):
                        _ = model(**input_data)
                    else:
                        # Assume it's a format that can be passed as *args
                        _ = model(input_data)
                except:
                    # If the model doesn't accept this input format, skip warmup
                    pass
        
        # Measure performance
        start_time = time.time()
        memory_start = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        with torch.no_grad():
            start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_event:
                start_event.record()
            
            # Run model
            if isinstance(input_data, torch.Tensor):
                output = model(input_data)
            elif isinstance(input_data, dict):
                output = model(**input_data)
            else:
                output = model(input_data)
            
            if end_event:
                end_event.record()
                end_event.synchronize()
                latency_ms = start_event.elapsed_time(end_event)
            else:
                # Fallback timing
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                latency_ms = (time.time() - start_time) * 1000
        
        memory_end = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_used_mb = (memory_end - memory_start) / (1024 * 1024)
        
        # Calculate throughput (tokens per second)
        if isinstance(output, torch.Tensor):
            tokens_processed = output.numel()
        else:
            tokens_processed = 1  # Fallback
        
        throughput = tokens_processed / max(latency_ms / 1000, 0.001)  # Avoid division by zero
        
        # Energy consumption is estimated (in real implementation, this would come from hardware sensors)
        energy_estimate = latency_ms * 0.1  # Simplified estimate
        
        # Accuracy drop - for now assume 0 for inference models
        accuracy_drop = 0.0
        
        return PerformanceMetrics(
            latency_ms=latency_ms,
            memory_usage_mb=memory_used_mb,
            throughput_tokens_per_sec=throughput,
            energy_consumption=energy_estimate,
            accuracy_drop=accuracy_drop
        )


class MLSuggestionEngine:
    """ML-based engine for suggesting optimal optimizations."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: List[str] = []
        self.is_trained = False
        self.sampler = MLSampler()
        
    def _combine_features(
        self,
        input_features: np.ndarray,
        hardware_features: np.ndarray,
        model_features: np.ndarray
    ) -> np.ndarray:
        """Combine all feature vectors into a single feature vector."""
        return np.concatenate([input_features, hardware_features, model_features])
    
    def train(self, training_data: List[OptimizationSelectionData], target_metric: str = "latency_ms"):
        """Train the ML models on collected data."""
        if not training_data:
            logger.warning("No training data provided for ML suggestion engine")
            return
        
        # Prepare features and targets
        X = []
        y = []
        
        for sample in training_data:
            combined_features = self._combine_features(
                sample.input_features,
                sample.hardware_features,
                sample.model_features
            )
            
            # For each optimization combination, create a training example
            for opt_combo, perf_metrics in sample.performance_outcomes.items():
                # Use the target metric as the prediction target
                if target_metric == "latency_ms":
                    target_value = perf_metrics.latency_ms
                elif target_metric == "memory_usage_mb":
                    target_value = perf_metrics.memory_usage_mb
                elif target_metric == "throughput_tokens_per_sec":
                    target_value = -perf_metrics.throughput_tokens_per_sec  # Negative because we want to maximize
                elif target_metric == "energy_consumption":
                    target_value = perf_metrics.energy_consumption
                elif target_metric == "accuracy_drop":
                    target_value = perf_metrics.accuracy_drop
                else:
                    target_value = perf_metrics.latency_ms  # Default to latency
                
                # Add features for which optimizations are applied
                opt_features = np.zeros(len(sample.selected_optimizations))
                for opt_name in opt_combo:
                    try:
                        idx = sample.selected_optimizations.index(opt_name)
                        opt_features[idx] = 1.0
                    except ValueError:
                        continue
                
                # Combine all features
                full_features = np.concatenate([combined_features, opt_features])
                
                X.append(full_features)
                y.append(target_value)
        
        if not X:
            logger.warning("No valid training samples found")
            return
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train multiple models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        
        best_model_name = None
        best_score = float('inf')
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            
            # Validate
            y_pred = model.predict(X_val_scaled)
            mse = mean_squared_error(y_val, y_pred)
            
            logger.info(f"{name} MSE: {mse:.4f}")
            
            if mse < best_score:
                best_score = mse
                best_model_name = name
        
        if best_model_name:
            self.models[target_metric] = models[best_model_name]
            self.scalers[target_metric] = scaler
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            self.is_trained = True
            
            logger.info(f"Best model for {target_metric}: {best_model_name} with MSE: {best_score:.4f}")
    
    def suggest_optimizations(
        self,
        model: nn.Module,
        input_data: Any,
        available_optimizations: List[str],
        hardware_info: Dict[str, Any],
        target_metric: str = "latency_ms",
        top_k: int = 3
    ) -> List[Tuple[List[str], float]]:
        """Suggest the best optimization combinations for the given inputs."""
        if not self.is_trained or target_metric not in self.models:
            logger.warning(f"Model not trained for {target_metric}, returning default suggestions")
            # Return some reasonable defaults
            return [([opt], 0.0) for opt in available_optimizations[:top_k]]
        
        # Extract features
        analyzer = get_complexity_analyzer()
        complexity_metrics = analyzer.analyze_input_complexity(input_data)
        
        model_features = self._extract_model_features(model)
        hw_features = self._extract_hardware_features(hardware_info)
        input_features = self._extract_input_features(complexity_metrics)
        
        combined_features = self._combine_features(input_features, hw_features, model_features)
        
        # Generate optimization combinations to evaluate
        opt_combinations = self._generate_optimization_combinations(available_optimizations)
        
        predictions = []
        for opt_combo in opt_combinations:
            # Create feature vector with optimization indicators
            opt_features = np.zeros(len(available_optimizations))
            for opt_name in opt_combo:
                try:
                    idx = available_optimizations.index(opt_name)
                    opt_features[idx] = 1.0
                except ValueError:
                    continue
            
            full_features = np.concatenate([combined_features, opt_features]).reshape(1, -1)
            scaled_features = self.scalers[target_metric].transform(full_features)
            
            predicted_value = self.models[target_metric].predict(scaled_features)[0]
            predictions.append((opt_combo, predicted_value))
        
        # Sort by predicted value (lower is better for most metrics)
        predictions.sort(key=lambda x: x[1])
        
        return predictions[:top_k]
    
    def _extract_model_features(self, model: nn.Module) -> np.ndarray:
        """Extract features from the model architecture."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_layers = len(list(model.modules()))
        
        # Count different layer types
        layer_counts = defaultdict(int)
        for module in model.modules():
            layer_type = type(module).__name__.lower()
            layer_counts[layer_type] += 1
        
        # Extract some key counts
        attention_layers = layer_counts.get('attention', 0) + layer_counts.get('multiheadattention', 0)
        linear_layers = layer_counts.get('linear', 0)
        conv_layers = layer_counts.get('conv2d', 0) + layer_counts.get('conv1d', 0)
        
        return np.array([
            total_params,
            trainable_params,
            num_layers,
            attention_layers,
            linear_layers,
            conv_layers,
            float(total_params > 1e9),  # Large model indicator
            float(conv_layers > 0)     # Has conv layers indicator
        ])
    
    def _extract_hardware_features(self, hardware_info: Dict[str, Any]) -> np.ndarray:
        """Extract features from hardware specifications."""
        gpu_memory_gb = hardware_info.get('gpu_memory_gb', 0)
        cpu_cores = hardware_info.get('cpu_cores', 4)
        cuda_cores = hardware_info.get('cuda_cores', 0)
        compute_capability = hardware_info.get('compute_capability', 0.0)
        is_gpu_available = hardware_info.get('is_gpu_available', False)
        
        return np.array([
            gpu_memory_gb,
            cpu_cores,
            cuda_cores,
            compute_capability,
            float(is_gpu_available)
        ])
    
    def _extract_input_features(self, complexity_metrics) -> np.ndarray:
        """Extract features from input complexity analysis."""
        # Handle both ComplexityMetrics objects and dictionaries
        if hasattr(complexity_metrics, 'sequence_length'):
            # It's a ComplexityMetrics object
            return np.array([
                complexity_metrics.sequence_length,
                complexity_metrics.vocabulary_richness,
                complexity_metrics.syntactic_complexity,
                complexity_metrics.semantic_density,
                complexity_metrics.numerical_content_ratio,
                complexity_metrics.special_character_ratio,
                complexity_metrics.complexity_score
            ])
        else:
            # It's a dictionary-like object
            return np.array([
                getattr(complexity_metrics, 'sequence_length', 0),
                getattr(complexity_metrics, 'vocabulary_richness', 0.0),
                getattr(complexity_metrics, 'syntactic_complexity', 0.0),
                getattr(complexity_metrics, 'semantic_density', 0.0),
                getattr(complexity_metrics, 'numerical_content_ratio', 0.0),
                getattr(complexity_metrics, 'special_character_ratio', 0.0),
                getattr(complexity_metrics, 'complexity_score', 0.0)
            ])
    
    def _generate_optimization_combinations(self, optimizations: List[str]) -> List[List[str]]:
        """Generate different combinations of optimizations to test."""
        # For simplicity, we'll test individual optimizations and some common pairs
        combinations = [[opt] for opt in optimizations]
        
        # Add some common pairs
        for i, opt1 in enumerate(optimizations):
            for j, opt2 in enumerate(optimizations[i+1:], i+1):
                combinations.append([opt1, opt2])
        
        # Add all optimizations combined
        if len(optimizations) > 1:
            combinations.append(optimizations[:])  # Copy of all optimizations
        
        return combinations
    
    def save_model(self, filepath: str):
        """Save the trained ML models to disk."""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"ML suggestion engine saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained ML models from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"ML suggestion engine loaded from {filepath}")


class AutoOptimizationSelector:
    """Main class for automatic optimization selection using ML."""
    
    def __init__(self):
        self.ml_engine = MLSuggestionEngine()
        self.performance_history: List[OptimizationSelectionData] = []
        self.model_family = None
        
    def set_model_family(self, model_family: ModelFamily):
        """Set the model family for optimization selection."""
        self.model_family = model_family
    
    def select_optimizations(
        self,
        model: nn.Module,
        input_data: Any,
        hardware_info: Optional[Dict[str, Any]] = None,
        target_metric: str = "latency_ms",
        max_optimizations: int = 5
    ) -> List[str]:
        """Select the best optimizations for the given model and input."""
        if hardware_info is None:
            hardware_info = self._get_default_hardware_info()
        
        # Get available optimizations
        opt_manager = get_optimization_manager()
        available_optimizations = opt_manager.get_available_optimizations()
        
        # Filter optimizations based on model family if known
        if self.model_family == ModelFamily.GLM:
            # GLM-specific optimizations might be preferred
            pass
        elif self.model_family == ModelFamily.QWEN:
            # Qwen-specific optimizations might be preferred
            pass
        
        # Use ML engine to suggest optimizations
        suggestions = self.ml_engine.suggest_optimizations(
            model=model,
            input_data=input_data,
            available_optimizations=available_optimizations,
            hardware_info=hardware_info,
            target_metric=target_metric,
            top_k=max_optimizations
        )
        
        # Return the best single combination (first suggestion)
        if suggestions:
            best_combo, _ = suggestions[0]
            return best_combo
        else:
            # Fallback: return some reasonable defaults
            return ["flash_attention", "kernel_fusion", "adaptive_batching"]
    
    def update_with_performance_feedback(
        self,
        model: nn.Module,
        input_data: Any,
        applied_optimizations: List[str],
        performance_metrics: PerformanceMetrics,
        hardware_info: Optional[Dict[str, Any]] = None
    ):
        """Update the ML model with performance feedback."""
        if hardware_info is None:
            hardware_info = self._get_default_hardware_info()
        
        # Collect sample data
        sample_data = self.ml_engine.sampler.collect_sample(
            model=model,
            input_data=input_data,
            available_optimizations=applied_optimizations,
            hardware_info=hardware_info,
            target_metric="latency"  # Placeholder
        )
        
        # Add to performance history
        self.performance_history.append(sample_data)
        
        # Retrain ML model periodically
        if len(self.performance_history) % 10 == 0:  # Retrain every 10 samples
            self.ml_engine.train(self.performance_history)
    
    def _get_default_hardware_info(self) -> Dict[str, Any]:
        """Get default hardware information."""
        return {
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0,
            'cpu_cores': os.cpu_count() or 4,
            'cuda_cores': 0,  # Would need to query GPU specifics
            'compute_capability': torch.cuda.get_device_capability()[0] + torch.cuda.get_device_capability()[1]/10 if torch.cuda.is_available() else 0.0,
            'is_gpu_available': torch.cuda.is_available()
        }
    
    def train_ml_model(self, training_data: Optional[List[OptimizationSelectionData]] = None):
        """Train the ML model with provided or stored training data."""
        data_to_use = training_data if training_data is not None else self.performance_history
        self.ml_engine.train(data_to_use)
    
    def save_state(self, filepath: str):
        """Save the selector state including ML models."""
        state = {
            'performance_history': self.performance_history,
            'model_family': self.model_family.value if self.model_family else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, default=str)  # Handle non-serializable objects
        
        # Save ML models separately
        ml_filepath = filepath.replace('.json', '_ml.pkl')
        self.ml_engine.save_model(ml_filepath)
        
        logger.info(f"Auto optimization selector state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load the selector state including ML models."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.performance_history = state.get('performance_history', [])
        model_family_str = state.get('model_family')
        if model_family_str:
            self.model_family = ModelFamily(model_family_str)
        
        # Load ML models separately
        ml_filepath = filepath.replace('.json', '_ml.pkl')
        if os.path.exists(ml_filepath):
            self.ml_engine.load_model(ml_filepath)
        
        logger.info(f"Auto optimization selector state loaded from {filepath}")


# Global instance
auto_selector = AutoOptimizationSelector()


def get_auto_selector() -> AutoOptimizationSelector:
    """Get the global auto optimization selector instance."""
    return auto_selector


logger.info("ML Optimization Selector module loaded successfully")


__all__ = [
    "OptimizationOutcome",
    "PerformanceMetrics",
    "OptimizationSelectionData",
    "MLSuggestionEngine",
    "AutoOptimizationSelector",
    "get_auto_selector",
    "auto_selector"
]