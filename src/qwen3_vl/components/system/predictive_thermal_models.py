"""
Predictive Thermal Modeling System
Comprehensive implementation of predictive thermal modeling to anticipate and prevent thermal issues.
"""
import time
import threading
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import statistics
from collections import deque, defaultdict
import math
from power_management import PowerConstraint
from thermal_management import ThermalManager, ThermalZone, CoolingDevice


@dataclass
class ThermalPredictionResult:
    """Result of thermal prediction with confidence and metadata."""
    predicted_temp: float
    confidence: float
    prediction_horizon: float = 30.0  # seconds
    algorithm_used: str = "ensemble"  # lstm, statistical, ensemble
    error_margin: float = 0.0

    def __post_init__(self):
        """Validate the ThermalPredictionResult after initialization."""
        if not isinstance(self.predicted_temp, (int, float)) or math.isnan(self.predicted_temp):
            raise ValueError(f"predicted_temp must be a valid number, got {self.predicted_temp}")

        if not isinstance(self.confidence, (int, float)) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

        if not isinstance(self.prediction_horizon, (int, float)) or self.prediction_horizon <= 0:
            raise ValueError(f"prediction_horizon must be positive, got {self.prediction_horizon}")

        if not isinstance(self.algorithm_used, str):
            raise ValueError(f"algorithm_used must be a string, got {type(self.algorithm_used)}")

        if not isinstance(self.error_margin, (int, float)) or self.error_margin < 0:
            raise ValueError(f"error_margin must be non-negative, got {self.error_margin}")


class PredictionAlgorithm(Enum):
    """Available prediction algorithms."""
    LSTM = "lstm"
    STATISTICAL = "statistical"
    ENSEMBLE = "ensemble"


class TemperatureForecastingModel:
    """
    Temperature forecasting model using multiple algorithms.
    Combines LSTM-based forecasting with statistical models and advanced ML techniques.
    """

    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        self.lstm_model = None  # Will be initialized when needed
        self.arima_model = None  # ARIMA model for time series forecasting
        self.models = {
            'lstm': self._predict_lstm,
            'arima': self._predict_arima,
            'statistical': self.predict_temperature_statistical,
            'ensemble': self.predict_temperature_ensemble
        }

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # For statistical model
        self.temp_history = deque(maxlen=100)
        self.feature_history = deque(maxlen=100)

    def _prepare_sequence_data(self, features: List[List[float]], sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequence data for LSTM training/prediction."""
        # Validate inputs
        if not isinstance(features, list):
            raise TypeError(f"features must be a list, got {type(features)}")

        if not isinstance(sequence_length, int) or sequence_length <= 0:
            raise ValueError(f"sequence_length must be a positive integer, got {sequence_length}")

        if len(features) == 0:
            return np.array([]), np.array([])

        if len(features) < sequence_length + 1:
            return np.array([]), np.array([])

        # Validate that all feature vectors have the same length
        if not all(len(f) == len(features[0]) for f in features):
            raise ValueError("All feature vectors must have the same length")

        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append([features[i + sequence_length][0]])  # Predict temperature (first feature)

        return np.array(X), np.array(y)

    def train_lstm_model(self, training_data: List[List[float]]):
        """
        Train LSTM model for temperature prediction.
        training_data format: [[temp, cpu_usage, power, ...], ...]
        """
        try:
            # Only import TensorFlow if needed (to avoid dependency issues)
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam

            X, y = self._prepare_sequence_data(training_data, self.sequence_length)

            if len(X) == 0:
                self.logger.warning("Not enough data to train LSTM model")
                return

            # Normalize the data
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            X_std[X_std == 0] = 1  # Avoid division by zero
            y_mean = np.mean(y)
            y_std = np.std(y)
            if y_std == 0:
                y_std = 1

            X_norm = (X - X_mean) / X_std
            y_norm = (y - y_mean) / y_std

            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])

            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

            # Train the model
            model.fit(X_norm, y_norm, batch_size=1, epochs=10, verbose=0)

            self.lstm_model = {
                'model': model,
                'X_mean': X_mean,
                'X_std': X_std,
                'y_mean': y_mean,
                'y_std': y_std
            }

            self.logger.info("LSTM model trained successfully")

        except ImportError:
            self.logger.warning("TensorFlow not available, LSTM model will not be trained")
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {e}")

    def _predict_lstm(self, recent_data: List[List[float]], prediction_horizon: float = 60.0) -> ThermalPredictionResult:
        """Predict temperature using LSTM model."""
        if self.lstm_model is None or len(recent_data) < self.sequence_length:
            # Fallback to statistical model
            return self.predict_temperature_statistical(recent_data)

        try:
            import tensorflow as tf

            # Prepare input sequence
            if len(recent_data) < self.sequence_length:
                # Pad with the last known values if not enough data
                needed = self.sequence_length - len(recent_data)
                padded_data = recent_data[:needed] + recent_data
                input_seq = np.array([padded_data[-self.sequence_length:]])
            else:
                input_seq = np.array([recent_data[-self.sequence_length:]])

            # Normalize input
            X_norm = (input_seq - self.lstm_model['X_mean']) / self.lstm_model['X_std']

            # Make prediction
            pred_norm = self.lstm_model['model'].predict(X_norm, verbose=0)

            # Denormalize output
            predicted_temp = pred_norm[0][0] * self.lstm_model['y_std'] + self.lstm_model['y_mean']

            # Calculate confidence based on prediction variance and model complexity
            # For now, use a fixed confidence for LSTM (can be enhanced with dropout uncertainty)
            confidence = 0.85  # High confidence for LSTM when model exists

            return ThermalPredictionResult(
                predicted_temp=float(predicted_temp),
                confidence=confidence,
                prediction_horizon=prediction_horizon,
                algorithm_used="lstm"
            )

        except Exception as e:
            self.logger.error(f"Error in LSTM prediction: {e}")
            # Fallback to statistical model
            return self.predict_temperature_statistical(recent_data)

    def _predict_arima(self, recent_data: List[List[float]], prediction_horizon: float = 60.0) -> ThermalPredictionResult:
        """Predict temperature using ARIMA model."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            import warnings
            warnings.filterwarnings("ignore")

            # Extract temperatures from recent data (first column)
            temperatures = [data[0] for data in recent_data]

            if len(temperatures) < 10:
                # Not enough data for ARIMA, fallback to statistical
                return self.predict_temperature_statistical(recent_data)

            # Fit ARIMA model
            # Use a simple (1,1,1) model as default, can be optimized based on data
            model = ARIMA(temperatures, order=(1, 1, 1))
            fitted_model = model.fit()

            # Forecast
            forecast_result = fitted_model.forecast(steps=1)
            predicted_temp = forecast_result[0]

            # Calculate confidence interval
            conf_int = fitted_model.get_forecast(steps=1).conf_int()
            lower_bound = conf_int.iloc[0, 0]
            upper_bound = conf_int.iloc[0, 1]

            # Calculate confidence based on prediction interval width
            interval_width = upper_bound - lower_bound
            confidence = max(0.1, min(0.95, 1.0 - interval_width / 20.0))  # Normalize based on expected range

            # Ensure predicted temperature is reasonable
            predicted_temp = max(20.0, min(100.0, predicted_temp))

            return ThermalPredictionResult(
                predicted_temp=float(predicted_temp),
                confidence=confidence,
                prediction_horizon=prediction_horizon,
                algorithm_used="arima",
                error_margin=(upper_bound - lower_bound) / 2
            )

        except ImportError:
            self.logger.warning("Statsmodels not available, ARIMA model will not be used")
            # Fallback to statistical model
            return self.predict_temperature_statistical(recent_data)
        except Exception as e:
            self.logger.error(f"Error in ARIMA prediction: {e}")
            # Fallback to statistical model
            return self.predict_temperature_statistical(recent_data)

    def predict_temperature_statistical(self, recent_data: List[List[float]], prediction_horizon: float = 60.0) -> ThermalPredictionResult:
        """Predict temperature using statistical model."""
        # Validate inputs
        if not isinstance(recent_data, list):
            raise TypeError(f"recent_data must be a list, got {type(recent_data)}")

        if not isinstance(prediction_horizon, (int, float)) or prediction_horizon <= 0:
            raise ValueError(f"prediction_horizon must be positive, got {prediction_horizon}")

        if len(recent_data) < 2:
            return ThermalPredictionResult(
                predicted_temp=60.0,  # Default safe temperature
                confidence=0.1,
                prediction_horizon=prediction_horizon,
                algorithm_used="statistical"
            )

        # Validate that all data points have at least one value (temperature)
        if not all(len(data_point) > 0 for data_point in recent_data):
            raise ValueError("Each data point must have at least one value (temperature)")

        # Extract temperatures from recent data (first column)
        temperatures = [data[0] for data in recent_data]

        # Validate that temperatures are numeric
        if not all(isinstance(temp, (int, float)) for temp in temperatures):
            raise ValueError("All temperatures must be numeric values")

        # Calculate trend using linear regression
        n = len(temperatures)
        x = list(range(n))

        # Calculate slope and intercept
        sum_x = sum(x)
        sum_y = sum(temperatures)
        sum_x2 = sum(i * i for i in x)
        sum_xy = sum(i * temp for i, temp in zip(x, temperatures))

        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        else:
            slope = 0

        # Calculate intercept
        intercept = (sum_y - slope * sum_x) / n

        # Predict next temperature based on trend
        next_time = n
        predicted_temp = slope * next_time + intercept

        # Calculate confidence based on consistency of trend
        if len(temperatures) >= 3:
            # Calculate variance of residuals
            residuals = [temp - (slope * i + intercept) for i, temp in enumerate(temperatures)]
            if len(residuals) > 0:
                variance = sum(r ** 2 for r in residuals) / len(residuals)
                std_dev = math.sqrt(variance)

                # Confidence decreases with higher variance
                confidence = max(0.1, min(0.9, 1.0 - std_dev / 10.0))
            else:
                confidence = 0.5
        else:
            confidence = 0.3  # Lower confidence with less data

        # Ensure predicted temperature is reasonable
        predicted_temp = max(20.0, min(100.0, predicted_temp))  # Clamp between 20-100°C

        return ThermalPredictionResult(
            predicted_temp=predicted_temp,
            confidence=confidence,
            prediction_horizon=prediction_horizon,
            algorithm_used="statistical"
        )

    def predict_temperature_ensemble(self, recent_data: List[List[float]], prediction_horizon: float = 60.0) -> ThermalPredictionResult:
        """Predict temperature using ensemble of models."""
        if len(recent_data) < 2:
            return self.predict_temperature_statistical(recent_data)

        # Get predictions from all available models
        predictions = []

        # Statistical prediction
        stat_result = self.predict_temperature_statistical(recent_data, prediction_horizon)
        predictions.append(('statistical', stat_result.predicted_temp, stat_result.confidence))

        # Try LSTM if available
        if self.lstm_model is not None:
            lstm_result = self._predict_lstm(recent_data, prediction_horizon)
            predictions.append(('lstm', lstm_result.predicted_temp, lstm_result.confidence))

        # Try ARIMA if available
        try:
            import statsmodels
            arima_result = self._predict_arima(recent_data, prediction_horizon)
            predictions.append(('arima', arima_result.predicted_temp, arima_result.confidence))
        except ImportError:
            pass  # Statsmodels not available

        # Weighted average based on confidence
        if len(predictions) == 1:
            # Only one prediction available
            model_name, temp, conf = predictions[0]
            return ThermalPredictionResult(
                predicted_temp=temp,
                confidence=conf,
                prediction_horizon=prediction_horizon,
                algorithm_used=model_name
            )
        elif len(predictions) > 1:
            # Calculate weighted average
            weighted_sum = sum(temp * conf for _, temp, conf in predictions)
            total_confidence = sum(conf for _, _, conf in predictions)

            if total_confidence > 0:
                combined_temp = weighted_sum / total_confidence
                combined_confidence = sum(conf for _, _, conf in predictions) / len(predictions)

                # Use the model with highest confidence as the algorithm used
                best_model = max(predictions, key=lambda x: x[2])[0]

                return ThermalPredictionResult(
                    predicted_temp=combined_temp,
                    confidence=combined_confidence,
                    prediction_horizon=prediction_horizon,
                    algorithm_used=best_model
                )

        # Fallback to statistical if no models worked
        return self.predict_temperature_statistical(recent_data, prediction_horizon)

    def predict_temperature(self, recent_data: List[List[float]], algorithm: PredictionAlgorithm = PredictionAlgorithm.ENSEMBLE,
                           prediction_horizon: float = 60.0) -> ThermalPredictionResult:
        """Predict temperature using specified algorithm."""
        if algorithm == PredictionAlgorithm.LSTM:
            return self._predict_lstm(recent_data, prediction_horizon)
        elif algorithm == PredictionAlgorithm.STATISTICAL:
            return self.predict_temperature_statistical(recent_data, prediction_horizon)
        else:  # ENSEMBLE
            return self.predict_temperature_ensemble(recent_data, prediction_horizon)


class WorkloadThermalAnalyzer:
    """
    Analyzes workload patterns and predicts their thermal impact.
    """
    
    def __init__(self):
        self.workload_patterns = defaultdict(lambda: {
            'count': 0,
            'total_duration': 0,
            'avg_cpu_usage': 0,
            'avg_gpu_usage': 0,
            'avg_power': 0,
            'avg_temp_rise': 0,
            'thermal_impact_samples': []
        })
        
        self.thermal_impact_profiles = {}
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    def update_workload_pattern(self, workload_data: Dict[str, Any]):
        """Update workload pattern with new data."""
        workload_type = workload_data.get('workload_type', 'unknown')
        
        pattern = self.workload_patterns[workload_type]
        pattern['count'] += 1
        
        # Update averages incrementally
        duration = workload_data.get('duration', 0)
        pattern['total_duration'] += duration
        
        cpu_usage = workload_data.get('cpu_usage_avg', 0)
        old_avg_cpu = pattern['avg_cpu_usage']
        pattern['avg_cpu_usage'] = ((old_avg_cpu * (pattern['count'] - 1) + cpu_usage) / 
                                   pattern['count'])
        
        gpu_usage = workload_data.get('gpu_usage_avg', 0)
        old_avg_gpu = pattern['avg_gpu_usage']
        pattern['avg_gpu_usage'] = ((old_avg_gpu * (pattern['count'] - 1) + gpu_usage) / 
                                   pattern['count'])
        
        power = workload_data.get('power_avg', 0)
        old_avg_power = pattern['avg_power']
        pattern['avg_power'] = ((old_avg_power * (pattern['count'] - 1) + power) / 
                               pattern['count'])
        
        # Store thermal impact if available
        temp_rise = workload_data.get('temp_rise', 0)
        if temp_rise != 0:
            pattern['avg_temp_rise'] = ((pattern['avg_temp_rise'] * (pattern['count'] - 1) + temp_rise) / 
                                       pattern['count'])
            pattern['thermal_impact_samples'].append({
                'compute_intensity': workload_data.get('compute_intensity', 0),
                'memory_bandwidth': workload_data.get('memory_bandwidth', 0),
                'duration': duration,
                'temp_rise': temp_rise
            })
    
    def predict_thermal_impact(self, workload_type: str, compute_intensity: float, 
                              memory_bandwidth: float, duration: float) -> Dict[str, float]:
        """Predict thermal impact of a workload."""
        pattern = self.workload_patterns.get(workload_type, None)
        
        if pattern is None or len(pattern['thermal_impact_samples']) < 3:
            # No historical data, use default estimates based on intensity
            cpu_temp_rise = compute_intensity * 10 + memory_bandwidth * 5
            gpu_temp_rise = compute_intensity * 15 + memory_bandwidth * 8
            peak_power = 10 + compute_intensity * 15 + memory_bandwidth * 5
            
            return {
                'cpu_temp_rise': cpu_temp_rise,
                'gpu_temp_rise': gpu_temp_rise,
                'peak_power': peak_power,
                'confidence': 0.3  # Low confidence without historical data
            }
        
        # Use historical data to make prediction
        samples = pattern['thermal_impact_samples']
        
        # Find similar workloads and interpolate
        temp_rises = []
        for sample in samples:
            # Calculate similarity (simpler approach)
            intensity_diff = abs(sample['compute_intensity'] - compute_intensity)
            bandwidth_diff = abs(sample['memory_bandwidth'] - memory_bandwidth)
            duration_diff = abs(sample['duration'] - duration) / sample['duration'] if sample['duration'] > 0 else 0
            
            similarity = 1.0 / (1.0 + intensity_diff + bandwidth_diff + duration_diff)
            temp_rises.append((sample['temp_rise'], similarity))
        
        # Weighted average based on similarity
        if temp_rises:
            weighted_sum = sum(temp * sim for temp, sim in temp_rises)
            similarity_sum = sum(sim for _, sim in temp_rises)
            
            if similarity_sum > 0:
                predicted_temp_rise = weighted_sum / similarity_sum
            else:
                predicted_temp_rise = statistics.mean([temp for temp, _ in temp_rises])
        else:
            predicted_temp_rise = pattern['avg_temp_rise']
        
        # Distribute between CPU and GPU based on typical patterns
        cpu_temp_rise = predicted_temp_rise * 0.6  # 60% to CPU
        gpu_temp_rise = predicted_temp_rise * 0.8  # 80% to GPU (more compute-intensive)
        
        # Estimate power based on intensity
        peak_power = pattern['avg_power'] * (0.5 + compute_intensity * 0.5 + memory_bandwidth * 0.3)
        
        # Calculate confidence based on number of similar samples
        confidence = min(0.9, len(samples) * 0.1)
        
        return {
            'cpu_temp_rise': cpu_temp_rise,
            'gpu_temp_rise': gpu_temp_rise,
            'peak_power': peak_power,
            'confidence': confidence
        }


class ThermalProfileCollector:
    """
    Collects thermal profile data for training predictive models.
    """
    
    def __init__(self, max_profile_size: int = 1000):
        self.profile_data = deque(maxlen=max_profile_size)
        self.max_profile_size = max_profile_size
        self.profile_stats = {
            'cpu_temp_stats': {'mean': 0, 'std': 0, 'min': float('inf'), 'max': float('-inf')},
            'gpu_temp_stats': {'mean': 0, 'std': 0, 'min': float('inf'), 'max': float('-inf')},
            'power_stats': {'mean': 0, 'std': 0, 'min': float('inf'), 'max': float('-inf')},
            'workload_correlations': {},
            'environmental_factors': {},
            'thermal_resistance_factors': {},  # How quickly temps change
            'cooling_efficiency': {}  # How effectively cooling works
        }

        # For calculating thermal resistance and cooling efficiency
        self.temp_change_history = deque(maxlen=50)
        self.cooling_effectiveness_history = deque(maxlen=50)

        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    def collect_thermal_profile(self, thermal_data: Dict[str, Any]):
        """Collect thermal profile data point."""
        # Validate input
        if not isinstance(thermal_data, dict):
            raise TypeError(f"thermal_data must be a dictionary, got {type(thermal_data)}")

        # Add timestamp if not present
        if 'timestamp' not in thermal_data:
            thermal_data['timestamp'] = time.time()

        # Validate timestamp
        if not isinstance(thermal_data['timestamp'], (int, float)):
            raise TypeError("timestamp must be numeric")

        # Calculate thermal resistance factors if we have previous data
        if self.profile_data:
            prev_data = self.profile_data[-1]
            current_time = thermal_data['timestamp']
            prev_time = prev_data['timestamp']
            time_diff = current_time - prev_time

            if time_diff > 0:
                # Calculate temperature change rates
                cpu_temp = thermal_data.get('cpu_temp', 0)
                prev_cpu_temp = prev_data.get('cpu_temp', 0)
                gpu_temp = thermal_data.get('gpu_temp', 0)
                prev_gpu_temp = prev_data.get('gpu_temp', 0)

                # Validate temperatures
                if not isinstance(cpu_temp, (int, float)) or not isinstance(prev_cpu_temp, (int, float)):
                    raise TypeError("CPU temperatures must be numeric")

                if not isinstance(gpu_temp, (int, float)) or not isinstance(prev_gpu_temp, (int, float)):
                    raise TypeError("GPU temperatures must be numeric")

                cpu_temp_change = cpu_temp - prev_cpu_temp
                gpu_temp_change = gpu_temp - prev_gpu_temp

                # Calculate thermal resistance factors
                cpu_thermal_resistance = cpu_temp_change / max(0.001, time_diff)  # degrees per second
                gpu_thermal_resistance = gpu_temp_change / max(0.001, time_diff)

                # Store in history for statistical analysis
                temp_change_entry = {
                    'timestamp': current_time,
                    'cpu_rate': cpu_thermal_resistance,
                    'gpu_rate': gpu_thermal_resistance,
                    'cpu_power': thermal_data.get('cpu_power', 0),
                    'gpu_power': thermal_data.get('gpu_power', 0),
                    'cpu_usage': thermal_data.get('cpu_usage', 0),
                    'gpu_usage': thermal_data.get('gpu_usage', 0)
                }

                # Validate numeric values in temp_change_entry
                for key, value in temp_change_entry.items():
                    if key != 'timestamp':  # timestamp already validated
                        if not isinstance(value, (int, float)):
                            raise TypeError(f"{key} must be numeric, got {type(value)}")

                self.temp_change_history.append(temp_change_entry)

        self.profile_data.append(thermal_data)

        # Update statistics
        self._update_statistics(thermal_data)
    
    def _update_statistics(self, thermal_data: Dict[str, Any]):
        """Update running statistics."""
        # Validate input
        if not isinstance(thermal_data, dict):
            raise TypeError(f"thermal_data must be a dictionary, got {type(thermal_data)}")

        # Update CPU temperature stats
        cpu_temp = thermal_data.get('cpu_temp', 0)
        if not isinstance(cpu_temp, (int, float)):
            raise TypeError(f"cpu_temp must be numeric, got {type(cpu_temp)}")

        if cpu_temp != 0:
            old_mean = self.profile_stats['cpu_temp_stats']['mean']
            old_count = len(self.profile_data) - 1 if len(self.profile_data) > 1 else 1
            new_count = len(self.profile_data)

            # Update mean incrementally
            new_mean = (old_mean * old_count + cpu_temp) / new_count
            self.profile_stats['cpu_temp_stats']['mean'] = new_mean

            # Update min/max
            self.profile_stats['cpu_temp_stats']['min'] = min(
                self.profile_stats['cpu_temp_stats']['min'], cpu_temp
            )
            self.profile_stats['cpu_temp_stats']['max'] = max(
                self.profile_stats['cpu_temp_stats']['max'], cpu_temp
            )

        # Update GPU temperature stats
        gpu_temp = thermal_data.get('gpu_temp', 0)
        if not isinstance(gpu_temp, (int, float)):
            raise TypeError(f"gpu_temp must be numeric, got {type(gpu_temp)}")

        if gpu_temp != 0:
            old_mean = self.profile_stats['gpu_temp_stats']['mean']
            old_count = len(self.profile_data) - 1 if len(self.profile_data) > 1 else 1
            new_count = len(self.profile_data)

            new_mean = (old_mean * old_count + gpu_temp) / new_count
            self.profile_stats['gpu_temp_stats']['mean'] = new_mean

            self.profile_stats['gpu_temp_stats']['min'] = min(
                self.profile_stats['gpu_temp_stats']['min'], gpu_temp
            )
            self.profile_stats['gpu_temp_stats']['max'] = max(
                self.profile_stats['gpu_temp_stats']['max'], gpu_temp
            )

        # Update power stats
        cpu_power = thermal_data.get('cpu_power', 0)
        gpu_power = thermal_data.get('gpu_power', 0)

        if not isinstance(cpu_power, (int, float)):
            raise TypeError(f"cpu_power must be numeric, got {type(cpu_power)}")

        if not isinstance(gpu_power, (int, float)):
            raise TypeError(f"gpu_power must be numeric, got {type(gpu_power)}")

        total_power = cpu_power + gpu_power

        if total_power > 0:
            old_mean = self.profile_stats['power_stats']['mean']
            old_count = len(self.profile_data) - 1 if len(self.profile_data) > 1 else 1
            new_count = len(self.profile_data)

            new_mean = (old_mean * old_count + total_power) / new_count
            self.profile_stats['power_stats']['mean'] = new_mean

            self.profile_stats['power_stats']['min'] = min(
                self.profile_stats['power_stats']['min'], total_power
            )
            self.profile_stats['power_stats']['max'] = max(
                self.profile_stats['power_stats']['max'], total_power
            )
    
    def get_thermal_profile_statistics(self) -> Dict[str, Any]:
        """Get comprehensive thermal profile statistics."""
        # Calculate standard deviations
        if len(self.profile_data) > 1:
            cpu_temps = [data['cpu_temp'] for data in self.profile_data if 'cpu_temp' in data]
            if cpu_temps:
                self.profile_stats['cpu_temp_stats']['std'] = statistics.stdev(cpu_temps)

            gpu_temps = [data['gpu_temp'] for data in self.profile_data if 'gpu_temp' in data]
            if gpu_temps:
                self.profile_stats['gpu_temp_stats']['std'] = statistics.stdev(gpu_temps)

            powers = [data.get('cpu_power', 0) + data.get('gpu_power', 0) for data in self.profile_data]
            if powers:
                self.profile_stats['power_stats']['std'] = statistics.stdev(powers)

        # Analyze workload correlations
        workload_types = [data.get('workload_type', 'unknown') for data in self.profile_data]
        unique_workloads = set(workload_types)

        workload_correlations = {}
        for wtype in unique_workloads:
            temps = [data['cpu_temp'] for data in self.profile_data
                    if data.get('workload_type') == wtype and 'cpu_temp' in data]
            if temps:
                workload_correlations[wtype] = {
                    'avg_temp': statistics.mean(temps),
                    'count': len(temps),
                    'temp_range': (min(temps), max(temps))
                }

        self.profile_stats['workload_correlations'] = workload_correlations

        # Environmental factors
        env_temps = [data.get('environment_temp', 25.0) for data in self.profile_data]
        if env_temps:
            self.profile_stats['environmental_factors'] = {
                'avg_env_temp': statistics.mean(env_temps),
                'env_temp_range': (min(env_temps), max(env_temps)),
                'correlation_with_cpu_temp': self._calculate_correlation(
                    [data.get('environment_temp', 25.0) for data in self.profile_data],
                    [data.get('cpu_temp', 0) for data in self.profile_data if 'cpu_temp' in data]
                )
            }

        # Add thermal resistance and cooling efficiency analysis
        self.profile_stats['thermal_resistance_factors'] = self.analyze_thermal_resistance()
        self.profile_stats['cooling_efficiency'] = self.analyze_cooling_efficiency()

        return self.profile_stats
    
    def _calculate_correlation(self, x_vals: List[float], y_vals: List[float]) -> float:
        """Calculate correlation between two sets of values."""
        # Filter out entries where either value is missing
        pairs = [(x, y) for x, y in zip(x_vals, y_vals) if x is not None and y is not None]
        
        if len(pairs) < 2:
            return 0.0
        
        x_vals, y_vals = zip(*pairs)
        
        # Calculate means
        x_mean = sum(x_vals) / len(x_vals)
        y_mean = sum(y_vals) / len(y_vals)
        
        # Calculate numerator and denominators for correlation
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
        sum_x_sq = sum((x - x_mean) ** 2 for x in x_vals)
        sum_y_sq = sum((y - y_mean) ** 2 for y in y_vals)
        
        if sum_x_sq == 0 or sum_y_sq == 0:
            return 0.0
        
        correlation = numerator / math.sqrt(sum_x_sq * sum_y_sq)
        return correlation
    
    def export_profile_data(self) -> List[Dict[str, Any]]:
        """Export thermal profile data for external analysis."""
        return list(self.profile_data)
    
    def get_training_data(self) -> List[List[float]]:
        """Get formatted training data for ML models."""
        training_data = []

        for data_point in self.profile_data:
            # Format: [temp, cpu_usage, gpu_usage, cpu_power, gpu_power, env_temp]
            temp = data_point.get('cpu_temp', 0)
            cpu_usage = data_point.get('cpu_usage', 0)
            gpu_usage = data_point.get('gpu_usage', 0)
            cpu_power = data_point.get('cpu_power', 0)
            gpu_power = data_point.get('gpu_power', 0)
            env_temp = data_point.get('environment_temp', 25.0)

            training_data.append([temp, cpu_usage, gpu_usage, cpu_power, gpu_power, env_temp])

        return training_data

    def analyze_thermal_resistance(self) -> Dict[str, float]:
        """Analyze thermal resistance factors for the system."""
        if len(self.temp_change_history) < 10:
            return {
                'cpu_thermal_resistance': 0.05,  # Default value
                'gpu_thermal_resistance': 0.05,
                'confidence': 0.3
            }

        # Calculate average thermal resistance for CPU and GPU
        cpu_rates = [entry['cpu_rate'] for entry in self.temp_change_history if abs(entry['cpu_rate']) > 0.001]
        gpu_rates = [entry['gpu_rate'] for entry in self.temp_change_history if abs(entry['gpu_rate']) > 0.001]

        cpu_resistance = statistics.mean(cpu_rates) if cpu_rates else 0.05
        gpu_resistance = statistics.mean(gpu_rates) if gpu_rates else 0.05

        # Calculate confidence based on data consistency
        cpu_confidence = 0.9 if len(cpu_rates) > 20 else 0.3 + (len(cpu_rates) / 20) * 0.6
        gpu_confidence = 0.9 if len(gpu_rates) > 20 else 0.3 + (len(gpu_rates) / 20) * 0.6

        return {
            'cpu_thermal_resistance': abs(cpu_resistance),
            'gpu_thermal_resistance': abs(gpu_resistance),
            'cpu_confidence': min(cpu_confidence, 0.95),
            'gpu_confidence': min(gpu_confidence, 0.95)
        }

    def analyze_cooling_efficiency(self) -> Dict[str, float]:
        """Analyze cooling system efficiency."""
        # This would analyze how effectively cooling systems reduce temperature
        # based on fan speeds, ambient temperature, and actual temperature changes

        if len(self.temp_change_history) < 5:
            return {
                'cpu_cooling_efficiency': 0.7,  # Default value
                'gpu_cooling_efficiency': 0.7,
                'confidence': 0.3
            }

        # Placeholder for cooling efficiency analysis
        # In a real implementation, this would look at:
        # - Fan speeds vs temperature changes
        # - Power consumption vs cooling effectiveness
        # - Time constants for cooling response

        return {
            'cpu_cooling_efficiency': 0.75,
            'gpu_cooling_efficiency': 0.70,
            'confidence': 0.8
        }

    def get_optimal_cooling_settings(self, target_temp: float, current_temp: float) -> Dict[str, float]:
        """Calculate optimal cooling settings to reach target temperature."""
        # Analyze thermal characteristics to determine optimal cooling
        thermal_stats = self.analyze_thermal_resistance()
        cooling_stats = self.analyze_cooling_efficiency()

        # Calculate temperature difference
        temp_diff = target_temp - current_temp

        # Calculate required cooling based on thermal resistance
        if temp_diff < 0:  # Need to cool down
            # Determine how much cooling is needed based on thermal resistance
            cpu_cooling_needed = abs(temp_diff) * thermal_stats['cpu_thermal_resistance'] / cooling_stats['cpu_cooling_efficiency']
            gpu_cooling_needed = abs(temp_diff) * thermal_stats['gpu_thermal_resistance'] / cooling_stats['gpu_cooling_efficiency']

            # Convert to fan percentages (0-100)
            cpu_fan_setting = min(100, max(20, int(50 + cpu_cooling_needed * 100)))
            gpu_fan_setting = min(100, max(20, int(50 + gpu_cooling_needed * 100)))
        else:  # Need to heat up (rare case, but for completeness)
            cpu_fan_setting = 20  # Minimal cooling
            gpu_fan_setting = 20  # Minimal cooling

        return {
            'cpu_fan_setting': cpu_fan_setting,
            'gpu_fan_setting': gpu_fan_setting,
            'estimated_time_to_target': abs(temp_diff) / max(0.1, thermal_stats['cpu_thermal_resistance']),
            'confidence': min(thermal_stats['cpu_confidence'], cooling_stats['confidence'])
        }

    def generate_thermal_model_parameters(self) -> Dict[str, Any]:
        """Generate parameters for thermal modeling based on collected data."""
        stats = self.get_thermal_profile_statistics()

        # Generate thermal model parameters
        thermal_params = {
            'cpu_thermal_constant': stats['thermal_resistance_factors']['cpu_thermal_resistance'],
            'gpu_thermal_constant': stats['thermal_resistance_factors']['gpu_thermal_resistance'],
            'cpu_cooling_factor': stats['cooling_efficiency']['cpu_cooling_efficiency'],
            'gpu_cooling_factor': stats['cooling_efficiency']['gpu_cooling_efficiency'],
            'ambient_influence': 0.1,  # How much ambient temp affects internal temp
            'power_to_temp_coefficient': {
                'cpu': 0.5,  # How much CPU power affects CPU temp
                'gpu': 0.8   # How much GPU power affects GPU temp
            }
        }

        return thermal_params


class ProactiveCoolingController:
    """
    Implements proactive cooling strategies based on thermal predictions.
    """

    def __init__(self, constraints: PowerConstraint, prediction_horizon: float = 60.0,
                 preemptive_threshold: float = 0.8, power_efficiency_mode: bool = True):
        self.constraints = constraints
        self.prediction_horizon = prediction_horizon
        self.preemptive_threshold = preemptive_threshold  # 80% of critical temp
        self.power_efficiency_mode = power_efficiency_mode  # Whether to optimize for power efficiency
        self.is_active = False
        self.cooling_thread = None
        self.performance_throttling_enabled = True  # Whether to enable performance throttling

        # Initialize logger
        self.logger = logging.getLogger(__name__)

    def _calculate_cooling_adjustment(self, current_temp: float, predicted_temp: float,
                                     critical_temp: float, prediction_horizon: float,
                                     confidence: float = 0.8) -> Dict[str, Union[int, float, List[str]]]:
        """Calculate appropriate cooling level and actions based on prediction."""
        # Validate inputs
        if not isinstance(current_temp, (int, float)):
            raise TypeError(f"current_temp must be numeric, got {type(current_temp)}")

        if not isinstance(predicted_temp, (int, float)):
            raise TypeError(f"predicted_temp must be numeric, got {type(predicted_temp)}")

        if not isinstance(critical_temp, (int, float)) or critical_temp <= 0:
            raise ValueError(f"critical_temp must be positive, got {critical_temp}")

        if not isinstance(prediction_horizon, (int, float)) or prediction_horizon <= 0:
            raise ValueError(f"prediction_horizon must be positive, got {prediction_horizon}")

        if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {confidence}")

        # Calculate how close we are to critical temperature
        current_ratio = current_temp / critical_temp
        predicted_ratio = predicted_temp / critical_temp

        # Calculate time to critical temperature
        temp_rate = max(0.001, (predicted_temp - current_temp) / (prediction_horizon / 60))  # degrees per minute
        time_to_critical = (critical_temp - current_temp) / temp_rate if temp_rate > 0 else float('inf')

        # Determine cooling strategy based on prediction
        cooling_actions = []

        # If predicted temperature is approaching critical, increase cooling aggressively
        if predicted_ratio > 0.95:  # Critical threshold
            cooling_level = 98
            cooling_actions.extend([
                "Maximum cooling activation",
                "Performance throttling if enabled",
                "Immediate power reduction"
            ])
        elif predicted_ratio > 0.9:  # Approaching critical
            cooling_level = 90
            cooling_actions.extend([
                "High cooling activation",
                "Prepare for performance throttling"
            ])
        elif predicted_ratio > 0.8:  # Approaching threshold
            cooling_level = 75
            cooling_actions.append("Increase cooling")
        elif predicted_ratio > 0.7:  # Approaching safe zone
            cooling_level = 60
            cooling_actions.append("Moderate cooling")
        elif predicted_ratio > 0.6:  # Safe but warming
            cooling_level = 50
            cooling_actions.append("Maintain cooling")
        else:  # Safe temperature
            cooling_level = max(25, int(40 - (0.6 - predicted_ratio) * 100))  # Reduce cooling to save power
            cooling_actions.append("Reduce cooling for power efficiency")

        # Adjust cooling based on prediction confidence
        if confidence < 0.5:
            # If confidence is low, be more conservative
            cooling_level = max(40, int(cooling_level * 0.8))
            cooling_actions.append("Low prediction confidence - conservative cooling")

        # Apply power efficiency considerations
        if self.power_efficiency_mode and cooling_level > 70:
            # In power efficiency mode, limit aggressive cooling
            cooling_level = min(cooling_level, 80)
            cooling_actions.append("Power efficiency mode - limiting maximum cooling")

        # Calculate performance throttling if enabled
        throttling_factor = 0.0
        if self.performance_throttling_enabled:
            if predicted_ratio > 0.95:
                throttling_factor = 0.4  # Reduce performance by 40%
            elif predicted_ratio > 0.9:
                throttling_factor = 0.25  # Reduce performance by 25%
            elif predicted_ratio > 0.8:
                throttling_factor = 0.15  # Reduce performance by 15%
            elif predicted_ratio > 0.7:
                throttling_factor = 0.05  # Reduce performance by 5%

        return {
            'cooling_level': cooling_level,
            'throttling_factor': throttling_factor,
            'time_to_critical': time_to_critical,
            'actions': cooling_actions
        }

    def get_cooling_recommendation(self, thermal_zone_name: str, current_temp: float,
                                  predictor) -> Dict[str, Any]:
        """Get cooling recommendation based on thermal prediction."""
        # Get prediction for this zone
        prediction_result = predictor.predict_temperature(thermal_zone_name,
                                                        prediction_horizon=self.prediction_horizon)

        # Calculate cooling adjustment
        cooling_adjustment = self._calculate_cooling_adjustment(
            current_temp=current_temp,
            predicted_temp=prediction_result.predicted_temp,
            critical_temp=self.constraints.max_cpu_temp_celsius if 'CPU' in thermal_zone_name
                         else self.constraints.max_gpu_temp_celsius,
            prediction_horizon=self.prediction_horizon,
            confidence=prediction_result.confidence
        )

        # Get current cooling level (this would come from thermal manager in real usage)
        current_cooling_level = 50  # Default value

        return {
            'zone_name': thermal_zone_name,
            'current_temp': current_temp,
            'predicted_temp': prediction_result.predicted_temp,
            'current_cooling_level': current_cooling_level,
            'recommended_cooling_level': cooling_adjustment['cooling_level'],
            'recommended_throttling_factor': cooling_adjustment['throttling_factor'],
            'confidence': prediction_result.confidence,
            'time_to_action': self.prediction_horizon,
            'time_to_critical': cooling_adjustment['time_to_critical'],
            'algorithm_used': prediction_result.algorithm_used,
            'recommended_actions': cooling_adjustment['actions']
        }

    def preemptive_cooling_action(self, thermal_manager: ThermalManager, predictor):
        """Execute preemptive cooling based on predictions."""
        # Get current thermal state
        thermal_zones = thermal_manager.get_thermal_state()

        for zone in thermal_zones:
            # Get prediction for this zone
            prediction_result = predictor.predict_temperature(zone.name,
                                                           prediction_horizon=self.prediction_horizon)

            # Calculate and apply cooling adjustment
            cooling_adjustment = self._calculate_cooling_adjustment(
                current_temp=zone.current_temp,
                predicted_temp=prediction_result.predicted_temp,
                critical_temp=zone.critical_temp,
                prediction_horizon=self.prediction_horizon,
                confidence=prediction_result.confidence
            )

            # Apply cooling adjustment
            thermal_manager._set_cooling_level(zone.name, cooling_adjustment['cooling_level'])

            # Apply performance throttling if needed
            if cooling_adjustment['throttling_factor'] > 0:
                thermal_manager._reduce_performance(cooling_adjustment['throttling_factor'])

            self.logger.info(f"Applied preemptive cooling for {zone.name}: "
                           f"current={zone.current_temp}°C, "
                           f"predicted={prediction_result.predicted_temp}°C, "
                           f"cooling={cooling_adjustment['cooling_level']}%, "
                           f"throttling={cooling_adjustment['throttling_factor']*100:.1f}%")

    def adaptive_cooling_strategy(self, thermal_manager: ThermalManager, predictor,
                                 workload_analyzer: WorkloadThermalAnalyzer = None):
        """Apply adaptive cooling based on workload predictions and thermal analysis."""
        # Get current thermal state
        thermal_zones = thermal_manager.get_thermal_state()

        for zone in thermal_zones:
            # Get prediction for this zone
            prediction_result = predictor.predict_temperature(zone.name,
                                                           prediction_horizon=self.prediction_horizon)

            # Calculate cooling adjustment
            cooling_adjustment = self._calculate_cooling_adjustment(
                current_temp=zone.current_temp,
                predicted_temp=prediction_result.predicted_temp,
                critical_temp=zone.critical_temp,
                prediction_horizon=self.prediction_horizon,
                confidence=prediction_result.confidence
            )

            # Apply workload-aware adjustments if analyzer is provided
            if workload_analyzer:
                # Get upcoming workload info if available
                # This would typically come from a workload scheduler
                pass  # Placeholder for workload-aware cooling

            # Apply cooling adjustment
            thermal_manager._set_cooling_level(zone.name, cooling_adjustment['cooling_level'])

            # Apply performance throttling if needed
            if cooling_adjustment['throttling_factor'] > 0:
                thermal_manager._reduce_performance(cooling_adjustment['throttling_factor'])

            self.logger.info(f"Adaptive cooling for {zone.name}: "
                           f"level={cooling_adjustment['cooling_level']}%, "
                           f"throttling={cooling_adjustment['throttling_factor']*100:.1f}%")

    def start_proactive_cooling(self, thermal_manager: ThermalManager, predictor,
                               workload_analyzer: WorkloadThermalAnalyzer = None,
                               check_interval: float = 10.0):
        """Start proactive cooling loop."""
        if self.is_active:
            return

        self.is_active = True

        def cooling_loop():
            while self.is_active:
                try:
                    if workload_analyzer:
                        self.adaptive_cooling_strategy(thermal_manager, predictor, workload_analyzer)
                    else:
                        self.preemptive_cooling_action(thermal_manager, predictor)
                    time.sleep(check_interval)
                except Exception as e:
                    self.logger.error(f"Error in proactive cooling loop: {e}")
                    time.sleep(check_interval)

        self.cooling_thread = threading.Thread(target=cooling_loop)
        self.cooling_thread.daemon = True
        self.cooling_thread.start()

        self.logger.info("Started proactive cooling controller")

    def stop_proactive_cooling(self):
        """Stop proactive cooling."""
        self.is_active = False
        if self.cooling_thread:
            self.cooling_thread.join(timeout=2.0)

        self.logger.info("Stopped proactive cooling controller")

    def enable_performance_throttling(self, enabled: bool = True):
        """Enable or disable performance throttling."""
        self.performance_throttling_enabled = enabled
        self.logger.info(f"Performance throttling {'enabled' if enabled else 'disabled'}")

    def set_power_efficiency_mode(self, enabled: bool):
        """Enable or disable power efficiency mode."""
        self.power_efficiency_mode = enabled
        self.logger.info(f"Power efficiency mode {'enabled' if enabled else 'disabled'}")


class ThermalPredictor:
    """
    Main predictive thermal modeling system that combines all components.
    """
    
    def __init__(self, constraints: PowerConstraint, prediction_horizon: float = 60.0):
        self.constraints = constraints
        self.prediction_horizon = prediction_horizon
        
        # Initialize components
        self.forecasting_model = TemperatureForecastingModel()
        self.workload_analyzer = WorkloadThermalAnalyzer()
        self.profile_collector = ThermalProfileCollector()
        
        # Historical data for predictions
        self.historical_data = deque(maxlen=500)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    def update_with_thermal_data(self, thermal_data: Dict[str, Any]):
        """Update the predictor with new thermal data."""
        # Add timestamp if not present
        if 'timestamp' not in thermal_data:
            thermal_data['timestamp'] = time.time()
        
        self.historical_data.append(thermal_data)
        self.profile_collector.collect_thermal_profile(thermal_data)
    
    def predict_temperature(self, thermal_zone_name: str, 
                          prediction_horizon: float = None,
                          algorithm: PredictionAlgorithm = PredictionAlgorithm.ENSEMBLE) -> ThermalPredictionResult:
        """Predict temperature for a specific thermal zone."""
        if prediction_horizon is None:
            prediction_horizon = self.prediction_horizon
        
        # Prepare features for prediction
        # Format: [temp, cpu_usage, gpu_usage, cpu_power, gpu_power, ...]
        features = []
        
        for data_point in self.historical_data:
            if thermal_zone_name == 'CPU':
                temp = data_point.get('cpu_temp', 0)
                cpu_usage = data_point.get('cpu_usage', 0)
                gpu_usage = data_point.get('gpu_usage', 0)
                cpu_power = data_point.get('cpu_power', 0)
                gpu_power = data_point.get('gpu_power', 0)
            elif thermal_zone_name == 'GPU':
                temp = data_point.get('gpu_temp', 0)
                cpu_usage = data_point.get('cpu_usage', 0)
                gpu_usage = data_point.get('gpu_usage', 0)
                cpu_power = data_point.get('cpu_power', 0)
                gpu_power = data_point.get('gpu_power', 0)
            else:
                # Default to CPU data if unknown zone
                temp = data_point.get('cpu_temp', 0)
                cpu_usage = data_point.get('cpu_usage', 0)
                gpu_usage = data_point.get('gpu_usage', 0)
                cpu_power = data_point.get('cpu_power', 0)
                gpu_power = data_point.get('gpu_power', 0)
            
            features.append([temp, cpu_usage, gpu_usage, cpu_power, gpu_power])
        
        if len(features) < 2:
            # Not enough data for meaningful prediction
            return ThermalPredictionResult(
                predicted_temp=60.0,  # Default safe temperature
                confidence=0.1,
                prediction_horizon=prediction_horizon,
                algorithm_used="statistical"
            )
        
        # Use the forecasting model to predict
        return self.forecasting_model.predict_temperature(
            features, 
            algorithm=algorithm, 
            prediction_horizon=prediction_horizon
        )
    
    def get_thermal_risk_assessment(self) -> Dict[str, Any]:
        """Get comprehensive thermal risk assessment."""
        if len(self.historical_data) < 10:
            # Not enough data for assessment
            return {
                'cpu_risk_level': 'normal',
                'gpu_risk_level': 'normal',
                'prediction_horizon': self.prediction_horizon,
                'recommended_actions': ['Continue monitoring'],
                'confidence': 0.3
            }
        
        # Get latest temperatures
        latest_data = self.historical_data[-1]
        cpu_temp = latest_data.get('cpu_temp', 0)
        gpu_temp = latest_data.get('gpu_temp', 0)
        
        # Get predictions
        cpu_prediction = self.predict_temperature('CPU')
        gpu_prediction = self.predict_temperature('GPU')
        
        # Assess risk levels
        cpu_risk_level = self._assess_thermal_risk(
            current_temp=cpu_temp,
            predicted_temp=cpu_prediction.predicted_temp,
            critical_temp=self.constraints.max_cpu_temp_celsius
        )
        
        gpu_risk_level = self._assess_thermal_risk(
            current_temp=gpu_temp,
            predicted_temp=gpu_prediction.predicted_temp,
            critical_temp=self.constraints.max_gpu_temp_celsius
        )
        
        # Generate recommended actions based on risk
        recommended_actions = self._generate_recommended_actions(
            cpu_risk_level, gpu_risk_level, cpu_prediction.confidence, gpu_prediction.confidence
        )
        
        # Overall confidence is the average of individual confidences
        overall_confidence = (cpu_prediction.confidence + gpu_prediction.confidence) / 2
        
        return {
            'cpu_risk_level': cpu_risk_level,
            'gpu_risk_level': gpu_risk_level,
            'prediction_horizon': self.prediction_horizon,
            'recommended_actions': recommended_actions,
            'confidence': overall_confidence,
            'cpu_prediction': cpu_prediction,
            'gpu_prediction': gpu_prediction
        }
    
    def _assess_thermal_risk(self, current_temp: float, predicted_temp: float, 
                           critical_temp: float) -> str:
        """Assess thermal risk level."""
        # Use the higher of current or predicted temperature
        temp_to_assess = max(current_temp, predicted_temp)
        
        ratio = temp_to_assess / critical_temp
        
        if ratio >= 0.95:
            return 'critical'
        elif ratio >= 0.85:
            return 'warning'
        elif ratio >= 0.7:
            return 'caution'
        else:
            return 'normal'
    
    def _generate_recommended_actions(self, cpu_risk: str, gpu_risk: str, 
                                    cpu_conf: float, gpu_conf: float) -> List[str]:
        """Generate recommended actions based on risk assessment."""
        actions = []
        
        # Add actions based on CPU risk
        if cpu_risk == 'critical':
            actions.extend([
                'Reduce CPU load immediately',
                'Increase CPU cooling to maximum',
                'Consider CPU frequency scaling down'
            ])
        elif cpu_risk == 'warning':
            actions.extend([
                'Monitor CPU temperature closely',
                'Prepare to reduce CPU load if temperature rises',
                'Increase CPU cooling if possible'
            ])
        elif cpu_risk == 'caution':
            actions.append('Continue monitoring CPU temperature')
        
        # Add actions based on GPU risk
        if gpu_risk == 'critical':
            actions.extend([
                'Reduce GPU load immediately',
                'Increase GPU cooling to maximum',
                'Consider reducing GPU compute intensity'
            ])
        elif gpu_risk == 'warning':
            actions.extend([
                'Monitor GPU temperature closely',
                'Prepare to reduce GPU load if temperature rises',
                'Increase GPU cooling if possible'
            ])
        elif gpu_risk == 'caution':
            actions.append('Continue monitoring GPU temperature')
        
        # Add general recommendations based on confidence
        if min(cpu_conf, gpu_conf) < 0.5:
            actions.append('Collect more thermal data to improve prediction accuracy')
        
        if not actions:
            actions.append('System thermal state is normal')
        
        return actions


# Integration with existing thermal management system
class PredictiveThermalManager(ThermalManager):
    """
    Extended thermal manager with predictive capabilities.
    """

    def __init__(self, constraints: PowerConstraint,
                 prediction_horizon: float = 60.0,
                 use_profiling: bool = True):
        super().__init__(constraints)

        # Add predictive components
        self.predictor = ThermalPredictor(constraints, prediction_horizon)
        self.cooling_controller = ProactiveCoolingController(constraints, prediction_horizon)
        self.workload_analyzer = WorkloadThermalAnalyzer()

        # Enable profiling if requested
        self.use_profiling = use_profiling
        if use_profiling:
            self.profile_collector = ThermalProfileCollector()

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Store previous thermal state for change detection
        self.previous_thermal_state = {}

    def get_thermal_state(self):
        """Get current thermal state and update predictor."""
        zones = super().get_thermal_state()

        # Gather system metrics for comprehensive thermal data
        try:
            import psutil
            cpu_usage = psutil.cpu_percent(interval=0.1)

            # Get CPU frequency for power estimation
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                freq_ratio = cpu_freq.current / max(cpu_freq.max, 1.0)
            else:
                freq_ratio = 1.0

            # Get memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent

        except ImportError:
            # If psutil not available, use defaults
            cpu_usage = 50.0
            freq_ratio = 1.0
            memory_usage = 50.0

        # Get GPU information if available
        gpu_info = self._get_gpu_info()
        gpu_usage = gpu_info.get('gpu_util', 0) if gpu_info else 0
        gpu_power = gpu_info.get('power_draw', 0) if gpu_info else 0

        # Calculate power estimates using existing power models if available
        try:
            from power_estimation_models import IntelI5_10210UPowerModel, NVidiaSM61PowerModel
            cpu_power_model = IntelI5_10210UPowerModel()
            gpu_power_model = NVidiaSM61PowerModel()

            cpu_power = cpu_power_model.estimate_power(cpu_usage/100.0, freq_ratio)
        except ImportError:
            # Use simple estimates if power models not available
            cpu_power = cpu_usage * 0.25  # Simple linear estimate

        # Update predictor with current data
        thermal_data = {
            'timestamp': time.time(),
            'cpu_temp': next((z.current_temp for z in zones if z.zone_type == 'CPU'), 40.0),
            'gpu_temp': next((z.current_temp for z in zones if z.zone_type == 'GPU'), 40.0),
            'cpu_usage': cpu_usage,
            'gpu_usage': gpu_usage,
            'cpu_power': cpu_power,
            'gpu_power': gpu_power,
            'memory_usage': memory_usage,
            'environment_temp': 25.0  # Default ambient temperature
        }

        # Update all predictive components
        self.predictor.update_with_thermal_data(thermal_data)

        if self.use_profiling:
            self.profile_collector.collect_thermal_profile(thermal_data)

        # Store for change detection
        self.previous_thermal_state = thermal_data.copy()

        return zones

    def get_predictive_thermal_summary(self) -> Dict[str, Any]:
        """Get thermal summary with predictive elements."""
        base_summary = self.get_thermal_summary()
        risk_assessment = self.predictor.get_thermal_risk_assessment()

        return {
            **base_summary,
            'predictive_risk_assessment': risk_assessment,
            'predictions': {
                'cpu_prediction': self.predictor.predict_temperature('CPU'),
                'gpu_prediction': self.predictor.predict_temperature('GPU')
            },
            'profile_statistics': self.profile_collector.get_thermal_profile_statistics() if self.use_profiling else {}
        }

    def start_predictive_management(self, monitoring_interval: float = 1.0,
                                   proactive_cooling: bool = True):
        """Start predictive thermal management."""
        # Start the base thermal management
        super().start_management(monitoring_interval)

        # Start proactive cooling if enabled
        if proactive_cooling:
            self.cooling_controller.start_proactive_cooling(
                thermal_manager=self,
                predictor=self.predictor,
                workload_analyzer=self.workload_analyzer,
                check_interval=monitoring_interval
            )

        self.logger.info("Started predictive thermal management")

    def stop_predictive_management(self):
        """Stop predictive thermal management."""
        # Stop proactive cooling
        self.cooling_controller.stop_proactive_cooling()

        # Stop base thermal management
        super().stop_management()

        self.logger.info("Stopped predictive thermal management")

    def register_workload_for_analysis(self, workload_data: Dict[str, Any]):
        """Register workload data for thermal impact analysis."""
        self.workload_analyzer.update_workload_pattern(workload_data)

    def predict_workload_thermal_impact(self, workload_type: str, compute_intensity: float,
                                       memory_bandwidth: float, duration: float) -> Dict[str, float]:
        """Predict thermal impact of a specific workload."""
        return self.workload_analyzer.predict_thermal_impact(
            workload_type, compute_intensity, memory_bandwidth, duration
        )

    def get_optimal_operating_settings(self) -> Dict[str, Any]:
        """Get optimal operating settings based on thermal profile."""
        if not self.use_profiling:
            return {
                'cpu_fan_speed': 50,
                'gpu_fan_speed': 50,
                'power_limit': self.constraints.max_cpu_power_watts,
                'suggested_workload_limit': 0.8
            }

        # Get current temperatures
        zones = self.get_thermal_state()
        cpu_temp = next((z.current_temp for z in zones if z.zone_type == 'CPU'), 60.0)

        # Calculate optimal cooling settings
        optimal_settings = self.profile_collector.get_optimal_cooling_settings(
            target_temp=min(self.constraints.max_cpu_temp_celsius * 0.7, cpu_temp - 5),  # Target safe temp
            current_temp=cpu_temp
        )

        return {
            'cpu_fan_speed': optimal_settings.get('cpu_fan_setting', 50),
            'gpu_fan_speed': optimal_settings.get('gpu_fan_setting', 50),
            'power_limit': self.constraints.max_cpu_power_watts * 0.9,  # 90% of max for safety
            'suggested_workload_limit': 0.85,  # 85% of max for thermal safety
            'confidence': optimal_settings.get('confidence', 0.7)
        }

    def adjust_for_predicted_thermal_events(self):
        """Adjust system settings based on predicted thermal events."""
        risk_assessment = self.predictor.get_thermal_risk_assessment()

        if risk_assessment['cpu_risk_level'] in ['warning', 'critical']:
            # Reduce CPU performance if high risk predicted
            cpu_prediction = risk_assessment['cpu_prediction']
            if cpu_prediction.confidence > 0.6:
                # Calculate required throttling based on predicted temperature
                temp_excess = max(0, cpu_prediction.predicted_temp - self.constraints.max_cpu_temp_celsius * 0.8)
                throttling_factor = min(0.3, temp_excess / 20.0)  # Up to 30% throttling
                self._reduce_performance(throttling_factor)

        if risk_assessment['gpu_risk_level'] in ['warning', 'critical']:
            # Log GPU risk for system awareness
            gpu_prediction = risk_assessment['gpu_prediction']
            self.logger.warning(f"Predicted GPU thermal risk: {gpu_prediction.predicted_temp}°C "
                              f"with {gpu_prediction.confidence:.2f} confidence")


def initialize_predictive_thermal_system(constraints: PowerConstraint = None) -> PredictiveThermalManager:
    """
    Initialize the predictive thermal system with proper configuration.

    Args:
        constraints: Power and thermal constraints for the system.
                    If None, default constraints will be used.

    Returns:
        PredictiveThermalManager instance
    """
    if constraints is None:
        constraints = PowerConstraint()

    # Create the predictive thermal manager
    thermal_manager = PredictiveThermalManager(constraints)

    # Optionally train the forecasting model with historical data if available
    try:
        # Try to load historical data for model training
        # This would typically come from saved profile data
        profile_collector = thermal_manager.profile_collector
        training_data = profile_collector.get_training_data()

        if len(training_data) >= 20:  # Need sufficient data to train
            thermal_manager.predictor.forecasting_model.train_lstm_model(training_data)
            thermal_manager.logger.info("LSTM model trained with historical data")
    except Exception as e:
        thermal_manager.logger.warning(f"Could not train LSTM model: {e}")

    return thermal_manager