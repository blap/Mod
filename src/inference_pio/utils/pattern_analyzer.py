"""
Pattern Analyzer Module

This module provides utilities for analyzing and predicting patterns in data,
which can be used for various optimization purposes in the Inference-PIO system.
"""

import numpy as np
from typing import List, Tuple, Any, Dict, Optional
from datetime import datetime
import time


class PatternAnalyzer:
    """
    Analyzes patterns in data sequences to identify trends and predict future values.
    """
    def __init__(self):
        self.patterns = {}
        self.history = []
    
    def add_data_point(self, key: str, value: Any, timestamp: Optional[float] = None):
        """Add a data point to the analyzer."""
        if timestamp is None:
            timestamp = time.time()
        
        self.history.append({
            'key': key,
            'value': value,
            'timestamp': timestamp
        })
        
        # Keep only recent history to avoid memory issues
        if len(self.history) > 1000:
            self.history = self.history[-500:]
    
    def analyze_trend(self, key: str, window_size: int = 10) -> Dict[str, float]:
        """Analyze trend for a specific key."""
        recent_values = [
            item['value'] for item in self.history 
            if item['key'] == key
        ][-window_size:]
        
        if len(recent_values) < 2:
            return {'slope': 0.0, 'trend_strength': 0.0}
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_values))
        y = np.array(recent_values)
        
        # Calculate slope
        if len(x) > 1:
            slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
        else:
            slope = 0.0
        
        # Calculate trend strength (correlation coefficient)
        if len(x) > 1 and x.std() > 0 and y.std() > 0:
            correlation = np.corrcoef(x, y)[0, 1]
            trend_strength = abs(correlation) if not np.isnan(correlation) else 0.0
        else:
            trend_strength = 0.0
        
        return {
            'slope': float(slope),
            'trend_strength': float(trend_strength)
        }
    
    def predict_next_value(self, key: str, window_size: int = 10) -> Optional[float]:
        """Predict the next value for a specific key based on trend."""
        trend_analysis = self.analyze_trend(key, window_size)
        recent_values = [
            item['value'] for item in self.history 
            if item['key'] == key
        ][-window_size:]
        
        if not recent_values:
            return None
        
        last_value = recent_values[-1]
        predicted_change = trend_analysis['slope']
        
        return float(last_value + predicted_change)
    
    def detect_periodicity(self, key: str, max_period: int = 100) -> Dict[str, Any]:
        """Detect periodic patterns in the data."""
        values = [
            item['value'] for item in self.history 
            if item['key'] == key
        ]
        
        if len(values) < 10:
            return {'period': None, 'strength': 0.0}
        
        # Convert to numpy array for autocorrelation
        values_array = np.array(values)
        n = len(values_array)
        
        # Calculate autocorrelation
        autocorr = []
        for lag in range(1, min(max_period, n//2)):
            corr = np.corrcoef(values_array[:-lag], values_array[lag:])[0, 1]
            if not np.isnan(corr):
                autocorr.append((lag, corr))
            else:
                autocorr.append((lag, 0.0))
        
        if not autocorr:
            return {'period': None, 'strength': 0.0}
        
        # Find the strongest correlation
        best_lag, best_corr = max(autocorr, key=lambda x: x[1])
        
        return {
            'period': best_lag if best_corr > 0.3 else None,  # Threshold for significance
            'strength': float(best_corr)
        }


class AccessPatternPredictor:
    """
    Specialized pattern analyzer for predicting access patterns.
    """
    def __init__(self):
        self.analyzer = PatternAnalyzer()
        self.access_frequency = {}
        self.last_access_times = {}
    
    def record_access(self, key: str, access_type: str = "read"):
        """Record an access event."""
        current_time = time.time()
        
        # Update access frequency
        if key not in self.access_frequency:
            self.access_frequency[key] = 0
        self.access_frequency[key] += 1
        
        # Update last access time
        self.last_access_times[key] = current_time
        
        # Add to analyzer
        self.analyzer.add_data_point(f"access_freq_{key}", self.access_frequency[key], current_time)
    
    def predict_next_access(self) -> List[Tuple[str, float]]:
        """Predict which keys are likely to be accessed next."""
        predictions = []
        
        for key in self.access_frequency.keys():
            # Use frequency trend to predict likelihood of access
            trend = self.analyzer.analyze_trend(f"access_freq_{key}")
            frequency = self.access_frequency[key]
            
            # Combine trend and frequency to get access probability
            access_probability = min(1.0, (frequency * (1 + trend['slope'])) / 10.0)
            access_probability = max(0.0, access_probability)  # Clamp to [0, 1]
            
            predictions.append((key, access_probability))
        
        # Sort by probability
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:10]  # Return top 10 predictions


def create_pattern_analyzer() -> PatternAnalyzer:
    """Factory function to create a pattern analyzer."""
    return PatternAnalyzer()


def create_access_pattern_predictor() -> AccessPatternPredictor:
    """Factory function to create an access pattern predictor."""
    return AccessPatternPredictor()