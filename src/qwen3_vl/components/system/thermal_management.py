"""
Thermal Management Module for Adaptive Algorithms

This module provides thermal management functionality for adaptive algorithms,
including temperature monitoring, thermal constraint management, and
thermal-aware computing utilities.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Union
import time
import threading
import logging
import psutil


@dataclass
class ThermalReading:
    """
    Represents a single thermal reading from a sensor.

    Attributes:
        sensor_name: Name of the thermal sensor
        temperature_celsius: Temperature reading in Celsius
        timestamp: Timestamp of the measurement
    """
    sensor_name: str
    temperature_celsius: float
    timestamp: float = 0.0

    def __post_init__(self):
        """Validate the thermal reading values."""
        if not self.sensor_name:
            raise ValueError("sensor_name must be non-empty")
        if self.temperature_celsius < -273.15:  # Absolute zero in Celsius
            raise ValueError("temperature_celsius cannot be below absolute zero")


@dataclass
class ThermalConstraint:
    """
    Defines thermal constraints for the system.

    Attributes:
        critical_temperature_celsius: Temperature at which system should take immediate action
        max_safe_temperature_celsius: Maximum safe operating temperature
        min_safe_temperature_celsius: Minimum safe operating temperature (for cold environments)
        thermal_gradient_threshold: Maximum allowed temperature change per minute
    """
    critical_temperature_celsius: float = 90.0
    max_safe_temperature_celsius: float = 80.0
    min_safe_temperature_celsius: float = -10.0
    thermal_gradient_threshold: float = 5.0  # degrees per minute

    def __post_init__(self):
        """Validate the thermal constraint values."""
        if self.critical_temperature_celsius <= self.max_safe_temperature_celsius:
            raise ValueError("critical_temperature_celsius must be greater than max_safe_temperature_celsius")
        if self.max_safe_temperature_celsius <= self.min_safe_temperature_celsius:
            raise ValueError("max_safe_temperature_celsius must be greater than min_safe_temperature_celsius")
        if self.thermal_gradient_threshold < 0:
            raise ValueError("thermal_gradient_threshold must be non-negative")


class ThermalManager:
    """
    Manages thermal state and provides thermal management utilities.
    """
    monitoring_thread: Optional[threading.Thread]

    def __init__(self, constraints: Optional[ThermalConstraint] = None):
        """
        Initialize the thermal manager.

        Args:
            constraints: Thermal constraints for the system (defaults to ThermalConstraint())
        """
        self.constraints = constraints or ThermalConstraint()
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.temperature_history: List[ThermalReading] = []
        self.max_history_size = 100
        self.thermal_lock = threading.Lock()

    def get_current_temperatures(self) -> List[ThermalReading]:
        """
        Get the current temperatures from system thermal sensors.

        Returns:
            List of ThermalReading objects with current temperature readings
        """
        with self.thermal_lock:
            readings = []
            timestamp = time.time()
            
            # Try to get temperatures from available sensors
            try:
                temps = psutil.sensors_temperatures()
                
                if not temps:
                    # If no sensors available, return a default reading
                    self.logger.warning("No temperature sensors found, returning default reading")
                    readings.append(ThermalReading(
                        sensor_name="default",
                        temperature_celsius=30.0,
                        timestamp=timestamp
                    ))
                else:
                    # Process available temperature sensors
                    for name, entries in temps.items():
                        for entry in entries:
                            reading = ThermalReading(
                                sensor_name=f"{name}_{entry.label or 'temp'}",
                                temperature_celsius=entry.current,
                                timestamp=timestamp
                            )
                            readings.append(reading)
                            
                            # Add to history
                            self._add_to_history(reading)
                            
            except AttributeError:
                # sensors_temperatures is not available on this platform
                self.logger.warning("Temperature sensors not available on this platform")
                readings.append(ThermalReading(
                    sensor_name="default",
                    temperature_celsius=30.0,
                    timestamp=timestamp
                ))
            
            return readings

    def _add_to_history(self, reading: ThermalReading) -> None:
        """
        Add a thermal reading to the history.

        Args:
            reading: ThermalReading to add to history
        """
        self.temperature_history.append(reading)
        if len(self.temperature_history) > self.max_history_size:
            self.temperature_history.pop(0)

    def get_average_temperature(self) -> float:
        """
        Get the average temperature across all sensors.

        Returns:
            Average temperature in Celsius
        """
        readings = self.get_current_temperatures()
        if not readings:
            return 30.0  # Default if no readings available
        
        total_temp = sum(reading.temperature_celsius for reading in readings)
        return total_temp / len(readings)

    def get_hottest_component_temperature(self) -> ThermalReading:
        """
        Get the temperature of the hottest component.

        Returns:
            ThermalReading for the hottest component
        """
        readings = self.get_current_temperatures()
        if not readings:
            return ThermalReading(sensor_name="default", temperature_celsius=30.0)
        
        return max(readings, key=lambda r: r.temperature_celsius)

    def is_thermal_violation(self, temperature: Optional[float] = None) -> Dict[str, bool]:
        """
        Check if current temperatures violate any thermal constraints.

        Args:
            temperature: Temperature to check (defaults to average temperature)

        Returns:
            Dictionary with violation status for each constraint
        """
        if temperature is None:
            temperature = self.get_average_temperature()

        violations = {
            'critical_temperature_violation': temperature >= self.constraints.critical_temperature_celsius,
            'max_safe_temperature_violation': temperature >= self.constraints.max_safe_temperature_celsius,
            'min_safe_temperature_violation': temperature <= self.constraints.min_safe_temperature_celsius,
        }

        # Check thermal gradient (rate of temperature change)
        if len(self.temperature_history) >= 2:
            # Get last two readings
            recent_readings = sorted(self.temperature_history[-2:], key=lambda r: r.timestamp)
            if len(recent_readings) == 2:
                temp_change = abs(recent_readings[1].temperature_celsius - recent_readings[0].temperature_celsius)
                time_diff = recent_readings[1].timestamp - recent_readings[0].timestamp
                if time_diff > 0:
                    rate_per_minute = (temp_change / time_diff) * 60
                    violations['thermal_gradient_violation'] = rate_per_minute > self.constraints.thermal_gradient_threshold
                else:
                    violations['thermal_gradient_violation'] = False
        else:
            violations['thermal_gradient_violation'] = False

        return violations

    def get_thermal_safety_margin(self) -> float:
        """
        Get the thermal safety margin (difference between current temp and critical temp).

        Returns:
            Thermal safety margin in Celsius (positive = safe, negative = critical)
        """
        avg_temp = self.get_average_temperature()
        return self.constraints.critical_temperature_celsius - avg_temp

    def get_thermal_efficiency_metrics(self) -> Dict[str, Union[float, str]]:
        """
        Get thermal efficiency metrics for the system.

        Returns:
            Dictionary containing thermal efficiency metrics
        """
        avg_temp = self.get_average_temperature()
        hottest = self.get_hottest_component_temperature()
        
        # Calculate efficiency metrics
        thermal_efficiency = 1.0 - (avg_temp / self.constraints.critical_temperature_celsius)
        thermal_safety_margin = self.get_thermal_safety_margin()
        
        return {
            'thermal_efficiency': max(0.0, thermal_efficiency),  # Clamp to 0-1 range
            'thermal_safety_margin_celsius': thermal_safety_margin,
            'average_temperature_celsius': avg_temp,
            'hottest_component_celsius': hottest.temperature_celsius,
            'hottest_component_name': hottest.sensor_name,
        }

    def start_monitoring(self, update_interval: float = 1.0) -> None:
        """
        Start continuous thermal monitoring.

        Args:
            update_interval: Interval between temperature updates in seconds
        """
        if self.monitoring:
            return

        self.monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(update_interval,),
            daemon=True
        )
        self.monitoring_thread.start()

    def stop_monitoring(self) -> None:
        """
        Stop continuous thermal monitoring.
        """
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)

    def _monitoring_loop(self, update_interval: float) -> None:
        """
        Internal monitoring loop that continuously updates thermal state.

        Args:
            update_interval: Interval between updates in seconds
        """
        while self.monitoring:
            try:
                self.get_current_temperatures()  # This also adds to history
                time.sleep(update_interval)
            except Exception as e:
                self.logger.error(f"Error in thermal monitoring loop: {e}")
                time.sleep(update_interval)

    def get_temperature_trend(self, window_minutes: int = 5) -> float:
        """
        Get the temperature trend over the specified time window.

        Args:
            window_minutes: Time window in minutes to calculate trend

        Returns:
            Temperature trend in degrees Celsius per minute (positive = increasing)
        """
        if len(self.temperature_history) < 2:
            return 0.0

        # Filter readings within the time window
        current_time = time.time()
        window_start = current_time - (window_minutes * 60)
        
        window_readings = [r for r in self.temperature_history if r.timestamp >= window_start]
        
        if len(window_readings) < 2:
            return 0.0

        # Sort by timestamp
        window_readings.sort(key=lambda r: r.timestamp)
        
        # Calculate trend using linear regression
        n = len(window_readings)
        timestamps = [r.timestamp for r in window_readings]
        temperatures = [r.temperature_celsius for r in window_readings]
        
        # Calculate means
        avg_time = sum(timestamps) / n
        avg_temp = sum(temperatures) / n
        
        # Calculate slope (trend)
        numerator = sum((timestamps[i] - avg_time) * (temperatures[i] - avg_temp) for i in range(n))
        denominator = sum((timestamps[i] - avg_time) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        # Slope in degrees per second, convert to degrees per minute
        trend_per_second = numerator / denominator
        trend_per_minute = trend_per_second * 60
        
        return trend_per_minute


def apply_thermal_throttling(temperature: float, max_temp: float = 80.0) -> float:
    """
    Calculate a throttling factor based on current temperature.

    Args:
        temperature: Current temperature in Celsius
        max_temp: Maximum safe temperature in Celsius

    Returns:
        Throttling factor (0.0-1.0, where 1.0 is no throttling)
    """
    if temperature <= max_temp * 0.8:  # Below 80% of max temp, no throttling
        return 1.0
    elif temperature >= max_temp:  # At or above max temp, maximum throttling
        return 0.1
    else:  # Between 80% and 100%, linear throttling
        throttling_ratio = (max_temp - temperature) / (max_temp - max_temp * 0.8)
        return max(0.1, throttling_ratio)  # Ensure at least 10% performance


if __name__ == "__main__":
    # Example usage
    print("Thermal Management Module - Example Usage")
    print("=" * 50)
    
    # Create thermal constraints
    constraints = ThermalConstraint(
        critical_temperature_celsius=90.0,
        max_safe_temperature_celsius=80.0,
        thermal_gradient_threshold=3.0
    )
    
    # Create thermal manager
    thermal_manager = ThermalManager(constraints)
    
    # Get current temperatures
    temperatures = thermal_manager.get_current_temperatures()
    print(f"Current temperatures: {[(t.sensor_name, t.temperature_celsius) for t in temperatures]}")
    
    # Get average temperature
    avg_temp = thermal_manager.get_average_temperature()
    print(f"Average temperature: {avg_temp:.2f}°C")
    
    # Get hottest component
    hottest = thermal_manager.get_hottest_component_temperature()
    print(f"Hottest component: {hottest.sensor_name} at {hottest.temperature_celsius:.2f}°C")
    
    # Check for violations
    violations = thermal_manager.is_thermal_violation()
    print(f"Thermal violations: {violations}")
    
    # Get thermal safety margin
    safety_margin = thermal_manager.get_thermal_safety_margin()
    print(f"Thermal safety margin: {safety_margin:.2f}°C")
    
    # Get efficiency metrics
    metrics = thermal_manager.get_thermal_efficiency_metrics()
    print(f"Efficiency metrics: {metrics}")
    
    # Get temperature trend
    trend = thermal_manager.get_temperature_trend(window_minutes=5)
    print(f"Temperature trend (5min): {trend:.2f}°C/min")
    
    # Example of thermal throttling
    throttling_factor = apply_thermal_throttling(avg_temp, constraints.max_safe_temperature_celsius)
    print(f"Thermal throttling factor: {throttling_factor:.2f}")