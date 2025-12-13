"""
Power Management Module for Adaptive Algorithms

This module provides power management functionality for adaptive algorithms,
including power state tracking, constraint management, and power-efficient
computing utilities.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import time
import psutil
import threading
import logging


@dataclass
class PowerState:
    """
    Represents the current power state of the system.

    Attributes:
        cpu_usage_percent: CPU usage percentage (0.0-100.0)
        gpu_usage_percent: GPU usage percentage (0.0-100.0)
        cpu_temp_celsius: CPU temperature in Celsius
        gpu_temp_celsius: GPU temperature in Celsius
        cpu_power_watts: CPU power consumption in watts
        gpu_power_watts: GPU power consumption in watts
        timestamp: Timestamp of the measurement
    """
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    cpu_temp_celsius: float = 25.0
    gpu_temp_celsius: float = 25.0
    cpu_power_watts: float = 0.0
    gpu_power_watts: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        """Validate the power state values."""
        if not 0.0 <= self.cpu_usage_percent <= 100.0:
            raise ValueError("cpu_usage_percent must be between 0.0 and 100.0")
        if not 0.0 <= self.gpu_usage_percent <= 100.0:
            raise ValueError("gpu_usage_percent must be between 0.0 and 100.0")
        if self.cpu_power_watts < 0.0:
            raise ValueError("cpu_power_watts must be non-negative")
        if self.gpu_power_watts < 0.0:
            raise ValueError("gpu_power_watts must be non-negative")


@dataclass
class PowerConstraint:
    """
    Defines power constraints for the system.

    Attributes:
        max_cpu_power_watts: Maximum allowed CPU power consumption in watts
        max_gpu_power_watts: Maximum allowed GPU power consumption in watts
        max_cpu_temp_celsius: Maximum allowed CPU temperature in Celsius
        max_gpu_temp_celsius: Maximum allowed GPU temperature in Celsius
        max_cpu_usage_percent: Maximum allowed CPU usage percentage (0.0-100.0)
        max_gpu_usage_percent: Maximum allowed GPU usage percentage (0.0-100.0)
    """
    max_cpu_power_watts: float = 45.0  # Default for Intel i5-10210U
    max_gpu_power_watts: float = 75.0  # Default for NVIDIA GPU
    max_cpu_temp_celsius: float = 85.0
    max_gpu_temp_celsius: float = 80.0
    max_cpu_usage_percent: float = 90.0
    max_gpu_usage_percent: float = 95.0

    def __post_init__(self):
        """Validate the power constraint values."""
        if self.max_cpu_power_watts <= 0.0:
            raise ValueError("max_cpu_power_watts must be positive")
        if self.max_gpu_power_watts <= 0.0:
            raise ValueError("max_gpu_power_watts must be positive")
        if self.max_cpu_temp_celsius <= 0.0:
            raise ValueError("max_cpu_temp_celsius must be positive")
        if self.max_gpu_temp_celsius <= 0.0:
            raise ValueError("max_gpu_temp_celsius must be positive")
        if not 0.0 <= self.max_cpu_usage_percent <= 100.0:
            raise ValueError("max_cpu_usage_percent must be between 0.0 and 100.0")
        if not 0.0 <= self.max_gpu_usage_percent <= 100.0:
            raise ValueError("max_gpu_usage_percent must be between 0.0 and 100.0")


class PowerMonitor:
    """
    Monitors system power state and provides power management utilities.
    """
    monitoring_thread: Optional[threading.Thread]

    def __init__(self, constraints: Optional[PowerConstraint] = None):
        """
        Initialize the power monitor.

        Args:
            constraints: Power constraints for the system (defaults to PowerConstraint())
        """
        self.constraints = constraints or PowerConstraint()
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.current_power_state = PowerState(timestamp=time.time())
        self.power_state_lock = threading.Lock()

    def get_current_power_state(self) -> PowerState:
        """
        Get the current power state of the system.

        Returns:
            Current PowerState object with system metrics
        """
        with self.power_state_lock:
            # Get CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # Get CPU temperature (if available)
            cpu_temp = 25.0  # Default value if not available
            try:
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps:
                    cpu_temp = temps['coretemp'][0].current
                elif 'cpu_thermal' in temps:  # Raspberry Pi, etc.
                    cpu_temp = temps['cpu_thermal'][0].current
            except (AttributeError, IndexError):
                # Temperature sensors not available
                pass
            
            # Get memory usage as a proxy for power consumption
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Estimate CPU power based on usage and temperature
            estimated_cpu_power = self._estimate_cpu_power(cpu_usage, cpu_temp)
            
            # Create power state object
            self.current_power_state = PowerState(
                cpu_usage_percent=cpu_usage,
                gpu_usage_percent=0.0,  # Placeholder - would require GPU-specific library
                cpu_temp_celsius=cpu_temp,
                gpu_temp_celsius=25.0,  # Placeholder
                cpu_power_watts=estimated_cpu_power,
                gpu_power_watts=0.0,  # Placeholder
                timestamp=time.time()
            )
            
            return self.current_power_state

    def _estimate_cpu_power(self, cpu_usage: float, cpu_temp: float) -> float:
        """
        Estimate CPU power consumption based on usage and temperature.

        Args:
            cpu_usage: CPU usage percentage (0.0-100.0)
            cpu_temp: CPU temperature in Celsius

        Returns:
            Estimated CPU power consumption in watts
        """
        # Simplified power estimation model
        # Base power (idle): ~5W
        # Max power (Intel i5-10210U): ~25W TDP (but can go higher under load)
        base_power = 5.0
        max_power = 25.0  # Conservative estimate for i5-10210U
        
        # Calculate power based on CPU usage
        usage_factor = cpu_usage / 100.0
        power_from_usage = base_power + (max_power - base_power) * usage_factor
        
        # Temperature adjustment (higher temp = higher power due to increased cooling needs)
        temp_factor = max(0.0, (cpu_temp - 25.0) / 60.0)  # Normalize temp effect
        power_from_temp = temp_factor * 5.0  # Up to 5W additional for high temp
        
        estimated_power = min(max_power, power_from_usage + power_from_temp)
        return estimated_power

    def is_power_violation(self, power_state: Optional[PowerState] = None) -> Dict[str, bool]:
        """
        Check if current power state violates any constraints.

        Args:
            power_state: Power state to check (defaults to current state)

        Returns:
            Dictionary with violation status for each constraint
        """
        if power_state is None:
            power_state = self.get_current_power_state()

        violations = {
            'cpu_power_violation': power_state.cpu_power_watts > self.constraints.max_cpu_power_watts,
            'gpu_power_violation': power_state.gpu_power_watts > self.constraints.max_gpu_power_watts,
            'cpu_temp_violation': power_state.cpu_temp_celsius > self.constraints.max_cpu_temp_celsius,
            'gpu_temp_violation': power_state.gpu_temp_celsius > self.constraints.max_gpu_temp_celsius,
            'cpu_usage_violation': power_state.cpu_usage_percent > self.constraints.max_cpu_usage_percent,
            'gpu_usage_violation': power_state.gpu_usage_percent > self.constraints.max_gpu_usage_percent,
        }

        return violations

    def start_monitoring(self, update_interval: float = 1.0) -> None:
        """
        Start continuous power state monitoring.

        Args:
            update_interval: Interval between power state updates in seconds
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
        Stop continuous power state monitoring.
        """
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)

    def _monitoring_loop(self, update_interval: float) -> None:
        """
        Internal monitoring loop that continuously updates power state.

        Args:
            update_interval: Interval between updates in seconds
        """
        while self.monitoring:
            try:
                self.get_current_power_state()
                time.sleep(update_interval)
            except Exception as e:
                self.logger.error(f"Error in power monitoring loop: {e}")
                time.sleep(update_interval)

    def get_power_efficiency_metrics(self) -> Dict[str, float]:
        """
        Get power efficiency metrics for the system.

        Returns:
            Dictionary containing power efficiency metrics
        """
        current_state = self.get_current_power_state()
        
        # Calculate efficiency metrics
        cpu_efficiency = current_state.cpu_usage_percent / max(1, current_state.cpu_power_watts) if current_state.cpu_power_watts > 0 else 0
        temp_safety_margin = self.constraints.max_cpu_temp_celsius - current_state.cpu_temp_celsius
        
        return {
            'cpu_efficiency': cpu_efficiency,  # Performance per watt
            'temp_safety_margin_celsius': temp_safety_margin,
            'current_cpu_power_watts': current_state.cpu_power_watts,
            'current_cpu_usage_percent': current_state.cpu_usage_percent,
            'current_cpu_temp_celsius': current_state.cpu_temp_celsius,
        }


def get_system_power_state() -> PowerState:
    """
    Convenience function to get the current system power state.

    Returns:
        Current PowerState object with system metrics
    """
    monitor = PowerMonitor()
    return monitor.get_current_power_state()


if __name__ == "__main__":
    # Example usage
    print("Power Management Module - Example Usage")
    print("=" * 50)
    
    # Create constraints
    constraints = PowerConstraint(
        max_cpu_power_watts=25.0,
        max_gpu_power_watts=50.0,
        max_cpu_temp_celsius=80.0,
        max_gpu_temp_celsius=75.0
    )
    
    # Create power monitor
    monitor = PowerMonitor(constraints)
    
    # Get current power state
    power_state = monitor.get_current_power_state()
    print(f"Current CPU Usage: {power_state.cpu_usage_percent}%")
    print(f"Current CPU Temp: {power_state.cpu_temp_celsius}°C")
    print(f"Current CPU Power: {power_state.cpu_power_watts}W")
    print(f"Timestamp: {power_state.timestamp}")
    
    # Check for violations
    violations = monitor.is_power_violation(power_state)
    print(f"Power violations: {violations}")
    
    # Get efficiency metrics
    metrics = monitor.get_power_efficiency_metrics()
    print(f"Efficiency metrics: {metrics}")