"""
System Validation Utilities for Enhanced Power and Thermal Management

This module provides validation, error handling, and system utilities for the
enhanced power and thermal management system.
"""

import time
import threading
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
import psutil
import subprocess
import math


@dataclass
class ValidationResult:
    """
    Represents the result of a validation operation.
    
    Attributes:
        is_valid (bool): Whether the validation passed
        message (str): Description of the validation result
        details (Optional[Dict[str, Any]]): Additional details about the validation
    """
    is_valid: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class ValidationError(Exception):
    """
    Base exception for validation errors.
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class HardwareValidationError(ValidationError):
    """
    Exception raised for hardware validation errors.
    """
    pass


class PowerValidationError(ValidationError):
    """
    Exception raised for power-related validation errors.
    """
    pass


class ThermalValidationError(ValidationError):
    """
    Exception raised for thermal validation errors.
    """
    pass


class ParameterValidationError(ValidationError):
    """
    Exception raised for parameter validation errors.
    """
    pass


class SystemValidator:
    """
    Validates system parameters, configurations, and states for power and thermal management.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_power_thresholds(self, 
                                current_power: float, 
                                max_allowed_power: float, 
                                component: str = "system") -> ValidationResult:
        """
        Validate power consumption against thresholds.
        
        Args:
            current_power: Current power consumption in watts
            max_allowed_power: Maximum allowed power in watts
            component: Component name for logging
            
        Returns:
            ValidationResult indicating if validation passed
        """
        if current_power > max_allowed_power:
            return ValidationResult(
                is_valid=False,
                message=f"Power consumption ({current_power}W) exceeds maximum allowed ({max_allowed_power}W) for {component}",
                details={
                    "current_power": current_power,
                    "max_allowed_power": max_allowed_power,
                    "component": component,
                    "excess": current_power - max_allowed_power
                }
            )
        
        return ValidationResult(
            is_valid=True,
            message=f"Power consumption ({current_power}W) is within limits for {component}",
            details={
                "current_power": current_power,
                "max_allowed_power": max_allowed_power,
                "component": component
            }
        )
    
    def validate_thermal_thresholds(self, 
                                   cpu_temp: float, 
                                   gpu_temp: float, 
                                   max_cpu_temp: float, 
                                   max_gpu_temp: float) -> ValidationResult:
        """
        Validate thermal thresholds for CPU and GPU.
        
        Args:
            cpu_temp: Current CPU temperature in Celsius
            gpu_temp: Current GPU temperature in Celsius
            max_cpu_temp: Maximum allowed CPU temperature
            max_gpu_temp: Maximum allowed GPU temperature
            
        Returns:
            ValidationResult indicating if validation passed
        """
        violations = []
        
        if cpu_temp > max_cpu_temp:
            violations.append(f"CPU temperature ({cpu_temp}°C) exceeds maximum ({max_cpu_temp}°C)")
            
        if gpu_temp > max_gpu_temp:
            violations.append(f"GPU temperature ({gpu_temp}°C) exceeds maximum ({max_gpu_temp}°C)")
            
        if violations:
            return ValidationResult(
                is_valid=False,
                message="; ".join(violations),
                details={
                    "cpu_temp": cpu_temp,
                    "gpu_temp": gpu_temp,
                    "max_cpu_temp": max_cpu_temp,
                    "max_gpu_temp": max_gpu_temp,
                    "violations": violations
                }
            )
        
        return ValidationResult(
            is_valid=True,
            message="Thermal thresholds are within acceptable limits",
            details={
                "cpu_temp": cpu_temp,
                "gpu_temp": gpu_temp,
                "max_cpu_temp": max_cpu_temp,
                "max_gpu_temp": max_gpu_temp
            }
        )
    
    def validate_frequency_transition(self, 
                                     current_freq: float, 
                                     target_freq: float, 
                                     max_transition_rate: float) -> ValidationResult:
        """
        Validate frequency transition against maximum allowed rate.
        
        Args:
            current_freq: Current frequency in MHz
            target_freq: Target frequency in MHz
            max_transition_rate: Maximum allowed transition rate in MHz/s
            
        Returns:
            ValidationResult indicating if validation passed
        """
        transition_delta = abs(target_freq - current_freq)
        
        if transition_delta > max_transition_rate:
            return ValidationResult(
                is_valid=False,
                message=f"Frequency transition ({transition_delta}MHz) exceeds maximum allowed rate ({max_transition_rate}MHz)",
                details={
                    "current_freq": current_freq,
                    "target_freq": target_freq,
                    "max_transition_rate": max_transition_rate,
                    "actual_transition": transition_delta
                }
            )
        
        return ValidationResult(
            is_valid=True,
            message=f"Frequency transition ({transition_delta}MHz) is within acceptable limits",
            details={
                "current_freq": current_freq,
                "target_freq": target_freq,
                "max_transition_rate": max_transition_rate,
                "actual_transition": transition_delta
            }
        )
    
    def validate_hardware_access(self, component: str) -> ValidationResult:
        """
        Validate access to hardware components.
        
        Args:
            component: Hardware component to validate
            
        Returns:
            ValidationResult indicating if validation passed
        """
        try:
            # In a real implementation, this would check actual hardware access
            # For simulation, we'll assume access is available
            if component.lower() in ["cpu", "gpu", "memory", "thermal", "power"]:
                return ValidationResult(
                    is_valid=True,
                    message=f"Access to {component} validated successfully",
                    details={"component": component}
                )
            else:
                return ValidationResult(
                    is_valid=False,
                    message=f"Unknown component: {component}",
                    details={"component": component}
                )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Failed to validate access to {component}: {str(e)}",
                details={"component": component, "error": str(e)}
            )
    
    def validate_parameter_range(self, 
                                value: float, 
                                min_val: float, 
                                max_val: float, 
                                param_name: str) -> ValidationResult:
        """
        Validate that a parameter is within the specified range.
        
        Args:
            value: Parameter value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            param_name: Name of the parameter for logging
            
        Returns:
            ValidationResult indicating if validation passed
        """
        if value < min_val or value > max_val:
            return ValidationResult(
                is_valid=False,
                message=f"Parameter '{param_name}' ({value}) is out of range [{min_val}, {max_val}]",
                details={
                    "param_name": param_name,
                    "value": value,
                    "min_val": min_val,
                    "max_val": max_val
                }
            )
        
        return ValidationResult(
            is_valid=True,
            message=f"Parameter '{param_name}' ({value}) is within range [{min_val}, {max_val}]",
            details={
                "param_name": param_name,
                "value": value,
                "min_val": min_val,
                "max_val": max_val
            }
        )
    
    def validate_system_resources(self, 
                                 cpu_usage: float, 
                                 memory_usage: float, 
                                 disk_usage: float) -> ValidationResult:
        """
        Validate system resource usage against thresholds.
        
        Args:
            cpu_usage: Current CPU usage percentage (0.0-100.0)
            memory_usage: Current memory usage percentage (0.0-100.0)
            disk_usage: Current disk usage percentage (0.0-100.0)
            
        Returns:
            ValidationResult indicating if validation passed
        """
        thresholds = {
            "cpu": 90.0,  # Percent
            "memory": 95.0,  # Percent
            "disk": 95.0  # Percent
        }
        
        violations = []
        
        if cpu_usage > thresholds["cpu"]:
            violations.append(f"CPU usage ({cpu_usage}%) exceeds threshold ({thresholds['cpu']}%)")
            
        if memory_usage > thresholds["memory"]:
            violations.append(f"Memory usage ({memory_usage}%) exceeds threshold ({thresholds['memory']}%)")
            
        if disk_usage > thresholds["disk"]:
            violations.append(f"Disk usage ({disk_usage}%) exceeds threshold ({thresholds['disk']}%)")
            
        if violations:
            return ValidationResult(
                is_valid=False,
                message="; ".join(violations),
                details={
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "disk_usage": disk_usage,
                    "thresholds": thresholds,
                    "violations": violations
                }
            )
        
        return ValidationResult(
            is_valid=True,
            message="System resource usage is within acceptable limits",
            details={
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "disk_usage": disk_usage,
                "thresholds": thresholds
            }
        )


class ErrorHandlingManager:
    """
    Manages error handling for power and thermal management operations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def handle_hardware_access_error(self,
                                   operation: str,
                                   component: str,
                                   error: Exception) -> Dict[str, Any]:
        """
        Handle errors during hardware access operations.
        
        Args:
            operation: Operation that failed
            component: Component that failed
            error: Exception that occurred
            
        Returns:
            Dictionary with error information and suggested actions
        """
        error_info = {
            "operation": operation,
            "component": component,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
            "suggested_action": self._get_hardware_error_suggestion(error),
            "fallback_available": self._has_hardware_fallback(component)
        }
        
        self.logger.error(f"Hardware access error in {operation} for {component}: {str(error)}")
        return error_info
    
    def handle_power_operation_error(self, 
                                   operation: str, 
                                   power_state: Dict[str, Any], 
                                   error: Exception) -> Dict[str, Any]:
        """
        Handle errors during power management operations.
        
        Args:
            operation: Operation that failed
            power_state: Current power state
            error: Exception that occurred
            
        Returns:
            Dictionary with error information and suggested actions
        """
        error_info = {
            "operation": operation,
            "power_state": power_state,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
            "suggested_action": self._get_power_error_suggestion(error),
            "fallback_available": True  # Power management typically has fallbacks
        }
        
        self.logger.error(f"Power operation error in {operation}: {str(error)}")
        return error_info
    
    def handle_thermal_operation_error(self, 
                                     operation: str, 
                                     thermal_state: Dict[str, Any], 
                                     error: Exception) -> Dict[str, Any]:
        """
        Handle errors during thermal management operations.
        
        Args:
            operation: Operation that failed
            thermal_state: Current thermal state
            error: Exception that occurred
            
        Returns:
            Dictionary with error information and suggested actions
        """
        error_info = {
            "operation": operation,
            "thermal_state": thermal_state,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
            "suggested_action": self._get_thermal_error_suggestion(error),
            "fallback_available": True  # Thermal management typically has fallbacks
        }
        
        self.logger.error(f"Thermal operation error in {operation}: {str(error)}")
        return error_info
    
    def handle_frequency_transition_error(self, 
                                        operation: str, 
                                        from_freq: float, 
                                        to_freq: float, 
                                        error: Exception) -> Dict[str, Any]:
        """
        Handle errors during frequency transitions.
        
        Args:
            operation: Operation that failed
            from_freq: Original frequency
            to_freq: Target frequency
            error: Exception that occurred
            
        Returns:
            Dictionary with error information and suggested actions
        """
        error_info = {
            "operation": operation,
            "from_freq": from_freq,
            "to_freq": to_freq,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
            "suggested_action": self._get_frequency_error_suggestion(error),
            "fallback_available": True  # Frequency transitions typically have fallbacks
        }
        
        self.logger.error(f"Frequency transition error from {from_freq}MHz to {to_freq}MHz: {str(error)}")
        return error_info
    
    def _get_hardware_error_suggestion(self, error: Exception) -> str:
        """Get suggested action for hardware errors."""
        error_str = str(error).lower()
        
        if "permission" in error_str or "access" in error_str:
            return "Check system permissions and run with elevated privileges"
        elif "not found" in error_str or "missing" in error_str:
            return "Verify hardware is properly connected and drivers are installed"
        elif "timeout" in error_str:
            return "Increase timeout or check hardware responsiveness"
        else:
            return "Retry operation or check hardware connection"
    
    def _get_power_error_suggestion(self, error: Exception) -> str:
        """Get suggested action for power management errors."""
        error_str = str(error).lower()
        
        if "constraint" in error_str or "limit" in error_str:
            return "Adjust power constraints or reduce system load"
        elif "not supported" in error_str:
            return "Check power management driver support"
        else:
            return "Check system power configuration"
    
    def _get_thermal_error_suggestion(self, error: Exception) -> str:
        """Get suggested action for thermal management errors."""
        error_str = str(error).lower()
        
        if "sensor" in error_str:
            return "Verify thermal sensors are functioning properly"
        elif "critical" in error_str or "overheat" in error_str:
            return "Implement immediate cooling measures"
        else:
            return "Check thermal management configuration"
    
    def _get_frequency_error_suggestion(self, error: Exception) -> str:
        """Get suggested action for frequency transition errors."""
        error_str = str(error).lower()
        
        if "unsupported" in error_str or "invalid" in error_str:
            return "Verify frequency is within supported range"
        elif "locked" in error_str:
            return "Check if frequency is locked by another process"
        else:
            return "Retry with smaller frequency increment"
    
    def _has_hardware_fallback(self, component: str) -> bool:
        """Check if a component has hardware fallback options."""
        # In a real implementation, this would check for actual fallback mechanisms
        return True


class ResourceCleanupManager:
    """
    Manages cleanup of system resources to prevent leaks and ensure proper shutdown.
    """
    
    def __init__(self):
        self.cleanup_functions: List[Callable[[], None]] = []
        self.cleanup_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def register_cleanup_function(self, cleanup_func: Callable[[], None]):
        """
        Register a cleanup function to be called during system shutdown.
        
        Args:
            cleanup_func: Function to call during cleanup (takes no args, returns None)
        """
        with self.cleanup_lock:
            self.cleanup_functions.append(cleanup_func)
            self.logger.debug(f"Registered cleanup function: {cleanup_func.__name__}")
    
    def cleanup_all_resources(self):
        """
        Execute all registered cleanup functions.
        """
        with self.cleanup_lock:
            for i, cleanup_func in enumerate(self.cleanup_functions):
                try:
                    cleanup_func()
                    self.logger.debug(f"Successfully executed cleanup function {i}: {cleanup_func.__name__}")
                except Exception as e:
                    self.logger.error(f"Error executing cleanup function {i}: {cleanup_func.__name__}: {str(e)}")
            
            # Clear the list after execution
            self.cleanup_functions.clear()


# Global instances for convenience
validator = SystemValidator()
error_handler = ErrorHandlingManager()
cleanup_manager = ResourceCleanupManager()


def get_system_validator() -> SystemValidator:
    """Get the global system validator instance."""
    return validator


def get_error_handler() -> ErrorHandlingManager:
    """Get the global error handler instance."""
    return error_handler


def get_cleanup_manager() -> ResourceCleanupManager:
    """Get the global cleanup manager instance."""
    return cleanup_manager


# For backward compatibility
SystemValidator = SystemValidator
ErrorHandlingManager = ErrorHandlingManager
ResourceCleanupManager = ResourceCleanupManager