"""
Enhanced Power Management System with Comprehensive Error Handling and Validation

This module extends the existing power management system with comprehensive
error handling, validation, and logging for system-level operations.
"""

import time
import threading
import psutil
import subprocess
import json
import logging
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager

# Import the accurate power estimation models
try:
    from power_estimation_models import IntelI5_10210UPowerModel, NVidiaSM61PowerModel, PowerProfiler
    POWER_MODELS_AVAILABLE = True
except ImportError:
    POWER_MODELS_AVAILABLE = False
    IntelI5_10210UPowerModel = None
    NVidiaSM61PowerModel = None
    PowerProfiler = None

# Import validation utilities
from system_validation_utils import (
    SystemValidator, ErrorHandlingManager, ValidationResult, 
    ValidationError, PowerValidationError, ParameterValidationError,
    ResourceCleanupManager, validator as system_validator, 
    error_handler as system_error_handler
)


@dataclass
class PowerConstraint:
    """Represents power and thermal constraints for the system"""
    max_cpu_power_watts: float = 25.0  # TDP of i5-10210U
    max_gpu_power_watts: float = 75.0  # Typical for mobile GPUs
    max_cpu_temp_celsius: float = 90.0
    max_gpu_temp_celsius: float = 85.0
    max_cpu_usage_percent: float = 90.0
    max_gpu_usage_percent: float = 85.0


@dataclass
class PowerState:
    """Current power and thermal state of the system"""
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    cpu_temp_celsius: float = 0.0
    gpu_temp_celsius: float = 0.0
    cpu_power_watts: float = 0.0
    gpu_power_watts: float = 0.0
    timestamp: float = 0.0


class PowerMode(Enum):
    """Different power modes for the system"""
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    POWER_SAVE = "power_save"
    THERMAL_MANAGEMENT = "thermal_management"


class EnhancedPowerAwareScheduler:
    """
    Enhanced Power-aware task scheduler with comprehensive error handling,
    validation, and logging for system-level operations.
    """

    def __init__(self, constraints: PowerConstraint):
        # Initialize logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Validate constraints
        self._validate_constraints(constraints)
        self.constraints = constraints
        
        self.tasks: List[Dict] = []
        self.running_tasks: List[Dict] = []
        self.power_state = PowerState()
        self.power_mode = PowerMode.BALANCED
        self.monitoring_thread = None
        self.is_monitoring = False
        self.task_queue_lock = threading.Lock()
        self.cleanup_manager = ResourceCleanupManager()
        
        # Initialize accurate power models if available
        self.cpu_power_model = IntelI5_10210UPowerModel() if POWER_MODELS_AVAILABLE else None
        self.gpu_power_model = NVidiaSM61PowerModel() if POWER_MODELS_AVAILABLE else None

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Register cleanup function
        self.cleanup_manager.register_cleanup_function(self._cleanup)

    def _validate_constraints(self, constraints: PowerConstraint) -> None:
        """Validate power constraints"""
        try:
            # Validate CPU power constraint
            cpu_power_result = system_validator.validate_parameter_range(
                constraints.max_cpu_power_watts, 1.0, 100.0, "max_cpu_power_watts"
            )
            if not cpu_power_result.is_valid:
                raise ParameterValidationError(
                    f"Invalid CPU power constraint: {cpu_power_result.message}",
                    details=cpu_power_result.details
                )
            
            # Validate GPU power constraint
            gpu_power_result = system_validator.validate_parameter_range(
                constraints.max_gpu_power_watts, 1.0, 200.0, "max_gpu_power_watts"
            )
            if not gpu_power_result.is_valid:
                raise ParameterValidationError(
                    f"Invalid GPU power constraint: {gpu_power_result.message}",
                    details=gpu_power_result.details
                )
            
            # Validate temperature constraints
            cpu_temp_result = system_validator.validate_parameter_range(
                constraints.max_cpu_temp_celsius, 50.0, 120.0, "max_cpu_temp_celsius"
            )
            if not cpu_temp_result.is_valid:
                raise ParameterValidationError(
                    f"Invalid CPU temperature constraint: {cpu_temp_result.message}",
                    details=cpu_temp_result.details
                )
            
            gpu_temp_result = system_validator.validate_parameter_range(
                constraints.max_gpu_temp_celsius, 50.0, 120.0, "max_gpu_temp_celsius"
            )
            if not gpu_temp_result.is_valid:
                raise ParameterValidationError(
                    f"Invalid GPU temperature constraint: {gpu_temp_result.message}",
                    details=gpu_temp_result.details
                )
            
            # Validate usage constraints
            cpu_usage_result = system_validator.validate_parameter_range(
                constraints.max_cpu_usage_percent, 50.0, 100.0, "max_cpu_usage_percent"
            )
            if not cpu_usage_result.is_valid:
                raise ParameterValidationError(
                    f"Invalid CPU usage constraint: {cpu_usage_result.message}",
                    details=cpu_usage_result.details
                )
            
            gpu_usage_result = system_validator.validate_parameter_range(
                constraints.max_gpu_usage_percent, 50.0, 100.0, "max_gpu_usage_percent"
            )
            if not gpu_usage_result.is_valid:
                raise ParameterValidationError(
                    f"Invalid GPU usage constraint: {gpu_usage_result.message}",
                    details=gpu_usage_result.details
                )
                
        except Exception as e:
            self.logger.error(f"Error validating constraints: {str(e)}")
            raise

    def get_system_power_state(self) -> PowerState:
        """Get current system power and thermal state with comprehensive error handling"""
        try:
            # Validate system resources before attempting to get power state
            resources_result = system_validator.validate_system_resources()
            if not resources_result.is_valid:
                self.logger.warning(f"System resource warning: {resources_result.message}")
            
            state = PowerState()

            # Get CPU usage and frequency
            try:
                state.cpu_usage_percent = psutil.cpu_percent(interval=0.1)
            except Exception as e:
                error_info = system_error_handler.handle_hardware_access_error(
                    "get_cpu_usage", "CPU", e
                )
                self.logger.warning(f"Failed to get CPU usage: {error_info['suggested_action']}")
                state.cpu_usage_percent = 0.0

            # Get CPU frequency for more accurate power estimation
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    current_freq = cpu_freq.current  # in MHz
                    max_freq = cpu_freq.max  # in MHz
                    base_freq = 1600.0  # Intel i5-10210U base frequency in MHz
                    freq_ratio = current_freq / base_freq
                else:
                    freq_ratio = 1.0  # Default to base frequency ratio
            except Exception as e:
                error_info = system_error_handler.handle_hardware_access_error(
                    "get_cpu_frequency", "CPU", e
                )
                self.logger.warning(f"Failed to get CPU frequency: {error_info['suggested_action']}")
                freq_ratio = 1.0

            # Get CPU temperature (try different methods)
            try:
                # Check if sensors_temperatures is available in psutil
                if hasattr(psutil, 'sensors_temperatures'):
                    temps = psutil.sensors_temperatures()
                    if temps:  # If sensors_temperatures returns data
                        if 'coretemp' in temps:
                            # Intel CPU temperature
                            state.cpu_temp_celsius = max([temp.current for temp in temps['coretemp']])
                        elif 'cpu_thermal' in temps:
                            # Raspberry Pi or other ARM systems
                            state.cpu_temp_celsius = temps['cpu_thermal'][0].current
                        else:
                            # Use first available temperature sensor
                            first_sensor = next(iter(temps.values()))
                            if first_sensor:
                                state.cpu_temp_celsius = first_sensor[0].current
                            else:
                                state.cpu_temp_celsius = 40.0
                    else:
                        # No temperature sensors available
                        state.cpu_temp_celsius = 40.0
                else:
                    # sensors_temperatures not available on this platform
                    state.cpu_temp_celsius = 40.0
            except Exception as e:
                error_info = system_error_handler.handle_hardware_access_error(
                    "get_cpu_temperature", "CPU", e
                )
                self.logger.warning(f"Failed to get CPU temperature: {error_info['suggested_action']}")
                state.cpu_temp_celsius = 40.0

            # Get GPU info (NVIDIA)
            try:
                gpu_info = self._get_gpu_info()
                if gpu_info:
                    state.gpu_usage_percent = gpu_info.get('gpu_util', 0.0)
                    state.gpu_temp_celsius = gpu_info.get('temperature', 0.0)
                    
                    # Validate thermal thresholds
                    thermal_result = system_validator.validate_thermal_thresholds(
                        state.cpu_temp_celsius, state.gpu_temp_celsius,
                        self.constraints.max_cpu_temp_celsius, self.constraints.max_gpu_temp_celsius
                    )
                    if not thermal_result.is_valid:
                        self.logger.warning(f"Thermal warning: {thermal_result.message}")
                    
                    # Use actual GPU power if available, otherwise estimate with our model
                    if 'power_draw' in gpu_info and gpu_info['power_draw'] > 0:
                        state.gpu_power_watts = gpu_info.get('power_draw', 0.0)
                    else:
                        # Estimate GPU power using our accurate model if available
                        if self.gpu_power_model and POWER_MODELS_AVAILABLE:
                            gpu_util = state.gpu_usage_percent / 100.0  # Convert to 0.0-1.0 range
                            state.gpu_power_watts = self.gpu_power_model.estimate_power(
                                utilization=gpu_util,
                                memory_utilization=gpu_util * 0.8  # Assume memory util is 80% of compute util
                            )
                        else:
                            # Fallback to simple estimation
                            state.gpu_power_watts = self.constraints.max_gpu_power_watts * (state.gpu_usage_percent / 100.0)
                else:
                    # No GPU detected, use our model if available
                    if self.gpu_power_model and POWER_MODELS_AVAILABLE:
                        state.gpu_power_watts = self.gpu_power_model.estimate_power(utilization=0.0, memory_utilization=0.0)
                    else:
                        state.gpu_power_watts = 0.0
            except Exception as e:
                error_info = system_error_handler.handle_hardware_access_error(
                    "get_gpu_info", "GPU", e
                )
                self.logger.warning(f"Failed to get GPU info: {error_info['suggested_action']}")
                state.gpu_usage_percent = 0.0
                state.gpu_temp_celsius = 0.0
                state.gpu_power_watts = 0.0

            # Calculate CPU power using the accurate model if available
            if self.cpu_power_model and POWER_MODELS_AVAILABLE:
                cpu_util = state.cpu_usage_percent / 100.0  # Convert to 0.0-1.0 range
                try:
                    state.cpu_power_watts = self.cpu_power_model.estimate_power(utilization=cpu_util, frequency_ratio=freq_ratio)
                except Exception as e:
                    error_info = system_error_handler.handle_power_operation_error(
                        "estimate_cpu_power", {"utilization": cpu_util, "frequency_ratio": freq_ratio}, e
                    )
                    self.logger.warning(f"Failed to estimate CPU power: {error_info['suggested_action']}")
                    state.cpu_power_watts = self.constraints.max_cpu_power_watts * (state.cpu_usage_percent / 100.0)
            else:
                # Fallback to simple estimation
                state.cpu_power_watts = self.constraints.max_cpu_power_watts * (state.cpu_usage_percent / 100.0)

            # Validate power thresholds
            power_result = system_validator.validate_power_thresholds(
                state.cpu_power_watts, state.gpu_power_watts,
                self.constraints.max_cpu_power_watts, self.constraints.max_gpu_power_watts
            )
            if not power_result.is_valid:
                self.logger.warning(f"Power warning: {power_result.message}")

            state.timestamp = time.time()
            self.power_state = state

            return state

        except Exception as e:
            error_info = system_error_handler.handle_power_operation_error(
                "get_system_power_state", {}, e
            )
            self.logger.error(f"Failed to get system power state: {error_info['error_message']}")
            raise

    def _get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get GPU information using nvidia-smi if available with error handling"""
        try:
            # Validate hardware access first
            hardware_result = system_validator.validate_hardware_access("gpu")
            if not hardware_result.is_valid:
                self.logger.debug(f"GPU not available: {hardware_result.message}")
                return None

            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                output = result.stdout.strip().split(', ')
                if len(output) >= 3:
                    return {
                        'gpu_util': float(output[0]),
                        'temperature': float(output[1]),
                        'power_draw': float(output[2])
                    }
        except subprocess.TimeoutExpired:
            self.logger.warning("nvidia-smi command timed out")
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"nvidia-smi command failed: {e}")
        except ValueError as e:
            self.logger.warning(f"Error parsing nvidia-smi output: {e}")
        except Exception as e:
            error_info = system_error_handler.handle_hardware_access_error(
                "_get_gpu_info", "GPU", e
            )
            self.logger.warning(f"Unexpected error getting GPU info: {error_info['error_message']}")
        
        return None

    def add_task(self, task_func: Callable, priority: int = 1, power_requirements: float = 1.0):
        """Add a task to the scheduler with power requirements and validation"""
        try:
            # Validate parameters
            priority_result = system_validator.validate_parameter_range(
                priority, 1, 10, "task_priority"
            )
            if not priority_result.is_valid:
                raise ParameterValidationError(
                    f"Invalid task priority: {priority_result.message}",
                    details=priority_result.details
                )
            
            power_req_result = system_validator.validate_parameter_range(
                power_requirements, 0.0, 1.0, "power_requirements"
            )
            if not power_req_result.is_valid:
                raise ParameterValidationError(
                    f"Invalid power requirements: {power_req_result.message}",
                    details=power_req_result.details
                )
            
            if task_func is None:
                raise ParameterValidationError("Task function cannot be None")
            
            task = {
                'id': len(self.tasks),
                'function': task_func,
                'priority': priority,
                'power_requirements': power_requirements,  # Relative power requirement (0.0-1.0)
                'created_at': time.time()
            }

            with self.task_queue_lock:
                self.tasks.append(task)
                # Sort by priority (higher priority first)
                self.tasks.sort(key=lambda x: x['priority'], reverse=True)

            self.logger.info(f"Added task {task['id']} with priority {priority} and power requirement {power_requirements}")

        except ParameterValidationError:
            raise
        except Exception as e:
            error_info = system_error_handler.handle_power_operation_error(
                "add_task", {"priority": priority, "power_requirements": power_requirements}, e
            )
            self.logger.error(f"Failed to add task: {error_info['error_message']}")
            raise

    def should_execute_task(self, task: Dict) -> bool:
        """Determine if a task should be executed based on current power state and constraints with error handling"""
        try:
            current_state = self.power_state

            # Check if we're in thermal management mode
            if self.power_mode == PowerMode.THERMAL_MANAGEMENT:
                return task['priority'] > 5  # Only execute high priority tasks

            # Check CPU constraints
            if current_state.cpu_usage_percent > self.constraints.max_cpu_usage_percent:
                self.logger.debug(f"CPU usage too high: {current_state.cpu_usage_percent}% > {self.constraints.max_cpu_usage_percent}%")
                return False

            if current_state.cpu_temp_celsius > self.constraints.max_cpu_temp_celsius * 0.9:
                self.logger.debug(f"CPU temperature too high: {current_state.cpu_temp_celsius}°C > {self.constraints.max_cpu_temp_celsius * 0.9}°C")
                return False  # Throttle if approaching thermal limit

            # Check GPU constraints if task might use GPU
            if current_state.gpu_temp_celsius > self.constraints.max_gpu_temp_celsius * 0.9:
                self.logger.debug(f"GPU temperature too high: {current_state.gpu_temp_celsius}°C > {self.constraints.max_gpu_temp_celsius * 0.9}°C")
                return False  # Throttle if GPU is approaching thermal limit

            # For power save mode, be more restrictive
            if self.power_mode == PowerMode.POWER_SAVE:
                should_execute = task['priority'] > 3 and current_state.cpu_usage_percent < 70.0
                if not should_execute:
                    self.logger.debug(f"Task not executed in power save mode - priority: {task['priority']}, CPU usage: {current_state.cpu_usage_percent}%")
                return should_execute

            # For performance mode, be less restrictive
            if self.power_mode == PowerMode.PERFORMANCE:
                return True

            # Balanced mode - normal constraints
            return True

        except Exception as e:
            error_info = system_error_handler.handle_power_operation_error(
                "should_execute_task", {"task_id": task.get('id', 'unknown')}, e
            )
            self.logger.error(f"Error determining if task should execute: {error_info['error_message']}")
            return False  # Default to not executing if there's an error

    def execute_tasks(self):
        """Execute tasks based on power and thermal constraints with comprehensive error handling"""
        try:
            with self.task_queue_lock:
                tasks_to_execute = []
                for task in self.tasks[:]:  # Create a copy to avoid modification during iteration
                    if self.should_execute_task(task):
                        tasks_to_execute.append(task)
                        # Remove from main task list
                        self.tasks.remove(task)
                        self.running_tasks.append(task)

            # Execute tasks in a thread pool
            for task in tasks_to_execute:
                self._execute_task_async(task)

        except Exception as e:
            error_info = system_error_handler.handle_power_operation_error(
                "execute_tasks", {}, e
            )
            self.logger.error(f"Error executing tasks: {error_info['error_message']}")

    def _execute_task_async(self, task: Dict):
        """Execute a task asynchronously with error handling"""
        def wrapper():
            try:
                result = task['function']()
                self.logger.info(f"Completed task {task['id']}")
                return result
            except Exception as e:
                self.logger.error(f"Error executing task {task['id']}: {str(e)}")
                error_info = system_error_handler.handle_power_operation_error(
                    f"execute_task_{task['id']}", {}, e
                )
                self.logger.error(f"Task execution error details: {error_info}")
            finally:
                with self.task_queue_lock:
                    if task in self.running_tasks:
                        self.running_tasks.remove(task)

        thread = threading.Thread(target=wrapper)
        thread.daemon = True
        thread.start()

    def start_monitoring(self, interval: float = 1.0):
        """Start monitoring system power and thermal state with error handling"""
        try:
            # Validate interval parameter
            interval_result = system_validator.validate_parameter_range(
                interval, 0.1, 10.0, "monitoring_interval"
            )
            if not interval_result.is_valid:
                raise ParameterValidationError(
                    f"Invalid monitoring interval: {interval_result.message}",
                    details=interval_result.details
                )
            
            if self.is_monitoring:
                self.logger.warning("Monitoring already active")
                return

            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            self.logger.info(f"Started power and thermal monitoring with interval {interval}s")

        except ParameterValidationError:
            raise
        except Exception as e:
            error_info = system_error_handler.handle_power_operation_error(
                "start_monitoring", {"interval": interval}, e
            )
            self.logger.error(f"Failed to start monitoring: {error_info['error_message']}")
            raise

    def stop_monitoring(self):
        """Stop monitoring system power and thermal state with error handling"""
        try:
            self.is_monitoring = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=2.0)
            self.logger.info("Stopped power and thermal monitoring")

        except Exception as e:
            error_info = system_error_handler.handle_power_operation_error(
                "stop_monitoring", {}, e
            )
            self.logger.error(f"Failed to stop monitoring: {error_info['error_message']}")
            raise

    def _monitor_loop(self, interval: float):
        """Main monitoring loop with error handling"""
        while self.is_monitoring:
            try:
                self.get_system_power_state()
                self._adjust_power_mode()
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                # Continue monitoring even if there's an error in one iteration
            finally:
                time.sleep(interval)

    def _adjust_power_mode(self):
        """Adjust power mode based on current thermal and power state with error handling"""
        try:
            state = self.power_state

            # Check for thermal emergency
            if (state.cpu_temp_celsius > self.constraints.max_cpu_temp_celsius or
                state.gpu_temp_celsius > self.constraints.max_gpu_temp_celsius):
                self.power_mode = PowerMode.THERMAL_MANAGEMENT
                self.logger.warning("Thermal emergency detected! Switching to thermal management mode")
                return

            # Check for high temperature (approaching limits)
            if (state.cpu_temp_celsius > self.constraints.max_cpu_temp_celsius * 0.8 or
                state.gpu_temp_celsius > self.constraints.max_gpu_temp_celsius * 0.8):
                self.power_mode = PowerMode.POWER_SAVE
                self.logger.info("High temperature detected! Switching to power save mode")
                return

            # Check for high usage
            if (state.cpu_usage_percent > 80.0 or state.gpu_usage_percent > 80.0):
                if self.power_mode != PowerMode.PERFORMANCE:
                    self.power_mode = PowerMode.BALANCED
            else:
                # If usage is low and we're not in thermal management, consider power save
                if (state.cpu_usage_percent < 30.0 and state.gpu_usage_percent < 30.0 and
                    state.cpu_temp_celsius < self.constraints.max_cpu_temp_celsius * 0.7 and
                    state.gpu_temp_celsius < self.constraints.max_gpu_temp_celsius * 0.7):
                    self.power_mode = PowerMode.POWER_SAVE
                else:
                    self.power_mode = PowerMode.BALANCED

        except Exception as e:
            error_info = system_error_handler.handle_power_operation_error(
                "_adjust_power_mode", {"current_state": self.power_state}, e
            )
            self.logger.error(f"Error adjusting power mode: {error_info['error_message']}")
            # Default to balanced mode if there's an error
            self.power_mode = PowerMode.BALANCED

    def get_power_mode(self) -> PowerMode:
        """Get current power mode"""
        return self.power_mode

    def set_power_mode(self, mode: PowerMode):
        """Set power mode manually with validation"""
        try:
            if not isinstance(mode, PowerMode):
                raise ParameterValidationError(f"Invalid power mode: {mode}, must be a PowerMode enum")
            
            self.power_mode = mode
            self.logger.info(f"Power mode set to {mode.value}")

        except ParameterValidationError:
            raise
        except Exception as e:
            error_info = system_error_handler.handle_power_operation_error(
                "set_power_mode", {"mode": str(mode)}, e
            )
            self.logger.error(f"Failed to set power mode: {error_info['error_message']}")
            raise

    def get_task_queue_status(self) -> Dict[str, Any]:
        """Get status of task queue with error handling"""
        try:
            return {
                'pending_tasks': len(self.tasks),
                'running_tasks': len(self.running_tasks),
                'power_mode': self.power_mode.value,
                'current_state': {
                    'cpu_usage': self.power_state.cpu_usage_percent,
                    'gpu_usage': self.power_state.gpu_usage_percent,
                    'cpu_temp': self.power_state.cpu_temp_celsius,
                    'gpu_temp': self.power_state.gpu_temp_celsius,
                    'cpu_power': self.power_state.cpu_power_watts,
                    'gpu_power': self.power_state.gpu_power_watts
                }
            }
        except Exception as e:
            error_info = system_error_handler.handle_power_operation_error(
                "get_task_queue_status", {}, e
            )
            self.logger.error(f"Failed to get task queue status: {error_info['error_message']}")
            raise

    def get_power_model_info(self) -> Dict[str, Any]:
        """Get information about the power models being used"""
        try:
            return {
                'cpu_power_model_available': POWER_MODELS_AVAILABLE,
                'gpu_power_model_available': POWER_MODELS_AVAILABLE,
                'cpu_power_model_type': 'IntelI5_10210UPowerModel' if self.cpu_power_model else 'Simple Estimation',
                'gpu_power_model_type': 'NVidiaSM61PowerModel' if self.gpu_power_model else 'Simple Estimation',
                'current_cpu_power_w': self.power_state.cpu_power_watts,
                'current_gpu_power_w': self.power_state.gpu_power_watts,
                'total_power_w': self.power_state.cpu_power_watts + self.power_state.gpu_power_watts
            }
        except Exception as e:
            error_info = system_error_handler.handle_power_operation_error(
                "get_power_model_info", {}, e
            )
            self.logger.error(f"Failed to get power model info: {error_info['error_message']}")
            raise

    def _cleanup(self):
        """Cleanup resources"""
        try:
            if self.is_monitoring:
                self.stop_monitoring()
            self.logger.info("Power scheduler cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during power scheduler cleanup: {str(e)}")


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 3
    HIGH = 5
    CRITICAL = 10


def create_power_efficient_task(task_func: Callable, priority: TaskPriority = TaskPriority.NORMAL):
    """Decorator to create a power-efficient task with validation"""
    def wrapper(*args, **kwargs):
        # This would contain additional power management logic
        return task_func(*args, **kwargs)
    return wrapper


# Backward compatibility - create an alias for the old class name
PowerAwareScheduler = EnhancedPowerAwareScheduler