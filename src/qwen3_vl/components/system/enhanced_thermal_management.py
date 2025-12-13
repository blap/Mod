"""
Enhanced Thermal Management System with Comprehensive Error Handling and Validation

This module extends the existing thermal management system with comprehensive
error handling, validation, and logging for system-level operations.
"""

import time
import threading
import psutil
import subprocess
import logging
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Import validation utilities
from system_validation_utils import (
    SystemValidator, ErrorHandlingManager, ValidationResult, 
    ValidationError, ThermalValidationError, ParameterValidationError,
    ResourceCleanupManager, validator as system_validator, 
    error_handler as system_error_handler
)
from enhanced_power_management import PowerState, PowerConstraint


@dataclass
class ThermalZone:
    """Represents a thermal zone in the system"""
    name: str
    current_temp: float
    critical_temp: float
    passive_temp: float
    zone_type: str


@dataclass
class CoolingDevice:
    """Represents a cooling device in the system"""
    name: str
    type: str  # fan, gpu_fan, etc.
    current_state: int  # 0-100%
    max_state: int
    min_state: int


class ThermalPolicy(Enum):
    """Thermal management policies"""
    PASSIVE = "passive"  # Reduce performance to reduce heat
    ACTIVE = "active"    # Increase cooling (fans, etc.)
    HYBRID = "hybrid"    # Combination of passive and active


class EnhancedThermalManager:
    """
    Enhanced Thermal management system with comprehensive error handling,
    validation, and logging for system-level operations.
    """

    def __init__(self, constraints: PowerConstraint):
        # Initialize logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Validate constraints
        self._validate_constraints(constraints)
        self.constraints = constraints
        
        self.thermal_zones: List[ThermalZone] = []
        self.cooling_devices: List[CoolingDevice] = []
        self.policy = ThermalPolicy.HYBRID
        self.is_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.control_thread: Optional[threading.Thread] = None
        self.callbacks: List[Callable] = []
        self.cleanup_manager = ResourceCleanupManager()

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Thresholds for thermal management
        self.critical_temp_threshold = 0.9  # 90% of max temp
        self.warning_temp_threshold = 0.8   # 80% of max temp
        self.safe_temp_threshold = 0.7      # 70% of max temp

        # Initialize thermal zones and cooling devices
        self._initialize_hardware()

        # Register cleanup function
        self.cleanup_manager.register_cleanup_function(self._cleanup)

    def _validate_constraints(self, constraints: PowerConstraint) -> None:
        """Validate thermal constraints"""
        try:
            # Validate CPU temperature constraint
            cpu_temp_result = system_validator.validate_parameter_range(
                constraints.max_cpu_temp_celsius, 50.0, 120.0, "max_cpu_temp_celsius"
            )
            if not cpu_temp_result.is_valid:
                raise ParameterValidationError(
                    f"Invalid CPU temperature constraint: {cpu_temp_result.message}",
                    details=cpu_temp_result.details
                )
            
            # Validate GPU temperature constraint
            gpu_temp_result = system_validator.validate_parameter_range(
                constraints.max_gpu_temp_celsius, 50.0, 120.0, "max_gpu_temp_celsius"
            )
            if not gpu_temp_result.is_valid:
                raise ParameterValidationError(
                    f"Invalid GPU temperature constraint: {gpu_temp_result.message}",
                    details=gpu_temp_result.details
                )
                
        except Exception as e:
            self.logger.error(f"Error validating thermal constraints: {str(e)}")
            raise

    def _initialize_hardware(self):
        """Initialize thermal zones and cooling devices with error handling"""
        try:
            # Initialize CPU thermal zone
            self.thermal_zones.append(ThermalZone(
                name="CPU",
                current_temp=40.0,
                critical_temp=self.constraints.max_cpu_temp_celsius,
                passive_temp=self.constraints.max_cpu_temp_celsius * 0.8,
                zone_type="CPU"
            ))

            # Initialize GPU thermal zone if available
            try:
                gpu_info = self._get_gpu_info()
                if gpu_info:
                    self.thermal_zones.append(ThermalZone(
                        name="GPU",
                        current_temp=gpu_info.get('temperature', 40.0),
                        critical_temp=self.constraints.max_gpu_temp_celsius,
                        passive_temp=self.constraints.max_gpu_temp_celsius * 0.8,
                        zone_type="GPU"
                    ))
            except Exception as e:
                error_info = system_error_handler.handle_hardware_access_error(
                    "_initialize_hardware", "GPU", e
                )
                self.logger.warning(f"GPU not available or error initializing: {error_info['error_message']}")

            # Initialize cooling devices (fans)
            self.cooling_devices.append(CoolingDevice(
                name="CPU Fan",
                type="fan",
                current_state=50,  # Start at 50% speed
                max_state=100,
                min_state=20
            ))

            if self._gpu_available():
                self.cooling_devices.append(CoolingDevice(
                    name="GPU Fan",
                    type="gpu_fan",
                    current_state=50,  # Start at 50% speed
                    max_state=100,
                    min_state=20
                ))

            self.logger.info(f"Initialized thermal zones: {[z.name for z in self.thermal_zones]}")
            self.logger.info(f"Initialized cooling devices: {[d.name for d in self.cooling_devices]}")

        except Exception as e:
            error_info = system_error_handler.handle_hardware_access_error(
                "_initialize_hardware", "system", e
            )
            self.logger.error(f"Failed to initialize hardware: {error_info['error_message']}")
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

    def _gpu_available(self) -> bool:
        """Check if GPU is available with error handling"""
        try:
            return self._get_gpu_info() is not None
        except Exception as e:
            error_info = system_error_handler.handle_hardware_access_error(
                "_gpu_available", "GPU", e
            )
            self.logger.warning(f"Error checking GPU availability: {error_info['error_message']}")
            return False

    def get_thermal_state(self) -> List[ThermalZone]:
        """Get current thermal state of all zones with comprehensive error handling"""
        try:
            for zone in self.thermal_zones:
                if zone.zone_type == "CPU":
                    # Check if sensors_temperatures is available in psutil
                    if hasattr(psutil, 'sensors_temperatures'):
                        try:
                            temps = psutil.sensors_temperatures()
                            if temps:  # If sensors_temperatures returns data
                                if 'coretemp' in temps:
                                    # Intel CPU temperature
                                    zone.current_temp = max([temp.current for temp in temps['coretemp']])
                                elif 'cpu_thermal' in temps:
                                    # Raspberry Pi or other ARM systems
                                    zone.current_temp = temps['cpu_thermal'][0].current
                                else:
                                    # Use first available temperature sensor
                                    first_sensor = next(iter(temps.values()))
                                    if first_sensor:
                                        zone.current_temp = first_sensor[0].current
                        except Exception as e:
                            error_info = system_error_handler.handle_hardware_access_error(
                                "get_cpu_temperature", "CPU", e
                            )
                            self.logger.warning(f"Failed to get CPU temperature: {error_info['suggested_action']}")
                    else:
                        # sensors_temperatures not available on this platform
                        pass  # Keep current value
                elif zone.zone_type == "GPU":
                    try:
                        gpu_info = self._get_gpu_info()
                        if gpu_info:
                            zone.current_temp = gpu_info.get('temperature', zone.current_temp)
                    except Exception as e:
                        error_info = system_error_handler.handle_hardware_access_error(
                            "get_gpu_temperature", "GPU", e
                        )
                        self.logger.warning(f"Failed to get GPU temperature: {error_info['suggested_action']}")

            return self.thermal_zones

        except Exception as e:
            error_info = system_error_handler.handle_thermal_operation_error(
                "get_thermal_state", {}, e
            )
            self.logger.error(f"Failed to get thermal state: {error_info['error_message']}")
            raise

    def get_cooling_state(self) -> List[CoolingDevice]:
        """Get current state of cooling devices with error handling"""
        try:
            # In a real implementation, this would query actual cooling device states
            # For simulation purposes, we'll just return the stored states
            return self.cooling_devices
        except Exception as e:
            error_info = system_error_handler.handle_hardware_access_error(
                "get_cooling_state", "cooling_devices", e
            )
            self.logger.error(f"Failed to get cooling state: {error_info['error_message']}")
            raise

    def register_callback(self, callback: Callable[[str, float], None]):
        """Register a callback to be called when thermal events occur with validation"""
        try:
            if not callable(callback):
                raise ParameterValidationError("Callback must be a callable function")
            
            self.callbacks.append(callback)
            self.logger.info(f"Registered thermal callback function")

        except ParameterValidationError:
            raise
        except Exception as e:
            error_info = system_error_handler.handle_thermal_operation_error(
                "register_callback", {}, e
            )
            self.logger.error(f"Failed to register callback: {error_info['error_message']}")
            raise

    def _notify_callbacks(self, event_type: str, temperature: float):
        """Notify all registered callbacks of a thermal event with error handling"""
        for i, callback in enumerate(self.callbacks):
            try:
                callback(event_type, temperature)
            except Exception as e:
                error_info = system_error_handler.handle_thermal_operation_error(
                    f"callback_{i}", {"event_type": event_type, "temperature": temperature}, e
                )
                self.logger.error(f"Error in thermal callback {i}: {error_info['error_message']}")

    def adjust_cooling(self, zone_name: str, temperature: float):
        """Adjust cooling based on temperature in a specific zone with validation and error handling"""
        try:
            # Validate parameters
            temp_result = system_validator.validate_parameter_range(
                temperature, -50.0, 200.0, "temperature"
            )
            if not temp_result.is_valid:
                raise ParameterValidationError(
                    f"Invalid temperature: {temp_result.message}",
                    details=temp_result.details
                )
            
            if not isinstance(zone_name, str):
                raise ParameterValidationError(f"Zone name must be a string, got {type(zone_name)}")
            
            # Find the thermal zone
            zone = next((z for z in self.thermal_zones if z.name.lower() == zone_name.lower()), None)
            if not zone:
                self.logger.warning(f"Thermal zone {zone_name} not found")
                return

            # Validate thermal thresholds
            thermal_result = system_validator.validate_thermal_thresholds(
                temperature if zone.zone_type == "CPU" else zone.current_temp,
                temperature if zone.zone_type == "GPU" else temperature,
                zone.critical_temp if zone.zone_type == "CPU" else self.constraints.max_gpu_temp_celsius,
                zone.critical_temp if zone.zone_type == "GPU" else self.constraints.max_gpu_temp_celsius
            )
            if not thermal_result.is_valid:
                self.logger.warning(f"Thermal validation warning: {thermal_result.message}")

            # Calculate how much above the safe threshold we are
            safe_temp = zone.passive_temp
            critical_temp = zone.critical_temp

            if temperature >= critical_temp:
                # Critical temperature - max cooling and performance reduction
                self._set_cooling_level(zone_name, 100)
                self._reduce_performance(0.3)  # Reduce performance by 30%
                self._notify_callbacks("critical_temp", temperature)
                self.logger.critical(f"CRITICAL TEMPERATURE in {zone_name}: {temperature}°C")
            elif temperature >= safe_temp:
                # Above safe threshold - increase cooling
                excess = (temperature - safe_temp) / (critical_temp - safe_temp)
                cooling_level = min(100, int(50 + excess * 50))  # Scale cooling 50-100%
                self._set_cooling_level(zone_name, cooling_level)
                if excess > 0.5:
                    self._reduce_performance(0.15)  # Reduce performance by 15% if significantly above safe
                self._notify_callbacks("high_temp", temperature)
                self.logger.warning(f"HIGH TEMPERATURE in {zone_name}: {temperature}°C, cooling to {cooling_level}%")
            else:
                # Temperature is safe - reduce cooling to save power
                cooling_level = max(30, int(50 - (safe_temp - temperature) / safe_temp * 20))
                self._set_cooling_level(zone_name, cooling_level)
                self._notify_callbacks("safe_temp", temperature)
                self.logger.info(f"SAFE TEMPERATURE in {zone_name}: {temperature}°C, cooling to {cooling_level}%")

        except ParameterValidationError:
            raise
        except Exception as e:
            error_info = system_error_handler.handle_thermal_operation_error(
                "adjust_cooling", {"zone_name": zone_name, "temperature": temperature}, e
            )
            self.logger.error(f"Failed to adjust cooling: {error_info['error_message']}")
            raise

    def _set_cooling_level(self, zone_name: str, level: int):
        """Set cooling level for a specific zone with validation and error handling"""
        try:
            # Validate parameters
            level_result = system_validator.validate_parameter_range(
                level, 0, 100, "cooling_level"
            )
            if not level_result.is_valid:
                raise ParameterValidationError(
                    f"Invalid cooling level: {level_result.message}",
                    details=level_result.details
                )
            
            if not isinstance(zone_name, str):
                raise ParameterValidationError(f"Zone name must be a string, got {type(zone_name)}")

            # In a real implementation, this would control actual cooling hardware
            # For simulation, we'll just log the action and update our internal state
            for device in self.cooling_devices:
                if zone_name.lower() in device.name.lower() or "cpu" in device.name.lower():
                    device.current_state = max(device.min_state, min(device.max_state, level))
                    self.logger.info(f"Set {device.name} to {device.current_state}%")

        except ParameterValidationError:
            raise
        except Exception as e:
            error_info = system_error_handler.handle_hardware_access_error(
                "_set_cooling_level", "cooling_device", e
            )
            self.logger.error(f"Failed to set cooling level: {error_info['error_message']}")
            raise

    def _reduce_performance(self, factor: float):
        """Reduce system performance to decrease heat generation with validation"""
        try:
            # Validate factor
            factor_result = system_validator.validate_parameter_range(
                factor, 0.0, 1.0, "performance_reduction_factor"
            )
            if not factor_result.is_valid:
                raise ParameterValidationError(
                    f"Invalid performance reduction factor: {factor_result.message}",
                    details=factor_result.details
                )

            # This is a simplified performance reduction
            # In a real implementation, this might involve:
            # - Throttling CPU frequency
            # - Reducing GPU clock speed
            # - Limiting number of active cores
            # - Throttling I/O operations
            self.logger.info(f"Reducing performance by factor: {factor}")

            # Notify registered callbacks about performance reduction
            self._notify_callbacks("performance_reduction", factor)

        except ParameterValidationError:
            raise
        except Exception as e:
            error_info = system_error_handler.handle_thermal_operation_error(
                "_reduce_performance", {"factor": factor}, e
            )
            self.logger.error(f"Failed to reduce performance: {error_info['error_message']}")
            raise

    def start_management(self, monitoring_interval: float = 1.0):
        """Start thermal management with validation and error handling"""
        try:
            # Validate interval parameter
            interval_result = system_validator.validate_parameter_range(
                monitoring_interval, 0.1, 10.0, "monitoring_interval"
            )
            if not interval_result.is_valid:
                raise ParameterValidationError(
                    f"Invalid monitoring interval: {interval_result.message}",
                    details=interval_result.details
                )
            
            if self.is_active:
                self.logger.warning("Thermal management already active")
                return

            self.is_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(monitoring_interval,)
            )
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()

            self.logger.info(f"Started thermal management with interval {monitoring_interval}s")

        except ParameterValidationError:
            raise
        except Exception as e:
            error_info = system_error_handler.handle_thermal_operation_error(
                "start_management", {"interval": monitoring_interval}, e
            )
            self.logger.error(f"Failed to start thermal management: {error_info['error_message']}")
            raise

    def stop_management(self):
        """Stop thermal management with error handling"""
        try:
            self.is_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=2.0)

            self.logger.info("Stopped thermal management")

        except Exception as e:
            error_info = system_error_handler.handle_thermal_operation_error(
                "stop_management", {}, e
            )
            self.logger.error(f"Failed to stop thermal management: {error_info['error_message']}")
            raise

    def _monitoring_loop(self, interval: float):
        """Main thermal monitoring loop with error handling"""
        while self.is_active:
            try:
                # Get current thermal state
                zones = self.get_thermal_state()

                # Adjust cooling for each zone
                for zone in zones:
                    self.adjust_cooling(zone.name, zone.current_temp)

                # Sleep for the specified interval
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in thermal monitoring loop: {str(e)}")
                # Continue monitoring even if there's an error in one iteration
                time.sleep(interval)

    def get_thermal_summary(self) -> Dict[str, Any]:
        """Get a summary of thermal state and management with error handling"""
        try:
            zones = self.get_thermal_state()
            cooling = self.get_cooling_state()

            return {
                "zones": [
                    {
                        "name": zone.name,
                        "current_temp": zone.current_temp,
                        "critical_temp": zone.critical_temp,
                        "passive_temp": zone.passive_temp,
                        "status": self._get_zone_status(zone)
                    }
                    for zone in zones
                ],
                "cooling_devices": [
                    {
                        "name": device.name,
                        "type": device.type,
                        "current_state": device.current_state,
                        "max_state": device.max_state,
                        "min_state": device.min_state
                    }
                    for device in cooling
                ],
                "policy": self.policy.value,
                "active": self.is_active
            }

        except Exception as e:
            error_info = system_error_handler.handle_thermal_operation_error(
                "get_thermal_summary", {}, e
            )
            self.logger.error(f"Failed to get thermal summary: {error_info['error_message']}")
            raise

    def _get_zone_status(self, zone: ThermalZone) -> str:
        """Get status of a thermal zone with error handling"""
        try:
            if zone.current_temp >= zone.critical_temp:
                return "critical"
            elif zone.current_temp >= zone.passive_temp:
                return "warning"
            else:
                return "normal"
        except Exception as e:
            error_info = system_error_handler.handle_thermal_operation_error(
                "_get_zone_status", {"zone": zone.name}, e
            )
            self.logger.error(f"Failed to get zone status: {error_info['error_message']}")
            return "unknown"

    def set_policy(self, policy: ThermalPolicy):
        """Set thermal management policy with validation"""
        try:
            if not isinstance(policy, ThermalPolicy):
                raise ParameterValidationError(f"Invalid thermal policy: {policy}, must be a ThermalPolicy enum")
            
            self.policy = policy
            self.logger.info(f"Thermal policy set to {policy.value}")

        except ParameterValidationError:
            raise
        except Exception as e:
            error_info = system_error_handler.handle_thermal_operation_error(
                "set_policy", {"policy": str(policy)}, e
            )
            self.logger.error(f"Failed to set thermal policy: {error_info['error_message']}")
            raise

    def _cleanup(self):
        """Cleanup resources"""
        try:
            if self.is_active:
                self.stop_management()
            self.logger.info("Thermal manager cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during thermal manager cleanup: {str(e)}")


class ThermalAwareTask:
    """
    A task that is aware of thermal conditions and can adjust its behavior
    based on the current thermal state of the system with comprehensive error handling.
    """

    def __init__(self, name: str, base_power: float = 1.0, thermal_manager: Optional[EnhancedThermalManager] = None):
        # Validate parameters
        if not isinstance(name, str) or not name.strip():
            raise ParameterValidationError("Task name must be a non-empty string")
        
        name_result = system_validator.validate_parameter_range(
            base_power, 0.0, 10.0, "base_power"
        )
        if not name_result.is_valid:
            raise ParameterValidationError(
                f"Invalid base power: {name_result.message}",
                details=name_result.details
            )
        
        self.name = name
        self.base_power = base_power  # Base power consumption (0.0-1.0)
        self.is_running = False
        self.thermal_manager = thermal_manager
        self.logger = logging.getLogger(__name__)

    def attach_thermal_manager(self, thermal_manager: EnhancedThermalManager):
        """Attach to a thermal manager to get thermal updates with validation"""
        try:
            if not isinstance(thermal_manager, EnhancedThermalManager):
                raise ParameterValidationError(
                    f"Thermal manager must be an EnhancedThermalManager instance, got {type(thermal_manager)}"
                )
            
            self.thermal_manager = thermal_manager
            self.logger.info(f"Attached thermal manager to task {self.name}")

        except ParameterValidationError:
            raise
        except Exception as e:
            error_info = system_error_handler.handle_thermal_operation_error(
                "attach_thermal_manager", {"task_name": self.name}, e
            )
            self.logger.error(f"Failed to attach thermal manager: {error_info['error_message']}")
            raise

    def execute_with_thermal_awareness(self, power_state: PowerState):
        """Execute the task considering thermal constraints with error handling"""
        try:
            if not self.thermal_manager:
                # Execute normally if no thermal manager
                self.logger.info(f"Executing {self.name} without thermal awareness (no thermal manager attached)")
                return self.execute()

            # Check thermal zones
            zones = self.thermal_manager.get_thermal_state()
            max_temp_ratio = 0.0

            for zone in zones:
                temp_ratio = zone.current_temp / zone.critical_temp if zone.critical_temp > 0 else 0
                max_temp_ratio = max(max_temp_ratio, temp_ratio)

            # Adjust execution based on thermal conditions
            if max_temp_ratio > 0.9:
                # Critical thermal conditions - skip execution
                self.logger.info(f"Skipping {self.name} due to critical thermal conditions (ratio: {max_temp_ratio:.2f})")
                return False
            elif max_temp_ratio > 0.8:
                # High thermal conditions - reduce intensity
                self.logger.info(f"Reducing intensity of {self.name} due to high thermal conditions (ratio: {max_temp_ratio:.2f})")
                return self.execute_reduced()
            else:
                # Normal conditions - execute normally
                self.logger.info(f"Executing {self.name} under normal thermal conditions (ratio: {max_temp_ratio:.2f})")
                return self.execute()

        except Exception as e:
            error_info = system_error_handler.handle_thermal_operation_error(
                "execute_with_thermal_awareness", {"task_name": self.name}, e
            )
            self.logger.error(f"Error in thermal-aware execution: {error_info['error_message']}")
            # Execute normally if thermal awareness fails
            return self.execute()

    def execute(self):
        """Execute the task normally with error handling"""
        try:
            self.is_running = True
            # Simulate task execution
            time.sleep(1)  # Replace with actual task logic
            self.is_running = False
            return True
        except Exception as e:
            self.logger.error(f"Error executing task {self.name}: {str(e)}")
            return False

    def execute_reduced(self):
        """Execute the task with reduced intensity with error handling"""
        try:
            self.is_running = True
            # Simulate reduced task execution
            time.sleep(0.5)  # Reduced work
            self.is_running = False
            return True
        except Exception as e:
            self.logger.error(f"Error executing reduced task {self.name}: {str(e)}")
            return False


# Backward compatibility - create an alias for the old class name
ThermalManager = EnhancedThermalManager