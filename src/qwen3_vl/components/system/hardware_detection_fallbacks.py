"""
Hardware Detection and Fallback Mechanisms for Power and Thermal Optimization System

This module implements comprehensive detection of CPU, GPU, temperature sensors, and other
hardware components at runtime, with appropriate fallback mechanisms when specific components
are not available. The system maintains full functionality even on systems with limited
hardware support.
"""

import os
import platform
import subprocess
import time
import logging
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass
import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class HardwareCapabilities:
    """Represents the detected hardware capabilities"""
    cpu_available: bool = False
    cpu_model: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0
    system_memory: int = 0  # in bytes
    gpu_available: bool = False
    gpu_vendor: Optional[str] = None
    gpu_model: Optional[str] = None
    gpu_memory: Optional[int] = None  # in bytes
    compute_capability: Optional[str] = None
    temperature_sensors_available: bool = False
    power_management_available: bool = False


class HardwareDetector:
    """Detects available hardware components and their capabilities"""
    
    def __init__(self):
        self.capabilities = HardwareCapabilities()
        self._detect_hardware()
        
    def _detect_hardware(self):
        """Detect all available hardware components"""
        logger.info("Starting hardware detection...")
        
        # Detect CPU
        self._detect_cpu()
        
        # Detect GPU
        self._detect_gpu()
        
        # Detect temperature sensors
        self._detect_temperature_sensors()
        
        # Detect power management capabilities
        self._detect_power_management()
        
        logger.info(f"Hardware detection completed: {self.capabilities}")
    
    def _detect_cpu(self):
        """Detect CPU capabilities"""
        try:
            self.capabilities.cpu_available = True
            self.capabilities.cpu_model = platform.processor()
            self.capabilities.cpu_cores = psutil.cpu_count(logical=False) or 0
            self.capabilities.cpu_threads = psutil.cpu_count(logical=True) or 0
            self.capabilities.system_memory = psutil.virtual_memory().total

            logger.info(f"CPU detected: {self.capabilities.cpu_model}, {self.capabilities.cpu_cores} cores, "
                       f"{self.capabilities.cpu_threads} threads, "
                       f"{self.capabilities.system_memory / (1024**3):.1f}GB RAM")
        except Exception as e:
            logger.warning(f"CPU detection failed: {e}")
            self.capabilities.cpu_available = False

    def _detect_gpu(self):
        """Detect GPU capabilities with fallbacks"""
        try:
            # Check if CUDA is available
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    # Get GPU properties for the first GPU
                    gpu_properties = torch.cuda.get_device_properties(0)

                    self.capabilities.gpu_available = True
                    self.capabilities.gpu_vendor = "NVIDIA"  # For now, assuming NVIDIA
                    self.capabilities.gpu_model = gpu_properties.name
                    self.capabilities.gpu_memory = gpu_properties.total_memory
                    self.capabilities.compute_capability = f"{gpu_properties.major}.{gpu_properties.minor}"

                    logger.info(f"GPU detected: {gpu_count} device(s), "
                               f"{gpu_properties.name}, "
                               f"{gpu_properties.total_memory / (1024**3):.1f}GB VRAM, "
                               f"Compute {self.capabilities.compute_capability}")
                else:
                    logger.warning("CUDA available but no GPU devices detected")
                    self._detect_gpu_alternative()
            else:
                logger.info("CUDA not available, attempting alternative GPU detection")
                self._detect_gpu_alternative()

        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            self._detect_gpu_alternative()
    
    def _detect_gpu_alternative(self):
        """Alternative GPU detection methods"""
        try:
            # Try using nvidia-smi if available
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                                   '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                gpu_info = result.stdout.strip().split('\n')[0].split(',')
                if len(gpu_info) >= 2:
                    gpu_name = gpu_info[0].strip()
                    gpu_memory_mb = int(gpu_info[1].strip())

                    self.capabilities.gpu_available = True
                    self.capabilities.gpu_vendor = "NVIDIA"
                    self.capabilities.gpu_model = gpu_name
                    self.capabilities.gpu_memory = gpu_memory_mb * 1024 * 1024  # Convert to bytes

                    logger.info(f"GPU detected via nvidia-smi: {gpu_name}, "
                               f"{gpu_memory_mb}MB VRAM")
                    return
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
            logger.debug("nvidia-smi not available or failed")

        # If no GPU detected, set defaults
        self.capabilities.gpu_available = False
        self.capabilities.gpu_vendor = None
        self.capabilities.gpu_model = None
        self.capabilities.gpu_memory = None
        self.capabilities.compute_capability = None

        logger.info("No GPU detected, will use CPU fallback")
    
    def _detect_temperature_sensors(self):
        """Detect temperature sensors with fallbacks"""
        try:
            # Check if psutil has temperature sensor support
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    self.capabilities.temperature_sensors_available = True
                    logger.info(f"Temperature sensors detected: {list(temps.keys())}")
                else:
                    logger.info("No temperature sensors available on this system")
            else:
                logger.info("Temperature sensors not available on this platform")
        except Exception as e:
            logger.warning(f"Temperature sensor detection failed: {e}")
        
        # Even if no sensors are available, we can still function
        if not self.capabilities.temperature_sensors_available:
            logger.info("Temperature sensor fallback will be used")
    
    def _detect_power_management(self):
        """Detect power management capabilities"""
        try:
            # Check if we can get power management info
            if platform.system() == "Windows":
                # On Windows, check if powercfg is available
                result = subprocess.run(['powercfg', '/list'], 
                                      capture_output=True, text=True, timeout=5)
                self.capabilities.power_management_available = (result.returncode == 0)
            elif platform.system() in ["Linux", "Darwin"]:
                # On Unix-like systems, check for power management files
                self.capabilities.power_management_available = (
                    os.path.exists('/sys/class/power_supply/') or
                    os.path.exists('/proc/acpi/battery/')
                )
            else:
                # For other systems, assume basic power management is available
                self.capabilities.power_management_available = True
                
            logger.info(f"Power management available: {self.capabilities.power_management_available}")
        except Exception as e:
            logger.warning(f"Power management detection failed: {e}")
            self.capabilities.power_management_available = False


class FallbackManager:
    """Manages fallback mechanisms when hardware components are not available"""
    
    def __init__(self, hardware_detector: HardwareDetector):
        self.hardware_detector = hardware_detector
        self.fallback_strategies = {}
        self._initialize_fallback_strategies()
    
    def _initialize_fallback_strategies(self):
        """Initialize fallback strategies for different hardware components"""
        self.fallback_strategies = {
            'gpu': self._gpu_fallback,
            'temperature': self._temperature_fallback,
            'power': self._power_fallback,
            'memory': self._memory_fallback
        }
        logger.info("Fallback strategies initialized")
    
    def _gpu_fallback(self, operation: str, *args, **kwargs) -> Any:
        """Fallback for GPU operations"""
        logger.warning(f"GPU not available, falling back for operation: {operation}")
        
        # For tensor operations, create on CPU instead
        if operation == 'tensor_creation':
            shape = args[0] if args else (1,)
            dtype = kwargs.get('dtype', torch.float32)
            return torch.empty(shape, dtype=dtype, device='cpu')
        
        # For computations, perform on CPU
        elif operation == 'computation':
            tensors = args
            # Perform computation on CPU tensors
            return [t.cpu() if isinstance(t, torch.Tensor) else t for t in tensors]
        
        # For memory allocation, use CPU memory
        elif operation == 'memory_allocation':
            size = args[0] if args else 0
            return torch.empty(size, device='cpu')
        
        # Default fallback
        else:
            logger.info(f"No specific fallback for operation {operation}, returning None")
            return None
    
    def _temperature_fallback(self, *args, **kwargs) -> float:
        """Fallback for temperature readings"""
        # Return a safe default temperature when sensors are not available
        default_temp = kwargs.get('default', 40.0)
        logger.info(f"Temperature sensors not available, using default: {default_temp}°C")
        return default_temp
    
    def _power_fallback(self, *args, **kwargs) -> float:
        """Fallback for power readings"""
        # Return a safe default power value when power management is not available
        default_power = kwargs.get('default', 10.0)  # 10W default
        logger.info(f"Power management not available, using default: {default_power}W")
        return default_power
    
    def _memory_fallback(self, *args, **kwargs) -> int:
        """Fallback for memory readings"""
        # Return available system memory when GPU memory is not available
        if self.hardware_detector.capabilities.system_memory > 0:
            return self.hardware_detector.capabilities.system_memory
        else:
            # Fallback to a reasonable default (8GB)
            default_memory = 8 * 1024 * 1024 * 1024  # 8GB
            logger.info(f"Memory detection not available, using default: {default_memory / (1024**3)}GB")
            return default_memory
    
    def execute_with_fallback(self, component: str, operation: str, *args, **kwargs) -> Any:
        """Execute an operation with appropriate fallback if needed"""
        capabilities = self.hardware_detector.capabilities
        
        # Check if the required hardware is available
        if component == 'gpu' and not capabilities.gpu_available:
            return self.fallback_strategies['gpu'](operation, *args, **kwargs)
        elif component == 'temperature' and not capabilities.temperature_sensors_available:
            return self.fallback_strategies['temperature'](*args, **kwargs)
        elif component == 'power' and not capabilities.power_management_available:
            return self.fallback_strategies['power'](*args, **kwargs)
        elif component == 'memory' and not capabilities.gpu_available:
            return self.fallback_strategies['memory'](*args, **kwargs)
        
        # If hardware is available, return None to indicate normal execution
        return None


class SafeHardwareInterface:
    """Safe interface that handles hardware operations with proper fallbacks"""
    
    def __init__(self):
        self.hardware_detector = HardwareDetector()
        self.fallback_manager = FallbackManager(self.hardware_detector)
        self.logger = logging.getLogger(__name__)
    
    def get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Safely get GPU information with fallback"""
        if not self.hardware_detector.capabilities.gpu_available:
            fallback_result = self.fallback_manager.execute_with_fallback(
                'gpu', 'info_query'
            )
            if fallback_result is not None:
                return {
                    'available': False,
                    'name': 'None',
                    'memory_total': 0,
                    'compute_capability': '0.0'
                }
        
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    props = torch.cuda.get_device_properties(0)
                    return {
                        'available': True,
                        'name': props.name,
                        'memory_total': props.total_memory,
                        'compute_capability': f"{props.major}.{props.minor}",
                        'multiprocessors': props.multi_processor_count,
                        'max_threads_per_mp': props.max_threads_per_block
                    }
        except Exception as e:
            self.logger.warning(f"GPU info query failed: {e}")
        
        return {
            'available': False,
            'name': 'None',
            'memory_total': 0,
            'compute_capability': '0.0'
        }
    
    def get_temperature(self, sensor_type: str = 'cpu') -> float:
        """Safely get temperature with fallback"""
        if not self.hardware_detector.capabilities.temperature_sensors_available:
            return self.fallback_manager.execute_with_fallback(
                'temperature', default=40.0
            )

        try:
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    if sensor_type == 'cpu' and 'coretemp' in temps:
                        return max([t.current for t in temps['coretemp']])
                    elif sensor_type == 'cpu' and 'cpu_thermal' in temps:
                        return temps['cpu_thermal'][0].current
                    elif sensor_type == 'gpu' and 'nvidia' in temps:
                        return max([t.current for t in temps['nvidia']])
                    elif sensor_type == 'gpu' and 'nct6776' in temps:
                        # Common GPU temp sensor name on some systems
                        return max([t.current for t in temps['nct6776']])

                    # If specific sensor not found, return first available
                    for sensor_name, sensor_temps in temps.items():
                        if sensor_temps:
                            return sensor_temps[0].current
        except Exception as e:
            self.logger.warning(f"Temperature reading failed: {e}")

        # Return fallback temperature
        return self.fallback_manager.execute_with_fallback(
            'temperature', default=45.0
        )
    
    def get_power_usage(self) -> float:
        """Safely get power usage with fallback"""
        if not self.hardware_detector.capabilities.power_management_available:
            return self.fallback_manager.execute_with_fallback(
                'power', default=15.0
            )
        
        try:
            # On NVIDIA systems, try to get power draw from nvidia-smi
            if self.hardware_detector.capabilities.gpu_available:
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=power.draw', 
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    power_str = result.stdout.strip()
                    if power_str and power_str.replace('.', '').isdigit():
                        return float(power_str)
        except Exception as e:
            self.logger.warning(f"Power usage reading failed: {e}")
        
        # Return CPU-based power estimate
        cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
        # Rough estimation: base power + usage * power_range
        return 5.0 + cpu_percent * 20.0  # 5W base + up to 20W additional

    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                       device: Optional[str] = None) -> torch.Tensor:
        """Safely allocate tensor with fallback to CPU if GPU unavailable"""
        target_device = device or 'cuda'

        if target_device == 'cuda' and not self.hardware_detector.capabilities.gpu_available:
            self.logger.info("GPU not available, allocating tensor on CPU")
            target_device = 'cpu'

        try:
            if target_device == 'cuda' and torch.cuda.is_available():
                # Check if we have enough GPU memory
                tensor_size = torch.Size(shape).numel() * torch.tensor([], dtype=dtype).element_size()
                available_gpu_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)

                if tensor_size > available_gpu_memory:
                    self.logger.warning(f"Not enough GPU memory, allocating on CPU instead")
                    target_device = 'cpu'

            return torch.empty(shape, dtype=dtype, device=target_device)
        except Exception as e:
            self.logger.warning(f"Tensor allocation failed on {target_device}: {e}, falling back to CPU")
            return torch.empty(shape, dtype=dtype, device='cpu')
    
    def get_memory_info(self) -> Dict[str, int]:
        """Get memory information with fallbacks"""
        memory_info = {}
        
        # System memory is always available
        memory_info['system_total'] = psutil.virtual_memory().total
        memory_info['system_available'] = psutil.virtual_memory().available
        
        # GPU memory if available
        if self.hardware_detector.capabilities.gpu_available:
            try:
                if torch.cuda.is_available():
                    memory_info['gpu_total'] = torch.cuda.get_device_properties(0).total_memory
                    memory_info['gpu_allocated'] = torch.cuda.memory_allocated()
                    memory_info['gpu_reserved'] = torch.cuda.memory_reserved()
                else:
                    # Fallback to nvidia-smi if PyTorch CUDA is not available
                    result = subprocess.run([
                        'nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
                        '--format=csv,noheader,nounits'
                    ], capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0:
                        memory_values = result.stdout.strip().split(', ')
                        if len(memory_values) >= 3:
                            memory_info['gpu_total'] = int(memory_values[0].strip()) * 1024 * 1024  # Convert MB to bytes
                            memory_info['gpu_allocated'] = int(memory_values[1].strip()) * 1024 * 1024  # Convert MB to bytes
            except Exception as e:
                self.logger.warning(f"GPU memory info retrieval failed: {e}")
        else:
            # Use fallback values
            memory_info['gpu_total'] = 0
            memory_info['gpu_allocated'] = 0
            memory_info['gpu_reserved'] = 0
        
        return memory_info

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent if os.path.exists('/') else 0,
            'timestamp': time.time()
        }

    def get_hardware_summary(self) -> Dict[str, Any]:
        """Get a summary of all hardware components"""
        return {
            'cpu_info': {
                'model': self.hardware_detector.capabilities.cpu_model,
                'cores': self.hardware_detector.capabilities.cpu_cores,
                'threads': self.hardware_detector.capabilities.cpu_threads,
            },
            'system_memory_gb': self.hardware_detector.capabilities.system_memory / (1024**3),
            'gpu_info': {
                'available': self.hardware_detector.capabilities.gpu_available,
                'name': self.hardware_detector.capabilities.gpu_model or 'N/A',
                'memory_gb': self.hardware_detector.capabilities.gpu_memory / (1024**3) if self.hardware_detector.capabilities.gpu_memory else 0,
                'compute_capability': self.hardware_detector.capabilities.compute_capability or 'N/A'
            },
            'temperature_sensors_available': self.hardware_detector.capabilities.temperature_sensors_available,
            'power_management_available': self.hardware_detector.capabilities.power_management_available
        }


def get_hardware_optimizer_config(hardware_interface: SafeHardwareInterface) -> Dict[str, Any]:
    """
    Generate hardware-optimized configuration based on detected capabilities
    """
    capabilities = hardware_interface.hardware_detector.capabilities

    config = {
        # CPU-specific optimizations
        'cpu_optimizations': {
            'use_multithreading': capabilities.cpu_available,
            'thread_count': min(8, capabilities.cpu_threads or 4),
            'cpu_affinity_enabled': True,
            'simd_optimizations': True,  # Assume SIMD is available on modern CPUs
        },

        # GPU-specific optimizations
        'gpu_optimizations': {
            'enabled': capabilities.gpu_available,
            'memory_fraction': 0.8 if capabilities.gpu_available else 0.0,
            'use_tensor_cores': capabilities.compute_capability and
                              capabilities.compute_capability >= '7.0',
            'batch_size_multiplier': 1.0 if capabilities.gpu_available else 0.5,
        },

        # Power and thermal management
        'power_management': {
            'enabled': capabilities.power_management_available,
            'temperature_monitoring': capabilities.temperature_sensors_available,
            'gpu_power_management': capabilities.gpu_available,
        },

        # Memory management
        'memory_management': {
            'use_gpu_memory_pool': capabilities.gpu_available,
            'cpu_memory_optimizations': True,
            'pinned_memory_enabled': capabilities.gpu_available,
        },

        # Hardware-specific parameters
        'hardware_specific': {
            'cpu_model': capabilities.cpu_model,
            'cpu_cores': capabilities.cpu_cores,
            'cpu_threads': capabilities.cpu_threads,
            'system_memory_gb': capabilities.system_memory / (1024**3),
            'gpu_available': capabilities.gpu_available,
            'gpu_model': capabilities.gpu_model or 'None',
            'gpu_memory_gb': capabilities.gpu_memory / (1024**3) if capabilities.gpu_memory else 0,
            'compute_capability': capabilities.compute_capability or '0.0',
        }
    }

    return config


def validate_hardware_compatibility(config: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate that the detected hardware is compatible with expected requirements
    """
    hardware_interface = SafeHardwareInterface()
    capabilities = hardware_interface.hardware_detector.capabilities

    validation_results = {
        'cpu_available': capabilities.cpu_available,
        'sufficient_cpu_cores': capabilities.cpu_cores >= 2,
        'sufficient_system_memory': capabilities.system_memory >= 4 * 1024 * 1024 * 1024,  # 4GB
        'gpu_available_if_needed': True,  # GPU is optional
        'temperature_sensors_available': capabilities.temperature_sensors_available,
        'power_management_available': capabilities.power_management_available,
    }

    # Log validation results
    for check, result in validation_results.items():
        status = "✓" if result else "✗"
        logger.info(f"Hardware validation - {check}: {status}")

    return validation_results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Hardware Detection and Fallback System")
    print("=" * 50)

    # Initialize the safe hardware interface
    hardware_interface = SafeHardwareInterface()

    # Get hardware capabilities
    caps = hardware_interface.hardware_detector.capabilities
    print(f"CPU Available: {caps.cpu_available}")
    print(f"CPU Model: {caps.cpu_model}")
    print(f"GPU Available: {caps.gpu_available}")
    if caps.gpu_available:
        print(f"  - GPU Model: {caps.gpu_model}")
        print(f"  - GPU Memory: {caps.gpu_memory / (1024**3):.1f}GB")
        print(f"  - Compute Capability: {caps.compute_capability}")
    print(f"Temperature Sensors: {caps.temperature_sensors_available}")
    print(f"Power Management: {caps.power_management_available}")

    # Get GPU info with fallback
    gpu_info = hardware_interface.get_gpu_info()
    print(f"GPU Info: {gpu_info}")

    # Get temperature with fallback
    cpu_temp = hardware_interface.get_temperature('cpu')
    gpu_temp = hardware_interface.get_temperature('gpu')
    print(f"CPU Temperature: {cpu_temp}°C")
    print(f"GPU Temperature: {gpu_temp}°C")

    # Get power usage with fallback
    power_usage = hardware_interface.get_power_usage()
    print(f"Estimated Power Usage: {power_usage}W")

    # Generate hardware-optimized config
    config = get_hardware_optimizer_config(hardware_interface)
    print(f"Hardware-Optimized Config: {config}")

    # Validate hardware compatibility
    validation = validate_hardware_compatibility(config)
    print(f"Hardware Validation: {validation}")

    print("\nHardware Detection and Fallback System initialized successfully!")