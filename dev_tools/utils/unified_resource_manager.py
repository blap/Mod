"""
Unified Resource Manager for Qwen3-VL Model

This module implements a comprehensive resource management system that coordinates
between memory, CPU, and thermal resources for optimal performance. The system
includes proper context managers for resource cleanup, robust error recovery
mechanisms, retry logic for critical operations, and unified resource tracking.

Key Features:
- Unified resource management across memory, CPU, and thermal resources
- Context managers for automatic resource cleanup
- Robust error handling and recovery mechanisms
- Retry logic for critical operations
- Resource monitoring and adaptive optimization
- Thread-safe operations
"""

import threading
import time
import logging
from typing import Dict, Any, Optional, Callable, List, Union
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import traceback
import weakref
from datetime import datetime

import sys
import os
# Add the src directory to the Python path to import the required modules
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# Also add the main project directory to find modules in the root
sys.path.insert(0, project_root)

# Import existing components
from src.qwen3_vl.components.memory.advanced_memory_pooling_system import AdvancedMemoryPoolingSystem, TensorType
from src.qwen3_vl.components.memory.advanced_memory_management_vl import VisionLanguageMemoryOptimizer
from src.qwen3_vl.components.optimization.enhanced_thermal_management import EnhancedThermalManager, ThermalAwareTask
from src.qwen3_vl.components.optimization.enhanced_power_management import PowerConstraint
from src.qwen3_vl.utils.centralized_metrics_collector import record_metric, record_timing, record_counter


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Enumeration for different types of resources"""
    MEMORY = "memory"
    CPU = "cpu"
    THERMAL = "thermal"
    GPU = "gpu"


class ResourceState(Enum):
    """Enumeration for resource states"""
    FREE = "free"
    ALLOCATED = "allocated"
    BUSY = "busy"
    ERROR = "error"
    DEGRADED = "degraded"


@dataclass
class ResourceInfo:
    """Information about a managed resource"""
    resource_id: str
    resource_type: ResourceType
    allocated_size: int
    state: ResourceState
    timestamp: float
    attributes: Dict[str, Any]
    owner: Optional[str] = None


class RetryManager:
    """Manages retry logic for critical operations with exponential backoff"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 0.1, max_delay: float = 5.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic
        
        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function execution
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    # Calculate delay with exponential backoff
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed. Last error: {str(e)}")
                    raise last_exception
        
        # This should never be reached, but included for type safety
        raise last_exception


class ResourceManager:
    """Base class for resource management with common functionality"""
    
    def __init__(self):
        self.resources: Dict[str, ResourceInfo] = {}
        self.lock = threading.RLock()
        self.retry_manager = RetryManager()
        
    def register_resource(self, resource_info: ResourceInfo) -> None:
        """Register a resource for tracking"""
        with self.lock:
            self.resources[resource_info.resource_id] = resource_info
            
    def unregister_resource(self, resource_id: str) -> Optional[ResourceInfo]:
        """Unregister a resource from tracking"""
        with self.lock:
            return self.resources.pop(resource_id, None)
            
    def get_resource(self, resource_id: str) -> Optional[ResourceInfo]:
        """Get resource information"""
        with self.lock:
            return self.resources.get(resource_id)
            
    def update_resource_state(self, resource_id: str, new_state: ResourceState) -> bool:
        """Update the state of a tracked resource"""
        with self.lock:
            if resource_id in self.resources:
                self.resources[resource_id].state = new_state
                return True
            return False


class MemoryResourceManager(ResourceManager):
    """Manages memory resources with integration to existing memory systems"""
    
    def __init__(self, memory_pooling_system: AdvancedMemoryPoolingSystem = None,
                 memory_optimizer: VisionLanguageMemoryOptimizer = None):
        super().__init__()
        self.memory_pooling_system = memory_pooling_system
        self.memory_optimizer = memory_optimizer
        self.system_memory_threshold = 0.8  # 80% threshold for system memory
        
    def allocate_tensor(self, tensor_type: TensorType, size: int, tensor_id: str) -> Optional[Any]:
        """Allocate memory for a tensor with retry logic"""
        def _allocate():
            if self.memory_pooling_system:
                return self.memory_pooling_system.allocate(tensor_type, size, tensor_id)
            elif self.memory_optimizer:
                return self.memory_optimizer.allocate_tensor_memory(
                    (size // 4,), dtype='float32', tensor_type=tensor_type.value
                )
            else:
                # Fallback to standard allocation
                import numpy as np
                return np.zeros(size // 4, dtype=np.float32)

        try:
            result = self.retry_manager.execute_with_retry(_allocate)
            # Check if result is not None (for numpy arrays, need to check if result is not None)
            if result is not None:
                # Register the resource
                resource_info = ResourceInfo(
                    resource_id=tensor_id,
                    resource_type=ResourceType.MEMORY,
                    allocated_size=size,
                    state=ResourceState.ALLOCATED,
                    timestamp=time.time(),
                    attributes={'tensor_type': tensor_type.value},
                    owner='memory_pooling_system' if self.memory_pooling_system else 'memory_optimizer'
                )
                self.register_resource(resource_info)
            return result
        except Exception as e:
            logger.error(f"Failed to allocate tensor {tensor_id}: {str(e)}")
            return None
    
    def deallocate_tensor(self, tensor_type: TensorType, tensor_id: str) -> bool:
        """Deallocate memory for a tensor with retry logic"""
        def _deallocate():
            if self.memory_pooling_system:
                return self.memory_pooling_system.deallocate(tensor_type, tensor_id)
            elif self.memory_optimizer:
                # For memory optimizer, we need to call free_tensor_memory
                # This requires keeping track of the allocated tensor
                tensor_info = self.get_resource(tensor_id)
                if tensor_info:
                    # In a real implementation, we would track the actual tensor object
                    self.memory_optimizer.free_tensor_memory(None, tensor_type.value)
                    return True
                else:
                    # If not in tracking, try to deallocate anyway with the provided tensor_type
                    self.memory_optimizer.free_tensor_memory(None, tensor_type.value)
                    return True
            return True  # Fallback success

        try:
            result = self.retry_manager.execute_with_retry(_deallocate)
            if result:
                self.unregister_resource(tensor_id)
            return result
        except Exception as e:
            logger.error(f"Failed to deallocate tensor {tensor_id}: {str(e)}")
            return False
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        try:
            memory_percent = psutil.virtual_memory().percent / 100.0
            return memory_percent > self.system_memory_threshold
        except Exception:
            # If we can't check system memory, assume normal conditions
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        stats = {}
        if self.memory_pooling_system:
            stats['pooling_system'] = self.memory_pooling_system.get_system_stats()
        if self.memory_optimizer:
            stats['optimizer'] = self.memory_optimizer.get_memory_stats()
        
        # System memory stats
        try:
            vm = psutil.virtual_memory()
            stats['system'] = {
                'total_gb': vm.total / (1024**3),
                'available_gb': vm.available / (1024**3),
                'used_gb': vm.used / (1024**3),
                'percent_used': vm.percent
            }
        except Exception as e:
            logger.warning(f"Could not get system memory stats: {e}")
        
        return stats


class CPUResourceManager(ResourceManager):
    """Manages CPU resources and performance optimization"""
    
    def __init__(self):
        super().__init__()
        self.cpu_usage_threshold = 80.0  # Percentage
        self.performance_mode = True
        self.active_threads = 0
        self.max_threads = psutil.cpu_count() or 4
        
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=1)
        except Exception:
            return 0.0
    
    def check_cpu_pressure(self) -> bool:
        """Check if system is under CPU pressure"""
        cpu_usage = self.get_cpu_usage()
        return cpu_usage > self.cpu_usage_threshold
    
    def optimize_thread_pool(self, base_workers: int = None) -> int:
        """Optimize thread pool size based on current system conditions"""
        if base_workers is None:
            base_workers = min(32, (self.max_threads or 4) + 4)
        
        cpu_usage = self.get_cpu_usage()
        
        if cpu_usage > 85:
            # High CPU pressure, reduce workers
            return max(1, base_workers // 2)
        elif cpu_usage > 70:
            # Moderate pressure, slightly reduce
            return max(2, int(base_workers * 0.75))
        else:
            # Normal conditions, use base workers
            return base_workers
    
    def execute_with_optimization(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with CPU optimization"""
        # Check if we're in performance mode
        if not self.performance_mode:
            # Add small delay to reduce CPU usage
            time.sleep(0.001)
        
        return func(*args, **kwargs)
    
    def get_cpu_stats(self) -> Dict[str, Any]:
        """Get CPU statistics"""
        stats = {}
        try:
            stats['cpu_percent'] = psutil.cpu_percent(interval=1)
            stats['cpu_count'] = psutil.cpu_count()
            stats['cpu_freq'] = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
            stats['load_avg'] = psutil.getloadavg()
        except Exception as e:
            logger.warning(f"Could not get CPU stats: {e}")
        
        stats['active_threads'] = self.active_threads
        stats['max_threads'] = self.max_threads
        stats['performance_mode'] = self.performance_mode
        
        return stats


class ThermalResourceManager(ResourceManager):
    """Manages thermal resources and thermal-aware operations"""
    
    def __init__(self, thermal_manager: EnhancedThermalManager = None):
        super().__init__()
        self.thermal_manager = thermal_manager
        self.thermal_threshold = 75.0  # Celsius
        self.performance_scaling = 1.0  # 1.0 = full performance, 0.5 = 50% performance
        
    def get_thermal_state(self) -> Dict[str, Any]:
        """Get current thermal state"""
        if self.thermal_manager:
            try:
                return self.thermal_manager.get_thermal_summary()
            except Exception as e:
                logger.warning(f"Could not get thermal summary: {e}")
                return {}
        else:
            # Fallback: create basic thermal info
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Use first available temperature sensor
                    first_sensor = next(iter(temps.values()))
                    if first_sensor:
                        return {
                            'zones': [{
                                'name': first_sensor[0].label or 'CPU',
                                'current_temp': first_sensor[0].current,
                                'critical_temp': getattr(first_sensor[0], 'high', 80.0),
                                'status': 'normal' if first_sensor[0].current < 70 else 'warning'
                            }],
                            'active': False
                        }
            except Exception:
                pass
            
            return {
                'zones': [{'name': 'default', 'current_temp': 30.0, 'critical_temp': 80.0, 'status': 'normal'}],
                'active': False
            }
    
    def check_thermal_pressure(self) -> bool:
        """Check if system is under thermal pressure"""
        try:
            thermal_state = self.get_thermal_state()
            # Check if thermal_state is a dictionary and has 'zones' key
            if isinstance(thermal_state, dict) and 'zones' in thermal_state:
                zones = thermal_state.get('zones', [])
                if isinstance(zones, list):
                    for zone in zones:
                        if isinstance(zone, dict) and zone.get('current_temp', 0) > self.thermal_threshold:
                            return True
        except Exception:
            # If there's any error in checking thermal pressure, assume no thermal pressure
            logger.warning("Error checking thermal pressure, assuming no thermal pressure")
        return False
    
    def adjust_performance_for_thermal(self) -> float:
        """Adjust performance scaling based on thermal conditions"""
        thermal_state = self.get_thermal_state()
        max_temp = 0
        critical_temp = 80.0
        
        if 'zones' in thermal_state:
            for zone in thermal_state['zones']:
                current_temp = zone.get('current_temp', 0)
                zone_critical = zone.get('critical_temp', 80.0)
                max_temp = max(max_temp, current_temp)
                critical_temp = max(critical_temp, zone_critical)
        
        # Calculate performance scaling based on temperature
        if max_temp >= critical_temp:
            self.performance_scaling = 0.3  # 30% performance
        elif max_temp >= critical_temp * 0.9:
            self.performance_scaling = 0.5  # 50% performance
        elif max_temp >= critical_temp * 0.8:
            self.performance_scaling = 0.7  # 70% performance
        elif max_temp >= critical_temp * 0.7:
            self.performance_scaling = 0.9  # 90% performance
        else:
            self.performance_scaling = 1.0  # 100% performance
            
        return self.performance_scaling
    
    def execute_thermal_aware(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with thermal awareness"""
        # Adjust performance if needed
        scaling = self.adjust_performance_for_thermal()
        
        if scaling < 1.0:
            # Add delay based on thermal pressure
            delay = (1.0 - scaling) * 0.01  # Up to 10ms delay
            time.sleep(delay)
        
        return func(*args, **kwargs)


class UnifiedResourceManager:
    """Main unified resource manager that coordinates all resource types"""
    
    def __init__(self, 
                 memory_pooling_system: AdvancedMemoryPoolingSystem = None,
                 memory_optimizer: VisionLanguageMemoryOptimizer = None,
                 thermal_manager: EnhancedThermalManager = None):
        self.memory_manager = MemoryResourceManager(memory_pooling_system, memory_optimizer)
        self.cpu_manager = CPUResourceManager()
        self.thermal_manager = ThermalResourceManager(thermal_manager)
        self.lock = threading.RLock()
        
        # System-wide thresholds
        self.memory_pressure_threshold = 0.8
        self.cpu_pressure_threshold = 80.0
        self.thermal_pressure_threshold = 75.0
        
        # Performance metrics
        self.metrics = {
            'allocation_count': 0,
            'deallocation_count': 0,
            'retry_count': 0,
            'error_count': 0
        }
        
        logger.info("Unified Resource Manager initialized")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health across all resource types"""
        with self.lock:
            memory_stats = self.memory_manager.get_memory_stats()
            cpu_stats = self.cpu_manager.get_cpu_stats()
            thermal_stats = self.thermal_manager.get_thermal_state()
            
            # Determine overall health
            memory_pressure = self.memory_manager.check_memory_pressure()
            cpu_pressure = self.cpu_manager.check_cpu_pressure()
            thermal_pressure = self.thermal_manager.check_thermal_pressure()
            
            overall_health = "healthy"
            if memory_pressure or cpu_pressure or thermal_pressure:
                overall_health = "under_pressure"
                if memory_pressure and cpu_pressure and thermal_pressure:
                    overall_health = "critical"
            
            return {
                'timestamp': time.time(),
                'overall_health': overall_health,
                'memory_pressure': memory_pressure,
                'cpu_pressure': cpu_pressure,
                'thermal_pressure': thermal_pressure,
                'memory_stats': memory_stats,
                'cpu_stats': cpu_stats,
                'thermal_stats': thermal_stats,
                'metrics': self.metrics.copy()
            }
    
    def allocate_resource(self, resource_type: ResourceType, size: int, resource_id: str, **kwargs) -> Optional[Any]:
        """Allocate a resource of specified type with unified management"""
        start_time = time.time()
        
        try:
            # Check system health before allocation
            health = self.get_system_health()
            if health['overall_health'] == 'critical':
                logger.warning(f"System under critical pressure, rejecting allocation for {resource_id}")
                return None
            
            # Perform allocation based on resource type
            if resource_type == ResourceType.MEMORY:
                tensor_type = kwargs.get('tensor_type', TensorType.KV_CACHE)
                result = self.memory_manager.allocate_tensor(tensor_type, size, resource_id)
            elif resource_type == ResourceType.CPU:
                # For CPU resources, we might want to limit concurrent operations
                if self.cpu_manager.active_threads >= self.cpu_manager.max_threads:
                    logger.warning(f"CPU thread limit reached for {resource_id}")
                    return None
                result = True  # CPU allocation is more about tracking usage
                self.cpu_manager.active_threads += 1
            else:
                logger.error(f"Unsupported resource type: {resource_type}")
                return None

            if result is not None:  # Changed from 'if result:' to handle numpy arrays properly
                self.metrics['allocation_count'] += 1

                # Record metrics
                record_metric(f"resource_allocation_size_{resource_type.value}", size, "resource", "UnifiedResourceManager")
                record_timing(f"resource_allocation_time_{resource_type.value}", time.time() - start_time, "UnifiedResourceManager")

                logger.debug(f"Allocated {resource_type.value} resource {resource_id}, size: {size}")
                return result
            else:
                logger.warning(f"Failed to allocate {resource_type.value} resource {resource_id}")
                return None
                
        except Exception as e:
            self.metrics['error_count'] += 1
            logger.error(f"Error allocating {resource_type.value} resource {resource_id}: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
    
    def deallocate_resource(self, resource_type: ResourceType, resource_id: str, **kwargs) -> bool:
        """Deallocate a resource with unified management"""
        start_time = time.time()

        try:
            # Perform deallocation based on resource type
            if resource_type == ResourceType.MEMORY:
                # Get the tensor type from kwargs or try to infer from resource tracking
                tensor_type = kwargs.get('tensor_type', TensorType.KV_CACHE)

                # Also check if we have the resource in our tracking to get the correct type
                resource_info = self.memory_manager.get_resource(resource_id)
                if resource_info and 'tensor_type' in resource_info.attributes:
                    tensor_type = TensorType(resource_info.attributes['tensor_type'])

                result = self.memory_manager.deallocate_tensor(tensor_type, resource_id)
            elif resource_type == ResourceType.CPU:
                self.cpu_manager.active_threads = max(0, self.cpu_manager.active_threads - 1)
                result = True
            else:
                logger.error(f"Unsupported resource type: {resource_type}")
                return False

            if result:
                self.metrics['deallocation_count'] += 1

                # Record metrics
                record_timing(f"resource_deallocation_time_{resource_type.value}", time.time() - start_time, "UnifiedResourceManager")

                logger.debug(f"Deallocated {resource_type.value} resource {resource_id}")
                return result
            else:
                logger.warning(f"Failed to deallocate {resource_type.value} resource {resource_id}")
                return False

        except Exception as e:
            self.metrics['error_count'] += 1
            logger.error(f"Error deallocating {resource_type.value} resource {resource_id}: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    def execute_with_resource_optimization(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with resource optimization"""
        # Check system health
        health = self.get_system_health()

        # Apply thermal awareness
        try:
            if isinstance(health.get('thermal_stats', {}), dict) and health['thermal_stats'].get('zones'):
                result = self.thermal_manager.execute_thermal_aware(func, *args, **kwargs)
            else:
                result = func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Error in thermal-aware execution: {e}, executing normally")
            result = func(*args, **kwargs)

        # Apply CPU optimization if needed
        try:
            if health.get('cpu_pressure', False):
                result = self.cpu_manager.execute_with_optimization(lambda: result)
        except Exception as e:
            logger.warning(f"Error in CPU optimization: {e}, executing normally")

        return result
    
    def get_resource_usage_summary(self) -> Dict[str, Any]:
        """Get a summary of resource usage"""
        health = self.get_system_health()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_health': health['overall_health'],
            'memory_usage_percent': health['memory_stats'].get('system', {}).get('percent_used', 0),
            'cpu_usage_percent': health['cpu_stats'].get('cpu_percent', 0),
            'max_temperature': max(
                [zone.get('current_temp', 0) for zone in health['thermal_stats'].get('zones', [])],
                default=0
            ),
            'active_allocations': len(self.memory_manager.resources),
            'cpu_active_threads': health['cpu_stats'].get('active_threads', 0),
            'performance_scaling': self.thermal_manager.performance_scaling,
            'metrics': self.metrics
        }
    
    def cleanup_all_resources(self) -> int:
        """Force cleanup of all managed resources"""
        cleaned_count = 0
        
        # Clean up memory resources
        for resource_id in list(self.memory_manager.resources.keys()):
            resource_info = self.memory_manager.get_resource(resource_id)
            if resource_info and resource_info.resource_type == ResourceType.MEMORY:
                if self.deallocate_resource(ResourceType.MEMORY, resource_id):
                    cleaned_count += 1
        
        # Clean up CPU resources
        self.cpu_manager.active_threads = 0
        
        logger.info(f"Cleaned up {cleaned_count} resources")
        return cleaned_count


class ResourceContextManager:
    """Context manager for unified resource management"""
    
    def __init__(self, resource_manager: UnifiedResourceManager, 
                 resource_type: ResourceType, size: int, resource_id: str, **kwargs):
        self.resource_manager = resource_manager
        self.resource_type = resource_type
        self.size = size
        self.resource_id = resource_id
        self.kwargs = kwargs
        self.allocated_resource = None
        self._entered = False
        self._exited = False
        self._exception_occurred = False
    
    def __enter__(self):
        if self._entered:
            raise RuntimeError("Context manager already entered")
        
        self._entered = True
        
        try:
            self.allocated_resource = self.resource_manager.allocate_resource(
                self.resource_type, self.size, self.resource_id, **self.kwargs
            )
            
            if self.allocated_resource is None:
                raise RuntimeError(f"Failed to allocate {self.resource_type.value} resource {self.resource_id}")
            
            return self.allocated_resource
        except Exception as e:
            logger.error(f"Error during resource allocation in context: {e}")
            raise
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self._exited:
            return False

        self._exited = True
        self._exception_occurred = exc_type is not None

        try:
            # Pass the tensor_type as a keyword argument for proper deallocation
            deallocate_kwargs = {}
            if self.resource_type == ResourceType.MEMORY:
                deallocate_kwargs['tensor_type'] = self.kwargs.get('tensor_type', TensorType.KV_CACHE)

            success = self.resource_manager.deallocate_resource(
                self.resource_type, self.resource_id, **deallocate_kwargs
            )

            if not success:
                logger.warning(f"Failed to deallocate {self.resource_type.value} resource {self.resource_id}")

            # Log if exception occurred
            if self._exception_occurred:
                logger.warning(f"Exception occurred in resource context {self.resource_id}: {exc_type.__name__}")

            return False  # Don't suppress exceptions
        except Exception as e:
            logger.error(f"Error during resource deallocation in context: {e}")
            raise


@contextmanager
def resource_context(resource_manager: UnifiedResourceManager,
                    resource_type: ResourceType, size: int, resource_id: str, **kwargs):
    """Context manager factory for unified resource management"""
    ctx = ResourceContextManager(resource_manager, resource_type, size, resource_id, **kwargs)
    try:
        yield ctx.__enter__()
    finally:
        ctx.__exit__(None, None, None)


# Global unified resource manager instance (singleton pattern)
class GlobalResourceManager:
    """Singleton class for global resource management"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.resource_manager = None
            self.initialized = True
    
    def set_resource_manager(self, resource_manager: UnifiedResourceManager):
        """Set the global resource manager"""
        self.resource_manager = resource_manager
    
    def get_resource_manager(self) -> Optional[UnifiedResourceManager]:
        """Get the global resource manager"""
        return self.resource_manager


def get_global_resource_manager() -> Optional[UnifiedResourceManager]:
    """Get the global resource manager instance"""
    return GlobalResourceManager().get_resource_manager()


def init_global_resource_manager(
    memory_pooling_system: AdvancedMemoryPoolingSystem = None,
    memory_optimizer: VisionLanguageMemoryOptimizer = None,
    thermal_manager: EnhancedThermalManager = None
) -> UnifiedResourceManager:
    """Initialize and set the global resource manager"""
    resource_manager = UnifiedResourceManager(
        memory_pooling_system, memory_optimizer, thermal_manager
    )
    GlobalResourceManager().set_resource_manager(resource_manager)
    return resource_manager


# Example usage and integration
def integrate_with_qwen3_vl():
    """
    Example of how to integrate the unified resource manager with Qwen3-VL
    """
    # Create or get existing memory systems
    memory_pooling_system = AdvancedMemoryPoolingSystem()
    memory_optimizer = VisionLanguageMemoryOptimizer()
    
    # Create power constraints for thermal management
    power_constraints = PowerConstraint(
        max_cpu_temp_celsius=85.0,
        max_gpu_temp_celsius=80.0,
        max_cpu_power_watts=45.0,
        max_gpu_power_watts=75.0
    )
    
    # Create thermal manager
    thermal_manager = EnhancedThermalManager(power_constraints)
    
    # Initialize unified resource manager
    resource_manager = init_global_resource_manager(
        memory_pooling_system,
        memory_optimizer,
        thermal_manager
    )
    
    return resource_manager


if __name__ == "__main__":
    print("Unified Resource Manager - Example Usage")
    print("=" * 50)
    
    # Integrate with Qwen3-VL
    resource_manager = integrate_with_qwen3_vl()
    
    # Example: Allocate memory resource using context manager
    print("\n1. Using resource context manager...")
    with resource_context(resource_manager, ResourceType.MEMORY, 1024*1024, "test_tensor_1", 
                         tensor_type=TensorType.KV_CACHE) as tensor:
        print(f"Allocated tensor resource: {type(tensor)}")
    
    # Example: Manual resource allocation/deallocation
    print("\n2. Manual resource allocation...")
    resource = resource_manager.allocate_resource(
        ResourceType.MEMORY, 2*1024*1024, "test_tensor_2", 
        tensor_type=TensorType.IMAGE_FEATURES
    )
    if resource:
        print(f"Manually allocated resource: {type(resource)}")
        resource_manager.deallocate_resource(ResourceType.MEMORY, "test_tensor_2")
    
    # Example: Check system health
    print("\n3. System health check...")
    health = resource_manager.get_system_health()
    print(f"Overall health: {health['overall_health']}")
    print(f"Memory pressure: {health['memory_pressure']}")
    print(f"CPU pressure: {health['cpu_pressure']}")
    print(f"Thermal pressure: {health['thermal_pressure']}")
    
    # Example: Resource usage summary
    print("\n4. Resource usage summary...")
    summary = resource_manager.get_resource_usage_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\nUnified Resource Manager example completed.")