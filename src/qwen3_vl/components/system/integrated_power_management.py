"""
Integrated Power Management Framework for Intel i5-10210U + NVIDIA SM61
This module integrates all power management components into a cohesive system
that optimizes performance while managing power consumption and heat generation.
"""
import time
import threading
import logging
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
from enum import Enum
import psutil
import subprocess

from power_management import PowerConstraint, PowerAwareScheduler, PowerState, PowerMode
from power_estimation_models import PowerProfiler
from thermal_management import ThermalManager, ThermalPolicy
from adaptive_algorithms import AdaptiveController, AdaptationStrategy, AdaptiveModelWrapper
from dvfs_controller import DVFSController, WorkloadBasedDVFS


@dataclass
class SystemHealth:
    """Comprehensive system health metrics"""
    cpu_usage: float
    gpu_usage: float
    cpu_temp: float
    gpu_temp: float
    cpu_power: float
    gpu_power: float
    timestamp: float
    power_mode: str
    thermal_status: str
    efficiency_score: float


class PowerManagementFramework:
    """
    Integrated power management framework that combines all components:
    - Power-aware scheduling
    - Thermal management
    - Adaptive algorithms
    - DVFS control
    """
    
    def __init__(self, constraints: Optional[PowerConstraint] = None):
        self.constraints = constraints or PowerConstraint()
        self.scheduler = PowerAwareScheduler(self.constraints)
        self.thermal_manager = ThermalManager(self.constraints)
        self.adaptive_controller = AdaptiveController(self.constraints)
        self.dvfs_controller = DVFSController()
        self.workload_dvfs = WorkloadBasedDVFS(self.dvfs_controller)

        # Initialize power profiler for accurate power estimation
        self.power_profiler = PowerProfiler()

        # System state
        self.is_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.management_thread: Optional[threading.Thread] = None
        self.system_health: Optional[SystemHealth] = None

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Register callbacks between components
        self._setup_component_callbacks()
    
    def _setup_component_callbacks(self):
        """Set up callbacks between components for coordinated management"""
        # Register thermal callbacks
        def thermal_callback(event_type: str, value: float):
            if event_type == "critical_temp":
                # In critical thermal state, adjust all components
                self.scheduler.set_power_mode(PowerMode.THERMAL_MANAGEMENT)
                self.adaptive_controller.set_strategy(AdaptationStrategy.THERMAL_AWARE)
            elif event_type == "high_temp":
                # In high temp, be more conservative
                if self.scheduler.get_power_mode() != PowerMode.THERMAL_MANAGEMENT:
                    self.scheduler.set_power_mode(PowerMode.POWER_SAVE)
        
        self.thermal_manager.register_callback(thermal_callback)
    
    def start_framework(self):
        """Start the integrated power management framework"""
        if self.is_active:
            return
        
        self.is_active = True
        
        # Start all component managers
        self.scheduler.start_monitoring(interval=2.0)
        self.thermal_manager.start_management()
        self.adaptive_controller.start_adaptation()

        if self.dvfs_controller.is_dvfs_available:
            self.dvfs_controller.start_adaptive_scaling(interval=3.0)
        
        # Start main management thread
        self.management_thread = threading.Thread(target=self._management_loop)
        self.management_thread.daemon = True
        self.management_thread.start()
        
        self.logger.info("Started integrated power management framework")
    
    def stop_framework(self):
        """Stop the integrated power management framework"""
        self.is_active = False
        
        # Stop all component managers
        self.scheduler.stop_monitoring()
        self.thermal_manager.stop_management()
        self.adaptive_controller.stop_adaptation()
        
        if self.dvfs_controller.is_dvfs_available:
            self.dvfs_controller.stop_adaptive_scaling()
        
        if self.management_thread:
            self.management_thread.join(timeout=3.0)
        
        self.logger.info("Stopped integrated power management framework")
    
    def _management_loop(self):
        """Main management loop that coordinates all components"""
        while self.is_active:
            try:
                # Get current system state
                power_state = self.scheduler.power_state
                thermal_zones = self.thermal_manager.get_thermal_state()
                
                # Update adaptive parameters based on current state
                self.adaptive_controller.update_parameters(power_state)
                
                # Execute scheduled tasks considering all constraints
                self.scheduler.execute_tasks()
                
                # Update system health
                self._update_system_health(power_state, thermal_zones)
                
                # Sleep for a short interval before next cycle
                time.sleep(1.0)
            except Exception as e:
                self.logger.error(f"Error in management loop: {str(e)}")
                time.sleep(1.0)  # Continue even if there's an error
    
    def _update_system_health(self, power_state: PowerState, thermal_zones: List):
        """Update system health metrics"""
        # Determine thermal status
        thermal_status = "normal"
        for zone in thermal_zones:
            if zone.current_temp >= zone.critical_temp:
                thermal_status = "critical"
                break
            elif zone.current_temp >= zone.passive_temp:
                thermal_status = "warning"
        
        # Calculate efficiency score (0.0-1.0, where 1.0 is most efficient)
        # This is a simplified efficiency calculation
        temp_ratio = max(
            power_state.cpu_temp_celsius / self.constraints.max_cpu_temp_celsius,
            power_state.gpu_temp_celsius / self.constraints.max_gpu_temp_celsius
        )
        power_ratio = max(
            power_state.cpu_power_watts / self.constraints.max_cpu_power_watts,
            power_state.gpu_power_watts / self.constraints.max_gpu_power_watts
        )
        
        # Efficiency is higher when temp and power are lower, but performance is maintained
        efficiency_score = max(0.0, min(1.0, 1.0 - (temp_ratio * 0.5 + power_ratio * 0.5)))
        
        self.system_health = SystemHealth(
            cpu_usage=power_state.cpu_usage_percent,
            gpu_usage=power_state.gpu_usage_percent,
            cpu_temp=power_state.cpu_temp_celsius,
            gpu_temp=power_state.gpu_temp_celsius,
            cpu_power=power_state.cpu_power_watts,
            gpu_power=power_state.gpu_power_watts,
            timestamp=power_state.timestamp,
            power_mode=self.scheduler.get_power_mode().value,
            thermal_status=thermal_status,
            efficiency_score=efficiency_score
        )
    
    def get_system_health(self) -> Optional[SystemHealth]:
        """Get current system health metrics"""
        return self.system_health
    
    def add_task(self, task_func: Callable, priority: int = 1, power_requirements: float = 1.0):
        """Add a task to the power-aware scheduler"""
        self.scheduler.add_task(task_func, priority, power_requirements)
    
    def execute_model_with_power_management(self, model, input_data, **kwargs):
        """Execute a model with integrated power management"""
        # Get current power state
        power_state = self.scheduler.power_state
        
        # Create adaptive model wrapper
        adaptive_model = AdaptiveModelWrapper(model, self.constraints)
        
        # Execute prediction with power awareness
        return adaptive_model.predict(input_data, power_state, **kwargs)
    
    def execute_training_with_power_management(self, model, training_data, **kwargs):
        """Execute model training with integrated power management"""
        # Get current power state
        power_state = self.scheduler.power_state
        
        # Create adaptive model wrapper
        adaptive_model = AdaptiveModelWrapper(model, self.constraints)
        
        # Execute training with power awareness
        return adaptive_model.fit(training_data, power_state, **kwargs)
    
    def set_adaptation_strategy(self, strategy: AdaptationStrategy):
        """Set the adaptation strategy for the system"""
        self.adaptive_controller.set_strategy(strategy)
        self.logger.info(f"Set adaptation strategy to {strategy.value}")
    
    def set_thermal_policy(self, policy: ThermalPolicy):
        """Set the thermal management policy"""
        self.thermal_manager.set_policy(policy)
        self.logger.info(f"Set thermal policy to {policy.value}")
    
    def get_framework_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the framework status"""
        return {
            "active": self.is_active,
            "scheduler_status": {
                "mode": self.scheduler.get_power_mode().value,
                "pending_tasks": len(self.scheduler.tasks),
                "running_tasks": len(self.scheduler.running_tasks),
            },
            "thermal_manager_status": self.thermal_manager.get_thermal_summary(),
            "adaptive_controller_status": self.adaptive_controller.get_adaptation_summary(),
            "dvfs_status": self.dvfs_controller.get_system_power_efficiency(),
            "system_health": self.system_health.__dict__ if self.system_health else None
        }
    
    def optimize_for_workload(self, workload_type: str):
        """Optimize the system for a specific workload type"""
        # Register workload profiles if not already registered
        from dvfs_controller import WORKLOAD_PROFILES
        for name, profile in WORKLOAD_PROFILES.items():
            self.workload_dvfs.register_workload_profile(name, profile)

        # Set the workload type for DVFS
        self.workload_dvfs.set_workload_type(workload_type)

        # Adjust strategies based on workload type
        if workload_type == "high_performance":
            self.scheduler.set_power_mode(PowerMode.PERFORMANCE)
            self.adaptive_controller.set_strategy(AdaptationStrategy.PERFORMANCE_FIRST)
            self.thermal_manager.set_policy(ThermalPolicy.HYBRID)
        elif workload_type == "power_efficient":
            self.scheduler.set_power_mode(PowerMode.POWER_SAVE)
            self.adaptive_controller.set_strategy(AdaptationStrategy.POWER_EFFICIENT)
            self.thermal_manager.set_policy(ThermalPolicy.PASSIVE)
        elif workload_type == "balanced":
            self.scheduler.set_power_mode(PowerMode.BALANCED)
            self.adaptive_controller.set_strategy(AdaptationStrategy.BALANCED)
            self.thermal_manager.set_policy(ThermalPolicy.HYBRID)
        else:
            self.logger.warning(f"Unknown workload type: {workload_type}")
    
    def get_power_consumption_estimate(self) -> Dict[str, float]:
        """Get estimated power consumption by component"""
        if not self.system_health:
            # Use power profiler to get current estimates if system health is not available
            cpu_power = self.power_profiler.get_current_cpu_power()
            gpu_power = self.power_profiler.get_current_gpu_power()
            total_power = cpu_power + gpu_power

            # Calculate efficiency score based on current power vs max allowed
            cpu_efficiency = 1.0 - min(cpu_power / self.constraints.max_cpu_power_watts, 1.0)
            gpu_efficiency = 1.0 - min(gpu_power / self.constraints.max_gpu_power_watts, 1.0)
            efficiency_score = min(cpu_efficiency, gpu_efficiency)

            return {
                "estimated_cpu_power": cpu_power,
                "estimated_gpu_power": gpu_power,
                "total_estimated_power": total_power,
                "power_efficiency_score": efficiency_score
            }

        # Use system health data if available
        return {
            "estimated_cpu_power": self.system_health.cpu_power,
            "estimated_gpu_power": self.system_health.gpu_power,
            "total_estimated_power": self.system_health.cpu_power + self.system_health.gpu_power,
            "power_efficiency_score": self.system_health.efficiency_score
        }

    def profile_workload_power(self, workload_func, *args, **kwargs):
        """Profile power consumption during a specific workload"""
        return self.power_profiler.profile_workload(workload_func, *args, **kwargs)

    def get_power_history(self) -> List[Dict]:
        """Get historical power consumption data"""
        return self.power_profiler.get_power_history()


class PowerManagedModel:
    """
    A wrapper for machine learning models that provides power management
    """
    
    def __init__(self, model, framework: PowerManagementFramework):
        self.model = model
        self.framework = framework
        self.logger = logging.getLogger(__name__)
    
    def predict(self, X, **kwargs):
        """Make a prediction with power management"""
        self.logger.info("Executing prediction with power management")
        return self.framework.execute_model_with_power_management(self.model, X, **kwargs)
    
    def fit(self, X, y, **kwargs):
        """Train the model with power management"""
        self.logger.info("Executing training with power management")
        training_data = (X, y)
        return self.framework.execute_training_with_power_management(self.model, training_data, **kwargs)


def create_optimized_framework() -> PowerManagementFramework:
    """Create and configure an optimized power management framework"""
    constraints = PowerConstraint(
        max_cpu_power_watts=25.0,  # Intel i5-10210U TDP
        max_gpu_power_watts=75.0,  # NVIDIA SM61 max power
        max_cpu_temp_celsius=90.0,
        max_gpu_temp_celsius=85.0,
        max_cpu_usage_percent=90.0,
        max_gpu_usage_percent=85.0
    )
    
    framework = PowerManagementFramework(constraints)
    
    # Set initial strategies
    framework.set_adaptation_strategy(AdaptationStrategy.BALANCED)
    framework.set_thermal_policy(ThermalPolicy.HYBRID)
    
    return framework


# Example usage and demonstration
if __name__ == "__main__":
    # Create and start the optimized framework
    framework = create_optimized_framework()
    framework.start_framework()
    
    # Simulate adding some tasks
    def cpu_intensive_task():
        # Simulate CPU-intensive work
        total = 0
        for i in range(1000000):
            total += i * i
        return total
    
    def gpu_intensive_task():
        # Simulate GPU-intensive work (would use actual GPU operations in real implementation)
        time.sleep(1)
        return "GPU task completed"
    
    # Add tasks with different priorities
    framework.add_task(cpu_intensive_task, priority=8)  # High priority
    framework.add_task(gpu_intensive_task, priority=5)  # Medium priority
    framework.add_task(cpu_intensive_task, priority=2)  # Low priority
    
    # Simulate optimizing for different workload types
    print("Optimizing for high performance workload...")
    framework.optimize_for_workload("high_performance")
    
    # Wait and check system health
    time.sleep(5)
    health = framework.get_system_health()
    if health:
        print(f"System Health: {health.__dict__}")
    
    print("\nOptimizing for power efficient workload...")
    framework.optimize_for_workload("power_efficient")
    
    # Wait and check system health again
    time.sleep(5)
    health = framework.get_system_health()
    if health:
        print(f"System Health: {health.__dict__}")
    
    # Get framework summary
    summary = framework.get_framework_summary()
    print(f"\nFramework Summary: {summary}")
    
    # Get power consumption estimate
    power_estimate = framework.get_power_consumption_estimate()
    print(f"Power Consumption Estimate: {power_estimate}")
    
    # Stop the framework
    framework.stop_framework()
    print("\nFramework stopped.")