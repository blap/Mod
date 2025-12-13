"""
Demonstration of the Integrated Power Management Framework for Intel i5-10210U + NVIDIA SM61
This script demonstrates how the power management system optimizes performance while 
managing power consumption and heat generation.
"""
import time
import threading
from power_management import PowerConstraint
from integrated_power_management import create_optimized_framework, PowerManagedModel


def cpu_intensive_task(task_id: int):
    """A CPU-intensive task that simulates computational work"""
    print(f"Task {task_id}: Starting CPU-intensive work...")
    
    # Simulate CPU-intensive computation
    total = 0
    for i in range(2000000):
        total += i * i
        if i % 500000 == 0:  # Print progress
            print(f"  Task {task_id}: Processed {i} iterations")
    
    print(f"Task {task_id}: Completed with result {total % 1000}")
    return total % 1000


def io_intensive_task(task_id: int):
    """An I/O-intensive task that simulates data processing"""
    print(f"Task {task_id}: Starting I/O-intensive work...")
    
    # Simulate I/O operations
    for i in range(5):
        time.sleep(0.5)  # Simulate I/O wait
        print(f"  Task {task_id}: I/O operation {i+1}/5 completed")
    
    print(f"Task {task_id}: I/O work completed")
    return f"Task {task_id} result"


def demo_power_management():
    """Demonstrate the power management framework"""
    print("=== Power Management Framework Demonstration ===\n")
    
    # Create and start the optimized framework
    framework = create_optimized_framework()
    framework.start_framework()
    
    print("Framework started. Current system status:")
    health = framework.get_system_health()
    if health:
        print(f"  CPU Usage: {health.cpu_usage}%")
        print(f"  GPU Usage: {health.gpu_usage}%")
        print(f"  CPU Temp: {health.cpu_temp}°C")
        print(f"  GPU Temp: {health.gpu_temp}°C")
        print(f"  Power Mode: {health.power_mode}")
        print(f"  Thermal Status: {health.thermal_status}")
        print(f"  Efficiency Score: {health.efficiency_score:.2f}")
    
    # Add various tasks with different priorities
    print("\nAdding tasks with different priorities...")
    framework.add_task(lambda: cpu_intensive_task(1), priority=8)  # High priority
    framework.add_task(lambda: io_intensive_task(2), priority=5)   # Medium priority
    framework.add_task(lambda: cpu_intensive_task(3), priority=3)  # Low priority
    framework.add_task(lambda: io_intensive_task(4), priority=10)  # Critical priority
    
    # Wait a bit to see the tasks execute
    print("\nWaiting for tasks to execute...")
    time.sleep(8)
    
    # Check system health again
    print("\nSystem health after task execution:")
    health = framework.get_system_health()
    if health:
        print(f"  CPU Usage: {health.cpu_usage}%")
        print(f"  GPU Usage: {health.gpu_usage}%")
        print(f"  CPU Temp: {health.cpu_temp}°C")
        print(f"  GPU Temp: {health.gpu_temp}°C")
        print(f"  Power Mode: {health.power_mode}")
        print(f"  Thermal Status: {health.thermal_status}")
        print(f"  Efficiency Score: {health.efficiency_score:.2f}")
    
    # Demonstrate workload optimization
    print("\n=== Demonstrating Workload Optimization ===")
    
    print("\nOptimizing for high performance workload...")
    framework.optimize_for_workload("high_performance")
    
    time.sleep(3)
    health = framework.get_system_health()
    if health:
        print(f"  System adjusted for high performance - Mode: {health.power_mode}")
    
    print("\nOptimizing for power efficient workload...")
    framework.optimize_for_workload("power_efficient")
    
    time.sleep(3)
    health = framework.get_system_health()
    if health:
        print(f"  System adjusted for power efficiency - Mode: {health.power_mode}")
    
    print("\nOptimizing for balanced workload...")
    framework.optimize_for_workload("balanced")
    
    time.sleep(3)
    health = framework.get_system_health()
    if health:
        print(f"  System adjusted for balanced operation - Mode: {health.power_mode}")
    
    # Get framework summary
    print("\n=== Framework Summary ===")
    summary = framework.get_framework_summary()
    print(f"Active: {summary['active']}")
    print(f"Scheduler Mode: {summary['scheduler_status']['mode']}")
    print(f"Pending Tasks: {summary['scheduler_status']['pending_tasks']}")
    print(f"Running Tasks: {summary['scheduler_status']['running_tasks']}")
    
    # Get power consumption estimate
    power_estimate = framework.get_power_consumption_estimate()
    print(f"\nPower Consumption Estimate:")
    print(f"  Estimated CPU Power: {power_estimate.get('estimated_cpu_power', 0):.2f}W")
    print(f"  Estimated GPU Power: {power_estimate.get('estimated_gpu_power', 0):.2f}W")
    print(f"  Total Estimated Power: {power_estimate.get('total_estimated_power', 0):.2f}W")
    print(f"  Power Efficiency Score: {power_estimate.get('power_efficiency_score', 0):.2f}")
    
    # Stop the framework
    framework.stop_framework()
    print("\nFramework stopped.")
    print("\n=== Power Management Framework Demonstration Complete ===")


def demo_adaptive_model():
    """Demonstrate the adaptive model wrapper"""
    print("\n=== Adaptive Model Demonstration ===")
    
    # Create a dummy model for demonstration
    class DummyModel:
        def predict(self, X):
            # Simulate model prediction
            time.sleep(0.5)  # Simulate processing time
            return [x * 2 for x in X]  # Simple transformation
        
        def fit(self, X, y):
            # Simulate model training
            time.sleep(1.0)  # Simulate training time
            return {"loss": 0.1, "epochs": 10}
    
    # Create framework
    framework = create_optimized_framework()
    framework.start_framework()
    
    # Create power-managed model
    dummy_model = DummyModel()
    power_managed_model = PowerManagedModel(dummy_model, framework)
    
    # Simulate some data
    input_data = [1, 2, 3, 4, 5]
    
    print("Executing prediction with power management...")
    prediction_result = power_managed_model.predict(input_data)
    print(f"Prediction result: {prediction_result}")
    
    print("Executing training with power management...")
    training_result = power_managed_model.fit(input_data, [2, 4, 6, 8, 10])
    print(f"Training result: {training_result}")
    
    # Stop framework
    framework.stop_framework()
    print("Adaptive model demonstration complete.")


if __name__ == "__main__":
    demo_power_management()
    demo_adaptive_model()