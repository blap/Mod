# Power Management System for Intel i5-10210U + NVIDIA SM61

## Overview

This power management system implements comprehensive techniques to optimize performance while managing power consumption and heat generation on the target hardware (Intel i5-10210U + NVIDIA SM61). The system includes:

- Power-aware scheduling algorithms
- Thermal management techniques
- Adaptive algorithms for power/thermal constraints
- Dynamic Voltage and Frequency Scaling (DVFS) where possible

## Components

### 1. Power-Aware Scheduler

The Power-Aware Scheduler manages task execution based on power and thermal constraints:

- Monitors CPU and GPU usage, temperature, and power consumption
- Adjusts task execution based on current system state
- Implements different power modes (Performance, Balanced, Power Save, Thermal Management)
- Prioritizes tasks based on their power requirements and priority levels

### 2. Thermal Management System

The Thermal Management System prevents overheating and maintains system stability:

- Monitors thermal zones (CPU and GPU)
- Controls cooling devices (fans)
- Implements thermal policies (Passive, Active, Hybrid)
- Adjusts cooling levels based on temperature thresholds
- Provides callbacks for thermal events

### 3. Adaptive Algorithms

The Adaptive Controller adjusts system behavior based on power and thermal constraints:

- Dynamically modifies performance factors
- Adjusts batch sizes and operation frequency
- Implements different adaptation strategies (Performance First, Power Efficient, Thermal Aware, Balanced)
- Maintains historical data for trend analysis
- Provides adaptive model wrapper for ML models

### 4. DVFS Controller

The DVFS Controller manages Dynamic Voltage and Frequency Scaling:

- Controls CPU frequency scaling on Linux and Windows
- Manages GPU power limits on NVIDIA systems
- Implements adaptive frequency scaling based on load and temperature
- Provides workload-based frequency optimization
- Supports different workload profiles

## Integration Framework

The Integrated Power Management Framework combines all components:

- Coordinated operation of all power management components
- Automatic adjustment based on system state
- Workload optimization for different scenarios
- Comprehensive system health monitoring
- Power consumption estimation

## Key Features

### Power Optimization
- Dynamic adjustment of performance parameters based on workload
- Power-aware task scheduling
- Frequency scaling to match performance requirements
- Efficient resource allocation

### Thermal Management
- Real-time temperature monitoring
- Proactive cooling adjustment
- Thermal event callbacks
- Temperature-based performance scaling

### Adaptive Behavior
- Automatic strategy selection based on system state
- Historical trend analysis
- Workload-specific optimizations
- Continuous adaptation to changing conditions

### Cross-Platform Support
- Linux support with full DVFS capabilities
- Windows support with power plan management
- Fallback mechanisms for systems without specific hardware support
- Robust error handling for unavailable sensors

## Usage Example

```python
from integrated_power_management import create_optimized_framework, PowerManagedModel

# Create and start the framework
framework = create_optimized_framework()
framework.start_framework()

# Add tasks with different priorities
framework.add_task(cpu_intensive_task, priority=8)
framework.add_task(io_intensive_task, priority=5)

# Optimize for different workload types
framework.optimize_for_workload("high_performance")
framework.optimize_for_workload("power_efficient")

# Execute ML models with power management
power_managed_model = PowerManagedModel(model, framework)
result = power_managed_model.predict(input_data)

# Monitor system health
health = framework.get_system_health()
print(f"Efficiency Score: {health.efficiency_score}")

# Stop the framework
framework.stop_framework()
```

## Performance Considerations

- The system balances performance and power consumption automatically
- Frequency scaling reduces power consumption during low-demand periods
- Thermal management prevents performance throttling due to overheating
- Adaptive algorithms maintain performance while respecting constraints
- Task scheduling prioritizes critical operations during thermal emergencies

## Hardware-Specific Optimizations

For Intel i5-10210U:
- TDP of 25W is respected
- Frequency scaling within CPU capabilities
- Temperature monitoring of CPU cores

For NVIDIA SM61:
- GPU power limits are managed via nvidia-smi
- Temperature monitoring for GPU
- Performance state adjustments for GPU

## Testing

The system includes comprehensive tests for:
- Individual component functionality
- Integration between components
- Edge cases and error conditions
- Performance under various load conditions

All tests pass successfully, ensuring the reliability of the power management system.