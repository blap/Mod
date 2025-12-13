# Hardware Detection and Fallback System for Power and Thermal Optimization

This repository contains a comprehensive hardware detection and fallback system for power and thermal optimization, specifically designed for the Intel i5-10210U + NVIDIA SM61 hardware platform but compatible with various configurations.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Components](#components)
6. [Fallback Mechanisms](#fallback-mechanisms)
7. [Validation](#validation)

## Overview

The hardware detection and fallback system provides a robust foundation for power and thermal optimization by:
- Detecting available hardware components at runtime
- Providing appropriate fallbacks when components are missing
- Optimizing operations based on detected hardware capabilities
- Ensuring reliable operation across different hardware configurations

## Features

- **Automatic Hardware Detection**: Detects CPU, GPU, temperature sensors, and power management capabilities
- **Comprehensive Fallbacks**: Graceful fallbacks when hardware components are unavailable
- **Power Management**: Adaptive power management based on thermal conditions
- **Thermal Management**: Proactive thermal management with performance throttling
- **Cross-Platform Compatibility**: Works on systems with or without dedicated GPUs
- **Performance Optimized**: Optimized for Intel i5-10210U + NVIDIA SM61 platform
- **Error Resilient**: Continues operation even when components fail

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/qwen3-vl.git

# Install required dependencies
pip install torch torchvision torchaudio
pip install psutil pynvml GPUtil
pip install cpuinfo  # Optional, for detailed CPU info
```

## Usage

### Basic Usage

```python
from src.qwen3_vl.components.system.hardware_detection_fallbacks import SafeHardwareInterface
from src.qwen3_vl.components.system.robust_power_management import create_robust_power_management_system
from src.qwen3_vl.components.system.robust_thermal_management import create_robust_thermal_management_system

# Create hardware-aware interface
hardware_interface = SafeHardwareInterface()

# Get detected hardware capabilities
caps = hardware_interface.hardware_detector.capabilities
print(f"CPU: {caps.cpu_available}, GPU: {caps.gpu_available}")

# Create power and thermal management systems
scheduler, thermal_manager = create_robust_power_management_system(hardware_interface)
thermal_manager = create_robust_thermal_management_system(hardware_interface)

# Use the systems with automatic fallbacks
cpu_temp = hardware_interface.get_temperature('cpu')  # Works with or without temp sensors
tensor = hardware_interface.allocate_tensor((100, 100))  # Falls back to CPU if GPU unavailable
```

### Advanced Usage

```python
# Generate hardware-optimized configuration
from src.qwen3_vl.components.system.hardware_detection_fallbacks import get_hardware_optimizer_config

config = get_hardware_optimizer_config(hardware_interface)
print(f"Optimized for: {config['hardware_specific']['cpu_model']} + {config['hardware_specific']['gpu_model']}")

# Validate hardware compatibility
from src.qwen3_vl.components.system.hardware_detection_fallbacks import validate_hardware_compatibility

compatibility = validate_hardware_compatibility(config)
print(f"Hardware compatibility: {compatibility}")
```

## Components

### 1. Hardware Detection System (`hardware_detection_fallbacks.py`)

The main hardware detection and fallback management system:

- **SafeHardwareInterface**: Main interface with detection and fallbacks
- **HardwareDetector**: Detects CPU, GPU, temperature sensors, and power management
- **FallbackManager**: Manages fallback mechanisms for missing components

### 2. Power Management System (`robust_power_management.py`)

Adaptive power management with hardware-aware optimizations:

- **RobustPowerAwareScheduler**: Power-aware task scheduler
- **PowerConstraint**: Defines system power and thermal limits
- **PowerState**: Represents current system power state

### 3. Thermal Management System (`robust_thermal_management.py`)

Proactive thermal management with performance protection:

- **RobustThermalManager**: Main thermal management system
- **ThermalAwareTask**: Tasks aware of thermal conditions
- **ThermalPolicy**: Different thermal management strategies

## Fallback Mechanisms

The system implements comprehensive fallback strategies:

### GPU Fallbacks
- Automatic fallback from GPU to CPU for tensor operations
- Memory allocation on CPU when GPU memory unavailable
- Performance adjustment based on available compute resources

### Temperature Fallbacks
- Default temperature values when sensors unavailable
- Conservative thermal management when no temperature data
- Continued operation without thermal monitoring

### Power Fallbacks
- CPU utilization-based power estimates when power management unavailable
- Conservative power management settings
- Continued operation without detailed power monitoring

### Memory Fallbacks
- CPU memory fallback when GPU memory exhausted
- Dynamic memory allocation based on availability
- Efficient memory pooling for both CPU and GPU

## Validation

The system has been thoroughly validated with:

- Hardware detection accuracy
- Fallback mechanism correctness
- Performance optimization effectiveness
- Error handling robustness
- Cross-platform compatibility

Run the validation test:
```bash
python test_hardware_fallbacks.py
```

## Performance Benefits

- **Reliability**: System continues operation regardless of hardware configuration
- **Optimization**: Performance optimized for detected hardware
- **Safety**: Thermal and power constraints respected
- **Flexibility**: Works on various hardware platforms
- **Maintainability**: Clean separation of concerns with fallback mechanisms

## Architecture Compatibility

The system is optimized for:
- **CPU**: Intel i5-10210U (4 cores, 8 threads, 15W TDP)
- **GPU**: NVIDIA SM61 (Maxwell architecture, compute capability 6.1)
- **Memory**: Systems with 8-16GB RAM

But maintains compatibility with:
- Various Intel/AMD CPU architectures
- Different NVIDIA GPU generations
- CPU-only systems
- Systems without temperature sensors

## License

This project is licensed under the MIT License - see the LICENSE file for details.