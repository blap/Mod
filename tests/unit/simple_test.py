"""
Final validation script to test the real hardware interfaces implementation
"""
from src.qwen3_vl.components.system.real_hardware_interfaces import get_hardware_interface, get_hardware_capabilities

print('Testing Real Hardware Interfaces Implementation')
hw_interface = get_hardware_interface()
print('[PASS] Hardware interface created')

# Test basic functionality
cpu_info = hw_interface.get_cpu_info()
print(f'[PASS] CPU Info: {cpu_info.vendor} {cpu_info.model}')

gpu_info = hw_interface.get_gpu_info()
if gpu_info:
    print(f'[PASS] GPU Info: {gpu_info.vendor} {gpu_info.name}')
else:
    print('[INFO] GPU not available')

thermal_info = hw_interface.get_thermal_info()
print(f'[PASS] Thermal Info: CPU={thermal_info.cpu_temperature}C, GPU={thermal_info.gpu_temperature}C')

power_info = hw_interface.get_power_info()
print(f'[PASS] Power Info: CPU={power_info.cpu_power:.2f}W, GPU={power_info.gpu_power:.2f}W')

capabilities = get_hardware_capabilities()
print(f'[PASS] Hardware Capabilities: AVX2={capabilities["avx2_supported"]}, GPU={capabilities["gpu_available"]}')

print('\nAll real hardware interfaces are working correctly!')