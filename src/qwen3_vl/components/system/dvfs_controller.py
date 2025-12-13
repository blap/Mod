"""
Dynamic Voltage and Frequency Scaling (DVFS) Implementation
This module implements DVFS techniques to optimize power consumption by adjusting
voltage and frequency based on workload requirements.
"""
import time
import os
import platform
import subprocess
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import psutil
import threading


@dataclass
class FrequencyState:
    """Represents a CPU frequency state (P-state)"""
    name: str
    frequency_mhz: int
    voltage_mv: Optional[int] = None
    power_watts: Optional[float] = None


@dataclass
class GpuFrequencyState:
    """Represents a GPU frequency state"""
    name: str
    core_clock_mhz: int
    memory_clock_mhz: int
    power_limit_watts: float


class DVFSController:
    """
    Controller for Dynamic Voltage and Frequency Scaling (DVFS)
    This implementation provides a cross-platform interface for frequency scaling,
    with specific implementations for different operating systems.
    """
    
    def __init__(self):
        self.system = platform.system().lower()
        self.is_dvfs_available = self._check_dvfs_availability()
        self.current_cpu_freq_state = None
        self.current_gpu_freq_state = None
        self.freq_states = self._get_available_frequency_states()
        self.gpu_freq_states = self._get_available_gpu_frequency_states()
        self.is_active = False
        self.scaling_thread: Optional[threading.Thread] = None

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _check_dvfs_availability(self) -> bool:
        """Check if DVFS is available on this system"""
        if self.system == "linux":
            # Check if cpufreq is available
            return os.path.exists("/sys/devices/system/cpu/cpu0/cpufreq/")
        elif self.system == "windows":
            # On Windows, we can use powercfg to adjust power plans
            try:
                result = subprocess.run(['powercfg', '/list'], 
                                      capture_output=True, text=True, timeout=5)
                return result.returncode == 0
            except:
                return False
        elif self.system == "darwin":  # macOS
            # macOS has limited DVFS support
            return False
        else:
            return False
    
    def _get_available_frequency_states(self) -> List[FrequencyState]:
        """Get available CPU frequency states"""
        states = []
        
        if self.system == "linux":
            try:
                # Read available frequencies from cpufreq
                with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies", "r") as f:
                    freqs = f.read().strip().split()
                    for freq in freqs:
                        states.append(FrequencyState(
                            name=f"P{len(states)}",
                            frequency_mhz=int(freq) // 1000  # Convert from kHz to MHz
                        ))
            except:
                # Fallback: create some common states
                states = [
                    FrequencyState("P0", 4200, power_watts=25.0),  # Max frequency
                    FrequencyState("P1", 3500, power_watts=20.0),
                    FrequencyState("P2", 2800, power_watts=15.0),
                    FrequencyState("P3", 2100, power_watts=10.0),
                    FrequencyState("P4", 1400, power_watts=7.0),   # Min frequency
                ]
        elif self.system == "windows":
            # Windows doesn't expose frequency states directly, so we create representative states
            states = [
                FrequencyState("High Performance", 4200, power_watts=25.0),
                FrequencyState("Balanced", 3000, power_watts=15.0),
                FrequencyState("Power Saver", 1800, power_watts=7.0),
            ]
        
        return states
    
    def _get_available_gpu_frequency_states(self) -> List[GpuFrequencyState]:
        """Get available GPU frequency states (NVIDIA-specific)"""
        states = []

        try:
            # Try to get GPU info using nvidia-smi
            result = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=name,performance.state,power.limit',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                # Parse the output to get GPU info
                # For NVIDIA SM61 (GeForce GTX 1050/1060 Mobile), we'll define typical states
                states = [
                    GpuFrequencyState("Max Performance", 1835, 4000, 75.0),
                    GpuFrequencyState("Balanced", 1600, 3500, 60.0),
                    GpuFrequencyState("Power Saver", 1200, 2500, 45.0),
                ]
        except:
            # Fallback - don't add any states if nvidia-smi is not available
            # This could be due to no GPU, no nvidia-smi, or other issues
            pass  # Return empty list if no GPU is available

        return states
    
    def get_current_frequency(self) -> Optional[int]:
        """Get current CPU frequency in MHz"""
        if self.system == "linux":
            try:
                with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r") as f:
                    return int(f.read().strip()) // 1000  # Convert from kHz to MHz
            except:
                # Fallback to psutil
                return int(psutil.cpu_freq().current)
        elif self.system == "windows":
            try:
                # Use wmic to get current CPU frequency
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'CurrentClockSpeed'], 
                    capture_output=True, text=True, timeout=5
                )
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    return int(lines[1].strip())
            except:
                # Fallback to psutil
                return int(psutil.cpu_freq().current)
        else:
            # Use psutil for other systems
            return int(psutil.cpu_freq().current)
    
    def get_max_frequency(self) -> Optional[int]:
        """Get maximum CPU frequency in MHz"""
        if self.system == "linux":
            try:
                with open("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "r") as f:
                    return int(f.read().strip()) // 1000  # Convert from kHz to MHz
            except:
                return int(psutil.cpu_freq().max)
        else:
            return int(psutil.cpu_freq().max)
    
    def get_min_frequency(self) -> Optional[int]:
        """Get minimum CPU frequency in MHz"""
        if self.system == "linux":
            try:
                with open("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq", "r") as f:
                    return int(f.read().strip()) // 1000  # Convert from kHz to MHz
            except:
                return int(psutil.cpu_freq().min)
        else:
            return int(psutil.cpu_freq().min)
    
    def set_frequency_state(self, state_name: str) -> bool:
        """Set CPU to a specific frequency state"""
        if not self.is_dvfs_available:
            self.logger.warning("DVFS not available on this system")
            return False
        
        state = next((s for s in self.freq_states if s.name == state_name), None)
        if not state:
            self.logger.error(f"Frequency state {state_name} not found")
            return False
        
        success = False
        if self.system == "linux":
            success = self._set_linux_frequency(state.frequency_mhz)
        elif self.system == "windows":
            success = self._set_windows_frequency(state_name)
        
        if success:
            self.current_cpu_freq_state = state
            self.logger.info(f"Set CPU frequency to {state.frequency_mhz} MHz ({state.name})")
        
        return success
    
    def _set_linux_frequency(self, target_freq_mhz: int) -> bool:
        """Set frequency on Linux systems"""
        try:
            # Convert MHz to kHz for the sysfs interface
            target_freq_khz = target_freq_mhz * 1000
            
            # Write to scaling_setspeed (if available) or scaling_max_freq
            freq_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed"
            if os.path.exists(freq_path):
                with open(freq_path, "w") as f:
                    f.write(str(target_freq_khz))
            else:
                # Fallback to setting max frequency
                max_freq_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"
                with open(max_freq_path, "w") as f:
                    f.write(str(target_freq_khz))
            
            return True
        except PermissionError:
            self.logger.error("Permission denied. Try running as root/administrator.")
            return False
        except Exception as e:
            self.logger.error(f"Error setting frequency: {str(e)}")
            return False
    
    def _set_windows_frequency(self, state_name: str) -> bool:
        """Set frequency on Windows systems using powercfg"""
        try:
            if state_name == "High Performance":
                # Set to high performance power plan
                subprocess.run(['powercfg', '/setactive', '8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c'], 
                             check=True, timeout=10)
            elif state_name == "Balanced":
                # Set to balanced power plan
                subprocess.run(['powercfg', '/setactive', '381b4222-f694-41f0-9685-ff5bb260df2e'], 
                             check=True, timeout=10)
            elif state_name == "Power Saver":
                # Set to power saver power plan
                subprocess.run(['powercfg', '/setactive', 'a1841308-3541-4fab-bc81-f71556f20b4a'], 
                             check=True, timeout=10)
            
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error setting Windows power plan: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error setting Windows frequency: {str(e)}")
            return False
    
    def set_gpu_frequency_state(self, state_name: str) -> bool:
        """Set GPU to a specific frequency state (NVIDIA-specific)"""
        if not self._gpu_available():
            self.logger.warning("GPU not available or nvidia-smi not found")
            return False
        
        state = next((s for s in self.gpu_freq_states if s.name == state_name), None)
        if not state:
            self.logger.error(f"GPU frequency state {state_name} not found")
            return False
        
        try:
            # Set GPU power limit
            subprocess.run([
                'nvidia-smi', 
                '-pl', str(state.power_limit_watts)  # Power limit in watts
            ], check=True, timeout=10)
            
            # For more advanced GPU frequency control, we would need to use nvidia-ml-py
            # or similar libraries, but this is a basic implementation
            self.current_gpu_freq_state = state
            self.logger.info(f"Set GPU power limit to {state.power_limit_watts}W ({state.name})")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error setting GPU power limit: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error setting GPU frequency: {str(e)}")
            return False
    
    def _gpu_available(self) -> bool:
        """Check if GPU is available"""
        try:
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def get_frequency_states(self) -> List[FrequencyState]:
        """Get available frequency states"""
        return self.freq_states
    
    def get_gpu_frequency_states(self) -> List[GpuFrequencyState]:
        """Get available GPU frequency states"""
        return self.gpu_freq_states
    
    def start_adaptive_scaling(self, interval: float = 2.0):
        """Start adaptive frequency scaling based on system load"""
        if self.is_active:
            return
        
        self.is_active = True
        self.scaling_thread = threading.Thread(
            target=self._adaptive_scaling_loop,
            args=(interval,)
        )
        self.scaling_thread.daemon = True
        self.scaling_thread.start()
        
        self.logger.info("Started adaptive frequency scaling")
    
    def stop_adaptive_scaling(self):
        """Stop adaptive frequency scaling"""
        self.is_active = False
        if self.scaling_thread:
            self.scaling_thread.join(timeout=2.0)
        
        self.logger.info("Stopped adaptive frequency scaling")
    
    def _adaptive_scaling_loop(self, interval: float):
        """Main adaptive scaling loop"""
        while self.is_active:
            try:
                # Get current system metrics
                cpu_usage = psutil.cpu_percent(interval=1)
                cpu_temp = self._get_cpu_temperature()
                
                # Determine appropriate frequency based on load and temperature
                target_state = self._determine_frequency_state(cpu_usage, cpu_temp)
                
                # Apply the frequency state
                if target_state:
                    self.set_frequency_state(target_state)
                
                # Sleep for the specified interval
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in adaptive scaling loop: {str(e)}")
                time.sleep(interval)  # Continue even if there's an error
    
    def _get_cpu_temperature(self) -> float:
        """Get current CPU temperature"""
        try:
            # Check if sensors_temperatures is available in psutil
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:  # If sensors_temperatures returns data
                    if 'coretemp' in temps:
                        # Intel CPU temperature
                        return max([temp.current for temp in temps['coretemp']])
                    elif 'cpu_thermal' in temps:
                        # Raspberry Pi or other ARM systems
                        return temps['cpu_thermal'][0].current
                    else:
                        # Use first available temperature sensor
                        first_sensor = next(iter(temps.values()))
                        if first_sensor:
                            return first_sensor[0].current
                        else:
                            # Default to 40°C if no temperature sensor found
                            return 40.0
                else:
                    # No temperature sensors available
                    return 40.0
            else:
                # sensors_temperatures not available on this platform
                return 40.0
        except:
            return 40.0
    
    def _determine_frequency_state(self, cpu_usage: float, cpu_temp: float) -> Optional[str]:
        """Determine appropriate frequency state based on CPU usage and temperature"""
        # Define temperature thresholds
        temp_critical = 90.0  # Critical temperature
        temp_warning = 80.0   # Warning temperature
        
        # Define usage thresholds
        usage_high = 80.0     # High usage
        usage_low = 20.0      # Low usage
        
        # Prioritize temperature over usage
        if cpu_temp >= temp_critical:
            # Critical temperature - use minimum frequency to cool down
            return self._get_min_frequency_state()
        elif cpu_temp >= temp_warning:
            # High temperature - use lower frequency
            if cpu_usage > usage_high:
                # High usage but high temp - balance performance and cooling
                return self._get_balanced_frequency_state()
            else:
                # Low usage, high temp - use lower frequency
                return self._get_lower_frequency_state()
        elif cpu_usage > usage_high:
            # High usage, acceptable temp - use higher frequency
            return self._get_max_frequency_state()
        elif cpu_usage < usage_low:
            # Low usage, acceptable temp - use lower frequency for power saving
            return self._get_min_frequency_state()
        else:
            # Medium usage, acceptable temp - use balanced frequency
            return self._get_balanced_frequency_state()
    
    def _get_max_frequency_state(self) -> Optional[str]:
        """Get the name of the highest frequency state"""
        if not self.freq_states:
            return None
        # Find state with highest frequency
        max_state = max(self.freq_states, key=lambda s: s.frequency_mhz)
        return max_state.name
    
    def _get_min_frequency_state(self) -> Optional[str]:
        """Get the name of the lowest frequency state"""
        if not self.freq_states:
            return None
        # Find state with lowest frequency
        min_state = min(self.freq_states, key=lambda s: s.frequency_mhz)
        return min_state.name
    
    def _get_balanced_frequency_state(self) -> Optional[str]:
        """Get the name of a balanced frequency state"""
        if not self.freq_states:
            return None
        # Find a middle state
        mid_idx = len(self.freq_states) // 2
        return self.freq_states[mid_idx].name
    
    def _get_lower_frequency_state(self) -> Optional[str]:
        """Get the name of a lower frequency state"""
        if not self.freq_states:
            return None
        # Find a state in the lower half
        low_idx = len(self.freq_states) // 4
        return self.freq_states[max(0, low_idx)].name
    
    def get_system_power_efficiency(self) -> Dict[str, float]:
        """Get metrics about system power efficiency"""
        current_freq = self.get_current_frequency()
        max_freq = self.get_max_frequency()
        min_freq = self.get_min_frequency()
        
        if max_freq and min_freq and max_freq != min_freq:
            efficiency_ratio = (current_freq - min_freq) / (max_freq - min_freq) if current_freq else 0.0
        else:
            efficiency_ratio = 0.5  # Default to 50% if we can't calculate
        
        cpu_usage = psutil.cpu_percent(interval=1)
        cpu_temp = self._get_cpu_temperature()
        
        return {
            'current_frequency_mhz': current_freq,
            'max_frequency_mhz': max_freq,
            'min_frequency_mhz': min_freq,
            'efficiency_ratio': efficiency_ratio,
            'cpu_usage_percent': cpu_usage,
            'cpu_temp_celsius': cpu_temp,
            'is_dvfs_available': self.is_dvfs_available
        }


class WorkloadBasedDVFS:
    """
    DVFS controller that adjusts frequency based on workload characteristics.
    """
    
    def __init__(self, dvfs_controller: DVFSController):
        self.dvfs_controller = dvfs_controller
        self.workload_profiles: Dict[str, Dict] = {}
        self.current_workload = "unknown"
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
    
    def register_workload_profile(self, name: str, profile: Dict[str, Any]):
        """Register a profile for a specific workload type"""
        self.workload_profiles[name] = profile
        self.logger.info(f"Registered workload profile: {name}")
    
    def set_workload_type(self, workload_type: str):
        """Set the current workload type to optimize frequency accordingly"""
        if workload_type not in self.workload_profiles:
            self.logger.warning(f"Workload profile '{workload_type}' not registered")
            self.current_workload = "unknown"
            return
        
        self.current_workload = workload_type
        profile = self.workload_profiles[workload_type]
        
        # Apply frequency settings based on workload profile
        if 'preferred_frequency_state' in profile:
            self.dvfs_controller.set_frequency_state(profile['preferred_frequency_state'])
        
        self.logger.info(f"Set workload type to {workload_type}")
    
    def execute_with_frequency_optimization(self, func, *args, **kwargs):
        """Execute a function with frequency optimized for the current workload"""
        # Determine if we need to adjust frequency based on function characteristics
        # This is a simplified example - in practice, you might analyze the function
        # or its expected resource usage
        
        if self.current_workload != "unknown":
            profile = self.workload_profiles[self.current_workload]
            if 'execution_frequency_state' in profile:
                self.dvfs_controller.set_frequency_state(profile['execution_frequency_state'])
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # Optionally, revert to a default state after execution
        # This would depend on the specific use case
        
        return result


# Example workload profiles
WORKLOAD_PROFILES = {
    "high_performance": {
        "description": "Workloads requiring maximum performance",
        "preferred_frequency_state": "High Performance",
        "execution_frequency_state": "High Performance",
        "priority": "high"
    },
    "power_efficient": {
        "description": "Power-efficient workloads",
        "preferred_frequency_state": "Power Saver",
        "execution_frequency_state": "Power Saver",
        "priority": "low"
    },
    "balanced": {
        "description": "General purpose workloads",
        "preferred_frequency_state": "Balanced",
        "execution_frequency_state": "Balanced",
        "priority": "medium"
    }
}


if __name__ == "__main__":
    # Example usage
    dvfs = DVFSController()
    
    print(f"DVFS available: {dvfs.is_dvfs_available}")
    print(f"Current frequency: {dvfs.get_current_frequency()} MHz")
    print(f"Max frequency: {dvfs.get_max_frequency()} MHz")
    print(f"Min frequency: {dvfs.get_min_frequency()} MHz")
    
    # Print available frequency states
    print("\nAvailable CPU frequency states:")
    for state in dvfs.get_frequency_states():
        print(f"  {state.name}: {state.frequency_mhz} MHz")
    
    # Print available GPU frequency states
    print("\nAvailable GPU frequency states:")
    for state in dvfs.get_gpu_frequency_states():
        print(f"  {state.name}: Core {state.core_clock_mhz} MHz, Memory {state.memory_clock_mhz} MHz, Power {state.power_limit_watts}W")
    
    # Example of using WorkloadBasedDVFS
    workload_dvfs = WorkloadBasedDVFS(dvfs)
    
    # Register workload profiles
    for name, profile in WORKLOAD_PROFILES.items():
        workload_dvfs.register_workload_profile(name, profile)
    
    # Set workload type and execute a function
    workload_dvfs.set_workload_type("high_performance")
    
    # Get system power efficiency metrics
    efficiency = dvfs.get_system_power_efficiency()
    print(f"\nSystem power efficiency: {efficiency}")