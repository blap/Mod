"""
Hardware Abstraction Layer for the Flexible Model System.

This module provides a unified interface for different hardware types
(CPU, GPU, TPU) with device detection, memory management, and performance
optimization features.
"""

import abc
import os
import platform
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import logging


class HardwareType(Enum):
    """Enumeration of supported hardware types."""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


class HardwareInterface(abc.ABC):
    """Abstract base class defining the hardware interface contract."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{device_id}")
        
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the name of the hardware device."""
        pass
    
    @property
    @abc.abstractmethod
    def type(self) -> HardwareType:
        """Return the type of hardware."""
        pass
    
    @abc.abstractmethod
    def initialize(self) -> bool:
        """Initialize the hardware device."""
        pass
    
    @abc.abstractmethod
    def cleanup(self) -> None:
        """Clean up resources associated with the hardware device."""
        pass
    
    @abc.abstractmethod
    def get_memory_info(self) -> Dict[str, Union[int, float]]:
        """Get memory information for the device."""
        pass
    
    @abc.abstractmethod
    def allocate_memory(self, size: int) -> Optional[object]:
        """Allocate memory on the device."""
        pass
    
    @abc.abstractmethod
    def free_memory(self, memory_handle: object) -> None:
        """Free allocated memory on the device."""
        pass
    
    @abc.abstractmethod
    def transfer_data(self, data: object, destination_device: 'HardwareInterface') -> object:
        """Transfer data between devices."""
        pass
    
    @abc.abstractmethod
    def get_performance_metrics(self) -> Dict[str, Union[int, float]]:
        """Get performance metrics for the device."""
        pass
    
    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if the hardware device is available."""
        pass


class CPUAdapter(HardwareInterface):
    """CPU hardware adapter implementation."""
    
    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self._initialized = False
        
    @property
    def name(self) -> str:
        """Return the name of the CPU."""
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            return f"{info['brand_raw']} ({platform.processor()})"
        except ImportError:
            return f"CPU-{self.device_id}"
    
    @property
    def type(self) -> HardwareType:
        return HardwareType.CPU
    
    def initialize(self) -> bool:
        """Initialize the CPU device."""
        try:
            # On CPU, initialization is mostly about checking system capabilities
            self._initialized = True
            self.logger.info(f"CPU device {self.device_id} initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize CPU device {self.device_id}: {str(e)}")
            return False
    
    def cleanup(self) -> None:
        """Clean up CPU resources."""
        self._initialized = False
        self.logger.info(f"CPU device {self.device_id} cleaned up")
    
    def get_memory_info(self) -> Dict[str, Union[int, float]]:
        """Get system memory information."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent_used': memory.percent,
                'free': memory.free
            }
        except ImportError:
            # Fallback if psutil is not available
            total_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            return {'total': total_memory, 'available': total_memory}
    
    def allocate_memory(self, size: int) -> Optional[object]:
        """Allocate memory on CPU."""
        if not self._initialized:
            raise RuntimeError("CPU device not initialized")
        
        try:
            # Allocate memory by creating a bytearray of the specified size
            memory_block = bytearray(size)
            self.logger.debug(f"Allocated {size} bytes on CPU")
            return memory_block
        except MemoryError:
            self.logger.error(f"Failed to allocate {size} bytes on CPU: insufficient memory")
            return None
    
    def free_memory(self, memory_handle: object) -> None:
        """Free allocated memory on CPU."""
        # In Python, memory is managed automatically by garbage collector
        # We just delete the reference here
        del memory_handle
        self.logger.debug("Memory freed on CPU")
    
    def transfer_data(self, data: object, destination_device: 'HardwareInterface') -> object:
        """Transfer data from CPU to another device."""
        if destination_device.type == HardwareType.CPU:
            # Same device type, just return the same data
            return data
        else:
            # For transfers to other devices, we typically convert to a format
            # compatible with the destination (this is a simplified implementation)
            return data
    
    def get_performance_metrics(self) -> Dict[str, Union[int, float]]:
        """Get CPU performance metrics."""
        try:
            import psutil
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'cpu_count': psutil.cpu_count(),
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
            }
        except ImportError:
            # Fallback if psutil is not available
            return {
                'cpu_percent': 0,
                'cpu_count': os.cpu_count(),
                'load_average': (0, 0, 0)
            }
    
    def is_available(self) -> bool:
        """Check if the CPU device is available."""
        return True


class GPUAdapter(HardwareInterface):
    """Base GPU adapter implementation."""
    
    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self.vendor = "Unknown"
        self._initialized = False
        
    @property
    def type(self) -> HardwareType:
        return HardwareType.GPU
    
    def get_memory_info(self) -> Dict[str, Union[int, float]]:
        """Get GPU memory information."""
        # This will be implemented by subclasses
        return {}
    
    def allocate_memory(self, size: int) -> Optional[object]:
        """Allocate memory on GPU."""
        # This will be implemented by subclasses
        return None
    
    def free_memory(self, memory_handle: object) -> None:
        """Free allocated memory on GPU."""
        # This will be implemented by subclasses
        pass
    
    def transfer_data(self, data: object, destination_device: 'HardwareInterface') -> object:
        """Transfer data between GPU and other devices."""
        # This will be implemented by subclasses
        return data
    
    def get_performance_metrics(self) -> Dict[str, Union[int, float]]:
        """Get GPU performance metrics."""
        # This will be implemented by subclasses
        return {}
    
    def is_available(self) -> bool:
        """Check if the GPU device is available."""
        # This will be implemented by subclasses
        return False


class NVIDIAGPUAdapter(GPUAdapter):
    """NVIDIA GPU adapter implementation."""
    
    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self.vendor = "NVIDIA"
        self.cuda_available = False
        self._cuda_handle = None
        
        # Try to initialize CUDA
        try:
            import torch
            if torch.cuda.is_available():
                self.cuda_available = True
                # Store device handle
                self._cuda_handle = torch.device(f'cuda:{self.device_id}')
        except ImportError:
            self.logger.warning("PyTorch not available, CUDA support disabled")
        except Exception as e:
            self.logger.warning(f"CUDA initialization failed: {str(e)}")
    
    @property
    def name(self) -> str:
        """Return the name of the NVIDIA GPU."""
        if self.cuda_available:
            try:
                import torch
                return torch.cuda.get_device_name(self.device_id)
            except Exception:
                return f"NVIDIA GPU-{self.device_id}"
        return f"NVIDIA GPU-{self.device_id} (unavailable)"
    
    def initialize(self) -> bool:
        """Initialize the NVIDIA GPU device."""
        if not self.cuda_available:
            self.logger.error(f"NVIDIA GPU {self.device_id} not available")
            return False
            
        try:
            import torch
            if self.device_id >= torch.cuda.device_count():
                self.logger.error(f"GPU device {self.device_id} does not exist")
                return False
                
            # Set the device
            torch.cuda.set_device(self.device_id)
            
            # Clear cache to free up memory
            torch.cuda.empty_cache()
            
            self._initialized = True
            self.logger.info(f"NVIDIA GPU {self.device_id} initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize NVIDIA GPU {self.device_id}: {str(e)}")
            return False
    
    def cleanup(self) -> None:
        """Clean up NVIDIA GPU resources."""
        if self._initialized:
            try:
                import torch
                torch.cuda.synchronize(self.device_id)
                torch.cuda.empty_cache()
            except Exception as e:
                self.logger.error(f"Error during NVIDIA GPU cleanup: {str(e)}")
                
            self._initialized = False
            self.logger.info(f"NVIDIA GPU {self.device_id} cleaned up")
    
    def get_memory_info(self) -> Dict[str, Union[int, float]]:
        """Get NVIDIA GPU memory information."""
        if not self.cuda_available:
            return {}
        
        try:
            import torch
            memory_stats = torch.cuda.memory_stats(self.device_id)
            return {
                'allocated_bytes.all.current': memory_stats.get('allocated_bytes.all.current', 0),
                'allocated_bytes.all.peak': memory_stats.get('allocated_bytes.all.peak', 0),
                'reserved_bytes.all.current': memory_stats.get('reserved_bytes.all.current', 0),
                'reserved_bytes.all.peak': memory_stats.get('reserved_bytes.all.peak', 0),
                'total': torch.cuda.get_device_properties(self.device_id).total_memory,
                'free': torch.cuda.get_device_properties(self.device_id).total_memory - 
                       memory_stats.get('reserved_bytes.all.current', 0)
            }
        except Exception as e:
            self.logger.error(f"Failed to get NVIDIA GPU memory info: {str(e)}")
            return {}
    
    def allocate_memory(self, size: int) -> Optional[object]:
        """Allocate memory on NVIDIA GPU."""
        if not self._initialized:
            raise RuntimeError("NVIDIA GPU device not initialized")
        
        try:
            import torch
            # Allocate a tensor of zeros with the specified size in bytes
            # Convert bytes to number of float32 elements (4 bytes per element)
            num_elements = size // 4
            if num_elements <= 0:
                num_elements = 1
            gpu_tensor = torch.zeros(num_elements, dtype=torch.float32, device=self._cuda_handle)
            self.logger.debug(f"Allocated {size} bytes on NVIDIA GPU")
            return gpu_tensor
        except Exception as e:
            self.logger.error(f"Failed to allocate {size} bytes on NVIDIA GPU: {str(e)}")
            return None
    
    def free_memory(self, memory_handle: object) -> None:
        """Free allocated memory on NVIDIA GPU."""
        if memory_handle is not None:
            del memory_handle
            import torch
            torch.cuda.empty_cache()
            self.logger.debug("Memory freed on NVIDIA GPU")
    
    def transfer_data(self, data: object, destination_device: 'HardwareInterface') -> object:
        """Transfer data from NVIDIA GPU to another device."""
        if destination_device.type == HardwareType.CPU:
            try:
                import torch
                # Transfer from GPU to CPU
                if isinstance(data, torch.Tensor) and data.is_cuda:
                    return data.cpu()
                return data
            except Exception as e:
                self.logger.error(f"Failed to transfer data from NVIDIA GPU to CPU: {str(e)}")
                return data
        elif destination_device.type == HardwareType.GPU and isinstance(destination_device, NVIDIAGPUAdapter):
            # Transfer between NVIDIA GPUs
            try:
                import torch
                if isinstance(data, torch.Tensor) and data.is_cuda:
                    return data.to(destination_device._cuda_handle)
                return data
            except Exception as e:
                self.logger.error(f"Failed to transfer data between NVIDIA GPUs: {str(e)}")
                return data
        else:
            # For other device types, first move to CPU then to destination
            cpu_data = self.transfer_data(data, CPUAdapter())
            return destination_device.transfer_data(cpu_data, destination_device)
    
    def get_performance_metrics(self) -> Dict[str, Union[int, float]]:
        """Get NVIDIA GPU performance metrics."""
        if not self.cuda_available:
            return {}
        
        try:
            import torch
            return {
                'gpu_utilization': torch.cuda.utilization(self.device_id) if hasattr(torch.cuda, 'utilization') else 0,
                'memory_utilization': (torch.cuda.memory_reserved(self.device_id) / 
                                     torch.cuda.get_device_properties(self.device_id).total_memory) * 100,
                'temperature': torch.cuda.temperature(self.device_id) if hasattr(torch.cuda, 'temperature') else 0,
                'power_draw': torch.cuda.power_draw(self.device_id) if hasattr(torch.cuda, 'power_draw') else 0,
                'fan_speed': torch.cuda.fan_speed(self.device_id) if hasattr(torch.cuda, 'fan_speed') else 0
            }
        except Exception as e:
            self.logger.error(f"Failed to get NVIDIA GPU performance metrics: {str(e)}")
            return {}
    
    def is_available(self) -> bool:
        """Check if the NVIDIA GPU device is available."""
        return self.cuda_available and self.device_id < self.get_device_count()
    
    def get_device_count(self) -> int:
        """Get the number of available NVIDIA GPUs."""
        try:
            import torch
            return torch.cuda.device_count()
        except:
            return 0


class AMDGPUAdapter(GPUAdapter):
    """AMD GPU adapter implementation."""
    
    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self.vendor = "AMD"
        self.rocm_available = False
        self._rocm_handle = None
        
        # Check for ROCm availability
        try:
            # Attempt to check if ROCm is available
            import torch
            # PyTorch supports ROCm through the hip backend
            if hasattr(torch.version, 'hip') and torch.version.hip:
                self.rocm_available = torch.cuda.is_available()  # ROCm uses the same API as CUDA
        except ImportError:
            self.logger.warning("PyTorch not available, ROCm support disabled")
        except Exception as e:
            self.logger.warning(f"ROCm initialization failed: {str(e)}")
    
    @property
    def name(self) -> str:
        """Return the name of the AMD GPU."""
        if self.rocm_available:
            try:
                import torch
                return torch.cuda.get_device_name(self.device_id)
            except Exception:
                return f"AMD GPU-{self.device_id}"
        return f"AMD GPU-{self.device_id} (unavailable)"
    
    def initialize(self) -> bool:
        """Initialize the AMD GPU device."""
        if not self.rocm_available:
            self.logger.error(f"AMD GPU {self.device_id} not available")
            return False
            
        try:
            import torch
            if self.device_id >= torch.cuda.device_count():
                self.logger.error(f"GPU device {self.device_id} does not exist")
                return False
                
            # Set the device
            torch.cuda.set_device(self.device_id)
            
            # Clear cache to free up memory
            torch.cuda.empty_cache()
            
            self._initialized = True
            self.logger.info(f"AMD GPU {self.device_id} initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize AMD GPU {self.device_id}: {str(e)}")
            return False
    
    def cleanup(self) -> None:
        """Clean up AMD GPU resources."""
        if self._initialized:
            try:
                import torch
                torch.cuda.synchronize(self.device_id)
                torch.cuda.empty_cache()
            except Exception as e:
                self.logger.error(f"Error during AMD GPU cleanup: {str(e)}")
                
            self._initialized = False
            self.logger.info(f"AMD GPU {self.device_id} cleaned up")
    
    def get_memory_info(self) -> Dict[str, Union[int, float]]:
        """Get AMD GPU memory information."""
        if not self.rocm_available:
            return {}
        
        try:
            import torch
            memory_stats = torch.cuda.memory_stats(self.device_id)
            return {
                'allocated_bytes.all.current': memory_stats.get('allocated_bytes.all.current', 0),
                'allocated_bytes.all.peak': memory_stats.get('allocated_bytes.all.peak', 0),
                'reserved_bytes.all.current': memory_stats.get('reserved_bytes.all.current', 0),
                'reserved_bytes.all.peak': memory_stats.get('reserved_bytes.all.peak', 0),
                'total': torch.cuda.get_device_properties(self.device_id).total_memory,
                'free': torch.cuda.get_device_properties(self.device_id).total_memory - 
                       memory_stats.get('reserved_bytes.all.current', 0)
            }
        except Exception as e:
            self.logger.error(f"Failed to get AMD GPU memory info: {str(e)}")
            return {}
    
    def allocate_memory(self, size: int) -> Optional[object]:
        """Allocate memory on AMD GPU."""
        if not self._initialized:
            raise RuntimeError("AMD GPU device not initialized")
        
        try:
            import torch
            # Allocate a tensor of zeros with the specified size in bytes
            # Convert bytes to number of float32 elements (4 bytes per element)
            num_elements = size // 4
            if num_elements <= 0:
                num_elements = 1
            gpu_tensor = torch.zeros(num_elements, dtype=torch.float32, device=f'cuda:{self.device_id}')
            self.logger.debug(f"Allocated {size} bytes on AMD GPU")
            return gpu_tensor
        except Exception as e:
            self.logger.error(f"Failed to allocate {size} bytes on AMD GPU: {str(e)}")
            return None
    
    def free_memory(self, memory_handle: object) -> None:
        """Free allocated memory on AMD GPU."""
        if memory_handle is not None:
            del memory_handle
            import torch
            torch.cuda.empty_cache()
            self.logger.debug("Memory freed on AMD GPU")
    
    def transfer_data(self, data: object, destination_device: 'HardwareInterface') -> object:
        """Transfer data from AMD GPU to another device."""
        if destination_device.type == HardwareType.CPU:
            try:
                import torch
                # Transfer from GPU to CPU
                if isinstance(data, torch.Tensor) and data.is_cuda:
                    return data.cpu()
                return data
            except Exception as e:
                self.logger.error(f"Failed to transfer data from AMD GPU to CPU: {str(e)}")
                return data
        elif destination_device.type == HardwareType.GPU and isinstance(destination_device, AMDGPUAdapter):
            # Transfer between AMD GPUs
            try:
                import torch
                if isinstance(data, torch.Tensor) and data.is_cuda:
                    return data.to(f'cuda:{destination_device.device_id}')
                return data
            except Exception as e:
                self.logger.error(f"Failed to transfer data between AMD GPUs: {str(e)}")
                return data
        else:
            # For other device types, first move to CPU then to destination
            cpu_data = self.transfer_data(data, CPUAdapter())
            return destination_device.transfer_data(cpu_data, destination_device)
    
    def get_performance_metrics(self) -> Dict[str, Union[int, float]]:
        """Get AMD GPU performance metrics."""
        if not self.rocm_available:
            return {}
        
        try:
            import torch
            return {
                'gpu_utilization': torch.cuda.utilization(self.device_id) if hasattr(torch.cuda, 'utilization') else 0,
                'memory_utilization': (torch.cuda.memory_reserved(self.device_id) / 
                                     torch.cuda.get_device_properties(self.device_id).total_memory) * 100,
                'temperature': torch.cuda.temperature(self.device_id) if hasattr(torch.cuda, 'temperature') else 0,
                'power_draw': torch.cuda.power_draw(self.device_id) if hasattr(torch.cuda, 'power_draw') else 0,
                'fan_speed': torch.cuda.fan_speed(self.device_id) if hasattr(torch.cuda, 'fan_speed') else 0
            }
        except Exception as e:
            self.logger.error(f"Failed to get AMD GPU performance metrics: {str(e)}")
            return {}
    
    def is_available(self) -> bool:
        """Check if the AMD GPU device is available."""
        return self.rocm_available and self.device_id < self.get_device_count()
    
    def get_device_count(self) -> int:
        """Get the number of available AMD GPUs."""
        try:
            import torch
            return torch.cuda.device_count()
        except:
            return 0


class TPUAdapter(HardwareInterface):
    """TPU hardware adapter implementation."""
    
    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self.tpu_available = False
        self._tpu_handle = None
        
        # Check for TPU availability
        try:
            import torch_xla.core.xla_model as xm
            import torch_xla.experimental.pjrt as pjrt
            # Check if TPUs are available
            if pjrt.device_type() == 'TPU' or len(xm.get_xla_supported_devices(devkind='TPU')) > 0:
                self.tpu_available = True
        except ImportError:
            self.logger.warning("PyTorch/XLA not available, TPU support disabled")
        except Exception as e:
            self.logger.warning(f"TPU initialization failed: {str(e)}")
    
    @property
    def name(self) -> str:
        """Return the name of the TPU."""
        if self.tpu_available:
            return f"TPU-{self.device_id}"
        return f"TPU-{self.device_id} (unavailable)"
    
    @property
    def type(self) -> HardwareType:
        return HardwareType.TPU
    
    def initialize(self) -> bool:
        """Initialize the TPU device."""
        if not self.tpu_available:
            self.logger.error(f"TPU {self.device_id} not available")
            return False
            
        try:
            import torch_xla.core.xla_model as xm
            # Get TPU device
            self._tpu_handle = xm.xla_device(device_type='TPU', devkind='TPU')
            
            self.logger.info(f"TPU {self.device_id} initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize TPU {self.device_id}: {str(e)}")
            return False
    
    def cleanup(self) -> None:
        """Clean up TPU resources."""
        # TPUs don't typically need explicit cleanup
        self.logger.info(f"TPU {self.device_id} cleaned up")
    
    def get_memory_info(self) -> Dict[str, Union[int, float]]:
        """Get TPU memory information."""
        # TPU memory information is not easily accessible through PyTorch/XLA
        # This is a placeholder implementation
        return {
            'total': 0,  # Placeholder - actual TPU memory varies by version
            'available': 0,
            'used': 0
        }
    
    def allocate_memory(self, size: int) -> Optional[object]:
        """Allocate memory on TPU."""
        if not self.tpu_available:
            raise RuntimeError("TPU device not available")
        
        try:
            import torch
            import torch_xla.core.xla_model as xm
            
            # Allocate a tensor of zeros with the specified size in bytes
            # Convert bytes to number of float32 elements (4 bytes per element)
            num_elements = size // 4
            if num_elements <= 0:
                num_elements = 1
            tpu_tensor = torch.zeros(num_elements, dtype=torch.float32).to(self._tpu_handle)
            self.logger.debug(f"Allocated {size} bytes on TPU")
            return tpu_tensor
        except Exception as e:
            self.logger.error(f"Failed to allocate {size} bytes on TPU: {str(e)}")
            return None
    
    def free_memory(self, memory_handle: object) -> None:
        """Free allocated memory on TPU."""
        if memory_handle is not None:
            del memory_handle
            import torch_xla.core.xla_model as xm
            xm.rendezvous("free_memory")  # Synchronize across TPU cores
            self.logger.debug("Memory freed on TPU")
    
    def transfer_data(self, data: object, destination_device: 'HardwareInterface') -> object:
        """Transfer data from TPU to another device."""
        if destination_device.type == HardwareType.CPU:
            try:
                import torch
                # Transfer from TPU to CPU
                if hasattr(data, 'cpu'):
                    return data.cpu()
                return data
            except Exception as e:
                self.logger.error(f"Failed to transfer data from TPU to CPU: {str(e)}")
                return data
        else:
            # For other device types, first move to CPU then to destination
            cpu_data = self.transfer_data(data, CPUAdapter())
            return destination_device.transfer_data(cpu_data, destination_device)
    
    def get_performance_metrics(self) -> Dict[str, Union[int, float]]:
        """Get TPU performance metrics."""
        # TPU metrics are not easily accessible through PyTorch/XLA
        # This is a placeholder implementation
        return {
            'core_utilization': 0,
            'interconnect_utilization': 0,
            'hbm_utilization': 0
        }
    
    def is_available(self) -> bool:
        """Check if the TPU device is available."""
        return self.tpu_available


class HardwareManager:
    """Manages hardware devices and provides device detection and enumeration."""
    
    def __init__(self):
        self.devices: List[HardwareInterface] = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self._detect_and_enumerate_devices()
    
    def _detect_and_enumerate_devices(self) -> None:
        """Detect and enumerate available hardware devices."""
        self.devices = []
        
        # Detect CPUs
        try:
            cpu_adapter = CPUAdapter(device_id=0)
            if cpu_adapter.is_available():
                self.devices.append(cpu_adapter)
        except Exception as e:
            self.logger.warning(f"Failed to detect CPU: {str(e)}")
        
        # Detect NVIDIA GPUs
        try:
            nvidia_gpu = NVIDIAGPUAdapter(device_id=0)
            if nvidia_gpu.is_available():
                # Add all available NVIDIA GPUs
                for i in range(nvidia_gpu.get_device_count()):
                    gpu = NVIDIAGPUAdapter(device_id=i)
                    if gpu.initialize():
                        self.devices.append(gpu)
                        gpu.cleanup()  # Clean up after adding to avoid resource conflicts
        except Exception as e:
            self.logger.warning(f"Failed to detect NVIDIA GPUs: {str(e)}")
        
        # Detect AMD GPUs
        try:
            amd_gpu = AMDGPUAdapter(device_id=0)
            if amd_gpu.is_available():
                # Add all available AMD GPUs
                for i in range(amd_gpu.get_device_count()):
                    gpu = AMDGPUAdapter(device_id=i)
                    if gpu.initialize():
                        self.devices.append(gpu)
                        gpu.cleanup()  # Clean up after adding to avoid resource conflicts
        except Exception as e:
            self.logger.warning(f"Failed to detect AMD GPUs: {str(e)}")
        
        # Detect TPUs
        try:
            tpu = TPUAdapter(device_id=0)
            if tpu.is_available():
                self.devices.append(tpu)
        except Exception as e:
            self.logger.warning(f"Failed to detect TPUs: {str(e)}")
    
    def get_devices_by_type(self, hardware_type: HardwareType) -> List[HardwareInterface]:
        """Get all devices of a specific type."""
        return [device for device in self.devices if device.type == hardware_type]
    
    def get_device_by_id(self, device_id: int, hardware_type: HardwareType) -> Optional[HardwareInterface]:
        """Get a specific device by ID and type."""
        devices = self.get_devices_by_type(hardware_type)
        for device in devices:
            if device.device_id == device_id:
                return device
        return None
    
    def get_best_device(self, preferred_type: HardwareType = HardwareType.GPU) -> Optional[HardwareInterface]:
        """Get the best available device, preferring the specified type."""
        # First, try to get the preferred device type
        devices = self.get_devices_by_type(preferred_type)
        if devices:
            return devices[0]  # Return the first available device of preferred type
        
        # If preferred type is not available, fall back to other types in order of preference
        fallback_order = [
            HardwareType.GPU,  # Prefer GPU over others
            HardwareType.TPU,  # Then TPU
            HardwareType.CPU   # Finally CPU
        ]
        
        for hw_type in fallback_order:
            if hw_type != preferred_type:
                devices = self.get_devices_by_type(hw_type)
                if devices:
                    return devices[0]
        
        return None
    
    def get_all_devices(self) -> List[HardwareInterface]:
        """Get all detected devices."""
        return self.devices.copy()


def get_hardware_manager() -> HardwareManager:
    """Get a singleton instance of the hardware manager."""
    if not hasattr(get_hardware_manager, '_instance'):
        get_hardware_manager._instance = HardwareManager()
    return get_hardware_manager._instance