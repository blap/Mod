"""
Hardware Detection System for Qwen3-VL Project
Supports Intel i5-10210U and i7-8700 CPU detection and optimization

This module provides runtime detection of specific CPU models and their capabilities,
enabling adaptive optimization based on detected hardware.
"""
import platform
import subprocess
import re
import psutil
import cpuinfo
import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CPUModel(Enum):
    """Enumeration of supported CPU models"""
    INTEL_I5_10210U = "Intel i5-10210U"
    INTEL_I7_8700 = "Intel i7-8700"
    UNKNOWN = "Unknown"


@dataclass
class CPUFeatures:
    """CPU features and capabilities"""
    model: CPUModel
    cores: int
    threads: int
    max_frequency: float  # in GHz
    l1_cache_size: int  # in bytes
    l2_cache_size: int  # in bytes
    l3_cache_size: int  # in bytes
    cache_line_size: int  # in bytes
    avx_support: bool
    avx2_support: bool
    sse_support: bool
    sse2_support: bool
    sse3_support: bool
    sse4_1_support: bool
    sse4_2_support: bool
    avx512_support: bool


class CPUDetector:
    """Hardware detection system that identifies specific CPUs at runtime"""
    
    def __init__(self):
        self.cpu_features: Optional[CPUFeatures] = None
        self._detected_cpu_model: Optional[CPUModel] = None
        
    def detect_cpu(self) -> CPUFeatures:
        """
        Detect the specific CPU model and its capabilities at runtime.
        
        Returns:
            CPUFeatures object containing detected CPU information
        """
        if self.cpu_features is not None:
            return self.cpu_features
            
        # Get CPU info using cpuinfo
        cpu_info = cpuinfo.get_cpu_info()
        
        # Extract relevant information
        processor_string = cpu_info.get('brand_raw', '').lower()
        flags = cpu_info.get('flags', [])
        
        # Detect specific CPU model
        self._detected_cpu_model = self._identify_cpu_model(processor_string)
        
        # Get hardware characteristics
        cores = psutil.cpu_count(logical=False)
        threads = psutil.cpu_count(logical=True)
        
        # Determine cache sizes based on CPU model
        l1_cache, l2_cache, l3_cache = self._get_cache_sizes(self._detected_cpu_model)
        
        # Determine frequency (best effort)
        max_frequency = self._get_max_frequency()
        
        # Determine SIMD capabilities
        cpu_features = CPUFeatures(
            model=self._detected_cpu_model,
            cores=cores,
            threads=threads,
            max_frequency=max_frequency,
            l1_cache_size=l1_cache,
            l2_cache_size=l2_cache,
            l3_cache_size=l3_cache,
            cache_line_size=64,  # Standard cache line size
            avx_support='avx' in flags,
            avx2_support='avx2' in flags,
            sse_support='sse' in flags,
            sse2_support='sse2' in flags,
            sse3_support='sse3' in flags,
            sse4_1_support='sse4_1' in flags,
            sse4_2_support='sse4_2' in flags,
            avx512_support='avx512f' in flags or 'avx512cd' in flags
        )
        
        self.cpu_features = cpu_features
        logger.info(f"Detected CPU: {cpu_features.model.value} with {cpu_features.cores} cores, "
                   f"{cpu_features.threads} threads, and {cpu_features.l3_cache_size / (1024*1024):.1f}MB L3 cache")
        
        return cpu_features
    
    def _identify_cpu_model(self, processor_string: str) -> CPUModel:
        """
        Identify the specific CPU model based on processor string.
        
        Args:
            processor_string: Lowercase processor brand string
            
        Returns:
            CPUModel enum value
        """
        processor_string = processor_string.lower()
        
        if 'i5-10210u' in processor_string:
            return CPUModel.INTEL_I5_10210U
        elif 'i7-8700' in processor_string:
            return CPUModel.INTEL_I7_8700
        else:
            # Try to identify based on other characteristics
            if 'intel' in processor_string:
                # Check for characteristics of known CPUs
                cores = psutil.cpu_count(logical=False)
                threads = psutil.cpu_count(logical=True)
                
                # i7-8700 has 6 cores, 12 threads
                if cores == 6 and threads == 12:
                    return CPUModel.INTEL_I7_8700
                # i5-10210U has 4 cores, 8 threads
                elif cores == 4 and threads == 8:
                    return CPUModel.INTEL_I5_10210U
            
            return CPUModel.UNKNOWN
    
    def _get_cache_sizes(self, cpu_model: CPUModel) -> Tuple[int, int, int]:
        """
        Get cache sizes based on CPU model.
        
        Args:
            cpu_model: The detected CPU model
            
        Returns:
            Tuple of (L1, L2, L3) cache sizes in bytes
        """
        if cpu_model == CPUModel.INTEL_I5_10210U:
            # Intel i5-10210U specifications
            return (32 * 1024, 256 * 1024, 6 * 1024 * 1024)  # 32KB per core, 256KB per core, 6MB shared
        elif cpu_model == CPUModel.INTEL_I7_8700:
            # Intel i7-8700 specifications
            return (32 * 1024, 256 * 1024, 12 * 1024 * 1024)  # 32KB per core, 256KB per core, 12MB shared
        else:
            # Default values for unknown CPUs
            cores = psutil.cpu_count(logical=False)
            return (32 * 1024, 256 * 1024, cores * 2 * 1024 * 1024)  # Estimate based on cores
    
    def _get_max_frequency(self) -> float:
        """
        Get the maximum CPU frequency.
        
        Returns:
            Maximum frequency in GHz
        """
        try:
            # Try to get frequency from cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            max_freq_mhz = cpu_info.get('max_freq', 0)
            if max_freq_mhz > 0:
                return max_freq_mhz / 1000.0  # Convert MHz to GHz
            
            # Fallback: try to get from system
            if platform.system() == "Windows":
                # Use wmic on Windows
                result = subprocess.run(
                    ["wmic", "cpu", "get", "MaxClockSpeed", "/value"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        if 'MaxClockSpeed=' in line:
                            try:
                                freq_mhz = int(line.split('=')[1].strip())
                                return freq_mhz / 1000.0  # Convert MHz to GHz
                            except (ValueError, IndexError):
                                pass
            elif platform.system() in ["Linux", "Darwin"]:
                # Try different methods on Unix-like systems
                try:
                    with open('/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq', 'r') as f:
                        freq_khz = int(f.read().strip())
                        return freq_khz / 1000000.0  # Convert kHz to GHz
                except (FileNotFoundError, ValueError):
                    # Try /proc/cpuinfo on Linux
                    try:
                        with open('/proc/cpuinfo', 'r') as f:
                            for line in f:
                                if 'cpu MHz' in line:
                                    try:
                                        freq_mhz = float(line.split(':')[1].strip())
                                        return freq_mhz / 1000.0  # Convert MHz to GHz
                                    except (ValueError, IndexError):
                                        continue
                    except FileNotFoundError:
                        pass
        
        except Exception as e:
            logger.warning(f"Could not determine CPU frequency: {e}")
        
        # Fallback to a reasonable default
        return 3.0  # 3.0 GHz as a reasonable default
    
    def get_cpu_features(self) -> CPUFeatures:
        """
        Get the detected CPU features. Will detect if not already done.
        
        Returns:
            CPUFeatures object with detected features
        """
        if self.cpu_features is None:
            return self.detect_cpu()
        return self.cpu_features
    
    def is_specific_model_detected(self, model: CPUModel) -> bool:
        """
        Check if a specific CPU model is detected.
        
        Args:
            model: CPUModel to check for
            
        Returns:
            True if the specified model is detected, False otherwise
        """
        detected = self.get_cpu_features()
        return detected.model == model
    
    def get_optimal_thread_count(self) -> int:
        """
        Get the optimal number of threads based on detected CPU.
        
        Returns:
            Recommended thread count
        """
        features = self.get_cpu_features()
        
        if features.model == CPUModel.INTEL_I7_8700:
            # i7-8700 has 6 cores with hyperthreading (12 threads), use most of them
            return min(10, features.threads)
        elif features.model == CPUModel.INTEL_I5_10210U:
            # i5-10210U has 4 cores with hyperthreading (8 threads)
            return min(6, features.threads)
        else:
            # For unknown CPUs, use a reasonable default
            return min(8, features.threads)


class HardwareOptimizer:
    """Hardware-specific optimizer that loads appropriate profiles based on detected hardware"""
    
    def __init__(self, cpu_detector: CPUDetector):
        self.cpu_detector = cpu_detector
        self.cpu_features = cpu_detector.get_cpu_features()
        self.optimization_profile = self._load_optimization_profile()
    
    def _load_optimization_profile(self) -> Dict[str, Any]:
        """
        Load CPU-specific optimization profile based on detected hardware.
        
        Returns:
            Dictionary containing optimization parameters
        """
        if self.cpu_features.model == CPUModel.INTEL_I7_8700:
            # i7-8700 profile: 6 cores, 12 threads, 12MB L3 cache
            return {
                'cpu_optimizations': {
                    'num_workers': 8,  # Use most of the available threads
                    'batch_size_multiplier': 1.5,  # Can handle larger batches
                    'memory_layout_optimization': True,
                    'simd_optimization_level': 'avx2',
                    'cache_optimization': True,
                    'l3_cache_size': self.cpu_features.l3_cache_size,
                    'use_hyperthreading': True
                },
                'memory_management': {
                    'memory_pool_size': 4 * 1024 * 1024 * 1024,  # 4GB for larger models
                    'cache_line_alignment': 64,
                    'l2_cache_blocking': True,
                    'l3_cache_blocking': True
                },
                'threading_model': {
                    'max_threads': 10,  # Leave 2 threads free for system
                    'thread_affinity_enabled': True,
                    'work_stealing_enabled': True
                },
                'simd_optimizations': {
                    'enable_avx2': self.cpu_features.avx2_support,
                    'enable_avx512': self.cpu_features.avx512_support,
                    'vector_width': 8 if self.cpu_features.avx2_support else 4,  # 8 floats for AVX2, 4 for SSE
                    'instruction_set': 'avx2' if self.cpu_features.avx2_support else 'sse'
                }
            }
        elif self.cpu_features.model == CPUModel.INTEL_I5_10210U:
            # i5-10210U profile: 4 cores, 8 threads, 6MB L3 cache
            return {
                'cpu_optimizations': {
                    'num_workers': 6,  # Use most of the available threads
                    'batch_size_multiplier': 1.0,  # Moderate batch sizes
                    'memory_layout_optimization': True,
                    'simd_optimization_level': 'avx2',
                    'cache_optimization': True,
                    'l3_cache_size': self.cpu_features.l3_cache_size,
                    'use_hyperthreading': True
                },
                'memory_management': {
                    'memory_pool_size': 3 * 1024 * 1024 * 1024,  # 3GB for moderate models
                    'cache_line_alignment': 64,
                    'l2_cache_blocking': True,
                    'l3_cache_blocking': True
                },
                'threading_model': {
                    'max_threads': 7,  # Leave 1 thread free for system
                    'thread_affinity_enabled': True,
                    'work_stealing_enabled': True
                },
                'simd_optimizations': {
                    'enable_avx2': self.cpu_features.avx2_support,
                    'enable_avx512': self.cpu_features.avx512_support,
                    'vector_width': 8 if self.cpu_features.avx2_support else 4,  # 8 floats for AVX2, 4 for SSE
                    'instruction_set': 'avx2' if self.cpu_features.avx2_support else 'sse'
                }
            }
        else:
            # Default profile for unknown CPUs
            return {
                'cpu_optimizations': {
                    'num_workers': max(2, self.cpu_features.threads // 2),
                    'batch_size_multiplier': 1.0,
                    'memory_layout_optimization': False,
                    'simd_optimization_level': 'none',
                    'cache_optimization': False,
                    'l3_cache_size': self.cpu_features.l3_cache_size,
                    'use_hyperthreading': True
                },
                'memory_management': {
                    'memory_pool_size': 2 * 1024 * 1024 * 1024,  # 2GB default
                    'cache_line_alignment': 64,
                    'l2_cache_blocking': False,
                    'l3_cache_blocking': False
                },
                'threading_model': {
                    'max_threads': self.cpu_features.threads,
                    'thread_affinity_enabled': False,
                    'work_stealing_enabled': False
                },
                'simd_optimizations': {
                    'enable_avx2': self.cpu_features.avx2_support,
                    'enable_avx512': self.cpu_features.avx512_support,
                    'vector_width': 4 if self.cpu_features.sse_support else 1,
                    'instruction_set': 'sse' if self.cpu_features.sse_support else 'none'
                }
            }
    
    def get_optimization_profile(self) -> Dict[str, Any]:
        """
        Get the loaded optimization profile.
        
        Returns:
            Dictionary containing optimization parameters
        """
        return self.optimization_profile
    
    def get_recommended_config(self) -> Dict[str, Any]:
        """
        Get recommended configuration based on detected hardware.
        
        Returns:
            Dictionary containing recommended configuration parameters
        """
        profile = self.get_optimization_profile()
        
        config = {
            # Threading configuration
            'num_threads': profile['threading_model']['max_threads'],
            'thread_affinity_enabled': profile['threading_model']['thread_affinity_enabled'],
            
            # Memory configuration
            'memory_pool_size': profile['memory_management']['memory_pool_size'],
            'cache_line_alignment': profile['memory_management']['cache_line_alignment'],
            
            # SIMD configuration
            'simd_instruction_set': profile['simd_optimizations']['instruction_set'],
            'enable_avx2': profile['simd_optimizations']['enable_avx2'],
            'enable_avx512': profile['simd_optimizations']['enable_avx512'],
            
            # Performance optimizations
            'enable_cache_optimization': profile['cpu_optimizations']['cache_optimization'],
            'use_hyperthreading': profile['cpu_optimizations']['use_hyperthreading'],
            'num_workers': profile['cpu_optimizations']['num_workers'],
        }
        
        return config


def get_hardware_optimizer() -> HardwareOptimizer:
    """
    Factory function to get a hardware optimizer with automatic CPU detection.
    
    Returns:
        HardwareOptimizer instance
    """
    detector = CPUDetector()
    return HardwareOptimizer(detector)


def get_detected_cpu_info() -> Dict[str, Any]:
    """
    Get information about the detected CPU in a simple dictionary format.
    
    Returns:
        Dictionary containing CPU information
    """
    detector = CPUDetector()
    features = detector.get_cpu_features()
    
    return {
        'model': features.model.value,
        'cores': features.cores,
        'threads': features.threads,
        'max_frequency_ghz': features.max_frequency,
        'l1_cache_mb': features.l1_cache_size / (1024 * 1024),
        'l2_cache_mb': features.l2_cache_size / (1024 * 1024),
        'l3_cache_mb': features.l3_cache_size / (1024 * 1024),
        'avx_support': features.avx_support,
        'avx2_support': features.avx2_support,
        'avx512_support': features.avx512_support,
        'is_i7_8700': detector.is_specific_model_detected(CPUModel.INTEL_I7_8700),
        'is_i5_10210u': detector.is_specific_model_detected(CPUModel.INTEL_I5_10210U),
        'recommended_threads': detector.get_optimal_thread_count()
    }


if __name__ == "__main__":
    print("Hardware Detection System for Qwen3-VL")
    print("=" * 50)
    
    # Initialize detector and optimizer
    detector = CPUDetector()
    optimizer = HardwareOptimizer(detector)
    
    # Get CPU features
    features = detector.get_cpu_features()
    print(f"Detected CPU: {features.model.value}")
    print(f"Cores: {features.cores}, Threads: {features.threads}")
    print(f"Max Frequency: {features.max_frequency:.2f} GHz")
    print(f"L1 Cache: {features.l1_cache_size / 1024:.0f} KB per core")
    print(f"L2 Cache: {features.l2_cache_size / 1024:.0f} KB per core")
    print(f"L3 Cache: {features.l3_cache_size / (1024*1024):.1f} MB shared")
    print(f"AVX2 Support: {features.avx2_support}")
    print(f"AVX512 Support: {features.avx512_support}")
    
    # Get optimization profile
    profile = optimizer.get_optimization_profile()
    print(f"\nOptimization Profile: {features.model.value}")
    print(f"Recommended workers: {profile['cpu_optimizations']['num_workers']}")
    print(f"Memory pool size: {profile['memory_management']['memory_pool_size'] / (1024*1024*1024):.1f} GB")
    print(f"Max threads: {profile['threading_model']['max_threads']}")
    print(f"SIMD instruction set: {profile['simd_optimizations']['instruction_set']}")
    
    # Get recommended config
    config = optimizer.get_recommended_config()
    print(f"\nRecommended Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nHardware detection system initialized successfully!")