"""
CPU-Specific Optimization Profiles for Intel i5-10210U and i7-8700
This module defines optimization profiles tailored to each specific CPU model
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List
from enum import Enum


class CPUModel(Enum):
    """Enumeration of supported CPU models"""
    INTEL_I5_10210U = "Intel i5-10210U"
    INTEL_I7_8700 = "Intel i7-8700"
    UNKNOWN = "Unknown"


@dataclass
class CPUSpecificConfig:
    """Base configuration for CPU-specific optimizations"""
    # General CPU settings
    cpu_model: CPUModel
    num_cores: int
    num_threads: int
    max_frequency: float  # in GHz
    l3_cache_size: int  # in bytes
    
    # Threading configuration
    num_workers: int
    max_concurrent_threads: int
    thread_affinity_enabled: bool
    use_hyperthreading: bool
    
    # Memory optimization
    memory_pool_size: int  # in bytes
    l1_cache_size: int  # in bytes
    l2_cache_size: int  # in bytes
    l3_cache_size_bytes: int  # in bytes
    cache_line_size: int  # in bytes
    memory_threshold: float  # 0.0 to 1.0
    
    # SIMD and vectorization
    simd_instruction_set: str
    enable_avx2: bool
    enable_avx512: bool
    vector_width: int  # number of floats processed simultaneously
    
    # Performance optimization
    batch_size_multiplier: float
    pipeline_depth: int
    pipeline_buffer_size: int
    enable_cache_optimization: bool
    cache_blocking_enabled: bool
    
    # Power and thermal management
    power_constraint: float  # 0.0 to 1.0
    thermal_constraint: float  # in Celsius
    performance_target: float  # 0.0 to 1.0


class CPUProfileFactory:
    """Factory for creating CPU-specific optimization profiles"""
    
    @staticmethod
    def create_i5_10210u_profile() -> CPUSpecificConfig:
        """
        Create optimization profile for Intel i5-10210U
        Specifications: 4 cores, 8 threads, 6MB L3 cache, up to 4.2GHz boost
        """
        return CPUSpecificConfig(
            cpu_model=CPUModel.INTEL_I5_10210U,
            num_cores=4,
            num_threads=8,
            max_frequency=4.2,  # GHz
            l3_cache_size=6 * 1024 * 1024,  # 6MB
            
            # Threading configuration
            num_workers=6,  # Use most threads but leave some for system
            max_concurrent_threads=7,  # Leave 1 thread free
            thread_affinity_enabled=True,
            use_hyperthreading=True,
            
            # Memory optimization
            memory_pool_size=3 * 1024 * 1024 * 1024,  # 3GB
            l1_cache_size=32 * 1024,  # 32KB per core
            l2_cache_size=256 * 1024,  # 256KB per core
            l3_cache_size_bytes=6 * 1024 * 1024 * 1024,  # 6MB
            cache_line_size=64,  # Standard 64 bytes
            memory_threshold=0.8,  # Use up to 80% of available memory
            
            # SIMD and vectorization
            simd_instruction_set="avx2",
            enable_avx2=True,
            enable_avx512=False,  # i5-10210U doesn't support AVX-512
            vector_width=8,  # 8 floats with AVX2 (256-bit registers)
            
            # Performance optimization
            batch_size_multiplier=1.0,  # Standard batch sizes
            pipeline_depth=3,  # Moderate pipeline depth
            pipeline_buffer_size=4,  # Moderate buffer size
            enable_cache_optimization=True,
            cache_blocking_enabled=True,
            
            # Power and thermal management
            power_constraint=0.85,  # Limit to 85% of TDP for thermal management
            thermal_constraint=75.0,  # Keep under 75°C
            performance_target=0.85  # Target 85% performance
        )
    
    @staticmethod
    def create_i7_8700_profile() -> CPUSpecificConfig:
        """
        Create optimization profile for Intel i7-8700
        Specifications: 6 cores, 12 threads, 12MB L3 cache, up to 4.6GHz boost
        """
        return CPUSpecificConfig(
            cpu_model=CPUModel.INTEL_I7_8700,
            num_cores=6,
            num_threads=12,
            max_frequency=4.6,  # GHz
            l3_cache_size=12 * 1024 * 1024,  # 12MB
            
            # Threading configuration
            num_workers=10,  # Use most threads but leave some for system
            max_concurrent_threads=11,  # Leave 1 thread free
            thread_affinity_enabled=True,
            use_hyperthreading=True,
            
            # Memory optimization
            memory_pool_size=5 * 1024 * 1024 * 1024,  # 5GB - larger pool for more cores
            l1_cache_size=32 * 1024,  # 32KB per core
            l2_cache_size=256 * 1024,  # 256KB per core
            l3_cache_size_bytes=12 * 1024 * 1024 * 1024,  # 12MB
            cache_line_size=64,  # Standard 64 bytes
            memory_threshold=0.85,  # Can use slightly more memory due to more cores
            
            # SIMD and vectorization
            simd_instruction_set="avx2",
            enable_avx2=True,
            enable_avx512=False,  # i7-8700 doesn't support AVX-512
            vector_width=8,  # 8 floats with AVX2 (256-bit registers)
            
            # Performance optimization
            batch_size_multiplier=1.5,  # Larger batches due to more cores/caches
            pipeline_depth=4,  # Deeper pipeline to keep more cores busy
            pipeline_buffer_size=6,  # Larger buffer for deeper pipeline
            enable_cache_optimization=True,
            cache_blocking_enabled=True,
            
            # Power and thermal management
            power_constraint=0.9,  # Can use more power due to higher TDP
            thermal_constraint=80.0,  # Can run slightly hotter due to better cooling
            performance_target=0.9  # Target 90% performance
        )
    
    @staticmethod
    def create_generic_profile(num_cores: int, num_threads: int, 
                             l3_cache_size: int) -> CPUSpecificConfig:
        """
        Create a generic optimization profile for unknown CPUs
        
        Args:
            num_cores: Number of physical CPU cores
            num_threads: Number of logical CPU threads
            l3_cache_size: L3 cache size in bytes
            
        Returns:
            CPUSpecificConfig for the generic profile
        """
        # Calculate parameters based on available resources
        memory_pool_size = min(8 * 1024 * 1024 * 1024,  # Max 8GB
                              max(2 * 1024 * 1024 * 1024,  # Min 2GB
                                  num_cores * 1024 * 1024 * 1024))  # 1GB per core
        
        return CPUSpecificConfig(
            cpu_model=CPUModel.UNKNOWN,
            num_cores=num_cores,
            num_threads=num_threads,
            max_frequency=3.0,  # Default frequency
            l3_cache_size=l3_cache_size,
            
            # Threading configuration (conservative)
            num_workers=min(num_threads, 8),  # Max 8 workers
            max_concurrent_threads=num_threads,
            thread_affinity_enabled=False,  # Only enable on known CPUs
            use_hyperthreading=True,
            
            # Memory optimization
            memory_pool_size=memory_pool_size,
            l1_cache_size=32 * 1024,  # Standard 32KB per core
            l2_cache_size=256 * 1024,  # Standard 256KB per core
            l3_cache_size_bytes=l3_cache_size,
            cache_line_size=64,  # Standard 64 bytes
            memory_threshold=0.7,  # Conservative memory usage
            
            # SIMD and vectorization (detect at runtime)
            simd_instruction_set="auto",
            enable_avx2=False,  # Will be detected at runtime
            enable_avx512=False,
            vector_width=4,  # Default to SSE width
            
            # Performance optimization
            batch_size_multiplier=1.0,  # Standard batch sizes
            pipeline_depth=3,  # Moderate pipeline
            pipeline_buffer_size=4,  # Standard buffer size
            enable_cache_optimization=False,  # Only enable on known CPUs
            cache_blocking_enabled=False,
            
            # Power and thermal management (conservative)
            power_constraint=0.75,  # Conservative power usage
            thermal_constraint=70.0,  # Conservative thermal limit
            performance_target=0.75  # Conservative performance target
        )


@dataclass
class AdaptiveOptimizationManager:
    """Manages adaptive optimizations based on detected CPU"""
    config: CPUSpecificConfig
    
    def get_thread_count(self) -> int:
        """Get the recommended number of threads based on CPU"""
        return self.config.max_concurrent_threads
    
    def get_memory_pool_size(self) -> int:
        """Get the recommended memory pool size based on CPU"""
        return self.config.memory_pool_size
    
    def get_batch_size_multiplier(self) -> float:
        """Get the recommended batch size multiplier based on CPU"""
        return self.config.batch_size_multiplier
    
    def get_pipeline_config(self) -> Dict[str, int]:
        """Get pipeline configuration based on CPU"""
        return {
            'depth': self.config.pipeline_depth,
            'buffer_size': self.config.pipeline_buffer_size
        }
    
    def get_cache_optimization_settings(self) -> Dict[str, Any]:
        """Get cache optimization settings based on CPU"""
        return {
            'enabled': self.config.enable_cache_optimization,
            'cache_blocking': self.config.cache_blocking_enabled,
            'l1_size': self.config.l1_cache_size,
            'l2_size': self.config.l2_cache_size,
            'l3_size': self.config.l3_cache_size_bytes,
            'line_size': self.config.cache_line_size
        }
    
    def get_simd_config(self) -> Dict[str, Any]:
        """Get SIMD configuration based on CPU"""
        return {
            'instruction_set': self.config.simd_instruction_set,
            'enable_avx2': self.config.enable_avx2,
            'enable_avx512': self.config.enable_avx512,
            'vector_width': self.config.vector_width
        }
    
    def get_power_thermal_config(self) -> Dict[str, float]:
        """Get power and thermal management configuration based on CPU"""
        return {
            'power_constraint': self.config.power_constraint,
            'thermal_constraint': self.config.thermal_constraint,
            'performance_target': self.config.performance_target
        }


def get_optimization_profile(cpu_model: CPUModel) -> AdaptiveOptimizationManager:
    """
    Get the appropriate optimization profile for the specified CPU model
    
    Args:
        cpu_model: The CPU model to get optimization profile for
        
    Returns:
        AdaptiveOptimizationManager with appropriate settings
    """
    if cpu_model == CPUModel.INTEL_I5_10210U:
        config = CPUProfileFactory.create_i5_10210u_profile()
    elif cpu_model == CPUModel.INTEL_I7_8700:
        config = CPUProfileFactory.create_i7_8700_profile()
    else:
        # For unknown CPUs, we'll need to determine parameters at runtime
        import psutil
        cores = psutil.cpu_count(logical=False)
        threads = psutil.cpu_count(logical=True)
        # Estimate L3 cache size (rough approximation)
        l3_cache_size = cores * 2 * 1024 * 1024  # 2MB per core estimate
        config = CPUProfileFactory.create_generic_profile(cores, threads, l3_cache_size)
    
    return AdaptiveOptimizationManager(config)


def get_cpu_specific_config(cpu_model: CPUModel, 
                           cores: int = None, 
                           threads: int = None, 
                           l3_cache_size: int = None) -> CPUSpecificConfig:
    """
    Get CPU-specific configuration based on model and optional parameters
    
    Args:
        cpu_model: The CPU model
        cores: Number of cores (for unknown CPUs)
        threads: Number of threads (for unknown CPUs)
        l3_cache_size: L3 cache size in bytes (for unknown CPUs)
        
    Returns:
        CPUSpecificConfig with appropriate settings
    """
    if cpu_model == CPUModel.INTEL_I5_10210U:
        return CPUProfileFactory.create_i5_10210u_profile()
    elif cpu_model == CPUModel.INTEL_I7_8700:
        return CPUProfileFactory.create_i7_8700_profile()
    else:
        # For unknown CPUs, use provided parameters or defaults
        cores = cores or psutil.cpu_count(logical=False)
        threads = threads or psutil.cpu_count(logical=True)
        l3_cache_size = l3_cache_size or (cores * 2 * 1024 * 1024)  # Estimate
        return CPUProfileFactory.create_generic_profile(cores, threads, l3_cache_size)


if __name__ == "__main__":
    print("CPU-Specific Optimization Profiles")
    print("=" * 40)
    
    # Test i5-10210U profile
    i5_profile = get_optimization_profile(CPUModel.INTEL_I5_10210U)
    print(f"Intel i5-10210U Profile:")
    print(f"  Threads: {i5_profile.get_thread_count()}")
    print(f"  Memory Pool: {i5_profile.get_memory_pool_size() / (1024*1024*1024):.1f} GB")
    print(f"  Batch Multiplier: {i5_profile.get_batch_size_multiplier()}")
    print(f"  SIMD Config: {i5_profile.get_simd_config()}")
    print()
    
    # Test i7-8700 profile
    i7_profile = get_optimization_profile(CPUModel.INTEL_I7_8700)
    print(f"Intel i7-8700 Profile:")
    print(f"  Threads: {i7_profile.get_thread_count()}")
    print(f"  Memory Pool: {i7_profile.get_memory_pool_size() / (1024*1024*1024):.1f} GB")
    print(f"  Batch Multiplier: {i7_profile.get_batch_size_multiplier()}")
    print(f"  SIMD Config: {i7_profile.get_simd_config()}")
    print()
    
    # Test unknown CPU profile
    unknown_profile = get_optimization_profile(CPUModel.UNKNOWN)
    print(f"Generic Profile:")
    print(f"  Threads: {unknown_profile.get_thread_count()}")
    print(f"  Memory Pool: {unknown_profile.get_memory_pool_size() / (1024*1024*1024):.1f} GB")
    print(f"  Batch Multiplier: {unknown_profile.get_batch_size_multiplier()}")
    print(f"  SIMD Config: {unknown_profile.get_simd_config()}")
    
    print("\nCPU-specific optimization profiles loaded successfully!")