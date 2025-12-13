"""
Unified Configuration System for Qwen3-VL Hardware Adaptation
Implements a configuration system that automatically adjusts parameters based on detected CPU
"""
import json
import os
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging
import yaml

from src.qwen3_vl.hardware.cpu_detector import CPUDetector, CPUModel, get_detected_cpu_info
from src.qwen3_vl.hardware.cpu_profiles import get_optimization_profile, CPUSpecificConfig
from src.qwen3_vl.hardware.adaptive_threading import get_threading_optimizer, ThreadingOptimizer
from src.qwen3_vl.hardware.adaptive_memory import get_memory_optimizer, MemoryOptimizer
from src.qwen3_vl.hardware.simd_optimizer import get_simd_optimizer, SIMDOptimizer


logger = logging.getLogger(__name__)


@dataclass
class HardwareAdaptationConfig:
    """Unified configuration for hardware adaptation"""
    # CPU-specific parameters
    cpu_model: str
    num_cores: int
    num_threads: int
    max_frequency: float
    l1_cache_size: int
    l2_cache_size: int
    l3_cache_size: int
    avx2_support: bool
    avx512_support: bool
    
    # Threading parameters
    num_workers: int
    max_concurrent_threads: int
    thread_affinity_enabled: bool
    use_hyperthreading: bool
    
    # Memory parameters
    memory_pool_size: int
    memory_threshold: float
    enable_memory_pooling: bool
    enable_cache_blocking: bool
    enable_memory_compression: bool
    enable_memory_swapping: bool
    
    # SIMD parameters
    simd_instruction_set: str
    enable_avx2: bool
    enable_avx512: bool
    vector_width: int
    
    # Performance parameters
    batch_size_multiplier: float
    pipeline_depth: int
    pipeline_buffer_size: int
    performance_target: float
    power_constraint: float
    thermal_constraint: float
    
    # Model-specific parameters that may be adjusted based on hardware
    model_specific_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.model_specific_params is None:
            self.model_specific_params = {}


class ConfigManager:
    """Manages hardware-adaptive configurations"""
    
    def __init__(self):
        self.cpu_detector = CPUDetector()
        self.cpu_features = self.cpu_detector.get_cpu_features()
        self.optimization_profile = get_optimization_profile(self.cpu_features.model)
        
        # Initialize optimizers to get their recommendations
        self.threading_optimizer = get_threading_optimizer()
        self.memory_optimizer = get_memory_optimizer()
        self.simd_optimizer = get_simd_optimizer()
        
        # Generate the unified configuration
        self.config = self._generate_unified_config()
        
        logger.info(f"Configuration manager initialized for {self.cpu_features.model.value}")
    
    def _generate_unified_config(self) -> HardwareAdaptationConfig:
        """Generate unified configuration based on detected hardware and optimizations"""
        # Get threading recommendations
        threading_stats = self.threading_optimizer.get_performance_stats()
        
        # Get memory recommendations
        memory_config = self.memory_optimizer.get_memory_config()
        
        # Get SIMD recommendations
        simd_config = self.simd_optimizer.get_simd_config()
        
        # Create unified config
        config = HardwareAdaptationConfig(
            # CPU-specific parameters
            cpu_model=self.cpu_features.model.value,
            num_cores=self.cpu_features.cores,
            num_threads=self.cpu_features.threads,
            max_frequency=self.cpu_features.max_frequency,
            l1_cache_size=self.cpu_features.l1_cache_size,
            l2_cache_size=self.cpu_features.l2_cache_size,
            l3_cache_size=self.cpu_features.l3_cache_size,
            avx2_support=self.cpu_features.avx2_support,
            avx512_support=self.cpu_features.avx512_support,
            
            # Threading parameters
            num_workers=threading_stats['recommended_workers'],
            max_concurrent_threads=threading_stats['cpu_thread_pool_size'],
            thread_affinity_enabled=self.optimization_profile.config.thread_affinity_enabled,
            use_hyperthreading=self.optimization_profile.config.use_hyperthreading,
            
            # Memory parameters
            memory_pool_size=memory_config.memory_pool_size,
            memory_threshold=memory_config.memory_threshold,
            enable_memory_pooling=memory_config.enable_memory_pooling,
            enable_cache_blocking=memory_config.enable_cache_blocking,
            enable_memory_compression=memory_config.enable_memory_compression,
            enable_memory_swapping=memory_config.enable_memory_swapping,
            
            # SIMD parameters
            simd_instruction_set=simd_config.instruction_set,
            enable_avx2=simd_config.enable_avx2,
            enable_avx512=simd_config.enable_avx512,
            vector_width=simd_config.vector_width,
            
            # Performance parameters
            batch_size_multiplier=self.optimization_profile.get_batch_size_multiplier(),
            pipeline_depth=self.optimization_profile.config.pipeline_depth,
            pipeline_buffer_size=self.optimization_profile.config.pipeline_buffer_size,
            performance_target=self.optimization_profile.config.performance_target,
            power_constraint=self.optimization_profile.config.power_constraint,
            thermal_constraint=self.optimization_profile.config.thermal_constraint
        )
        
        return config
    
    def get_config(self) -> HardwareAdaptationConfig:
        """
        Get the current hardware-adaptive configuration
        
        Returns:
            HardwareAdaptationConfig object
        """
        return self.config
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get the configuration as a dictionary
        
        Returns:
            Dictionary representation of the configuration
        """
        return asdict(self.config)
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model-specific configuration parameters based on hardware
        
        Returns:
            Dictionary containing model-specific parameters
        """
        config_dict = self.get_config_dict()
        
        # Extract model-specific parameters
        model_config = {
            # Threading configuration for model execution
            'num_workers': config_dict['num_workers'],
            'max_concurrent_threads': config_dict['max_concurrent_threads'],
            
            # Memory configuration for model
            'memory_pool_size': config_dict['memory_pool_size'],
            'enable_memory_pooling': config_dict['enable_memory_pooling'],
            
            # Performance optimization parameters
            'batch_size_multiplier': config_dict['batch_size_multiplier'],
            'pipeline_depth': config_dict['pipeline_depth'],
            'pipeline_buffer_size': config_dict['pipeline_buffer_size'],
            
            # Hardware-specific optimizations
            'simd_instruction_set': config_dict['simd_instruction_set'],
            'enable_avx2': config_dict['enable_avx2'],
            'enable_avx512': config_dict['enable_avx512'],
            
            # Power and thermal constraints
            'power_constraint': config_dict['power_constraint'],
            'thermal_constraint': config_dict['thermal_constraint'],
        }
        
        return model_config
    
    def update_model_specific_params(self, params: Dict[str, Any]):
        """
        Update model-specific parameters in the configuration
        
        Args:
            params: Dictionary of model-specific parameters to update
        """
        self.config.model_specific_params.update(params)
        logger.info(f"Updated model-specific parameters: {list(params.keys())}")
    
    def save_config(self, filepath: str):
        """
        Save the configuration to a file
        
        Args:
            filepath: Path to save the configuration file
        """
        config_dict = self.get_config_dict()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")
    
    def load_config(self, filepath: str) -> HardwareAdaptationConfig:
        """
        Load configuration from a file
        
        Args:
            filepath: Path to load the configuration file from
            
        Returns:
            Loaded HardwareAdaptationConfig object
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Create a new config object from the loaded dictionary
        config = HardwareAdaptationConfig(
            cpu_model=config_dict['cpu_model'],
            num_cores=config_dict['num_cores'],
            num_threads=config_dict['num_threads'],
            max_frequency=config_dict['max_frequency'],
            l1_cache_size=config_dict['l1_cache_size'],
            l2_cache_size=config_dict['l2_cache_size'],
            l3_cache_size=config_dict['l3_cache_size'],
            avx2_support=config_dict['avx2_support'],
            avx512_support=config_dict['avx512_support'],
            num_workers=config_dict['num_workers'],
            max_concurrent_threads=config_dict['max_concurrent_threads'],
            thread_affinity_enabled=config_dict['thread_affinity_enabled'],
            use_hyperthreading=config_dict['use_hyperthreading'],
            memory_pool_size=config_dict['memory_pool_size'],
            memory_threshold=config_dict['memory_threshold'],
            enable_memory_pooling=config_dict['enable_memory_pooling'],
            enable_cache_blocking=config_dict['enable_cache_blocking'],
            enable_memory_compression=config_dict['enable_memory_compression'],
            enable_memory_swapping=config_dict['enable_memory_swapping'],
            simd_instruction_set=config_dict['simd_instruction_set'],
            enable_avx2=config_dict['enable_avx2'],
            enable_avx512=config_dict['enable_avx512'],
            vector_width=config_dict['vector_width'],
            batch_size_multiplier=config_dict['batch_size_multiplier'],
            pipeline_depth=config_dict['pipeline_depth'],
            pipeline_buffer_size=config_dict['pipeline_buffer_size'],
            performance_target=config_dict['performance_target'],
            power_constraint=config_dict['power_constraint'],
            thermal_constraint=config_dict['thermal_constraint'],
            model_specific_params=config_dict.get('model_specific_params', {})
        )
        
        logger.info(f"Configuration loaded from {filepath}")
        return config
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all optimizations applied based on hardware
        
        Returns:
            Dictionary containing optimization summary
        """
        return {
            'detected_cpu': self.cpu_features.model.value,
            'cpu_cores': self.cpu_features.cores,
            'cpu_threads': self.cpu_features.threads,
            'l3_cache_mb': self.cpu_features.l3_cache_size / (1024 * 1024),
            'simd_support': {
                'avx2': self.cpu_features.avx2_support,
                'avx512': self.cpu_features.avx512_support
            },
            'threading_optimizations': {
                'recommended_workers': self.config.num_workers,
                'max_threads': self.config.max_concurrent_threads,
                'thread_affinity': self.config.thread_affinity_enabled
            },
            'memory_optimizations': {
                'pool_size_gb': self.config.memory_pool_size / (1024 * 1024 * 1024),
                'cache_blocking': self.config.enable_cache_blocking,
                'compression': self.config.enable_memory_compression
            },
            'simd_optimizations': {
                'instruction_set': self.config.simd_instruction_set,
                'vector_width': self.config.vector_width
            },
            'performance_optimizations': {
                'batch_size_multiplier': self.config.batch_size_multiplier,
                'pipeline_depth': self.config.pipeline_depth,
                'target_performance': self.config.performance_target
            }
        }


class UnifiedHardwareOptimizer:
    """Unified optimizer that coordinates all hardware-specific optimizations"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.cpu_detector = CPUDetector()
        self.cpu_features = self.cpu_detector.get_cpu_features()
        
        logger.info(f"Unified hardware optimizer initialized for {self.cpu_features.model.value}")
    
    def get_optimal_config(self) -> HardwareAdaptationConfig:
        """
        Get the optimal configuration for the detected hardware
        
        Returns:
            HardwareAdaptationConfig with optimal settings
        """
        return self.config_manager.get_config()
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration optimized for the detected hardware
        
        Returns:
            Dictionary containing model configuration
        """
        return self.config_manager.get_model_config()
    
    def apply_optimizations_to_model(self, model) -> Any:
        """
        Apply all hardware-specific optimizations to a model
        
        Args:
            model: The model to optimize
            
        Returns:
            Optimized model
        """
        # Apply threading optimizations
        threading_optimizer = get_threading_optimizer()
        threading_optimizer.allocate_model_tensors(model)
        
        # Apply memory optimizations
        memory_optimizer = get_memory_optimizer()
        memory_optimizer.allocate_model_tensors(model)
        
        # Apply SIMD optimizations
        simd_optimizer = get_simd_optimizer()
        model = simd_optimizer.apply_model_optimizations(model)
        
        logger.info(f"Applied all hardware-specific optimizations to model for {self.cpu_features.model.value}")
        return model
    
    def get_hardware_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive report about the detected hardware and optimizations
        
        Returns:
            Dictionary containing hardware report
        """
        cpu_info = get_detected_cpu_info()
        optimization_summary = self.config_manager.get_optimization_summary()
        
        return {
            'cpu_info': cpu_info,
            'optimization_summary': optimization_summary,
            'configuration': self.config_manager.get_config_dict()
        }
    
    def save_hardware_config(self, filepath: str):
        """
        Save the hardware-specific configuration to a file
        
        Args:
            filepath: Path to save the configuration
        """
        self.config_manager.save_config(filepath)
    
    def load_hardware_config(self, filepath: str) -> HardwareAdaptationConfig:
        """
        Load hardware-specific configuration from a file
        
        Args:
            filepath: Path to load the configuration from
            
        Returns:
            Loaded HardwareAdaptationConfig
        """
        return self.config_manager.load_config(filepath)


def get_unified_optimizer() -> UnifiedHardwareOptimizer:
    """
    Get a unified hardware optimizer configured for the detected CPU
    
    Returns:
        UnifiedHardwareOptimizer instance
    """
    return UnifiedHardwareOptimizer()


def get_adaptive_config() -> HardwareAdaptationConfig:
    """
    Get the adaptive configuration for the detected hardware
    
    Returns:
        HardwareAdaptationConfig instance
    """
    optimizer = get_unified_optimizer()
    return optimizer.get_optimal_config()


def get_model_hardware_config() -> Dict[str, Any]:
    """
    Get model-specific configuration optimized for the detected hardware
    
    Returns:
        Dictionary containing model hardware configuration
    """
    optimizer = get_unified_optimizer()
    return optimizer.get_model_config()


class ConfigAdapterFactory:
    """Factory for creating configuration adapters"""
    
    @staticmethod
    def create_for_cpu_model(cpu_model: CPUModel) -> ConfigManager:
        """
        Create a configuration manager for a specific CPU model
        This is a simplified implementation - in practice, you would need to 
        temporarily override the CPU detection to create config for specific models
        
        Args:
            cpu_model: The CPU model to create configuration for
            
        Returns:
            ConfigManager instance
        """
        # For this implementation, we'll just return the default config manager
        # since the system is designed to work with the currently detected CPU
        return ConfigManager()


if __name__ == "__main__":
    print("Unified Configuration System for Qwen3-VL Hardware Adaptation")
    print("=" * 60)
    
    # Test the configuration system
    print("Initializing unified hardware optimizer...")
    optimizer = get_unified_optimizer()
    
    # Get the optimal config
    config = optimizer.get_optimal_config()
    print(f"Generated optimal config for {config.cpu_model}")
    print(f"  Cores: {config.num_cores}, Threads: {config.num_threads}")
    print(f"  L3 Cache: {config.l3_cache_size / (1024*1024):.0f} MB")
    print(f"  Memory Pool: {config.memory_pool_size / (1024*1024*1024):.1f} GB")
    print(f"  SIMD: {config.simd_instruction_set} (width: {config.vector_width})")
    print(f"  Batch Multiplier: {config.batch_size_multiplier}")
    print()
    
    # Get model config
    model_config = optimizer.get_model_config()
    print("Model Configuration:")
    for key, value in list(model_config.items())[:10]:  # Show first 10 items
        print(f"  {key}: {value}")
    print("  ...")
    print()
    
    # Get hardware report
    report = optimizer.get_hardware_report()
    print("Hardware Report:")
    cpu_info = report['cpu_info']
    print(f"  Detected CPU: {cpu_info['model']}")
    print(f"  Cores: {cpu_info['cores']}, Threads: {cpu_info['threads']}")
    print(f"  L3 Cache: {cpu_info['l3_cache_mb']:.1f} MB")
    print(f"  AVX2 Support: {cpu_info['avx2_support']}")
    print()
    
    # Save config to file
    optimizer.save_hardware_config("test_config.json")
    print("Configuration saved to test_config.json")
    
    # Load config from file
    loaded_config = optimizer.load_hardware_config("test_config.json")
    print(f"Configuration loaded: {loaded_config.cpu_model}")
    
    # Clean up test file
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")
    
    print("\nUnified configuration system implementation completed!")