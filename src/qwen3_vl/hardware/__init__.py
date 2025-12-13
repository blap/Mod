"""
Main Hardware Adaptation Module for Qwen3-VL
Provides a unified interface to the entire hardware adaptation system
"""
from typing import Dict, Any, Optional
import logging

# Import all components
from src.qwen3_vl.hardware.cpu_detector import (
    CPUDetector, CPUModel, get_hardware_optimizer, get_detected_cpu_info
)
from src.qwen3_vl.hardware.cpu_profiles import (
    CPUProfileFactory, get_optimization_profile, AdaptiveOptimizationManager
)
from src.qwen3_vl.hardware.adaptive_threading import (
    AdaptiveThreadingModel, ThreadingOptimizer, ThreadingModelFactory
)
from src.qwen3_vl.hardware.adaptive_memory import (
    AdaptiveMemoryManager, MemoryOptimizer, MemoryManagerFactory
)
from src.qwen3_vl.hardware.simd_optimizer import (
    SIMDOperationManager, SIMDOptimizer, SIMDManagerFactory
)
from src.qwen3_vl.hardware.unified_config import (
    ConfigManager, UnifiedHardwareOptimizer
)
from src.qwen3_vl.hardware.performance_optimizer import (
    PerformancePathOptimizer, ComprehensivePerformanceOptimizer
)
from src.qwen3_vl.hardware.integration import (
    HardwareAdaptiveModel, apply_hardware_optimizations_to_model,
    get_hardware_adaptive_config, hardware_optimized_inference,
    HardwareAdaptiveTrainer, initialize_hardware_adaptation,
    get_hardware_summary, integrate_with_qwen3vl_pipeline
)

# Set up logging
logger = logging.getLogger(__name__)


class Qwen3VLHardwareAdapter:
    """
    Main adapter class that provides a unified interface to all hardware adaptation features
    """
    
    def __init__(self):
        # Initialize all components
        self.cpu_detector = CPUDetector()
        self.cpu_features = self.cpu_detector.get_cpu_features()
        
        self.unified_optimizer = UnifiedHardwareOptimizer()
        self.threading_optimizer = ThreadingOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        self.simd_optimizer = SIMDOptimizer()
        self.performance_optimizer = ComprehensivePerformanceOptimizer()
        
        logger.info(f"Qwen3-VL Hardware Adapter initialized for {self.cpu_features.model.value}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information
        
        Returns:
            Dictionary containing system information
        """
        return {
            'cpu_info': get_detected_cpu_info(),
            'config_summary': self.unified_optimizer.config_manager.get_config_dict(),
            'optimization_summary': self.performance_optimizer.get_cpu_specific_optimizations()
        }
    
    def optimize_model(self, model) -> Any:
        """
        Apply all hardware optimizations to a model
        
        Args:
            model: The model to optimize
            
        Returns:
            Optimized model
        """
        return apply_hardware_optimizations_to_model(model)
    
    def run_optimized_inference(self, model, inputs, **kwargs) -> Any:
        """
        Run hardware-optimized inference
        
        Args:
            model: The model to run
            inputs: Input data
            **kwargs: Additional parameters
            
        Returns:
            Model output
        """
        return hardware_optimized_inference(model, inputs, **kwargs)
    
    def get_optimal_config(self) -> Dict[str, Any]:
        """
        Get the optimal configuration for the current hardware
        
        Returns:
            Dictionary containing optimal configuration
        """
        return get_hardware_adaptive_config()
    
    def create_adaptive_trainer(self, model, optimizer) -> HardwareAdaptiveTrainer:
        """
        Create a hardware-adaptive trainer
        
        Args:
            model: The model to train
            optimizer: The optimizer to use
            
        Returns:
            HardwareAdaptiveTrainer instance
        """
        return HardwareAdaptiveTrainer(model, optimizer)
    
    def get_recommendations(self) -> Dict[str, str]:
        """
        Get performance recommendations for the current hardware
        
        Returns:
            Dictionary containing recommendations
        """
        return self.performance_optimizer.get_cpu_specific_optimizations()['recommendations']
    
    def run_benchmarks(self) -> Dict[str, Any]:
        """
        Run comprehensive hardware benchmarks
        
        Returns:
            Dictionary containing benchmark results
        """
        return self.performance_optimizer.run_performance_benchmarks()
    
    def get_hardware_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive hardware report
        
        Returns:
            Dictionary containing hardware report
        """
        return self.performance_optimizer.get_optimization_report()
    
    def shutdown(self):
        """Shutdown the hardware adapter and clean up resources"""
        self.threading_optimizer.shutdown()
        logger.info("Qwen3-VL Hardware Adapter shut down")


# Global instance for easy access
_hardware_adapter: Optional[Qwen3VLHardwareAdapter] = None


def get_hardware_adapter() -> Qwen3VLHardwareAdapter:
    """
    Get the global hardware adapter instance
    
    Returns:
        Qwen3VLHardwareAdapter instance
    """
    global _hardware_adapter
    if _hardware_adapter is None:
        _hardware_adapter = Qwen3VLHardwareAdapter()
    return _hardware_adapter


def get_optimal_model_config() -> Dict[str, Any]:
    """
    Get the optimal model configuration for the current hardware
    
    Returns:
        Dictionary containing optimal model configuration
    """
    adapter = get_hardware_adapter()
    return adapter.get_optimal_config()


def optimize_model_for_hardware(model) -> Any:
    """
    Apply hardware optimizations to a model using the global adapter
    
    Args:
        model: The model to optimize
        
    Returns:
        Optimized model
    """
    adapter = get_hardware_adapter()
    return adapter.optimize_model(model)


def run_hardware_optimized_inference(model, inputs, **kwargs) -> Any:
    """
    Run hardware-optimized inference using the global adapter
    
    Args:
        model: The model to run
        inputs: Input data
        **kwargs: Additional parameters
        
    Returns:
        Model output
    """
    adapter = get_hardware_adapter()
    return adapter.run_optimized_inference(model, inputs, **kwargs)


def get_hardware_info() -> Dict[str, Any]:
    """
    Get comprehensive hardware information using the global adapter
    
    Returns:
        Dictionary containing hardware information
    """
    adapter = get_hardware_adapter()
    return adapter.get_system_info()


def get_hardware_recommendations() -> Dict[str, str]:
    """
    Get hardware-specific recommendations using the global adapter
    
    Returns:
        Dictionary containing recommendations
    """
    adapter = get_hardware_adapter()
    return adapter.get_recommendations()


def shutdown_hardware_adapter():
    """Shutdown the global hardware adapter"""
    global _hardware_adapter
    if _hardware_adapter:
        _hardware_adapter.shutdown()
        _hardware_adapter = None


# Initialize the system when module is imported
def _initialize():
    """Initialize the hardware adaptation system"""
    try:
        adapter = get_hardware_adapter()
        logger.info("Qwen3-VL Hardware Adaptation System initialized successfully")
        return adapter
    except Exception as e:
        logger.error(f"Failed to initialize hardware adaptation system: {e}")
        raise


# Initialize on import
_adapter = _initialize()


if __name__ == "__main__":
    print("Qwen3-VL Hardware Adaptation System")
    print("=" * 40)
    
    # Get the adapter
    adapter = get_hardware_adapter()
    
    # Get system info
    system_info = adapter.get_system_info()
    cpu_info = system_info['cpu_info']
    print(f"CPU Model: {cpu_info['model']}")
    print(f"Cores: {cpu_info['cores']}, Threads: {cpu_info['threads']}")
    print(f"L3 Cache: {cpu_info['l3_cache_mb']:.1f} MB")
    print(f"AVX2 Support: {cpu_info['avx2_support']}")
    print()
    
    # Get recommendations
    recommendations = adapter.get_recommendations()
    print("Performance Recommendations:")
    for key, value in recommendations.items():
        print(f"  {key}: {value}")
    print()
    
    # Run benchmarks
    benchmarks = adapter.run_benchmarks()
    print("Benchmark Results:")
    for key, value in benchmarks.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Get hardware report
    report = adapter.get_hardware_report()
    print(f"Optimization Strategy: {report['optimization_strategy']}")
    
    # Test model optimization
    import torch
    import torch.nn as nn
    
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
        
        def forward(self, x):
            return self.linear(x)
    
    test_model = TestModel()
    print(f"\nTesting model optimization...")
    
    optimized_model = adapter.optimize_model(test_model)
    print("Model optimization completed successfully")
    
    # Test inference
    test_input = torch.randn(2, 10)
    output = adapter.run_optimized_inference(optimized_model, test_input)
    print(f"Inference completed, output shape: {output.shape}")
    
    # Shutdown
    adapter.shutdown()
    
    print("\nQwen3-VL Hardware Adaptation System completed!")