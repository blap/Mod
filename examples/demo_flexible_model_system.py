"""
Demonstration script for the flexible model system that supports Qwen3-VL project 
and Qwen/Qwen3-4B-Instruct-2507 model and other potential model additions.
"""

import torch
import torch.nn as nn
from src.models.flexible_model_system import get_flexible_model_system
from src.models.model_registry import ModelSpec
from src.models.config_manager import get_config_manager
from src.models.adaptive_memory_manager import get_memory_manager
from src.models.model_loader import get_model_loader
from src.models.hardware_optimizer import get_hardware_optimizer
from src.models.plugin_system import get_plugin_manager
from src.models.optimization_strategies import get_optimization_manager
from src.models.performance_optimizer import get_performance_optimizer
from src.models.config_validator import get_config_validator
from src.models.model_adapter import create_unified_interface


def demonstrate_flexible_model_system():
    """Demonstrate the flexible model system capabilities."""
    print("=== Flexible Model System Demonstration ===\n")

    # Get the main system
    system = get_flexible_model_system()
    print("[OK] Initialized flexible model system")

    # Show available models
    available_models = system.list_available_models()
    print(f"[OK] Available models: {available_models}")

    # Show system information
    system_info = system.get_system_info()
    print(f"[OK] System info: {system_info['hardware']['gpu_count']} GPU(s), "
          f"{system_info['hardware']['memory_gb']:.2f}GB RAM")

    # Demonstrate model registration
    print("\n--- Demonstrating Model Registration ---")

    # Register a sample model (using a simple linear model for demonstration)
    try:
        from src.models.base_model import Qwen3VLModel
        from src.qwen3_vl.config.base_config import Qwen3VLConfig

        success = system.register_model(
            name="Demo-Model-1B",
            model_class=Qwen3VLModel,
            config_class=Qwen3VLConfig,
            supported_dtypes=["float16", "float32"],
            required_memory_gb=4.0,
            max_sequence_length=2048,
            description="Demo model for testing flexible system",
            model_type="language"
        )

        if success:
            print("[OK] Successfully registered Demo-Model-1B")
        else:
            print("[ERROR] Failed to register Demo-Model-1B")
    except ImportError:
        print("[WARN] Could not import Qwen3VLModel, using placeholder")
        # For demo purposes, use a simple model
        success = system.register_model(
            name="Demo-Model-1B",
            model_class=nn.Linear,
            config_class=dict,
            supported_dtypes=["float16", "float32"],
            required_memory_gb=4.0,
            max_sequence_length=2048,
            description="Demo model for testing flexible system",
            model_type="language"
        )
        if success:
            print("[OK] Successfully registered Demo-Model-1B (placeholder)")

    # Show updated available models
    updated_models = system.list_available_models()
    print(f"[OK] Updated available models: {updated_models}")

    # Demonstrate configuration management
    print("\n--- Demonstrating Configuration Management ---")
    config_manager = get_config_manager()

    # Get and adapt configuration for hardware
    config = config_manager.load_config("Qwen3-VL")
    system_memory = system.memory_manager.get_system_memory_info()
    adapted_config = config_manager.adapt_config_for_hardware(
        config, system_memory["available_gb"]
    )
    print(f"[OK] Adapted configuration for hardware: {adapted_config['performance_profile']} profile")

    # Demonstrate memory management
    print("\n--- Demonstrating Memory Management ---")
    memory_manager = get_memory_manager()

    # Get memory information
    gpu_memory = memory_manager.get_gpu_memory_info()
    if gpu_memory:
        print(f"[OK] GPU Memory: {gpu_memory['total_gb']:.2f}GB total, "
              f"{gpu_memory['available_gb']:.2f}GB available")
    else:
        print("[OK] No GPU detected, using CPU memory management")

    # Demonstrate hardware optimization
    print("\n--- Demonstrating Hardware Optimization ---")
    hardware_optimizer = get_hardware_optimizer()

    # Get hardware spec
    hw_spec = hardware_optimizer.get_hardware_spec()
    print(f"[OK] Hardware spec: {hw_spec.cpu_count} CPUs, "
          f"{hw_spec.memory_gb:.2f}GB memory, "
          f"CUDA available: {hw_spec.cuda_available}")

    # Demonstrate optimization strategies
    print("\n--- Demonstrating Optimization Strategies ---")
    opt_manager = get_optimization_manager()

    # Get optimization recommendations
    recommendations = system.get_optimization_recommendations(
        "Qwen3-VL",
        hardware_memory_gb=system_info['hardware']['memory_gb']
    )
    print(f"[OK] Optimization recommendations for Qwen3-VL: {len(recommendations['recommendations'])} strategies")

    # Demonstrate performance optimization
    print("\n--- Demonstrating Performance Optimization ---")
    perf_optimizer = get_performance_optimizer()

    # Get performance config
    perf_config = perf_optimizer.get_performance_config("Qwen3-VL")
    if perf_config:
        print(f"[OK] Performance config: batch_size={perf_config.batch_size}, "
              f"profile={perf_config.performance_level.value}")
    else:
        print("[OK] No specific performance config found for Qwen3-VL")

    # Demonstrate plugin system
    print("\n--- Demonstrating Plugin System ---")
    plugin_manager = get_plugin_manager()

    # Show loaded plugins
    plugins = plugin_manager.get_all_plugins()
    print(f"[OK] Loaded {len(plugins)} plugins")

    # Apply plugins to registry
    plugin_manager.apply_plugins_to_registry()
    print("[OK] Applied plugins to model registry")

    # Demonstrate model loading (with a simple example)
    print("\n--- Demonstrating Model Loading ---")
    model_loader = get_model_loader()

    # Calculate model size category
    hw_optimizer = get_hardware_optimizer()
    model_size_category = hw_optimizer.get_model_size_category(4000000000)  # 4B params
    print(f"[OK] Model size category for 4B params: {model_size_category}")

    # Show optimization recommendations
    print("\n--- Optimization Recommendations ---")
    recommendations = system.get_optimization_recommendations(
        "Qwen3-VL",
        hardware_memory_gb=8.0,
        target_performance="balanced"
    )
    print(f"[OK] Recommendations for Qwen3-VL on 8GB system:")
    for rec in recommendations["recommendations"]:
        print(f"  - {rec}")

    print("\n=== Flexible Model System Demo Complete ===")
    print("The system provides:")
    print("- [OK] Model registry supporting multiple architectures")
    print("- [OK] Configurable loading for different models")
    print("- [OK] Adaptive memory management")
    print("- [OK] Hardware optimization profiles")
    print("- [OK] Plugin system for easy model additions")
    print("- [OK] Model-specific optimization strategies")
    print("- [OK] Performance scaling based on model size")
    print("- [OK] Configuration validation")
    print("- [OK] Unified model adapter interface")


def demonstrate_model_adapter():
    """Demonstrate the model adapter functionality."""
    print("\n=== Model Adapter Demonstration ===")

    # Create a simple model for demonstration
    simple_model = nn.Linear(10, 5)

    # Create unified interface
    unified_interface = create_unified_interface(simple_model, "default", {})

    # Test the interface
    test_input = torch.randn(2, 10)
    output = unified_interface.forward(test_input)

    print(f"[OK] Created unified interface")
    print(f"[OK] Forward pass successful: input shape {test_input.shape}, output shape {output.shape}")

    # Test device movement
    if torch.cuda.is_available():
        unified_interface.to("cuda")
        print("[OK] Moved model to CUDA")
        unified_interface.to("cpu")  # Move back to CPU for consistency
    else:
        print("[OK] CUDA not available, staying on CPU")

    print("[OK] Model adapter demonstration complete")


if __name__ == "__main__":
    print("Starting Flexible Model System Demonstration...")
    
    try:
        demonstrate_flexible_model_system()
        demonstrate_model_adapter()
        
        print("\n[SUCCESS] All demonstrations completed successfully!")
        print("The flexible model system is ready to support Qwen3-VL, Qwen3-4B-Instruct-2507, and other models.")
        
    except Exception as e:
        print(f"\n[ERROR] Error during demonstration: {e}")
        import traceback
        traceback.print_exc()