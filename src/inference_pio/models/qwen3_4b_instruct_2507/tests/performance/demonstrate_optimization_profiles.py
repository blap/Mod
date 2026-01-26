"""
Demonstration of the Optimization Profile System for Inference-PIO.

This script demonstrates how to use the optimization profile system to configure
different optimization strategies for the GLM-4-7, Qwen3-4b-instruct-2507, 
Qwen3-coder-30b, and Qwen3-vl-2b models.
"""

from src.inference_pio.common.optimization_profiles import (
    get_profile_manager,
    PerformanceProfile,
    MemoryEfficientProfile,
    BalancedProfile,
    GLM47Profile,
    Qwen34BProfile,
    Qwen3CoderProfile,
    Qwen3VLProfile
)
from src.inference_pio.common.config_loader import get_config_loader
from src.inference_pio.models.glm_4_7.config import GLM47Config
from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen34BInstruct2507Config
from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig


def demonstrate_basic_profiles():
    """Demonstrate basic optimization profiles."""
    print("=== Basic Optimization Profiles ===")
    
    profile_manager = get_profile_manager()
    
    # Create and register a performance profile
    perf_profile = PerformanceProfile(
        name="high_performance",
        description="High performance settings for low-latency inference"
    )
    profile_manager.register_profile("high_performance", perf_profile)
    
    # Create and register a memory efficient profile
    mem_profile = MemoryEfficientProfile(
        name="memory_efficient",
        description="Memory efficient settings for constrained environments"
    )
    profile_manager.register_profile("memory_efficient", mem_profile)
    
    # Create and register a balanced profile
    balanced_profile = BalancedProfile(
        name="balanced",
        description="Balanced performance and memory usage"
    )
    profile_manager.register_profile("balanced", balanced_profile)
    
    print("Registered basic optimization profiles:")
    for profile_name in profile_manager.list_profiles():
        print(f"  - {profile_name}")
    
    print()


def demonstrate_model_specific_profiles():
    """Demonstrate model-specific optimization profiles."""
    print("=== Model-Specific Optimization Profiles ===")
    
    profile_manager = get_profile_manager()
    
    # Create model-specific profiles
    glm_profile = GLM47Profile(
        name="glm_optimized",
        description="GLM-4.7 specific optimizations",
        use_glm_attention_patterns=True,
        glm_attention_pattern_sparsity=0.4,
        use_glm_memory_efficient_kv=True
    )
    profile_manager.register_profile("glm_optimized", glm_profile)
    
    qwen3_4b_profile = Qwen34BProfile(
        name="qwen3_4b_optimized",
        description="Qwen3-4B specific optimizations",
        use_qwen3_attention_optimizations=True,
        qwen3_attention_sparsity_ratio=0.35
    )
    profile_manager.register_profile("qwen3_4b_optimized", qwen3_4b_profile)
    
    qwen3_coder_profile = Qwen3CoderProfile(
        name="qwen3_coder_optimized",
        description="Qwen3-Coder specific optimizations",
        use_qwen3_coder_code_optimizations=True,
        code_generation_temperature=0.1
    )
    profile_manager.register_profile("qwen3_coder_optimized", qwen3_coder_profile)
    
    qwen3_vl_profile = Qwen3VLProfile(
        name="qwen3_vl_optimized",
        description="Qwen3-VL specific optimizations",
        use_qwen3_vl_attention_optimizations=True,
        use_multimodal_attention=True
    )
    profile_manager.register_profile("qwen3_vl_optimized", qwen3_vl_profile)
    
    print("Registered model-specific optimization profiles:")
    for profile_name in ["glm_optimized", "qwen3_4b_optimized", "qwen3_coder_optimized", "qwen3_vl_optimized"]:
        print(f"  - {profile_name}")
    
    print()


def demonstrate_profile_application():
    """Demonstrate applying profiles to model configurations."""
    print("=== Applying Profiles to Model Configurations ===")
    
    profile_manager = get_profile_manager()
    
    # Create model configurations
    glm_config = GLM47Config()
    qwen3_4b_config = Qwen34BInstruct2507Config()
    qwen3_coder_config = Qwen3Coder30BConfig()
    qwen3_vl_config = Qwen3VL2BConfig()
    
    print("Original GLM-4.7 config - gradient_checkpointing:", glm_config.gradient_checkpointing)
    print("Original Qwen3-4B config - gradient_checkpointing:", qwen3_4b_config.gradient_checkpointing)
    print("Original Qwen3-Coder config - gradient_checkpointing:", qwen3_coder_config.gradient_checkpointing)
    print("Original Qwen3-VL config - gradient_checkpointing:", qwen3_vl_config.gradient_checkpointing)
    
    # Apply performance profile to all configs
    profile_manager.apply_profile_to_config("high_performance", glm_config)
    profile_manager.apply_profile_to_config("high_performance", qwen3_4b_config)
    profile_manager.apply_profile_to_config("high_performance", qwen3_coder_config)
    profile_manager.apply_profile_to_config("high_performance", qwen3_vl_config)
    
    print("\nAfter applying performance profile:")
    print("GLM-4.7 config - gradient_checkpointing:", glm_config.gradient_checkpointing)
    print("Qwen3-4B config - gradient_checkpointing:", qwen3_4b_config.gradient_checkpointing)
    print("Qwen3-Coder config - gradient_checkpointing:", qwen3_coder_config.gradient_checkpointing)
    print("Qwen3-VL config - gradient_checkpointing:", qwen3_vl_config.gradient_checkpointing)
    
    # Apply memory efficient profile to Qwen3-Coder
    profile_manager.apply_profile_to_config("memory_efficient", qwen3_coder_config)
    print("\nAfter applying memory efficient profile to Qwen3-Coder:")
    print("Qwen3-Coder config - gradient_checkpointing:", qwen3_coder_config.gradient_checkpointing)
    print("Qwen3-Coder config - max_memory_ratio:", qwen3_coder_config.max_memory_ratio)
    
    print()


def demonstrate_config_loader_integration():
    """Demonstrate integration with the config loader."""
    print("=== Config Loader Integration ===")
    
    config_loader = get_config_loader()
    
    # Create a GLM config from a performance profile
    success = config_loader.create_config_from_profile(
        model_type='glm',
        profile_name='high_performance',
        config_name='glm_perf_config'
    )
    
    if success:
        glm_perf_config = config_loader.config_manager.get_config('glm_perf_config')
        print("Created GLM config from performance profile")
        print(f"  - gradient_checkpointing: {glm_perf_config.gradient_checkpointing}")
        print(f"  - use_quantization: {glm_perf_config.use_quantization}")
        print(f"  - quantization_bits: {glm_perf_config.quantization_bits}")
    
    # Create a Qwen3-Coder config from a model-specific profile
    success = config_loader.create_config_from_profile(
        model_type='qwen3_coder',
        profile_name='qwen3_coder_optimized',
        config_name='qwen3_coder_custom_config'
    )
    
    if success:
        qwen3_coder_custom_config = config_loader.config_manager.get_config('qwen3_coder_custom_config')
        print("\nCreated Qwen3-Coder config from model-specific profile")
        print(f"  - use_qwen3_coder_code_optimizations: {qwen3_coder_custom_config.use_qwen3_coder_code_optimizations}")
        print(f"  - code_generation_temperature: {qwen3_coder_custom_config.code_generation_temperature}")
        print(f"  - use_flash_attention_2: {qwen3_coder_custom_config.use_flash_attention_2}")
    
    print()


def demonstrate_profile_management():
    """Demonstrate profile management capabilities."""
    print("=== Profile Management ===")
    
    profile_manager = get_profile_manager()
    
    # List all profiles
    print("All registered profiles:")
    for profile_name in profile_manager.list_profiles():
        metadata = profile_manager.get_profile_metadata(profile_name)
        print(f"  - {profile_name}: {metadata['description']} (type: {metadata['type']})")
    
    print()
    
    # Show profile history (though initially empty for newly created profiles)
    perf_profile = profile_manager.get_profile("high_performance")
    if perf_profile:
        history = profile_manager.get_profile_history("high_performance")
        print(f"History for 'high_performance' profile: {len(history)} versions")
    
    print()


def main():
    """Main demonstration function."""
    print("Optimization Profile System Demonstration for Inference-PIO\n")
    
    demonstrate_basic_profiles()
    demonstrate_model_specific_profiles()
    demonstrate_profile_application()
    demonstrate_config_loader_integration()
    demonstrate_profile_management()
    
    print("Demonstration completed successfully!")


if __name__ == "__main__":
    main()