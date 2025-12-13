"""
Updated usage example for Qwen3-VL model demonstrating the consolidated module structure
"""
from src.qwen3_vl.config import get_default_config, create_unified_config_manager
from src.qwen3_vl.components.models import Qwen3VLModel
from src.qwen3_vl.inference import Qwen3VLInference
from src.qwen3_vl.memory_management import GeneralMemoryManager, KVCacheOptimizer


def main():
    print("Qwen3-VL Updated Usage Example with Consolidated Modules")
    print("="*60)

    # Step 1: Get configuration using the unified configuration system
    config_manager = create_unified_config_manager()
    config = get_default_config()
    
    # Get optimized configuration
    optimized_config = config_manager.get_config("balanced")
    print(f"Using optimized configuration with {optimized_config.num_hidden_layers} layers")

    # Step 2: Initialize memory management
    memory_manager = GeneralMemoryManager(config=optimized_config)
    kv_cache_optimizer = KVCacheOptimizer(config=optimized_config)
    print("Memory management initialized")

    # Step 3: Create the model
    model = Qwen3VLModel(config=optimized_config)
    print("Model created successfully")

    # Step 4: Apply memory optimizations
    optimized_model = kv_cache_optimizer.apply_optimizations(model)
    print("Memory optimizations applied")

    # Step 5: Create inference instance
    inference = Qwen3VLInference(optimized_model)
    print("Inference engine ready")

    # Example text input
    text = "Describe the content of this image."

    # Generate response (with dummy image since we're using a dummy model)
    response = inference.generate_response(
        text=text,
        max_new_tokens=50,
        temperature=0.7
    )

    print(f"Input: {text}")
    print(f"Response: {response}")

    # Run a multimodal task example
    multimodal_response = inference.run_multimodal_task(
        text="What objects do you see in this image?",
        image_path="dummy_path",  # Using dummy path for demonstration
        task_type="vqa"
    )

    print(f"VQA Response: {multimodal_response}")

    print("\nConsolidated module example completed successfully!")
    print("\nKey features demonstrated:")
    print("- Unified configuration system")
    print("- Memory management with KV cache optimization")
    print("- Inference pipeline with multimodal support")
    print("- Clean import structure from consolidated modules")


if __name__ == "__main__":
    main()