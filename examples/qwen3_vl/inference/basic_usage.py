"""
Basic usage example for Qwen3-VL model demonstrating the consolidated module structure
"""
from src.qwen3_vl.config import get_default_config
from src.qwen3_vl.components.models import Qwen3VLModel
from src.qwen3_vl.inference import Qwen3VLInference
from src.qwen3_vl.memory_management import GeneralMemoryManager


def main():
    print("Qwen3-VL Basic Usage Example with Consolidated Modules")
    print("="*55)

    # Get configuration
    config = get_default_config()
    print(f"Configuration loaded: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")

    # Initialize memory management
    memory_manager = GeneralMemoryManager(config=config)
    print("Memory management initialized")

    # Create the model
    model = Qwen3VLModel(config=config)
    print("Model created successfully")

    # Create inference instance
    inference = Qwen3VLInference(model)

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
    print("\nThis example demonstrates:")
    print("- Clean import structure from consolidated modules")
    print("- Unified configuration system")
    print("- Integrated memory management")


if __name__ == "__main__":
    main()