"""
Example usage of the Streaming Computation System with Models

This script demonstrates how to use the streaming computation system with the four models:
- GLM-4-7
- Qwen3-4b-instruct-2507
- Qwen3-coder-30b
- Qwen3-vl-2b
"""

import time
from concurrent.futures import as_completed

import torch

from src.inference_pio.models.glm_4_7_flash.config import GLM47Config
from src.inference_pio.models.glm_4_7_flash.model import GLM47Model
from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen34BInstruct2507Config
from src.inference_pio.models.qwen3_4b_instruct_2507.model import Qwen34BInstruct2507Model
from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig
from src.inference_pio.models.qwen3_coder_30b.model import Qwen3Coder30BModel
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel


def example_glm47_streaming():
    """Example of using streaming with GLM-4-7 model."""
    print("Setting up GLM-4-7 model with streaming...")

    # Create config for GLM-4-7
    config = GLM47Config(
        model_path="THUDM/glm-4-9b",  # Using a placeholder for the example
        torch_dtype="float16",
        device_map="cpu",  # Use CPU for the example
        use_flash_attention_2=False,  # Disable for CPU
        use_sparse_attention=False,
        use_tensor_parallelism=False,
    )

    # Create model instance
    model = GLM47Model(config)

    # Setup streaming computation
    model.setup_streaming_computation(max_concurrent_requests=2, buffer_size=10)

    # Example prompts
    prompts = [
        "Hello, how are you?",
        "What is the weather today?",
        "Tell me a joke.",
        "Explain quantum computing.",
    ]

    print("Submitting requests to GLM-4-7 streaming engine...")

    # Submit requests
    futures = []
    for i, prompt in enumerate(prompts):
        # Tokenize the prompt
        inputs = model.get_tokenizer()(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        future = model.submit_stream_request(f"glm47_req_{i}", inputs)
        futures.append(future)

    # Collect results
    for future in futures:
        result = future.result(timeout=30.0)
        if result.error:
            print(f"Error in GLM-4-7 request {result.request_id}: {result.error}")
        else:
            print(f"GLM-4-7 result for {result.request_id}: {result.result}")

    # Clean up
    model.streaming_engine.stop()


def example_qwen3_4b_streaming():
    """Example of using streaming with Qwen3-4b-instruct-2507 model."""
    print("\nSetting up Qwen3-4b-instruct-2507 model with streaming...")

    # Create config for Qwen3-4b-instruct-2507
    config = Qwen34BInstruct2507Config(
        model_path="Qwen/Qwen3-4B-Instruct-2507",  # Using a placeholder for the example
        torch_dtype="float16",
        device_map="cpu",  # Use CPU for the example
        use_flash_attention_2=False,  # Disable for CPU
        use_sparse_attention=False,
        use_tensor_parallelism=False,
    )

    # Create model instance
    model = Qwen34BInstruct2507Model(config)

    # Setup streaming computation
    model.setup_streaming_computation(max_concurrent_requests=2, buffer_size=10)

    # Example prompts
    prompts = [
        "Hello, how are you?",
        "What is the weather today?",
        "Tell me a joke.",
        "Explain quantum computing.",
    ]

    print("Submitting requests to Qwen3-4b-instruct-2507 streaming engine...")

    # Submit requests
    futures = []
    for i, prompt in enumerate(prompts):
        # Tokenize the prompt
        inputs = model.get_tokenizer()(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        future = model.submit_stream_request(f"qwen3_4b_req_{i}", inputs)
        futures.append(future)

    # Collect results
    for future in futures:
        result = future.result(timeout=30.0)
        if result.error:
            print(f"Error in Qwen3-4b request {result.request_id}: {result.error}")
        else:
            print(f"Qwen3-4b result for {result.request_id}: {result.result}")

    # Clean up
    model.streaming_engine.stop()


def example_qwen3_coder_streaming():
    """Example of using streaming with Qwen3-coder-30b model."""
    print("\nSetting up Qwen3-coder-30b model with streaming...")

    # Create config for Qwen3-coder-30b
    config = Qwen3Coder30BConfig(
        model_path="Qwen/Qwen3-Coder-30B",  # Using a placeholder for the example
        torch_dtype="float16",
        device_map="cpu",  # Use CPU for the example
        use_flash_attention_2=False,  # Disable for CPU
        use_sparse_attention=False,
        use_tensor_parallelism=False,
    )

    # Create model instance
    model = Qwen3Coder30BModel(config)

    # Setup streaming computation
    model.setup_streaming_computation(max_concurrent_requests=2, buffer_size=10)

    # Example prompts
    prompts = [
        "Write a Python function to calculate factorial.",
        "How do I reverse a linked list?",
        "Explain the difference between stack and queue.",
    ]

    print("Submitting requests to Qwen3-coder-30b streaming engine...")

    # Submit requests
    futures = []
    for i, prompt in enumerate(prompts):
        # Tokenize the prompt
        inputs = model.get_tokenizer()(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        future = model.submit_stream_request(f"qwen3_coder_req_{i}", inputs)
        futures.append(future)

    # Collect results
    for future in futures:
        result = future.result(timeout=30.0)
        if result.error:
            print(f"Error in Qwen3-coder request {result.request_id}: {result.error}")
        else:
            print(f"Qwen3-coder result for {result.request_id}: {result.result}")

    # Clean up
    model.streaming_engine.stop()


def example_qwen3_vl_streaming():
    """Example of using streaming with Qwen3-vl-2b model."""
    print("\nSetting up Qwen3-vl-2b model with streaming...")

    # Create config for Qwen3-vl-2b
    config = Qwen3VL2BConfig(
        model_path="Qwen/Qwen3-VL-2B",  # Using a placeholder for the example
        torch_dtype="float16",
        device_map="cpu",  # Use CPU for the example
        use_flash_attention_2=False,  # Disable for CPU
        use_sparse_attention=False,
        use_tensor_parallelism=False,
    )

    # Create model instance
    model = Qwen3VL2BModel(config)

    # Setup streaming computation
    model.setup_streaming_computation(max_concurrent_requests=2, buffer_size=10)

    # Example prompts (for text-only since we don't have images in this example)
    prompts = [
        "Describe what you see in this image.",
        "What is the relationship between objects in the image?",
        "Analyze the scene in the image.",
    ]

    print("Submitting requests to Qwen3-vl-2b streaming engine...")

    # Submit requests
    futures = []
    for i, prompt in enumerate(prompts):
        # Tokenize the prompt
        inputs = model.get_tokenizer()(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        future = model.submit_stream_request(f"qwen3_vl_req_{i}", inputs)
        futures.append(future)

    # Collect results
    for future in futures:
        result = future.result(timeout=30.0)
        if result.error:
            print(f"Error in Qwen3-vl request {result.request_id}: {result.error}")
        else:
            print(f"Qwen3-vl result for {result.request_id}: {result.result}")

    # Clean up
    model.streaming_engine.stop()


def main():
    """Run all examples."""
    print("Starting examples of streaming computation with different models...\n")

    try:
        example_glm47_streaming()
    except Exception as e:
        print(f"Error in GLM-4-7 example: {e}")

    try:
        example_qwen3_4b_streaming()
    except Exception as e:
        print(f"Error in Qwen3-4b example: {e}")

    try:
        example_qwen3_coder_streaming()
    except Exception as e:
        print(f"Error in Qwen3-coder example: {e}")

    try:
        example_qwen3_vl_streaming()
    except Exception as e:
        print(f"Error in Qwen3-vl example: {e}")

    print("\nAll examples completed!")


if __name__ == "__main__":
    main()
