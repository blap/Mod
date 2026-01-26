"""
Example: Using Asynchronous Multimodal Processing with Qwen3-VL-2B Model

This example demonstrates how to use the asynchronous multimodal processing system
with the Qwen3-VL-2B model for efficient vision-language tasks.
"""

import asyncio
import torch
from PIL import Image
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig


async def example_async_multimodal_processing():
    """Example of using the asynchronous multimodal processing system."""
    print("Example: Asynchronous Multimodal Processing with Qwen3-VL-2B")
    print("=" * 60)

    # Create configuration with async processing enabled
    config = Qwen3VL2BConfig()
    config.enable_async_multimodal_processing = True
    config.async_max_concurrent_requests = 4
    config.async_buffer_size = 100
    config.async_batch_timeout = 0.05
    config.enable_async_batching = True
    config.device = "cpu"  # Use CPU for this example

    # Disable heavy optimizations to speed up initialization
    config.use_flash_attention_2 = False
    config.use_sparse_attention = False
    config.use_sliding_window_attention = False
    config.use_paged_attention = False
    config.enable_disk_offloading = False
    config.enable_intelligent_pagination = False
    config.use_quantization = False
    config.use_tensor_decomposition = False
    config.use_structured_pruning = False

    # Create and initialize the plugin
    plugin = create_qwen3_vl_2b_instruct_plugin()
    success = plugin.initialize(**config.__dict__)
    
    if not success:
        print("Failed to initialize plugin")
        return

    print(f"Plugin initialized successfully: {plugin.metadata.name}")
    print(f"Async multimodal processing enabled: {config.enable_async_multimodal_processing}")

    # Create sample multimodal inputs
    sample_inputs = [
        {
            'text': 'Describe this image in detail.',
            'image': Image.new('RGB', (224, 224), color='red')
        },
        {
            'text': 'What objects do you see in this picture?',
            'image': Image.new('RGB', (224, 224), color='blue')
        },
        {
            'text': 'Analyze the content of this image.',
            'image': Image.new('RGB', (224, 224), color='green')
        }
    ]

    print(f"\nProcessing {len(sample_inputs)} multimodal inputs asynchronously...")

    # Process inputs asynchronously
    start_time = asyncio.get_event_loop().time()
    
    tasks = []
    for i, inp in enumerate(sample_inputs):
        task = plugin._model.async_process_multimodal_request(
            text=inp['text'],
            image=inp['image'],
            request_id=f"req_{i}"
        )
        tasks.append(task)

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    end_time = asyncio.get_event_loop().time()
    total_time = end_time - start_time

    print(f"Processed {len(results)} inputs in {total_time:.4f} seconds")
    print(f"Average time per input: {total_time / len(results):.4f} seconds")

    # Process batch of inputs
    print(f"\nProcessing batch of {len(sample_inputs)} multimodal inputs asynchronously...")
    
    batch_start_time = asyncio.get_event_loop().time()
    batch_results = await plugin._model.async_process_batch_multimodal_requests(sample_inputs)
    batch_end_time = asyncio.get_event_loop().time()
    batch_total_time = batch_end_time - batch_start_time

    print(f"Batch processed in {batch_total_time:.4f} seconds")
    print(f"Inputs processed: {len(batch_results)}")

    # Compare with synchronous processing
    print(f"\nComparing with synchronous processing...")
    
    sync_start_time = asyncio.get_event_loop().time()
    sync_results = []
    for inp in sample_inputs:
        # Synchronous processing
        processed_data = plugin._model.preprocessor.preprocess(
            text=inp['text'],
            image=inp['image']
        )
        result = plugin._model.forward(processed_data)
        sync_results.append(result)
    sync_end_time = asyncio.get_event_loop().time()
    sync_total_time = sync_end_time - sync_start_time

    print(f"Synchronous processing took {sync_total_time:.4f} seconds")
    print(f"Async processing took {total_time:.4f} seconds")
    
    if sync_total_time > 0:
        speedup = sync_total_time / total_time if total_time > 0 else float('inf')
        print(f"Speedup: {speedup:.2f}x")

    # Show async processing statistics
    if hasattr(plugin._model, '_async_multimodal_manager'):
        stats = plugin._model._async_multimodal_manager.get_stats()
        print(f"\nAsync Processing Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    print("\nExample completed successfully!")


def run_example():
    """Run the async multimodal processing example."""
    print("Running example of asynchronous multimodal processing for Qwen3-VL-2B...")
    
    # Run the async example
    asyncio.run(example_async_multimodal_processing())


if __name__ == "__main__":
    run_example()