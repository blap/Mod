"""
Demonstration of Asynchronous Multimodal Processing for Qwen3-VL-2B Model

This script demonstrates the implementation and usage of the asynchronous multimodal processing system
for the Qwen3-VL-2B model. It showcases how text and image inputs can be processed efficiently
in an asynchronous manner to optimize performance and resource utilization.
"""

import asyncio
import time
import torch
from PIL import Image
import numpy as np
import logging
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig


def setup_logging():
    """Setup logging for the demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('async_multimodal_demo.log')
        ]
    )


async def demo_async_multimodal_processing():
    """Demonstrate asynchronous multimodal processing capabilities."""
    print("=" * 80)
    print("ASYNC MULTIMODAL PROCESSING DEMONSTRATION FOR QWEN3-VL-2B")
    print("=" * 80)

    # Create plugin instance
    plugin = create_qwen3_vl_2b_instruct_plugin()

    # Configure the plugin for async multimodal processing
    config = Qwen3VL2BConfig()
    config.enable_async_multimodal_processing = True
    config.async_max_concurrent_requests = 4
    config.async_buffer_size = 100
    config.async_batch_timeout = 0.05
    config.enable_async_batching = True
    config.device = "cpu"  # Use CPU for demonstration

    print(f"Initializing Qwen3-VL-2B plugin with async multimodal processing...")
    success = plugin.initialize(**config.__dict__)
    if not success:
        print("Failed to initialize plugin")
        return

    print("Plugin initialized successfully")
    print(f"Model loaded: {plugin.is_loaded}")
    print(f"Async multimodal processing enabled: {hasattr(plugin._model, 'async_process_multimodal_request')}")

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
        },
        {
            'text': 'Generate a caption for this image.',
            'image': Image.new('RGB', (224, 224), color='yellow')
        }
    ]

    print(f"\nProcessing {len(sample_inputs)} multimodal requests asynchronously...")

    # Time the async processing
    start_time = time.time()
    
    # Process requests asynchronously
    results = await plugin._model.async_process_batch_multimodal_requests(sample_inputs)
    
    end_time = time.time()
    total_time = end_time - start_time

    print(f"Async processing completed in {total_time:.4f} seconds")
    print(f"Average time per request: {total_time / len(sample_inputs):.4f} seconds")
    print(f"Results received: {len([r for r in results if r is not None])}")

    # Process individual requests asynchronously
    print(f"\nProcessing individual multimodal requests asynchronously...")
    
    individual_results = []
    for i, inp in enumerate(sample_inputs):
        start_time = time.time()
        result = await plugin._model.async_process_multimodal_request(
            text=inp['text'],
            image=inp['image']
        )
        end_time = time.time()
        processing_time = end_time - start_time
        individual_results.append(result)
        print(f"Request {i+1} processed in {processing_time:.4f}s")
    
    print(f"Individual async processing completed")
    print(f"Results received: {len([r for r in individual_results if r is not None])}")

    # Demonstrate performance comparison
    print(f"\nDemonstrating performance comparison between sync and async processing...")

    # Synchronous processing for comparison
    sync_start_time = time.time()
    sync_results = []
    for inp in sample_inputs:
        # Fallback to synchronous processing
        inputs = plugin._model.preprocessor.preprocess(text=inp['text'], image=inp['image'])
        result = plugin._model.forward(inputs)
        sync_results.append(result)
    sync_end_time = time.time()
    sync_total_time = sync_end_time - sync_start_time

    print(f"Synchronous processing completed in {sync_total_time:.4f} seconds")
    print(f"Average time per request (sync): {sync_total_time / len(sample_inputs):.4f} seconds")

    # Show comparison
    print(f"\nPERFORMANCE COMPARISON:")
    print(f"  Async processing time: {total_time:.4f}s")
    print(f"  Sync processing time:  {sync_total_time:.4f}s")
    if sync_total_time > 0:
        speedup = sync_total_time / total_time if total_time > 0 else float('inf')
        print(f"  Speedup: {speedup:.2f}x")

    # Show async processing statistics
    if hasattr(plugin._model, '_async_multimodal_manager'):
        stats = plugin._model._async_multimodal_manager.get_stats()
        print(f"\nASYNC PROCESSING STATISTICS:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    print("=" * 80)
    print("ASYNC MULTIMODAL PROCESSING DEMONSTRATION COMPLETED")
    print("=" * 80)


def demo_sync_multimodal_processing():
    """Demonstrate synchronous multimodal processing for comparison."""
    print("\nSYNC MULTIMODAL PROCESSING DEMONSTRATION FOR COMPARISON")
    print("-" * 60)

    # Create plugin instance
    plugin = create_qwen3_vl_2b_instruct_plugin()

    # Configure the plugin (without async processing for comparison)
    config = Qwen3VL2BConfig()
    config.enable_async_multimodal_processing = False  # Disable async processing
    config.device = "cpu"  # Use CPU for demonstration

    print(f"Initializing Qwen3-VL-2B plugin with synchronous processing...")
    success = plugin.initialize(**config.__dict__)
    if not success:
        print("Failed to initialize plugin")
        return

    print("Plugin initialized successfully")

    # Create sample multimodal inputs
    sample_inputs = [
        {
            'text': 'Describe this image in detail.',
            'image': Image.new('RGB', (224, 224), color='red')
        },
        {
            'text': 'What objects do you see in this picture?',
            'image': Image.new('RGB', (224, 224), color='blue')
        }
    ]

    print(f"Processing {len(sample_inputs)} multimodal requests synchronously...")

    # Time the sync processing
    start_time = time.time()
    
    sync_results = []
    for inp in sample_inputs:
        # Synchronous processing
        inputs = plugin._model.preprocessor.preprocess(text=inp['text'], image=inp['image'])
        result = plugin._model.forward(inputs)
        sync_results.append(result)
    
    end_time = time.time()
    total_time = end_time - start_time

    print(f"Synchronous processing completed in {total_time:.4f} seconds")
    print(f"Average time per request: {total_time / len(sample_inputs):.4f} seconds")
    print(f"Results received: {len([r for r in sync_results if r is not None])}")


async def main():
    """Main function to run the demonstration."""
    setup_logging()
    
    print("Starting demonstration of asynchronous multimodal processing for Qwen3-VL-2B model...")
    
    # Run async processing demonstration
    await demo_async_multimodal_processing()
    
    # Run sync processing demonstration for comparison
    demo_sync_multimodal_processing()
    
    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())