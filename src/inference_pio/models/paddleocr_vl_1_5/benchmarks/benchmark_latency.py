"""
Latency Benchmark for PaddleOCR-VL-1.5
"""

import time
import torch
import os
import sys
from PIL import Image
import numpy as np

# Adjust path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.inference_pio.models.paddleocr_vl_1_5.plugin import create_paddleocr_vl_1_5_plugin

def run_benchmark():
    print("Initializing plugin...")
    plugin = create_paddleocr_vl_1_5_plugin()

    # Check if we should run with real model
    # For CI/CD environments without GPU, loading 0.9B model is slow.
    # We will attempt it but maybe on a smaller dummy if possible?
    # No, we must benchmark the real thing.

    try:
        start_load = time.time()
        plugin.load_model()
        load_time = time.time() - start_load
        print(f"Model Load Time: {load_time:.2f}s")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create dummy image
    image = Image.new('RGB', (448, 448), color='white')

    inputs = {
        "image": image,
        "task": "ocr",
        "max_new_tokens": 20 # Short generation for latency test
    }

    print("Warmup...")
    try:
        plugin.infer(inputs)
    except Exception as e:
        print(f"Inference failed: {e}")
        return

    print("Running Latency Benchmark (5 iterations)...")
    latencies = []
    for i in range(5):
        start = time.time()
        plugin.infer(inputs)
        latencies.append(time.time() - start)
        print(f"Iter {i+1}: {latencies[-1]:.4f}s")

    avg_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)

    print(f"Average Latency: {avg_latency:.4f}s")
    print(f"P99 Latency: {p99_latency:.4f}s")

if __name__ == "__main__":
    run_benchmark()
