"""
Throughput Benchmark for PaddleOCR-VL-1.5
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
    plugin.load_model()

    # Create dummy image
    image = Image.new('RGB', (448, 448), color='white')

    # We want to generate many tokens to measure token/s
    inputs = {
        "image": image,
        "task": "ocr",
        "max_new_tokens": 100
    }

    print("Running Throughput Benchmark...")
    start = time.time()
    result = plugin.infer(inputs)
    duration = time.time() - start

    # Estimate tokens (approximation since we don't have token count easily exposed from infer string)
    # Ideally we use the tokenizer to count.
    tokenizer = plugin.model_wrapper.processor.tokenizer
    num_tokens = len(tokenizer.encode(result))

    tokens_per_sec = num_tokens / duration
    print(f"Generated {num_tokens} tokens in {duration:.2f}s")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/s")

if __name__ == "__main__":
    run_benchmark()
