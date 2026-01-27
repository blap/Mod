
import sys
import time
from unittest.mock import MagicMock
import torch
from PIL import Image
import numpy as np

# Mock transformers.AutoModelForVision2Seq to avoid ImportError
import transformers
transformers.AutoModelForVision2Seq = MagicMock()

# Now import the target module
from src.inference_pio.models.qwen3_vl_2b.image_tokenization import ImageTokenizationConfig, ImageTokenizer

def generate_random_image(size=(1024, 1024)):
    return Image.fromarray(np.random.randint(0, 255, size + (3,), dtype=np.uint8))

def main():
    # Enable compression to test the optimization
    config = ImageTokenizationConfig(
        image_size=448,
        enable_batch_processing=True,
        enable_quantization=False,
        enable_compression=True,
        compression_ratio=0.5
    )

    # This will use BasicImageProcessor because the path H:/... doesn't exist
    tokenizer = ImageTokenizer(config)

    # Create batch of images
    batch_size = 32
    print(f"Generating {batch_size} random images...")
    images = [generate_random_image() for _ in range(batch_size)]

    print("Starting benchmark with compression enabled...")
    start_time = time.time()

    output = tokenizer.batch_tokenize(images)

    end_time = time.time()
    duration = end_time - start_time

    print(f"Batch processing of {batch_size} images took {duration:.4f} seconds")
    print(f"Average time per image: {duration/batch_size:.4f} seconds")
    print(f"Output shape: {output['pixel_values'].shape}")

if __name__ == "__main__":
    main()
