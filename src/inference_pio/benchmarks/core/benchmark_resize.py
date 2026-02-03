import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def generate_random_image(size=(1024, 1024)):
    return Image.fromarray(np.random.randint(0, 255, size + (3,), dtype=np.uint8))


def old_pipeline(image, image_size=448, compression_ratio=0.5):
    # 1. Resize to image_size (PIL)
    img_resized = image.resize((image_size, image_size))

    # 2. To Tensor & Normalize
    arr = np.array(img_resized).astype(np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = (tensor / 127.5) - 1.0
    tensor = tensor.unsqueeze(0)

    # 3. Compress (Interpolate)
    ratio = compression_ratio**0.5
    target_size = int(image_size * ratio)  # 448 * 0.707 = 316

    compressed = F.interpolate(
        tensor, size=(target_size, target_size), mode="bilinear", align_corners=False
    )
    return compressed


def new_pipeline(image, image_size=448, compression_ratio=0.5):
    # 1. Calc target size
    ratio = compression_ratio**0.5
    target_size = int(image_size * ratio)  # 316

    # 2. Resize directly to target_size (PIL)
    img_resized = image.resize((target_size, target_size))

    # 3. To Tensor & Normalize
    arr = np.array(img_resized).astype(np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = (tensor / 127.5) - 1.0
    tensor = tensor.unsqueeze(0)

    # No interpolation needed
    return tensor


def main():
    image = generate_random_image()
    iterations = 100

    # Warmup
    old_pipeline(image)
    new_pipeline(image)

    start = time.time()
    for _ in range(iterations):
        old_pipeline(image)
    end = time.time()
    print(f"Old pipeline: {end - start:.4f}s")

    start = time.time()
    for _ in range(iterations):
        new_pipeline(image)
    end = time.time()
    print(f"New pipeline: {end - start:.4f}s")

    # Correctness check
    # Note: Interpolation results might differ slightly from PIL resize due to algorithm differences
    # But roughly should be similar shape
    res_old = old_pipeline(image)
    res_new = new_pipeline(image)
    print(f"Old shape: {res_old.shape}")
    print(f"New shape: {res_new.shape}")


if __name__ == "__main__":
    main()
