
import time
import torch
from PIL import Image
import numpy as np

# Mock config
class Config:
    image_size = 448

config = Config()

def resize_images(images):
    resized = []
    for img in images:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        resized.append(img.resize((config.image_size, config.image_size)))
    return resized

def old_processing(resized_images):
    results = []
    for img in resized_images:
        image_array = np.array(img).astype(np.float32)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = (image_tensor / 127.5) - 1.0
        results.append(image_tensor.unsqueeze(0))
    return torch.cat(results, dim=0)

def new_processing(resized_images):
    # Convert to tensors (uint8)
    tensors = [torch.from_numpy(np.array(img)) for img in resized_images]

    # Stack
    batch = torch.stack(tensors) # (B, H, W, C)

    # Permute and cast
    batch = batch.permute(0, 3, 1, 2).float()

    # Normalize
    batch = (batch / 127.5) - 1.0

    return batch

def generate_random_image(size=(1024, 1024)):
    return Image.fromarray(np.random.randint(0, 255, size + (3,), dtype=np.uint8))

def main():
    batch_size = 32
    print(f"Generating {batch_size} random images...")
    images = [generate_random_image() for _ in range(batch_size)]

    print("Resizing images...")
    resized_images = resize_images(images)

    # Benchmark Old
    start = time.time()
    batch_old = old_processing(resized_images)
    end = time.time()
    print(f"Old processing: {end - start:.4f}s")

    # Benchmark New
    start = time.time()
    batch_new = new_processing(resized_images)
    end = time.time()
    print(f"New processing: {end - start:.4f}s")

    # Verify correctness
    assert torch.allclose(batch_old, batch_new, atol=1e-5)
    print("Results match!")

if __name__ == "__main__":
    main()
