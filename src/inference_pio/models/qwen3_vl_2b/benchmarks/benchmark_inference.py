"""
Benchmark for Qwen3-VL-2B
"""
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel

def benchmark_inference(iterations=5, max_new_tokens=20):
    config = Qwen3VL2BConfig(
        hidden_size=256, num_attention_heads=8, num_hidden_layers=4, vocab_size=1000,
        vision_hidden_size=64, vision_num_attention_heads=4, vision_num_hidden_layers=2,
        vision_patch_size=2, vision_image_size=8
    )
    model = Qwen3VL2BModel(config)

    input_ids = Tensor([1, 10])
    input_ids.load([1.0]*10)

    pixels = Tensor([1, 3, 8, 8])
    pixels.fill(0.5)

    print(f"Benchmarking Qwen3VL2BModel (Multimodal)...")

    # Warmup
    model.generate(input_ids, pixel_values=pixels, max_new_tokens=2)

    start_time = time.time()
    total_tokens = 0

    for i in range(iterations):
        t0 = time.time()
        out = model.generate(input_ids, pixel_values=pixels, max_new_tokens=max_new_tokens)
        dt = time.time() - t0
        # For Qwen3-VL, out length is input text + generated.
        gen_tokens = out.shape[1] - input_ids.shape[1]
        total_tokens += gen_tokens
        print(f"Iter {i+1}: {gen_tokens} tokens in {dt:.4f}s ({gen_tokens/dt:.2f} t/s)")

    total_time = time.time() - start_time
    if total_time > 0:
        print(f"Average Speed: {total_tokens/total_time:.2f} tokens/sec")

if __name__ == "__main__":
    benchmark_inference()
