"""
Benchmark for GLM-4.7-Flash
"""
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.glm_4_7_flash.config import GLM47FlashConfig
from src.inference_pio.models.glm_4_7_flash.model import GLM47FlashModel

def benchmark_inference(iterations=5, max_new_tokens=20):
    config = GLM47FlashConfig(
        hidden_size=256,
        num_attention_heads=8,
        num_layers=4,
        vocab_size=1000,
        max_position_embeddings=512
    )
    model = GLM47FlashModel(config)

    input_ids = Tensor([1, 10])
    input_ids.load([1.0]*10)

    print(f"Benchmarking GLM47FlashModel...")

    # Warmup
    model.generate(input_ids, max_new_tokens=2)

    start_time = time.time()
    total_tokens = 0

    for i in range(iterations):
        t0 = time.time()
        out = model.generate(input_ids, max_new_tokens=max_new_tokens)
        dt = time.time() - t0
        gen_tokens = out.shape[1] - input_ids.shape[1]
        total_tokens += gen_tokens
        print(f"Iter {i+1}: {gen_tokens} tokens in {dt:.4f}s ({gen_tokens/dt:.2f} t/s)")

    total_time = time.time() - start_time
    if total_time > 0:
        print(f"Average Speed: {total_tokens/total_time:.2f} tokens/sec")

if __name__ == "__main__":
    benchmark_inference()
