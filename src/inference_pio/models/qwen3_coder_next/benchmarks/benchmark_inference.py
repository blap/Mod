"""
Benchmark for Qwen3-Coder-Next
"""
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.qwen3_coder_next.config import create_qwen3_coder_next_config
from src.inference_pio.models.qwen3_coder_next.model import Qwen3CoderNextForCausalLM

def benchmark_inference(iterations=5, max_new_tokens=20):
    # Use a medium config
    config = create_qwen3_coder_next_config(
        hidden_size=256,
        num_hidden_layers=4,
        vocab_size=1000,
        num_attention_heads=8,
        intermediate_size=512,
        max_position_embeddings=512,
        deltanet_query_key_heads=8,
        deltanet_value_heads=8,
        deltanet_head_dim=32,
        hybrid_block_pattern=["deltanet", "attention"],
        num_experts=8,
        num_experts_per_tok=2
    )
    model = Qwen3CoderNextForCausalLM(config)

    input_ids = Tensor([1, 10])
    input_ids.load([1.0]*10)

    print(f"Benchmarking Qwen3CoderNextForCausalLM...")

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
