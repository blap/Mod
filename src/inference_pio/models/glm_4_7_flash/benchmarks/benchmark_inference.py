"""
Standardized Benchmark for GLM-4.7-Flash
"""
import sys
import os
import time
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.glm_4_7_flash.model import GLM47FlashModel
from src.inference_pio.models.glm_4_7_flash.config import GLM47FlashConfig
from src.inference_pio.common.processing.prompt_utils import format_math_prompt, format_mcq_prompt

def run_benchmark():
    print("="*60)
    print("Benchmark: GLM-4.7-Flash (Standardized)")
    print("="*60)

    start_load = time.time()
    config = GLM47FlashConfig()
    try:
        model = GLM47FlashModel(config)
        print(f"Model Loaded in {time.time() - start_load:.2f}s")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    prompts = [
        ("General", "Hello, artificial intelligence is"),
        ("Math", format_math_prompt("Solve 2x + 5 = 15")),
        ("MCQ", format_mcq_prompt("What is the capital of France? A) London B) Paris C) Berlin"))
    ]

    tokenizer = model._tokenizer

    for p_name, prompt_text in prompts:
        print(f"\n--- Running {p_name} Task ---")
        if tokenizer:
            try:
                 if hasattr(tokenizer, 'encode'):
                    ids = tokenizer.encode(prompt_text)
                 else: ids = [1, 2, 3]

                 if isinstance(ids, list):
                     input_tensor = Tensor([1, len(ids)]); input_tensor.load([float(x) for x in ids])
                 elif isinstance(ids, Tensor): input_tensor = ids
                 else: input_tensor = Tensor([1, 10]); input_tensor.fill(1.0)
            except: input_tensor = Tensor([1, 10]); input_tensor.fill(1.0)
        else:
            input_tensor = Tensor([1, 10]); input_tensor.fill(1.0)

        max_new_tokens = 50
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024

        start_time = time.time()
        try:
            output = model.generate(input_tensor, max_new_tokens=max_new_tokens)
        except Exception as e:
            print(f"Generation failed: {e}")
            continue

        end_time = time.time()
        mem_after = process.memory_info().rss / 1024 / 1024

        total_time = end_time - start_time
        tps = max_new_tokens / total_time if total_time > 0 else 0

        print(f"Time: {total_time:.4f}s | TPS: {tps:.2f} | Mem Delta: {mem_after - mem_before:.2f} MB")

    print("\n" + "="*60)
    print("Benchmark Complete!")

if __name__ == "__main__":
    run_benchmark()
