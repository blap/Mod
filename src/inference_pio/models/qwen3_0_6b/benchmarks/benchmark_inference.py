"""
Standardized Benchmark for Qwen3-0.6B
Tests:
1. Thinking Mode (Standard)
2. Non-Thinking Mode
3. Math Prompting
4. MCQ Prompting
"""
import sys
import os
import time
import psutil

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.qwen3_0_6b.config import Qwen3_0_6bConfig as Qwen3_0_6B_Config
from src.inference_pio.models.qwen3_0_6b.model import Qwen3_0_6B_Model
from src.inference_pio.common.processing.prompt_utils import (
    format_math_prompt,
    format_mcq_prompt,
    apply_chat_template
)

def run_benchmark():
    print("="*60)
    print("Benchmark: Qwen3-0.6B (Standardized)")
    print("="*60)

    # 1. Instantiate Model
    print("Loading Model...")
    start_load = time.time()
    config = Qwen3_0_6B_Config()
    try:
        model = Qwen3_0_6B_Model(config)
        print(f"Model Loaded in {time.time() - start_load:.2f}s")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    tokenizer = model._tokenizer
    if not tokenizer:
        print("Warning: No tokenizer found. Benchmark limited.")
        return

    # 2. Define Tasks
    tasks = []

    # Task 1: Thinking Mode (Default)
    messages_think = [{"role": "user", "content": "Explain quantum entanglement."}]
    prompt_think = apply_chat_template(tokenizer, messages_think, enable_thinking=True, tokenize=False)
    tasks.append(("Thinking Mode", prompt_think))

    # Task 2: Non-Thinking Mode
    messages_no_think = [{"role": "user", "content": "Explain quantum entanglement."}]
    prompt_no_think = apply_chat_template(tokenizer, messages_no_think, enable_thinking=False, tokenize=False)
    tasks.append(("Non-Thinking Mode", prompt_no_think))

    # Task 3: Math
    math_query = format_math_prompt("Solve for x: 3x + 9 = 21")
    messages_math = [{"role": "user", "content": math_query}]
    prompt_math = apply_chat_template(tokenizer, messages_math, enable_thinking=True, tokenize=False)
    tasks.append(("Math Task", prompt_math))

    # Task 4: MCQ
    mcq_query = format_mcq_prompt("What is the speed of light? A) 3e8 m/s B) 300 km/h C) Infinite")
    messages_mcq = [{"role": "user", "content": mcq_query}]
    prompt_mcq = apply_chat_template(tokenizer, messages_mcq, enable_thinking=False, tokenize=False) # MCQ usually strict answer
    tasks.append(("MCQ Task", prompt_mcq))

    # Run Tasks
    for task_name, prompt_text in tasks:
        print(f"\n--- Running {task_name} ---")
        print(f"Prompt Start: {prompt_text[:100]!r}...")

        try:
             ids = tokenizer.encode(prompt_text)
             if isinstance(ids, list):
                 input_tensor = Tensor([1, len(ids)]); input_tensor.load([float(x) for x in ids])
             elif isinstance(ids, Tensor):
                 input_tensor = ids
             else:
                 input_tensor = Tensor([1, 10]); input_tensor.fill(1.0)
        except Exception as e:
             print(f"Tokenizer error: {e}. Skipping.")
             continue

        # Benchmark
        max_new_tokens = 30 # Short generation for speed
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

        # Verify output structure (mock check since weights are random)
        # In real scenario, we'd check for <think> tags in the decoded output.
        # decoded = tokenizer.decode(output...)
        # if "Thinking" in task_name: assert "<think>" in decoded (conceptually)

    print("\n" + "="*60)
    print("Benchmark Complete!")

if __name__ == "__main__":
    run_benchmark()
