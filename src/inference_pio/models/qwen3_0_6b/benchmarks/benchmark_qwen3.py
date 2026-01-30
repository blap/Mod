
import time
import torch
import logging
from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Benchmark")

def benchmark_qwen3():
    logger.info("Starting Qwen3-0.6B Benchmark...")

    # Measure Initialization Time
    start_init = time.time()
    plugin = create_qwen3_0_6b_plugin()
    plugin.initialize()
    init_time = time.time() - start_init
    logger.info(f"Initialization Time: {init_time:.4f}s")

    prompts = [
        "Explain the theory of relativity. /think",
        "Explain the theory of relativity. /no_think"
    ]

    for prompt in prompts:
        mode = "Thinking" if "/think" in prompt else "Non-Thinking"
        logger.info(f"\n--- Benchmarking {mode} Mode ---")

        # Warmup
        plugin.generate_text("Warmup", max_new_tokens=10)

        # Measure Latency and Throughput
        start_gen = time.time()
        output = plugin.generate_text(prompt, max_new_tokens=100)
        total_time = time.time() - start_gen

        # Estimate token count (simple split for now, real benchmark would use tokenizer)
        # Using the plugin's tokenizer if available
        if hasattr(plugin, 'tokenize'):
            input_ids = plugin.tokenize(prompt)['input_ids']
            # Handle potential batch dimension
            if isinstance(input_ids[0], list) or (torch.is_tensor(input_ids) and input_ids.ndim > 1):
                input_ids = input_ids[0]

            output_ids = plugin.tokenize(output)['input_ids']
            if isinstance(output_ids[0], list) or (torch.is_tensor(output_ids) and output_ids.ndim > 1):
                output_ids = output_ids[0]

            gen_tokens = len(output_ids)
        else:
            gen_tokens = len(output.split()) * 1.3 # Rough estimate

        tps = gen_tokens / total_time

        logger.info(f"Prompt: {prompt}")
        logger.info(f"Total Time: {total_time:.4f}s")
        logger.info(f"Generated Tokens: {gen_tokens}")
        logger.info(f"Tokens Per Second (TPS): {tps:.2f}")

    # Memory Stats
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.info(f"\nPeak GPU Memory: {peak_mem:.2f} GB")

    plugin.cleanup()

if __name__ == "__main__":
    benchmark_qwen3()
