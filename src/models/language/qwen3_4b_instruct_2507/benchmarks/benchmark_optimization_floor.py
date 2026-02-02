import os
import sys
import time

import numpy as np
import torch

# Add repo root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from src.models.qwen3_4b_instruct_2507.attention.paged_attention import (
    Qwen34BPagedAttention,
)
from src.models.qwen3_4b_instruct_2507.config import Qwen34BInstruct2507Config
from src.models.qwen3_4b_instruct_2507.kv_cache.paged_kv_cache import PagedKVCache
from src.models.qwen3_4b_instruct_2507.kv_cache.scheduler import (
    ContinuousBatchingScheduler,
)


def run_benchmark(num_requests=10, prompt_len=128, gen_len=128, batch_size=4):
    print(
        f"Running Benchmark: Qwen3-4B (Requests={num_requests}, Prompt={prompt_len}, Gen={gen_len}, Batch={batch_size})"
    )

    config = Qwen34BInstruct2507Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Setup
    kv_cache = PagedKVCache(
        num_layers=1,
        num_heads=config.num_attention_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        max_num_blocks=2048,
        block_size=16,
        dtype=dtype,
        device=device,
    )
    attn = (
        Qwen34BPagedAttention(config, layer_idx=0, kv_cache=kv_cache)
        .to(device)
        .to(dtype)
    )
    scheduler = ContinuousBatchingScheduler(kv_cache, max_batch_size=batch_size)

    # Add Requests
    for i in range(num_requests):
        scheduler.add_request(
            f"req_{i}", list(range(prompt_len)), max_new_tokens=gen_len
        )

    start_time = time.time()
    total_tokens_generated = 0
    first_token_times = {}

    # Simulation Loop
    step = 0
    while scheduler.waiting_queue or scheduler.running_queue:
        iter_start = time.time()

        # 1. Schedule
        batch_reqs, block_tables, seq_lens, input_ids = scheduler.schedule()

        if not batch_reqs:
            break

        # 2. Model Forward (Simulated Latency for pure attention + linear)
        # We use the real attention module, but inputs are dummy tensors
        # Shape: [batch, 1 or prompt_len, hidden]
        # We need to reconstruct the input tensor shape correctly
        # Prefill requests have len > 1, decode has len = 1.
        # For simplicity in this benchmark, we treat the batch as a flat list of tokens but attention expects [bsz, seq_len, ...]
        # Since standard SDPA doesn't support ragged batches natively without nested tensor or padding,
        # we will simulate the compute cost by running attention on the longest sequence in the batch (padding simulation).

        max_len = max(
            len(req.prompt_token_ids) if not req.generated_token_ids else 1
            for req in batch_reqs
        )

        # Create dummy hidden states
        hidden_states = torch.randn(
            len(batch_reqs), max_len, config.hidden_size, device=device, dtype=dtype
        )

        # Run Attention
        with torch.no_grad():
            output, _, _ = attn(
                hidden_states,
                hidden_states,
                hidden_states,
                block_tables=block_tables,
                seq_lens=seq_lens,
            )

        torch.cuda.synchronize() if device == "cuda" else None

        # 3. Update Status
        current_time = time.time()
        for req in batch_reqs:
            if not req.generated_token_ids:
                first_token_times[req.request_id] = current_time - start_time

            # Simulate sampling 1 token
            scheduler.update_request_status(req, 123)
            total_tokens_generated += 1

        scheduler.free_finished_requests()
        step += 1

    end_time = time.time()
    duration = end_time - start_time

    avg_ttft = np.mean(list(first_token_times.values()))
    tps = total_tokens_generated / duration

    print(f"Results:")
    print(f"  - Total Duration: {duration:.2f} s")
    print(f"  - Total Tokens: {total_tokens_generated}")
    print(f"  - Throughput: {tps:.2f} tokens/s")
    print(f"  - Avg TTFT: {avg_ttft*1000:.2f} ms")


if __name__ == "__main__":
    run_benchmark()
