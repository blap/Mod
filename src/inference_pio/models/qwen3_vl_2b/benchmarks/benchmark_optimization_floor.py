import time
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from src.inference_pio.models.qwen3_vl_2b.kv_cache.paged_kv_cache import PagedKVCache
from src.inference_pio.models.qwen3_vl_2b.attention.paged_attention import Qwen3VL2BPagedAttention
from src.inference_pio.models.qwen3_vl_2b.kv_cache.scheduler import ContinuousBatchingScheduler
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig

def run_benchmark(num_requests=10, prompt_len=128, gen_len=128, batch_size=4):
    print(f"Running Benchmark: Qwen3-VL-2B (Requests={num_requests}, Prompt={prompt_len}, Gen={gen_len}, Batch={batch_size})")

    config = Qwen3VL2BConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    kv_cache = PagedKVCache(
        num_layers=1, num_heads=config.num_attention_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        max_num_blocks=2048, block_size=16, dtype=dtype, device=device
    )
    attn = Qwen3VL2BPagedAttention(config, layer_idx=0, kv_cache=kv_cache).to(device).to(dtype)
    scheduler = ContinuousBatchingScheduler(kv_cache, max_batch_size=batch_size)

    for i in range(num_requests):
        scheduler.add_request(f"req_{i}", list(range(prompt_len)), max_new_tokens=gen_len)

    start_time = time.time()
    total_tokens_generated = 0
    first_token_times = {}

    step = 0
    while scheduler.waiting_queue or scheduler.running_queue:
        batch_reqs, block_tables, seq_lens, input_ids = scheduler.schedule()
        if not batch_reqs: break

        max_len = max(len(req.prompt_token_ids) if not req.generated_token_ids else 1 for req in batch_reqs)
        hidden_states = torch.randn(len(batch_reqs), max_len, config.hidden_size, device=device, dtype=dtype)

        with torch.no_grad():
            output, _, _ = attn(hidden_states, block_tables=block_tables, seq_lens=seq_lens)

        current_time = time.time()
        for req in batch_reqs:
            if not req.generated_token_ids:
                first_token_times[req.request_id] = current_time - start_time
            scheduler.update_request_status(req, 123)
            total_tokens_generated += 1

        scheduler.free_finished_requests()
        step += 1

    duration = time.time() - start_time
    tps = total_tokens_generated / duration
    avg_ttft = np.mean(list(first_token_times.values()))

    print(f"Results:")
    print(f"  - Total Duration: {duration:.2f} s")
    print(f"  - Total Tokens: {total_tokens_generated}")
    print(f"  - Throughput: {tps:.2f} tokens/s")
    print(f"  - Avg TTFT: {avg_ttft*1000:.2f} ms")

if __name__ == "__main__":
    run_benchmark()
