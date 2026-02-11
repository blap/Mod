from typing import List, Dict, Tuple
from ...core.engine.backend import Tensor
from ...models.qwen3_coder_next.model import Qwen3CoderNextForCausalLM

class BatchManager:
    """
    Manages continuous batching state (block tables, sequences).
    """
    def __init__(self, model: Qwen3CoderNextForCausalLM, block_size: int = 16):
        self.model = model
        self.block_size = block_size
        self.requests: Dict[int, Dict] = {} # id -> {input_ids, output_ids, block_table}
        self.free_blocks: List[int] = list(range(1024)) # Mock allocator

    def add_request(self, req_id: int, input_ids: List[int]):
        # Allocate blocks
        num_blocks = (len(input_ids) + self.block_size - 1) // self.block_size
        blocks = [self.free_blocks.pop() for _ in range(num_blocks)]

        self.requests[req_id] = {
            "input_ids": input_ids,
            "blocks": blocks,
            "status": "prefill"
        }

    def step(self):
        # 1. Select active requests
        # 2. Prepare batch inputs (flattened)
        # 3. Prepare block_tables tensor
        # 4. Run model.forward(..., use_paged_attn=True)
        # 5. Sample
        # 6. Update requests
        pass
