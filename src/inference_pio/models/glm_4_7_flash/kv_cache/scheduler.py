"""
GLM-4.7-Flash Continuous Batching Scheduler

This module implements iteration-level scheduling (Orca-style) for continuous batching.
"""

import dataclasses
from typing import List, Optional, Dict, Tuple
import torch
import logging

from .paged_kv_cache import PagedKVCache

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Request:
    """Represents a generation request."""
    request_id: str
    prompt_token_ids: List[int]
    max_new_tokens: int
    generated_token_ids: List[int] = dataclasses.field(default_factory=list)
    block_table: List[int] = dataclasses.field(default_factory=list)
    finished: bool = False

    @property
    def total_len(self):
        return len(self.prompt_token_ids) + len(self.generated_token_ids)


class ContinuousBatchingScheduler:
    """
    Scheduler for Continuous Batching (Iteration Level Scheduling) for GLM-4.7.
    """

    def __init__(
        self,
        kv_cache: PagedKVCache,
        max_batch_size: int = 32,
        max_context_len: int = 4096
    ):
        self.kv_cache = kv_cache
        self.max_batch_size = max_batch_size
        self.max_context_len = max_context_len

        self.waiting_queue: List[Request] = []
        self.running_queue: List[Request] = []

        self.block_size = kv_cache.block_size

    def add_request(self, request_id: str, prompt_token_ids: List[int], max_new_tokens: int):
        request = Request(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens
        )
        self.waiting_queue.append(request)

    def _allocate_blocks_for_request(self, request: Request, num_tokens_to_alloc: int) -> bool:
        num_blocks_needed = (num_tokens_to_alloc + self.block_size - 1) // self.block_size
        try:
            new_blocks = self.kv_cache.allocate_blocks(num_blocks_needed)
            request.block_table.extend(new_blocks)
            return True
        except RuntimeError:
            return False

    def schedule(self) -> Tuple[List[Request], List[List[int]], List[int], List[int]]:
        # 1. Add new requests
        while self.waiting_queue and len(self.running_queue) < self.max_batch_size:
            candidate_request = self.waiting_queue[0]
            num_prompt_tokens = len(candidate_request.prompt_token_ids)
            if self._allocate_blocks_for_request(candidate_request, num_prompt_tokens):
                self.waiting_queue.pop(0)
                self.running_queue.append(candidate_request)
            else:
                break

        # 2. Allocate for running requests
        requests_to_preempt = []
        for request in self.running_queue:
            current_len = request.total_len
            if current_len > 0 and current_len % self.block_size == 0:
                if not self._allocate_blocks_for_request(request, 1):
                    requests_to_preempt.append(request)

        for req in requests_to_preempt:
            self.running_queue.remove(req)
            self.kv_cache.free_blocks(req.block_table)

        # 3. Prepare outputs
        batch_requests = []
        batch_block_tables = []
        batch_seq_lens = []
        batch_input_ids = []

        for req in self.running_queue:
            batch_requests.append(req)
            batch_block_tables.append(req.block_table)

            if not req.generated_token_ids:
                batch_input_ids.extend(req.prompt_token_ids)
                batch_seq_lens.append(len(req.prompt_token_ids))
            else:
                batch_input_ids.append(req.generated_token_ids[-1])
                batch_seq_lens.append(req.total_len)

        return batch_requests, batch_block_tables, batch_seq_lens, batch_input_ids

    def update_request_status(self, request: Request, new_token_id: int):
        request.generated_token_ids.append(new_token_id)
        if len(request.generated_token_ids) >= request.max_new_tokens:
            request.finished = True

    def free_finished_requests(self):
        finished_indices = []
        for i, req in enumerate(self.running_queue):
            if req.finished:
                finished_indices.append(i)
                self.kv_cache.free_blocks(req.block_table)

        for i in sorted(finished_indices, reverse=True):
            self.running_queue.pop(i)
