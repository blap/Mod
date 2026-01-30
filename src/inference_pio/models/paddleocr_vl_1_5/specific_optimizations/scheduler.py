"""
Continuous Batching Scheduler for PaddleOCR-VL-1.5

This module implements an Iteration Level Scheduler (Orca-style) for managing
incoming requests and executing them in continuous batches.
"""

import threading
import queue
import uuid
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Request:
    request_id: str
    prompt: str
    image: Optional[object]
    arrival_time: float
    max_tokens: int
    generated_tokens: List[int]
    finished: bool = False
    seq_id: Optional[int] = None

class ContinuousBatchingScheduler:
    def __init__(self, max_batch_size: int = 16, max_num_seqs: int = 256):
        self.max_batch_size = max_batch_size
        self.waiting_queue: queue.Queue = queue.Queue()
        self.running_queue: List[Request] = []
        self.finished_queue: queue.Queue = queue.Queue()
        self.lock = threading.Lock()

        # Sequence ID management
        self.next_seq_id = 0

    def add_request(self, prompt: str, image=None, max_tokens: int = 512) -> str:
        request_id = str(uuid.uuid4())
        import time
        req = Request(
            request_id=request_id,
            prompt=prompt,
            image=image,
            arrival_time=time.time(),
            max_tokens=max_tokens,
            generated_tokens=[]
        )
        self.waiting_queue.put(req)
        return request_id

    def get_next_batch(self) -> List[Request]:
        """
        Form the next batch by combining running requests and new requests from queue.
        Iteration Level Scheduling.
        """
        with self.lock:
            # 1. Filter finished requests
            active_running = []
            for req in self.running_queue:
                if req.finished:
                    self.finished_queue.put(req)
                    # Release seq_id resource here ideally
                else:
                    active_running.append(req)
            self.running_queue = active_running

            # 2. Add new requests if space allows
            spaces_available = self.max_batch_size - len(self.running_queue)
            while spaces_available > 0 and not self.waiting_queue.empty():
                try:
                    new_req = self.waiting_queue.get_nowait()
                    # Assign seq_id
                    new_req.seq_id = self.next_seq_id
                    self.next_seq_id += 1

                    self.running_queue.append(new_req)
                    spaces_available -= 1
                except queue.Empty:
                    break

            return list(self.running_queue)

    def mark_finished(self, request_id: str):
        with self.lock:
            for req in self.running_queue:
                if req.request_id == request_id:
                    req.finished = True
                    break

    def has_pending_work(self) -> bool:
        with self.lock:
            return not self.waiting_queue.empty() or len(self.running_queue) > 0
