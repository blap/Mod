from typing import List, Dict, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass
import logging
from ...core.engine.backend import Tensor

logger = logging.getLogger(__name__)

@dataclass
class BatchRequest:
    req_id: int
    input_ids: List[float]
    status: str = "PENDING" # PENDING, RUNNING, COMPLETED
    output: Optional[Tensor] = None

class BatchManager:
    """
    Serial Batch Manager implementation.
    Processes requests in a strict FIFO queue.
    LIMITATION: Does not interleave tokens (no continuous batching).
    """
    def __init__(self, model: Any, block_size: int = 16):
        self.model = model
        self.block_size = block_size
        self.request_queue = deque()
        self.running_request: Optional[BatchRequest] = None
        self.completed_history: Dict[int, BatchRequest] = {}

    def add_request(self, req_id: int, input_ids: List[float]):
        req = BatchRequest(req_id=req_id, input_ids=input_ids)
        self.request_queue.append(req)
        logger.info(f"Request {req_id} added to queue. Queue size: {len(self.request_queue)}")

    def step(self) -> Optional[Tensor]:
        """
        Process the next request in the queue to completion.
        Returns the output tensor of the processed request, or None if queue empty.
        """
        if not self.request_queue:
            return None

        # FCFS: Pop from left
        req = self.request_queue.popleft()
        req.status = "RUNNING"
        self.running_request = req

        logger.info(f"BatchManager: Processing Request {req.req_id} serially (Blocking FCFS).")

        try:
            # Convert to Tensor
            t = Tensor([1, len(req.input_ids)])
            t.load(req.input_ids)

            # Execute Model (Blocking)
            # This uses the standardized static KV cache generation loop
            output_tensor = self.model.generate(t)

            req.output = output_tensor
            req.status = "COMPLETED"
            self.completed_history[req.req_id] = req

            self.running_request = None
            return output_tensor

        except Exception as e:
            logger.error(f"Error processing request {req.req_id}: {e}")
            req.status = "FAILED"
            self.completed_history[req.req_id] = req
            self.running_request = None
            return None

    def get_status(self, req_id: int) -> str:
        # Check running
        if self.running_request and self.running_request.req_id == req_id:
            return self.running_request.status
        # Check queue
        for req in self.request_queue:
            if req.req_id == req_id:
                return req.status
        # Check history
        if req_id in self.completed_history:
            return self.completed_history[req_id].status

        return "UNKNOWN"
