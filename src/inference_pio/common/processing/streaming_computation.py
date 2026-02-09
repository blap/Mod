"""
Streaming Computation System for Continuous Processing
Dependency-Free
"""

import asyncio
import logging
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Union

from ...core.engine.backend import Module, Tensor

logger = logging.getLogger(__name__)

@dataclass
class StreamRequest:
    id: str
    data: Any
    callback: Optional[Callable] = None
    priority: int = 0
    timestamp: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp == 0.0: self.timestamp = time.time()

    def __lt__(self, other):
        if self.priority != other.priority: return self.priority < other.priority
        return self.timestamp < other.timestamp

@dataclass
class StreamResult:
    request_id: str
    result: Any
    error: Optional[Exception] = None
    processing_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

class StreamingComputationEngine:
    def __init__(
        self,
        model: Module,
        max_concurrent_requests: int = 4,
        buffer_size: int = 100,
        batch_timeout: float = 0.1,
        enable_batching: bool = True,
        device: str = "cpu",
    ):
        self.model = model
        self.max_concurrent_requests = max_concurrent_requests
        self.buffer_size = buffer_size
        self.batch_timeout = batch_timeout
        self.enable_batching = enable_batching
        self.device = device

        self.request_queue = queue.PriorityQueue(maxsize=buffer_size)
        self.result_queue = queue.Queue(maxsize=buffer_size)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_requests)
        self.processing_thread = None
        self.is_running = False
        self.stats = {"requests_processed": 0, "avg_processing_time": 0.0, "total_processing_time": 0.0, "active_requests": 0}
        self.batch_buffer = []
        self.batch_lock = threading.Lock()

        logger.info(f"StreamingEngine init: max_concurrent={max_concurrent_requests}, batching={enable_batching}, device={device}")

    def start(self):
        if self.is_running: return
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_requests, daemon=True)
        self.processing_thread.start()
        logger.info("Started streaming engine")

    def stop(self):
        if not self.is_running: return
        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        self.executor.shutdown(wait=True)
        logger.info("Stopped streaming engine")

    def submit_request(self, request: StreamRequest) -> Future:
        if not self.is_running: raise RuntimeError("Engine not running")
        future = Future()
        try:
            self.request_queue.put_nowait((request.priority, request, future))
        except queue.Full:
            logger.warning("Request queue full")
            future.set_exception(RuntimeError("Queue full"))
        return future

    def _process_requests(self):
        while self.is_running:
            try:
                priority, request, future = self.request_queue.get(timeout=0.1)
                if self.enable_batching:
                    with self.batch_lock:
                        self.batch_buffer.append((priority, request, future))
                        if len(self.batch_buffer) >= self.max_concurrent_requests:
                            self._process_batch()
                        else:
                            threading.Timer(self.batch_timeout, self._process_batch_if_ready).start()
                else:
                    self._process_single_request(request, future)
            except queue.Empty: continue
            except Exception as e: logger.error(f"Error in loop: {e}")

    def _process_batch_if_ready(self):
        with self.batch_lock:
            if len(self.batch_buffer) > 0: self._process_batch()

    def _process_batch(self):
        with self.batch_lock:
            if not self.batch_buffer: return
            items = self.batch_buffer.copy()
            self.batch_buffer.clear()

        items.sort(key=lambda x: x[0])
        futures = []
        for _, req, fut in items:
            f = self.executor.submit(self._process_single_request_sync, req)
            futures.append((fut, f))

        for orig_fut, proc_fut in futures:
            try: orig_fut.set_result(proc_fut.result())
            except Exception as e: orig_fut.set_exception(e)

    def _process_single_request(self, request, future):
        def run():
            try: future.set_result(self._process_single_request_sync(request))
            except Exception as e: future.set_exception(e)
        self.executor.submit(run)

    def _process_single_request_sync(self, request: StreamRequest) -> StreamResult:
        start_time = time.time()
        self.stats["active_requests"] += 1
        try:
            data = self._prepare_input(request.data)
            # Backend ops are implicitly no-grad
            result = self.model(data)

            res = StreamResult(request.id, result, processing_time=time.time()-start_time, metadata=request.metadata)
            self._update_stats(res.processing_time)
            if request.callback:
                try: request.callback(res)
                except Exception: pass
            return res
        except Exception as e:
            self._update_stats(time.time()-start_time)
            return StreamResult(request.id, None, error=e, processing_time=time.time()-start_time, metadata=request.metadata)
        finally:
            self.stats["active_requests"] -= 1

    def _prepare_input(self, data: Any) -> Any:
        if isinstance(data, Tensor): return data.to(self.device)
        elif isinstance(data, dict):
            return {k: (v.to(self.device) if isinstance(v, Tensor) else v) for k, v in data.items()}
        # Basic type conversion if needed, but backend handles list->tensor in creation usually
        return data

    def _update_stats(self, t):
        self.stats["requests_processed"] += 1
        self.stats["total_processing_time"] += t
        self.stats["avg_processing_time"] = self.stats["total_processing_time"] / self.stats["requests_processed"]

    def get_stats(self): return self.stats.copy()

    def generate_stream(self, prompts, max_new_tokens=512, **kwargs):
        if isinstance(prompts, str): prompts = [prompts]
        for i, prompt in enumerate(prompts):
            req_id = f"gen_{int(time.time())}_{i}"
            # Simplified generation call assumption
            # In real usage, this would prepare tensors
            req = StreamRequest(id=req_id, data=prompt, metadata={"prompt": prompt})
            fut = self.submit_request(req)
            try: yield fut.result(timeout=30.0)
            except Exception as e: yield StreamResult(req_id, None, error=e)

class StreamingComputationManager:
    def __init__(self): self.engines = {}
    def register_engine(self, name, engine): self.engines[name] = engine
    def get_engine(self, name): return self.engines.get(name)
    def start_all_engines(self):
        for e in self.engines.values(): e.start()
    def stop_all_engines(self):
        for e in self.engines.values(): e.stop()

streaming_manager = StreamingComputationManager()

def create_streaming_engine(model, name, **kwargs):
    engine = StreamingComputationEngine(model, **kwargs)
    streaming_manager.register_engine(name, engine)
    return engine

__all__ = ["StreamRequest", "StreamResult", "StreamingComputationEngine", "streaming_manager", "create_streaming_engine"]
