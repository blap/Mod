"""
Streaming Computation System for Continuous Processing

This module implements a centralized streaming computation system for continuous processing
across multiple models in the Inference-PIO system. The system reduces latency for
continuous inference and improves hardware resource utilization.
"""

import asyncio
import logging
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class StreamRequest:
    """Represents a streaming computation request."""

    id: str
    data: Any
    callback: Optional[Callable] = None
    priority: int = 0  # Lower numbers have higher priority
    timestamp: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def __lt__(self, other):
        """Define comparison for priority queue ordering."""
        # Compare by priority first, then by timestamp for tie-breaking
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


@dataclass
class StreamResult:
    """Represents a streaming computation result."""

    request_id: str
    result: Any
    error: Optional[Exception] = None
    processing_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class StreamingComputationEngine:
    """
    Centralized streaming computation engine for continuous processing.

    This engine manages a queue of incoming requests and processes them continuously
    to reduce latency and improve resource utilization.
    """

    def __init__(
        self,
        model: nn.Module,
        max_concurrent_requests: int = 4,
        buffer_size: int = 100,
        batch_timeout: float = 0.1,  # seconds
        enable_batching: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize the streaming computation engine.

        Args:
            model: The neural network model to use for computations
            max_concurrent_requests: Maximum number of concurrent requests to process
            buffer_size: Size of the internal request buffer
            batch_timeout: Timeout for batching requests (seconds)
            enable_batching: Whether to enable request batching for efficiency
            device: Device to run computations on (defaults to model's device)
        """
        self.model = model
        self.max_concurrent_requests = max_concurrent_requests
        self.buffer_size = buffer_size
        self.batch_timeout = batch_timeout
        self.enable_batching = enable_batching
        self.device = device or next(model.parameters()).device

        # Internal queues and buffers
        self.request_queue = queue.PriorityQueue(maxsize=buffer_size)
        self.result_queue = queue.Queue(maxsize=buffer_size)

        # Threading and async components
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_requests)
        self.processing_thread = None
        self.is_running = False

        # Statistics
        self.stats = {
            "requests_processed": 0,
            "avg_processing_time": 0.0,
            "total_processing_time": 0.0,
            "active_requests": 0,
        }

        # Batch processing components
        self.batch_buffer = []
        self.batch_lock = threading.Lock()

        logger.info(
            f"Initialized StreamingComputationEngine with max_concurrent={max_concurrent_requests}, "
            f"batching={enable_batching}, device={self.device}"
        )

    def start(self):
        """Start the streaming computation engine."""
        if self.is_running:
            logger.warning("Streaming computation engine already running")
            return

        self.is_running = True
        self.processing_thread = threading.Thread(
            target=self._process_requests, daemon=True
        )
        self.processing_thread.start()
        logger.info("Started streaming computation engine")

    def stop(self):
        """Stop the streaming computation engine."""
        if not self.is_running:
            return

        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)  # Wait up to 5 seconds

        self.executor.shutdown(wait=True)
        logger.info("Stopped streaming computation engine")

    def submit_request(self, request: StreamRequest) -> Future:
        """
        Submit a request to the streaming computation engine.

        Args:
            request: The stream request to submit

        Returns:
            A Future object that can be used to get the result
        """
        if not self.is_running:
            raise RuntimeError("Streaming computation engine is not running")

        # Create a future to return
        future = Future()

        # Add request to queue with priority (lower number = higher priority)
        queue_item = (request.priority, request, future)
        try:
            self.request_queue.put_nowait(queue_item)
        except queue.Full:
            logger.warning("Request queue is full, dropping request")
            future.set_exception(RuntimeError("Request queue is full"))

        return future

    def _process_requests(self):
        """Main processing loop for handling requests."""
        while self.is_running:
            try:
                # Get a request from the queue
                priority, request, future = self.request_queue.get(timeout=0.1)

                if self.enable_batching:
                    # Add to batch buffer
                    with self.batch_lock:
                        self.batch_buffer.append((priority, request, future))

                        # Process batch if it's full or timeout reached
                        if len(self.batch_buffer) >= self.max_concurrent_requests:
                            self._process_batch()
                        else:
                            # Start a timer to process the batch after timeout
                            timer = threading.Timer(
                                self.batch_timeout, self._process_batch_if_ready
                            )
                            timer.daemon = True
                            timer.start()
                else:
                    # Process immediately without batching
                    self._process_single_request(request, future)

            except queue.Empty:
                # Queue is empty, continue loop
                continue
            except Exception as e:
                logger.error(f"Error in request processing loop: {e}")

    def _process_batch_if_ready(self):
        """Process batch if it has accumulated requests."""
        with self.batch_lock:
            if len(self.batch_buffer) > 0:
                self._process_batch()

    def _process_batch(self):
        """Process a batch of requests."""
        with self.batch_lock:
            if not self.batch_buffer:
                return

            batch_items = self.batch_buffer.copy()
            self.batch_buffer.clear()

        # Sort by priority (lower number = higher priority)
        batch_items.sort(key=lambda x: x[0])

        # Process each item in the batch concurrently
        futures = []
        for priority, request, original_future in batch_items:
            future = self.executor.submit(self._process_single_request_sync, request)
            futures.append((original_future, future))

        # Wait for all to complete and set results
        for original_future, processing_future in futures:
            try:
                result = processing_future.result()
                original_future.set_result(result)
            except Exception as e:
                original_future.set_exception(e)

    def _process_single_request(self, request: StreamRequest, future: Future):
        """Process a single request asynchronously."""

        def run():
            try:
                result = self._process_single_request_sync(request)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        self.executor.submit(run)

    def _process_single_request_sync(self, request: StreamRequest) -> StreamResult:
        """Process a single request synchronously."""
        start_time = time.time()
        self.stats["active_requests"] += 1

        try:
            # Move data to appropriate device
            processed_data = self._prepare_input(request.data)

            # Perform the actual computation
            with torch.no_grad():  # Disable gradients for inference
                result = self.model(processed_data)

            # Prepare result
            stream_result = StreamResult(
                request_id=request.id,
                result=result,
                processing_time=time.time() - start_time,
                metadata=request.metadata,
            )

            # Update statistics
            self._update_stats(stream_result.processing_time)

            # Execute callback if provided
            if request.callback:
                try:
                    request.callback(stream_result)
                except Exception as e:
                    logger.error(f"Error in request callback: {e}")

            return stream_result

        except Exception as e:
            # Handle errors
            stream_result = StreamResult(
                request_id=request.id,
                result=None,
                error=e,
                processing_time=time.time() - start_time,
                metadata=request.metadata,
            )

            # Update statistics
            self._update_stats(stream_result.processing_time)

            return stream_result
        finally:
            self.stats["active_requests"] -= 1

    def _prepare_input(self, data: Any) -> Any:
        """Prepare input data for the model."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            # Recursively move tensors in dictionary to device
            prepared_data = {}
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    prepared_data[key] = value.to(self.device)
                else:
                    prepared_data[key] = value
            return prepared_data
        else:
            # For other types, try to convert to tensor if possible
            try:
                return torch.as_tensor(data).to(self.device)
            except Exception:
                # If conversion fails, return as-is
                return data

    def _update_stats(self, processing_time: float):
        """Update processing statistics."""
        self.stats["requests_processed"] += 1
        self.stats["total_processing_time"] += processing_time

        # Calculate average processing time
        count = self.stats["requests_processed"]
        self.stats["avg_processing_time"] = self.stats["total_processing_time"] / count

    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.stats.copy()

    def generate_stream(
        self,
        prompts: Union[str, List[str], Generator],
        max_new_tokens: int = 512,
        **kwargs,
    ) -> Generator[StreamResult, None, None]:
        """
        Generate streaming outputs for continuous processing.

        Args:
            prompts: Single prompt, list of prompts, or generator of prompts
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation arguments

        Yields:
            StreamResult objects for each generated output
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        # Prepare generation arguments
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": kwargs.get("do_sample", True),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "return_dict_in_generate": True,
            "output_scores": False,
        }
        gen_kwargs.update(kwargs)

        # Process each prompt
        for i, prompt in enumerate(prompts):
            # Create a request ID
            req_id = f"gen_{int(time.time())}_{i}"

            # Prepare the input
            tokenizer = getattr(self.model, "get_tokenizer", lambda: None)()
            if tokenizer:
                inputs = tokenizer(
                    prompt, return_tensors="pt", padding=True, truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                # If no tokenizer, assume raw tensor input
                inputs = prompt

            # Create and submit request
            request = StreamRequest(id=req_id, data=inputs, metadata={"prompt": prompt})

            future = self.submit_request(request)

            try:
                result = future.result(timeout=30.0)  # 30 second timeout
                yield result
            except Exception as e:
                yield StreamResult(request_id=req_id, result=None, error=e)


class StreamingComputationManager:
    """
    Manager for multiple streaming computation engines.

    This class manages multiple streaming engines for different models,
    allowing for coordinated processing across models.
    """

    def __init__(self):
        self.engines: Dict[str, StreamingComputationEngine] = {}
        self.active_streams: Dict[str, Any] = {}

    def register_engine(self, name: str, engine: StreamingComputationEngine):
        """Register a streaming computation engine."""
        self.engines[name] = engine
        logger.info(f"Registered streaming engine: {name}")

    def get_engine(self, name: str) -> Optional[StreamingComputationEngine]:
        """Get a registered streaming computation engine."""
        return self.engines.get(name)

    def start_all_engines(self):
        """Start all registered engines."""
        for name, engine in self.engines.items():
            try:
                engine.start()
                logger.info(f"Started engine: {name}")
            except Exception as e:
                logger.error(f"Failed to start engine {name}: {e}")

    def stop_all_engines(self):
        """Stop all registered engines."""
        for name, engine in self.engines.items():
            try:
                engine.stop()
                logger.info(f"Stopped engine: {name}")
            except Exception as e:
                logger.error(f"Failed to stop engine {name}: {e}")

    def submit_request_to_engine(
        self, engine_name: str, request: StreamRequest
    ) -> Optional[Future]:
        """Submit a request to a specific engine."""
        engine = self.get_engine(engine_name)
        if engine:
            return engine.submit_request(request)
        else:
            logger.error(f"Engine {engine_name} not found")
            return None

    def get_engine_stats(self, engine_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific engine."""
        engine = self.get_engine(engine_name)
        if engine:
            return engine.get_stats()
        else:
            return None

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all engines."""
        stats = {}
        for name, engine in self.engines.items():
            stats[name] = engine.get_stats()
        return stats


# Global manager instance for convenience
streaming_manager = StreamingComputationManager()


def create_streaming_engine(
    model: nn.Module,
    name: str,
    max_concurrent_requests: int = 4,
    buffer_size: int = 100,
    batch_timeout: float = 0.1,
    enable_batching: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> StreamingComputationEngine:
    """
    Create and register a streaming computation engine.

    Args:
        model: The model to use for computations
        name: Name for the engine
        max_concurrent_requests: Max concurrent requests
        buffer_size: Buffer size for requests
        batch_timeout: Batch timeout in seconds
        enable_batching: Whether to enable batching
        device: Device to run on

    Returns:
        Created StreamingComputationEngine
    """
    engine = StreamingComputationEngine(
        model=model,
        max_concurrent_requests=max_concurrent_requests,
        buffer_size=buffer_size,
        batch_timeout=batch_timeout,
        enable_batching=enable_batching,
        device=device,
    )

    streaming_manager.register_engine(name, engine)
    return engine


__all__ = [
    "StreamRequest",
    "StreamResult",
    "StreamingComputationEngine",
    "StreamingComputationManager",
    "streaming_manager",
    "create_streaming_engine",
]
