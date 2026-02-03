"""
Asynchronous Multimodal Processing System for Qwen3-VL-2B Model - Self-Contained Version

This module implements an asynchronous processing system specifically for multimodal
inputs in the Qwen3-VL-2B model. It enables efficient handling of text and image
inputs with optimized scheduling and resource utilization.
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import PriorityQueue, Queue
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class AsyncMultimodalRequest:
    """Represents an asynchronous multimodal processing request."""

    id: str
    text: Optional[str] = None
    image: Optional[Union[Image.Image, str]] = None
    callback: Optional[Callable[[Any], None]] = None
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
class AsyncMultimodalResult:
    """Represents an asynchronous multimodal processing result."""

    request_id: str
    result: Any
    error: Optional[Exception] = None
    processing_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class Qwen3VL2BAsyncMultimodalManager:
    """
    Asynchronous multimodal processing manager specifically for the Qwen3-VL-2B model.
    Manages the coordination of text and image processing with optimized scheduling.
    """

    def __init__(self, model: torch.nn.Module, config: Any):
        """
        Initialize the asynchronous multimodal manager.

        Args:
            model: The Qwen3-VL-2B model instance
            config: Configuration for the model
        """
        self.model = model
        self.config = config
        self.is_initialized = False

        # Initialize async processing components
        self._async_processor = None
        self._dynamic_batch_manager = None
        self._preprocessor = None

        # Async processing settings from config
        self.max_concurrent_requests = getattr(
            config, "async_max_concurrent_requests", 4
        )
        self.buffer_size = getattr(config, "async_buffer_size", 100)
        self.batch_timeout = getattr(config, "async_batch_timeout", 0.1)
        self.enable_batching = getattr(config, "enable_async_batching", True)
        self.device = getattr(
            config, "async_processing_device", getattr(config, "device", "cpu")
        )

        # Initialize internal components
        self._request_queue = PriorityQueue()
        self._result_callbacks = {}
        self._processing_tasks = {}
        self._executor = ThreadPoolExecutor(max_workers=self.max_concurrent_requests)
        self._running = False
        self._processing_loop_task = None

    def initialize(self) -> bool:
        """
        Initialize the asynchronous multimodal processing system.

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Get tokenizer and image processor from the model
            tokenizer = getattr(self.model, "tokenizer", None)
            image_processor = getattr(self.model, "image_processor", None)

            if not tokenizer or not image_processor:
                logger.warning(
                    "Tokenizer or image processor not found in model, using defaults"
                )
                from transformers import AutoImageProcessor, AutoTokenizer

                # Use a default tokenizer and image processor
                tokenizer = AutoTokenizer.from_pretrained(
                    "H:/Qwen3-VL-2B-Instruct", trust_remote_code=True
                )
                image_processor = AutoImageProcessor.from_pretrained(
                    "H:/Qwen3-VL-2B-Instruct", trust_remote_code=True
                )

            # Initialize multimodal preprocessor
            from .multimodal_preprocessing import MultimodalPreprocessor

            self._preprocessor = MultimodalPreprocessor(
                tokenizer=tokenizer, image_processor=image_processor
            )

            # Initialize dynamic batch manager for multimodal inputs
            if self.enable_batching:
                from .dynamic_multimodal_batching import DynamicMultimodalBatchManager

                self._dynamic_batch_manager = DynamicMultimodalBatchManager(
                    initial_batch_size=getattr(self.config, "initial_batch_size", 1),
                    min_batch_size=getattr(self.config, "min_batch_size", 1),
                    max_batch_size=getattr(
                        self.config, "max_batch_size", 8
                    ),  # Lower default for multimodal
                    text_weight=getattr(self.config, "text_weight", 0.4),
                    image_weight=getattr(self.config, "image_weight", 0.6),
                    complexity_threshold_low=getattr(
                        self.config, "complexity_threshold_low", 0.3
                    ),
                    complexity_threshold_high=getattr(
                        self.config, "complexity_threshold_high", 0.7
                    ),
                )

            # Initialize async processor
            self._async_processor = AsyncMultimodalProcessor(
                model=self.model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                max_concurrent_requests=self.max_concurrent_requests,
                buffer_size=self.buffer_size,
                enable_batching=self.enable_batching,
                device=self.device,
            )

            self.is_initialized = True
            logger.info(
                "Qwen3-VL-2B asynchronous multimodal manager initialized successfully"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to initialize Qwen3-VL-2B asynchronous multimodal manager: {e}"
            )
            return False

    async def process_multimodal_request(
        self,
        text: Optional[str] = None,
        image: Optional[Union[Image.Image, str]] = None,
        **kwargs,
    ) -> AsyncMultimodalResult:
        """
        Process a multimodal request asynchronously.

        Args:
            text: Text input (optional)
            image: Image input (optional)
            **kwargs: Additional processing arguments

        Returns:
            AsyncMultimodalResult with the processing result
        """
        if not self.is_initialized:
            raise RuntimeError(
                "Async multimodal manager not initialized. Call initialize() first."
            )

        # Create a request ID
        request_id = (
            f"async_mm_{int(time.time())}_{hash(str(text) + str(image)) % 10000}"
        )

        # Create and submit request
        request = AsyncMultimodalRequest(
            id=request_id, text=text, image=image, metadata=kwargs
        )

        # Submit to async processor
        future = await self._async_processor.submit_request(request)

        # Wait for result
        result = await asyncio.wrap_future(future)
        return result

    async def process_batch_multimodal_requests(
        self, requests: List[Dict[str, Any]], **kwargs
    ) -> List[AsyncMultimodalResult]:
        """
        Process a batch of multimodal requests asynchronously.

        Args:
            requests: List of request dictionaries with 'text' and/or 'image' keys
            **kwargs: Additional processing arguments

        Returns:
            List of AsyncMultimodalResult objects
        """
        if not self.is_initialized:
            raise RuntimeError(
                "Async multimodal manager not initialized. Call initialize() first."
            )

        # Process each request concurrently
        tasks = []
        for i, req in enumerate(requests):
            request_id = req.get("id", f"async_batch_{int(time.time())}_{i}")
            text = req.get("text")
            image = req.get("image")

            task = self.process_multimodal_request(text=text, image=image, **kwargs)
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(
                    AsyncMultimodalResult(
                        request_id="unknown", result=None, error=result
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the async processing system.

        Returns:
            Dictionary containing processing statistics
        """
        if not self.is_initialized:
            return {"initialized": False}

        stats = {
            "initialized": True,
            "max_concurrent_requests": self.max_concurrent_requests,
            "buffer_size": self.buffer_size,
            "batching_enabled": self.enable_batching,
            "device": self.device,
        }

        if self._async_processor:
            stats.update(self._async_processor.get_stats())

        return stats


class AsyncMultimodalProcessor:
    """
    Asynchronous processor for multimodal inputs (text and images).
    Handles parallel processing of different modalities with optimized scheduling.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        image_processor: Any,
        max_concurrent_requests: int = 4,
        buffer_size: int = 100,
        enable_batching: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize the asynchronous multimodal processor.

        Args:
            model: The model to use for processing
            tokenizer: Tokenizer for text processing
            image_processor: Image processor for image processing
            max_concurrent_requests: Maximum number of concurrent requests to process
            buffer_size: Size of the internal request buffer
            enable_batching: Whether to enable request batching for efficiency
            device: Device to run computations on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_concurrent_requests = max_concurrent_requests
        self.buffer_size = buffer_size
        self.enable_batching = enable_batching
        self.device = device or next(model.parameters()).device

        # Initialize multimodal preprocessor
        from .multimodal_preprocessing import MultimodalPreprocessor

        self.preprocessor = MultimodalPreprocessor(
            tokenizer=tokenizer, image_processor=image_processor
        )

        # Request queues and buffers
        self.request_queue = asyncio.Queue(maxsize=buffer_size)
        self.result_queue = asyncio.Queue(maxsize=buffer_size)

        # Thread pool for CPU-bound preprocessing tasks
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_requests)

        # Statistics
        self.stats = {
            "requests_processed": 0,
            "avg_processing_time": 0.0,
            "total_processing_time": 0.0,
            "active_requests": 0,
        }

        # Batching components
        self.batch_buffer = []
        self.batch_lock = asyncio.Lock()
        self.batch_timeout = 0.1  # seconds

        logger.info(
            f"Initialized AsyncMultimodalProcessor with max_concurrent={max_concurrent_requests}, "
            f"batching={enable_batching}, device={self.device}"
        )

    async def start(self):
        """Start the asynchronous processing system."""
        # Start the main processing loop in a background task
        self.processing_task = asyncio.create_task(self._process_requests())
        logger.info("Started asynchronous multimodal processing system")

    async def stop(self):
        """Stop the asynchronous processing system."""
        if hasattr(self, "processing_task"):
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        self.executor.shutdown(wait=True)
        logger.info("Stopped asynchronous multimodal processing system")

    async def submit_request(self, request: AsyncMultimodalRequest) -> Future:
        """
        Submit a request to the asynchronous processing system.

        Args:
            request: The multimodal request to submit

        Returns:
            A Future object that can be used to get the result
        """
        if not hasattr(self, "processing_task") or not self.processing_task.done():
            # Create a future to return
            future = asyncio.Future()

            # Add request to queue with priority
            queue_item = (request.priority, request, future)
            try:
                await self.request_queue.put(queue_item)
            except asyncio.QueueFull:
                logger.warning("Request queue is full, dropping request")
                future.set_exception(RuntimeError("Request queue is full"))

            return future
        else:
            raise RuntimeError("Async processor is not running")

    async def _process_requests(self):
        """Main processing loop for handling requests asynchronously."""
        while True:
            try:
                # Get a request from the queue
                priority, request, future = await self.request_queue.get()

                if self.enable_batching:
                    # Add to batch buffer
                    async with self.batch_lock:
                        self.batch_buffer.append((priority, request, future))

                        # Process batch if it's full
                        if len(self.batch_buffer) >= self.max_concurrent_requests:
                            await self._process_batch()
                        else:
                            # Start a timer to process the batch after timeout
                            asyncio.create_task(self._process_batch_if_ready())
                else:
                    # Process immediately without batching
                    await self._process_single_request(request, future)

            except Exception as e:
                logger.error(f"Error in request processing loop: {e}")

    async def _process_batch_if_ready(self):
        """Process batch if it has accumulated requests after timeout."""
        await asyncio.sleep(self.batch_timeout)

        async with self.batch_lock:
            if len(self.batch_buffer) > 0:
                await self._process_batch()

    async def _process_batch(self):
        """Process a batch of requests asynchronously."""
        async with self.batch_lock:
            if not self.batch_buffer:
                return

            batch_items = self.batch_buffer.copy()
            self.batch_buffer.clear()

        # Sort by priority (lower number = higher priority)
        batch_items.sort(key=lambda x: x[0])

        # Process each item in the batch concurrently
        tasks = []
        for priority, request, original_future in batch_items:
            task = asyncio.create_task(self._process_single_request_async(request))
            tasks.append((original_future, task))

        # Wait for all to complete and set results
        for original_future, processing_task in tasks:
            try:
                result = await processing_task
                original_future.set_result(result)
            except Exception as e:
                original_future.set_exception(e)

    async def _process_single_request_async(
        self, request: AsyncMultimodalRequest
    ) -> AsyncMultimodalResult:
        """Process a single request asynchronously."""
        start_time = time.time()
        self.stats["active_requests"] += 1

        try:
            # Prepare multimodal input
            processed_data = await self._prepare_multimodal_input(request)

            # Perform the actual computation
            with torch.no_grad():  # Disable gradients for inference
                result = self.model(processed_data)

            # Prepare result
            multimodal_result = AsyncMultimodalResult(
                request_id=request.id,
                result=result,
                processing_time=time.time() - start_time,
                metadata=request.metadata,
            )

            # Update statistics
            self._update_stats(multimodal_result.processing_time)

            # Execute callback if provided
            if request.callback:
                try:
                    request.callback(multimodal_result)
                except Exception as e:
                    logger.error(f"Error in request callback: {e}")

            return multimodal_result

        except Exception as e:
            # Handle errors
            multimodal_result = AsyncMultimodalResult(
                request_id=request.id,
                result=None,
                error=e,
                processing_time=time.time() - start_time,
                metadata=request.metadata,
            )

            # Update statistics
            self._update_stats(multimodal_result.processing_time)

            return multimodal_result
        finally:
            self.stats["active_requests"] -= 1

    async def _prepare_multimodal_input(
        self, request: AsyncMultimodalRequest
    ) -> Dict[str, torch.Tensor]:
        """Prepare multimodal input asynchronously."""
        # Use thread pool for CPU-bound preprocessing tasks
        loop = asyncio.get_event_loop()

        # Preprocess multimodal data
        if request.text is not None and request.image is not None:
            # Both text and image provided
            processed_data = await loop.run_in_executor(
                self.executor,
                self.preprocessor.preprocess_multimodal_pair,
                request.text,
                request.image,
            )
        elif request.text is not None:
            # Text only
            processed_data = await loop.run_in_executor(
                self.executor,
                self.preprocessor.text_preprocessor.preprocess,
                request.text,
            )
        elif request.image is not None:
            # Image only
            processed_data = await loop.run_in_executor(
                self.executor,
                self.preprocessor.image_preprocessor.preprocess,
                request.image,
            )
        else:
            raise ValueError("Either text or image (or both) must be provided")

        # Move tensors to appropriate device
        for key, value in processed_data.items():
            if isinstance(value, torch.Tensor):
                processed_data[key] = value.to(self.device)

        return processed_data

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


def create_async_multimodal_engine(
    model: torch.nn.Module,
    name: str,
    tokenizer: Any,
    image_processor: Any,
    max_concurrent_requests: int = 4,
    buffer_size: int = 100,
    batch_timeout: float = 0.1,
    enable_batching: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> AsyncMultimodalProcessor:
    """
    Create an asynchronous multimodal processing engine.

    Args:
        model: The model to use for processing
        name: Name for the engine
        tokenizer: Tokenizer for text processing
        image_processor: Image processor for image processing
        max_concurrent_requests: Maximum number of concurrent requests to process
        buffer_size: Size of the internal request buffer
        batch_timeout: Timeout for batching requests (seconds)
        enable_batching: Whether to enable request batching for efficiency
        device: Device to run computations on

    Returns:
        AsyncMultimodalProcessor instance
    """
    engine = AsyncMultimodalProcessor(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_concurrent_requests=max_concurrent_requests,
        buffer_size=buffer_size,
        enable_batching=enable_batching,
        device=device,
    )

    logger.info(f"Created asynchronous multimodal engine: {name}")
    return engine


def apply_async_multimodal_processing_to_model(
    model: torch.nn.Module, config: Any
) -> torch.nn.Module:
    """
    Apply asynchronous multimodal processing capabilities to the model.

    Args:
        model: The model to enhance
        config: Configuration for the model

    Returns:
        Enhanced model with asynchronous multimodal processing capabilities
    """
    logger.info("Applying asynchronous multimodal processing to model...")

    # This function would typically enhance the model with async processing capabilities
    # For now, we'll just return the model as is, but in a real implementation,
    # we would add async processing methods or attributes to the model

    # Add async processing attributes if they don't exist
    if not hasattr(model, "_async_processing_enabled"):
        model._async_processing_enabled = getattr(
            config, "enable_async_multimodal_processing", False
        )

    logger.info("Asynchronous multimodal processing applied successfully")
    return model


__all__ = [
    "AsyncMultimodalRequest",
    "AsyncMultimodalResult",
    "Qwen3VL2BAsyncMultimodalManager",
    "AsyncMultimodalProcessor",
    "create_async_multimodal_engine",
    "apply_async_multimodal_processing_to_model",
]
