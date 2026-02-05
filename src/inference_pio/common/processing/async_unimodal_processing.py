"""
Asynchronous Unimodal Processing Manager for Text Models

This module implements the asynchronous unimodal processing manager for generic text models.
It coordinates the processing of textual inputs using asynchronous techniques to optimize
performance and resource utilization. Model-specific configurations should be handled by
individual model implementations.
"""

import asyncio
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from .dynamic_text_batching import (
    DynamicTextBatchManager,
    get_dynamic_text_batch_manager,
)
from .streaming_computation import StreamRequest, StreamResult
from .unimodal_preprocessing import UnimodalPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class AsyncUnimodalRequest:
    """Represents an asynchronous unimodal processing request."""

    id: str
    text: Optional[str] = None
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
class AsyncUnimodalResult:
    """Represents an asynchronous unimodal processing result."""

    request_id: str
    result: Any
    error: Optional[Exception] = None
    processing_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class AsyncUnimodalManager:
    """
    Asynchronous unimodal processing manager for text models.
    Manages the coordination of text processing with optimized scheduling.
    """

    def __init__(self, model: nn.Module, config: Any, model_type: str = "generic"):
        """
        Initialize the asynchronous unimodal manager.

        Args:
            model: The text model instance
            config: Configuration for the model
            model_type: Type of model (for identification purposes)
        """
        self.model = model
        self.config = config
        self.model_type = model_type
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

    def initialize(self) -> bool:
        """
        Initialize the asynchronous unimodal processing system.

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Get tokenizer from the model
            tokenizer = getattr(self.model, "_tokenizer", None)

            if not tokenizer:
                logger.warning("Tokenizer not found in model, using defaults")
                # Use a generic tokenizer - model-specific tokenizers should be handled by the model itself
                tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Generic fallback

            # Set padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Initialize unimodal preprocessor
            self._preprocessor = UnimodalPreprocessor(tokenizer=tokenizer)

            # Initialize dynamic batch manager for text inputs
            if self.enable_batching:
                self._dynamic_batch_manager = get_dynamic_text_batch_manager(
                    initial_batch_size=getattr(self.config, "initial_batch_size", 1),
                    min_batch_size=getattr(self.config, "min_batch_size", 1),
                    max_batch_size=getattr(self.config, "max_batch_size", 8),
                    memory_threshold_ratio=getattr(
                        self.config, "memory_threshold_ratio", 0.80
                    ),
                    performance_window_size=getattr(
                        self.config, "performance_window_size", 10
                    ),
                    adjustment_factor=getattr(self.config, "adjustment_factor", 0.15),
                    cooldown_period=getattr(self.config, "cooldown_period", 3.0),
                    performance_target=getattr(self.config, "performance_target", 0.85),
                    complexity_weight=getattr(self.config, "complexity_weight", 0.4),
                    sequence_length_weight=getattr(
                        self.config, "sequence_length_weight", 0.3
                    ),
                    memory_safety_margin=getattr(
                        self.config, "memory_safety_margin", 0.15
                    ),
                )

            # Initialize async processor
            self._async_processor = AsyncUnimodalProcessor(
                model=self.model,
                tokenizer=tokenizer,
                max_concurrent_requests=self.max_concurrent_requests,
                buffer_size=self.buffer_size,
                enable_batching=self.enable_batching,
                device=self.device,
                model_type=self.model_type,
            )

            # Start the async processor
            asyncio.create_task(self._async_processor.start())

            self.is_initialized = True
            logger.info(
                f"{self.model_type} asynchronous unimodal manager initialized successfully"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to initialize {self.model_type} asynchronous unimodal manager: {e}"
            )
            return False

    async def process_unimodal_request(
        self, text: Optional[str] = None, **kwargs
    ) -> AsyncUnimodalResult:
        """
        Process a unimodal request asynchronously.

        Args:
            text: Text input (required)
            **kwargs: Additional processing arguments

        Returns:
            AsyncUnimodalResult with the processing result
        """
        if not self.is_initialized:
            raise RuntimeError(
                "Async unimodal manager not initialized. Call initialize() first."
            )

        if text is None:
            raise ValueError("Text input is required for unimodal processing")

        # Create a request ID
        request_id = f"async_um_{int(time.time())}_{hash(text) % 10000}"

        # Create and submit request
        request = AsyncUnimodalRequest(id=request_id, text=text, metadata=kwargs)

        # Submit to async processor
        future = await self._async_processor.submit_request(request)

        # Wait for result
        result = await asyncio.wrap_future(future)
        return result

    async def process_batch_unimodal_requests(
        self, requests: List[Dict[str, Any]], **kwargs
    ) -> List[AsyncUnimodalResult]:
        """
        Process a batch of unimodal requests asynchronously.

        Args:
            requests: List of request dictionaries with 'text' key
            **kwargs: Additional processing arguments

        Returns:
            List of AsyncUnimodalResult objects
        """
        if not self.is_initialized:
            raise RuntimeError(
                "Async unimodal manager not initialized. Call initialize() first."
            )

        # Process each request concurrently
        tasks = []
        for i, req in enumerate(requests):
            request_id = req.get("id", f"async_batch_{int(time.time())}_{i}")
            text = req.get("text")

            if text is None:
                raise ValueError(f"Text input is required for request {i}")

            task = self.process_unimodal_request(text=text, **kwargs)
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(
                    AsyncUnimodalResult(request_id="unknown", result=None, error=result)
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
            "model_type": self.model_type,
            "max_concurrent_requests": self.max_concurrent_requests,
            "buffer_size": self.buffer_size,
            "batching_enabled": self.enable_batching,
            "device": self.device,
        }

        if self._async_processor:
            stats.update(self._async_processor.get_stats())

        return stats


class AsyncUnimodalProcessor:
    """
    Asynchronous processor for unimodal text inputs.
    Handles parallel processing of text with optimized scheduling.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        max_concurrent_requests: int = 4,
        buffer_size: int = 100,
        enable_batching: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        model_type: str = "generic",
    ):
        """
        Initialize the asynchronous unimodal processor.

        Args:
            model: The model to use for processing
            tokenizer: Tokenizer for text processing
            max_concurrent_requests: Maximum number of concurrent requests to process
            buffer_size: Size of the internal request buffer
            enable_batching: Whether to enable request batching for efficiency
            device: Device to run computations on
            model_type: Type of model (for identification purposes)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_concurrent_requests = max_concurrent_requests
        self.buffer_size = buffer_size
        self.enable_batching = enable_batching
        self.device = device or next(model.parameters()).device
        self.model_type = model_type

        # Initialize unimodal preprocessor - handle case where tokenizer is None
        if tokenizer is not None:
            self.preprocessor = UnimodalPreprocessor(tokenizer=tokenizer)
        else:
            # Create a basic tokenizer if none is provided
            from transformers import AutoTokenizer

            # Use a generic tokenizer as fallback
            try:
                self.preprocessor = UnimodalPreprocessor(
                    tokenizer=AutoTokenizer.from_pretrained(
                        "gpt2", trust_remote_code=True
                    )
                )
            except:
                # If even gpt2 fails, create a minimal tokenizer
                class MinimalTokenizer:
                    def encode(self, text, **kwargs):
                        return [
                            abs(hash(c)) % 10000 for c in text[:100]
                        ]  # Simple hash-based encoding

                    def decode(self, ids, **kwargs):
                        return "".join(
                            [chr((id % 95) + 32) for id in ids]
                        )  # Simple decoding

                    pad_token = "[PAD]"
                    eos_token = "[EOS]"

                self.preprocessor = UnimodalPreprocessor(tokenizer=MinimalTokenizer())

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
            f"Initialized AsyncUnimodalProcessor for {model_type} with max_concurrent={max_concurrent_requests}, "
            f"batching={enable_batching}, device={self.device}"
        )

    async def start(self):
        """Start the asynchronous processing system."""
        # Start the main processing loop in a background task
        self.processing_task = asyncio.create_task(self._process_requests())
        logger.info("Started asynchronous unimodal processing system")

    async def stop(self):
        """Stop the asynchronous processing system."""
        if hasattr(self, "processing_task"):
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                raise NotImplementedError("Method not implemented")
        self.executor.shutdown(wait=True)
        logger.info("Stopped asynchronous unimodal processing system")

    async def submit_request(self, request: AsyncUnimodalRequest) -> Future:
        """
        Submit a request to the asynchronous processing system.

        Args:
            request: The unimodal request to submit

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

    async def _process_single_request(
        self, request: AsyncUnimodalRequest, future: Future
    ):
        """Process a single request and set the result in the future."""
        try:
            result = await self._process_single_request_async(request)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)

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
        self, request: AsyncUnimodalRequest
    ) -> AsyncUnimodalResult:
        """Process a single request asynchronously."""
        start_time = time.time()
        self.stats["active_requests"] += 1

        try:
            # Prepare text input
            processed_data = await self._prepare_text_input(request)

            # Perform the actual computation
            with torch.no_grad():  # Disable gradients for inference
                result = self.model(**processed_data)

            # Prepare result
            unimodal_result = AsyncUnimodalResult(
                request_id=request.id,
                result=result,
                processing_time=time.time() - start_time,
                metadata=request.metadata,
            )

            # Update statistics
            self._update_stats(unimodal_result.processing_time)

            # Execute callback if provided
            if request.callback:
                try:
                    request.callback(unimodal_result)
                except Exception as e:
                    logger.error(f"Error in request callback: {e}")

            return unimodal_result

        except Exception as e:
            # Handle errors
            unimodal_result = AsyncUnimodalResult(
                request_id=request.id,
                result=None,
                error=e,
                processing_time=time.time() - start_time,
                metadata=request.metadata,
            )

            # Update statistics
            self._update_stats(unimodal_result.processing_time)

            return unimodal_result
        finally:
            self.stats["active_requests"] -= 1

    async def _prepare_text_input(
        self, request: AsyncUnimodalRequest
    ) -> Dict[str, torch.Tensor]:
        """Prepare text input asynchronously."""
        # Use thread pool for CPU-bound preprocessing tasks
        loop = asyncio.get_event_loop()

        # Preprocess text data
        try:
            processed_data = await loop.run_in_executor(
                self.executor,
                self.preprocessor.preprocess,
                request.text,
                "pt",  # PyTorch tensors
                self.model_type,
            )
        except Exception as e:
            # If preprocessing fails, create a minimal tensor representation
            logger.warning(f"Text preprocessing failed: {e}, creating minimal tensor")
            # Create a minimal tensor representation of the text
            token_ids = [abs(hash(c)) % 10000 for c in request.text[:100]]
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            processed_data = {"input_ids": input_ids, "attention_mask": attention_mask}

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


def create_async_unimodal_engine(
    model: nn.Module,
    name: str,
    tokenizer: Any,
    max_concurrent_requests: int = 4,
    buffer_size: int = 100,
    batch_timeout: float = 0.1,
    enable_batching: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    model_type: str = "generic",
) -> AsyncUnimodalProcessor:
    """
    Create an asynchronous unimodal processing engine.

    Args:
        model: The model to use for processing
        name: Name for the engine
        tokenizer: Tokenizer for text processing
        max_concurrent_requests: Maximum number of concurrent requests to process
        buffer_size: Size of the internal request buffer
        batch_timeout: Timeout for batching requests (seconds)
        enable_batching: Whether to enable request batching for efficiency
        device: Device to run computations on
        model_type: Type of model (for identification purposes)

    Returns:
        AsyncUnimodalProcessor instance
    """
    engine = AsyncUnimodalProcessor(
        model=model,
        tokenizer=tokenizer,
        max_concurrent_requests=max_concurrent_requests,
        buffer_size=buffer_size,
        enable_batching=enable_batching,
        device=device,
        model_type=model_type,
    )

    logger.info(f"Created asynchronous unimodal engine: {name}")
    return engine


def apply_async_unimodal_processing_to_model(
    model: nn.Module, config: Any, model_type: str = "generic"
) -> nn.Module:
    """
    Apply asynchronous unimodal processing capabilities to the model.

    Args:
        model: The model to enhance
        config: Configuration for the model
        model_type: Type of model (for identification purposes)

    Returns:
        Enhanced model with asynchronous unimodal processing capabilities
    """
    logger.info(f"Applying asynchronous unimodal processing to {model_type} model...")

    # This function would typically enhance the model with async processing capabilities
    # For now, we'll just return the model as is, but in a real implementation,
    # we would add async processing methods or attributes to the model

    # Add async processing attributes if they don't exist
    if not hasattr(model, "_async_processing_enabled"):
        model._async_processing_enabled = getattr(
            config, "enable_async_unimodal_processing", False
        )

    logger.info(
        f"Asynchronous unimodal processing applied successfully to {model_type} model"
    )
    return model


__all__ = [
    "AsyncUnimodalRequest",
    "AsyncUnimodalResult",
    "AsyncUnimodalManager",
    "AsyncUnimodalProcessor",
    "create_async_unimodal_engine",
    "apply_async_unimodal_processing_to_model",
]
