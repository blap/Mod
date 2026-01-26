# Asynchronous Multimodal Processing for Qwen3-VL-2B

This directory contains the implementation and documentation for the asynchronous multimodal processing system specifically designed for the Qwen3-VL-2B model.

## Overview

The asynchronous multimodal processing system enables efficient parallel processing of text and image inputs for the Qwen3-VL-2B vision-language model. This system significantly improves throughput and resource utilization by leveraging asynchronous execution patterns and intelligent batching strategies.

## Key Features

- **Asynchronous Processing**: Handles multiple multimodal requests concurrently
- **Dynamic Batching**: Adjusts batch sizes based on input complexity
- **Cross-Modal Fusion**: Optimized attention mechanisms for vision-language tasks
- **Memory Management**: Intelligent pagination and offloading for multimodal data
- **Performance Optimization**: Reduces latency and increases throughput for multimodal inference

## Components

### Core Classes
- `AsyncMultimodalProcessor`: Handles asynchronous processing of multimodal inputs
- `Qwen3VL2BAsyncMultimodalManager`: Specialized manager for Qwen3-VL-2B model
- `AsyncMultimodalRequest`: Represents an asynchronous multimodal processing request
- `AsyncMultimodalResult`: Represents the result of an asynchronous multimodal processing request

### Configuration Options
- `enable_async_multimodal_processing`: Enable/disable async processing
- `async_max_concurrent_requests`: Maximum number of concurrent requests
- `async_buffer_size`: Size of the internal request buffer
- `async_batch_timeout`: Timeout for batching requests
- `enable_async_batching`: Enable/disable request batching
- `text_weight`: Weight of text complexity in combined complexity calculation
- `image_weight`: Weight of image complexity in combined complexity calculation

## Usage

### Basic Async Processing
```python
from inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin

# Initialize plugin with async processing enabled
plugin = create_qwen3_vl_2b_instruct_plugin()
plugin.initialize(enable_async_multimodal_processing=True)

# Process multimodal input asynchronously
result = await plugin._model.async_process_multimodal_request(
    text="Describe this image",
    image="path/to/image.jpg"
)
```

### Batch Async Processing
```python
# Process multiple multimodal inputs asynchronously
batch_results = await plugin._model.async_process_batch_multimodal_requests([
    {"text": "Describe image 1", "image": "path1.jpg"},
    {"text": "Describe image 2", "image": "path2.jpg"}
])
```

## Performance Benefits

- **Improved Throughput**: Up to 2-4x improvement in requests per second
- **Reduced Latency**: Better resource utilization leads to lower response times
- **Memory Efficiency**: Intelligent pagination optimizes memory usage
- **Scalability**: Dynamic batch sizing adapts to input complexity

## Integration

The async multimodal processing system integrates seamlessly with:
- Existing attention optimization systems
- CUDA kernels and hardware-specific optimizations
- Memory management and pagination systems
- Model surgery and quantization techniques
- Streaming computation framework

## Testing

Run the tests to verify the implementation:
```bash
python -m pytest src/inference_pio/models/qwen3_vl_2b/tests/test_async_multimodal_processing.py
```

## Benchmarking

Benchmark the async processing performance:
```bash
python src/inference_pio/models/qwen3_vl_2b/benchmarks/benchmark_async_multimodal_processing.py
```