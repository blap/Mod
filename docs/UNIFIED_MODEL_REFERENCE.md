# Unified Model Reference for Inference-PIO

## Overview

This document provides a comprehensive reference for all models in the Inference-PIO system, highlighting their unique features, optimizations, and usage patterns within the self-contained plugin architecture.

## Model Comparison

| Model | Type | Parameters | Primary Use Case | Specialized Optimizations |
|-------|------|------------|------------------|-------------------------|
| GLM-4.7 | Text Generation | 4.7B | Advanced reasoning, text understanding | Reasoning-focused attention, memory-efficient processing |
| Qwen3-Coder-30B | Code Generation | 30B | Code completion, generation, understanding | Syntax-aware attention, multi-language processing |
| Qwen3-VL-2B | Vision-Language | 2B | Image understanding, multimodal tasks | Cross-modal attention, vision-language fusion |
| Qwen3-4B-Instruct-2507 | Instruction Following | 4B | Conversational AI, instruction following | Instruction-tuned attention, safety mechanisms |

## Common Architecture Elements

All models in the Inference-PIO system follow the same self-contained plugin architecture:

```
src/inference_pio/models/[model_name]/
├── __init__.py
├── config.py
├── model.py
├── plugin.py
├── attention/
├── benchmarks/
├── cuda_kernels/
├── fused_layers/
├── kv_cache/
├── linear_optimizations/
├── prefix_caching/
├── rotary_embeddings/
├── tensor_parallel/
└── tests/
```

## Model-Specific Implementations

### GLM-4.7 (`glm_4_7`)

**Description**: Advanced reasoning language model with 4.7B parameters.

**Key Features**:
- Transformer-based architecture optimized for reasoning tasks
- Advanced attention mechanisms
- Hardware-specific optimizations
- Comprehensive test suite

**Specialized Optimizations**:
- Reasoning-focused attention patterns
- Memory-efficient processing for complex tasks
- Specialized rotary embeddings for reasoning
- GLM-specific attention optimizations in `plugin/glm47_specific_optimizations.py`

**Usage Example**:
```python
from inference_pio import create_glm_4_7_flash_plugin

plugin = create_glm_4_7_flash_plugin()
plugin.initialize()
plugin.load_model()
result = plugin.infer("Solve this mathematical equation: 2x + 5 = 15")
plugin.cleanup()
```

### Qwen3-Coder-30B (`qwen3_coder_30b`)

**Description**: Code generation and understanding model with 30B parameters.

**Key Features**:
- Specialized for code-related tasks
- Multi-language support
- Advanced context understanding
- Optimized for long contexts

**Specialized Optimizations**:
- Syntax-aware attention mechanisms
- Code-specific KV-cache optimizations
- Multi-language processing optimizations
- Qwen3-specific attention optimizations in `specific_optimizations/`

**Usage Example**:
```python
from inference_pio import create_qwen3_coder_30b_plugin

plugin = create_qwen3_coder_30b_plugin()
plugin.initialize()
plugin.load_model()
result = plugin.generate_text("Write a Python function to calculate factorial")
plugin.cleanup()
```

### Qwen3-VL-2B (`qwen3_vl_2b`)

**Description**: Vision-language multimodal model with 2B parameters.

**Key Features**:
- Multimodal processing (text and images)
- Vision transformer components
- Cross-modal attention mechanisms
- Specialized for visual understanding

**Specialized Optimizations**:
- Cross-modal attention mechanisms
- Vision-language fusion optimizations
- Efficient image processing pipelines
- Multimodal-specific optimizations in various subdirectories

**Usage Example**:
```python
from inference_pio import create_qwen3_vl_2b_instruct_plugin
from PIL import Image

plugin = create_qwen3_vl_2b_instruct_plugin()
plugin.initialize()
plugin.load_model()

# For multimodal input
image_path = "path/to/image.jpg"
result = plugin.infer({
    "text": "Describe this image:",
    "image": image_path
})

plugin.cleanup()
```

### Qwen3-4B-Instruct-2507 (`qwen3_4b_instruct_2507`)

**Description**: Instruction-following language model with 4B parameters.

**Key Features**:
- Fine-tuned for instruction following
- Conversational capabilities
- Safety and alignment features
- Context-aware responses

**Specialized Optimizations**:
- Instruction-following attention patterns
- Safety and alignment optimizations
- Conversational context management
- Qwen3-specific instruction optimizations in `specific_optimizations/`

**Usage Example**:
```python
from inference_pio import create_qwen3_4b_instruct_2507_plugin

plugin = create_qwen3_4b_instruct_2507_plugin()
plugin.initialize()
plugin.load_model()

# Chat completion
messages = [
    {"role": "user", "content": "Explain quantum computing in simple terms"}
]
result = plugin.chat_completion(messages)

plugin.cleanup()
```

## Plugin Interface Compliance

All models implement the standardized plugin interface:

### Core Methods
- `initialize(**kwargs)`: Initialize the plugin with configuration
- `load_model(config=None)`: Load the model with optional configuration
- `infer(data)`: Perform inference on input data
- `cleanup()`: Clean up resources used by the plugin

### Text Model Extensions
- `tokenize(text, **kwargs)`: Tokenize input text
- `detokenize(token_ids, **kwargs)`: Decode token IDs back to text
- `generate_text(prompt, max_new_tokens=512, **kwargs)`: Generate text from prompt
- `chat_completion(messages, max_new_tokens=1024, **kwargs)`: Chat completion interface

## Performance Optimizations

Each model includes state-of-the-art optimizations tailored to its architecture:

### Attention Mechanisms
- FlashAttention 2.0: Memory-efficient attention with reduced computational complexity
- Sparse Attention: Attention with sparse connectivity patterns for long sequences
- Sliding Window Attention: Local attention window for efficient processing of long sequences
- Multi-Query/Grouped-Query Attention: Reduced KV-cache memory usage
- Paged Attention: Memory-efficient attention with paged KV-cache management

### Memory Optimizations
- KV-Cache Compression: Quantization and low-rank compression of KV-cache
- Paged KV-Cache: Memory-efficient KV-cache management with paging
- Prefix Caching: Caching of common prefixes for efficient reuse
- Gradient Checkpointing: Memory-efficient training with recomputation
- Tensor Parallelism: Model parallelism across multiple devices

### Hardware Optimizations
- CUDA Kernels: Custom kernels for NVIDIA GPU acceleration
- Fused Operations: Combined operations to reduce memory transfers
- Mixed Precision: Efficient use of FP16/BF16 for performance
- Tensor Cores: Utilization of NVIDIA Tensor Cores for acceleration

## Testing and Validation

Each model includes:
- Comprehensive unit tests
- Integration tests
- Performance benchmarks
- Accuracy verification tests
- Memory usage tests

Test files are located in the `tests/` subdirectory of each model's directory.

## Configuration Management

Each model has its own configuration class that extends the base configuration with model-specific parameters:

- Hardware optimization settings
- Memory management parameters
- Attention mechanism selection
- Performance tuning parameters
- Model-specific hyperparameters

Configuration files are located at `config.py` in each model's directory.