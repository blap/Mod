# Inference-PIO Models Summary

This document summarizes the implemented models and plugins in the Inference-PIO framework. All models are **fully self-contained**, **dependency-free** (no PyTorch/NumPy/Transformers), and utilize the custom **C/CUDA backend** (`libtensor_ops`) for all operations.

## Implemented Models

All models implement the standardized `TextModelPluginInterface` with uniform `tokenize`, `detokenize`, and `infer_batch` methods.

### 1. Qwen3-Coder-Next
- **Type**: Hybrid (DeltaNet + Attention) with MoE (Mixture-of-Experts)
- **Plugin**: `src/inference_pio/models/qwen3_coder_next/`
- **Key Features**:
    - Efficient MoE routing using C kernels (`gather_by_value`, `scatter_add_by_index`).
    - Hybrid block pattern (Attention + DeltaNet).
    - Fully implemented in `model.py` using `backend.Tensor`.
- **Status**: Complete with Tests & Benchmarks.

### 2. GLM-4.7-Flash
- **Type**: Large Language Model (Attention-based)
- **Plugin**: `src/inference_pio/models/glm_4_7_flash/`
- **Key Features**:
    - Multi-Head Attention with correct reshaping and `scaled_dot_product_attention`.
    - `swiglu`, `rope`, `rms_norm` C kernels.
- **Status**: Complete with Tests & Benchmarks.

### 3. Qwen3-VL-2B
- **Type**: Multimodal (Vision-Language)
- **Plugin**: `src/inference_pio/models/qwen3_vl_2b/`
- **Key Features**:
    - Vision Transformer Encoder (resizing via C `image_resize_bilinear`).
    - Multimodal Projector and Text Decoder.
    - Standardized text interface (image input via tuple).
- **Status**: Complete with Tests & Benchmarks.

### 4. Qwen3-4B-Instruct-2507
- **Type**: Causal Language Model
- **Plugin**: `src/inference_pio/models/qwen3_4b_instruct_2507/`
- **Key Features**:
    - Optimized generation loop.
    - Fixed RoPE/Architecture.
    - Standardized Batch Inference.
- **Status**: Complete with Tests & Benchmarks.

### 5. Qwen3-0.6B
- **Type**: Small Language Model
- **Plugin**: `src/inference_pio/models/qwen3_0_6b/`
- **Key Features**:
    - Compact architecture, optimized for speed.
    - Fully strictly typed and standardized.
- **Status**: Complete with Tests & Benchmarks.

### 6. Qwen3-Coder-30B
- **Type**: Large Language Model (Code Specialized)
- **Plugin**: `src/inference_pio/models/qwen3_coder_30b/`
- **Key Features**:
    - Deep architecture handling (60+ layers).
    - KV Cache optimization.
    - Standardized `generate_text` flow.
- **Status**: Complete with Tests & Benchmarks.

## Optimizations & Backend

### Custom C/CUDA Backend (`libtensor_ops`)
- **No Stubs**: All operations are real C/C++ implementations.
- **Primitives**: `matmul`, `linear`, `rope`, `swiglu`, `rms_norm`, `conv2d`, `scaled_dot_product_attention`, `gather`, `scatter_add`, `moe` ops.
- **Memory**: Pinned memory support, Custom Allocator, `safetensors` loading via `mmap`.

### Standardized Plugin Interface
All plugins strictly adhere to `TextModelPluginInterface`:
- `tokenize(text) -> List[float]`: Robust tokenization with fallback.
- `detokenize(ids) -> str`: Robust decoding.
- `infer_batch(requests) -> List[str]`: Integrated with `BatchManager`.
- `generate_text(prompt) -> str`: Standard generation entry point.

### Scheduling & Batching
- **BatchManager**: Implements Serial Batching (FCFS) for all models.
- **HybridScheduler**: Supports CUDA Streams and async offloading/prefetching.

## Directory Structure
Each model directory follows a strict self-contained structure:
```
models/<model_name>/
├── __init__.py
├── config.py           # Configuration classes
├── model.py            # Model implementation (Module, Tensor ops)
├── plugin.py           # Plugin Interface (Standardized)
├── architecture.py     # (Optional) Architecture specifics
├── README.md           # Model documentation
├── benchmarks/         # Performance scripts
│   └── benchmark_inference.py
└── tests/              # Unit tests
    └── unit/
        ├── test_model.py
        └── test_plugin.py
```

## Usage
- **Tests**: Use `python -m unittest discover tests` or model-specific tests.
- **Verification**: Run `python verify_textual_standardization.py` to ensure compliance.
