# Inference-PIO Models Summary

This document summarizes the implemented models and plugins in the Inference-PIO framework. All models are self-contained, dependency-free (no PyTorch/NumPy), and utilize the custom C/CUDA backend (`libtensor_ops`).

## Implemented Models

### 1. Qwen3-Coder-Next
- **Type**: Hybrid (DeltaNet + Attention) with MoE (Mixture-of-Experts)
- **Plugin**: `src/inference_pio/models/qwen3_coder_next/`
- **Key Features**: Efficient MoE routing using `gather_by_value`/`scatter_add_by_index`, hybrid block pattern.
- **Status**: Complete with Tests & Benchmarks.

### 2. GLM-4.7-Flash
- **Type**: Large Language Model (Attention-based)
- **Plugin**: `src/inference_pio/models/glm_4_7_flash/`
- **Key Features**: Multi-Head Attention with correct reshaping, RoPE, RMSNorm.
- **Status**: Complete with Tests & Benchmarks.

### 3. Qwen3-VL-2B
- **Type**: Multimodal (Vision-Language)
- **Plugin**: `src/inference_pio/models/qwen3_vl_2b/`
- **Key Features**: Vision Transformer Encoder, Text Decoder, Multimodal Projector.
- **Status**: Complete with Tests & Benchmarks.

### 4. Qwen3-4B-Instruct-2507
- **Type**: Causal Language Model
- **Plugin**: `src/inference_pio/models/qwen3_4b_instruct_2507/`
- **Key Features**: Optimized generation loop, fixed RoPE/Architecture.
- **Status**: Complete with Tests & Benchmarks.

### 5. Qwen3-0.6B
- **Type**: Small Language Model
- **Plugin**: `src/inference_pio/models/qwen3_0_6b/`
- **Key Features**: Compact architecture, optimized for speed.
- **Status**: Existing implementation.

### 6. Qwen3-Coder-30B
- **Type**: Large Language Model (Code Specialized)
- **Plugin**: `src/inference_pio/models/qwen3_coder_30b/`
- **Key Features**: Deep architecture, KV Cache, Dynamic Offloading support.
- **Status**: Complete with Tests & Benchmarks.

## Dynamic Offloading & Hybrid Scheduling
All implemented models now support **Dynamic Offloading** via `HybridScheduler`. This allows layers to be automatically migrated between CPU and GPU during inference based on memory pressure, enabling execution of large models on consumer hardware.

- **Hook**: `scheduler.check_migration_policy(layer_idx, layer)` inside `forward` loop.
- **Mechanism**: Detects memory constraints and proactively moves upcoming layers to GPU while evicting used layers to CPU if necessary.

## Directory Structure
Each model directory follows a self-contained structure:
```
models/<model_name>/
├── __init__.py
├── config.py           # Configuration classes
├── model.py            # Model implementation (Module, Tensor ops)
├── plugin.py           # Plugin Interface
├── architecture.py     # (Optional) Architecture specifics if separated
├── README.md           # Model documentation
├── benchmarks/         # Performance scripts
│   └── benchmark_inference.py
└── tests/              # Unit tests
    └── unit/
        ├── test_model.py
        └── test_plugin.py
```

## Testing & Benchmarking
- **Tests**: Use `python -m unittest ...` to run unit tests in `tests/unit/`.
- **Benchmarks**: Execute `benchmark_inference.py` to measure tokens/sec and latency.
