# Qwen3-Coder-Next Plugin

This plugin implements the Qwen3-Coder-Next model using the custom C/CUDA backend of Inference-PIO.

## Features
- **Dependency-Free**: Uses only `backend.Tensor` and C/CUDA kernels (no PyTorch/NumPy).
- **Hybrid Architecture**: Combines DeltaNet and Attention layers.
- **MoE Routing**: Implements Mixture-of-Experts with `topk`, `gather`, and `scatter_add` primitives.
- **Self-Contained**: Includes configuration, model implementation, tests, and benchmarks.

## Usage
```python
from src.inference_pio.models.qwen3_coder_next.plugin import create_qwen3_coder_next_plugin
plugin = create_qwen3_coder_next_plugin()
plugin.initialize()
# Assume tensor_input is prepared
# output = plugin.infer(tensor_input)
```

## Testing
Run unit tests:
```bash
python -m unittest src/inference_pio/models/qwen3_coder_next/tests/unit/test_model.py
```

## Benchmarks
Run benchmark script:
```bash
python src/inference_pio/models/qwen3_coder_next/benchmarks/benchmark_inference.py
```
