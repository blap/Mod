# Qwen3-0.6B

**Qwen3-0.6B** is the latest generation of large language models in the Qwen series, optimized for efficiency and reasoning. This plugin integrates the model into the Inference-PIO system with advanced optimizations for "Thinking Mode" and hardware efficiency.

## Model Highlights

*   **Dual Mode:** Seamless switching between **Thinking Mode** (complex reasoning, math, coding) and **Non-Thinking Mode** (efficient dialogue).
*   **Parameters:** 0.6B (0.44B Non-Embedding).
*   **Architecture:** 28 Layers, 16 Attention Heads (GQA), 32k Context Length.
*   **Language Support:** 100+ languages.

## Usage

### Basic Usage

```python
from src.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin

plugin = create_qwen3_0_6b_plugin()
response = plugin.generate_text("Explain quantum entanglement.")
print(response)
```

### Thinking Mode Switching

You can dynamically switch modes using prompt tags:

*   **Default:** Thinking Mode is enabled by default (`enable_thinking=True` in config).
*   **Disable Thinking:** Add `/no_think` to the prompt.
*   **Force Thinking:** Add `/think` to the prompt.

```python
# Efficient mode (No Thinking)
response = plugin.generate_text("Hello, how are you? /no_think")

# Reasoning mode (Thinking)
response = plugin.generate_text("Solve this complex math problem... /think")
```

## Optimizations

This implementation includes the mandatory **Optimization Floor** plus specific enhancements for reasoning models:

### Standard Optimizations
*   **Flash Attention 2 / SDPA:** Fast attention kernels.
*   **Fused Kernels:** RoPE, RMSNorm, SwiGLU MLP.
*   **Paged KV Cache:** Efficient memory management.
*   **Continuous Batching:** High-throughput scheduling.

### Thinking Mode Optimizations
*   **Thought-Aware KV Cache Compression:** Compresses the KV cache of the `<think>...</think>` segment after reasoning concludes to free up VRAM for the final response.
*   **Dynamic Repetition Penalty:** Applies aggressive penalties (1.5) specifically during thought generation to prevent loops, relaxing them for the final answer.
*   **Long-Sequence RoPE Scaling:** Uses `float32` precision for rotary embeddings to maintain numerical stability in long chains of thought (>8k tokens).

## Citation

```bibtex
@misc{qwen3technicalreport,
      title={Qwen3 Technical Report},
      author={Qwen Team},
      year={2025},
      eprint={2505.09388},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.09388},
}
```
