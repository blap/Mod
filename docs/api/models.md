# Model Reference

## 1. Supported Models

| Model | Type | Params | Optimizations |
|-------|------|--------|---------------|
| **GLM-4.7-Flash** | Text/Reasoning | 4.7B | MoE (4 active), Reasoning Attention |
| **Qwen3-4B-Instruct** | Instruction | 4B | Context Management, Safety |
| **Qwen3-Coder-30B** | Code | 30B | Syntax-Aware Attention, Long Context |
| **Qwen3-VL-2B** | Multimodal | 2B | Vision Fusion, Cross-Attention |

## 2. Standardized Structure

Every model plugin in `src/inference_pio/models/` follows this layout:

```
models/<name>/
├── config.py           # Model-specific config
├── model.py            # Core logic
├── plugin.py           # Interface adapter
├── attention/          # Attention kernels (Flash, Sparse)
├── kv_cache/           # Compression & Paging
├── fused_layers/       # Optimized implementations
└── tests/              # Unit & Integration tests
```

## 3. Implementation Details

### GLM-4.7-Flash
Optimized for complex reasoning. Uses a mixture-of-experts architecture.
*   **Special Feature:** `ReasoningAttention` for improved chain-of-thought.
*   **Usage:**
    ```python
    plugin.infer("Solve 2x + 5 = 15")
    ```

### Qwen3-4B-Instruct-2507
Fine-tuned for chat and instruction following.
*   **Special Feature:** Safety alignment and conversational context caching.
*   **Usage:**
    ```python
    plugin.chat_completion([{"role": "user", "content": "Hello"}])
    ```

### Qwen3-Coder-30B
A large model for code generation.
*   **Special Feature:** Multi-language syntax awareness and long-context optimizations for file-level coding.
*   **Usage:**
    ```python
    plugin.generate_text("def fibonacci(n):")
    ```

### Qwen3-VL-2B
Vision-Language model.
*   **Special Feature:** Single-pipeline image resizing and tokenization.
*   **Usage:**
    ```python
    plugin.infer({"text": "Describe image", "image": "path/to/img.jpg"})
    ```

## 4. Configuration

Models use specific config classes (e.g., `Qwen3VL2BConfig`) extending `BaseModelConfig`.
Common parameters:
*   `use_flash_attention_2`: bool
*   `load_in_4bit`: bool
*   `max_new_tokens`: int

## 5. Optimizations
All models leverage:
*   **FlashAttention 2.0**
*   **Paged KV Cache** (vLLM style)
*   **Rotary Embeddings (RoPE)**
*   **Tensor Parallelism** (Optional)
