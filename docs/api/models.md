# Model Reference

## 1. Supported Models

| Model | Type | Params | Optimizations |
|-------|------|--------|---------------|
| **GLM-4.7-Flash** | Text/Reasoning | 4.7B | MoE (4 active), Reasoning Attention |
| **Qwen3-4B-Instruct** | Instruction | 4B | Context Management, Safety |
| **Qwen3-Coder-30B** | Code | 30B | Syntax-Aware Attention, Long Context |
| **Qwen3-VL-2B** | Multimodal | 2B | Vision Fusion, Cross-Attention |
| **Qwen3-0.6B** | Text/Reasoning | 0.6B | Thinking Mode, Lightweight Reasoning |

## 2. Standardized Structure

Every model plugin in `src/inference_pio/models/` follows this layout:

```
src/inference_pio/models/<name>/
├── __init__.py         # Module entry point
├── config.py           # Model-specific config
├── model.py            # Core logic
├── plugin.py           # Interface adapter
├── plugin_manifest.json # Plugin metadata and discovery info
├── architecture/       # Architecture-specific implementations
├── attention/          # Attention mechanisms (Flash, Sparse, etc.)
├── fused_layers/       # Fused layer implementations
├── kv_cache/           # KV cache management and compression
├── mlp/                # MLP implementations
├── rotary_embeddings/  # Rotary embedding implementations
├── specific_optimizations/ # Model-specific optimizations
├── configs/            # Configuration files
├── tests/              # Unit & Integration tests
├── benchmarks/         # Benchmark implementations
└── README.md           # Model-specific documentation
```

## 3. Documentation Standards for Models

Each model plugin must adhere to the project's documentation standards:

### Module Documentation
Each model's `__init__.py` file must include a module-level docstring that:
- Describes the model's purpose and functionality
- Lists the main classes and functions exported
- Mentions the self-contained nature of the model

### Class Documentation
All classes within the model must include:
- A class-level docstring describing the class's purpose
- Constructor documentation with parameter details
- Method documentation following Google style

### Method Documentation
All public methods must include:
- A brief description of the method's purpose
- Complete parameter documentation with type hints
- Return value documentation
- Exception documentation
- Usage examples when beneficial

For detailed standards, see [DOCSTRINGS.md](../standards/DOCSTRINGS.md) and [COMMENTS.md](../standards/COMMENTS.md).

## 4. Implementation Details

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

### Qwen3-0.6B
Ultra-lightweight reasoning model.
*   **Special Feature:** "Thinking Mode" which can be toggled via `/think` command in prompts to activate detailed reasoning chains.
*   **Usage:**
    ```python
    plugin.infer("/think Explain the implications of quantum entanglement")
    ```

## 5. Configuration

Models use specific config classes (e.g., `Qwen3VL2BConfig`) extending `BaseModelConfig`.
Common parameters:
*   `use_flash_attention_2`: bool
*   `load_in_4bit`: bool
*   `max_new_tokens`: int

## 6. Optimizations
All models leverage:
*   **FlashAttention 2.0**
*   **Paged KV Cache** (vLLM style)
*   **Rotary Embeddings (RoPE)**
*   **Tensor Parallelism** (Optional)
