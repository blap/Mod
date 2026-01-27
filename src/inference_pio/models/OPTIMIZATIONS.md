# Optimization Standards & Implementation Floor

This document defines the mandatory set of optimizations ("The Floor") that every model in this repository must implement. The goal is to ensure consistent, high-performance inference across all supported architectures.

## 1. Categorization

Models are divided into two primary categories:
*   **Text Models:** Pure language models (e.g., Qwen3-4B-Instruct, Qwen3-Coder-30B, GLM-4.7-Flash).
*   **Multimodal Models:** Models processing vision/audio + text (e.g., Qwen3-VL-2B).

## 2. Text Model Floor (Mandatory)

All text-based models must implement the following:

### A. Memory Management
*   **Paged KV Cache:**
    *   Must use a vLLM-style block table approach to manage Key-Value cache memory.
    *   No contiguous memory allocation for future tokens.
    *   Must support dynamic block allocation/deallocation.
    *   *Implementation:* Local `kv_cache/paged_kv_cache.py` module in each model.

### B. Throughput & Latency
*   **Continuous Batching:**
    *   Must support "Iteration Level Scheduling" (Orca-style).
    *   New requests can join the running batch at any step.
    *   Must include a local Scheduler to manage request queues and block tables.
*   **Flash Attention 2 / SDPA:**
    *   Must use `torch.nn.functional.scaled_dot_product_attention` or optimized custom kernels.
    *   Must support Paged Attention variants of the kernels.

### C. Compute Efficiency
*   **Fused Kernels:**
    *   **RoPE (Rotary Positional Embeddings):** Must use cached cos/sin tables.
    *   **RMSNorm / LayerNorm:** Must use fused kernels (Triton/CUDA).
    *   **Fused MLP (SwiGLU):** The Up-projection, Gate-projection, and Down-projection should be optimized, preferably fusing the activation (SwiGLU) with the matrix multiplication where possible.

## 3. Multimodal Model Floor (Mandatory)

Multimodal models must implement **all Text Model optimizations** (for their language backbone) plus:

### A. Vision/Audio Specific
*   **Tensor Pagination:**
    *   Large input tensors (e.g., high-res images, video frames) must be pageable to host RAM/Disk to avoid VRAM OOM.
    *   *Implementation:* `specific_optimizations/` or `visual_resource_compression/`.
*   **Resizing & Projection Optimization:**
    *   Avoid redundant interpolation. Combine resizing and normalization steps where possible.

## 4. Future Roadmap (Planned)

The following optimizations are planned for future inclusion in the floor:

*   **Speculative Decoding:**
    *   Using smaller draft models (e.g., Qwen3-VL-2B drafting for Qwen3-Coder-30B) to accelerate generation.
*   **FP8 / INT8 Quantization:**
    *   Standardizing W8A16 or W8A8 inference for 30B+ parameter models.
*   **Thinking / Long-Context Optimization:**
    *   Specific kernel fusion for "Chain of Thought" patterns (extremely long output sequences).
