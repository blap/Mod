# Inference-PIO

Inference-PIO is a modular, high-performance inference engine for Large Language Models (LLMs) and Vision-Language Models (VLMs). It features a plugin-based architecture for both hardware backends (CPU, CUDA, OpenCL) and model architectures.

## Key Features

*   **Modular Architecture:** Everything is a plugin. Add new models or hardware support without modifying the core engine.
*   **Custom C/C++ Backend:** High-performance tensor operations implemented in C (CPU) and CUDA/OpenCL (GPU).
*   **Memory Efficiency:**
    *   **Safetensors Context:** Low-memory footprint loading using `mmap` and streaming.
    *   **Unified Memory Manager:** Tracks allocations across devices.
    *   **Flash Attention & Paged Attention:** Optimized attention kernels.
*   **Hybrid Scheduling:** Dynamic offloading between CPU and GPU based on workload and VRAM availability.
*   **Hardware Support:**
    *   **CPU:** Optimized AVX2/AVX512 kernels, OpenMP multithreading.
    *   **NVIDIA GPU:** CUDA kernels with FP16/Int8 support.
    *   **AMD/Intel GPU:** OpenCL kernels via dynamic driver loading.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-org/inference-pio.git
    cd inference-pio
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Build the backend:**
    ```bash
    python3 build_ops.py
    ```

## Usage

### Running a Model

```bash
# Example: Run Qwen3-0.6B
python3 src/inference_pio/models/qwen3_0_6b/benchmarks/benchmark_inference.py
```

### Running Tests

```bash
python3 -m pytest tests/
```

## Documentation & Guides

Learn how to extend Inference-PIO:

*   [**Creating Model Plugins**](docs/GUIDE_MODEL_PLUGINS.md): How to add support for new LLM/VLM architectures.
*   [**Creating CPU Plugins**](docs/GUIDE_CPU_PLUGINS.md): How to implement optimized CPU backends.
*   [**Creating GPU Plugins**](docs/GUIDE_GPU_PLUGINS.md): How to implement CUDA or OpenCL backends.
