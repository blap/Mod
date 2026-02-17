# Inference-PIO

Inference-PIO is a modular, high-performance inference engine for Large Language Models (LLMs) and Vision-Language Models (VLMs). It features a plugin-based architecture for both hardware backends (CPU, CUDA, OpenCL) and model architectures, implemented via a custom C/C++ backend.

## Key Features

*   **Modular Architecture:** Everything is a plugin. Add new models or hardware support without modifying the core engine.
*   **Custom C/C++ Backend:** High-performance tensor operations implemented in C (CPU) and CUDA/OpenCL (GPU).
*   **Cross-Platform Build System:**
    *   **Universal Build:** `build_ops.py` auto-detects host OS and available compilers (GCC, MSVC, MinGW, WSL).
    *   **Cross-Compilation:** Build Windows DLLs from Linux (via MinGW) and Linux `.so` from Windows (via WSL).
    *   **Wheel Support:** Package everything into a `.whl` for easy distribution.
*   **Memory Efficiency:**
    *   **Safetensors Context:** Low-memory footprint loading using `mmap` and streaming.
    *   **Unified Memory Manager:** Tracks allocations across devices.
    *   **Flash Attention & Paged Attention:** Optimized attention kernels.
*   **Hybrid Scheduling:** Dynamic offloading between CPU and GPU based on workload and VRAM availability.
*   **Hardware Support:**
    *   **CPU:** Optimized AVX2/AVX512 kernels, OpenMP multithreading.
    *   **NVIDIA GPU:** CUDA kernels with FP16/Int8 support.
    *   **AMD/Intel GPU:** OpenCL kernels via dynamic driver loading.

## Installation & Build

### 1. Clone the repository
```bash
git clone https://github.com/your-org/inference-pio.git
cd inference-pio
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Build the Backend

Inference-PIO uses a custom build script `build_ops.py` that handles compilation for CPU, CUDA, and OpenCL backends.

#### Native Build (Linux or Windows)
Run the following command to build for your current OS:
```bash
python3 build_ops.py
```
*   **Linux:** Requires `gcc`. builds `.so` files.
*   **Windows:** Requires `cl` (MSVC) or `gcc` (MinGW). Builds `.dll` files.

#### Cross-Compilation

**Building Windows Artifacts on Linux:**
Install MinGW-w64 (`sudo apt install mingw-w64`) and run:
```bash
python3 build_ops.py
```
The script will automatically detect `x86_64-w64-mingw32-gcc` and build Windows DLLs alongside native Linux binaries.

**Building Linux Artifacts on Windows:**
Ensure you have WSL (Windows Subsystem for Linux) installed and `gcc` available inside it (`wsl gcc --version`). Run:
```bash
python build_ops.py
```
The script will detect `wsl` and invoke GCC inside the Linux environment to build `.so` files, saving them to the Windows filesystem.

### 4. Create a Wheel Package
To create a distributable Python wheel containing all compiled artifacts:
```bash
pip install setuptools wheel
python3 setup.py bdist_wheel
```
The resulting `.whl` file will be in `dist/` and can be installed via `pip install dist/inference_pio-*.whl`.

## Usage

### Running a Model

```bash
# Example: Run Qwen3-0.6B Benchmark
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
