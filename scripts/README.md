# Scripts Directory

This directory contains essential utility scripts for setting up, building, and maintaining the Qwen3-VL model project. These scripts automate common development tasks and ensure proper environment configuration. The scripts are organized into subdirectories based on their functionality.

## Organization

Scripts are categorized into the following subdirectories:

### build/
Contains scripts related to building and compiling the project, especially CUDA extensions:
- `build_cuda_extensions.py` - Builds and compiles SM61-optimized CUDA kernels for the Qwen3-VL model
- `setup_cuda.py` - CUDA kernel setup configuration for SM61 architecture
- `verify_cuda_integration.py` - Comprehensive verification of CUDA integration in the Qwen3-VL model

### env/
Contains scripts for environment setup and management:
- `setup_env.py` - Environment setup script that installs dependencies and sets up the development environment

### utils/
Contains utility scripts for maintenance and miscellaneous tasks:
- `cleanup_cache_dirs.py` - Removes Python cache directories like `__pycache__` and `.pytest_cache` that may have been created before proper `.gitignore` configuration

## Usage

### Environment Setup
To set up the development environment:
```bash
python env/setup_env.py
```

For development environment with additional dependencies:
```bash
python env/setup_env.py --dev
```

### CUDA Kernel Compilation
To build the SM61-optimized CUDA extensions:
```bash
python build/build_cuda_extensions.py
```

This script will:
1. Clean any existing build artifacts
2. Compile the CUDA kernels optimized for SM61 architecture
3. Install the extensions

### CUDA Integration Verification
To verify the CUDA integration is working correctly:
```bash
python build/verify_cuda_integration.py
```

This script performs comprehensive tests including:
- CUDA availability and basic functionality
- CUDA extensions compilation
- CUDA wrapper functionality
- Optimized model components
- Fallback mechanisms
- Model capacity preservation
- Build and setup script functionality

### Cache Cleanup
To remove Python cache directories:
```bash
python utils/cleanup_cache_dirs.py
```

This is particularly useful after updating the `.gitignore` file to ensure no unwanted cache files remain in the repository.

## Important Notes

1. **CUDA Requirements**: The CUDA-related scripts require a compatible NVIDIA GPU and properly installed CUDA toolkit.

2. **Architecture Optimization**: The CUDA kernels are specifically optimized for SM61 architecture (commonly found in NVIDIA GTX 10-series and newer GPUs).

3. **Environment Variables**: Some scripts may require specific environment variables to be set for CUDA development.

4. **Dependencies**: Ensure all required dependencies are installed before running the build scripts.

## Best Practices

1. Always run `env/setup_env.py` first when setting up a new development environment
2. Use `build/verify_cuda_integration.py` after building CUDA extensions to ensure everything is working properly
3. Run `utils/cleanup_cache_dirs.py` periodically to keep the project clean
4. Check CUDA compatibility before attempting to build CUDA extensions