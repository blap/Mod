# Benchmark Discovery System Implementation

## Overview

This implementation provides a comprehensive benchmark discovery mechanism that automatically finds and runs benchmark functions across the project's standardized benchmark structure. The system replaces hardcoded imports with a flexible, automated discovery process.

## Components

### 1. Central Discovery Module (`benchmarks/discovery.py`)
- **Purpose**: Main discovery engine that scans the entire project for benchmark functions
- **Features**:
  - Discovers benchmark functions in standardized directory structure (unit/, integration/, performance/)
  - Identifies functions that start with 'run_' or 'benchmark_'
  - Categorizes benchmarks by type and model
  - Provides safe execution with error handling
  - Generates comprehensive reports

### 2. Updated Main Benchmark Runner (`benchmarks/run_all_inference_benchmarks.py`)
- **Purpose**: Updated to use the new discovery mechanism instead of hardcoded imports
- **Features**:
  - Automatically discovers all performance benchmarks
  - Runs benchmarks for all models without manual specification
  - Maintains backward compatibility with existing result aggregation

### 3. Model-Specific Discovery Modules
Created specialized discovery modules for each model:
- `src/inference_pio/models/glm_4_7/benchmarks/discovery.py`
- `src/inference_pio/models/qwen3_4b_instruct_2507/benchmarks/discovery.py`
- `src/inference_pio/models/qwen3_coder_30b/benchmarks/discovery.py`
- `src/inference_pio/models/qwen3_vl_2b/benchmarks/discovery.py`
- `src/inference_pio/plugin_system/benchmarks/discovery.py`

Each provides:
- Model-specific benchmark discovery
- Category filtering capabilities
- Dedicated result saving

## Key Features

### Automatic Discovery
- Scans all benchmark directories (unit, integration, performance)
- Finds functions with 'run_' or 'benchmark_' prefixes
- Works with the existing standardized directory structure

### Flexible Execution
- Can run all benchmarks or filter by category/model
- Safe execution with comprehensive error handling
- Detailed reporting and result aggregation

### Standardized Structure Support
- Respects existing directory hierarchy
- Works with unit/, integration/, performance/ subdirectories
- Maintains compatibility with existing benchmark implementations

## Benefits

1. **Eliminates Hardcoded Imports**: No need to manually import each benchmark function
2. **Automatic Scaling**: New benchmarks are automatically discovered
3. **Maintainability**: Changes to benchmark structure don't require runner updates
4. **Flexibility**: Easy to filter by category, model, or other criteria
5. **Robustness**: Comprehensive error handling and graceful degradation

## Usage Examples

### Run all benchmarks in the project:
```python
from benchmarks.discovery import discover_and_run_all_benchmarks
results = discover_and_run_all_benchmarks()
```

### Run model-specific benchmarks:
```python
from src.inference_pio.models.glm_4_7.benchmarks.discovery import run_glm_4_7_benchmarks
results = run_glm_4_7_benchmarks()
```

### Run only performance benchmarks for a model:
```python
from src.inference_pio.models.qwen3_vl_2b.benchmarks.discovery import run_qwen3_vl_2b_performance_benchmarks
results = run_qwen3_vl_2b_performance_benchmarks()
```

## Integration

The system seamlessly integrates with the existing benchmark infrastructure:
- Maintains all existing result formats (JSON/CSV)
- Preserves the same reporting structure
- Compatible with existing benchmark implementations
- No changes needed to individual benchmark functions
