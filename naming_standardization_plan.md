# Naming Standardization Plan

## Overview
This document outlines the plan to standardize test and benchmark file naming conventions according to the project's architecture and organization standards.

## Current Issues Identified

### 1. Inconsistent Test File Naming
- Some files use `test_*.py` pattern (correct)
- Others use `*_test.py` pattern (incorrect)
- Mixed patterns within the same directory

### 2. Misplaced Benchmark Files
- Benchmark files scattered across directories
- Not organized in the `benchmarks/` directory structure
- Mixed with test files

### 3. Inconsistent Directory Structure
- Some models have tests in `tests/` subdirectory
- Others have tests directly in model directory
- Missing standardized structure

## Standardized Naming Convention

### Test Files
- **Unit tests**: `test_{component_name}.py` (e.g., `test_attention.py`, `test_config_loading.py`)
- **Integration tests**: `test_{component_name}_integration.py` (e.g., `test_plugin_integration.py`)
- **End-to-end tests**: `test_{feature_name}_end_to_end.py` (e.g., `test_model_inference_end_to_end.py`)
- **Performance tests**: `test_{component_name}_performance.py` (e.g., `test_inference_performance.py`)

### Benchmark Files
- **Performance benchmarks**: `benchmark_{metric_name}.py` (e.g., `benchmark_inference_speed.py`, `benchmark_memory_usage.py`)
- **Comparison benchmarks**: `benchmark_{comparison_type}_comparison.py` (e.g., `benchmark_optimized_vs_unoptimized_comparison.py`)
- **Accuracy benchmarks**: `benchmark_{model_name}_accuracy.py` (e.g., `benchmark_model_accuracy.py`)

## Directory Structure
```
src/
└── inference_pio/
    ├── models/
    │   └── {model_name}/
    │       └── tests/
    │           ├── unit/
    │           ├── integration/
    │           └── performance/
    ├── common/
    │   └── tests/
    │       ├── unit/
    │       ├── integration/
    │       └── performance/
    └── plugin_system/
        └── tests/
            ├── unit/
            ├── integration/
            └── performance/
benchmarks/
└── {model_name}/
    ├── unit/
    ├── integration/
    └── performance/
```

## Specific File Renames Needed

### Test Files to Rename (from *_test.py to test_*.py pattern)
- `simple_verification_test.py` → `test_simple_verification.py`
- `standalone_test_utils_test.py` → `test_standalone_test_utils.py`
- `test_final_verification.py` → `test_final_verification.py` (already correct)
- `test_test_utils.py` → `test_test_utils.py` (already correct)

### Files in dev_tools/tests/ that need renaming
- `test_*.py` files in dev_tools/tests/ need to be categorized properly

### Model-specific files that need reorganization
- Files in `src/inference_pio/models/*/tests/` need to be organized by type (unit, integration, performance)

## Implementation Steps

### Step 1: Create missing directory structures
- Ensure all models have unit, integration, performance subdirectories in their tests/

### Step 2: Rename files to follow standard pattern
- Convert all `*_test.py` to `test_*.py` pattern

### Step 3: Move files to appropriate categories
- Organize files by test type (unit, integration, performance)

### Step 4: Update benchmark files
- Move all benchmark files to benchmarks/ directory structure
- Rename benchmark files to follow standard pattern

### Step 5: Update imports and references
- Update any imports that reference moved/renamed files
- Update documentation that references specific file paths