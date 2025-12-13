# Qwen3-VL-2B-Instruct Test Suite

This directory contains the comprehensive test suite for the Qwen3-VL-2B-Instruct project, covering memory management, CPU optimizations, and system integration.

## Test Suite Structure

```
tests/
├── unit/
│   ├── test_memory_management.py      # Unit tests for memory management system
│   ├── test_cpu_optimizations.py      # Unit tests for CPU optimization system
│   └── test_improvements_validation.py # Tests for improvements and fixes
├── integration/
│   ├── test_optimization_pipeline.py  # Integration tests for optimization pipeline
│   └── test_component_interaction.py  # Tests for component interactions
├── performance/
│   └── test_performance_regression.py # Performance regression tests
├── conftest.py                       # Shared fixtures and configuration
└── test_suite.py                     # Main test suite runner
```

## Test Categories

### 1. Unit Tests
- **Memory Management**: Tests for `AdvancedMemoryPool`, `MemoryDefragmenter`, `CacheAwareMemoryManager`, `VisionLanguageMemoryOptimizer`
- **CPU Optimizations**: Tests for `IntelCPUOptimizedPreprocessor`, `IntelOptimizedPipeline`, `AdaptiveIntelOptimizer`, and related components
- **Improvements**: Tests for thread safety, error handling, memory management improvements

### 2. Integration Tests
- **Optimization Pipeline**: Tests for the end-to-end optimization pipeline
- **Component Interaction**: Tests for how different system components work together

### 3. Performance Tests
- **Regression Tests**: Tests to ensure performance doesn't degrade over time
- **Benchmarking**: Performance measurement and comparison tests

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run specific test category:
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only  
pytest tests/integration/

# Performance tests only
pytest tests/performance/
```

### Run with specific options:
```bash
# Verbose output with coverage
pytest tests/ -v --cov=src/

# Run with specific markers
pytest tests/ -m "cpu"  # Only CPU-related tests

# Run specific test file
pytest tests/unit/test_memory_management.py
```

## Test Markers

The test suite uses the following markers:
- `multimodal`: Multimodal processing tests
- `cpu`: CPU-specific optimization tests  
- `gpu`: GPU-related tests
- `performance`: Performance benchmark tests
- `accuracy`: Accuracy validation tests

## Test Coverage

The test suite provides comprehensive coverage including:

### Memory Management
- Memory pool allocation and deallocation
- Memory defragmentation
- Cache-aware memory layouts
- Thread-safe memory operations
- Specialized memory pools for different tensor types
- Memory pool statistics and monitoring

### CPU Optimizations
- Intel-specific optimizations
- CPU pipeline stages
- Adaptive optimization parameters
- Thread-safe preprocessing
- Cache-optimized operations
- Power and thermal management

### Error Handling
- Invalid parameter validation
- Graceful error recovery
- Resource cleanup
- Exception handling in concurrent operations

### Thread Safety
- Concurrent memory operations
- Thread-safe pipeline operations
- Synchronized resource access
- Race condition prevention

### Performance
- Allocation/deallocation performance
- Pipeline throughput
- Memory fragmentation over time
- Concurrent operation efficiency
- Resource utilization efficiency

## Best Practices Followed

1. **Isolation**: Each test is independent and doesn't rely on global state
2. **Repeatability**: Tests produce consistent results across runs
3. **Speed**: Tests are optimized to run quickly
4. **Clarity**: Test names clearly describe what is being tested
5. **Maintainability**: Tests follow DRY principles and are easy to modify
6. **Comprehensiveness**: Tests cover both positive and negative scenarios

## Adding New Tests

When adding new tests:

1. Place unit tests in `tests/unit/`
2. Place integration tests in `tests/integration/` 
3. Place performance tests in `tests/performance/`
4. Use descriptive test names following the pattern `test_[what]_[condition]`
5. Include proper assertions and error handling
6. Use fixtures from `conftest.py` where appropriate
7. Add appropriate markers for test categorization