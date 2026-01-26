# Test Structure Configuration

## Overview
This document describes the new standardized test directory structure for the Inference-PIO project.

## Directory Structure
```
tests/
├── unit/                 # Unit tests for individual components
│   ├── models/           # Unit tests for specific models
│   ├── common/           # Unit tests for common components  
│   └── utils/            # Unit tests for utility functions
├── integration/          # Integration tests for component interactions
│   ├── models/           # Integration tests for specific models
│   ├── common/           # Integration tests for common components
│   └── end_to_end/       # End-to-end integration tests
└── performance/          # Performance and benchmarking tests
    ├── models/           # Performance tests for specific models
    └── common/           # Performance tests for common components
```

## Migration Guide
- All existing tests have been preserved and are accessible from both locations
- New tests should be added to the standardized structure
- Legacy test locations remain for backward compatibility

## Test Categories

### Unit Tests (`tests/unit/`)
- Test individual functions and classes in isolation
- Fast execution, no external dependencies
- Focus on logic correctness

### Integration Tests (`tests/integration/`)
- Test interactions between multiple components
- May involve external dependencies
- Validate system behavior

### Performance Tests (`tests/performance/`)
- Benchmark performance metrics
- Stress test system capabilities
- Monitor regression over time

## Running Tests

### All tests:
```bash
pytest
```

### Specific category:
```bash
pytest tests/unit/
pytest tests/integration/  
pytest tests/performance/
```

### Specific model:
```bash
pytest tests/unit/models/
pytest tests/integration/models/test_qwen3_vl_end_to_end.py
```

## Backward Compatibility
- Original test locations maintained via file copies
- All existing CI/CD pipelines and scripts continue to work
- Gradual migration path for new tests