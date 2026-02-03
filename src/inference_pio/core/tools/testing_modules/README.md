# Modular Testing Framework for Mod Project

This document describes the new modular testing framework implemented in the Mod project. The framework separates different aspects of testing into independent, reusable modules.

## Overview

The modular testing framework consists of five independent testing modules:

1. **Functional Testing Module** - Tests complete user workflows and system functionality
2. **Performance Testing Module** - Benchmarks performance metrics and resource usage
3. **Integration Testing Module** - Tests interactions between multiple components
4. **Regression Testing Module** - Ensures changes don't break existing functionality
5. **Unit Testing Module** - Tests individual units of code in isolation

Each module is completely independent and can be used separately or together through the main orchestrator.

## Directory Structure

```
testing_modules/
├── __init__.py          # Main orchestrator and entry point
├── functional_testing.py  # Functional testing module
├── performance_testing.py # Performance testing module
├── integration_testing.py # Integration testing module
├── regression_testing.py  # Regression testing module
└── unit_testing.py       # Unit testing module
```

## Usage

### Individual Module Usage

Each testing module can be used independently:

```python
from testing_modules.functional_testing import ModelFunctionalTest

class MyModelFunctionalTest(ModelFunctionalTest):
    def get_model_plugin_class(self):
        return MyModelPluginClass
    
    def test_specific_functionality(self):
        # Your test implementation
        pass
```

### Using the Orchestrator

The main orchestrator allows running multiple test types together:

```python
from testing_modules import TestOrchestrator, TestType

# Create orchestrator
orchestrator = TestOrchestrator()

# Run specific test types
results = orchestrator.run_tests_by_type(TestType.FUNCTIONAL)

# Run multiple test types
results = orchestrator.run_selected_tests([
    TestType.UNIT, 
    TestType.INTEGRATION, 
    TestType.FUNCTIONAL
])

# Run all tests
all_results = orchestrator.run_all_tests()
```

### Running Existing Tests with New Modules

Existing tests have been updated to use the new modular structure:

- `test_real_functional.py` now extends `ModelFunctionalTest`
- `test_real_performance.py` now extends `ModelPerformanceTest`
- `test_real_integration.py` now extends `ModelIntegrationTest`
- `test_real_regression.py` now extends `ModelRegressionTest`

## Module Details

### Functional Testing Module

Provides base classes for functional testing:
- `FunctionalTestBase` - Base class for all functional tests
- `ModelFunctionalTest` - For testing model plugins
- `PluginFunctionalTest` - For testing plugins

### Performance Testing Module

Provides performance benchmarking capabilities:
- `PerformanceTestBase` - Base class with performance measurement utilities
- `ModelPerformanceTest` - For benchmarking model performance
- `PluginPerformanceTest` - For benchmarking plugin performance

### Integration Testing Module

Focuses on component interaction testing:
- `IntegrationTestBase` - Base class for integration tests
- `ModelIntegrationTest` - For testing model integration
- `PipelineIntegrationTest` - For testing pipeline integration
- `PluginIntegrationTest` - For testing plugin integration

### Regression Testing Module

Ensures consistency over time:
- `RegressionTestBase` - Base class with baseline comparison utilities
- `ModelRegressionTest` - For model regression testing
- `FeatureRegressionTest` - For specific feature regression testing
- `SystemRegressionTest` - For system-level regression testing

### Unit Testing Module

Tests individual components in isolation:
- `UnitTestBase` - Base class for unit tests
- `ModelUnitTest` - For unit testing model plugins
- `PluginUnitTest` - For unit testing plugins
- `ComponentUnitTest` - For unit testing generic components

## Benefits

1. **Independence**: Each module can be used independently
2. **Reusability**: Common functionality is abstracted into base classes
3. **Maintainability**: Clear separation of concerns makes code easier to maintain
4. **Scalability**: Easy to add new test types or extend existing ones
5. **Flexibility**: Tests can be run individually or as part of orchestrated suites

## Best Practices

1. Extend the appropriate base class for your test type
2. Implement required abstract methods like `get_model_plugin_class()`
3. Use the orchestrator for running multiple test types together
4. Follow the naming conventions for test methods (`test_*`)
5. Keep tests focused and independent of each other

## Migration Notes

Existing tests in the `tests/` directory have been updated to use the new modular structure. The original tests in the root directory have been updated to extend the appropriate base classes from the new modules.

For custom model implementations, update your test classes to extend the appropriate base class from the testing modules rather than the original base classes.