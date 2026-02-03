# Test Naming Conventions

This document establishes clear and consistent naming conventions for test classes and methods in the Inference-PIO project. Following these conventions improves code readability, maintainability, and makes it easier to understand test purposes at a glance.

## 1. General Principles

- **Descriptive**: Names should clearly describe what is being tested and under what conditions.
- **Consistent**: Follow the same patterns throughout the codebase.
- **Specific**: Be as specific as possible about the scenario being tested.
- **Readable**: Names should be easily understood by other developers.

## 2. Test Class Naming Conventions

### 2.1 Unit Test Classes
- Format: `Test[UnitName]` or `Test[UnitName][Scenario]`
- Examples:
  - `TestClassUnderTest` - Tests for a specific class
  - `TestClassUnderTestEdgeCases` - Edge case tests for a specific class
  - `TestPluginManagerInitialization` - Tests for plugin manager initialization

### 2.2 Integration Test Classes
- Format: `Test[IntegrationName]Integration` or `Test[SystemComponent1][SystemComponent2]Integration`
- Examples:
  - `TestPluginManagerModelIntegration` - Tests for plugin manager and model integration
  - `TestPipelineExecutionIntegration` - Tests for pipeline execution integration

### 2.3 Model-Specific Test Classes
- Format: `Test[ModelName][Feature]` or `Test[ModelName][Feature]Integration`
- Examples:
  - `TestQwen3_0_6B_Inference` - Tests for Qwen3-0.6B inference
  - `TestGLM_4_7_Flash_TokenizationIntegration` - Integration tests for GLM-4.7-Flash tokenization

### 2.4 Benchmark Classes
- Format: `[Feature]Benchmarks` or `[Component]Benchmarks`
- Examples:
  - `LLMInferenceSpeedBenchmarks` - Benchmarks for LLM inference speed
  - `ModelInitializationBenchmarks` - Benchmarks for model initialization
  - `PluginLoadingBenchmarks` - Benchmarks for plugin loading

## 3. Test Method Naming Conventions

### 3.1 General Format
- Format: `test_[feature_under_test]_[scenario]_[expected_outcome]`
- All test methods must start with `test_` (including benchmarks running under `unittest`).
- Use underscores to separate words.
- Keep names concise but descriptive.

### 3.2 Common Patterns

#### 3.2.1 Basic Functionality Tests
- Format: `test_[method_name]_with_[input_type]_returns_[expected_result]`
- Examples:
  - `test_initialize_with_valid_config_returns_true`
  - `test_infer_with_string_input_returns_processed_output`

#### 3.2.2 Error Condition Tests
- Format: `test_[method_name]_with_[invalid_condition]_raises_[exception_type]`
- Examples:
  - `test_tokenize_with_non_string_input_raises_TypeError`
  - `test_load_model_with_invalid_config_raises_ValueError`

#### 3.2.3 State Transition Tests
- Format: `test_[initial_state]_after_[action]_becomes_[final_state]`
- Examples:
  - `test_uninitialized_plugin_after_initialize_becomes_initialized`
  - `test_loaded_model_after_cleanup_becomes_unloaded`

#### 3.2.4 Configuration Tests
- Format: `test_[feature]_with_[configuration]_behaves_as_expected`
- Examples:
  - `test_inference_with_batch_size_1_behaves_as_expected`
  - `test_tokenization_with_special_tokens_enabled_works_properly`

#### 3.2.5 Integration Tests
- Format: `test_[component1_action]_affects_[component2_behavior]_correctly`
- Examples:
  - `test_plugin_registration_affects_manager_list_correctly`
  - `test_pipeline_stage_creation_enables_model_parallelism_correctly`

#### 3.2.6 Benchmark Methods
- Format: `test_[model/component]_[scenario]_[metric]`
- Examples:
  - `test_qwen3_0_6b_inference_speed`
  - `test_glm_4_7_flash_tokenization_latency`
  - `test_plugin_initialization_memory_usage`

### 3.3 Descriptive Words for Scenarios
Use these standardized terms in test method names:

- **Input types**: `with_string_input`, `with_list_input`, `with_dict_config`, `with_empty_data`
- **Conditions**: `with_valid_config`, `with_invalid_config`, `with_missing_params`, `with_extra_params`
- **States**: `when_initialized`, `when_not_initialized`, `after_cleanup`, `before_setup`
- **Expected outcomes**: `returns_expected_result`, `raises_ValueError`, `modifies_state_correctly`

## 4. Examples of Good vs Bad Naming

### 4.1 Test Class Names
```
❌ Bad:
- MyTest
- Test1
- PluginTests

✅ Good:
- TestPluginManager
- TestQwen3_0_6B_Inference
- TestMemoryManagementIntegration
```

### 4.2 Test Method Names
```
❌ Bad:
- test1()
- check_something()
- test_func()

✅ Good:
- test_initialize_with_valid_config_returns_true()
- test_infer_with_string_input_returns_processed_output()
- test_tokenize_with_non_string_input_raises_TypeError()
```

## 5. File Naming and Directory Structure

### 5.1 File Names
- Format: `test_[feature].py` or `test_[component]_[aspect].py`
- Examples:
  - `test_inference_speed.py`
  - `test_model_functionality.py`
  - `test_plugin_management.py`

### 5.2 Directory Structure
The project follows a structure where tests are organized by model or component.

```
tests/
├── unit/                       # Generic unit tests
├── integration/                # Generic integration tests
├── models/                     # Model-specific tests
│   ├── qwen3_0_6b/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── performance/
│   └── glm_4_7_flash/
│       └── ...
└── ...

benchmarks/
├── performance/                # Pure performance benchmarks
├── functional/                 # Functional benchmarks
└── integration/                # Integration benchmarks
```

## 6. Enforcement

These conventions should be followed for:
- All new test files
- Refactoring of existing tests
- Code reviews (reviewers should check for adherence)
- Automated checks (to be implemented in pre-commit hooks)
