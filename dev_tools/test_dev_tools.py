"""
Simple test to verify the dev tools package works correctly
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import directly from the modules since relative imports don't work when running as script
from debugging_utils import (
    tensor_debugger, model_debugger, debug_tracer,
    enable_debugging, disable_debugging, print_debug_summary
)
from profiling_tools import (
    global_profiler, bottleneck_detector, enable_profiling,
    disable_profiling, profile_block
)
from config_validation import (
    global_config_validator, config_manager, setup_qwen3_vl_schemas,
    validate_config_arg
)
from model_inspection import (
    ModelInspector, ParameterAnalyzer, ArchitectureVisualizer,
    create_model_diff_report, analyze_model_capacity
)
from automated_testing import (
    optimization_validator, integration_suite, regression_suite,
    hardware_tester, test_reporter, run_comprehensive_tests,
    setup_optimization_tests, setup_integration_tests
)
from documentation_generator import (
    DocumentationGenerator, DocParser, ModelDocGenerator
)
from code_quality import (
    quality_checker, pylint_runner, flake8_runner, black_formatter,
    code_metrics, report_generator, run_quality_checks, generate_quality_report
)
from benchmarking_tools import (
    benchmark_suite, model_benchmark_suite, optimization_benchmark_suite,
    hardware_benchmark_suite, benchmark_reporter, run_model_benchmarks,
    compare_optimizations, run_hardware_benchmarks
)

# Import the main system class
from __init__ import DeveloperExperienceSystem


def test_dev_tools():
    """Test that all dev tools can be imported and used"""
    print("Testing Qwen3-VL Developer Experience Tools...")
    
    # Initialize the development system
    dev_system = DeveloperExperienceSystem()
    dev_system.initialize_project()
    print("SUCCESS: DeveloperExperienceSystem initialized")
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(100, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.linear(x))
            return x
    
    model = TestModel()
    print("SUCCESS: Test model created")
    
    # Test model inspection
    dev_system.inspect_model(model, input_shape=(1, 100))
    print("SUCCESS: Model inspection completed")
    
    # Test configuration validation
    test_config = {
        "model": "qwen3_vl_2b",
        "transformer_layers": 2,
        "attention_heads": 2,
        "hidden_size": 64,
        "sequence_length": 128,
        "batch_size": 1,
        "device": "cpu"
    }
    
    # Save test config
    from config_validation import config_manager
    config_manager.set_config("test_config", test_config)
    config_manager.save_config("test_config", "test_config.json")
    
    # Validate config
    dev_system.validate_model_configuration("test_config.json")
    print("SUCCESS: Configuration validation completed")
    
    # Create test data
    input_data = torch.randn(2, 100)
    target_data = torch.randn(2, 10)
    
    # Test benchmarking
    benchmark_results = dev_system.benchmark_model(model, input_data, target_data)
    print("SUCCESS: Benchmarking completed")
    
    # Test quality checks (on the dev_tools directory)
    quality_results = dev_system.run_quality_checks(".")
    print("SUCCESS: Quality checks completed")
    
    # Generate documentation
    dev_system.generate_documentation(model=model, model_name="TestModel")
    print("SUCCESS: Documentation generation completed")
    
    # Run a subset of comprehensive tests
    test_results = dev_system.run_comprehensive_tests(model, (input_data, target_data))
    print("SUCCESS: Comprehensive tests completed")
    
    # Clean up test files
    import os
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")
    
    print("\nSUCCESS: All dev tools tests passed successfully!")
    return True


if __name__ == "__main__":
    success = test_dev_tools()
    if success:
        print("\nAll tests passed! The Qwen3-VL Developer Experience Tools are working correctly.")
    else:
        print("\nSome tests failed!")
        exit(1)