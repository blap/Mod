"""
Developer Experience Enhancement System for Qwen3-VL Model

This module integrates all the developer experience tools into a unified system
that significantly improves the development workflow for the Qwen3-VL model.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn

# Import all the tools we've created
# Use relative imports when used as a package, absolute imports when run as script
try:
    from .debugging_utils import (
        tensor_debugger, model_debugger, debug_tracer,
        enable_debugging, disable_debugging, print_debug_summary
    )
    from .profiling_tools import (
        global_profiler, bottleneck_detector, enable_profiling,
        disable_profiling, profile_block
    )
    from .config_validation import (
        global_config_validator, config_manager, setup_qwen3_vl_schemas,
        validate_config_arg
    )
    from .model_inspection import (
        ModelInspector, ParameterAnalyzer, ArchitectureVisualizer,
        create_model_diff_report, analyze_model_capacity
    )
    from .automated_testing import (
        optimization_validator, integration_suite, regression_suite,
        hardware_tester, test_reporter, run_comprehensive_tests,
        setup_optimization_tests, setup_integration_tests
    )
    from .documentation_generator import (
        DocumentationGenerator, DocParser, ModelDocGenerator
    )
    from .code_quality import (
        quality_checker, pylint_runner, flake8_runner, black_formatter,
        code_metrics, report_generator, run_quality_checks, generate_quality_report
    )
    from .benchmarking_tools import (
        benchmark_suite, model_benchmark_suite, optimization_benchmark_suite,
        hardware_benchmark_suite, benchmark_reporter, run_model_benchmarks,
        compare_optimizations, run_hardware_benchmarks
    )
except ImportError:
    # Fallback to absolute imports when run as script
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


class DeveloperExperienceSystem:
    """Main system that integrates all developer experience tools"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.doc_generator = DocumentationGenerator()
        self.model_inspector = ModelInspector()
        self.param_analyzer = ParameterAnalyzer()
        
        # Setup standard configurations
        setup_qwen3_vl_schemas()
        setup_optimization_tests()
        setup_integration_tests()
    
    def initialize_project(self):
        """Initialize the development environment with all tools"""
        print("Initializing Qwen3-VL Developer Experience System...")
        
        # Create necessary directories
        dirs_to_create = [
            self.project_root / "docs",
            self.project_root / "tests",
            self.project_root / "benchmarks",
            self.project_root / "reports"
        ]
        
        for directory in dirs_to_create:
            directory.mkdir(exist_ok=True)
        
        # Enable all debugging and profiling by default
        enable_debugging()
        enable_profiling()
        
        print("SUCCESS: Developer Experience System initialized")
    
    def validate_model_configuration(self, config_path: str) -> bool:
        """Validate model configuration"""
        try:
            result = config_manager.load_config(config_path, validate=True)
            print(f"SUCCESS: Configuration validation passed: {config_path}")
            return True
        except Exception as e:
            print(f"FAILED: Configuration validation failed: {e}")
            return False
    
    def inspect_model(self, model: nn.Module, input_shape: tuple = None):
        """Comprehensive model inspection"""
        print("Running model inspection...")
        
        summary = self.model_inspector.inspect_model(model, input_shape)
        self.model_inspector.print_model_summary()
        
        # Analyze parameters
        param_stats = self.param_analyzer.analyze_parameters(model)
        large_params = self.param_analyzer.find_large_parameters()
        sparse_layers = self.param_analyzer.find_sparse_layers()
        
        if large_params:
            print(f"Found {len(large_params)} layers with large parameters: {large_params}")
        
        if sparse_layers:
            print(f"Found {len(sparse_layers)} potentially sparse layers: {sparse_layers}")
        
        print("SUCCESS: Model inspection completed")
    
    def run_comprehensive_tests(self, model: nn.Module, test_data: Optional[tuple] = None):
        """Run all types of tests"""
        print("Running comprehensive tests...")
        
        results = run_comprehensive_tests(model, test_data)
        
        # Print summary
        opt_passed = sum(1 for r in results['optimization'] if r.passed)
        int_passed = sum(1 for r in results['integration'] if r.passed)
        hw_passed = sum(1 for r in results['hardware'] if r.passed)
        
        total_opt = len(results['optimization'])
        total_int = len(results['integration'])
        total_hw = len(results['hardware'])
        
        print(f"SUCCESS: Tests completed - Optimization: {opt_passed}/{total_opt}, Integration: {int_passed}/{total_int}, Hardware: {hw_passed}/{total_hw}")
        
        return results
    
    def benchmark_model(self, model: nn.Module, input_data: torch.Tensor, 
                       target_data: Optional[torch.Tensor] = None):
        """Run comprehensive benchmarks"""
        print("Running benchmarks...")
        
        results = run_model_benchmarks(model, input_data, target_data)
        
        print(f"SUCCESS: Benchmarks completed")
        return results
    
    def compare_model_versions(self, original_model: nn.Module, 
                              new_model: nn.Module, input_data: torch.Tensor):
        """Compare two model versions"""
        print("Comparing model versions...")
        
        comparison = compare_optimizations(original_model, new_model, input_data)
        
        print(f"SUCCESS: Model comparison completed - Speedup: {comparison['improvements']['speedup_factor']:.2f}x")
        return comparison
    
    def generate_documentation(self, source_dir: str = None, model: nn.Module = None, 
                              model_name: str = "Qwen3-VL"):
        """Generate comprehensive documentation"""
        print("Generating documentation...")
        
        if source_dir:
            self.doc_generator.generate_code_docs(source_dir)
        
        if model:
            self.doc_generator.generate_model_docs(model, model_name)
        
        print("SUCCESS: Documentation generated")
    
    def run_quality_checks(self, directory: str = "."):
        """Run code quality checks"""
        print("Running quality checks...")
        
        results = run_quality_checks(directory)
        generate_quality_report(results, output_dir=str(self.project_root / "reports" / "quality"))
        
        print(f"SUCCESS: Quality checks completed - Found {len(results['issues'])} issues")
        return results
    
    def generate_performance_report(self, output_path: str = None):
        """Generate a comprehensive performance report"""
        if output_path is None:
            output_path = str(self.project_root / "reports" / "performance_report.txt")
        
        # This would typically combine results from various tools
        report_content = [
            "Qwen3-VL Model Development Report",
            "=" * 40,
            "",
            "This report combines insights from all development tools.",
            "",
            "For detailed results, check individual tool outputs in the reports directory."
        ]
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_content))
        
        print(f"SUCCESS: Performance report generated: {output_path}")
    
    def setup_test_environment(self):
        """Setup a comprehensive test environment"""
        print("Setting up test environment...")
        
        # Create test configuration
        test_config = {
            "model": "qwen3_vl_test",
            "transformer_layers": 4,  # Smaller for testing
            "attention_heads": 8,
            "hidden_size": 512,
            "sequence_length": 128,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "optimizer": "adamw",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "precision": "fp32"  # Use FP32 for testing stability
        }
        
        config_manager.set_config("test_config", test_config)
        config_manager.save_config("test_config", str(self.project_root / "tests" / "test_config.json"))
        
        print("SUCCESS: Test environment setup completed")
    
    def run_development_workflow(self, model: nn.Module, input_data: torch.Tensor, 
                                target_data: Optional[torch.Tensor] = None):
        """Run the complete development workflow"""
        print("=" * 60)
        print("Starting Complete Development Workflow for Qwen3-VL Model")
        print("=" * 60)
        
        # 1. Model inspection
        self.inspect_model(model)
        
        # 2. Run comprehensive tests
        test_results = self.run_comprehensive_tests(model, (input_data, target_data) if target_data else None)
        
        # 3. Run benchmarks
        benchmark_results = self.benchmark_model(model, input_data, target_data)
        
        # 4. Run quality checks
        quality_results = self.run_quality_checks()
        
        # 5. Generate documentation
        self.generate_documentation(model=model)
        
        # 6. Generate performance report
        self.generate_performance_report()
        
        print("=" * 60)
        print("Development Workflow Completed Successfully!")
        print("=" * 60)
        
        return {
            'tests': test_results,
            'benchmarks': benchmark_results,
            'quality': quality_results
        }


def create_cli():
    """Create command-line interface for the developer tools"""
    parser = argparse.ArgumentParser(description="Qwen3-VL Developer Experience Tools")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Configuration validation command
    config_parser = subparsers.add_parser('validate-config', help='Validate model configuration')
    config_parser.add_argument('config_path', help='Path to configuration file')
    
    # Model inspection command
    inspect_parser = subparsers.add_parser('inspect-model', help='Inspect model architecture')
    inspect_parser.add_argument('model_path', help='Path to model file')
    
    # Testing command
    test_parser = subparsers.add_parser('run-tests', help='Run comprehensive tests')
    test_parser.add_argument('--model-path', help='Path to model file')
    test_parser.add_argument('--config-path', help='Path to configuration file')
    
    # Benchmarking command
    bench_parser = subparsers.add_parser('benchmark', help='Run model benchmarks')
    bench_parser.add_argument('--model-path', help='Path to model file')
    bench_parser.add_argument('--input-shape', help='Input shape as comma-separated values (e.g., 1,128,768)')
    
    # Quality check command
    quality_parser = subparsers.add_parser('check-quality', help='Run code quality checks')
    quality_parser.add_argument('directory', help='Directory to check', default='.')
    
    # Full workflow command
    workflow_parser = subparsers.add_parser('run-workflow', help='Run complete development workflow')
    workflow_parser.add_argument('--model-path', help='Path to model file')
    workflow_parser.add_argument('--config-path', help='Path to configuration file')
    
    return parser


def handle_cli_command(args):
    """Handle CLI commands"""
    dev_system = DeveloperExperienceSystem()
    
    if args.command == 'validate-config':
        success = dev_system.validate_model_configuration(args.config_path)
        return 0 if success else 1
    
    elif args.command == 'inspect-model':
        # This would require loading the model
        print("Model inspection from CLI requires loading the model in Python")
        print("Use the Python API for model inspection")
        return 0
    
    elif args.command == 'run-tests':
        print("Running tests from CLI requires specific model and test data")
        print("Use the Python API for comprehensive testing")
        return 0
    
    elif args.command == 'benchmark':
        print("Benchmarking from CLI requires specific model and input data")
        print("Use the Python API for benchmarking")
        return 0
    
    elif args.command == 'check-quality':
        results = run_quality_checks(args.directory)
        print(f"Quality check completed. Found {len(results['issues'])} issues.")
        return 0
    
    elif args.command == 'run-workflow':
        print("Running complete workflow from CLI requires specific model and data")
        print("Use the Python API for the complete workflow")
        return 0
    
    else:
        print("Unknown command. Use --help for available commands.")
        return 1


def example_usage():
    """Example of using the complete system"""
    print("=== Qwen3-VL Developer Experience System Example ===")
    
    # Initialize the system
    dev_system = DeveloperExperienceSystem()
    dev_system.initialize_project()
    
    # Create a simple model for demonstration
    class SimpleQwen3VL(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True),
                num_layers=4
            )
            self.classifier = nn.Linear(256, 10)
        
        def forward(self, x):
            x = self.transformer(x)
            x = x.mean(dim=1)  # Global average pooling
            x = self.classifier(x)
            return x
    
    model = SimpleQwen3VL()
    
    # Create sample data
    input_data = torch.randn(4, 64, 256)  # (batch, seq_len, features)
    target_data = torch.randint(0, 10, (4,))  # Classification targets
    
    # Run the complete development workflow
    results = dev_system.run_development_workflow(model, input_data, target_data)
    
    print("\nDevelopment workflow completed successfully!")
    print(f"Test results: {len(results['tests'])} test suites executed")
    print(f"Benchmark results: {len(results['benchmarks'])} benchmark types")
    print(f"Quality results: {len(results['quality']['issues'])} issues found")


def quick_start_guide():
    """Provide a quick start guide for developers"""
    guide = """
# Qwen3-VL Developer Experience Quick Start Guide

## Installation
```bash
# Recommended: Use the main project requirements for consistency
pip install -r requirements.txt  # Core dependencies from project root
pip install -r requirements-dev.txt  # Development dependencies from project root
pip install -e .
```

## Initialize Development Environment
```python
from dev_tools import DeveloperExperienceSystem

dev_system = DeveloperExperienceSystem()
dev_system.initialize_project()
```

## Validate Configuration
```python
# Validate your model configuration
dev_system.validate_model_configuration('path/to/config.json')
```

## Inspect Your Model
```python
# Inspect model architecture and parameters
dev_system.inspect_model(your_model, input_shape=(1, 128, 768))
```

## Run Comprehensive Tests
```python
# Run all tests (optimization, integration, hardware)
test_results = dev_system.run_comprehensive_tests(your_model, test_data)
```

## Benchmark Performance
```python
# Run performance benchmarks
benchmark_results = dev_system.benchmark_model(your_model, input_data)
```

## Compare Model Versions
```python
# Compare optimization impact
comparison = dev_system.compare_model_versions(original_model, optimized_model, input_data)
```

## Generate Documentation
```python
# Generate comprehensive documentation
dev_system.generate_documentation(model=your_model, model_name='MyQwen3VL')
```

## Run Quality Checks
```python
# Check code quality
quality_results = dev_system.run_quality_checks('./src')
```

## Complete Development Workflow
```python
# Run the entire workflow in one command
results = dev_system.run_development_workflow(model, input_data, target_data)
```

## Command Line Interface
```bash
# Validate configuration
python -m dev_tools validate-config path/to/config.json

# Run quality checks
python -m dev_tools check-quality ./src

# More commands available, use --help for details
python -m dev_tools --help
```

## Key Features

1. **Debugging Utilities**: Advanced tensor and model debugging
2. **Performance Profiling**: Bottleneck detection and visualization
3. **Configuration Validation**: Catch misconfigurations early
4. **Model Inspection**: Architecture analysis and visualization
5. **Automated Testing**: Comprehensive test suites
6. **Documentation Generation**: Auto-generate docs
7. **Code Quality**: Linting and formatting tools
8. **Benchmarking**: Performance comparison tools
"""
    return guide


if __name__ == "__main__":
    # Check if running as CLI
    if len(sys.argv) > 1:
        parser = create_cli()
        args = parser.parse_args()
        exit_code = handle_cli_command(args)
        sys.exit(exit_code)
    else:
        # Run example
        example_usage()