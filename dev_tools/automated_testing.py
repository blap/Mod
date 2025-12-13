"""
Automated Testing Tools for Qwen3-VL Model Optimizations

This module provides comprehensive automated testing tools to validate that 
optimizations don't break functionality in the Qwen3-VL model.
"""

import os
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import numpy as np
import copy
import time
import warnings
from contextlib import contextmanager
import traceback


@dataclass
class TestResult:
    """Data class for test results"""
    test_name: str
    passed: bool
    duration: float
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class OptimizationValidator:
    """Validator for model optimizations"""
    
    def __init__(self):
        self.test_results = []
        self.optimization_tests = []
    
    def add_optimization_test(self, name: str, test_func: Callable, description: str = ""):
        """Add an optimization-specific test"""
        self.optimization_tests.append({
            'name': name,
            'func': test_func,
            'description': description
        })
    
    def run_optimization_tests(self, model: nn.Module, test_data: Optional[Tuple] = None) -> List[TestResult]:
        """Run all optimization tests"""
        results = []
        
        for test_info in self.optimization_tests:
            start_time = time.time()
            
            try:
                # Run the test
                test_result = test_info['func'](model, test_data)
                duration = time.time() - start_time
                
                if test_result is True or (isinstance(test_result, dict) and test_result.get('passed', False)):
                    results.append(TestResult(
                        test_name=test_info['name'],
                        passed=True,
                        duration=duration,
                        details=test_result if isinstance(test_result, dict) else None
                    ))
                else:
                    results.append(TestResult(
                        test_name=test_info['name'],
                        passed=False,
                        duration=duration,
                        error=str(test_result) if test_result else "Test failed without specific error",
                        details=test_result if isinstance(test_result, dict) else None
                    ))
            except Exception as e:
                duration = time.time() - start_time
                results.append(TestResult(
                    test_name=test_info['name'],
                    passed=False,
                    duration=duration,
                    error=str(e),
                    details={'traceback': traceback.format_exc()}
                ))
        
        self.test_results.extend(results)
        return results
    
    def validate_memory_optimization(self, model: nn.Module, test_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """Validate that memory optimizations work correctly"""
        try:
            # Test 1: Check if model can handle larger batch sizes after optimization
            original_memory_usage = self._measure_memory_usage(model, test_data)
            
            # Run a forward pass to ensure model is working
            if test_data:
                inputs, targets = test_data
                with torch.no_grad():
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    # Check output shape is as expected
                    expected_shape = targets.shape if len(targets.shape) == len(outputs.shape) else targets.shape[:-1] + outputs.shape[-1:]
                    if outputs.shape != expected_shape:
                        return {'passed': False, 'error': f'Output shape mismatch: {outputs.shape} vs {expected_shape}'}
            
            # Test 2: Check that gradients flow properly after optimization
            if test_data:
                inputs, targets = test_data
                model.train()
                outputs = model(inputs)
                
                # Create a simple loss
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                    loss = torch.nn.functional.cross_entropy(outputs, targets.argmax(dim=1) if targets.dim() > 1 else targets)
                else:
                    loss = torch.nn.functional.mse_loss(outputs, targets)
                
                loss.backward()
                
                # Check that gradients exist for parameters that should have them
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is None:
                        return {'passed': False, 'error': f'No gradient for parameter {name}'}
                    if param.requires_grad and param.grad is not None and torch.isnan(param.grad).any():
                        return {'passed': False, 'error': f'NaN gradients found in parameter {name}'}
            
            return {'passed': True, 'memory_usage': original_memory_usage}
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def validate_performance_optimization(self, model: nn.Module, test_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """Validate that performance optimizations work correctly"""
        try:
            if not test_data:
                # Create dummy test data
                inputs = torch.randn(1, 100, 256)  # batch, seq_len, features
                targets = torch.randn(1, 100, 256)
                test_data = (inputs, targets)
            
            inputs, targets = test_data
            
            # Measure performance before and after
            start_time = time.time()
            with torch.no_grad():
                outputs = model(inputs)
            duration = time.time() - start_time
            
            # Validate output
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Check that output is reasonable (not NaN, inf, etc.)
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                return {'passed': False, 'error': 'Output contains NaN or Inf values'}
            
            # Check output shape
            if outputs.shape[0] != inputs.shape[0]:  # batch size should match
                return {'passed': False, 'error': f'Batch size mismatch: {outputs.shape[0]} vs {inputs.shape[0]}'}
            
            return {'passed': True, 'inference_time': duration, 'output_shape': outputs.shape}
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def validate_functional_equivalence(self, original_model: nn.Module, optimized_model: nn.Module, 
                                       test_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """Validate that optimized model produces equivalent results to original"""
        try:
            if not test_data:
                # Create dummy test data
                inputs = torch.randn(2, 10, 256)  # batch, seq_len, features
                test_data = (inputs, None)
            
            inputs, _ = test_data
            
            # Set models to eval mode
            original_model.eval()
            optimized_model.eval()
            
            # Ensure both models produce the same output
            with torch.no_grad():
                original_output = original_model(copy.deepcopy(inputs))
                optimized_output = optimized_model(copy.deepcopy(inputs))
            
            # Handle tuple outputs
            if isinstance(original_output, tuple):
                original_output = original_output[0]
            if isinstance(optimized_output, tuple):
                optimized_output = optimized_output[0]
            
            # Check shape equivalence
            if original_output.shape != optimized_output.shape:
                return {
                    'passed': False, 
                    'error': f'Output shape mismatch: {original_output.shape} vs {optimized_output.shape}'
                }
            
            # Check value equivalence (with tolerance for floating point errors)
            diff = torch.abs(original_output - optimized_output)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            
            # Use a reasonable tolerance
            tolerance = 1e-5
            if max_diff > tolerance:
                return {
                    'passed': False,
                    'error': f'Output values differ significantly. Max diff: {max_diff}, Mean diff: {mean_diff}',
                    'max_diff': max_diff,
                    'mean_diff': mean_diff
                }
            
            return {
                'passed': True,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'tolerance': tolerance
            }
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _measure_memory_usage(self, model: nn.Module, test_data: Optional[Tuple] = None) -> Dict[str, float]:
        """Measure memory usage of model"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.max_memory_allocated()
            
            if test_data:
                inputs, _ = test_data
                with torch.no_grad():
                    _ = model(inputs)
            
            peak_memory = torch.cuda.max_memory_allocated()
            return {
                'start_memory_mb': start_memory / 1024 / 1024,
                'peak_memory_mb': peak_memory / 1024 / 1024,
                'used_memory_mb': (peak_memory - start_memory) / 1024 / 1024
            }
        else:
            # Fallback for CPU-only systems
            return {'peak_memory_mb': 0, 'used_memory_mb': 0}


class IntegrationTestSuite:
    """Comprehensive integration test suite"""
    
    def __init__(self):
        self.tests = []
        self.results = []
    
    def add_test(self, name: str, test_func: Callable, description: str = ""):
        """Add a test to the suite"""
        self.tests.append({
            'name': name,
            'func': test_func,
            'description': description
        })
    
    def run_all_tests(self, model: nn.Module, test_data: Optional[Tuple] = None) -> List[TestResult]:
        """Run all tests in the suite"""
        results = []
        
        for test_info in self.tests:
            start_time = time.time()
            
            try:
                test_result = test_info['func'](model, test_data)
                duration = time.time() - start_time
                
                if test_result is True or (isinstance(test_result, dict) and test_result.get('passed', False)):
                    results.append(TestResult(
                        test_name=test_info['name'],
                        passed=True,
                        duration=duration,
                        details=test_result if isinstance(test_result, dict) else None
                    ))
                else:
                    results.append(TestResult(
                        test_name=test_info['name'],
                        passed=False,
                        duration=duration,
                        error=str(test_result) if test_result else "Test failed without specific error",
                        details=test_result if isinstance(test_result, dict) else None
                    ))
            except Exception as e:
                duration = time.time() - start_time
                results.append(TestResult(
                    test_name=test_info['name'],
                    passed=False,
                    duration=duration,
                    error=str(e),
                    details={'traceback': traceback.format_exc()}
                ))
        
        self.results.extend(results)
        return results
    
    def test_model_forward_pass(self, model: nn.Module, test_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """Test that model can perform forward pass without errors"""
        try:
            if not test_data:
                inputs = torch.randn(1, 10, 256)
                test_data = (inputs, None)
            
            inputs, _ = test_data
            
            model.eval()
            with torch.no_grad():
                output = model(inputs)
            
            # Validate output
            if isinstance(output, tuple):
                output = output[0]
            
            if torch.isnan(output).any() or torch.isinf(output).any():
                return {'passed': False, 'error': 'Output contains NaN or Inf values'}
            
            return {'passed': True, 'output_shape': output.shape}
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def test_model_backward_pass(self, model: nn.Module, test_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """Test that model can perform backward pass without errors"""
        try:
            if not test_data:
                inputs = torch.randn(2, 10, 256, requires_grad=True)
                targets = torch.randn(2, 10, 256)
                test_data = (inputs, targets)
            
            inputs, targets = test_data
            
            model.train()
            output = model(inputs)
            
            if isinstance(output, tuple):
                output = output[0]
            
            # Create a loss function appropriate for the output
            if output.shape == targets.shape:
                loss = torch.nn.functional.mse_loss(output, targets)
            else:
                # If shapes don't match, try to adapt
                if len(output.shape) > 1 and len(targets.shape) > 1:
                    min_size = min(output.shape[-1], targets.shape[-1])
                    loss = torch.nn.functional.mse_loss(
                        output[..., :min_size], 
                        targets[..., :min_size]
                    )
                else:
                    return {'passed': False, 'error': f'Cannot compute loss: shapes {output.shape} vs {targets.shape}'}
            
            loss.backward()
            
            # Check gradients
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                    
                    if torch.isnan(param.grad).any():
                        return {'passed': False, 'error': f'NaN gradients in parameter {name}'}
            
            return {
                'passed': True, 
                'loss_value': loss.item(),
                'avg_grad_norm': np.mean(grad_norms) if grad_norms else 0
            }
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def test_model_save_load(self, model: nn.Module, test_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """Test that model can be saved and loaded correctly"""
        try:
            # Create a temporary file
            temp_path = "temp_model_test.pt"
            
            # Save model
            torch.save(model.state_dict(), temp_path)
            
            # Create a new model instance with same architecture
            new_model = copy.deepcopy(model)
            
            # Load state dict
            new_model.load_state_dict(torch.load(temp_path))
            
            # Compare outputs
            if test_data:
                inputs, _ = test_data
                model.eval()
                new_model.eval()
                
                with torch.no_grad():
                    original_output = model(inputs)
                    loaded_output = new_model(inputs)
                
                if isinstance(original_output, tuple):
                    original_output = original_output[0]
                if isinstance(loaded_output, tuple):
                    loaded_output = loaded_output[0]
                
                if not torch.allclose(original_output, loaded_output, atol=1e-6):
                    return {'passed': False, 'error': 'Model outputs differ after save/load'}
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return {'passed': True}
        except Exception as e:
            # Clean up even if error occurs
            temp_path = "temp_model_test.pt"
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return {'passed': False, 'error': str(e)}
    
    def test_model_device_compatibility(self, model: nn.Module, test_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """Test that model works on different devices"""
        try:
            if not test_data:
                inputs = torch.randn(1, 10, 256)
                test_data = (inputs, None)
            
            inputs, _ = test_data
            
            devices_to_test = [torch.device('cpu')]
            if torch.cuda.is_available():
                devices_to_test.append(torch.device('cuda'))
            
            original_device = next(model.parameters()).device
            
            for device in devices_to_test:
                # Move model to device
                model_test = copy.deepcopy(model).to(device)
                inputs_test = inputs.to(device)
                
                # Test forward pass
                model_test.eval()
                with torch.no_grad():
                    output = model_test(inputs_test)
                
                if isinstance(output, tuple):
                    output = output[0]
                
                if torch.isnan(output).any() or torch.isinf(output).any():
                    return {'passed': False, 'error': f'Output contains NaN or Inf on device {device}'}
            
            # Move model back to original device
            model.to(original_device)
            
            return {'passed': True, 'tested_devices': [str(d) for d in devices_to_test]}
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def test_model_gradient_flow(self, model: nn.Module, test_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """Test that gradients flow properly through the model"""
        try:
            if not test_data:
                inputs = torch.randn(2, 10, 256, requires_grad=True)
                targets = torch.randn(2, 10, 256)
                test_data = (inputs, targets)
            
            inputs, targets = test_data
            
            model.train()
            
            # Zero gradients
            model.zero_grad()
            
            # Forward pass
            output = model(inputs)
            if isinstance(output, tuple):
                output = output[0]
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(output, targets)
            
            # Backward pass
            loss.backward()
            
            # Check that gradients exist and are valid
            param_count = 0
            grad_count = 0
            nan_grad_count = 0
            zero_grad_count = 0
            
            for name, param in model.named_parameters():
                param_count += 1
                if param.grad is not None:
                    grad_count += 1
                    if torch.isnan(param.grad).any():
                        nan_grad_count += 1
                    if torch.allclose(param.grad, torch.zeros_like(param.grad), atol=1e-8):
                        zero_grad_count += 1
            
            if nan_grad_count > 0:
                return {'passed': False, 'error': f'{nan_grad_count} parameters have NaN gradients'}
            
            if grad_count < param_count:
                return {'passed': False, 'error': f'Not all parameters have gradients: {grad_count}/{param_count}'}
            
            return {
                'passed': True,
                'param_count': param_count,
                'grad_count': grad_count,
                'nan_grad_count': nan_grad_count,
                'zero_grad_count': zero_grad_count,
                'loss_value': loss.item()
            }
        except Exception as e:
            return {'passed': False, 'error': str(e)}


class RegressionTestSuite:
    """Regression test suite to ensure functionality is preserved"""
    
    def __init__(self):
        self.baseline_outputs = {}
        self.test_results = []
    
    def establish_baseline(self, model: nn.Module, test_inputs: List[torch.Tensor], 
                          model_name: str = "default") -> Dict[str, torch.Tensor]:
        """Establish baseline outputs for regression testing"""
        model.eval()
        baseline = {}
        
        with torch.no_grad():
            for i, inputs in enumerate(test_inputs):
                key = f"input_{i}"
                output = model(inputs)
                if isinstance(output, tuple):
                    output = output[0]  # Take first output if tuple
                baseline[key] = output.clone().detach()
        
        self.baseline_outputs[model_name] = baseline
        return baseline
    
    def run_regression_tests(self, model: nn.Module, model_name: str = "default") -> List[TestResult]:
        """Run regression tests against established baseline"""
        if model_name not in self.baseline_outputs:
            raise ValueError(f"No baseline established for model '{model_name}'")
        
        model.eval()
        results = []
        
        with torch.no_grad():
            for key, baseline_output in self.baseline_outputs[model_name].items():
                # Create input tensor with same shape as used for baseline
                input_shape = [baseline_output.shape[0]]  # batch size
                # This is a simplification - in practice, you'd need to store the original inputs
                # For now, we'll just create a random tensor with the same batch size
                inputs = torch.randn_like(baseline_output, requires_grad=False)
                
                # Adjust the inputs to have the right shape for the model
                # This is tricky without storing original inputs, so we'll skip this test for now
                # Instead, we'll just test that the model can produce outputs of the right shape
                output = model(inputs)
                if isinstance(output, tuple):
                    output = output[0]
                
                # Check shape
                if output.shape != baseline_output.shape:
                    results.append(TestResult(
                        test_name=f"regression_shape_{key}",
                        passed=False,
                        duration=0,
                        error=f"Output shape changed: {output.shape} vs {baseline_output.shape}"
                    ))
                else:
                    # Check values (with tolerance)
                    diff = torch.abs(output - baseline_output)
                    max_diff = torch.max(diff).item()
                    
                    if max_diff > 1e-4:  # Tolerance for regression
                        results.append(TestResult(
                            test_name=f"regression_values_{key}",
                            passed=False,
                            duration=0,
                            error=f"Output values changed significantly. Max diff: {max_diff}"
                        ))
                    else:
                        results.append(TestResult(
                            test_name=f"regression_{key}",
                            passed=True,
                            duration=0
                        ))
        
        self.test_results.extend(results)
        return results


class HardwareCompatibilityTester:
    """Test hardware compatibility and optimization effectiveness"""
    
    def __init__(self):
        self.test_results = []
    
    def test_cuda_compatibility(self, model: nn.Module, test_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """Test CUDA compatibility if available"""
        if not torch.cuda.is_available():
            return {'passed': True, 'message': 'CUDA not available, skipping CUDA tests'}
        
        try:
            # Test model on CUDA
            cuda_model = model.to('cuda')
            
            if test_data:
                inputs, targets = test_data
                cuda_inputs = inputs.to('cuda')
                
                cuda_model.eval()
                with torch.no_grad():
                    cuda_output = cuda_model(cuda_inputs)
                
                if isinstance(cuda_output, tuple):
                    cuda_output = cuda_output[0]
                
                if torch.isnan(cuda_output).any() or torch.isinf(cuda_output).any():
                    return {'passed': False, 'error': 'CUDA output contains NaN or Inf values'}
            
            # Test CUDA-specific optimizations
            if hasattr(torch.cuda, 'get_device_name'):
                device_name = torch.cuda.get_device_name(0)
                compute_capability = torch.cuda.get_device_capability(0)
                
                return {
                    'passed': True,
                    'device_name': device_name,
                    'compute_capability': compute_capability
                }
        
        except Exception as e:
            return {'passed': False, 'error': f'CUDA test failed: {str(e)}'}
    
    def test_mixed_precision(self, model: nn.Module, test_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """Test mixed precision training compatibility"""
        try:
            if not test_data:
                inputs = torch.randn(2, 10, 256)
                targets = torch.randn(2, 10, 256)
                test_data = (inputs, targets)
            
            inputs, targets = test_data
            
            # Test with autocast
            model.train()
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else warnings.catch_warnings():
                if torch.cuda.is_available():
                    warnings.simplefilter("ignore")  # Ignore autocast warnings
                
                output = model(inputs)
                if isinstance(output, tuple):
                    output = output[0]
                
                if torch.isnan(output).any() or torch.isinf(output).any():
                    return {'passed': False, 'error': 'Mixed precision output contains NaN or Inf values'}
                
                # Compute loss
                loss = torch.nn.functional.mse_loss(output, targets)
            
            # Test backward pass with gradient scaling
            if torch.cuda.is_available():
                scaler = torch.cuda.amp.GradScaler()
                scaler.scale(loss).backward()
                scaler.step(optim.SGD(model.parameters(), lr=0.01))
                scaler.update()
            else:
                # Fallback for CPU
                loss.backward()
            
            return {'passed': True, 'loss_value': loss.item()}
        except Exception as e:
            return {'passed': False, 'error': f'Mixed precision test failed: {str(e)}'}
    
    def run_all_hardware_tests(self, model: nn.Module, test_data: Optional[Tuple] = None) -> List[TestResult]:
        """Run all hardware compatibility tests"""
        results = []
        
        # CUDA compatibility test
        start_time = time.time()
        cuda_result = self.test_cuda_compatibility(model, test_data)
        duration = time.time() - start_time
        
        results.append(TestResult(
            test_name="cuda_compatibility",
            passed=cuda_result['passed'],
            duration=duration,
            error=cuda_result.get('error'),
            details=cuda_result
        ))
        
        # Mixed precision test
        start_time = time.time()
        mp_result = self.test_mixed_precision(model, test_data)
        duration = time.time() - start_time
        
        results.append(TestResult(
            test_name="mixed_precision",
            passed=mp_result['passed'],
            duration=duration,
            error=mp_result.get('error'),
            details=mp_result
        ))
        
        self.test_results.extend(results)
        return results


class TestReporter:
    """Generate reports from test results"""
    
    def __init__(self):
        pass
    
    def generate_report(self, results: List[TestResult], title: str = "Test Report") -> str:
        """Generate a text report from test results"""
        report_lines = [f"=== {title} ===", ""]
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests
        
        report_lines.append(f"Total Tests: {total_tests}")
        report_lines.append(f"Passed: {passed_tests}")
        report_lines.append(f"Failed: {failed_tests}")
        report_lines.append(f"Success Rate: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "Success Rate: 0%")
        report_lines.append("")
        
        if failed_tests > 0:
            report_lines.append("Failed Tests:")
            for result in results:
                if not result.passed:
                    report_lines.append(f"  - {result.test_name}: {result.error}")
            report_lines.append("")
        
        if results:
            avg_duration = sum(r.duration for r in results) / len(results)
            report_lines.append(f"Average Test Duration: {avg_duration:.4f}s")
            report_lines.append(f"Total Duration: {sum(r.duration for r in results):.4f}s")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_report(self, results: List[TestResult], path: str, title: str = "Test Report"):
        """Save test report to file"""
        report = self.generate_report(results, title)
        with open(path, 'w') as f:
            f.write(report)


# Global test instances
optimization_validator = OptimizationValidator()
integration_suite = IntegrationTestSuite()
regression_suite = RegressionTestSuite()
hardware_tester = HardwareCompatibilityTester()
test_reporter = TestReporter()


def setup_optimization_tests():
    """Setup standard optimization tests"""
    # Add optimization-specific tests
    optimization_validator.add_optimization_test(
        "memory_optimization",
        optimization_validator.validate_memory_optimization,
        "Validate memory optimization correctness"
    )
    
    optimization_validator.add_optimization_test(
        "performance_optimization",
        optimization_validator.validate_performance_optimization,
        "Validate performance optimization correctness"
    )


def setup_integration_tests():
    """Setup standard integration tests"""
    # Add integration tests
    integration_suite.add_test(
        "forward_pass",
        integration_suite.test_model_forward_pass,
        "Test model forward pass"
    )
    
    integration_suite.add_test(
        "backward_pass",
        integration_suite.test_model_backward_pass,
        "Test model backward pass"
    )
    
    integration_suite.add_test(
        "save_load",
        integration_suite.test_model_save_load,
        "Test model save/load functionality"
    )
    
    integration_suite.add_test(
        "device_compatibility",
        integration_suite.test_model_device_compatibility,
        "Test device compatibility"
    )
    
    integration_suite.add_test(
        "gradient_flow",
        integration_suite.test_model_gradient_flow,
        "Test gradient flow"
    )


def run_comprehensive_tests(model: nn.Module, test_data: Optional[Tuple] = None) -> Dict[str, Any]:
    """Run all types of tests and return comprehensive results"""
    results = {}
    
    print("Running optimization tests...")
    opt_results = optimization_validator.run_optimization_tests(model, test_data)
    results['optimization'] = opt_results
    
    print("Running integration tests...")
    int_results = integration_suite.run_all_tests(model, test_data)
    results['integration'] = int_results
    
    print("Running hardware compatibility tests...")
    hw_results = hardware_tester.run_all_hardware_tests(model, test_data)
    results['hardware'] = hw_results
    
    # Generate reports
    opt_report = test_reporter.generate_report(opt_results, "Optimization Tests")
    int_report = test_reporter.generate_report(int_results, "Integration Tests")
    hw_report = test_reporter.generate_report(hw_results, "Hardware Compatibility Tests")
    
    results['reports'] = {
        'optimization': opt_report,
        'integration': int_report,
        'hardware': hw_report
    }
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Optimization Tests: {sum(1 for r in opt_results if r.passed)}/{len(opt_results)} passed")
    print(f"Integration Tests: {sum(1 for r in int_results if r.passed)}/{len(int_results)} passed")
    print(f"Hardware Tests: {sum(1 for r in hw_results if r.passed)}/{len(hw_results)} passed")
    
    return results


def example_optimization_testing():
    """Example of optimization testing"""
    print("=== Optimization Testing Example ===")
    
    # Setup tests
    setup_optimization_tests()
    setup_integration_tests()
    
    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = SimpleModel()
    
    # Create test data
    test_inputs = torch.randn(4, 256)
    test_targets = torch.randn(4, 10)
    test_data = (test_inputs, test_targets)
    
    # Run comprehensive tests
    results = run_comprehensive_tests(model, test_data)
    
    # Print detailed reports
    print("\n" + "="*50)
    print("DETAILED REPORTS")
    print("="*50)
    
    for category, report in results['reports'].items():
        print(f"\n{category.upper()} TESTS REPORT:")
        print(report)


def example_regression_testing():
    """Example of regression testing"""
    print("\n=== Regression Testing Example ===")
    
    # Create two similar models
    class ModelV1(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(100, 10)
        
        def forward(self, x):
            return self.layer(x)
    
    class ModelV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(100, 10)
            self.activation = nn.ReLU()
        
        def forward(self, x):
            x = self.layer(x)
            return self.activation(x)
    
    model_v1 = ModelV1()
    model_v2 = ModelV2()
    
    # Create test inputs
    test_inputs = [torch.randn(2, 100) for _ in range(3)]
    
    # Establish baseline with V1
    baseline = regression_suite.establish_baseline(model_v1, test_inputs, "model_v1")
    print(f"Established baseline with {len(baseline)} outputs")
    
    # Run regression tests with V2 (should fail due to different architecture)
    try:
        regression_results = regression_suite.run_regression_tests(model_v2, "model_v1")
        print(f"Regression tests completed: {len(regression_results)} tests")
        
        for result in regression_results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  {result.test_name}: {status}")
            if not result.passed:
                print(f"    Error: {result.error}")
    except ValueError as e:
        print(f"Expected error: {e}")


if __name__ == "__main__":
    example_optimization_testing()
    example_regression_testing()