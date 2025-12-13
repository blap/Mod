"""
Accuracy Preservation Tests for Qwen3-VL Model Optimizations
This module implements tests to validate that performance optimizations do not result in quality loss.
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, List
import pytest
from dataclasses import dataclass

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qwen3_vl.config.config import Qwen3VLConfig
from models.modeling_qwen3_vl_phase2 import Qwen3VLForConditionalGeneration


@dataclass
class AccuracyTestConfig:
    """Configuration for accuracy preservation tests"""
    # Model parameters
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    vocab_size: int = 1000
    max_position_embeddings: int = 128
    
    # Test parameters
    tolerance: float = 1e-5  # Tolerance for numerical differences
    similarity_threshold: float = 0.95  # Threshold for output similarity
    max_relative_error: float = 0.01  # Maximum allowed relative error (1%)
    
    # Input parameters
    batch_size: int = 2
    sequence_length: int = 32
    image_size: int = 224


class AccuracyPreservationTester:
    """Class to test accuracy preservation across optimizations"""
    
    def __init__(self, config: AccuracyTestConfig = None):
        self.config = config or AccuracyTestConfig()
    
    def create_models(self) -> Tuple[nn.Module, nn.Module]:
        """Create baseline and optimized models for comparison"""
        # Create configuration for baseline model (no optimizations)
        baseline_config = Qwen3VLConfig()
        baseline_config.hidden_size = self.config.hidden_size
        baseline_config.num_hidden_layers = self.config.num_hidden_layers
        baseline_config.num_attention_heads = self.config.num_attention_heads
        baseline_config.vocab_size = self.config.vocab_size
        baseline_config.max_position_embeddings = self.config.max_position_embeddings
        
        # Disable optimizations for baseline
        baseline_config.use_sparsity = False
        baseline_config.use_gradient_checkpointing = False
        baseline_config.use_moe = False
        baseline_config.use_flash_attention_2 = False
        baseline_config.use_dynamic_sparse_attention = False
        baseline_config.use_adaptive_depth = False
        baseline_config.use_context_adaptive_positional_encoding = False
        baseline_config.use_conditional_feature_extraction = False
        
        # Create baseline model
        baseline_model = Qwen3VLForConditionalGeneration(baseline_config)
        baseline_model.eval()
        
        # Create configuration for optimized model
        optimized_config = Qwen3VLConfig()
        optimized_config.hidden_size = self.config.hidden_size
        optimized_config.num_hidden_layers = self.config.num_hidden_layers
        optimized_config.num_attention_heads = self.config.num_attention_heads
        optimized_config.vocab_size = self.config.vocab_size
        optimized_config.max_position_embeddings = self.config.max_position_embeddings
        
        # Enable optimizations for optimized model
        optimized_config.use_sparsity = True
        optimized_config.sparsity_ratio = 0.5
        optimized_config.exit_threshold = 0.75
        optimized_config.use_gradient_checkpointing = True
        optimized_config.use_moe = True
        optimized_config.moe_num_experts = 4
        optimized_config.moe_top_k = 2
        optimized_config.use_flash_attention_2 = True
        optimized_config.use_dynamic_sparse_attention = True
        optimized_config.use_adaptive_depth = True
        optimized_config.use_context_adaptive_positional_encoding = True
        optimized_config.use_conditional_feature_extraction = True
        
        # Create optimized model
        optimized_model = Qwen3VLForConditionalGeneration(optimized_config)
        optimized_model.eval()
        
        # Copy weights from baseline to optimized model to ensure same starting point
        optimized_model.load_state_dict(baseline_model.state_dict(), strict=False)
        
        return baseline_model, optimized_model
    
    def create_test_inputs(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create standardized test inputs"""
        # Create text input
        input_ids = torch.randint(
            0, self.config.vocab_size, 
            (self.config.batch_size, self.config.sequence_length)
        ).to(device)
        
        # Create image input
        pixel_values = torch.randn(
            self.config.batch_size, 3, 
            self.config.image_size, self.config.image_size
        ).to(device)
        
        return input_ids, pixel_values
    
    def test_forward_pass_similarity(self) -> Dict[str, Any]:
        """Test that forward pass outputs are similar between models"""
        print("Testing forward pass similarity...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create models
        baseline_model, optimized_model = self.create_models()
        baseline_model = baseline_model.to(device)
        optimized_model = optimized_model.to(device)
        
        # Create test inputs
        input_ids, pixel_values = self.create_test_inputs(device)
        
        # Run forward pass on both models
        with torch.no_grad():
            baseline_output = baseline_model(
                input_ids=input_ids, 
                pixel_values=pixel_values
            )
            optimized_output = optimized_model(
                input_ids=input_ids, 
                pixel_values=pixel_values
            )
        
        # Compare outputs
        if isinstance(baseline_output, torch.Tensor):
            baseline_tensor = baseline_output
            optimized_tensor = optimized_output
        else:
            # If output is a dataclass or tuple, compare the first tensor
            baseline_tensor = baseline_output[0] if isinstance(baseline_output, (tuple, list)) else baseline_output.last_hidden_state
            optimized_tensor = optimized_output[0] if isinstance(optimized_output, (tuple, list)) else optimized_output.last_hidden_state
        
        # Calculate metrics
        absolute_diff = torch.abs(baseline_tensor - optimized_tensor)
        relative_diff = absolute_diff / (torch.abs(baseline_tensor) + self.config.tolerance)
        
        max_abs_diff = torch.max(absolute_diff).item()
        max_rel_diff = torch.max(relative_diff).item()
        mean_abs_diff = torch.mean(absolute_diff).item()
        mean_rel_diff = torch.mean(relative_diff).item()
        
        # Check if differences are within acceptable bounds
        abs_diff_acceptable = max_abs_diff < self.config.tolerance
        rel_diff_acceptable = max_rel_diff < self.config.max_relative_error
        
        results = {
            'max_absolute_difference': max_abs_diff,
            'max_relative_difference': max_rel_diff,
            'mean_absolute_difference': mean_abs_diff,
            'mean_relative_difference': mean_rel_diff,
            'absolute_difference_acceptable': abs_diff_acceptable,
            'relative_difference_acceptable': rel_diff_acceptable,
            'baseline_output_shape': list(baseline_tensor.shape),
            'optimized_output_shape': list(optimized_tensor.shape)
        }
        
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Max relative difference: {max_rel_diff:.2e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.2e}")
        print(f"  Mean relative difference: {mean_rel_diff:.2e}")
        print(f"  Difference acceptable: {abs_diff_acceptable and rel_diff_acceptable}")
        
        return results
    
    def test_generation_similarity(self) -> Dict[str, Any]:
        """Test that generation outputs are similar between models"""
        print("Testing generation similarity...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create models
        baseline_model, optimized_model = self.create_models()
        baseline_model = baseline_model.to(device)
        optimized_model = optimized_model.to(device)
        
        # Create test inputs
        input_ids, pixel_values = self.create_test_inputs(device)
        
        # Generate with both models
        with torch.no_grad():
            baseline_generated = baseline_model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=0
            )
            
            optimized_generated = optimized_model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=0
            )
        
        # Compare generated sequences
        generation_match = torch.equal(baseline_generated, optimized_generated)
        sequence_similarity = (baseline_generated == optimized_generated).float().mean().item()
        
        results = {
            'generation_match': generation_match,
            'sequence_similarity': sequence_similarity,
            'baseline_generated_shape': list(baseline_generated.shape),
            'optimized_generated_shape': list(optimized_generated.shape)
        }
        
        print(f"  Generation match: {generation_match}")
        print(f"  Sequence similarity: {sequence_similarity:.4f}")
        
        return results
    
    def test_gradient_flow(self) -> Dict[str, Any]:
        """Test that gradients flow properly through optimized model"""
        print("Testing gradient flow...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create models with gradients enabled
        baseline_model, optimized_model = self.create_models()
        baseline_model = baseline_model.to(device)
        optimized_model = optimized_model.to(device)
        
        # Create test inputs with gradients enabled
        input_ids = torch.randint(
            0, self.config.vocab_size, 
            (self.config.batch_size, self.config.sequence_length),
            device=device,
            requires_grad=False
        )
        pixel_values = torch.randn(
            self.config.batch_size, 3, 
            self.config.image_size, self.config.image_size,
            device=device,
            requires_grad=True
        )
        
        # Test gradient flow in optimized model
        optimized_model.train()
        output = optimized_model(input_ids=input_ids, pixel_values=pixel_values)
        
        # Calculate a dummy loss
        if isinstance(output, torch.Tensor):
            loss = output.mean()
        else:
            loss = output[0].mean() if isinstance(output, (tuple, list)) else output.last_hidden_state.mean()
        
        # Backpropagate
        loss.backward()
        
        # Check if gradients exist for parameters
        param_count = 0
        grad_count = 0
        for param in optimized_model.parameters():
            param_count += 1
            if param.grad is not None:
                grad_count += 1
        
        gradient_flow_ok = grad_count == param_count and grad_count > 0
        
        results = {
            'total_parameters': param_count,
            'parameters_with_gradients': grad_count,
            'gradient_flow_ok': gradient_flow_ok
        }
        
        print(f"  Parameters with gradients: {grad_count}/{param_count}")
        print(f"  Gradient flow OK: {gradient_flow_ok}")
        
        return results
    
    def test_numerical_stability(self) -> Dict[str, Any]:
        """Test numerical stability of optimized model"""
        print("Testing numerical stability...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create optimized model
        _, optimized_model = self.create_models()
        optimized_model = optimized_model.to(device)
        optimized_model.eval()
        
        # Create test inputs
        input_ids, pixel_values = self.create_test_inputs(device)
        
        # Run multiple forward passes to check for consistency
        outputs = []
        for _ in range(5):
            with torch.no_grad():
                output = optimized_model(input_ids=input_ids, pixel_values=pixel_values)
                if isinstance(output, torch.Tensor):
                    outputs.append(output.clone())
                else:
                    outputs.append(output[0].clone() if isinstance(output, (tuple, list)) else output.last_hidden_state.clone())
        
        # Calculate variance across runs
        if len(outputs) > 1:
            output_stack = torch.stack(outputs)
            variance = torch.var(output_stack, dim=0)
            max_variance = torch.max(variance).item()
            mean_variance = torch.mean(variance).item()
        else:
            max_variance = 0
            mean_variance = 0
        
        # Check for NaN or infinity
        has_nan = any(torch.isnan(output).any().item() for output in outputs)
        has_inf = any(torch.isinf(output).any().item() for output in outputs)
        
        numerical_stability_ok = not (has_nan or has_inf) and max_variance < 1e-6
        
        results = {
            'max_variance': max_variance,
            'mean_variance': mean_variance,
            'has_nan': has_nan,
            'has_inf': has_inf,
            'numerical_stability_ok': numerical_stability_ok
        }
        
        print(f"  Max variance across runs: {max_variance:.2e}")
        print(f"  Has NaN: {has_nan}")
        print(f"  Has Inf: {has_inf}")
        print(f"  Numerical stability OK: {numerical_stability_ok}")
        
        return results
    
    def run_all_accuracy_tests(self) -> Dict[str, Any]:
        """Run all accuracy preservation tests"""
        print("=" * 80)
        print("ACCURACY PRESERVATION TESTS FOR QWEN3-VL OPTIMIZATIONS")
        print("=" * 80)
        
        results = {}
        
        # Run forward pass similarity test
        results['forward_pass_similarity'] = self.test_forward_pass_similarity()
        
        # Run generation similarity test
        results['generation_similarity'] = self.test_generation_similarity()
        
        # Run gradient flow test
        results['gradient_flow'] = self.test_gradient_flow()
        
        # Run numerical stability test
        results['numerical_stability'] = self.test_numerical_stability()
        
        # Generate summary
        summary = self.generate_summary(results)
        
        print("\n" + "=" * 80)
        print("ACCURACY PRESERVATION TEST SUMMARY")
        print("=" * 80)
        
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        return {
            'results': results,
            'summary': summary
        }
    
    def generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of all accuracy test results"""
        summary = {}
        
        # Forward pass similarity summary
        if 'forward_pass_similarity' in results:
            forward_results = results['forward_pass_similarity']
            summary['forward_pass_similar'] = (
                forward_results['absolute_difference_acceptable'] and 
                forward_results['relative_difference_acceptable']
            )
            summary['max_abs_diff_forward'] = forward_results['max_absolute_difference']
            summary['max_rel_diff_forward'] = forward_results['max_relative_difference']
        
        # Generation similarity summary
        if 'generation_similarity' in results:
            gen_results = results['generation_similarity']
            summary['generation_similar'] = gen_results['generation_match']
            summary['sequence_similarity_score'] = gen_results['sequence_similarity']
        
        # Gradient flow summary
        if 'gradient_flow' in results:
            grad_results = results['gradient_flow']
            summary['gradient_flow_preserved'] = grad_results['gradient_flow_ok']
        
        # Numerical stability summary
        if 'numerical_stability' in results:
            stability_results = results['numerical_stability']
            summary['numerically_stable'] = stability_results['numerical_stability_ok']
        
        # Overall accuracy preservation
        overall_pass = (
            summary.get('forward_pass_similar', True) and
            summary.get('gradient_flow_preserved', True) and
            summary.get('numerically_stable', True)
        )
        summary['overall_accuracy_preserved'] = overall_pass
        
        return summary


def run_accuracy_preservation_tests():
    """Run all accuracy preservation tests"""
    config = AccuracyTestConfig()
    tester = AccuracyPreservationTester(config)
    
    results = tester.run_all_accuracy_tests()
    
    return results


if __name__ == "__main__":
    results = run_accuracy_preservation_tests()