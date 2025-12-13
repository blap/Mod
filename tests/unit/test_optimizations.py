"""
Comprehensive Tests for Qwen3-VL Optimizations
Tests for INT8 quantization, sparsification, pruning, and adaptive precision optimizations
"""
import unittest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import sys
import os

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from int8_quantization_optimization import INT8Quantizer, INT8QuantizationConfig, apply_int8_quantization_to_model
from visual_token_sparsification import VisualTokenSparsifier, SparsificationConfig, apply_visual_token_sparsification
from model_pruning_optimization import ModelPruner, PruningConfig, apply_pruning_to_model
from adaptive_precision_optimization import AdaptivePrecisionController, AdaptivePrecisionConfig, apply_adaptive_precision_to_model


class MockModel(nn.Module):
    """Mock model for testing purposes."""
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(1000, 512)
        self.linear1 = nn.Linear(512, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.mlp = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512)
        )
        self.lm_head = nn.Linear(512, 1000)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed_tokens(input_ids)
        x = self.linear1(x)
        x = self.linear2(x)
        
        # Self-attention
        x_attn, _ = self.attention(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
        x = x + x_attn.transpose(0, 1)
        
        # MLP
        x = x + self.mlp(x)
        
        # Output
        output = self.lm_head(x)
        return output


class TestINT8Quantization(unittest.TestCase):
    """Test cases for INT8 quantization."""
    
    def setUp(self):
        self.model = MockModel()
        self.config = INT8QuantizationConfig(
            quantization_mode="static",
            activation_bits=8,
            weight_bits=8,
            quantize_embeddings=True,
            quantize_attention=True,
            quantize_mlp=True,
            quantize_ln=False
        )

    def test_quantization_application(self):
        """Test that quantization can be applied to the model."""
        quantized_model, info = apply_int8_quantization_to_model(self.model, self.config)
        
        # Check that the model was returned
        self.assertIsNotNone(quantized_model)
        self.assertIsInstance(info, dict)
        
        # Check that quantization info contains expected keys
        expected_keys = [
            'config', 'quantization_mode', 'weight_bits', 
            'activation_bits', 'quantize_embeddings', 
            'quantize_attention', 'quantize_mlp', 'quantize_ln'
        ]
        for key in expected_keys:
            self.assertIn(key, info)

    def test_quantization_metrics(self):
        """Test quantization metrics calculation."""
        quantizer = INT8Quantizer(self.config)
        
        # Create sample tensors
        original_tokens = torch.randn(2, 100, 512)
        sparsified_tokens = torch.randn(2, 50, 512)  # Half the tokens
        
        metrics = quantizer.calculate_sparsity_metrics(original_tokens, sparsified_tokens)
        
        # Check that metrics are calculated correctly
        self.assertIn('token_reduction_ratio', metrics)
        self.assertIn('memory_reduction_percent', metrics)
        self.assertGreaterEqual(metrics['token_reduction_ratio'], 0.0)
        self.assertLessEqual(metrics['token_reduction_ratio'], 1.0)


class TestVisualTokenSparsification(unittest.TestCase):
    """Test cases for visual token sparsification."""
    
    def setUp(self):
        self.model = MockModel()
        self.config = SparsificationConfig(
            sparsity_ratio=0.5,
            sparsity_method="top_k",
            min_tokens_per_image=16,
            max_tokens_per_image=256,
            apply_to_vision_encoder=False,  # Since our mock model doesn't have a vision encoder
            apply_to_cross_attention=False,
            apply_to_self_attention=False
        )

    def test_sparsification_application(self):
        """Test that sparsification can be applied to the model."""
        sparsified_model, info = apply_visual_token_sparsification(self.model, self.config)
        
        # Check that the model was returned
        self.assertIsNotNone(sparsified_model)
        self.assertIsInstance(info, dict)
        
        # Check that sparsification info contains expected keys
        expected_keys = [
            'config', 'sparsity_ratio', 'sparsity_method',
            'apply_to_vision_encoder', 'apply_to_cross_attention', 'apply_to_self_attention'
        ]
        for key in expected_keys:
            self.assertIn(key, info)

    def test_sparsification_metrics(self):
        """Test sparsification metrics calculation."""
        sparsifier = VisualTokenSparsifier(self.config)
        
        # Create sample tensors
        original_tokens = torch.randn(2, 100, 512)
        sparsified_tokens = torch.randn(2, 50, 512)  # Half the tokens
        
        metrics = sparsifier.calculate_sparsity_metrics(original_tokens, sparsified_tokens)
        
        # Check that metrics are calculated correctly
        self.assertIn('token_reduction_ratio', metrics)
        self.assertIn('memory_reduction_percent', metrics)
        self.assertGreaterEqual(metrics['token_reduction_ratio'], 0.0)
        self.assertLessEqual(metrics['token_reduction_ratio'], 1.0)


class TestModelPruning(unittest.TestCase):
    """Test cases for model pruning."""
    
    def setUp(self):
        self.model = MockModel()
        self.config = PruningConfig(
            pruning_method="unstructured",
            pruning_ratio=0.2,
            pruning_schedule="one_shot",
            num_pruning_steps=1,
            initial_sparsity=0.0,
            prune_embeddings=False,
            prune_attention=True,
            prune_mlp=True,
            prune_output_layers=False
        )

    def test_pruning_application(self):
        """Test that pruning can be applied to the model."""
        pruned_model, info = apply_pruning_to_model(self.model, self.config)
        
        # Check that the model was returned
        self.assertIsNotNone(pruned_model)
        self.assertIsInstance(info, dict)
        
        # Check that pruning info contains expected keys
        expected_keys = [
            'config', 'pruning_method', 'pruning_ratio', 
            'pruning_schedule', 'prune_embeddings', 
            'prune_attention', 'prune_mlp', 'prune_output_layers'
        ]
        for key in expected_keys:
            self.assertIn(key, info)

    def test_pruning_metrics(self):
        """Test pruning metrics calculation."""
        pruner = ModelPruner(self.config)
        
        # Calculate metrics
        metrics = pruner.calculate_pruning_metrics(self.model, self.model)
        
        # Check that metrics are calculated
        self.assertIn('original_parameter_count', metrics)
        self.assertIn('pruned_parameter_count', metrics)
        self.assertIn('zero_parameter_count', metrics)
        self.assertIn('parameter_reduction_percent', metrics)
        
        # For same models, parameter counts should be the same
        self.assertEqual(metrics['original_parameter_count'], metrics['pruned_parameter_count'])


class TestAdaptivePrecision(unittest.TestCase):
    """Test cases for adaptive precision."""
    
    def setUp(self):
        self.model = MockModel()
        self.config = AdaptivePrecisionConfig(
            base_precision="fp16",
            enable_dynamic_precision=True,
            min_precision="int8",
            max_precision="fp32",
            enable_layerwise_precision=True,
            enable_input_adaptive_precision=True,
            enable_system_adaptive_precision=True
        )

    def test_adaptive_precision_application(self):
        """Test that adaptive precision can be applied to the model."""
        adaptive_model, info = apply_adaptive_precision_to_model(self.model, self.config)
        
        # Check that the model was returned
        self.assertIsNotNone(adaptive_model)
        self.assertIsInstance(info, dict)
        
        # Check that precision info contains expected keys
        expected_keys = [
            'config', 'base_precision', 'enable_dynamic_precision',
            'min_precision', 'max_precision', 'enable_layerwise_precision',
            'enable_input_adaptive_precision', 'enable_system_adaptive_precision'
        ]
        for key in expected_keys:
            self.assertIn(key, info)

    def test_precision_manager(self):
        """Test the precision manager functionality."""
        from adaptive_precision_optimization import PrecisionManager
        
        manager = PrecisionManager(self.config)
        
        # Test getting precision for different layer types
        embedding_precision = manager.get_precision_for_layer("embed_tokens")
        attention_precision = manager.get_precision_for_layer("attention")
        mlp_precision = manager.get_precision_for_layer("mlp.0")
        output_precision = manager.get_precision_for_layer("lm_head")
        
        # Check that precisions are valid
        self.assertIn(embedding_precision, [torch.float32, torch.float16, torch.bfloat16, torch.int8])
        self.assertIn(attention_precision, [torch.float32, torch.float16, torch.bfloat16, torch.int8])
        self.assertIn(mlp_precision, [torch.float32, torch.float16, torch.bfloat16, torch.int8])
        self.assertIn(output_precision, [torch.float32, torch.float16, torch.bfloat16, torch.int8])


class TestIntegration(unittest.TestCase):
    """Integration tests for all optimizations."""
    
    def setUp(self):
        self.model = MockModel()
        self.int8_config = INT8QuantizationConfig(quantization_mode="static")
        self.sparsification_config = SparsificationConfig(sparsity_ratio=0.5)
        self.pruning_config = PruningConfig(pruning_ratio=0.2, pruning_schedule="one_shot")
        self.adaptive_precision_config = AdaptivePrecisionConfig(base_precision="fp16")

    def test_sequential_optimizations(self):
        """Test applying optimizations sequentially."""
        # Apply INT8 quantization
        quantized_model, _ = apply_int8_quantization_to_model(self.model, self.int8_config)
        
        # Apply sparsification
        sparsified_model, _ = apply_visual_token_sparsification(quantized_model, self.sparsification_config)
        
        # Apply pruning
        pruned_model, _ = apply_pruning_to_model(sparsified_model, self.pruning_config)
        
        # Apply adaptive precision
        adaptive_model, _ = apply_adaptive_precision_to_model(pruned_model, self.adaptive_precision_config)
        
        # Check that the final model is not None
        self.assertIsNotNone(adaptive_model)

    def test_optimization_metrics(self):
        """Test metrics calculation for optimizations."""
        # Apply all optimizations
        model, _ = apply_int8_quantization_to_model(self.model, self.int8_config)
        model, _ = apply_visual_token_sparsification(model, self.sparsification_config)
        model, _ = apply_pruning_to_model(model, self.pruning_config)
        model, _ = apply_adaptive_precision_to_model(model, self.adaptive_precision_config)
        
        # Create dummy data loader for benchmarking
        class DummyDataLoader:
            def __iter__(self):
                # Create a single batch of dummy data
                input_ids = torch.randint(0, 1000, (2, 10))
                yield {'input_ids': input_ids}
        
        dummy_loader = DummyDataLoader()
        
        # Test benchmarking functions (these will run but may not produce meaningful results with mock model)
        from int8_quantization_optimization import INT8Quantizer
        from visual_token_sparsification import VisualTokenSparsifier
        from model_pruning_optimization import ModelPruner
        from adaptive_precision_optimization import AdaptivePrecisionController
        
        # Test INT8 quantization benchmarking
        int8_quantizer = INT8Quantizer(self.int8_config)
        int8_metrics = int8_quantizer.benchmark_quantization_impact(self.model, model, dummy_loader)
        
        # Test sparsification benchmarking
        sparsifier = VisualTokenSparsifier(self.sparsification_config)
        sparsity_metrics = sparsifier.benchmark_sparsification_impact(self.model, model, dummy_loader)
        
        # Test pruning benchmarking
        pruner = ModelPruner(self.pruning_config)
        pruning_metrics = pruner.benchmark_pruning_impact(self.model, model, dummy_loader)
        
        # Test adaptive precision benchmarking
        adaptive_controller = AdaptivePrecisionController(self.adaptive_precision_config)
        adaptive_metrics = adaptive_controller.benchmark_adaptive_precision_impact(self.model, model, dummy_loader)
        
        # Check that metrics dictionaries have required keys
        required_keys = ['original_avg_time_ms', 'speedup', 'original_model_size_mb', 'num_test_batches']
        for metrics in [int8_metrics, sparsity_metrics, pruning_metrics, adaptive_metrics]:
            for key in required_keys:
                self.assertIn(key, metrics)


def run_tests():
    """Run all tests."""
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTest(unittest.makeSuite(TestINT8Quantization))
    suite.addTest(unittest.makeSuite(TestVisualTokenSparsification))
    suite.addTest(unittest.makeSuite(TestModelPruning))
    suite.addTest(unittest.makeSuite(TestAdaptivePrecision))
    suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    
    return result


if __name__ == "__main__":
    print("Running comprehensive tests for Qwen3-VL optimizations...")
    print("=" * 60)
    
    # Run tests
    test_result = run_tests()
    
    if test_result.wasSuccessful():
        print("\nAll tests passed! ✓")
    else:
        print("\nSome tests failed! ✗")
        for failure in test_result.failures:
            print(f"FAILURE: {failure[0]}")
            print(f"Details: {failure[1]}")
        for error in test_result.errors:
            print(f"ERROR: {error[0]}")
            print(f"Details: {error[1]}")