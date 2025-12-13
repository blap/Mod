"""
Standardized Test Suite for Qwen3-VL Model Components

This module provides a comprehensive, standardized test suite for all Qwen3-VL model components
following pytest best practices and ensuring consistency across all tests.
"""

import pytest
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
import warnings
from dataclasses import dataclass
import sys
from pathlib import Path

# Add the project root to the path to import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestConfig:
    """Standardized tests for Qwen3-VL configuration."""

    def test_config_initialization(self):
        """Test basic configuration initialization."""
        from src.qwen3_vl.config.config import Qwen3VLConfig
        
        config = Qwen3VLConfig(
            hidden_size=1024,
            num_attention_heads=32,  # Maintaining 32 attention heads as required
            num_key_value_heads=32,
            max_position_embeddings=2048,
            vocab_size=50000
        )

        assert config.hidden_size == 1024
        assert config.num_attention_heads == 32  # Confirming 32 attention heads are preserved
        assert config.num_key_value_heads == 32
        assert config.max_position_embeddings == 2048
        assert config.vocab_size == 50000

    def test_config_default_values(self):
        """Test configuration with default values."""
        from src.qwen3_vl.config.config import Qwen3VLConfig
        
        config = Qwen3VLConfig()

        # Check that default values are properly set
        assert config.hidden_size > 0
        assert config.num_attention_heads > 0  # Should be 32 for full capacity
        # num_key_value_heads can be None, which is valid
        if config.num_key_value_heads is not None:
            assert config.num_key_value_heads >= 0

    def test_config_capacity_preservation(self):
        """Test that model capacity is preserved (32 transformer layers and 32 attention heads)."""
        from src.qwen3_vl.config.config import Qwen3VLConfig
        
        config = Qwen3VLConfig()

        # Verify that the configuration maintains the required capacity
        assert config.num_hidden_layers == 32, f"num_hidden_layers must be 32 to preserve full capacity, got {config.num_hidden_layers}"
        assert config.num_attention_heads == 32, f"num_attention_heads must be 32 to preserve full capacity, got {config.num_attention_heads}"
        assert config.vision_num_hidden_layers == 24, f"vision_num_hidden_layers must be 24, got {config.vision_num_hidden_layers}"
        assert config.vision_num_attention_heads == 16, f"vision_num_attention_heads must be 16, got {config.vision_num_attention_heads}"


class TestModelInitialization:
    """Tests for model initialization."""

    def test_model_creation(self):
        """Test that the model can be created successfully."""
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        
        config = Qwen3VLConfig(
            hidden_size=256,  # Smaller for testing
            num_attention_heads=8,  # Reduced for testing
            num_key_value_heads=8,
            max_position_embeddings=128,
            attention_dropout_prob=0.0,
            num_hidden_layers=4,  # Reduced for testing
            vision_num_hidden_layers=4,  # Reduced for testing
            vocab_size=1000,
            vision_hidden_size=256,
            vision_num_attention_heads=8
        )
        
        model = Qwen3VLForConditionalGeneration(config)
        
        # Verify model components are created
        assert hasattr(model, 'vision_tower')
        assert hasattr(model, 'multi_modal_projector')
        assert hasattr(model, 'language_model')
        assert hasattr(model, 'config')
        
        # Verify layer counts
        assert len(model.language_model.layers) == config.num_hidden_layers
        assert len(model.vision_tower.layers) == config.vision_num_hidden_layers
        
        # Verify attention head counts
        assert model.config.num_attention_heads == config.num_attention_heads

    def test_model_forward_pass(self):
        """Test that the model can perform a forward pass."""
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        
        config = Qwen3VLConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=128,
            attention_dropout_prob=0.0,
            num_hidden_layers=2,  # Small for testing
            vision_num_hidden_layers=2,  # Small for testing
            vocab_size=1000,
            vision_hidden_size=256,
            vision_num_attention_heads=8
        )
        model = Qwen3VLForConditionalGeneration(config)
        model.eval()

        # Create test inputs
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)  # Standard image size

        # Forward pass
        with torch.no_grad():
            output = model(input_ids=input_ids, pixel_values=pixel_values)

        # Check output shape
        assert output.shape[0] == batch_size
        assert torch.all(torch.isfinite(output)), "Output should contain finite values"

    def test_text_only_forward_pass(self):
        """Test that the model can handle text-only inputs."""
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        
        config = Qwen3VLConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=128,
            attention_dropout_prob=0.0,
            num_hidden_layers=2,  # Small for testing
            vision_num_hidden_layers=2,  # Small for testing
            vocab_size=1000,
            vision_hidden_size=256,
            vision_num_attention_heads=8
        )
        model = Qwen3VLForConditionalGeneration(config)
        model.eval()

        # Create text-only inputs
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        with torch.no_grad():
            output = model(input_ids=input_ids)

        # Check output shape
        assert output.shape[0] == batch_size
        assert torch.all(torch.isfinite(output)), "Output should contain finite values"

    def test_image_only_forward_pass(self):
        """Test that the model can handle image-only inputs."""
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        
        config = Qwen3VLConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=128,
            attention_dropout_prob=0.0,
            num_hidden_layers=2,  # Small for testing
            vision_num_hidden_layers=2,  # Small for testing
            vocab_size=1000,
            vision_hidden_size=256,
            vision_num_attention_heads=8
        )
        model = Qwen3VLForConditionalGeneration(config)
        model.eval()

        # Create image-only inputs
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224)

        # Forward pass
        with torch.no_grad():
            output = model(pixel_values=pixel_values)

        # Check output shape
        assert output.shape[0] == batch_size
        assert torch.all(torch.isfinite(output)), "Output should contain finite values"


class TestModelCapacity:
    """Tests for model capacity preservation."""

    def test_full_capacity_preservation(self):
        """Test that full model capacity is preserved (32 layers and 32 attention heads)."""
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        
        # Create a config with full capacity
        config = Qwen3VLConfig()

        # Verify the config has the expected values
        assert config.num_hidden_layers == 32, f"Expected 32 hidden layers, got {config.num_hidden_layers}"
        assert config.num_attention_heads == 32, f"Expected 32 attention heads, got {config.num_attention_heads}"

        # Create the model
        model = Qwen3VLForConditionalGeneration(config)

        # Verify the model has the expected number of layers
        assert len(model.language_model.layers) == config.num_hidden_layers, \
            f"Model should have {config.num_hidden_layers} layers, got {len(model.language_model.layers)}"

        # Verify the model has the expected number of attention heads
        assert model.config.num_attention_heads == config.num_attention_heads, \
            f"Model should have {config.num_attention_heads} attention heads, got {model.config.num_attention_heads}"

        # Verify the vision model also maintains its capacity
        assert len(model.vision_tower.layers) == config.vision_num_hidden_layers, \
            f"Vision model should have {config.vision_num_hidden_layers} layers, got {len(model.vision_tower.layers)}"

        print(f"V Model capacity preserved: {config.num_hidden_layers} layers, {config.num_attention_heads} heads")


class TestModelGeneration:
    """Tests for model generation functionality."""

    def test_generate_text_only(self):
        """Test text-only generation."""
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        
        config = Qwen3VLConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            attention_dropout_prob=0.0,
            num_hidden_layers=2,  # Small for testing
            vision_num_hidden_layers=2,  # Small for testing
            vocab_size=500,
            vision_hidden_size=128,
            vision_num_attention_heads=4
        )
        model = Qwen3VLForConditionalGeneration(config)
        model.eval()

        # Create input
        batch_size, seq_len = 1, 5
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Generate
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=10,
                do_sample=False
            )

        # Check output shape
        assert generated.shape[0] == batch_size
        assert generated.shape[1] >= seq_len  # Should have at least the original tokens plus new ones
        assert torch.all(generated >= 0) and torch.all(generated < config.vocab_size), "Generated tokens should be valid"

    def test_generate_multimodal(self):
        """Test multimodal generation."""
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        
        config = Qwen3VLConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            attention_dropout_prob=0.0,
            num_hidden_layers=2,  # Small for testing
            vision_num_hidden_layers=2,  # Small for testing
            vocab_size=500,
            vision_hidden_size=128,
            vision_num_attention_heads=4
        )
        model = Qwen3VLForConditionalGeneration(config)
        model.eval()

        # Create inputs
        batch_size, seq_len = 1, 5
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)

        # Generate
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=10,
                do_sample=False
            )

        # Check output shape
        assert generated.shape[0] == batch_size
        assert torch.all(generated >= 0) and torch.all(generated < config.vocab_size), "Generated tokens should be valid"


# Run tests if this file is executed directly
if __name__ == "__main__":
    import unittest
    
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add tests to the suite
    suite.addTest(unittest.makeSuite(TestConfig))
    suite.addTest(unittest.makeSuite(TestModelInitialization))
    suite.addTest(unittest.makeSuite(TestModelCapacity))
    suite.addTest(unittest.makeSuite(TestModelGeneration))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print results
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun}")