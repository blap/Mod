"""
Test suite for Qwen3 model configurations.

This module contains tests for the configuration classes of Qwen3 models,
ensuring they follow the self-contained configuration pattern and properly
detect and use the H drive when available.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from src.inference_pio.models.qwen3_0_6b.config import Qwen3_0_6B_Config, Qwen3_0_6B_DynamicConfig
from src.inference_pio.models.qwen3_coder_next.config import Qwen3CoderNextConfig, Qwen3CoderNextDynamicConfig


class TestQwen3_0_6B_Config(unittest.TestCase):
    """Test cases for Qwen3-0.6B configuration."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = Qwen3_0_6B_Config()

    def test_model_name_and_path(self):
        """Test that model name and path are set correctly."""
        self.assertEqual(self.config.model_name, "qwen3_0_6b")
        # Check that path is set to H drive by default
        self.assertEqual(self.config.model_path, "H:/Qwen3-0.6B")

    def test_architecture_parameters(self):
        """Test that architecture parameters are set correctly."""
        self.assertEqual(self.config.hidden_size, 896)
        self.assertEqual(self.config.num_attention_heads, 14)
        self.assertEqual(self.config.num_key_value_heads, 2)
        self.assertEqual(self.config.num_hidden_layers, 24)
        self.assertEqual(self.config.intermediate_size, 4864)
        self.assertEqual(self.config.vocab_size, 151936)
        self.assertEqual(self.config.max_position_embeddings, 32768)
        self.assertEqual(self.config.rms_norm_eps, 1e-6)
        self.assertEqual(self.config.rope_theta, 1000000.0)

    def test_thinking_mode_settings(self):
        """Test that thinking mode settings are configured."""
        self.assertTrue(self.config.enable_thinking)
        self.assertEqual(self.config.thinking_temperature, 0.6)
        self.assertEqual(self.config.thinking_top_p, 0.95)
        self.assertEqual(self.config.thinking_top_k, 20)

    def test_get_model_specific_params(self):
        """Test that model-specific parameters are returned correctly."""
        params = self.config.get_model_specific_params()
        self.assertIn('hidden_size', params)
        self.assertIn('num_attention_heads', params)
        self.assertIn('vocab_size', params)
        self.assertEqual(params['hidden_size'], 896)

    def test_dynamic_config_creation(self):
        """Test that dynamic config can be created."""
        dynamic_config = Qwen3_0_6B_DynamicConfig()
        self.assertIsInstance(dynamic_config, Qwen3_0_6B_Config)


class TestQwen3CoderNextConfig(unittest.TestCase):
    """Test cases for Qwen3-Coder-Next configuration."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = Qwen3CoderNextConfig()

    def test_model_name_and_path(self):
        """Test that model name and path are set correctly."""
        self.assertEqual(self.config.model_name, "qwen3_coder_next")
        # Check that path is set to H drive by default
        self.assertEqual(self.config.model_path, "H:/Qwen3-Coder-Next")

    def test_architecture_parameters(self):
        """Test that architecture parameters are set correctly."""
        self.assertEqual(self.config.hidden_size, 2048)
        self.assertEqual(self.config.num_hidden_layers, 48)
        self.assertEqual(self.config.max_position_embeddings, 262144)
        self.assertEqual(self.config.vocab_size, 152064)
        self.assertEqual(self.config.layer_norm_eps, 1e-6)
        self.assertEqual(self.config.rope_theta, 1000000.0)

    def test_hybrid_architecture(self):
        """Test that hybrid architecture parameters are set correctly."""
        self.assertEqual(len(self.config.hybrid_block_pattern), 48)  # Should match num_hidden_layers
        self.assertEqual(self.config.num_attention_heads, 16)
        self.assertEqual(self.config.num_key_value_heads, 2)
        self.assertEqual(self.config.deltanet_value_heads, 32)
        self.assertEqual(self.config.num_experts, 512)
        self.assertEqual(self.config.num_activated_experts, 10)

    def test_thinking_mode_disabled(self):
        """Test that thinking mode is disabled for coder model."""
        self.assertFalse(self.config.thinking_mode)
        self.assertFalse(self.config.enable_thinking)

    def test_get_model_specific_params(self):
        """Test that model-specific parameters are returned correctly."""
        params = self.config.get_model_specific_params()
        self.assertIn('hidden_size', params)
        self.assertIn('num_hidden_layers', params)
        self.assertIn('num_experts', params)
        self.assertEqual(params['hidden_size'], 2048)

    def test_dynamic_config_creation(self):
        """Test that dynamic config can be created."""
        dynamic_config = Qwen3CoderNextDynamicConfig()
        self.assertIsInstance(dynamic_config, Qwen3CoderNextConfig)


class TestHDriveDetection(unittest.TestCase):
    """Test cases for H drive detection logic."""

    @patch('os.path.exists')
    def test_h_drive_detection_for_qwen3_0_6b(self, mock_exists):
        """Test that H drive path is used when available."""
        mock_exists.return_value = True
        config = Qwen3_0_6B_Config(model_path="")
        self.assertEqual(config.model_path, "H:/Qwen3-0.6B")

    @patch('os.path.exists')
    def test_h_drive_detection_for_qwen3_coder_next(self, mock_exists):
        """Test that H drive path is used when available."""
        mock_exists.return_value = True
        config = Qwen3CoderNextConfig(model_path="")
        self.assertEqual(config.model_path, "H:/Qwen3-Coder-Next")


if __name__ == '__main__':
    unittest.main()