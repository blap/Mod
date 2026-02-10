"""
Unit tests for Qwen3-0.6B Plugin
"""
import unittest
import sys
import os

# Adicionando o caminho para o módulo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from src.inference_pio.models.qwen3_0_6b.plugin import Qwen3_0_6B_Plugin
from src.inference_pio.models.qwen3_0_6b.config import Qwen3_0_6B_Config


class TestQwen3_0_6B_Plugin(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.plugin = Qwen3_0_6B_Plugin()
    
    def test_plugin_initialization(self):
        """Test that the plugin initializes correctly."""
        self.assertIsNotNone(self.plugin)
        self.assertIsNotNone(self.plugin.metadata)
        self.assertEqual(self.plugin.metadata.name, "Qwen3-0.6B")
        self.assertEqual(self.plugin.metadata.model_size, "0.6B")
    
    def test_get_model_info(self):
        """Test getting model info."""
        info = self.plugin.get_model_info()
        self.assertIn("name", info)
        self.assertEqual(info["name"], "Qwen3-0.6B")
    
    def test_get_model_parameters(self):
        """Test getting model parameters."""
        params = self.plugin.get_model_parameters()
        self.assertIn("total_parameters", params)
        self.assertEqual(params["total_parameters"], 600000000)
    
    def test_tokenize_and_detokenize_with_fallback(self):
        """Test tokenize and detokenize methods with fallback."""
        text = "Hello world"
        tokens = self.plugin.tokenize(text)
        # Even if tokenizer is not loaded, it should return a list
        self.assertIsInstance(tokens, list)
        
        # Test detokenize with empty list (fallback case)
        detokenized = self.plugin.detokenize([])
        self.assertIsInstance(detokenized, str)


if __name__ == '__main__':
    unittest.main()