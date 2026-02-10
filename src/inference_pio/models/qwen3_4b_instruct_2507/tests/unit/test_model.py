"""
Unit tests for Qwen3-4B-Instruct-2507 Model
"""
import unittest
import sys
import os

# Adicionando o caminho para o módulo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from src.inference_pio.models.qwen3_4b_instruct_2507.model import Qwen3_4B_Instruct_2507_Model
from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen3_4B_Instruct_2507_Config


class TestQwen3_4B_Instruct_2507_Model(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = Qwen3_4B_Instruct_2507_Config()
        self.model = Qwen3_4B_Instruct_2507_Model(self.config)
    
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model.config)
        self.assertEqual(self.model.config.hidden_size, 4096)
        self.assertEqual(self.model.config.num_attention_heads, 32)
        self.assertEqual(self.model.config.num_hidden_layers, 32)
    
    def test_forward_method_exists(self):
        """Test that the model has a forward method."""
        self.assertTrue(hasattr(self.model, 'forward'))
        self.assertTrue(callable(getattr(self.model, 'forward')))
    
    def test_generate_method_exists(self):
        """Test that the model has a generate method."""
        self.assertTrue(hasattr(self.model, 'generate'))
        self.assertTrue(callable(getattr(self.model, 'generate')))


if __name__ == '__main__':
    unittest.main()