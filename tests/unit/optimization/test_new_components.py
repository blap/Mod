
import unittest
import torch
import sys
import os

# Adjust path to find the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.qwen3_vl.optimization.dynamic_sparse import DynamicSparseAttention, VisionDynamicSparseAttention
from src.qwen3_vl.optimization.adaptive_computation import AdaptiveAttention, AdaptiveMLP

class MockConfig:
    def __init__(self):
        self.hidden_size = 64
        self.num_attention_heads = 4
        self.attention_bias = True
        self.intermediate_size = 256
        self.sparsity_factor = 2
        self.local_window_size = 4
        self.adaptive_threshold = 0.5
        self.mlp_importance_threshold = 0.5

class TestOptimizationComponents(unittest.TestCase):
    def setUp(self):
        self.config = MockConfig()
        self.device = torch.device("cpu")

    def test_dynamic_sparse_attention(self):
        print("\n1. Testing DynamicSparseAttention...")
        model = DynamicSparseAttention(self.config).to(self.device)

        # Batch=2, Seq=16, Dim=64
        hidden_states = torch.randn(2, 16, 64).to(self.device)

        output, _, _ = model(hidden_states)

        self.assertEqual(output.shape, (2, 16, 64))
        print("  Forward pass successful")

    def test_adaptive_attention(self):
        print("\n2. Testing AdaptiveAttention...")
        model = AdaptiveAttention(self.config).to(self.device)

        hidden_states = torch.randn(2, 16, 64).to(self.device)

        output, _, _ = model(hidden_states)
        self.assertEqual(output.shape, (2, 16, 64))
        print("  Forward pass successful")

    def test_adaptive_mlp(self):
        print("\n3. Testing AdaptiveMLP...")
        model = AdaptiveMLP(self.config).to(self.device)

        hidden_states = torch.randn(2, 16, 64).to(self.device)

        output = model(hidden_states)
        self.assertEqual(output.shape, (2, 16, 64))
        print("  Forward pass successful")

if __name__ == '__main__':
    unittest.main()
