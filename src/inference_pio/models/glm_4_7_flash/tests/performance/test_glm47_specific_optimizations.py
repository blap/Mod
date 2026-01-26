"""
Comprehensive Test Suite for GLM-4.7-Flash Specific Optimizations with Real Parameters

This module tests the GLM-4.7-Flash specific optimizations implemented in the model
using the actual model parameters and architecture.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
import sys
import os

# Add the project root to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.inference_pio.models.glm_4_7_flash.optimizations.glm_specific_optimizations import (
    GLM47OptimizationConfig,
    GLM47AttentionOptimizer,
    GLM47FFNOptimizer,
    GLM47LayerNormOptimizer,
    GLM47ResidualOptimizer,
    GLM47RotaryEmbedding,
    GLM47AttentionPatternOptimizer,
    GLM47KVCachemanager,
    GLM47GroupProcessor,
    GLM47SwiGLU,
    apply_glm47_specific_optimizations,
    get_glm47_optimization_report
)
from src.inference_pio.models.glm_4_7.config import GLM47FlashConfig

# TestGLM47OptimizationConfig

    """Test cases for GLM47OptimizationConfig."""
    
    def default_config_values(self)():
        """Test that default configuration values are set correctly."""
        config = GLM47OptimizationConfig()
        
        assert_true(config.use_glm_attention_patterns)
        assert_equal(config.glm_attention_pattern_sparsity)
        assert_equal(config.glm_attention_window_size, 1024)
        assert_true(config.use_glm_ffn_optimization)
        assert_equal(config.glm_ffn_expansion_ratio)
        assert_equal(config.glm_ffn_group_size, 128)
        assert_true(config.use_glm_memory_efficient_kv)
        assert_equal(config.glm_kv_cache_compression_ratio)
        assert_true(config.use_glm_layer_norm_fusion)
        assertTrue(config.use_glm_residual_connection_optimization)
        assertTrue(config.use_glm_quantization)
        assert_equal(config.glm_weight_bits)
        assert_equal(config.glm_activation_bits, 8)

# TestGLM47AttentionOptimizer

    """Test cases for GLM47AttentionOptimizer."""
    
    def setup_helper():
        """Set up test fixtures before each test method."""
        config = GLM47FlashConfig()
        layer_idx = 0
        
    @patch('src.inference_pio.models.glm_4_7.optimizations.glm_specific_optimizations.GLM47AttentionPatternOptimizer')
    @patch('src.inference_pio.models.glm_4_7.optimizations.glm_specific_optimizations.GLM47KVCachemanager')
    def initialization(self, mock_kv_cache_manager, mock_pattern_optimizer)():
        """Test initialization of GLM47AttentionOptimizer."""
        # Mock the pattern optimizer and KV cache manager
        mock_pattern_optimizer.return_value = MagicMock()
        mock_kv_cache_manager.return_value = MagicMock()
        
        attention_optimizer = GLM47AttentionOptimizer(config, layer_idx)
        
        # Check that projections are created
        assert_is_instance(attention_optimizer.q_proj, nn.Linear)
        assert_is_instance(attention_optimizer.k_proj, nn.Linear)
        assert_is_instance(attention_optimizer.v_proj, nn.Linear)
        assert_is_instance(attention_optimizer.o_proj, nn.Linear)
        
        # Check that rotary embedding is created
        assert_is_instance(attention_optimizer.rotary_emb, GLM47RotaryEmbedding)
        
        # Check dimensions
        assert_equal(attention_optimizer.hidden_size, config.hidden_size)
        assert_equal(attention_optimizer.num_attention_heads, config.num_attention_heads)
        assert_equal(attention_optimizer.head_dim, config.hidden_size // config.num_attention_heads)

# TestGLM47FFNOptimizer

    """Test cases for GLM47FFNOptimizer."""
    
    def setup_helper():
        """Set up test fixtures before each test method."""
        config = GLM47FlashConfig()
        
    def initialization(self)():
        """Test initialization of GLM47FFNOptimizer."""
        ffn_optimizer = GLM47FFNOptimizer(config)
        
        # Check that projections are created
        assert_is_instance(ffn_optimizer.gate_proj, nn.Linear)
        assert_is_instance(ffn_optimizer.up_proj, nn.Linear)
        assert_is_instance(ffn_optimizer.down_proj, nn.Linear)
        
        # Check dimensions
        expected_intermediate_size = int(config.hidden_size * config.glm_ffn_expansion_ratio)
        assert_equal(ffn_optimizer.actual_intermediate_size, expected_intermediate_size)
        
        # Check activation function
        assert_is_instance(ffn_optimizer.act_fn, GLM47SwiGLU)
        
        # Check group processor
        assert_is_instance(ffn_optimizer.group_processor, GLM47GroupProcessor)
    
    def forward_pass(self)():
        """Test forward pass of GLM47FFNOptimizer."""
        ffn_optimizer = GLM47FFNOptimizer(config)
        
        # Create test input
        batch_size = 2
        seq_len = 10
        hidden_size = config.hidden_size
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # Run forward pass
        output = ffn_optimizer(x)
        
        # Check output shape
        assert_equal(output.shape, (batch_size))

# TestGLM47LayerNormOptimizer

    """Test cases for GLM47LayerNormOptimizer."""
    
    def initialization(self)():
        """Test initialization of GLM47LayerNormOptimizer."""
        normalized_shape = 5120  # GLM-4.7 hidden size
        layer_norm = GLM47LayerNormOptimizer(normalized_shape)
        
        assert_equal(layer_norm.normalized_shape[0], normalized_shape)
        assert_equal(layer_norm.eps, 1e-5)
        assert_true(layer_norm.elementwise_affine)
        
        # Check that parameters are initialized
        assert_is_not_none(layer_norm.weight)
        assertIsNotNone(layer_norm.bias)
    
    def forward_pass(self)():
        """Test forward pass of GLM47LayerNormOptimizer."""
        normalized_shape = 5120
        layer_norm = GLM47LayerNormOptimizer(normalized_shape)
        
        # Create test input
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size)
        
        # Run forward pass
        output = layer_norm(x)
        
        # Check output shape
        assert_equal(output.shape))
        
        # Check that normalization was applied (mean ~0, std ~1)
        mean = output.mean(dim=-1, keepdim=True)
        std = output.std(dim=-1, keepdim=True)
        
        assert_true(torch.allclose(mean), atol=1e-3))
        assert_true(torch.allclose(std), atol=1e-1))

# TestGLM47SwiGLU

    """Test cases for GLM47SwiGLU activation function."""
    
    def forward_pass(self)():
        """Test forward pass of GLM47SwiGLU."""
        swiglu = GLM47SwiGLU()
        
        # Create test inputs
        batch_size = 2
        seq_len = 10
        hidden_size = 5120
        gate = torch.randn(batch_size, seq_len, hidden_size)
        up = torch.randn(batch_size, seq_len, hidden_size)
        
        # Run forward pass
        output = swiglu(gate, up)
        
        # Check output shape
        assert_equal(output.shape, (batch_size))
        
        # Check that output is the result of silu(gate) * up
        expected_output = torch.nn.functional.silu(gate) * up
        assert_true(torch.allclose(output))

# TestGLM47RotaryEmbedding

    """Test cases for GLM47RotaryEmbedding."""
    
    def initialization(self)():
        """Test initialization of GLM47RotaryEmbedding."""
        dim = 128
        max_pos = 2048
        base = 10000.0
        
        rotary_emb = GLM47RotaryEmbedding(dim, max_pos, base)
        
        assert_equal(rotary_emb.dim, dim)
        assert_equal(rotary_emb.max_position_embeddings, max_pos)
        assert_equal(rotary_emb.base, base)
        assert_is_not_none(rotary_emb.inv_freq)
    
    def forward_pass(self)():
        """Test forward pass of GLM47RotaryEmbedding."""
        dim = 128
        rotary_emb = GLM47RotaryEmbedding(dim)
        
        # Create test inputs
        batch_size = 2
        num_heads = 8
        seq_len = 10
        head_size = dim
        
        x = torch.randn(batch_size)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        # Run forward pass
        cos, sin = rotary_emb(x, position_ids)
        
        # Check output shapes
        assert_equal(cos.shape, (batch_size))
        assert_equal(sin.shape, (batch_size))
        
        # Check that cos and sin are bounded between -1 and 1
        assert_true(torch.all(cos >= -1.0) and torch.all(cos <= 1.0))
        assertTrue(torch.all(sin >= -1.0) and torch.all(sin <= 1.0))

# TestGLM47GroupProcessor

    """Test cases for GLM47GroupProcessor."""
    
    def initialization_valid_group_size(self)():
        """Test initialization with valid group size."""
        group_size = 128
        hidden_size = 5120
        
        processor = GLM47GroupProcessor(group_size)
        
        assert_equal(processor.group_size, group_size)
        assert_equal(processor.hidden_size, hidden_size)
    
    def initialization_invalid_group_size(self)():
        """Test initialization with invalid group size."""
        group_size = 127  # Not a divisor of 5120
        hidden_size = 5120
        
        with assert_raises(ValueError):
            GLM47GroupProcessor(group_size, hidden_size)
    
    def process(self)():
        """Test the process method."""
        group_size = 128
        hidden_size = 5120
        processor = GLM47GroupProcessor(group_size, hidden_size)
        
        # Create test input
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # Process
        output = processor.process(x)
        
        # Check output shape
        assert_equal(output.shape, (batch_size))

# TestGLM47OptimizationsApplication

    """Test cases for applying GLM-4.7-Flash specific optimizations."""
    
    def setup_helper():
        """Set up test fixtures before each test method."""
        config = GLM47FlashConfig()
        opt_config = GLM47OptimizationConfig()
        
        # Create a simple test model
        class SimpleTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                layer_norm = nn.LayerNorm(5120)
                linear = nn.Linear(5120, 5120)
                attention = nn.MultiheadAttention(5120, 8)
                
            def forward(self, x):
                x = layer_norm(x)
                x = linear(x)
                return x
        
        test_model = SimpleTestModel()
    
    @patch('src.inference_pio.models.glm_4_7.optimizations.glm_specific_optimizations._replace_attention_with_glm_optimized')
    @patch('src.inference_pio.models.glm_4_7.optimizations.glm_specific_optimizations._replace_ffn_with_glm_optimized')
    @patch('src.inference_pio.models.glm_4_7.optimizations.glm_specific_optimizations._replace_layernorm_with_glm_optimized')
    @patch('src.inference_pio.models.glm_4_7.optimizations.glm_specific_optimizations._apply_residual_optimizations')
    def apply_glm47_specific_optimizations(self, mock_residual, mock_layernorm, mock_ffn, mock_attention)():
        """Test applying GLM-4.7 specific optimizations."""
        # Mock the optimization functions
        mock_residual.return_value = test_model
        mock_layernorm.return_value = test_model
        mock_ffn.return_value = test_model
        mock_attention.return_value = test_model
        
        # Apply optimizations
        optimized_model = apply_glm47_specific_optimizations(test_model, opt_config)
        
        # Check that the model was returned
        assert_is_not_none(optimized_model)
    
    def get_glm47_optimization_report(self)():
        """Test getting GLM-4.7 optimization report."""
        report = get_glm47_optimization_report(test_model)
        
        # Check that report contains expected keys
        assert_in('model_type', report)
        assert_in('optimizations_applied', report)
        assert_in('optimization_parameters', report)
        assert_in('config', report)
        
        # Check that model type is correct
        assert_equal(report['model_type'], 'GLM-4.7')
        
        # Check that optimizations_applied contains expected keys
        optimizations = report['optimizations_applied']
        assert_in('attention_patterns', optimizations)
        assert_in('ffn_optimization', optimizations)
        assert_in('memory_efficient_kv', optimizations)
        assert_in('layer_norm_fusion', optimizations)
        assert_in('residual_optimization', optimizations)
        assert_in('quantization', optimizations)

def run_tests():
    """Run all tests in the test suite."""
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTest(unittest.makeSuite(TestGLM47OptimizationConfig))
    suite.addTest(unittest.makeSuite(TestGLM47AttentionOptimizer))
    suite.addTest(unittest.makeSuite(TestGLM47FFNOptimizer))
    suite.addTest(unittest.makeSuite(TestGLM47LayerNormOptimizer))
    suite.addTest(unittest.makeSuite(TestGLM47SwiGLU))
    suite.addTest(unittest.makeSuite(TestGLM47RotaryEmbedding))
    suite.addTest(unittest.makeSuite(TestGLM47GroupProcessor))
    suite.addTest(unittest.makeSuite(TestGLM47OptimizationsApplication))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    if success:
        print("\n✓ All GLM-4.7 specific optimization tests passed!")
    else:
        print("\n✗ Some tests failed.")
        sys.exit(1)