"""
Comprehensive Tests for Pipeline Parallelism System
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.inference_pio.common.pipeline_parallel import (
    PipelineConfig,
    PipelineStage,
    PipelineBalancer,
    PipelineParallel,
    PipelineParallelManager,
    create_pipeline_parallel_config,
    split_model_for_pipeline
)
from src.inference_pio.models.glm_4_7.config import GLM47Config
from src.inference_pio.models.glm_4_7.model import GLM47Model
from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen34BInstruct2507Config
from src.inference_pio.models.qwen3_4b_instruct_2507.model import Qwen34BInstruct2507Model
from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig
from src.inference_pio.models.qwen3_coder_30b.model import Qwen3Coder30BModel
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel

class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for testing."""
    
    def __init__(self, hidden_size=128):
        super().__init__()
        attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        norm1 = nn.LayerNorm(hidden_size)
        norm2 = nn.LayerNorm(hidden_size)
        ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
    def forward(self, x):
        # Multihead attention expects (seq_len, batch_size, embed_dim)
        x = x.transpose(0, 1)  # Change from (batch, seq, embed) to (seq, batch, embed)
        attn_out, _ = attention(x, x, x)
        x = norm1(x + attn_out)
        ff_out = ffn(x)
        x = norm2(x + ff_out)
        x = x.transpose(0, 1)  # Change back to (batch, seq, embed)
        return x

class SimpleTestModel(nn.Module):
    """Simple test model for pipeline parallelism testing."""
    
    def __init__(self, hidden_size=128, num_layers=6, vocab_size=1000):
        super().__init__()
        embed = nn.Embedding(vocab_size, hidden_size)
        layers = nn.ModuleList([
            SimpleTransformerBlock(hidden_size) for _ in range(num_layers)
        ])
        norm = nn.LayerNorm(hidden_size)
        head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids):
        x = embed(input_ids)
        for layer in layers:
            x = layer(x)
        x = norm(x)
        logits = head(x)
        return logits

# TestPipelineParallelism

    """Comprehensive tests for pipeline parallelism system."""
    
    def setup_helper():
        """Set up test fixtures."""
        hidden_size = 64
        vocab_size = 100
        seq_length = 10
        batch_size = 2
        
    def pipeline_config_creation(self)():
        """Test creating pipeline configuration."""
        config = PipelineConfig(
            num_stages=3,
            microbatch_size=2,
            enable_activation_offloading=True,
            pipeline_schedule='1f1b'
        )
        
        assert_equal(config.num_stages, 3)
        assert_equal(config.microbatch_size, 2)
        assert_true(config.enable_activation_offloading)
        assert_equal(config.pipeline_schedule)
        
    def create_pipeline_parallel_config_helper(self)():
        """Test helper function for creating pipeline config."""
        config = create_pipeline_parallel_config(
            num_stages=2,
            microbatch_size=4,
            enable_activation_offloading=False,
            pipeline_schedule='gpipe'
        )
        
        assert_equal(config.num_stages, 2)
        assert_equal(config.microbatch_size, 4)
        assert_false(config.enable_activation_offloading)
        assert_equal(config.pipeline_schedule)
        
    def split_model_for_pipeline(self)():
        """Test splitting a model into pipeline stages."""
        model = SimpleTestModel(hidden_size=hidden_size, num_layers=6, vocab_size=vocab_size)
        
        # Test splitting into 3 stages
        num_stages = 3
        stage_models = split_model_for_pipeline(model, num_stages)
        
        assert_equal(len(stage_models), num_stages)
        
        # Check that all layers are preserved
        original_transformer_blocks = sum(1 for m in model.modules() if isinstance(m, SimpleTransformerBlock))
        split_transformer_blocks = sum(
            sum(1 for m in stage.modules() if isinstance(m, SimpleTransformerBlock)) 
            for stage in stage_models
        )
        
        assert_equal(original_transformer_blocks, split_transformer_blocks)
        
    def pipeline_stage_initialization(self)():
        """Test initializing a pipeline stage."""
        config = PipelineConfig(num_stages=1)
        stage_model = nn.Linear(hidden_size, hidden_size)
        
        stage = PipelineStage(
            stage_id=0,
            model_part=stage_model,
            config=config,
            input_device='cpu',
            output_device='cpu'
        )
        
        assert_equal(stage.stage_id, 0)
        assert_is_instance(stage.model_part, nn.Linear)
        
    def pipeline_stage_forward(self)():
        """Test forward pass through a pipeline stage."""
        config = PipelineConfig(num_stages=1)
        stage_model = nn.Linear(hidden_size, hidden_size)
        
        stage = PipelineStage(
            stage_id=0,
            model_part=stage_model,
            config=config,
            input_device='cpu',
            output_device='cpu'
        )
        
        input_tensor = torch.randn(batch_size, seq_length, hidden_size)
        output = stage(input_tensor)
        
        assert_equal(output.shape, (batch_size))
        
    def pipeline_balancer(self)():
        """Test pipeline balancer functionality."""
        config = PipelineConfig(num_stages=3)
        balancer = PipelineBalancer(config)
        
        # Record some fake timing data
        balancer.record_stage_time(0, 0.1)
        balancer.record_stage_time(1, 0.15)
        balancer.record_stage_time(2, 0.08)
        balancer.record_stage_time(0, 0.12)  # Second measurement for stage 0
        
        avg_times = balancer.get_average_stage_times()
        assert_equal(len(avg_times), 3)
        assertAlmostEqual(avg_times[0], 0.11, places=2)  # Average of 0.1 and 0.12
        assertAlmostEqual(avg_times[1], 0.15, places=2)
        assertAlmostEqual(avg_times[2], 0.08, places=2)
        
    def pipeline_parallel_cpu(self)():
        """Test pipeline parallel on CPU."""
        config = PipelineConfig(
            num_stages=2,
            stage_device_mapping=['cpu', 'cpu'],
            microbatch_size=1
        )
        
        model = SimpleTestModel(hidden_size=64, num_layers=4, vocab_size=100)
        pipeline_model = PipelineParallel(model, config)
        
        # Test forward pass
        input_tensor = torch.randint(0, 100, (batch_size, seq_length))
        output = pipeline_model(input_tensor)
        
        assert_equal(output.shape[0], batch_size)
        assert_equal(output.shape[1], seq_length)
        assert_equal(output.shape[2], 100)  # vocab size
        
    def pipeline_parallel_manager(self)():
        """Test pipeline parallel manager."""
        manager = PipelineParallelManager()
        
        config = PipelineConfig(num_stages=2)
        model = SimpleTestModel(hidden_size=64, num_layers=4, vocab_size=100)
        
        pipeline_model = manager.create_pipeline_model(model, config)
        
        # Test stats
        stats = manager.get_pipeline_stats(pipeline_model)
        assert_equal(stats['num_stages'], 2)
        assert_in('stage_times', stats)
        assert_in('devices_used', stats)
        
        # Cleanup
        manager.cleanup_model(pipeline_model)
        
    def pipeline_generate_with_simple_model(self)():
        """Test generation functionality with pipeline."""
        config = PipelineConfig(
            num_stages=2,
            stage_device_mapping=['cpu', 'cpu'],
            microbatch_size=1
        )
        
        model = SimpleTestModel(hidden_size=64, num_layers=4, vocab_size=100)
        pipeline_model = PipelineParallel(model, config)
        
        # Test generation
        input_tensor = torch.randint(0, 100, (1, 5))  # Smaller input for generation test
        output = pipeline_model.generate_with_pipeline(input_tensor, max_new_tokens=3)
        
        # The output shape depends on the generation implementation
        # For this simple test, we just check that it runs without error
        assert_is_not_none(output)
        
    def edge_cases(self)():
        """Test edge cases and error conditions."""
        # Test with 1 stage (should work like regular model)
        config = PipelineConfig(num_stages=1)
        model = SimpleTestModel(hidden_size=64)
        pipeline_model = PipelineParallel(model, config)
        
        input_tensor = torch.randint(0, 100, (2, 5))
        output = pipeline_model(input_tensor)
        assert_equal(output.shape[2], 100)  # vocab size

# TestModelIntegration

    """Test pipeline parallelism integration with actual models."""
    
    def setup_helper():
        """Set up test fixtures."""
        # Use minimal configurations for testing
        glm_config = GLM47Config(
            model_path="test_model",  # Will use fallback
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=4,
            vocab_size=1000,
            torch_dtype="float32",
            device_map="cpu",
            enable_disk_offloading=False,
            enable_intelligent_pagination=False,
            use_quantization=False
        )
        
        qwen_config = Qwen34BInstruct2507Config(
            model_path="test_model",  # Will use fallback
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=4,
            vocab_size=1000,
            torch_dtype="float32",
            device_map="cpu",
            enable_disk_offloading=False,
            enable_intelligent_pagination=False,
            use_quantization=False
        )
        
        qwen_coder_config = Qwen3Coder30BConfig(
            model_path="test_model",  # Will use fallback
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=4,
            vocab_size=1000,
            torch_dtype="float32",
            device_map="cpu",
            enable_disk_offloading=False,
            enable_intelligent_pagination=False,
            use_quantization=False
        )
        
        qwen_vl_config = Qwen3VL2BConfig(
            model_path="test_model",  # Will use fallback
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=4,
            vocab_size=1000,
            torch_dtype="float32",
            device_map="cpu",
            enable_disk_offloading=False,
            enable_intelligent_pagination=False,
            use_quantization=False
        )
    
    def glm_pipeline_integration(self)():
        """Test GLM-4-7 model with pipeline parallelism."""
        # Create a minimal model instance
        model = GLM47Model(glm_config)
        
        # Test that the model has pipeline parallelism attributes
        assert_true(hasattr(model))
        
        # Test forward pass (with pipeline disabled by default)
        input_tensor = torch.randint(0, 1000, (1, 10))
        output = model._model(input_tensor)
        assert_is_not_none(output)
    
    def qwen3_4b_pipeline_integration(self)():
        """Test Qwen3-4b-instruct-2507 model with pipeline parallelism."""
        # Create a minimal model instance
        model = Qwen34BInstruct2507Model(qwen_config)
        
        # Test that the model has pipeline parallelism attributes
        assert_true(hasattr(model))
        
        # Test forward pass (with pipeline disabled by default)
        input_tensor = torch.randint(0))
        output = model._model(input_tensor)
        assert_is_not_none(output)
    
    def qwen3_coder_pipeline_integration(self)():
        """Test Qwen3-coder-30b model with pipeline parallelism."""
        # Create a minimal model instance
        model = Qwen3Coder30BModel(qwen_coder_config)
        
        # Test that the model has pipeline parallelism attributes
        assert_true(hasattr(model))
        
        # Test forward pass (with pipeline disabled by default)
        input_tensor = torch.randint(0))
        output = model._model(input_tensor)
        assert_is_not_none(output)
    
    def qwen3_vl_pipeline_integration(self)():
        """Test Qwen3-vl-2b model with pipeline parallelism."""
        # Create a minimal model instance
        model = Qwen3VL2BModel(qwen_vl_config)
        
        # Test that the model has pipeline parallelism attributes
        assert_true(hasattr(model))
        
        # Test forward pass (with pipeline disabled by default)
        input_tensor = torch.randint(0))
        output = model._model(input_tensor)
        assert_is_not_none(output)

if __name__ == '__main__':
    run_tests(test_functions)