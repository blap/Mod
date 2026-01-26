"""
Tests for Pipeline Parallelism System
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
from src.inference_pio.common.pipeline_parallel import (
    PipelineConfig,
    PipelineStage,
    PipelineBalancer,
    PipelineParallel,
    PipelineParallelManager,
    create_pipeline_parallel_config,
    split_model_for_pipeline
)

class SimpleTestModel(nn.Module):
    """Simple test model for pipeline parallelism testing."""
    
    def __init__(self, hidden_size=128, num_layers=4):
        super().__init__()
        layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        activation = nn.ReLU()
        
    def forward(self, x):
        for layer in layers:
            x = activation(layer(x))
        return x

# TestPipelineParallel

    """Test cases for pipeline parallelism system."""
    
    def setup_helper():
        """Set up test fixtures."""
        hidden_size = 128
        batch_size = 4
        test_model = SimpleTestModel(hidden_size=hidden_size, num_layers=6)
        test_input = torch.randn(batch_size, hidden_size)
        
    def pipeline_config_creation(self)():
        """Test creating pipeline configuration."""
        config = PipelineConfig(
            num_stages=3,
            microbatch_size=2,
            enable_activation_offloading=True
        )
        
        assert_equal(config.num_stages, 3)
        assert_equal(config.microbatch_size, 2)
        assert_true(config.enable_activation_offloading)
        
    def create_pipeline_parallel_config_helper(self)():
        """Test helper function for creating pipeline config."""
        config = create_pipeline_parallel_config(
            num_stages=2,
            microbatch_size=4,
            enable_activation_offloading=False
        )
        
        assert_equal(config.num_stages, 2)
        assert_equal(config.microbatch_size, 4)
        assert_false(config.enable_activation_offloading)
        
    def split_model_for_pipeline(self)():
        """Test splitting a model into pipeline stages."""
        num_stages = 3
        stage_models = split_model_for_pipeline(test_model)
        
        assert_equal(len(stage_models), num_stages)
        
        # Check that all layers are preserved
        total_layers = sum(len(list(stage.modules())) for stage in stage_models)
        original_total = len(list(test_model.modules()))
        
        # Account for the fact that split_model_for_pipeline creates Sequential containers
        # which add extra modules, so we just verify the number of Linear layers
        original_linear_layers = sum(1 for m in test_model.modules() if isinstance(m, nn.Linear))
        split_linear_layers = sum(sum(1 for m in stage.modules() if isinstance(m, nn.Linear)) 
                                 for stage in stage_models)
        
        assert_equal(original_linear_layers, split_linear_layers)
        
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
        
        input_tensor = torch.randn(batch_size, hidden_size)
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
        
        avg_times = balancer.get_average_stage_times()
        assert_equal(len(avg_times), 3)
        assert_greater(avg_times[1], avg_times[2])  # Stage 1 should have higher avg time
        
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def pipeline_parallel_with_cuda(self)():
        """Test pipeline parallel with CUDA devices."""
        config = PipelineConfig(
            num_stages=2,
            stage_device_mapping=['cuda:0', 'cuda:1'] if torch.cuda.device_count() > 1 else ['cuda:0', 'cuda:0'],
            microbatch_size=2
        )
        
        # Create a smaller model for testing
        model = SimpleTestModel(hidden_size=64, num_layers=4)
        pipeline_model = PipelineParallel(model, config)
        
        # Test forward pass
        input_tensor = torch.randn(4, 64).cuda()
        output = pipeline_model(input_tensor)
        
        assert_equal(output.shape, (4))
        
    def pipeline_parallel_cpu(self)():
        """Test pipeline parallel on CPU."""
        config = PipelineConfig(
            num_stages=2,
            stage_device_mapping=['cpu', 'cpu'],
            microbatch_size=2
        )
        
        model = SimpleTestModel(hidden_size=64, num_layers=4)
        pipeline_model = PipelineParallel(model, config)
        
        # Test forward pass
        input_tensor = torch.randn(4, 64)
        output = pipeline_model(input_tensor)
        
        assert_equal(output.shape, (4))
        
    def pipeline_parallel_manager(self)():
        """Test pipeline parallel manager."""
        manager = PipelineParallelManager()
        
        config = PipelineConfig(num_stages=2)
        model = SimpleTestModel(hidden_size=64, num_layers=4)
        
        pipeline_model = manager.create_pipeline_model(model, config)
        
        # Test stats
        stats = manager.get_pipeline_stats(pipeline_model)
        assert_equal(stats['num_stages'], 2)
        assert_in('stage_times', stats)
        
        # Cleanup
        manager.cleanup_model(pipeline_model)
        
    def pipeline_generate_with_simple_model(self)():
        """Test generation functionality with pipeline."""
        config = PipelineConfig(
            num_stages=2,
            stage_device_mapping=['cpu', 'cpu'],
            microbatch_size=1
        )
        
        model = SimpleTestModel(hidden_size=64, num_layers=4)
        pipeline_model = PipelineParallel(model, config)
        
        # Test generation
        input_tensor = torch.randn(1, 64)
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
        
        input_tensor = torch.randn(2, 64)
        output = pipeline_model(input_tensor)
        assert_equal(output.shape, (2))

if __name__ == '__main__':
    run_tests(test_functions)