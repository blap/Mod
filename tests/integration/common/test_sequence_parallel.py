"""
Test suite for sequence parallelism system.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
from src.inference_pio.common.sequence_parallel import (
    SequenceParallel,
    SequenceParallelConfig,
    create_sequence_parallel_config,
    split_sequence_for_parallel
)

class SimpleTestModel(nn.Module):
    """Simple test model for sequence parallelism testing."""
    
    def __init__(self, hidden_size=128, num_layers=4):
        super().__init__()
        layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        for layer in layers:
            x = torch.relu(layer(x))
        x = norm(x)
        return x

# TestSequenceParallel

    """Test cases for sequence parallelism system."""
    
    def setup_helper():
        """Set up test fixtures."""
        hidden_size = 128
        seq_len = 64
        batch_size = 2
        model = SimpleTestModel(hidden_size=hidden_size, num_layers=4)
        
    def sequence_parallel_config_creation(self)():
        """Test creation of sequence parallel configuration."""
        config = create_sequence_parallel_config(
            num_segments=2,
            sequence_split_method='chunk',
            enable_sequence_overlap=True,
            overlap_size=32
        )
        
        assert_equal(config.num_segments, 2)
        assert_equal(config.sequence_split_method, 'chunk')
        assert_true(config.enable_sequence_overlap)
        assert_equal(config.overlap_size)
        
    def sequence_parallel_basic_forward(self)():
        """Test basic forward pass with sequence parallelism."""
        config = create_sequence_parallel_config(num_segments=2)
        seq_parallel = SequenceParallel(model, config)
        
        # Create test input
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        output = seq_parallel(input_tensor)
        
        # Check output shape
        assert_equal(output.shape, (batch_size))
        
    def sequence_parallel_different_methods(self)():
        """Test different sequence parallel methods."""
        for method in ['chunk', 'stride']:
            with subTest(method=method):
                config = SequenceParallelConfig(
                    num_segments=2,
                    sequence_split_method=method
                )
                seq_parallel = SequenceParallel(model, config)
                
                input_tensor = torch.randn(batch_size, seq_len, hidden_size)
                output = seq_parallel(input_tensor)
                
                assert_equal(output.shape, (batch_size))
                
    def sequence_parallel_with_overlap(self)():
        """Test sequence parallelism with overlap enabled."""
        config = SequenceParallelConfig(
            num_segments=2,
            enable_sequence_overlap=True,
            overlap_size=16
        )
        seq_parallel = SequenceParallel(model, config)
        
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        output = seq_parallel(input_tensor)
        
        assert_equal(output.shape, (batch_size))
        
    def sequence_parallel_ring_algorithm(self)():
        """Test sequence parallelism with ring algorithm."""
        config = SequenceParallelConfig(
            num_segments=2,
            sequence_parallel_algorithm='ring',
            ring_chunk_size=16
        )
        seq_parallel = SequenceParallel(model, config)
        
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        output = seq_parallel(input_tensor)
        
        assert_equal(output.shape, (batch_size))
        
    def split_sequence_for_parallel(self)():
        """Test utility function to split model for sequence parallelism."""
        num_segments = 2
        segments = split_sequence_for_parallel(model, num_segments)

        assert_equal(len(segments), num_segments)

        # Check that the layers are properly split
        # The split function only splits the main layers, not other components like norm
        original_layers_params = sum(p.numel() for p in model.layers.parameters())
        split_params = sum(sum(p.numel() for p in seg.parameters()) for seg in segments)

        # The split parameters should equal the original layer parameters
        assert_equal(original_layers_params, split_params)
        
    def sequence_parallel_generate(self)():
        """Test generate method with sequence parallelism."""
        config = SequenceParallelConfig(
            num_segments=2,
            sequence_parallel_algorithm='1d'
        )
        seq_parallel = SequenceParallel(model, config)
        
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        
        # Test generate method (should fall back to forward for this simple model)
        output = seq_parallel.generate_with_sequence_parallel(
            input_tensor,
            max_new_tokens=10
        )
        
        # For this simple model, it should return the same shape as forward
        assert_equal(output.shape, (batch_size))
        
    def sequence_parallel_different_devices(self)():
        """Test sequence parallelism with different device mappings."""
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            device_mapping = ['cuda:0', 'cuda:1'] if torch.cuda.device_count() >= 2 else ['cuda:0', 'cuda:0']
            
            config = SequenceParallelConfig(
                num_segments=2,
                segment_device_mapping=device_mapping
            )
            seq_parallel = SequenceParallel(model, config)
            
            input_tensor = torch.randn(batch_size, seq_len, hidden_size)
            output = seq_parallel(input_tensor)
            
            assert_equal(output.shape, (batch_size))
        else:
            # Skip this test if CUDA is not available or has only one GPU
            skipTest("CUDA not available or only one GPU")
            
    def sequence_parallel_edge_cases(self)():
        """Test edge cases for sequence parallelism."""
        # Test with single segment (should behave like original model)
        config = SequenceParallelConfig(num_segments=1)
        seq_parallel = SequenceParallel(model, config)
        
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        output = seq_parallel(input_tensor)
        
        assert_equal(output.shape, (batch_size))
        
        # Test with more segments than layers
        config = SequenceParallelConfig(num_segments=10)  # More segments than model layers
        seq_parallel = SequenceParallel(model, config)
        
        output = seq_parallel(input_tensor)
        assert_equal(output.shape, (batch_size))

if __name__ == '__main__':
    run_tests(test_functions)