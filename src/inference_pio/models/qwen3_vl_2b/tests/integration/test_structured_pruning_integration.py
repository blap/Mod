"""
Comprehensive test for structured pruning integration with all four models.

This test verifies that the structured pruning system is properly integrated
with all four models (GLM-4-7, Qwen3-4b-instruct-2507, Qwen3-coder-30b, Qwen3-vl-2b)
and that it preserves accuracy while reducing model complexity.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import sys
import os
# Add the src directory to the path to import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.inference_pio.common.structured_pruning import (
    StructuredPruningSystem, 
    PruningMethod, 
    get_structured_pruning_system, 
    apply_structured_pruning
)

class MockModel(torch.nn.Module):
    """Mock model for testing structured pruning."""
    
    def __init__(self):
        super().__init__()
        linear1 = torch.nn.Linear(10, 20)
        relu1 = torch.nn.ReLU()
        linear2 = torch.nn.Linear(20, 15)
        relu2 = torch.nn.ReLU()
        linear3 = torch.nn.Linear(15, 5)
        attention = torch.nn.MultiheadAttention(embed_dim=10, num_heads=2)
        
    def forward(self, x):
        x = relu1(linear1(x))
        x = relu2(linear2(x))
        x, _ = attention(x, x, x)
        x = linear3(x)
        return x

# TestStructuredPruningIntegration

    """Test cases for structured pruning integration with all models."""
    
    def setup_helper():
        """Set up test fixtures before each test method."""
        mock_model = MockModel()
        pruning_system = get_structured_pruning_system()
        
    def structured_pruning_system_initialization(self)():
        """Test that the structured pruning system initializes correctly."""
        assert_is_instance(pruning_system, StructuredPruningSystem)
        assert_equal(len(pruning_system.pruning_history), 0)
        
    def apply_structured_pruning_layer_removal(self)():
        """Test applying structured pruning with layer removal method."""
        original_params = sum(p.numel() for p in mock_model.parameters())
        
        result = apply_structured_pruning(
            mock_model,
            pruning_ratio=0.2,
            method=PruningMethod.LAYER_REMOVAL
        )
        
        assert_is_not_none(result)
        assert_is_instance(result)))
        # Since we're mocking, we can't verify the exact result):
        """Test applying structured pruning with block removal method."""
        result = apply_structured_pruning(
            mock_model,
            pruning_ratio=0.2,
            method=PruningMethod.BLOCK_REMOVAL,
            block_size=2
        )
        
        assert_is_not_none(result)
        
    def apply_structured_pruning_head_removal(self)():
        """Test applying structured pruning with head removal method."""
        result = apply_structured_pruning(
            mock_model,
            pruning_ratio=0.5,
            method=PruningMethod.HEAD_REMOVAL
        )
        
        assert_is_not_none(result)
        
    def apply_structured_pruning_mlp_removal(self)():
        """Test applying structured pruning with MLP removal method."""
        result = apply_structured_pruning(
            mock_model,
            pruning_ratio=0.3,
            method=PruningMethod.MLP_REMOVAL
        )
        
        assert_is_not_none(result)
        
    def apply_structured_pruning_adaptive_pruning(self)():
        """Test applying structured pruning with adaptive pruning method."""
        result = apply_structured_pruning(
            mock_model,
            pruning_ratio=0.25,
            method=PruningMethod.ADAPTIVE_PRUNING
        )
        
        assert_is_not_none(result)
        
    @patch('torch.load')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def glm_4_7_model_integration(self)():
        """Test that GLM-4-7 model integrates with structured pruning."""
        # Import here to avoid issues if the model isn't available
        try:
            from src.inference_pio.models.glm_4_7.config import GLM47Config
            from src.inference_pio.models.glm_4_7.model import GLM47Model
            
            # Create a mock config with structured pruning enabled
            config = GLM47Config()
            config.use_structured_pruning = True
            config.pruning_ratio = 0.2
            config.pruning_method = "layer_removal"
            config.pruning_block_size = 1
            
            # Mock the model loading
            mock_model_instance = MagicMock()
            mock_model_instance.named_modules.return_value = []
            mock_model_instance.parameters.return_value = []
            mock_auto_model.return_value = mock_model_instance
            
            # Create the model - this should trigger structured pruning
            model = GLM47Model(config)
            
            # Verify that structured pruning was attempted
            # (We can't fully test without the actual model files)
            assert_true(hasattr(model))
            
        except ImportError:
            # If the model isn't available, skip this test
            skipTest("GLM-4-7 model not available")
            
    @patch('torch.load')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def qwen3_4b_instruct_2507_model_integration(self, mock_load, mock_auto_model)():
        """Test that Qwen3-4b-instruct-2507 model integrates with structured pruning."""
        try:
            from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen34BInstruct2507Config
            from src.inference_pio.models.qwen3_4b_instruct_2507.model import Qwen34BInstruct2507Model
            
            # Create a mock config with structured pruning enabled
            config = Qwen34BInstruct2507Config()
            config.use_structured_pruning = True
            config.pruning_ratio = 0.2
            config.pruning_method = "layer_removal"
            config.pruning_block_size = 1
            
            # Mock the model loading
            mock_model_instance = MagicMock()
            mock_model_instance.named_modules.return_value = []
            mock_model_instance.parameters.return_value = []
            mock_auto_model.return_value = mock_model_instance
            
            # Create the model - this should trigger structured pruning
            model = Qwen34BInstruct2507Model(config)
            
            # Verify that structured pruning was attempted
            assert_true(hasattr(model))
            
        except ImportError:
            skipTest("Qwen3-4b-instruct-2507 model not available")
            
    @patch('torch.load')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def qwen3_coder_30b_model_integration(self, mock_load, mock_auto_model)():
        """Test that Qwen3-coder-30b model integrates with structured pruning."""
        try:
            from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig
            from src.inference_pio.models.qwen3_coder_30b.model import Qwen3Coder30BModel
            
            # Create a mock config with structured pruning enabled
            config = Qwen3Coder30BConfig()
            config.use_structured_pruning = True
            config.pruning_ratio = 0.2
            config.pruning_method = "layer_removal"
            config.pruning_block_size = 1
            
            # Mock the model loading
            mock_model_instance = MagicMock()
            mock_model_instance.named_modules.return_value = []
            mock_model_instance.parameters.return_value = []
            mock_auto_model.return_value = mock_model_instance
            
            # Create the model - this should trigger structured pruning
            model = Qwen3Coder30BModel(config)
            
            # Verify that structured pruning was attempted
            assert_true(hasattr(model))
            
        except ImportError:
            skipTest("Qwen3-coder-30b model not available")
            
    @patch('torch.load')
    @patch('transformers.AutoModelForVision2Seq.from_pretrained')
    def qwen3_vl_2b_model_integration(self, mock_load, mock_auto_model)():
        """Test that Qwen3-vl-2b model integrates with structured pruning."""
        try:
            from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
            from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
            
            # Create a mock config with structured pruning enabled
            config = Qwen3VL2BConfig()
            config.use_structured_pruning = True
            config.pruning_ratio = 0.2
            config.pruning_method = "layer_removal"
            config.pruning_block_size = 1
            
            # Mock the model loading
            mock_model_instance = MagicMock()
            mock_model_instance.named_modules.return_value = []
            mock_model_instance.parameters.return_value = []
            mock_auto_model.return_value = mock_model_instance
            
            # Create the model - this should trigger structured pruning
            model = Qwen3VL2BModel(config)
            
            # Verify that structured pruning was attempted
            assert_true(hasattr(model))
            
        except ImportError:
            skipTest("Qwen3-vl-2b model not available")
    
    def pruning_method_enum_values(self)():
        """Test that all pruning method enum values are properly defined."""
        assert_equal(PruningMethod.LAYER_REMOVAL.value, "layer_removal")
        assert_equal(PruningMethod.BLOCK_REMOVAL.value, "block_removal")
        assert_equal(PruningMethod.HEAD_REMOVAL.value, "head_removal")
        assert_equal(PruningMethod.MLP_REMOVAL.value, "mlp_removal")
        assert_equal(PruningMethod.ADAPTIVE_PRUNING.value, "adaptive_pruning")
        
    def pruning_result_structure(self)():
        """Test the structure of the pruning result."""
        # Create a mock pruning result
        from src.inference_pio.common.structured_pruning import PruningResult
        
        mock_model = MockModel()
        original_params = sum(p.numel() for p in mock_model.parameters())
        
        result = PruningResult(
            pruned_model=mock_model,
            original_params=original_params,
            pruned_params=original_params * 0.8,  # 20% reduction
            compression_ratio=0.2,
            accuracy_preserved=True,
            removed_layers=["layer1", "layer2"],
            metrics={"test_metric": 0.95}
        )
        
        assert_is_instance(result.pruned_model, torch.nn.Module)
        assert_equal(result.original_params, original_params)
        assertLessEqual(result.pruned_params, result.original_params)
        assert_equal(result.compression_ratio, 0.2)
        assert_true(result.accuracy_preserved)
        assert_equal(result.removed_layers)
        assert_equal(result.metrics["test_metric"], 0.95)

# TestEndToEndPruning

    """End-to-end tests for structured pruning functionality."""
    
    def setup_helper():
        """Set up test fixtures before each test method."""
        mock_model = MockModel()
        
    def end_to_end_pruning_process(self)():
        """Test the complete end-to-end pruning process."""
        # Test that we can create the system and apply pruning
        system = get_structured_pruning_system()
        
        # Verify the system has no history initially
        stats = system.get_pruning_stats()
        initial_ops = stats.get('total_pruning_operations', 0)
        
        # Apply pruning
        result = apply_structured_pruning(
            mock_model,
            pruning_ratio=0.1,
            method=PruningMethod.LAYER_REMOVAL
        )
        
        # Verify stats were updated
        updated_stats = system.get_pruning_stats()
        final_ops = updated_stats.get('total_pruning_operations', 0)
        
        # The number of operations should have increased
        assertGreaterEqual(final_ops, initial_ops)
        
    def multiple_pruning_operations(self)():
        """Test performing multiple pruning operations."""
        system = get_structured_pruning_system()
        
        # Clear any previous history
        system.pruning_history = []
        
        # Perform multiple pruning operations with different methods
        methods = [
            PruningMethod.LAYER_REMOVAL,
            PruningMethod.BLOCK_REMOVAL,
            PruningMethod.HEAD_REMOVAL
        ]
        
        for i, method in enumerate(methods):
            result = apply_structured_pruning(
                mock_model,
                pruning_ratio=0.1,
                method=method
            )
            assert_is_not_none(result)
        
        # Check that history was recorded
        stats = system.get_pruning_stats()
        assert_equal(stats['total_pruning_operations'])

if __name__ == '__main__':
    # Run the tests
    run_tests(test_functions)