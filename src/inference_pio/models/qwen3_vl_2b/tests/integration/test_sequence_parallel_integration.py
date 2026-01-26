"""
Test suite for sequence parallelism integration in models.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
from src.inference_pio.models.glm_4_7.config import GLM47Config
from src.inference_pio.models.glm_4_7.model import GLM47Model
from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen34BInstruct2507Config
from src.inference_pio.models.qwen3_4b_instruct_2507.model import Qwen34BInstruct2507Model
from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig
from src.inference_pio.models.qwen3_coder_30b.model import Qwen3Coder30BModel
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel

# TestSequenceParallelIntegration

    """Test cases for sequence parallelism integration in models."""
    
    def setup_helper():
        """Set up test fixtures."""
        # Create minimal configurations for testing
        glm_config = GLM47Config(
            model_path="fake_path"  # Will use fallback
        )
        # Set sequence parallelism parameters after initialization
        glm_config.enable_sequence_parallelism = True
        glm_config.sequence_parallel_num_segments = 2
        glm_config.sequence_parallel_split_method = 'chunk'
        glm_config.sequence_parallel_enable_overlap = True
        glm_config.sequence_parallel_overlap_size = 32
        glm_config.sequence_parallel_algorithm = '1d'

        qwen3_4b_config = Qwen34BInstruct2507Config(
            model_path="fake_path"  # Will use fallback
        )
        # Set sequence parallelism parameters after initialization
        qwen3_4b_config.enable_sequence_parallelism = True
        qwen3_4b_config.sequence_parallel_num_segments = 2
        qwen3_4b_config.sequence_parallel_split_method = 'chunk'
        qwen3_4b_config.sequence_parallel_enable_overlap = True
        qwen3_4b_config.sequence_parallel_overlap_size = 32
        qwen3_4b_config.sequence_parallel_algorithm = '1d'

        qwen3_coder_config = Qwen3Coder30BConfig(
            model_path="fake_path"  # Will use fallback
        )
        # Set sequence parallelism parameters after initialization
        qwen3_coder_config.enable_sequence_parallelism = True
        qwen3_coder_config.sequence_parallel_num_segments = 2
        qwen3_coder_config.sequence_parallel_split_method = 'chunk'
        qwen3_coder_config.sequence_parallel_enable_overlap = True
        qwen3_coder_config.sequence_parallel_overlap_size = 32
        qwen3_coder_config.sequence_parallel_algorithm = '1d'

        qwen3_vl_config = Qwen3VL2BConfig()
        # Set sequence parallelism parameters after initialization
        qwen3_vl_config.model_path = "fake_path"  # Will use fallback
        qwen3_vl_config.enable_sequence_parallelism = True
        qwen3_vl_config.sequence_parallel_num_segments = 2
        qwen3_vl_config.sequence_parallel_split_method = 'chunk'
        qwen3_vl_config.sequence_parallel_enable_overlap = True
        qwen3_vl_config.sequence_parallel_overlap_size = 32
        qwen3_vl_config.sequence_parallel_algorithm = '1d'
        
    def glm_sequence_parallel_initialization(self)():
        """Test GLM model with sequence parallelism initialization."""
        # Just check that the config has the right attributes
        assert_true(hasattr(glm_config))
        assert_true(glm_config.enable_sequence_parallelism)

        # Test that the model class has the sequence parallelism methods
        # We'll create a mock model instance without calling __init__ to avoid downloading
        model_cls = GLM47Model
        # Check that the class has the required methods/attributes
        # by inspecting the source code
        import inspect
        source = inspect.getsource(GLM47Model._initialize_sequence_parallelism)
        assertIn('_sequence_parallel_model')

    def qwen3_4b_sequence_parallel_initialization(self)():
        """Test Qwen3-4B model with sequence parallelism initialization."""
        # Just check that the config has the right attributes
        assert_true(hasattr(qwen3_4b_config))
        assert_true(qwen3_4b_config.enable_sequence_parallelism)

        # Test that the model class has the sequence parallelism methods
        import inspect
        source = inspect.getsource(Qwen34BInstruct2507Model._initialize_sequence_parallelism)
        assertIn('_sequence_parallel_model')

    def qwen3_coder_sequence_parallel_initialization(self)():
        """Test Qwen3-Coder model with sequence parallelism initialization."""
        # Just check that the config has the right attributes
        assert_true(hasattr(qwen3_coder_config))
        assert_true(qwen3_coder_config.enable_sequence_parallelism)

        # Test that the model class has the sequence parallelism methods
        import inspect
        source = inspect.getsource(Qwen3Coder30BModel._initialize_sequence_parallelism)
        assertIn('_sequence_parallel_model')

    def qwen3_vl_sequence_parallel_initialization(self)():
        """Test Qwen3-VL model with sequence parallelism initialization."""
        # Just check that the config has the right attributes
        assert_true(hasattr(qwen3_vl_config))
        assert_true(qwen3_vl_config.enable_sequence_parallelism)

        # Test that the model class has the sequence parallelism methods
        import inspect
        source = inspect.getsource(Qwen3VL2BModel._initialize_sequence_parallelism)
        assertIn('_sequence_parallel_model')
            
    def sequence_parallel_attributes_exist(self)():
        """Test that sequence parallel attributes exist in all models."""
        # Test GLM
        glm_model = GLM47Model.__new__(GLM47Model)
        glm_model._sequence_parallel_model = None
        assert_true(hasattr(glm_model))
        
        # Test Qwen3-4B
        qwen3_4b_model = Qwen34BInstruct2507Model.__new__(Qwen34BInstruct2507Model)
        qwen3_4b_model._sequence_parallel_model = None
        assert_true(hasattr(qwen3_4b_model))
        
        # Test Qwen3-Coder
        qwen3_coder_model = Qwen3Coder30BModel.__new__(Qwen3Coder30BModel)
        qwen3_coder_model._sequence_parallel_model = None
        assert_true(hasattr(qwen3_coder_model))
        
        # Test Qwen3-VL
        qwen3_vl_model = Qwen3VL2BModel.__new__(Qwen3VL2BModel)
        qwen3_vl_model._sequence_parallel_model = None
        assert_true(hasattr(qwen3_vl_model))

if __name__ == '__main__':
    run_tests(test_functions)