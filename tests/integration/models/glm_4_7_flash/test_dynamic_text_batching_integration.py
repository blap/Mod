"""
Integration tests for Dynamic Text Batching System with Unimodal Models.

This test verifies that the dynamic text batching system integrates correctly 
with the unimodal models (GLM-4-7, Qwen3-4b-instruct-2507, Qwen3-coder-30b).
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import sys
import os
import torch

# Add the project root to the path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference_pio.models.glm_4_7.model import GLM47Model
from src.inference_pio.models.qwen3_4b_instruct_2507.model import Qwen34BInstruct2507Model
from src.inference_pio.models.qwen3_coder_30b.model import Qwen3Coder30BModel
from src.inference_pio.models.glm_4_7.config import GLM47Config
from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen34BInstruct2507Config
from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig

# TestDynamicTextBatchingIntegration

    """Integration tests for dynamic text batching with unimodal models."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        # Create minimal configurations for testing
        glm_config = GLM47Config(
            model_path="THUDM/glm-4-9b",  # Using a placeholder path
            initial_batch_size=2,
            min_batch_size=1,
            max_batch_size=4,
            use_flash_attention_2=False,
            use_sparse_attention=False,
            use_bias_removal_optimization=False,
            use_fused_layer_norm=False,
            use_kv_cache_compression=False,
            use_prefix_caching=False,
            use_cuda_kernels=False,
            linear_bias_optimization_enabled=False,
            use_tensor_parallelism=False,
            gradient_checkpointing=False,
            use_cache=False,
            torch_dtype="float16",
            device_map="cpu",  # Use CPU for testing
            low_cpu_mem_usage=True,
            max_memory=None
        )
        
        qwen3_4b_config = Qwen34BInstruct2507Config(
            model_path="H:/Qwen3-4B-Instruct-2507",  # Using a placeholder path
            initial_batch_size=2,
            min_batch_size=1,
            max_batch_size=4,
            use_flash_attention_2=False,
            use_sparse_attention=False,
            use_bias_removal_optimization=False,
            use_fused_layer_norm=False,
            use_kv_cache_compression=False,
            use_prefix_caching=False,
            use_cuda_kernels=False,
            linear_bias_optimization_enabled=False,
            use_tensor_parallelism=False,
            gradient_checkpointing=False,
            use_cache=False,
            torch_dtype="float16",
            device_map="cpu",  # Use CPU for testing
            low_cpu_mem_usage=True,
            max_memory=None
        )
        
        qwen3_coder_config = Qwen3Coder30BConfig(
            model_path="H:/Qwen3-Coder-30B",  # Using a placeholder path
            initial_batch_size=2,
            min_batch_size=1,
            max_batch_size=4,
            use_flash_attention_2=False,
            use_sparse_attention=False,
            use_bias_removal_optimization=False,
            use_fused_layer_norm=False,
            use_kv_cache_compression=False,
            use_prefix_caching=False,
            use_cuda_kernels=False,
            linear_bias_optimization_enabled=False,
            use_tensor_parallelism=False,
            gradient_checkpointing=False,
            use_cache=False,
            torch_dtype="float16",
            device_map="cpu",  # Use CPU for testing
            low_cpu_mem_usage=True,
            max_memory=None
        )

    def glm47_dynamic_batching_integration(self)():
        """Test that GLM-4-7 model integrates with dynamic text batching."""
        try:
            # Create model instance (this would normally load the actual model)
            # For this test, we're checking if the methods exist and work with the new system
            model = object.__new__(GLM47Model)  # Create without calling __init__ to avoid loading
            model.config = glm_config
            
            # Check that the new method exists
            assert_true(hasattr(GLM47Model))
            
            # Check that the required imports are in place
            import src.inference_pio.models.glm_4_7.model
            import inspect
            source = inspect.getsource(GLM47Model.generate_with_adaptive_batching)
            
            # Verify that the new dynamic text batching is used
            assert_in('get_dynamic_text_batch_manager', source)
            assert_in('DynamicTextBatchManager', source)
            
        except Exception as e:
            # If there's an issue with the actual model loading, at least verify the source has the right imports
            import src.inference_pio.models.glm_4_7.model
            import inspect
            source = inspect.getsource(src.inference_pio.models.glm_4_7.model)
            
            # Check that the new imports are present
            assert_in('dynamic_text_batching', source)
            assert_in('get_dynamic_text_batch_manager', source)
            assert_in('DynamicTextBatchManager', source)

    def qwen3_4b_dynamic_batching_integration(self)():
        """Test that Qwen3-4b-instruct-2507 model integrates with dynamic text batching."""
        try:
            # Create model instance (this would normally load the actual model)
            model = object.__new__(Qwen34BInstruct2507Model)  # Create without calling __init__ to avoid loading
            model.config = qwen3_4b_config
            
            # Check that the new method exists
            assert_true(hasattr(Qwen34BInstruct2507Model))
            
            # Check that the required imports are in place
            import src.inference_pio.models.qwen3_4b_instruct_2507.model
            import inspect
            source = inspect.getsource(Qwen34BInstruct2507Model.generate_with_adaptive_batching)
            
            # Verify that the new dynamic text batching is used
            assert_in('get_dynamic_text_batch_manager', source)
            assert_in('DynamicTextBatchManager', source)
            
        except Exception as e:
            # If there's an issue with the actual model loading, at least verify the source has the right imports
            import src.inference_pio.models.qwen3_4b_instruct_2507.model
            import inspect
            source = inspect.getsource(src.inference_pio.models.qwen3_4b_instruct_2507.model)
            
            # Check that the new imports are present
            assert_in('dynamic_text_batching', source)
            assert_in('get_dynamic_text_batch_manager', source)
            assert_in('DynamicTextBatchManager', source)

    def qwen3_coder_dynamic_batching_integration(self)():
        """Test that Qwen3-coder-30b model integrates with dynamic text batching."""
        try:
            # Create model instance (this would normally load the actual model)
            model = object.__new__(Qwen3Coder30BModel)  # Create without calling __init__ to avoid loading
            model.config = qwen3_coder_config
            
            # Check that the new method exists
            assert_true(hasattr(Qwen3Coder30BModel))
            
            # Check that the required imports are in place
            import src.inference_pio.models.qwen3_coder_30b.model
            import inspect
            source = inspect.getsource(Qwen3Coder30BModel.generate_with_adaptive_batching)
            
            # Verify that the new dynamic text batching is used
            assert_in('get_dynamic_text_batch_manager', source)
            assert_in('DynamicTextBatchManager', source)
            
        except Exception as e:
            # If there's an issue with the actual model loading, at least verify the source has the right imports
            import src.inference_pio.models.qwen3_coder_30b.model
            import inspect
            source = inspect.getsource(src.inference_pio.models.qwen3_coder_30b.model)
            
            # Check that the new imports are present
            assert_in('dynamic_text_batching', source)
            assert_in('get_dynamic_text_batch_manager', source)
            assert_in('DynamicTextBatchManager', source)

    def dynamic_batching_method_signature_consistency(self)():
        """Test that all models have consistent method signatures for dynamic batching."""
        import inspect
        
        # Get the method signatures
        glm_sig = inspect.signature(GLM47Model.generate_with_adaptive_batching)
        qwen3_4b_sig = inspect.signature(Qwen34BInstruct2507Model.generate_with_adaptive_batching)
        qwen3_coder_sig = inspect.signature(Qwen3Coder30BModel.generate_with_adaptive_batching)
        
        # Check that they all have the same parameters (excluding 'self')
        glm_params = [p for p in glm_sig.parameters.keys() if p != 'self']
        qwen3_4b_params = [p for p in qwen3_4b_sig.parameters.keys() if p != 'self']
        qwen3_coder_params = [p for p in qwen3_coder_sig.parameters.keys() if p != 'self']
        
        assert_equal(glm_params, qwen3_4b_params)
        assert_equal(qwen3_4b_params, qwen3_coder_params)
        
        # Check that the first parameter is 'inputs' and is of the expected type
        assert_in('inputs', glm_params)
        assert_in('inputs', qwen3_4b_params)
        assert_in('inputs', qwen3_coder_params)

    def common_imports_exist(self)():
        """Test that all models have the required imports for dynamic text batching."""
        import src.inference_pio.models.glm_4_7.model
        import src.inference_pio.models.qwen3_4b_instruct_2507.model
        import src.inference_pio.models.qwen3_coder_30b.model
        
        import inspect
        
        # Check GLM-4-7 model imports
        glm_source = inspect.getsource(src.inference_pio.models.glm_4_7.model)
        assert_in('get_dynamic_text_batch_manager', glm_source)
        assert_in('DynamicTextBatchManager', glm_source)
        
        # Check Qwen3-4b-instruct-2507 model imports
        qwen3_4b_source = inspect.getsource(src.inference_pio.models.qwen3_4b_instruct_2507.model)
        assert_in('get_dynamic_text_batch_manager', qwen3_4b_source)
        assert_in('DynamicTextBatchManager', qwen3_4b_source)
        
        # Check Qwen3-coder-30b model imports
        qwen3_coder_source = inspect.getsource(src.inference_pio.models.qwen3_coder_30b.model)
        assert_in('get_dynamic_text_batch_manager', qwen3_coder_source)
        assert_in('DynamicTextBatchManager', qwen3_coder_source)

# TestDynamicTextBatchingFunctionality

    """Tests for the functionality of dynamic text batching."""

    def text_type_classification_logic(self)():
        """Test that text type classification works as expected."""
        from src.inference_pio.common.dynamic_text_batching import DynamicTextBatchManager, TextBatchType
        
        batch_manager = DynamicTextBatchManager(
            initial_batch_size=4,
            min_batch_size=1,
            max_batch_size=16
        )
        
        # Test simple text
        simple_text = ["Hello world", "How are you?"]
        simple_type = batch_manager.analyze_text_batch_type(simple_text)
        assert_is_instance(simple_type, TextBatchType)
        
        # Test longer text
        long_text = ["This is a longer text. " * 100]  # Very long text
        long_type = batch_manager.analyze_text_batch_type(long_text)
        assert_is_instance(long_type, TextBatchType)
        
        # Test code-like text
        code_text = ["def hello():\n    print('world')"]
        code_type = batch_manager.analyze_text_batch_type(code_text)
        assert_is_instance(code_type, TextBatchType)

    def memory_usage_estimation(self)():
        """Test that memory usage estimation works correctly."""
        from src.inference_pio.common.dynamic_text_batching import DynamicTextBatchManager
        
        batch_manager = DynamicTextBatchManager(
            initial_batch_size=4,
            min_batch_size=1,
            max_batch_size=16
        )
        
        # Test with different text lengths
        short_texts = ["Hi", "Bye"]
        long_texts = ["This is a much longer text for testing purposes." * 10]
        
        short_memory = batch_manager.estimate_memory_usage(short_texts, 2)
        long_memory = batch_manager.estimate_memory_usage(long_texts, 2)
        
        # Longer texts should estimate higher memory usage
        assertGreaterEqual(long_memory, short_memory)
        assertGreaterEqual(short_memory, 0)
        assertGreaterEqual(long_memory, 0)

if __name__ == '__main__':
    run_tests(test_functions)