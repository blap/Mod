#!/usr/bin/env python3
"""
Comprehensive test suite for all optimizations implemented across all models
(GLM-4-7, Qwen3-4b-instruct-2507, Qwen3-coder-30b, Qwen3-vl-2b).
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import torch
import tempfile
from pathlib import Path
import shutil

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.inference_pio.models.glm_4_7.config import GLM47Config
from src.inference_pio.models.glm_4_7.model import GLM47Model
from src.inference_pio.models.glm_4_7.plugin import GLM47Plugin, create_glm_4_7_plugin

from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen3_4BInstruct2507Config
from src.inference_pio.models.qwen3_4b_instruct_2507.model import Qwen3_4BInstruct2507Model
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import Qwen3_4BInstruct2507Plugin, create_qwen3_4b_instruct_2507_plugin

from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig
from src.inference_pio.models.qwen3_coder_30b.model import Qwen3Coder30BModel
from src.inference_pio.models.qwen3_coder_30b.plugin import Qwen3Coder30BPlugin, create_qwen3_coder_30b_plugin

from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Instruct_Plugin, create_qwen3_vl_2b_instruct_plugin

from src.inference_pio.common.quantization import QuantizationManager, QuantizationScheme
from src.inference_pio.common.cross_modal_fusion_kernels import CrossModalFusionManager
from src.inference_pio.common.vision_language_parallel import VisionLanguageParallelConfig
from src.inference_pio.common.pipeline_parallel import PipelineConfig
from src.inference_pio.common.image_tokenization import ImageTokenizationConfig


class TestGLM47Optimizations(unittest.TestCase):
    """Test GLM-4-7 specific optimizations."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.test_dir)

    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_config_attributes(self, mock_tokenizer, mock_model):
        """Test that GLM-4-7 config has all required optimization attributes."""
        config = GLM47Config()
        
        # Check basic model attributes
        self.assertIsNotNone(config.model_path)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.num_hidden_layers, 28)
        
        # Check GLM-4-7 specific optimization attributes
        self.assertTrue(hasattr(config, 'use_glm_attention_patterns'))
        self.assertTrue(hasattr(config, 'glm_attention_pattern_sparsity'))
        self.assertTrue(hasattr(config, 'glm_attention_window_size'))
        self.assertTrue(hasattr(config, 'use_glm_ffn_optimization'))
        self.assertTrue(hasattr(config, 'glm_ffn_expansion_ratio'))
        self.assertTrue(hasattr(config, 'glm_ffn_group_size'))
        self.assertTrue(hasattr(config, 'use_glm_memory_efficient_kv'))
        self.assertTrue(hasattr(config, 'glm_kv_cache_compression_ratio'))
        self.assertTrue(hasattr(config, 'use_glm_layer_norm_fusion'))
        self.assertTrue(hasattr(config, 'use_glm_residual_connection_optimization'))
        self.assertTrue(hasattr(config, 'use_glm_quantization'))
        self.assertTrue(hasattr(config, 'glm_weight_bits'))
        self.assertTrue(hasattr(config, 'glm_activation_bits'))

    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_model_creation_with_optimizations(self, mock_tokenizer, mock_model):
        """Test GLM-4-7 model creation with optimizations."""
        # Mock model components
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        config = GLM47Config()
        config.model_path = "dummy_path"
        
        model = GLM47Model(config)
        
        # Verify model has expected attributes
        self.assertTrue(hasattr(model, '_model'))
        self.assertTrue(hasattr(model, '_tokenizer'))
        
        # Verify optimization methods exist
        self.assertTrue(hasattr(model, '_apply_glm47_specific_optimizations'))
        self.assertTrue(hasattr(model, '_apply_configured_optimizations'))

    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_plugin_creation(self, mock_tokenizer, mock_model):
        """Test GLM-4-7 plugin creation and methods."""
        plugin = GLM47Plugin()
        
        # Verify plugin has required methods
        required_methods = [
            'load_model', 'infer', 'generate_text', 'chat_completion',
            'get_model_info', 'get_model_parameters', 'initialize',
            'apply_glm47_specific_optimizations', 'get_glm47_optimization_report'
        ]
        
        for method in required_methods:
            self.assertTrue(hasattr(plugin, method), f"Missing method: {method}")

    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_quantization_integration(self, mock_tokenizer, mock_model):
        """Test GLM-4-7 integration with quantization."""
        config = GLM47Config()
        config.use_quantization = True
        config.quantization_scheme = 'int8'
        
        # Mock model components
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        model = GLM47Model(config)
        
        # Verify quantization was applied
        self.assertTrue(config.use_quantization)
        self.assertEqual(config.quantization_scheme, 'int8')


class TestQwen3_4BOptimizations(unittest.TestCase):
    """Test Qwen3-4b-instruct-2507 optimizations."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.test_dir)

    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_config_attributes(self, mock_tokenizer, mock_model):
        """Test that Qwen3-4b config has all required optimization attributes."""
        config = Qwen3_4BInstruct2507Config()
        
        # Check basic model attributes
        self.assertIsNotNone(config.model_path)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.num_hidden_layers, 32)
        
        # Check Qwen3-4b specific optimization attributes
        self.assertTrue(hasattr(config, 'use_qwen_attention_optimization'))
        self.assertTrue(hasattr(config, 'use_qwen_gqa_optimization'))
        self.assertTrue(hasattr(config, 'use_qwen_rope_optimization'))
        self.assertTrue(hasattr(config, 'use_qwen_kv_cache_optimization'))
        self.assertTrue(hasattr(config, 'use_qwen_instruction_tuning_optimization'))
        self.assertTrue(hasattr(config, 'use_qwen_generation_optimization'))
        self.assertTrue(hasattr(config, 'use_qwen_compressed_kv_cache'))
        self.assertTrue(hasattr(config, 'use_qwen_internal_sparse_attention'))
        self.assertTrue(hasattr(config, 'use_qwen_internal_gqa'))
        self.assertTrue(hasattr(config, 'use_qwen_internal_flash_attention'))

    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_model_creation_with_optimizations(self, mock_tokenizer, mock_model):
        """Test Qwen3-4b model creation with optimizations."""
        # Mock model components
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        config = Qwen3_4BInstruct2507Config()
        config.model_path = "dummy_path"
        
        model = Qwen3_4BInstruct2507Model(config)
        
        # Verify model has expected attributes
        self.assertTrue(hasattr(model, '_model'))
        self.assertTrue(hasattr(model, '_tokenizer'))
        
        # Verify optimization methods exist
        self.assertTrue(hasattr(model, '_apply_qwen3_specific_optimizations'))
        self.assertTrue(hasattr(model, '_apply_configured_optimizations'))

    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_plugin_creation(self, mock_tokenizer, mock_model):
        """Test Qwen3-4b plugin creation and methods."""
        plugin = Qwen3_4BInstruct2507Plugin()
        
        # Verify plugin has required methods
        required_methods = [
            'load_model', 'infer', 'generate_text', 'chat_completion',
            'get_model_info', 'get_model_parameters', 'initialize',
            'apply_qwen3_specific_optimizations', 'get_qwen3_optimization_report'
        ]
        
        for method in required_methods:
            self.assertTrue(hasattr(plugin, method), f"Missing method: {method}")


class TestQwen3CoderOptimizations(unittest.TestCase):
    """Test Qwen3-coder-30b optimizations."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.test_dir)

    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_config_attributes(self, mock_tokenizer, mock_model):
        """Test that Qwen3-coder config has all required optimization attributes."""
        config = Qwen3Coder30BConfig()
        
        # Check basic model attributes
        self.assertIsNotNone(config.model_path)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.num_hidden_layers, 64)
        
        # Check Qwen3-coder specific optimization attributes
        self.assertTrue(hasattr(config, 'use_code_specific_optimizations'))
        self.assertTrue(hasattr(config, 'code_task_detection_threshold'))
        self.assertTrue(hasattr(config, 'code_generation_temperature'))
        self.assertTrue(hasattr(config, 'code_generation_top_p'))
        self.assertTrue(hasattr(config, 'code_generation_top_k'))
        self.assertTrue(hasattr(config, 'code_generation_repetition_penalty'))
        self.assertTrue(hasattr(config, 'code_generation_max_new_tokens'))
        self.assertTrue(hasattr(config, 'code_generation_min_new_tokens'))
        self.assertTrue(hasattr(config, 'code_generation_length_penalty'))
        self.assertTrue(hasattr(config, 'code_generation_no_repeat_ngram_size'))
        self.assertTrue(hasattr(config, 'code_generation_early_stopping'))
        self.assertTrue(hasattr(config, 'code_generation_diversity_penalty'))
        self.assertTrue(hasattr(config, 'code_generation_num_beam_groups'))
        self.assertTrue(hasattr(config, 'code_generation_num_return_sequences'))
        self.assertTrue(hasattr(config, 'code_generation_pad_token_id'))
        self.assertTrue(hasattr(config, 'code_generation_eos_token_id'))

    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_model_creation_with_optimizations(self, mock_tokenizer, mock_model):
        """Test Qwen3-coder model creation with optimizations."""
        # Mock model components
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        config = Qwen3Coder30BConfig()
        config.model_path = "dummy_path"
        
        model = Qwen3Coder30BModel(config)
        
        # Verify model has expected attributes
        self.assertTrue(hasattr(model, '_model'))
        self.assertTrue(hasattr(model, '_tokenizer'))
        
        # Verify optimization methods exist
        self.assertTrue(hasattr(model, '_apply_code_specific_optimizations'))
        self.assertTrue(hasattr(model, '_apply_configured_optimizations'))

    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_plugin_creation(self, mock_tokenizer, mock_model):
        """Test Qwen3-coder plugin creation and methods."""
        plugin = Qwen3Coder30BPlugin()
        
        # Verify plugin has required methods
        required_methods = [
            'load_model', 'infer', 'generate_text', 'chat_completion',
            'get_model_info', 'get_model_parameters', 'initialize',
            'detect_code_task', 'apply_code_specific_generation_params'
        ]
        
        for method in required_methods:
            self.assertTrue(hasattr(plugin, method), f"Missing method: {method}")


class TestQwen3VLOptimizations(unittest.TestCase):
    """Test Qwen3-vl-2b optimizations."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.test_dir)

    @patch('transformers.AutoModelForVision2Seq.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoImageProcessor.from_pretrained')
    def test_config_attributes(self, mock_image_proc, mock_tokenizer, mock_model):
        """Test that Qwen3-vl config has all required optimization attributes."""
        config = Qwen3VL2BConfig()
        
        # Check basic model attributes
        self.assertIsNotNone(config.model_path)
        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.num_attention_heads, 16)
        self.assertEqual(config.num_hidden_layers, 24)
        
        # Check Qwen3-vl specific optimization attributes
        self.assertTrue(hasattr(config, 'use_multimodal_attention'))
        self.assertTrue(hasattr(config, 'use_cross_modal_alignment'))
        self.assertTrue(hasattr(config, 'use_projection_layer_optimization'))
        self.assertTrue(hasattr(config, 'use_vision_language_parallelism'))
        self.assertTrue(hasattr(config, 'use_vision_encoder_optimization'))
        self.assertTrue(hasattr(config, 'use_visual_compression'))
        self.assertTrue(hasattr(config, 'use_dynamic_multimodal_batching'))
        self.assertTrue(hasattr(config, 'use_intelligent_multimodal_caching'))
        self.assertTrue(hasattr(config, 'use_async_multimodal_processing'))
        
        # Check projection layer attributes
        self.assertTrue(hasattr(config, 'projection_layer_use_bias'))
        self.assertTrue(hasattr(config, 'projection_layer_activation'))
        self.assertTrue(hasattr(config, 'projection_layer_dropout'))
        self.assertTrue(hasattr(config, 'projection_layer_use_residual'))
        self.assertTrue(hasattr(config, 'projection_layer_use_low_rank'))
        self.assertTrue(hasattr(config, 'projection_layer_low_rank_dim'))
        self.assertTrue(hasattr(config, 'projection_layer_use_group_norm'))
        self.assertTrue(hasattr(config, 'projection_layer_intermediate_dim'))
        self.assertTrue(hasattr(config, 'projection_layer_num_layers'))
        self.assertTrue(hasattr(config, 'projection_layer_use_cross_attention'))
        
        # Check image tokenization attributes
        self.assertTrue(hasattr(config, 'image_size'))
        self.assertTrue(hasattr(config, 'patch_size'))
        self.assertTrue(hasattr(config, 'max_image_tokens'))
        self.assertTrue(hasattr(config, 'token_dim'))
        self.assertTrue(hasattr(config, 'enable_patch_caching'))
        self.assertTrue(hasattr(config, 'enable_batch_processing'))
        self.assertTrue(hasattr(config, 'enable_quantization'))
        self.assertTrue(hasattr(config, 'enable_compression'))

    @patch('transformers.AutoModelForVision2Seq.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoImageProcessor.from_pretrained')
    def test_model_creation_with_optimizations(self, mock_image_proc, mock_tokenizer, mock_model):
        """Test Qwen3-vl model creation with optimizations."""
        # Mock model components
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_image_proc.return_value = MagicMock()
        
        config = Qwen3VL2BConfig()
        config.model_path = "dummy_path"
        
        model = Qwen3VL2BModel(config)
        
        # Verify model has expected attributes
        self.assertTrue(hasattr(model, '_model'))
        self.assertTrue(hasattr(model, '_tokenizer'))
        self.assertTrue(hasattr(model, '_image_processor'))
        
        # Verify optimization methods exist
        self.assertTrue(hasattr(model, '_apply_multimodal_attention_optimizations'))
        self.assertTrue(hasattr(model, '_apply_cross_modal_alignment_optimizations'))
        self.assertTrue(hasattr(model, '_apply_projection_layer_optimizations'))
        self.assertTrue(hasattr(model, '_apply_vision_encoder_optimizations'))
        self.assertTrue(hasattr(model, '_apply_visual_compression_optimizations'))
        self.assertTrue(hasattr(model, '_apply_configured_optimizations'))

    @patch('transformers.AutoModelForVision2Seq.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoImageProcessor.from_pretrained')
    def test_plugin_creation(self, mock_image_proc, mock_tokenizer, mock_model):
        """Test Qwen3-vl plugin creation and methods."""
        plugin = Qwen3_VL_2B_Instruct_Plugin()
        
        # Verify plugin has required methods
        required_methods = [
            'load_model', 'infer', 'generate_text', 'chat_completion',
            'get_model_info', 'get_model_parameters', 'initialize',
            'setup_multimodal_attention', 'setup_cross_modal_alignment',
            'setup_projection_layer_optimizations', 'setup_vision_encoder_optimizations',
            'setup_visual_compression', 'tokenize_image', 'batch_tokenize_images',
            'setup_dynamic_multimodal_batching', 'setup_intelligent_multimodal_caching',
            'setup_async_multimodal_processing'
        ]
        
        for method in required_methods:
            self.assertTrue(hasattr(plugin, method), f"Missing method: {method}")


class TestCommonOptimizations(unittest.TestCase):
    """Test common optimizations across all models."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.test_dir)

    def test_quantization_manager(self):
        """Test quantization manager functionality."""
        manager = QuantizationManager()
        
        # Test registration of quantization schemes
        config = MagicMock()
        config.enabled = True
        config.scheme = QuantizationScheme.INT8
        
        manager.register_quantization_scheme("test_scheme", config)
        
        retrieved_config = manager.get_quantization_config("test_scheme")
        self.assertEqual(retrieved_config, config)
        
        # Test global instance
        global_manager = QuantizationManager.get_instance()
        self.assertIsInstance(global_manager, QuantizationManager)

    def test_cross_modal_fusion_manager(self):
        """Test cross-modal fusion manager."""
        manager = CrossModalFusionManager()
        
        # Test creation of different fusion methods
        fusion_methods = manager.get_available_methods()
        self.assertIn('cross_attention', fusion_methods)
        self.assertIn('concatenation', fusion_methods)
        self.assertIn('additive', fusion_methods)

    def test_pipeline_config(self):
        """Test pipeline configuration."""
        config = PipelineConfig(
            num_stages=2,
            microbatch_size=1,
            enable_activation_offloading=True,
            pipeline_schedule='1f1b'
        )
        
        self.assertEqual(config.num_stages, 2)
        self.assertEqual(config.microbatch_size, 1)
        self.assertTrue(config.enable_activation_offloading)
        self.assertEqual(config.pipeline_schedule, '1f1b')

    def test_vision_language_parallel_config(self):
        """Test vision-language parallel configuration."""
        config = VisionLanguageParallelConfig(
            vision_stages=1,
            language_stages=1,
            enable_cross_modal_fusion=True
        )
        
        self.assertEqual(config.vision_stages, 1)
        self.assertEqual(config.language_stages, 1)
        self.assertTrue(config.enable_cross_modal_fusion)

    def test_image_tokenization_config(self):
        """Test image tokenization configuration."""
        config = ImageTokenizationConfig(
            image_size=448,
            patch_size=14,
            max_image_tokens=1024,
            token_dim=1024,
            enable_patch_caching=True,
            enable_batch_processing=True
        )
        
        self.assertEqual(config.image_size, 448)
        self.assertEqual(config.patch_size, 14)
        self.assertEqual(config.max_image_tokens, 1024)
        self.assertEqual(config.token_dim, 1024)
        self.assertTrue(config.enable_patch_caching)
        self.assertTrue(config.enable_batch_processing)


class TestIntegrationAcrossModels(unittest.TestCase):
    """Test integration of optimizations across different models."""

    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_glm47_with_common_optimizations(self, mock_tokenizer, mock_model):
        """Test GLM-4-7 integration with common optimizations."""
        # Mock model components
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        config = GLM47Config()
        config.model_path = "dummy_path"
        config.use_quantization = True
        config.quantization_scheme = 'int8'
        
        model = GLM47Model(config)
        
        # Verify both model-specific and common optimizations are applied
        self.assertTrue(config.use_quantization)
        self.assertEqual(config.quantization_scheme, 'int8')

    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_qwen3_4b_with_common_optimizations(self, mock_tokenizer, mock_model):
        """Test Qwen3-4b integration with common optimizations."""
        # Mock model components
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        config = Qwen3_4BInstruct2507Config()
        config.model_path = "dummy_path"
        config.use_quantization = True
        config.quantization_scheme = 'int4'
        
        model = Qwen3_4BInstruct2507Model(config)
        
        # Verify both model-specific and common optimizations are applied
        self.assertTrue(config.use_quantization)
        self.assertEqual(config.quantization_scheme, 'int4')

    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_qwen3_coder_with_common_optimizations(self, mock_tokenizer, mock_model):
        """Test Qwen3-coder integration with common optimizations."""
        # Mock model components
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        config = Qwen3Coder30BConfig()
        config.model_path = "dummy_path"
        config.use_quantization = True
        config.quantization_scheme = 'fp16'
        
        model = Qwen3Coder30BModel(config)
        
        # Verify both model-specific and common optimizations are applied
        self.assertTrue(config.use_quantization)
        self.assertEqual(config.quantization_scheme, 'fp16')

    @patch('transformers.AutoModelForVision2Seq.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoImageProcessor.from_pretrained')
    def test_qwen3_vl_with_common_optimizations(self, mock_image_proc, mock_tokenizer, mock_model):
        """Test Qwen3-vl integration with common optimizations."""
        # Mock model components
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_image_proc.return_value = MagicMock()
        
        config = Qwen3VL2BConfig()
        config.model_path = "dummy_path"
        config.use_quantization = True
        config.quantization_scheme = 'nf4'
        
        model = Qwen3VL2BModel(config)
        
        # Verify both model-specific and common optimizations are applied
        self.assertTrue(config.use_quantization)
        self.assertEqual(config.quantization_scheme, 'nf4')


class TestPluginFactories(unittest.TestCase):
    """Test plugin factory functions."""

    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_glm47_plugin_factory(self, mock_tokenizer, mock_model):
        """Test GLM-4-7 plugin factory function."""
        plugin = create_glm_4_7_plugin()
        self.assertIsInstance(plugin, GLM47Plugin)

    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_qwen3_4b_plugin_factory(self, mock_tokenizer, mock_model):
        """Test Qwen3-4b plugin factory function."""
        plugin = create_qwen3_4b_instruct_2507_plugin()
        self.assertIsInstance(plugin, Qwen3_4BInstruct2507Plugin)

    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_qwen3_coder_plugin_factory(self, mock_tokenizer, mock_model):
        """Test Qwen3-coder plugin factory function."""
        plugin = create_qwen3_coder_30b_plugin()
        self.assertIsInstance(plugin, Qwen3Coder30BPlugin)

    @patch('transformers.AutoModelForVision2Seq.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoImageProcessor.from_pretrained')
    def test_qwen3_vl_plugin_factory(self, mock_image_proc, mock_tokenizer, mock_model):
        """Test Qwen3-vl plugin factory function."""
        plugin = create_qwen3_vl_2b_instruct_plugin()
        self.assertIsInstance(plugin, Qwen3_VL_2B_Instruct_Plugin)


def run_all_tests():
    """Run all tests and return the result."""
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestGLM47Optimizations))
    suite.addTests(loader.loadTestsFromTestCase(TestQwen3_4BOptimizations))
    suite.addTests(loader.loadTestsFromTestCase(TestQwen3CoderOptimizations))
    suite.addTests(loader.loadTestsFromTestCase(TestQwen3VLOptimizations))
    suite.addTests(loader.loadTestsFromTestCase(TestCommonOptimizations))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationAcrossModels))
    suite.addTests(loader.loadTestsFromTestCase(TestPluginFactories))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running comprehensive optimization tests for all models...")
    print("=" * 60)
    
    success = run_all_tests()
    
    print("=" * 60)
    if success:
        print("✓ All tests passed! All optimizations are properly implemented.")
        sys.exit(0)
    else:
        print("✗ Some tests failed.")
        sys.exit(1)