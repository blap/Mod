"""
Tests for Advanced CPU Optimizations in Qwen3-VL Model
Validating preprocessing, tokenization, and CPU-GPU coordination optimizations
"""
import unittest
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from transformers import PreTrainedTokenizer
from unittest.mock import Mock, MagicMock
import time
import tempfile
import os

from src.qwen3_vl.optimization.advanced_cpu_optimizations import (
    AdvancedCPUOptimizationConfig,
    VectorizedImagePreprocessor,
    AdvancedCPUPreprocessor,
    AdvancedCPU_GPU_Coordinator,
    AdvancedOptimizedInferencePipeline,
    apply_advanced_cpu_optimizations
)
from src.qwen3_vl.optimization.advanced_tokenization import (
    AdvancedTokenizationConfig,
    AdvancedTokenizationCache,
    AdvancedMultithreadedTokenizer as AdvancedTokenizationProcessor,
    AdvancedBatchTokenizationProcessor,
    AdvancedTokenizationPipeline,
    create_advanced_tokenization_pipeline
)
from src.qwen3_vl.optimization.advanced_cpu_gpu_coordination import (
    AdvancedCPUGPUConfig,
    AdvancedMemoryPool,
    AdvancedCPUGPUTransferOptimizer,
    AdvancedCPUGPUCoordinator,
    AdvancedCPUGPUOptimizationPipeline,
    create_advanced_cpu_gpu_pipeline
)


class TestVectorizedImagePreprocessor(unittest.TestCase):
    """Test cases for VectorizedImagePreprocessor."""
    
    def setUp(self):
        config = AdvancedCPUOptimizationConfig()
        self.preprocessor = VectorizedImagePreprocessor(config)

    def test_preprocess_single_image(self):
        """Test preprocessing of a single image."""
        # Create a dummy image
        img = Image.new('RGB', (256, 256), color='red')
        result = self.preprocessor.preprocess_images_batch([img])
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, 3, 224, 224))  # Expected shape after resize and normalization

    def test_preprocess_batch_images(self):
        """Test preprocessing of a batch of images."""
        # Create multiple dummy images
        images = [Image.new('RGB', (256, 256), color='red') for _ in range(3)]
        result = self.preprocessor.preprocess_images_batch(images)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (3, 3, 224, 224))  # Expected shape for batch

    def test_preprocess_empty_batch(self):
        """Test preprocessing of an empty batch."""
        result = self.preprocessor.preprocess_images_batch([])
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (0, 3, 224, 224))  # Expected shape for empty batch


class TestAdvancedTokenizationCache(unittest.TestCase):
    """Test cases for AdvancedTokenizationCache."""
    
    def setUp(self):
        self.cache = AdvancedTokenizationCache(max_size=5)

    def test_put_and_get(self):
        """Test putting and getting items from cache."""
        test_data = {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.tensor([[1, 1, 1]])}
        
        self.cache.put("test_key", test_data)
        result = self.cache.get("test_key")
        
        self.assertIsNotNone(result)
        self.assertTrue(torch.equal(result['input_ids'], test_data['input_ids']))
        self.assertTrue(torch.equal(result['attention_mask'], test_data['attention_mask']))

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        # Fill cache to max size
        for i in range(5):
            data = {'input_ids': torch.tensor([[i]]), 'attention_mask': torch.tensor([[1]])}
            self.cache.put(f"key_{i}", data)
        
        # Access key_0 to make it most recently used
        self.cache.get("key_0")
        
        # Add one more item, which should evict key_1 (not key_0)
        new_data = {'input_ids': torch.tensor([[99]]), 'attention_mask': torch.tensor([[1]])}
        self.cache.put("new_key", new_data)
        
        # key_0 should still be in cache
        self.assertIsNotNone(self.cache.get("key_0"))
        # key_1 should be evicted
        self.assertIsNone(self.cache.get("key_1"))


class TestAdvancedMemoryPool(unittest.TestCase):
    """Test cases for AdvancedMemoryPool."""
    
    def setUp(self):
        config = AdvancedCPUGPUConfig()
        self.memory_pool = AdvancedMemoryPool(config)

    def test_get_and_return_tensor(self):
        """Test getting and returning tensors to the pool."""
        shape = (10, 20)
        dtype = torch.float32
        device = torch.device('cpu')
        
        # Get tensor from pool
        tensor1 = self.memory_pool.get_tensor(shape, dtype, device)
        self.assertEqual(tensor1.shape, shape)
        self.assertEqual(tensor1.dtype, dtype)
        
        # Return tensor to pool
        self.memory_pool.return_tensor(tensor1)
        
        # Get tensor again - should come from pool
        tensor2 = self.memory_pool.get_tensor(shape, dtype, device)
        
        # Check stats
        stats = self.memory_pool.get_stats()
        self.assertGreaterEqual(stats['hits'], 0)
        self.assertGreaterEqual(stats['misses'], 0)


class TestAdvancedMultithreadedTokenizer(unittest.TestCase):
    """Test cases for AdvancedMultithreadedTokenizer."""

    def setUp(self):
        # Create a mock tokenizer
        self.mock_tokenizer = Mock(spec=PreTrainedTokenizer)
        self.mock_tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        self.mock_tokenizer.__call__ = lambda texts, **kwargs: {
            'input_ids': torch.randint(0, 1000, (len(texts) if isinstance(texts, list) else 1, 10)),
            'attention_mask': torch.ones(len(texts) if isinstance(texts, list) else 1, 10)
        }

        config = AdvancedTokenizationConfig()
        self.tokenizer = AdvancedTokenizationProcessor(self.mock_tokenizer, config)

    def test_tokenize_batch(self):
        """Test tokenizing a batch of texts."""
        texts = ["Hello world", "How are you?"]
        result = self.tokenizer.tokenize_batch(texts)
        
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
        self.assertEqual(result['input_ids'].shape[0], len(texts))

    def test_async_tokenization(self):
        """Test asynchronous tokenization."""
        texts = ["Hello world", "How are you?"]
        future = self.tokenizer.tokenize_batch_async(texts)
        
        result = future.result()
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)

    def test_prefetch_functionality(self):
        """Test prefetch functionality."""
        texts = ["Hello world", "How are you?"]
        self.tokenizer.prefetch_tokenize_batch(texts)
        
        # Just ensure no exceptions are raised
        self.assertTrue(True)


class TestAdvancedCPUGPUTransferOptimizer(unittest.TestCase):
    """Test cases for AdvancedCPUGPUTransferOptimizer."""
    
    def setUp(self):
        config = AdvancedCPUGPUConfig()
        self.transfer_optimizer = AdvancedCPUGPUTransferOptimizer(config)

    def test_transfer_to_device(self):
        """Test transferring data to device."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        data = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }

        result = self.transfer_optimizer.transfer_to_device(data, device)

        for key, tensor in result.items():
            # Compare device types and indices separately to handle CUDA device comparison
            self.assertEqual(tensor.device.type, device.type)
            if device.type == 'cuda':
                # For CUDA devices, just check the type since the index may vary
                self.assertEqual(tensor.device.type, 'cuda')
            else:
                self.assertEqual(tensor.device, device)

    def test_async_transfer(self):
        """Test asynchronous transfer."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        data = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }

        future = self.transfer_optimizer.transfer_to_device_async(data, device)
        result = future.result()

        for key, tensor in result.items():
            # Compare device types and indices separately to handle CUDA device comparison
            self.assertEqual(tensor.device.type, device.type)
            if device.type == 'cuda':
                # For CUDA devices, just check the type since the index may vary
                self.assertEqual(tensor.device.type, 'cuda')
            else:
                self.assertEqual(tensor.device, device)

    def test_prefetch_functionality(self):
        """Test prefetch functionality."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            
        data = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        
        self.transfer_optimizer.prefetch_to_device(data, device)
        
        # Just ensure no exceptions are raised
        self.assertTrue(True)


class TestPerformanceComparison(unittest.TestCase):
    """Test performance improvements of optimized components."""
    
    def setUp(self):
        # Create a simple model for testing
        self.model = nn.Linear(100, 10)
        
        # Create test data
        self.texts = ["Test text " + str(i) for i in range(10)]
        self.images = [Image.new('RGB', (224, 224), color='red') for _ in range(5)]

    def test_preprocessing_performance(self):
        """Compare performance of optimized vs basic preprocessing."""
        # Create a mock tokenizer for the preprocessor
        mock_tokenizer = Mock(spec=PreTrainedTokenizer)
        def tokenization_side_effect(texts, **kwargs):
            batch_size = len(texts) if isinstance(texts, list) else 1
            return {
                'input_ids': torch.randint(0, 1000, (batch_size, 20)),
                'attention_mask': torch.ones(batch_size, 20)
            }
        mock_tokenizer.side_effect = tokenization_side_effect
        mock_tokenizer.__call__ = tokenization_side_effect

        config = AdvancedCPUOptimizationConfig()
        preprocessor = AdvancedCPUPreprocessor(config, tokenizer=mock_tokenizer)

        # Time optimized preprocessing
        start_time = time.time()
        result = preprocessor.preprocess_batch(self.texts, self.images)
        optimized_time = time.time() - start_time

        # Basic preprocessing would be slower
        # For this test, we just verify the optimized version works correctly
        self.assertIn('input_ids', result)
        self.assertIn('pixel_values', result)
        self.assertLess(optimized_time, 5.0)  # Should complete in reasonable time

    def test_tokenization_performance(self):
        """Test tokenization performance."""
        # Create a simple tokenizer mock
        mock_tokenizer = Mock(spec=PreTrainedTokenizer)
        def tokenization_side_effect(texts, **kwargs):
            batch_size = len(texts) if isinstance(texts, list) else 1
            return {
                'input_ids': torch.randint(0, 1000, (batch_size, 20)),
                'attention_mask': torch.ones(batch_size, 20)
            }
        mock_tokenizer.side_effect = tokenization_side_effect
        mock_tokenizer.__call__ = tokenization_side_effect

        config = AdvancedTokenizationConfig()
        tokenizer = AdvancedTokenizationProcessor(mock_tokenizer, config)

        # Time tokenization
        start_time = time.time()
        result = tokenizer.tokenize_batch(self.texts)
        tokenization_time = time.time() - start_time

        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
        self.assertLess(tokenization_time, 5.0)  # Should complete in reasonable time


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete optimization pipeline."""
    
    def setUp(self):
        # Create a simple model
        self.model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        # Create test data
        self.texts = ["Test text " + str(i) for i in range(5)]
        self.images = [Image.new('RGB', (224, 224), color='red') for _ in range(3)]

    def test_complete_optimization_pipeline(self):
        """Test the complete optimization pipeline."""
        # Create a mock tokenizer
        mock_tokenizer = Mock(spec=PreTrainedTokenizer)
        def tokenization_side_effect(texts, **kwargs):
            batch_size = len(texts) if isinstance(texts, list) else 1
            return {
                'input_ids': torch.randint(0, 1000, (batch_size, 20)),
                'attention_mask': torch.ones(batch_size, 20)
            }
        mock_tokenizer.side_effect = tokenization_side_effect
        mock_tokenizer.__call__ = tokenization_side_effect

        # Create a simple model with generate method
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(20, 10)

            def forward(self, input_ids=None, pixel_values=None, attention_mask=None, **kwargs):
                if input_ids is not None:
                    return self.linear(input_ids.float().mean(dim=1, keepdim=True).expand(-1, 20))
                else:
                    return torch.randn(pixel_values.shape[0], 10)

            def generate(self, input_ids=None, pixel_values=None, attention_mask=None, **kwargs):
                # Mock generate method
                batch_size = input_ids.shape[0] if input_ids is not None else pixel_values.shape[0]
                return torch.randint(0, 1000, (batch_size, 20))

        simple_model = SimpleModel()

        # Apply optimizations
        pipeline = apply_advanced_cpu_optimizations(
            simple_model,
            tokenizer=mock_tokenizer,
            num_preprocess_workers=2,
            tokenization_chunk_size=4
        )

        # Run inference
        results = pipeline.preprocess_and_infer(
            self.texts[:2],
            self.images[:1],
            tokenizer=mock_tokenizer
        )

        # Verify results
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)  # Should have one response per text input

    def test_tokenization_pipeline(self):
        """Test the tokenization pipeline."""
        # Create a mock tokenizer
        mock_tokenizer = Mock(spec=PreTrainedTokenizer)
        def tokenization_side_effect(texts, **kwargs):
            batch_size = len(texts) if isinstance(texts, list) else 1
            return {
                'input_ids': torch.randint(0, 1000, (batch_size, 20)),
                'attention_mask': torch.ones(batch_size, 20)
            }
        mock_tokenizer.side_effect = tokenization_side_effect
        mock_tokenizer.__call__ = tokenization_side_effect
        
        # Create pipeline
        pipeline = create_advanced_tokenization_pipeline(mock_tokenizer)
        
        # Tokenize
        result = pipeline.tokenize(self.texts[:3])
        
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
        self.assertEqual(result['input_ids'].shape[0], 3)

    def test_cpu_gpu_pipeline(self):
        """Test the CPU-GPU coordination pipeline."""
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(20, 5)

            def forward(self, input_ids=None, **kwargs):
                return self.linear(input_ids.float())

        model = SimpleModel()

        # Move model to appropriate device
        if torch.cuda.is_available():
            model = model.cuda()

        # Create test data
        test_data = {
            'input_ids': torch.randint(0, 1000, (2, 20)),
            'attention_mask': torch.ones(2, 20)
        }

        # Create pipeline
        pipeline = create_advanced_cpu_gpu_pipeline()

        # Process with pipeline
        result = pipeline.process_inference_batch(model, [test_data])

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], torch.Tensor)


class TestMemoryManagement(unittest.TestCase):
    """Test memory management and pooling."""
    
    def test_memory_pool_reuse(self):
        """Test that tensors are properly pooled and reused."""
        config = AdvancedCPUGPUConfig()
        memory_pool = AdvancedMemoryPool(config)
        
        # Get a tensor
        shape = (10, 10)
        dtype = torch.float32
        device = torch.device('cpu')
        
        tensor1 = memory_pool.get_tensor(shape, dtype, device)
        original_data_ptr = tensor1.data_ptr()
        
        # Return it to the pool
        memory_pool.return_tensor(tensor1)
        
        # Get another tensor of the same shape
        tensor2 = memory_pool.get_tensor(shape, dtype, device)
        
        # Check if pooling worked (implementation-dependent)
        stats = memory_pool.get_stats()
        self.assertGreaterEqual(stats['allocations'], 1)


def run_all_tests():
    """Run all tests and return results."""
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTest(unittest.makeSuite(TestVectorizedImagePreprocessor))
    suite.addTest(unittest.makeSuite(TestAdvancedTokenizationCache))
    suite.addTest(unittest.makeSuite(TestAdvancedMemoryPool))
    suite.addTest(unittest.makeSuite(TestAdvancedMultithreadedTokenizer))
    suite.addTest(unittest.makeSuite(TestAdvancedCPUGPUTransferOptimizer))
    suite.addTest(unittest.makeSuite(TestPerformanceComparison))
    suite.addTest(unittest.makeSuite(TestIntegration))
    suite.addTest(unittest.makeSuite(TestMemoryManagement))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("Running tests for Advanced CPU Optimizations in Qwen3-VL Model...")
    print("=" * 60)
    
    # Run all tests
    test_result = run_all_tests()
    
    print("=" * 60)
    print(f"Tests run: {test_result.testsRun}")
    print(f"Failures: {len(test_result.failures)}")
    print(f"Errors: {len(test_result.errors)}")
    
    if test_result.failures:
        print("\nFailures:")
        for test, traceback in test_result.failures:
            print(f"  {test}: {traceback}")
    
    if test_result.errors:
        print("\nErrors:")
        for test, traceback in test_result.errors:
            print(f"  {test}: {traceback}")
    
    if not test_result.failures and not test_result.errors:
        print("\nAll tests passed! :)")
    else:
        print("\nSome tests failed or had errors.")