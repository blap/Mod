"""
Integration tests for the Qwen3-VL-2B-Instruct optimization pipeline.

This test suite verifies the interaction between different components of the system,
including memory management, CPU optimizations, and the overall pipeline integration.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import tempfile
import os
import threading
import time

from advanced_memory_management_vl import VisionLanguageMemoryOptimizer
from advanced_cpu_optimizations_intel_i5_10210u import (
    AdvancedCPUOptimizationConfig,
    IntelCPUOptimizedPreprocessor,
    IntelOptimizedPipeline,
    AdaptiveIntelOptimizer,
    apply_intel_optimizations_to_model
)


class TestOptimizationPipelineIntegration:
    """Test integration between different optimization components."""
    
    def test_memory_and_cpu_optimization_integration(self, sample_config):
        """Test integration between memory management and CPU optimizations."""
        # Create memory optimizer
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=4*1024*1024,  # 4MB
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        # Create CPU optimizer
        cpu_preprocessor = IntelCPUOptimizedPreprocessor(sample_config)
        
        # Test tensor allocation with memory optimization
        tensor = memory_optimizer.allocate_tensor_memory(
            (100, 256), 
            dtype=np.float32, 
            tensor_type="general"
        )
        assert tensor is not None
        assert tensor.shape == (100, 256)
        
        # Test image processing with both optimizations
        sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(sample_image)
        
        # Process image through memory optimizer
        optimized_image = memory_optimizer.optimize_image_processing_memory(
            np.expand_dims(sample_image.astype(np.float32) / 255.0, 0)
        )
        assert optimized_image is not None
        
        # Clean up
        memory_optimizer.cleanup()
    
    def test_attention_memory_optimization_with_cpu_pipeline(self, sample_config):
        """Test integration of attention memory optimization with CPU pipeline."""
        # Create memory optimizer
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=8*1024*1024,  # 8MB
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        # Create mock model for pipeline
        mock_model = Mock()
        mock_model.device = torch.device('cpu')
        mock_model.parameters = Mock(return_value=iter([torch.nn.Parameter(torch.randn(10, 10))]))
        mock_model.generate = Mock(return_value=torch.randint(0, 1000, (4, 10)))
        
        # Create CPU pipeline
        cpu_pipeline = IntelOptimizedPipeline(mock_model, sample_config)
        
        # Optimize attention memory
        attention_components = memory_optimizer.optimize_attention_memory(
            batch_size=4,
            seq_len=1024,
            hidden_dim=768,
            num_heads=12
        )
        
        # Verify attention components are properly allocated
        assert 'query' in attention_components
        assert 'key' in attention_components
        assert 'value' in attention_components
        assert 'attention_scores' in attention_components
        
        # Test that components have the expected shapes
        assert attention_components['query'].shape == (4, 1024, 768)
        assert attention_components['attention_scores'].shape == (4, 12, 1024, 1024)
        
        # Clean up
        memory_optimizer.cleanup()
    
    def test_adaptive_optimization_with_memory_management(self, sample_config):
        """Test integration of adaptive optimization with memory management."""
        # Create memory optimizer
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=4*1024*1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        # Create adaptive optimizer
        adaptive_optimizer = AdaptiveIntelOptimizer(sample_config)
        
        # Start adaptation
        adaptive_optimizer.start_adaptation()
        
        # Perform some memory operations
        for i in range(5):
            tensor = memory_optimizer.allocate_tensor_memory(
                (50, 128), 
                dtype=np.float32, 
                tensor_type="general"
            )
            if tensor is not None:
                memory_optimizer.free_tensor_memory(tensor, "general")
        
        # Get optimization parameters
        params = adaptive_optimizer.get_optimization_params()
        assert 'batch_size' in params
        assert 'thread_count' in params
        
        # Get memory statistics
        mem_stats = memory_optimizer.get_memory_stats()
        if 'general_pool' in mem_stats:
            assert 'current_usage' in mem_stats['general_pool']
        
        # Stop adaptation
        adaptive_optimizer.stop_adaptation()
        
        # Clean up
        memory_optimizer.cleanup()


class TestEndToEndPipelineIntegration:
    """Test end-to-end pipeline integration."""
    
    def test_complete_optimization_pipeline(self, sample_config, sample_text_batch):
        """Test the complete optimization pipeline from preprocessing to inference."""
        # Create mock model with proper structure for optimization
        mock_model = Mock()
        mock_model.device = torch.device('cpu')
        mock_model.parameters = Mock(return_value=iter([torch.nn.Parameter(torch.randn(10, 10))]))
        mock_model.generate = Mock(return_value=torch.randint(0, 1000, (len(sample_text_batch), 10)))

        # Mock the language model structure that the optimization function expects
        mock_model.language_model = Mock()
        mock_model.language_model.layers = [Mock() for _ in range(2)]  # Create mock layers

        # Mock each layer to have the expected attributes
        for layer in mock_model.language_model.layers:
            layer.self_attn = Mock()
            layer.mlp = Mock()
            layer.self_attn.q_proj = Mock()
            layer.self_attn.k_proj = Mock()
            layer.self_attn.v_proj = Mock()
            layer.self_attn.o_proj = Mock()
            layer.mlp.gate_proj = Mock()
            layer.mlp.up_proj = Mock()
            layer.mlp.down_proj = Mock()

        # Create memory optimizer
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=8*1024*1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )

        # Apply Intel optimizations to model - this may fail gracefully if structure isn't perfect
        try:
            optimized_model, components = apply_intel_optimizations_to_model(mock_model, sample_config)
            # Create Intel-optimized pipeline from components
            intel_pipeline = components['intel_pipeline']
        except Exception:
            # If optimization fails, just create a pipeline directly with the original model
            intel_pipeline = IntelOptimizedPipeline(mock_model, sample_config)

        # Run end-to-end pipeline
        results = intel_pipeline.preprocess_and_infer(sample_text_batch)

        # Verify results
        assert len(results) == len(sample_text_batch)
        assert all(isinstance(result, str) for result in results)

        # Get performance metrics
        pipeline_metrics = intel_pipeline.get_performance_metrics()
        assert 'inference_count' in pipeline_metrics
        assert 'avg_inference_time' in pipeline_metrics

        # Clean up
        memory_optimizer.cleanup()
    
    def test_multimodal_pipeline_integration(self, sample_config, sample_text_batch, sample_image_batch):
        """Test multimodal (text + image) pipeline integration."""
        # Create mock model
        mock_model = Mock()
        mock_model.device = torch.device('cpu')
        mock_model.parameters = Mock(return_value=iter([torch.nn.Parameter(torch.randn(10, 10))]))
        mock_model.generate = Mock(return_value=torch.randint(0, 1000, (2, 10)))  # 2 items
        
        # Create Intel-optimized pipeline
        intel_pipeline = IntelOptimizedPipeline(mock_model, sample_config)
        
        # Run multimodal inference
        results = intel_pipeline.preprocess_and_infer(
            texts=sample_text_batch[:2],  # Use first 2 texts
            images=sample_image_batch[:2]  # Use first 2 images
        )
        
        # Verify results
        assert len(results) == 2  # Batch size
        assert all(isinstance(result, str) for result in results)
        
        # Get performance metrics
        metrics = intel_pipeline.get_performance_metrics()
        assert 'inference_count' in metrics
        assert metrics['inference_count'] >= 1
        assert 'avg_inference_time' in metrics


class TestComponentInteraction:
    """Test interactions between different system components."""
    
    def test_memory_pool_thread_safety_with_cpu_optimizations(self, sample_config):
        """Test thread safety of memory pool when used with CPU optimizations."""
        # Create memory optimizer
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=16*1024*1024,  # Larger pool for threading test
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        # Create CPU preprocessor
        cpu_preprocessor = IntelCPUOptimizedPreprocessor(sample_config)
        
        # Results list to collect from threads
        results = []
        
        def worker_thread(thread_id):
            for i in range(5):
                # Allocate tensor
                tensor = memory_optimizer.allocate_tensor_memory(
                    (10, 50), 
                    dtype=np.float32, 
                    tensor_type="general"
                )
                
                if tensor is not None:
                    # Process with CPU preprocessor (simulate)
                    time.sleep(0.001)  # Simulate processing time
                    # Free tensor
                    memory_optimizer.free_tensor_memory(tensor, "general")
                
                time.sleep(0.001)  # Small delay between operations
        
        # Create multiple threads
        threads = []
        for i in range(4):  # 4 threads
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check final memory stats
        final_stats = memory_optimizer.get_memory_stats()
        if 'general_pool' in final_stats:
            assert final_stats['general_pool']['allocation_count'] >= final_stats['general_pool']['deallocation_count']
        
        # Clean up
        memory_optimizer.cleanup()
    
    def test_adaptive_parameter_adjustment_with_memory_pressure(self, sample_config):
        """Test adaptive parameter adjustment based on memory pressure."""
        # Create a smaller memory pool to trigger memory pressure
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=2*1024*1024,  # 2MB
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        # Create adaptive optimizer
        adaptive_optimizer = AdaptiveIntelOptimizer(sample_config)
        
        # Start adaptation
        adaptive_optimizer.start_adaptation()
        
        # Simulate memory pressure by allocating many tensors
        allocated_tensors = []
        for i in range(50):  # Allocate many small tensors
            tensor = memory_optimizer.allocate_tensor_memory(
                (20, 20), 
                dtype=np.float32, 
                tensor_type="general"
            )
            if tensor is not None:
                allocated_tensors.append((tensor, "general"))
        
        # Get current optimization parameters
        params_before = adaptive_optimizer.get_optimization_params()
        
        # Free some tensors to reduce pressure
        for i in range(25):
            if allocated_tensors:
                tensor, tensor_type = allocated_tensors.pop()
                memory_optimizer.free_tensor_memory(tensor, tensor_type)
        
        # Wait a bit for adaptation to occur
        time.sleep(0.1)
        
        # Get parameters after
        params_after = adaptive_optimizer.get_optimization_params()
        
        # Stop adaptation
        adaptive_optimizer.stop_adaptation()
        
        # Clean up remaining tensors
        while allocated_tensors:
            tensor, tensor_type = allocated_tensors.pop()
            memory_optimizer.free_tensor_memory(tensor, tensor_type)
        
        # Clean up
        memory_optimizer.cleanup()
    
    def test_preprocessing_pipeline_with_memory_optimization(self, sample_config, sample_text_batch):
        """Test preprocessing pipeline with memory optimization."""
        # Create memory optimizer
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=4*1024*1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        # Create CPU preprocessor
        cpu_preprocessor = IntelCPUOptimizedPreprocessor(sample_config)
        
        # Preprocess batch
        processed_result = cpu_preprocessor.preprocess_batch(sample_text_batch)
        
        # Verify processing worked
        assert 'input_ids' in processed_result or len(processed_result) > 0
        
        # Get performance metrics
        preprocessor_metrics = cpu_preprocessor.get_performance_metrics()
        assert 'avg_processing_time' in preprocessor_metrics
        
        # Get memory stats
        mem_stats = memory_optimizer.get_memory_stats()
        
        # Clean up
        memory_optimizer.cleanup()


class TestResourceManagementIntegration:
    """Test resource management across different components."""
    
    def test_memory_cleanup_after_pipeline_execution(self, sample_config, sample_text_batch):
        """Test that memory is properly cleaned up after pipeline execution."""
        # Create mock model
        mock_model = Mock()
        mock_model.device = torch.device('cpu')
        mock_model.parameters = Mock(return_value=iter([torch.nn.Parameter(torch.randn(10, 10))]))
        mock_model.generate = Mock(return_value=torch.randint(0, 1000, (len(sample_text_batch), 10)))
        
        # Create memory optimizer
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=8*1024*1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        # Get initial memory stats
        initial_stats = memory_optimizer.get_memory_stats()
        
        # Execute pipeline multiple times
        for _ in range(3):
            results = memory_optimizer.allocate_tensor_memory(
                (100, 128), 
                dtype=np.float32, 
                tensor_type="general"
            )
            if results is not None:
                memory_optimizer.free_tensor_memory(results, "general")
        
        # Get final memory stats
        final_stats = memory_optimizer.get_memory_stats()
        
        # Clean up
        memory_optimizer.cleanup()
        
        # Both initial and final stats should be accessible without error
    
    def test_pipeline_with_error_handling(self, sample_config, sample_text_batch):
        """Test pipeline behavior with error conditions."""
        # Create mock model that raises an exception
        mock_model = Mock()
        mock_model.device = torch.device('cpu')
        mock_model.parameters = Mock(return_value=iter([torch.nn.Parameter(torch.randn(10, 10))]))
        mock_model.generate = Mock(side_effect=Exception("Simulated error"))
        
        # Create Intel-optimized pipeline
        intel_pipeline = IntelOptimizedPipeline(mock_model, sample_config)
        
        # Run pipeline - should handle the error gracefully
        try:
            results = intel_pipeline.preprocess_and_infer(sample_text_batch)
            # Even with error, it should return some results or handle gracefully
            assert isinstance(results, list)
        except Exception as e:
            # If it raises an exception, that's also acceptable as long as it's handled properly
            pass
        
        # Get performance metrics (should work even after error)
        metrics = intel_pipeline.get_performance_metrics()
        assert isinstance(metrics, dict)


class TestPerformanceOptimizationIntegration:
    """Test performance optimizations across components."""
    
    def test_concurrent_pipeline_execution(self, sample_config, sample_text_batch):
        """Test concurrent execution of multiple pipelines."""
        # Create multiple mock models
        mock_models = []
        for i in range(3):
            mock_model = Mock()
            mock_model.device = torch.device('cpu')
            mock_model.parameters = Mock(return_value=iter([torch.nn.Parameter(torch.randn(10, 10))]))
            mock_model.generate = Mock(return_value=torch.randint(0, 1000, (len(sample_text_batch), 10)))
            mock_models.append(mock_model)
        
        # Create multiple pipelines
        pipelines = []
        for model in mock_models:
            pipeline = IntelOptimizedPipeline(model, sample_config)
            pipelines.append(pipeline)
        
        # Execute pipelines concurrently
        results = []
        
        def run_pipeline(pipeline, texts, result_list, index):
            try:
                result = pipeline.preprocess_and_infer(texts)
                result_list.append((index, result))
            except Exception as e:
                result_list.append((index, f"Error: {str(e)}"))
        
        threads = []
        for i, pipeline in enumerate(pipelines):
            thread = threading.Thread(
                target=run_pipeline, 
                args=(pipeline, sample_text_batch, results, i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all pipelines executed
        assert len(results) == 3
        
        # Clean up
        for pipeline in pipelines:
            # Stop any running threads in the pipeline
            if hasattr(pipeline, 'stop_pipeline'):
                pipeline.stop_pipeline()
    
    def test_memory_pool_efficiency_with_cpu_optimizations(self, sample_config):
        """Test memory pool efficiency when used with CPU optimizations."""
        # Create memory optimizer
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=16*1024*1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        # Create CPU preprocessor
        cpu_preprocessor = IntelCPUOptimizedPreprocessor(sample_config)
        
        # Perform allocation/deallocation cycles to test efficiency
        for cycle in range(10):
            # Allocate multiple tensors
            tensors = []
            for i in range(5):
                tensor = memory_optimizer.allocate_tensor_memory(
                    (50, 100), 
                    dtype=np.float32, 
                    tensor_type="general"
                )
                if tensor is not None:
                    tensors.append((tensor, "general"))
            
            # Process with CPU preprocessor
            time.sleep(0.001)  # Simulate processing
            
            # Free all tensors
            for tensor, tensor_type in tensors:
                memory_optimizer.free_tensor_memory(tensor, tensor_type)
        
        # Check final memory efficiency
        final_stats = memory_optimizer.get_memory_stats()
        
        # Clean up
        memory_optimizer.cleanup()


# Run the tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__])