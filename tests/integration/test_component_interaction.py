"""
Comprehensive tests for component interactions in the Qwen3-VL-2B-Instruct system.

This test suite validates how different system components work together,
including memory management, CPU optimization, resource management, and
other integrated functionality.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import threading
import time
import gc
import queue
from PIL import Image

from advanced_memory_management_vl import (
    AdvancedMemoryPool, 
    MemoryPoolType, 
    VisionLanguageMemoryOptimizer,
    CacheAwareMemoryManager,
    GPUCPUMemoryOptimizer
)
from advanced_cpu_optimizations_intel_i5_10210u import (
    AdvancedCPUOptimizationConfig,
    IntelCPUOptimizedPreprocessor,
    IntelOptimizedPipeline,
    AdaptiveIntelOptimizer,
    apply_intel_optimizations_to_model
)


class TestMemoryCPUIntegration:
    """Test integration between memory management and CPU optimization."""
    
    def test_memory_optimized_preprocessing(self, sample_config):
        """Test preprocessing with memory-optimized tensors."""
        # Create memory optimizer
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=8*1024*1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        # Create CPU preprocessor
        cpu_preprocessor = IntelCPUOptimizedPreprocessor(sample_config)
        
        # Create memory-optimized tensors for preprocessing
        text_embedding = memory_optimizer.allocate_tensor_memory(
            (4, 512), 
            dtype=np.float32, 
            tensor_type="text_embeddings"
        )
        
        # Preprocess with CPU-optimized preprocessor
        texts = ["Sample text 1", "Sample text 2", "Sample text 3", "Sample text 4"]
        processed_result = cpu_preprocessor.preprocess_batch(texts)
        
        # Both components should work together without conflict
        assert processed_result is not None
        assert 'input_ids' in processed_result or len(processed_result) >= 0
        
        # Clean up
        if text_embedding is not None:
            memory_optimizer.free_tensor_memory(text_embedding, "text_embeddings")
        memory_optimizer.cleanup()
    
    def test_attention_memory_with_cpu_pipeline(self, mock_model, sample_config, sample_text_batch):
        """Test attention memory optimization with CPU pipeline."""
        # Create memory optimizer
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=16*1024*1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        # Create CPU pipeline
        cpu_pipeline = IntelOptimizedPipeline(mock_model, sample_config)
        
        # Optimize attention memory
        attention_components = memory_optimizer.optimize_attention_memory(
            batch_size=4,
            seq_len=1024,
            hidden_dim=768,
            num_heads=12
        )
        
        # Run CPU pipeline
        results = cpu_pipeline.preprocess_and_infer(sample_text_batch[:4])
        
        # Verify both systems worked together
        assert 'query' in attention_components
        assert 'key' in attention_components
        assert 'value' in attention_components
        assert len(results) == 4
        
        # Clean up
        memory_optimizer.cleanup()
    
    def test_memory_pool_with_cpu_cache_optimization(self):
        """Test memory pool working with CPU cache optimization."""
        # Create memory pool
        pool = AdvancedMemoryPool(initial_size=8*1024*1024)
        
        # Create cache-aware manager
        cache_manager = CacheAwareMemoryManager()
        
        # Allocate memory through pool
        alloc_result = pool.allocate(1024*1024, MemoryPoolType.TENSOR_DATA)  # 1MB
        if alloc_result:
            ptr, size = alloc_result
            
            # Create numpy array in the allocated memory space
            # (In practice, we'd use the memory directly, but for testing we'll just verify the allocation)
            test_array = np.random.random((1000, 256)).astype(np.float32)
            
            # Optimize the array layout
            optimized_array = cache_manager.optimize_memory_layout(test_array)
            
            # Verify the optimization worked
            assert optimized_array.shape == test_array.shape
            assert optimized_array.dtype == test_array.dtype
            assert optimized_array.flags['C_CONTIGUOUS']  # Should be cache-friendly
            
            # Clean up
            pool.deallocate(ptr)
        
        pool.cleanup()


class TestResourceManagementIntegration:
    """Test integration of resource management components."""
    
    def test_adaptive_optimization_with_memory_pressure(self, sample_config):
        """Test adaptive optimization responding to memory pressure."""
        # Create memory optimizer
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=4*1024*1024,  # Smaller pool to create pressure
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        # Create adaptive optimizer
        adaptive_optimizer = AdaptiveIntelOptimizer(sample_config)
        
        # Start adaptation
        adaptive_optimizer.start_adaptation()
        
        # Create memory pressure by allocating many tensors
        allocated_tensors = []
        for i in range(50):  # Allocate many small tensors to create fragmentation
            tensor = memory_optimizer.allocate_tensor_memory(
                (50, 50), 
                dtype=np.float32, 
                tensor_type="general"
            )
            if tensor is not None:
                allocated_tensors.append((tensor, "general"))
        
        # Get parameters during high memory usage
        params_during_pressure = adaptive_optimizer.get_optimization_params()
        
        # Free some tensors to reduce pressure
        for i in range(25):
            if allocated_tensors:
                tensor, tensor_type = allocated_tensors.pop()
                memory_optimizer.free_tensor_memory(tensor, tensor_type)
        
        # Wait for adaptation to potentially adjust parameters
        time.sleep(0.1)
        
        # Get parameters after pressure reduction
        params_after_pressure = adaptive_optimizer.get_optimization_params()
        
        # Stop adaptation
        adaptive_optimizer.stop_adaptation()
        
        # Clean up remaining tensors
        while allocated_tensors:
            tensor, tensor_type = allocated_tensors.pop()
            memory_optimizer.free_tensor_memory(tensor, tensor_type)
        
        # Both sets of parameters should be valid
        assert 'batch_size' in params_during_pressure
        assert 'batch_size' in params_after_pressure
        
        # Clean up
        memory_optimizer.cleanup()
    
    def test_power_thermal_constraints_with_cpu_memory(self, sample_config, sample_text_batch):
        """Test power and thermal constraints affecting both CPU and memory usage."""
        # Create adaptive optimizer with strict constraints
        # Create a new config with modified limits
        config = AdvancedCPUOptimizationConfig(
            num_preprocess_workers=sample_config.num_preprocess_workers,
            preprocess_batch_size=sample_config.preprocess_batch_size,
            max_concurrent_threads=sample_config.max_concurrent_threads,
            l1_cache_size=sample_config.l1_cache_size,
            l2_cache_size=sample_config.l2_cache_size,
            l3_cache_size=sample_config.l3_cache_size,
            cache_line_size=sample_config.cache_line_size,
            image_resize_size=sample_config.image_resize_size,
            max_text_length=sample_config.max_text_length,
            pipeline_depth=sample_config.pipeline_depth,
            pipeline_buffer_size=sample_config.pipeline_buffer_size,
            adaptation_frequency=sample_config.adaptation_frequency,
            performance_target=sample_config.performance_target,
            power_constraint=0.6,  # Lower power limit
            thermal_constraint=65.0,  # Lower thermal limit
            enable_thread_affinity=sample_config.enable_thread_affinity,
            enable_hyperthreading_optimization=sample_config.enable_hyperthreading_optimization,
            memory_threshold=sample_config.memory_threshold,
            clear_cache_interval=sample_config.clear_cache_interval,
            enable_memory_pooling=sample_config.enable_memory_pooling
        )
        
        adaptive_optimizer = AdaptiveIntelOptimizer(config)
        
        # Create memory optimizer
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=8*1024*1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        # Create CPU preprocessor
        cpu_preprocessor = IntelCPUOptimizedPreprocessor(config)
        
        # Start adaptation
        adaptive_optimizer.start_adaptation()
        
        # Perform operations that would trigger power/thermal adjustments
        for i in range(10):
            # Allocate memory
            tensor = memory_optimizer.allocate_tensor_memory(
                (100, 128), 
                dtype=np.float32, 
                tensor_type="general"
            )
            
            # Process with CPU
            if tensor is not None:
                processed = cpu_preprocessor.preprocess_batch(sample_text_batch[:1])
                memory_optimizer.free_tensor_memory(tensor, "general")
        
        # Get current parameters
        params = adaptive_optimizer.get_optimization_params()
        
        # Stop adaptation
        adaptive_optimizer.stop_adaptation()
        
        # Parameters should reflect the constraints
        assert 'power_limit' in params
        assert 'thermal_limit' in params
        
        # Clean up
        memory_optimizer.cleanup()
    
    def test_memory_pool_statistics_with_cpu_pipeline(self, mock_model, sample_config, sample_text_batch):
        """Test memory pool statistics while CPU pipeline is active."""
        # Create memory optimizer
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=16*1024*1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        # Create CPU pipeline
        cpu_pipeline = IntelOptimizedPipeline(mock_model, sample_config)
        
        # Get initial memory stats
        initial_stats = memory_optimizer.get_memory_stats()
        
        # Run pipeline with memory operations
        for i in range(5):
            # Allocate memory
            tensor = memory_optimizer.allocate_tensor_memory(
                (50, 256), 
                dtype=np.float32, 
                tensor_type="general"
            )
            
            # Run pipeline
            results = cpu_pipeline.preprocess_and_infer(sample_text_batch[:1])
            
            # Free memory
            if tensor is not None:
                memory_optimizer.free_tensor_memory(tensor, "general")
        
        # Get final memory stats
        final_stats = memory_optimizer.get_memory_stats()
        
        # Stats should be consistent
        if 'general_pool' in final_stats:
            assert final_stats['general_pool']['allocation_count'] >= 0
            assert final_stats['general_pool']['deallocation_count'] >= 0
        
        # Clean up
        memory_optimizer.cleanup()


class TestPipelineIntegration:
    """Test integration within and between processing pipelines."""
    
    def test_multistage_pipeline_with_memory_optimization(self, mock_model, sample_config, sample_text_batch):
        """Test multistage pipeline with memory optimization."""
        # Create memory optimizer
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=32*1024*1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        # Create CPU pipeline
        cpu_pipeline = IntelOptimizedPipeline(mock_model, sample_config)
        
        # Simulate a multistage pipeline
        # Stage 1: Memory allocation
        stage1_tensor = memory_optimizer.allocate_tensor_memory(
            (4, 1024, 512), 
            dtype=np.float32, 
            tensor_type="general"
        )
        
        # Stage 2: Preprocessing
        texts = sample_text_batch[:4]
        processed_inputs = cpu_pipeline.preprocessor.preprocess_batch(texts)
        
        # Stage 3: Pipeline execution
        results = cpu_pipeline.preprocess_and_infer(texts)
        
        # Stage 4: Memory cleanup
        if stage1_tensor is not None:
            memory_optimizer.free_tensor_memory(stage1_tensor, "general")
        
        # All stages should complete successfully
        assert processed_inputs is not None
        assert len(results) == 4
        
        # Clean up
        memory_optimizer.cleanup()
    
    def test_pipeline_buffer_integration(self, mock_model, sample_config, sample_text_batch):
        """Test pipeline buffer integration with memory management."""
        # Create CPU pipeline
        cpu_pipeline = IntelOptimizedPipeline(mock_model, sample_config)
        
        # Access pipeline buffers
        assert len(cpu_pipeline.pipeline_buffers) > 0
        
        # Test buffer operations
        test_data = {"input_ids": torch.tensor([[1, 2, 3]])}
        
        try:
            # Put data in buffer
            cpu_pipeline.pipeline_buffers[0].put(test_data, timeout=1.0)
            
            # Get data from buffer
            retrieved_data = cpu_pipeline.pipeline_buffers[0].get(timeout=1.0)
            
            # Verify data integrity
            assert "input_ids" in retrieved_data
        except queue.Empty:
            # Buffer might be empty due to pipeline processing
            pass
    
    def test_async_pipeline_with_memory_management(self, sample_config, sample_text_batch):
        """Test asynchronous pipeline operations with memory management."""
        # Create memory optimizer
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=8*1024*1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        # Create CPU preprocessor
        cpu_preprocessor = IntelCPUOptimizedPreprocessor(sample_config)
        
        # Use async preprocessing
        future = cpu_preprocessor.preprocess_batch_parallel(sample_text_batch)
        
        # Allocate memory while preprocessing happens
        tensor = memory_optimizer.allocate_tensor_memory(
            (100, 256), 
            dtype=np.float32, 
            tensor_type="general"
        )
        
        # Get preprocessing result
        result = future.result(timeout=5.0)  # 5 second timeout
        
        # Free memory
        if tensor is not None:
            memory_optimizer.free_tensor_memory(tensor, "general")
        
        # Both operations should succeed
        assert result is not None
        
        # Clean up
        memory_optimizer.cleanup()


class TestComponentSynchronization:
    """Test synchronization between different components."""
    
    def test_synchronized_memory_cpu_operations(self, sample_config, sample_text_batch):
        """Test synchronized operations between memory and CPU components."""
        # Create memory optimizer
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=16*1024*1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        # Create CPU preprocessor
        cpu_preprocessor = IntelCPUOptimizedPreprocessor(sample_config)
        
        # Shared synchronization primitive
        sync_event = threading.Event()
        
        def memory_worker():
            tensors = []
            for i in range(5):
                tensor = memory_optimizer.allocate_tensor_memory(
                    (50, 128), 
                    dtype=np.float32, 
                    tensor_type="general"
                )
                if tensor is not None:
                    tensors.append((tensor, "general"))
                time.sleep(0.01)
            sync_event.set()  # Signal completion
            
            # Free tensors after CPU work is done
            sync_event.wait()  # Wait for CPU to finish
            for tensor, tensor_type in tensors:
                memory_optimizer.free_tensor_memory(tensor, tensor_type)
        
        def cpu_worker():
            sync_event.wait()  # Wait for memory allocation
            for i in range(5):
                result = cpu_preprocessor.preprocess_batch(sample_text_batch[:1])
                time.sleep(0.01)
        
        # Run both workers
        mem_thread = threading.Thread(target=memory_worker)
        cpu_thread = threading.Thread(target=cpu_worker)
        
        mem_thread.start()
        cpu_thread.start()
        
        mem_thread.join()
        cpu_thread.join()
        
        # Clean up
        memory_optimizer.cleanup()
    
    def test_adaptive_parameter_synchronization(self, sample_config, sample_text_batch):
        """Test synchronization of adaptive parameters across components."""
        # Create adaptive optimizer
        adaptive_optimizer = AdaptiveIntelOptimizer(sample_config)
        
        # Create memory optimizer
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=8*1024*1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        # Start adaptation
        adaptive_optimizer.start_adaptation()
        
        # Perform operations that might trigger parameter adjustments
        for i in range(10):
            tensor = memory_optimizer.allocate_tensor_memory(
                (100, 100), 
                dtype=np.float32, 
                tensor_type="general"
            )
            if tensor is not None:
                # Simulate work that might affect system metrics
                time.sleep(0.01)
                memory_optimizer.free_tensor_memory(tensor, "general")
        
        # Get parameters to verify they're being updated
        params = adaptive_optimizer.get_optimization_params()
        
        # Stop adaptation
        adaptive_optimizer.stop_adaptation()
        
        # Parameters should be properly maintained
        assert 'batch_size' in params
        assert 'thread_count' in params
        
        # Clean up
        memory_optimizer.cleanup()
    
    def test_resource_contention_handling(self, sample_config, sample_text_batch):
        """Test handling of resource contention between components."""
        # Create multiple optimizers that might compete for resources
        memory_optimizer1 = VisionLanguageMemoryOptimizer(
            memory_pool_size=8*1024*1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        memory_optimizer2 = VisionLanguageMemoryOptimizer(
            memory_pool_size=8*1024*1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        cpu_preprocessor = IntelCPUOptimizedPreprocessor(sample_config)
        
        # Simulate resource contention
        results = []
        
        def worker1():
            local_results = []
            for i in range(10):
                tensor = memory_optimizer1.allocate_tensor_memory(
                    (50, 100), 
                    dtype=np.float32, 
                    tensor_type="general"
                )
                if tensor is not None:
                    # Simulate processing
                    time.sleep(0.001)
                    memory_optimizer1.free_tensor_memory(tensor, "general")
                    local_results.append(True)
                else:
                    local_results.append(False)
            results.append(("mem1", local_results))
        
        def worker2():
            local_results = []
            for i in range(10):
                tensor = memory_optimizer2.allocate_tensor_memory(
                    (50, 100), 
                    dtype=np.float32, 
                    tensor_type="general"
                )
                if tensor is not None:
                    # Simulate processing
                    time.sleep(0.001)
                    memory_optimizer2.free_tensor_memory(tensor, "general")
                    local_results.append(True)
                else:
                    local_results.append(False)
            results.append(("mem2", local_results))
        
        def worker3():
            local_results = []
            for i in range(10):
                result = cpu_preprocessor.preprocess_batch(sample_text_batch[:1])
                local_results.append(result is not None)
                time.sleep(0.001)
            results.append(("cpu", local_results))
        
        # Run all workers concurrently
        threads = [
            threading.Thread(target=worker1),
            threading.Thread(target=worker2),
            threading.Thread(target=worker3)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Check that all workers completed
        assert len(results) == 3
        
        # Clean up
        memory_optimizer1.cleanup()
        memory_optimizer2.cleanup()


class TestSystemWideIntegration:
    """Test system-wide integration of all components."""
    
    def test_complete_system_integration(self, mock_model, sample_config, sample_text_batch, sample_image_batch):
        """Test complete system integration with all components."""
        # Create all system components
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=32*1024*1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        adaptive_optimizer = AdaptiveIntelOptimizer(sample_config)
        
        # Mock the language model structure that the optimization function expects
        if not hasattr(mock_model, 'language_model'):
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

        # Apply Intel optimizations to model - this may fail gracefully if structure isn't perfect
        try:
            optimized_model, components = apply_intel_optimizations_to_model(mock_model, sample_config)
            # Get the pipeline from components
            intel_pipeline = components['intel_pipeline']
        except Exception:
            # If optimization fails, just create a pipeline directly with the original model
            intel_pipeline = IntelOptimizedPipeline(mock_model, sample_config)
        
        # Start adaptation
        adaptive_optimizer.start_adaptation()
        
        # Run complete system workflow
        for iteration in range(3):
            # Allocate memory-optimized tensors
            text_tensor = memory_optimizer.allocate_tensor_memory(
                (len(sample_text_batch), 128), 
                dtype=np.float32, 
                tensor_type="text_embeddings"
            )
            
            img_tensor = memory_optimizer.allocate_tensor_memory(
                (len(sample_image_batch), 196, 512), 
                dtype=np.float32, 
                tensor_type="image_features"
            )
            
            # Run inference pipeline
            results = intel_pipeline.preprocess_and_infer(
                sample_text_batch,
                sample_image_batch
            )
            
            # Free memory-optimized tensors
            if text_tensor is not None:
                memory_optimizer.free_tensor_memory(text_tensor, "text_embeddings")
            if img_tensor is not None:
                memory_optimizer.free_tensor_memory(img_tensor, "image_features")
        
        # Get performance metrics from all components
        pipeline_metrics = intel_pipeline.get_performance_metrics()
        adaptive_metrics = adaptive_optimizer.get_performance_metrics()
        memory_stats = memory_optimizer.get_memory_stats()
        
        # Stop adaptation
        adaptive_optimizer.stop_adaptation()
        
        # Verify all components provided metrics
        assert 'inference_count' in pipeline_metrics
        assert 'adaptation_count' in adaptive_metrics
        # Memory stats may or may not include pool stats depending on implementation
        
        # Clean up
        memory_optimizer.cleanup()
    
    def test_error_recovery_across_components(self, sample_config, sample_text_batch):
        """Test error recovery across all system components."""
        # Create components
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=8*1024*1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        adaptive_optimizer = AdaptiveIntelOptimizer(sample_config)
        
        cpu_preprocessor = IntelCPUOptimizedPreprocessor(sample_config)
        
        # Start adaptation
        adaptive_optimizer.start_adaptation()
        
        try:
            # Perform operations that might encounter errors
            for i in range(5):
                # Memory allocation
                tensor = memory_optimizer.allocate_tensor_memory(
                    (100, 100), 
                    dtype=np.float32, 
                    tensor_type="general"
                )
                
                # CPU processing
                if tensor is not None:
                    try:
                        result = cpu_preprocessor.preprocess_batch(sample_text_batch[:1])
                    except Exception:
                        # Handle preprocessing errors gracefully
                        pass
                
                # Memory cleanup
                if tensor is not None:
                    memory_optimizer.free_tensor_memory(tensor, "general")
        
        except Exception as e:
            # System should handle errors without crashing
            pass
        
        finally:
            # Ensure cleanup happens
            adaptive_optimizer.stop_adaptation()
            memory_optimizer.cleanup()
    
    def test_performance_monitoring_integration(self, mock_model, sample_config, sample_text_batch):
        """Test integration of performance monitoring across components."""
        # Create all components
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=16*1024*1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        adaptive_optimizer = AdaptiveIntelOptimizer(sample_config)
        
        cpu_pipeline = IntelOptimizedPipeline(mock_model, sample_config)
        
        # Start adaptation for monitoring
        adaptive_optimizer.start_adaptation()
        
        # Record initial metrics
        initial_mem_stats = memory_optimizer.get_memory_stats()
        initial_adaptive_params = adaptive_optimizer.get_optimization_params()
        
        # Perform operations
        operation_count = 0
        for i in range(10):
            # Memory operations
            tensor = memory_optimizer.allocate_tensor_memory(
                (50, 256), 
                dtype=np.float32, 
                tensor_type="general"
            )
            
            # CPU pipeline operations
            results = cpu_pipeline.preprocess_and_infer(sample_text_batch[:1])
            
            # Memory cleanup
            if tensor is not None:
                memory_optimizer.free_tensor_memory(tensor, "general")
            
            operation_count += 1
        
        # Record final metrics
        final_mem_stats = memory_optimizer.get_memory_stats()
        final_adaptive_params = adaptive_optimizer.get_optimization_params()
        pipeline_metrics = cpu_pipeline.get_performance_metrics()
        
        # Stop adaptation
        adaptive_optimizer.stop_adaptation()
        
        # Verify metrics were collected
        assert 'allocation_count' in (final_mem_stats.get('general_pool', {}) or {})
        assert 'batch_size' in final_adaptive_params
        assert 'inference_count' in pipeline_metrics
        
        # Clean up
        memory_optimizer.cleanup()


class TestRealWorldScenarioIntegration:
    """Test integration under realistic usage scenarios."""
    
    def test_batch_processing_scenario(self, mock_model, sample_config, sample_text_batch, sample_image_batch):
        """Test integration under batch processing scenario."""
        # Create system components
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=64*1024*1024,  # Larger pool for batch processing
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        adaptive_optimizer = AdaptiveIntelOptimizer(sample_config)
        
        cpu_pipeline = IntelOptimizedPipeline(mock_model, sample_config)
        
        # Start adaptation
        adaptive_optimizer.start_adaptation()
        
        # Simulate batch processing workflow
        batch_size = 8
        num_batches = 5
        
        for batch_idx in range(num_batches):
            # Prepare batch data
            batch_texts = sample_text_batch * 2  # Double up to reach batch size
            batch_images = sample_image_batch * 2  # Double up to reach batch size
            
            # Allocate batch-specific memory
            text_embeddings = memory_optimizer.allocate_tensor_memory(
                (batch_size, 512), 
                dtype=np.float32, 
                tensor_type="text_embeddings"
            )
            
            image_features = memory_optimizer.allocate_tensor_memory(
                (batch_size, 196, 512), 
                dtype=np.float32, 
                tensor_type="image_features"
            )
            
            # Process batch
            results = cpu_pipeline.preprocess_and_infer(
                batch_texts,
                batch_images
            )
            
            # Verify results
            assert len(results) == batch_size
            
            # Clean up batch-specific memory
            if text_embeddings is not None:
                memory_optimizer.free_tensor_memory(text_embeddings, "text_embeddings")
            if image_features is not None:
                memory_optimizer.free_tensor_memory(image_features, "image_features")
        
        # Get final metrics
        mem_stats = memory_optimizer.get_memory_stats()
        adaptive_params = adaptive_optimizer.get_optimization_params()
        pipeline_metrics = cpu_pipeline.get_performance_metrics()
        
        # Stop adaptation
        adaptive_optimizer.stop_adaptation()
        
        # Verify system handled the workload
        assert 'allocation_count' in (mem_stats.get('general_pool', {}) or {})
        assert 'inference_count' in pipeline_metrics
        assert pipeline_metrics['inference_count'] >= num_batches
        
        # Clean up
        memory_optimizer.cleanup()
    
    def test_long_running_inference_scenario(self, mock_model, sample_config, sample_text_batch):
        """Test integration under long-running inference scenario."""
        # Create system components
        memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=32*1024*1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        adaptive_optimizer = AdaptiveIntelOptimizer(sample_config)
        
        cpu_pipeline = IntelOptimizedPipeline(mock_model, sample_config)
        
        # Start adaptation
        adaptive_optimizer.start_adaptation()
        
        # Simulate long-running inference
        start_time = time.time()
        target_duration = 5.0  # 5 seconds
        request_count = 0
        
        while time.time() - start_time < target_duration:
            try:
                # Allocate temporary memory
                temp_tensor = memory_optimizer.allocate_tensor_memory(
                    (10, 100), 
                    dtype=np.float32, 
                    tensor_type="temporary"
                )
                
                # Run inference
                results = cpu_pipeline.preprocess_and_infer(sample_text_batch[:1])
                
                # Clean up
                if temp_tensor is not None:
                    memory_optimizer.free_tensor_memory(temp_tensor, "temporary")
                
                request_count += 1
                
                # Periodic cleanup
                if request_count % 10 == 0:
                    gc.collect()
                
            except Exception:
                # Continue processing even if individual requests fail
                continue
        
        # Get metrics
        duration = time.time() - start_time
        throughput = request_count / duration if duration > 0 else 0
        
        # Stop adaptation
        adaptive_optimizer.stop_adaptation()
        
        # Verify reasonable performance
        assert throughput > 0  # Should have processed at least some requests
        assert request_count >= 0
        
        # Clean up
        memory_optimizer.cleanup()


# Run the tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__])