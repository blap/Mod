"""
Comprehensive tests for the multi-model support framework.
"""

import unittest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from src.models.multi_model_framework import (
    ModelRegistry,
    ModelFactory,
    MultiModelManager,
    ModelOrchestrator,
    ResourceManager,
    InterModelCommunicator,
    ModelLifecycleManager,
    PerformanceMonitor,
    ModelVersionManager
)
from src.models.model_registry import ModelSpec


class TestMultiModelFramework(unittest.TestCase):
    """Test cases for the multi-model support framework."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock models for testing
        self.mock_model_1 = nn.Linear(10, 5)
        self.mock_model_2 = nn.Linear(20, 10)

        # Create a mock config class that is callable
        class MockConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        # Create mock specs
        self.spec_1 = ModelSpec(
            name="test_model_1",
            model_class=nn.Linear,
            config_class=MockConfig,
            supported_dtypes=["float32"],
            required_memory_gb=1.0,
            max_sequence_length=512,
            description="Test model 1",
            model_type="test"
        )

        self.spec_2 = ModelSpec(
            name="test_model_2",
            model_class=nn.Linear,
            config_class=MockConfig,
            supported_dtypes=["float32"],
            required_memory_gb=2.0,
            max_sequence_length=1024,
            description="Test model 2",
            model_type="test"
        )

    def test_model_registry(self):
        """Test the ModelRegistry functionality."""
        registry = ModelRegistry()
        
        # Register models
        self.assertTrue(registry.register_model(self.spec_1))
        self.assertTrue(registry.register_model(self.spec_2))
        
        # Check registration
        self.assertTrue(registry.is_model_registered("test_model_1"))
        self.assertTrue(registry.is_model_registered("test_model_2"))
        
        # Get model specs
        spec1 = registry.get_model_spec("test_model_1")
        spec2 = registry.get_model_spec("test_model_2")
        self.assertIsNotNone(spec1)
        self.assertIsNotNone(spec2)
        self.assertEqual(spec1.name, "test_model_1")
        self.assertEqual(spec2.name, "test_model_2")
        
        # Get all specs
        all_specs = registry.get_all_model_specs()
        self.assertEqual(len(all_specs), 2)
        
        # Get names
        names = registry.get_model_names()
        self.assertIn("test_model_1", names)
        self.assertIn("test_model_2", names)
        
        # Get by type
        typed_specs = registry.get_model_by_type("test")
        self.assertEqual(len(typed_specs), 2)
        
        # Unregister
        self.assertTrue(registry.unregister_model("test_model_1"))
        self.assertFalse(registry.is_model_registered("test_model_1"))

    def test_model_factory(self):
        """Test the ModelFactory functionality."""
        registry = ModelRegistry()
        factory = ModelFactory(registry)

        # Register a model spec
        registry.register_model(self.spec_1)

        # Create model using config (since nn.Linear doesn't have from_pretrained)
        with patch.object(self.spec_1.config_class, '__call__', return_value={}):
            model = factory.create_model("test_model_1")  # Create from config instead of pretrained

        self.assertIsNotNone(model)

        # List available models
        models = factory.list_available_models()
        self.assertIn("test_model_1", models)

        # Get model info
        info = factory.get_model_info("test_model_1")
        self.assertIsNotNone(info)

    def test_resource_manager(self):
        """Test the ResourceManager functionality."""
        manager = ResourceManager()

        # Register models with resource requirements
        manager.register_model_resources("model_a", required_memory=2.0, required_gpus=1)
        manager.register_model_resources("model_b", required_memory=4.0, required_gpus=2)

        # Allocate resources
        alloc_a = manager.allocate_resources("model_a", requested_memory=2.0, requested_gpus=1)
        alloc_b = manager.allocate_resources("model_b", requested_memory=4.0, requested_gpus=2)

        self.assertIsNotNone(alloc_a)
        self.assertIsNotNone(alloc_b)
        self.assertEqual(alloc_a['allocated_memory'], 2.0)

        # Check memory allocation
        self.assertEqual(alloc_a['allocated_memory'], 2.0)
        self.assertEqual(alloc_b['allocated_memory'], 4.0)

        # For GPU allocation, check if system has GPUs
        # If no GPUs are available, the allocation will be 0 regardless of request
        # The important thing is that the system handles this gracefully
        self.assertGreaterEqual(alloc_a['allocated_gpus'], 0)  # At least 0 GPUs allocated
        self.assertGreaterEqual(alloc_b['allocated_gpus'], 0)  # At least 0 GPUs allocated
        # And it shouldn't exceed what was requested
        self.assertLessEqual(alloc_a['allocated_gpus'], 1)
        self.assertLessEqual(alloc_b['allocated_gpus'], 2)

        # Check total allocation
        total_alloc = manager.get_total_allocation()
        self.assertEqual(total_alloc['total_allocated_memory'], 6.0)
        expected_gpus = alloc_a['allocated_gpus'] + alloc_b['allocated_gpus']
        self.assertEqual(total_alloc['total_allocated_gpus'], expected_gpus)

        # Free resources
        manager.free_resources("model_a")
        alloc_after_free = manager.get_total_allocation()
        self.assertEqual(alloc_after_free['total_allocated_memory'], 4.0)
        self.assertEqual(alloc_after_free['total_allocated_gpus'], alloc_b['allocated_gpus'])

    def test_inter_model_communicator(self):
        """Test the InterModelCommunicator functionality."""
        communicator = InterModelCommunicator()
        
        # Register models
        model_a = Mock()
        model_b = Mock()
        
        communicator.register_model("model_a", model_a)
        communicator.register_model("model_b", model_b)
        
        # Send message between models
        message_data = {"key": "value", "tensor": torch.tensor([1, 2, 3])}
        communicator.send_message("model_a", "model_b", "test_message", message_data)
        
        # Get messages for model_b
        messages = communicator.get_messages("model_b", "test_message")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["data"]["key"], "value")
        
        # Clear messages
        communicator.clear_messages("model_b", "test_message")
        messages_after_clear = communicator.get_messages("model_b", "test_message")
        self.assertEqual(len(messages_after_clear), 0)

    def test_model_lifecycle_manager(self):
        """Test the ModelLifecycleManager functionality."""
        lifecycle_manager = ModelLifecycleManager()
        
        # Create mock model
        mock_model = Mock()
        
        # Register model lifecycle
        lifecycle_manager.register_model_lifecycle("test_model", mock_model)
        
        # Check initial state
        self.assertEqual(lifecycle_manager.get_model_state("test_model"), "created")
        
        # Activate model
        lifecycle_manager.activate_model("test_model")
        self.assertEqual(lifecycle_manager.get_model_state("test_model"), "active")
        
        # Deactivate model
        lifecycle_manager.deactivate_model("test_model")
        self.assertEqual(lifecycle_manager.get_model_state("test_model"), "inactive")
        
        # Destroy model
        lifecycle_manager.destroy_model("test_model")
        self.assertEqual(lifecycle_manager.get_model_state("test_model"), "destroyed")
        
        # Check if model is removed
        self.assertNotIn("test_model", lifecycle_manager._models)

    def test_performance_monitor(self):
        """Test the PerformanceMonitor functionality."""
        monitor = PerformanceMonitor()
        
        # Start monitoring
        monitor.start_monitoring("model_1")
        
        # Record metrics
        monitor.record_metric("model_1", "latency", 0.123)
        monitor.record_metric("model_1", "throughput", 10.0)
        monitor.record_metric("model_1", "memory_usage", 2.5)
        
        # Get metrics
        metrics = monitor.get_metrics("model_1")
        self.assertIn("latency", metrics)
        self.assertIn("throughput", metrics)
        self.assertIn("memory_usage", metrics)
        
        # Get aggregated metrics
        agg_metrics = monitor.get_aggregated_metrics()
        self.assertIn("model_1", agg_metrics)
        
        # Stop monitoring
        monitor.stop_monitoring("model_1")
        self.assertNotIn("model_1", monitor._monitored_models)

    def test_model_version_manager(self):
        """Test the ModelVersionManager functionality."""
        version_manager = ModelVersionManager()
        
        # Register model versions
        version_manager.register_model_version("my_model", "v1.0", "path/to/v1")
        version_manager.register_model_version("my_model", "v2.0", "path/to/v2")
        
        # Get latest version
        latest = version_manager.get_latest_version("my_model")
        self.assertEqual(latest, "v2.0")
        
        # Get version info
        version_info = version_manager.get_version_info("my_model", "v1.0")
        self.assertEqual(version_info["path"], "path/to/v1")
        
        # Check compatibility
        is_compat = version_manager.check_compatibility("v1.0", "v2.0")
        self.assertTrue(is_compat)  # Default is compatible
        
        # Get all versions
        all_versions = version_manager.get_all_versions("my_model")
        self.assertIn("v1.0", all_versions)
        self.assertIn("v2.0", all_versions)

    def test_model_orchestrator(self):
        """Test the ModelOrchestrator functionality."""
        orchestrator = ModelOrchestrator()
        
        # Create mock models
        model_a = Mock()
        model_b = Mock()
        
        # Add models to orchestrator
        orchestrator.add_model("model_a", model_a)
        orchestrator.add_model("model_b", model_b)
        
        # Check models are added
        self.assertEqual(len(orchestrator.get_active_models()), 2)
        self.assertIn("model_a", orchestrator.get_active_models())
        self.assertIn("model_b", orchestrator.get_active_models())
        
        # Remove model
        orchestrator.remove_model("model_a")
        self.assertEqual(len(orchestrator.get_active_models()), 1)
        self.assertNotIn("model_a", orchestrator.get_active_models())

    def test_multi_model_manager(self):
        """Test the MultiModelManager functionality."""
        manager = MultiModelManager()

        # Register model specs
        manager.model_registry.register_model(self.spec_1)
        manager.model_registry.register_model(self.spec_2)

        # Check available models
        available = manager.list_available_models()
        self.assertIn("test_model_1", available)
        self.assertIn("test_model_2", available)

        # Create models
        with patch.object(self.spec_1.config_class, '__call__', return_value={}):
            model_1 = manager.create_model("test_model_1", model_id="model_1")

        with patch.object(self.spec_2.config_class, '__call__', return_value={}):
            model_2 = manager.create_model("test_model_2", model_id="model_2")

        # Check models are created and managed
        active_models = manager.get_active_models()
        self.assertIn("model_1", active_models)
        self.assertIn("model_2", active_models)

        # Get specific model
        retrieved_model = manager.get_model("model_1")
        self.assertIsNotNone(retrieved_model)

        # Get model info
        model_info = manager.get_model_info("model_1")
        self.assertIsNotNone(model_info)

        # Activate models
        manager.activate_model("model_1")
        manager.activate_model("model_2")

        # Check states
        self.assertEqual(manager.get_model_state("model_1"), "active")
        self.assertEqual(manager.get_model_state("model_2"), "active")

        # Deactivate model
        manager.deactivate_model("model_1")
        self.assertEqual(manager.get_model_state("model_1"), "inactive")

        # Get system info
        system_info = manager.get_system_info()
        self.assertIn("active_models", system_info)
        self.assertIn("resource_usage", system_info)

        # Destroy model
        manager.destroy_model("model_1")
        self.assertNotIn("model_1", manager.get_active_models())
        self.assertEqual(manager.get_model_state("model_1"), "destroyed")


class TestIntegration(unittest.TestCase):
    """Test integration between different components."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = MultiModelManager()
        
        # Create test model specs
        self.spec_1 = ModelSpec(
            name="integration_test_model_1",
            model_class=nn.Linear,
            config_class=dict,
            supported_dtypes=["float32"],
            required_memory_gb=1.0,
            max_sequence_length=512,
            description="Integration test model 1",
            model_type="test"
        )
        
        self.spec_2 = ModelSpec(
            name="integration_test_model_2",
            model_class=nn.Linear,
            config_class=dict,
            supported_dtypes=["float32"],
            required_memory_gb=2.0,
            max_sequence_length=1024,
            description="Integration test model 2",
            model_type="test"
        )

    def test_full_workflow(self):
        """Test a full workflow of creating, managing, and destroying models."""
        # Register model specs
        self.manager.model_registry.register_model(self.spec_1)
        self.manager.model_registry.register_model(self.spec_2)

        # Create models with resource allocation
        model_1 = self.manager.create_model(
            "integration_test_model_1",
            model_id="model_1",
            config={"required_memory_gb": 1.0}
        )

        model_2 = self.manager.create_model(
            "integration_test_model_2",
            model_id="model_2",
            config={"required_memory_gb": 2.0}
        )

        # Verify models are created and registered
        self.assertIsNotNone(model_1)
        self.assertIsNotNone(model_2)
        self.assertIn("model_1", self.manager.get_active_models())
        self.assertIn("model_2", self.manager.get_active_models())

        # Check resource allocation
        resource_usage = self.manager.get_resource_usage()
        self.assertGreater(resource_usage["total_allocated_memory"], 0)

        # Activate models
        self.manager.activate_model("model_1")
        self.manager.activate_model("model_2")

        # Verify activation
        self.assertEqual(self.manager.get_model_state("model_1"), "active")
        self.assertEqual(self.manager.get_model_state("model_2"), "active")

        # Simulate some processing and record metrics
        self.manager.performance_monitor.record_metric("model_1", "latency", 0.1)
        self.manager.performance_monitor.record_metric("model_2", "latency", 0.15)

        # Get performance metrics
        metrics = self.manager.get_performance_metrics("model_1")
        self.assertIsNotNone(metrics)

        # Send a message between models
        self.manager.inter_model_communicator.register_model("model_1", model_1)
        self.manager.inter_model_communicator.register_model("model_2", model_2)
        self.manager.inter_model_communicator.send_message(
            "model_1", "model_2", "test_signal", {"data": "hello"}
        )

        # Verify message was sent
        messages = self.manager.inter_model_communicator.get_messages("model_2", "test_signal")
        self.assertEqual(len(messages), 1)

        # Deactivate and destroy models
        self.manager.deactivate_model("model_1")
        self.manager.destroy_model("model_1")
        self.manager.destroy_model("model_2")

        # Verify cleanup
        self.assertNotIn("model_1", self.manager.get_active_models())
        self.assertNotIn("model_2", self.manager.get_active_models())
        self.assertEqual(self.manager.get_model_state("model_1"), "destroyed")
        self.assertEqual(self.manager.get_model_state("model_2"), "destroyed")

    def test_version_compatibility(self):
        """Test model versioning and compatibility."""
        # Register model versions
        self.manager.version_manager.register_model_version("test_model", "v1.0", "/path/v1")
        self.manager.version_manager.register_model_version("test_model", "v2.0", "/path/v2")
        
        # Check latest version
        latest = self.manager.version_manager.get_latest_version("test_model")
        self.assertEqual(latest, "v2.0")
        
        # Check compatibility
        is_compatible = self.manager.version_manager.check_compatibility("v1.0", "v2.0")
        self.assertTrue(is_compatible)

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Try to create a non-existent model
        with self.assertRaises(ValueError):
            self.manager.create_model("non_existent_model", model_id="bad_model")
        
        # Try to get a non-existent model
        self.assertIsNone(self.manager.get_model("non_existent_model"))
        
        # Try to activate a non-existent model
        self.assertFalse(self.manager.activate_model("non_existent_model"))
        
        # Try to deactivate a non-existent model
        self.assertFalse(self.manager.deactivate_model("non_existent_model"))
        
        # Try to destroy a non-existent model
        self.assertFalse(self.manager.destroy_model("non_existent_model"))


if __name__ == '__main__':
    # Run the tests
    unittest.main()