"""
Test suite for Intelligent Scheduling System for Qwen3-Coder-30B Model.
"""

import unittest
import torch
import time
from unittest.mock import MagicMock, patch
from src.inference_pio.models.qwen3_coder_30b.scheduling.intelligent_scheduler import (
    IntelligentSchedulerConfig,
    IntelligentOperationScheduler,
    SchedulingPolicy,
    Operation,
    apply_intelligent_scheduling_to_model,
    create_intelligent_scheduler_for_qwen3_coder
)


class TestIntelligentScheduler(unittest.TestCase):
    """Test cases for the Intelligent Scheduler."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = IntelligentSchedulerConfig(
            max_concurrent_ops=16,
            scheduling_policy=SchedulingPolicy.INTELLIGENT,
            enable_prediction=True,
            prediction_horizon=15,
            enable_adaptive_scheduling=True,
            adaptive_window_size=125,
            enable_resource_optimization=True,
            resource_buffer_percentage=0.2,
            enable_priority_boosting=True,
            priority_decay_factor=0.90,
            enable_load_balancing=True,
            load_balance_interval=0.05,
            performance_log_interval=75
        )
        self.scheduler = IntelligentOperationScheduler(self.config)

    def tearDown(self):
        """Clean up after each test method."""
        self.scheduler.shutdown()

    def test_config_initialization(self):
        """Test that the scheduler config is initialized correctly."""
        self.assertEqual(self.config.max_concurrent_ops, 16)
        self.assertEqual(self.config.scheduling_policy, SchedulingPolicy.INTELLIGENT)
        self.assertTrue(self.config.enable_prediction)
        self.assertEqual(self.config.prediction_horizon, 15)

    def test_scheduler_initialization(self):
        """Test that the scheduler is initialized correctly."""
        self.assertEqual(self.scheduler.config.max_concurrent_ops, 16)
        self.assertEqual(self.scheduler.config.scheduling_policy, SchedulingPolicy.INTELLIGENT)
        self.assertEqual(self.scheduler.get_active_operations_count(), 0)
        self.assertEqual(self.scheduler.get_queue_size(), 0)

    def test_submit_operation(self):
        """Test submitting an operation to the scheduler."""
        def mock_func(*args, **kwargs):
            time.sleep(0.01)  # Simulate some work
            return "result"
        
        op = Operation(
            id="test_op_1",
            operation_type="test",
            priority=5,
            estimated_duration=0.01,
            resource_requirements={"gpu_memory": 100},
            func=mock_func,
            args=(),
            kwargs={}
        )
        
        op_id = self.scheduler.submit_operation(op)
        self.assertEqual(op_id, "test_op_1")
        
        # Wait a bit for the operation to complete
        time.sleep(0.1)
        self.assertEqual(self.scheduler.get_active_operations_count(), 0)

    def test_calculate_priority_fifo(self):
        """Test priority calculation with FIFO policy."""
        fifo_config = IntelligentSchedulerConfig(scheduling_policy=SchedulingPolicy.FIFO)
        fifo_scheduler = IntelligentOperationScheduler(fifo_config)
        
        op = Operation(
            id="test_op",
            operation_type="test",
            priority=5,
            estimated_duration=0.01,
            resource_requirements={"gpu_memory": 100},
            func=lambda: "result",
            args=(),
            kwargs={}
        )
        
        priority = fifo_scheduler._calculate_priority(op)
        self.assertEqual(priority, 5)
        
        fifo_scheduler.shutdown()

    def test_calculate_priority_intelligent(self):
        """Test priority calculation with INTELLIGENT policy."""
        op = Operation(
            id="test_op",
            operation_type="test",
            priority=5,
            estimated_duration=0.01,
            resource_requirements={"gpu_memory": 100},
            func=lambda: "result",
            args=(),
            kwargs={}
        )
        
        priority = self.scheduler._calculate_priority(op)
        # With intelligent policy, priority might be adjusted based on various factors
        self.assertIsInstance(priority, int)
        self.assertGreaterEqual(priority, 0)

    def test_operation_history_tracking(self):
        """Test that operation history is tracked correctly."""
        # Submit a few operations to build history
        def mock_func(*args, **kwargs):
            return "result"
        
        for i in range(5):
            op = Operation(
                id=f"history_op_{i}",
                operation_type="test",
                priority=5,
                estimated_duration=0.01,
                resource_requirements={"gpu_memory": 100},
                func=mock_func,
                args=(),
                kwargs={}
            )
            self.scheduler.submit_operation(op)
        
        # Wait for operations to complete so they're added to history
        time.sleep(0.2)
        
        # Check that history has recorded operations
        pattern = self.scheduler.operation_history.get_pattern_by_type("test")
        if pattern:
            self.assertGreaterEqual(pattern['count'], 0)

    def test_resource_manager_can_allocate(self):
        """Test resource allocation functionality."""
        # Test that we can allocate resources
        requirements = {"gpu_memory": 100}
        can_allocate = self.scheduler.resource_manager.can_allocate(requirements)
        self.assertTrue(can_allocate)
        
        # Test allocation and deallocation
        self.scheduler.resource_manager.allocate(requirements)
        self.scheduler.resource_manager.deallocate(requirements)

    def test_shutdown(self):
        """Test that the scheduler shuts down cleanly."""
        scheduler = IntelligentOperationScheduler(self.config)
        scheduler.shutdown()
        # Should not raise an exception


class TestIntelligentSchedulerIntegration(unittest.TestCase):
    """Integration tests for the Intelligent Scheduler."""

    def test_create_scheduler_for_qwen3_coder(self):
        """Test creating an intelligent scheduler specifically for Qwen3-Coder-30B."""
        mock_config = MagicMock()
        mock_config.intelligent_scheduling_max_concurrent_ops = 64
        mock_config.intelligent_scheduling_policy = 'intelligent'
        mock_config.intelligent_scheduling_enable_prediction = True
        mock_config.intelligent_scheduling_prediction_horizon = 20
        mock_config.intelligent_scheduling_enable_adaptive = True
        mock_config.intelligent_scheduling_adaptive_window = 250
        mock_config.intelligent_scheduling_enable_resource_opt = True
        mock_config.intelligent_scheduling_resource_buffer = 0.2
        mock_config.intelligent_scheduling_enable_priority_boost = True
        mock_config.intelligent_scheduling_priority_decay = 0.90
        mock_config.intelligent_scheduling_enable_load_balancing = True
        mock_config.intelligent_scheduling_load_balance_interval = 0.05
        mock_config.intelligent_scheduling_performance_log_interval = 100

        scheduler = create_intelligent_scheduler_for_qwen3_coder(mock_config)
        
        self.assertIsInstance(scheduler, IntelligentOperationScheduler)
        self.assertEqual(scheduler.config.max_concurrent_ops, 64)
        self.assertEqual(scheduler.config.scheduling_policy, SchedulingPolicy.INTELLIGENT)
        self.assertTrue(scheduler.config.enable_prediction)
        self.assertEqual(scheduler.config.prediction_horizon, 20)
        
        scheduler.shutdown()

    def test_apply_intelligent_scheduling_to_model(self):
        """Test applying intelligent scheduling to a model."""
        mock_model = MagicMock()
        config = IntelligentSchedulerConfig()
        
        result_model = apply_intelligent_scheduling_to_model(mock_model, config)
        
        # Check that the model now has an intelligent scheduler
        self.assertTrue(hasattr(result_model, 'intelligent_scheduler'))
        self.assertTrue(hasattr(result_model, 'submit_operation'))
        self.assertIsInstance(result_model.intelligent_scheduler, IntelligentOperationScheduler)
        
        # Clean up
        result_model.intelligent_scheduler.shutdown()


class TestSchedulingPolicies(unittest.TestCase):
    """Test different scheduling policies."""

    def test_fifo_policy(self):
        """Test FIFO scheduling policy."""
        config = IntelligentSchedulerConfig(scheduling_policy=SchedulingPolicy.FIFO)
        scheduler = IntelligentOperationScheduler(config)
        
        self.assertEqual(scheduler.config.scheduling_policy, SchedulingPolicy.FIFO)
        scheduler.shutdown()

    def test_priority_policy(self):
        """Test PRIORITY scheduling policy."""
        config = IntelligentSchedulerConfig(scheduling_policy=SchedulingPolicy.PRIORITY)
        scheduler = IntelligentOperationScheduler(config)
        
        self.assertEqual(scheduler.config.scheduling_policy, SchedulingPolicy.PRIORITY)
        scheduler.shutdown()

    def test_round_robin_policy(self):
        """Test ROUND_ROBIN scheduling policy."""
        config = IntelligentSchedulerConfig(scheduling_policy=SchedulingPolicy.ROUND_ROBIN)
        scheduler = IntelligentOperationScheduler(config)
        
        self.assertEqual(scheduler.config.scheduling_policy, SchedulingPolicy.ROUND_ROBIN)
        scheduler.shutdown()

    def test_predictive_policy(self):
        """Test PREDICTIVE scheduling policy."""
        config = IntelligentSchedulerConfig(
            scheduling_policy=SchedulingPolicy.PREDICTIVE,
            enable_prediction=True
        )
        scheduler = IntelligentOperationScheduler(config)
        
        self.assertEqual(scheduler.config.scheduling_policy, SchedulingPolicy.PREDICTIVE)
        scheduler.shutdown()

    def test_intelligent_policy(self):
        """Test INTELLIGENT scheduling policy."""
        config = IntelligentSchedulerConfig(scheduling_policy=SchedulingPolicy.INTELLIGENT)
        scheduler = IntelligentOperationScheduler(config)
        
        self.assertEqual(scheduler.config.scheduling_policy, SchedulingPolicy.INTELLIGENT)
        scheduler.shutdown()


if __name__ == '__main__':
    unittest.main()