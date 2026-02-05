"""
Qwen3-4B-Instruct-2507 Intelligent Scheduling System

This module implements an intelligent scheduling system with predictive and advanced scheduling policies
for the Qwen3-4B-Instruct-2507 model. The system includes predictive scheduling, intelligent operation prioritization,
and adaptive resource allocation based on workload patterns.
"""

import enum
import time
import heapq
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import threading
import queue
import torch


logger = logging.getLogger(__name__)


class SchedulingPolicy(enum.Enum):
    """Different scheduling policies available."""
    FIFO = "fifo"
    PRIORITY = "priority"
    ROUND_ROBIN = "round_robin"
    PREDICTIVE = "predictive"
    INTELLIGENT = "intelligent"


@dataclass
class IntelligentSchedulerConfig:
    """Configuration for intelligent scheduling."""
    
    max_concurrent_ops: int = 32  # Higher for 4B model
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.INTELLIGENT
    enable_prediction: bool = True
    prediction_horizon: int = 15  # Extended for instruction-following tasks
    enable_adaptive_scheduling: bool = True
    adaptive_window_size: int = 150  # Larger window for adaptive algorithms
    enable_resource_optimization: bool = True
    resource_buffer_percentage: float = 0.15  # 15% buffer for resource allocation
    enable_priority_boosting: bool = True
    priority_decay_factor: float = 0.92  # Slightly faster decay for responsiveness
    enable_load_balancing: bool = True
    load_balance_interval: float = 0.08  # Faster load balancing for instruction tasks
    performance_log_interval: int = 75  # Log performance every N operations


@dataclass
class Operation:
    """Represents an operation to be scheduled."""
    
    id: str
    operation_type: str
    priority: int
    estimated_duration: float
    resource_requirements: Dict[str, Any]
    func: Callable
    args: tuple
    kwargs: dict
    submission_time: float = None
    scheduled_time: float = None
    completion_time: float = None
    
    def __post_init__(self):
        if self.submission_time is None:
            self.submission_time = time.time()


class OperationHistory:
    """Tracks historical operation data for predictive scheduling."""
    
    def __init__(self, window_size: int = 150):
        self.window_size = window_size
        self.history: List[Operation] = []
        
    def add_operation(self, op: Operation):
        self.history.append(op)
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
    def get_pattern_by_type(self, op_type: str):
        """Get historical patterns for a specific operation type."""
        ops = [op for op in self.history if op.operation_type == op_type]
        if not ops:
            return None
            
        avg_duration = sum(op.estimated_duration for op in ops) / len(ops)
        frequency = len(ops) / (time.time() - min(op.submission_time for op in ops)) if ops else 0
        
        return {
            'avg_duration': avg_duration,
            'frequency': frequency,
            'count': len(ops)
        }


class PerformanceMonitor:
    """Monitors scheduling performance metrics."""
    
    def __init__(self, log_interval: int = 75):
        self.log_interval = log_interval
        self.completed_ops_count = 0
        self.total_wait_time = 0.0
        self.total_execution_time = 0.0
        self.total_ops_count = 0
        
    def record_completion(self, op: Operation):
        self.completed_ops_count += 1
        self.total_ops_count += 1
        wait_time = (op.scheduled_time or op.submission_time) - op.submission_time
        exec_time = (op.completion_time or time.time()) - (op.scheduled_time or op.submission_time)
        self.total_wait_time += wait_time
        self.total_execution_time += exec_time
        
        if self.completed_ops_count % self.log_interval == 0:
            avg_wait = self.total_wait_time / self.completed_ops_count
            avg_exec = self.total_execution_time / self.completed_ops_count
            logger.info(
                f"Scheduling Performance - "
                f"Avg Wait: {avg_wait:.4f}s, "
                f"Avg Exec: {avg_exec:.4f}s, "
                f"Total Ops: {self.total_ops_count}"
            )


class IntelligentOperationScheduler:
    """
    Advanced operation scheduler with predictive and intelligent scheduling policies for Qwen3-4B-Instruct-2507.
    """
    
    def __init__(self, config: IntelligentSchedulerConfig):
        self.config = config
        self.operation_queue = queue.PriorityQueue()
        self.active_operations: Dict[str, Operation] = {}
        self.operation_history = OperationHistory(window_size=config.adaptive_window_size)
        self.performance_monitor = PerformanceMonitor(log_interval=config.performance_log_interval)
        self.resource_manager = ResourceManager()
        self.prediction_cache: Dict[str, Any] = {}
        self.lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
        # Thread for background scheduling operations
        self.scheduler_thread = threading.Thread(target=self._scheduler_worker, daemon=True)
        self.scheduler_thread.start()
        
        logger.info(f"Initialized IntelligentOperationScheduler with policy: {config.scheduling_policy.value}")

    def submit_operation(self, op: Operation) -> str:
        """Submit an operation for scheduling."""
        with self.lock:
            # Calculate dynamic priority based on policy
            final_priority = self._calculate_priority(op)
            
            # Add to queue with priority (lower number = higher priority)
            self.operation_queue.put((-final_priority, time.time(), op))
            logger.debug(f"Submitted operation {op.id} with priority {final_priority}")
            
            return op.id

    def _calculate_priority(self, op: Operation) -> int:
        """Calculate the final priority based on the selected policy."""
        base_priority = op.priority
        
        if self.config.scheduling_policy == SchedulingPolicy.PRIORITY:
            return base_priority
        elif self.config.scheduling_policy == SchedulingPolicy.ROUND_ROBIN:
            return base_priority + int(time.time() * 1000) % 100  # Add timestamp to break ties
        elif self.config.scheduling_policy == SchedulingPolicy.PREDICTIVE:
            return self._predictive_priority(op)
        elif self.config.scheduling_policy == SchedulingPolicy.INTELLIGENT:
            return self._intelligent_priority(op)
        else:  # FIFO
            return base_priority

    def _predictive_priority(self, op: Operation) -> int:
        """Calculate priority based on predictive analysis."""
        if not self.config.enable_prediction:
            return op.priority
            
        # Check if we have historical data for this operation type
        pattern = self.operation_history.get_pattern_by_type(op.operation_type)
        if pattern:
            # Boost priority if this operation type tends to be frequent
            freq_factor = min(pattern['frequency'] * 10, 50)  # Cap at 50
            duration_factor = max(50 - (pattern['avg_duration'] * 10), 0)  # Shorter ops get boost
            return op.priority + int(freq_factor + duration_factor)
        
        return op.priority

    def _intelligent_priority(self, op: Operation) -> int:
        """Calculate priority using multiple intelligent factors."""
        if not self.config.enable_prediction:
            base_priority = op.priority
        else:
            # Get historical pattern if available
            pattern = self.operation_history.get_pattern_by_type(op.operation_type)
            if pattern:
                # Combine frequency, duration, and recency factors
                freq_factor = min(pattern['frequency'] * 15, 50)
                duration_factor = max(50 - (pattern['avg_duration'] * 10), 0)
                
                # Consider recency - recently active operation types get priority
                recent_ops = [o for o in self.operation_history.history 
                             if o.operation_type == op.operation_type and 
                                time.time() - o.completion_time < 30.0 if o.completion_time else True]
                recency_factor = min(len(recent_ops) * 10, 30)
                
                base_priority = op.priority + int(freq_factor + duration_factor + recency_factor)
            else:
                base_priority = op.priority
        
        # Apply priority decay if enabled
        if self.config.enable_priority_boosting:
            time_since_submission = time.time() - op.submission_time
            decay_multiplier = 1.0 + (time_since_submission * 0.01)  # Small boost over time
            base_priority = int(base_priority * decay_multiplier)
        
        return base_priority

    def _scheduler_worker(self):
        """Background worker that schedules operations based on availability and policy."""
        while not self.shutdown_event.is_set():
            try:
                # Check if we can schedule more operations
                if len(self.active_operations) < self.config.max_concurrent_ops and not self.operation_queue.empty():
                    # Get the highest priority operation
                    _, _, op = self.operation_queue.get_nowait()
                    
                    # Check resource availability
                    if self.config.enable_resource_optimization:
                        if not self.resource_manager.can_allocate(op.resource_requirements):
                            # Put back in queue and wait
                            self.operation_queue.put((-self._calculate_priority(op), time.time(), op))
                            time.sleep(0.01)
                            continue
                    
                    # Allocate resources and start operation
                    with self.lock:
                        self.active_operations[op.id] = op
                        op.scheduled_time = time.time()
                        
                        if self.config.enable_resource_optimization:
                            self.resource_manager.allocate(op.resource_requirements)
                    
                    # Execute operation in a separate thread
                    exec_thread = threading.Thread(
                        target=self._execute_operation, 
                        args=(op,),
                        daemon=True
                    )
                    exec_thread.start()
                    
                time.sleep(0.001)  # Small delay to prevent busy waiting
                
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in scheduler worker: {e}")
                time.sleep(0.01)

    def _execute_operation(self, op: Operation):
        """Execute a single operation and handle completion."""
        try:
            # Actually execute the operation
            result = op.func(*op.args, **op.kwargs)
            
            # Record completion
            with self.lock:
                op.completion_time = time.time()
                if op.id in self.active_operations:
                    del self.active_operations[op.id]
                
                # Free up resources
                if self.config.enable_resource_optimization:
                    self.resource_manager.deallocate(op.resource_requirements)
                
                # Add to history for future predictions
                self.operation_history.add_operation(op)
                self.performance_monitor.record_completion(op)
                
            logger.debug(f"Completed operation {op.id}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing operation {op.id}: {e}")
            with self.lock:
                if op.id in self.active_operations:
                    del self.active_operations[op.id]
                if self.config.enable_resource_optimization:
                    self.resource_manager.deallocate(op.resource_requirements)

    def get_active_operations_count(self) -> int:
        """Get the count of currently active operations."""
        with self.lock:
            return len(self.active_operations)

    def get_queue_size(self) -> int:
        """Get the size of the operation queue."""
        return self.operation_queue.qsize()

    def shutdown(self):
        """Shutdown the scheduler gracefully."""
        self.shutdown_event.set()
        self.scheduler_thread.join(timeout=2.0)
        logger.info("IntelligentOperationScheduler shut down")


class ResourceManager:
    """Manages resource allocation for operations."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.allocated_resources: Dict[str, float] = {}
        # Higher limits for 4B model
        self.resource_limits = {
            'gpu_memory': 1024 * 1024 * 1024 * 12,  # 12GB GPU memory
            'cpu_cores': 24,
            'bandwidth': 1500,  # arbitrary units
        }
        
    def can_allocate(self, requirements: Dict[str, Any]) -> bool:
        """Check if resources can be allocated for the given requirements."""
        with self.lock:
            for resource, amount in requirements.items():
                if resource in self.resource_limits:
                    currently_allocated = self.allocated_resources.get(resource, 0)
                    if currently_allocated + amount > self.resource_limits[resource]:
                        return False
            return True
    
    def allocate(self, requirements: Dict[str, Any]):
        """Allocate resources for an operation."""
        with self.lock:
            for resource, amount in requirements.items():
                current = self.allocated_resources.get(resource, 0)
                self.allocated_resources[resource] = current + amount
    
    def deallocate(self, requirements: Dict[str, Any]):
        """Deallocate resources for an operation."""
        with self.lock:
            for resource, amount in requirements.items():
                current = self.allocated_resources.get(resource, 0)
                self.allocated_resources[resource] = max(0, current - amount)


def apply_intelligent_scheduling_to_model(model: torch.nn.Module, config: IntelligentSchedulerConfig) -> torch.nn.Module:
    """
    Apply intelligent scheduling to the model.
    
    Args:
        model: The model to apply scheduling to
        config: Intelligent scheduling configuration
    
    Returns:
        Model with intelligent scheduling applied
    """
    # Create and attach the intelligent scheduler to the model
    model.intelligent_scheduler = IntelligentOperationScheduler(config)
    
    # Add a method to the model to submit operations
    def submit_operation(op: Operation):
        return model.intelligent_scheduler.submit_operation(op)
    
    model.submit_operation = submit_operation
    
    logger.info("Applied intelligent scheduling to Qwen3-4B-Instruct-2507 model")
    return model


def create_intelligent_scheduler_for_qwen3_4b(config) -> IntelligentOperationScheduler:
    """
    Create an intelligent scheduler specifically configured for Qwen3-4B-Instruct-2507.
    
    Args:
        config: Base configuration with model-specific overrides
        
    Returns:
        IntelligentOperationScheduler: The created scheduler
    """
    intelligent_config = IntelligentSchedulerConfig(
        max_concurrent_ops=getattr(config, 'intelligent_scheduling_max_concurrent_ops', 32),
        scheduling_policy=SchedulingPolicy(getattr(config, 'intelligent_scheduling_policy', 'intelligent')),
        enable_prediction=getattr(config, 'intelligent_scheduling_enable_prediction', True),
        prediction_horizon=getattr(config, 'intelligent_scheduling_prediction_horizon', 15),
        enable_adaptive_scheduling=getattr(config, 'intelligent_scheduling_enable_adaptive', True),
        adaptive_window_size=getattr(config, 'intelligent_scheduling_adaptive_window', 150),
        enable_resource_optimization=getattr(config, 'intelligent_scheduling_enable_resource_opt', True),
        resource_buffer_percentage=getattr(config, 'intelligent_scheduling_resource_buffer', 0.15),
        enable_priority_boosting=getattr(config, 'intelligent_scheduling_enable_priority_boost', True),
        priority_decay_factor=getattr(config, 'intelligent_scheduling_priority_decay', 0.92),
        enable_load_balancing=getattr(config, 'intelligent_scheduling_enable_load_balancing', True),
        load_balance_interval=getattr(config, 'intelligent_scheduling_load_balance_interval', 0.08),
        performance_log_interval=getattr(config, 'intelligent_scheduling_performance_log_interval', 75)
    )
    
    return IntelligentOperationScheduler(intelligent_config)


__all__ = [
    "IntelligentSchedulerConfig",
    "IntelligentOperationScheduler", 
    "SchedulingPolicy",
    "Operation",
    "apply_intelligent_scheduling_to_model",
    "create_intelligent_scheduler_for_qwen3_4b"
]