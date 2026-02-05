"""
Intelligent Scheduling System for Qwen3-4B-Instruct-2507 Model - Init Module

This module initializes the intelligent scheduling system for the Qwen3-4B-Instruct-2507 model.
"""

from .intelligent_scheduler import (
    IntelligentSchedulerConfig,
    IntelligentOperationScheduler,
    apply_intelligent_scheduling_to_model,
    create_intelligent_scheduler_for_qwen3_4b
)

__all__ = [
    "IntelligentSchedulerConfig",
    "IntelligentOperationScheduler",
    "apply_intelligent_scheduling_to_model",
    "create_intelligent_scheduler_for_qwen3_4b"
]