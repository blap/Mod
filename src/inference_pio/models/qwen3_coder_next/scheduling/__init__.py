"""
Intelligent Scheduling System for Qwen3-Coder-Next Model - Init Module

This module initializes the intelligent scheduling system for the Qwen3-Coder-Next model.
"""

from .intelligent_scheduler import (
    IntelligentSchedulerConfig,
    IntelligentOperationScheduler,
    apply_intelligent_scheduling_to_model,
    create_intelligent_scheduler_for_qwen3_coder_next
)

__all__ = [
    "IntelligentSchedulerConfig",
    "IntelligentOperationScheduler",
    "apply_intelligent_scheduling_to_model",
    "create_intelligent_scheduler_for_qwen3_coder_next"
]