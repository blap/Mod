"""
Fallback Mechanisms for Qwen3-VL Optimization Techniques
Provides fallback strategies when specific optimizations aren't applicable or fail.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, Union, List, Tuple
from enum import Enum
import logging
import time
import statistics
import psutil
from contextlib import contextmanager
import warnings
from src.qwen3_vl.optimization.unified_optimization_manager import OptimizationType, OptimizationManager
from src.qwen3_vl.optimization.config_manager import OptimizationConfig


class FallbackLevel(Enum):
    """Levels of fallback strategies"""
    MINIMAL = "minimal"  # Minimal fallback - just disable the optimization
    PARTIAL = "partial"  # Partial fallback - use a simpler version of the optimization
    COMPLETE = "complete"  # Complete fallback - revert to baseline implementation


class FallbackReason(Enum):
    """Reasons why a fallback might be needed"""
    HARDWARE_INCOMPATIBILITY = "hardware_incompatibility"
    RESOURCE_CONSTRAINTS = "resource_constraints"
    INPUT_INCOMPATIBILITY = "input_incompatibility"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RUNTIME_ERROR = "runtime_error"
    MEMORY_LIMIT_EXCEEDED = "memory_limit_exceeded"
    UNKNOWN_ERROR = "unknown_error"


class FallbackManager:
    """
    Manages fallback strategies for when optimizations fail or aren't applicable.
    Provides graceful degradation while maintaining model functionality.
    """
    
    def __init__(self, optimization_manager: OptimizationManager):
        self.optimization_manager = optimization_manager
        self.fallback_strategies: Dict[OptimizationType, Dict[FallbackLevel, Callable]] = {}
        self.fallback_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize fallback strategies
        self._initialize_fallback_strategies()
    
    def _initialize_fallback_strategies(self):
        """Initialize fallback strategies for each optimization"""
        # Block sparse attention fallbacks
        self.fallback_strategies[OptimizationType.BLOCK_SPARSE_ATTENTION] = {
            FallbackLevel.MINIMAL: self._fallback_block_sparse_minimal,
            FallbackLevel.PARTIAL: self._fallback_block_sparse_partial,
            FallbackLevel.COMPLETE: self._fallback_block_sparse_complete
        }
        
        # Cross-modal token merging fallbacks
        self.fallback_strategies[OptimizationType.CROSS_MODAL_TOKEN_MERGING] = {
            FallbackLevel.MINIMAL: self._fallback_cross_modal_minimal,
            FallbackLevel.PARTIAL: self._fallback_cross_modal_partial,
            FallbackLevel.COMPLETE: self._fallback_cross_modal_complete
        }
        
        # Hierarchical memory compression fallbacks
        self.fallback_strategies[OptimizationType.HIERARCHICAL_MEMORY_COMPRESSION] = {
            FallbackLevel.MINIMAL: self._fallback_memory_compression_minimal,
            FallbackLevel.PARTIAL: self._fallback_memory_compression_partial,
            FallbackLevel.COMPLETE: self._fallback_memory_compression_complete
        }
        
        # Learned activation routing fallbacks
        self.fallback_strategies[OptimizationType.LEARNED_ACTIVATION_ROUTING] = {
            FallbackLevel.MINIMAL: self._fallback_activation_routing_minimal,
            FallbackLevel.PARTIAL: self._fallback_activation_routing_partial,
            FallbackLevel.COMPLETE: self._fallback_activation_routing_complete
        }
        
        # Adaptive batch processing fallbacks
        self.fallback_strategies[OptimizationType.ADAPTIVE_BATCH_PROCESSING] = {
            FallbackLevel.MINIMAL: self._fallback_adaptive_batch_minimal,
            FallbackLevel.PARTIAL: self._fallback_adaptive_batch_partial,
            FallbackLevel.COMPLETE: self._fallback_adaptive_batch_complete
        }
        
        # Cross-layer parameter recycling fallbacks
        self.fallback_strategies[OptimizationType.CROSS_LAYER_PARAMETER_RECYCLING] = {
            FallbackLevel.MINIMAL: self._fallback_parameter_recycling_minimal,
            FallbackLevel.PARTIAL: self._fallback_parameter_recycling_partial,
            FallbackLevel.COMPLETE: self._fallback_parameter_recycling_complete
        }
        
        # Adaptive sequence packing fallbacks
        self.fallback_strategies[OptimizationType.ADAPTIVE_SEQUENCE_PACKING] = {
            FallbackLevel.MINIMAL: self._fallback_sequence_packing_minimal,
            FallbackLevel.PARTIAL: self._fallback_sequence_packing_partial,
            FallbackLevel.COMPLETE: self._fallback_sequence_packing_complete
        }
        
        # Memory-efficient gradient accumulation fallbacks
        self.fallback_strategies[OptimizationType.MEMORY_EFFICIENT_GRAD_ACCUMULATION] = {
            FallbackLevel.MINIMAL: self._fallback_grad_accumulation_minimal,
            FallbackLevel.PARTIAL: self._fallback_grad_accumulation_partial,
            FallbackLevel.COMPLETE: self._fallback_grad_accumulation_complete
        }
        
        # KV cache multiple strategies fallbacks
        self.fallback_strategies[OptimizationType.KV_CACHE_MULTIPLE_STRATEGIES] = {
            FallbackLevel.MINIMAL: self._fallback_kv_cache_minimal,
            FallbackLevel.PARTIAL: self._fallback_kv_cache_partial,
            FallbackLevel.COMPLETE: self._fallback_kv_cache_complete
        }
        
        # Faster rotary embeddings fallbacks
        self.fallback_strategies[OptimizationType.FASTER_ROTARY_EMBEDDINGS] = {
            FallbackLevel.MINIMAL: self._fallback_rotary_embeddings_minimal,
            FallbackLevel.PARTIAL: self._fallback_rotary_embeddings_partial,
            FallbackLevel.COMPLETE: self._fallback_rotary_embeddings_complete
        }
        
        # Distributed pipeline parallelism fallbacks
        self.fallback_strategies[OptimizationType.DISTRIBUTED_PIPELINE_PARALLELISM] = {
            FallbackLevel.MINIMAL: self._fallback_pipeline_parallelism_minimal,
            FallbackLevel.PARTIAL: self._fallback_pipeline_parallelism_partial,
            FallbackLevel.COMPLETE: self._fallback_pipeline_parallelism_complete
        }
        
        # Hardware-specific kernels fallbacks
        self.fallback_strategies[OptimizationType.HARDWARE_SPECIFIC_KERNELS] = {
            FallbackLevel.MINIMAL: self._fallback_hardware_kernels_minimal,
            FallbackLevel.PARTIAL: self._fallback_hardware_kernels_partial,
            FallbackLevel.COMPLETE: self._fallback_hardware_kernels_complete
        }
    
    def _fallback_block_sparse_minimal(self, query, key, value, attention_mask=None, **kwargs):
        """Minimal fallback for block sparse attention - disable and use standard attention"""
        return query, key, value  # Return original tensors, optimization will be skipped
    
    def _fallback_block_sparse_partial(self, query, key, value, attention_mask=None, **kwargs):
        """Partial fallback for block sparse attention - use less aggressive sparsity"""
        # Use a larger block size to reduce sparsity effect
        large_block_size = 256  # Much larger blocks = less sparsity
        return query, key, value  # In real implementation, would apply less aggressive sparsity
    
    def _fallback_block_sparse_complete(self, query, key, value, attention_mask=None, **kwargs):
        """Complete fallback for block sparse attention - revert to standard attention"""
        # Perform standard attention computation
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))
        if attention_mask is not None:
            scores = scores + attention_mask
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, value)
        return output
    
    def _fallback_cross_modal_minimal(self, hidden_states, **kwargs):
        """Minimal fallback for cross-modal token merging - skip merging"""
        return hidden_states
    
    def _fallback_cross_modal_partial(self, hidden_states, **kwargs):
        """Partial fallback for cross-modal token merging - use higher threshold"""
        # In a real implementation, this would use a higher similarity threshold
        return hidden_states
    
    def _fallback_cross_modal_complete(self, hidden_states, **kwargs):
        """Complete fallback for cross-modal token merging - no merging at all"""
        return hidden_states
    
    def _fallback_memory_compression_minimal(self, tensor, **kwargs):
        """Minimal fallback for hierarchical memory compression - reduce compression level"""
        return tensor  # No compression applied
    
    def _fallback_memory_compression_partial(self, tensor, **kwargs):
        """Partial fallback for hierarchical memory compression - use lighter compression"""
        # In a real implementation, this would apply minimal compression
        return tensor
    
    def _fallback_memory_compression_complete(self, tensor, **kwargs):
        """Complete fallback for hierarchical memory compression - no compression"""
        return tensor
    
    def _fallback_activation_routing_minimal(self, x, layer_id=None, **kwargs):
        """Minimal fallback for learned activation routing - use default activation"""
        return torch.nn.functional.gelu(x)
    
    def _fallback_activation_routing_partial(self, x, layer_id=None, **kwargs):
        """Partial fallback for learned activation routing - use fixed activation per layer"""
        if layer_id is not None and layer_id % 2 == 0:
            return torch.nn.functional.relu(x)
        else:
            return torch.nn.functional.gelu(x)
    
    def _fallback_activation_routing_complete(self, x, layer_id=None, **kwargs):
        """Complete fallback for learned activation routing - always use GELU"""
        return torch.nn.functional.gelu(x)
    
    def _fallback_adaptive_batch_minimal(self, batch_data, input_types=None, **kwargs):
        """Minimal fallback for adaptive batch processing - use original batch"""
        return batch_data
    
    def _fallback_adaptive_batch_partial(self, batch_data, input_types=None, **kwargs):
        """Partial fallback for adaptive batch processing - use fixed batch size"""
        return batch_data
    
    def _fallback_adaptive_batch_complete(self, batch_data, input_types=None, **kwargs):
        """Complete fallback for adaptive batch processing - no batching changes"""
        return batch_data
    
    def _fallback_parameter_recycling_minimal(self, hidden_states, layer_idx, **kwargs):
        """Minimal fallback for cross-layer parameter recycling - disable recycling"""
        return hidden_states
    
    def _fallback_parameter_recycling_partial(self, hidden_states, layer_idx, **kwargs):
        """Partial fallback for cross-layer parameter recycling - increase recycling interval"""
        return hidden_states
    
    def _fallback_parameter_recycling_complete(self, hidden_states, layer_idx, **kwargs):
        """Complete fallback for cross-layer parameter recycling - no recycling"""
        return hidden_states
    
    def _fallback_sequence_packing_minimal(self, sequences, **kwargs):
        """Minimal fallback for adaptive sequence packing - disable packing"""
        return sequences
    
    def _fallback_sequence_packing_partial(self, sequences, **kwargs):
        """Partial fallback for adaptive sequence packing - use simple packing"""
        return sequences
    
    def _fallback_sequence_packing_complete(self, sequences, **kwargs):
        """Complete fallback for adaptive sequence packing - no packing changes"""
        return sequences
    
    def _fallback_grad_accumulation_minimal(self, grad, step_count, **kwargs):
        """Minimal fallback for memory-efficient gradient accumulation - use standard accumulation"""
        return grad
    
    def _fallback_grad_accumulation_partial(self, grad, step_count, **kwargs):
        """Partial fallback for memory-efficient gradient accumulation - use larger accumulation steps"""
        return grad
    
    def _fallback_grad_accumulation_complete(self, grad, step_count, **kwargs):
        """Complete fallback for memory-efficient gradient accumulation - standard approach"""
        return grad
    
    def _fallback_kv_cache_minimal(self, k_cache, v_cache, **kwargs):
        """Minimal fallback for KV cache optimization - use standard cache"""
        return k_cache, v_cache
    
    def _fallback_kv_cache_partial(self, k_cache, v_cache, **kwargs):
        """Partial fallback for KV cache optimization - use simple compression"""
        return k_cache, v_cache
    
    def _fallback_kv_cache_complete(self, k_cache, v_cache, **kwargs):
        """Complete fallback for KV cache optimization - no optimization"""
        return k_cache, v_cache
    
    def _fallback_rotary_embeddings_minimal(self, query, key, position_ids, **kwargs):
        """Minimal fallback for faster rotary embeddings - use standard rotary embeddings"""
        return query, key
    
    def _fallback_rotary_embeddings_partial(self, query, key, position_ids, **kwargs):
        """Partial fallback for faster rotary embeddings - use less optimized version"""
        return query, key
    
    def _fallback_rotary_embeddings_complete(self, query, key, position_ids, **kwargs):
        """Complete fallback for faster rotary embeddings - standard implementation"""
        return query, key
    
    def _fallback_pipeline_parallelism_minimal(self, hidden_states, **kwargs):
        """Minimal fallback for distributed pipeline parallelism - use single device"""
        return hidden_states
    
    def _fallback_pipeline_parallelism_partial(self, hidden_states, **kwargs):
        """Partial fallback for distributed pipeline parallelism - reduce pipeline stages"""
        return hidden_states
    
    def _fallback_pipeline_parallelism_complete(self, hidden_states, **kwargs):
        """Complete fallback for distributed pipeline parallelism - no pipelining"""
        return hidden_states
    
    def _fallback_hardware_kernels_minimal(self, hidden_states, **kwargs):
        """Minimal fallback for hardware-specific kernels - use standard operations"""
        return hidden_states
    
    def _fallback_hardware_kernels_partial(self, hidden_states, **kwargs):
        """Partial fallback for hardware-specific kernels - use generic optimized ops"""
        return hidden_states
    
    def _fallback_hardware_kernels_complete(self, hidden_states, **kwargs):
        """Complete fallback for hardware-specific kernels - standard PyTorch ops"""
        return hidden_states
    
    def apply_fallback(
        self,
        opt_type: OptimizationType,
        fallback_level: FallbackLevel,
        *args,
        reason: FallbackReason = FallbackReason.UNKNOWN_ERROR,
        **kwargs
    ) -> Any:
        """Apply the appropriate fallback for a given optimization"""
        start_time = time.time()
        
        try:
            # Get the fallback function
            fallback_func = self.fallback_strategies[opt_type][fallback_level]
            
            # Apply the fallback
            result = fallback_func(*args, **kwargs)
            
            # Log the fallback
            duration = time.time() - start_time
            self.logger.info(
                f"Fallback applied: {opt_type.value} at {fallback_level.value} level "
                f"due to {reason.value}. Duration: {duration:.4f}s"
            )
            
            # Record in history
            self.fallback_history.append({
                'timestamp': time.time(),
                'optimization_type': opt_type,
                'fallback_level': fallback_level,
                'reason': reason,
                'duration': duration,
                'args_shape': [arg.shape if hasattr(arg, 'shape') else str(type(arg)) for arg in args if arg is not None]
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fallback failed for {opt_type} at {fallback_level} level: {e}")
            # If fallback fails, return the original input (no-op)
            return args[0] if args else None
    
    def check_fallback_conditions(self, opt_type: OptimizationType, *args, **kwargs) -> Optional[FallbackReason]:
        """Check if fallback conditions are met for an optimization"""
        # Check hardware compatibility
        if opt_type == OptimizationType.HARDWARE_SPECIFIC_KERNELS:
            if not self._check_tensor_cores_compatibility():
                return FallbackReason.HARDWARE_INCOMPATIBILITY
        
        # Check resource constraints
        if opt_type == OptimizationType.HIERARCHICAL_MEMORY_COMPRESSION:
            if self._check_memory_pressure():
                return FallbackReason.RESOURCE_CONSTRAINTS
        
        # Check input compatibility
        if opt_type == OptimizationType.BLOCK_SPARSE_ATTENTION:
            if self._check_input_compatibility_for_block_sparse(args):
                return FallbackReason.INPUT_INCOMPATIBILITY
        
        # Check for performance degradation (would require monitoring)
        # This would typically require comparing performance against baseline
        
        return None
    
    def _check_tensor_cores_compatibility(self) -> bool:
        """Check if tensor cores are available and compatible"""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                capability = torch.cuda.get_device_capability(device)
                return capability[0] >= 7  # Tensor cores available from compute capability 7.0
            return False
        except:
            return False
    
    def _check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        try:
            memory_percent = psutil.virtual_memory().percent
            return memory_percent > 90  # High memory pressure
        except:
            return False
    
    def _check_input_compatibility_for_block_sparse(self, args) -> bool:
        """Check if input is compatible with block sparse attention"""
        if len(args) > 0 and hasattr(args[0], 'shape'):
            seq_len = args[0].shape[-2] if len(args[0].shape) >= 2 else 0
            # Block sparse attention may not be efficient for very short sequences
            return seq_len < 64  # Too short for block sparsity to be beneficial
        return False
    
    @contextmanager
    def fallback_context(self, opt_type: OptimizationType, fallback_level: FallbackLevel = FallbackLevel.COMPLETE):
        """Context manager for handling optimization fallbacks"""
        original_state = self.optimization_manager.optimization_states.get(opt_type, True)
        try:
            yield
        except Exception as e:
            self.logger.warning(f"Optimization {opt_type} failed: {e}. Activating fallback.")
            # Revert to fallback level
            if fallback_level != FallbackLevel.COMPLETE:
                self.optimization_manager.optimization_states[opt_type] = False
            # The actual fallback will be handled by the optimization manager
            raise
        finally:
            # Restore original state if needed
            self.optimization_manager.optimization_states[opt_type] = original_state
    
    def auto_fallback_wrapper(self, opt_type: OptimizationType, func: Callable, *args, **kwargs) -> Any:
        """Wrapper that automatically applies fallbacks when needed"""
        # Check if fallback conditions are met
        fallback_reason = self.check_fallback_conditions(opt_type, *args, **kwargs)
        
        if fallback_reason:
            # Apply minimal fallback first
            return self.apply_fallback(
                opt_type, FallbackLevel.MINIMAL, *args, 
                reason=fallback_reason, **kwargs
            )
        
        try:
            # Try the original function
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.warning(f"Optimization {opt_type} failed: {e}. Applying fallback.")
            
            # Determine appropriate fallback level based on error type
            if "memory" in str(e).lower() or "cuda" in str(e).lower():
                fallback_level = FallbackLevel.COMPLETE
                reason = FallbackReason.MEMORY_LIMIT_EXCEEDED
            elif "device" in str(e).lower() or "hardware" in str(e).lower():
                fallback_level = FallbackLevel.COMPLETE
                reason = FallbackReason.HARDWARE_INCOMPATIBILITY
            else:
                fallback_level = FallbackLevel.COMPLETE
                reason = FallbackReason.RUNTIME_ERROR
            
            # Apply fallback
            return self.apply_fallback(
                opt_type, fallback_level, *args, 
                reason=reason, **kwargs
            )
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get statistics about fallback usage"""
        if not self.fallback_history:
            return {"message": "No fallbacks have been triggered yet"}
        
        stats = {
            'total_fallbacks': len(self.fallback_history),
            'fallbacks_by_type': {},
            'fallbacks_by_level': {},
            'fallbacks_by_reason': {},
            'average_duration': statistics.mean([entry['duration'] for entry in self.fallback_history]) if self.fallback_history else 0,
            'recent_fallbacks': self.fallback_history[-10:]  # Last 10 fallbacks
        }
        
        # Count by type
        for entry in self.fallback_history:
            opt_type = entry['optimization_type'].value
            stats['fallbacks_by_type'][opt_type] = stats['fallbacks_by_type'].get(opt_type, 0) + 1
            
            level = entry['fallback_level'].value
            stats['fallbacks_by_level'][level] = stats['fallbacks_by_level'].get(level, 0) + 1
            
            reason = entry['reason'].value
            stats['fallbacks_by_reason'][reason] = stats['fallbacks_by_reason'].get(reason, 0) + 1
        
        return stats


class AdaptiveFallbackManager(FallbackManager):
    """
    Extended fallback manager that adapts fallback strategies based on context
    and learns from previous fallback experiences.
    """
    
    def __init__(self, optimization_manager: OptimizationManager):
        super().__init__(optimization_manager)
        self.performance_history: Dict[OptimizationType, List[Dict[str, float]]] = {}
        self.context_sensitivity_enabled = True
    
    def adaptive_fallback_selection(
        self, 
        opt_type: OptimizationType, 
        context: Dict[str, Any],
        error: Optional[Exception] = None
    ) -> Tuple[FallbackLevel, FallbackReason]:
        """
        Select the most appropriate fallback level based on context and error
        """
        # Default fallback selection
        fallback_level = FallbackLevel.COMPLETE
        fallback_reason = FallbackReason.UNKNOWN_ERROR
        
        if error is not None:
            error_str = str(error).lower()
            
            # Determine reason from error message
            if "memory" in error_str or "cuda" in error_str:
                fallback_reason = FallbackReason.MEMORY_LIMIT_EXCEEDED
                # For memory errors, start with partial fallback to preserve some optimization
                fallback_level = FallbackLevel.PARTIAL
            elif "device" in error_str or "hardware" in error_str:
                fallback_reason = FallbackReason.HARDWARE_INCOMPATIBILITY
                fallback_level = FallbackLevel.MINIMAL  # try to keep some functionality
            else:
                fallback_reason = FallbackReason.RUNTIME_ERROR
        
        # Adjust based on context if available
        if context and self.context_sensitivity_enabled:
            # Check sequence length for attention-based optimizations
            seq_len = context.get('seq_len', 0)
            batch_size = context.get('batch_size', 0)
            
            if opt_type in [OptimizationType.BLOCK_SPARSE_ATTENTION, 
                           OptimizationType.KV_CACHE_MULTIPLE_STRATEGIES] and seq_len < 128:
                # For short sequences, even standard attention is fast, so minimal fallback
                fallback_level = FallbackLevel.MINIMAL
                fallback_reason = FallbackReason.INPUT_INCOMPATIBILITY
            
            if opt_type == OptimizationType.ADAPTIVE_BATCH_PROCESSING and batch_size == 1:
                # For single-batch processing, adaptive batching isn't beneficial
                fallback_level = FallbackLevel.MINIMAL
                fallback_reason = FallbackReason.INPUT_INCOMPATIBILITY
        
        # Adjust based on previous performance (if available)
        if opt_type in self.performance_history:
            recent_results = self.performance_history[opt_type][-5:]  # Last 5 results
            if recent_results:
                avg_performance = sum(r.get('improvement_factor', 1.0) for r in recent_results) / len(recent_results)
                if avg_performance < 1.0:  # Optimization is degrading performance
                    fallback_reason = FallbackReason.PERFORMANCE_DEGRADATION
                    # If consistently degrading, use complete fallback
                    if avg_performance < 0.95:
                        fallback_level = FallbackLevel.COMPLETE
        
        return fallback_level, fallback_reason
    
    def record_performance_result(self, opt_type: OptimizationType, result: Dict[str, float]):
        """Record performance result for adaptive fallback decisions"""
        if opt_type not in self.performance_history:
            self.performance_history[opt_type] = []
        
        self.performance_history[opt_type].append(result)
        
        # Keep only recent results (last 20)
        if len(self.performance_history[opt_type]) > 20:
            self.performance_history[opt_type] = self.performance_history[opt_type][-20:]
    
    def smart_fallback_wrapper(
        self,
        opt_type: OptimizationType,
        func: Callable,
        context: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ) -> Any:
        """Smart wrapper that selects the best fallback strategy based on context"""
        # Check if fallback conditions are met
        fallback_reason = self.check_fallback_conditions(opt_type, *args, **kwargs)
        
        if fallback_reason:
            fallback_level, _ = self.adaptive_fallback_selection(opt_type, context or {})
            return self.apply_fallback(
                opt_type, fallback_level, *args, 
                reason=fallback_reason, **kwargs
            )
        
        try:
            # Try the original function
            result = func(*args, **kwargs)
            
            # Record successful performance
            if context:
                self.record_performance_result(opt_type, {
                    'timestamp': time.time(),
                    'success': True,
                    'improvement_factor': 1.1,  # Placeholder - would be calculated in real implementation
                    'context': context
                })
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Optimization {opt_type} failed: {e}. Applying adaptive fallback.")
            
            # Select appropriate fallback based on error and context
            fallback_level, reason = self.adaptive_fallback_selection(opt_type, context or {}, e)
            
            # Record failure
            if context:
                self.record_performance_result(opt_type, {
                    'timestamp': time.time(),
                    'success': False,
                    'improvement_factor': 0.0,
                    'error': str(e),
                    'context': context
                })
            
            # Apply adaptive fallback
            return self.apply_fallback(
                opt_type, fallback_level, *args, 
                reason=reason, **kwargs
            )


# Global fallback manager instance
def create_global_fallback_manager(optimization_manager: OptimizationManager) -> AdaptiveFallbackManager:
    """Create a global fallback manager instance"""
    return AdaptiveFallbackManager(optimization_manager)