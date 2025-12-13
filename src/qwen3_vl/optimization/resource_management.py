"""Resource Management System for Qwen3-VL Optimization Techniques
Coordinates memory and compute resources between all optimization techniques to ensure
efficient utilization and prevent resource conflicts.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import time
import psutil
from dataclasses import dataclass
import math


@dataclass
class ResourceRequest:
    """Represents a resource request from an optimization."""
    optimization_name: str
    memory_requested: int  # in bytes
    compute_requested: float  # fraction of available compute
    duration_estimate: float  # in seconds
    priority: int = 0  # Lower number means higher priority


class ResourceManager:
    """Main resource manager that coordinates memory and compute resources between optimizations."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize resource pools
        self._initialize_resource_pools()
        
        # Resource allocation tracking
        self.resource_allocations = {}
        self.resource_requests_queue = []
        
        # Initialize resource policies
        self.resource_policies = self._initialize_resource_policies()
        
        self.logger.info("Resource Manager initialized")
    
    def _initialize_resource_pools(self):
        """Initialize memory and compute resource pools."""
        # Memory pools
        if torch.cuda.is_available():
            # Get GPU memory info
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            self.gpu_memory_pool = {
                'total': gpu_memory,
                'available': gpu_memory,
                'reserved': 0,
                'used': 0
            }
        else:
            # Use system memory as approximation
            system_memory = psutil.virtual_memory().total
            self.gpu_memory_pool = {
                'total': system_memory,
                'available': system_memory,
                'reserved': 0,
                'used': 0
            }
        
        # CPU memory pool
        self.cpu_memory_pool = {
            'total': psutil.virtual_memory().total,
            'available': psutil.virtual_memory().available,
            'reserved': 0,
            'used': 0
        }
        
        # Compute resources
        self.gpu_compute_pool = {
            'total_units': 100.0,  # Represented as percentage
            'available_units': 100.0,
            'used_units': 0.0
        }
        
        self.cpu_compute_pool = {
            'total_units': psutil.cpu_count() * 100.0,  # Total CPU percentage available
            'available_units': psutil.cpu_count() * 100.0,
            'used_units': 0.0
        }
    
    def _initialize_resource_policies(self) -> Dict[str, Any]:
        """Initialize resource allocation policies."""
        return {
            'memory_reservation_policy': 'conservative',  # conservative, aggressive, balanced
            'compute_sharing_policy': 'fair',  # fair, priority_based, proportional
            'allocation_timeout': 10.0,  # seconds
            'fallback_on_resource_exhaustion': True,
            'memory_fragmentation_tolerance': 0.1,  # 10% tolerance
            'priority_boost_for_critical_optimizations': True
        }
    
    def request_resources(self, request: ResourceRequest) -> Tuple[bool, Dict[str, Any]]:
        """Process a resource request from an optimization."""
        # Check if resources are available
        memory_available = self._check_memory_availability(request)
        compute_available = self._check_compute_availability(request)
        
        if memory_available and compute_available:
            # Allocate resources
            allocation = self._allocate_resources(request)
            return True, allocation
        else:
            # Resources not available, try to free up resources or return failure
            freed_resources = self._try_free_resources(request)
            
            # Re-check availability after freeing resources
            memory_available = self._check_memory_availability(request)
            compute_available = self._check_compute_availability(request)
            
            if memory_available and compute_available:
                allocation = self._allocate_resources(request)
                return True, allocation
            else:
                # Return failure with information about available resources
                available_resources = {
                    'gpu_memory_available': self.gpu_memory_pool['available'],
                    'cpu_memory_available': self.cpu_memory_pool['available'],
                    'gpu_compute_available': self.gpu_compute_pool['available_units'],
                    'cpu_compute_available': self.cpu_compute_pool['available_units']
                }
                
                if self.resource_policies['fallback_on_resource_exhaustion']:
                    # Provide fallback allocation with reduced resources
                    fallback_allocation = self._allocate_resources_fallback(request)
                    return True, fallback_allocation
                else:
                    return False, available_resources
    
    def _check_memory_availability(self, request: ResourceRequest) -> bool:
        """Check if memory is available for the request."""
        # Check GPU memory if CUDA is available
        if torch.cuda.is_available():
            gpu_required = request.memory_requested if request.optimization_name in ['attention', 'flash_attention'] else 0
            if gpu_required > self.gpu_memory_pool['available']:
                return False
        
        # Check CPU memory
        cpu_required = request.memory_requested if request.optimization_name in ['cpu_optimization', 'memory_management'] else 0
        if cpu_required > self.cpu_memory_pool['available']:
            return False
        
        return True
    
    def _check_compute_availability(self, request: ResourceRequest) -> bool:
        """Check if compute resources are available for the request."""
        # Check GPU compute
        gpu_required = request.compute_requested if request.optimization_name in ['attention', 'flash_attention'] else 0
        if gpu_required * 100 > self.gpu_compute_pool['available_units']:
            return False
        
        # Check CPU compute
        cpu_required = request.compute_requested if request.optimization_name in ['cpu_optimization', 'memory_management'] else 0
        if cpu_required * 100 > self.cpu_compute_pool['available_units']:
            return False
        
        return True
    
    def _allocate_resources(self, request: ResourceRequest) -> Dict[str, Any]:
        """Allocate resources to an optimization."""
        allocation = {
            'optimization_name': request.optimization_name,
            'memory_allocated': 0,
            'compute_allocated': 0.0,
            'allocation_time': time.time()
        }
        
        # Allocate memory
        if torch.cuda.is_available() and request.optimization_name in ['attention', 'flash_attention']:
            self.gpu_memory_pool['available'] -= request.memory_requested
            self.gpu_memory_pool['used'] += request.memory_requested
            allocation['memory_allocated'] = request.memory_requested
            self.resource_allocations[request.optimization_name] = {
                'gpu_memory': request.memory_requested,
                'cpu_memory': 0,
                'gpu_compute': request.compute_requested * 100,
                'cpu_compute': 0,
                'allocation_time': allocation['allocation_time']
            }
        elif request.optimization_name in ['cpu_optimization', 'memory_management']:
            self.cpu_memory_pool['available'] -= request.memory_requested
            self.cpu_memory_pool['used'] += request.memory_requested
            allocation['memory_allocated'] = request.memory_requested
            self.resource_allocations[request.optimization_name] = {
                'gpu_memory': 0,
                'cpu_memory': request.memory_requested,
                'gpu_compute': 0,
                'cpu_compute': request.compute_requested * 100,
                'allocation_time': allocation['allocation_time']
            }
        
        # Allocate compute
        if request.optimization_name in ['attention', 'flash_attention']:
            compute_units = request.compute_requested * 100
            self.gpu_compute_pool['available_units'] -= compute_units
            self.gpu_compute_pool['used_units'] += compute_units
            allocation['compute_allocated'] = compute_units
        elif request.optimization_name in ['cpu_optimization', 'memory_management']:
            compute_units = request.compute_requested * 100
            self.cpu_compute_pool['available_units'] -= compute_units
            self.cpu_compute_pool['used_units'] += compute_units
            allocation['compute_allocated'] = compute_units
        
        return allocation
    
    def _allocate_resources_fallback(self, request: ResourceRequest) -> Dict[str, Any]:
        """Allocate reduced resources as fallback when full resources aren't available."""
        allocation = {
            'optimization_name': request.optimization_name,
            'memory_allocated': 0,
            'compute_allocated': 0.0,
            'allocation_time': time.time(),
            'fallback_applied': True
        }
        
        # Allocate reduced memory
        reduced_memory = min(request.memory_requested, self.gpu_memory_pool['available'] // 2)
        if torch.cuda.is_available() and request.optimization_name in ['attention', 'flash_attention']:
            self.gpu_memory_pool['available'] -= reduced_memory
            self.gpu_memory_pool['used'] += reduced_memory
            allocation['memory_allocated'] = reduced_memory
            self.resource_allocations[request.optimization_name] = {
                'gpu_memory': reduced_memory,
                'cpu_memory': 0,
                'gpu_compute': (request.compute_requested * 100) / 2,  # Reduced compute allocation
                'cpu_compute': 0,
                'allocation_time': allocation['allocation_time']
            }
        else:
            reduced_memory = min(request.memory_requested, self.cpu_memory_pool['available'] // 2)
            self.cpu_memory_pool['available'] -= reduced_memory
            self.cpu_memory_pool['used'] += reduced_memory
            allocation['memory_allocated'] = reduced_memory
            self.resource_allocations[request.optimization_name] = {
                'gpu_memory': 0,
                'cpu_memory': reduced_memory,
                'gpu_compute': 0,
                'cpu_compute': (request.compute_requested * 100) / 2,  # Reduced compute allocation
                'allocation_time': allocation['allocation_time']
            }
        
        # Allocate reduced compute
        reduced_compute = (request.compute_requested * 100) / 2
        if request.optimization_name in ['attention', 'flash_attention']:
            self.gpu_compute_pool['available_units'] -= reduced_compute
            self.gpu_compute_pool['used_units'] += reduced_compute
            allocation['compute_allocated'] = reduced_compute
        elif request.optimization_name in ['cpu_optimization', 'memory_management']:
            self.cpu_compute_pool['available_units'] -= reduced_compute
            self.cpu_compute_pool['used_units'] += reduced_compute
            allocation['compute_allocated'] = reduced_compute
        
        self.logger.warning(f"Applied fallback resource allocation for {request.optimization_name}")
        return allocation
    
    def _try_free_resources(self, request: ResourceRequest) -> bool:
        """Try to free up resources by releasing unused allocations."""
        freed = False
        
        # Free GPU memory if available
        if torch.cuda.is_available():
            current_gpu_memory = torch.cuda.memory_allocated()
            reserved_gpu_memory = self.gpu_memory_pool['total'] - self.gpu_memory_pool['available']
            
            # Calculate how much memory we can potentially free
            potentially_freeable = reserved_gpu_memory - current_gpu_memory
            if potentially_freeable > request.memory_requested:
                # Try to free some memory by clearing cache
                torch.cuda.empty_cache()
                self.gpu_memory_pool['available'] = self.gpu_memory_pool['total'] - torch.cuda.memory_allocated()
                freed = True
        
        # Free CPU memory if needed
        if self.cpu_memory_pool['available'] < request.memory_requested:
            # Try to free CPU memory
            psutil.virtual_memory()
            self.cpu_memory_pool['available'] = psutil.virtual_memory().available
            freed = True
        
        return freed
    
    def release_resources(self, optimization_name: str):
        """Release resources allocated to an optimization."""
        if optimization_name in self.resource_allocations:
            alloc_info = self.resource_allocations[optimization_name]
            
            # Release memory
            if alloc_info['gpu_memory'] > 0:
                self.gpu_memory_pool['available'] += alloc_info['gpu_memory']
                self.gpu_memory_pool['used'] -= alloc_info['gpu_memory']
            
            if alloc_info['cpu_memory'] > 0:
                self.cpu_memory_pool['available'] += alloc_info['cpu_memory']
                self.cpu_memory_pool['used'] -= alloc_info['cpu_memory']
            
            # Release compute
            if alloc_info['gpu_compute'] > 0:
                self.gpu_compute_pool['available_units'] += alloc_info['gpu_compute']
                self.gpu_compute_pool['used_units'] -= alloc_info['gpu_compute']
            
            if alloc_info['cpu_compute'] > 0:
                self.cpu_compute_pool['available_units'] += alloc_info['cpu_compute']
                self.cpu_compute_pool['used_units'] -= alloc_info['cpu_compute']
            
            # Remove allocation record
            del self.resource_allocations[optimization_name]
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        return {
            'gpu_memory': {
                'total': self.gpu_memory_pool['total'],
                'available': self.gpu_memory_pool['available'],
                'used': self.gpu_memory_pool['used'],
                'utilization': self.gpu_memory_pool['used'] / self.gpu_memory_pool['total'] if self.gpu_memory_pool['total'] > 0 else 0
            },
            'cpu_memory': {
                'total': self.cpu_memory_pool['total'],
                'available': self.cpu_memory_pool['available'],
                'used': self.cpu_memory_pool['used'],
                'utilization': self.cpu_memory_pool['used'] / self.cpu_memory_pool['total'] if self.cpu_memory_pool['total'] > 0 else 0
            },
            'gpu_compute': {
                'total': self.gpu_compute_pool['total_units'],
                'available': self.gpu_compute_pool['available_units'],
                'used': self.gpu_compute_pool['used_units'],
                'utilization': self.gpu_compute_pool['used_units'] / self.gpu_compute_pool['total_units'] if self.gpu_compute_pool['total_units'] > 0 else 0
            },
            'cpu_compute': {
                'total': self.cpu_compute_pool['total_units'],
                'available': self.cpu_compute_pool['available_units'],
                'used': self.cpu_compute_pool['used_units'],
                'utilization': self.cpu_compute_pool['used_units'] / self.cpu_compute_pool['total_units'] if self.cpu_compute_pool['total_units'] > 0 else 0
            },
            'active_allocations': list(self.resource_allocations.keys()),
            'allocation_details': self.resource_allocations
        }


class MemoryManager:
    """Memory-specific resource management for optimization techniques."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize memory pools
        self._initialize_memory_pools()
        
        # Memory optimization strategies
        self.memory_strategies = self._initialize_memory_strategies()
        
        self.logger.info("Memory Manager initialized")
    
    def _initialize_memory_pools(self):
        """Initialize memory pools for different types of tensors."""
        # Memory pools for different tensor types
        self.tensor_pools = {
            'attention_weights': torch.nn.Module(),  # Will store reusable tensors
            'kv_cache': torch.nn.Module(),
            'intermediate': torch.nn.Module(),
            'embeddings': torch.nn.Module(),
            'activations': torch.nn.Module()
        }
        
        # Pre-allocated memory blocks
        self.pre_allocated_blocks = {}
        
        # Memory fragmentation tracking
        self.memory_fragmentation = 0.0
    
    def _initialize_memory_strategies(self) -> Dict[str, Any]:
        """Initialize memory optimization strategies."""
        return {
            'tensor_caching': True,
            'memory_pooling': True,
            'lazy_allocation': True,
            'memory_sharing': True,
            'precision_optimization': True,
            'batch_processing_optimization': True
        }
    
    def get_memory_efficient_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                                   tensor_type: str = 'general') -> torch.Tensor:
        """Get a memory-efficient tensor based on type and hardware capabilities."""
        # Apply memory optimizations based on tensor type
        if tensor_type == 'attention_weights':
            # For attention weights, consider using half precision if possible
            if self.memory_strategies['precision_optimization'] and dtype == torch.float32:
                dtype = torch.float16
        elif tensor_type == 'kv_cache':
            # For KV cache, consider using low-rank approximation
            if self.memory_strategies['memory_pooling']:
                # Apply low-rank approximation if specified
                pass  # Would implement low-rank approximation here
        elif tensor_type == 'embeddings':
            # For embeddings, consider memory layout optimizations
            if self.memory_strategies['memory_sharing']:
                # Align dimensions for better memory access
                shape = self._align_tensor_shape_for_memory(shape)
        
        # Create tensor with optimized parameters
        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        tensor = torch.empty(shape, dtype=dtype, device=device)
        
        return tensor
    
    def _align_tensor_shape_for_memory(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Align tensor shape for better memory access patterns."""
        # For NVIDIA SM61, align to multiples of 32 for better memory coalescing
        if torch.cuda.is_available():
            # Check if we're on an SM61 GPU
            device_prop = torch.cuda.get_device_properties(torch.cuda.current_device())
            if device_prop.major == 6 and device_prop.minor == 1:  # SM61
                # Align the last dimension (usually head dimension or feature dimension)
                new_shape = list(shape)
                if len(new_shape) > 0:
                    # Align to multiples of 32
                    aligned_dim = math.ceil(new_shape[-1] / 32) * 32
                    new_shape[-1] = aligned_dim
                return tuple(new_shape)
        
        return shape
    
    def release_unused_tensors(self):
        """Release tensors that are no longer needed."""
        # Clear PyTorch's memory cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Reset fragmentation counter
        self.memory_fragmentation = 0.0


class ComputeResourceManager:
    """Compute resource management for optimization techniques."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize compute resource pools
        self._initialize_compute_pools()
        
        # Compute scheduling policies
        self.scheduling_policies = self._initialize_scheduling_policies()
        
        self.logger.info("Compute Resource Manager initialized")
    
    def _initialize_compute_pools(self):
        """Initialize compute resource pools."""
        if torch.cuda.is_available():
            # Get compute capabilities
            device_prop = torch.cuda.get_device_properties(0)
            self.compute_capabilities = {
                'major': device_prop.major,
                'minor': device_prop.minor,
                'warp_size': device_prop.warp_size,
                'max_threads_per_block': device_prop.max_threads_per_block,
                'max_shared_memory_per_block': device_prop.shared_mem_per_block
            }
        else:
            # Default values for CPU
            self.compute_capabilities = {
                'major': 0,
                'minor': 0,
                'warp_size': 32,
                'max_threads_per_block': 1024,
                'max_shared_memory_per_block': 48 * 1024  # 48KB
            }
        
        # Compute scheduling queues
        self.compute_queues = {
            'high_priority': [],
            'normal_priority': [],
            'low_priority': []
        }
    
    def _initialize_scheduling_policies(self) -> Dict[str, Any]:
        """Initialize compute scheduling policies."""
        return {
            'priority_scheduling': True,
            'load_balancing': True,
            'batch_size_optimization': True,
            'sequence_length_grouping': True
        }
    
    def schedule_compute_operation(self, operation: Callable, 
                                  priority: str = 'normal',
                                  input_tensor: Optional[torch.Tensor] = None) -> Any:
        """Schedule a compute operation based on priority and resource availability."""
        # Add operation to appropriate queue
        queue = self.compute_queues[priority]
        queue.append({
            'operation': operation,
            'scheduled_time': time.time(),
            'input_tensor': input_tensor
        })
        
        # Process queue based on scheduling policy
        if self.scheduling_policies['priority_scheduling']:
            # Process high priority first, then normal, then low
            for priority_level in ['high_priority', 'normal_priority', 'low_priority']:
                if self.compute_queues[priority_level]:
                    item = self.compute_queues[priority_level].pop(0)
                    result = item['operation'](item['input_tensor'])
                    return result
        else:
            # Process in FIFO order
            if queue:
                item = queue.pop(0)
                result = item['operation'](item['input_tensor'])
                return result
    
    def optimize_batch_processing(self, batch_size: int) -> int:
        """Optimize batch size based on compute resources."""
        if self.scheduling_policies['batch_size_optimization']:
            # Adjust batch size based on available resources
            if torch.cuda.is_available():
                # On GPU, use a batch size that fits well with the compute capabilities
                max_batch_size = int(self.compute_capabilities['max_threads_per_block'] / 32)  # Threads per warp
                return min(batch_size, max_batch_size)
            else:
                # On CPU, consider number of cores
                return min(batch_size, psutil.cpu_count())
        return batch_size


class OptimizationResourceManager:
    """Resource manager that coordinates all optimization techniques."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize sub-managers
        self.resource_manager = ResourceManager(config)
        self.memory_manager = MemoryManager(config)
        self.compute_manager = ComputeResourceManager(config)
        
        # Track resource usage by optimization type
        self.optimization_resource_usage = {}
        
        # Initialize optimization resource profiles
        self.optimization_profiles = self._initialize_optimization_profiles()
        
        self.logger.info("Optimization Resource Manager initialized")
    
    def _initialize_optimization_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Initialize resource profiles for different optimizations."""
        return {
            'block_sparse_attention': {
                'memory_factor': 0.5,  # 50% of standard attention memory
                'compute_factor': 0.7,  # 70% of standard attention compute
                'priority': 2  # Medium priority
            },
            'cross_modal_token_merging': {
                'memory_factor': 0.6,  # 60% of standard memory
                'compute_factor': 0.8,  # 80% of standard compute
                'priority': 3  # Lower priority
            },
            'hierarchical_memory_compression': {
                'memory_factor': 0.3,  # 30% of original memory
                'compute_factor': 1.1,  # Slightly more compute
                'priority': 1  # High priority
            },
            'learned_activation_routing': {
                'memory_factor': 0.8,  # 80% of standard memory
                'compute_factor': 0.9,  # 90% of standard compute
                'priority': 2  # Medium priority
            },
            'adaptive_batch_processing': {
                'memory_factor': 0.9,  # 90% of standard memory
                'compute_factor': 0.8,  # 80% of standard compute
                'priority': 4  # Lower priority
            },
            'cross_layer_parameter_recycling': {
                'memory_factor': 0.7,  # 70% of standard memory
                'compute_factor': 0.8,  # 80% of standard compute
                'priority': 1  # High priority
            },
            'adaptive_sequence_packing': {
                'memory_factor': 0.7,  # 70% of standard memory
                'compute_factor': 0.9,  # 90% of standard compute
                'priority': 3  # Lower priority
            },
            'memory_efficient_grad_accumulation': {
                'memory_factor': 0.4,  # 40% of standard memory
                'compute_factor': 1.2,  # More compute for accumulation
                'priority': 5  # Lowest priority (only during training)
            },
            'kv_cache_optimization': {
                'memory_factor': 0.3,  # 30% of standard KV cache memory
                'compute_factor': 0.9,  # 90% of standard compute
                'priority': 1  # High priority
            },
            'faster_rotary_embeddings': {
                'memory_factor': 0.9,  # 90% of standard memory
                'compute_factor': 0.7,  # 70% of standard compute
                'priority': 2  # Medium priority
            },
            'distributed_pipeline_parallelism': {
                'memory_factor': 0.6,  # 60% of memory per stage
                'compute_factor': 0.9,  # 90% of compute per stage
                'priority': 6  # Low priority (mainly for inference)
            },
            'hardware_specific_kernels': {
                'memory_factor': 0.8,  # 80% of standard memory
                'compute_factor': 0.6,  # 60% of standard compute
                'priority': 0  # Highest priority
            }
        }
    
    def allocate_resources_for_optimization(self, optimization_name: str, 
                                           input_shape: Tuple[int, ...]) -> Tuple[bool, Dict[str, Any]]:
        """Allocate resources for a specific optimization."""
        if optimization_name not in self.optimization_profiles:
            self.logger.warning(f"Unknown optimization: {optimization_name}, using default profile")
            profile = {
                'memory_factor': 1.0,
                'compute_factor': 1.0,
                'priority': 3
            }
        else:
            profile = self.optimization_profiles[optimization_name]
        
        # Calculate resource requirements based on input shape and profile
        base_memory = self._calculate_base_memory_requirement(input_shape)
        required_memory = int(base_memory * profile['memory_factor'])
        
        # Calculate compute requirement (as fraction of available compute)
        base_compute = self._calculate_base_compute_requirement(input_shape)
        required_compute = base_compute * profile['compute_factor']
        
        # Create resource request
        request = ResourceRequest(
            optimization_name=optimization_name,
            memory_requested=required_memory,
            compute_requested=required_compute,
            duration_estimate=0.1,  # Placeholder duration
            priority=profile['priority']
        )
        
        # Process request through resource manager
        success, allocation = self.resource_manager.request_resources(request)
        
        if success:
            # Track resource usage by optimization
            self.optimization_resource_usage[optimization_name] = {
                'allocated_memory': allocation['memory_allocated'],
                'allocated_compute': allocation['compute_allocated'],
                'allocation_time': allocation['allocation_time'],
                'fallback_applied': allocation.get('fallback_applied', False)
            }
        
        return success, allocation
    
    def _calculate_base_memory_requirement(self, shape: Tuple[int, ...]) -> int:
        """Calculate base memory requirement for a tensor of given shape."""
        # Calculate total number of elements
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        
        # Assume float32 (4 bytes per element) as base
        return total_elements * 4
    
    def _calculate_base_compute_requirement(self, shape: Tuple[int, ...]) -> float:
        """Calculate base compute requirement for operations on tensor of given shape."""
        # For attention-like operations, compute scales with sequence length squared
        if len(shape) >= 3:
            seq_len = shape[1]  # Assume second dimension is sequence length
            # Attention computation is roughly O(seq_len^2 * hidden_size)
            hidden_size = shape[-1]  # Last dimension is usually hidden size
            compute_requirement = (seq_len ** 2) * hidden_size
            return min(compute_requirement / 1e8, 1.0)  # Normalize and cap at 1.0
        
        return 0.5  # Default compute requirement
    
    def release_resources_for_optimization(self, optimization_name: str):
        """Release resources allocated for a specific optimization."""
        self.resource_manager.release_resources(optimization_name)
        
        if optimization_name in self.optimization_resource_usage:
            del self.optimization_resource_usage[optimization_name]
    
    def get_optimization_resource_summary(self) -> Dict[str, Any]:
        """Get summary of resource usage by optimizations."""
        return {
            'resource_status': self.resource_manager.get_resource_status(),
            'optimization_resource_usage': self.optimization_resource_usage,
            'optimization_profiles': self.optimization_profiles
        }
    
    def optimize_tensor_allocation(self, shape: Tuple[int, ...], 
                                  dtype: torch.dtype, 
                                  optimization_type: str) -> torch.Tensor:
        """Optimize tensor allocation based on optimization type and hardware."""
        # Get appropriate tensor from memory manager
        tensor = self.memory_manager.get_memory_efficient_tensor(shape, dtype, optimization_type)
        
        return tensor


def create_optimization_resource_manager(config) -> OptimizationResourceManager:
    """Factory function to create an optimization resource manager."""
    return OptimizationResourceManager(config)


# Example usage and testing
if __name__ == "__main__":
    # Mock config for testing
    class MockConfig:
        def __init__(self):
            self.memory_reservation_policy = 'conservative'
            self.compute_sharing_policy = 'fair'
            self.allocation_timeout = 10.0
            self.fallback_on_resource_exhaustion = True
            self.memory_fragmentation_tolerance = 0.1
            self.priority_boost_for_critical_optimizations = True
    
    config = MockConfig()
    
    # Create resource manager
    resource_manager = create_optimization_resource_manager(config)
    
    # Test resource allocation for different optimizations
    print("Testing resource allocation for optimizations...")
    
    # Define test shapes for different optimization types
    test_shapes = [
        ((1, 8, 512, 512), 'attention_weights'),  # Attention weights tensor
        ((1, 8, 1024, 64), 'kv_cache'),           # KV cache tensor
        ((1, 512, 4096), 'intermediate'),         # FFN intermediate tensor
        ((1, 576, 1152), 'embeddings'),           # Vision embeddings
        ((2, 128, 2048), 'activations')           # Activation tensor
    ]
    
    optimization_names = [
        'block_sparse_attention',
        'cross_modal_token_merging',
        'hierarchical_memory_compression',
        'learned_activation_routing',
        'adaptive_batch_processing'
    ]
    
    for i, (shape, tensor_type) in enumerate(test_shapes):
        opt_name = optimization_names[i % len(optimization_names)]
        
        print(f"  Testing {opt_name} with shape {shape}...")
        
        # Request resources for this optimization
        success, allocation = resource_manager.allocate_resources_for_optimization(opt_name, shape)
        
        print(f"    Allocation success: {success}")
        if success:
            print(f"      Memory allocated: {allocation['memory_allocated']:,} bytes")
            print(f"      Compute allocated: {allocation['compute_allocated']:.2f}%")
            if 'fallback_applied' in allocation:
                print(f"      Fallback applied: {allocation['fallback_applied']}")
        
        # Test tensor allocation optimization
        optimized_tensor = resource_manager.optimize_tensor_allocation(shape, torch.float32, tensor_type)
        print(f"      Optimized tensor shape: {optimized_tensor.shape}")
        
        # Release resources
        resource_manager.release_resources_for_optimization(opt_name)
        print(f"    Resources released for {opt_name}\n")
    
    # Test resource status
    print("Current resource status:")
    status = resource_manager.get_optimization_resource_summary()
    print(f"  GPU Memory utilization: {status['resource_status']['gpu_memory']['utilization']:.2%}")
    print(f"  CPU Memory utilization: {status['resource_status']['cpu_memory']['utilization']:.2%}")
    print(f"  Active allocations: {len(status['optimization_resource_usage'])}")
    
    print("\nResource management system tests completed successfully!")