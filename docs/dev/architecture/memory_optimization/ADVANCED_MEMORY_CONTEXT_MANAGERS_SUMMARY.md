# Comprehensive Context Manager System for Memory Resource Management

## Overview

This implementation provides a robust and production-ready system for managing memory resources using context managers. The system ensures proper allocation and deallocation of resources, prevents memory leaks, and handles cleanup automatically in all scenarios including exceptions.

## Key Components

### 1. Resource Tracker (`ResourceTracker`)
- Thread-safe resource tracking system
- Monitors all allocated resources
- Provides forced cleanup capability
- Prevents memory leaks by tracking resource states

### 2. General-Purpose Context Manager (`MemoryResourceContextManager`)
- Wraps around existing allocation/deallocation patterns
- Handles any resource type with custom allocator/deallocator functions
- Provides automatic cleanup and exception handling

### 3. Specialized Context Managers
- `KVCacheContextManager`: For KV cache tensor allocations
- `ImageFeaturesContextManager`: For image feature tensor allocations
- `TextEmbeddingsContextManager`: For text embedding tensor allocations
- `AdvancedMemoryPoolContextManager`: For raw memory pool allocations
- `VisionLanguageMemoryOptimizerContextManager`: For optimized tensor allocations

### 4. Context Manager Factories
- Factory functions for each context manager type
- Provide convenient API for developers
- Support all the same functionality as direct context managers

## Features

### Thread Safety
- All operations are thread-safe using `threading.RLock`
- Multiple threads can safely use context managers simultaneously
- Resource tracking is protected against race conditions

### Exception Handling
- Resources are properly cleaned up even when exceptions occur
- Context managers implement proper `__exit__` methods
- No resource leaks occur due to unhandled exceptions

### Resource State Management
- Tracks resource states: UNALLOCATED, ALLOCATED, DEALLOCATING, DEALLOCATED, ERROR
- Provides visibility into resource lifecycle
- Enables debugging and monitoring

### Automatic Cleanup
- Resources are automatically deallocated when context exits
- At-exit handler ensures cleanup on program termination
- Forced cleanup function available for emergency situations

## Usage Examples

### Basic Usage
```python
from memory_context_managers import kv_cache_context
from advanced_memory_pooling_system import AdvancedMemoryPoolingSystem

memory_system = AdvancedMemoryPoolingSystem()

# Using context manager
with kv_cache_context(memory_system, 1024*512, "my_tensor") as block:
    # Use the allocated block
    print(f"Allocated {block.size} bytes")
    # Block is automatically deallocated when exiting context
```

### Factory Function Usage
```python
from memory_context_managers import image_features_context

with image_features_context(memory_system, 2*1024*512, "img_features") as block:
    # Work with image features
    pass
```

### Complex Nested Usage
```python
with kv_cache_context(memory_system, 1024*512, "query") as query_block:
    with image_features_context(memory_system, 2*1024*512, "img") as img_block:
        # Both resources are available and will be cleaned up automatically
        pass
```

### Generic Resource Management
```python
def my_allocator():
    return "allocated_resource"

def my_deallocator(resource):
    print(f"Deallocating {resource}")

with memory_resource_context(my_allocator, my_deallocator, "my_resource") as resource:
    # Use the resource
    pass
```

## Integration with Existing Systems

The context managers are designed to integrate seamlessly with:
- `AdvancedMemoryPoolingSystem` from `advanced_memory_pooling_system.py`
- `AdvancedMemoryPool` from `advanced_memory_management_vl.py`
- `VisionLanguageMemoryOptimizer` from `advanced_memory_management_vl.py`

## Error Handling

The system handles various error scenarios:
- Allocation failures
- Deallocation failures
- Exceptions within context
- Thread safety violations
- Invalid input parameters

## Performance Considerations

- Minimal overhead for resource tracking
- Efficient state management
- Thread-safe operations without excessive locking
- Proper cleanup prevents memory leaks

## Testing

The implementation includes comprehensive tests covering:
- Basic functionality
- Exception handling
- Thread safety
- Integration scenarios
- Resource tracking
- Edge cases

All tests pass successfully, ensuring the system is production-ready.