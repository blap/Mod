# Core Interfaces

This document describes the specialized interfaces for different functional components in the Inference-PIO project.

## Overview

Interfaces are designed to promote separation of concerns and facilitate code maintenance. Each interface represents a distinct concept within the system, ensuring modularity and extensibility.

## Interface Types

### 1. MemoryManagerInterface
Interface for memory management operations, including tensor pagination, smart swapping, and memory statistics.

### 2. DistributedExecutionManagerInterface
Interface for distributed execution operations, including multi-GPU execution simulation and model partitioning.

### 3. TensorCompressionManagerInterface
Interface for tensor compression operations, including compression of model weights and activations.

### 4. SecurityManagerInterface
Interface for security operations, including file access validation and network security checks.

### 5. KernelFusionManagerInterface
Interface for kernel fusion operations and model operation optimizations.

### 6. AdaptiveBatchingManagerInterface
Interface for adaptive batch sizing operations based on performance metrics.

### 7. ModelSurgeryManagerInterface
Interface for model surgery operations to identify and remove non-essential components.

### 8. PipelineManagerInterface
Interface for disk-based pipeline operations for inference.

### 9. ShardingManagerInterface
Interface for extreme model sharding operations into hundreds of small fragments.

## Usage

To use an interface, import it and implement it in your class:

```python
from src.inference_pio.common.interfaces.memory_interface import MemoryManagerInterface

class MyMemoryManager(MemoryManagerInterface):
    def setup_memory_management(self, **kwargs) -> bool:
        # Specific implementation
        pass

    # Implement other required methods...
```

## Benefits

- **Clarity**: Each interface has a well-defined responsibility.
- **Flexibility**: Facilitates swapping implementations.
- **Testability**: Interfaces make it easier to create mocks for testing.
- **Maintainability**: Changes in one part of the system do not affect others.
