# Qwen3-VL API Reference

This document provides a comprehensive reference for the Qwen3-VL API, highlighting the new consolidated module structure.

## Package Structure

The Qwen3-VL package is organized into logical modules for better maintainability and usage:

### Main Package Access

```python
# Import the main package
import qwen3_vl

# Or import from the source directory
from src import qwen3_vl
```

### Core Modules

#### Attention Mechanisms (`src/attention/` and `src/components/attention/`)

```python
# From the attention module
from src.attention import (
    Qwen3VLAttention,
    FlashAttention2,
    BlockSparseAttention,
    DynamicSparseAttention,
    Qwen3VLRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv
)

# From the components attention module
from src.components.attention import (
    OptimizedQwen3VLAttention,
    AttentionMechanisms,
    DynamicSparseAttention as ComponentDynamicSparseAttention
)
```

#### Models (`src/models/` and `src/qwen3_vl/models/`)

```python
# Core model interfaces
from src.models import (
    Qwen3VLModel,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
    Qwen3VLPreTrainedModel
)

# Model components
from src.qwen3_vl.components.models import Qwen3VLModel as ConsolidatedQwen3VLModel
```

#### Memory Management (`src/qwen3_vl/memory_management/`)

```python
# Memory management utilities
from src.qwen3_vl.memory_management import (
    KVCacheOptimizer,
    GeneralMemoryManager,
    VisionLanguageMemoryManager,
    SparseMemoryManager,
    MemoryPool,
    MemoryEfficientAttention
)

# Specific memory managers
from src.qwen3_vl.memory_management.kv_cache_optimizer import KVCacheOptimizer
from src.qwen3_vl.memory_management.general_memory_manager import GeneralMemoryManager
from src.qwen3_vl.memory_management.vision_language_memory_manager import VisionLanguageMemoryManager
```

#### Configuration (`src/qwen3_vl/config/`)

```python
# Configuration system
from src.qwen3_vl.config import (
    Qwen3VLConfig,
    UnifiedConfig,
    MemoryConfig,
    CPUConfig,
    GPUConfig,
    PowerManagementConfig,
    OptimizationConfig,
    UnifiedConfigManager
)

# Configuration utilities
from src.qwen3_vl.config.unified_config import (
    get_default_config,
    create_unified_config_manager,
    get_hardware_optimized_config
)
```

#### Inference (`src/qwen3_vl/inference/`)

```python
# Inference utilities
from src.qwen3_vl.inference import (
    Qwen3VLInference,
    UnifiedInferencePipeline,
    DataPipelineOptimization
)

# Generation utilities
from src.qwen3_vl.inference.generation import generate_text, generate_multimodal
from src.qwen3_vl.inference.cli import main as cli_main
```

#### Optimization (`src/components/optimization/` and `src/qwen3_vl/optimization/`)

```python
# Optimization utilities
from src.components.optimization import (
    AdaptiveSparse,
    AdaptivePrecision,
    DistributedPipelineParallelism,
    FasterRotaryEmbeddings,
    MemoryEfficientGradientAccumulation,
    AdaptiveSequencePacking,
    CrossLayerParameterRecycling,
    AdaptiveBatchProcessing,
    HardwareSpecificKernels,
    KVCachingOptimization,
    LearnedActivationRouting,
    HierarchicalMemoryCompression,
    CrossModalTokenMerging
)

# Consolidated optimization
from src.qwen3_vl.optimization import (
    UnifiedOptimizationOrchestrator,
    UnifiedArchitecture,
    UnifiedOptimizationManager,
    OrderOptimizer,
    MemoryOptimizationIntegrator,
    KernelFusion,
    FallbackManager
)
```

## Key Classes and Functions

### Model Classes

#### Qwen3VLModel
```python
from src.qwen3_vl.components.models import Qwen3VLModel

# Initialize the model
model = Qwen3VLModel(config=qwen3_vl_config)
```

#### Qwen3VLInference
```python
from src.qwen3_vl.inference import Qwen3VLInference

# Initialize inference engine
inference_engine = Qwen3VLInference(model=model)
response = inference_engine.generate_response(text="Describe this image", max_new_tokens=50)
```

### Configuration Management

#### Unified Configuration System
```python
from src.qwen3_vl.config import (
    UnifiedConfig,
    get_default_config,
    create_unified_config_manager
)

# Get default configuration
config = get_default_config()

# Create configuration manager
manager = create_unified_config_manager()

# Get optimized configuration for different scenarios
minimal_config = manager.get_config("minimal")
balanced_config = manager.get_config("balanced")
aggressive_config = manager.get_config("aggressive")

# Get hardware-optimized configuration
hardware_specs = {
    "gpu_memory": 6 * 1024 * 1024 * 1024,  # 6GB
    "cpu_cores": 4,
    "memory_gb": 8
}
hw_config = manager.get_hardware_optimized_config(hardware_specs)
```

### Memory Management

#### Memory Manager
```python
from src.qwen3_vl.memory_management import (
    GeneralMemoryManager,
    KVCacheOptimizer,
    VisionLanguageMemoryManager
)

# Initialize memory manager
memory_manager = GeneralMemoryManager(config=config)

# Use KV cache optimizer
kv_cache_optimizer = KVCacheOptimizer(config=config)

# For multimodal scenarios
vl_memory_manager = VisionLanguageMemoryManager(config=config)
```

### Optimization Components

#### Mixture of Experts (MoE)
```python
from src.qwen3_vl.components.routing import MoeLayer

# Initialize MoE layer
moe_layer = MoeLayer(
    hidden_size=config.hidden_size,
    num_experts=config.optimization_config.moe_num_experts,
    top_k=config.optimization_config.moe_top_k
)
```

#### Adaptive Attention
```python
from src.qwen3_vl.optimization import AdaptiveAttention

# Initialize adaptive attention
adaptive_attention = AdaptiveAttention(
    config=config,
    attention_type="flash"  # or "sparse", "dynamic_sparse"
)
```

## Module Imports

With the consolidated structure, here are the recommended import patterns:

### Standard Imports
```python
# For general usage
from src.qwen3_vl import Qwen3VLModel, Qwen3VLConfig

# For specific components
from src.qwen3_vl.inference import Qwen3VLInference
from src.qwen3_vl.memory_management import KVCacheOptimizer
from src.qwen3_vl.config import UnifiedConfig
```

### Component-Specific Imports
```python
# Attention mechanisms
from src.attention import FlashAttention2
from src.components.attention import OptimizedQwen3VLAttention

# Optimization components
from src.components.optimization import AdaptivePrecision
from src.qwen3_vl.optimization import UnifiedOptimizationManager

# Memory management
from src.qwen3_vl.memory_management import GeneralMemoryManager
```

### Legacy Compatibility
```python
# The package maintains backward compatibility
from src.qwen3_vl import (
    # Old-style imports still work
    Qwen3VLModel,
    Qwen3VLAttention,
    FlashAttention2
)
```

## Key Interfaces

### ConfigurableComponent
```python
from src.qwen3_vl.components.system.interfaces import ConfigurableComponent

class MyComponent(ConfigurableComponent):
    def __init__(self, config):
        super().__init__(config)
        
    def configure(self, config):
        # Apply configuration
        pass
```

### MemoryManager
```python
from src.qwen3_vl.components.system.interfaces import MemoryManager

class MyMemoryManager(MemoryManager):
    def allocate(self, size):
        # Custom allocation logic
        pass
        
    def deallocate(self, ptr):
        # Custom deallocation logic
        pass
```

## Usage Patterns

### Creating and Configuring a Model
```python
from src.qwen3_vl.config import get_default_config, create_unified_config_manager
from src.qwen3_vl.components.models import Qwen3VLModel
from src.qwen3_vl.inference import Qwen3VLInference

# Get configuration
config = get_default_config()
config_manager = create_unified_config_manager()
optimized_config = config_manager.get_config("balanced")

# Create model
model = Qwen3VLModel(config=optimized_config)

# Create inference engine
inference_engine = Qwen3VLInference(model)
```

### Using Memory Management
```python
from src.qwen3_vl.memory_management import GeneralMemoryManager, KVCacheOptimizer

# Initialize memory management
memory_manager = GeneralMemoryManager(config=config)
kv_optimizer = KVCacheOptimizer(config=config)

# Apply optimizations
optimized_model = kv_optimizer.apply_optimizations(model)
```

This consolidated API structure provides clear organization while maintaining flexibility and backward compatibility.