# Standardized Model Architecture for Inference-PIO

## Overview

This document provides a standardized architecture reference for all models in the Inference-PIO system. It consolidates the implementation details, optimizations, and interfaces for GLM-4-7, Qwen3-4b-instruct-2507, Qwen3-coder-30b, and Qwen3-vl-2b models.

## Model Directory Structure

Each model follows the same standardized directory structure:

```
src/inference_pio/models/[model_name]/
├── __init__.py
├── config.py
├── model.py
├── plugin.py
├── attention/
│   ├── __init__.py
│   ├── flash_attention.py
│   ├── sparse_attention.py
│   ├── paged_attention.py
│   ├── sliding_window_attention.py
│   └── multi_query_attention.py
├── benchmarks/
│   ├── __init__.py
│   ├── benchmark_throughput.py
│   ├── benchmark_inference_speed.py
│   ├── benchmark_memory_usage.py
│   ├── benchmark_power_efficiency.py
│   ├── benchmark_optimization_impact.py
│   └── benchmark_accuracy.py
├── cuda_kernels/
│   ├── __init__.py
│   └── optimizations.py
├── fused_layers/
│   ├── __init__.py
│   └── fused_layer_norm.py
├── kv_cache/
│   ├── __init__.py
│   └── compression_techniques.py
├── linear_optimizations/
│   ├── __init__.py
│   └── bias_removal.py
├── prefix_caching/
│   ├── __init__.py
│   └── prefix_cache_manager.py
├── rotary_embeddings/
│   ├── __init__.py
│   └── rotary_embedding.py
├── tensor_parallel/
│   ├── __init__.py
│   └── tensor_parallel_layers.py
├── tests/
│   ├── __init__.py
│   ├── test_plugin_integration.py
│   ├── test_model_loading.py
│   ├── test_inference.py
│   ├── test_attention.py
│   ├── test_optimizations.py
│   └── test_end_to_end.py
└── specific_optimizations/
    ├── __init__.py
    └── [model_specific_optimizations].py
```

## Standardized Plugin Interface

All models implement the same standardized plugin interface:

```python
from inference_pio.common.standard_plugin_interface import ModelPluginInterface

class BaseModelPlugin(ModelPluginInterface):
    def __init__(self):
        # Initialize with model-specific metadata
        metadata = ModelPluginMetadata(
            name="[MODEL_NAME]",
            version="1.0.0",
            author="[AUTHOR]",
            description="[DESCRIPTION]",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=["torch", "transformers", "accelerate"],
            compatibility={
                "torch_version": ">=2.0.0",
                "transformers_version": ">=4.30.0",
                "python_version": ">=3.8",
                "min_memory_gb": [MEMORY_GB]
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            model_architecture="[ARCHITECTURE]",
            model_size="[SIZE]",
            required_memory_gb=[REQUIRED_MEMORY_GB],
            supported_modalities=["text"],  # or ["text", "image"] for multimodal
            license="MIT",
            tags=[...],
            model_family="[FAMILY]",
            num_parameters=[NUM_PARAMS],
            test_coverage=0.95,
            validation_passed=True
        )
        super().__init__(metadata)
        
        # Initialize common attributes
        self._model = None
        self._tokenizer = None
        self._config = None
        self._compiled_model = None
        
        # Initialize optimization managers
        self._memory_manager = None
        self._tensor_paging_manager = None
        self._paging_enabled = False
        self._adaptive_batch_manager = None
        self._distributed_simulation_manager = None
        self._tensor_compressor = None
        self._compression_enabled = False
        self._disk_offloader = None
        self._disk_tensor_offloading_manager = None
        self._offloading_enabled = False
        self._activation_offloading_manager = None
        self._unimodal_preprocessor = None
```

## Model-Specific Implementations

### GLM-4-7 Model

**Model Characteristics:**
- Architecture: Transformer-based language model optimized for advanced reasoning
- Parameters: 4.7 billion
- Memory Requirement: 16 GB
- Modalities: Text-only

**Key Optimizations:**
- Reasoning-focused attention patterns
- Memory-efficient processing for complex tasks
- Specialized rotary embeddings for reasoning

### Qwen3-4B-Instruct-2507 Model

**Model Characteristics:**
- Architecture: Transformer-based language model optimized for instruction following
- Parameters: 4 billion
- Memory Requirement: 8 GB
- Modalities: Text-only

**Key Optimizations:**
- Instruction-following attention patterns
- Safety and alignment optimizations
- Conversational context management

### Qwen3-Coder-30B Model

**Model Characteristics:**
- Architecture: Transformer-based language model optimized for coding tasks
- Parameters: 30 billion
- Memory Requirement: 16 GB
- Modalities: Text-only

**Key Optimizations:**
- Syntax-aware attention mechanisms
- Code-specific KV-cache optimizations
- Multi-language processing optimizations

### Qwen3-VL-2B Model

**Model Characteristics:**
- Architecture: Transformer-based multimodal model optimized for vision-language tasks
- Parameters: 2 billion
- Memory Requirement: 6 GB
- Modalities: Text and Image

**Key Optimizations:**
- Cross-modal attention mechanisms
- Vision-language fusion optimizations
- Efficient image processing pipelines

## Standardized Configuration Classes

Each model has a standardized configuration class:

```python
from dataclasses import dataclass
from typing import Optional, List, Union

@dataclass
class BaseModelConfig:
    # Model identification
    model_path: str = "[DEFAULT_PATH]"
    model_name: str = "[MODEL_NAME]"
    
    # Device settings
    device: str = "cpu"  # Will be set dynamically during initialization
    
    # Model architecture parameters
    hidden_size: int = [HIDDEN_SIZE]
    num_attention_heads: int = [NUM_HEADS]
    num_hidden_layers: int = [NUM_LAYERS]
    max_position_embeddings: int = [MAX_POS]
    intermediate_size: int = [INTERMEDIATE_SIZE]
    vocab_size: int = [VOCAB_SIZE]
    
    # Memory optimization settings
    gradient_checkpointing: bool = True
    use_cache: bool = True
    torch_dtype: str = "float16"
    device_map: str = "auto"
    low_cpu_mem_usage: bool = True
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_new_tokens: int = 512
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    
    # Optimization flags
    use_flash_attention_2: bool = True
    use_sparse_attention: bool = True
    use_multi_query_attention: bool = True
    use_grouped_query_attention: bool = True
    use_paged_attention: bool = True
    use_sliding_window_attention: bool = True
    use_fused_layer_norm: bool = True
    use_bias_removal_optimization: bool = True
    use_tensor_parallelism: bool = False
    
    # KV-cache compression settings
    use_kv_cache_compression: bool = True
    kv_cache_compression_method: str = "combined"
    kv_cache_quantization_bits: int = 8
    
    # Prefix caching settings
    use_prefix_caching: bool = True
    prefix_cache_max_size: int = 1024 * 1024 * 256  # 256MB
    
    # CUDA kernels settings
    use_cuda_kernels: bool = True
    cuda_kernel_gelu_enabled: bool = True
    cuda_kernel_matmul_enabled: bool = True
    cuda_kernel_softmax_enabled: bool = True
    cuda_kernel_attention_enabled: bool = True
    cuda_kernel_mlp_enabled: bool = True
    cuda_kernel_layernorm_enabled: bool = True
    
    # Runtime memory optimization settings
    torch_compile_mode: str = "reduce-overhead"
    torch_compile_fullgraph: bool = False
    torch_compile_dynamic: bool = True
    enable_cudnn_benchmark: bool = True
    enable_memory_efficient_attention: bool = True
```

## Standardized Model Loading Process

All models follow the same standardized loading process:

```python
def load_model(self, config: Optional[BaseConfig] = None) -> nn.Module:
    """
    Load the model with all optimizations applied.
    
    Args:
        config: Model configuration (optional)
    
    Returns:
        Loaded and optimized model instance
    """
    try:
        if config is not None:
            self._config = config

        logger.info(f"Loading {self.metadata.name} model from: {self._config.model_path}")

        # Set device configuration
        device = getattr(self._config, 'device', 'cpu')
        if device == 'cpu':
            self._config.device_map = 'cpu'
        elif device.startswith('cuda'):
            self._config.device_map = device
        else:
            self._config.device_map = 'auto'

        # Create the base model
        self._model = BaseModelClass(self._config)

        # Apply model-specific optimizations
        self._apply_model_specific_optimizations()

        # Load tokenizer
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._config.model_path,
            trust_remote_code=True
        )

        # Set padding token if not present
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Apply runtime optimizations
        if hasattr(self._config, 'torch_compile_mode') and self._config.torch_compile_mode:
            self._model = torch.compile(
                self._model,
                mode=self._config.torch_compile_mode,
                fullgraph=self._config.torch_compile_fullgraph,
                dynamic=self._config.torch_compile_dynamic
            )

        self.is_loaded = True
        logger.info(f"{self.metadata.name} model loaded successfully")
        return self._model
    except Exception as e:
        logger.error(f"Failed to load {self.metadata.name} model: {e}")
        raise e
```

## Standardized Inference Process

All models implement the same standardized inference process:

```python
def infer(self, data: Any) -> Any:
    """
    Perform inference on the given data.
    
    Args:
        data: Input data for inference
    
    Returns:
        Inference results
    """
    if self._model is None or self._tokenizer is None:
        self.load_model()

    if not isinstance(data, str):
        raise ValueError(f"{self.metadata.name} model expects string input")

    # Handle empty input
    if not data.strip():
        logger.warning("Empty input provided, returning empty string")
        return ""

    try:
        # Tokenize input
        inputs = self._tokenizer(
            data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=getattr(self._config, 'max_position_embeddings', 8192)
        )

        # Move inputs to the same device as the model
        device = next(self._model.parameters()).device if self._model is not None else torch.device('cpu')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Use compiled model if available, otherwise use original model
        model_to_use = self.get_compiled_model()

        # Generate output with model-specific parameters
        with torch.no_grad():
            outputs = model_to_use.generate(
                **inputs,
                max_length=min(len(inputs['input_ids'][0]) + self._config.max_new_tokens, 
                              getattr(self._config, 'max_position_embeddings', 8192)),
                pad_token_id=self._config.pad_token_id,
                do_sample=self._config.do_sample,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
                top_k=self._config.top_k,
                repetition_penalty=self._config.repetition_penalty,
                num_return_sequences=1,
            )

        # Decode output
        generated_text = self._tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise e
```

## Standardized Cleanup Process

All models implement the same standardized cleanup process:

```python
def cleanup(self) -> bool:
    """
    Clean up resources used by the plugin.
    
    Returns:
        True if cleanup was successful, False otherwise
    """
    try:
        # Clean up model
        if hasattr(self, '_model') and self._model is not None:
            del self._model
            self._model = None
            
        # Clean up tokenizer
        if hasattr(self, '_tokenizer') and self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
            
        # Clean up compiled model
        if hasattr(self, '_compiled_model') and self._compiled_model is not None:
            del self._compiled_model
            self._compiled_model = None

        # Force memory cleanup
        self.force_memory_cleanup()

        # Clean up optimization managers
        if hasattr(self, '_disk_tensor_offloading_manager') and self._disk_tensor_offloading_manager:
            self._disk_tensor_offloading_manager.stop_proactive_management()

        if hasattr(self, '_disk_offloader') and self._disk_offloader:
            self._disk_offloader.cleanup()
            self._disk_offloader = None
            self._disk_tensor_offloading_manager = None
            self._offloading_enabled = False

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        self.is_loaded = False
        self.is_active = False

        logger.info(f"{self.metadata.name} plugin cleaned up successfully")
        return True
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return False
```

## Model-Specific Optimizations

Each model implements its own specific optimizations while maintaining compatibility with the standard interface:

### GLM-4-7 Specific Optimizations
- Reasoning-focused attention patterns
- Memory-efficient processing for complex tasks
- Specialized rotary embeddings for reasoning

### Qwen3-4B-Instruct-2507 Specific Optimizations
- Instruction-following attention patterns
- Safety and alignment optimizations
- Conversational context management

### Qwen3-Coder-30B Specific Optimizations
- Syntax-aware attention mechanisms
- Code-specific KV-cache optimizations
- Multi-language processing optimizations

### Qwen3-VL-2B Specific Optimizations
- Cross-modal attention mechanisms
- Vision-language fusion optimizations
- Efficient image processing pipelines

## Testing Standards

All models follow the same testing standards:

```python
# Standard test structure for all models
class TestModelPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = create_model_plugin()
        self.config = ModelConfig()
        
    def test_initialization(self):
        # Test plugin initialization
        success = self.plugin.initialize(config=self.config)
        self.assertTrue(success)
        
    def test_model_loading(self):
        # Test model loading
        model = self.plugin.load_model()
        self.assertIsNotNone(model)
        
    def test_inference(self):
        # Test inference functionality
        result = self.plugin.infer("Test input")
        self.assertIsInstance(result, str)
        
    def test_generation(self):
        # Test text generation
        result = self.plugin.generate_text("Test prompt")
        self.assertIsInstance(result, str)
        
    def test_cleanup(self):
        # Test cleanup functionality
        success = self.plugin.cleanup()
        self.assertTrue(success)
```

## Performance Benchmarks

All models include standardized performance benchmarks:

- Throughput benchmarks
- Inference speed benchmarks
- Memory usage benchmarks
- Power efficiency benchmarks
- Optimization impact benchmarks
- Accuracy benchmarks

## Conclusion

This standardized architecture ensures consistency across all models in the Inference-PIO system while allowing for model-specific optimizations. Each model maintains its unique characteristics while adhering to the common interface and structure.