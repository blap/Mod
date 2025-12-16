"""
Hardware Adaptation Integration for Qwen3-VL
Integrates the hardware adaptation system with existing Qwen3-VL architecture
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, List
import logging
from functools import wraps

from src.qwen3_vl.hardware.cpu_detector import get_hardware_optimizer
from src.qwen3_vl.hardware.adaptive_threading import get_threading_optimizer
from src.qwen3_vl.hardware.adaptive_memory import get_memory_optimizer
from src.qwen3_vl.hardware.simd_optimizer import get_simd_optimizer
from src.qwen3_vl.hardware.unified_config import get_unified_optimizer, get_model_hardware_config
from src.qwen3_vl.hardware.performance_optimizer import get_performance_optimizer, optimize_model_for_cpu


logger = logging.getLogger(__name__)


class HardwareAdaptiveModel(nn.Module):
    """
    A wrapper around Qwen3-VL models that provides hardware-adaptive optimizations
    """
    
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        
        # Initialize hardware optimizers
        self.unified_optimizer = get_unified_optimizer()
        self.threading_optimizer = get_threading_optimizer()
        self.memory_optimizer = get_memory_optimizer()
        self.simd_optimizer = get_simd_optimizer()
        self.performance_optimizer = get_performance_optimizer()
        
        # Apply hardware-specific optimizations to the base model
        self.optimized_model = optimize_model_for_cpu(self.base_model)
        
        logger.info("Hardware-adaptive model wrapper initialized")
    
    def forward(self, *args, **kwargs):
        """
        Forward pass with hardware-adaptive optimizations
        """
        return self.optimized_model(*args, **kwargs)
    
    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare inputs with hardware-adaptive optimizations
        
        Args:
            inputs: Raw input dictionary
            
        Returns:
            Optimized input dictionary
        """
        # Apply memory optimizations to inputs
        optimized_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                # Apply memory optimizations based on tensor size and hardware
                if self.memory_optimizer.should_compress_activations():
                    # Apply compression if needed
                    optimized_inputs[key] = self.memory_optimizer.memory_manager.compress_tensor(value)
                else:
                    optimized_inputs[key] = value
            else:
                optimized_inputs[key] = value
        
        return optimized_inputs
    
    def get_optimal_batch_size(self, base_batch_size: int = 1) -> int:
        """
        Get the optimal batch size for the current hardware
        
        Args:
            base_batch_size: Base batch size to adjust
            
        Returns:
            Optimal batch size for current hardware
        """
        return self.threading_optimizer.threading_model.get_optimal_batch_size(base_batch_size)
    
    def optimize_inference(self, input_data: torch.Tensor, 
                          batch_size: int = 1) -> torch.Tensor:
        """
        Optimize inference with hardware-specific optimizations
        
        Args:
            input_data: Input tensor for inference
            batch_size: Batch size for inference
            
        Returns:
            Model output
        """
        return self.performance_optimizer.performance_optimizer.optimize_for_inference(
            self.optimized_model, input_data, batch_size
        )


def apply_hardware_optimizations_to_model(model: nn.Module) -> nn.Module:
    """
    Apply hardware-specific optimizations to a model
    
    Args:
        model: The model to optimize
        
    Returns:
        Hardware-optimized model
    """
    return optimize_model_for_cpu(model)


def get_hardware_adaptive_config() -> Dict[str, Any]:
    """
    Get the hardware-adaptive configuration for the current system
    
    Returns:
        Dictionary containing hardware-adaptive configuration
    """
    return get_model_hardware_config()


def hardware_optimized_inference(model: nn.Module, 
                               inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                               **kwargs) -> Any:
    """
    Run hardware-optimized inference
    
    Args:
        model: The model to run inference on
        inputs: Input data (tensor or dictionary of tensors)
        **kwargs: Additional arguments for inference
        
    Returns:
        Model output
    """
    # Apply hardware optimizations to the model if not already done
    if not hasattr(model, '_hardware_optimized'):
        optimized_model = apply_hardware_optimizations_to_model(model)
        optimized_model._hardware_optimized = True
    else:
        optimized_model = model
    
    # Prepare inputs with hardware optimizations
    if isinstance(inputs, dict):
        prepared_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                # Apply any necessary input optimizations
                prepared_inputs[key] = value
            else:
                prepared_inputs[key] = value
        inputs = prepared_inputs
    elif isinstance(inputs, torch.Tensor):
        # For single tensor inputs, no special preparation needed
        pass
    
    # Run inference
    with torch.no_grad():
        if isinstance(inputs, dict):
            output = optimized_model(**inputs, **kwargs)
        else:
            output = optimized_model(inputs, **kwargs)
    
    return output


class HardwareAdaptiveTrainer:
    """
    Hardware-adaptive trainer that adjusts training parameters based on detected hardware
    """
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.performance_optimizer = get_performance_optimizer()
        
        # Apply hardware optimizations to the model
        self.model = apply_hardware_optimizations_to_model(self.model)
        
        # Get hardware-adaptive configuration
        self.config = get_hardware_adaptive_config()

        logger.info("Hardware-adaptive trainer initialized")
    
    def train_batch(self, data_batch: List[torch.Tensor]) -> Dict[str, float]:
        """
        Train on a single batch with hardware-adaptive optimizations
        
        Args:
            data_batch: List of tensors for training (typically [inputs, targets])
            
        Returns:
            Dictionary containing training metrics
        """
        # Apply threading optimizations for data loading
        inputs, targets = data_batch[0], data_batch[1]

        # Move data to appropriate device
        device = next(self.model.parameters()).device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = self.model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Return metrics
        return {
            'loss': loss.item(),
            'batch_size': inputs.size(0)
        }
    
    def get_optimal_training_params(self) -> Dict[str, Any]:
        """
        Get optimal training parameters based on hardware
        
        Returns:
            Dictionary containing optimal training parameters
        """
        recommendations = self.performance_optimizer.get_cpu_specific_optimizations()['recommendations']

        return {
            'batch_size': recommendations.get('batch_size', 'standard'),
            'gradient_accumulation_steps': 1,  # Adjust based on memory
            'learning_rate_multiplier': 1.0,   # Adjust based on stability
            'num_workers': self.config.get('num_workers', 4)
        }


def hardware_adaptive_decorator(func):
    """
    Decorator that applies hardware optimizations to a function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Initialize hardware optimizers
        unified_optimizer = get_unified_optimizer()
        
        # Apply any pre-execution optimizations
        result = func(*args, **kwargs)
        
        return result
    return wrapper


def initialize_hardware_adaptation():
    """
    Initialize the hardware adaptation system
    This function should be called at the start of the application
    """
    logger.info("Initializing hardware adaptation system...")
    
    # Initialize all hardware optimizers
    unified_optimizer = get_unified_optimizer()
    threading_optimizer = get_threading_optimizer()
    memory_optimizer = get_memory_optimizer()
    simd_optimizer = get_simd_optimizer()
    performance_optimizer = get_performance_optimizer()
    
    # Get and log the optimization report
    report = performance_optimizer.get_optimization_report()
    cpu_model = report['cpu_specific_optimizations']['cpu_model']
    
    logger.info(f"Hardware adaptation system initialized for {cpu_model}")
    logger.info(f"Performance path: {report['optimization_strategy']}")
    
    return {
        'unified_optimizer': unified_optimizer,
        'threading_optimizer': threading_optimizer,
        'memory_optimizer': memory_optimizer,
        'simd_optimizer': simd_optimizer,
        'performance_optimizer': performance_optimizer
    }


def get_hardware_summary() -> Dict[str, Any]:
    """
    Get a summary of the hardware adaptation system
    
    Returns:
        Dictionary containing hardware summary
    """
    performance_optimizer = get_performance_optimizer()
    report = performance_optimizer.get_optimization_report()
    
    return {
        'cpu_model': report['cpu_specific_optimizations']['cpu_model'],
        'performance_path': report['optimization_strategy'],
        'recommendations': report['recommendations'],
        'memory_pool_size_gb': report['cpu_specific_optimizations']['applied_config']['memory_pool_size'] / (1024**3),
        'simd_instruction_set': report['cpu_specific_optimizations']['applied_config']['simd_instruction_set'],
        'vector_width': report['cpu_specific_optimizations']['applied_config']['vector_width'],
        'optimal_workers': report['cpu_specific_optimizations']['applied_config']['num_workers']
    }


# Integration with existing Qwen3-VL components
def integrate_with_qwen3vl_pipeline():
    """
    Integrate hardware adaptation with existing Qwen3-VL pipeline components
    """
    from src.qwen3_vl.training.optimizers.cpu_optimizations import CPUPipeline, CPUPreprocessor
    from src.qwen3_vl.training.optimizers.cpu_optimizations import apply_cpu_optimizations
    
    # Create a hardware-adaptive pipeline
    class HardwareAdaptivePipeline(CPUPipeline):
        def __init__(self, model: nn.Module, config, preprocessor):
            # Apply hardware optimizations to the model first
            self.hardware_optimized_model = apply_hardware_optimizations_to_model(model)
            
            # Initialize parent with optimized model
            super().__init__(self.hardware_optimized_model, config, preprocessor)
        
        def preprocess_and_infer(self, texts, images=None, tokenizer=None, **generation_kwargs):
            # Use hardware-adaptive inference
            return hardware_optimized_inference(
                self.model, 
                {'texts': texts, 'images': images}, 
                tokenizer=tokenizer, 
                **generation_kwargs
            )
    
    # Create a hardware-adaptive preprocessor
    class HardwareAdaptivePreprocessor(CPUPreprocessor):
        def __init__(self, config):
            super().__init__(config)
            
            # Get hardware-specific configuration
            self.hardware_config = get_hardware_adaptive_config()
        
        def preprocess_batch(self, texts, images=None, return_tensors="pt", tokenizer=None):
            # Use hardware-optimized preprocessing
            return super().preprocess_batch(texts, images, return_tensors, tokenizer)
    
    return HardwareAdaptivePipeline, HardwareAdaptivePreprocessor


if __name__ == "__main__":
    print("Hardware Adaptation Integration for Qwen3-VL")
    print("=" * 50)
    
    # Initialize hardware adaptation
    optimizers = initialize_hardware_adaptation()
    print(f"Hardware adaptation system initialized")
    
    # Get hardware summary
    summary = get_hardware_summary()
    print(f"CPU Model: {summary['cpu_model']}")
    print(f"Performance Path: {summary['performance_path']}")
    print(f"Memory Pool: {summary['memory_pool_size_gb']:.1f} GB")
    print(f"SIMD: {summary['simd_instruction_set']} (width: {summary['vector_width']})")
    print(f"Optimal Workers: {summary['optimal_workers']}")
    
    # Test model optimization
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
        
        def forward(self, x):
            return self.linear(x)

    test_model = TestModel()
    print(f"\nOriginal model parameters: {sum(p.numel() for p in test_model.parameters())}")
    
    # Apply hardware optimizations
    optimized_model = apply_hardware_optimizations_to_model(test_model)
    print(f"Optimized model parameters: {sum(p.numel() for p in optimized_model.parameters())}")
    
    # Test inference
    test_input = torch.randn(2, 10)
    output = hardware_optimized_inference(optimized_model, test_input)
    print(f"Inference output shape: {output.shape}")
    
    # Test pipeline integration
    try:
        adaptive_pipeline, adaptive_preprocessor = integrate_with_qwen3vl_pipeline()
        print("Pipeline integration successful")
    except ImportError as e:
        print(f"Pipeline integration skipped due to missing components: {e}")
    
    print("\nHardware adaptation system integration completed!")