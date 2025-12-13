"""
Model-specific optimization strategies (quantization, sparsity, etc.).

This module provides model-specific optimization strategies appropriate for each model.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import logging
from enum import Enum


class OptimizationType(Enum):
    """Types of optimizations."""
    QUANTIZATION = "quantization"
    SPARSITY = "sparsity"
    PRUNING = "pruning"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    LAYER_FUSION = "layer_fusion"
    MEMORY_OPTIMIZATION = "memory_optimization"


@dataclass
class OptimizationConfig:
    """Configuration for a specific optimization."""
    optimization_type: OptimizationType
    enabled: bool = False
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class OptimizationStrategy:
    """Base class for optimization strategies."""
    
    def __init__(self, name: str, optimization_type: OptimizationType):
        self.name = name
        self.optimization_type = optimization_type
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def apply(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """
        Apply the optimization to a model.

        Args:
            model: Model to optimize
            config: Optimization configuration

        Returns:
            Optimized model
        """
        raise RuntimeError("OptimizationStrategy.apply() is an abstract method that must be "
                         "implemented by subclasses. Use a specific optimization strategy instead.")


class QuantizationStrategy(OptimizationStrategy):
    """Quantization optimization strategy."""
    
    def __init__(self):
        super().__init__("Quantization", OptimizationType.QUANTIZATION)
    
    def apply(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Apply quantization to the model."""
        if not config.enabled:
            return model
        
        quantization_type = config.parameters.get("type", "dynamic")
        dtype = config.parameters.get("dtype", torch.qint8)
        
        if quantization_type == "dynamic":
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=dtype
            )
        elif quantization_type == "static":
            # Static quantization requires calibration
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            # Calibration would happen here with sample data
            torch.quantization.convert(model, inplace=True)
            quantized_model = model
        elif quantization_type == "qat":  # Quantization Aware Training
            model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            quantized_model = torch.quantization.prepare_qat(model, inplace=False)
        else:
            self._logger.warning(f"Unknown quantization type: {quantization_type}")
            return model
        
        self._logger.info(f"Applied {quantization_type} quantization to model")
        return quantized_model


class SparsityStrategy(OptimizationStrategy):
    """Sparsity optimization strategy."""
    
    def __init__(self):
        super().__init__("Sparsity", OptimizationType.SPARSITY)
    
    def apply(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Apply sparsity to the model."""
        if not config.enabled:
            return model
        
        sparsity_ratio = config.parameters.get("sparsity_ratio", 0.5)
        method = config.parameters.get("method", "magnitude")
        
        if method == "magnitude":
            # Apply magnitude-based pruning
            import torch.nn.utils.prune as prune
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    try:
                        prune.l1_unstructured(module, name='weight', amount=sparsity_ratio)
                        self._logger.info(f"Applied {sparsity_ratio*100}% magnitude pruning to {name}")
                    except Exception as e:
                        self._logger.warning(f"Could not prune {name}: {e}")
        elif method == "topk":
            # Custom top-k sparsity implementation
            self._apply_topk_sparsity(model, sparsity_ratio)
        else:
            self._logger.warning(f"Unknown sparsity method: {method}")
            return model
        
        self._logger.info(f"Applied {sparsity_ratio*100}% sparsity to model using {method} method")
        return model
    
    def _apply_topk_sparsity(self, model: nn.Module, sparsity_ratio: float):
        """Apply top-k sparsity to model weights."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name and len(param.shape) > 1:  # Only apply to weight matrices
                    num_params = param.numel()
                    k = int(num_params * (1 - sparsity_ratio))
                    
                    # Get top-k values
                    flat_param = param.view(-1)
                    topk_values, topk_indices = torch.topk(torch.abs(flat_param), k, largest=True)
                    
                    # Create mask
                    mask = torch.zeros_like(flat_param)
                    mask[topk_indices] = 1
                    
                    # Apply mask
                    param.data = param.data * mask.view(param.shape)


class PruningStrategy(OptimizationStrategy):
    """Pruning optimization strategy."""
    
    def __init__(self):
        super().__init__("Pruning", OptimizationType.PRUNING)
    
    def apply(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Apply pruning to the model."""
        if not config.enabled:
            return model
        
        pruning_ratio = config.parameters.get("pruning_ratio", 0.2)
        method = config.parameters.get("method", "l1")
        
        import torch.nn.utils.prune as prune
        
        if method == "l1":
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    try:
                        prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                        # Remove the reparameterization and permanently apply the mask
                        prune.remove(module, 'weight')
                        self._logger.info(f"Applied {pruning_ratio*100}% L1 pruning to {name}")
                    except Exception as e:
                        self._logger.warning(f"Could not prune {name}: {e}")
        elif method == "random":
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    try:
                        prune.random_unstructured(module, name='weight', amount=pruning_ratio)
                        prune.remove(module, 'weight')
                        self._logger.info(f"Applied {pruning_ratio*100}% random pruning to {name}")
                    except Exception as e:
                        self._logger.warning(f"Could not prune {name}: {e}")
        else:
            self._logger.warning(f"Unknown pruning method: {method}")
            return model
        
        self._logger.info(f"Applied {pruning_ratio*100}% pruning to model using {method} method")
        return model


class OptimizationManager:
    """Manager for applying model-specific optimization strategies."""
    
    def __init__(self):
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._strategies: Dict[OptimizationType, OptimizationStrategy] = {
            OptimizationType.QUANTIZATION: QuantizationStrategy(),
            OptimizationType.SPARSITY: SparsityStrategy(),
            OptimizationType.PRUNING: PruningStrategy(),
        }
        
        # Model-specific optimization configurations
        self._model_configs: Dict[str, List[OptimizationConfig]] = {
            "Qwen3-VL": [
                OptimizationConfig(
                    optimization_type=OptimizationType.QUANTIZATION,
                    enabled=True,
                    parameters={"type": "dynamic", "dtype": torch.qint8}
                ),
                OptimizationConfig(
                    optimization_type=OptimizationType.SPARSITY,
                    enabled=True,
                    parameters={"sparsity_ratio": 0.3, "method": "magnitude"}
                )
            ],
            "Qwen3-4B-Instruct-2507": [
                OptimizationConfig(
                    optimization_type=OptimizationType.QUANTIZATION,
                    enabled=True,
                    parameters={"type": "static", "dtype": torch.qint8}
                ),
                OptimizationConfig(
                    optimization_type=OptimizationType.PRUNING,
                    enabled=True,
                    parameters={"pruning_ratio": 0.2, "method": "l1"}
                )
            ]
        }
    
    def register_strategy(self, strategy: OptimizationStrategy) -> bool:
        """
        Register a new optimization strategy.
        
        Args:
            strategy: OptimizationStrategy instance
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            self._strategies[strategy.optimization_type] = strategy
            self._logger.info(f"Registered optimization strategy: {strategy.name}")
            return True
        except Exception as e:
            self._logger.error(f"Error registering optimization strategy: {e}")
            return False
    
    def get_optimization_configs(self, model_name: str) -> List[OptimizationConfig]:
        """
        Get optimization configurations for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of OptimizationConfig instances
        """
        return self._model_configs.get(model_name, [])
    
    def set_model_config(self, model_name: str, configs: List[OptimizationConfig]) -> bool:
        """
        Set optimization configurations for a specific model.
        
        Args:
            model_name: Name of the model
            configs: List of OptimizationConfig instances
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._model_configs[model_name] = configs
            self._logger.info(f"Set optimization configs for {model_name}")
            return True
        except Exception as e:
            self._logger.error(f"Error setting optimization configs for {model_name}: {e}")
            return False
    
    def apply_optimizations(
        self,
        model: nn.Module,
        model_name: str,
        custom_configs: Optional[List[OptimizationConfig]] = None
    ) -> nn.Module:
        """
        Apply optimizations to a model based on its name or custom configurations.
        
        Args:
            model: Model to optimize
            model_name: Name of the model
            custom_configs: Custom optimization configurations (optional)
            
        Returns:
            Optimized model
        """
        configs = custom_configs or self.get_optimization_configs(model_name)
        
        optimized_model = model
        
        for config in configs:
            if config.enabled and config.optimization_type in self._strategies:
                strategy = self._strategies[config.optimization_type]
                try:
                    optimized_model = strategy.apply(optimized_model, config)
                    self._logger.info(f"Applied {strategy.name} optimization to {model_name}")
                except Exception as e:
                    self._logger.error(f"Error applying {strategy.name} optimization: {e}")
        
        return optimized_model
    
    def get_optimization_recommendations(
        self,
        model_name: str,
        hardware_memory_gb: float,
        target_performance: str = "balanced"  # "speed", "memory", "balanced"
    ) -> List[OptimizationConfig]:
        """
        Get optimization recommendations based on hardware and performance target.
        
        Args:
            model_name: Name of the model
            hardware_memory_gb: Available hardware memory in GB
            target_performance: Performance target ("speed", "memory", "balanced")
            
        Returns:
            List of recommended OptimizationConfig instances
        """
        recommendations = []
        
        # Memory-constrained optimizations
        if hardware_memory_gb < 8.0:
            # Apply quantization to save memory
            recommendations.append(
                OptimizationConfig(
                    optimization_type=OptimizationType.QUANTIZATION,
                    enabled=True,
                    parameters={"type": "dynamic", "dtype": torch.qint8}
                )
            )
            
            # Apply sparsity to save memory
            sparsity_ratio = 0.3 if hardware_memory_gb < 4.0 else 0.2
            recommendations.append(
                OptimizationConfig(
                    optimization_type=OptimizationType.SPARSITY,
                    enabled=True,
                    parameters={"sparsity_ratio": sparsity_ratio, "method": "magnitude"}
                )
            )
        elif hardware_memory_gb > 16.0 and target_performance == "speed":
            # For high-memory systems with performance target, minimal optimizations
            recommendations.append(
                OptimizationConfig(
                    optimization_type=OptimizationType.QUANTIZATION,
                    enabled=False
                )
            )
        else:
            # Default optimizations for balanced systems
            recommendations.extend(self.get_optimization_configs(model_name))
        
        return recommendations


# Global optimization manager instance
optimization_manager = OptimizationManager()


def get_optimization_manager() -> OptimizationManager:
    """
    Get the global optimization manager instance.
    
    Returns:
        OptimizationManager instance
    """
    return optimization_manager